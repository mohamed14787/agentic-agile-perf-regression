import json
import os
import re
from camel.messages import BaseMessage
from agile_team_cline.agents.base import make_agent, zh_prompt
from agile_team_cline.config import MAX_TASKS_PER_SPRINT, TASK_TOKEN_LIMITS, make_architect_model
from agile_team_cline.shared import extract_json, step_with_retry
from agile_team_cline.workspace_discovery import get_existing_class_names

_ARCHITECT_GOAL = (
    "You design software. When asked, you output a task list in a strict pipe-delimited format. "
    "No explanations. No markdown. Follow the format exactly."
)

_CONTRACT_PROMPT = (
    "⚡ Design a minimal JSON contract for this feature.\n"
    "Feature: {feature}\n"
    "Stack: {stack_name}\n\n"
    "Output ONLY a JSON object (no prose, no markdown fences) with this exact shape:\n"
    "{{\n"
    '  "entities": [{{"name": "Note", "fields": {{"id": "UUID", "title": "string", "body": "string"}}}}],\n'
    '  "endpoints": [{{"method": "GET", "path": "/notes/{{id}}", "response": "Note"}}],\n'
    '  "schemas": {{}}\n'
    "}}\n"
    "Include EVERY entity named in the feature — INCLUDING parent/owner entities referenced only through a relationship, even if they have no dedicated endpoint "
    "(e.g. 'An Owner has Pets' ⇒ add an Owner entity AND give Pet an ownerId field linking to it). "
    "For every relationship between entities, add the foreign-key field (e.g. petId, ownerId) on the owning side.\n"
    "A computed/aggregated RESULT shaped for a response (a total, summary, report, count, projection) that is NOT stored in its own table is a response DTO, not an entity: either leave it out of 'entities' or mark it {{\"name\": \"...\", \"dto\": true}} — never give it a table. RULE OF THUMB: if a type appears ONLY as an endpoint 'response' (never written by POST/PUT/PATCH, never referenced as a field by another entity), it MUST be marked \"dto\": true — regardless of how it is named.\n"
    "Keep it small: ≤6 entities, ≤8 endpoints. JSON only."
)

_CONTRACT_PROMPT_ZH = (
    "⚡ 为此feature设计一个最小化的JSON contract。\n"
    "Feature: {feature}\n"
    "Stack: {stack_name}\n\n"
    "仅输出一个JSON对象（不要说明文字，不要markdown代码块），结构严格如下：\n"
    "{{\n"
    '  "entities": [{{"name": "Note", "fields": {{"id": "UUID", "title": "string", "body": "string"}}}}],\n'
    '  "endpoints": [{{"method": "GET", "path": "/notes/{{id}}", "response": "Note"}}],\n'
    '  "schemas": {{}}\n'
    "}}\n"
    "必须包含feature中提到的每一个entity——包括只通过关系被引用的父/拥有者entity，即使它没有专属endpoint"
    "（例如'An Owner has Pets' ⇒ 须新增Owner entity，并给Pet加上指向它的ownerId字段）。"
    "对entity之间的每一个关系，在拥有方加上外键字段（例如petId, ownerId）。\n"
    "保持精简：≤6个entities，≤8个endpoints。仅输出JSON。"
)

# Sprint-1 scaffold rules, per stack family. The templates below were Spring-only for
# every stack until 2026-08-02: 12 of the first 15 TeaStore trunk features hallucinated a
# phantom Spring config class (PersistenceConfig, JerseyConfig, JpaConfig,
# RestApplicationConfig, DataInitializerConfig, ApplicationConfig, Application.java) into a
# JAX-RS/EclipseLink reactor with no Spring on the classpath — 171 ORPHAN ARTIFACT /
# CONTRACT DRIFT hits that _vf_ceiling could only detect after the fact, at ≥2 cline
# refinement rounds each. The instruction also fought the pipeline's own rules: sprint.py
# refuses to delete persistence.xml on non-Spring stacks, and dual_track's deployability
# gate FAILS a run whose generated application.properties re-points the EMF at H2. Prevent
# rather than detect: tell each stack the truth about its own configuration model.
#
# The "spring" entry is FROZEN — it reproduces the pre-refactor prompt character for
# character (the 15-feature dataset was generated with it, so any drift invalidates
# cross-feature comparison). Do not edit it.
_SPRINT1_CONFIG = {
    "spring": {
        "file_examples": "pom.xml, application.properties, WebSocketConfig.java",
        "rules": (
            "Spring Boot rules: configure the datasource ONLY in application.properties (target the H2 in-memory database). "
            "NEVER generate a persistence.xml — Spring Boot ignores it and it conflicts with application.properties. "
            "Any JPA/Hibernate configuration must be a Java @Configuration class, not persistence.xml.\n"
        ),
        "rules_zh": (
            "Spring Boot规则：数据源只能在application.properties中配置（指向H2内存数据库）。"
            "绝不要生成persistence.xml——Spring Boot会忽略它，且与application.properties冲突。"
            "任何JPA/Hibernate配置都必须是Java的@Configuration类，而非persistence.xml。\n"
        ),
        "examples": (
            "T1|Initialize Maven project|config|1|pom.xml\n"
            "T2|Configure database and WebSocket|config|1|application.properties\n"
        ),
    },
    "jakarta": {
        "file_examples": "pom.xml, persistence.xml, web.xml",
        "rules": (
            "Jakarta EE rules — there is NO Spring on this classpath. This stack is JAX-RS (Jersey) on Tomcat "
            "with JPA/EclipseLink, and all configuration is DECLARATIVE XML:\n"
            "- src/main/resources/META-INF/persistence.xml — the persistence unit and its <class> list. "
            "This file is REQUIRED here; deleting or omitting it destroys the persistence layer.\n"
            "- src/main/webapp/WEB-INF/web.xml — the Jersey servlet and its resource packages.\n"
            "- pom.xml — dependencies and build config.\n"
            "NEVER emit any of the following, they do not work on this stack and become dead files that "
            "fail the build gates: a @Configuration/@Bean/@SpringBootApplication class; ANY *Config.java "
            "(PersistenceConfig, JerseyConfig, JpaConfig, RestApplicationConfig, DataInitializerConfig, "
            "ApplicationConfig, WebSocketConfig — the whole family); a jakarta.ws.rs.core.Application or "
            "ResourceConfig subclass; an Application.java or any class with a main() method; or "
            "src/main/resources/application.properties (nothing reads it, and it silently overrides the "
            "real datasource).\n"
            "These descriptors ALREADY EXIST in this brownfield module — EDIT them in place. Emit a task "
            "only for a file this feature genuinely changes; emit no Sprint-1 tasks at all if it changes none.\n"
        ),
        "rules_zh": (
            "Jakarta EE规则——此classpath上没有Spring。本stack是Tomcat上的JAX-RS (Jersey) 加 JPA/EclipseLink，"
            "所有配置都是声明式XML：\n"
            "- src/main/resources/META-INF/persistence.xml —— persistence unit及其<class>列表。"
            "此文件在这里是必需的；删除或省略它会破坏持久化层。\n"
            "- src/main/webapp/WEB-INF/web.xml —— Jersey servlet及其resource packages。\n"
            "- pom.xml —— 依赖与构建配置。\n"
            "绝不要生成以下任何一项，它们在本stack上无效，会变成使构建门禁失败的死文件："
            "@Configuration/@Bean/@SpringBootApplication类；任何 *Config.java"
            "（PersistenceConfig, JerseyConfig, JpaConfig, RestApplicationConfig, DataInitializerConfig, "
            "ApplicationConfig, WebSocketConfig —— 整个家族）；jakarta.ws.rs.core.Application 或 "
            "ResourceConfig 的子类；Application.java 或任何带 main() 方法的类；"
            "以及 src/main/resources/application.properties（没有任何代码读取它，且会静默覆盖真正的数据源）。\n"
            "这些描述符在本brownfield模块中已经存在——请就地编辑。仅为本feature确实修改的文件产出任务；"
            "如果一个都不修改，就不要输出任何Sprint 1任务。\n"
        ),
        "examples": (
            "T1|Add the feature's dependencies to the build|config|1|pom.xml\n"
            "T2|Register the new entity in the persistence unit|config|1|persistence.xml\n"
        ),
    },
    "generic": {
        "file_examples": "the dependency manifest, the settings module",
        "rules": (
            "Configuration rules: use THIS stack's own dependency manifest and its native settings/config "
            "module. Do NOT emit Spring artifacts (@Configuration classes, application.properties) or Java EE "
            "descriptors (persistence.xml, web.xml) — they belong to other stacks and do nothing here.\n"
        ),
        "rules_zh": (
            "配置规则：使用本stack自己的依赖清单及其原生的settings/config模块。"
            "不要产出Spring产物（@Configuration类、application.properties）或Java EE描述符"
            "（persistence.xml、web.xml）——它们属于其他stack，在这里不起任何作用。\n"
        ),
        "examples": (
            "T1|Declare the project dependencies|config|1|requirements.txt\n"
            "T2|Configure the database connection|config|1|settings.py\n"
        ),
    },
}

# Sprint 1 only — architecture scaffold tasks
_ARCHITECT_SPRINT1_PROMPT = (
    "⚡ List the Sprint 1 architecture files to build for: {feature}\n"
    "Stack: {stack_name}.\n"
    "Sprint 1 = Architecture only: project config, build files, database config.\n"
    "Output ONLY lines in this exact format (no headers, no extra text):\n"
    "ID|title|type|sprint|target_file\n"
    "Where type is one of: config\n"
    "Where sprint must be 1 for ALL tasks.\n"
    "Where target_file is the exact filename (e.g. {file_examples}).\n"
    "{rules}"
    "Max {max_tasks} tasks. Example:\n"
    "{examples}"
    "Now output Sprint 1 lines only for {feature}:"
)

_ARCHITECT_SPRINT1_PROMPT_ZH = (
    "⚡ 列出为以下feature构建Sprint 1架构文件所需的任务：{feature}\n"
    "Stack: {stack_name}。\n"
    "Sprint 1 = 仅架构：项目配置、构建文件、数据库配置。\n"
    "仅按以下精确格式输出行（无标题，无多余文字）：\n"
    "ID|title|type|sprint|target_file\n"
    "其中type须为：config\n"
    "其中sprint对所有任务都必须为1。\n"
    "其中target_file为精确文件名（例如{file_examples}）。\n"
    "{rules}"
    "最多{max_tasks}个任务。示例：\n"
    "{examples}"
    "现在仅为{feature}输出Sprint 1的行："
)


def _sprint1_config_family(stack: dict) -> str:
    """Which Sprint-1 configuration model does this stack use — "spring", "jakarta"
    (Java without Spring) or "generic" (everything else)?

    Three families, not two: python-fastapi.json declares no framework_family and
    framework "FastAPI", so _is_spring returns False for it — a bare `not _is_spring`
    split would hand FastAPI persistence.xml/web.xml instructions. Check _is_spring
    first (its default-True keeps under-specified stacks on the frozen Spring path),
    then narrow the remainder by language."""
    # Deferred import: sprint.py imports this module lazily in several places, so the
    # reciprocal must stay lazy too or the package cannot be imported.
    from agile_team_cline.sprint import _is_spring

    if _is_spring(stack):
        return "spring"
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    return "jakarta" if is_java else "generic"


def _build_sprint1_prompt(feature: str, stack: dict) -> str:
    """Render the Sprint-1 architect prompt for this stack's configuration family.
    Production and the tests both go through here, so a test cannot pass while the
    call site renders something else."""
    cfg = _SPRINT1_CONFIG[_sprint1_config_family(stack)]
    # Same AGILE_CHINESE gate picks the template and its matching rules block, so the two
    # can never end up in different languages.
    return zh_prompt(_ARCHITECT_SPRINT1_PROMPT, _ARCHITECT_SPRINT1_PROMPT_ZH).format(
        feature=feature,
        stack_name=stack["name"],
        max_tasks=MAX_TASKS_PER_SPRINT,
        file_examples=cfg["file_examples"],
        rules=zh_prompt(cfg["rules"], cfg["rules_zh"]),
        examples=cfg["examples"],
    )

# Per-sprint planner — context-aware, carries prior output forward
_NEXT_SPRINT_PROMPT = (
    "⚡ Plan Sprint {sprint_num} ({sprint_type}) for: {feature}\n"
    "Stack: {stack_name}.\n\n"
    "Files already built in previous sprints:\n{existing_files}\n\n"
    "Contract (entities and endpoints):\n{contract_summary}\n\n"
    "{controller_mapping}"
    "{qa_section}"
    "Output ONLY the next {max_tasks} tasks in this exact format:\n"
    "ID|title|type|sprint|target_file\n"
    "Where type is one of: entity, repository, service, controller, config, test\n"
    "Where sprint must be {sprint_num} for ALL tasks.\n"
    "Where target_file is the exact filename to create or update.\n"
    "RULES:\n"
    "- In Java, each public class, interface, or repository MUST be declared in its own individual file matching the class/interface name (e.g. User.java, Note.java, UserRepository.java, NoteRepository.java). Never group multiple public classes or repositories into a single file (like Entities.java or Repositories.java) as this violates Java syntax and causes compile failures.\n"
    "- Do NOT repeat files already built. Build on top of existing work.\n"
    "Output {sprint_num} tasks now:"
)

_NEXT_SPRINT_PROMPT_ZH = (
    "⚡ 为以下feature规划Sprint {sprint_num}（{sprint_type}）：{feature}\n"
    "Stack: {stack_name}。\n\n"
    "之前sprint已构建的文件：\n{existing_files}\n\n"
    "Contract（entities和endpoints）：\n{contract_summary}\n\n"
    "{controller_mapping}"
    "{qa_section}"
    "仅按以下精确格式输出接下来的{max_tasks}个任务：\n"
    "ID|title|type|sprint|target_file\n"
    "其中type须为：entity, repository, service, controller, config, test 之一\n"
    "其中所有任务的sprint必须为{sprint_num}。\n"
    "其中target_file为要创建或更新的精确文件名。\n"
    "规则：\n"
    "- 在Java中，每个public类、接口或repository必须声明在与类/接口名匹配的独立文件中（例如User.java, Note.java, UserRepository.java, NoteRepository.java）。绝不要把多个public类或repository合并到一个文件中（如Entities.java或Repositories.java），这违反Java语法并导致编译失败。\n"
    "- 不要重复已构建的文件。在已有工作基础上继续。\n"
    "现在输出{sprint_num}个任务："
)


def filter_tasks_for_brownfield(tasks: list, workspace_state: dict) -> list:
    """Remove tasks that would regenerate existing files in a brownfield project.

    For brownfield (existing workspace), don't regenerate scaffold or existing classes.
    Only generate NEW features for this milestone.
    """
    if not workspace_state.get("is_brownfield"):
        return tasks

    existing_classes = get_existing_class_names(workspace_state)

    filtered = []
    skipped = []

    for task in tasks:
        target = (task.get("target_file") or "").replace(".java", "")

        # Check if this class already exists
        if target in existing_classes:
            skipped.append(target)
            continue

        # Skip scaffold config files if brownfield
        if target in ("JpaConfig", "WebMvcConfig", "CacheConfiguration", "application.properties", "pom.xml"):
            skipped.append(target)
            continue

        filtered.append(task)

    if skipped:
        print(f"   ⏭️  Skipped {len(skipped)} existing class(es): {', '.join(skipped[:3])}")

    return filtered


def _merge_contracts(base: dict, addition: dict) -> dict:
    """Union two contracts: concatenate entities (normalize_contract dedups them by
    canonical name, unioning fields), dedup endpoints by (method, path), merge schemas.
    Lets a new feature reuse the existing entities (Pet, Visit) and add its own."""
    eps = list(base.get("endpoints", []) or []) + list(addition.get("endpoints", []) or [])
    seen, uniq = set(), []
    for ep in eps:
        k = (ep.get("method"), ep.get("path")) if isinstance(ep, dict) else (None, str(ep))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(ep)
    merged = {
        "entities": _entity_list(base) + _entity_list(addition),
        "endpoints": uniq,
        "schemas": {**(base.get("schemas") or {}), **(addition.get("schemas") or {})},
        "feature": addition.get("feature", base.get("feature", "")),
    }
    if base.get("database"):
        merged["database"] = base["database"]
    return merged


def _is_java_brownfield_scaffolded(output_dir: str) -> bool:
    """True when the workspace already has a Maven/Gradle build file AND a
    @SpringBootApplication main class on disk — i.e. the Sprint-1 scaffold
    (project config, build files, main application class) is already done, so
    the camel Sprint-1 task-list call would only re-plan work that already exists."""
    if not (os.path.exists(os.path.join(output_dir, "pom.xml"))
            or os.path.exists(os.path.join(output_dir, "build.gradle"))):
        return False
    src_root = os.path.join(output_dir, "src/main/java")
    if not os.path.isdir(src_root):
        return False
    for root, _, files in os.walk(src_root):
        for f in files:
            if f.endswith("Application.java"):
                return True
    return False


def run_sprint_planning(feature: str, stack: dict, existing_contract=None, prior_issues=None,
                        output_dir=None, arch_context=None) -> dict:
    """Plans Sprint 1 (architecture scaffold) only. Subsequent sprints planned on-the-fly.

    One LLM call: the architect emits the Sprint-1 task list directly. The
    earlier PO-ask + architect-warmup rounds were discarded anyway.

    `output_dir` enables the workspace-aware architect (cline plan mode, when
    AGILE_ARCHITECT_BACKEND=cline) and the deterministic existing-entity lock.

    `arch_context` is the ARCHITECTURE.md generated by Cline's read-only workspace analysis.
    If provided, it's included in the prompt so the architect understands existing patterns
    and can design compatible new components.
    """
    tokens = TASK_TOKEN_LIMITS["architect_planning"]
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"

    if is_java and output_dir and _is_java_brownfield_scaffolded(output_dir):
        # Brownfield Java: pom.xml + a @SpringBootApplication main class already exist on
        # disk, so the Sprint-1 scaffold (project config, build files, main app class) is
        # already done — the camel task-list call only re-planned already-existing files
        # (e.g. "Create Spring Boot main application class" for a repo that already has
        # PetClinicApplication.java). Sprint 1 then runs with an empty backlog: the
        # architect sprint self-skips when the contract is detailed, and the dev sprint
        # no-ops on an empty task list.
        print("   ⏭️  Skipping Sprint-1 scaffold planning (brownfield: pom.xml + *Application.java already exist)")
        backlog = []
    else:
        architect = make_agent("Software Architect", _ARCHITECT_GOAL, tokens, model=make_architect_model(tokens))

        # Include architecture context in prompt if available
        arch_note = ""
        if arch_context:
            arch_note = (
                "\n\n📋 EXISTING CODEBASE ARCHITECTURE:\n"
                "──────────────────────────────────────\n"
                f"{arch_context}\n"
                "──────────────────────────────────────\n"
                "^ Design new components following existing patterns (entity fields, service structure, annotations).\n"
                "^ Do NOT create conflicting configurations or duplicate classes.\n\n"
            )

        # arch_note stays a post-format concatenation: ARCHITECTURE.md is arbitrary text
        # and may contain braces that str.format would choke on.
        prompt = arch_note + _build_sprint1_prompt(feature, stack)
        resp = step_with_retry(architect, BaseMessage.make_user_message(role_name="User", content=prompt))

        raw = resp.msgs[0].content if resp.msgs else ""
        backlog = _parse_task_lines(raw)
        # Force all tasks to sprint 1 with globally unique IDs
        for t in backlog:
            t["sprint"] = 1
            if not t["id"].startswith("S1-"):
                t["id"] = f"S1-{t['id']}"
        backlog = backlog[:MAX_TASKS_PER_SPRINT]

    # Derive entities/endpoints from the feature description so developers
    # receive a real contract rather than the /health stub.
    # `existing_contract` may be `{}` (Blackboard.read returns {} when the file
    # is missing) — treat that as absent.
    if existing_contract:
        # AGILE_INCREMENTAL: derive the NEW feature's contract and union it with the
        # existing one so its entities get added (Pet/Visit reused, Vaccination/Invoice
        # added). Default (flag off) keeps the prior reuse-verbatim behaviour.
        if os.getenv("AGILE_INCREMENTAL") == "1":
            new_c = _architect_derive(feature, stack, tokens, output_dir, prior_issues) or _default_contract(feature)
            contract = _merge_contracts(existing_contract, new_c)
        else:
            contract = existing_contract
    else:
        contract = _architect_derive(feature, stack, tokens, output_dir) or _default_contract(feature)
    contract = normalize_contract(contract, stack, output_dir=output_dir)
    # Lock the contract to the real on-disk schema so the architect can't retype/drop
    # existing fields (the redesign drift the plan-mode probe exposed).
    if output_dir:
        contract = _lock_existing_entity_schemas(contract, output_dir)
        # Drift-control A: snap id/FK types onto on-disk truth + record _pinned_ids.
        contract = reconcile_contract_ids(contract, output_dir)
    # Completeness gate: every type an endpoint returns/accepts must have a class the pipeline
    # will create (a contract entity/DTO) or that already exists on disk. Otherwise the dev
    # sprint authors a controller referencing an undefined type and the module cannot compile
    # (silent until build). Detect deterministically and surface it on the contract so the
    # gap is dataset-visible instead of a mystery red build. (LLM re-derive of the missing
    # types is the follow-up remediation once model quota is available.)
    missing = _contract_missing_types(contract, output_dir)
    if missing:
        contract.setdefault("_warnings", []).append(
            "contract_incomplete: endpoints reference undefined type(s): "
            + ", ".join(sorted(missing)))
        print(f"   ⚠️  Contract completeness: endpoints reference undefined type(s) "
              f"with no entity/DTO/on-disk class: {sorted(missing)}")
    return {"contract": contract, "sprint_backlog": backlog,
            "incomplete_types": sorted(missing)}


def plan_next_sprint(feature: str, sprint_num: int, sprint_type: str,
                     blackboard, stack: dict, workspace_state: dict = None) -> list:
    """
    Plans a single sprint using the current blackboard state.
    Carries forward: generated files (with class summaries), contract, test report.
    """
    bb_data = blackboard.read("blackboard")
    existing_files = bb_data.get("generated_files", [])
    contract = blackboard.read("contract")
    test_report = blackboard.read("test_report") if sprint_type == "QA" else {}
    output_dir = str(blackboard.output_dir)

    # Build compact file summary (filename + first non-empty line = package/class declaration)
    file_summaries = []
    for fname in existing_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    head = ""
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("//") and not line.startswith("#"):
                            head = line[:80]
                            break
                file_summaries.append(f"  {fname}: {head}")
            except Exception:
                file_summaries.append(f"  {fname}")
        else:
            file_summaries.append(f"  {fname}")
    files_str = "\n".join(file_summaries) if file_summaries else "  (none yet)"

    # Compact contract summary — entity names + endpoints only
    entities = []
    if isinstance(contract, dict):
        entities_data = contract.get("entities", {})
        if isinstance(entities_data, dict):
            entities = list(entities_data.keys())
        elif isinstance(entities_data, list):
            entities = [e.get("name", str(e)) if isinstance(e, dict) else str(e) for e in entities_data]
            
    endpoints = [
        f"{ep.get('method','?')} {ep.get('path','?')}" if isinstance(ep, dict) else str(ep)
        for ep in (contract.get("endpoints", []) if isinstance(contract, dict) else [])
    ]
    contract_summary = (
        f"Entities: {', '.join(entities) or 'TBD'}\n"
        f"Endpoints: {', '.join(endpoints) or 'TBD'}"
    )

    # QA section: surface issues from test report
    qa_section = ""
    if sprint_type == "QA" and isinstance(test_report, dict):
        issues = test_report.get("issues", [])
        gaps = test_report.get("coverage_gaps", [])
        if issues or gaps:
            qa_section = (
                f"QA issues to cover:\n"
                f"{json.dumps(issues[:5], separators=(',',':'))}\n"
                f"Coverage gaps: {json.dumps(gaps[:3], separators=(',',':'))}\n\n"
            )

    # Authoritative controller partitioning logic
    controller_mapping = ""
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if is_java and isinstance(contract, dict) and entities:
        controllers = _resource_controllers(contract, entities)
        mappings = []
        for c in controllers:
            eps = _controller_endpoints(contract, c, entities)
            mappings.append(f"  - {c}Controller.java: must own endpoints [{', '.join(eps)}]")
        if mappings:
            controller_mapping = (
                "Authoritative Controller Partitioning Rules (nested sub-resources belong to the parent controller):\n"
                + "\n".join(mappings)
                + "\nDo NOT plan or create any controller other than the ones listed above. All endpoints must be implemented in their owning controllers.\n\n"
            )

    tokens = TASK_TOKEN_LIMITS["architect_planning"]
    planner = make_agent("Software Architect", _ARCHITECT_GOAL, tokens, model=make_architect_model(tokens))

    prompt = zh_prompt(_NEXT_SPRINT_PROMPT, _NEXT_SPRINT_PROMPT_ZH).format(
        sprint_num=sprint_num,
        sprint_type=sprint_type,
        feature=feature,
        stack_name=stack["name"],
        existing_files=files_str,
        contract_summary=contract_summary,
        controller_mapping=controller_mapping,
        qa_section=qa_section,
        max_tasks=MAX_TASKS_PER_SPRINT,
    )

    msg = BaseMessage.make_user_message(role_name="User", content=prompt)
    response = step_with_retry(planner, msg)
    raw = response.msgs[0].content if response.msgs else ""

    tasks = _parse_task_lines(raw)
    for t in tasks:
        t["sprint"] = sprint_num  # enforce correct sprint number
        if not t["id"].startswith(f"S{sprint_num}-"):
            t["id"] = f"S{sprint_num}-{t['id']}"

    # BROWNFIELD: Filter out tasks that would regenerate existing classes
    if workspace_state:
        tasks = filter_tasks_for_brownfield(tasks, workspace_state)

    return tasks[:MAX_TASKS_PER_SPRINT]


def _parse_task_lines(text: str) -> list:
    tasks = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        if line.lower().startswith("id|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        task_id, title, task_type, sprint_raw = parts[0], parts[1], parts[2], parts[3]
        target_file = parts[4] if len(parts) > 4 else None
        try:
            sprint = int(sprint_raw)
        except ValueError:
            sprint = 2
        tasks.append({
            "id": task_id,
            "title": title,
            "type": task_type.lower(),
            "sprint": sprint,
            "target_file": target_file,
        })
    return tasks


def _default_contract(feature: str) -> dict:
    return {
        "entities": [],
        "endpoints": [{"method": "GET", "path": "/health", "response": {"status": "string", "uptime": "string"}}],
        "schemas": {},
        "feature": feature,
    }


# Presentation tier (enhancement #2, FullStack-Agent shape). Appended to the architect prompt
# ONLY for non-Spring (jakarta) stacks when AGILE_PRESENTATION=1 — so the Spring/PetClinic
# prompt is byte-identical and pure-backend features still emit no views.
_VIEWS_GUIDANCE = (
    "\nPRESENTATION TIER: if (and ONLY if) the feature requires a web UI page/screen — a servlet "
    "that fetches data over REST and renders a JSP view — add a top-level \"views\" array to the "
    "JSON. Each view: {\"name\": \"ProductReviews\", \"route\": \"/product/reviews\", "
    "\"shows\": \"<entity or DTO rendered on the page>\", \"consumes\": \"<backend resource the "
    "servlet fetches over REST>\", \"jsp\": \"product_reviews.jsp\"}. A pure backend feature with no "
    "UI page must NOT include \"views\". Never put UI pages in \"endpoints\" — endpoints are "
    "REST/JAX-RS resources; views are servlet+JSP."
)


def _presentation_on(stack: dict) -> bool:
    """True when the presentation tier (servlet/JSP) is active: AGILE_PRESENTATION=1 AND a
    non-Spring (jakarta) stack. Keeps the frozen Spring baseline byte-identical."""
    from agile_team_cline.sprint import _is_spring
    return os.getenv("AGILE_PRESENTATION", "0") == "1" and not _is_spring(stack)


def _derive_contract(feature: str, stack: dict, tokens: int, prior_issues=None) -> dict:
    """Ask the architect for a real contract (entities + endpoints) from the feature."""
    architect = make_agent(
        "Software Architect",
        "Design a minimal JSON contract from a feature description. JSON only.",
        tokens,
        model=make_architect_model(tokens),
    )
    prompt = zh_prompt(_CONTRACT_PROMPT, _CONTRACT_PROMPT_ZH).format(feature=feature, stack_name=stack["name"])
    if prior_issues:
        # Fix-forward: let the architect see the previous feature's unresolved problems
        # so the new feature integrates cleanly and avoids repeating them.
        prompt += (
            "\n\nThe existing system (built by previous features) has these unresolved issues. "
            "Design the new feature to integrate with the existing entities and avoid repeating them:\n"
            + json.dumps(prior_issues[:6], separators=(",", ":"))
        )
    if _presentation_on(stack):
        prompt += _VIEWS_GUIDANCE
    try:
        resp = step_with_retry(architect, BaseMessage.make_user_message(role_name="User", content=prompt))
    except Exception as e:
        print(f"⚠️  Contract derivation failed: {e}. Falling back to default contract.")
        return None

    raw = resp.msgs[0].content if resp.msgs else ""
    contract = extract_json(raw)
    if not isinstance(contract, dict) or not contract:
        return None
    contract["feature"] = feature
    return contract


# System prompt for the workspace-aware (cline plan-mode) architect. The probe showed
# cline reads the files and stops hallucinating synonym entities, but its instinct is
# to REDESIGN the schema (retype ids, change path style) — so the rules below force it
# to conform, and the deterministic entity-lock below is the backstop.
_CLINE_ARCHITECT_SYSTEM = (
    "You are a software architect in plan mode: you read the existing source files in this project "
    "to learn the entities, their EXACT field names and types, and REST path conventions already in use. "
    "THEN design a minimal JSON contract for the requested feature. HARD RULES:\n"
    "- READ the existing code with your tools: search_codebase (regex search across the workspace), "
    "read_files (open files by path), and run_commands (read-only commands like `find`/`ls` work in "
    "this plan mode). If a search returns 0 results, retry with a broader or different query before "
    "concluding the workspace is empty. Do not call ask_question — there is no human to respond. You "
    "MUST actually open files (do not guess).\n"
    "- ALWAYS read the entity base class(es) (e.g. BaseEntity / NamedEntity in src/main/java/.../model) "
    "to learn the EXACT id Java type the project uses (e.g. Integer), and give new entities AND their "
    "scalar foreign keys that SAME id type. NEVER introduce UUID/Long if existing entities use Integer.\n"
    "- REUSE existing entities by their EXACT class name (if a Pet entity exists, use 'Pet' — never "
    "invent a synonym like 'Patient').\n"
    "- PRESERVE existing field names, field types, and endpoint path conventions EXACTLY. You may only "
    "ADD new entities, fields, or endpoints — never restyle, retype, or drop what already exists.\n"
    "- Reference other entities with scalar foreign-key fields (e.g. petId), NOT object references.\n"
    "- A computed/aggregated RESULT shaped for a response (a total, summary, report, count, projection) "
    "that is NOT stored in its own table is a response DTO, not an entity: leave it out of 'entities' or "
    "mark it {\"name\": \"...\", \"dto\": true}; never give it a table. If a type appears ONLY as an endpoint "
    "'response' (never written by POST/PUT/PATCH, never referenced as a field by another entity), it MUST "
    "be marked \"dto\": true — regardless of its name.\n"
    "Output ONLY one JSON object (no prose, no markdown) with this exact shape (the id type shown is an "
    "EXAMPLE — use whatever the existing entities actually use):\n"
    "{\"entities\": [{\"name\": \"Pet\", \"fields\": {\"id\": \"Integer\", \"name\": \"string\"}}], "
    "\"endpoints\": [{\"method\": \"GET\", \"path\": \"/pets\", \"response\": \"Pet\"}], \"schemas\": {}}"
)


def _derive_contract_cline(feature: str, stack: dict, output_dir: str, prior_issues=None) -> dict:
    """Workspace-aware architect: a cline PLAN-mode session reads the existing code and
    derives the contract, so it reuses on-disk entities instead of hallucinating
    synonyms. Returns None on failure (caller falls back to the camel architect)."""
    from agile_team_cline.cline_runner import run_cline_with_retry
    # AGILE_ARCHITECT_THINKING: "0"/unset → none; "1" → xhigh (maps to
    # reasoning_effort=max); or an explicit cline level (none|low|medium|high|xhigh)
    # passed through verbatim. Was previously read but hardcoded off at the call sites.
    # Longer timeout ensures plan sessions have time to handle retries if tool format issues occur.
    _raw_thinking = os.getenv("AGILE_ARCHITECT_THINKING", "0")
    thinking = {"0": "none", "1": "xhigh"}.get(_raw_thinking, _raw_thinking)
    timeout = int(os.getenv("AGILE_ARCHITECT_CLINE_TIMEOUT", "1800"))
    brief = (f"Design a JSON contract for this feature: {feature}\n"
             f"Stack: {stack.get('name','')}. Read the existing entities/controllers already in this "
             f"project and reuse them by exact name; only add what the feature needs. Keep it small "
             f"(<=8 entities). IMPORTANT: match the EXISTING model's id strategy — inspect the base "
             f"class / existing entities (e.g. BaseEntity) and give new entities AND their foreign-key "
             f"fields the SAME id type the project already uses (do not introduce UUID if the project "
             f"uses Integer/Long, or vice-versa). Output ONLY the JSON contract object.")
    if prior_issues:
        brief += ("\nKnown unresolved issues to integrate with cleanly (do not repeat):\n"
                  + json.dumps(prior_issues[:6], separators=(",", ":")))
    # RAG grounding (opt-in): inject the on-disk entity inventory so the architect reuses
    # existing entities/ids instead of inventing synonyms. Default off → brief byte-identical
    # (Spring baseline unaffected); the TeaStore runs set AGILE_ARCHITECT_INVENTORY=1.
    if os.getenv("AGILE_ARCHITECT_INVENTORY", "0") == "1":
        brief += _render_existing_inventory(output_dir)
    if _presentation_on(stack):
        brief += _VIEWS_GUIDANCE
    res = run_cline_with_retry(output_dir, brief, plan=True, thinking=thinking,
                               system=_CLINE_ARCHITECT_SYSTEM, timeout=timeout, title="architect")
    if not res.success or not res.text:
        print("⚠️  cline plan-mode architect failed; falling back to camel architect.")
        return None
    contract = extract_json(res.text)
    if not isinstance(contract, dict) or not contract:
        print(f"⚠️  cline output (first 300 chars): {repr(res.text[:300])}")
        # The plan-mode session usually DID the analysis (it prints "BaseEntity uses
        # Integer..." etc.) but emitted a prose summary instead of the JSON contract.
        # Falling straight back to the camel architect produces a DIFFERENT decomposition
        # (breaks EN/CN comparability and triggers extra coverage sprints), so first give
        # cline one more turn: feed back its own analysis and demand JSON only.
        if os.getenv("AGILE_ARCHITECT_JSON_RETRY", "1") == "1":
            print("⚠️  cline architect output not parseable as JSON; retrying for JSON-only output...")
            retry_brief = (
                "You already analyzed this project's existing code and produced this analysis:\n\n"
                f"{res.text.strip()[:4000]}\n\n"
                f"Now output ONLY the JSON contract object for this feature — no prose, no markdown, "
                f"no explanation, just the JSON object. Feature: {feature}\n"
                "Required shape: {\"entities\": [{\"name\": \"Pet\", \"fields\": {\"id\": \"Integer\", "
                "\"name\": \"string\"}}], \"endpoints\": [{\"method\": \"GET\", \"path\": \"/pets\", "
                "\"response\": \"Pet\"}], \"schemas\": {}}\n"
                "Match the existing model's id type EXACTLY as you found it above."
            )
            res2 = run_cline_with_retry(output_dir, retry_brief, plan=True, thinking=thinking,
                                        system=_CLINE_ARCHITECT_SYSTEM, timeout=timeout,
                                        title="architect-json-retry")
            if res2.success and res2.text:
                contract = extract_json(res2.text)
        if not isinstance(contract, dict) or not contract:
            print("⚠️  cline architect output still not parseable as JSON; falling back to camel architect.")
            return None
    contract["feature"] = feature
    return contract


def _architect_derive(feature: str, stack: dict, tokens: int, output_dir=None, prior_issues=None) -> dict:
    """Pick the architect backend: cline plan-mode (workspace-aware) when
    AGILE_ARCHITECT_BACKEND=cline and a workspace exists, else the camel model.
    Cline failures fall back to camel so derivation never hard-fails."""
    if os.getenv("AGILE_ARCHITECT_BACKEND", "camel").lower() == "cline" and output_dir:
        c = _derive_contract_cline(feature, stack, output_dir, prior_issues=prior_issues)
        if c:
            return c
    return _derive_contract(feature, stack, tokens, prior_issues=prior_issues)


# Java declared-type → contract logical type, for locking the contract to on-disk code.
_JAVA_TO_LOGICAL = {
    "String": "string", "UUID": "UUID", "Long": "Long", "long": "Long",
    "Integer": "number", "int": "number", "BigDecimal": "number",
    "Double": "number", "double": "number", "Float": "number", "float": "number",
    "Boolean": "boolean", "boolean": "boolean",
    "LocalDate": "string", "LocalDateTime": "string", "Instant": "string",
}


def _existing_entities_on_disk(output_dir: str) -> dict:
    """Map EntityName -> {field: logical_type} for @Entity classes already on disk
    (parses `private <Type> <field>;`). The truth the contract is locked to."""
    out = {}
    root = os.path.join(output_dir, "src", "main", "java")
    walk = root if os.path.isdir(root) else output_dir
    for dp, dirs, files in os.walk(walk):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".java"):
                continue
            try:
                src = open(os.path.join(dp, fn)).read()
            except Exception:
                continue
            if "@Entity" not in src:
                continue
            m = re.search(r"\bclass\s+(\w+)", src)
            if not m:
                continue
            fields = {}
            for ftype, fname in re.findall(r"private\s+([\w<>.]+)\s+(\w+)\s*;", src):
                base = ftype.split("<")[0].split(".")[-1]
                fields[fname] = _JAVA_TO_LOGICAL.get(base, base)
            if fields:
                out[m.group(1)] = fields
    return out


def _on_disk_type_names(output_dir: str) -> set:
    """Lowercased class names (`.java` file stems) already present on disk — the cheap,
    parse-free 'does this type already exist' signal used by _merge_ordinal_twin_entities to
    tell a positional-alias phantom (FirstProduct beside the baseline's Product) from a
    genuinely new entity. Covers ALL on-disk types (transport POJOs, @Entity classes, DTOs),
    not just @Entity ones, since the ordinal base often names a plain transport type."""
    names = set()
    if not output_dir or not os.path.isdir(output_dir):
        return names
    for dp, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if fn.endswith(".java") and fn != "package-info.java":
                names.add(fn[:-len(".java")].lower())
    return names


def _lock_existing_entity_schemas(contract: dict, output_dir: str) -> dict:
    """For any contract entity that already exists on disk, force its existing field
    NAMES and TYPES to match the real code (on-disk wins); the contract may only ADD
    new fields. Also infers the ID generation strategy from existing entities and applies
    it to new entities. Neutralizes the architect 'redesign drift' the probe exposed.
    Expects canonical list-form entities (run after normalize_contract). Idempotent."""
    if not isinstance(contract, dict):
        return contract
    on_disk = _existing_entities_on_disk(output_dir)

    # Lock existing entities to on-disk schema
    by_lower = {k.lower(): v for k, v in on_disk.items()} if on_disk else {}
    for e in contract.get("entities", []):
        if not isinstance(e, dict):
            continue
        locked = by_lower.get(str(e.get("name", "")).lower())
        if not locked:
            continue
        fields = e.get("fields") if isinstance(e.get("fields"), dict) else {}
        merged = dict(locked)  # on-disk names+types win
        for k, v in fields.items():  # keep genuinely new contract fields
            if k not in merged:
                merged[k] = v
        e["fields"] = merged

    # NOTE: id-strategy is intentionally NOT stamped here. The model owns id-type
    # decisions (it reads the existing ARCHITECTURE.md / entities and chooses to
    # match them). Forcing a strategy here is what produced the UUID-vs-Integer
    # drift; existing-entity FIELD locking above is kept, ids are left to the model.
    return contract


def _entities_on_disk_resolved(output_dir: str) -> dict:
    """Map EntityName -> {field: RAW Java type} for @Entity classes on disk, with the
    `extends` chain resolved through @MappedSuperclass bases (Pet → NamedEntity →
    BaseEntity gives Pet an `Integer id`). Unlike _existing_entities_on_disk this
    keeps exact Java types (Integer vs Long matters for id reconciliation) and the
    class regex is anchored to the declaration line so javadoc prose ("base class
    for…") is never parsed as a class."""
    classes = {}   # name -> {"fields": {fname: JavaType}, "extends": parent|None, "entity": bool}
    root = os.path.join(output_dir, "src", "main", "java")
    walk = root if os.path.isdir(root) else output_dir
    decl_re = re.compile(
        r"^[ \t]*(?:public\s+|protected\s+|abstract\s+|final\s+|static\s+)*class\s+(\w+)"
        r"(?:<[^>\n]*>)?(?:\s+extends\s+([\w.]+))?", re.M)
    field_re = re.compile(r"(?:private|protected)\s+([\w<>.]+)\s+(\w+)\s*;")
    for dp, dirs, files in os.walk(walk):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".java"):
                continue
            try:
                src = open(os.path.join(dp, fn)).read()
            except Exception:
                continue
            is_entity = "@Entity" in src
            if not is_entity and "@MappedSuperclass" not in src:
                continue
            m = decl_re.search(src)
            if not m:
                continue
            fields = {}
            for ftype, fname in field_re.findall(src):
                fields[fname] = ftype.split("<")[0].split(".")[-1]
            classes[m.group(1)] = {
                "fields": fields,
                "extends": (m.group(2) or "").split(".")[-1] or None,
                "entity": is_entity,
            }
    out = {}
    for name, info in classes.items():
        if not info["entity"]:
            continue
        merged, cur, hops = {}, name, 0
        while cur in classes and hops < 8:  # child fields win over inherited ones
            for f, t in classes[cur]["fields"].items():
                merged.setdefault(f, t)
            cur = classes[cur]["extends"]
            hops += 1
            if not cur:
                break
        if merged:
            out[name] = merged
    return out


def _render_existing_inventory(output_dir: str) -> str:
    """RAG-style grounding block (ISSTA-2025 lightweight RAG / De-Hallucinator): a
    deterministic inventory of the @Entity classes already on disk — exact name, id type, and
    fields (inheritance resolved) — injected into the architect brief so it REUSES existing
    entities instead of inventing synonyms or re-typing ids, without relying on the flaky
    plan-mode read. Returns '' when nothing is on disk (greenfield) so the brief is unchanged.
    Read-only; reuses _entities_on_disk_resolved (the same on-disk truth the gates use)."""
    inv = _entities_on_disk_resolved(output_dir)
    if not inv:
        return ""
    lines = []
    for name in sorted(inv):
        fields = inv[name]
        id_t = fields.get("id", "?")
        body = ", ".join(f"{f}:{t}" for f, t in fields.items())
        lines.append(f"- {name} (id: {id_t}) — fields: {body}")
    return ("\n\nEXISTING ENTITIES ALREADY ON DISK (reuse by EXACT name and id type; do NOT "
            "re-create, rename, or invent a synonym for any of these):\n" + "\n".join(lines))


def _extract_conventions(output_dir: str) -> dict:
    """Drift-control mechanism E (DRIFT_CONTROL_PLAN.md): a deterministic, no-LLM
    snapshot of conventions actually realized on disk at a green checkpoint —
    id types, package layout, and the live route map. Persisted as
    conventions.json and fed into the next milestone's brief, replacing the
    flaky per-milestone ARCHITECTURE.md rebuild and the `_inferred_patterns`
    guess. Only observed/compiled facts are recorded — never model choices."""
    import os
    from agile_team_cline.sprint import _controller_mappings

    on_disk = _entities_on_disk_resolved(output_dir)
    id_types = {name: fields["id"] for name, fields in on_disk.items() if fields.get("id")}

    packages = set()
    root = os.path.join(output_dir, "src", "main", "java")
    if os.path.isdir(root):
        for dirpath, _dirs, files in os.walk(root):
            _dirs[:] = [d for d in _dirs if d not in ("target", ".git") and not d.startswith(".")]
            if any(f.endswith(".java") for f in files):
                pkg = os.path.relpath(dirpath, root).replace(os.sep, ".")
                if pkg != ".":
                    packages.add(pkg)

    routes = sorted(f"{method} {path}" for method, path in _controller_mappings(output_dir))

    return {"id_types": id_types, "packages": sorted(packages), "routes": routes}


# Exact Java id types we trust as pins; model-chosen logical types outside this set
# (e.g. "number") are ambiguous and are not pinned.
_PINNABLE_ID_TYPES = {"Integer", "Long", "UUID", "String"}


def reconcile_contract_ids(contract: dict, output_dir: str) -> dict:
    """Drift-control mechanism A (see DRIFT_CONTROL_PLAN.md): one-time deterministic
    reconciliation of the contract against on-disk truth, run right after derivation +
    entity-locking. Existing entities get their `id` snapped to the real (inheritance-
    resolved) Java type; every `*Id` FK pointing at an existing entity snaps to that
    entity's id type. New entities keep the model's choice. Pins land in
    contract['_pinned_ids'] (exact Java types) for the brief, the skeleton emitter and
    the gates. Idempotent; every snap is logged as `contract_reconciliation:`."""
    if not isinstance(contract, dict) or not output_dir:
        return contract
    on_disk = _entities_on_disk_resolved(output_dir)
    disk_ids = {n: f["id"] for n, f in on_disk.items() if f.get("id")}
    if not disk_ids:
        return contract

    id_by_lower = {n.lower(): t for n, t in disk_ids.items()}

    # The project's dominant id type (e.g. Integer). New entities are NOT on disk, so
    # the existing-entity reconciliation below can't snap them — without this a new
    # entity's own primary key drifts from convention (e.g. M6 Prescription got UUID
    # while every other entity used Integer, a latent, untested type inconsistency).
    snap_new_pk = os.getenv("AGILE_RECONCILE_NEW_PK", "1") == "1"
    from collections import Counter
    dominant_id = Counter(disk_ids.values()).most_common(1)[0][0] if disk_ids else None

    def _ref_id_type(fk: str, fields: dict):
        # petId -> Pet.id. Prefer the contract's own relationship field naming the
        # target entity (e.g. `type: "PetType"` alongside `typeId`) — this is the
        # only reliable signal when the FK's name-stem ("type") doesn't textually
        # relate to the target entity's name ("PetType"). Falls back to an exact
        # entity-name match, then loose prefix/suffix (vetId ↔ Veterinarian); the
        # loose fallback is last because it can match an unrelated entity that
        # happens to share a substring (e.g. a stray "Type" entity).
        stem = fk[:-2]
        sib = fields.get(stem) or fields.get(stem[:1].lower() + stem[1:])
        if isinstance(sib, str) and sib.lower() in id_by_lower:
            return id_by_lower[sib.lower()]
        ref = stem.lower()
        if ref in id_by_lower:
            return id_by_lower[ref]
        for k, t in id_by_lower.items():
            if k.startswith(ref) or ref.startswith(k):
                return t
        return None

    entities = contract.get("entities")
    if isinstance(entities, list):
        for e in entities:
            if not isinstance(e, dict) or not isinstance(e.get("fields"), dict):
                continue
            name, fields = str(e.get("name", "")), e["fields"]
            disk_id = id_by_lower.get(name.lower())
            if disk_id and fields.get("id") != disk_id:
                print(f"contract_reconciliation: {name}.id {fields.get('id')!r} → {disk_id} (on-disk)")
                fields["id"] = disk_id
            elif (not disk_id and snap_new_pk and dominant_id
                  and fields.get("id") and fields.get("id") != dominant_id):
                print(f"contract_reconciliation: {name}.id {fields.get('id')!r} → {dominant_id} "
                      f"(new entity → project convention)")
                fields["id"] = dominant_id
            for fk in [f for f in fields if f != "id" and f.lower().endswith("id") and len(f) > 2]:
                want = _ref_id_type(fk, fields)
                if want and fields.get(fk) != want:
                    print(f"contract_reconciliation: {name}.{fk} {fields.get(fk)!r} → {want} (FK → on-disk id)")
                    fields[fk] = want

    # Pin map: on-disk entities (truth) + new contract entities whose chosen id is
    # already an exact Java type. Disk wins on collision.
    pinned = {}
    if isinstance(entities, list):
        for e in entities:
            if isinstance(e, dict) and isinstance(e.get("fields"), dict):
                t = e["fields"].get("id")
                if e.get("name") and t in _PINNABLE_ID_TYPES:
                    pinned[e["name"]] = t
    pinned.update(disk_ids)
    contract["_pinned_ids"] = dict(sorted(pinned.items()))
    return contract


def _backfill_relationship_entities(contract: dict, on_disk: set = None) -> None:
    """Deterministic guarantee: if an entity declares a `<X>Id` foreign key but the
    `X` entity is missing (the LLM added the link but dropped the parent — e.g. Pet
    has ownerId but no Owner entity), synthesize a minimal `X` entity. Keeps the
    domain model whole without depending on the derivation prompt alone.

    `on_disk` (2026-07-04): names of persisted classes that already exist in the
    on-disk codebase (from _existing_entities_on_disk(output_dir).keys(), which follow
    TeaStore's `Persistence<X>` naming convention). Checked IN ADDITION to the contract's
    own entity list before synthesizing a stub — a scalar FK to a genuinely-existing
    entity the model correctly did NOT redefine (StockMovement.productId -> the real
    PersistenceProduct, which the architect explicitly reasoned NOT to redefine because
    the instructions say "only ADD new entities") was previously mistaken for one the
    model forgot to define, producing a phantom duplicate `Product` entity with its own
    table (also the root cause of the ProductA/ProductB phantoms in the copurchase run:
    fields productAId/productBId triggered the same blind spot). Brownfield-only guard;
    greenfield behavior (on_disk empty/None, e.g. PetClinic Pet.ownerId -> Owner) is
    unchanged."""
    on_disk_lower = {str(n).lower() for n in (on_disk or ())}
    data = contract.get("entities")
    if isinstance(data, dict):
        existing = {k.lower(): k for k in data}
        def fields_of(v): return v if isinstance(v, dict) else {}
        items = list(data.items())
    elif isinstance(data, list):
        existing = {}
        for e in data:
            if isinstance(e, dict) and e.get("name"):
                existing[e["name"].lower()] = e["name"]
        def fields_of(e): return e.get("fields", {}) if isinstance(e, dict) else {}
        items = [(e.get("name", ""), e) for e in data if isinstance(e, dict)]
    else:
        return

    referenced = set()
    for _name, val in items:
        for fld in fields_of(val):
            if fld.lower().endswith("id") and len(fld) > 2 and fld.lower() != "id":
                referenced.add(fld[:-2])  # ownerId -> owner

    def _on_disk_match(r: str) -> bool:
        # Same prefix/suffix matching, PLUS a plain suffix match so "product" (from
        # productId) resolves against "persistenceproduct" — the Persistence<X> prefix
        # means the bare name is a SUFFIX of the disk class, not a prefix.
        return any(r == k or k.startswith(r) or r.startswith(k) or k.endswith(r)
                   for k in on_disk_lower)

    def already_present(ref: str) -> bool:
        r = ref.lower()
        # Exact, or abbreviation/prefix either way (vetId ↔ Veterinarian, ownerId ↔ Owner).
        if any(r == k or k.startswith(r) or r.startswith(k) for k in existing) or _on_disk_match(r):
            return True
        # A trailing single differentiator letter (productAId/productBId -> "producta"/
        # "productb" — the real copurchase run's phantom-entity trigger, two references
        # to ONE existing entity disambiguated A/B) still names the SAME on-disk entity
        # once that one letter is stripped. Guarded to len>3 so short real names aren't
        # over-stripped (e.g. a genuine "Ta" wouldn't shrink to the single letter "T").
        if len(r) > 3 and r[-1].isalpha():
            return _on_disk_match(r[:-1])
        return False

    for ref in referenced:
        if ref and not already_present(ref):
            target = ref[:1].upper() + ref[1:]
            stub = {"id": "UUID", "name": "string"}
            if isinstance(data, dict):
                data[target] = stub
            else:
                data.append({"name": target, "fields": stub})
            existing[ref.lower()] = target


def entity_names(contract: dict) -> list:
    if not isinstance(contract, dict):
        return []
    data = contract.get("entities", [])
    if isinstance(data, dict):
        return list(data.keys())
    if isinstance(data, list):
        return [e.get("name", "") if isinstance(e, dict) else str(e) for e in data]
    return []


# A computed/response value object (a total, summary, report, projection, …) that is
# NOT stored in its own table is a response DTO, not a persisted JPA entity. When the
# architect lists one in `entities`, the planner would otherwise emit @Entity + @Table +
# repository + CREATE TABLE for a thing that is never stored (TeaStore run #15:
# `ProductSalesSummary` → phantom `product_sales_summary` table + repo; run 2026-06-20:
# `CategoryProductCount` → phantom `category_product_count` table + SCHEMA-DDL dispute).
# These names are emitted as plain POJOs instead. The suffix list deliberately excludes
# anything a real domain entity is named (Product/Order/User/Review/Wishlist/Invoice/Visit/…
# never match; `Count` is case-sensitive so Account/Discount do NOT match).
# Trailing `s?` on the plural-only stems (Stat/Statistic/Metric/Total/Count) is REQUIRED: normalize_contract
# runs _singularize first, so an architect-named "...Statistics" reaches _dto_names as the
# SINGULAR "...Statistic" — without the optional s it would miss and the result gets modeled as a
# persistent @Entity (sales-inv test, 2026-06-16). The other stems already have singular forms.
# Entry/Group/Row/Node = nested response-component names an architect emits for a multi-level
# aggregate (e.g. related-breakdown's RelatedProductEntry / RelatedCategoryGroup, run #35,
# 2026-06-23): they are computed rows of a parent DTO, never persisted, but got modeled as
# @Entity @Table -> dead CREATE TABLE across 3 dialects + a phantom join table + a SCHEMA-DDL
# dispute that thrashed 3 refinement rounds. NOTE: `Item` is deliberately EXCLUDED — OrderItem /
# PersistenceOrderItem is a REAL persistent TeaStore entity; matching it would suppress its table.
_DTO_NAME_RE = re.compile(
    r"(Summary|Result|Report|Response|Projection|Aggregate|Stats?|Statistics?|Metrics?|"
    r"View|Dto|DTO|Totals?|Counts?|Breakdown|Snapshot|Overview|Entry|Group|Row|Node)$")

# Lifecycle/audit metadata field-name signal (item-6 corroboration guard, see
# _structural_dto_names): a field literally named type/status/state, or ending in "At"
# (createdAt/reportedAt/requestedAt/joinedAt/filedAt — the dominant Java timestamp-field
# convention in these briefs), marks its owner as a standalone persisted domain record
# rather than a computed rollup.
_LIFECYCLE_FIELD_RE = re.compile(r"(type|status|state|at)$", re.IGNORECASE)


def _structural_dto_names(items: list, contract: dict) -> set:
    """Names that are STRUCTURALLY response-only projections — a name-independent backstop
    for the suffix regex (runs #47-51: the architect set no `dto` flag and the projection
    names ProductCoPurchase / ChurnRiskUser / ProductPriceVolatility / NearDuplicatePair match
    NO suffix, so they were modeled as persistent @Entity + dead CREATE TABLE; `ChurnRiskUser`
    even ends in `User`, a real entity, so suffix matching is unfixable here). This is the
    paper-[4] deterministic-structural approach: an entity E is a computed DTO when ALL hold —
      (1) E appears as (or inside) some endpoint's `response` type (List<E>, Set<E>, E),
      (2) no OTHER entity references E (no field typed E, no `<e>Id` scalar FK pointing at E),
      (3) no write endpoint (POST/PUT/PATCH) targets E (by response or request body).
    The discriminator is (1): a genuinely-new PERSISTENT entity used only internally by the
    query (e.g. volatility's PriceHistory) never appears as a response type, so it is NEVER
    reclassified and keeps its table. Persistence-prefixed names (TeaStore's persistent-entity
    convention) are excluded outright. Contract-only; the caller still lets an explicit
    `dto: false` / `persistent: true` flag override."""
    if not isinstance(contract, dict):
        return set()
    endpoints = contract.get("endpoints") or []
    if not isinstance(endpoints, list) or not endpoints:
        return set()
    names = [str(e.get("name", "")).strip() for e in items]
    names = [n for n in names if n]
    if not names:
        return set()
    _word = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

    def _field_pairs(e):
        f = e.get("fields")
        if isinstance(f, dict):
            return [(str(k), str(v)) for k, v in f.items()]
        if isinstance(f, list):
            pairs = []
            for it in f:
                if isinstance(it, dict):
                    pairs.append((str(it.get("name", "")), str(it.get("type", ""))))
                else:
                    pairs.append((str(it), ""))
            return pairs
        return []

    def _has_lifecycle_field(e: dict) -> bool:
        for fname, _ftype in _field_pairs(e):
            if _LIFECYCLE_FIELD_RE.search(fname or ""):
                return True
        return False

    appears_in_response, appears_in_request, written = set(), set(), set()
    for ep in endpoints:
        if not isinstance(ep, dict):
            continue
        method = str(ep.get("method", "")).strip().upper()
        resp_toks = set(_word.findall(str(ep.get("response", "") or "")))
        body_toks = set(_word.findall(
            str(ep.get("request", "") or ep.get("body", "") or "")))
        for n in names:
            if n in resp_toks:
                appears_in_response.add(n)
            if n in body_toks:
                appears_in_request.add(n)
            if method in ("POST", "PUT", "PATCH") and (n in body_toks or n in resp_toks):
                written.add(n)

    referenced = set()
    for e in items:
        owner = str(e.get("name", "")).strip()
        for fname, ftype in _field_pairs(e):
            ftoks = set(_word.findall(ftype))
            flow = fname.lower()
            for n in names:
                if n == owner:
                    continue
                if n in ftoks or flow == n.lower() + "id":
                    referenced.add(n)

    owners = {str(o).lower() for o in _resource_controllers(contract, names) if o}
    by_name = {str(e.get("name", "")).strip(): e for e in items}

    out = set()
    for n in names:
        if n.startswith("Persistence"):
            continue  # project's persistent-entity naming convention — never a DTO
        # (a) response-only projection DTO: returned by an endpoint, never stored/referenced.
        # Corroboration guard (item-6, 2026-07-02): criteria (1)(2)(3) alone also match a
        # genuinely-persisted record exposed ONLY via GET and never referenced elsewhere
        # (e.g. PointsTransaction: id/accountId/orderId/delta/type/createdAt) — demoting it
        # drops its repository -> compile-fatal red build. A computed/aggregate projection
        # (the validated adversarial set: ChurnRiskUser/ProductCoPurchase/
        # ProductPriceVolatility/NearDuplicatePair) carries no lifecycle/audit metadata of
        # its own; a standalone domain record does — a `type`/`status`/`state` field, or a
        # `...At` timestamp field (createdAt/reportedAt/requestedAt/joinedAt/filedAt is the
        # dominant convention across the TrainTicket briefs). A false DEMOTE is fatal
        # (missing repository); a false KEEP is merely a dead table (survivable) — so bias
        # toward keeping: block demotion when this signal is present, even though it may
        # occasionally over-protect a projection that happens to carry e.g. a
        # "generatedAt" timestamp (a false KEEP, the safe failure mode).
        if (n in appears_in_response and n not in referenced and n not in written
                and not _has_lifecycle_field(by_name.get(n, {}))):
            out.add(n)
            continue
        # (b) request-only input DTO: a type used SOLELY as an HTTP request body (e.g.
        # EarnRequest, GroupOrderRequest) — never returned, never referenced by another
        # entity, and not itself a REST resource owner. The architect keeps naming these in
        # `entities`; modeling them as @Entity + @Table (the pre-fix behaviour) produced
        # phantom tables and broke compiles. A persisted record that merely happens to be
        # named "…Request" (e.g. UpgradeRequest, which is also RETURNED) appears in a
        # response, so it fails this test and is correctly kept as an entity.
        if (n in appears_in_request and n not in appears_in_response
                and n not in referenced and n.lower() not in owners):
            out.add(n)
    return out


def _dto_names(contract: dict) -> set:
    """Set of contract entity names that are computed/response DTOs, not persistent
    entities — by an explicit flag (`dto: true` / `persistent: false`), failing that a
    name-suffix heuristic, failing that a deterministic structural check (response-only,
    referenced by nobody, never written). An explicit `dto: false` / `persistent: true`
    keeps it an entity even if a heuristic matches (the flag is authoritative)."""
    data = contract.get("entities", []) if isinstance(contract, dict) else []
    if isinstance(data, dict):
        items = [{"name": k} for k in data.keys()]
    elif isinstance(data, list):
        items = [e if isinstance(e, dict) else {"name": str(e)} for e in data]
    else:
        items = []
    structural = _structural_dto_names(items, contract)
    out = set()
    for e in items:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        if e.get("dto") is False or e.get("persistent") is True:
            continue  # explicitly persistent — never treat as a DTO
        if (e.get("dto") or e.get("persistent") is False
                or _DTO_NAME_RE.search(name) or name in structural):
            out.add(name)
    return out


# --- Contract normalization (deterministic IR pass) -----------------------------
_SINGULAR_KEEP = ("ss", "us", "is", "as", "os")  # status, address, basis, ...


def _singularize(word: str) -> str:
    w = (word or "").strip()
    lw = w.lower()
    if len(w) <= 3:
        return w
    if lw.endswith("ies"):
        return w[:-3] + "y"
    if lw.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if lw.endswith("s") and not lw.endswith(_SINGULAR_KEEP):
        return w[:-1]
    return w


def _snake(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name or "").lower()
    return re.sub(r"__+", "_", s).strip("_")


def _pascal(name: str) -> str:
    n = (name or "").strip()
    return n[:1].upper() + n[1:] if n else n


def _entity_list(contract: dict) -> list:
    """Coerce the entities field (dict-form, list-of-dicts, or list-of-strings)
    to a uniform mutable list of {name, fields(dict)[, table][, dto][, persistent]} dicts.

    Carries `dto`/`persistent` through untouched (2026-07-04): these are the model's own
    explicit signal that an entity is a computed response value, not a persisted record
    (see _dto_names). Dropping them here — the very first coercion step normalize_contract
    runs — meant a correct `"dto": true` from the architect was silently destroyed before
    _dto_names ever got a chance to honor it, so a genuinely-computed value (TeaStore
    stockledger probe's LowStockEntry) still got an id+table injected downstream."""
    data = contract.get("entities", [])
    out = []
    if isinstance(data, dict):
        for k, v in data.items():
            out.append({"name": str(k), "fields": v if isinstance(v, dict) else {}})
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and e.get("name"):
                ent = {"name": str(e["name"]),
                       "fields": e["fields"] if isinstance(e.get("fields"), dict) else {}}
                if e.get("table"):
                    ent["table"] = e["table"]
                for k in ("dto", "persistent"):
                    if k in e:
                        ent[k] = e[k]
                out.append(ent)
            elif isinstance(e, str):
                out.append({"name": e, "fields": {}})
    return out


def _entity_field_names(e: dict) -> set:
    """Lowercased non-id field names of an entity entry."""
    fields = e.get("fields") if isinstance(e, dict) else None
    if not isinstance(fields, dict):
        return set()
    return {str(k).lower() for k in fields.keys() if str(k).lower() != "id"}


def _base_type(s: str) -> str:
    """Lowercased element type, stripped of collection wrappers/punctuation:
    'List<Product>'/'Product[]' -> 'product'. NOT singularized (a DTO legitimately named
    'ProductSales' must stay 'productsales', or it would falsely match a phantom 'ProductSale')."""
    s = re.sub(r"^\s*(list|set|collection|array|page|iterable)\s*<", "", str(s or ""), flags=re.I)
    s = s.strip().split("<")[0].split("[")[0].strip().strip("<>[]{}() ")
    return s.lower()


def _name_variant(a: str, b: str) -> bool:
    """True if names a, b are near-variants — purely structural, no hardcoded names:
    a plural of the other (foo/foos) or one a suffix of the other with the shorter ≥4 chars
    (Type/PetType). Catches the hallucinated-duplicate pair without a name list."""
    al, bl = a.lower(), b.lower()
    if al == bl:
        return False
    if al + "s" == bl or bl + "s" == al:
        return True
    short, long = (al, bl) if len(al) <= len(bl) else (bl, al)
    return len(short) >= 4 and long.endswith(short)


def _collect_referenced_types(contract: dict) -> set:
    """Lowercased names of types the contract actually USES: endpoint response/request/body
    values, endpoint path owners, and FK targets (a `<x>Id` field, or a field typed as an
    entity). An entity referenced by none of these is an orphan."""
    refs = set()
    names = [n for n in entity_names(contract) if n]
    name_set = {n.lower() for n in names}
    eps = contract.get("endpoints", []) if isinstance(contract, dict) else []
    for ep in eps:
        if not isinstance(ep, dict):
            continue
        for k in ("response", "request", "body", "result", "returns", "dto", "schema"):
            v = ep.get(k)
            if isinstance(v, str) and v.strip():
                refs.add(_base_type(v))
    for owner in _resource_controllers(contract, names):
        if owner:
            refs.add(owner.lower())
    for e in _entity_list(contract):
        for fld, typ in (e.get("fields") or {}).items():
            m = re.match(r"(.+?)_?id$", str(fld), re.I)
            if m and m.group(1):
                refs.add(_singularize(m.group(1)).lower())
            tb = _base_type(str(typ))
            if tb in name_set:
                refs.add(tb)
    return refs


# Wrapper / container / built-in type tokens that an endpoint payload string may contain but
# which are NOT domain types requiring a generated class (so they never count as "missing").
_PAYLOAD_WRAPPERS = {
    "response", "responseentity", "httpentity", "list", "set", "map", "optional", "page",
    "collection", "iterable", "arraylist", "linkedlist", "hashmap", "hashset", "treemap",
    "treeset", "linkedhashmap", "linkedhashset", "concurrenthashmap", "void", "object",
    "body", "request", "string", "long", "integer", "int", "double", "float", "boolean",
    "short", "byte", "char", "character", "bigdecimal", "biginteger", "date", "localdate",
    "localdatetime", "localtime", "instant", "uuid", "number", "t", "e", "k", "v",
    "edu", "fudan", "common", "util",
}


def _ondisk_class_names(output_dir: str) -> set:
    """Lowercased names of every class/interface/enum/record declared on disk under
    output_dir. In a brownfield project these are already-compilable types, so an endpoint
    that references one is NOT missing even if the architect left it out of `entities`."""
    import os
    names = set()
    decl = re.compile(r"\b(?:class|interface|enum|record)\s+(\w+)")
    if not output_dir or not os.path.isdir(output_dir):
        return names
    for dp, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs
                   if d not in ("target", ".git", "node_modules") and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".java"):
                continue
            try:
                src = open(os.path.join(dp, fn)).read()
            except Exception:
                continue
            for m in decl.findall(src):
                names.add(m.lower())
    return names


def _contract_missing_types(contract: dict, output_dir: str = None) -> set:
    """Domain types an endpoint returns/accepts that have NO class the pipeline will create
    (a contract entity or DTO) and that do NOT already exist on disk. Such a type makes the
    dev sprint author a controller referencing an undefined class → the module cannot compile
    (silent until build). Deterministic; digs the payload type out of Response<X>/List<X>/X[].
    """
    if not isinstance(contract, dict):
        return set()
    tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
    generic = re.compile(r"<([^<>]+)>")
    referenced = set()
    for ep in (contract.get("endpoints") or []):
        if not isinstance(ep, dict):
            continue
        for k in ("response", "request", "body", "result", "returns"):
            v = ep.get(k)
            if isinstance(v, dict):
                v = " ".join(str(x) for x in v.values())
            s = str(v or "")
            for t in tok.findall(s):
                if t[:1].isupper() and t.lower() not in _PAYLOAD_WRAPPERS:
                    referenced.add(t)
            # Inside generic brackets is unambiguously a TYPE position, so flag a payload
            # there even if the architect wrote it lowercase (Response<voucherInfo>) — the
            # uppercase-only rule above would silently pass it. (Innermost brackets only:
            # [^<>] can't cross a nested '<'.)
            for inner in generic.findall(s):
                for t in tok.findall(inner):
                    if len(t) > 2 and t.lower() not in _PAYLOAD_WRAPPERS:
                        referenced.add(t)
    if not referenced:
        return set()
    defined = {str(n).lower() for n in entity_names(contract) if n}
    defined |= _ondisk_class_names(output_dir) if output_dir else set()
    return {t for t in referenced if t.lower() not in defined}


def _dedup_duplicate_entities(contract: dict) -> dict:
    """Drop a contract entity that is referenced NOWHERE and is a structural twin — same
    non-id field set OR a near-variant name — of a type that IS referenced. Targets the
    'architect hallucinated a redundant parallel entity' defect (`ProductSale` beside the
    used `ProductSales` DTO; `Type` beside `PetType`). Drop-only, never rename; the double
    condition (orphan AND has an anchored twin) means a real, used entity can't be removed.
    Purely structural — no hardcoded entity names. Idempotent."""
    if not isinstance(contract, dict):
        return contract
    ents = _entity_list(contract)
    if len(ents) < 2:
        return contract
    referenced = _collect_referenced_types(contract)
    fieldsets = {e["name"]: _entity_field_names(e) for e in ents}
    names_lower = {e["name"].lower() for e in ents}
    keep, dropped = [], []
    for e in ents:
        n = e["name"]
        if n.lower() in referenced:
            keep.append(e)
            continue
        twin = None
        for o in ents:  # an anchored (referenced) twin among the other entities
            if o["name"] == n:
                continue
            if o["name"].lower() in referenced and (
                    (fieldsets[n] and fieldsets[n] == fieldsets[o["name"]])
                    or _name_variant(n, o["name"])):
                twin = o["name"]
                break
        if not twin:  # an anchored twin that isn't itself an entity (e.g. a response DTO)
            for ref in referenced:
                if ref not in names_lower and _name_variant(n, ref):
                    twin = ref
                    break
        if twin:
            dropped.append((n, twin))
        else:
            keep.append(e)
    if not dropped:
        return contract
    contract["entities"] = keep
    for d, t in dropped:
        print(f"   🧹 Contract dedup: dropped redundant entity '{d}' (unreferenced twin of '{t}')")
    return contract


_ORDINAL_PREFIXES = ("first", "second", "third", "left", "right", "source",
                     "target", "other", "primary", "secondary")


def _ordinal_twin_base(name: str) -> str:
    """If `name` looks like a positional/ordinal variant of some base type (First
    Product/SecondProduct, ProductA/ProductB, Product1/Product2), return that base name;
    else "". Purely structural — no hardcoded entity names, only positional markers.
    Targets the 'self-comparison feature modeled as two separate entity types instead of
    comparing one type against itself' defect (confirmed live: a near-duplicate-detection
    contract created FirstProduct/SecondProduct beside the real Product entity — both
    referenced only by each other's DTO, so the orphan-twin dedup above never fires,
    and 3 sprints of entity+repository+test work got spent on a phantom pair)."""
    low = name.lower()
    for p in _ORDINAL_PREFIXES:
        if low.startswith(p) and len(name) > len(p) and (name[len(p):len(p) + 1].isupper()):
            return name[len(p):]
    m = re.match(r"^(.+?)([ABab12])$", name)
    if m and len(m.group(1)) >= 3:
        return m.group(1)
    return ""


def _merge_ordinal_twin_entities(contract: dict, on_disk_types: set = None) -> dict:
    """Detect entities that are ordinal/positional variants of a common base (FirstProduct/
    SecondProduct, ProductA/ProductB, Product1/Product2) and remove the phantom(s), rewriting
    every string reference to the dropped name(s) throughout the contract (DTO field types,
    endpoint specs, relationships). Two cases:

      • BASE ALREADY EXISTS — the base type is real, either declared in this contract OR
        already on disk in the staged baseline (e.g. TeaStore's Product). The ordinal names
        are then positional ALIASES of it, never new entities: DROP them ALL and repoint every
        reference at the real base. Fires even for a LONE ordinal (FirstProduct with no
        SecondProduct twin) — a self-comparison feature must reuse the existing type, not mint
        one. This is the copurchase defect: FirstProduct beside the baseline Product was merged
        into a surviving `FirstProduct` phantom, which VF then flagged as an unbuildable orphan
        (contract declares it → must create; nothing references it → must delete; unresolvable).

      • BASE ABSENT everywhere — a genuinely new self-comparison type. Require an IDENTICAL
        field set (same bar as _dedup_duplicate_entities) and collapse the pair into ONE kept
        entity, so two genuinely different entities sharing a naming pattern are never merged.

    Runs before any dev/test task is planned, so a phantom never reaches the coverage backlog."""
    if not isinstance(contract, dict):
        return contract
    ents = _entity_list(contract)
    on_disk = {t.lower() for t in (on_disk_types or set())}
    fieldsets = {e["name"]: _entity_field_names(e) for e in ents}
    groups: dict = {}
    for e in ents:
        base = _ordinal_twin_base(e["name"])
        if base:
            groups.setdefault(base.lower(), []).append(e["name"])
    renames = {}
    drop = set()
    aliased = set()   # dropped because the base type already exists (a positional alias)
    for base_lower, names in groups.items():
        base_canonical = _ordinal_twin_base(names[0]) or _pascal(base_lower)
        existing = next((e["name"] for e in ents if e["name"].lower() == base_lower), None)
        if existing is not None or base_lower in on_disk:
            # positional aliases of a real type — drop them ALL, repoint to the base.
            keep = existing or base_canonical
            for n in names:
                if n != keep:
                    renames[n] = keep
                    drop.add(n)
                    aliased.add(n)
            continue
        # base absent everywhere: only collapse a genuine identical-field PAIR into one survivor
        if len(names) < 2:
            continue
        first_fields = fieldsets[names[0]]
        if not all(fieldsets[n] == first_fields for n in names):
            continue  # structurally different — not a twin pair, leave alone
        keep = names[0]
        for n in names:
            if n != keep:
                renames[n] = keep
                drop.add(n)
    if not renames:
        return contract

    def _rewrite(obj):
        if isinstance(obj, str):
            return renames.get(obj, obj)
        if isinstance(obj, list):
            return [_rewrite(v) for v in obj]
        if isinstance(obj, dict):
            return {k: _rewrite(v) for k, v in obj.items()}
        return obj

    # Surviving entities are rewritten too (their field TYPE refs to a dropped phantom become the
    # base), but they can never be renamed — a surviving name is never a rename key (dropped
    # names were filtered out), and dict keys (field names) are preserved by _rewrite.
    contract["entities"] = [_rewrite(e) for e in ents if e["name"] not in drop]
    for key in list(contract.keys()):
        if key != "entities":
            contract[key] = _rewrite(contract[key])
    for dropped, kept in renames.items():
        if dropped in aliased:
            print(f"   🧹 Contract dedup: dropped ordinal phantom '{dropped}' — base type "
                  f"'{kept}' already exists (positional alias, not a new entity); repointed "
                  f"references to '{kept}'")
        else:
            print(f"   🧹 Contract dedup: merged ordinal-twin entity '{dropped}' into '{kept}' "
                  f"(identical fields — a self-comparison feature doesn't need two entity types)")
    return contract


def _normalize_views(contract: dict) -> None:
    """Canonicalize the optional presentation `views` array in place (enhancement #2):
    PascalCase name, dedup by name, ensure a leading-slash `route` and a `.jsp` page name
    (derived from the name when absent). No-op when there is no `views` key, so every
    backend-only / Spring contract is byte-identical."""
    views = contract.get("views")
    if not isinstance(views, list):
        return
    seen, out = set(), []
    for v in views:
        if not isinstance(v, dict):
            continue
        name = _pascal(str(v.get("name", "")).strip())
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        route = str(v.get("route") or "").strip()
        if not route:
            route = "/" + _snake(name).replace("_", "-")
        if not route.startswith("/"):
            route = "/" + route
        jsp = str(v.get("jsp") or "").strip() or (_snake(name) + ".jsp")
        # Keep only the bare file name — the model often prefixes "WEB-INF/pages/", and
        # build_coverage_tasks + the presentation gate add that prefix themselves (else the
        # path doubles to …/WEB-INF/pages/WEB-INF/pages/x.jsp).
        jsp = jsp.replace("\\", "/").rsplit("/", 1)[-1]
        if not jsp.endswith(".jsp"):
            jsp += ".jsp"
        nv = {"name": name, "route": route, "jsp": jsp}
        if v.get("shows"):
            nv["shows"] = str(v["shows"])
        if v.get("consumes"):
            nv["consumes"] = str(v["consumes"])
        out.append(nv)
    contract["views"] = out


def normalize_contract(contract: dict, stack: dict, output_dir: str = None) -> dict:
    """Deterministic IR normalization + validation. Turns an LLM-proposed contract
    into a complete, canonical, conflict-free spec before it drives the
    deterministic backlog — consolidating the scattered Category-B fixes:
      • entities → canonical singular PascalCase, de-duplicated (Pets+Pet → Pet)
      • every PERSISTED entity gets an `id` field and a canonical snake_case `table`
        name, so @Table and any schema agree by construction (a `dto`-flagged or
        reclassified computed value object gets neither)
      • relationship-parent entities backfilled (Pet.ownerId ⇒ Owner), skipping
        entities that already exist on disk when `output_dir` is given (brownfield)
      • endpoints validated (malformed dropped)
      • DB vendor pinned for the config generators (one source of truth)
    Idempotent."""
    if not isinstance(contract, dict):
        return contract
    on_disk_entities = set(_existing_entities_on_disk(output_dir)) if output_dir else set()

    merged = {}
    for e in _entity_list(contract):
        e["name"] = _pascal(_singularize(e["name"]))
        key = e["name"].lower()
        if not key:
            continue
        if key in merged:
            merged[key]["fields"].update(e.get("fields", {}))
            if e.get("table"):
                merged[key]["table"] = e["table"]
        else:
            merged[key] = e
    contract["entities"] = list(merged.values())

    # backfill relationship parents on the now-canonical list
    _backfill_relationship_entities(contract, on_disk=on_disk_entities)

    # guarantee an id field and a canonical table name on every PERSISTED entity — a
    # computed/response DTO (explicit `dto: true`, or matched by the name/structural
    # reclassifier) must NOT get either: an id+table turns a value object into a phantom
    # persisted entity (TeaStore stockledger probe: LowStockEntry, dto:true from the
    # architect, still got id+table injected here before this fix, then a downstream
    # pass "reconciled" the id type it should never have had in the first place).
    dto_names = _dto_names(contract)
    for e in contract["entities"]:
        if not isinstance(e, dict):
            continue
        if e.get("name") in dto_names:
            continue
        fields = e["fields"] if isinstance(e.get("fields"), dict) else {}
        if "id" not in {k.lower() for k in fields}:
            fields = {"id": "UUID", **fields}
        e["fields"] = fields
        e.setdefault("table", _snake(e["name"]))

    # validate endpoints: keep only well-formed {method, path, ...}
    eps = contract.get("endpoints", [])
    contract["endpoints"] = (
        [ep for ep in eps if isinstance(ep, dict) and ep.get("method") and ep.get("path")]
        if isinstance(eps, list) else []
    )

    # merge ordinal/positional twin entities (FirstProduct+SecondProduct -> Product) before
    # the orphan-twin gate below: these ARE mutually referenced (each other's DTO), so the
    # "unreferenced" condition that gate requires never fires for them on its own. Pass the
    # on-disk type names so a phantom whose base already exists in the staged baseline (e.g.
    # TeaStore's Product) is dropped outright, not merged into a surviving twin.
    contract = _merge_ordinal_twin_entities(
        contract, on_disk_types=(on_disk_entities | _on_disk_type_names(output_dir)))

    # drop hallucinated duplicate entities (orphan + anchored twin) — runs after entities
    # and endpoints are canonical so the reference graph is complete.
    contract = _dedup_duplicate_entities(contract)

    # canonicalize the optional presentation tier (no-op when there is no `views` key)
    _normalize_views(contract)

    # pin the DB vendor so config generators read one fact, not their own guess
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if is_java and not contract.get("database"):
        contract["database"] = {"vendor": "H2", "mode": "in-memory"}
    return contract


def _segment_to_entity(seg: str, entities: list) -> str:
    """Map an endpoint's top-level path segment (usually a plural noun) to an
    entity/controller name."""
    # Strip path-segment word separators (e.g. "health-alerts" / "health_alerts")
    # so kebab/snake-case segments compare equal to the camel-case entity name
    # ("HealthAlert") — otherwise no entity matches and the fallback below
    # derives a malformed name like "Health-alert".
    s = seg.strip().lower().replace("-", "").replace("_", "")
    for e in entities:
        el = e.lower()
        # Handle regular plurals plus the y→ies form (specialty → specialties).
        if (s == el or s == el + "s" or s == el + "es"
                or (el.endswith("y") and s == el[:-1] + "ies")
                or s.startswith(el)):
            return e
    if s.endswith("ies") and len(s) > 3:
        singular = s[:-3] + "y"
    elif s.endswith("s") and len(s) > 1:
        singular = s[:-1]
    else:
        singular = s
    return singular[:1].upper() + singular[1:]


def _endpoint_owner(ep: dict, entities: list) -> str:
    """The single controller that owns an endpoint = the entity of its TOP-LEVEL
    path segment. A nested endpoint like /pets/{id}/vaccinations is owned by Pet,
    NOT Vaccination — so each endpoint belongs to exactly one controller."""
    path = ep.get("path", "") if isinstance(ep, dict) else ""
    if not path or path == "/health":
        return ""
    stripped = path.strip("/")
    seg = stripped.split("/")[0] if stripped else ""
    if not seg or seg.startswith("{"):
        return ""
    return _segment_to_entity(seg, entities)


def _resource_controllers(contract: dict, entities: list) -> list:
    """One controller per distinct top-level endpoint resource. Sub-resource
    endpoints (e.g. /orders/{id}/items) fold into the parent (/orders) controller,
    so a child entity like OrderItem gets NO controller of its own."""
    endpoints = contract.get("endpoints", []) if isinstance(contract, dict) else []
    out, seen = [], set()
    for ep in endpoints:
        name = _endpoint_owner(ep, entities)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append(name)
    return out


def _controller_endpoints(contract: dict, name: str, entities: list) -> list:
    """The exact, disjoint endpoint list a controller owns (by top-level segment).
    Passing this explicitly prevents two controllers from both implementing the
    same sub-resource (the dead-code/duplication failure)."""
    endpoints = contract.get("endpoints", []) if isinstance(contract, dict) else []
    out = []
    for ep in endpoints:
        if _endpoint_owner(ep, entities) == name:
            method = ep.get("method", "?") if isinstance(ep, dict) else "?"
            path = ep.get("path", "?") if isinstance(ep, dict) else "?"
            out.append(f"{method} {path}")
    return out


def build_coverage_tasks(contract: dict, stack: dict, existing_files=None,
                         mapped_endpoints=None, workspace_state: dict = None) -> list:
    """Deterministic, dependency-ordered Java backlog that fully covers the
    contract: entity -> repository -> service per entity, plus one controller per
    top-level endpoint resource (sub-resources fold into the parent controller).

    Returns [] for non-Java stacks. If `existing_files` is given, only tasks whose
    target file is absent are returned (used by the coverage safety-net gate);
    if None, the full backlog is returned (used by the planner).

    `mapped_endpoints` (a set of (HTTP_METHOD, normalized_path) actually served on
    disk, from sprint._controller_mappings) makes controller coverage ENDPOINT-aware:
    a controller file existing no longer counts as covered — if it's missing some of
    its contract endpoints, an EXTEND task (marked coverage_extend=True) is emitted so
    the missing routes get added in place. Without it, controllers dedup by file
    existence as before (used by the upfront planner)."""
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if not is_java:
        return []
    ext = stack.get("file_extension", ".java")
    entities = [e for e in entity_names(contract) if e]
    if not entities:
        return []
    existing = None
    if existing_files is not None:
        existing = {os.path.basename(f).lower() for f in existing_files}

    tasks = []

    def add(prefix, name, suffix, ttype, title):
        fname = f"{name}{suffix}{ext}"
        if existing is not None and fname.lower() in existing:
            return
        tasks.append({"id": f"{prefix}-{name}", "title": title, "type": ttype, "target_file": fname})

    # Layering depends on the framework family. Spring: entity→repository→service per
    # entity, + one @RestController per resource. Non-Spring (JAX-RS/EclipseLink, e.g.
    # TeaStore): entity→repository→endpoint, NO service layer, and the resource layer is
    # a JAX-RS @Path *Endpoint, not a *Controller (TeaStore e2e run #2: the Spring layering
    # spawned phantom ProductService/ProductController tasks for a model that has neither).
    from agile_team_cline.sprint import _is_spring, _norm_endpoint_path
    spring = _is_spring(stack)

    # Split computed/response DTOs out of the persistent-entity set: a DTO gets a plain
    # POJO task only — NO @Entity, @Table, repository, service, or CREATE TABLE (which is
    # what made `ProductSalesSummary` a phantom-table entity in run #15). Resource/endpoint
    # derivation below still sees the full `entities` list (a DTO owns no path, so it gets
    # no controller anyway), keeping endpoint ownership unchanged.
    dto = _dto_names(contract)
    persistent = [n for n in entities if n not in dto]
    dtos = [n for n in entities if n in dto]

    for n in dtos:
        add("DTO", n, "", "dto",
            f"Create the {n} response DTO as a PLAIN POJO: the contract fields plus getters and a "
            f"matching all-args constructor. It is a COMPUTED/response value object — do NOT annotate "
            f"it @Entity, do NOT add @Table or any JPA mapping, do NOT create a {n}Repository, and do "
            f"NOT add a database table for it. This type is serialized by the framework onto the wire: "
            f"open one of the project's EXISTING transport/DTO classes and copy its exact shape "
            f"(constructors — including a no-arg one if siblings have it — getters, naming style). "
            f"Name it {n} exactly as the contract declares, and the endpoint/repository code that "
            f"produces it MUST reference {n} (typed return, not ad-hoc maps) — an unreferenced DTO "
            f"is a defect.")
    # Java SE registration rule — non-Spring only. `mvn test` bootstraps the persistence
    # unit in Java SE mode, where the provider is NOT required to auto-discover annotated
    # classes and EclipseLink does not. An @Entity missing from persistence.xml's <class>
    # list compiles cleanly, then throws "The abstract schema type 'X' is unknown" on the
    # first query naming it — and the stack trace points at the REPOSITORY that ran the
    # query, never at the descriptor that is actually wrong (volatility 2026-07-28: 11
    # errors over 4 test classes, all tracing to the repository; the fix was one <class>
    # line). Spring Boot scans the classpath itself and has no persistence.xml, so it must
    # never be told this. vf_entity_registered is the safety net if the coder skips it.
    _register = "" if spring else (
        " Then REGISTER it in the persistence unit: add a <class> entry with the entity's "
        "FULLY-QUALIFIED name to src/main/resources/META-INF/persistence.xml, alongside the "
        "entries already there. The tests bootstrap JPA in Java SE mode, where EclipseLink "
        "does NOT auto-discover annotated classes — an unregistered entity compiles but "
        "every query against it fails with \"The abstract schema type is unknown\". Edit "
        "that file in place; do not create a second persistence.xml.")
    for n in persistent:
        add("ENT", n, "", "entity",
            f"Create the {n} JPA entity with the fields defined in the contract.{_register}")
    for n in persistent:
        # Reconcile the repository name against on-disk reality. TeaStore's @Entity classes
        # are Persistence-prefixed (PersistenceProduct) but their repositories drop the prefix
        # (ProductRepository); deriving the repo as f"{n}Repository" yields a phantom
        # PersistenceProductRepository that exact-filename dedup never catches (the real file
        # is ProductRepository.java), so a dead duplicate repo is generated alongside a
        # correct out-of-band patch to the real one — runs #36/#37/#38/#42, 5×. Prefer the
        # prefix-stripped owner ONLY when that repo actually exists on disk and the literal
        # one does not, so a project that genuinely names a repo PersistenceXRepository (or
        # any non-TeaStore stack) is unaffected.
        repo_owner = n
        if (existing is not None and n.startswith("Persistence") and len(n) > len("Persistence")
                and f"{n[len('Persistence'):]}Repository{ext}".lower() in existing
                and f"{n}Repository{ext}".lower() not in existing):
            repo_owner = n[len("Persistence"):]
        if spring:
            add("REPO", repo_owner, "Repository", "repository", f"Create {repo_owner}Repository extending JpaRepository for {n}")
        else:
            add("REPO", repo_owner, "Repository", "repository",
                f"Create {repo_owner}Repository following the project's EXISTING repository pattern (a singleton "
                f"repository over an EntityManager / the project's AbstractCRUDEndpoint contract). Do NOT "
                f"use Spring Data or JpaRepository.")
    if spring:
        for n in persistent:
            others = [e for e in entities if e != n]
            others_str = ", ".join(others) if others else "none"
            add("SVC", n, "Service", "service",
                f"Create {n}Service containing ONLY {n}'s own business and persistence logic. "
                f"Do NOT add methods that primarily operate on other entities ({others_str}) — each of those has its OWN service that owns its logic. "
                f"When an operation needs another entity, inject and delegate to that entity's service or repository; never copy or re-implement it here.")

    res_suffix = "Controller" if spring else "Endpoint"
    res_type = "controller" if spring else "endpoint"
    for r in _resource_controllers(contract, entities):
        owned = _controller_endpoints(contract, r, entities)
        fname = f"{r}{res_suffix}{ext}"
        if mapped_endpoints is None:
            # No disk-route info (upfront planner): dedup by file existence as before.
            owned_str = ", ".join(owned) if owned else "(none)"
            if spring:
                add("CTL", r, "Controller", "controller",
                    f"Create {r}Controller exposing EXACTLY these REST endpoints and no others: [{owned_str}]. "
                    f"Do NOT implement any endpoint outside this list — other controllers own those. "
                    f"Delegate all business logic to the relevant service; do not reimplement another entity's logic.")
            else:
                add("CTL", r, "Endpoint", "endpoint",
                    f"Create {r}Endpoint as a JAX-RS resource (@Path with @GET/@POST/@PUT/@DELETE) exposing "
                    f"EXACTLY these endpoints and no others: [{owned_str}]. Follow the project's existing "
                    f"*Endpoint pattern; do NOT use Spring. Put query logic in {r}Repository, not the endpoint.")
            continue
        # Endpoint-aware: which of this resource's owned routes aren't served yet?
        missing = [ep for ep in owned
                   if (ep.split(" ", 1)[0].upper(),
                       _norm_endpoint_path(ep.split(" ", 1)[1] if " " in ep else "")) not in mapped_endpoints]
        if not missing:
            continue  # resource serves all its contract endpoints — nothing to do
        present = [ep for ep in owned if ep not in missing]
        if spring:
            if present:
                tasks.append({"id": f"CTL-{r}", "type": "controller", "target_file": fname,
                              "coverage_extend": True,
                              "title": (f"Extend {r}Controller to ADD these missing REST endpoints "
                                        f"(keep all existing ones): [{', '.join(missing)}]. Delegate all "
                                        f"business logic to the relevant service; do not reimplement another "
                                        f"entity's logic.")})
            else:
                tasks.append({"id": f"CTL-{r}", "type": "controller", "target_file": fname,
                              "title": (f"Create {r}Controller exposing EXACTLY these REST endpoints and no "
                                        f"others: [{', '.join(missing)}]. Do NOT implement any endpoint outside "
                                        f"this list — other controllers own those. Delegate all business logic "
                                        f"to the relevant service; do not reimplement another entity's logic.")})
        else:
            # Non-Spring: if the *Endpoint file already exists on disk, EXTEND it in place
            # (the existing JAX-RS resource gains the missing route) instead of creating a
            # duplicate. Existence is the disk basename set in `existing`.
            file_exists = existing is not None and fname.lower() in existing
            if file_exists or present:
                tasks.append({"id": f"CTL-{r}", "type": "endpoint", "target_file": fname,
                              "coverage_extend": True,
                              "title": (f"Extend the existing {r}Endpoint (JAX-RS resource) to ADD these missing "
                                        f"endpoint(s) (keep all existing ones): [{', '.join(missing)}]. Follow the "
                                        f"existing @Path/@GET pattern; put query logic in {r}Repository, not the "
                                        f"endpoint. Do NOT use Spring.")})
            else:
                tasks.append({"id": f"CTL-{r}", "type": "endpoint", "target_file": fname,
                              "title": (f"Create {r}Endpoint as a JAX-RS resource (@Path/@GET/...) exposing EXACTLY "
                                        f"these endpoint(s) and no others: [{', '.join(missing)}]. Follow the "
                                        f"project's existing *Endpoint pattern; query logic in {r}Repository. "
                                        f"Do NOT use Spring.")})

    # ── Presentation tier (enhancement #2): servlet + JSP per contract view ──────────
    # Appended AFTER all backend tasks (entity/repo/endpoint) so they land in a later sprint
    # and the backend API exists first (FullStack-Agent backend→frontend ordering). Jakarta +
    # AGILE_PRESENTATION + a `views` section only — a backend-only or Spring contract is unchanged.
    if not spring and os.getenv("AGILE_PRESENTATION", "0") == "1":
        for v in (contract.get("views") or []):
            if not isinstance(v, dict) or not v.get("name"):
                continue
            name, route = v["name"], v.get("route", "")
            jsp = v.get("jsp") or f"{_snake(name)}.jsp"
            shows = v.get("shows") or "the data"
            consumes = v.get("consumes") or shows
            sfname = f"{name}Servlet{ext}"
            if existing is None or sfname.lower() not in existing:
                tasks.append({"id": f"WEB-{name}", "type": "servlet", "target_file": sfname,
                    "title": (f"Create {name}Servlet following the project's EXISTING servlet pattern "
                              f"(extend AbstractUIServlet, see ProductServlet): annotate @WebServlet(\"{route}\"); "
                              f"in handleGETRequest, fetch {shows} via the registryclient LoadBalanced*Operations "
                              f"for '{consumes}', set the request attributes, and forward to WEB-INF/pages/{jsp}. "
                              f"Do NOT call the persistence service directly — go through registryclient. Do NOT "
                              f"use Spring.")})
            else:
                # Servlet exists in brownfield — generate an edit task to add the registryclient wiring.
                # vf_handoff_wired will verify the LoadBalanced*Operations call is present.
                tasks.append({"id": f"WEB-{name}", "type": "servlet", "target_file": sfname,
                    "title": (f"Edit the EXISTING {sfname} in-place to add the registryclient handoff: "
                              f"in handleGETRequest, call the registryclient LoadBalanced*Operations "
                              f"for '{consumes}', set the request attributes for {shows}, and forward "
                              f"to WEB-INF/pages/{jsp}. Do NOT call the persistence service directly — "
                              f"go through registryclient. Preserve ALL existing logic. Do NOT use Spring.")})
            jrel = os.path.join("src", "main", "webapp", "WEB-INF", "pages", jsp)
            if existing is None or jsp.lower() not in existing:
                tasks.append({"id": f"JSP-{name}", "type": "view", "target_file": jrel,
                    "title": (f"Create the {jsp} JSP page under src/main/webapp/WEB-INF/pages rendering the "
                              f"request attributes {name}Servlet sets (the {shows} data); follow the existing "
                              f"product.jsp layout (JSTL/EL over the request attributes).")})

    # BROWNFIELD: Filter out tasks that would regenerate existing classes
    if workspace_state:
        tasks = filter_tasks_for_brownfield(tasks, workspace_state)

    # VeriMAP: attach per-task verification function descriptors (no-op when AGILE_VERIFY_AWARE=0)
    if os.getenv("AGILE_VERIFY_AWARE", "0") != "0":
        from agile_team_cline.verify_lib import attach_verifications
        tasks = [attach_verifications(t, contract, stack) for t in tasks]

    return tasks


# ── Deterministic controller skeleton emitter ─────────────────────────────────
# Writes a compiling @RestController whose mappings exist by construction (so the
# coverage gate is satisfied WITHOUT depending on cline obeying a prompt) and whose
# bodies throw UnsupportedOperationException — a TODO marker cline fills later. This
# decouples "the endpoints exist & are wired" (deterministic) from "the endpoints do
# the right thing" (LLM). Pairs with the small per-controller units from grouping.
SKELETON_STUB_MARKER = 'UnsupportedOperationException("TODO: implement'


def _skel_norm_path(path: str) -> str:
    p = re.sub(r"\{[^}]*\}", "{}", (path or "").strip())
    return "/" + p.strip("/")


def _skel_pathvar_type(var: str, pinned_ids: dict = None, body_type: str = None) -> str:
    # Body only throws, so any consistent type compiles — but a wrong skeleton type
    # anchors cline on it during body-fill (id-type drift). Resolve id-like vars from
    # the contract's _pinned_ids (petId → Pet.id type, bare id → body_type's id type);
    # fall back to Long. Non-id vars stay String.
    v = (var or "").lower()
    if "id" not in v:
        return "String"
    pins = {k.lower(): t for k, t in (pinned_ids or {}).items()}
    if pins:
        if v == "id" and body_type and body_type.lower() in pins:
            return pins[body_type.lower()]
        ref = v[:-2] if v.endswith("id") and len(v) > 2 else ""
        if ref:
            if ref in pins:
                return pins[ref]
            for k, t in pins.items():
                if k.startswith(ref) or ref.startswith(k):
                    return t
        if len(set(pins.values())) == 1:  # project uses one id type everywhere
            return next(iter(pins.values()))
    return "Long"


def _skel_cap(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]", "", s or "")
    return s[:1].upper() + s[1:] if s else s


def _skel_method_name(http: str, path: str) -> str:
    verb = {"GET": "get", "POST": "create", "PUT": "update",
            "DELETE": "delete", "PATCH": "patch"}.get(http.upper(), "handle")
    statics = [s for s in path.strip("/").split("/") if s and not s.startswith("{")]
    vars_ = re.findall(r"\{([^}]*)\}", path)
    name = verb + "".join(_skel_cap(s) for s in statics)
    if vars_:
        name += "By" + "".join(_skel_cap(v) for v in vars_)
    return name or (verb + "Endpoint")


def emit_controller_skeleton(class_name: str, base_path: str, endpoints: list,
                             package: str, output_dir: str, body_type: str = None,
                             pinned_ids: dict = None) -> str:
    """Write <package>/<class_name>.java: a compiling @RestController where each
    'METHOD /path' in `endpoints` becomes an @*Mapping method whose body throws
    UnsupportedOperationException. `body_type` (a same-package type, e.g. 'Pet') sets the
    return/@RequestBody type; None → ResponseEntity<?> / Object. Endpoints are deduped by
    (verb, var-normalized path) to avoid duplicate mappings. Returns the relpath, or None
    if nothing emittable."""
    ann = {"GET": "GetMapping", "POST": "PostMapping", "PUT": "PutMapping",
           "DELETE": "DeleteMapping", "PATCH": "PatchMapping"}
    base = "/" + base_path.strip("/") if base_path and base_path.strip("/") else ""
    ret = f"ResponseEntity<{body_type}>" if body_type else "ResponseEntity<?>"
    seen, used, methods = set(), set(), []
    for ep in endpoints:
        parts = ep.split(None, 1)
        if len(parts) != 2:
            continue
        http, path = parts[0].upper(), parts[1].strip()
        if http not in ann:
            continue
        key = (http, _skel_norm_path(path))
        if key in seen:
            continue
        seen.add(key)
        sub = path[len(base):] if base and path.startswith(base) else path
        sub = "/" + sub.strip("/")
        mapping = f"@{ann[http]}" + (f'("{sub}")' if sub != "/" else "")
        params = [f"@PathVariable {_skel_pathvar_type(v, pinned_ids, body_type)} {v}"
                  for v in re.findall(r"\{([^}]*)\}", path)]
        if http in ("POST", "PUT"):
            params.append(f"@RequestBody {body_type or 'Object'} body")
        mname = _skel_method_name(http, path)
        while mname in used:
            mname += "_"
        used.add(mname)
        methods.append(
            f"    {mapping}\n"
            f"    public {ret} {mname}({', '.join(params)}) {{\n"
            f'        throw new UnsupportedOperationException("TODO: implement {class_name}.{mname}");\n'
            f"    }}\n")
    if not methods:
        return None
    cls_ann = "@RestController\n" + (f'@RequestMapping("{base}")\n' if base else "")
    src = (f"package {package};\n\n"
           "import org.springframework.http.ResponseEntity;\n"
           "import org.springframework.web.bind.annotation.*;\n\n"
           f"{cls_ann}public class {class_name} {{\n\n"
           + "\n".join(methods) + "}\n")
    rel = os.path.join("src", "main", "java", *package.split("."), f"{class_name}.java")
    dest = os.path.join(output_dir, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w") as f:
        f.write(src)
    return rel


def count_skeleton_stubs(output_dir: str) -> int:
    """Count UnsupportedOperationException TODO stubs left in the tree. A milestone with
    stubs remaining is structurally present but NOT implemented, so it must not be
    checkpointed 'complete' even though it compiles."""
    n = 0
    root = os.path.join(output_dir, "src", "main", "java")
    for dp, dn, fns in os.walk(root if os.path.isdir(root) else output_dir):
        dn[:] = [d for d in dn if d not in ("target", ".git") and not d.startswith(".")]
        for fn in fns:
            if fn.endswith(".java"):
                try:
                    with open(os.path.join(dp, fn)) as f:
                        n += f.read().count(SKELETON_STUB_MARKER)
                except Exception:
                    pass
    return n


def reconcile_view_collisions(contract: dict, output_dir: str, namespace: str = "/api") -> dict:
    """Re-namespace contract REST endpoints whose (verb, path) is already served by a
    legacy Thymeleaf view @Controller, so the generated REST controllers don't clash with
    the view controllers at boot (Spring 'Ambiguous mapping' — compiles, fails at context
    startup). Whole-resource: if ANY of a resource's endpoints collide, all of that
    resource's not-yet-namespaced endpoints get the prefix (one consistent base path).
    Relocates the URL; drops nothing — the endpoint's logic is still generated. Probe-
    driven, deterministic, idempotent. No-op if the probe is unavailable or there are no
    view routes (e.g. greenfield)."""
    if not isinstance(contract, dict) or not isinstance(contract.get("endpoints"), list):
        return contract
    try:
        from agile_team_cline import java_probe
    except Exception:
        return contract

    def _norm(p):
        p = re.sub(r"\{[^}]*\}", "{}", (p or "").strip())
        return "/" + p.strip("/")

    ns = namespace.strip("/")
    view_owned = {(str(e["http"]).upper(), _norm(e["path"]))
                  for c in java_probe.controllers(output_dir)
                  for e in c["endpoints"] if not e["serves_json"]}
    if not view_owned:
        return contract

    entities = [e for e in entity_names(contract) if e]
    colliding = set()
    for ep in contract["endpoints"]:
        if not isinstance(ep, dict):
            continue
        path = ep.get("path", "")
        if path.strip("/").split("/")[:1] == [ns]:
            continue  # already namespaced
        if (str(ep.get("method", "GET")).upper(), _norm(path)) in view_owned:
            colliding.add(_endpoint_owner(ep, entities))
    if not colliding:
        return contract

    import copy
    contract = copy.deepcopy(contract)
    pfx = "/" + ns
    seen, out = set(), []
    for ep in contract["endpoints"]:
        if isinstance(ep, dict):
            path = ep.get("path", "")
            already = path.strip("/").split("/")[:1] == [ns]
            if (not already and path.startswith("/")
                    and _endpoint_owner(ep, entities) in colliding):
                ep["path"] = pfx + path
            key = (str(ep.get("method", "")).upper(), ep.get("path", ""))
            if key in seen:
                continue
            seen.add(key)
        out.append(ep)
    contract["endpoints"] = out
    return contract
