import json
from camel.messages import BaseMessage
from agile_team_cline.agents.base import make_agent, zh_prompt
from agile_team_cline.agents.roles import ARCHITECT, QA
from agile_team_cline.config import TASK_TOKEN_LIMITS, TOTAL_SPRINTS
from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.shared import extract_json, step_with_retry

# Optional tree-sitter Java probe (deterministic, no JVM, parses non-compiling trees).
# When present it replaces the regex/basename heuristics for controller detection,
# fix-routing, and truncated-file detection; absent, the legacy regex paths are used.
try:
    from agile_team_cline import java_probe
    _HAS_TS = True
except Exception:
    _HAS_TS = False

_ARCHITECT_EXPAND_PROMPT = (
    "⚡ Expand this contract with full field types and validation rules. JSON only.\n"
    "Contract: {contract}"
)

_ARCHITECT_EXPAND_PROMPT_ZH = (
    "⚡ 扩展此 contract，补全完整的字段类型和校验规则。仅输出JSON。\n"
    "Contract: {contract}"
)

_QA_REPORT_PROMPT = (
    "⚡ Review these source files against the contract. Output JSON only.\n"
    "Source Files:\n{files_content}\n\n"
    "Contract: {contract}\n"
    'Output format: {{"passed": [], "issues": [{{"file": "...", "concern": "..."}}], "coverage_gaps": []}}'
)

_QA_REPORT_PROMPT_ZH = (
    "⚡ 依据 contract 审查以下源文件，仅输出JSON。\n"
    "Source Files:\n{files_content}\n\n"
    "Contract: {contract}\n"
    '输出格式：{{"passed": [], "issues": [{{"file": "...", "concern": "..."}}], "coverage_gaps": []}}'
)

def run_sprint(sprint_num: int, tasks: list, blackboard: Blackboard, stack: dict, model, refinement_context: dict = None, total_sprints: int = None) -> dict:
    # The QA sprint number is assigned DYNAMICALLY by the pipeline (2 + dev chunks);
    # the caller passes it so the QA dispatch (which runs the deterministic Maven
    # gate) fires on the real final sprint. Falling back to the constant misroutes
    # the QA sprint as a dev sprint whenever the feature's chunk count != the
    # constant, silently skipping the build gate (false-green accepts).
    if total_sprints is None:
        total_sprints = TOTAL_SPRINTS

    if sprint_num == 1:
        _run_architect_sprint(blackboard, stack, model)
        return _run_dev_sprint(tasks, blackboard, stack, model)
    elif sprint_num == total_sprints:
        return _run_qa_sprint(tasks, blackboard, stack, model)
    else:
        return _run_dev_sprint(tasks, blackboard, stack, model, refinement_context=refinement_context)


def _run_architect_sprint(blackboard: Blackboard, stack: dict, model) -> dict:
    contract = blackboard.read("contract") or {}
    if contract.get("entities") and contract.get("endpoints"):
        print("ℹ️ Contract is already detailed. Skipping architect expand pass to prevent overwriting.")
        return {"contract_expanded": False, "skipped": True}

    agent = make_agent(ARCHITECT["role"], ARCHITECT["goal"], TASK_TOKEN_LIMITS["integration"], model=model)
    prompt = zh_prompt(_ARCHITECT_EXPAND_PROMPT, _ARCHITECT_EXPAND_PROMPT_ZH).format(
        contract=json.dumps(contract, separators=(",", ":")),
    )
    msg = BaseMessage.make_user_message(role_name="User", content=prompt)
    response = step_with_retry(agent, msg)
    raw = response.msgs[0].content if response.msgs else ""
    expanded = extract_json(raw)
    if expanded:
        blackboard.write("contract", expanded)
    return {"contract_expanded": bool(expanded)}


def _python_structural_summary(source: str) -> str:
    """Top-level exports of a Python module: classes (+ methods), functions,
    and assigned/imported names. Lets a developer see what a sibling module
    actually exposes instead of inventing symbols to import."""
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    def _sig(fn):
        args = [a.arg for a in fn.args.args]
        if fn.args.vararg:
            args.append("*" + fn.args.vararg.arg)
        if fn.args.kwarg:
            args.append("**" + fn.args.kwarg.arg)
        prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {fn.name}({', '.join(args)})"

    out = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            out.append(f"class {node.name}" + (f"({', '.join(bases)})" if bases else "") + ":")
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.append(f"  {_sig(sub)}")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(_sig(node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.append(f"{t.id} = ...")
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(f"{node.target.id} = ...")
    return "\n".join(out)


def _get_structural_summary(output_dir: str, filepath: str) -> str:
    import os
    full_path = ""
    if "/" in filepath or "\\" in filepath:
        full_path = os.path.join(output_dir, filepath)
    else:
        for root, _, files in os.walk(output_dir):
            if filepath in files:
                full_path = os.path.join(root, filepath)
                break
    if not full_path or not os.path.exists(full_path):
        return ""

    if full_path.endswith(".py"):
        try:
            with open(full_path, "r") as f:
                return _python_structural_summary(f.read())
        except Exception:
            return ""

    try:
        with open(full_path, "r") as f:
            lines = f.readlines()

        summary_lines = []
        in_class = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("package "):
                summary_lines.append(stripped)
            elif "class " in stripped or "enum " in stripped or "interface " in stripped:
                in_class = True
                summary_lines.append(stripped)
            elif in_class and (stripped.startswith("private ") or stripped.startswith("protected ") or stripped.startswith("public ")):
                # Method or constructor signature — cross-file consumers need
                # method names + types or they drift (e.g. controller calling
                # `getNotes()` when service exposes `getAllNotes()`).
                if "(" in stripped:
                    sig = stripped.split("{", 1)[0].rstrip()
                    # Skip blocks that opened on the next line — we don't track
                    # multi-line signatures; the truncated head is good enough.
                    if not sig.endswith(";"):
                        sig = sig.rstrip() + (";" if sig.endswith(")") else "")
                    summary_lines.append(f"  {sig}")
                elif "=" not in stripped and stripped.endswith(";"):
                    summary_lines.append(f"  {stripped}")
                elif "=" in stripped:
                    idx = stripped.find("=")
                    summary_lines.append(f"  {stripped[:idx].strip()};")
        return "\n".join(summary_lines)
    except Exception:
        return ""


# Java invariants the agentic coder must honour — distilled from _DEVELOPER_PROMPT's
# PRESERVATION & QUALITY CONTRACT. These are the rules the deterministic gates
# (source hygiene, repair.py) and javac depend on, so they go in the brief verbatim.
def _is_spring(stack: dict) -> bool:
    """True for Spring stacks (default), False for non-Spring (e.g. Jakarta/JAX-RS).
    Explicit framework_family wins; otherwise infer from the framework string."""
    fam = stack.get("framework_family")
    if fam:
        return fam == "spring"
    return "spring" in stack.get("framework", "Spring").lower()


def _build_cline_brief(tasks: list, blackboard: Blackboard, stack: dict,
                       refinement_context: dict = None) -> str:
    """Compose one agentic brief covering all `tasks`, reusing the same authoritative
    context the camel path injected (contract + symbol-table interfaces + test
    directives) but instructing cline to author/edit the files in place itself."""
    import os
    contract = blackboard.read("contract")
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"

    # Module scope: when AGILE_BUILD_MODULE names a discovered module, lead the
    # brief with a hard instruction to stay inside that module's subtree. Empty when
    # unset, so the single-module (petclinic) brief is byte-identical.
    scope_hdr = ""
    _target = os.getenv("AGILE_BUILD_MODULE", "").strip()
    if _target:
        _ws = blackboard.read("workspace_mapping") or {}
        _targets = [t.strip() for t in _target.split(",") if t.strip()]
        _mods = [m for m in _ws.get("modules", []) if m["path"] in _targets]
        if len(_mods) == 1:
            _mod = _mods[0]
            _pkg = (_mod.get("package_root") or "").replace(".", "/")
            scope_hdr = (
                f"SCOPE — work ONLY inside the Maven module at '{_target}'.\n"
                f"- Create/edit Java main-source files only under '{_target}/src/main/java/{_pkg}/...'.\n"
                f"- Create/edit Java test files only under '{_target}/src/test/java/{_pkg}/...' — "
                f"NEVER at a bare 'src/test/java/...' path relative to the repo root; that directory "
                f"is outside this module and Maven will never run tests placed there.\n"
                f"- Do NOT create or modify files in any other module or directory.\n"
                f"- Follow that module's existing package layout and code patterns.\n\n"
            )
        elif len(_mods) > 1:
            # Cross-module: allow several module subtrees, but forbid wiring a
            # direct Maven dependency between services — they integrate over REST.
            _lines = "\n".join(
                f"  - '{m['path']}/src/main/java/{(m.get('package_root') or '').replace('.', '/')}/...' (main)\n"
                f"  - '{m['path']}/src/test/java/{(m.get('package_root') or '').replace('.', '/')}/...' (tests)"
                for m in _mods
            )
            scope_hdr = (
                "SCOPE — work ONLY inside these Maven modules:\n"
                f"{_lines}\n"
                "- Test files go under that SAME module's 'src/test/java/...' — NEVER at a bare "
                "'src/test/java/...' path relative to the repo root; that directory is outside "
                "every module and Maven will never run tests placed there.\n"
                "- Do NOT create or modify files in any other module or directory.\n"
                "- Follow each module's existing package layout and code patterns.\n"
                "- These services talk to each other ONLY over REST through the registry/"
                "load balancer (registryclient) — do NOT add a direct compile/Maven "
                "dependency between services.\n\n"
            )

    targets = []
    for t in tasks:
        fname = t.get("target_file") or _title_to_filename(t["title"], stack["file_extension"])
        targets.append((t, fname))
    tasks_block = "\n".join(
        f"- {t['title']}  →  target file: {fname}  (type: {t.get('type','default')})"
        for t, fname in targets
    )

    # Drift-control B: pinned id types extracted from disk at contract reconciliation.
    # ~10 lines of facts — read-first steering stays intact, the brief stays lean.
    pinned = contract.get("_pinned_ids") if isinstance(contract, dict) else None
    pinned_block = ""
    if pinned:
        pinned_block = ("PINNED ID TYPES (on-disk truth — confirm by reading the code; NEVER "
                        "contradict these in ids, foreign keys, path variables or repositories):\n"
                        + "".join(f"  {n}.id: {t}\n" for n, t in pinned.items()) + "\n")

    # Refinement issues to fix (only those touching our target files — plus frozen-suite
    # failures, which are exempt from ownership on purpose).
    refine_block = ""
    if refinement_context:
        target_names = {fname for _t, fname in targets}
        # scope="frozen_suite" is set by _parse_surefire ONLY for a thrown exception it
        # traced to a specific production file. Those escape the target-file filter because
        # the file that threw usually owns no backlog task (the backlog is contract-derived
        # over new types, while the defect sits in a pre-existing file edited in place), and
        # a filtered-out issue means refinement never even sees the red test. Every other
        # unowned issue is still dropped — the filter is widened by marker, not by file.
        issues = [i for i in refinement_context.get("qa_issues", [])
                  if i.get("file") in target_names or not i.get("file")
                  or i.get("scope") == "frozen_suite"]
        unowned = sorted({i["file"] for i in issues
                          if i.get("scope") == "frozen_suite" and i.get("file")
                          and i["file"] not in target_names})
        po_notes = refinement_context.get("po_notes", "")
        if issues or po_notes:
            refine_block = "⚠️ FIX THESE PREVIOUSLY-REPORTED ISSUES (keep all other logic intact):\n"
            if issues:
                refine_block += json.dumps(issues, separators=(",", ":")) + "\n"
            if unowned:
                refine_block += (
                    f"NOTE — {', '.join(unowned)} is outside this batch's target files, but the "
                    f"frozen test suite fails because of it. EDIT it in place: do not create a new "
                    f"class, do not move the method elsewhere, and do NOT modify any test file "
                    f"(the tests are frozen). Fix only what makes the operation execute.\n")
            if po_notes:
                refine_block += f"PO rejection notes: {po_notes}\n"
            refine_block += "\n"

            # T3b: embed the failing files' current content so cline doesn't burn
            # exploration iterations re-reading exactly the files it needs to fix
            # (glm-5.1 refinement sessions ran 9-11 iterations, ~4 of which were
            # reads of these same files).
            issue_files = []
            for i in issues:
                f = i.get("file")
                if f and f not in issue_files:
                    issue_files.append(f)
            embedded = []
            for fname in issue_files[:3]:
                rel = blackboard.find_existing(fname)
                if not rel:
                    continue
                try:
                    lines = (blackboard.output_dir / rel).read_text().splitlines()
                except Exception:
                    continue
                # Embed only what fits WHOLE. A truncated embed is worse than none, because
                # the header below tells cline not to re-read the file: rfm's
                # UserRepository.java is 400+ lines and its defect is at :392, so a 250-line
                # window would hide the exact line the round exists to fix. Oversized files
                # are left to cline's own read_file tool (the pre-2026-08-02 issue set was
                # newly-authored files, all comfortably under the cap — this only bites now
                # that frozen-suite routing can name a large pre-existing file).
                if len(lines) > 250:
                    continue
                embedded.append(f"--- {fname} ---\n" + "\n".join(lines))
            if embedded:
                refine_block += ("CURRENT CONTENT OF FILES TO FIX (use this content directly; do not "
                                 "re-read these files):\n" + "\n\n".join(embedded) + "\n\n")

    # VeriMAP: inject per-task VF failure messages so cline self-refines on the exact gap.
    if refinement_context:
        _vf_failures = refinement_context.get("vf_failures", {})
        if _vf_failures:
            _vf_msgs = []
            for _t, _fname in targets:
                _msgs = _vf_failures.get(_t.get("id", ""), [])
                if _msgs:
                    _vf_msgs.append(_fname + ":\n" + "\n".join(f"  - {m}" for m in _msgs))
            if _vf_msgs:
                if not refine_block:
                    refine_block = "⚠️ FIX THESE PREVIOUSLY-REPORTED ISSUES (keep all other logic intact):\n"
                refine_block += ("\nVERIFICATION FAILURES (fix these first — the task cannot be "
                                 "accepted until they pass):\n"
                                 + "\n".join(_vf_msgs) + "\n\n")

    # Lean brief: cline is an agent with read_file/list_files/search_files tools — let it
    # READ the existing code itself instead of pre-injecting the whole tree/file contents
    # /symbol tables and forbidding exploration (the old qwen-era bloat that blinded the
    # model to base classes like BaseEntity → id-type drift). Keep only the invariants
    # that prevent build breaks.
    if is_java:
        placement = (
            "PLACEMENT (Maven): Java under src/main/java/<package-as-dirs>/, tests under "
            "src/test/java/<package-as-dirs>/, resources (application.properties, schema.sql) under "
            "src/main/resources/. File name MUST equal the public top-level type name.\n")
        rules = (
            "1. READ BEFORE WRITING: open the entities/services/controllers you touch AND the base "
            "classes they extend (e.g. BaseEntity). Match the existing id Java type, field names, "
            "packages and conventions EXACTLY — only ADD, never restyle/retype/drop existing code. A "
            "new entity's id AND its scalar foreign keys MUST use the SAME id type the existing "
            "entities use.\n"
            "2. No Lombok (not on the classpath — it breaks the build): write plain constructors and "
            "getters/setters by hand.\n"
            "3. Spring Boot 3 / Jakarta: use jakarta.* (not javax.*) for persistence/validation/"
            "transaction/servlet. Exactly ONE class annotated @SpringBootApplication in the project.\n"
            "4. Exactly ONE public top-level type per file; the file name equals that type's name.\n"
            "5. schema.sql is H2-only (no SERIAL / gen_random_uuid() / CREATE EXTENSION); @Table(name=...) "
            "must match the table names. Never write persistence.xml. Whenever you add a NEW @Entity, "
            "also add its CREATE TABLE to db/h2/schema.sql AND mirror the same DDL to "
            "db/mysql/schema.sql and db/postgres/schema.sql in the same change — ddl-auto=none means a "
            "missing table 500s every endpoint that touches the entity at runtime. If you ADD a column "
            "to an EXISTING table (e.g. a new field on Pet), you MUST also update every INSERT for that "
            "table in db/h2/data.sql (and db/mysql/, db/postgres/) — rewrite positional INSERTs "
            "(INSERT INTO pets VALUES (...)) to explicit column-name form "
            "(INSERT INTO pets (name, birth_date, type_id, owner_id) VALUES (...)) so the legacy seed "
            "data still loads; a column-count mismatch fails schema init and breaks the whole app context.\n"
            "6. Each entity's persistence/query logic lives in its OWN service; controllers and other "
            "services inject and DELEGATE rather than re-implementing it.\n"
            "7. WIRE WHAT YOU WRITE: every public service method you add must be invoked from the "
            "request flow of this milestone (a controller endpoint or another service method that is "
            "itself reachable) — no orphaned helper methods. Computed/derived values (e.g. an invoice "
            "total) must be recalculated wherever the underlying data changes (e.g. when a line item "
            "is added/updated/removed), not just defined and never called.\n"
            "8. Batch ALL initial file reads into ONE read_files call (it accepts multiple paths). Each "
            "separate tool call costs a full conversation resend.\n"
            "9. Do NOT re-read a file you just edited to verify — trust the edit diff. Read only the "
            "files you will modify or directly mirror.\n"
            "10. Implement ONLY the target files listed in TASKS. Do not create additional entities, "
            "services, controllers, or other classes beyond what's listed, even if you think the "
            "feature would benefit from them.\n"
            "11. For files that already exist, prefer a targeted diff/edit over rewriting the whole "
            "file — touch only the lines the task requires and preserve everything else "
            "byte-for-byte.\n")
        if not _is_spring(stack):
            # Non-Spring (Jakarta/JAX-RS) override: keep the framework-neutral rules, replace
            # the Spring-specific assertions (@SpringBootApplication, Spring Data, schema.sql,
            # @Service layering) with JAX-RS/EclipseLink-truthful guidance.
            rules = (
                "1. READ BEFORE WRITING: open the entities/endpoints/repositories you touch AND the "
                "base classes they extend. Match the existing id Java type, field names, packages and "
                "conventions EXACTLY — including the class-naming patterns of sibling files in the "
                "same package (if existing entities are split into a Persistence-prefixed @Entity plus "
                "a plain transport class, a new entity follows that same split and naming). Only ADD, "
                "never restyle/retype/drop existing code. A new entity's id AND its scalar foreign "
                "keys MUST use the SAME id type existing entities use.\n"
                "2. No Lombok (not on the classpath): write plain constructors and getters/setters by "
                "hand.\n"
                "3. This project is JAX-RS (Jersey) on Tomcat — there is NO Spring and NO "
                "@SpringBootApplication. Expose REST via JAX-RS resource classes (@Path/@GET/@POST/"
                "@PUT/@DELETE), mirroring the existing endpoints; register them the way the project "
                "already does. Use jakarta.* (not javax.*) for persistence/servlet APIs.\n"
                "4. Exactly ONE public top-level type per file; the file name equals that type's name.\n"
                "5. Persistence is JPA/EclipseLink, NOT Spring Data — follow the existing repository "
                "pattern (e.g. a singleton *Repository over an EntityManager / the project's "
                "AbstractCRUDEndpoint contract). Do NOT introduce JpaRepository or Spring Data. Do NOT "
                "invent schema.sql/data.sql; match the project's existing DB configuration.\n"
                "6. Keep each entity's persistence/query logic in the project's existing "
                "repository/endpoint structure; reuse shared base classes rather than re-implementing "
                "CRUD.\n"
                "7. WIRE WHAT YOU WRITE: every new method must be reachable from the request flow of "
                "this milestone (a JAX-RS endpoint or a method an endpoint calls) — no orphaned "
                "helpers. Recompute derived values wherever the underlying data changes.\n"
                "8. Batch ALL initial file reads into ONE read_files call (it accepts multiple paths). "
                "Each separate tool call costs a full conversation resend.\n"
                "9. Do NOT re-read a file you just edited to verify — trust the edit diff. Read only the "
                "files you will modify or directly mirror.\n"
                "10. Implement ONLY the target files listed in TASKS. Do not create additional classes "
                "beyond what's listed, even if you think the feature would benefit from them.\n"
                "11. For files that already exist, prefer a targeted diff/edit over rewriting the whole "
                "file — touch only the lines the task requires and preserve everything else "
                "byte-for-byte.\n")
    else:
        placement = ("PLACEMENT: follow the project's existing layout; place modules where imports "
                     "expect them.\n")
        rules = ("1. READ the relevant existing files before editing; match their conventions and call "
                 "only members that actually exist.\n"
                 "2. Extend files in place; preserve unrelated logic. Keep imports resolvable.\n"
                 "3. Batch ALL initial file reads into ONE read_files call (it accepts multiple paths). "
                 "Each separate tool call costs a full conversation resend.\n"
                 "4. Do NOT re-read a file you just edited to verify — trust the edit diff. Read only "
                 "the files you will modify or directly mirror.\n"
                 "5. Implement ONLY the target files listed in TASKS. Do not create additional modules "
                 "or classes beyond what's listed, even if you think the feature would benefit from "
                 "them.\n"
                 "6. For files that already exist, prefer a targeted diff/edit over rewriting the whole "
                 "file — touch only the lines the task requires and preserve everything else "
                 "byte-for-byte.\n")

    # Implementation constraint (dual-track junior "induced-inefficiency" mode): the junior
    # skips the architect, so a brief on the feature text never reaches it — dual_track passes
    # it here as an env var and it is injected verbatim into the coder brief. Empty when unset,
    # so every non-induced run (senior, petclinic, scale-mode) is byte-identical.
    _impl_constraint = os.getenv("AGILE_IMPL_CONSTRAINT", "").strip()
    constraint_block = (f"\n\n{_impl_constraint}\n" if _impl_constraint else "")

    return (
        scope_hdr +
        f"You are working inside an existing {stack.get('framework', stack.get('name',''))} project "
        f"rooted at the current directory.\n"
        "USE YOUR TOOLS (search_codebase, read_files, run_commands) to inspect the existing code before "
        "you write — learn the current entities, their id/field types, packages and conventions, then "
        "make your changes consistent with them. Create or edit ONLY each task's target file, IN PLACE, "
        "and actually WRITE it (the task is unfinished until the target file exists). Do NOT run "
        "mvn/gradle and do NOT touch files outside your targets — a deterministic build+repair step "
        "runs after you and owns whole-project compilation.\n\n"
        f"TASKS:\n{tasks_block}\n\n"
        f"{refine_block}"
        f"{pinned_block}"
        f"CONTRACT (authoritative data model & endpoints for the new feature):\n"
        f"{json.dumps(contract, separators=(',', ':'))}\n\n"
        f"RULES:\n{rules}"
        f"{placement}"
        "Do NOT recreate pom.xml or application.properties if they exist; do not delete unrelated files."
        f"{constraint_block}"
    )


def _validate_entity_semantics(blackboard: Blackboard, tasks: list, results: dict) -> list:
    """Validate that generated Java entities match contract semantics. Returns list of
    (task_id, reason) for failed validations. Checks: ID type strategy, relationship
    vs primitive patterns. Does NOT block the pipeline; logged as warnings for review."""
    import os
    import re
    contract = blackboard.read("contract") or {}
    failures = []

    for task in tasks:
        task_id = task.get("id")
        if task_id not in results or not results[task_id]:
            continue  # Task didn't complete; skip semantic check

        rel_path = results[task_id]
        entity_name = task.get("entity_name") or task.get("title", "").split()[0]
        if not (entity_name and entity_name in {e.get("name") for e in contract.get("entities", [])}):
            continue  # Not an entity task; skip

        try:
            full_path = blackboard.output_dir / rel_path
            content = full_path.read_text()
        except Exception:
            continue  # Can't read file; skip

        entity_spec = next(
            (e for e in contract.get("entities", []) if e.get("name") == entity_name),
            None
        )
        if not entity_spec:
            continue

        # Check ID type strategy — ONLY when the contract explicitly pins one.
        # The model owns id-type decisions (matching the existing model), so an
        # unset strategy means "don't police it" rather than "must be UUID".
        inferred_strategy = entity_spec.get("_id_strategy")
        if inferred_strategy == "UUID":
            if "GenerationType.IDENTITY" in content:
                failures.append((task_id, f"Entity {entity_name} uses IDENTITY; expected UUID"))
            if re.search(r"private\s+Long\s+id\s*;", content):
                failures.append((task_id, f"Entity {entity_name} uses Long id; expected UUID"))
        elif inferred_strategy == "IDENTITY":
            if "GenerationType.UUID" in content:
                failures.append((task_id, f"Entity {entity_name} uses UUID; expected IDENTITY"))

        # Check for primitive foreign keys instead of relationships
        for field_name, field_type in entity_spec.get("fields", {}).items():
            if field_name.endswith("Id") and field_name != "id" and field_type != "string":
                # Should be @ManyToOne relationship, not primitive
                if re.search(rf"private\s+{re.escape(field_type)}\s+{re.escape(field_name)}\s*;", content):
                    parent = field_name[:-2]  # petId -> pet
                    if "@ManyToOne" not in content or f"private {parent[0].upper() + parent[1:]} {field_name[:-2]}" not in content:
                        failures.append(
                            (task_id, f"Field {field_name} is primitive; should be @ManyToOne "
                             f"relationship to {parent[0].upper() + parent[1:]}")
                        )

    return failures


def _run_cline_for_tasks(tasks: list, blackboard: Blackboard, stack: dict,
                         refinement_context: dict = None) -> dict:
    """Run ONE or more agentic cline session(s) for `tasks`. Separates test generation into
    simpler per-entity sessions to avoid brief explosion. Returns {task_id: resolved_path}."""
    from agile_team_cline.cline_runner import run_cline_with_retry
    import os
    output_dir = os.path.abspath(str(blackboard.output_dir))

    # Separate test tasks from core (entity/service/controller) tasks
    is_java = stack.get("language", "").lower() == "java"
    core_tasks = [t for t in tasks if not (is_java and ("test" in t.get("type", "").lower()))]
    test_tasks = [t for t in tasks if is_java and "test" in t.get("type", "").lower()]

    # AGILE_AUTHOR_TESTS=0 (default) means NO test authoring — and that must also cover
    # type:test tasks the LLM sprint-planner emits, not just the deterministic test gate.
    # Otherwise these run slow per-file 600s+ sessions in front of the feature work
    # (TeaStore run #3 burned ~6h on LLM-planned test tasks that timed out).
    if test_tasks and os.getenv("AGILE_AUTHOR_TESTS", "0") != "1":
        print(f"   ⏭️  AGILE_AUTHOR_TESTS=0 — skipping {len(test_tasks)} LLM-planned test task(s)")
        test_tasks = []

    results = {}

    # Process core (entity/service/controller) tasks in one session
    if core_tasks:
        results.update(_run_cline_session_for_tasks(core_tasks, blackboard, stack, refinement_context))

    # Process test tasks individually (smaller briefs = better completion rates)
    if test_tasks:
        for test_task in test_tasks:
            before = blackboard.snapshot_src()
            brief = _build_simple_test_brief(test_task, blackboard, stack)
            output_dir = os.path.abspath(str(blackboard.output_dir))
            result = run_cline_with_retry(output_dir, brief, title=test_task["title"],
                                          timeout=int(os.getenv("AGILE_TEST_CLINE_TIMEOUT", "1800")), thinking=False)

            blackboard.record_changed_since(before)
            fname = test_task.get("target_file") or _title_to_filename(test_task["title"], stack["file_extension"])
            rel = blackboard.find_existing(fname)
            if rel:
                results[test_task["id"]] = rel
                blackboard.clear_failure(test_task["id"])
            else:
                results[test_task["id"]] = ""
                blackboard.record_failure(test_task["id"])
                print(f"   ⚠️ Test file {fname} not generated for {test_task.get('title', '')}; "
                      f"this is a known cline limitation. Continuing with other tests.")

    return results


def _build_simple_test_brief(test_task: dict, blackboard: Blackboard, stack: dict) -> str:
    """Build a minimal test-generation brief for a single test file. Embed entity/
    service/repository/controller content so cline doesn't have to discover them,
    which saves time and reduces timeout risk. Requirements branch on `test_task["kind"]`
    ('unit' | 'webmvc' | 'datajpa' | 'integration') so the brief never asks for both
    @SpringBootTest and Mockito mocking, and integration tests are explicitly allowed
    for the 'integration' kind."""
    import re
    # entity_name = the class under test, derived from the authoritative target_file.
    # NOT from the verbose title: stripping "Test" out of a ~400-char title yields a
    # garbage string that poisons both the embed lookup and the write path.
    import os
    target_file = test_task.get("target_file") or _title_to_filename(test_task["title"], stack["file_extension"])
    kind = test_task.get("kind", "unit")
    if test_task.get("entity_name"):
        entity_name = test_task["entity_name"]
    elif test_task.get("resource_name"):
        entity_name = test_task["resource_name"]
    elif target_file.endswith("Test.java"):
        entity_name = os.path.basename(target_file)[:-len("Test.java")]
    elif target_file.endswith(".java"):
        entity_name = os.path.basename(target_file)[:-len(".java")]
    else:
        entity_name = os.path.basename(target_file)
    test_class_name = os.path.basename(target_file)[:-len(".java")]

    # Locate related classes on disk to (a) embed their source and (b) derive the real
    # package — services/controllers live in sub-packages (.vet, .owner), not the root
    # petclinic package. find_existing matches by basename across the tree.
    src_rel = blackboard.find_existing(f"{entity_name}.java")
    svc_rel = blackboard.find_existing(f"{entity_name}Service.java")
    repo_rel = blackboard.find_existing(f"{entity_name}Repository.java")
    ctrl_rel = (blackboard.find_existing(f"{entity_name}Controller.java")
                or blackboard.find_existing(f"{entity_name}RestController.java"))

    def _read(rel):
        if not rel:
            return ""
        try:
            return (blackboard.output_dir / rel).read_text()
        except Exception:
            return ""

    context_parts = []
    for label, rel in ((f"{entity_name}.java", src_rel),
                        (f"{entity_name}Service.java", svc_rel),
                        (f"{entity_name}Repository.java", repo_rel),
                        (os.path.basename(ctrl_rel) if ctrl_rel else "", ctrl_rel)):
        content = _read(rel)
        if content:
            context_parts.append(f"=== {label} ===\n{content}")
    context_block = "\n\n".join(context_parts) + "\n\n" if context_parts else ""

    # Derive the test path + package from the located source relpath so the test lands
    # in the SAME package as the class under test. Fall back to the root package (Spring
    # only — the jakarta/JAX-RS stack has no fixed root package to fall back to, and
    # entity_name always resolves to a real on-disk file for the 'repository' kind since
    # the feature adds a method to an EXISTING repository class).
    default_pkg = "org.springframework.samples.petclinic" if _is_spring(stack) else ""
    package = default_pkg
    test_path = (f"src/test/java/{default_pkg.replace('.', '/')}/{test_class_name}.java"
                 if default_pkg else f"src/test/java/{test_class_name}.java")
    pkg_src = src_rel or svc_rel or ctrl_rel
    if pkg_src:
        m = re.search(r"src/main/java/(.+)/[^/]+\.java$", pkg_src.replace("\\", "/"))
        if m:
            pkg_path = m.group(1)
            package = pkg_path.replace("/", ".")
            test_path = f"src/test/java/{pkg_path}/{test_class_name}.java"

    if not _is_spring(stack):
        # Jakarta/JAX-RS (TeaStore): no Spring test slices exist to reuse (@WebMvcTest/
        # @DataJpaTest/@SpringBootTest all require Spring Boot on the classpath, which
        # this stack does not have). Every kind collapses to ONE recipe: a plain JUnit 4
        # test that directly calls the repository method(s) this task is about. This is
        # what makes the row pass the dual-track feature_exercised gate
        # (agile_team_cline/dual_track.py _feature_exercised) — the Kieker weave scope
        # profiles *Repository classes, so the test must actually invoke the new method,
        # not just construct the repository.
        exemplar_rel = blackboard.find_existing("RepositoryTest.java")
        emf_rel = blackboard.find_existing("EMFManagerInitializer.java")
        for label, rel in (("RepositoryTest.java (structural exemplar — copy this pattern)",
                            exemplar_rel),
                           ("EMFManagerInitializer.java", emf_rel)):
            content = _read(rel)
            if content:
                context_parts.append(f"=== {label} ===\n{content}")
        context_block = "\n\n".join(context_parts) + "\n\n" if context_parts else ""

        # TeaStore convention: persistence-tier tests sit in the module's ROOT test
        # package (RepositoryTest.java is `package tools.descartes.teastore.persistence;`,
        # NOT the `.domain` subpackage the repository classes themselves live in). Prefer
        # the exemplar's own package over the source class's package derived above.
        if exemplar_rel:
            m = re.search(r"src/test/java/(.+)/[^/]+\.java$", exemplar_rel.replace("\\", "/"))
            if m:
                pkg_path = m.group(1)
                package = pkg_path.replace("/", ".")
                test_path = f"src/test/java/{pkg_path}/{test_class_name}.java"

        common_tail_jakarta = (
            "- Import ONLY classes that actually exist in the reference code above\n"
            "- Construct test data with the EXACT types and constructors from the reference "
            "code (e.g. if an id field is String use a String literal, if UUID use "
            "UUID.randomUUID(), if Long/Integer use a numeric literal — never substitute one "
            "for another)\n"
            "- Use proper assertions: org.junit.Assert.assertEquals, assertTrue, assertNotNull\n"
            f"- MUST write the complete file to this exact path: {test_path}\n"
            "- Do NOT stop until the file is written to disk."
        )
        requirements = (
            "- Use plain JUnit 4 ONLY: import org.junit.Test, org.junit.Before, org.junit.Assert. "
            "Do NOT use JUnit 5 / org.junit.jupiter.*, Mockito, MockMvc, or any Spring/CDI "
            "annotation — @RunWith, @ExtendWith, @Autowired, @SpringBootTest, @WebMvcTest, "
            "@DataJpaTest do NOT apply to this project (it is plain JAX-RS + JPA/EclipseLink, "
            "no Spring on the classpath)\n"
            "- In an @Before method, call "
            "tools.descartes.teastore.persistence.repository.EMFManagerInitializer"
            ".initializeEMF() to set up the in-memory test database, exactly as the project's "
            "own RepositoryTest.java does\n"
            f"- The test MUST directly invoke the repository method(s) this task is about on "
            f"{entity_name}.REPOSITORY (the project's singleton-repository access pattern) and "
            "assert on the returned values — do not merely construct/reference the repository, "
            "actually CALL the method under test with realistic arguments\n"
            "- Follow the exact structural pattern of RepositoryTest.java above (and "
            "EMFManagerInitializer.java if present): singleton repository access via "
            "REPOSITORY, an @Before method calling EMFManagerInitializer.initializeEMF(), "
            "plain JUnit 4 @Test methods\n"
            + common_tail_jakarta
        )
        return (
            f"Write a single test file: {test_path}\n\n"
            f"REFERENCE CODE (already exists on disk):\n{context_block}\n"
            "Test requirements:\n"
            f"- Test class name: {test_class_name}\n"
            f"- Package: {package}\n"
            f"{requirements}"
        )

    common_tail = (
        "- Import ONLY classes that actually exist in the reference code above\n"
        "- Construct test data with the EXACT types and constructors from the reference code "
        "(e.g. if an id field is String use a String literal, if UUID use UUID.randomUUID(), "
        "if Long/Integer use a numeric literal — never substitute one for another)\n"
        "- Use proper assertions: assertEquals, assertTrue, assertThrows\n"
        "- Spring Boot 4 ships Jackson 3: for JSON serialization import "
        "tools.jackson.databind.ObjectMapper — NEVER com.fasterxml.jackson.databind "
        "(Jackson 2 is not on the classpath and the test will not compile)\n"
        f"- MUST write the complete file to this exact path: {test_path}\n"
        "- Do NOT stop until the file is written to disk."
    )

    if kind == "webmvc":
        requirements = (
            f"- Use JUnit 5 with @WebMvcTest({entity_name}.class)\n"
            "- Use @MockBean for every service the controller depends on\n"
            "- Autowire MockMvc; drive each declared endpoint via perform(...), asserting "
            "status codes and JSON response shape for a representative case\n"
            "- Do NOT use @SpringBootTest; do NOT start the full application context\n"
            "- Do NOT test Spring framework code; only test this controller's request handling\n"
            + common_tail
        )
    elif kind == "datajpa":
        requirements = (
            "- Use JUnit 5 with @DataJpaTest\n"
            f"- Autowire the REAL {entity_name}Repository — do NOT mock it; this test runs "
            "against the project's actual H2 database with schema.sql applied\n"
            f"- Save a new {entity_name} instance via the repository, then assert it can be "
            "found by id afterwards with its persisted fields matching what was saved\n"
            "- Write 2-4 test methods (save+find, plus any other repository finder methods "
            "that exist)\n"
            + common_tail
        )
    elif kind == "integration":
        contract = blackboard.read("contract") or {}
        endpoints = []
        try:
            from agile_team_cline.ceremonies.planning import _resource_controllers, _controller_endpoints, entity_names
            entities = entity_names(contract)
            endpoints = _controller_endpoints(contract, entity_name, entities)
        except Exception:
            pass
        endpoints_block = ("\n".join(f"  {e}" for e in endpoints)) if endpoints else "  (see contract below)"
        requirements = (
            "- Use JUnit 5 with @SpringBootTest and @AutoConfigureMockMvc; autowire MockMvc\n"
            f"- This IS an integration test — do not mock collaborators\n"
            f"- Drive a realistic create-then-read flow over these endpoints:\n{endpoints_block}\n"
            "- Use a POST/PUT to create or update data, then a GET to read it back\n"
            "- Assert 2xx statuses AND that any computed/derived field in the GET response "
            "(e.g. a total, count, or aggregate) reflects the data just written "
            "(non-zero/non-default where applicable) — not just that the field is present\n"
            f"- Contract for this resource: {json.dumps(contract, separators=(',', ':'))[:1500]}\n"
            + common_tail
        )
    else:  # 'unit' (service)
        requirements = (
            "- Use JUnit 5 with Mockito (@ExtendWith(MockitoExtension.class), @Mock its "
            "repositories/collaborators, @InjectMocks the service)\n"
            "- Do NOT use @SpringBootTest\n"
            "- Write 5-8 test methods covering: create, read, update, delete, and edge cases\n"
            "- Do NOT write integration tests; focus on unit tests\n"
            "- Do NOT test Spring framework code; only test business logic\n"
            + common_tail
        )

    return (
        f"Write a single test file: {test_path}\n\n"
        f"REFERENCE CODE (already exists on disk):\n{context_block}\n"
        "Test requirements:\n"
        f"- Test class name: {test_class_name}\n"
        f"- Package: {package}\n"
        f"{requirements}"
    )


def _run_cline_session_for_tasks(tasks: list, blackboard: Blackboard, stack: dict,
                                  refinement_context: dict = None) -> dict:
    """Internal: run a single cline session for a list of core (non-test) tasks."""
    from agile_team_cline.cline_runner import run_cline_with_retry
    import os
    output_dir = os.path.abspath(str(blackboard.output_dir))

    # (#2) Don't re-issue a task whose target file already exists. Creation/coverage
    # sprints re-generating a present file is the main non-convergence multiplier;
    # refinement DELIBERATELY re-edits existing files, so it is exempt. If nothing is
    # left to build, skip the cline session (and its cold-start overhead) entirely.
    if refinement_context:
        pending = list(tasks)
    else:
        # (#2/E) A task is "already satisfied" only when THIS run actually wrote its
        # target — not merely when the file exists on disk. In brownfield, generated_files
        # is pre-seeded with the whole existing tree (ScrumMaster.init_sprint), so a raw
        # find_existing() match on an UNTOUCHED pre-existing file (e.g. an "add a method to
        # ProductRepository.java" task) would FALSE-SKIP the session and report green while
        # writing nothing (TeaStore perf runs #33-35, 2026-06-23). Compare each target
        # against the pre-development baseline snapshot: a target counts as produced only
        # when it is NEW (absent from baseline) or MODIFIED since it (mtime/size changed).
        # Untouched baseline files stay pending so cline runs. Greenfield is unaffected
        # (its baseline is empty/scaffold, so newly-created files still dedupe as before).
        _cur = blackboard.snapshot_src()
        _base = blackboard.read("baseline_src") or {}

        def _produced_this_run(t):
            fname = t.get("target_file") or _title_to_filename(t["title"], stack["file_extension"])
            rel = blackboard.find_existing(fname)
            if not rel:
                return False  # not on disk → must build
            prev, meta = _base.get(rel), _cur.get(rel)
            return prev is None or (meta is not None and list(prev) != list(meta))

        pending = [t for t in tasks if not _produced_this_run(t)]

    if not pending:
        results = {}
        for task in tasks:
            fname = task.get("target_file") or _title_to_filename(task["title"], stack["file_extension"])
            rel = blackboard.find_existing(fname)
            results[task["id"]] = rel
            if rel:
                blackboard.clear_failure(task["id"])
        print(f"   ⏭️  cline: all {len(tasks)} target file(s) already present — skipping session")
        return results

    brief = _build_cline_brief(pending, blackboard, stack, refinement_context)
    before = blackboard.snapshot_src()
    title = pending[0]["title"] if len(pending) == 1 else f"{len(pending)} tasks"
    result = run_cline_with_retry(output_dir, brief, title=title)

    # Record every new/modified project file (append-ordered, deduped) so baseline
    # slicing, QA, coverage and the last-green guard keep seeing an accurate list.
    blackboard.record_changed_since(before)

    # (#3) TASK COMPLETION VALIDATION: Check that every assigned task has its target
    # file on disk. Cline sessions can return success while partially completing tasks
    # (e.g., creating 4/6 entities but stopping). This is a critical safeguard: if any
    # task lacks its target file after cline finishes, fail hard instead of proceeding
    # with incomplete work that breaks downstream sprints.
    results = {}
    incomplete = []
    for task in pending:
        fname = task.get("target_file") or _title_to_filename(task["title"], stack["file_extension"])
        rel = blackboard.find_existing(fname)
        if rel:
            results[task["id"]] = rel
            blackboard.clear_failure(task["id"])
        else:
            incomplete.append((task["id"], fname, task.get("title", "")))
            results[task["id"]] = ""
            blackboard.record_failure(task["id"])

    if incomplete:
        # Task completion failure: cline did not write all target files despite the
        # explicit instruction. This is a hard failure because downstream sprints
        # depend on all entities/repos/services existing.
        incomplete_str = "\n   ".join(
            f"✗ {task_id}: {fname} — {title}" for task_id, fname, title in incomplete)
        print(f"   ❌ TASK COMPLETION FAILURE: cline did not write {len(incomplete)} target file(s):")
        print(f"   {incomplete_str}")
        print(f"   → This is a critical issue: downstream sprints depend on all tasks completing.")
        print(f"   → cline session marked tasks as failed; proceeding with incomplete work will cause")
        print(f"   → compilation errors and broken references.")
        if not result.success:
            print(f"   → cline session also reported finish_reason={result.finish_reason}")
        # Return early with failures marked so downstream detects the broken state
        return results

    # All pending tasks completed successfully; now validate semantic correctness
    semantic_failures = _validate_entity_semantics(blackboard, pending, results)
    if semantic_failures:
        print(f"   ⚠️ SEMANTIC VALIDATION WARNINGS ({len(semantic_failures)} issue(s) found):")
        for task_id, reason in semantic_failures:
            print(f"      • {task_id}: {reason}")
        print(f"   → These will be caught by compilation/QA; proceeding with caution.")

    # All pending tasks completed successfully
    if not result.success:
        print(f"   ⚠️ cline session finish_reason={result.finish_reason} but all {len(pending)} "
              f"target file(s) exist; continuing with caution")
    return results


def _run_dev_sprint(tasks: list, blackboard: Blackboard, stack: dict, model, refinement_context: dict = None) -> dict:
    """Delegate code authoring/editing to the Cline CLI. `model` is accepted for
    signature compatibility (callers pass the camel model) but unused — cline runs
    its own provider. Granularity is per-sprint by default (one session for all
    tasks); AGILE_CLINE_GRANULARITY=task runs one session per task for A/B."""
    import os
    if not tasks:
        return {}
    granularity = os.getenv("AGILE_CLINE_GRANULARITY", "sprint")
    if granularity == "task":
        results = {}
        for task in tasks:
            results.update(_run_cline_for_tasks([task], blackboard, stack, refinement_context))
    else:
        results = _run_cline_for_tasks(tasks, blackboard, stack, refinement_context)

    # VeriMAP: run per-task VFs and store failures on blackboard for acceptance intersect.
    # mode="shadow" logs only; mode="1" enforces rejection. No-op when AGILE_VERIFY_AWARE=0.
    _vf_mode = os.getenv("AGILE_VERIFY_AWARE", "0")
    if _vf_mode != "0":
        try:
            from agile_team_cline.verify_lib import run_task_vfs
            vf_failures = run_task_vfs(tasks, blackboard, stack, _vf_mode)
            blackboard.write("vf_failures", vf_failures)
        except Exception as _vf_exc:
            print(f"   ⚠️ VF check error (non-blocking): {_vf_exc}")

    return results


def _restore_missing_poms(output_dir: str) -> list:
    """Tree-integrity guard before sprint-side builds. Delegates to repair.py's
    _ensure_tree_integrity (pom snapshot + baseline manifest) so every build path —
    repair._compile_raw/_compile_main and the sprint-side mvn gates — runs the same
    guards. Name kept for existing callers; returns [] (callers ignore the value)."""
    from agile_team_cline.repair import _ensure_tree_integrity
    _ensure_tree_integrity(output_dir)
    return []


def _run_maven_compile(output_dir: str) -> list:
    """Runs `mvn clean test-compile` in output_dir and parses any compilation errors.
    Scoped to AGILE_BUILD_MODULE (+ -am) when set, so a multi-module reactor compiles
    only the feature's subtree instead of every module — otherwise a pre-existing
    baseline failure in an unrelated module (e.g. TeaStore's `image` module) gets
    misattributed to the feature and burns refinement rounds."""
    import os
    import subprocess

    _restore_missing_poms(output_dir)

    pom_path = os.path.join(output_dir, "pom.xml")
    if not os.path.exists(pom_path):
        return []

    timeout = int(os.getenv("AGILE_MVN_TIMEOUT", "300"))
    print("🛠️ Running `mvn clean test-compile` to physically verify the build...")
    try:
        from agile_team_cline.repair import _mvn_compile_argv
        res = subprocess.run(
            _mvn_compile_argv(["clean", "test-compile"]),
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print("⚠️ Maven compile check timed out.")
        return [{"file": "pom.xml", "concern": f"Maven compilation timed out after {timeout} seconds."}]
    except Exception as e:
        print(f"⚠️ Failed to run maven compile: {e}")
        return []

    return _parse_maven_issues(res.stdout, output_dir, res.returncode)


def _run_maven_tests(output_dir: str) -> dict:
    """Physically EXECUTE the test suite with `mvn test` (no clean — reuses the tree
    repair already compiled) and parse the Surefire summary, so the PO judges on real
    execution results instead of the QA model guessing "zero passing tests". Returns
    {ran, tests_run, tests_passed, tests_failed, issues}. Assertion/error failures
    become file-scoped issues mapped to <FailingTestClass>.java so refinement has a
    concrete target — kept DISTINCT from compile errors and never folded into
    report["compiles"] (the last-green guard must keep meaning "compiles", not "green
    tests", or a red test would trigger a rollback)."""
    import os
    import subprocess

    _restore_missing_poms(output_dir)

    pom_path = os.path.join(output_dir, "pom.xml")
    if not os.path.exists(pom_path):
        return {"ran": False, "tests_run": 0, "tests_passed": 0, "tests_failed": 0, "issues": []}

    timeout = int(os.getenv("AGILE_MVN_TEST_TIMEOUT", "600"))
    print("🧪 Running `mvn test` to physically execute the test suite...")
    try:
        from agile_team_cline.repair import _mvn_compile_argv
        res = subprocess.run(
            # -DtrimStackTrace=false: surefire's default trimming deletes every frame
            # between the test method and the throw site, which is precisely the range
            # that names the production file at fault. Trimmed, the rfm 2026-08-02 trace
            # showed ONLY CustomerRfmSegmentRepositoryTest.java:251; untrimmed it carries
            # 49 frames whose first project frame is UserRepository.getRfmSegments — the
            # actual defect. _parse_surefire below cannot route a fix without this.
            _mvn_compile_argv(["test"]) + ["-DtrimStackTrace=false"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"⚠️ Maven test run timed out after {timeout}s.")
        return {"ran": False, "tests_run": 0, "tests_passed": 0, "tests_failed": 0,
                "issues": [{"file": "pom.xml", "concern": f"Test execution timed out after {timeout}s."}]}
    except Exception as e:
        print(f"⚠️ Failed to run maven test: {e}")
        return {"ran": False, "tests_run": 0, "tests_passed": 0, "tests_failed": 0, "issues": []}

    return _parse_surefire(res.stdout, output_dir)


def _parse_surefire(stdout: str, output_dir: str = None) -> dict:
    """Pull Surefire's aggregate totals + map each failing test back to its source
    file. The LAST 'Tests run: …' line is Surefire's final 'Results:' grand total.

    `output_dir` (optional) enables production-file attribution: a failing test is
    normally reported against <FailingTestClass>.java, but the refinement brief is
    filtered to the current tasks' target files, so a defect in a file that owns no
    backlog task is unreachable — refinement literally cannot open it (rfm 2026-08-02:
    two rounds, six red tests, and UserRepository.java never touched because the only
    task was the DTO). When given, consult the surefire XML's stack traces and re-point
    such an issue at the file that actually threw, marking it scope="frozen_suite" so
    _build_cline_brief lets it through the ownership filter.

    Omitting `output_dir` keeps the old behaviour exactly — only thrown exceptions with
    a resolvable production frame are ever re-pointed; everything else still reports
    against the test class with no marker."""
    import re
    owners = {}
    if output_dir:
        try:
            from agile_team_cline.verify_lib import owning_source_files
            owners = owning_source_files(output_dir)
        except Exception as e:  # attribution is a bonus; never fail the test gate on it
            print(f"⚠️ Could not attribute test failures to source files: {e}")
    totals = re.findall(
        r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)",
        stdout)
    if not totals:
        # Surefire produced no parseable summary (e.g. no test sources discovered).
        return {"ran": True, "tests_run": 0, "tests_passed": 0, "tests_failed": 0, "issues": []}

    run, fail, err, skip = (int(x) for x in totals[-1])
    failed = fail + err
    passed = max(run - failed - skip, 0)
    issues = []
    if failed:
        print(f"❌ Tests: {run} run, {failed} failed, {skip} skipped.")
        seen = set()
        for cls, method in re.findall(r"([A-Z]\w*(?:Test|Tests|IT))\.(\w+)", stdout):
            key = (cls, method)
            if key in seen:
                continue
            seen.add(key)
            owner = owners.get(f"{cls}.java")
            if owner:
                src, exc = owner
                issues.append({
                    "file": src,
                    "scope": "frozen_suite",
                    "concern": (f"Test failure: {cls}.{method} did not pass. {src} THREW at run "
                                f"time — {exc}. Fix {src} so the operation runs to completion. "
                                f"This is about the code EXECUTING, not about how fast it is."),
                })
            else:
                issues.append({
                    "file": f"{cls}.java",
                    "concern": (f"Test failure: {cls}.{method} did not pass. Fix the code under "
                                f"test (or the assertion if it tests behavior the contract does not "
                                f"require) so the suite is green."),
                })
        if not issues:
            issues.append({"file": "pom.xml",
                           "concern": f"{failed} test(s) failed — see Surefire output."})
    else:
        print(f"✅ Tests: {run} run, all passed ({skip} skipped).")
    return {"ran": True, "tests_run": run, "tests_passed": passed,
            "tests_failed": failed, "issues": issues}


def _parse_maven_issues(stdout: str, output_dir: str, returncode: int) -> list:
    """Transform `mvn` output into report issues (with cross-file drift
    enrichment). All low-level [ERROR] parsing + owner/drift detection lives in
    ONE place — repair.parse_maven_errors — so this and the deterministic repair
    pass can never drift apart. Split out from _run_maven_compile so the repair
    pass can hand its already-run compile output straight here instead of
    recompiling the identical tree."""
    if returncode == 0:
        print("✅ Maven build compiled successfully!")
        return []

    print("❌ Maven build failed. Parsing compilation errors...")
    from agile_team_cline.repair import parse_maven_errors

    issues = []
    drift = {}  # consumer filename -> set of referenced owner type simple-names
    for rec in parse_maven_errors(stdout):
        if rec["kind"] == "goal_failure":
            issue = {"file": "pom.xml", "concern": rec["headline"]}
            if issue not in issues:
                issues.append(issue)
            continue

        filename = rec["file"]
        if rec["col"]:
            line_info = f" at line {rec['line']}, col {rec['col']}"
        elif rec["line"]:
            line_info = f" at line {rec['line']}"
        else:
            line_info = ""
        concern = f"Compilation Error{line_info}: {rec['headline']}"
        if not any(it["file"] == filename and it["concern"] == concern for it in issues):
            issues.append({"file": filename, "concern": concern})

        for t in rec["owners"]:
            if f"{t}.java" != filename:
                drift.setdefault(filename, set()).add(t)

    # For each consumer with drift, append ONE enriched issue carrying the real
    # public interface of the referenced type(s), so refinement makes the caller
    # match the producer (which is contract-anchored) instead of inventing members.
    for fname, owner_types in drift.items():
        sigs = []
        for t in sorted(owner_types):
            summary = _get_structural_summary(output_dir, f"{t}.java")
            if summary:
                sigs.append(f"{t} actual public interface:\n{summary}")
        if sigs:
            issues.append({
                "file": fname,
                "concern": ("CROSS-FILE DRIFT: this file calls members that the referenced type(s) "
                            "do not declare. Change THIS file to use ONLY the real members listed "
                            "below — do not invent accessors and do not change the referenced "
                            "types.\n" + "\n\n".join(sigs)),
            })

    # If no specific errors parsed but compile failed, append general error
    if not issues:
        issues.append({"file": "pom.xml", "concern": "Maven compilation failed with non-zero exit code. Check project configuration and source files."})

    return issues


import re as _re_hygiene

# javax.* packages that moved to jakarta.* under Spring Boot 3 / Jakarta EE 9+.
_JAVAX_TO_JAKARTA = ("persistence", "validation", "transaction", "annotation", "servlet")
_PUBLIC_TYPE_RE = _re_hygiene.compile(
    r"\bpublic\s+(?:final\s+|abstract\s+|sealed\s+|non-sealed\s+)*"
    r"(?:class|interface|enum|record)\s+([A-Za-z_]\w*)")


def _javax_to_jakarta(code: str) -> str:
    for pkg in _JAVAX_TO_JAKARTA:
        code = code.replace(f"javax.{pkg}", f"jakarta.{pkg}")
    return code


def _java_public_type(src: str) -> str:
    m = _PUBLIC_TYPE_RE.search(src)
    return m.group(1) if m else ""


import re as _re

# PostgreSQL-only idioms the LLM emits in schema.sql that fail on the H2 in-memory
# DB the generated apps default to. Column defaults are unnecessary anyway because
# entity ids use JPA @GeneratedValue, so stripping them is safe.
_PG_DEFAULT_UUID_RE = _re.compile(r"\s+DEFAULT\s+(?:gen_random_uuid|uuid_generate_v4)\s*\(\s*\)", _re.IGNORECASE)
_PG_EXTENSION_RE = _re.compile(r"^\s*CREATE\s+EXTENSION[^;]*;\s*$\n?", _re.IGNORECASE | _re.MULTILINE)


def _h2_safe_sql(sql: str) -> str:
    """Strip PostgreSQL-only constructs so schema.sql runs on H2."""
    sql = _PG_EXTENSION_RE.sub("", sql)
    sql = _PG_DEFAULT_UUID_RE.sub("", sql)
    return sql


def _run_source_hygiene(blackboard: Blackboard, stack: dict) -> list:
    """Deterministic source-tree fixes that prompt rules can't reliably enforce.
    Runs before every Java compile so the build/PO see a clean tree:
      1. javax.* -> jakarta.* (Spring Boot 3 invariant).
      2. Exactly one @SpringBootApplication (delete extra entry points).
      3. A public type's file MUST be named after it (javac hard requirement).
      4. Drop test-source files that duplicate a main-source class FQN.
      5. Drop persistence.xml + scrub PostgreSQL idioms (Spring Boot config invariant).
    Mutates the blackboard's generated_files list to track renames/removals."""
    import os
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if not is_java:
        return []

    output_dir = str(blackboard.output_dir)
    bb = blackboard.read("blackboard")
    gen = bb.get("generated_files", [])
    changes = []

    def ap(rel):
        return os.path.join(output_dir, rel)

    def is_test(rel):
        return "src/test/" in rel.replace("\\", "/")

    # 1. javax -> jakarta, scoped to AGILE_BUILD_MODULE (+ its declared deps) when set.
    # `gen` is brownfield-pre-seeded with the WHOLE workspace tree (not just this
    # feature's files — see project_brownfield_false_skip_fix.md), so an unscoped rewrite
    # touches every javax hit in the repo: TeaStore runs rewrote 5 unrelated files in
    # modules outside the build scope (kieker.rabbitmq servlets, webui/ErrorServlet) that
    # then rode along into both dual-track workspaces and, in chained-history mode, into
    # the trunk as part of an unrelated feature's "commit". Module paths are relative
    # (e.g. "services/tools.descartes.teastore.persistence"), matching AGILE_BUILD_MODULE's
    # own convention (sprint.py:198). Unset -> unscoped, unchanged behavior.
    _scope_mods = [m.strip() for m in (
        os.getenv("AGILE_BUILD_MODULE", "") + "," + os.getenv("AGILE_BUILD_MODULE_DEPS", "")
    ).split(",") if m.strip()]

    def _in_scope(rel: str) -> bool:
        if not _scope_mods:
            return True
        relp = rel.replace("\\", "/")
        return any(relp == m or relp.startswith(m + "/") for m in _scope_mods)

    for rel in gen:
        if not rel.endswith(".java"):
            continue
        p = ap(rel)
        if not os.path.exists(p):
            continue
        try:
            src = open(p).read()
        except Exception:
            continue
        fixed = _javax_to_jakarta(src)
        if fixed != src:
            if not _in_scope(rel):
                print(f"   ⚠️ javax→jakarta hit outside build scope, NOT rewritten: {rel}",
                      flush=True)
                continue
            open(p, "w").write(fixed)
            changes.append(f"javax→jakarta:{os.path.basename(rel)}")

    # 2. Single @SpringBootApplication entry point, then align ONLY the kept
    #    bootstrap file to its class name. Filename↔class alignment is restricted
    #    to the bootstrap class because the pipeline is filename-authoritative for
    #    every other layer (coverage builds <Entity>.java / <Entity>Service.java) —
    #    renaming a service/entity file to a mis-declared class name corrupts the
    #    layer. The bootstrap class name is arbitrary and unreferenced, so aligning
    #    it is safe and clears the "class X should be in X.java" javac error.
    main_apps = []
    for rel in gen:
        if not rel.endswith(".java") or is_test(rel):
            continue
        p = ap(rel)
        try:
            if os.path.exists(p) and "@SpringBootApplication" in open(p).read():
                main_apps.append(rel)
        except Exception:
            pass
    keep = sorted(main_apps)[0] if main_apps else None
    for rel in main_apps:
        if rel == keep:
            continue
        try:
            os.remove(ap(rel))
        except Exception:
            pass
        if rel in gen:
            gen.remove(rel)
        changes.append(f"removed duplicate @SpringBootApplication:{os.path.basename(rel)}")
    if keep and os.path.exists(ap(keep)):
        tname = _java_public_type(open(ap(keep)).read())
        stem = os.path.splitext(os.path.basename(keep))[0]
        newrel = os.path.join(os.path.dirname(keep), f"{tname}.java")
        if tname and tname != stem and not os.path.exists(ap(newrel)):
            try:
                os.rename(ap(keep), ap(newrel))
                gen[gen.index(keep)] = newrel
                changes.append(f"renamed bootstrap {os.path.basename(keep)}→{tname}.java")
            except Exception:
                pass
    elif not main_apps and _is_spring(stack):
        # Zero entry points generated — the app compiles but cannot boot. Synthesize
        # a minimal @SpringBootApplication in the project's package so it is runnable.
        # Spring-only: a non-Spring app (JAX-RS/servlet) has no @SpringBootApplication
        # and synthesizing one corrupts the tree (see TeaStore e2e run #1).
        pkg = ""
        for rel in gen:
            if rel.endswith(".java") and not is_test(rel) and os.path.exists(ap(rel)):
                for line in open(ap(rel)):
                    line = line.strip()
                    if line.startswith("package ") and line.endswith(";"):
                        pkg = line[8:-1].strip()
                        break
            if pkg:
                break
        newrel = os.path.join("src", "main", "java", pkg.replace(".", "/"), "Application.java") if pkg else ""
        if pkg and not os.path.exists(ap(newrel)):
            code = (f"package {pkg};\n\n"
                    "import org.springframework.boot.SpringApplication;\n"
                    "import org.springframework.boot.autoconfigure.SpringBootApplication;\n\n"
                    "@SpringBootApplication\n"
                    "public class Application {\n\n"
                    "    public static void main(String[] args) {\n"
                    "        SpringApplication.run(Application.class, args);\n"
                    "    }\n\n"
                    "}\n")
            os.makedirs(os.path.dirname(ap(newrel)), exist_ok=True)
            open(ap(newrel), "w").write(code)
            gen.append(newrel)
            changes.append("synthesized missing @SpringBootApplication: Application.java")

    # 3. drop test-source files duplicating a main-source class
    main_types = set()
    for rel in gen:
        if rel.endswith(".java") and not is_test(rel) and os.path.exists(ap(rel)):
            t = _java_public_type(open(ap(rel)).read())
            if t:
                main_types.add(t)
    for rel in list(gen):
        if not (rel.endswith(".java") and is_test(rel)) or not os.path.exists(ap(rel)):
            continue
        t = _java_public_type(open(ap(rel)).read())
        if t and t in main_types:
            try:
                os.remove(ap(rel))
            except Exception:
                pass
            if rel in gen:
                gen.remove(rel)
            changes.append(f"removed test dup of {t}")

    # 5. Spring Boot config invariants. persistence.xml is a JPA-native descriptor
    #    that conflicts with application.properties (it forces a second, usually
    #    PostgreSQL, dialect); Spring Boot ignores it, the PO rejects it. schema.sql
    #    routinely leaks PostgreSQL-only idioms that crash on the H2 default DB.
    for rel in list(gen):
        base = os.path.basename(rel)
        # Spring-only: persistence.xml conflicts with application.properties in Spring Boot,
        # but is the REQUIRED JPA descriptor in a plain JPA/EclipseLink app (TeaStore) —
        # deleting it there destroys the persistence layer (see e2e run #1).
        if base == "persistence.xml" and _is_spring(stack) and os.path.exists(ap(rel)):
            try:
                os.remove(ap(rel))
            except Exception:
                pass
            if rel in gen:
                gen.remove(rel)
            changes.append("removed persistence.xml (Spring Boot uses application.properties)")
        elif base == "schema.sql" and os.path.exists(ap(rel)):
            sql = open(ap(rel)).read()
            scrubbed = _h2_safe_sql(sql)
            if scrubbed != sql:
                open(ap(rel), "w").write(scrubbed)
                changes.append("scrubbed PostgreSQL idioms from schema.sql")

    # Deleted module pom.xml files are restored by _restore_missing_poms() inside
    # _run_maven_compile (reads pom_snapshot.json) so the guard runs before EVERY build,
    # not only at QA-sprint entry.

    if changes:
        bb["generated_files"] = gen
        blackboard.write("blackboard", bb)
        print(f"🧹 Source hygiene: {len(changes)} fix(es) — {changes}")
    return changes


def _resolve_path(output_dir: str, filename: str) -> str:
    import os
    if "/" in filename or "\\" in filename:
        p = os.path.join(output_dir, filename)
        return p if os.path.exists(p) else ""
    for root, _, files in os.walk(output_dir):
        if filename in files:
            return os.path.join(root, filename)
    return ""


def _module_toplevel_names(tree) -> set:
    """Names a Python module exposes at top level (defs, classes, assignments,
    and re-imported names)."""
    import ast
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _run_python_checks(output_dir: str, files: list) -> list:
    """Deterministic gate for Python: syntax + intra-project import resolution.
    Catches the cross-file drift the LLM QA misses (e.g. `from database import
    Note` when database.py never defines Note). No deps installed, no network."""
    import ast
    import os

    issues = []
    modules = {}  # stem -> (basename, tree, names)
    for fname in files:
        if not fname.endswith(".py"):
            continue
        path = _resolve_path(output_dir, fname)
        if not path:
            continue
        try:
            src = open(path).read()
        except Exception:
            continue
        base = os.path.basename(fname)
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            issues.append({"file": base, "concern": f"SyntaxError: {e.msg} at line {e.lineno}"})
            continue
        stem = os.path.splitext(base)[0]
        modules[stem] = (base, tree, _module_toplevel_names(tree))

    local_stems = set(modules)
    for stem, (base, tree, _names) in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                target = node.module.split(".")[-1]
                if target in local_stems and target != stem:
                    exported = modules[target][2]
                    for alias in node.names:
                        if alias.name == "*":
                            continue
                        if alias.name not in exported:
                            issues.append({
                                "file": base,
                                "concern": (f"Imports '{alias.name}' from '{node.module}', "
                                            f"but {target}.py does not define it."),
                            })
    return issues


def _has_endpoint_layer(output_dir: str, files: list, stack: dict) -> bool:
    """True if any generated file defines a route handler."""
    lang = stack.get("language", "").lower()
    if lang == "python":
        markers = ["@app.", "@router."]
    else:
        # Route-mapping annotations only. NOT bare "@RestController" — it is a
        # substring of "@RestControllerAdvice" (exception handlers), which would
        # falsely register the endpoint layer as present.
        markers = ["@GetMapping", "@PostMapping", "@PutMapping",
                   "@DeleteMapping", "@PatchMapping", "@RequestMapping"]
    for fname in files:
        path = _resolve_path(output_dir, fname)
        if not path:
            continue
        try:
            content = open(path).read()
        except Exception:
            continue
        if any(m in content for m in markers):
            return True
    return False


def _norm_endpoint_path(path: str) -> str:
    """Canonicalize a REST path for matching: strip path-variable NAMES (so
    /pets/{petId} == /pets/{id}), normalize slashes. '/pets/{petId}/visits' →
    '/pets/{}/visits'."""
    import re
    p = re.sub(r"\{[^}]*\}", "{}", (path or "").strip())
    p = "/" + p.strip("/")
    return p.rstrip("/") or "/"


def _controller_for_path(path: str) -> str:
    """Best-guess controller filename owning a path, from its deepest static
    segment: '/pets/{petId}/vaccinations/{id}' → 'VaccinationController.java'."""
    import re
    segs = [s for s in (path or "").strip("/").split("/") if s and not s.startswith("{")]
    res = segs[-1] if segs else "Api"
    name = "".join(part[:1].upper() + part[1:] for part in re.split(r"[-_]", res) if part)
    if len(name) > 1 and name.endswith("s") and not name.endswith("ss"):
        name = name[:-1]  # crude singularize: vaccinations → Vaccination, pets → Pet
    return f"{name}Controller.java"


def _ondisk_controller_kind(output_dir: str, target_file: str):
    """'RestController' / 'Controller(view)' for an existing controller file (matched by
    basename via the tree-sitter probe), or None if the probe is unavailable or no such
    controller is parsed. Lets the coverage gate avoid routing REST endpoint work into a
    Thymeleaf view @Controller (the PetController /pets collision)."""
    if not _HAS_TS:
        return None
    import os
    base = os.path.basename(target_file).lower()
    for c in java_probe.controllers(output_dir):
        if os.path.basename(c["file"]).lower() == base:
            return c["kind"]
    return None


def _rest_controller_name(target_file: str) -> str:
    """PetController.java → PetRestController.java — a distinct REST controller so the
    legacy view controller of the same name is left untouched."""
    if target_file.endswith("Controller.java"):
        return target_file[:-len("Controller.java")] + "RestController.java"
    return target_file


def _emit_skeleton_for_controller_task(t, target_file, contract, blackboard, output_dir):
    """Deterministically emit a compiling @RestController skeleton for a coverage
    controller task: derive the owned endpoints + base path from the contract and
    colocate the file in the resource entity's package. Returns the written relpath, or
    None (caller falls back to a cline CREATE task)."""
    from agile_team_cline.ceremonies.planning import (
        emit_controller_skeleton, entity_names, _controller_endpoints)
    tid = t.get("id", "")
    resource = tid.split("-", 1)[1] if "-" in tid else ""
    if not resource and target_file.endswith("Controller.java"):
        resource = target_file[:-len("Controller.java")].replace("Rest", "")
    if not resource:
        return None
    entities = [e for e in entity_names(contract) if e]
    owned = _controller_endpoints(contract, resource, entities)
    if not owned:
        return None
    # base path = the top-level segment the owned endpoints share
    base = ""
    first = owned[0].split(None, 1)
    if len(first) == 2:
        seg = first[1].strip("/").split("/")[0]
        if seg and not seg.startswith("{"):
            base = "/" + seg
    # colocate with the resource's entity (then service) package on disk
    package, body_type = None, None
    for probe in (f"{resource}.java", f"{resource}Service.java"):
        rel = blackboard.find_existing(probe)
        if rel:
            rel = rel.replace("\\", "/")
            if "src/main/java/" in rel:
                package = rel.split("src/main/java/", 1)[1].rsplit("/", 1)[0].replace("/", ".")
                if probe == f"{resource}.java":
                    body_type = resource
                break
    if not package:
        return None
    pinned = contract.get("_pinned_ids") if isinstance(contract, dict) else None
    return emit_controller_skeleton(target_file[:-len(".java")], base, owned,
                                    package, output_dir, body_type=body_type,
                                    pinned_ids=pinned)


def _jaxrs_mappings(output_dir: str) -> set:
    """Set of (HTTP_METHOD, normalized_path) served by JAX-RS resources across EVERY
    module's src tree. TeaStore-style endpoints are invisible to the Spring scanner, so
    without this the coverage gate thinks every JAX-RS route is 'uncovered' and demands a
    Spring controller (TeaStore run #5: 6× CheapestController).

    Routes are resolved THROUGH inheritance: a concrete @Path resource is credited with the
    HTTP-method routes on its own body AND those inherited from its superclass or any
    implemented interface — TeaStore's *Endpoint classes carry only a class-level @Path and
    inherit all CRUD routes from AbstractCRUDEndpoint, which the old own-annotations-only scan
    missed (false-RED; runs #8/#9). Resolution is purely structural: no base-class name or
    route set is hardcoded, so it works for any JAX-RS base class or interface."""
    import os
    import re
    from agile_team_cline.workspace_discovery import _find_src_roots
    http_anns = ("GET", "POST", "PUT", "DELETE", "HEAD", "PATCH")
    ann_re = re.compile(r'@(' + "|".join(http_anns) + r')\b')
    roots = _find_src_roots(output_dir) or [os.path.join(output_dir, "src", "main", "java")]

    def _own_routes(src: str) -> list:
        """(HTTP, subpath) for each method handler declared in this file's own body."""
        out = []
        for m in ann_re.finditer(src):
            # window from the HTTP annotation to the method DECLARATION (its @Path, if any).
            # Cut at the method keyword, not the first '{' — a path variable like
            # @Path("cheapest/{limit}") contains '{' and would truncate it.
            tail = src[m.end(): m.end() + 300]
            mdecl = re.search(r'\b(?:public|protected|private)\b', tail)
            window = tail[:mdecl.start()] if mdecl else tail
            pm = re.search(r'@Path\s*\(\s*"([^"]*)"', window)
            out.append((m.group(1), pm.group(1) if pm else ""))
        return out

    # Pass 1: one record per class/interface that is a resource or a route-bearing ancestor.
    records = {}
    for root in roots:
        for dirpath, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                if not fn.endswith(".java"):
                    continue
                try:
                    src = open(os.path.join(dirpath, fn)).read()
                except Exception:
                    continue
                if "@Path" not in src and not ann_re.search(src):
                    continue  # neither a mounted resource nor a route-bearing base/interface
                decl = re.search(r'\b(?:class|interface)\s+(\w+)', src)
                if not decl:
                    continue
                name = decl.group(1)
                # class-level @Path (the resource mount), taken from the pre-declaration header
                header = src[:decl.start()]
                cb = re.search(r'@Path\s*\(\s*"([^"]*)"', header)
                # ancestors: `extends X` and/or `implements A, B, ...` — generics stripped so a
                # bound like `class X<T extends Number>` can't be mistaken for a superclass.
                hdr = re.sub(r'<[^>]*>', '', src[decl.start():].split("{", 1)[0])
                parents = []
                ext = re.search(r'\bextends\s+([\w.]+)', hdr)
                if ext:
                    parents.append(ext.group(1).split(".")[-1])
                impl = re.search(r'\bimplements\s+([^{]+)', hdr)
                if impl:
                    for p in impl.group(1).split(","):
                        p = p.strip().split(".")[-1]
                        if p:
                            parents.append(p)
                is_abstract = bool(
                    re.search(r'\b(?:abstract\s+class|interface)\s+' + re.escape(name) + r'\b', src))
                records[name] = {
                    "base": cb.group(1) if cb else None,
                    "parents": parents,
                    "abstract": is_abstract,
                    "own": _own_routes(src),
                }

    def _inherited(parents, seen):
        """Own routes of every ancestor reachable via extends/implements (cycle-guarded;
        unknown ancestors outside the workspace are simply skipped)."""
        out = []
        for p in parents:
            if p in seen:
                continue
            seen.add(p)
            rec = records.get(p)
            if not rec:
                continue
            out += rec["own"]
            out += _inherited(rec["parents"], seen)
        return out

    # Pass 2: emit own + inherited routes for every concrete, @Path-mounted resource.
    mapped = set()
    for name, rec in records.items():
        if rec["base"] is None or rec["abstract"]:
            continue  # not a mounted resource (e.g. an abstract base or a bare interface)
        base = rec["base"]
        for http, sub in rec["own"] + _inherited(rec["parents"], {name}):
            full = _norm_endpoint_path((base.rstrip("/") + "/" + sub.strip("/")).strip("/"))
            mapped.add((http, full))
    return mapped


def _controller_mappings(output_dir: str) -> set:
    """Set of (HTTP_METHOD, normalized_path) actually served by Spring controllers OR
    JAX-RS resources anywhere under src/ — scanned across the WHOLE tree, not the
    current-milestone slice, since a path may be owned by a file built earlier.

    Uses the tree-sitter probe when available (JSON-serving routes only, so a Thymeleaf
    @Controller view can't falsely cover a REST endpoint; AST-correct base paths). Falls
    back to the regex scanner below otherwise. JAX-RS routes are always unioned in."""
    jax = _jaxrs_mappings(output_dir)
    if _HAS_TS:
        return java_probe.controller_mappings(output_dir) | jax
    import os
    import re
    ann_method = {"GetMapping": "GET", "PostMapping": "POST", "PutMapping": "PUT",
                  "DeleteMapping": "DELETE", "PatchMapping": "PATCH"}
    mapped = set()
    src_root = os.path.join(output_dir, "src", "main", "java")
    walk_root = src_root if os.path.isdir(src_root) else output_dir
    for dirpath, dirs, files in os.walk(walk_root):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".java"):
                continue
            try:
                src = open(os.path.join(dirpath, fn)).read()
            except Exception:
                continue
            if "@RestController" not in src and "@Controller" not in src:
                continue
            # Class-level base path (the @RequestMapping before `class`).
            header = src.split("class ", 1)[0]
            mb = re.search(r'@RequestMapping\s*\(\s*(?:value\s*=\s*|path\s*=\s*)?[{\s]*"([^"]*)"', header)
            base = mb.group(1) if mb else ""

            def _full(sub: str) -> str:
                joined = (base.rstrip("/") + "/" + (sub or "").strip("/")).strip("/")
                return _norm_endpoint_path(joined)

            for ann, http in ann_method.items():
                for mm in re.finditer(r'@' + ann + r'\b\s*(\(([^)]*)\))?', src):
                    pm = re.search(r'"([^"]*)"', mm.group(2) or "")
                    mapped.add((http, _full(pm.group(1) if pm else "")))
            # Method-level @RequestMapping(method=RequestMethod.X, ...).
            for mm in re.finditer(r'@RequestMapping\s*\(([^)]*method[^)]*)\)', src):
                args = mm.group(1)
                pm = re.search(r'"([^"]*)"', args)
                sub = pm.group(1) if pm else ""
                for httpm in re.findall(r'RequestMethod\.(\w+)', args):
                    mapped.add((httpm.upper(), _full(sub)))
    return mapped | jax


def _run_contract_coverage(output_dir: str, contract: dict, files: list, stack: dict) -> list:
    """Endpoint-coverage gate. For Java, flags EACH contract endpoint that has no
    matching controller @*Mapping (path-variable names ignored). The old check only
    asked "does ANY route handler exist?", so a single controller masked whole
    resources with no controller at all (e.g. the M2 pet/vaccination CRUD)."""
    if not isinstance(contract, dict):
        return []
    real = [e for e in contract.get("endpoints", [])
            if isinstance(e, dict) and e.get("path") not in (None, "", "/health")]
    if not real:
        return []

    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if not is_java:
        # FastAPI folds many routes into few files — per-endpoint matching is
        # unreliable there, so keep the coarse "is the route layer present?" check.
        if _has_endpoint_layer(output_dir, files, stack):
            return []
        listed = ", ".join(f"{e.get('method','?')} {e.get('path','?')}" for e in real[:8])
        return [{
            "file": "(endpoint layer)",
            "concern": (f"Contract declares {len(real)} endpoint(s) but no generated file "
                        f"implements a route handler — the controller/router layer is missing. "
                        f"Endpoints: {listed}"),
        }]

    mapped = _controller_mappings(output_dir)
    uncovered = [e for e in real
                 if (str(e.get("method", "GET")).upper(), _norm_endpoint_path(e.get("path", ""))) not in mapped]
    if not uncovered:
        return []
    if not _is_spring(stack):
        # JAX-RS: target the resource's *Endpoint (extend in place), not a Spring controller.
        # Resource owner = the entity of the TOP-LEVEL path segment (the same authority
        # build_coverage_tasks uses for its target_file), NOT the deepest segment: an action
        # sub-path like /products/best-sellers is owned by the products collection →
        # ProductEndpoint, not BestSellerEndpoint. Keeps this human-readable concern aligned
        # with the deterministic rebuild (was: _controller_for_path → deepest segment).
        from agile_team_cline.ceremonies.planning import _endpoint_owner, entity_names
        ents = entity_names(contract)
        by_ep = {}
        for e in uncovered:
            owner = _endpoint_owner(e, ents) or _controller_for_path(e.get("path", "")).replace("Controller.java", "")
            ep = f"{owner}Endpoint"
            by_ep.setdefault(ep, []).append(f"{e.get('method','?')} {e.get('path','?')}")
        return [{
            "file": ep,
            "concern": (f"Contract endpoint(s) {', '.join(eps)} are not served by any JAX-RS "
                        f"@Path resource — add them to {ep} (extend the existing resource with "
                        f"@GET/@POST + @Path; do NOT create a Spring controller). Put query logic "
                        f"in the matching *Repository, not the endpoint."),
        } for ep, eps in by_ep.items()]
    by_ctrl = {}
    for e in uncovered:
        ctrl = _controller_for_path(e.get("path", ""))
        # Don't pin a REST endpoint onto a Thymeleaf view controller of the same name —
        # target a distinct *RestController instead (the PetController /pets collision).
        if _ondisk_controller_kind(output_dir, ctrl) == "Controller(view)":
            ctrl = _rest_controller_name(ctrl)
        by_ctrl.setdefault(ctrl, []).append(
            f"{e.get('method','?')} {e.get('path','?')}")
    return [{
        "file": ctrl,
        "concern": (f"Contract endpoint(s) {', '.join(eps)} have no matching "
                    f"@GetMapping/@PostMapping/@RequestMapping in any controller — implement "
                    f"them in {ctrl}, delegating business logic to the owning service."),
    } for ctrl, eps in by_ctrl.items()]


def _webservlet_routes(output_dir: str) -> set:
    """Set of normalized URL patterns served by Jakarta @WebServlet classes across every
    module's src tree (enhancement #2 — presentation tier). Handles @WebServlet("/x"),
    @WebServlet({"/x","/y"}), and @WebServlet(urlPatterns = "/x" | {"/x"}). Mirrors the
    structural, regex-based approach of _jaxrs_mappings."""
    import os
    import re
    from agile_team_cline.workspace_discovery import _find_src_roots
    roots = _find_src_roots(output_dir) or [os.path.join(output_dir, "src", "main", "java")]
    ann_re = re.compile(r'@WebServlet\s*\(([^)]*)\)', re.S)
    str_re = re.compile(r'"([^"]*)"')
    out = set()
    for root in roots:
        for dp, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                if not fn.endswith(".java"):
                    continue
                try:
                    src = open(os.path.join(dp, fn)).read()
                except Exception:
                    continue
                for m in ann_re.finditer(src):
                    for s in str_re.findall(m.group(1)):
                        if s.startswith("/"):
                            out.add(_norm_endpoint_path(s))
    return out


def _webapp_pages(output_dir: str) -> set:
    """Lowercased basenames of every *.jsp under any src/main/webapp tree (enhancement #2)."""
    import os
    out = set()
    for dp, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        if "webapp" not in dp.split(os.sep):
            continue
        for fn in files:
            if fn.endswith(".jsp"):
                out.add(fn.lower())
    return out


def _run_presentation_coverage(output_dir: str, contract: dict, stack: dict) -> list:
    """Presentation-tier coverage gate (enhancement #2, FullStack-Agent per-tier verification):
    for each contract `view`, flag a missing servlet (no @WebServlet serving its route) or a
    missing JSP page. Jakarta + a non-empty `views` section only; never sets report["compiles"]
    — failures become file-scoped issues that feed the same refinement loop as the other gates."""
    if not isinstance(contract, dict) or _is_spring(stack):
        return []
    views = contract.get("views")
    if not isinstance(views, list) or not views:
        return []
    served = _webservlet_routes(output_dir)
    pages = _webapp_pages(output_dir)
    issues = []
    for v in views:
        if not isinstance(v, dict) or not v.get("name"):
            continue
        name = v["name"]
        route = _norm_endpoint_path(v.get("route", ""))
        # basename-only: _webapp_pages keys on bare file names, and the model sometimes
        # stores a "WEB-INF/pages/x.jsp" path in the view (defense in depth vs _normalize_views).
        jsp = (v.get("jsp") or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
        if route and route not in served:
            issues.append({"file": f"{name}Servlet.java",
                "concern": (f"Presentation gap: no @WebServlet serves the view route "
                            f"'{v.get('route')}'. Add {name}Servlet (extend AbstractUIServlet, "
                            f"@WebServlet(\"{v.get('route')}\")), fetch data via registryclient, and "
                            f"forward to WEB-INF/pages/{v.get('jsp')}.")})
        if jsp and jsp not in pages:
            issues.append({"file": v.get("jsp"),
                "concern": (f"Presentation gap: the JSP page '{v.get('jsp')}' for view '{name}' is "
                            f"missing under src/main/webapp/WEB-INF/pages — create it to render the "
                            f"servlet's request attributes.")})
    return issues


def count_presentation_gaps(output_dir: str, contract: dict, stack: dict) -> int:
    """Number of unmet presentation-tier views (enhancement #2) — used by the milestone
    checkpoint guard so a dropped webui layer can't false-green as Complete. Returns 0 when the
    contract has no `views` (every backend-only / Spring run), so behavior is unchanged."""
    return len(_run_presentation_coverage(output_dir, contract, stack))


def _run_smoke_gate(output_dir: str, contract: dict):
    """Boot the app (mvn spring-boot:run) and GET-probe its routes for 5xx. Returns
    (issues, summary) when the smoke run executed, or None if it didn't run at all
    (e.g. no pom.xml — non-Maven stack already filtered by the caller)."""
    import os
    import re
    from agile_team_cline import smoke
    from agile_team_cline.ceremonies.planning import _extract_conventions

    conv = _extract_conventions(output_dir)
    paths = set()
    for r in conv.get("routes", []):
        if r.startswith("GET "):
            paths.add(r[len("GET "):])
    if isinstance(contract, dict):
        for e in contract.get("endpoints", []):
            if isinstance(e, dict) and str(e.get("method", "GET")).upper() == "GET":
                p = e.get("path")
                if p:
                    paths.add(p)

    result = smoke.run_smoke(output_dir, sorted(paths))
    if not result.get("ran"):
        if result.get("boot_error"):
            return ([{
                "file": "src/main/resources/application.properties",
                "concern": ("RUNTIME SMOKE FAILURE: the application failed to start under "
                            f"`mvn spring-boot:run`: {result['boot_error'][-800:]}"),
            }], "boot failed")
        return None

    issues = []
    if result["failed"]:
        ctrl_map = {}
        if _HAS_TS:
            for c in java_probe.controllers(output_dir):
                for e in c["endpoints"]:
                    if e["http"] == "GET":
                        ctrl_map[_norm_endpoint_path(e["path"])] = os.path.basename(c["file"])
        for f in result["failed"]:
            detail = f.get("detail", "")
            if re.search(r'Table\s+"?[A-Z0-9_]+"?\s+not found', detail, re.I):
                target = "src/main/resources/db/h2/schema.sql"
            else:
                target = ctrl_map.get(_norm_endpoint_path(f["path"])) or _controller_for_path(f["path"])
            issues.append({
                "file": target,
                "concern": (f"RUNTIME SMOKE FAILURE: GET {f['path']} returned "
                            f"{f['status']} when the app was actually run. "
                            f"Response: {detail}"),
            })
    summary = f"{result['probed']} routes probed, {len(result['failed'])} failed"
    return (issues, summary)


def _run_schema_coverage(output_dir: str, blackboard: Blackboard) -> list:
    """Flag @Entity classes (current-milestone slice) with no matching CREATE TABLE in
    db/h2/schema.sql. With ddl-auto=none (the project default) every endpoint touching
    such an entity 500s at runtime with 'Table "X" not found' — this gate catches it
    deterministically at QA time, before the smoke gate has to run the app."""
    if not _HAS_TS:
        return []
    import os
    import re
    props_path = os.path.join(output_dir, "src", "main", "resources", "application.properties")
    try:
        with open(props_path) as f:
            props = f.read()
    except OSError:
        props = ""
    m = re.search(r"ddl-auto\s*=\s*(\S+)", props)
    if m and m.group(1).strip().lower() not in ("none", "validate"):
        return []  # auto-create/update schemas don't need DDL coverage

    bb = blackboard.read("blackboard")
    baseline = bb.get("milestone_file_baseline", 0)
    files = bb.get("generated_files", [])[baseline:]
    java_paths = []
    for rel in files:
        rel_norm = rel.replace("\\", "/")
        if rel_norm.endswith(".java") and "src/main/java" in rel_norm:
            full = os.path.join(output_dir, rel)
            if os.path.exists(full):
                java_paths.append(full)
    if not java_paths:
        return []

    entities = java_probe.entity_tables(java_paths)
    if not entities:
        return []

    schema_path = os.path.join(output_dir, "src", "main", "resources", "db", "h2", "schema.sql")
    try:
        with open(schema_path) as f:
            schema_text = f.read()
    except OSError:
        schema_text = ""
    existing = java_probe.schema_tables(schema_text)
    missing = {table: path for table, path in entities.items() if table not in existing}
    if not missing:
        return []

    issues = []
    for table, path in sorted(missing.items()):
        rel = os.path.relpath(path, output_dir).replace("\\", "/")
        issues.append({
            "file": "src/main/resources/db/h2/schema.sql",
            "concern": (
                f"SCHEMA GAP: the @Entity in {rel} maps to table '{table}' but "
                f"db/h2/schema.sql has no CREATE TABLE '{table}'. With ddl-auto=none, "
                f"every endpoint touching this entity will fail at runtime with "
                f"Table \"{table.upper()}\" not found. Add a CREATE TABLE {table} "
                f"(...) statement to db/h2/schema.sql with columns matching the "
                f"entity's fields (and FK columns for its relationships), and mirror "
                f"the same DDL in db/mysql/schema.sql and db/postgres/schema.sql."
            ),
        })
    return issues


def ensure_contract_coverage(feature: str, blackboard: Blackboard, stack: dict, model,
                             scrum_master=None, total_sprints: int = 5):
    """Coverage safety-net (runs after the sprint loop). Java now plans the full
    vertical deterministically up front, so this should normally find nothing —
    but it still catches anything missing using the SAME contract-derived rule
    (entity→repository→service per entity, controller per endpoint resource). For
    FastAPI (folded routes) it only checks the route-handler layer, which is the
    one reliably detectable signal there. Returns built tasks so the caller can
    fold them into the backlog for review/refinement."""
    from agile_team_cline.ceremonies.planning import build_coverage_tasks

    contract = blackboard.read("contract")
    bb = blackboard.read("blackboard")
    all_files = bb.get("generated_files", [])
    baseline = bb.get("milestone_file_baseline", 0)
    files = all_files[baseline:]
    output_dir = str(blackboard.output_dir)

    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if is_java:
        # Dedup ("does this file already exist?") MUST be answered against the WHOLE
        # tree on disk — all milestones + brownfield — NOT the current-milestone slice
        # (all_files[baseline:]). With the slice, at M2 every M1/brownfield file looks
        # "missing", so the gate floods the backlog with re-create-existing tasks: those
        # regenerate duplicate classes and, being dependency-sorted ahead of controllers,
        # starve the controller tasks at the tail of refinement (PetRestController never
        # gets dispatched). Walk the disk so existing entities/repos/services are skipped.
        import os
        from agile_team_cline.workspace_discovery import _find_src_roots
        disk_files = []
        # Multi-module aware: walk EVERY Maven module's src tree (P4.1), not just a single
        # <root>/src/main/java. On a reactor the single-root walk goes blind to all modules
        # the moment any file lands at the phantom <root>/src/main/java, so existing entities
        # read as "missing" and get rebuilt (TeaStore e2e run #1). Single-module (petclinic)
        # returns exactly [<root>/src/main/java] → identical behavior.
        src_roots = _find_src_roots(output_dir) or [os.path.join(output_dir, "src", "main", "java")]
        for _root in src_roots:
            for _dp, _dn, _fns in os.walk(_root):
                _dn[:] = [d for d in _dn if d not in ("target", ".git") and not d.startswith(".")]
                disk_files.extend(f for f in _fns if f.endswith(".java"))
        # Endpoint-aware: feed the routes actually served on disk so a controller
        # that exists but is missing contract endpoints yields an EXTEND task,
        # instead of being silently skipped because its file is present.
        mapped = _controller_mappings(output_dir)
        tasks = build_coverage_tasks(contract, stack, existing_files=disk_files, mapped_endpoints=mapped)
        # Two-layer dedup fix: build_coverage_tasks dedups on the milestone slice, so a
        # legacy brownfield controller (e.g. a Thymeleaf PetController serving zero REST
        # routes) yields a CREATE task — which the session-level skip then drops because
        # the file exists, so its endpoints are never added and the coverage gate loops.
        # Use the authoritative on-disk check to re-route those as in-place EXTENDs.
        for t in (tasks or []):
            if t.get("type") != "controller":
                continue
            target = t["target_file"]

            if t.get("coverage_extend"):
                # Endpoint-aware extend tasks name the target by resource convention
                # (e.g. Pet -> PetController.java) with no file attribution — if that
                # file is the legacy Thymeleaf view controller (the Pet /pets
                # collision), reroute to the distinct *RestController exactly like
                # the unmarked-CREATE path below, instead of bolting REST endpoints
                # (and new constructor deps) onto the view controller.
                kind = _ondisk_controller_kind(output_dir, target)
                if kind != "Controller(view)":
                    continue  # genuine REST controller missing endpoints — extend as planned
                rest_target = _rest_controller_name(target)
                if blackboard.find_existing(rest_target):
                    t["target_file"] = rest_target
                    t["title"] = ("Extend the existing " + rest_target[:-5]
                                  + " in place to ADD the missing contract REST endpoints "
                                    "(keep the existing file/class). " + t["title"])
                    continue
                rel = _emit_skeleton_for_controller_task(t, rest_target, contract, blackboard, output_dir)
                if rel:
                    t["target_file"] = rest_target
                    t["title"] = ("EDIT the existing file " + rest_target + " IN PLACE and SAVE it: "
                                  "every method body currently throws UnsupportedOperationException "
                                  "— replace each one with a real implementation that delegates to "
                                  "the owning service. You MUST rewrite the file; do NOT leave any "
                                  "UnsupportedOperationException. Keep the method signatures and the "
                                  "@*Mapping annotations exactly as they are.")
                else:
                    t["target_file"] = rest_target
                    t["coverage_extend"] = False
                    t["title"] = ("Create a NEW @RestController class " + rest_target[:-5]
                                  + " in a brand-new file " + rest_target + " — do NOT modify the existing "
                                    "view controller. " + t["title"])
                continue

            if blackboard.find_existing(target):
                # The file exists — REST controller (safe to extend) or a Thymeleaf view
                # @Controller of the same name (the PetController /pets collision)?
                kind = _ondisk_controller_kind(output_dir, target)
                if kind != "Controller(view)":
                    # Existing REST controller missing endpoints → extend in place (cline).
                    t["coverage_extend"] = True
                    t["title"] = ("Extend the existing " + target[:-5]
                                  + " in place to ADD the missing contract REST endpoints "
                                    "(keep the existing file/class). " + t["title"])
                    continue
                # View collision → the REST controller is a distinct file.
                target = _rest_controller_name(target)
                # …but cline may already have created that REST controller in a prior
                # round — never overwrite real work with a skeleton; extend it instead.
                if blackboard.find_existing(target):
                    t["target_file"] = target
                    t["coverage_extend"] = True
                    t["title"] = ("Extend the existing " + target[:-5]
                                  + " in place to ADD the missing contract REST endpoints "
                                    "(keep the existing file/class). " + t["title"])
                    continue
            # `target` is a NEW controller file. Emit a deterministic, compiling skeleton
            # (mappings present; bodies throw UnsupportedOperationException) so coverage +
            # compile pass WITHOUT depending on cline writing the file — the prompt only
            # has to FILL bodies, not create the whole controller. Fall back to a clean
            # CREATE task if emission can't resolve the package/endpoints.
            rel = _emit_skeleton_for_controller_task(t, target, contract, blackboard, output_dir)
            if rel:
                t["target_file"] = target
                t["coverage_extend"] = True
                t["title"] = ("EDIT the existing file " + target + " IN PLACE and SAVE it: "
                              "every method body currently throws UnsupportedOperationException "
                              "— replace each one with a real implementation that delegates to "
                              "the owning service. You MUST rewrite the file; do NOT leave any "
                              "UnsupportedOperationException. Keep the method signatures and the "
                              "@*Mapping annotations exactly as they are.")
            elif target != t["target_file"]:
                # Emission failed but we must NOT edit the legacy view controller.
                t["target_file"] = target
                t["title"] = ("Create a NEW @RestController class " + target[:-5]
                              + " in a brand-new file " + target + " — do NOT modify the existing "
                                "view controller. " + t["title"])
    else:
        endpoints = [e for e in (contract.get("endpoints") or [])
                     if isinstance(e, dict) and e.get("path") not in (None, "", "/health")] \
                     if isinstance(contract, dict) else []
        if endpoints and not _has_endpoint_layer(output_dir, files, stack):
            tasks = [{"id": "COV-1",
                      "title": "Implement all contract API endpoints as FastAPI routes",
                      "type": "router", "target_file": "main.py"}]
        else:
            tasks = []

    # A3: schema coverage repair — new @Entity classes with no matching CREATE TABLE
    # in db/h2/schema.sql 500 at runtime under ddl-auto=none. Append one task that
    # rides the extend path (bypasses the file-exists dedup) so cline edits the
    # existing schema.sql files in place.
    if is_java and stack.get("build_tool") == "Maven":
        schema_issues = _run_schema_coverage(output_dir, blackboard)
        if schema_issues:
            tasks = tasks or []
            tasks.append({
                "id": "SCHEMA-DDL",
                "title": (
                    "Add the missing CREATE TABLE statement(s) to "
                    "src/main/resources/db/h2/schema.sql (and mirror the same DDL in "
                    "db/mysql/schema.sql and db/postgres/schema.sql) so these entities "
                    "don't 500 with 'Table not found' at runtime:\n"
                    + "\n".join(f"- {i['concern']}" for i in schema_issues)
                ),
                "type": "config",
                "coverage_extend": True,
                "target_file": "src/main/resources/db/h2/schema.sql",
            })

    if not tasks:
        return None

    sprint_num = (bb.get("sprint_num", total_sprints) or total_sprints) + 50
    for t in tasks:
        t["sprint"] = sprint_num
    if scrum_master is not None:
        scrum_master.advance_sprint(sprint_num)

    print(f"🧩 Coverage gate: building missing files from contract: "
          f"{[t['target_file'] for t in tasks]}")
    # Controllers that already exist but lack endpoints are EXTEND tasks: run them
    # as an in-place edit (refinement_context) so the per-file dedup in
    # _run_cline_for_tasks doesn't skip them for already having a target file.
    create = [t for t in tasks if not t.get("coverage_extend")]
    extend = [t for t in tasks if t.get("coverage_extend")]
    if create:
        _run_dev_sprint(create, blackboard, stack, model)
    if extend:
        refine = {"qa_issues": [{"file": t["target_file"], "concern": t["title"]} for t in extend],
                  "po_notes": "Add the missing contract endpoints to these existing controllers. "
                              "Implement any business rule (validation, conflict/overlap checks) as a "
                              "method on the owning entity's service and call it from the controller — "
                              "do not put business logic or repository scans inside the controller."}
        _run_dev_sprint(extend, blackboard, stack, model, refinement_context=refine)
    # Refresh QA so the verdict reflects the newly built files.
    _run_qa_sprint(tasks, blackboard, stack, model)
    return tasks


def _build_test_tasks(blackboard: Blackboard, stack: dict) -> list:
    """Test tasks for the CURRENT milestone, deduped via find_existing — re-QA rounds
    (tests already present) yield no tasks, so authoring is naturally one-time:
      - 'unit'        one per new *Service.java    (Mockito unit test)
      - 'webmvc'      one per new *Controller.java (@WebMvcTest slice test)
      - 'datajpa'     one per new @Entity          (real H2 + schema.sql — fails if
                       the table is missing, backstopping the schema gate)
      - 'integration' one per new top-level resource (@SpringBootTest, drives a
                       contract create-then-read flow, backstopping the wiring rule)
    """
    import os
    import re
    is_spring = _is_spring(stack)
    bb = blackboard.read("blackboard")
    baseline = bb.get("milestone_file_baseline", 0)
    milestone = list(bb.get("generated_files", [])[baseline:])
    # Brownfield: a milestone that only EXTENDS pre-existing files (e.g. adds a method to
    # an already-on-disk *Repository.java/*Endpoint.java) never appears in generated_files
    # (that list is pre-seeded with the whole tree — see #2/E in sprint.py:_run_cline_session_
    # for_tasks), so it would be silently invisible here too and get ZERO test coverage even
    # with AGILE_AUTHOR_TESTS=1 (teastore-probe-affinity, 2026-07-05: UserRepository.java and
    # UserEndpoint.java were both extended, not created, and both were skipped). Union in every
    # file whose mtime/size changed since the pre-development baseline snapshot.
    _cur = blackboard.snapshot_src()
    _base = blackboard.read("baseline_src") or {}
    _modified = [rel for rel, meta in _cur.items()
                 if _base.get(rel) is not None and list(_base[rel]) != list(meta)]
    for rel in _modified:
        if rel not in milestone:
            milestone.append(rel)
    milestone_basenames = {os.path.basename(rel) for rel in milestone}

    # Feature-target scoping (AGILE_TEST_SCOPE, default "contract"): a greenfield/
    # regenerated milestone can contain many untouched-by-the-feature scaffolding
    # classes (e.g. every Repository in the module) alongside the one the feature
    # actually changed. Authoring a dedicated test for each of those is pure token
    # waste (test-authoring = the dominant cost driver — 64% of all tokens spent
    # historically). Scope unit/webmvc/repository/endpoint tests to classes whose
    # resource is named by the contract's endpoints, OR whose source references a
    # feature-declared DTO/endpoint-path token (covers the class actually holding
    # the new method even when it differs from the endpoint's nominal resource
    # owner, e.g. a cross-entity join method placed on a sibling repository).
    # Fails OPEN (no filtering) when the contract yields no usable signal, so a
    # feature is never shipped with zero test coverage.
    _test_scope = os.getenv("AGILE_TEST_SCOPE", "contract")
    _feature_resources, _feature_tokens = set(), set()
    if _test_scope != "off":
        try:
            from agile_team_cline.ceremonies.planning import entity_names, _resource_controllers
            _contract = blackboard.read("contract") or {}
            _entities = entity_names(_contract)
            _feature_resources = {n.lower() for n in _resource_controllers(_contract, _entities) if n}
            _raw_entities = _contract.get("entities", []) if isinstance(_contract, dict) else []
            if isinstance(_raw_entities, list):
                _feature_tokens = {e.get("name", "").lower() for e in _raw_entities
                                    if isinstance(e, dict) and e.get("dto") and e.get("name")}
            for ep in (_contract.get("endpoints", []) if isinstance(_contract, dict) else []):
                path = ep.get("path", "") if isinstance(ep, dict) else ""
                for seg in re.split(r"[/\-_{}]+", path):
                    seg = seg.strip().lower()
                    if len(seg) >= 5 and seg.isalpha():
                        _feature_tokens.add(seg)
        except Exception:
            _feature_resources, _feature_tokens = set(), set()
        if not _feature_resources:
            _test_scope = "off"  # no usable signal — don't suppress anything

    # Feature tag for chained-trunk test naming. On a chained trunk a per-class test
    # (e.g. ProductRepositoryTest.java) authored by an EARLIER feature already exists on
    # the seeded baseline, so the natural <stem>Test.java below would be skipped as
    # "already present" — leaving THIS feature's new methods on {stem} untested and thus
    # never profiled (feature_exercised=False: the teastore-hist-dupes defect). When that
    # happens we author the test under a unique, feature-specific name instead. Prefer the
    # response DTO this milestone added (e.g. ProductNearDuplicate); fall back to the
    # feature/workspace id so a tag is always available.
    _feature_tag = ""
    try:
        from agile_team_cline.ceremonies.planning import _dto_names
        _new_dtos = [d for d in _dto_names(blackboard.read("contract") or {})
                     if f"{d}.java" in milestone_basenames]
        if _new_dtos:
            _feature_tag = sorted(_new_dtos, key=len, reverse=True)[0]
    except Exception:
        _feature_tag = ""
    if not _feature_tag:
        _wid = re.sub(r"-(senior|junior)$", "", os.path.basename(str(blackboard.output_dir).rstrip("/")))
        _feature_tag = "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", _wid) if p)

    tasks, seen, entity_files = [], set(), []
    for rel in milestone:
        base = os.path.basename(rel)
        if "Test" in base or not base.endswith(".java"):
            continue
        if base.endswith("Service.java"):
            kind = "unit"
        elif base.endswith("Controller.java"):
            kind = "webmvc"
        elif base.endswith("Repository.java"):
            if is_spring:
                continue  # Spring Data repositories are interfaces — nothing to unit test
            kind = "repository"  # jakarta: this IS where the feature/perf method lives
        elif base.endswith("Endpoint.java") and not is_spring:
            kind = "endpoint"  # jakarta/JAX-RS resource — the project has no @WebMvcTest
                               # equivalent, so exercise it the same way as a repository
        else:
            try:
                if "@Entity" in (blackboard.output_dir / rel).read_text():
                    entity_files.append(rel)
            except Exception:
                pass
            continue
        stem = base[:-len(".java")]
        if _test_scope != "off" and kind in ("unit", "webmvc", "repository", "endpoint"):
            _suffix = {"unit": "Service", "webmvc": "Controller",
                       "repository": "Repository", "endpoint": "Endpoint"}[kind]
            _resource_name = stem[:-len(_suffix)] if stem.endswith(_suffix) else stem
            if _resource_name.lower() not in _feature_resources:
                try:
                    _src_lower = (blackboard.output_dir / rel).read_text().lower()
                except Exception:
                    _src_lower = ""
                if not any(tok and tok in _src_lower for tok in _feature_tokens):
                    continue
        test_file = f"{stem}Test.java"
        if test_file in seen or blackboard.find_existing(test_file):
            # Chained-trunk: the per-class test already exists (an earlier feature authored
            # it) and does NOT cover the method(s) THIS milestone added to {stem}. Do not
            # skip — author under a unique feature-tagged name so the new methods get a
            # real, profiled test instead of silently landing feature_exercised=False.
            _kind_suffix = {"unit": "Service", "webmvc": "Controller",
                            "repository": "Repository", "endpoint": "Endpoint"}.get(kind, "")
            _uniq = f"{_feature_tag}{_kind_suffix}Test.java" if _feature_tag else ""
            if not _uniq or _uniq in seen or blackboard.find_existing(_uniq):
                continue  # no unique name available, or already authored — idempotent
            test_file = _uniq
        seen.add(test_file)
        if kind == "unit":
            how = ("a JUnit 5 unit test using Mockito (@ExtendWith(MockitoExtension.class), @Mock "
                   "its repositories/collaborators, @InjectMocks the service) asserting its core "
                   "business methods with stubbed repository behavior")
        elif kind == "repository":
            how = (
                f"a plain JUnit 4 test that directly invokes the repository method(s) added or "
                f"changed on {stem} in this milestone and asserts on their returned values — the "
                f"test MUST call the method under test, not merely construct the repository. "
                f"Follow the project's own RepositoryTest.java pattern: "
                f"EMFManagerInitializer.initializeEMF() in an @Before method, singleton access "
                f"via {stem}.REPOSITORY. Use org.junit.Test/Before/Assert only — no JUnit 5, "
                f"Mockito, or Spring annotations"
            )
        elif kind == "endpoint":
            how = (
                f"a plain JUnit 4 test that directly calls the JAX-RS resource method(s) added "
                f"or changed on {stem} in this milestone (new instance, real EMFManagerInitializer"
                f".initializeEMF() in an @Before, no HTTP client / no Jersey test container) and "
                f"asserts on the returned values — the test MUST call the method under test, not "
                f"merely construct the endpoint. If the method returns a response DTO, ALSO "
                f"construct that DTO directly and round-trip it through the project's configured "
                f"JSON provider (or at minimum assert it has a no-arg constructor and public "
                f"getters) so a class that can't be marshalled to JSON fails this test instead of "
                f"shipping unexercised. Use org.junit.Test/Before/Assert only — no JUnit 5, "
                f"Mockito, or Spring annotations"
            )
        else:
            how = ("a Spring @WebMvcTest slice test using MockMvc and @MockBean for the service(s) it "
                   "calls, asserting each declared endpoint's status and JSON for a representative case")
        _test_class = test_file[:-len(".java")]
        title = (f"Create {_test_class} verifying {stem}: {how}. Before writing, READ {stem}.java and the "
                 f"entity/DTO classes it uses to copy their EXACT types. Construct id and field values of "
                 f"the SAME declared type as the code under test (e.g. if an id field is String use a "
                 f"String literal, if UUID use UUID.randomUUID(), if Long use a long) — never substitute "
                 f"UUID for String or vice-versa. Call ONLY methods/constructors/fields that actually "
                 f"exist on {stem} and its collaborators with their real parameter types; do not invent "
                 f"symbols. The test MUST compile and pass.")
        tasks.append({"id": f"TEST-{_test_class}", "title": title,
                      "target_file": test_file, "type": "test", "kind": kind,
                      "subject": stem})

    if not is_spring:
        # datajpa/integration below require @DataJpaTest/@SpringBootTest — Spring-only.
        return tasks

    # datajpa: one per new @Entity in this milestone — real H2 + schema.sql, so a
    # missing CREATE TABLE (schema gate's failure mode) fails this test in `mvn test`.
    for rel in entity_files:
        stem = os.path.basename(rel)[:-len(".java")]
        test_file = f"{stem}RepositoryTest.java"
        if test_file in seen or blackboard.find_existing(test_file):
            continue
        seen.add(test_file)
        title = (
            f"Create {stem}RepositoryTest: a @DataJpaTest that autowires the REAL "
            f"{stem}Repository (do NOT mock it — this test runs against the project's "
            f"actual H2 database with schema.sql applied), saves a new {stem} instance, "
            f"and asserts it can be found by id afterwards with its persisted fields "
            f"matching what was saved. Before writing, READ {stem}.java and "
            f"{stem}Repository.java to copy their EXACT field names, types and "
            f"constructors. Call ONLY methods/fields that actually exist. The test MUST "
            f"compile and pass — if it fails with 'Table not found', that is a real bug "
            f"in db/h2/schema.sql, NOT in this test."
        )
        tasks.append({"id": f"TEST-{stem}-DataJpa", "title": title,
                      "target_file": test_file, "type": "test", "kind": "datajpa",
                      "entity_name": stem, "subject": stem})

    # integration: one per new top-level resource controller in this milestone —
    # drives a contract create-then-read flow end-to-end, so a value that's
    # "computed" but never recalculated (the dead-code bug class) fails the test.
    contract = blackboard.read("contract") or {}
    try:
        from agile_team_cline.ceremonies.planning import _resource_controllers, _controller_endpoints, entity_names
        entities = entity_names(contract)
        resources = _resource_controllers(contract, entities)
    except Exception:
        resources, entities = [], []
    for r in resources:
        if f"{r}Controller.java" not in milestone_basenames and f"{r}RestController.java" not in milestone_basenames:
            continue
        test_file = f"{r}IntegrationTest.java"
        if test_file in seen or blackboard.find_existing(test_file):
            continue
        seen.add(test_file)
        endpoints = _controller_endpoints(contract, r, entities)
        endpoints_str = "; ".join(endpoints) if endpoints else "(see contract)"
        title = (
            f"Create {r}IntegrationTest: a @SpringBootTest with @AutoConfigureMockMvc "
            f"driving a realistic create-then-read flow over these contract endpoints "
            f"for {r}: [{endpoints_str}]. Use MockMvc to perform a POST/PUT that creates "
            f"or updates data, then a GET that reads it back, asserting a 2xx status AND "
            f"that any computed/derived field in the response (e.g. a total, count, or "
            f"aggregate) reflects the data that was just written (non-zero/non-default "
            f"where applicable) — not just that the field is present. Before writing, "
            f"READ the relevant controller/service/entity classes to copy their EXACT "
            f"request/response shapes, field names and types; call ONLY endpoints and "
            f"fields that actually exist. The test MUST compile and pass."
        )
        tasks.append({"id": f"TEST-{r}-Integration", "title": title,
                      "target_file": test_file, "type": "test", "kind": "integration",
                      "resource_name": r})
    return tasks


def _untargeted_test_tasks(test_tasks: list, blackboard: Blackboard) -> list:
    """Return the authored test tasks whose test file never references the task's
    `subject` class. Word-boundary match on comment-stripped source, so a javadoc
    mention or the test's own lookalike class name (FooTest contains Foo but not
    \\bFoo\\b) never counts as a reference. Tasks without a subject (e.g. integration
    tests that drive routes, not classes) and tasks whose file is missing (already
    recorded as failures by the authoring path) are skipped."""
    import re as _re
    from agile_team_cline.verify_lib import _strip_comments
    stale = []
    for t in test_tasks:
        subject = t.get("subject")
        fname = t.get("target_file", "")
        if not subject or not fname:
            continue
        rel = blackboard.find_existing(fname)
        if not rel:
            continue
        try:
            src = _strip_comments((blackboard.output_dir / rel).read_text())
        except Exception:
            continue
        if not _re.search(rf'\b{_re.escape(subject)}\b', src):
            stale.append(t)
    return stale


def _run_qa_sprint(tasks: list, blackboard: Blackboard, stack: dict, model) -> dict:
    import os
    # Author real tests (one dedicated, deduped cline session) BEFORE verification so
    # the suite actually exists to execute. On re-QA rounds the tests already exist, so
    # _build_test_tasks returns nothing and no session runs.
    is_java_stack = (stack.get("language", "").lower() == "java"
                     or stack.get("file_extension") == ".java")
    # Test authoring (#4 test-coverage gate): author per-entity unit/webmvc/datajpa/
    # integration tests so the suite actually exercises the new feature (a milestone with
    # zero tests for its new entity is how the M6 Prescription UUID/Integer id-drift and
    # the GET /prescriptions contract bug both shipped unexercised). Toggle with
    # AGILE_AUTHOR_TESTS (run scripts set =1). The old ~600s/test "writes nothing" failure
    # was backend-specific and no longer applies. Code default
    # stays 0 so non-Maven/ad-hoc entry points don't pay for authoring unless they opt in.
    if (is_java_stack and stack.get("build_tool") == "Maven"
            and os.getenv("AGILE_AUTHOR_TESTS", "0") == "1"):
        test_tasks = _build_test_tasks(blackboard, stack)
        if test_tasks:
            print(f"🧪 Authoring {len(test_tasks)} test file(s) via cline before verification...")
            _run_cline_for_tasks(test_tasks, blackboard, stack)
            # Test-target fidelity: an authored test must reference its subject class —
            # the minimum condition for it to possibly test that subject. One generic
            # rule for every kind/stack; one hardened re-issue, then record the failure.
            stale = _untargeted_test_tasks(test_tasks, blackboard)
            if stale:
                for t in stale:
                    subject = t["subject"]
                    print(f"   🔁 authored {t.get('target_file', '?')} never references "
                          f"{subject} — re-issuing once with a hardened brief")
                    t = dict(t)
                    t["title"] = (f"{t['title']} IMPORTANT: the previous attempt at "
                                  f"{t.get('target_file', '')} never referenced {subject} at all "
                                  f"(it tested something else). Rewrite it so it instantiates/"
                                  f"invokes {subject} itself and asserts on {subject}'s results.")
                    _run_cline_for_tasks([t], blackboard, stack)
                still = _untargeted_test_tasks(stale, blackboard)
                for t in still:
                    blackboard.record_failure(t["id"])
                    print(f"   ⚠️ {t.get('target_file', '?')} still does not reference "
                          f"{t['subject']} after re-issue — recorded as failed")

    # Deterministic source-tree hygiene + compiler-error-driven semantic repair
    # before any verification, so the compile and the PO judge a reconciled tree
    # (fixes javax/jakarta, dup/zero main class, class/file-name mismatch, stray
    # test-source dups, intra-project import typos, missing repo derived-finders).
    from agile_team_cline.repair import repair_sources
    _repair_changes, compile_result = repair_sources(blackboard, stack)

    # AGILE_QA_THINKING=1 routes the QA review through the extended-thinking model.
    # make_agent ignores the max_tokens arg when a model is passed, so we build the
    # thinking model at 16384 — the reasoning tokens share max_tokens with the JSON
    # report, and the default 4096 would let thinking starve/truncate the answer.
    if os.getenv("AGILE_QA_THINKING") == "1":
        from agile_team_cline.config import make_thinking_model
        qa_model = make_thinking_model(16384)
    else:
        qa_model = model
    agent = make_agent(QA["role"], QA["goal"], TASK_TOKEN_LIMITS["qa_report"], model=qa_model)
    bb = blackboard.read("blackboard")
    all_files = bb.get("generated_files", [])
    baseline = bb.get("milestone_file_baseline", 0)
    # QA only audits files from the current milestone — carryover files are
    # already-accepted prior work and don't belong in this verdict.
    files = all_files[baseline:]
    if not files:
        # Resumed/re-attempted milestone: init_sprint set the baseline to cover
        # ALL files already on disk (including this milestone's own prior-attempt
        # output), so the slice above is empty and QA would review nothing. Fall
        # back to the milestone's full backlog target files that exist on disk.
        seen = set()
        for task in bb.get("sprint_backlog", []):
            fname = task.get("target_file") or _title_to_filename(task["title"], stack["file_extension"])
            existing = blackboard.find_existing(fname)
            if existing and existing not in seen:
                seen.add(existing)
                files.append(existing)
    contract = blackboard.read("contract")
    output_dir = str(blackboard.output_dir)

    # Identify the files touched in the current sprint
    current_sprint_files = set()
    for task in tasks:
        fname = task.get("target_file") or _title_to_filename(task["title"], stack["file_extension"])
        current_sprint_files.add(fname)

    import os
    files_parts = []
    for filename in files:
        filepath = ""
        if "/" in filename or "\\" in filename:
            filepath = os.path.join(output_dir, filename)
        else:
            for root, _, walk_files in os.walk(output_dir):
                if filename in walk_files:
                    filepath = os.path.join(root, filename)
                    break
        if filepath and os.path.exists(filepath):
            # ALWAYS pass the full content of the file to the QA Engineer so they can fully verify completeness and correctness!
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                files_parts.append(f"--- File: {filename} ---\n{content}\n")
            except Exception:
                pass

    files_str = "\n".join(files_parts) if files_parts else "none"

    prompt = zh_prompt(_QA_REPORT_PROMPT, _QA_REPORT_PROMPT_ZH).format(
        files_content=files_str,
        contract=json.dumps(contract, separators=(",", ":")),
    )
    msg = BaseMessage.make_user_message(role_name="User", content=prompt)
    response = step_with_retry(agent, msg)
    raw = response.msgs[0].content if response.msgs else ""
    report = extract_json(raw) or {"passed": [], "issues": [], "coverage_gaps": []}

    # Physically verify compilation if stack uses Maven. Reuse the compile the
    # repair pass already ran on this exact tree (repair_sources returns it) so we
    # don't recompile the identical sources; only fall back to a fresh compile if
    # repair couldn't supply one (e.g. no pom yet).
    if stack.get("build_tool") == "Maven":
        if compile_result is not None:
            compile_issues = compile_result["issues"]
            report["compiles"] = compile_result["compiles"]
            # test_compiles defaults to compiles for older/None results.
            report["test_compiles"] = compile_result.get("test_compiles", compile_result["compiles"])
        else:
            compile_issues = _run_maven_compile(output_dir)
            report["compiles"] = not compile_issues
            report["test_compiles"] = report["compiles"]
        # report["compiles"] now means MAIN (production) compiles — the hard gate the
        # last-green guard and PO acceptance ride on. A broken generated test leaves
        # main green but test_compiles False; its compile errors are still recorded
        # below as file-scoped issues so the bad test is flagged, not the whole build.
        #
        # Truncation gate: a morpheus SIGKILL mid-write leaves a syntactically broken
        # .java (the VaccinationController EOF bug). The probe flags it deterministically;
        # folding it into compile_issues with report["compiles"]=False lets the G1 scoping
        # reject ONLY that file for refinement, instead of it silently poisoning the build.
        if _HAS_TS:
            changed_paths = []
            for fname in current_sprint_files:
                if not fname.endswith(".java"):
                    continue
                rel = blackboard.find_existing(fname)
                if rel:
                    changed_paths.append(os.path.join(output_dir, rel))
            bad = java_probe.parse_errors(changed_paths)
            if bad:
                print(f"🔎 Parse gate: {len(bad)} truncated/unparseable file(s) flagged.")
                report["compiles"] = False
                report["test_compiles"] = False
                compile_issues = list(compile_issues) + [
                    {"file": os.path.basename(b),
                     "concern": "Truncated/unparseable Java (incomplete write) — regenerate this file in full."}
                    for b in bad]
        if compile_issues:
            report.setdefault("issues", []).extend(compile_issues)

        # Gate-mismatch fix: the PO requires "compiled AND tested successfully", so
        # actually EXECUTE the suite — but only when the TEST sources compile (a
        # red test tree can't run, and `mvn test` would just re-report the compile
        # error). Failures become file-scoped issues → refinement; they are
        # deliberately NOT folded into report["compiles"] so the last-green guard
        # never rolls back a compiling app on a red test.
        if report.get("compiles") and report.get("test_compiles"):
            test_run = _run_maven_tests(output_dir)
            report["tests_run"] = test_run["tests_run"]
            report["tests_passed"] = test_run["tests_passed"]
            report["tests_failed"] = test_run["tests_failed"]
            report["tests"] = (f"{test_run['tests_run']} executed, "
                               f"{test_run['tests_passed']} passed, "
                               f"{test_run['tests_failed']} failed")
            if test_run["issues"]:
                report.setdefault("issues", []).extend(test_run["issues"])

        # Runtime smoke gate: only when the build AND tests are green — boot the
        # app for real and GET-probe its routes for 5xx. Catches missing-table
        # (schema) and dead-computation bugs that a clean compile+test misses.
        # Never sets report["compiles"]; failures become file-scoped issues that
        # feed the same refinement loop.
        if (report.get("compiles") and report.get("test_compiles")
                and report.get("tests_failed", 0) == 0
                and os.getenv("AGILE_SMOKE", "1") != "0"):
            smoke_issues = _run_smoke_gate(output_dir, contract)
            if smoke_issues is not None:
                issues, smoke_summary = smoke_issues
                report["smoke"] = smoke_summary
                if issues:
                    print(f"🔎 Smoke gate: {smoke_summary} — {len(issues)} issue(s).")
                    report.setdefault("issues", []).extend(issues)

    # Python has no compile step — run a deterministic syntax + intra-project
    # import-resolution gate so non-importable apps don't get accepted.
    if stack.get("language", "").lower() == "python":
        py_issues = _run_python_checks(output_dir, files)
        if py_issues:
            print(f"🔎 Python gate: {len(py_issues)} syntax/import-drift issue(s) flagged.")
            report.setdefault("issues", []).extend(py_issues)

    # All stacks: flag a dropped endpoint layer (contract vs generated files).
    coverage_issues = _run_contract_coverage(output_dir, contract, files, stack)
    if coverage_issues:
        print(f"🔎 Coverage gate: {coverage_issues[0]['concern']}")
        report.setdefault("issues", []).extend(coverage_issues)

    # Presentation-tier coverage (enhancement #2): flag a missing servlet/JSP for any contract
    # view (jakarta + `views` only). File-scoped issues → refinement; never sets report["compiles"].
    presentation_issues = _run_presentation_coverage(output_dir, contract, stack)
    if presentation_issues:
        print(f"🔎 Presentation gate: {len(presentation_issues)} missing servlet/JSP for contract view(s).")
        report.setdefault("issues", []).extend(presentation_issues)

    # Java/Maven: flag new @Entity classes with no matching CREATE TABLE in
    # db/h2/schema.sql — with ddl-auto=none these compile fine but 500 at runtime.
    if stack.get("build_tool") == "Maven":
        schema_issues = _run_schema_coverage(output_dir, blackboard)
        if schema_issues:
            print(f"🔎 Schema gate: {len(schema_issues)} table(s) missing from db/h2/schema.sql.")
            report.setdefault("issues", []).extend(schema_issues)

    # CRITICAL GATE: If compilation failed, reject ALL tasks (not just flag issues).
    # This prevents silent failures where broken code is accepted, causing downstream
    # cascading compilation errors. Compilation is THE hard gate that determines acceptance.
    if not report.get("compiles", True):
        # Soften: reject only the tasks whose target file is named in the compile
        # errors, so clean tasks stay acceptable and the blanket record_failure that
        # used to trip the critical-task halt (aborting before refinement) is gone.
        # Fall back to rejecting all only when no sprint task owns the error (e.g. a
        # pom/cross-file error). report["compiles"] stays False either way, so the
        # last-green guard + PO still prevent shipping a non-compiling tree.
        err_files = {os.path.basename(i.get("file", "")).lower()
                     for i in compile_issues if i.get("file")}

        def _implicated(t):
            fn = t.get("target_file") or _title_to_filename(t["title"], stack["file_extension"])
            return os.path.basename(fn).lower() in err_files

        reject = [t for t in tasks if _implicated(t)] or list(tasks)
        print(f"🚫 COMPILATION GATE: build does not compile — rejecting "
              f"{len(reject)}/{len(tasks)} implicated task(s) for refinement.")
        report["accepted"] = [t.get("id") for t in tasks if t not in reject]
        report["rejected"] = [t.get("id") for t in reject]
        report["reason"] = (
            "Compilation failed; implicated tasks rejected for refinement. Fix the "
            "compilation errors listed above."
        )
        for task in reject:
            blackboard.record_failure(task.get("id"))

    blackboard.write("test_report", report)
    return report


def _title_to_filename(title: str, ext: str) -> str:
    # "Write NoteController.java" → "NoteController.java"
    # "Write note_router.py" → "note_router.py"
    words = title.split()
    for word in reversed(words):
        if ext in word:
            return word
        if word and word[0].isupper() and word.isidentifier():
            return word + ext
    fallback = (words[-1] if words else "Unknown") + ext
    print(f"⚠️  _title_to_filename: could not parse filename from title {title!r}; falling back to {fallback!r}")
    return fallback
