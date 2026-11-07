import json
import os
import re
from camel.messages import BaseMessage
from agile_team_cline.agents.base import make_agent, zh_prompt
from agile_team_cline.config import TASK_TOKEN_LIMITS, MAX_TASKS_PER_SPRINT, make_architect_model
from agile_team_cline.shared import step_with_retry
from agile_team_cline.shared.blackboard import Blackboard

_ANALYST_GOAL = (
    "Senior software architect auditing generated code for technical debt. "
    "Output fix tasks in strict pipe-delimited format only. No prose, no markdown."
)

_DEBT_PROMPT = (
    "Audit these source files for technical debt. Output ONLY fix tasks in this exact format:\n"
    "FIX_ID|description of fix|target_file|type\n"
    "Where type is one of: entity, service, controller, config, test, default\n"
    "Where target_file is REQUIRED and MUST be the exact filename to modify or create\n"
    "(e.g. NotesApplication.java, NoteController.java). Do NOT leave target_file empty.\n\n"
    "Look for:\n"
    "- Duplicate methods across services (e.g. joinRoom defined in two services)\n"
    "- No-op stub methods that should have real implementations\n"
    "- Missing classes that are imported but not generated\n"
    "- Missing @SpringBootApplication main class\n"
    "- Redundant service logic that should be consolidated\n"
    "- Broken or placeholder code marked with TODO/FIXME\n\n"
    "Max {max_tasks} fix tasks. If no debt found output exactly: NONE\n\n"
    "Generated files:\n{files}\n\n"
    "Contract: {contract}\n\n"
    "Output fix tasks:"
)

_DEBT_PROMPT_ZH = (
    "审查以下源文件中的技术债务。仅以此精确格式输出修复任务：\n"
    "FIX_ID|description of fix|target_file|type\n"
    "其中type须为以下之一：entity, service, controller, config, test, default\n"
    "其中target_file为必填项，必须是要修改或创建的精确文件名\n"
    "（例如 NotesApplication.java, NoteController.java）。target_file不可留空。\n\n"
    "重点关注：\n"
    "- 跨service的重复方法（例如joinRoom在两个service中都有定义）\n"
    "- 应有真实实现却是空壳stub的方法\n"
    "- 被import但未生成的缺失类\n"
    "- 缺失的@SpringBootApplication主类\n"
    "- 应合并的冗余service逻辑\n"
    "- 标记TODO/FIXME的损坏或占位代码\n\n"
    "最多{max_tasks}个修复任务。若未发现债务，精确输出：NONE\n\n"
    "Generated files:\n{files}\n\n"
    "Contract: {contract}\n\n"
    "输出修复任务："
)


def run_tech_debt_sprint(blackboard: Blackboard, stack: dict, model) -> dict:
    from agile_team_cline.sprint import _run_dev_sprint

    bb_data = blackboard.read("blackboard")
    all_files = bb_data.get("generated_files", [])
    baseline = bb_data.get("milestone_file_baseline", 0)
    # Audit only this milestone's files; older milestones already passed review.
    generated_files = all_files[baseline:]
    contract = blackboard.read("contract")
    output_dir = str(blackboard.output_dir)

    # Cap to the most recent N — auditing too many files blows the prompt and
    # gives the architect nothing actionable.
    max_audit = int(os.getenv("AGILE_TECH_DEBT_MAX_FILES", "8"))
    max_lines = int(os.getenv("AGILE_TECH_DEBT_MAX_LINES", "60"))
    audit_files = generated_files[-max_audit:]

    files_parts = []
    for filename in audit_files:
        filepath = ""
        for root, _, files in os.walk(output_dir):
            if filename in files:
                filepath = os.path.join(root, filename)
                break
        if filepath and os.path.exists(filepath):
            try:
                lines = open(filepath).readlines()[:max_lines]
                content = "".join(lines)
                files_parts.append(f"--- {filename} ---\n{content}\n")
            except Exception:
                pass

    if not files_parts:
        return {"fixed": [], "skipped": "no files found"}

    files_str = "\n".join(files_parts)

    # Architect audits for debt
    tokens = TASK_TOKEN_LIMITS["architect_planning"]
    analyst = make_agent("Software Architect", _ANALYST_GOAL, tokens,
                         model=make_architect_model(tokens))
    prompt = zh_prompt(_DEBT_PROMPT, _DEBT_PROMPT_ZH).format(
        files=files_str,
        contract=json.dumps(contract, separators=(",", ":")),
        max_tasks=MAX_TASKS_PER_SPRINT,
    )
    msg = BaseMessage.make_user_message(role_name="User", content=prompt)
    response = step_with_retry(analyst, msg)
    raw = response.msgs[0].content if response.msgs else ""

    if not raw or raw.strip().upper().startswith("NONE"):
        return {"fixed": [], "debt_found": False}

    # Parse into dev-sprint-compatible tasks
    debt_tasks = _parse_debt_tasks(raw, stack["file_extension"])
    if not debt_tasks:
        return {"fixed": [], "debt_found": False}

    # Stamp a sprint number beyond current max so _run_dev_sprint bookkeeping works.
    # Re-read here — bb_data above was captured before refinement rounds may have
    # advanced sprint_num, so reusing it can collide with refinement sprint IDs.
    current_bb = blackboard.read("blackboard") or {}
    tech_sprint_num = current_bb.get("sprint_num", 0) + 1
    for t in debt_tasks:
        t["sprint"] = tech_sprint_num

    results = _run_dev_sprint(debt_tasks, blackboard, stack, model)
    fixed_files = [v for v in results.values() if v]

    return {
        "fixed": fixed_files,
        "debt_found": True,
        "tasks": [t["title"] for t in debt_tasks],
    }


_CLASS_NAME_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]+)\b")


def _parse_debt_tasks(text: str, ext: str) -> list:
    tasks = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        if line.lower().startswith("fix_id") or line.lower().startswith("id|"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        fix_id, description, target_file = parts[0], parts[1], parts[2]
        task_type = parts[3].lower() if len(parts) > 3 else "default"

        # Empty target_file used to silently become "default.java" via the
        # downstream filename fallback. Try to recover a class name from the
        # description; if that fails, skip the task entirely.
        if not target_file:
            m = _CLASS_NAME_RE.search(description)
            if m:
                target_file = m.group(1)
                print(f"⚠️  tech-debt: target_file empty for {fix_id!r}; derived {target_file!r} from description")
            else:
                print(f"⚠️  tech-debt: skipping {fix_id!r} — no target_file and no class name in description: {description!r}")
                continue

        if "." not in target_file:
            target_file += ext
        tasks.append({
            "id": fix_id,
            "title": description,
            "type": task_type,
            "target_file": target_file,
        })
    return tasks[:MAX_TASKS_PER_SPRINT]
