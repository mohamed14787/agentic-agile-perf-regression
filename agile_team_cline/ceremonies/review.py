import json
import re
from camel.messages import BaseMessage
from agile_team_cline.agents.base import make_agent, zh_prompt
from agile_team_cline.config import TASK_TOKEN_LIMITS
from agile_team_cline.shared import extract_json, step_with_retry

_QA_GOAL = "Present test findings to the Product Owner. Be concise — one short summary only."

_PO_GOAL = "Review QA findings and output an acceptance verdict as a JSON object. No prose."

_PO_VERDICT_PROMPT = (
    "⚡ Based on this QA report, evaluate the tasks in the backlog and output a JSON object matching this schema (no prose, no markdown fences):\n"
    '{{\n'
    '  "accepted": ["ID_OF_ACCEPTED_TASK_1", "ID_OF_ACCEPTED_TASK_2"],\n'
    '  "rejected": ["ID_OF_REJECTED_TASK_1"],\n'
    '  "notes": "Brief explanation of rejection reasons and compilation issues."\n'
    '}}\n\n'
    "QA Report:\n{test_report}\n\n"
    "Backlog (each entry is {{id, title}}):\n{backlog}\n\n"
    "RULES:\n"
    "- Every task ID from the backlog must be categorized as EITHER accepted or rejected (never both, and no omissions).\n"
    "- accepted/rejected MUST contain ONLY the bare task IDs (e.g., [\"S1-1\", \"S2-1\"]), NOT titles or description strings.\n"
    "- Judge ONLY on the QA report's concrete evidence: the \"compiles\" boolean, the \"tests\" summary (executed/passed/failed counts), and the \"issues\" list. Do NOT assume tests are missing or failing when the report shows otherwise.\n"
    "- \"compiles\" reflects the PRODUCTION code; \"test_compiles\" reflects the test sources. If \"compiles\" is true, the application builds — a false \"test_compiles\" or a compile issue in a *Test.java file is NOT a build-wide failure. Reject only the specific test file's task if one exists; never reject the whole backlog over a broken test.\n"
    "- ACCEPT a task when the build compiles, the report shows tests executed with zero failures, and no issue's \"file\" corresponds to that task. When the report shows tests passed, treat functionality as verified — do NOT reject for \"unverified\" reasons.\n"
    "- REJECT a task only when there is real evidence against it: a compilation error, a failed test, or an issue whose \"file\" is the task's file. If \"tests_failed\" > 0, reject the tasks for the files named in the failing-test issues.\n"
    "- An issue whose \"concern\" starts with \"RUNTIME SMOKE FAILURE\" is hard evidence from actually running the built application — always REJECT the task owning that issue's \"file\", even if compilation and tests are otherwise green.\n"
    "- Reject a task ONLY if the QA report contains hard evidence against its file: a compile error, a failed test, or an issue entry naming that file. Code-style concerns, suspected logic problems you inferred by reading code, or \"this might not work at runtime\" reasoning are NOT grounds for rejection — accept the task and put the concern in notes.\n"
    "- Output ONLY the raw JSON object."
)

_PO_VERDICT_PROMPT_ZH = (
    "⚡ 根据以下QA报告，评估backlog中的各项任务，输出符合此schema的JSON对象（不要有说明文字，不要markdown代码块）：\n"
    '{{\n'
    '  "accepted": ["ID_OF_ACCEPTED_TASK_1", "ID_OF_ACCEPTED_TASK_2"],\n'
    '  "rejected": ["ID_OF_REJECTED_TASK_1"],\n'
    '  "notes": "Brief explanation of rejection reasons and compilation issues."\n'
    '}}\n\n'
    "QA Report:\n{test_report}\n\n"
    "Backlog (each entry is {{id, title}}):\n{backlog}\n\n"
    "规则：\n"
    "- backlog中每个task ID必须被归入accepted或rejected两者之一（不可同时出现在两边，也不可遗漏）。\n"
    "- accepted/rejected中只能放裸task ID（例如[\"S1-1\", \"S2-1\"]），不能放标题或描述文字。\n"
    "- 只依据QA报告中的具体证据来判断：\"compiles\"布尔值、\"tests\"汇总（执行/通过/失败数）、以及\"issues\"列表。报告显示正常时，不要凭空假设测试缺失或失败。\n"
    "- \"compiles\"反映生产代码；\"test_compiles\"反映测试源码。若\"compiles\"为true，说明应用本身可构建——某个*Test.java的\"test_compiles\"为false或其编译问题，不构成整体构建失败。仅拒绝该测试文件对应的task（如存在）；不要因为一个坏测试拒绝整个backlog。\n"
    "- 当构建通过编译、报告显示测试执行且零失败、且没有任何issue的\"file\"指向该task时，ACCEPT该task。报告显示测试通过即视为功能已验证——不要以\"未验证\"为由拒绝。\n"
    "- 只有存在真实不利证据时才REJECT一个task：编译错误、失败的测试、或某条issue的\"file\"正是该task的文件。若\"tests_failed\" > 0，拒绝失败测试issue中所指文件对应的task。\n"
    "- 任何\"concern\"以\"RUNTIME SMOKE FAILURE\"开头的issue，都是真实运行已构建应用得到的硬证据——即使编译和测试均为绿色，也必须REJECT该issue的\"file\"所属的task。\n"
    "- 仅当QA报告对某文件含有硬证据时才REJECT其task：编译错误、失败的测试、或指名该文件的issue条目。代码风格问题、你通过读代码推测的潜在逻辑问题、或\"这在运行时可能不work\"之类的推断，均不构成拒绝理由——应ACCEPT该task，并将顾虑写入notes。\n"
    "- 只输出原始JSON对象。"
)

# Match leading IDs like S1-T1, S2-1, T3 — anything before the first colon/space.
_ID_PREFIX_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_\-]*)")


def _normalize_id(raw, valid_ids: set) -> str:
    """Strip 'S2-1: title' → 'S2-1'. Return '' if no recognizable ID."""
    if not isinstance(raw, str):
        return ""
    # Cheap path: already a bare ID
    if raw in valid_ids:
        return raw
    m = _ID_PREFIX_RE.match(raw)
    if m and m.group(1) in valid_ids:
        return m.group(1)
    # Fallback: split on common separators
    for sep in (":", " ", "\t", "—", "-"):
        candidate = raw.split(sep, 1)[0].strip()
        if candidate in valid_ids:
            return candidate
    return ""


def run_sprint_review(test_report: dict, backlog: list, evidence: list = None) -> dict:
    tokens = TASK_TOKEN_LIMITS["review"]
    qa = make_agent("QA Engineer", _QA_GOAL, tokens)
    po = make_agent("Product Owner", _PO_GOAL, tokens)

    # Expose IDs distinctly so the PO can quote them verbatim.
    backlog_summary = json.dumps(
        [{"id": t.get("id", ""), "title": t.get("title", "")} for t in backlog],
        separators=(",", ":"),
    )
    valid_ids = {t.get("id", "") for t in backlog if t.get("id")}
    report_str = json.dumps(test_report, separators=(",", ":"))

    # Deterministic disk evidence gathered at verdict time (pipeline._task_evidence):
    # settles existence claims so the judges can't reject on-disk code as "missing"
    # when the file post-dates the QA report's context (coverage-gate/test-authoring
    # files). Appended AFTER .format() so the JSON needs no brace escaping.
    evidence_block = ""
    if evidence:
        evidence_block = (
            "\n\nDISK EVIDENCE (deterministic, gathered at verdict time — AUTHORITATIVE "
            "for file existence):\n" + json.dumps(evidence, separators=(",", ":")) +
            "\n- A task whose file shows \"exists\":true IS on disk right now — NEVER "
            "reject it as missing/not-provided; judge only compile/test/issue evidence.\n"
            "- \"deterministic_checks\":\"all pass\" means the task's verification "
            "functions pass against the current tree.\n"
            "- \"exists\":false is settled too: the file is genuinely absent."
        )

    # Round 1: QA presents findings
    qa_msg = step_with_retry(qa, BaseMessage.make_user_message(
        role_name="User",
        content=(f"Present this QA report to the Product Owner:\n{report_str}\n"
                 f"Backlog: {backlog_summary}{evidence_block}"),
    ))
    qa_text = qa_msg.msgs[0].content if qa_msg.msgs else f"QA report: {report_str}"

    # Round 2: PO outputs verdict JSON
    verdict_prompt = zh_prompt(_PO_VERDICT_PROMPT, _PO_VERDICT_PROMPT_ZH).format(
        test_report=report_str,
        backlog=backlog_summary,
    ) + evidence_block
    po_resp = step_with_retry(po, BaseMessage.make_user_message(role_name="QA Engineer", content=verdict_prompt))

    raw = po_resp.msgs[0].content if po_resp.msgs else ""
    result = extract_json(raw)
    if not result:
        all_ids = [t.get("id", "") for t in backlog]
        # Fallback to reject everything so developer gets a chance to refine, and pass full PO response as notes
        result = {"accepted": [], "rejected": all_ids, "notes": raw if raw else (qa_text if qa_text else "")}
    else:
        # Normalize accepted/rejected — PO often returns "<id>: <title>" strings
        # despite the prompt; strip back to bare IDs and discard anything we
        # don't recognise so the refinement loop matches reliably.
        for key in ("accepted", "rejected"):
            entries = result.get(key, [])
            if not isinstance(entries, list):
                entries = []
            normalized = []
            for entry in entries:
                bare = _normalize_id(entry, valid_ids)
                if bare and bare not in normalized:
                    normalized.append(bare)
            result[key] = normalized

        if isinstance(result.get("notes"), dict):
            if "summary" in result["notes"]:
                result["notes"] = str(result["notes"]["summary"])
            else:
                result["notes"] = json.dumps(result["notes"])
        elif result and not isinstance(result.get("notes"), str):
            result["notes"] = str(result.get("notes", ""))
    return result
