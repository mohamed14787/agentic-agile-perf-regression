#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

from agile_team_cline.config import make_model, make_architect_model, make_thinking_model, TASK_TOKEN_LIMITS, TOTAL_SPRINTS, MAX_TASKS_PER_SPRINT
from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.scrum_master import ScrumMaster
from agile_team_cline.ceremonies.planning import run_sprint_planning, plan_next_sprint, build_coverage_tasks
from agile_team_cline.ceremonies.review import run_sprint_review
from agile_team_cline.sprint import run_sprint, _title_to_filename
from agile_team_cline.agents.base import make_agent
from agile_team_cline.cline_runner import run_cline_with_retry
from camel.messages import BaseMessage
from agile_team_cline.shared import step_with_retry, extract_json
from agile_team_cline.workspace_discovery import discover_workspace_state, should_skip_pom_regeneration

_STACKS_DIR = Path(__file__).parent / "stacks"

_DECOMPOSE_PROMPT = (
    "You are a Senior Product Manager. Your job is to take a large, complex software feature request "
    "and decompose it into a list of 2 to 3 smaller, incremental, sequential milestones. "
    "Each milestone must build on top of the previous one. "
    "Output ONLY a raw JSON array of strings representing the milestones. No explanations. No markdown formatting. "
    "Example request: 'Build a Social Media Feed system with posts, follows, likes, comments, notifications, and timeline aggregation using Spring Boot and WebSocket'\n"
    "Example output:\n"
    "[\n"
    "  \"Build a Social Media system with posts and likes using Spring Boot\",\n"
    "  \"Add follows and timeline aggregation to the Social Media system using Spring Boot\",\n"
    "  \"Add comments, real-time notifications, and WebSocket support to the Social Media system\"\n"
    "]\n\n"
    "Now decompose this request:\n"
    "'{feature}'"
)


def _dedup_backlog(backlog: list) -> list:
    """Collapse tasks that target the same file (or share an id) to a single
    entry, keeping the most recent. The coverage safety-net re-emits tasks for
    files it rebuilds (e.g. SVC-Visit) that already exist in the backlog; left
    un-deduped these duplicate the rejected set every refinement round so it can
    never shrink to zero (the PO even rejects them as duplicates)."""
    seen = {}
    order = []
    for t in backlog:
        key = t.get("target_file") or t.get("id")
        if key not in seen:
            order.append(key)
        seen[key] = t  # last occurrence wins (most recent state)
    return [seen[k] for k in order]


def _judgeable_backlog(backlog: list) -> list:
    """The subset of the backlog the PO is allowed to judge. With AGILE_AUTHOR_TESTS off
    (default), type:test tasks are never authored — sprint.py drops them before any cline
    session runs — so handing them to the verdict only manufactures phantom 'disputed'
    labels (no code exists, so the PO can never find evidence; after 2 rejects they go
    disputed). Exclude them from the judged list ONLY; the persisted blackboard backlog is
    untouched (other logic reads it). Same predicate sprint.py uses to skip them."""
    if os.getenv("AGILE_AUTHOR_TESTS", "0") == "1":
        return backlog
    return [t for t in backlog if "test" not in str(t.get("type", "")).lower()]


def _task_evidence(backlog: list, bb, stack: dict = None) -> list:
    """Deterministic per-task disk evidence gathered AT VERDICT TIME for the QA/PO prompts.
    Invariant: a judge must see the tree as it exists when judging. Files created late in
    the sprint (coverage gate, test authoring) post-date the QA context assembly, which
    made the PO reject on-disk code as 'entirely missing' in 4 of the last 5 runs — an
    ordering bug, not a hallucination. Existence claims are settled by this block; the
    model judges only quality. Non-blocking: any error yields partial/empty evidence."""
    import datetime
    evidence = []
    for t in backlog:
        entry = {"id": t.get("id", "")}
        target = os.path.basename(str(t.get("target_file") or ""))
        if not target:
            evidence.append(entry)
            continue
        entry["file"] = target
        try:
            rel = bb.find_existing(target)
            if rel:
                st = os.stat(os.path.join(str(bb.output_dir), rel))
                entry.update({
                    "exists": True, "path": rel, "bytes": st.st_size,
                    "modified": datetime.datetime.fromtimestamp(st.st_mtime)
                                .isoformat(timespec="seconds"),
                })
            else:
                entry["exists"] = False
        except Exception:
            pass
        evidence.append(entry)
    # Freshest deterministic VF verdicts (same staleness class: dev-sprint VF results
    # pre-date the coverage gate). Re-run is cheap and side-effect-free.
    try:
        from agile_team_cline.verify_lib import run_task_vfs
        vf_now = run_task_vfs([t for t in backlog if t.get("verifications")],
                              bb, stack or {}, "1")
        for entry in evidence:
            fails = vf_now.get(entry.get("id", ""))
            if entry.get("id"):
                entry["deterministic_checks"] = fails if fails else "all pass"
    except Exception:
        pass
    return evidence


def _vf_floor(summary: dict, backlog: list, bb, stack: dict, test_report: dict) -> dict:
    """Deterministic FLOOR on the LLM Product Owner verdict.

    Runs #37-51: the PO disputed CTL-*/REPO-* tasks whose code is provably on disk — 9/9
    'missing endpoint' disputes verified false. The PO can reject but cannot see the tree; the
    deterministic VFs can. So after the verdict, any rejected/disputed task whose deterministic
    VFs ALL pass (endpoint_served / file_exists / parse_ok …) on a GREEN tree is moved to
    accepted: ground truth overrides the LLM. This is paper-[4]'s 'don't trust AI-judging-AI'
    applied to the PO itself — a floor it cannot dip below. Runs independent of
    AGILE_VERIFY_AWARE (the perf runs leave VFs off, but the floor still runs its own check
    here). Conservative: no-op on a non-compiling tree (nothing to certify); a task with no
    deterministic VF, or any VF that fails or errors, is left to the PO untouched (the floor
    never manufactures an accept it cannot prove)."""
    if not isinstance(summary, dict):
        return summary
    candidates = list(dict.fromkeys(
        list(summary.get("rejected", [])) + list(summary.get("disputed", []))))
    if not candidates:
        return summary
    if not (isinstance(test_report, dict) and test_report.get("compiles")):
        return summary  # never certify against a red tree
    try:
        from agile_team_cline.verify_lib import attach_verifications, VF_REGISTRY
    except Exception:
        return summary
    contract = bb.read("contract") or {}
    by_id = {t.get("id"): t for t in backlog}
    output_dir = str(bb.output_dir)
    overridden = []
    for tid in candidates:
        task = by_id.get(tid)
        if not task:
            continue
        try:
            attach_verifications(task, contract, stack)
        except Exception:
            continue
        det = [v for v in task.get("verifications", []) if "vf" in v]
        if not det:
            continue  # no deterministic check -> leave to the PO
        ran, ok = 0, True
        for vfd in det:
            fn = VF_REGISTRY.get(vfd["vf"])
            if fn is None:
                continue
            try:
                msg = fn(output_dir, vfd.get("args", {}), bb)
            except Exception:
                ok = False  # an erroring VF is not positive evidence -> never certify
                break
            ran += 1
            if msg:
                ok = False
                break
        if ok and ran >= 1:
            overridden.append(tid)
    if overridden:
        summary["accepted"] = list(dict.fromkeys(list(summary.get("accepted", [])) + overridden))
        summary["rejected"] = [t for t in summary.get("rejected", []) if t not in overridden]
        if summary.get("disputed"):
            summary["disputed"] = [t for t in summary["disputed"] if t not in overridden]
        print(f"   🟢 [VF floor] deterministic VFs pass on disk — overriding PO verdict, "
              f"accepting: {overridden}")
    return summary


_CEILING_KW = ("endpoint", "controller", "@path", "resource", "repository", "jax-rs", "jaxrs")


def _ceiling_targets(backlog: list, accepted: list) -> list:
    """Accepted tasks that produced REST/repository surface — the ones a duplicate-route /
    misplaced-repo collision implicates (matched by type, id prefix, target file, or title;
    the dupes accepts were S-prefixed sprint tasks, so title/type matching is required)."""
    by_id = {t.get("id"): t for t in backlog}
    out = []
    for tid in accepted:
        t = by_id.get(tid) or {}
        ttype = str(t.get("type", "")).lower()
        title = str(t.get("title", "")).lower()
        tgt = str(t.get("target_file", "")).lower()
        idu = str(tid).upper()
        if (ttype in ("controller", "endpoint", "repository")
                or idu.startswith(("CTL-", "REPO-", "ENDPOINT-"))
                or tgt.endswith(("endpoint.java", "controller.java", "repository.java"))
                or any(k in title for k in _CEILING_KW)):
            out.append(tid)
    return out


def _all_code_targets(backlog: list, accepted: list) -> list:
    """Accepted code-producing tasks (entity/dto/controller/endpoint/repository/service) —
    the whole-tree contract-fidelity checks (contract drift, orphan artifacts) aren't scoped
    to one file, so any of these could be implicated; unlike _ceiling_targets there is no
    narrower signal to match against."""
    by_id = {t.get("id"): t for t in backlog}
    out = []
    for tid in accepted:
        t = by_id.get(tid) or {}
        if str(t.get("type", "")).lower() in (
                "entity", "dto", "controller", "endpoint", "repository", "service"):
            out.append(tid)
    return out


def _all_accepted_targets(backlog: list, accepted: list) -> list:
    """Every currently accepted task, unfiltered by type or target_file. Used by the
    frozen-suite ceiling check, which cannot narrow to file ownership the way
    _ceiling_targets/_all_code_targets do: a failing test's owning source file is often
    not ANY current task's target_file (RFM 2026-08-02 — UserRepository.getRfmSegments()
    threw on 6/6 CustomerRfmSegment{Endpoint,Repository}Test cases, but the only task in
    the backlog was the DTO, so the PO verdict reasoned the failures were 'not part of
    task scope' and accepted anyway, twice, independently, ~15M tokens for two attempts
    that both then died at the Phase D/E equivalence gate). A red suite makes every
    accept in the batch suspect until proven otherwise, so this check has no narrower
    target function to fall back on."""
    return list(accepted)


def _frozen_suite_msg(test_report: dict) -> str:
    """Whole-tree check: is the frozen suite (the senior-authored tests, unmodified by
    the junior) still red? Mirrors vf_contract_types_realized's ceiling role but for
    test execution instead of contract fidelity — Sprint Review's per-task verdict can
    rationalize a failing test away as belonging to no current task ('not part of task
    scope'); this check ignores task ownership entirely, exactly the blind spot that
    let it happen. Returns '' (no-op) on a clean or non-executed suite."""
    if not isinstance(test_report, dict):
        return ""
    failed = test_report.get("tests_failed", 0)
    if not failed:
        return ""
    names = sorted({i.get("file") for i in test_report.get("issues", []) if i.get("file")})
    shown = ", ".join(names[:8]) + ("…" if len(names) > 8 else "")
    return (f"{failed} frozen-suite test(s) still failing ({shown or 'no file mapped'}) — "
            f"a red suite is never acceptable, regardless of which backlog task's files "
            f"the failures are attributed to.")


def _vf_ceiling(summary: dict, backlog: list, bb, stack: dict, test_report: dict,
                final: bool = False) -> dict:
    """Deterministic CEILING on the PO verdict: whole-tree checks the PO's per-task verdict
    never runs against, so an 'accept' on a task that never entered rejection can still hide
    a defect the compiler can't see. Three checks, all re-opening implicated accepts — at the
    initial verdict downgrading to rejected (refinement fixes it in place); at the closing
    verdict downgrading to disputed (honest final label — never ship a known defect green):

    - `vf_no_duplicate_route` (run #51 dupes false-green): the feature was cloned into a
      parallel @Path('products') resource + a repository in the wrong package, and the PO
      accepted over QA's own collision warning. JAX-RS only.
    - `vf_contract_types_realized` (2026-07-05 stockledger probe): a task can be accepted
      without QA/PO ever flagging it, yet still leave a contract type dead/absent or a new
      file this run created wired to nothing (PersistenceConfig.java in that run) — the VF
      floor only upgrades false rejections, it never downgrades an accept nothing rejected in
      the first place, so this whole-tree check needs its own ceiling entry, same as dupes.
    - `_frozen_suite_msg` (2026-08-02 rfm false-accept): the senior track's equivalent defect
      (a JPQL aggregate over a converter-mapped column throwing ClassCastException) was caught
      by name via a task-scoped VF ('RUNTIME EXECUTION DEFECT') because refinement had an
      explicit task to pin it to; the junior track's identical defect landed inside a
      generically-scoped task, so QA's own test run coming back red never blocked the verdict.
      A whole-tree, ownership-blind check closes that gap the same way the other two do.

    No-op when the tree is clean or when the implicated VF errors."""
    if not isinstance(summary, dict):
        return summary
    accepted = list(summary.get("accepted", []))
    if not accepted:
        return summary
    try:
        from agile_team_cline.verify_lib import vf_no_duplicate_route, vf_contract_types_realized
    except Exception:
        return summary
    spring = ("spring" in str(stack.get("framework", "")).lower()
              or "spring" in str(stack.get("name", "")).lower())
    checks = [
        (lambda: vf_no_duplicate_route(str(bb.output_dir), {"is_spring": spring}, bb),
         "duplicate-route / shadow-resource", _ceiling_targets),
        (lambda: vf_contract_types_realized(str(bb.output_dir), {}, bb),
         "contract drift / orphan artifact", _all_code_targets),
        (lambda: _frozen_suite_msg(test_report),
         "frozen-suite test failure", _all_accepted_targets),
    ]
    bucket = "disputed" if final else "rejected"
    for run_check, label, target_fn in checks:
        accepted = list(summary.get("accepted", []))
        if not accepted:
            break
        try:
            msg = run_check()
        except Exception:
            msg = None
        if not msg:
            continue
        targets = target_fn(backlog, accepted)
        if not targets:
            continue
        summary["accepted"] = [t for t in accepted if t not in targets]
        summary[bucket] = list(dict.fromkeys(list(summary.get(bucket, [])) + targets))
        summary["notes"] = (str(summary.get("notes", "")) + f"\n[VF ceiling] {msg}").strip()
        action = "disputing (final)" if final else "re-opening for refinement"
        print(f"   🔴 [VF ceiling] {label} — {action}: {targets}")
        print(f"        {msg}")
    return summary


def _arch_valid(content: str) -> bool:
    # A real analysis is a multi-section markdown doc. An 8s model bail
    # ("I need the repository root path...") is short and heading-free —
    # must never overwrite a good prior milestone's analysis.
    return bool(content) and len(content) >= 1000 and "##" in content


def _append_milestone_delivered(output_dir: str, milestone_idx: int, conventions: dict) -> None:
    """Deterministically append a '## Milestone N delivered' section to
    ARCHITECTURE.md from the reconciled contract + on-disk conventions (drift-control
    mechanism E). No LLM call, so a model bail can never zero the file. No-op if
    ARCHITECTURE.md doesn't exist yet (first milestone's LLM analysis owns creation)."""
    arch_file = os.path.join(output_dir, "ARCHITECTURE.md")
    try:
        with open(arch_file) as f:
            prior = f.read()
    except OSError:
        return
    if not _arch_valid(prior):
        return

    lines = [f"\n\n## Milestone {milestone_idx} delivered\n"]
    id_types = conventions.get("id_types") or {}
    if id_types:
        lines.append("Entities (id types):")
        for name, t in id_types.items():
            lines.append(f"- {name}.id: {t}")
    routes = conventions.get("routes") or []
    if routes:
        lines.append("\nEndpoints:")
        for r in routes:
            lines.append(f"- {r}")
    packages = conventions.get("packages") or []
    if packages:
        lines.append("\nPackages:")
        for p in packages:
            lines.append(f"- {p}")

    try:
        with open(arch_file, "a") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"   ⚠️  Could not append to ARCHITECTURE.md: {e}")


def _render_architecture_md(output_dir: str) -> str:
    """Deterministic, no-LLM fallback for ARCHITECTURE.md when both plan-mode
    analysis attempts bail. Renders the same on-disk-verified facts Mechanism E
    already computes (id types, packages, routes) plus the project's Spring Boot
    version, so the file is always _arch_valid and _append_milestone_delivered
    can fire on subsequent milestones."""
    from agile_team_cline.ceremonies.planning import _extract_conventions
    from agile_team_cline.repair import _project_version_hint

    conventions = _extract_conventions(output_dir)
    version = _project_version_hint(output_dir)

    lines = ["# Architecture Analysis", "", "## Existing Architecture", ""]
    lines.append("This section is generated deterministically from the conventions observed on disk "
                  "at the last green checkpoint (no LLM read of the source tree was available).")
    lines.append("")
    lines.append("### Framework Details")
    lines.append(f"Spring Boot version: {version or 'unknown'}")
    lines.append("")

    lines.append("### Package Structure")
    for p in conventions.get("packages") or []:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("### Entity Patterns")
    for name, t in (conventions.get("id_types") or {}).items():
        lines.append(f"- {name}.id: {t}")
    lines.append("")

    lines.append("### Existing Endpoints")
    for r in conventions.get("routes") or []:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("### Key Constraints")
    lines.append("- New entities must use the id types, package layout, and route "
                  "conventions above unless the contract specifies otherwise.")

    return "\n".join(lines) + "\n"


def _analyze_and_document_architecture(output_dir: str, milestone_idx: int) -> str:
    """Analyze workspace architecture using Cline in read-only mode and generate architecture.md.

    This runs Cline in plan mode (-p) to examine the existing codebase and document:
    - Package structure and entity relationships
    - Service/Repository/Controller patterns
    - Framework setup (Spring version, annotations)
    - Existing configurations and constraints

    Returns the content of architecture.md and writes it to disk.
    """
    arch_file = os.path.join(output_dir, "ARCHITECTURE.md")

    # Deterministic-first (default): _render_architecture_md builds a valid
    # ARCHITECTURE.md from on-disk-verified conventions (packages, id types, routes)
    # at a fraction of the cost of the LLM plan-mode scan (309K-610K tokens/run) —
    # the downstream cline architect (_derive_contract_cline) reads entities/base
    # classes itself regardless, so the LLM scan was largely duplicated work.
    # Set AGILE_ARCH_ANALYSIS=llm to restore the old plan-mode analysis.
    if os.getenv("AGILE_ARCH_ANALYSIS", "deterministic") != "llm":
        arch_content = _render_architecture_md(output_dir)
        try:
            with open(arch_file, "w") as f:
                f.write(arch_content)
            print(f"   ✅ Architecture documented (deterministic): {arch_file} ({len(arch_content)} bytes)")
        except Exception as e:
            print(f"   ⚠️  Could not write ARCHITECTURE.md: {e}")
        return arch_content

    analysis_brief = (
        "Analyze this Java/Spring codebase and document these findings as markdown:\n"
        "## Existing Architecture\n\n"
        "### Package Structure\n"
        "List all packages and their purpose.\n\n"
        "### Entity Patterns\n"
        "For each entity: name, field types (UUID/String/Long/@Id strategy), relationships.\n\n"
        "### Service & Repository Patterns\n"
        "Show method signatures and naming conventions used.\n\n"
        "### Framework Details\n"
        "Spring Boot version, Jakarta/javax imports, @Configuration classes, Hibernate setup.\n\n"
        "### Key Constraints\n"
        "What patterns MUST be followed for new code.\n"
    )

    analysis_system = (
        "You are a code analyst in read-only plan mode. Explore the project's source with your "
        "available tools: search_codebase (regex search across the workspace), read_files (open one "
        "or more files by path), and run_commands (read-only shell commands like `find` or `ls` work "
        "in this mode). You MUST actually search and open files — do not guess from file names alone. "
        "If a search returns 0 results, retry with a broader or different query (e.g. drop a path "
        "prefix, search for a class name instead of a package) before concluding the workspace is "
        "empty. Do not call ask_question — there is no human to respond. Start by running a broad "
        "search or `find` for *.java files under src/main/java, then read the entities and their base "
        "classes. When documenting Entity Patterns, report each entity's EXACT id Java type by reading "
        "the base class (e.g. BaseEntity) it extends."
    )
    print(f"   📊 Analyzing workspace architecture (read-only mode)...")
    arch_content = ""
    for attempt in range(2):
        result = run_cline_with_retry(
            output_dir,
            analysis_brief,
            plan=True,  # Read-only mode
            timeout=600,
            title="Architecture Analysis",
            system=analysis_system,
        )
        if result.success and _arch_valid(result.text):
            arch_content = result.text
            break
        preview = (result.text or "").strip()[:150]
        print(f"   ⚠️  Architecture analysis output invalid (attempt {attempt + 1}: "
              f"{len(result.text or '')} bytes, finish={result.finish_reason}): {preview!r}")

    if not arch_content:
        # Both attempts bailed — keep a valid prior milestone's analysis if present.
        try:
            with open(arch_file) as f:
                prior = f.read()
        except OSError:
            prior = ""
        if _arch_valid(prior):
            print(f"   ↩️  Keeping prior ARCHITECTURE.md ({len(prior)} bytes)")
            return prior
        # Both LLM attempts bailed and there's no usable prior analysis — fall back
        # to a deterministic render from on-disk-verified conventions so the file
        # is always _arch_valid and Mechanism E's milestone-delivered append can fire.
        arch_content = _render_architecture_md(output_dir)
        print(f"   ↩️  Falling back to deterministic ARCHITECTURE.md render")

    try:
        with open(arch_file, "w") as f:
            f.write(arch_content)
        if _arch_valid(arch_content):
            print(f"   ✅ Architecture documented: {arch_file} ({len(arch_content)} bytes)")
        else:
            print(f"   ⚠️  Wrote ARCHITECTURE.md but content is not valid ({len(arch_content)} bytes)")
    except Exception as e:
        print(f"   ⚠️  Could not write ARCHITECTURE.md: {e}")

    return arch_content


def _snapshot_sources(blackboard) -> dict:
    """Capture the current generated source tree (contents + file list) so a
    later refinement round that breaks compilation can be rolled back. Also
    snapshots the full project file SET (src/** + pom.xml) — under agentic cline
    editing a later round may CREATE files not in generated_files, and since
    Maven compiles the whole src/ tree those orphans would survive a content-only
    restore and keep the build red. `src_set` lets _restore_sources delete them."""
    import os
    bb = blackboard.read("blackboard")
    out = str(blackboard.output_dir)
    files = {}
    for rel in bb.get("generated_files", []):
        p = os.path.join(out, rel)
        if os.path.exists(p):
            try:
                files[rel] = open(p).read()
            except Exception:
                pass
    return {"files": files, "gen": list(bb.get("generated_files", [])),
            "src_set": set(blackboard.snapshot_src().keys())}


def _restore_sources(blackboard, snap: dict) -> None:
    import os
    out = str(blackboard.output_dir)
    for rel, content in snap["files"].items():
        p = os.path.join(out, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            open(p, "w").write(content)
        except Exception:
            pass
    # Delete project files created since the green snapshot (orphans that would
    # otherwise keep the whole-tree Maven compile broken). Only files captured by
    # snapshot_src (src/** + pom.xml) are considered, never the JSON state.
    snap_set = snap.get("src_set")
    if snap_set is not None:
        for rel in blackboard.snapshot_src().keys():
            if rel not in snap_set:
                try:
                    os.remove(os.path.join(out, rel))
                except OSError:
                    pass
    bb = blackboard.read("blackboard")
    bb["generated_files"] = list(snap["gen"])
    blackboard.write("blackboard", bb)


def load_stack(stack_name: str) -> dict:
    path = _STACKS_DIR / f"{stack_name}.json"
    if not path.exists():
        available = [p.stem for p in _STACKS_DIR.glob("*.json")]
        print(f"[ERROR] Stack '{stack_name}' not found. Available: {available}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


# ── Deterministic Java scaffold (cline fork) ──────────────────────────────
# pom.xml / application.properties are pure boilerplate, but cline's config-task
# session dies nondeterministically on a tool-format slip and never writes them,
# leaving no build file → the maven gate can't run. Seed them from a known-good
# Spring Boot 3 + H2 template so the gate always has a pom; cline then skips them
# (its brief already says "do NOT recreate pom.xml/application.properties").
_SCAFFOLD_POM = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
        <relativePath/>
    </parent>
    <groupId>com.example</groupId>
    <artifactId>demo</artifactId>
    <version>0.0.1-SNAPSHOT</version>
    <name>demo</name>
    <properties>
        <java.version>17</java.version>
    </properties>
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-validation</artifactId>
        </dependency>
        <dependency>
            <groupId>com.h2database</groupId>
            <artifactId>h2</artifactId>
            <scope>runtime</scope>
        </dependency>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
</project>
"""

_SCAFFOLD_PROPERTIES = """spring.datasource.url=jdbc:h2:mem:testdb;DB_CLOSE_DELAY=-1
spring.datasource.driverClassName=org.h2.Driver
spring.datasource.username=sa
spring.datasource.password=
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect
spring.jpa.hibernate.ddl-auto=update
spring.h2.console.enabled=true
"""


def _seed_java_scaffold(bb, stack, workspace_state=None) -> None:
    """Write pom.xml + application.properties from a known-good template if absent,
    recording them so the incremental backlog filter drops the flaky config tasks.

    Skip if brownfield project (existing pom.xml + config)."""
    is_java = stack.get("language", "").lower() == "java" or stack.get("file_extension") == ".java"
    if not is_java:
        return

    # Skip scaffold seeding if brownfield (existing build config)
    is_brownfield = workspace_state.get("is_brownfield", False) if workspace_state else False
    if is_brownfield:
        print(f"   ⏭️  Skipping scaffold seeding (existing build config)")
        return

    seeded = []
    for name, content in (("pom.xml", _SCAFFOLD_POM), ("application.properties", _SCAFFOLD_PROPERTIES)):
        resolved = bb.resolve_path(name, content)
        if not (bb.output_dir / resolved).exists():
            seeded.append(bb.write_file(name, content))
    for rel in seeded:
        bb.record_file(rel)
    if seeded:
        print(f"   🔧 Seeded deterministic scaffold: {', '.join(seeded)}")


_CLINEIGNORE_CONTENT = """# Auto-generated by agile_team_cline. Keeps Cline's workspace scan on source only.
# Build artifacts (Maven/Gradle)
target/
build/
out/
.gradle/
gradle/
*.jar
*.war
*.class

# Pipeline-generated session logs (large, never source)
*_cline.jsonl
token_usage.jsonl
camel_token_usage.jsonl
*.log
milestone_checkpoint.json

# Dependencies
node_modules/
**/node_modules/

# Python
.venv/
venv/
__pycache__/
*.pyc

# IDE / OS
.idea/
.vscode/
*.iml
.DS_Store

# Env
.env
.env.*
"""


def _init_cline_ignore(output_dir: str) -> None:
    """Drop a .clineignore into the workspace so Cline skips build artifacts and
    session logs on every scan. No-op if one already exists (e.g. resumed run)."""
    try:
        ignore_path = Path(output_dir) / ".clineignore"
        if ignore_path.exists():
            return
        ignore_path.parent.mkdir(parents=True, exist_ok=True)
        ignore_path.write_text(_CLINEIGNORE_CONTENT)
    except Exception:
        pass


def _module_scope(workspace_state: dict) -> dict:
    """Resolve the Phase 3 single-module run scope from AGILE_BUILD_MODULE (the same
    module path Phase 0 compiles and Phase 1 discovers). Returns
    {target, deps, ignore} where `ignore` is every OTHER discovered module's path
    (siblings to hide from cline), or None when unset/invalid (no scoping)."""
    target = os.getenv("AGILE_BUILD_MODULE", "").strip()
    if not target:
        return None
    mods = {m["path"] for m in workspace_state.get("modules", [])}
    # AGILE_BUILD_MODULE may name several comma-separated modules (Phase 7 cross-module
    # feature). Each is validated independently; unknown entries are dropped with a
    # warning. Single-module runs (one valid entry) are unchanged.
    targets = [t.strip() for t in target.split(",") if t.strip()]
    for t in targets:
        if t not in mods:
            print(f"   ⚠️ AGILE_BUILD_MODULE entry '{t}' not among discovered modules — ignored")
    valid = [t for t in targets if t in mods]
    if not valid:
        print(f"   ⚠️ AGILE_BUILD_MODULE='{target}' not among discovered modules — scope skipped")
        return None
    deps = {d.strip() for d in os.getenv("AGILE_BUILD_MODULE_DEPS", "").split(",") if d.strip()}
    keep = set(valid) | deps
    return {"target": ",".join(valid), "targets": valid, "deps": deps, "ignore": sorted(mods - keep)}


def _snapshot_pom_files(output_dir: str, bb) -> None:
    """Snapshot all pom.xml files right after workspace setup, to a DEDICATED store
    (pom_snapshot.json). The build gate (_run_maven_compile) restores any that cline
    deletes mid-session. Must NOT piggyback on the 'blackboard' store — that store is
    rewritten with a fixed schema each sprint, which silently drops extra keys."""
    if bb.read("pom_snapshot"):
        return
    snap = {}
    for dirpath, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        if "pom.xml" in files:
            p = os.path.join(dirpath, "pom.xml")
            try:
                snap[os.path.relpath(p, output_dir)] = open(p).read()
            except Exception:
                pass
    bb.write("pom_snapshot", snap)


def _apply_module_scope_ignore(output_dir: str, workspace_state: dict) -> None:
    """Append non-target module dirs to .clineignore so cline's codebase scan stays on
    the target module + its shared deps. No-op when not module-scoped."""
    scope = _module_scope(workspace_state)
    if not scope or not scope["ignore"]:
        return
    try:
        with open(Path(output_dir) / ".clineignore", "a") as f:
            f.write("\n# Phase 3 module scope: hide non-target modules\n")
            for p in scope["ignore"]:
                f.write(f"{p}/\n")
    except Exception:
        pass
    print(f"   🎯 Scoped to module: {scope['target']}"
          + (f" (+deps {', '.join(sorted(scope['deps']))})" if scope["deps"] else "")
          + f" — hiding {len(scope['ignore'])} sibling module(s)")


def run_pipeline(feature: str, stack_name: str = "java-spring", output_dir: str = "./workspace") -> dict:
    from dotenv import load_dotenv
    load_dotenv()

    stack = load_stack(stack_name)
    os.environ["AGILE_OUTPUT_DIR"] = str(output_dir)
    _init_cline_ignore(output_dir)
    bb = Blackboard(output_dir)
    sm = ScrumMaster(bb)
    model = make_model(TASK_TOKEN_LIMITS["default"])

    print(f"\n{'='*55}")
    print(f"  Agile Pipeline")
    print(f"  Large Feature: {feature}")
    print(f"  Stack        : {stack['name']}")
    print(f"  Output       : {output_dir}")
    print(f"{'='*55}\n")

    # PHASE 1: Workspace Discovery (brownfield detection)
    print("📍 Analyzing workspace structure...")
    workspace_state = discover_workspace_state(output_dir)
    bb.write("workspace_mapping", workspace_state)
    _apply_module_scope_ignore(output_dir, workspace_state)
    _snapshot_pom_files(output_dir, bb)

    if workspace_state.get("is_brownfield"):
        print(f"✅ Detected existing project (brownfield mode)")
        print(f"   📦 Package root: {workspace_state.get('package_root', 'unknown')}")
        print(f"   📊 Existing files: {len(workspace_state.get('existing_files', []))}")
        mods = workspace_state.get("modules", [])
        if len(mods) > 1:
            print(f"   🧩 Modules: {len(mods)} — " +
                  ", ".join(f"{m['name']}({m['file_count']})" for m in mods[:6]) +
                  (" …" if len(mods) > 6 else ""))
        if workspace_state.get("existing_entities"):
            print(f"   🔒 Existing entities: {', '.join(workspace_state['existing_entities'].keys())}")
    else:
        print(f"🆕 Fresh workspace (greenfield mode)")
    print()

    # 1. Decompose the large feature request into smaller, incremental milestones.
    # Disabled by default — multiplies the whole pipeline by N milestones, blowing
    # token budgets. Opt in with AGILE_ENABLE_DECOMPOSE=1 for genuinely huge features.
    import os as _os_decomp
    enable_decompose = _os_decomp.getenv("AGILE_ENABLE_DECOMPOSE", "0") == "1"

    milestones = None
    if enable_decompose:
        print("🧩 Decomposing feature into incremental milestones...")
        pm_model = make_architect_model(TASK_TOKEN_LIMITS["integration"])
        pm_agent = make_agent(
            "Product Manager",
            "You decompose large requirements into clean, sequential, incremental feature milestones. Output JSON array only.",
            TASK_TOKEN_LIMITS["integration"],
            model=pm_model
        )
        decompose_prompt = _DECOMPOSE_PROMPT.format(feature=feature)
        msg = BaseMessage.make_user_message(role_name="User", content=decompose_prompt)
        try:
            response = step_with_retry(pm_agent, msg)
            raw = response.msgs[0].content if response.msgs else ""
            milestones = extract_json(raw)
        except Exception as e:
            print(f"⚠️ PM Decomposition failed: {e}. Falling back to single feature.")
            milestones = None

    # AGILE_MILESTONES_FILE: drive the milestone loop with an explicit JSON array of
    # feature strings (one per milestone) instead of LLM decomposition — used to build
    # several distinct features into one shared workspace sequentially.
    ms_file = _os_decomp.getenv("AGILE_MILESTONES_FILE")
    if ms_file:
        try:
            with open(ms_file) as _f:
                loaded = json.load(_f)
            if isinstance(loaded, list) and loaded:
                milestones = loaded
                print(f"🧩 Loaded {len(milestones)} explicit milestone(s) from {ms_file}")
        except Exception as e:
            print(f"⚠️ Could not load AGILE_MILESTONES_FILE: {e}")

    if not isinstance(milestones, list) or not milestones:
        milestones = [feature]

    print(f"\n🎯 Sequential Milestones Planned:")
    for idx, milestone in enumerate(milestones, 1):
        print(f"   Milestone {idx}: {milestone}")
    print(f"{'='*55}\n")

    # 2. Iterate through each milestone sequentially
    final_summary = {}
    completed_milestones = []
    # Resume: if a prior run checkpointed completed milestones into this same
    # workspace, skip them and build on what's already on disk. The blackboard
    # (contract, prior QA issues) was loaded from output_dir above, so the skipped
    # milestones' state is preserved. Only fully-completed milestones are skipped;
    # a partially-built milestone is rebuilt, reusing its on-disk files.
    resume_from = 0
    _ckpt_path = Path(output_dir) / "milestone_checkpoint.json"
    if _ckpt_path.exists():
        try:
            _ck = json.loads(_ckpt_path.read_text())
            _raw_completed = _ck.get("completed", [])
            # Identity check: a checkpoint only describes THIS run's progress if its
            # recorded milestone text matches the milestone at that index in the current
            # `milestones` list. Without this, a checkpoint file that bled in from a
            # DIFFERENT feature (e.g. via trunk-seeding rsync copying a prior feature's
            # scratch files into a new workspace) gets trusted blindly — the pipeline
            # then reports "already complete" and skips real implementation entirely,
            # silently re-accepting the OLD feature's code as if it were the new one.
            # See 2026-07-14 teastore-hist-dupes incident: admitted=False,
            # feature_exercised=False — milestone 1 was skipped because a leaked
            # milestone_checkpoint.json from teastore-hist-copurchase was on disk.
            completed_milestones = []
            _mismatch = False
            for _m in _raw_completed:
                _idx = _m.get("index")
                _expected = milestones[_idx - 1] if _idx and 1 <= _idx <= len(milestones) else None
                if _expected is not None and _m.get("milestone") == _expected:
                    completed_milestones.append(_m)
                else:
                    _mismatch = True
            if _mismatch:
                print(f"   ⚠️ checkpoint milestone text does not match this run's feature — "
                      f"discarding stale/foreign checkpoint entries, starting fresh")
            if completed_milestones:
                resume_from = max(m["index"] for m in completed_milestones)
                final_summary = completed_milestones[-1].get("summary", {})
                print(f"⏩ Resuming: milestones {[m['index'] for m in completed_milestones]} already complete — continuing from Milestone {resume_from + 1}\n")
        except Exception as _e:
            print(f"   (could not read checkpoint, starting fresh: {_e})")
    # Only milestones recorded as complete are skipped. A milestone that did NOT compile
    # is NOT recorded (see checkpoint write below), so membership — not max(index) —
    # decides skipping: a non-compiling milestone is re-attempted, even if a later one
    # somehow got recorded.
    completed_indices = {m["index"] for m in completed_milestones}
    for milestone_idx, milestone in enumerate(milestones, 1):
        if milestone_idx in completed_indices:
            continue
        print(f"\n🚀 Starting Milestone {milestone_idx}/{len(milestones)}: {milestone}")
        print(f"{'='*55}")

        # Initialize the sprint for this milestone. ScrumMaster scans existing files dynamically.
        sm.init_sprint(milestone, stack_name)

        # Seed the deterministic build scaffold before any dev sprint so the maven
        # gate always has a pom; recorded files make the incremental filter below
        # drop the (cline-flaky) pom/properties config tasks. No-op once they exist.
        # Skip if brownfield (existing build config).
        _seed_java_scaffold(bb, stack, workspace_state)

        # ── PRE-PLANNING: Architecture Analysis (Cline read-only) ──
        # Analyze existing codebase and generate ARCHITECTURE.md so architect has full context
        # of existing patterns, frameworks, constraints before planning new features. Run the
        # (flaky, nondeterministic) LLM analysis only once per workspace — once ARCHITECTURE.md
        # holds a valid analysis, later milestones reuse it as-is and get their own
        # incremental facts via _append_milestone_delivered + the conventions pinned block.
        arch_context = ""
        if workspace_state.get("existing_files"):
            _arch_file = os.path.join(str(output_dir), "ARCHITECTURE.md")
            try:
                _prior_arch = open(_arch_file).read()
            except OSError:
                _prior_arch = ""
            if _arch_valid(_prior_arch):
                arch_context = _prior_arch
            else:
                arch_context = _analyze_and_document_architecture(str(output_dir), milestone_idx)

        # Read the existing contract if it exists on the blackboard, to preserve it across milestones!
        existing_contract = bb.read("contract")

        # Drift-control mechanism E: fold the prior milestone's on-disk-verified
        # conventions (id types, packages, live routes) into the architect brief as
        # a short pinned-facts block. Deterministic, observed facts only — no
        # model guesses (supersedes the old _inferred_patterns propagation).
        _conv_path = Path(output_dir) / "conventions.json"
        if _conv_path.exists():
            try:
                conventions = json.loads(_conv_path.read_text())
            except Exception:
                conventions = {}
            conv_lines = []
            if conventions.get("id_types"):
                conv_lines.append("Existing entity id types (do not change): " +
                                   ", ".join(f"{n}.id={t}" for n, t in conventions["id_types"].items()))
            if conventions.get("packages"):
                conv_lines.append("Existing source packages: " + ", ".join(conventions["packages"]))
            if conventions.get("routes"):
                conv_lines.append("Already-implemented endpoints (do not duplicate): " +
                                   ", ".join(conventions["routes"]))
            if conventions.get("id_types"):
                _id_types = list(conventions["id_types"].values())
                _dominant = max(set(_id_types), key=_id_types.count)
                conv_lines.append(f"New entities should use the dominant existing id type "
                                  f"({_dominant}) unless the contract says otherwise.")
            if conv_lines:
                conv_block = ("\n\nVERIFIED CONVENTIONS FROM PRIOR MILESTONE (on-disk truth):\n"
                               + "\n".join(f"- {l}" for l in conv_lines) + "\n")
                arch_context = (arch_context or "") + conv_block

        # Incremental mode: merge the new feature into the existing workspace, reuse
        # already-built files, and feed the prior feature's unresolved QA issues forward.
        incremental = _os_decomp.getenv("AGILE_INCREMENTAL") == "1"
        prior_issues = bb.read("test_report").get("issues", []) if incremental else None

        # ── Ceremony: Sprint 1 Planning (Architecture scaffold) ──
        print(f"📋 [M{milestone_idx}] Sprint Planning — Sprint 1: Architecture (Architect, single call)...")
        planning_result = run_sprint_planning(milestone, stack, existing_contract=existing_contract,
                                               prior_issues=prior_issues, output_dir=output_dir,
                                               arch_context=arch_context)
        contract = planning_result.get("contract", {})
        # Re-namespace REST endpoints that collide with a legacy view @Controller (same
        # verb+path) under /api, so generated REST controllers don't clash with the view
        # controllers at boot (Spring ambiguous mapping). Done here — before the contract
        # is persisted and before the Java dev backlog is derived from it — so the dev
        # tasks, coverage gate, and skeleton emitter all see the conflict-free spec.
        from agile_team_cline.ceremonies.planning import reconcile_view_collisions
        contract = reconcile_view_collisions(contract, str(output_dir))
        backlog = planning_result.get("sprint_backlog", [])
        # Don't rebuild scaffold files (pom.xml, application.properties, …) that an
        # earlier milestone already produced — would clobber working config.
        if incremental:
            _existing_base = {_os_decomp.path.basename(f).lower()
                              for f in bb.read("blackboard").get("generated_files", [])}
            backlog = [t for t in backlog
                       if (t.get("target_file", "") or "").lower() not in _existing_base]
        bb.write("contract", contract)
        bb.write("sprint_backlog", backlog)
        sm.set_backlog(backlog)
        print(f"   Sprint 1 backlog: {len(backlog)} tasks")

        # Sprint structure: 1=Architecture, middle=Development, last=QA.
        # For Java (a file-fan-out stack) the development backlog is derived
        # deterministically from the contract — entity→repository→service per
        # entity, plus one controller per top-level endpoint resource — so no
        # layer can be dropped by the per-sprint task budget. The number of dev
        # sprints is sized to fit that backlog. Non-Java keeps LLM planning.
        is_java = stack.get("language", "").lower() == "java"
        dev_chunks = None
        if is_java:
            # Incremental: skip entity/repo/service/controller files already built by a
            # prior milestone, so only the new feature's files are generated.
            _cov_existing = bb.read("blackboard").get("generated_files", []) if incremental else None
            workspace_state = bb.read("workspace_mapping") or {}
            dev_backlog = build_coverage_tasks(contract, stack, existing_files=_cov_existing, workspace_state=workspace_state)
            if dev_backlog:
                dev_chunks = [dev_backlog[i:i + MAX_TASKS_PER_SPRINT]
                              for i in range(0, len(dev_backlog), MAX_TASKS_PER_SPRINT)]
        if dev_chunks:
            total_sprints = 2 + len(dev_chunks)  # arch + dev sprints + QA
            print(f"   Coverage-aware plan: {sum(len(c) for c in dev_chunks)} dev tasks "
                  f"across {len(dev_chunks)} sprint(s); total sprints = {total_sprints}")
        else:
            total_sprints = TOTAL_SPRINTS

        # ── Sprints Loop (1 to total_sprints) ──────────────────
        for sprint in range(1, total_sprints + 1):
            # Plan sprint 2+ on-the-fly using prior sprint outputs
            if sprint > 1:
                if sprint == total_sprints:
                    sprint_type = "QA"
                    if os.getenv("AGILE_AUTHOR_TESTS") == "1":
                        print(f"\n📋 [M{milestone_idx}] Sprint Planning — Sprint {sprint}: QA (Architect, context-aware)...")
                        workspace_state = bb.read("workspace_mapping") or {}
                        new_tasks = plan_next_sprint(milestone, sprint, sprint_type, bb, stack, workspace_state=workspace_state)
                    else:
                        # QA gate (_run_qa_sprint) runs unconditionally below regardless of the
                        # planned list — with test authoring off there's nothing for the QA
                        # sprint to plan, and the camel call only produced phantom task IDs
                        # (e.g. "verify X" labels, hallucinated "already exists" dev tasks)
                        # that never execute but still enter the backlog/PO-verdict surface.
                        new_tasks = []
                elif dev_chunks is not None:
                    # Deterministic, contract-derived development backlog (Java).
                    print(f"\n📋 [M{milestone_idx}] Sprint Planning — Sprint {sprint}: Development (contract-derived coverage)...")
                    new_tasks = dev_chunks[sprint - 2]
                    for t in new_tasks:
                        t["sprint"] = sprint
                else:
                    sprint_type = "Development"
                    print(f"\n📋 [M{milestone_idx}] Sprint Planning — Sprint {sprint}: Development (Architect, context-aware)...")
                    workspace_state = bb.read("workspace_mapping") or {}
                    new_tasks = plan_next_sprint(milestone, sprint, sprint_type, bb, stack, workspace_state=workspace_state)

                # Merge into full backlog and write to blackboard
                backlog = backlog + new_tasks
                bb.write("sprint_backlog", backlog)
                sm.set_backlog(backlog)
                print(f"   Sprint {sprint} backlog: {len(new_tasks)} tasks — {[t['title'] for t in new_tasks]}")

            if sprint == 1:
                print(f"\n🏗️  [M{milestone_idx}] Sprint 1: Architecture...")
            elif sprint == total_sprints:
                print(f"\n🧪 [M{milestone_idx}] Sprint {sprint}: QA...")
            else:
                print(f"💻 [M{milestone_idx}] Sprint {sprint}: Development...")

            sm.advance_sprint(sprint)
            sprint_tasks = sm.tasks_for_sprint(sprint)
            for task in sprint_tasks:
                sm.mark_task(task["id"], "in_progress")
            results = run_sprint(sprint, sprint_tasks, bb, stack, model, total_sprints=total_sprints)

            # CRITICAL: Detect if entity/repository tasks failed to be created. This is
            # a hard failure because downstream tasks (services, controllers) depend on
            # them existing. If cline returns success but didn't create critical files,
            # halt the pipeline rather than proceeding with broken references.
            bb_data = bb.read("blackboard") or {}
            failed_tasks = bb_data.get("failed_tasks", [])
            # Only abort when an entity/repo task's target FILE is genuinely absent
            # on disk. A task rejected for a compile error (file present) must flow to
            # the refinement loop — that's where cline gets space to fix it — instead
            # of killing the whole run.
            failed_critical = []
            for t in sprint_tasks:
                if t.get("type") in ("entity", "repository") and t["id"] in failed_tasks:
                    fname = t.get("target_file") or _title_to_filename(t["title"], stack["file_extension"])
                    if not bb.find_existing(fname):
                        failed_critical.append(t["id"])
            if failed_critical:
                failed_str = ", ".join(sorted(failed_critical))
                print(f"❌ HALTING PIPELINE: Critical entity/repository tasks failed: {failed_str}")
                print(f"   These tasks did not produce target files despite cline session completion.")
                print(f"   Downstream sprints depend on these files existing.")
                raise RuntimeError(f"Critical task completion failure: {failed_str}")

            for task in sprint_tasks:
                sm.mark_task(task["id"], "done")
            sm.standup(sprint)

        # ── Coverage Gate: ensure the contract's required layers were built ──
        from agile_team_cline.sprint import ensure_contract_coverage
        cov_tasks = ensure_contract_coverage(milestone, bb, stack, model, scrum_master=sm, total_sprints=total_sprints)
        if cov_tasks:
            backlog = backlog + cov_tasks

        # Collapse duplicate tasks (coverage-gate re-emits vs original backlog)
        # so the refinement loop's rejected set can converge to zero.
        backlog = _dedup_backlog(backlog)
        bb.write("sprint_backlog", backlog)
        sm.set_backlog(backlog)

        # ── Ceremony: Sprint Review ────────────────────────────
        print(f"✅ [M{milestone_idx}] Sprint Review (QA → PO verdict)...")
        test_report = bb.read("test_report")
        summary = run_sprint_review(test_report, _judgeable_backlog(backlog),
                                    evidence=_task_evidence(_judgeable_backlog(backlog), bb, stack))
        # Deterministic floor: drop PO rejections the VFs prove present+compiling on disk,
        # so the documented false-negatives never seed the refinement loop (runs #37-51).
        summary = _vf_floor(summary, backlog, bb, stack, test_report)
        # Deterministic ceiling: re-open PO accepts that hide a duplicate-route / shadow
        # resource so refinement consolidates them (run #51 dupes false-green).
        summary = _vf_ceiling(summary, backlog, bb, stack, test_report, final=False)
        bb.write("sprint_summary", summary)

        # ── Dependency Ordering for Refinement ─────────────
        def get_task_priority(task_id: str, backlog: list) -> int:
            """Determine task priority for refinement based on dependency type.

            Lower priority = fix first.
            Order: Entity → Repository → Service → Controller → Test
            """
            # Find the task in backlog to get its type
            task = next((t for t in backlog if t["id"] == task_id), None)
            if not task:
                return 999  # Unknown tasks go last

            task_type = task.get("type", "").lower()
            task_title = task.get("title", "").lower()

            # Determine priority based on type
            if task_type == "entity" or task_id.startswith("ENT-"):
                return 0  # Entities first (no dependencies)
            elif task_type == "repository" or task_id.startswith("REPO-"):
                return 1  # Repositories (depend on entities)
            elif task_type == "service" or task_id.startswith("SVC-"):
                return 2  # Services (depend on repositories)
            elif task_type == "controller" or task_id.startswith("CTL-"):
                return 3  # Controllers (depend on services)
            elif task_type == "config" or task_id.startswith("S1-T"):
                return -1  # Config/architecture first (no data dependencies)
            elif task_type == "test" or task_id.startswith(("TEST-", "S3-", "S4-")):
                return 4  # Tests last (depend on everything)
            else:
                return 2  # Default to middle priority

        def sort_tasks_by_dependency(rejected_ids: list, backlog: list) -> list:
            """Sort rejected task IDs by dependency order."""
            return sorted(rejected_ids, key=lambda tid: get_task_priority(tid, backlog))

        # ── Iterative Refinement & Bug-Fixing Loop ─────────────
        rejected = summary.get('rejected', [])
        # Sort rejected tasks by dependency order (config → entity → repo → service → controller → test)
        rejected = sort_tasks_by_dependency(rejected, backlog)

        refinement_round = 1
        max_refinement_rounds = int(os.getenv("AGILE_MAX_REFINEMENT_ROUNDS", "10"))
        # T1a: per-task rejection cap. `rejected` here is the PO's initial verdict,
        # so each task already has one rejection on the books.
        reject_counts = {tid: 1 for tid in rejected}
        disputed = []
        # (C) Cap the rejected tasks attempted per round so one bloated session can't
        # run away and hit the cline timeout; the overflow rolls to the next round.
        # 2 (optimized) = process 2 tasks per refinement session for Qwen (slower model).
        # Can override with AGILE_REFINE_BATCH env var.
        refine_batch = int(os.getenv("AGILE_REFINE_BATCH", "2"))
        from agile_team_cline.sprint import _run_qa_sprint

        # Last-green guard: once the build compiles, snapshot the source tree. If a
        # later refinement round breaks compilation (the LLM regenerating files for
        # logic nits re-introduces a compile error), revert to that snapshot and stop
        # — compilation is the primary success criterion, so never regress it.
        guard_on = stack.get("build_tool") == "Maven"
        green_snapshot = None
        if guard_on and isinstance(test_report, dict) and test_report.get("compiles"):
            green_snapshot = _snapshot_sources(bb)
            green_snapshot["summary"], green_snapshot["test_report"] = summary, test_report
            green_snapshot["tests_failed"] = test_report.get("tests_failed", 0)

        thinking_model = make_thinking_model(max(TASK_TOKEN_LIMITS.values()))
        while rejected and refinement_round <= max_refinement_rounds:
            print(f"\n🔄 [M{milestone_idx}] Iterative Refinement — Round {refinement_round} (Addressing PO feedback to fix rejected tasks)...")
            # (C) Attempt only a bounded batch this round; defer the overflow.
            batch_ids = rejected[:refine_batch] if refine_batch else list(rejected)
            deferred = [tid for tid in rejected if tid not in batch_ids]

            rejected_tasks = [t for t in backlog if t["id"] in batch_ids]
            if not rejected_tasks:
                print(f"   ⚠️  No backlog tasks matched rejected IDs {batch_ids} — stopping refinement.")
                break
            for task_id in batch_ids:
                sm.mark_task(task_id, "in_progress")
            current_sprint = total_sprints + refinement_round
            sm.advance_sprint(current_sprint)

            refinement_context = {
                "qa_issues": test_report.get("issues", []) if isinstance(test_report, dict) else [],
                "po_notes": summary.get("notes", ""),
                "vf_failures": bb.read("vf_failures") or {},  # VeriMAP: per-task VF feedback
            }
            print(f"🛠️  Fixing tasks: {[t['title'] for t in rejected_tasks]}")
            run_sprint(current_sprint, rejected_tasks, bb, stack, thinking_model, refinement_context=refinement_context, total_sprints=total_sprints)

            for task_id in batch_ids:
                sm.mark_task(task_id, "done")
            sm.standup(current_sprint)

            # Re-run QA review — pass the rejected tasks so QA sees the fixed
            # files at full content (not as historical summaries).
            print("🧪 Re-running QA review...")
            test_report = _run_qa_sprint(rejected_tasks, bb, stack, model)

            # (A) Re-verdict ONLY the batch under refinement, then carry prior accepts
            # forward — the PO no longer re-litigates already-accepted tasks against a
            # QA report that didn't re-examine them (the old whole-backlog verdict is
            # what let a single failing task keep the whole milestone in the loop).
            print("✅ Re-running Sprint Review...")
            round_summary = run_sprint_review(test_report, rejected_tasks,
                                              evidence=_task_evidence(rejected_tasks, bb, stack))
            still_rejected = [tid for tid in round_summary.get('rejected', []) if tid in batch_ids]
            # VeriMAP acceptance intersect: a task with deterministic VF failures cannot be
            # accepted regardless of PO verdict. Additive-only — never removes a rejection.
            if os.getenv("AGILE_VERIFY_AWARE", "0") not in ("0", "shadow"):
                _vf_fails = bb.read("vf_failures") or {}
                _vf_rejected = [tid for tid in batch_ids
                                if tid in _vf_fails and _vf_fails[tid]]
                still_rejected = list(dict.fromkeys(still_rejected + _vf_rejected))
            newly_accepted = [tid for tid in batch_ids if tid not in still_rejected]
            summary['accepted'] = list(dict.fromkeys(summary.get('accepted', []) + newly_accepted))

            # T1a: a second rejection of the same task never flipped the verdict in
            # glm-5.1 data — it's the same PO opinion re-applied. Move it to
            # "disputed" instead of looping again. Disputed != failed: the
            # milestone proceeds (the code may well be fine, as proven for
            # Invoice.total in the glm-5.1 run).
            still_disputing = []
            for tid in still_rejected:
                reject_counts[tid] = reject_counts.get(tid, 1) + 1
                if reject_counts[tid] >= 2:
                    disputed.append(tid)
                    print(f"   ⚖️  {tid} disputed (rejected {reject_counts[tid]}x) — proceeding without further refinement")
                else:
                    still_disputing.append(tid)

            # Rotate: try not-yet-attempted rejects before retrying ones that failed.
            rejected = deferred + still_disputing
            summary['rejected'] = rejected
            summary['disputed'] = list(dict.fromkeys(summary.get('disputed', []) + disputed))
            summary['notes'] = round_summary.get('notes', summary.get('notes', ''))
            bb.write("sprint_summary", summary)

            # Last-green guard: track the latest compiling tree but keep refining
            # ALL rounds (a later round may reach a better green). The fallback to
            # the latest green happens after the loop, only if it ends non-compiling
            # or with more failing tests than this snapshot. Never let a tree with
            # MORE failing tests than the current snapshot replace it.
            # VeriMAP: don't snapshot a tree that has deterministic VF failures — that would
            # let the guard restore a false-green tree. NL VFs never gate the snapshot.
            _vf_mode = os.getenv("AGILE_VERIFY_AWARE", "0")
            _vf_clean = (_vf_mode in ("0", "shadow")
                         or not (bb.read("vf_failures") or {}))
            if (guard_on and isinstance(test_report, dict) and test_report.get("compiles")
                    and _vf_clean
                    and (green_snapshot is None
                         or test_report.get("tests_failed", 0) <= green_snapshot.get("tests_failed", 0))):
                green_snapshot = _snapshot_sources(bb)
                green_snapshot["summary"], green_snapshot["test_report"] = summary, test_report
                green_snapshot["tests_failed"] = test_report.get("tests_failed", 0)

            refinement_round += 1
            # (B) Stop early when a round fixed nothing and there's nothing left to
            # try — kills the no-op rounds that burned the back half of the loop.
            # (With a batch cap, deferred work still pending counts as progress so
            # the queue keeps draining.)
            if not newly_accepted and not deferred and not disputed:
                print(f"   🛑 Refinement made no progress (no rejected task fixed) — stopping early after round {refinement_round - 1}.")
                break

        # ── Contract escalation: VF → architect second look ───────────────
        # A dispute rooted in a CONTRACT-fidelity VF failure (contract drift / orphan artifact)
        # is unresolvable by editing code — VF verifies code AGAINST the contract and this loop
        # only edits code. Give refinement the authority to escalate: hand the architect the exact
        # VF issue for a second look, and if it drops the phantom type (only-shrink), apply that
        # bounded correction + purge the orphans, then re-verify. Guarded/capped/senior-only/
        # fail-open — see ceremonies/contract_escalation.py.
        # Two post-loop recoveries for defects the code-only loop can't fix: (1) escalation
        # drops a phantom IN the contract; (2) orphan purge deletes cline-authored dead files
        # NOT in the contract (e.g. Spring ApplicationConfig/JpaConfig hallucinated into a
        # non-Spring reactor) that VF flags ORPHAN and the loop thrashed on. Both act only on
        # DISPUTED tasks (refinement already gave up) — see ceremonies/contract_escalation.py.
        try:
            from agile_team_cline.ceremonies.contract_escalation import (
                escalate_contract_defect, purge_unresolved_orphans)
            _disputed = list(dict.fromkeys(list(summary.get("disputed", [])) + list(rejected)))
            _esc = escalate_contract_defect(bb, stack, model, _disputed, milestone_idx=milestone_idx)
            _purge = purge_unresolved_orphans(bb, _disputed, milestone_idx=milestone_idx)
        except Exception as _e:
            _esc = _purge = None
            print(f"   ⚠️  post-loop recovery skipped: {_e}")
        if (_esc and _esc.get("revised")) or (_purge and _purge.get("purged")):
            if _esc and _esc.get("revised"):
                print(f"   🧭 [M{milestone_idx}] Contract escalation applied — dropped {_esc['dropped']}, "
                      f"purged {_esc['purged']} orphan artifact(s); re-verifying.")
            if _purge and _purge.get("purged"):
                print(f"   🧹 [M{milestone_idx}] Orphan purge removed {_purge['purged']} dead "
                      f"artifact(s): {_purge['purged_files']}; re-verifying.")
            _fix_tasks = [t for t in backlog if t["id"] in _disputed] or backlog
            test_report = _run_qa_sprint(_fix_tasks, bb, stack, model)
            summary = run_sprint_review(test_report, _fix_tasks,
                                        evidence=_task_evidence(_fix_tasks, bb, stack))
            rejected = summary.get("rejected", [])
            bb.write("test_report", test_report)
            bb.write("sprint_summary", summary)
            # Protect the purge from the last-green guard: a compiling re-verified tree with the
            # phantom/orphans removed becomes the new green baseline (never restore the pre-purge tree).
            if guard_on and isinstance(test_report, dict) and test_report.get("compiles"):
                green_snapshot = _snapshot_sources(bb)
                green_snapshot["summary"], green_snapshot["test_report"] = summary, test_report
                green_snapshot["tests_failed"] = test_report.get("tests_failed", 0)

        # Last-green guard (post-loop): if refinement ended on a non-compiling tree,
        # or with MORE failing tests than the latest green snapshot, restore that
        # snapshot — the milestone never regresses below a build that compiled with
        # fewer failing tests, even though every round was allowed to run.
        if guard_on and green_snapshot is not None and isinstance(test_report, dict):
            not_compiling = not test_report.get("compiles")
            tests_regressed = (not not_compiling
                                and test_report.get("tests_failed", 0) > green_snapshot.get("tests_failed", 0))
            if not_compiling or tests_regressed:
                if not_compiling:
                    print("↩️  Refinement ended on a non-compiling tree — restoring the latest green build.")
                else:
                    print(f"↩️  Refinement ended with more failing tests "
                          f"({test_report.get('tests_failed', 0)}) than the last green build "
                          f"({green_snapshot.get('tests_failed', 0)}) — restoring the latest green build.")
                _restore_sources(bb, green_snapshot)
                test_report = green_snapshot["test_report"]
                summary = green_snapshot["summary"]
                bb.write("test_report", test_report)
                bb.write("sprint_summary", summary)
                rejected = summary.get("rejected", [])

        # ── Tech Debt Sprint ──────────────────────────────
        if os.getenv("AGILE_TECH_DEBT", "0") == "1":
            from agile_team_cline.ceremonies.tech_debt import run_tech_debt_sprint
            print(f"\n🔧 [M{milestone_idx}] Tech Debt Sprint (Architect audit → Developer fixes)...")
            try:
                debt_result = run_tech_debt_sprint(bb, stack, model)
                if debt_result.get("debt_found"):
                    print(f"   Debt tasks : {debt_result.get('tasks', [])}")
                    print(f"   Fixed files: {debt_result.get('fixed', [])}")
                
                    # Run a QA check on the tech-debt changes to verify they didn't break compilation
                    print("🧪 Running post-tech-debt QA review to verify build integrity...")
                    from agile_team_cline.sprint import _run_qa_sprint
                    mock_qa_tasks = []
                    for idx, fname in enumerate(debt_result.get("fixed", [])):
                        mock_qa_tasks.append({
                            "id": f"TD-{idx+1}",
                            "title": f"Verify Tech-Debt Fix for {fname}",
                            "target_file": fname
                        })
                
                    test_report = _run_qa_sprint(mock_qa_tasks, bb, stack, model)
                
                    # If the compile/QA report has issues, re-enter the refinement loop!
                    if test_report.get("issues"):
                        print("⚠️ Tech-debt fixes introduced compilation or quality issues! Re-entering refinement loop...")
                        summary = run_sprint_review(test_report, mock_qa_tasks)
                        rejected = summary.get('rejected', [])
                    
                        refinement_round = 1
                        max_refinement_rounds = 3  # Allow up to 3 refinement rounds for tech-debt fixes
                    
                        while rejected and refinement_round <= max_refinement_rounds:
                            print(f"\n🔄 [M{milestone_idx}] Tech-Debt Refinement — Round {refinement_round} (Fixing broken tech-debt changes)...")
                            for task_id in rejected:
                                sm.mark_task(task_id, "in_progress")
                        
                            rejected_tasks = [t for t in mock_qa_tasks if t["id"] in rejected]
                            current_sprint = 20 + refinement_round  # distinct sprint range
                            sm.advance_sprint(current_sprint)
                        
                            refinement_context = {
                                "qa_issues": test_report.get("issues", []) if isinstance(test_report, dict) else [],
                                "po_notes": summary.get("notes", ""),
                            }
                            print(f"🛠️  Fixing tech-debt tasks: {[t['title'] for t in rejected_tasks]}")
                            run_sprint(current_sprint, rejected_tasks, bb, stack, thinking_model, refinement_context=refinement_context, total_sprints=total_sprints)

                            for task_id in rejected:
                                sm.mark_task(task_id, "done")
                            sm.standup(current_sprint)
                        
                            print("🧪 Re-running QA review...")
                            test_report = _run_qa_sprint(rejected_tasks, bb, stack, model)
                        
                            print("✅ Re-running Sprint Review...")
                            summary = run_sprint_review(test_report, mock_qa_tasks)
                            bb.write("sprint_summary", summary)
                        
                            rejected = summary.get('rejected', [])
                            refinement_round += 1
                    
                        # Update final PO summary matching the full backlog
                        summary = run_sprint_review(test_report, _judgeable_backlog(backlog),
                                                    evidence=_task_evidence(_judgeable_backlog(backlog), bb, stack))
                        bb.write("sprint_summary", summary)
                else:
                    print("   No tech debt identified.")
            except Exception as e:
                print(f"   Tech debt sprint failed: {e}")
        else:
            print(f"\n🔧 [M{milestone_idx}] Tech Debt Sprint skipped (set AGILE_TECH_DEBT=1 to enable)")

        # Closing deterministic floor: a task that survived refinement as 'disputed' but is
        # provably present+compiling on disk is accepted — the PO's blind-spot never reaches
        # the final dataset label (covers rejected AND disputed in the settled summary).
        final_summary = _vf_floor(summary, backlog, bb, stack, test_report)
        # Closing ceiling: if a duplicate-route / shadow resource still stands, downgrade the
        # implicated accepts to disputed — never ship a known collision as green.
        final_summary = _vf_ceiling(final_summary, backlog, bb, stack, test_report, final=True)
        # A milestone is "Complete" (checkpointed + skippable on resume) ONLY if it
        # compiles. Recording a non-compiling milestone false-greens the run: it prints
        # "Complete", returns rc=0, and on resume is skipped — so the build is never
        # finished. Gate on test_report["compiles"] (the same hard signal the last-green
        # guard uses); a red milestone is left out of the checkpoint and re-attempted.
        milestone_compiles = bool(isinstance(test_report, dict) and test_report.get("compiles"))
        # Deterministic controller skeletons compile (bodies throw UnsupportedOperationException)
        # but are NOT implemented — a milestone with stubs left is structurally present, not done.
        # Gate "complete" on compiles AND zero remaining stubs so a skeleton-only build can't
        # false-green (the same risk the compiles-gate itself guards against).
        from agile_team_cline.ceremonies.planning import count_skeleton_stubs
        stub_count = count_skeleton_stubs(str(output_dir))
        # Presentation-tier completeness (enhancement #2): a contract `view` with no servlet/JSP
        # on disk is a dropped webui layer that compiles green — gate "complete" on zero gaps so
        # it can't false-green. 0 for any contract without `views` (Spring/backend-only unchanged).
        from agile_team_cline.sprint import count_presentation_gaps
        pres_gaps = count_presentation_gaps(str(output_dir), contract, stack)
        # Red tests are a structural defect too — gate "complete" on zero failing
        # tests so a milestone that compiles but ships broken behavior isn't
        # checkpointed as done. AGILE_CHECKPOINT_TESTS=0 reverts to compile-only.
        tests_red = bool(isinstance(test_report, dict) and test_report.get("tests_failed", 0)
                          and os.getenv("AGILE_CHECKPOINT_TESTS", "1") == "1")
        if milestone_compiles and stub_count == 0 and not tests_red and pres_gaps == 0:
            # Checkpoint to disk so the record survives even if a LATER milestone dies
            # (e.g. retries exhausted on a gateway 504). Generated files already persist
            # in the single workspace; this preserves the run's progress/summary.
            completed_milestones.append({"index": milestone_idx, "milestone": milestone,
                                         "summary": summary, "compiles": True})
            try:
                with open(Path(output_dir) / "milestone_checkpoint.json", "w") as _ckpt:
                    json.dump({"total": len(milestones), "completed": completed_milestones}, _ckpt, indent=2)
            except Exception as _e:
                print(f"   (checkpoint write failed: {_e})")
            # Drift-control mechanism E: snapshot deterministic, on-disk-verified
            # conventions (id types, package layout, live route map) for the next
            # milestone — replaces the per-milestone _inferred_patterns guess.
            try:
                from agile_team_cline.ceremonies.planning import _extract_conventions
                _conventions = _extract_conventions(str(output_dir))
                with open(Path(output_dir) / "conventions.json", "w") as _conv:
                    json.dump(_conventions, _conv, indent=2)
                _append_milestone_delivered(str(output_dir), milestone_idx, _conventions)
            except Exception as _e:
                print(f"   (conventions write failed: {_e})")
            print(f"\n🏁 Milestone {milestone_idx} Complete!")
        elif milestone_compiles and stub_count:
            print(f"\n⚠️  Milestone {milestone_idx} compiles but has {stub_count} unimplemented "
                  f"skeleton stub(s) (UnsupportedOperationException) — NOT checkpointed; the method "
                  f"bodies must be filled. Re-attempted on the next run.")
        elif milestone_compiles and tests_red:
            print(f"\n⚠️  Milestone {milestone_idx} compiles but has "
                  f"{test_report.get('tests_failed', 0)} failing test(s) — NOT checkpointed; "
                  f"a resumed run will re-execute this milestone.")
        elif milestone_compiles and pres_gaps:
            print(f"\n⚠️  Milestone {milestone_idx} compiles but has {pres_gaps} missing "
                  f"presentation-tier file(s) (servlet/JSP for a contract view) — NOT checkpointed; "
                  f"the web UI layer must be built. Re-attempted on the next run.")
        else:
            print(f"\n⚠️  Milestone {milestone_idx} did NOT compile (compiles: false) — NOT "
                  f"checkpointed; it will be re-attempted on the next run instead of being "
                  f"skipped as complete.")
        print(f"{'='*55}\n")

    print(f"\n{'='*55}")
    print(f"  All Milestones Complete — Final Results")
    print(f"  Accepted : {final_summary.get('accepted', [])}")
    print(f"  Rejected : {final_summary.get('rejected', [])}")
    print(f"  Disputed : {final_summary.get('disputed', [])}")
    print(f"  Notes    : {final_summary.get('notes', '')}")
    print(f"{'='*55}\n")
    return final_summary


def main():
    parser = argparse.ArgumentParser(description="Agile multi-agent software pipeline")
    parser.add_argument("feature", help="Feature request (e.g. 'Build a Notes REST API')")
    parser.add_argument("--stack", default="java-spring", help="Stack preset (default: java-spring)")
    parser.add_argument("--output-dir", default="./workspace", help="Output directory (default: ./workspace)")
    args = parser.parse_args()
    try:
        run_pipeline(args.feature, args.stack, args.output_dir)
    except Exception as e:
        # A milestone died (e.g. gateway 504 after exhausting retries). Completed
        # milestones are already on disk and recorded in milestone_checkpoint.json;
        # report that instead of dumping a raw traceback, so prior work isn't "lost".
        ckpt = Path(args.output_dir) / "milestone_checkpoint.json"
        done = []
        if ckpt.exists():
            try:
                done = json.loads(ckpt.read_text()).get("completed", [])
            except Exception:
                pass
        print(f"\n❌ Run aborted: {type(e).__name__}: {e}")
        print(f"   Completed milestones preserved: {[m['index'] for m in done]} (see {ckpt})")
        print(f"   Re-run to resume from the last completed milestone.")
        sys.exit(1)


if __name__ == "__main__":
    main()
