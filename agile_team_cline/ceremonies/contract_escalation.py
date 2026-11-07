"""VF → architect escalation (2026-07-12).

A dispute rooted in a CONTRACT-fidelity VF failure (CONTRACT DRIFT / ORPHAN ARTIFACT that names a
contract-declared type) is structurally unresolvable by the refinement loop: VF verifies CODE
against the contract, and the refinement loop only edits code — it can never satisfy a
self-contradictory contract (declare FirstProduct → must create it; nothing references it → must
delete it). The only stage empowered to fix a contract defect is the architect.

This module gives the refinement loop the authority to escalate: hand the architect the exact VF
issue for a SECOND LOOK, and if its revised contract DROPS the offending type — and ONLY shrinks the
contract (never grows the build post-loop) — apply that bounded correction and purge the orphan
artifacts the phantom spawned.

Guards (all must hold): env AGILE_CONTRACT_ESCALATION != "0" (default on); once per run (bb flag);
the architect must actually drop an implicated type (no-repeat); the revision must add no new types
(only-shrink); fail-open on any error. The dual-track JUNIOR runs a frozen seeded contract and MUST
NOT diverge, so dual_track disables this for the junior track.
"""
import os
import re

_FIDELITY_RE = re.compile(r"CONTRACT DRIFT|ORPHAN ARTIFACT")
# a dropped contract type drags its role artifacts + tests with it
_ROLE_SUFFIXES = ("", "Repository", "Endpoint", "Controller", "Service",
                  "RepositoryTest", "EndpointTest", "ControllerTest", "ServiceTest", "Test")


def _implicated_types(vf_failures: dict, disputed_ids, contract: dict) -> set:
    """Contract-declared type names named in an unresolved CONTRACT-fidelity VF failure among the
    disputed tasks — the defects code edits provably can't fix."""
    from agile_team_cline.verify_lib import _contract_type_names
    declared = _contract_type_names(contract or {})
    hits = set()
    for tid in disputed_ids or []:
        for msg in (vf_failures.get(tid) or []):
            if not isinstance(msg, str) or not _FIDELITY_RE.search(msg):
                continue
            for name in declared:
                if re.search(rf"\b{re.escape(name)}\b", msg):
                    hits.add(name)
    return hits


def _build_directive(vf_failures: dict, disputed_ids, implicated) -> str:
    """The 'issue reported' to the architect: the verbatim VF evidence plus an instruction to
    remove the phantom type(s) rather than re-declare them."""
    msgs = []
    for tid in disputed_ids or []:
        for m in (vf_failures.get(tid) or []):
            if isinstance(m, str) and _FIDELITY_RE.search(m):
                msgs.append(f"- {m}")
    uniq = list(dict.fromkeys(msgs))
    return (
        "CONTRACT-FIDELITY DEFECT — SECOND LOOK REQUIRED.\n"
        f"The type(s) {', '.join(sorted(implicated))} declared in your previous contract could not "
        "be realized by iterative code refinement — the verifier flagged them as unresolvable "
        "orphans / contract drift (declared, so the code must create them; referenced by nothing "
        "real, so they must be deleted — a contradiction no code edit can satisfy). This means the "
        "type does not belong in the contract: it is a phantom (typically a self-comparison feature "
        "modeled as a separate entity instead of reusing the EXISTING one). REMOVE these type(s) "
        "from the contract entirely, or replace each with the existing type it is aliasing. Do NOT "
        "re-declare them and do NOT introduce new entities. Verifier evidence:\n" + "\n".join(uniq)
    )


def _purge_orphan_artifacts(output_dir: str, dropped_types, baseline_rels) -> list:
    """Delete NEW (non-baseline) .java files whose class is a dropped type or a dropped type + a
    known role/test suffix. Never touches a baseline file."""
    if not output_dir or not os.path.isdir(output_dir):
        return []
    targets = {f"{t}{suf}.java" for t in dropped_types for suf in _ROLE_SUFFIXES}
    baseline_bases = {os.path.basename(r) for r in (baseline_rels or [])}
    purged = []
    for dp, dirs, files in os.walk(output_dir):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        for fn in files:
            if fn in targets and fn not in baseline_bases:
                try:
                    os.remove(os.path.join(dp, fn))
                    purged.append(fn)
                except Exception:
                    pass
    return purged


_ORPHAN_FILE_RE = re.compile(r"ORPHAN ARTIFACT:\s+([A-Za-z0-9_]+\.java)\b")


def purge_unresolved_orphans(bb, disputed_ids, milestone_idx: int = 1):
    """Delete cline-authored ORPHAN artifacts the refinement loop provably could NOT resolve.

    Distinct from escalate_contract_defect: that fixes phantoms IN the contract. This fixes
    NON-contract dead files cline hallucinated — e.g. Spring `ApplicationConfig.java` /
    `JpaConfig.java` emitted into a non-Spring Jakarta reactor. VF flags them ORPHAN ARTIFACT
    ("created this run but nothing references it"), but the loop tries to WIRE THEM IN rather
    than delete, thrashing to the round cap; worse, a config class importing unavailable Spring
    deps fails test-compile and sinks the whole module → non-equivalent (copurchase 2026-07-11).

    Safety limiters (deletion is only as safe as these): a file is purged ONLY if it is
      • named in an ORPHAN ARTIFACT VF message among the DISPUTED tasks (refinement already
        gave up on it after 2 rejects — the strongest available 'unresolvable' signal), AND
      • NOT a baseline file (created this run), AND
      • NOT a contract type's own artifact ({type}{role}.java — those belong to escalation).
    Env AGILE_ORPHAN_PURGE != "0" (default on); fail-open (never raises). Returns
    {purged: int, purged_files: [...]} or None."""
    if os.getenv("AGILE_ORPHAN_PURGE", "1") == "0":
        return None
    try:
        from agile_team_cline.verify_lib import _contract_type_names
        vf = bb.read("vf_failures") or {}
        contract = bb.read("contract") or {}
        output_dir = str(bb.output_dir)
        baseline_bases = {os.path.basename(r) for r in (bb.read("baseline_src") or {}).keys()}
        # contract types' own artifacts are escalation's job, not ours — never touch them here
        contract_artifacts = {f"{t}{suf}.java"
                              for t in _contract_type_names(contract) for suf in _ROLE_SUFFIXES}

        wanted = set()
        for tid in disputed_ids or []:
            for msg in (vf.get(tid) or []):
                if not isinstance(msg, str):
                    continue
                for m in _ORPHAN_FILE_RE.findall(msg):
                    if m not in baseline_bases and m not in contract_artifacts:
                        wanted.add(m)
        if not wanted:
            return None

        purged = []
        for dp, dirs, files in os.walk(output_dir):
            dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
            for fn in files:
                if fn in wanted:
                    try:
                        os.remove(os.path.join(dp, fn))
                        purged.append(fn)
                    except Exception:
                        pass
        if not purged:
            return None
        # drop the now-satisfied ORPHAN failures so they don't re-trigger downstream
        purged_stems = {p[:-5] for p in purged}  # strip ".java"
        cleaned = {tid: [msg for msg in msgs
                         if not any(s in (msg or "") for s in purged_stems)]
                   for tid, msgs in vf.items()}
        bb.write("vf_failures", {tid: m for tid, m in cleaned.items() if m})
        print(f"   🧹 [M{milestone_idx}] Orphan purge: deleted {len(purged)} unresolved "
              f"cline-authored orphan artifact(s): {sorted(set(purged))}")
        return {"purged": len(purged), "purged_files": sorted(set(purged))}
    except Exception as e:
        print(f"   ⚠️  orphan purge error (skipped): {e}")
        return None


def escalate_contract_defect(bb, stack, model, disputed_ids, milestone_idx: int = 1):
    """Fire the VF→architect escalation. Returns None (didn't fire / no change) or a dict
    {revised: bool, dropped: [...], purged: int, purged_files: [...]}. Fail-open: never raises."""
    if os.getenv("AGILE_CONTRACT_ESCALATION", "1") == "0":
        return None
    try:
        if bb.read("_contract_escalated"):
            return None
        vf = bb.read("vf_failures") or {}
        contract = bb.read("contract") or {}
        implicated = _implicated_types(vf, disputed_ids, contract)
        if not implicated:
            return None

        from agile_team_cline.config import TASK_TOKEN_LIMITS
        from agile_team_cline.ceremonies.planning import _architect_derive, normalize_contract
        from agile_team_cline.verify_lib import _contract_type_names

        output_dir = str(bb.output_dir)
        feature = bb.read("feature") or contract.get("feature") or ""
        directive = _build_directive(vf, disputed_ids, implicated)
        print(f"   🧭 [M{milestone_idx}] Contract escalation: architect second look — "
              f"unresolvable contract-fidelity defect on {sorted(implicated)}")

        tokens = TASK_TOKEN_LIMITS["architect_planning"]
        revised = _architect_derive(feature, stack, tokens, output_dir, prior_issues=[directive])
        bb.write("_contract_escalated", True)   # cap: once per run, even on a no-op result
        if not revised:
            return {"revised": False, "implicated": sorted(implicated)}
        revised = normalize_contract(revised, stack, output_dir=output_dir)

        orig_lower = {n.lower() for n in _contract_type_names(contract)}
        new_lower = {n.lower() for n in _contract_type_names(revised)}
        dropped_lower = orig_lower - new_lower
        added = new_lower - orig_lower
        impl_lower = {n.lower() for n in implicated}

        if not (impl_lower & dropped_lower):
            print("   ↩️  Escalation: architect re-emitted the same type(s) — no change applied.")
            return {"revised": False, "implicated": sorted(implicated)}
        if added:
            print(f"   ↩️  Escalation: revision would ADD {sorted(added)} — too risky post-loop; "
                  "skipping.")
            return {"revised": False, "implicated": sorted(implicated)}

        dropped = [n for n in _contract_type_names(contract) if n.lower() in dropped_lower]
        bb.write("contract", revised)
        baseline_rels = list((bb.read("baseline_src") or {}).keys())
        purged = _purge_orphan_artifacts(output_dir, dropped, baseline_rels)
        # drop stale VF failures naming the removed types so they don't re-trigger downstream
        cleaned = {tid: [m for m in msgs
                         if not any(re.search(rf"\b{re.escape(d)}\b", m) for d in dropped)]
                   for tid, msgs in vf.items()}
        bb.write("vf_failures", {tid: m for tid, m in cleaned.items() if m})
        print(f"   ✅ Escalation applied: dropped {dropped} from contract; purged {len(purged)} "
              f"orphan artifact(s): {purged}")
        return {"revised": True, "dropped": dropped, "purged": len(purged), "purged_files": purged}
    except Exception as e:
        print(f"   ⚠️  contract escalation error (skipped): {e}")
        return None
