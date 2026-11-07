#!/usr/bin/env python3
"""check_single_injection.py — verify a junior's single-anti-pattern claim against its diff.

`injected_inefficiencies.json` is the junior's OWN self-report (audit MEDIUM 11: "injection
labels are unverified self-reports ... no static check that e.g. n_plus_one_queries is actually
present in the diff"). That was tolerable when the brief asked for as many anti-patterns as
applied — several claims, several chances one landed. It is not tolerable under the single-pick
brief: with exactly one claimed defect, a claim that never landed produces a sample that looks
like a clean converged implementation and is silently indistinguishable from one.

This checks what can be checked STATICALLY and cheaply:
  * shape      — exactly one id in `injected`, the other seven in `skipped`, every catalog id
                 accounted for exactly once, no unknown ids;
  * locus      — the class named in `where` is a file that ACTUALLY DIFFERS between the senior
                 and junior workspaces. A claim pointing at a byte-identical file did not land.
  * evidence   — a per-pattern source heuristic (a nested loop for missing_preindex_rescan, a
                 repository call inside a loop for n_plus_one_queries, ...) on the changed file.

It reports; it does not gate. A heuristic miss is weak evidence, not proof of absence — the
`where`/diff check is the reliable signal and the heuristic is a hint for manual review. Nothing
here should silently reject a sample.

Usage:
    python3 scripts/check_single_injection.py teastore-hist-priceoutlier
    python3 scripts/check_single_injection.py --all
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS_ROOT = os.path.join(REPO, os.getenv("WORKSPACE_ROOT", "teastore-dataset"))

CATALOG = ["push_down_avoidance", "missing_preindex_rescan", "missing_memoization",
           "n_plus_one_queries", "no_early_termination", "wrong_data_structure",
           "memory_allocation_bloat", "redundant_mapping"]

# Source-level hints. Deliberately loose: they raise confidence when they fire and prove nothing
# when they do not, because any of these patterns can be written in a shape no regex anticipates.
_HINTS = {
    "missing_preindex_rescan": (r"for\s*\([^)]*\)[^{]*\{(?:[^{}]|\{[^{}]*\})*for\s*\(",
                                "a nested loop over collections"),
    # Must cover BOTH shapes: a repository round-trip AND a raw EntityManager lookup.
    # priceoutlier's junior (2026-07-28) wrote the textbook N+1 as `em.find(Category.class, id)`
    # inside the per-product loop; a REPOSITORY-only regex missed it and reported "no textual
    # sign", which reads as a failed injection when the injection was in fact perfect.
    "n_plus_one_queries":      (r"for\s*\((?:[^{}]|\{[^{}]*\})*?"
                                r"(?:(?:REPOSITORY|repository)\s*\.\s*get|\bem\s*\.\s*(?:find|createQuery)|"
                                r"getEntityManager\(\))",
                                "a per-entity lookup (repository call or em.find) inside a loop"),
    "wrong_data_structure":    (r"(?:List|ArrayList)<[^>]*>[^;]*;(?:[^;]*;){0,40}?\.contains\(",
                                "a List.contains() membership test"),
    "missing_memoization":     (r"for\s*\((?:[^{}]|\{[^{}]*\})*?\bnew\s+HashSet",
                                "a set rebuilt inside a loop"),
    "push_down_avoidance":     (r"getAll\w*\(\)|createQuery\([^)]*SELECT\s+\w+\s+FROM\s+\w+\s*\"",
                                "an unfiltered fetch processed in Java"),
    "no_early_termination":    (r"\.sort\(|Collections\.sort\(|\.sorted\(",
                                "a full sort where top-N would do"),
    "memory_allocation_bloat": (r"for\s*\((?:[^{}]|\{[^{}]*\})*?new\s+(?:ArrayList|HashMap|String)",
                                "fresh allocations inside a loop"),
    "redundant_mapping":       (r"for\s*\((?:[^{}]|\{[^{}]*\})*?new\s+\w*(?:DTO|Response|Dto)\b",
                                "DTO construction inside a loop"),
}


SINGLE_PICK_MARKER = "CHOOSE EXACTLY ONE"


def _regime(feature_id: str, dataset: str) -> str:
    """'single-pick' | 'menu' | 'no-brief' | 'unknown' — which brief this junior was given,
    read from `junior.induced_brief` on the row.

    Without this, `--all` reports FAIL for every one of the ten juniors already in the dataset,
    purely because they were built under the old menu brief that asks for as many anti-patterns
    as apply. That is a different regime, not a defect, and burying the real single-pick failures
    under ten expected ones makes the tool useless. The brief is recorded verbatim on the row as
    `junior.induced_brief`, so the regime is read from the run itself rather than assumed."""
    if not os.path.isfile(dataset):
        return "unknown"
    row = None
    for line in open(dataset, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("feature_id") == feature_id and not r.get("superseded"):
            row = r
    if row is None:
        return "unknown"          # no row at all (e.g. a run still in flight)
    brief = (row.get("junior") or {}).get("induced_brief")
    if not brief:
        # no brief: nothing self-reported, so the absence of injected_inefficiencies.json is
        # correct, not a defect. (regression_source == "scale" is a pre-2026-08-29 tag still
        # honoured on old/superseded rows.)
        return "no-brief" if row.get("regression_source") in ("induced", "scale", None) else "unknown"
    return "single-pick" if SINGLE_PICK_MARKER in brief else "menu"


def _java_files(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("target", ".git")]
        for f in fn:
            if f.endswith(".java") and not f.startswith("._"):
                p = os.path.join(dp, f)
                out[os.path.relpath(p, root)] = p
    return out


def check(feature_id: str, dataset: str = None) -> dict:
    jun = os.path.join(WS_ROOT, f"{feature_id}-junior")
    sen = os.path.join(WS_ROOT, f"{feature_id}-senior")
    dataset = dataset or os.path.join(REPO, "perf_dataset_teastore_history.jsonl")
    res = {"feature": feature_id, "ok": True, "notes": [],
           "regime": _regime(feature_id, dataset)}

    def bad(msg):
        res["ok"] = False
        res["notes"].append(("FAIL", msg))

    def good(msg):
        res["notes"].append(("ok", msg))

    if not os.path.isdir(jun):
        bad(f"no junior workspace at {jun}")
        return res

    path = os.path.join(jun, "injected_inefficiencies.json")
    if not os.path.isfile(path):
        if res["regime"] in ("no-brief", "scale"):
            good("no-brief run (no anti-pattern catalog) — nothing self-reported, "
                 "mechanism is diff-derived")
        else:
            bad("injected_inefficiencies.json missing — the junior did not self-report")
        return res
    try:
        rep = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        bad(f"injected_inefficiencies.json unparseable: {e}")
        return res

    inj = rep.get("injected") or []
    skp = rep.get("skipped") or []
    res["injected"] = [i.get("id") for i in inj if isinstance(i, dict)]

    # --- shape. The one-injection rule applies only to juniors actually given the single-pick
    # brief; a menu-mode junior listing several is behaving as instructed.
    if len(inj) != 1 and res["regime"] == "menu":
        res["notes"].append(("warn", f"menu-mode junior with {len(inj)} injected — the "
                                     f"single-pick rule does not apply; attribution to one "
                                     f"anti-pattern is not possible for this sample"))
    elif len(inj) != 1:
        bad(f"expected EXACTLY 1 injected anti-pattern, found {len(inj)}: {res['injected']}")
    else:
        good(f"exactly one injected: {res['injected'][0]}")
    ids = [i.get("id") for i in (inj + skp) if isinstance(i, dict)]
    unknown = [i for i in ids if i not in CATALOG]
    if unknown:
        bad(f"unknown catalog id(s): {unknown}")
    missing = [c for c in CATALOG if c not in ids]
    if missing:
        bad(f"catalog ids never accounted for: {missing}")
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        bad(f"id(s) listed twice: {sorted(dupes)}")
    if not (unknown or missing or dupes):
        good(f"all {len(CATALOG)} catalog ids accounted for exactly once")

    if not inj:
        return res

    # Every claim is audited, not just the single-pick case. Until 2026-08-15 this function
    # returned here unless exactly one injection was reported, so the locus and evidence checks
    # below never ran for 21 of the 25 features — only 4 juniors report a single injection, the
    # rest report 3-8 under the menu brief. `--all` was therefore verifying the bookkeeping
    # (catalog ids accounted for) and nothing about the code.
    if not os.path.isdir(sen):
        res["notes"].append(("warn", f"no senior workspace at {sen} — cannot diff"))
        return res
    sj, jj = _java_files(sen), _java_files(jun)
    changed = {r for r in (set(sj) & set(jj))
               if open(sj[r], 'rb').read() != open(jj[r], 'rb').read()}
    added = set(jj) - set(sj)
    touched = changed | added
    res["n_changed_files"] = len(touched)

    if len(inj) == 1:
        res["claim"], res["where"] = inj[0].get("id"), (inj[0].get("where") or "")
    res["claims"] = []
    accounted = set()

    for item in inj:
        if not isinstance(item, dict):
            bad(f"injected entry is not an object: {item!r}")
            continue
        claim, where = item.get("id"), (item.get("where") or "")
        entry = {"claim": claim, "where": where}
        cls = where.split(".")[0].strip() if where else ""
        hit = [r for r in touched if cls and os.path.basename(r) == f"{cls}.java"]

        # --- locus: did the claimed class actually change?
        if not where:
            bad(f"{claim}: `where` is empty — the claim cannot be located in the diff")
        elif not cls:
            bad(f"{claim}: `where` = {where!r} does not name a class")
        elif hit:
            good(f"{claim}: `where` points at {os.path.basename(hit[0])}, which differs "
                 f"from the senior")
            accounted.update(hit)
        else:
            bad(f"{claim}: `where` names {cls}.java but that file is identical to the "
                f"senior's (or absent) — the claimed injection did not land. "
                f"{len(touched)} file(s) did change: "
                f"{sorted(os.path.basename(t) for t in touched)[:5]}")
        entry["located"] = bool(hit)

        # --- evidence: pattern heuristic on the changed file
        if hit and claim in _HINTS:
            rx, desc = _HINTS[claim]
            src = open(jj[hit[0]], encoding="utf-8", errors="ignore").read()
            if re.search(rx, src, re.S):
                good(f"{claim}: source shows {desc}")
                entry["evidence"] = "found"
            else:
                res["notes"].append(
                    ("warn", f"{claim}: no textual sign of {desc} in {cls}.java — heuristic "
                             f"only, confirm by reading the diff"))
                entry["evidence"] = "not-found"
        else:
            entry["evidence"] = "no-heuristic"
        res["claims"].append(entry)

    # --- converse: which changed files does no claim account for?
    #
    # This is REPORTED, NOT FAILED, and the distinction is the whole point. Senior and junior
    # implement the feature INDEPENDENTLY from the same ticket, so the feature's own files
    # (its entity, its endpoint, the repository it extends) differ between the two tracks by
    # construction — that divergence is the experiment, not a defect. Measured across the
    # corpus: 23 of 25 features have at least one such file. Failing on it would fail almost
    # every honest sample, and a check that always fails gets switched off.
    #
    # The list is a human shortlist, and it must be read knowing this diff is the WRONG
    # REFERENCE for attribution. Measured 2026-08-16: `bundles` was long carried as the
    # "3 changes, 1 reported" two-variable example on the strength of a senior-vs-junior diff.
    # Against its actual baseline (tp-13 trunk snapshot) BOTH tracks are pure insertions —
    # senior +140/-0, junior +139/-0 — so nothing was modified and bundles is a clean
    # single-injection sample. The senior-vs-junior diff had manufactured the alarm.
    #
    # The correct reference for F8/F9 is the TRUNK SNAPSHOT
    # (teastore-dataset/teastore-trunk_history/tp-NN-*.tar.gz; all 25 exist, resolvable via
    # chain_manifest.json), which lets the feature's own additions be subtracted. Until this
    # function diffs against that, treat every entry below as "look here", never as evidence.
    unexplained = sorted(touched - accounted)
    res["unexplained_files"] = [os.path.basename(u) for u in unexplained]
    if not touched:
        bad("no Java file differs from the senior — nothing was injected")
    elif unexplained:
        res["notes"].append(
            ("warn", f"{len(unexplained)} changed file(s) no claim accounts for: "
                     f"{[os.path.basename(u) for u in unexplained][:5]} — expected for the "
                     f"feature's own files (both tracks implement it independently), but this "
                     f"is also where an unreported second variable would hide; review before "
                     f"citing this pair as single-variable"))
    else:
        good(f"every changed file ({len(touched)}) is accounted for by a claim")
    return res


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        feats = sorted({d[:-len("-junior")] for d in os.listdir(WS_ROOT)
                        if d.endswith("-junior")})
    elif args:
        feats = args
    else:
        print(__doc__.strip().split("Usage:")[-1])
        return 1

    worst = 0
    for f in feats:
        r = check(f)
        head = "PASS" if r["ok"] else "FAIL"
        inj = r.get("injected")
        extra = f"  [{r['regime']}]"
        if inj:
            extra += f"  injected={inj[0] if len(inj) == 1 else f'{len(inj)} patterns'}"
        print(f"\n[{head}] {f}{extra}")
        for lvl, msg in r["notes"]:
            print(f"   {'✓' if lvl == 'ok' else '!' if lvl == 'warn' else '✗'} {msg}")
        if not r["ok"]:
            worst = 1
    return worst


if __name__ == "__main__":
    sys.exit(main())
