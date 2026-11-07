#!/usr/bin/env python3
"""apply_setup_split.py — rebuild a profile with the fixture-loading phase removed.

WHY
---
A `PERF_only` profile is one JVM session that does two unrelated things in sequence: the
exerciser loads its fixtures (hundreds to thousands of `createEntity` / `updateEntity` /
`getAllEntities` calls) and only then invokes the feature query `repeats` times. Kieker
records both, so steps B->C->D see a component set in which the test scaffolding vastly
outnumbers the thing under test. Measured on the 2026-08-17 batch:

  * clv     13,977 setup records vs    30 measured ones (99.8% scaffolding)
  * rfm     13,517 setup records vs 10,110 measured ones
  * feature share of the step-C denominator: 0.0%-15% on every feature step D missed

That is not a modelling subtlety, it decides verdicts. Step B's rule is deliberately
two-sided (the paper's Table I carries negative MPD rows), so a `createEntity` that happened
to run 68% FASTER in the junior session enters `deviated` with a large negative MD and drags
the subsystem total down. rfm's feature method got 37% slower and its demand scale came out
at x0.725 — the fixture moved the number, not the code.

The paper profiles a running system under production load, where no fixture phase exists.
Removing it restores that setting rather than departing from it.

WHAT
----
Per profile directory: find the first invocation of the feature under test (`meta.json`'s
`only` field, which the harness already records), drop every record emitted before it, and
rewrite summary.csv / samples.csv / samples_tagged.csv / traces.json from what remains into
a sibling `<profile>_split/`. Nothing is modified in place — the unsplit profile stays
exactly as measured, so the two can be compared and the change can be undone by deletion.

The split is on `tin`, not on a signature allowlist, so a method called during BOTH phases
keeps only its measured-phase samples. That distinction is load-bearing, not theoretical: a
corpus-wide scan finds 11 straddling (signature, workspace) pairs across 9 workspaces, almost
all `getEntityClass` — JPA plumbing that the fixtures call AND the feature calls. They are
concentrated in the JUNIOR track, because the induced anti-pattern re-enters the repository
per row:

    personalbundles-junior  getEntityClass   1,754 before / 81,540 after
    turnover-junior         getEntityClass     472 /  26,580
    lift-junior             getEntityClass     840 /   9,360
    substitutes-senior      getEntityClass + getAllEntities

A signature allowlist would have to choose one bucket for the whole method: drop it and
personalbundles loses 81,540 real measured-phase calls; keep it and 1,754 fixture calls
contaminate the comparison. The timestamp rule splits the method instead, which is the only
answer that is right for both halves.

USAGE
-----
    python3 scripts/apply_setup_split.py                       # every workspace with a profile_only/
    python3 scripts/apply_setup_split.py --only lift --only clv
    python3 scripts/apply_setup_split.py --profile-dir profile_only --max-gb 8

Then point the algorithm at the result:

    python3 scripts/run_paper_algorithm.py --model paper_artifacts/teastore_model.json \\
        --profile-dir profile_only_split --out <fresh file>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiling"))
import parse_kieker as pk                                            # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO, "teastore-dataset")


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / (1024 ** 3)


def split_one(profile: str, out_dir: str) -> dict:
    """Rebuild `profile` into `out_dir` keeping only measured-phase records."""
    meta_path = os.path.join(profile, "meta.json")
    if not os.path.isfile(meta_path):
        return {"status": "skipped", "reason": "no meta.json"}
    with open(meta_path) as fh:
        meta = json.load(fh)
    target = (meta.get("only") or "").strip()
    if not target:
        return {"status": "skipped", "reason": "meta.json has no 'only' target"}

    records = pk.parse_dir_full(profile)
    if not records:
        return {"status": "skipped", "reason": "no records (profile produced no traces)"}

    kept, cutoff = pk.drop_setup(records, target)
    if cutoff is None:
        # The feature never ran. Splitting on a guessed boundary would silently fabricate a
        # measurement, so refuse and say so.
        return {"status": "failed", "reason": f"target {target!r} never invoked",
                "n_records": len(records)}

    os.makedirs(out_dir, exist_ok=True)
    times: dict[str, list[int]] = {}
    for r in kept:
        times.setdefault(r["signature"], []).append(r["exec_ns"])

    pk.write_summary(pk.summarise(times), os.path.join(out_dir, "summary.csv"))
    pk.write_raw(times, os.path.join(out_dir, "samples.csv"))
    pk.write_raw_tagged(pk.parse_run_tagged(profile, records=kept),
                        os.path.join(out_dir, "samples_tagged.csv"))
    pk.write_traces(pk.build_call_graph(profile, records=kept),
                    os.path.join(out_dir, "traces.json"))

    out_meta = dict(meta)
    out_meta.update({
        "setup_split": True,
        "setup_split_target": target,
        "setup_split_cutoff_tin": cutoff,
        "records_total": len(records),
        "records_kept": len(kept),
        "source_profile": os.path.basename(profile.rstrip("/")),
    })
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(out_meta, fh)

    return {"status": "ok", "target": target, "n_records": len(records),
            "n_kept": len(kept), "n_sigs": len(times),
            "pct_dropped": 100.0 * (len(records) - len(kept)) / len(records)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile-dir", default="profile_only",
                    help="profile subdirectory to split (default: profile_only)")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these features (repeatable, bare name e.g. 'lift')")
    ap.add_argument("--max-gb", type=float, default=4.0,
                    help="refuse to start a workspace when free disk would drop below this "
                         "(default 4). The split keeps the measured phase, which is tiny for "
                         "most features but ~95%% of the profile for dupes/crosssell.")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even when the _split directory already exists")
    args = ap.parse_args()

    workspaces = sorted(
        d for d in os.listdir(DATASET_DIR)
        if d.startswith("teastore-hist-")
        and os.path.isdir(os.path.join(DATASET_DIR, d, args.profile_dir))
    )
    if args.only:
        wanted = set(args.only)
        workspaces = [w for w in workspaces
                      if w.replace("teastore-hist-", "").rsplit("-", 1)[0] in wanted]
    if not workspaces:
        print("no matching workspaces", file=sys.stderr)
        return 2

    print(f"{len(workspaces)} workspace(s), profile dir {args.profile_dir!r}, "
          f"{free_gb(DATASET_DIR):.1f} GB free\n")
    ok = skipped = failed = 0
    for w in workspaces:
        profile = os.path.join(DATASET_DIR, w, args.profile_dir)
        out_dir = os.path.join(DATASET_DIR, w, args.profile_dir + "_split")
        label = w.replace("teastore-hist-", "")
        if os.path.isdir(out_dir) and not args.force:
            print(f"  {label:34s} exists (use --force to rebuild)")
            skipped += 1
            continue
        if free_gb(DATASET_DIR) < args.max_gb:
            print(f"  {label:34s} STOPPED — free disk {free_gb(DATASET_DIR):.1f} GB "
                  f"< --max-gb {args.max_gb}", file=sys.stderr)
            failed += 1
            break
        try:
            res = split_one(profile, out_dir)
        except Exception as exc:                                       # noqa: BLE001
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"  {label:34s} ERROR {exc}", file=sys.stderr)
            failed += 1
            continue
        if res["status"] == "ok":
            print(f"  {label:34s} {res['n_records']:>9,} -> {res['n_kept']:>7,} records "
                  f"({res['pct_dropped']:5.1f}% dropped, {res['n_sigs']:>3} sigs) "
                  f"target={res['target']}")
            ok += 1
        elif res["status"] == "skipped":
            print(f"  {label:34s} skipped — {res['reason']}")
            skipped += 1
        else:
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"  {label:34s} FAILED — {res['reason']}", file=sys.stderr)
            failed += 1

    print(f"\n{ok} split, {skipped} skipped, {failed} failed. "
          f"{free_gb(DATASET_DIR):.1f} GB free.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
