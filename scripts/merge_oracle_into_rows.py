#!/usr/bin/env python3
"""merge_oracle_into_rows.py — fold system-level oracle results into the dataset rows.

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
The oracle runs out-of-band: `scripts/run_oracle_session.sh` takes ~83 minutes per pair and
drops `oracle_stats.json` into the junior workspace hours or days after the row was written.
`dual_track._system_oracle_row` already knows how to read that file, but the only thing that
calls it is `remeasure_chain.py`, which re-profiles the whole pair. Re-measuring 14 pairs to
copy 14 JSON files into 14 rows would cost a day and would change the perf numbers underneath
the analysis. This script does the merge and nothing else.

Consequence of it never having run: every analysis of the oracle results so far has been
computed by reading the loose `oracle_stats.json` files directly, while the shipped dataset
still says `system_verdict: null` and still labels orphans, revenuepareto and bundles as true
negatives. Two artifacts, two answers.

WHAT IT WRITES
--------------
`perf_dataset_teastore_history.jsonl` is append-only with last-wins semantics (36 lines, 25
features), so an updated row is APPENDED rather than edited in place. The previous version
stays on the record, which is the property that lets a reviewer see that the labels changed
and when. A timestamped backup is taken first regardless.

Only the oracle-derived fields are recomputed (`system_verdict`, `oracle`). Everything else in
the row is copied forward untouched — this script must never become a second place where
measurement fields are derived.

USAGE
-----
    python3 scripts/merge_oracle_into_rows.py --dry-run     # show what would change
    python3 scripts/merge_oracle_into_rows.py
    python3 scripts/merge_oracle_into_rows.py --only teastore-hist-churn
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "agile_team_cline"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import dual_track as dt                                              # noqa: E402
try:
    import feature_cost_driver                                       # noqa: E402
except ImportError:
    feature_cost_driver = None

HISTORY = os.path.join(REPO, "perf_dataset_teastore_history.jsonl")
DATASET_DIR = os.path.join(REPO, "teastore-dataset")


def load_rows(path: str):
    """Every line, in order, plus the last-wins view keyed by feature_id."""
    lines = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                lines.append(json.loads(ln))
    latest = {}
    for r in lines:
        latest[r["feature_id"]] = r
    return lines, latest


def _summarise(block: dict) -> str:
    o = block.get("oracle") or {}
    if o.get("excluded"):
        return f"EXCLUDED ({o.get('excluded_class')})"
    if not o.get("available"):
        return "not measured"
    share = o.get("feature_share_of_pooled_pct")
    return (f"pooled={block.get('system_verdict')} feature={o.get('feature_verdict')} "
            f"featureMPD={o.get('feature_mpd_pct')}% "
            f"share={share}% err={o.get('error_rate_pct', {}).get('candidate')}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=HISTORY)
    ap.add_argument("--only", action="append", default=None,
                    help="feature_id to merge (repeatable); default every feature with a result")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lines, latest = load_rows(args.history)
    print(f"{len(lines)} lines, {len(latest)} features in {os.path.basename(args.history)}")

    updated = []
    unchanged = 0
    for fid in sorted(latest):
        if args.only and fid not in args.only:
            continue
        junior = os.path.join(DATASET_DIR, f"{fid}-junior")
        if not os.path.isdir(junior):
            continue
        block = dt._system_oracle_row(junior)
        # Attach the checked-in fixture/deployment cardinality table so a reader of
        # the row itself can see which cost dimension was measured at the wrong scale.
        if block["oracle"].get("available") and feature_cost_driver is not None:
            block["oracle"]["data_fidelity"] = feature_cost_driver.get(fid)
        old = latest[fid]
        # Nothing to merge, and nothing already merged: leave the row alone rather than
        # appending an identical copy. An append-only log is only readable if every entry
        # represents a real change.
        if not (block["oracle"].get("available") or block["oracle"].get("excluded")):
            continue
        same = (old.get("system_verdict") == block["system_verdict"]
                and old.get("oracle") == block["oracle"])
        if same:
            unchanged += 1
            continue
        row = dict(old)
        row.update(block)
        row["oracle_merged_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        updated.append((fid, old, row))

    if not updated:
        print(f"nothing to merge ({unchanged} already current)")
        return 0

    print(f"\n{len(updated)} row(s) to update, {unchanged} already current:\n")
    for fid, old, new in updated:
        print(f"  {fid.replace('teastore-hist-', ''):<16}")
        print(f"      was: {_summarise(old)}")
        print(f"      now: {_summarise(new)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.history}.bak-{stamp}"
    shutil.copy2(args.history, backup)
    with open(args.history, "a", encoding="utf-8") as fh:
        for _, _, row in updated:
            fh.write(json.dumps(row) + "\n")

    lines2, latest2 = load_rows(args.history)
    print(f"\nbacked up -> {os.path.basename(backup)}")
    print(f"appended {len(updated)} row(s); file now {len(lines2)} lines, "
          f"{len(latest2)} features")
    n_sys = sum(1 for r in latest2.values()
                if (r.get("oracle") or {}).get("available"))
    n_exc = sum(1 for r in latest2.values() if (r.get("oracle") or {}).get("excluded"))
    print(f"features with a system-level result: {n_sys}; excluded with reason: {n_exc}; "
          f"still pending: {len(latest2) - n_sys - n_exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
