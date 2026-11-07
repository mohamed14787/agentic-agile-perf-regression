#!/usr/bin/env python3
"""dataset_view.py — canonical "last row wins" view + honest counts over a perf dataset.

WHY THIS EXISTS (2026-07-22): the perf_dataset_*.jsonl files are APPEND LOGS. Every
(re)run of dual_track.py appends a fresh row for the feature; the whole pipeline reads
them "last row wins" per feature_id (see scripts/lib_trunk_chain.sh). That preserves the
full provenance of a feature's attempts (e.g. churn landed on its 3rd/5th roll — valuable
research history we deliberately keep) BUT it means the raw file has NO uniqueness
invariant: a naive `grep '"admitted": true' | wc -l` over the raw log DOUBLE-COUNTS
superseded rows and, worse, counts a stale `admitted:true` that a later attempt overturned
(volatility flips true->false across its rows). On the history dataset that gap is a naive
count of 13 admitted vs the correct 8.

This tool is the ONE safe reader: it collapses the append log to one row per feature_id
(last wins), prints the correct summary, and — with --write — emits a canonical
`<dataset>.latest.jsonl` (exactly one row per feature) that downstream consumers / a
supervisor can read without needing to know the last-row-wins rule. It NEVER mutates the
append log (provenance stays intact).

Usage:
  python scripts/dataset_view.py [dataset.jsonl]        # summary to stdout
  python scripts/dataset_view.py [dataset.jsonl] --write # also write <dataset>.latest.jsonl
  python scripts/dataset_view.py [dataset.jsonl] --json  # machine-readable summary

Exit code is non-zero when the naive all-rows admitted count disagrees with the correct
last-row-wins count, so this can double as a CI/pre-commit guard against anyone wiring a
naive counter.
"""
import argparse
import json
import sys
from collections import OrderedDict


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def latest_by_feature(rows):
    """One row per feature_id, last occurrence wins — the canonical view the pipeline uses."""
    latest = OrderedDict()
    for r in rows:
        fid = r.get("feature_id")
        if fid is None:
            continue
        latest[fid] = r  # last write for this id wins
    return latest


def _seq(row):
    ch = row.get("chain") or {}
    s = ch.get("sequence")
    return s if isinstance(s, int) else 1 << 30


def summarize(path):
    rows = load_rows(path)
    latest = latest_by_feature(rows)
    canon = sorted(latest.values(), key=_seq)

    naive_admitted = sum(1 for r in rows if r.get("admitted") is True)
    correct_admitted = sum(1 for r in canon if r.get("admitted") is True)

    per_feature = []
    for r in canon:
        fid = r.get("feature_id")
        attempts = sum(1 for x in rows if x.get("feature_id") == fid)
        per_feature.append({
            "sequence": (r.get("chain") or {}).get("sequence"),
            "feature_id": fid,
            "admitted": r.get("admitted"),
            "feature_exercised": r.get("feature_exercised"),
            "regression_source": r.get("regression_source"),
            "attempts": attempts,
        })
    return {
        "dataset": path,
        "total_rows": len(rows),
        "distinct_features": len(latest),
        "admitted": correct_admitted,
        "naive_admitted_over_all_rows": naive_admitted,
        "count_discrepancy": naive_admitted - correct_admitted,
        "features": per_feature,
    }


def write_latest(path, latest):
    out = path + ".latest.jsonl"
    canon = sorted(latest.values(), key=_seq)
    with open(out, "w", encoding="utf-8") as fh:
        for r in canon:
            fh.write(json.dumps(r) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default="perf_dataset_teastore_history.jsonl")
    ap.add_argument("--write", action="store_true",
                    help="also write <dataset>.latest.jsonl (one canonical row per feature)")
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args()

    s = summarize(args.dataset)

    if args.json:
        print(json.dumps(s, indent=2))
    else:
        print(f"dataset: {s['dataset']}")
        print(f"  append-log rows : {s['total_rows']}")
        print(f"  distinct features: {s['distinct_features']}")
        print(f"  admitted (last-row-wins, CORRECT): {s['admitted']}")
        if s["count_discrepancy"]:
            print(f"  ⚠️  naive count over ALL rows      : {s['naive_admitted_over_all_rows']} "
                  f"(off by {s['count_discrepancy']:+d} — superseded rows; use this tool, not grep)")
        print()
        hdr = f"  {'seq':>3}  {'feature_id':<30} {'admitted':>8} {'exercised':>9} {'src':>8} {'rolls':>5}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for f in s["features"]:
            print(f"  {str(f['sequence'] if f['sequence'] is not None else '?'):>3}  "
                  f"{f['feature_id']:<30} {str(f['admitted']):>8} "
                  f"{str(f['feature_exercised']):>9} {str(f['regression_source']):>8} "
                  f"{f['attempts']:>5}")

    if args.write:
        out = write_latest(args.dataset, latest_by_feature(load_rows(args.dataset)))
        print(f"\n  ✅ canonical view written: {out} ({s['distinct_features']} rows)")

    # non-zero when a naive reader would miscount — lets this double as a guard.
    return 1 if s["count_discrepancy"] else 0


if __name__ == "__main__":
    sys.exit(main())
