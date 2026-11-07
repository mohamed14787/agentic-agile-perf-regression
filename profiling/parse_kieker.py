#!/usr/bin/env python3
"""
parse_kieker.py — parse Kieker OperationExecutionRecord .dat files from a profiling run.

Replicates the LOCAL performance signal from:
  Liao et al., "Early Detection of Performance Regressions by Bridging Local
  Performance Data and Architectural Models", ICSE 2025.

Each iteration dir contains one or more Kieker .dat files.  Each $1 record holds:
  $1;<writeTs>;<operationSignature>;<sessionId>;<traceId>;<tin>;<tout>;<hostname>;<eoi>;<ess>
Execution time = tout - tin (nanoseconds).

Output:
  <run_root>/summary.csv   — per-method stats across all iterations
  <run_root>/raw.csv       — every individual execution (optional with --raw)

Usage:
  python3 profiling/parse_kieker.py <run_root> [--raw]
"""
import csv
import json
import math
import re
import os
import sys
from collections import defaultdict


def _oer_record_ids(dirpath: str) -> tuple[str, ...]:
    """Resolve which $N record id(s) mean OperationExecutionRecord in this directory.
    Kieker assigns record-type ids per run via kieker.map ($1 is only conventional — if
    another record type registers first, hardcoding $1 silently parses nothing). Falls
    back to ("$1",) when no .map file is present (older runs, hand-made fixtures)."""
    ids = set()
    try:
        entries = os.listdir(dirpath)
    except OSError:
        return ("$1",)
    for fn in entries:
        if not fn.endswith(".map"):
            continue
        try:
            with open(os.path.join(dirpath, fn), encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("$") and "=" in line:
                        rid, cls = line.split("=", 1)
                        if cls.strip().endswith("OperationExecutionRecord"):
                            ids.add(rid.strip())
        except OSError:
            continue
    return tuple(sorted(ids)) or ("$1",)


def _read_dat(fpath: str, times: dict[str, list[int]],
              rec_ids: tuple[str, ...] = ("$1",)) -> None:
    """Accumulate exec times from one Kieker text .dat file into `times`."""
    prefixes = tuple(i + ";" for i in rec_ids)
    with open(fpath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(prefixes):
                continue
            parts = line.split(";")
            if len(parts) < 7:
                continue
            sig = parts[2]
            try:
                tin  = int(parts[5])
                tout = int(parts[6])
            except ValueError:
                continue
            elapsed = tout - tin
            if elapsed >= 0:
                times[sig].append(elapsed)


def _read_dat_full(fpath: str, records: list[dict],
                    rec_ids: tuple[str, ...] = ("$1",)) -> None:
    """Like `_read_dat` but keeps every field Upgrade A/B need: traceId (for grouping and
    the raw-sample export) and eoi/ess (call-tree depth/order, for `build_call_graph`).
    Kept as a separate reader (not folded into `_read_dat`) so the hot aggregate path used
    by `summarise`/`compare_kieker` stays untouched and provably byte-identical."""
    prefixes = tuple(i + ";" for i in rec_ids)
    with open(fpath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(prefixes):
                continue
            parts = line.split(";")
            if len(parts) < 10:
                continue
            sig = parts[2]
            trace_id = parts[4]
            try:
                tin  = int(parts[5])
                tout = int(parts[6])
                eoi  = int(parts[8])
                ess  = int(parts[9])
            except ValueError:
                continue
            elapsed = tout - tin
            if elapsed < 0:
                continue
            records.append({"signature": sig, "trace_id": trace_id,
                            "exec_ns": elapsed, "eoi": eoi, "ess": ess, "tin": tin})


def parse_dir(root: str) -> dict[str, list[int]]:
    """Parse every *.dat under `root` (ANY layout) into {signature: [exec_ns, ...]}.
    Used for a single profiled test run (e.g. dual-track <out>/profile/), which has no
    iter_NN/ structure."""
    times: dict[str, list[int]] = defaultdict(list)
    for dp, _dirs, files in os.walk(root):
        dat = [f for f in files if f.endswith(".dat")]
        if not dat:
            continue
        rec_ids = _oer_record_ids(dp)
        for fname in dat:
            _read_dat(os.path.join(dp, fname), times, rec_ids)
    return dict(times)


def parse_run(run_root: str) -> dict[str, list[int]]:
    """Return {operationSignature: [exec_time_ns, ...]} across all iter_NN/ iterations."""
    times: dict[str, list[int]] = defaultdict(list)
    iter_dirs = sorted(
        d for d in os.listdir(run_root)
        if d.startswith("iter_") and os.path.isdir(os.path.join(run_root, d))
    )
    if not iter_dirs:
        sys.exit(f"No iter_NN/ directories found in {run_root}")

    for iter_dir in iter_dirs:
        for sig, vals in parse_dir(os.path.join(run_root, iter_dir)).items():
            times[sig].extend(vals)
    return dict(times)


def _iteration_label(root: str, dirpath: str) -> str:
    """The iter_NN path component between `root` and `dirpath`, if any, else 'run' — mirrors
    `parse_dir`'s ANY-layout walk so the tagged reader accepts the same directory shapes
    (dual_track's flat `profile/iter_NN/kieker-.../` and the boost script's identical layout)."""
    rel = os.path.relpath(dirpath, root)
    for part in rel.split(os.sep):
        if part.startswith("iter_"):
            return part
    return "run"


def parse_dir_full(root: str) -> list[dict]:
    """Parse every *.dat under `root` (any layout) keeping full per-record detail
    (signature/trace_id/exec_ns/eoi/ess/tin/iteration) — the Upgrade A/B reader. Unlike
    `parse_dir`, nothing is aggregated here; callers group/tag/reconstruct as needed.

    The "iteration" label here is purely directory-derived (iter_NN/ path component, or
    "run" when absent) and is NOT rank-rebucketed — `build_call_graph` groups records by
    (iteration, trace_id) to reconstruct call trees, and a trace's records must stay
    together. Rank-based `rep_NN` sub-labeling for the in-session repeat protocol happens
    downstream in `parse_run_tagged`, per signature, where trace identity no longer
    matters."""
    records: list[dict] = []
    for dp, _dirs, files in os.walk(root):
        dat = [f for f in files if f.endswith(".dat")]
        if not dat:
            continue
        rec_ids = _oer_record_ids(dp)
        iteration = _iteration_label(root, dp)
        batch: list[dict] = []
        for fname in dat:
            _read_dat_full(os.path.join(dp, fname), batch, rec_ids)
        for rec in batch:
            rec["iteration"] = iteration
        records.extend(batch)
    return records


def _read_repeats(root: str) -> int | None:
    """Read the in-session repeat count the profiling harness recorded, if any.

    Written by kieker_profile.sh / dual_track.py as `<root>/meta.json` whenever the
    perf-exercise workload ran: `{"repeats": N, "protocol": "in-session"}`. N is the number
    of times each catalog-wide query was invoked inside the single profiled JVM — Liao et
    al.'s protocol is 30 repetitions inside one Kieker session, not 30 separate test runs
    (confirmed against their own replication package: every `kieker.dat` in
    `Component_perf_res/` carries one Kieker metadata record and per-operation call counts
    that are exact multiples of 30). Absent for older profiles or when the exerciser was
    off; callers fall back to the single "run" label used before this existed."""
    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        repeats = int(meta.get("repeats") or 0)
        return repeats if repeats > 1 else None
    except (OSError, ValueError, TypeError):
        return None


def method_name(signature: str) -> str:
    """Bare method name out of a fully-qualified Kieker operation signature."""
    m = re.search(r"\.(\w+)\(", signature or "")
    return m.group(1) if m else ""


def setup_cutoff(records: list[dict], target: str) -> int | None:
    """`tin` of the FIRST invocation of `target`, or None if it never ran.

    In a PERF_only profile the exerciser loads its fixtures (hundreds of `createEntity`
    calls) and only then invokes the feature query `repeats` times. Everything emitted
    before that first invocation is test scaffolding, not the system serving the feature's
    workload — the paper profiles a running system under load, where no such phase exists.
    """
    tins = [r["tin"] for r in records if method_name(r["signature"]) == target]
    return min(tins) if tins else None


def drop_setup(records: list[dict], target: str) -> tuple[list[dict], int | None]:
    """Records from the measured phase only: those at or after `target`'s first call.

    Returns (kept, cutoff). When `target` never ran, nothing is dropped and the cutoff is
    None — refusing to guess a boundary is the only safe answer, and the caller can see it
    happened rather than silently comparing a setup-only profile.
    """
    cut = setup_cutoff(records, target)
    if cut is None:
        return records, None
    return [r for r in records if r["tin"] >= cut], cut


def parse_run_tagged(run_root: str, records: list[dict] = None
                     ) -> dict[str, list[tuple[str, int]]]:
    """Return {signature: [(iteration_label, exec_time_ns), ...]} — the Upgrade A
    iteration-preserving counterpart to `parse_run`'s pooled `[exec_time_ns, ...]`. Works
    over any layout `parse_dir_full` accepts (iteration label falls back to 'run' when no
    iter_NN/ path component is present).

    When `run_root/meta.json` records an in-session repeat count (paper-aligned protocol:
    one Kieker session — kieker_profile.sh/dual_track.py both default to ONE iter_NN/
    directory now — with the catalog-wide queries invoked `repeats` times inside it) AND
    every record in this run in fact carries the SAME single iteration label (confirming
    there is really only one session to split, not several `iter_NN/` directories from the
    old multi-invocation protocol), each signature's records are re-labeled
    `rep_00`..`rep_{repeats-1}` by rank of execution order (`tin`). This turns the one flat
    block the paper's own data structure would otherwise degenerate to into `repeats`
    blocks for the bootstrap in paper_detect.py — matching their 30 in-session repetitions
    rather than this project's earlier 5 separate `mvn test` invocations. Confirmed against
    Liao et al.'s own replication package: this exact logic recovers 30 blocks of 5 calls
    each from their `Component_perf_res/L2_Original/kieker.dat`.

    Rebucketing happens HERE, per signature, not in `parse_dir_full` —
    `build_call_graph` groups records by (iteration, trace_id) to reconstruct call trees,
    and per-signature rank rebucketing would split a single trace's records across
    different labels depending on which signatures it happened to call, corrupting the
    call graph. Once collapsed into per-signature (iteration, exec_ns) pairs here, trace
    identity no longer matters, so it is safe. Multi-iteration profiles (several `iter_NN/`
    directories — the pre-alignment protocol) are left exactly as before: rebucketing only
    engages when there is genuinely one session to split."""
    tagged: dict[str, list[tuple[str, int]]] = defaultdict(list)
    by_sig_recs: dict[str, list[dict]] = defaultdict(list)
    all_labels: set[str] = set()
    for rec in (parse_dir_full(run_root) if records is None else records):
        by_sig_recs[rec["signature"]].append(rec)
        all_labels.add(rec["iteration"])

    repeats = _read_repeats(run_root)
    single_session = repeats and len(all_labels) == 1
    for sig, recs in by_sig_recs.items():
        if single_session:
            recs = sorted(recs, key=lambda r: r["tin"])
            n = len(recs)
            for rank, r in enumerate(recs):
                bucket = min(repeats - 1, (rank * repeats) // n)
                tagged[sig].append((f"rep_{bucket:02d}", r["exec_ns"]))
        else:
            for r in recs:
                tagged[sig].append((r["iteration"], r["exec_ns"]))
    return dict(tagged)


def percentile(sorted_vals: list[int], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def summarise(times: dict[str, list[int]]) -> list[dict]:
    rows = []
    for sig, vals in sorted(times.items()):
        vals_s = sorted(vals)
        n = len(vals)
        mean = sum(vals_s) / n
        variance = sum((v - mean) ** 2 for v in vals_s) / n if n > 1 else 0.0
        stddev = math.sqrt(variance)
        rows.append({
            "operation_signature": sig,
            "n_calls": n,
            "mean_ns": round(mean, 1),
            "stddev_ns": round(stddev, 1),
            "min_ns": vals_s[0],
            "p50_ns": round(percentile(vals_s, 50), 1),
            "p95_ns": round(percentile(vals_s, 95), 1),
            "p99_ns": round(percentile(vals_s, 99), 1),
            "max_ns": vals_s[-1],
        })
    rows.sort(key=lambda r: -r["mean_ns"])
    return rows


def write_raw(times: dict[str, list[int]], out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["operation_signature", "exec_time_ns"])
        for sig, vals in sorted(times.items()):
            for v in vals:
                w.writerow([sig, v])


def write_raw_tagged(tagged: dict[str, list[tuple[str, int]]], out_path: str) -> None:
    """Upgrade A: every raw execution with its iteration label kept (not pooled), so a
    later re-analysis (per-iteration blocking, warmup exclusion) doesn't need to re-run
    profiling — the iteration boundary that `write_raw`/`samples.csv` discards survives
    here as `<profile_dir>/samples_tagged.csv`."""
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["iteration", "operation_signature", "exec_time_ns"])
        for sig, pairs in sorted(tagged.items()):
            for iteration, v in pairs:
                w.writerow([iteration, sig, v])


def build_call_graph(root: str, records: list[dict] = None) -> dict:
    """Reconstruct the call tree from traceId/eoi/ess (Kieker's own convention) and
    aggregate edge call-counts across every trace in `root` (any layout `parse_dir_full`
    accepts). Edges are aggregated PER ITERATION then summed — a traceId is only unique
    within one JVM run, so summing raw trace ids across iterations would silently merge
    unrelated traces.

    Algorithm per trace: sort its records by eoi (execution order); walk keeping a
    depth-keyed stack of open calls. A record whose ess == len(stack) is a child of
    whatever is on top of the stack (or a root if the stack is empty); push it at
    position ess. Kieker emits one record per call (tin/tout both on the same line), so
    "still open" is tracked purely by ess-compare, not a literal push/pop event stream.

    Returns {"edges": [{"caller","callee","calls"}, ...], "entry_points": [sig, ...]}.
    """
    records = parse_dir_full(root) if records is None else records
    by_iter_trace: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for rec in records:
        by_iter_trace[rec["iteration"]][rec["trace_id"]].append(rec)

    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    entry_points: set[str] = set()

    for _iteration, traces in by_iter_trace.items():
        for _trace_id, trace_records in traces.items():
            trace_records.sort(key=lambda r: r["eoi"])
            stack: list[str] = []
            for rec in trace_records:
                sig, ess = rec["signature"], rec["ess"]
                # Defensive: a malformed/truncated trace could report an ess deeper than
                # the stack has tracked — clamp rather than index past the end.
                depth = min(ess, len(stack))
                caller = stack[depth - 1] if depth > 0 else None
                if caller is None:
                    entry_points.add(sig)
                else:
                    edge_counts[(caller, sig)] += 1
                stack = stack[:depth]
                stack.append(sig)

    edges = [{"caller": c, "callee": e, "calls": n}
             for (c, e), n in sorted(edge_counts.items(), key=lambda kv: -kv[1])]
    return {"edges": edges, "entry_points": sorted(entry_points)}


def write_traces(graph: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=1)


def write_summary(rows: list[dict], out_path: str) -> None:
    fields = ["operation_signature", "n_calls", "mean_ns", "stddev_ns",
              "min_ns", "p50_ns", "p95_ns", "p99_ns", "max_ns"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = sys.argv[1:]
    emit_raw = "--raw" in args
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        sys.exit("Usage: parse_kieker.py <run_root> [--raw]")
    run_root = positional[0]
    if not os.path.isdir(run_root):
        sys.exit(f"Not a directory: {run_root}")

    times = parse_run(run_root)
    if not times:
        sys.exit("No OperationExecutionRecord ($1) entries found.")

    rows = summarise(times)
    total_calls = sum(r["n_calls"] for r in rows)

    summary_path = os.path.join(run_root, "summary.csv")
    write_summary(rows, summary_path)
    print(f"  {len(rows)} operations, {total_calls} total calls → {summary_path}")

    if emit_raw:
        raw_path = os.path.join(run_root, "raw.csv")
        write_raw(times, raw_path)
        print(f"  raw records → {raw_path}")

    print("\nTop 10 by mean execution time:")
    print(f"  {'mean_ms':>10}  {'p99_ms':>10}  {'n':>6}  operation")
    print(f"  {'-'*10}  {'-'*10}  {'-'*6}  {'-'*60}")
    for r in rows[:10]:
        print(f"  {r['mean_ns']/1e6:>10.3f}  {r['p99_ns']/1e6:>10.3f}  "
              f"{r['n_calls']:>6}  {r['operation_signature']}")


if __name__ == "__main__":
    main()
