#!/usr/bin/env python3
"""system_oracle_stats.py — the system-level counterpart to the Kieker local signal.

Parses a JMeter CSV results file (.jtl, header: timeStamp,elapsed,label,responseCode,...) into
per-request response times and computes the same paper-faithful verdict already used for the
local signal: Wilcoxon rank-sum + Cliff's Delta (`profiling/nonparametric_stats.py` — the ONE
shared statistical helper for both signals, per the system oracle protocol §3; no new
statistical code here, only a second caller).

Usage (library):
    from system_oracle_stats import parse_jtl, compute_verdict
    baseline = parse_jtl("senior/results.jtl")     # {label: [elapsed_ms, ...]}
    candidate = parse_jtl("junior/results.jtl")
    verdict = compute_verdict(baseline, candidate)  # includes overall + per-label breakdown

CLI:
    python3 system_oracle_stats.py <baseline.jtl> <candidate.jtl> [--out stats.json]
"""
import collections
import csv
import glob
import json
import os
import sys
from collections import defaultdict

try:
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta


def parse_jtl(path: str, discard_warmup_s: float = 0.0) -> dict[str, list[int]]:
    """Parse a JMeter CSV-format .jtl into {sampler_label: [elapsed_ms, ...]}, successful
    samples only (failed samples during a real oracle run mean the app errored under load —
    a functional problem, not a performance data point, and would poison the timing
    distribution with error-page response times).

    `discard_warmup_s` drops every sample whose timestamp falls within that many seconds of
    THIS FILE's first sample. The paper "exclude[s] the initial 5-minute warm-up period data and
    retain[s] the subsequent 10-minute data as steady-state performance data" (§IV-C). Our
    JMeter scenario maps its `warmup` property onto `ThreadGroup.ramp_time`, which RAMPS
    threads up but still records every sample, so without this the ramp — when the JVM is
    still JIT-compiling and caches are cold — is averaged into the steady-state result.

    ONE CUTOFF PER FILE, WHICH IS WHY CALLERS MUST NOT HAND IT A MERGED JTL. An earlier version
    of this docstring claimed the discard was "applied per input file before merging"; it was
    not. `run_oracle.sh` calls `_concat_jtl` to merge the per-repeat JTLs and passed the merged
    file here, so the single cutoff discarded repeat 1's ramp and RETAINED repeats 2 and 3's —
    and every repeat restarts the docker stack, so every repeat has its own cold ramp. Measured
    on reorder: 2,520 senior / 2,507 junior Feature samples of ramp survived, 16.4% of the pool.
    No verdict changed there (Feature delta 0.997 either way) but pooled MPD moved +15.7% ->
    +11.7%, so part of the pooled shift was ramp artifact; on a smaller-effect pair that
    asymmetry could flip a pooled verdict. Use `parse_jtl_repeats` for a multi-repeat run.
    """
    times: dict[str, list[int]] = defaultdict(list)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                rows.append((int(row["timeStamp"]), row))
            except (KeyError, ValueError):
                continue
    if not rows:
        return {}
    cutoff = min(ts for ts, _ in rows) + discard_warmup_s * 1000.0
    for ts, row in rows:
        if ts < cutoff:
            continue
        if row.get("success", "").lower() != "true":
            continue
        try:
            elapsed = int(row["elapsed"])
        except (KeyError, ValueError):
            continue
        times[row.get("label", "")].append(elapsed)
    return dict(times)


def jtl_error_stats(path: str, discard_warmup_s: float = 0.0) -> dict:
    """Failed-sample accounting over the same steady-state window `parse_jtl` measures.

    `parse_jtl` drops failed samples on purpose (an error page's response time is not a
    performance data point), which means a run where the app fell over under load and a run
    where it served every request cleanly produce timing distributions that look equally
    healthy. On the big-effect pairs that difference IS the result: a junior whose query is
    122x slower may not "run slower", it may time out, and the verdict would then be computed
    from whichever minority of requests survived. Counted here, per label and overall, so a
    degraded run is visible in the row instead of inferable only from the JMeter console.

    Follows the same per-repeat discovery as `parse_jtl_repeats`: each repeat's own ramp is
    excluded, so the rate describes steady state rather than cold start.
    """
    totals: dict[str, int] = defaultdict(int)
    failures: dict[str, int] = defaultdict(int)
    for p in repeat_jtls(path):
        rows = []
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        rows.append((int(row["timeStamp"]), row))
                    except (KeyError, ValueError):
                        continue
        except OSError:
            continue
        if not rows:
            continue
        cutoff = min(ts for ts, _ in rows) + discard_warmup_s * 1000.0
        for ts, row in rows:
            if ts < cutoff:
                continue
            label = row.get("label", "")
            totals[label] += 1
            if row.get("success", "").lower() != "true":
                failures[label] += 1
    total = sum(totals.values())
    failed = sum(failures.values())
    per_label = {}
    for label in sorted(totals):
        n, bad = totals[label], failures.get(label, 0)
        per_label[label] = {"samples": n, "failed": bad,
                            "error_rate_pct": round(bad / n * 100.0, 3) if n else None}
    return {"samples": total, "failed": failed,
            "error_rate_pct": round(failed / total * 100.0, 3) if total else None,
            "per_label": per_label}


def jtl_response_bytes(path: str, discard_warmup_s: float = 0.0) -> dict:
    """Per-label response-size accounting over the same steady-state window `parse_jtl` measures.

    Every gate before this one confirms the request was SENT and came back `success=true`; none
    of them confirm the response carried any actual data. A handler that silently returns an
    empty list on every call is indistinguishable, by timing or error rate alone, from one that
    works — this is exactly how six feature endpoints went unnoticed for weeks, each returning
    an identical byte count (150 or 179) across 5,000+ calls despite six different JSON schemas.
    That was found once, by hand, spot-checking response sizes. This makes it a standing,
    per-row field instead: `distinct_bytes` is how many different response sizes the label saw,
    and `constant_bytes` is true when that count is exactly one — every response, across the
    whole run, was byte-for-byte the same size. Real, varying data essentially never does that
    at any real sample count; a constant size at 30+ samples is the same signature the manual
    check found.
    """
    totals: dict[str, list[int]] = defaultdict(list)
    for p in repeat_jtls(path):
        rows = []
        try:
            with open(p, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    try:
                        rows.append((int(row["timeStamp"]), row))
                    except (KeyError, ValueError):
                        continue
        except OSError:
            continue
        if not rows:
            continue
        cutoff = min(ts for ts, _ in rows) + discard_warmup_s * 1000.0
        for ts, row in rows:
            if ts < cutoff:
                continue
            if row.get("success", "").lower() != "true":
                continue
            try:
                nbytes = int(row["bytes"])
            except (KeyError, ValueError):
                continue
            totals[row.get("label", "")].append(nbytes)
    per_label = {}
    for label, vals in sorted(totals.items()):
        distinct = sorted(set(vals))
        per_label[label] = {
            "n": len(vals),
            "distinct_bytes": len(distinct),
            "min_bytes": distinct[0],
            "max_bytes": distinct[-1],
            "constant_bytes": len(distinct) == 1,
        }
    return per_label


#: Below this many samples, a coincidentally-constant byte count is not yet evidence of a
#: degenerate response — matches the paper's own n=30 protocol used elsewhere in this pipeline.
_RESPONSE_NONEMPTY_MIN_N = 30

#: `constant_bytes` alone is NOT sufficient evidence of an empty response: the load generator
#: hits a fixed entity id with fixed query parameters against a database that is generated once
#: and never mutated during a run, so any deterministic GET legitimately returns the exact same
#: byte count on every call — real, large payloads included (measured on this corpus: 544B up
#: to 76,089B, all constant, all genuine data). The actual empty-response signature found by
#: hand was six different JSON schemas all serializing to the identical 179 bytes — an
#: empty-array wrapper is schema-independent, so its size floor sits far below any observed
#: real single-record payload (544B is the smallest genuine one on this corpus). 200 is chosen
#: to sit strictly between the two: below every real payload observed, above the empty ones.
_RESPONSE_NONEMPTY_MAX_DEGENERATE_BYTES = 200


def response_nonempty_gate(baseline_bytes: dict, candidate_bytes: dict) -> dict:
    """Did the Feature endpoint (and its redirect children) return a real payload?

    Checks every `_is_feature_label` label in each arm: `constant_bytes=True` at
    `_RESPONSE_NONEMPTY_MIN_N`+ samples AND a size at or below
    `_RESPONSE_NONEMPTY_MAX_DEGENERATE_BYTES` means the payload was the same small size on
    every single call — the empty/near-empty signature, not merely a deterministic one.
    `response_nonempty=False` (or `None` when the feature was never exercised) means the row's
    `feature_verdict` should be read as unconfirmed rather than a clean true negative, until the
    mechanism is diagnosed (see docs/PROBLEMS-FOUND-2026-08-23.md item O2).
    """
    exercised = (any(_is_feature_label(l) for l in baseline_bytes)
                and any(_is_feature_label(l) for l in candidate_bytes))
    if not exercised:
        return {"response_nonempty": None,
                "response_nonempty_detail": {
                    "reason": "no Feature label recorded in one or both arms"}}
    flagged = []
    for arm_name, bytes_by_label in (("baseline", baseline_bytes), ("candidate", candidate_bytes)):
        for label, info in bytes_by_label.items():
            if (_is_feature_label(label) and info.get("n", 0) >= _RESPONSE_NONEMPTY_MIN_N
                    and info.get("constant_bytes")
                    and info.get("min_bytes", 0) <= _RESPONSE_NONEMPTY_MAX_DEGENERATE_BYTES):
                flagged.append({"arm": arm_name, "label": label,
                                "bytes": info["min_bytes"], "n": info["n"]})
    return {"response_nonempty": len(flagged) == 0,
            "response_nonempty_detail": {"flagged": flagged} if flagged else {}}


def _one_verdict(baseline_ms: list[int], candidate_ms: list[int]) -> dict:
    if not baseline_ms or not candidate_ms:
        return {"system_verdict": "unlabelled", "reason": "insufficient samples",
                "baseline_n": len(baseline_ms), "candidate_n": len(candidate_ms)}
    p_value = wilcoxon_rank_sum(baseline_ms, candidate_ms)
    delta, magnitude = cliffs_delta(baseline_ms, candidate_ms)
    mean_b = sum(baseline_ms) / len(baseline_ms)
    mean_c = sum(candidate_ms) / len(candidate_ms)
    mpd = ((mean_c - mean_b) / mean_b * 100.0) if mean_b > 0 else None
    # Mirrors the paper's decision rule exactly (the system oracle protocol §3): significant
    # AND non-negligible effect AND the candidate is the slower side.
    is_regression = p_value <= 0.05 and magnitude != "negligible" and delta > 0
    return {
        "system_verdict": "regression" if is_regression else "none",
        "p_value": round(p_value, 5),
        "cliffs_delta": round(delta, 3),
        "effect_magnitude": magnitude,
        "mpd_pct": round(mpd, 2) if mpd is not None else None,
        "baseline_n": len(baseline_ms), "candidate_n": len(candidate_ms),
        "baseline_mean_ms": round(mean_b, 2), "candidate_mean_ms": round(mean_c, 2),
    }


#: Sampler label for the request that exercises the feature under test. The JMeter plan names
#: it "Feature" regardless of which endpoint it points at, so this key is stable across rows.
FEATURE_LABEL = "Feature"


def jtl_throughput(path: str, target_rps: float, discard_warmup_s: float = 0.0) -> dict:
    """Did the load generator actually deliver its target rate, per repeat?

    WHY THIS IS A RECORDED FIELD AND NOT A PRINTOUT. `ConstantThroughputTimer` can only PAUSE
    threads, never accelerate them, so a generator that cannot keep up produces a
    valid-looking JTL at the wrong load level — and load level is the independent variable of
    the whole experiment. Nothing downstream notices. run_oracle.sh has printed this check
    since the first session, but into a log nobody reads at analysis time; a row that cannot
    say whether its own workload was delivered cannot be audited.

    It also makes the single-machine isolation deficit MEASURABLE rather than merely disclosed.
    The load generator shares host cores with the application (macOS exposes no CPU affinity
    API, so it cannot be pinned away from it). Starvation would show up here first, and it is
    most likely on the saturating features, where the application pins CPUs for minutes.

    Counting follows run_oracle.sh: a TransactionController writes a synthetic PARENT row
    beside its children, and the target refers to parents only.
    """
    per_repeat = []
    for jtl in repeat_jtls(path):
        try:
            with open(jtl, newline="", encoding="utf-8", errors="replace") as fh:
                rows = [(r["label"], int(r["timeStamp"])) for r in csv.DictReader(fh)
                        if r.get("timeStamp", "").isdigit()]
        except (OSError, KeyError):
            continue
        if len(rows) < 2:
            continue
        labels = {l for l, _ in rows}
        children = {l for l in labels
                    if l.rsplit("-", 1)[0] in labels and l.rsplit("-", 1)[-1].isdigit()}
        cut = min(t for _, t in rows) + discard_warmup_s * 1000.0
        steady = [(l, t) for l, t in rows if t >= cut]
        if not steady:
            continue
        span = (max(t for _, t in steady) - cut) / 1000.0
        if span <= 0:
            continue
        counts = collections.Counter(l for l, _ in steady)
        got = sum(v for k, v in counts.items() if k not in children) / span
        per_repeat.append(round(got, 2))

    if not per_repeat or target_rps <= 0:
        return {"target_rps": target_rps, "achieved_rps_per_repeat": per_repeat,
                "worst_drift_pct": None, "starved": None,
                "note": "not computable — no usable timestamps"}
    worst = min(per_repeat)
    drift = (worst - target_rps) / target_rps * 100.0
    return {
        "target_rps": target_rps,
        "achieved_rps_per_repeat": per_repeat,
        "worst_drift_pct": round(drift, 2),
        # 10% matches the OFF-TARGET threshold run_oracle.sh has always printed against.
        "starved": drift < -10.0,
    }


def _is_feature_label(label: str) -> bool:
    """The Feature sampler, plus any redirect children JMeter names `Feature-0`, `Feature-1`...

    Matching on the bare name alone would leave the children pooled into the browse-only
    figure, which is exactly the traffic that block exists to remove.
    """
    return label == FEATURE_LABEL or label.startswith(FEATURE_LABEL + "-")


def repeat_jtls(path: str) -> list[str]:
    """The per-repeat JTLs behind a merged `oracle_results.jtl`, or [path] if there are none.

    `run_oracle.sh` writes `oracle_results_r1.jtl`, `_r2`, `_r3` next to the merged file, so the
    repeat boundaries the merge destroys are recoverable from the filesystem without changing
    the JTL format.
    """
    base, ext = os.path.splitext(path)
    reps = sorted(glob.glob(f"{base}_r*{ext}"))
    return reps or [path]


def parse_jtl_repeats(path: str, discard_warmup_s: float = 0.0
                      ) -> tuple[dict[str, list[int]], list[dict[str, list[int]]]]:
    """Parse a multi-repeat oracle run, discarding each repeat's OWN ramp.

    Returns (pooled, per_repeat). Each repeat restarts the docker stack and therefore has its
    own cold start; applying one cutoff to the merged file only ever discards the first repeat's
    ramp. Falls back to single-file behaviour when no per-repeat JTLs exist.
    """
    per_repeat = [parse_jtl(p, discard_warmup_s) for p in repeat_jtls(path)]
    pooled: dict[str, list[int]] = defaultdict(list)
    for rep in per_repeat:
        for label, vals in rep.items():
            pooled[label].extend(vals)
    return dict(pooled), per_repeat


def _per_repeat_deltas(baseline_reps: list, candidate_reps: list) -> list[dict]:
    """Mean/delta per repeat, pooled across labels — the blocking factor the design created.

    The oracle interleaves senior and junior per repeat precisely so that session-level warming
    is shared. Pooling every sample into one rank test then throws that pairing away: with
    ~150k samples the test reports p~0 for drift-level shifts. Measured on churn, whose pooled
    row reads as a uniform ~+7%: the per-repeat deltas are -9.4%, +8.4%, +5.3% over a monotone
    senior warming curve (8.43 -> 6.79 -> 5.97 ms). That spread is the honest picture, and with
    three repeats the median of per-repeat deltas is a more defensible headline than the pooled
    test. Reported, not substituted for the pooled verdict.
    """
    out = []
    for i, (b, c) in enumerate(zip(baseline_reps, candidate_reps), start=1):
        bv = [v for vals in b.values() for v in vals]
        cv = [v for vals in c.values() for v in vals]
        if not bv or not cv:
            continue
        b_mean = sum(bv) / len(bv)
        c_mean = sum(cv) / len(cv)
        entry = {"repeat": i, "baseline_mean_ms": b_mean, "candidate_mean_ms": c_mean,
                 "mpd_pct": (c_mean - b_mean) / b_mean * 100.0 if b_mean else None,
                 "baseline_n": len(bv), "candidate_n": len(cv)}
        fb, fc = b.get(FEATURE_LABEL, []), c.get(FEATURE_LABEL, [])
        if fb and fc:
            fb_mean = sum(fb) / len(fb)
            fc_mean = sum(fc) / len(fc)
            entry["feature_mpd_pct"] = ((fc_mean - fb_mean) / fb_mean * 100.0
                                        if fb_mean else None)
            entry["feature_n"] = {"baseline": len(fb), "candidate": len(fc)}
        out.append(entry)
    return out


def compute_verdict(baseline: dict[str, list[int]], candidate: dict[str, list[int]],
                    baseline_repeats: list = None, candidate_repeats: list = None) -> dict:
    """Overall (all samplers pooled) verdict — the paper's system-level signal is the whole
    workload's response-time distribution, not one endpoint — plus a per-label breakdown kept
    for auditability (which page/request drove the verdict), same 'raw over verdicts'
    discipline as the local Kieker signal.

    `feature_verdict` isolates the one sampler that actually reaches the changed code. It is
    reported ALONGSIDE the pooled verdict, never instead of it, because the two answer different
    questions and both belong in the row:

      * `system_verdict` (pooled) — "is the system slower for a user doing what users do?" This
        is the paper's question, and dilution is part of its answer: the feature endpoint is one
        request in seven, so a regression confined to it moves the pooled mean by roughly a
        seventh of its own size. A pooled "none" over a feature-level "regression" is a real
        finding, not a measurement failure.
      * `feature_verdict` — "is the changed code slower end-to-end?" This is what the local
        Kieker signal predicts, so it is the like-for-like comparison when scoring the paper's
        algorithm, whose input is component-level data about exactly this code.

    `feature_samples` records how many samples backed it: a feature verdict computed from a
    handful of requests deserves less weight than the pooled one, and hiding that behind a bare
    verdict string would be the same mistake as reporting a verdict for a workload that never
    reached the endpoint."""
    overall = _one_verdict(
        [v for vals in baseline.values() for v in vals],
        [v for vals in candidate.values() for v in vals],
    )
    per_label = {}
    for label in sorted(set(baseline) | set(candidate)):
        per_label[label] = _one_verdict(baseline.get(label, []), candidate.get(label, []))
    overall["per_label"] = per_label

    # BROWSE-ONLY POOLED — the like-for-like comparison with Liao et al.'s own oracle.
    #
    # Their TeaStore workload is browse+login only and contains NO feature endpoint, so their
    # pooled response-time distribution is 100% browse traffic. Ours is 1-in-7 `Feature` by
    # construction: the gated sampler exists precisely because the browse path never reaches
    # the changed code (see the oracle-workload finding). That makes our pooled number
    # structurally MORE feature-sensitive than theirs, so `system_verdict` and their published
    # pooled effect are not comparable as published.
    #
    # This block applies the identical pooling rule with the Feature sampler and its redirect
    # children removed. Report it beside `system_verdict` whenever the comparison is against
    # the paper; report `system_verdict` when the question is "what did OUR workload see".
    # Neither replaces `feature_verdict`, which answers the third, different question.
    browse = {k: v for k, v in baseline.items() if not _is_feature_label(k)}
    browse_c = {k: v for k, v in candidate.items() if not _is_feature_label(k)}
    if browse or browse_c:
        bo = _one_verdict([v for vals in browse.values() for v in vals],
                          [v for vals in browse_c.values() for v in vals])
        bo["labels_excluded"] = sorted(k for k in set(baseline) | set(candidate)
                                       if _is_feature_label(k))
        overall["browse_only"] = bo

    feature = per_label.get(FEATURE_LABEL)
    if feature is None:
        # No Feature sampler in this JTL: either a browse-only run, or a run predating the
        # workload fix. Say so explicitly — a missing key would read as "no regression".
        overall["feature_verdict"] = None
        overall["feature_exercised_by_workload"] = False
        overall["feature_samples"] = {"baseline_n": 0, "candidate_n": 0}
    else:
        overall["feature_verdict"] = feature["system_verdict"]
        overall["feature_exercised_by_workload"] = True
        overall["feature_mpd_pct"] = feature.get("mpd_pct")
        overall["feature_cliffs_delta"] = feature.get("cliffs_delta")
        overall["feature_p_value"] = feature.get("p_value")
        overall["feature_samples"] = {"baseline_n": feature.get("baseline_n", 0),
                                      "candidate_n": feature.get("candidate_n", 0)}

    # Per-repeat breakdown: see _per_repeat_deltas. Absent (not empty) when the caller did not
    # supply repeats, so a consumer can tell "not computed" from "computed, no spread".
    if baseline_repeats and candidate_repeats:
        deltas = _per_repeat_deltas(baseline_repeats, candidate_repeats)
        if deltas:
            overall["per_repeat"] = deltas
            spread = [d["mpd_pct"] for d in deltas if d.get("mpd_pct") is not None]
            if spread:
                overall["per_repeat_mpd_median_pct"] = sorted(spread)[len(spread) // 2]
                overall["per_repeat_mpd_range_pct"] = [min(spread), max(spread)]
    return overall


def write_stats(verdict: dict, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)


def main() -> None:
    args = sys.argv[1:]
    out_path = None
    discard = 0.0
    pooled_warmup = False
    target_rps = 0.0
    pos = []
    i = 0
    while i < len(args):
        if args[i] == "--out":
            out_path = args[i + 1]; i += 2
        elif args[i] == "--discard-warmup-s":
            discard = float(args[i + 1]); i += 2
        elif args[i] == "--pooled-warmup":
            pooled_warmup = True; i += 1
        elif args[i] == "--target-rps":
            target_rps = float(args[i + 1]); i += 2
        else:
            pos.append(args[i]); i += 1
    if len(pos) < 2:
        sys.exit("usage: system_oracle_stats.py <baseline.jtl> <candidate.jtl> [--out stats.json] "
                 "[--discard-warmup-s N] [--pooled-warmup] [--target-rps R]")
    # Per-repeat by default: each repeat restarts the stack and has its own ramp, so a single
    # cutoff over the merged JTL discards only the first one. --pooled-warmup restores the old
    # behaviour for comparison against previously published numbers.
    if pooled_warmup:
        baseline, candidate = parse_jtl(pos[0], discard), parse_jtl(pos[1], discard)
        b_reps = c_reps = None
    else:
        baseline, b_reps = parse_jtl_repeats(pos[0], discard)
        candidate, c_reps = parse_jtl_repeats(pos[1], discard)
    verdict = compute_verdict(baseline, candidate, b_reps, c_reps)
    verdict["warmup_discard"] = {
        "seconds": discard,
        "mode": "merged (single cutoff)" if pooled_warmup else "per-repeat",
        "repeats_found": len(b_reps) if b_reps else 1,
    }
    # Load-generator health, per arm. See jtl_throughput: an under-delivering generator yields
    # a clean-looking JTL at the wrong load level, and load level is the experiment's
    # independent variable. Recorded so the row can be audited without its run log.
    if target_rps > 0:
        verdict["load_generator"] = {
            "baseline": jtl_throughput(pos[0], target_rps, discard),
            "candidate": jtl_throughput(pos[1], target_rps, discard),
        }
        starved = [k for k, v in verdict["load_generator"].items() if v.get("starved")]
        if starved:
            verdict["load_generator"]["warning"] = (
                f"load generator under-delivered on: {', '.join(starved)} — the arms were not "
                f"compared at the same offered load, so this row's verdict is not trustworthy")
    # Failed samples never reach the verdict (parse_jtl drops them), so their COUNT is the only
    # record that the measured system was healthy while it was measured. See jtl_error_stats.
    # Always per-repeat here, whatever --pooled-warmup does to the verdict: that flag exists to
    # reproduce previously published numbers, and there is no reason to reproduce a ramp
    # accounting error in a health metric that no prior row carries.
    verdict["error_rates"] = {
        "baseline": jtl_error_stats(pos[0], discard),
        "candidate": jtl_error_stats(pos[1], discard),
    }
    # response_nonempty gate: a request that returns success=true with a fixed-size, likely-
    # empty payload passes every check above. See jtl_response_bytes/response_nonempty_gate.
    b_bytes = jtl_response_bytes(pos[0], discard)
    c_bytes = jtl_response_bytes(pos[1], discard)
    verdict["response_bytes"] = {"baseline": b_bytes, "candidate": c_bytes}
    verdict.update(response_nonempty_gate(b_bytes, c_bytes))
    if out_path:
        write_stats(verdict, out_path)
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_label"}, indent=1))
    print(f"\n{len(verdict['per_label'])} sampler label(s); "
          f"{sum(1 for v in verdict['per_label'].values() if v.get('system_verdict') == 'regression')} "
          f"flagged regression at the per-label level.")


if __name__ == "__main__":
    main()
