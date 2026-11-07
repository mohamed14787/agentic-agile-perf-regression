#!/usr/bin/env python3
"""
compare_kieker.py — per-method local regression report between two Kieker runs.

The dual-track local signal (Liao et al., ICSE'25): given a BASELINE run (senior track)
and a CANDIDATE run (junior track) profiled with the same suite, match operations by
signature and flag those whose mean execution time regressed beyond a threshold.

A method is flagged REGRESSION when ALL hold (so noise is not mistaken for a regression):
  • candidate mean exceeds baseline mean by >= --threshold-pct (default 20%), AND
  • the absolute increase exceeds --min-delta-ms (default 0.5 ms) — a noise floor, AND
  • both sides have >= --min-calls samples (default 5) — a single-call mean is noise, AND
  • the noise guard clears. Two noise guards exist:
      - RAW SAMPLES available on both sides (preferred — the paper's own machinery):
        Wilcoxon rank-sum p <= 0.05 AND Cliff's Delta magnitude not "negligible".
        Rank-based, so robust to the outlier-heavy JVM distributions where the mean
        guard under-detects (PERF-307-dualval: +181.8% mean held "suspect" by 2σ
        because a few GC/JIT outliers pushed stddev above the mean).
      - summaries only (fallback, older data): Welch-style 2-sigma guard on means
        (delta >= 2·sqrt(sb²/nb + sc²/nc)).
A method that clears the pct+abs thresholds but fails the sample/noise guards is reported
as status "suspect" (never counted as REGRESSION) so borderline cases stay visible.

RUN-OFFSET NORMALIZATION (2026-07-26). The two runs are separate JVM invocations executed
back-to-back, so machine state (thermal/JIT/background load) shifts the WHOLE run by a
multiplicative factor that has nothing to do with code. Measured live on the chained-trunk
reorder pair: every method — including `ProductRepository.getEntityClass()`, a one-line
`return X.class` that is byte-identical in both tracks — came out 16–51% "faster" in the
candidate, median offset -30.5%. That offset SWAMPED the 20% threshold and inverted the
sign of a real +24% regression on the feature method, which the report then recorded as
`n_regressions: 0`. The rank guard made it worse, not better: a systematic offset is
statistically real, so Wilcoxon returned p=0.0 on `getEntityClass()`.

The fix: when the caller supplies `control_sigs` — signatures whose SOURCE is identical in
both tracks (dual_track derives them from `_feature_targets`, the same diff that drives the
feature-exercised gate) — estimate the run offset as the MEDIAN of candidate/baseline mean
ratios over those controls, and (when `apply_offset=True`) divide it out of the candidate
side (means AND raw samples, so the rank guard sees the corrected distribution too). Median
over many controls is robust to a few contaminated ones. Requires >= `min_controls` usable
controls; below that the offset is not estimated at all.
Both the raw and the normalized delta are reported, so the correction is always auditable.
With `control_sigs=None` (the default) behaviour is byte-identical to the pre-2026-07-26
module.

DEFAULT CHANGED TO apply_offset=False (2026-07-30). Liao et al.'s local measurement has no
such correction — it doesn't need one, because their 30 repetitions happen INSIDE one Kieker
session (confirmed against their own replication package), so there is no cross-invocation
machine-state offset to correct for in the first place. This project's profiling harnesses
now default to the same in-session protocol (kieker_profile.sh, dual_track.py — see
FeaturePerfExerciseTest), which removes the offset at its source instead of correcting for
it after the fact. The offset estimate itself is STILL computed and reported
(`run_offset_ratio`/`run_offset_pct`/`n_control_methods`/`offset_estimated`) whenever
`control_sigs` are supplied — it now serves as a contamination diagnostic: if a pair's
senior and junior really were captured as separate JVM sessions (e.g. `mvn_test_interleaved`
falling back to multiple iterations), a nonzero offset on unchanged control methods says so.
Pass `apply_offset=True` to restore the pre-2026-07-30 corrected behavior for a sensitivity
comparison.

Programmatic use (e.g. from dual_track.py):
  from compare_kieker import compare_runs, compare_summaries
  result = compare_runs(senior_run_dir, junior_run_dir)   # -> dict

CLI:
  python3 profiling/compare_kieker.py <baseline_run> <candidate_run> [--threshold-pct 20]
                                       [--min-delta-ms 0.5] [--out compare.csv]
"""
import csv
import os
import sys

# Reusable parsing from the sibling module (works both as script and import).
try:
    from parse_kieker import parse_run, summarise
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta
except ImportError:  # imported from another package (e.g. agile_team_cline)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_kieker import parse_run, summarise
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta


def _rows_by_sig(run_root: str) -> dict[str, dict]:
    """Parse a run root into {operation_signature: summary_row}."""
    return {r["operation_signature"]: r for r in summarise(parse_run(run_root))}


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# Methods that are source-identical across tracks — so formally valid controls — but whose
# timing behaviour is nothing like the analytic read queries under test. Excluded from the
# DRIFT ESTIMATE only; they are still compared and still reported as ordinary rows.
#
# Audit 2026-08-05 (finding K4). Census of teastore-hist-reorder-junior/profile/iter_01
# (3,203 records): createEntity on Product/OrderItem/Order accounted for 1,155 records (36%)
# and getEntityClass on Product/User/Category/Order for 1,040 (32%) — 68% of the profile, while
# the operation actually under test (getReplenishmentForecast, 40 calls) did not reach the top
# eight. Because the drift ratio is a MEDIAN over controls, these two families outvoted every
# read-path control and effectively defined the correction on their own.
#
# Why that is wrong: every chained-history feature is an analytic READ query, so createEntity
# appears only as test-fixture construction — a write-heavy, transactional, flush-bound path
# with entirely different cache, lock and I/O behaviour from the query being corrected.
# getEntityClass is a one-line accessor returning a constant, so its "timing" is dominated by
# probe overhead rather than by machine speed, which is precisely the quantity the ratio is
# supposed to estimate.
#
# Upstream TeaStore's own aop.xml excludes getEntityClass from instrumentation entirely, for all
# five repositories. Enforcing it there is not possible in this harness (see profiling/aop.xml
# for the NoClassDefFoundError that a concrete-aspect hits under Surefire's classloader), so the
# equivalent rule is applied here, at the point where it actually changes a number.
_CONTROL_EXCLUDED_METHODS = (
    "getEntityClass",   # constant-returning accessor; measures the probe, not the machine
    "createEntity",     # fixture construction: write path, absent from every feature's read path
    "updateEntity",
    "deleteEntity",
)


def _is_usable_control(sig: str) -> bool:
    """True when `sig` may contribute to the machine-drift estimate.

    Matches on the method name as it appears before the argument list, so it is unaffected by
    return type, modifiers or parameter spelling in the Kieker signature."""
    name = sig.split("(")[0].rsplit(".", 1)[-1]
    return name not in _CONTROL_EXCLUDED_METHODS


def estimate_run_offset(baseline: dict[str, dict], candidate: dict[str, dict],
                        control_sigs, min_calls: int = 5,
                        min_controls: int = 5) -> dict:
    """Multiplicative machine-speed offset between two runs, from CONTROL methods only.

    A control is a signature whose source is identical in both tracks, so any difference in
    its timing is machine state by construction. The offset is the median of
    candidate_mean/baseline_mean over the controls that carry >= min_calls samples on both
    sides. Returns {ratio, n_controls, applied, reason} — `applied` is False (ratio 1.0)
    whenever there is not enough evidence, so an unreliable correction is never applied
    silently.

    Controls whose method name is in `_CONTROL_EXCLUDED_METHODS` are dropped first: being
    source-identical makes a method a valid control only if its timing tracks machine speed,
    and fixture writes and constant-returning accessors do not (see that constant). The count
    of dropped controls is reported as `n_excluded` so the filter can never shrink the control
    pool silently."""
    if not control_sigs:
        return {"ratio": 1.0, "n_controls": 0, "n_excluded": 0, "applied": False,
                "reason": "no control signatures supplied"}
    usable = [s for s in control_sigs if _is_usable_control(s)]
    n_excluded = len(control_sigs) - len(usable)
    ratios = []
    for sig in usable:
        b, c = baseline.get(sig), candidate.get(sig)
        if not b or not c:
            continue
        if b.get("n_calls", 0) < min_calls or c.get("n_calls", 0) < min_calls:
            continue
        if not b.get("mean_ns"):
            continue
        ratios.append(c["mean_ns"] / b["mean_ns"])
    if len(ratios) < min_controls:
        return {"ratio": 1.0, "n_controls": len(ratios), "n_excluded": n_excluded,
                "applied": False,
                "reason": (f"only {len(ratios)} usable control(s) < min_controls={min_controls}"
                           + (f" ({n_excluded} excluded as fixture/accessor)" if n_excluded else ""))}
    return {"ratio": _median(ratios), "n_controls": len(ratios), "n_excluded": n_excluded,
            "applied": True,
            "reason": ("median candidate/baseline ratio over unchanged control methods"
                       + (f", excluding {n_excluded} fixture/accessor control(s)"
                          if n_excluded else ""))}


def compare_summaries(baseline: dict[str, dict], candidate: dict[str, dict],
                      threshold_pct: float = 20.0, min_delta_ms: float = 0.5,
                      min_calls: int = 5,
                      baseline_raw: dict[str, list] = None,
                      candidate_raw: dict[str, list] = None,
                      control_sigs=None, min_controls: int = 5,
                      apply_offset: bool = False, offset_override: dict = None,
                      exclude_controls: bool = True) -> dict:
    """Compare two {sig: summary_row} maps. Returns a structured report.

    When `baseline_raw`/`candidate_raw` ({sig: [exec_ns, ...]}) are supplied, the noise
    guard for a signature present in both is Wilcoxon rank-sum + Cliff's Delta (the
    paper's own tests, robust to JVM outliers); otherwise the Welch 2-sigma guard on
    summary means applies, byte-identical to the pre-raw behavior.

    `control_sigs` (signatures identical in both tracks) lets a run-offset be ESTIMATED —
    see the module docstring. It is only DIVIDED OUT of every verdict below when
    `apply_offset=True` (default False, paper-literal — see the 2026-07-30 addendum in the
    module docstring); otherwise the estimate is reported for diagnostic purposes only and
    every verdict uses the raw candidate distribution. `delta_pct_raw` always preserves the
    uncorrected number for audit, whichever mode is active."""
    import math
    min_delta_ns = min_delta_ms * 1e6
    baseline_raw = baseline_raw or {}
    candidate_raw = candidate_raw or {}
    offset = estimate_run_offset(baseline, candidate, control_sigs, min_calls, min_controls)
    # An ISOLATED profile (PERF_only) contains the feature's own method plus setup and little
    # else — 7 unique operations on lift, against 20 for the sweep — so only 2 controls clear
    # min_calls and `estimate_run_offset` gives up (applied=False). Note what is lost: with
    # apply_offset=False (the default since 2026-07-30, paper-literal — Liao et al. do not
    # normalize) no correction was going to be applied anyway. What a starved control pool
    # destroys is the CONTAMINATION DIAGNOSTIC — `offset_estimated` can no longer tell you
    # whether this session drifted at all, on exactly the runs whose verdicts matter most.
    #
    # `offset_override` restores it from a SWEEP profile of the same pair measured in the same
    # session: same machine state, same drift, but ~22 usable controls instead of 2. It is a
    # diagnostic input only — it does not change `apply_offset`, so the paper-literal path
    # stays uncorrected either way.
    if offset_override and not offset["applied"]:
        offset = dict(offset_override)
        offset["reason"] = (f"estimated from the sweep profile of the same session "
                            f"({offset.get('n_controls')} controls); the isolated profile had "
                            f"too few usable controls")
    corrected = offset["applied"] and apply_offset
    k = offset["ratio"] if corrected else 1.0
    rows = []
    for sig in sorted(set(baseline) | set(candidate)):
        b = baseline.get(sig)
        c = candidate.get(sig)
        b_mean = b["mean_ns"] if b else None
        c_mean = c["mean_ns"] if c else None
        # drift-corrected candidate: the decision variable for every guard below
        c_mean_norm = (c_mean / k) if c_mean is not None else None
        p_value = cliff = cliff_label = None
        delta_pct_raw = None
        if b is None:
            status, delta_pct = "NEW", None
        elif c is None:
            status, delta_pct = "REMOVED", None
        else:
            delta_ns = c_mean_norm - b_mean
            delta_pct = (delta_ns / b_mean * 100.0) if b_mean > 0 else None
            delta_pct_raw = ((c_mean - b_mean) / b_mean * 100.0) if b_mean > 0 else None
            over_thresholds = (delta_pct is not None and delta_pct >= threshold_pct
                               and delta_ns >= min_delta_ns)
            enough_samples = (b["n_calls"] >= min_calls and c["n_calls"] >= min_calls)
            raw_b, raw_c = baseline_raw.get(sig), candidate_raw.get(sig)
            # scale the candidate's per-call samples by the same factor, so the rank guard
            # judges the corrected distribution instead of re-confirming the machine offset
            if raw_c and k != 1.0:
                raw_c = [v / k for v in raw_c]
            if raw_b and raw_c:
                # Rank-based guard (preferred): significant AND non-negligible effect,
                # with the direction requirement that candidate tends slower (delta>0).
                p_value = wilcoxon_rank_sum(raw_b, raw_c)
                cliff, cliff_label = cliffs_delta(raw_b, raw_c)
                clears_noise = (p_value <= 0.05 and cliff_label != "negligible"
                                and cliff > 0)
            else:
                # Welch-style 2-sigma guard: with per-call stddev available on both
                # sides, the mean delta must clear twice the combined standard error.
                # Graceful when stddev is absent (older summaries): guard doesn't apply.
                sb, sc = (b.get("stddev_ns"), c.get("stddev_ns"))
                if (sb is not None and sc is not None
                        and b["n_calls"] > 1 and c["n_calls"] > 1):
                    sc = sc / k          # same scaling as the candidate mean
                    sem = math.sqrt(sb ** 2 / b["n_calls"] + sc ** 2 / c["n_calls"])
                    clears_noise = delta_ns >= 2.0 * sem
                else:
                    clears_noise = True
            if over_thresholds and enough_samples and clears_noise:
                status = "REGRESSION"
            elif over_thresholds:
                status = "suspect"   # big delta, but too few samples / inside noise band
            else:
                status = "ok"
        rows.append({
            "operation_signature": sig,
            "baseline_mean_ns": b_mean,
            "candidate_mean_ns": c_mean,
            "candidate_mean_ns_norm": c_mean_norm,
            "delta_ns": (c_mean_norm - b_mean) if (b and c) else None,
            "delta_pct": round(delta_pct, 1) if delta_pct is not None else None,
            "delta_pct_raw": round(delta_pct_raw, 1) if delta_pct_raw is not None else None,
            "is_control": bool(control_sigs and sig in control_sigs),
            "baseline_n": b["n_calls"] if b else 0,
            "candidate_n": c["n_calls"] if c else 0,
            "p_value": round(p_value, 5) if p_value is not None else None,
            "cliffs_delta": round(cliff, 3) if cliff is not None else None,
            "effect": cliff_label,
            "status": status,
        })
    # worst regressions first, then by absolute slowdown
    rows.sort(key=lambda r: (r["status"] != "REGRESSION",
                             -(r["delta_pct"] or -1e18)))
    # A CONTROL method is source-identical in both tracks, so a code-caused regression on one is
    # impossible by construction: any delta it shows is machine drift, which is exactly why it
    # was chosen as a control in the first place. Counting it as a regression is a category
    # error, and a live one — the corpus's personalbundles row reports its only REGRESSION on
    # `getCoPurchaseReach`, a control (is_control=True, +23.1%, p=0.039), while the actual
    # feature method sits at p=0.60. Under the isolated workload it is worse: only 2 controls
    # clear min_calls, so the offset estimator never engages (see estimate_run_offset's
    # min_controls) and the raw drift lands on all of them at once — measured on lift, 3 of 4
    # "regressions" were controls.
    #
    # The rows keep their status and their is_control flag: a control that trips the threshold
    # is the single best diagnostic that this run's drift was not removed, and hiding it would
    # cost more than it saves. Only the HEADLINE COUNTS exclude them, so `n_regressions` means
    # "methods whose code could explain the slowdown".
    #
    # `exclude_controls=False` for the TRUNK comparison (parent trunk vs current trunk), where
    # the same source-identity test means something else entirely. There, "control" == a method
    # that existed before this feature — and whether those got slower IS the cumulative-drift
    # question C8 asks. Excluding them makes C8 structurally incapable of reporting anything.
    # Same predicate, opposite meaning, because the two sides being compared differ: two TRACKS
    # of one feature vs two STATES of the trunk.
    def _headline(status):
        return [r for r in rows if r["status"] == status
                and not (exclude_controls and r["is_control"])]

    regressions = _headline("REGRESSION")
    suspects = _headline("suspect")
    control_flagged = [r for r in rows
                       if r["is_control"] and r["status"] in ("REGRESSION", "suspect")]
    return {
        "threshold_pct": threshold_pct,
        "min_delta_ms": min_delta_ms,
        "min_calls": min_calls,
        # run-offset provenance. "normalized": whether the report below was corrected by
        # it (apply_offset=True and enough controls). "offset_estimated": whether a
        # reliable estimate exists at all, regardless of whether it was applied — the
        # contamination diagnostic described in the module docstring.
        "run_offset_ratio": round(offset["ratio"], 4),
        "run_offset_pct": round((offset["ratio"] - 1.0) * 100.0, 1),
        "normalized": corrected,
        "offset_estimated": offset["applied"],
        "n_control_methods": offset["n_controls"],
        "normalization_note": offset["reason"] if apply_offset else (
            offset["reason"] if not offset["applied"]
            else "offset estimated but not applied (apply_offset=False, paper-literal)"),
        "n_methods_compared": sum(1 for r in rows
                                  if r["status"] in ("ok", "REGRESSION", "suspect")),
        "n_regressions": len(regressions),
        "n_suspect": len(suspects),
        "regressed": [r["operation_signature"] for r in regressions],
        "suspect": [r["operation_signature"] for r in suspects],
        # Controls that tripped the threshold anyway. NOT a code signal — a contamination
        # diagnostic. Non-empty means this run's drift survived into the report, so every
        # borderline verdict in it should be read with that in mind.
        "n_control_flagged": len(control_flagged),
        "control_flagged": [r["operation_signature"] for r in control_flagged],
        "rows": rows,
    }


def compare_runs(baseline_run: str, candidate_run: str,
                 threshold_pct: float = 20.0, min_delta_ms: float = 0.5,
                 min_calls: int = 5, control_sigs=None,
                 apply_offset: bool = False) -> dict:
    """Parse two run roots and compare them — raw per-call samples included, so the
    rank-based guard applies (this is the paper-faithful path)."""
    raw_b, raw_c = parse_run(baseline_run), parse_run(candidate_run)
    return compare_summaries(
        {r["operation_signature"]: r for r in summarise(raw_b)},
        {r["operation_signature"]: r for r in summarise(raw_c)},
        threshold_pct, min_delta_ms, min_calls,
        baseline_raw=raw_b, candidate_raw=raw_c, control_sigs=control_sigs,
        apply_offset=apply_offset)


def write_compare_csv(result: dict, out_path: str) -> None:
    fields = ["operation_signature", "baseline_mean_ns", "candidate_mean_ns",
              "candidate_mean_ns_norm", "delta_ns", "delta_pct", "delta_pct_raw",
              "is_control", "baseline_n", "candidate_n",
              "p_value", "cliffs_delta", "effect", "status"]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(result["rows"])


def _ms(ns):
    return f"{ns/1e6:.3f}" if ns is not None else "—"


def main() -> None:
    args = sys.argv[1:]
    threshold_pct = 20.0
    min_delta_ms = 0.5
    min_calls = 5
    out_path = None
    apply_offset = False
    pos = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--threshold-pct":
            threshold_pct = float(args[i + 1]); i += 2
        elif a == "--min-delta-ms":
            min_delta_ms = float(args[i + 1]); i += 2
        elif a == "--min-calls":
            min_calls = int(args[i + 1]); i += 2
        elif a == "--out":
            out_path = args[i + 1]; i += 2
        elif a == "--apply-offset":
            apply_offset = True; i += 1
        else:
            pos.append(a); i += 1
    if len(pos) < 2:
        sys.exit("Usage: compare_kieker.py <baseline_run> <candidate_run> "
                 "[--threshold-pct N] [--min-delta-ms N] [--min-calls N] [--out compare.csv] "
                 "[--apply-offset]")
    baseline_run, candidate_run = pos[0], pos[1]
    result = compare_runs(baseline_run, candidate_run, threshold_pct, min_delta_ms, min_calls,
                          apply_offset=apply_offset)

    if out_path is None:
        out_path = os.path.join(candidate_run, "compare.csv")
    write_compare_csv(result, out_path)

    print(f"  baseline : {baseline_run}")
    print(f"  candidate: {candidate_run}")
    if result.get("normalized"):
        print(f"  run offset: {result['run_offset_pct']:+.1f}% divided out "
              f"({result['n_control_methods']} unchanged control methods)")
    else:
        print(f"  run offset: NOT normalized — {result.get('normalization_note')}")
    print(f"  {result['n_methods_compared']} methods compared, "
          f"{result['n_regressions']} REGRESSION(s), {result['n_suspect']} suspect "
          f"(>= +{threshold_pct:.0f}%, +{min_delta_ms}ms, n>={min_calls}, 2σ) → {out_path}")
    print(f"\n  {'baseline_ms':>12}  {'candidate_ms':>12}  {'delta_%':>9}  "
          f"{'p':>8}  {'effect':>10}  status  operation")
    print(f"  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*10}  ------  {'-'*40}")
    shown = [r for r in result["rows"]
             if r["status"] in ("REGRESSION", "suspect", "NEW", "REMOVED")] \
        or result["rows"][:10]
    for r in shown[:20]:
        dp = f"{r['delta_pct']:+.1f}" if r["delta_pct"] is not None else "—"
        pv = f"{r['p_value']:.4f}" if r.get("p_value") is not None else "—"
        eff = r.get("effect") or "—"
        print(f"  {_ms(r['baseline_mean_ns']):>12}  {_ms(r['candidate_mean_ns']):>12}  "
              f"{dp:>9}  {pv:>8}  {eff:>10}  {r['status']:<6}  {r['operation_signature']}")


if __name__ == "__main__":
    main()
