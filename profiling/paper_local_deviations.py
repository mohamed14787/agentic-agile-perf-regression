#!/usr/bin/env python3
"""
paper_local_deviations.py — STEP B of Liao et al., ICSE'25, implemented literally.

  "Early Detection of Performance Regressions by Bridging Local Performance Data and
   Architectural Models", §III-B ("Identifying local performance deviations").

The paper's rule, and nothing else:

    Wilcoxon rank-sum at the 5% level  AND  Cliff's Delta magnitude not "negligible"
    => this local component has a performance deviation.
    Then, for those components only, compute MD (mean difference) of local performance
    before vs after. Positive MD = decreased performance (longer execution time).

WHY THIS IS A SEPARATE MODULE FROM compare_kieker.py. `compare_kieker.compare_summaries`
answers a different question. It ANDs three extra gates onto the paper's rule — the candidate
mean must exceed baseline by >= 20%, the absolute delta must exceed 0.5 ms, and both sides
need >= 5 calls — because its job is to decide whether to fail a dual-track pipeline run,
where a false alarm is expensive. Those gates are this project's, not the paper's, and they
change which components get flagged: a component with a 4% slowdown across 40,000 calls is
"ok" to compare_kieker and a genuine local deviation to the paper. Since the point here is to
evaluate the PAPER'S algorithm, the paper's rule has to be applied unmodified. Both reports
are produced and compared in the final evaluation, so the difference is quantified rather
than assumed away.

RUN-OFFSET NORMALIZATION — DEFAULT CHANGED TO "none" (2026-07-30). The paper repeats the
component test 30 times INSIDE one Kieker session (confirmed directly against their own
replication package — every kieker.dat in Component_perf_res/ carries one Kieker metadata
record and per-operation call counts that are exact multiples of 30), not across 30 separate
test invocations. It has no cross-invocation machine-state offset to normalize because it
never crosses a JVM boundary between repetitions. `--normalize control-median` was added
here (2026-07-26) to correct for a real problem this project HAD: profiling ran 5 SEPARATE
`mvn test` invocations, and a byte-identical one-line `getEntityClass()` came out 16-51%
"faster" in the candidate track, median offset -30.5% (see compare_kieker.py's module
docstring). The fix for that problem is now upstream of this module: kieker_profile.sh and
dual_track.py default to ONE in-session profile per track (FeaturePerfExerciseTest repeats
the queries 30 times inside that one JVM), which removes the cross-invocation offset at its
source rather than correcting for it after measurement. `--normalize none` is therefore now
the default and the paper-literal path; `--normalize control-median` remains available as a
sensitivity comparison.

NEITHER MODE IS THE SAFE ONE (corrected 2026-08-07). This docstring previously claimed that a
nonzero estimated offset under the in-session protocol "would mean a pair was NOT actually
profiled in-session". That inference is wrong, and acting on it mislabels a real regression as
a profiling fault. Control methods are unchanged CODE, not an unchanged ENVIRONMENT: when the
candidate does more work, they slow down too, and the estimator cannot tell that from drift.

  * On the paper's own L2 data the offset IS drift — x1.1033 / x1.1018 / x1.0467, measured on
    components nowhere near the injection. Leaving it in contributes more to step C's sum than
    the injected regression does at Low and Medium (see paper_propagate's L2 note).
  * On teastore-hist-lift it is NOT drift. The junior adds
    `CategoryRepository.resolveCategoryName` (9,240 calls against the senior's 0) plus 9,208
    extra `getEntityClass` calls — 2.36x the instrumented work per iteration in `profile_only`.
    Its x1.1729 offset is a second-order consequence of the regression under test, and dividing
    it out removes real signal: step C's deviation drops from +138.3% to +102.6%.

So `none` is exposed to machine drift and `control-median` is exposed to workload-coupled
slowdown. Report both, and check the call-volume ratio (`measurement.call_volumes`, written by
scripts/remeasure_chain.py) before deciding which number to trust for a given pair.

Control signatures are read from the `is_control` column of an existing `compare.csv`, which
`dual_track` already derived from the same source-diff that drives the feature-exercised gate
(`_feature_targets`), so no new notion of "unchanged method" is invented here.

Programmatic use:
  from paper_local_deviations import load_samples, identify_deviations
  rep = identify_deviations(baseline_raw, candidate_raw, control_sigs=controls)

CLI:
  python3 profiling/paper_local_deviations.py <senior_profile_dir> <junior_profile_dir> \
      [--normalize none|control-median] [--controls-from compare.csv] [--out dev.json]
"""
import argparse
import csv
import json
import os
import sys

try:
    from parse_kieker import summarise
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta
    from compare_kieker import estimate_run_offset
except ImportError:  # imported from another package (e.g. agile_team_cline)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parse_kieker import summarise
    from nonparametric_stats import wilcoxon_rank_sum, cliffs_delta
    from compare_kieker import estimate_run_offset

# csv's default field-size cap is 128 KB; Kieker signatures are long but nowhere near that.
# Raised anyway because samples.csv files here reach 140 MB and a single malformed line
# would otherwise abort the whole load with an opaque error.
csv.field_size_limit(1 << 24)

P_THRESHOLD = 0.05          # paper: "the statistical test at the 5% level of significance"


def load_samples(path: str) -> dict[str, list[int]]:
    """Load `profile/samples.csv` (operation_signature,exec_time_ns) -> {sig: [ns, ...]}."""
    out: dict[str, list[int]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sig = row["operation_signature"]
            try:
                out.setdefault(sig, []).append(int(row["exec_time_ns"]))
            except (TypeError, ValueError):
                continue
    return out


def load_samples_tagged(path: str) -> dict[str, dict[str, list[int]]]:
    """Load `profile/samples_tagged.csv` -> {sig: {iteration: [ns, ...]}}.

    The iteration boundary is the independent replicate for the bootstrap in step D
    (calls within one iteration share JIT/GC/cache state and are NOT independent), which
    is exactly what plain samples.csv throws away.
    """
    out: dict[str, dict[str, list[int]]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sig, it = row["operation_signature"], row["iteration"]
            try:
                out.setdefault(sig, {}).setdefault(it, []).append(int(row["exec_time_ns"]))
            except (TypeError, ValueError):
                continue
    return out


def per_iteration_stats(tagged: dict[str, dict[str, list[int]]]) -> dict[str, dict[str, dict]]:
    """Collapse tagged samples to {sig: {iteration: {"sum": float, "n": int}}}.

    Precomputed so a block-bootstrap replicate in step D costs O(n_sigs * n_iters) instead of
    re-summing ~10^6 raw samples per replicate: the mean of a resampled iteration set is just
    sum(selected sums) / sum(selected counts). Without this, 1000 replicates over two tracks
    is ~10^9 Python-level additions.
    """
    return {sig: {it: {"sum": float(sum(v)), "n": len(v)} for it, v in iters.items()}
            for sig, iters in tagged.items()}


def load_control_sigs(compare_csv: str) -> list[str]:
    """Read control signatures from the `is_control` column of an existing compare.csv."""
    controls = []
    with open(compare_csv, newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("is_control", "")).strip().lower() in ("true", "1", "yes"):
                controls.append(row["operation_signature"])
    return controls


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def identify_deviations(baseline_raw: dict[str, list], candidate_raw: dict[str, list],
                        control_sigs=None, normalize: str = "none",
                        min_controls: int = 5) -> dict:
    """Apply the paper's step-B rule to raw per-signature samples.

    Returns {components: [...], deviated: [...], new_components, removed_components,
             run_offset: {...}, rule, n_compared}. `deviated` is the subset of `components`
             where the paper's rule fires; step C consumes exactly that subset.
    """
    base_sum = {r["operation_signature"]: r for r in summarise(baseline_raw)}
    cand_sum = {r["operation_signature"]: r for r in summarise(candidate_raw)}

    if normalize == "control-median":
        offset = estimate_run_offset(base_sum, cand_sum, control_sigs,
                                     min_calls=1, min_controls=min_controls)
    elif normalize == "none":
        offset = {"ratio": 1.0, "n_controls": 0, "applied": False,
                  "reason": "normalization disabled (--normalize none, paper-literal)"}
    else:
        raise ValueError(f"unknown normalize mode: {normalize!r}")
    k = offset["ratio"] if offset["applied"] else 1.0

    shared = sorted(set(base_sum) & set(cand_sum))
    components = []
    for sig in shared:
        raw_b = baseline_raw[sig]
        raw_c = [x / k for x in candidate_raw[sig]] if k != 1.0 else candidate_raw[sig]

        p_value = wilcoxon_rank_sum(raw_b, raw_c)
        delta, magnitude = cliffs_delta(raw_b, raw_c)
        # The paper's rule verbatim: significant AND effect size beyond negligible. Note it is
        # two-sided -- a component that got FASTER is also a "deviation" (the paper's Table I
        # carries negative MPD rows, e.g. TeaStore L3-Low at -1.08 ms), so direction is
        # recorded in MD's sign rather than used as a filter here.
        deviated = p_value <= P_THRESHOLD and magnitude != "negligible"

        b_mean, c_mean = _mean(raw_b), _mean(raw_c)
        components.append({
            "operation_signature": sig,
            "deviated": deviated,
            "p_value": p_value,
            "cliffs_delta": delta,
            "magnitude": magnitude,
            # MD, the paper's mean difference. Positive => decreased performance (slower).
            "md_ns": c_mean - b_mean,
            "md_ms": (c_mean - b_mean) / 1e6,
            "baseline_mean_ns": b_mean,
            "candidate_mean_ns": c_mean,
            "candidate_mean_ns_raw": _mean(candidate_raw[sig]),
            "baseline_n": len(raw_b),
            "candidate_n": len(raw_c),
        })

    components.sort(key=lambda c: -abs(c["md_ns"]))
    return {
        "rule": "wilcoxon p <= 0.05 AND cliffs delta magnitude != negligible (Liao et al. III-B)",
        "normalize": normalize,
        "run_offset": offset,
        "n_compared": len(shared),
        "components": components,
        "deviated": [c for c in components if c["deviated"]],
        "new_components": sorted(set(cand_sum) - set(base_sum)),
        "removed_components": sorted(set(base_sum) - set(cand_sum)),
    }


def _summary_line(rep: dict) -> str:
    off = rep["run_offset"]
    slower = sum(1 for c in rep["deviated"] if c["md_ns"] > 0)
    return (f"{len(rep['deviated'])}/{rep['n_compared']} components deviated "
            f"({slower} slower, {len(rep['deviated']) - slower} faster)  "
            f"normalize={rep['normalize']} offset={'x%.4f' % off['ratio'] if off['applied'] else 'n/a'}"
            f" controls={off['n_controls']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline_profile", help="senior (original) profile dir with samples.csv")
    ap.add_argument("candidate_profile", help="junior (updated) profile dir with samples.csv")
    ap.add_argument("--normalize", choices=["none", "control-median"], default="none")
    ap.add_argument("--controls-from", default=None,
                    help="compare.csv whose is_control column names the control methods "
                         "(default: <candidate_profile>/../compare.csv)")
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    ap.add_argument("--top", type=int, default=15, help="rows to print (default 15)")
    args = ap.parse_args()

    b_path = os.path.join(args.baseline_profile, "samples.csv")
    c_path = os.path.join(args.candidate_profile, "samples.csv")
    for p in (b_path, c_path):
        if not os.path.isfile(p):
            print(f"!! missing {p}", file=sys.stderr)
            return 1

    controls = None
    cfrom = args.controls_from or os.path.join(
        os.path.dirname(os.path.abspath(args.candidate_profile)), "compare.csv")
    if args.normalize == "control-median":
        if os.path.isfile(cfrom):
            controls = load_control_sigs(cfrom)
            print(f"controls: {len(controls)} from {cfrom}")
        else:
            print(f"!! no compare.csv at {cfrom} — cannot identify controls; "
                  f"report will be UNNORMALIZED and flagged", file=sys.stderr)

    print(f"loading {b_path} ...")
    base = load_samples(b_path)
    print(f"loading {c_path} ...")
    cand = load_samples(c_path)

    rep = identify_deviations(base, cand, control_sigs=controls, normalize=args.normalize)
    print("\n" + _summary_line(rep))
    if not rep["run_offset"]["applied"]:
        print(f"  offset NOT applied: {rep['run_offset']['reason']}")

    print(f"\n{'MD (ms)':>10} {'p':>9} {'delta':>7} {'effect':>10} {'dev':>4}  signature")
    for c in rep["components"][:args.top]:
        print(f"{c['md_ms']:>10.4f} {c['p_value']:>9.3g} {c['cliffs_delta']:>7.3f} "
              f"{c['magnitude']:>10} {'YES' if c['deviated'] else '-':>4}  "
              f"{c['operation_signature'][:78]}")
    if rep["new_components"]:
        print(f"\nNEW in candidate ({len(rep['new_components'])}):")
        for s in rep["new_components"][:10]:
            print(f"  {s[:100]}")
    if rep["removed_components"]:
        print(f"\nREMOVED from baseline ({len(rep['removed_components'])}):")
        for s in rep["removed_components"][:10]:
            print(f"  {s[:100]}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rep, fh, indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
