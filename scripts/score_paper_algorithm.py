#!/usr/bin/env python3
"""
score_paper_algorithm.py — score the algorithm's predictions against the oracle ground truth.

Builds the confusion matrix the paper reports in its Table I: the algorithm's model-based
PREDICTION (from `paper_algorithm_results.jsonl`, step D) against the measured system-level
ORACLE (from each junior workspace's `oracle_stats.json`, produced by scripts/run_oracle.sh).

    TP — predicted regression, oracle confirms regression
    FP — predicted regression, oracle sees none          (false alarm)
    TN — predicted clean,      oracle sees none
    FN — predicted clean,      oracle confirms regression (missed)

GROUND TRUTH IS THE FEATURE-SCOPED VERDICT, NOT THE POOLED ONE. `oracle_stats.json` carries
both, and `profiling/system_oracle_stats.compute_verdict` says which is which at the point of
production: `system_verdict` pools every sampler, `feature_verdict` isolates the one sampler
that reaches the changed code. The algorithm's input is component data about exactly that code,
so `feature_verdict` is the like-for-like label; scoring against the pooled verdict asks the
detector to predict the behaviour of endpoints it was never shown.

The difference is not academic. On `reorder` the feature endpoint measures +1240% (delta 0.997,
p=0), while every other sampler — running byte-identical webui code on both tracks — measures
17-37% FASTER, so the pooled delta lands at -0.013 ("negligible", verdict "none"). Two effects
push that way: whole-run host drift (the same +-30% artefact already corrected for the local
Kieker signal), and closed-loop coupling, where threads blocked on the slow feature request
lower the concurrency reaching every other endpoint and genuinely speed it up. Pooling here is
not merely diluted, it is anti-correlated with the regression under test. The pooled numbers
stay in the output beside the feature ones — the dilution is a real finding about workload-level
detection — but they cannot serve as the label.

An oracle that never called the feature (`feature_exercised_by_workload` false, or the key
absent in stats files written before the Feature sampler existed) is reported as UNMEASURED and
excluded, for the same reason an unrun oracle is: it is a verdict about unmodified code, not
evidence of no regression.

BASELINE COMPARISON. The same oracle also scores this repo's pre-existing local-only detector:
the `REGRESSION` rows in each junior workspace's `compare.csv` (mean delta >= 20% AND >= 0.5 ms
AND >= 5 calls AND a rank-based noise guard). That detector never consults an architectural
model — it flags a feature as regressed whenever any single method regressed locally. Scoring
both against one oracle is what answers the thesis's actual question: does the paper's
architectural-model bridge beat naive local thresholding on LLM-scale-induced regressions?

Features with no oracle result are reported as UNSCORED and excluded from the matrix rather
than silently counted as negatives — an unrun oracle is missing data, not a "no regression".

Features whose dual-track pair is not a VALID sample are excluded for the same reason. Two
disqualifiers, both read from the dataset row:
  * `admitted: false` — the pair failed the equivalence gate, so senior and junior do not
    implement the same behaviour and any performance difference between them is uninterpretable.
  * `measurement.workload_asymmetry` — the two tracks were profiled a different number of
    times, or one track has profile data the other lacks, in a profile dir a result is computed
    from. The local comparison is then invalid regardless of drift normalization.
teastore-hist-volatility hits both: its junior throws JPQLException on getPriceVolatility (11
test errors, BUILD FAILURE) and therefore only ever produced 1 profiling iteration against the
senior's 5 (2,696 samples vs 338,294). Scoring a detector against a sample whose "regression"
is a crashing query would measure the wrong thing.

`measurement.call_volumes` is REPORTED but never disqualifying. A call-count ratio far from 1
at EQUAL replicate counts is what an N+1 regression looks like from the outside — lift's junior
adds `CategoryRepository.resolveCategoryName` (9,240 calls vs the senior's 0) for a 2.36x ratio
in `profile_only` that IS the regression under test. A driver fault and a real N+1 cannot be
told apart from call counts, so the skew is surfaced next to the score and left to the analyst
rather than silently deciding the sample set.

Usage:
  python3 scripts/score_paper_algorithm.py
  python3 scripts/score_paper_algorithm.py --variant "norm=control-median,weight=none,boot=block"
"""
import argparse
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO, "teastore-dataset")


def load_predictions(path: str, variant: str = None) -> dict:
    """Last-row-wins per (feature, variant), matching the repo's append-log convention."""
    rows = {}
    if not os.path.isfile(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if variant and r.get("variant") != variant:
                continue
            rows[(r.get("feature"), r.get("variant"))] = r
    return rows


def load_oracle(feature: str) -> dict | None:
    """Read the measured system-level verdict for one feature, if the oracle has been run."""
    p = os.path.join(DATASET_DIR, f"teastore-hist-{feature}-junior", "oracle_stats.json")
    if not os.path.isfile(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def _feature_share_pct(o: dict):
    """The Feature label's request-weighted share of pooled response time, senior side.

    Mirrors `dual_track._feature_share_pct`; duplicated rather than imported because this
    script must run against a bare results directory without the pipeline package on the path.
    """
    feat = (o.get("per_label") or {}).get("Feature")
    if not feat or not o.get("baseline_mean_ms") or not o.get("baseline_n"):
        return None
    den = o["baseline_mean_ms"] * o["baseline_n"]
    if not den:
        return None
    return round((feat.get("baseline_mean_ms") or 0.0) * (feat.get("baseline_n") or 0)
                 / den * 100.0, 3)


def oracle_ground_truth(o: dict) -> tuple[bool | None, str | None]:
    """(actual_regression, why_unusable) for one oracle result.

    Reads the feature-scoped verdict — see the module docstring for why the pooled one cannot
    serve as the label. The exercised check is `is not True` rather than a falsy test because
    the key is absent entirely in stats files written before the Feature sampler existed, and
    those are exactly the runs this has to catch."""
    if o.get("feature_exercised_by_workload") is not True:
        return None, ("the load generator never called the feature endpoint — run predates "
                      "the Feature sampler, or the sampler was disabled")
    verdict = o.get("feature_verdict")
    if verdict is None:
        return None, "feature sampler ran but produced no verdict"
    return verdict == "regression", None


def load_data_fidelity(dataset: str) -> dict:
    """{feature_short_name: violated bool_or_None} from the dataset row's oracle.data_fidelity
    block (scripts/feature_cost_driver.py, merged in by merge_oracle_into_rows.py). Not read
    from oracle_stats.json — data_fidelity is a static per-feature annotation about the method's
    source, not a measurement, so it only ever lives in the merged dataset row. Same last-row-
    wins/superseded-skip convention as `sample_validity`."""
    out = {}
    if not os.path.isfile(dataset):
        return out
    with open(dataset) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("superseded"):
                continue
            fid = (r.get("feature_id") or "").replace("teastore-hist-", "")
            if not fid:
                continue
            fidelity = (r.get("oracle") or {}).get("data_fidelity")
            out[fid] = fidelity.get("violated") if fidelity else None
    return out


def validity_split(row: dict) -> str:
    """Assigns a row to its validity split in precedence order:
    saturation is checked first because it already dominates the row's interpretation (no finite
    step-D prediction at all); data-fidelity violations are next; response-degenerate is last.
    'honored' is every row not caught by any of the three."""
    if row.get("saturated"):
        return "saturated"
    if row.get("fidelity_violated"):
        return "fidelity-violated"
    if row.get("response_nonempty") is False:
        return "response-degenerate"
    return "honored"


def split_confusion(rows: list[dict]) -> dict:
    """{split_name: confusion matrix} — confusion matrices across validity splits."""
    buckets: dict[str, list[tuple[bool, bool]]] = {}
    for r in rows:
        buckets.setdefault(r["split"], []).append((r["predicted"], r["actual"]))
    return {name: confusion(pairs) for name, pairs in buckets.items()}


def sample_validity(dataset: str) -> dict:
    """{feature_short_name: None if the pair is a valid sample, else why it is not}.

    Last row wins per feature_id, matching the repo's append-log convention; superseded rows
    describe implementations that are no longer on disk and are ignored."""
    out = {}
    if not os.path.isfile(dataset):
        return out
    with open(dataset) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("superseded"):
                continue
            fid = (r.get("feature_id") or "").replace("teastore-hist-", "")
            if not fid:
                continue
            why = None
            dep = r.get("deployability") or {}
            if r.get("admitted") is not True:
                # Report the reason that actually blocked admission, not just the first
                # field on the row: a pair can pass the equivalence gate and still be
                # rejected as undeployable (copurchase), and saying "both green, N shared
                # tests pass" as the exclusion reason would be actively misleading.
                if (r.get("equivalence") or {}).get("equivalent") is False:
                    why = f"not admitted — {r['equivalence'].get('reason', 'n/a')}"
                elif dep.get("deployable") is False:
                    bad = next((d.get("reason") for k, d in dep.items()
                                if isinstance(d, dict) and d.get("deployable") is False), "")
                    why = f"not admitted — undeployable: {bad}"
                else:
                    why = "not admitted"
            elif (r.get("measurement") or {}).get("workload_asymmetry"):
                why = f"asymmetric workload — {r['measurement']['workload_asymmetry']}"
            out[fid] = why
    return out


def call_volume_skew(dataset: str) -> dict:
    """{feature: {profile_dir: ratio}} for pairs whose junior executed a very different number
    of instrumented calls at equal replicate counts. Reported beside the score, never used to
    exclude — see the module docstring."""
    out = {}
    if not os.path.isfile(dataset):
        return out
    with open(dataset) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("superseded"):
                continue
            fid = (r.get("feature_id") or "").replace("teastore-hist-", "")
            if not fid:
                continue
            vols = ((r.get("measurement") or {}).get("call_volumes") or {})
            skew = {p: v.get("call_ratio") for p, v in vols.items()
                    if isinstance(v, dict) and v.get("skewed")}
            out[fid] = skew
    return {f: s for f, s in out.items() if s}


def load_local_only_detector(feature: str) -> dict | None:
    """This repo's existing local threshold detector: any REGRESSION row in compare.csv."""
    p = os.path.join(DATASET_DIR, f"teastore-hist-{feature}-junior", "compare.csv")
    if not os.path.isfile(p):
        return None
    n_reg = n_suspect = n = 0
    regressed = []
    with open(p, newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            status = (row.get("status") or "").strip()
            if status == "REGRESSION":
                n_reg += 1
                regressed.append(row["operation_signature"])
            elif status == "suspect":
                n_suspect += 1
    return {"predicted_regression": n_reg > 0, "n_regression_rows": n_reg,
            "n_suspect": n_suspect, "n_rows": n, "regressed": regressed}


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (predicted, actual) -> confusion matrix and derived rates."""
    tp = sum(1 for p, a in pairs if p and a)
    fp = sum(1 for p, a in pairs if p and not a)
    tn = sum(1 for p, a in pairs if not p and not a)
    fn = sum(1 for p, a in pairs if not p and a)
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    acc = (tp + tn) / len(pairs) if pairs else None
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "n": len(pairs),
            "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def _fmt(x, spec="6.3f"):
    return "   n/a" if x is None else format(x, spec)


def _outcome(pred: bool, actual: bool) -> str:
    return {(True, True): "TP", (True, False): "FP",
            (False, False): "TN", (False, True): "FN"}[(pred, actual)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # The default used to be `paper_algorithm_results.jsonl` — the 2026-07-27 full-sweep run,
    # superseded twice since and contradicted by every later file. A stale default is worse
    # than no default: it produces a plausible-looking score from the wrong inputs, silently.
    ap.add_argument("--results",
                    default=os.path.join(REPO, "paper_algorithm_results_split.jsonl"))
    ap.add_argument("--variant", default=None,
                    help="score only this variant id (default: score each separately)")
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "paper_algorithm_scores.json"))
    ap.add_argument("--dataset",
                    default=os.path.join(REPO, "perf_dataset_teastore_history.jsonl"),
                    help="dataset used to decide which pairs are valid samples")
    ap.add_argument("--score-invalid", action="store_true",
                    help="include pairs that failed the equivalence/symmetry gate (not "
                         "meaningful; provided only to reproduce a pre-quarantine score)")
    args = ap.parse_args()

    preds = load_predictions(args.results, args.variant)
    if not preds:
        print(f"!! no predictions in {args.results} — run scripts/run_paper_algorithm.py first",
              file=sys.stderr)
        return 1

    # norm=none (paper-literal — see run_paper_algorithm.py) is the headline variant and
    # printed/reported first; the rest fall back to plain alphabetical order.
    variants = sorted({v for _f, v in preds},
                      key=lambda v: (0 if v.startswith("norm=none") else 1, v))
    features = sorted({f for f, _v in preds})

    oracles = {f: load_oracle(f) for f in features}
    validity = sample_validity(args.dataset)
    fidelity = load_data_fidelity(args.dataset)
    invalid = {f: validity.get(f) for f in features
               if validity.get(f) and not args.score_invalid}
    # An oracle file existing is not the same as it having measured the feature; classify each
    # one before it can become a label.
    truth, unmeasured = {}, {}
    for f in features:
        if f in invalid or oracles[f] is None:
            continue
        actual, why = oracle_ground_truth(oracles[f])
        if why:
            unmeasured[f] = why
        else:
            truth[f] = actual
    scored = [f for f in features if f in truth]
    unscored = [f for f in features if oracles[f] is None and f not in invalid]

    skew = {f: s for f, s in call_volume_skew(args.dataset).items() if f in scored}
    report = {"variants": {}, "scored_features": scored, "unscored_features": unscored,
              "unmeasured_features": unmeasured, "invalid_samples": invalid,
              "call_volume_skew": skew}

    print(f"features with predictions: {len(features)}   with oracle ground truth: {len(scored)}")
    if unscored:
        print(f"UNSCORED (oracle not run — excluded from the matrix, NOT counted as negatives):")
        print(f"  {', '.join(unscored)}")
    if unmeasured:
        print("UNMEASURED (oracle ran but never exercised the changed code — a verdict about "
              "unmodified endpoints, excluded for the same reason as an unrun oracle):")
        for f, why in sorted(unmeasured.items()):
            print(f"  {f}: {why}")
    if invalid:
        print("EXCLUDED (not a valid dual-track sample — scoring a detector against these "
              "would measure the wrong thing):")
        for f, why in sorted(invalid.items()):
            print(f"  {f}: {why}")
    if skew:
        print("CALL-VOLUME SKEW (reported, still scored — at equal replicate counts this is "
              "also what an N+1 regression looks like):")
        for f, s in sorted(skew.items()):
            print(f"  {f}: " + ", ".join(f"{p} junior/senior={r}" for p, r in sorted(s.items())))

    if not scored:
        print("\nNo oracle results yet, so no confusion matrix can be computed.")
        print("Predictions are still summarised below.\n")
        for v in variants:
            hits = [f for f in features if preds.get((f, v), {}).get("predicted_regression")]
            print(f"  {v:<44} predicts regression for {len(hits)}: "
                  f"{', '.join(hits) if hits else '(none)'}")
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\nwrote {args.out}")
        return 0

    # --- paper's algorithm, per variant
    for v in variants:
        pairs, rows = [], []
        for f in scored:
            r = preds.get((f, v))
            if not r:
                continue
            # SATURATION IS NOT "PREDICTED CLEAN". When the queueing network saturates, step D
            # returns predicted_regression=None and omits the response-time fields entirely —
            # there is no finite distribution to test. `bool(None)` silently scored those as a
            # clean prediction, which is the opposite of what a saturated station means: the
            # model is saying the station is pinned at 100% utilisation. Two of the fourteen
            # features with a system-level result (reach, reorder) are saturated, and both are
            # measured regressions, so the old path turned two true positives into false
            # negatives. It also raised KeyError on `mpd_response_time_ms`, which is absent on
            # exactly those rows — the crash was masked only because the scorer had not been
            # run since a saturated row acquired an oracle.
            saturated = bool(r.get("saturated_stations"))
            pred = bool(r.get("predicted_regression")) or saturated
            actual = truth[f]
            pairs.append((pred, actual))
            row = {
                "feature": f, "predicted": pred, "actual": actual,
                "saturated": saturated,
                "saturated_stations": r.get("saturated_stations") or [],
                "outcome": _outcome(pred, actual),
                # T2.1 (audit check): the scorer previously read only
                # feature_exercised_by_workload and feature_verdict — nothing about whether
                # the response actually carried data, whether the method's cost path is known
                # to diverge from deployment, or whether the load generator kept pace. All
                # three now ride beside every row instead of being invisible in a bare 0/1.
                "response_nonempty": oracles[f].get("response_nonempty"),
                "fidelity_violated": fidelity.get(f),
                "load_generator_starved": any(
                    (oracles[f].get("load_generator") or {}).get(arm, {}).get("starved")
                    for arm in ("baseline", "candidate")),
                "predicted_mpd_ms": r.get("mpd_response_time_ms"),
                "oracle_mpd_pct": oracles[f].get("feature_mpd_pct"),
                "oracle_cliffs_delta": oracles[f].get("feature_cliffs_delta"),
                "predicted_effect": r.get("effect_magnitude"),
                # kept beside the label, never as the label — see the module docstring
                "oracle_pooled_verdict": oracles[f].get("system_verdict"),
                "oracle_pooled_mpd_pct": oracles[f].get("mpd_pct"),
                # The share of end-to-end response time the feature's endpoint actually holds.
                # Step C's denominator is supposed to carry this; the isolated profile sets it
                # to 1, so a demand scale cannot be read without it beside it.
                "oracle_feature_share_pct": _feature_share_pct(oracles[f]),
                "demand_scale": r["demand_scale"],
                # step_b is absent on saturated rows for the same reason the response-time
                # fields are: step D returns early, before the full report is assembled.
                "n_deviated": (r.get("step_b") or {}).get("n_deviated"),
            }
            row["split"] = validity_split(row)
            rows.append(row)
        cm = confusion(pairs)
        report["variants"][v] = {"confusion": cm, "rows": rows,
                                 "splits": split_confusion(rows)}

        print(f"\n=== {v} ===")
        print(f"  TP={cm['TP']} FP={cm['FP']} TN={cm['TN']} FN={cm['FN']}   "
              f"precision={_fmt(cm['precision'])} recall={_fmt(cm['recall'])} "
              f"f1={_fmt(cm['f1'])} accuracy={_fmt(cm['accuracy'])}")
        n_sat = sum(1 for r in rows if r["saturated"])
        if n_sat:
            print(f"  ({n_sat} row(s) saturated — counted as a predicted regression, not as "
                  f"clean; step D has no finite prediction for them)")
        splits = report["variants"][v]["splits"]
        print("  validity split: " + ", ".join(
            f"{name}={sp['n']} (f1={_fmt(sp['f1'])})" for name, sp in sorted(splits.items())))
        print(f"  {'feature':<14} {'out':>4} {'pred MPD ms':>12} {'feat MPD %':>12} "
              f"{'feat delta':>11} {'pred effect':>12} {'pooled':>11} {'share%':>7} {'scale':>10}")
        for r in rows:
            print(f"  {r['feature']:<14} {r['outcome']:>4} "
                  f"{_fmt(r['predicted_mpd_ms'], '12.4f')} "
                  f"{_fmt(r['oracle_mpd_pct'], '12.2f')} "
                  f"{_fmt(r['oracle_cliffs_delta'], '11.3f')} "
                  f"{(r['predicted_effect'] or 'saturated'):>12} "
                  f"{str(r['oracle_pooled_verdict']):>11} "
                  f"{_fmt(r['oracle_feature_share_pct'], '7.2f')} x{r['demand_scale']:.5f}")

    # --- baseline: local-only threshold detector
    base_pairs, base_rows = [], []
    for f in scored:
        loc = load_local_only_detector(f)
        if loc is None:
            continue
        actual = truth[f]
        base_pairs.append((loc["predicted_regression"], actual))
        base_row = {"feature": f, "predicted": loc["predicted_regression"],
                    "actual": actual,
                    "outcome": _outcome(loc["predicted_regression"], actual),
                    "n_regression_rows": loc["n_regression_rows"],
                    "n_rows": loc["n_rows"],
                    # parameter check: the algorithm's rows carry response_nonempty/fidelity_violated
                    # (T2.1); the baseline detector's did not, so its errors couldn't be told
                    # apart from the algorithm's on the same validity axes. The join is free —
                    # both are oracle-side row properties keyed by feature, not a measurement
                    # of the baseline detector itself. No "saturated" key: station saturation
                    # is a step-D/queueing-model artifact this threshold detector never has.
                    "response_nonempty": oracles[f].get("response_nonempty"),
                    "fidelity_violated": fidelity.get(f)}
        base_row["split"] = validity_split(base_row)
        base_rows.append(base_row)
    if base_pairs:
        bcm = confusion(base_pairs)
        base_splits = split_confusion(base_rows)
        report["baseline_local_only"] = {"confusion": bcm, "rows": base_rows,
                                         "splits": base_splits}
        print(f"\n=== BASELINE: local-only threshold detector (compare.csv REGRESSION rows) ===")
        print(f"  TP={bcm['TP']} FP={bcm['FP']} TN={bcm['TN']} FN={bcm['FN']}   "
              f"precision={_fmt(bcm['precision'])} recall={_fmt(bcm['recall'])} "
              f"f1={_fmt(bcm['f1'])} accuracy={_fmt(bcm['accuracy'])}")
        print("  validity split: " + ", ".join(
            f"{name}={sp['n']} (f1={_fmt(sp['f1'])})" for name, sp in sorted(base_splits.items())))
        for r in base_rows:
            print(f"  {r['feature']:<14} {r['outcome']:>4}  "
                  f"{r['n_regression_rows']}/{r['n_rows']} methods flagged")

        print("\n=== does the architectural-model bridge beat local thresholding? ===")
        for v in variants:
            cm = report["variants"][v]["confusion"]
            print(f"  {v:<44} f1={_fmt(cm['f1'])}  vs baseline f1={_fmt(bcm['f1'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
