#!/usr/bin/env python3
"""
run_paper_algorithm.py — run Liao et al.'s ICSE'25 algorithm over the dual-track dataset.

Applies steps B (paper_local_deviations) -> C (paper_propagate) -> D (paper_detect) to every
senior/junior pair that has profile data, and appends one row per (feature, variant) to
`paper_algorithm_results.jsonl`.

VARIANTS. Three of the algorithm's inputs are places where this dataset forces a documented
choice, so each is swept rather than silently fixed:

  --normalize none | control-median
      Whether cross-run machine drift is divided out before the paper's step-B test. `none`
      (the default, 2026-07-30) is paper-literal — the paper has no such correction because its
      30 repetitions happen INSIDE one Kieker session, so there is no cross-invocation drift to
      correct for. This project's profiling harnesses now default to the same in-session
      protocol (FeaturePerfExerciseTest repeats each query 30x in one JVM — see
      kieker_profile.sh/dual_track.py), which removes the offset at its source rather than
      correcting for it after the fact. `control-median` remains available as a sensitivity
      run and as a way to audit a specific pair for contamination (a large estimated offset
      under the in-session protocol means that pair was not actually profiled cleanly).
  --weight none | calls
      Whether the top-level subsystem sum is unweighted (paper-literal) or weighted by root
      invocation count.
  --bootstrap block | iid
      Whether step D's distributions resample whole iterations (correct — iterations are the
      independent replicate) or individual calls. With the in-session protocol, "iteration"
      here means the 30 in-session repeats (see parse_kieker.py's rank-based `rep_NN`
      labeling), not 30 separate `mvn test` invocations as it did before 2026-07-30.

The default sweep runs the paper-literal corner FIRST (headline) plus two sensitivity corners
so the difference is visible without a combinatorial explosion; `--all-variants` runs the full
cross product.

Follows the repo's append-log convention: rows accumulate, and `scripts/dataset_view.py`-style
last-row-wins collapsing applies per (feature_id, variant).

Usage:
  python3 scripts/run_paper_algorithm.py --model paper_artifacts/teastore_model.json
  python3 scripts/run_paper_algorithm.py --model M.json --only volatility --replicates 2000
"""
import argparse
import datetime
import json
import os
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "profiling"))

from paper_detect import detect                                    # noqa: E402
from paper_local_deviations import load_control_sigs               # noqa: E402

DATASET_DIR = os.path.join(REPO, "teastore-dataset")

def discover_features(dataset_dir: str, profile_dir: str) -> list[str]:
    """Features whose senior AND junior workspaces both hold a samples.csv, read from disk.

    This replaced a hardcoded 10-feature list that had drifted into being wrong in both
    directions: it named features with no usable profile, omitted every feature that had one,
    and its comment claimed bundles/orphans/clv were "senior-only, no junior exists" when 25
    juniors exist on disk (the real problem was always missing profiles, not missing code).
    A list that has to be edited by hand to stay true is a list that will be false.
    """
    if not os.path.isdir(dataset_dir):
        return []
    found = set()
    for entry in os.listdir(dataset_dir):
        if not entry.startswith("teastore-hist-"):
            continue
        for suffix in ("-senior", "-junior"):
            if entry.endswith(suffix):
                found.add(entry[len("teastore-hist-"):-len(suffix)])
    usable = []
    for feature in sorted(found):
        stem = os.path.join(dataset_dir, "teastore-hist-" + feature)
        if all(os.path.isfile(os.path.join(stem + t, profile_dir, "samples.csv"))
               for t in ("-senior", "-junior")):
            usable.append(feature)
    return usable

DEFAULT_VARIANTS = [
    {"normalize": "none", "weight": "none", "bootstrap": "block"},              # headline: paper-literal
    {"normalize": "control-median", "weight": "none", "bootstrap": "block"},    # sensitivity: drift-corrected
    {"normalize": "control-median", "weight": "calls", "bootstrap": "block"},
]


def _workload_label(profile_dir: str) -> str:
    """Plain-language description of the workload a profile directory was measured under.

    `profile`         the functional suite — every test, JUnit-sized fixtures
    `profile_only*`   the perf exerciser driving ONE query at production cardinality
    `*_split`         the same, with the fixture-loading phase removed by timestamp
    `*_rep*`          a senior-vs-senior replicate, i.e. the noise floor
    """
    if not profile_dir.startswith("profile_only"):
        return "full-sweep"
    parts = ["perf-exercise-isolated"]
    if "_rep" in profile_dir:
        parts.append("replicate")
    if profile_dir.endswith("_split"):
        parts.append("setup-split")
    return "+".join(parts)


def _variant_id(v: dict) -> str:
    return f"norm={v['normalize']},weight={v['weight']},boot={v['bootstrap']}"


def _paths(feature: str, profile_dir: str = "profile") -> dict:
    sen = os.path.join(DATASET_DIR, f"teastore-hist-{feature}-senior")
    jun = os.path.join(DATASET_DIR, f"teastore-hist-{feature}-junior")
    return {
        "senior_dir": sen, "junior_dir": jun,
        "senior_profile": os.path.join(sen, profile_dir),
        "junior_profile": os.path.join(jun, profile_dir),
        "compare_csv": os.path.join(jun, "compare.csv"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="architectural model descriptor JSON")
    ap.add_argument("--out", default=os.path.join(REPO, "paper_algorithm_results.jsonl"))
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to these features (repeatable)")
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260727,
                    help="RNG seed for step D's bootstrap resampling, passed through to "
                         "paper_detect.detect() and recorded in each row's variant_params. "
                         "Provenance-only (parameter check): the default reproduces every existing "
                         "row; a different value demonstrates the seed-fragile detection "
                         "band (x1.728-x1.861) instead of asserting it.")
    ap.add_argument("--profile-dir", default="profile_only",
                    help="profile subdirectory to read per workspace. Default 'profile_only' is "
                         "the PERF_only isolated profile, the only workload valid for step C "
                         "(specification F6). Pass 'profile' for the full sweep against "
                         "assertion-sized fixtures — retained for comparison only: it cannot "
                         "expose these regressions and its call mix drowns the target in step C "
                         "(measured on reorder, the feature fell to 0.238%% of station demand).")
    ap.add_argument("--class-aware", action="store_true",
                    help="route sampled requests by workload class in step D instead of sending "
                         "every request through every station with mix-averaged demands. The "
                         "faithful analog of SimQPN's coloured QPN; makes step D detect LESS "
                         "(the default under-disperses, sd 13.2 ms vs ~21 ms, which inflates "
                         "Cliff's Delta). Opt-in: the Table I match was validated on the default.")
    ap.add_argument("--all-variants", action="store_true",
                    help="full cross product of normalize x weight x bootstrap")
    ap.add_argument("--servers", type=int, default=None, metavar="M",
                    help="override the model's per-station server count. The descriptor carries "
                         "servers=2 because it is parsed from Liao et al.'s teastore_model.qpe, "
                         "i.e. it describes THEIR deployment. Our oracle caps every container at "
                         "1.0 CPU (scripts/oracle_compose.py), so `--servers 1` is the setting "
                         "that describes the system our own measurements actually ran on. It "
                         "moves the step-D boundaries (detection x1.75->x1.65, saturation "
                         "x9.18->x4.59). Leave unset to reproduce the paper's model, which is "
                         "what validate_against_paper.py must keep using.")
    args = ap.parse_args()

    with open(args.model) as fh:
        model = json.load(fh)

    if args.servers is not None:
        if args.servers < 1:
            ap.error("--servers must be >= 1")
        # Recorded in provenance, not applied silently: the server count changes both step-D
        # boundaries, so a row must say which deployment its verdict describes.
        model["servers"] = {k: args.servers for k in (model.get("servers") or model["demands"])}
        model["provenance"] = (f"{model.get('provenance', '')} | servers overridden to "
                               f"{args.servers} per station (deployment-matched: our oracle "
                               f"runs 1.0 CPU per container)")

    if args.all_variants:
        variants = [{"normalize": n, "weight": w, "bootstrap": b}
                    for n in ("control-median", "none")
                    for w in ("none", "calls")
                    for b in ("block", "iid")]
    else:
        variants = DEFAULT_VARIANTS

    features = args.only or discover_features(DATASET_DIR, args.profile_dir)
    if not features:
        print(f"\n!! no feature has samples.csv in both tracks under '{args.profile_dir}/'.\n"
              f"   Nothing to run. Profile a pair first, or pass --profile-dir.")
        return 2
    started = datetime.datetime.now().isoformat(timespec="seconds")
    rows, failures = [], []

    for feature in features:
        p = _paths(feature, args.profile_dir)
        missing = [k for k in ("senior_profile", "junior_profile")
                   if not os.path.isfile(os.path.join(p[k], "samples.csv"))]
        if missing:
            print(f"[{feature}] SKIP — no samples.csv in {', '.join(missing)}")
            failures.append({"feature": feature, "error": f"missing samples.csv: {missing}"})
            continue

        controls = (load_control_sigs(p["compare_csv"])
                    if os.path.isfile(p["compare_csv"]) else None)

        for v in variants:
            vid = _variant_id(v)
            try:
                rep = detect(p["senior_profile"], p["junior_profile"], model,
                             weight=v["weight"], normalize=v["normalize"],
                             bootstrap=v["bootstrap"], n_replicates=args.replicates,
                             control_sigs=controls, seed=args.seed,
                             class_aware=args.class_aware)
            except Exception as exc:                                  # noqa: BLE001
                print(f"[{feature}] {vid}  ERROR: {exc}")
                traceback.print_exc(limit=3)
                failures.append({"feature": feature, "variant": vid, "error": str(exc)})
                continue

            row = {
                "feature_id": f"teastore-hist-{feature}",
                "feature": feature,
                "variant": vid,
                # seed recorded alongside normalize/weight/bootstrap (parameter check) — dict(v, ...)
                # rather than mutating v itself, since v is DEFAULT_VARIANTS' shared entry.
                "variant_params": dict(v, seed=args.seed),
                "run_at": started,
                "senior_dir": os.path.relpath(p["senior_dir"], REPO),
                "junior_dir": os.path.relpath(p["junior_dir"], REPO),
                # Provenance. Rows written before 2026-08-15 do not carry this, and a consumer
                # doing last-row-wins cannot otherwise tell an F6-valid isolated result from a
                # full-sweep one — they differ by orders of magnitude in station demand share.
                "profile_dir": args.profile_dir,
                # The old expression tested for equality with "profile_only" and called
                # everything else "full-sweep", so every `profile_only_split` row was labelled
                # full-sweep while `profile_dir` beside it correctly said otherwise — a
                # provenance field contradicting its own neighbour on all 75 shipped rows.
                # Derive it from the prefix so any future `_split`/`_rep` variant is described
                # by the workload that actually produced it.
                "workload": _workload_label(args.profile_dir),
                "n_control_signatures": len(controls) if controls else 0,
                **rep,
            }
            rows.append(row)
            if rep.get("saturated"):
                # No response-time distribution exists, so there is no verdict — and "no verdict"
                # must not print as "clean".
                print(f"[{feature:<12}] {vid:<44} SATURATED   "
                      f"{', '.join(rep['saturated_stations'])}  "
                      f"scale x{rep['demand_scale']:.5f}")
                continue
            verdict = "REGRESSION" if rep["predicted_regression"] else "clean     "
            print(f"[{feature:<12}] {vid:<44} {verdict}  "
                  f"MPD {rep['mpd_response_time_ms']:+9.4f} ms  "
                  f"scale x{rep['demand_scale']:.5f}  "
                  f"dev {rep['step_b']['n_deviated']}/{rep['step_b']['n_compared']}  "
                  f"p={rep['p_value']:.3g}")

    with open(args.out, "a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nappended {len(rows)} rows to {args.out}")

    print("\n=== predicted regressions, per variant ===")
    for v in variants:
        vid = _variant_id(v)
        hits = [r["feature"] for r in rows
                if r["variant"] == vid and r["predicted_regression"]]
        print(f"  {vid:<44} {len(hits)}: {', '.join(hits) if hits else '(none)'}")

    if failures:
        print(f"\n!! {len(failures)} failure(s):")
        for f in failures:
            print(f"   {f}")

    # A driver that writes nothing and exits 0 is the project's own recurring failure mode
    # reproduced at the driver layer: the batch reports success, the dataset never changes, and
    # the gap is found weeks later. Producing no rows is an error, not a quiet outcome.
    if not rows:
        print("\n!! NO ROWS PRODUCED. Every feature failed or was skipped — see above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
