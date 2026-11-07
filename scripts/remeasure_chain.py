#!/usr/bin/env python3
"""remeasure_chain.py — re-derive the perf fields of the chained-trunk dataset WITHOUT
re-running any LLM.

Why this exists (2026-07-26 audit, docs/PIPELINE-AUDIT-2026-07-26.md): every admitted row
carried `n_regressions: 0` because the senior and junior tracks were profiled in separate
back-to-back blocks and compared on absolute means. Machine drift between the two blocks
reached -30% — larger than the 20% detection threshold — and on the reorder pair it inverted
the sign of a real +24% regression. Two fixes landed:

  • compare_kieker run-offset normalization (divide out the median candidate/baseline ratio
    over methods whose source is identical in both tracks), and
  • dual_track.mvn_test_interleaved (alternate S,J,S,J and swap the leading slot).

Both are measurement-side, so the existing senior/junior workspaces — the LLM-authored code —
are already correct on disk and are never touched here. Only `mvn test` is re-run.

Two modes:
  --recompute-only   post-hoc normalization of the profiles already on disk (seconds; the
                     interleaving fix cannot apply retroactively, so drift is only corrected,
                     not prevented). Useful as a cross-check.
  (default)          re-run the frozen suite interleaved under Kieker, then recompute
                     (~1 min per profiled run; 11 features x 2 tracks x --iters).

The dataset's LAST row per feature_id is the one updated: it is the row that describes the
code currently in the workspaces. Earlier duplicate rows describe superseded code that no
longer exists on disk, so they are marked `superseded: true` rather than rewritten with
measurements of a different implementation. The original file is backed up first.

Usage:
  python scripts/remeasure_chain.py --recompute-only
  python scripts/remeasure_chain.py                     # --iters defaults to 1 (paper protocol)
  python scripts/remeasure_chain.py --features teastore-hist-reorder
"""
import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "profiling"))

import parse_kieker                                    # noqa: E402
import compare_kieker                                  # noqa: E402
from agile_team_cline import dual_track as dt          # noqa: E402

# -Dsurefire.runOrder=alphabetical pins the class execution order. The persistence pom sets
# reuseForks=false/forkCount=1, so every test class gets a fresh JVM, a fresh in-memory H2 and
# fresh static *Repository.REPOSITORY singletons — which means how much data a feature method
# is profiled against depends on which classes ran in its fork and in what order. Surefire's
# default runOrder is `filesystem`, which is not stable across machines or checkouts, and the
# existing dataset shows a 100x spread in profiled call counts across features that all use the
# same fixed 30-product DataGenerator (audit BLOCKER 0). A command-line property, deliberately
# NOT a pom edit: the frozen-suite invariant the equivalence gate rests on requires the 22
# workspaces' sources to stay byte-identical.
MVN_ARGS = ["-pl", "services/tools.descartes.teastore.persistence", "-am",
            "-Dsurefire.runOrder=alphabetical"]
ARCHIVE = os.path.join(ROOT, "profiling", "runs", "pre-normalization-2026-07-26")


def log(msg=""):
    print(msg, flush=True)


# ── baselines: a feature's targets are diffed against the trunk as it stood BEFORE it ──
def baseline_for(seq: int, manifest: dict, tmp: str) -> str:
    """Extract the trunk snapshot this feature was staged from (genesis for sequence 1)."""
    if seq <= 1:
        return manifest.get("genesis_baseline") or os.path.join(ROOT, "targets", "TeaStore")
    prev = [t for t in manifest["test_points"] if t["sequence"] == seq - 1]
    if not prev:
        raise SystemExit(f"no snapshot for sequence {seq - 1} — cannot diff feature {seq}")
    snap = os.path.join(ROOT, prev[0]["snapshot"])
    dest = os.path.join(tmp, f"baseline_seq{seq:02d}")
    if not os.path.isdir(dest):
        os.makedirs(dest, exist_ok=True)
        with tarfile.open(snap) as tar:
            tar.extractall(dest)
    return dest


def archive_old(feature: str, senior: str, junior: str) -> None:
    """Keep the pre-fix summaries/comparison for audit (raw .dat is superseded, not kept)."""
    dest = os.path.join(ARCHIVE, feature)
    os.makedirs(dest, exist_ok=True)
    for name, ws in (("senior", senior), ("junior", junior)):
        src = os.path.join(ws, "profile", "summary.csv")
        if os.path.isfile(src) and not os.path.isfile(os.path.join(dest, f"{name}-summary.csv")):
            shutil.copy2(src, os.path.join(dest, f"{name}-summary.csv"))
    old_cmp = os.path.join(junior, "compare.csv")
    if os.path.isfile(old_cmp) and not os.path.isfile(os.path.join(dest, "compare.csv")):
        shutil.copy2(old_cmp, os.path.join(dest, "compare.csv"))


def feature_method(feature: str) -> str | None:
    """The one query this feature added, from scripts/feature_methods.json (or None)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_methods.json")
    try:
        with open(path) as fh:
            feats = json.load(fh)["features"]
    except (OSError, ValueError, KeyError):
        return None
    key = feature if feature.startswith("teastore-hist-") else "teastore-hist-" + feature
    entry = feats.get(feature) or feats.get(key)
    return entry["method"] if entry else None


def measure(feature: str, senior: str, junior: str, iters: int) -> tuple:
    """Interleaved profiled run against both tracks, isolated to this feature's own query.

    Without DUAL_PERF_only the exerciser runs the full 25-query sweep, whose relative call
    counts describe the sweep rather than a request mix — measured on reorder, the feature under
    test fell to 0.238% of station demand and a 22x local regression predicted +0.03 ms
    system-wide. Isolation is what makes the profile valid for the paper's propagation step
    (specification F6), so it is set here rather than left to the caller's environment.
    """
    os.environ["DUAL_KIEKER_ITERS"] = str(iters)
    method = feature_method(feature)
    if method:
        os.environ["DUAL_PERF_only"] = method
    else:
        # Better to sweep loudly than to silently profile the wrong thing; the gate will
        # reject the result either way.
        os.environ.pop("DUAL_PERF_only", None)
        log(f"   ⚠️  {feature}: no entry in feature_methods.json — profiling the FULL SWEEP, "
            f"which is F6-invalid for propagation")
    res = dt.mvn_test_interleaved(senior, junior, MVN_ARGS)
    if method:
        measure_replicate(feature, senior, verdict=res[0])
        measure_sweep(feature, senior, junior, iters, verdict=res)
    return res


def replicate_dirname() -> str:
    """Sibling of the current profile dir holding the senior's SECOND profile (S')."""
    return dt._profile_dirname() + "_rep"


def measure_replicate(feature: str, senior: str, verdict: dict = None) -> None:
    """Profile the SENIOR a second time, unchanged, into `<profile dir>_rep/`.

    Every pair in this corpus is senior-vs-junior, where the junior carries a deliberately
    induced inefficiency. That yields 25 positives and ZERO true negatives, so nothing in the
    dataset measures what the detector does when there is nothing to find — and a detector's
    false-positive rate is not optional information when the dataset exists to benchmark
    detectors. Re-profiling the same tree gives a pair that is IDENTICAL BY CONSTRUCTION: no
    diff, no injected anti-pattern, same commit. Any method the comparator flags across it is a
    false positive, and the size of that set is this session's noise floor.

    It also prices the verdicts around it. A pair reporting a 31% local regression means one
    thing if S vs S' is clean and quite another if S vs S' flags four methods on its own — and
    until now a row carried no way to tell those apart (the I4 personalbundles artifact was
    exactly this question, answered only retrospectively and by hand).

    Run immediately after the isolated pass, before the sweep, so S and S' are adjacent in time
    and share machine state as closely as the protocol allows — the same reasoning that makes
    the S/J iterations interleave rather than run in blocks.

    Non-fatal: the S/J measurement is the row's primary artifact and is already on disk.
    Set DUAL_REPLICATE=0 to skip (saves ~45 s/feature and forfeits the noise floor).
    """
    if os.getenv("DUAL_REPLICATE", "1") != "1":
        return
    base = dt._profile_dirname()
    os.environ["DUAL_PROFILE_DIR"] = base + "_rep"    # read before _profile_dirname() is called
    try:
        log(f"   ── replicate pass (senior again, unchanged) → {base}_rep/ "
            f"[noise floor / true-negative sample]")
        dt.mvn_test(senior, MVN_ARGS, kieker=True, known_verdict=verdict)
    except Exception as exc:                                          # noqa: BLE001
        log(f"   ⚠️  {feature}: replicate pass failed ({exc}) — the S/J measurement stands; "
            f"this row will carry no noise floor")
    finally:
        os.environ.pop("DUAL_PROFILE_DIR", None)


def replicate_noise(senior: str, sen_targets: dict) -> dict | None:
    """Compare the senior against its own replicate: the detector's false-positive set.

    Uses the same comparator, thresholds and guards as the real S/J comparison — a noise floor
    measured by a gentler instrument than the one under test would be worthless.

    `exclude_controls=False` here, and the reason is the mirror image of the S/J case. There, a
    control is a source-identical method whose flagging signals contamination rather than code,
    so it is kept out of the headline. Here EVERY method is source-identical, so excluding
    controls would empty the result by construction and report a perfect noise floor for any
    input whatsoever. The whole measurement is the count that exclusion would delete.
    """
    base_dir = os.path.join(senior, dt._profile_dirname())
    rep_dir = os.path.join(senior, replicate_dirname())
    if not (os.path.isdir(base_dir) and os.path.isdir(rep_dir)):
        return None
    try:
        a_times = parse_kieker.parse_dir(base_dir)
        b_times = parse_kieker.parse_dir(rep_dir)
        if not (a_times and b_times):
            return None
        a_ops = {r["operation_signature"]: r for r in parse_kieker.summarise(a_times)}
        b_ops = {r["operation_signature"]: r for r in parse_kieker.summarise(b_times)}
        controls = dt._control_signatures((a_ops, sen_targets), (b_ops, sen_targets))
        cmp = compare_kieker.compare_summaries(
            a_ops, b_ops,
            threshold_pct=float(os.getenv("DUAL_KIEKER_THRESHOLD_PCT", "20")),
            min_delta_ms=float(os.getenv("DUAL_KIEKER_MIN_DELTA_MS", "0.5")),
            min_calls=int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")),
            baseline_raw=a_times, candidate_raw=b_times, control_sigs=controls,
            exclude_controls=False)
    except Exception:                                                 # noqa: BLE001
        return None
    n_fp = cmp["n_regressions"]
    return {
        "pair": "senior vs senior-replicate (identical source)",
        "n_methods_compared": cmp["n_methods_compared"],
        # Flagged across identical code: false positives by construction.
        "n_false_positives": n_fp,
        "false_positives": cmp["regressed"],
        "n_suspect": cmp["n_suspect"],
        "run_offset_pct": cmp["run_offset_pct"],
        "offset_estimated": cmp["offset_estimated"],
        # The headline: did the detector stay silent on a pair with nothing to find?
        "clean": n_fp == 0,
        "note": "true-negative control. Any regression here is noise, not code; a same-session "
                "S/J verdict should not be read as a code signal below this floor.",
    }


def sweep_repeats() -> int:
    """In-session repeat count for the SWEEP pass. Six, against the isolated pass's thirty.

    The sweep exists for two comparisons — trunk_regression (C8) and the run-offset diagnostic —
    and both reach it through `parse_kieker.parse_dir` → `summarise`, which pools every call of a
    signature into one distribution. Neither reads `rep_NN` blocks; the block bootstrap in
    paper_detect.py runs on the ISOLATED profile, which keeps all 30. So the sweep does not need
    30 for the paper's protocol, and 30 is expensive: measured, 2.95 GB per track against 0.20 GB
    at 2 for the SAME 33 operations, i.e. ~140 GB versus ~10 GB across the batch.

    SIX, not two, and the reason is a hard arithmetic floor rather than a taste for margin.
    `estimate_run_offset` ignores any control with fewer than `min_calls=5` samples, and in the
    sweep **each feature query is invoked exactly once per repeat**. At 2 repeats those methods
    carry 2 calls and are all discarded: measured on lift, 32 control signatures fall to 20 after
    the fixture/accessor filter and then to 4 after the call-count filter — one short of
    `min_controls=5`, so the diagnostic reports `applied: False` and the row silently loses its
    contamination check. At 6 repeats those same queries carry 6 calls and ~14 controls survive.

    5 would technically clear the bar, and is rejected deliberately: it puts every one of those
    controls exactly ON the `>= min_calls` boundary, where a single lost sample drops it out.
    6 is the first value with margin. (Constructors, at one call per RUN rather than per repeat,
    stay below the floor at any realistic repeat count — they are not the ones being bought.)"""
    try:
        return max(1, int(os.getenv("DUAL_SWEEP_repeats", "6")))
    except ValueError:
        return 6


def measure_sweep(feature: str, senior: str, junior: str, iters: int,
                  verdict: tuple = None) -> None:
    """A second profiled pass over BOTH tracks with the exerciser unrestricted, into `profile/`.

    Isolation buys F6 validity for the paper's propagation step and costs everything the
    isolated profile no longer contains. Two things depend on what it dropped:

      • trunk_regression (C8) compares the parent senior against the current senior over
        methods they SHARE — i.e. previously-landed feature methods. The isolated profile has
        none of them (verified on lift: 0 of 9 prior-feature methods present, against 9 of 9 in
        the sweep), so the cumulative-drift question degenerates to setup noise.
      • the run-offset CONTAMINATION DIAGNOSTIC needs >= min_controls usable controls. Isolated
        lift yields 2; the sweep yields ~22.

    Both are answered by a sweep of the same pair in the same session — same machine state,
    same drift — which the L1 scoping fix now makes safe: -Dtest=FeaturePerfExerciseTest keeps
    it to ONE warm Kieker session, so the 61-cold-JVM defect that made sweep profiles worthless
    cannot recur. The sweep became valid for C8 exactly when isolation made it invalid for B/C/D.

    Note this profiles BOTH tracks, not seniors only: C8 alone would need just the seniors, but
    a run offset is a candidate/baseline ratio and has no meaning from one side.

    ~2 min per feature. Failure here is NOT fatal — the isolated profile is the batch's primary
    artifact and is already on disk; C8 and the diagnostic degrade to unavailable, which the
    row records rather than hides.
    """
    saved = os.environ.get("DUAL_PERF_only")
    saved_repeats = os.environ.get("DUAL_PERF_repeats")
    os.environ.pop("DUAL_PERF_only", None)          # unrestricted -> _profile_dirname() = profile
    os.environ["DUAL_PERF_repeats"] = str(sweep_repeats())
    try:
        log(f"   ── sweep pass (unrestricted exerciser, repeats={sweep_repeats()}) → profile/ "
            "[for trunk_regression + the offset diagnostic]")
        # The frozen-suite verdict is carried over from the isolated pass: same trees, same
        # session, and re-running the whole suite here only re-derives it (see the
        # `known_verdict` contract in dual_track.mvn_test_interleaved).
        dt.mvn_test_interleaved(senior, junior, MVN_ARGS, known_verdict=verdict)
    except Exception as exc:                                          # noqa: BLE001
        log(f"   ⚠️  {feature}: sweep pass failed ({exc}) — the isolated measurement stands; "
            f"trunk_regression and the offset diagnostic will be unavailable for this row")
    finally:
        if saved is not None:
            os.environ["DUAL_PERF_only"] = saved
        if saved_repeats is None:
            os.environ.pop("DUAL_PERF_repeats", None)
        else:
            os.environ["DUAL_PERF_repeats"] = saved_repeats


def sweep_offset(senior: str, junior: str, sen_targets: dict, jun_targets: dict) -> dict | None:
    """Run-offset estimate from the pair's SWEEP profiles, or None if they are not on disk.

    The isolated profile is too small to estimate a machine-drift offset from — 2 usable
    controls on lift, against `min_controls=5`. That does not change any verdict (the default
    is apply_offset=False, paper-literal), but it removes the DIAGNOSTIC: without it a row
    cannot say whether its session drifted, and a drifting session is precisely when a
    borderline verdict should not be trusted. The sweep pass measures the same two tracks in
    the same session with ~22 usable controls, so it answers the question the isolated data
    cannot. Diagnostic only — never applied as a correction here.

    The control signatures are re-derived from the SWEEP operations, not reused from the
    isolated comparison. Reusing them silently defeats the whole point: the isolated control
    set names methods the isolated run saw, and measured on lift only 2 of them clear
    min_calls in the sweep data, so the estimate gives up exactly as before. Derived from the
    sweep's own operations it finds 13 usable controls and a +10.4% drift.
    """
    sen_sweep = os.path.join(senior, "profile")
    jun_sweep = os.path.join(junior, "profile")
    if not (os.path.isdir(sen_sweep) and os.path.isdir(jun_sweep)):
        return None
    try:
        s_ops = {r["operation_signature"]: r
                 for r in parse_kieker.summarise(parse_kieker.parse_dir(sen_sweep))}
        j_ops = {r["operation_signature"]: r
                 for r in parse_kieker.summarise(parse_kieker.parse_dir(jun_sweep))}
    except Exception:                                                 # noqa: BLE001
        return None
    controls = dt._control_signatures((s_ops, sen_targets), (j_ops, jun_targets))
    off = compare_kieker.estimate_run_offset(
        s_ops, j_ops, controls, int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")))
    if not off.get("applied"):
        # Say so. Returning a bare None here is how a shrunken sweep workload disabled the
        # contamination diagnostic for a whole pilot without anything noticing: C8 kept working,
        # the verdict still looked clean, and the only trace was `offset_estimated: false` deep
        # in the row. The reason string carries the actual arithmetic (how many controls were
        # excluded as fixtures vs dropped for too few calls), which is what makes it fixable.
        log(f"   ⚠️  offset diagnostic unavailable from the sweep: {off.get('reason')} "
            f"[{len(controls)} control signature(s) offered] — this row records no "
            f"contamination check; consider raising DUAL_SWEEP_repeats")
        return None
    return off


def gate_pair_detail(feature: str, senior: str, junior: str) -> dict:
    """Run scripts/measurement_gate.py over a freshly measured pair and return the FULL report.
    Persists every field `measurement_gate.Check` carries into the row via `validity_block`.

    The measurement protocol requires the gate after EVERY pair, not at the end of the batch: a batch
    that discovers on its last day that the harness was stale has already spent the runtime.
    Wiring it here is the difference between the gate being a tool and the gate being in the
    loop — it was standalone until 2026-08-16 (Review audit) and nothing called it.

    UNVERIFIABLE checks (a legacy profile with no v2 meta) do NOT block: a pair measured by
    this very function always writes v2 meta, so an unverifiable claim here means the profile
    did not come from this run at all, which the C5b staleness check reports separately.
    """
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import measurement_gate                                   # noqa: E402
    # gate_pair takes PROFILE dirs, not workspace dirs, and the subdirectory depends on the
    # workload this run used — the same convention the engine writes by.
    pdir = dt._profile_dirname()
    checks, _ok = measurement_gate.gate_pair(os.path.join(senior, pdir),
                                             os.path.join(junior, pdir), feature)
    failed = [c for c in checks if c.status == measurement_gate.FAIL]
    for c in checks:
        if c.status != measurement_gate.PASS:
            log(f"      {c}")
    return {
        "ok": not failed,
        "checks": [{"id": c.cid, "status": c.status, "detail": c.detail} for c in checks],
        "fingerprint": measurement_gate.harness_fingerprint(),
        "checked_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def gate_pair_ok(feature: str, senior: str, junior: str) -> bool:
    """True iff the pair is green. Thin wrapper over `gate_pair_detail` for callers that only
    need the boolean (kept so nothing outside this module has to change)."""
    return gate_pair_detail(feature, senior, junior)["ok"]


def load_profiles(ws: str) -> tuple:
    """({sig: summary}, {sig: [ns,...]}) from a workspace's profile dir.

    Follows the engine's own convention (dual_track._profile_dirname): an isolated run
    (DUAL_PERF_only set, which measure() always sets when the feature has a mapped method)
    writes `profile_only/`, a full sweep writes `profile/`. Reading a hardcoded `profile/`
    here would silently load the PREVIOUS sweep's numbers after an isolated re-measurement.
    """
    times = parse_kieker.parse_dir(os.path.join(ws, dt._profile_dirname()))
    rows = parse_kieker.summarise(times)
    return {r["operation_signature"]: r for r in rows}, times


def profile_iterations(ws: str, profile_dir: str = "profile") -> int:
    """Count iter_NN directories actually on disk. NOT the value the dataset row claims:
    a track that CRASHES mid-suite leaves fewer iterations than were requested, and the
    stale row still reports the requested count. teastore-hist-volatility shipped
    `profile_iterations: 1` for the junior against 5 for the senior — 2,696 samples vs
    338,294, a 125x workload asymmetry — and nothing in the pipeline noticed, because the
    comparator only ever sees pooled means. Counting from disk makes the asymmetry
    checkable (see `check_symmetry`).

    JVM SESSIONS, not replicates: under the in-session protocol this is 1 for every profile,
    and the replicate axis lives in samples_tagged.csv instead (see `_profile_totals`)."""
    d = os.path.join(ws, profile_dir)
    if not os.path.isdir(d):
        return 0
    return sum(1 for n in os.listdir(d)
               if n.startswith("iter_") and os.path.isdir(os.path.join(d, n)))


# Profile dirs a published result is computed from — the values `run_paper_algorithm.py
# --profile-dir` accepts. Only these disqualify a pair when present on one track and not the
# other; any other `profile*` dir is exploratory and is checked for symmetry when it happens
# to exist on both sides.
CONSUMED_PROFILE_DIRS = ("profile", "profile_only")


def profile_dirs_of(ws: str) -> list[str]:
    """Every `profile*` directory in a workspace, `profile` first.

    The paper algorithm no longer reads only `profile`: `run_paper_algorithm.py --profile-dir`
    selects between the full sweep and the `PERF_only` isolated profiles, and step C's
    subsystem denominator changes by more than an order of magnitude between them. A validity
    gate that inspects only `profile` therefore certifies a directory the result was not
    computed from.
    """
    if not os.path.isdir(ws):
        return []
    dirs = sorted(n for n in os.listdir(ws)
                  if n.startswith("profile") and os.path.isdir(os.path.join(ws, n)))
    return sorted(dirs, key=lambda n: (n != "profile", n))


def _profile_totals(ws: str, profile_dir: str) -> tuple[int, int]:
    """(n_replicates, n_calls) for one profile dir, from the DURABLE csv artifacts.

    `samples_tagged.csv`'s `iteration` labels are the replicate axis under either protocol:
    `iter_NN` for the pre-2026-07-30 multi-invocation runs, `rep_NN` for the in-session one
    (parse_kieker.parse_run_tagged rebuckets a single Kieker session into the paper's 30
    blocks). Counting `iter_NN` DIRECTORIES instead — what `profile_iterations` does — reports
    1 for every in-session profile, which would make the replicate check vacuous on exactly
    the profiles the paper algorithm now reads.

    Falls back to raw .dat only when the csvs are absent: .dat is superseded here and has been
    pruned under disk pressure before, which would otherwise read as 0 calls and fire a false
    asymmetry alarm.
    """
    d = os.path.join(ws, profile_dir)
    summ, tagged = os.path.join(d, "summary.csv"), os.path.join(d, "samples_tagged.csv")

    n_calls = None
    if os.path.isfile(summ):                      # small file; authoritative call totals
        with open(summ, newline="") as fh:
            n_calls = sum(int(float(r["n_calls"] or 0)) for r in csv.DictReader(fh))

    if os.path.isfile(tagged):
        # csv.reader over one indexed column, not DictReader: these files reach 5.4M rows
        # (reorder/profile_paper) and building a dict per row makes a validity check that
        # should be incidental cost minutes of wall time.
        with open(tagged, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd, None)
            if header and "iteration" in header:
                i = header.index("iteration")
                labels, n = set(), 0
                for row in rd:
                    labels.add(row[i])
                    n += 1
                return len(labels), (n_calls if n_calls is not None else n)

    if n_calls is not None:
        return profile_iterations(ws, profile_dir), n_calls
    return (profile_iterations(ws, profile_dir),
            sum(r["n_calls"] for r in
                (parse_kieker.summarise(parse_kieker.parse_dir(d)) or [])))


def check_symmetry(feature: str, senior: str, junior: str,
                   profile_dirs: list[str] = None) -> tuple[str | None, dict]:
    """Check every profile dir the two tracks share. Returns (blocking_reason, volume_report).

    TWO DISTINCT SIGNALS, deliberately not merged (they were, until 2026-08-07):

      * REPLICATE asymmetry — the tracks were profiled a different number of times. This is
        always a measurement fault: control-offset normalization divides out a multiplicative
        offset between two runs of THE SAME workload and cannot rescue a comparison where one
        side simply ran less (audit BLOCKER 0). teastore-hist-volatility is the case that
        motivated the gate: its junior crashed out at 1 iteration against the senior's 5.
        This is the BLOCKING signal.

      * CALL-VOLUME asymmetry at EQUAL replicates — both tracks ran the same workload the same
        number of times, and the junior still executed a different number of instrumented
        calls. That is not automatically invalid, because it is exactly what an N+1 regression
        looks like: teastore-hist-lift's junior adds `CategoryRepository.resolveCategoryName`
        (9,240 calls against the senior's 0) plus 9,208 extra `getEntityClass` calls, giving a
        2.36x call ratio in `profile_only` that IS the regression under test. Blocking on it
        would discard the genuine positives this dataset exists to contain. It is REPORTED, not
        blocking, and the analyst decides — a driver fault and a real N+1 are indistinguishable
        from call counts alone.
    """
    if profile_dirs is None:
        # UNION, not intersection: a profile dir the algorithm CONSUMES but that exists on only
        # one side is itself disqualifying, and intersecting would silently skip it. Not
        # hypothetical — teastore-hist-volatility's senior has no profile dir at all, and an
        # intersection returns an empty check list and reports the pair as clean.
        profile_dirs = sorted(set(profile_dirs_of(senior)) | set(profile_dirs_of(junior)),
                              key=lambda n: (n != "profile", n))
    blocking, volumes = [], {}
    if not profile_dirs:
        return (f"no profile data in either workspace for {feature} — nothing to compare",
                volumes)
    for p in profile_dirs:
        missing = [w for w, ws in (("senior", senior), ("junior", junior))
                   if not os.path.isdir(os.path.join(ws, p))]
        if missing:
            # One-sided absence blocks only for the dirs a result is actually computed from
            # (`run_paper_algorithm.py --profile-dir`). Exploratory profiles kept on one track
            # — teastore-hist-churn's senior-only `profile_largescale` — are not a validity
            # defect and must not disqualify an otherwise sound pair.
            if p in CONSUMED_PROFILE_DIRS:
                blocking.append(f"{p}: absent from the {' and '.join(missing)} workspace — "
                                f"one track has profile data the other does not")
            volumes[p] = {"missing": missing, "consumed": p in CONSUMED_PROFILE_DIRS}
            continue
        s_it, s_n = _profile_totals(senior, p)
        j_it, j_n = _profile_totals(junior, p)
        ratio = (j_n / s_n) if s_n else 0.0
        volumes[p] = {"senior_replicates": s_it, "junior_replicates": j_it,
                      "senior_calls": s_n, "junior_calls": j_n,
                      "call_ratio": round(ratio, 4),
                      "skewed": bool(s_n and not (0.5 <= ratio <= 2.0))}
        if s_it != j_it or not s_n or not j_n:
            blocking.append(
                f"{p}: replicates senior={s_it} junior={j_it}, calls senior={s_n} "
                f"junior={j_n} — the two tracks did not run the same number of times, so "
                f"the comparison is not valid regardless of normalization")
    return ("; ".join(blocking) or None), volumes


def recompute(feature: str, seq: int, senior: str, junior: str, baseline: str,
              senior_res: dict, junior_res: dict, write_csv: bool = True) -> dict:
    """Rebuild every perf-derived field of the row from the profiles now on disk."""
    sen_ops, sen_raw = load_profiles(senior)
    jun_ops, jun_raw = load_profiles(junior)
    senior_res["operations"], senior_res["raw_times"] = sen_ops, sen_raw
    junior_res["operations"], junior_res["raw_times"] = jun_ops, jun_raw

    sen_targets = dt._feature_targets(senior, baseline)
    jun_targets = dt._feature_targets(junior, baseline)
    controls = dt._control_signatures((sen_ops, sen_targets), (jun_ops, jun_targets))

    cmp = compare_kieker.compare_summaries(
        sen_ops, jun_ops,
        threshold_pct=float(os.getenv("DUAL_KIEKER_THRESHOLD_PCT", "20")),
        min_delta_ms=float(os.getenv("DUAL_KIEKER_MIN_DELTA_MS", "0.5")),
        min_calls=int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")),
        baseline_raw=sen_raw, candidate_raw=jun_raw, control_sigs=controls,
        offset_override=sweep_offset(senior, junior, sen_targets, jun_targets))
    # Guarded by write_csv so --dry-run really is read-only. It was not: this write sat
    # ahead of the --dry-run check in main(), so a "report only" run still rewrote every
    # compare.csv — which is how the repo reached a state where the CSVs carried
    # normalized deltas and perf_dataset_teastore_history.jsonl still carried the raw
    # pre-normalization ones. The two must not be able to diverge.
    if write_csv:
        compare_kieker.write_compare_csv(cmp, os.path.join(junior, "compare.csv"))

    local = {k: cmp[k] for k in
             ("threshold_pct", "min_delta_ms", "min_calls",
              "run_offset_ratio", "run_offset_pct", "normalized",
              "offset_estimated", "n_control_methods", "normalization_note",
              "n_methods_compared", "n_regressions", "n_suspect", "regressed", "suspect",
              # controls that tripped the threshold: not a code signal, a contamination
              # diagnostic. Non-empty => this session's drift survived into the report.
              "n_control_flagged", "control_flagged")}
    sen_ex = dt._feature_exercised(senior_res, sen_targets)
    jun_ex = dt._feature_exercised(junior_res, jun_targets)
    fe = (sen_ex and jun_ex) if (sen_ex is not None and jun_ex is not None) else None
    verdict = dt.equivalence_verdict(senior_res, junior_res)
    return {"local": local, "sen_ex": sen_ex, "jun_ex": jun_ex, "feature_exercised": fe,
            "verdict": verdict, "senior_res": senior_res, "junior_res": junior_res,
            "sen_targets": sen_targets, "n_controls": len(controls),
            # None when the replicate pass was skipped or failed — absent, not clean.
            "replicate": replicate_noise(senior, sen_targets)}


def trunk_regression(seq: int, manifest: dict, feature: str, senior: str,
                     senior_res: dict, sen_targets: dict) -> dict:
    """Cumulative drift vs the parent's senior — now that BOTH profiles come from this same
    re-measurement session, plus control normalization, this is finally a code signal.

    Reads the SWEEP profile (`profile/`) on both sides, never the isolated one. This question is
    about methods the two trunk states SHARE — the features landed before this one — and an
    isolated profile contains none of them: lift's `profile_only` holds 0 of the 9 prior-feature
    methods, the sweep holds 9 of 9. Pointed at isolated data the comparison does not fail, it
    silently degenerates to setup noise, which is worse. `measure_sweep` produces this artifact
    in the same session as the isolated one."""
    prev = [t for t in manifest["test_points"] if t["sequence"] == seq - 1]
    if not prev:
        return None
    parent = os.path.join(ROOT, prev[0]["senior_out_dir"])
    prev_times = parse_kieker.parse_dir(os.path.join(parent, "profile"))
    cur_times = parse_kieker.parse_dir(os.path.join(senior, "profile"))
    if not prev_times or not cur_times:
        return None
    prev_ops = {r["operation_signature"]: r for r in parse_kieker.summarise(prev_times)}
    cur_ops = {r["operation_signature"]: r for r in parse_kieker.summarise(cur_times)}
    controls = dt._control_signatures((prev_ops, sen_targets), (cur_ops, sen_targets))
    cmp = compare_kieker.compare_summaries(
        prev_ops, cur_ops,
        threshold_pct=float(os.getenv("DUAL_KIEKER_THRESHOLD_PCT", "20")),
        min_delta_ms=float(os.getenv("DUAL_KIEKER_MIN_DELTA_MS", "0.5")),
        min_calls=int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")),
        baseline_raw=prev_times, candidate_raw=cur_times,
        control_sigs=controls,
        # see dual_track._trunk_regression: pre-existing methods getting slower is the signal
        exclude_controls=False)
    block = {k: cmp[k] for k in
             ("threshold_pct", "min_delta_ms", "min_calls",
              "run_offset_ratio", "run_offset_pct", "normalized",
              "offset_estimated", "n_control_methods", "normalization_note",
              "n_methods_compared", "n_regressions", "n_suspect", "regressed", "suspect",
              # controls that tripped the threshold: not a code signal, a contamination
              # diagnostic. Non-empty => this session's drift survived into the report.
              "n_control_flagged", "control_flagged")}
    block["parent_feature_id"] = prev[0]["feature_id"]
    block["new_methods"] = [r["operation_signature"] for r in cmp["rows"]
                            if r["status"] == "NEW"]
    return block


# The junior brief was rewritten after F17 (turnover), which changed program SEMANTICS rather
# than cost because the old brief let a junior treat an impure read like `LocalDateTime.now()`
# as a loop-invariant. Features 1–18 and 19–25 were therefore generated under different
# instructions. Left undisclosed that is a confound sitting inside the labels; recorded as a
# column it is a stratification variable a consumer can control for, documented in the dataset specification.
_BRIEF_V2_FROM_SEQ = 19


def junior_brief_protocol(seq: int) -> str | None:
    """Which junior-brief revision generated this feature's inefficiency, from its chain seq."""
    if not isinstance(seq, int) or seq < 1:
        return None
    return ("v2-impure-invariants" if seq >= _BRIEF_V2_FROM_SEQ
            else "v1-pure-invariants")


def validity_block(res: dict, tr: dict, mode: str) -> dict:
    """An explicit, machine-readable statement of what this row's measurement can support.

    The corpus rule is "silently missing is never acceptable; explicitly excluded always
    is", and until now the dataset broke it in the one place it mattered most: a row that was
    never validly measured looked exactly like a row measured cleanly with no regression found.
    Every downstream consumer then had to re-derive validity from a scatter of nullable fields,
    and the scorer's ground-truth bug came from doing that wrong once.

    Four states, deliberately not three — `unmeasured` must never collapse into "no regression":

      unmeasured  the feature code was never observed running (feature_exercised null/False).
                  Null is the common shape: a track failed to COMPILE, so its profiler recorded
                  nothing. There is no signal here of any kind, positive or negative.
      invalid     measured, but the comparison itself does not hold — the tracks did different
                  amounts of work, which no normalization can repair.
      degraded    a real comparison, with a named caveat attached (contaminated controls, a
                  noise floor that is not clean, or a missing cross-check). Usable WITH the
                  caveat; the caveats are listed so a consumer can filter on them.
      valid       a clean comparison with its cross-checks present.
    """
    reasons = []
    fe = res.get("feature_exercised")
    local = res.get("local") or {}
    rep = res.get("replicate")

    if res.get("asymmetry"):
        status = "invalid"
        reasons.append(f"workload asymmetry: {res['asymmetry']}")
    elif fe is not True:
        status = "unmeasured"
        reasons.append("feature code never observed executing "
                       f"(feature_exercised={fe!r}) — this is NOT evidence of no regression")
    else:
        status = "valid"
        # Controls are source-identical by construction, so one tripping the threshold means
        # this session's drift reached the report — the verdict is about the machine, not code.
        if local.get("n_control_flagged"):
            status = "degraded"
            reasons.append(f"{local['n_control_flagged']} source-identical control method(s) "
                           f"flagged — session drift reached the comparison")
        # No offset estimate => nothing checked whether this session drifted. The verdict is not
        # wrong, but it is unverified in a specific way the row should not hide: a shrunken sweep
        # starved this check through the whole first pilot while every other field looked clean.
        if not local.get("offset_estimated"):
            status = "degraded"
            reasons.append("no run-offset estimate — too few usable control methods, so this "
                           "row carries no check on whether the session drifted")
        if rep is None:
            status = "degraded"
            reasons.append("no replicate (S') pass — this row has no measured noise floor, so "
                           "its regression sizes cannot be priced against same-session noise")
        elif not rep.get("clean"):
            status = "degraded"
            reasons.append(f"noise floor not clean: {rep['n_false_positives']} false "
                           f"positive(s) on identical source — read local regressions at or "
                           f"below this size as noise")
        if tr is None:
            status = "degraded"
            reasons.append("no trunk regression (C8) comparison available for this row")
    # `gate_detail`, threaded in via `res["gate_detail"]`, carries the full
    # `measurement_gate.gate_pair` report when this run actually executed the gate.
    gate_detail = res.get("gate_detail")
    if gate_detail:
        measurement_gate = {"status": "green" if gate_detail["ok"] else "red",
                            "checks": gate_detail["checks"],
                            "fingerprint": gate_detail["fingerprint"],
                            "checked_at": gate_detail["checked_at"]}
    else:
        measurement_gate = "green"
    return {"status": status, "reasons": reasons,
            "measured_by": mode,
            "measurement_gate": measurement_gate}


def apply_to_row(row: dict, res: dict, tr: dict, mode: str, iters: int,
                 senior: str, junior: str, seq: int = None) -> dict:
    """Write the recomputed measurement back into a dataset row, in place."""
    without = lambda r: {k: v for k, v in r.items() if k not in ("operations", "raw_times")}
    row["senior"].update(without(res["senior_res"]))
    row["senior"]["perf"] = dt._perf_from_ops(res["senior_res"])
    row["senior"]["feature_exercised"] = res["sen_ex"]
    row["junior"].update(without(res["junior_res"]))
    row["junior"]["perf"] = dt._perf_from_ops(res["junior_res"])
    row["junior"]["feature_exercised"] = res["jun_ex"]
    row["equivalence"] = res["verdict"]
    row["local_regression"] = res["local"]
    row["trunk_regression"] = tr
    row["feature_exercised"] = res["feature_exercised"]
    # S vs S' on identical source: the detector's false-positive set for THIS session, and the
    # corpus's only true-negative sample. None = the pass was skipped or failed, which is not
    # the same as a clean floor and must not read as one.
    row["replicate_noise"] = res.get("replicate")
    if seq is not None:
        row["junior_brief_protocol"] = junior_brief_protocol(seq)
    # Deployability is re-derived here for the same reason the perf fields are: the gate
    # postdates every row in the file, so without this the backfill would leave an
    # undeployable pair sitting at admitted=True.
    _dep = {t: dt._deployability(o) for t, o in (("senior", senior), ("junior", junior))}
    row["deployability"] = {"deployable": all(d["deployable"] for d in _dep.values()), **_dep}
    row["admitted"] = dt._admit_row(
        res["verdict"]["equivalent"], row.get("status") == "infra_abort",
        res["feature_exercised"],
        os.getenv("DUAL_REQUIRE_FEATURE_EXERCISED", "1") == "1",
        row["deployability"]["deployable"])
    # The system oracle runs out-of-band (scripts/run_oracle.sh), long after the row was
    # written, and drops oracle_stats.json into the junior workspace. dual_track already
    # knows how to merge it; it simply has not run since the oracle results were collected,
    # which is why system_verdict was null in all 21 rows despite four results existing.
    row.update(dt._system_oracle_row(junior))
    row["run_id"] = f"{row.get('feature_id')}@{row.get('started_at')}"
    row["measurement"] = {
        "remeasured_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "profile_iterations": iters,
        "senior_profile_iterations": res["senior_res"].get("profile_iterations"),
        "junior_profile_iterations": res["junior_res"].get("profile_iterations"),
        # Blocking: the tracks were profiled a different number of times, across ANY profile
        # dir they share (not just `profile` — the paper algorithm also reads `profile_only`).
        "workload_asymmetry": res.get("asymmetry"),
        # Reported only. Call-count skew at equal replicates is indistinguishable from the
        # regression under test, so it never disqualifies a pair. See check_symmetry.
        "call_volumes": res.get("call_volumes"),
        "run_offset_normalized": res["local"]["normalized"],
        "run_offset_pct": res["local"]["run_offset_pct"],
        "n_control_methods": res["local"]["n_control_methods"],
        "llm_rerun": False,
        "note": "perf fields re-derived from the on-disk workspaces; implementation code "
                "unchanged (see docs/PIPELINE-AUDIT-2026-07-26.md)",
    }
    row["measurement"]["sweep_repeats"] = sweep_repeats()
    row["measurement"]["replicate"] = (
        "absent" if res.get("replicate") is None else "measured")
    # Which classes did NOT get a vote on equivalence, and the cap that bounded the run. Recorded
    # because an exclusion a consumer cannot see is indistinguishable from a suite that never had
    # those tests: `equivalent: true` means "the tests that ran agreed", and this says which ran.
    row["equivalence"]["excluded_classes"] = os.getenv(
        "DUAL_VERDICT_EXCLUDE", dt._VERDICT_EXCLUDE_DEFAULT) or None
    row["equivalence"]["fork_timeout_s"] = os.getenv(
        "DUAL_VERDICT_TIMEOUT", dt._VERDICT_TIMEOUT_DEFAULT) or None
    row["equivalence"]["excluded_reason"] = (
        "large-scale PERFORMANCE fixtures are excluded from the FUNCTIONAL verdict; the "
        "performance question they pose is answered by the Kieker profiles instead. Normal-scale "
        "tests still decide correctness.")
    if res.get("asymmetry"):
        row["measurement"]["note"] += (
            " | WORKLOAD ASYMMETRY: the two tracks did different amounts of work, so the "
            "local comparison in this row is NOT valid — normalization cannot correct it")
    # Last: it summarises the fields written above, so it must not be computed before them.
    row["validity"] = validity_block(res, tr, mode)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="perf_dataset_teastore_history.jsonl")
    ap.add_argument("--manifest", default="chain_manifest.json")
    # JVM SESSIONS, not repetitions. 1 = Liao et al.'s protocol (30 in-session repeats via
    # -Dteastore.perf.repeats), which is also what parse_run_tagged needs to rebucket into the
    # paper's 30 bootstrap blocks — with >1 iteration label it cannot, and step D degrades to
    # that many coarse blocks instead. See run_teastore_feature.sh for the full rationale.
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--features", nargs="*", default=None)
    ap.add_argument("--recompute-only", action="store_true",
                    help="skip mvn; normalize the profiles already on disk")
    ap.add_argument("--dry-run", action="store_true", help="report, do not write the dataset")
    args = ap.parse_args()

    os.chdir(ROOT)
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk")
    manifest = json.load(open(args.manifest))
    points = sorted(manifest["test_points"], key=lambda t: t["sequence"])
    if args.features:
        points = [t for t in points if t["feature_id"] in args.features]

    ds_rows = [json.loads(l) for l in open(args.dataset) if l.strip()]
    last_idx = {}
    for i, r in enumerate(ds_rows):
        last_idx[r.get("feature_id")] = i

    mode = "recompute-only (post-hoc normalization)" if args.recompute_only \
        else f"interleaved re-measurement ({args.iters} iterations)"
    log(f"\n{'='*72}\n  RE-MEASUREMENT — {mode}\n"
        f"  {len(points)} feature(s), dataset={args.dataset}\n{'='*72}")

    tmp = tempfile.mkdtemp(prefix="remeasure_")
    results, failures, asymmetric, unrecoverable = [], [], [], []
    gate_failed = []
    try:
        for tp in points:
            seq, fid = tp["sequence"], tp["feature_id"]
            senior = os.path.join(ROOT, tp["senior_out_dir"])
            junior = senior.replace("-senior", "-junior")
            log(f"\n######## [{seq}/{len(points)}] {fid} ########")
            if not os.path.isdir(junior):
                log(f"  ⏭️  no junior workspace ({junior}) — skipping")
                failures.append((fid, "no junior workspace"))
                continue

            # Never overwrite a real measurement with an empty one. `--recompute-only`
            # derives everything from profile/ on disk, so a feature whose profile data is
            # GONE (teastore-hist-dupes: deleted after its re-measurement attempt wrote 30 GB
            # of Kieker records and had to be killed — audit BLOCKER 0) would otherwise be
            # rewritten with 0 controls, normalized=False and feature_exercised=null,
            # destroying the numbers that are still in the row.
            if args.recompute_only and not (profile_iterations(senior)
                                            and profile_iterations(junior)):
                log(f"  ⏭️  no profile data on disk (senior={profile_iterations(senior)} "
                    f"junior={profile_iterations(junior)} iterations) — row left untouched; "
                    f"re-measure with a full run to recover it")
                failures.append((fid, "no profile data on disk"))
                unrecoverable.append(fid)
                continue

            archive_old(fid, senior, junior)
            baseline = baseline_for(seq, manifest, tmp)

            gate_detail = None  # recompute-only never re-runs the gate; validity_block says so
            if args.recompute_only:
                sres = {k: ds_rows[last_idx[fid]]["senior"].get(k) for k in
                        ("build_success", "mvn_rc", "tests_run", "failures", "errors",
                         "skipped", "tests_pass")}
                jres = {k: ds_rows[last_idx[fid]]["junior"].get(k) for k in
                        ("build_success", "mvn_rc", "tests_run", "failures", "errors",
                         "skipped", "tests_pass")}
                # profile_iterations comes from DISK, never from the stale row — see
                # profile_iterations() for why the row's own number cannot be trusted.
                sres["profile_iterations"] = profile_iterations(senior)
                jres["profile_iterations"] = profile_iterations(junior)
            else:
                sres, jres = measure(fid, senior, junior, args.iters)
                # ── gate the pair NOW, not at the end of the batch (measurement gate protocol) ──
                # A red gate means the measurement is not what the protocol claims (wrong
                # call count, stale harness, 61 cold JVMs instead of one in-session run).
                # Writing that row would put an unusable measurement into the dataset with
                # nothing marking it as such — the F23-F25 defect, which is exactly how three
                # admitted rows ended up carrying invalid local timings. Skip the row, keep
                # the batch going, and report at the end.
                gate_detail = gate_pair_detail(fid, senior, junior)
                if not gate_detail["ok"]:
                    log(f"  ⛔ {fid}: MEASUREMENT GATE RED — row not written, "
                        f"re-measure this pair (see the check list above)")
                    gate_failed.append(fid)
                    continue

            asym, volumes = check_symmetry(fid, senior, junior)
            if asym:
                log(f"  ⚠️  ASYMMETRIC WORKLOAD — {asym}")
                asymmetric.append((fid, asym))
            for p, v in volumes.items():
                if v.get("skewed"):  # absent on the one-sided-profile entries built above
                    log(f"  ℹ️  call volume {p}: senior={v['senior_calls']} "
                        f"junior={v['junior_calls']} (ratio {v['call_ratio']}) at equal "
                        f"replicates — reported, NOT disqualifying: this is also what an N+1 "
                        f"regression looks like")

            res = recompute(fid, seq, senior, junior, baseline, sres, jres,
                            write_csv=not args.dry_run)
            res["asymmetry"] = asym
            res["call_volumes"] = volumes
            res["gate_detail"] = gate_detail
            tr = trunk_regression(seq, manifest, fid, senior, res["senior_res"],
                                  res["sen_targets"])
            lr = res["local"]
            log(f"  offset {lr['run_offset_pct']:+.1f}% over {lr['n_control_methods']} controls "
                f"(normalized={lr['normalized']}) → {lr['n_regressions']} regression(s), "
                f"{lr['n_suspect']} suspect | exercised s={res['sen_ex']} j={res['jun_ex']}")
            if lr["regressed"]:
                for s in lr["regressed"]:
                    log(f"      ↳ REGRESSION {s.split('(')[0].rsplit('.', 1)[-1]}")
            results.append((fid, seq, res, tr, senior, junior))

        if args.dry_run:
            log("\n(--dry-run: dataset not written, no compare.csv rewritten)")
            return 0

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = f"{args.dataset}.bak-{stamp}"
        shutil.copy2(args.dataset, backup)
        log(f"\n  backup: {backup}")

        touched = {fid for fid, _s, _r, _t, _sw, _jw in results}
        for fid in touched:
            for i, r in enumerate(ds_rows):
                if r.get("feature_id") == fid and i != last_idx[fid]:
                    r["superseded"] = True
                    r.setdefault("superseded_reason",
                                 "earlier attempt; the workspaces on disk hold the "
                                 "implementation measured in the row that supersedes this one")
        for fid, seq, res, tr, sw, jw in results:
            apply_to_row(ds_rows[last_idx[fid]], res, tr, mode, args.iters, sw, jw, seq=seq)

        # Rows whose junior never COMPILED (feature_exercised null, not False) carry no
        # measurement at all and were never cleaned: lib_trunk_chain's _prune_unexercised_row
        # only fires immediately before a FRESH run of that feature id, and these features
        # are now "done", so it can never reach them (HIGH 7). Drop them here, where the
        # dataset is already being rewritten under a backup.
        pruned = [r for r in ds_rows
                  if r.get("feature_exercised") is None and r.get("admitted") is not True]
        for r in pruned:
            log(f"  🧹 dropping unmeasured row {r.get('feature_id')}@{r.get('started_at')} "
                f"(feature_exercised=null, junior never compiled)")
        ds_rows = [r for r in ds_rows if r not in pruned]
        for r in ds_rows:
            r.setdefault("run_id", f"{r.get('feature_id')}@{r.get('started_at')}")

        # A row that could NOT be re-derived still carries its pre-normalization numbers, and
        # unlike a superseded row nothing else in the file replaces it — so without an explicit
        # marker it is indistinguishable from a freshly recomputed one. Flag it instead.
        for fid in unrecoverable:
            for r in ds_rows:
                if r.get("feature_id") == fid and not r.get("superseded"):
                    r["measurement_stale"] = {
                        "reason": "profile data absent from the workspace; local_regression and "
                                  "trunk_regression here predate run-offset normalization and "
                                  "are drift-contaminated (docs/PIPELINE-AUDIT-2026-07-26.md "
                                  "BLOCKER 1/3)",
                        "recover_with": "scripts/remeasure_chain.py --features "
                                        f"{fid}",
                    }

        tmp_ds = args.dataset + ".tmp"
        with open(tmp_ds, "w") as fh:
            for r in ds_rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp_ds, args.dataset)
        log(f"  dataset updated: {args.dataset} ({len(results)} row(s) re-measured)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    log(f"\n{'='*72}")
    tot = sum(r[2]["local"]["n_regressions"] for r in results)
    sus = sum(r[2]["local"]["n_suspect"] for r in results)
    norm = sum(1 for r in results if r[2]["local"]["normalized"])
    log(f"  {len(results)} feature(s) re-measured | normalized: {norm}/{len(results)}")
    log(f"  TOTAL regressions: {tot}   suspect: {sus}")
    if failures:
        log(f"  skipped: {failures}")
    if asymmetric:
        log(f"\n  ⚠️  {len(asymmetric)} feature(s) with an INVALID (asymmetric) comparison — "
            f"their local_regression must not be read as a code signal:")
        for fid, why in asymmetric:
            log(f"      {fid}: {why}")
    if gate_failed:
        log(f"\n  ⛔ {len(gate_failed)} feature(s) FAILED THE MEASUREMENT GATE and were NOT "
            f"written to the dataset — re-measure these:")
        for fid in gate_failed:
            log(f"      {fid}")
    log(f"{'='*72}")
    # A gate-red pair is a failed batch, not a partial success: exiting 0 here is what let
    # three admitted rows carry invalid local timings (F23-F25).
    return 1 if gate_failed else 0


if __name__ == "__main__":
    sys.exit(main())
