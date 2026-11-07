#!/usr/bin/env python3
"""
measurement_gate.py — refuse to accept a profile that did not measure what it claims to.

Run this after EVERY pair, not at the end of a batch. Five separate incidents in this project's
history were a correct fix that the dataset silently never picked up; each of them is caught by
one of the checks below, and each cost days because the batch reported success throughout.

WHAT THIS DOES NOT CHECK, AND WHY
---------------------------------
Do NOT gate on `kieker.map` mentioning the exerciser. That check is invalid:

  * `kieker.map` holds only the two record-type mappings ($0=KiekerMetadataRecord,
    $1=OperationExecutionRecord). Application class names never appear in it.
  * `profiling/aop.xml` weave-excludes `..*Test`, so the exerciser is absent from the `.dat`
    files too, BY DESIGN — weaving around-advice into a test method generates bad bytecode.
  * Measured 2026-08-15: 0 of 4,424 `kieker.map` files mention it, INCLUDING `lift` and
    `reorder`, whose `profile_only` runs are the two known-good profiles in the corpus.

A gate that fails on good runs gets disabled by the third day. The checks below are the
implementable form of the same question, and lift/reorder pass them.

THE CHECKS
----------
C1  feature method present in `summary.csv`      — the workload reached the changed code at all
C2  `n_calls == repeats` on the feature method   — the exerciser drove it exactly once per
                                                   repeat. Tighter than `% repeats == 0`, which
                                                   also passes at 60 calls (setup + repeats, or
                                                   driven twice) and so admits a workload that is
                                                   not the protocol.
C3  `meta.json.repeats` == expected              — the paper-literal in-session protocol was
                                                   REQUESTED (see C7 for whether it happened)
C4  `meta.json.only` == the feature method       — F6: the ISOLATED workload, not the 25-query
                                                   sweep (which leaves the feature under test at
                                                   0.238% of station demand)
C5  harness fingerprint matches current source   — the profile is not older than the last
                                                   measurement-code change
C6  (pair only) both tracks agree on the         — F7: a pair measured half-old/half-new is
    feature-method call count                      worse than no measurement
C7  exactly ONE Kieker session in the profile    — the run really was one warm JVM

WHY C7 EXISTS, AND WHY C3 IS NOT ENOUGH. `meta.json` is written BY the harness: `repeats: 30`
is a claim the harness makes about itself, and it stayed true-looking while being false. The
persistence pom inherits `<reuseForks>false</reuseForks>`, so profiling the whole suite forks a
fresh JVM per test class and every fork writes its own Kieker session into the same output dir.
Measured 2026-08-15: the three `profile/` sweeps hold **61, 63 and 59 sessions** while their
`meta.json` all say `{"repeats": 30, "protocol": "in-session"}`. Those profiles are 60 cold JVMs,
which breaks `parse_run_tagged`'s block bootstrap (it rebuckets the pool into 30 fake blocks) and
`build_call_graph` (trace ids are unique per JVM only, so merging invents edges). C7 counts the
`$0` metadata records on disk instead of believing the metadata file — one session, one `$0`.
The clean `profile_only/` pairs have exactly 1.

KNOWN GAP THIS GATE DOES NOT CLOSE: rep_00 is not exchangeable with rep_01..29. All setup
records precede the first query, and the first repeat is JIT-cold — measured on lift, the feature
method's rep_00 mean is 7.73 ms against 2.19 ms for rep_01..29 (3.5x; junior 52.17 vs 16.72).
Both tracks are contaminated so the demand-scale ratio partly cancels, but not exactly for
asymmetric workloads. Handle it in the bootstrap, not here.

LEGACY PROFILES. `meta.json` carried only `repeats` and `protocol` before 2026-08-15, so C4 and
C5 are UNVERIFIABLE for anything profiled earlier — including lift/reorder. Those report as
UNVERIFIABLE, not PASS, and the gate still fails unless `--allow-legacy` is passed explicitly.
Silence about an uncheckable claim is the failure mode this whole file exists to prevent.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURE_METHODS = os.path.join(REPO, "scripts", "feature_methods.json")
EXPECTED_REPEATS = 30

# Files whose content defines what a measurement means. A profile taken before any of these
# changed is stale by definition, whatever its timestamp says.
HARNESS_FILES = [
    os.path.join(REPO, "profiling", "aop.xml"),
    os.path.join(REPO, "scripts", "kieker_profile.sh"),
]
_EXERCISER_PATH = os.environ.get(
    "PERF_EXERCISER_FILE",
    os.path.join(REPO, "teastore-dataset", "teastore-trunk", "services",
                 "tools.descartes.teastore.persistence", "src", "test", "java", "tools",
                 "descartes", "teastore", "persistence", "FeaturePerfExerciseTest.java"),
)
if os.path.isfile(_EXERCISER_PATH):
    HARNESS_FILES.append(_EXERCISER_PATH)

PASS, FAIL, UNVERIFIABLE = "PASS", "FAIL", "UNVERIFIABLE"


class Check:
    def __init__(self, cid, status, detail):
        self.cid, self.status, self.detail = cid, status, detail

    def __repr__(self):
        return f"{self.status:<12} {self.cid}  {self.detail}"


def _stamp(mtime):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))


def harness_fingerprint():
    """sha256 over the files that define what a measurement means."""
    h = hashlib.sha256()
    for p in HARNESS_FILES:
        h.update(os.path.basename(p).encode())
        with open(p, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def load_feature_methods(mapping_file=None):
    path = mapping_file or os.environ.get("FEATURE_METHODS_FILE", FEATURE_METHODS)
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh).get("features", {})


def feature_method_for(feature_id, method=None, mapping_file=None):
    """Accepts an explicit method name, a full feature_id, or the bare suffix."""
    if method:
        return method
    env_method = os.environ.get("PERF_only") or os.environ.get("DUAL_PERF_only")
    if env_method:
        return env_method
    feats = load_feature_methods(mapping_file)
    if feature_id in feats:
        return feats[feature_id]["method"]
    key = feature_id if feature_id.startswith("teastore-hist-") else "teastore-hist-" + feature_id
    if key in feats:
        return feats[key]["method"]
    raise KeyError(
        f"{feature_id} is not in {mapping_file or FEATURE_METHODS}; pass --method <methodName> "
        f"or set PERF_only / FEATURE_METHODS_FILE"
    )


def count_sessions(profile_dir):
    """Number of Kieker sessions in a profile, counted from the data rather than the metadata.

    One session writes exactly one `$0` KiekerMetadataRecord. Counting session DIRECTORIES is
    equivalent and far cheaper than scanning ~500 MB of `.dat`, so directories are counted when
    present and the `$0` records are only read as a fallback. Returns None when the raw output
    has already been pruned (parse-then-delete), in which case the check is UNVERIFIABLE rather
    than passed — an absent artifact is not evidence of a good run.
    """
    if not os.path.isdir(profile_dir):
        return None
    iters = sorted(d for d in os.listdir(profile_dir) if d.startswith("iter_"))
    sessions = 0
    seen_raw = False
    for it in iters:
        it_path = os.path.join(profile_dir, it)
        if not os.path.isdir(it_path):
            continue
        kdirs = [d for d in os.listdir(it_path) if d.startswith("kieker-")]
        if kdirs:
            seen_raw = True
            sessions += len(kdirs)
    return sessions if seen_raw else None


def _summary_row(profile_dir, method):
    path = os.path.join(profile_dir, "summary.csv")
    if not os.path.exists(path):
        return None, f"no summary.csv in {profile_dir}"
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if method in row.get("operation_signature", ""):
                return row, None
    return None, f"{method} absent from summary.csv"


def gate_profile(profile_dir, feature_id, repeats=EXPECTED_REPEATS, fingerprint=None, method=None):
    """Run C1-C5 against one profile directory. Returns a list of Check."""
    method = feature_method_for(feature_id, method=method)
    fingerprint = fingerprint or harness_fingerprint()
    checks = []

    meta_path = os.path.join(profile_dir, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)
    legacy = "meta_version" not in meta

    # C1 — did the workload reach the changed code?
    row, err = _summary_row(profile_dir, method)
    if row is None:
        checks.append(Check("C1 feature-method-present", FAIL, err))
        # Everything downstream reads from that row; stop here rather than emit noise.
        return checks
    n_calls = int(float(row["n_calls"]))
    checks.append(Check("C1 feature-method-present", PASS, f"{method} n_calls={n_calls}"))

    # C2 — driven by the exerciser exactly once per repeat, or incidental fixture calls?
    if n_calls == repeats:
        checks.append(Check("C2 calls-equal-repeats", PASS, f"{n_calls} == {repeats}"))
    elif n_calls % repeats == 0:
        checks.append(Check("C2 calls-equal-repeats", FAIL,
                            f"{n_calls} is a multiple of {repeats} but not equal to it — the "
                            f"method is being called more than once per repeat (setup calls, or "
                            f"a second driver), so its timings are not one-query-per-repeat"))
    else:
        checks.append(Check("C2 calls-equal-repeats", FAIL,
                            f"{n_calls} != {repeats} — the perf exerciser did not drive this "
                            f"method; these are incidental calls"))

    # C3 — paper-literal in-session repetition
    if meta.get("repeats") == repeats:
        checks.append(Check("C3 repeats-match", PASS, f"meta.repeats={repeats}"))
    else:
        checks.append(Check("C3 repeats-match", FAIL,
                            f"meta.repeats={meta.get('repeats')!r}, expected {repeats}"))

    # C4 — F6 isolation
    if legacy:
        checks.append(Check("C4 isolated-workload", UNVERIFIABLE,
                            "meta.json predates the `only` field (2026-08-15); isolation cannot "
                            "be confirmed from this profile's artifacts"))
    elif meta.get("only") == method:
        checks.append(Check("C4 isolated-workload", PASS, f"meta.only={method}"))
    else:
        checks.append(Check("C4 isolated-workload", FAIL,
                            f"meta.only={meta.get('only')!r}, expected {method} — this looks "
                            f"like the full sweep, which is F6-invalid for propagation"))

    # C5 — is the profile older than the measurement code?
    if legacy:
        checks.append(Check("C5 harness-current", UNVERIFIABLE,
                            "meta.json records no harness fingerprint"))
    elif meta.get("harness_fingerprint") == fingerprint:
        checks.append(Check("C5 harness-current", PASS, fingerprint))
    else:
        checks.append(Check("C5 harness-current", FAIL,
                            f"profile fingerprint {meta.get('harness_fingerprint')!r} != current "
                            f"{fingerprint} — measurement code changed since this run"))

    # C5b — mtime fallback, and a check in its own right. The fingerprint only exists on v2
    # profiles, but staleness is checkable on any of them: `demandforecast`'s profile was
    # written at 01:12 and the exerciser gained its feature method at 13:50 the same day, so
    # that profile ran a workload in which the method did not exist as a query. That is the
    # single most repeated incident in this project's history.
    newest_harness = max(os.path.getmtime(p) for p in HARNESS_FILES if os.path.exists(p))
    summary_mtime = os.path.getmtime(os.path.join(profile_dir, "summary.csv"))
    if summary_mtime >= newest_harness:
        checks.append(Check("C5b profile-newer-than-harness", PASS, _stamp(summary_mtime)))
    else:
        checks.append(Check("C5b profile-newer-than-harness", FAIL,
                            f"profile {_stamp(summary_mtime)} predates measurement code "
                            f"{_stamp(newest_harness)} — it measured a workload that no longer "
                            f"exists"))

    # C7 — one warm JVM, counted from the data. This is the check `meta.json` cannot make
    # about itself; see the module docstring.
    sessions = count_sessions(profile_dir)
    if sessions is None:
        checks.append(Check("C7 single-session", UNVERIFIABLE,
                            "no raw Kieker output left in the profile (pruned, or never "
                            "written) — session count cannot be confirmed"))
    elif sessions == 1:
        checks.append(Check("C7 single-session", PASS, "1 Kieker session"))
    else:
        checks.append(Check("C7 single-session", FAIL,
                            f"{sessions} Kieker sessions — this is {sessions} cold JVMs "
                            f"(surefire reuseForks=false forks per test class), not the "
                            f"in-session protocol meta.json claims. Scope the run to the "
                            f"exerciser (-Dtest=FeaturePerfExerciseTest)."))

    return checks


def gate_pair(senior_dir, junior_dir, feature_id, repeats=EXPECTED_REPEATS, method=None):
    """C1-C5 on both tracks plus C6 pair symmetry. Returns (checks, ok)."""
    fp = harness_fingerprint()
    checks = []
    for label, d in (("senior", senior_dir), ("junior", junior_dir)):
        for c in gate_profile(d, feature_id, repeats, fp, method=method):
            checks.append(Check(f"[{label}] {c.cid}", c.status, c.detail))

    # C6 — F7 pair symmetry. Only meaningful if both tracks produced a row.
    method = feature_method_for(feature_id, method=method)
    srow, _ = _summary_row(senior_dir, method)
    jrow, _ = _summary_row(junior_dir, method)
    if srow and jrow:
        sn, jn = int(float(srow["n_calls"])), int(float(jrow["n_calls"]))
        if sn == jn:
            checks.append(Check("C6 pair-symmetry", PASS, f"both tracks n_calls={sn}"))
        else:
            # A call-count skew is not automatically disqualifying — an N+1 injection legitimately
            # multiplies calls of the methods BELOW the feature. It is disqualifying on the
            # feature method itself, which the exerciser drives a fixed number of times.
            checks.append(Check("C6 pair-symmetry", FAIL,
                                f"feature-method calls differ: senior={sn} junior={jn}; the "
                                f"exerciser drives this method a fixed number of times, so the "
                                f"two tracks did not run the same workload"))
    else:
        checks.append(Check("C6 pair-symmetry", FAIL, "one or both tracks have no feature-method row"))

    ok = all(c.status == PASS for c in checks)
    return checks, ok


def _report(checks, allow_legacy):
    worst_fail = [c for c in checks if c.status == FAIL]
    unver = [c for c in checks if c.status == UNVERIFIABLE]
    for c in checks:
        print(f"  {c}")
    print()
    if worst_fail:
        print(f"GATE RED — {len(worst_fail)} check(s) failed. Do not admit this measurement.")
        return 1
    if unver and not allow_legacy:
        print(f"GATE RED — {len(unver)} check(s) UNVERIFIABLE (legacy profile). Re-profile, or "
              f"pass --allow-legacy to accept the measurement with those claims unchecked.")
        return 1
    if unver:
        print(f"GATE AMBER — passed, but {len(unver)} claim(s) could not be verified "
              f"(--allow-legacy). Record this in the row's validity block.")
        return 0
    print("GATE GREEN — measurement is what it claims to be.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("feature", help="feature id ('lift' or 'teastore-hist-lift')")
    ap.add_argument("--method", default=None,
                    help="explicit target method name (overrides feature_methods.json lookup)")
    ap.add_argument("--dataset", default=os.path.join(REPO, "teastore-dataset"))
    ap.add_argument("--profile-dir", default="profile_only",
                    help="subdirectory to gate (default: profile_only, the isolated workload)")
    ap.add_argument("--repeats", type=int, default=EXPECTED_REPEATS)
    ap.add_argument("--allow-legacy", action="store_true",
                    help="accept profiles whose meta.json predates the `only`/fingerprint fields")
    args = ap.parse_args()

    stem = args.feature if os.path.isdir(os.path.join(args.dataset, args.feature + "-senior")) \
        else (args.feature if args.feature.startswith("teastore-hist-")
              else "teastore-hist-" + args.feature)
    sen = os.path.join(args.dataset, stem + "-senior", args.profile_dir)
    jun = os.path.join(args.dataset, stem + "-junior", args.profile_dir)

    print(f"\nmeasurement gate — {stem} ({args.profile_dir})")
    print(f"harness fingerprint: {harness_fingerprint()}\n")
    checks, _ = gate_pair(sen, jun, args.feature, args.repeats, method=args.method)
    return _report(checks, args.allow_legacy)


if __name__ == "__main__":
    sys.exit(main())
