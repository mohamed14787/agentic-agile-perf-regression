#!/usr/bin/env python3
"""Dual-Track Implementation Engine — thesis Component 2.

Generates a functionally-equivalent PAIR of implementations of ONE feature:
  • Senior track — large LLM  → efficient implementation
  • Junior track — small LLM  → naturally less-efficient implementation
    (the ground-truth, fault-injection-free performance regression)

Both tracks implement an IDENTICAL frozen spec (the senior-authored contract) and
are judged by the SAME unit-test suite, so the implementation model SCALE is the
only independent variable. A pair enters the dataset only if BOTH compile and pass
the same tests (functional equivalence) — a broken junior is an invalid sample.

KEEP-ASIDE / ADDITIVE: this module orchestrates the EXISTING pipeline as
subprocesses and consumes its output artifacts (contract.json, test_report.json).
It imports and modifies NO pipeline internals, so the still-evolving
performance-maximization work stays untouched. Process isolation per track also
avoids cline's global-once provider configure() bleeding between models.

Profiling (exec time / CPU / memory — Component 3) is intentionally left as stubs.

Flow:
  A. SENIOR runs the full pipeline (plans → authors tests → implements).         [plan + impl]
  B. Stage a fresh JUNIOR workspace; seed it with senior's frozen contract.json
     + senior's src/test/** suite so the spec & tests are identical.             [freeze + fork]
  C. JUNIOR runs the pipeline with the small model, tests frozen (not re-authored). [impl only]
  D. Equivalence gate: run the SAME test suite against both; admit iff both green. [equivalence]
  E. Append one dataset row (perf-metric fields stubbed for Component 3).         [dataset]

Usage:
  python -m agile_team_cline.dual_track \
      --feature "<feature text>" --stack java-jakarta \
      --baseline TeaStore --out-prefix teastore-cheapest-dual \
      --senior-model <large-id> --junior-model <small-id> \
      [--maven-test-args "-pl services/...,utilities/... -am"]
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time


# ── profiling (Component 3): Kieker local per-method timing ───────────────────
# Reuse the standalone profiling harness so the dual-track engine and the CLI profiler
# share ONE measurement definition (argLine + parser + comparator).
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "profiling"))
try:
    import kieker_args
    import parse_kieker
    import compare_kieker
    _KIEKER_AVAILABLE = True
except Exception:  # profiling is optional; engine still runs the equivalence gate
    _KIEKER_AVAILABLE = False

_KIEKER_ENABLED = _KIEKER_AVAILABLE and os.getenv("DUAL_KIEKER", "1") == "1"


def _profile_summary(profile_dir: str) -> tuple[dict, dict]:
    """Parse a profiled test run (pooled across profile/iter_NN/) into
    ({signature: stats}, {signature: [exec_ns, ...]}).
    Keeps the full distribution (stddev/p50/min/max, not just mean/p99): the comparator's
    noise guards need it, and the thesis dataset exists to benchmark regression
    DETECTORS — stripping the distribution would strip their input. Persists summary.csv
    AND samples.csv (Kieker plan Upgrade A: every raw per-call duration, durable even if
    iteration dirs are later cleaned) next to the raw .dat files. The raw dict feeds the
    comparator's Wilcoxon/Cliff's-Delta guard — rank-based, so the outlier-heavy JVM
    distributions that defeat a mean-2σ guard still score correctly."""
    times = parse_kieker.parse_dir(profile_dir)
    rows = parse_kieker.summarise(times)
    if rows:
        parse_kieker.write_summary(rows, os.path.join(profile_dir, "summary.csv"))
        parse_kieker.write_raw(times, os.path.join(profile_dir, "samples.csv"))
        # Upgrade A completion: iteration + trace_id kept (samples.csv above pools them).
        tagged = parse_kieker.parse_run_tagged(profile_dir)
        parse_kieker.write_raw_tagged(tagged, os.path.join(profile_dir, "samples_tagged.csv"))
        # Upgrade B: call-graph reconstruction (traceId/eoi/ess) — additive, non-fatal.
        try:
            graph = parse_kieker.build_call_graph(profile_dir)
            parse_kieker.write_traces(graph, os.path.join(profile_dir, "traces.json"))
        except Exception as e:
            print(f"   ⚠️ call-graph reconstruction skipped: {e}", flush=True)
    keep = ("operation_signature", "mean_ns", "stddev_ns", "min_ns",
            "p50_ns", "p95_ns", "p99_ns", "max_ns", "n_calls")
    return ({r["operation_signature"]: {k: r[k] for k in keep} for r in rows}, times)


# Every top-level key `system_oracle_stats` can put
# in `oracle_stats.json` MUST be triaged here. A new
# key added to the producer and forgotten here now fails `test_every_producer_key_is_triaged`
# instead of shipping silently. "carry" = the key (or a documented rename/derivation, see
# `_ORACLE_KEY_ROW_NAME` below) reaches `row["oracle"]`. "raw-detail" = deliberately left out
# — either too large for a row (`per_label`, `response_bytes`: full per-sampler breakdown,
# available via the `path` field) or superseded by a carried derived field (`error_rates` ->
# `error_rate_pct`; `effect_magnitude`/`baseline_n`/`candidate_n`/`baseline_mean_ms`/
# `candidate_mean_ms` -> the `feature_*`/pooled fields already carried are what scoring uses).
ORACLE_STATS_KEY_DISPOSITION = {
    "system_verdict": "carry",
    "p_value": "carry",
    "cliffs_delta": "carry",
    "effect_magnitude": "raw-detail",
    "mpd_pct": "carry",
    "baseline_n": "raw-detail",
    "candidate_n": "raw-detail",
    "baseline_mean_ms": "raw-detail",
    "candidate_mean_ms": "raw-detail",
    "per_label": "raw-detail",
    "browse_only": "carry",
    "feature_verdict": "carry",
    "feature_exercised_by_workload": "carry",
    "feature_mpd_pct": "carry",
    "feature_cliffs_delta": "carry",
    "feature_p_value": "carry",
    "feature_samples": "carry",
    "per_repeat": "carry",
    "per_repeat_mpd_median_pct": "carry",
    "per_repeat_mpd_range_pct": "carry",
    "warmup_discard": "carry",
    "load_generator": "carry",
    "error_rates": "carry",
    "response_bytes": "raw-detail",
    "response_nonempty": "carry",
    "response_nonempty_detail": "carry",
}

# Producer key -> row["oracle"] key, for the "carry" keys that are renamed or compactly
# derived rather than copied verbatim. A "carry" key absent from this map is assumed to
# survive under its own name — checked by the same test.
_ORACLE_KEY_ROW_NAME = {
    "error_rates": "error_rate_pct",
    "per_repeat": "per_repeat_feature_mpd_pct",
}


def _system_oracle_row(junior_out: str) -> dict:
    """System-level oracle block. Kept as a separate, optional post-step per
    profiling/system_oracle_stats.py (this module never runs Docker/JMeter itself — see
    scripts/run_oracle.sh): if `run_oracle.sh <senior> <junior> <ticket>` was run
    separately and left `oracle_stats.json` in the junior workspace, its verdict is merged
    into the row; otherwise the row carries an honest null placeholder (no silent 'none'
    verdict from data that was never collected)."""
    stats_path = os.path.join(junior_out, "oracle_stats.json")
    if not os.path.isfile(stats_path):
        # A pair can be un-measurable rather than merely unmeasured, and the two must not read
        # the same. copurchase's junior overrides the datasource at main scope to a driver the
        # deployed service does not ship, so persistence never starts and no load test can
        # produce a verdict — predicted by the deployability gate, confirmed in the container
        # log. `run_oracle_subset.sh` records that as oracle_excluded.json rather than leaving
        # the slot looking pending forever.
        excl_path = os.path.join(junior_out, "oracle_excluded.json")
        if os.path.isfile(excl_path):
            try:
                with open(excl_path, encoding="utf-8") as fh:
                    excl = json.load(fh)
            except Exception:
                excl = {}
            return {"system_verdict": None,
                    "oracle": {"isolation": "single-machine", "available": False,
                               "excluded": True,
                               "excluded_class": excl.get("class"),
                               "excluded_reason": excl.get("reason"),
                               "path": excl_path}}
        return {"system_verdict": None, "oracle": {"isolation": "single-machine",
                                                    "available": False}}
    try:
        with open(stats_path, encoding="utf-8") as fh:
            stats = json.load(fh)
    except Exception:
        return {"system_verdict": None, "oracle": {"isolation": "single-machine",
                                                    "available": False}}
    # `feature_exercised_by_workload` decides whether system_verdict means anything. A run whose
    # workload never called the feature's endpoint produces a perfectly well-formed "none" from
    # browse traffic alone — which is how four rows came to carry a verdict about code the load
    # generator never touched. Carrying the flag (and the feature-only verdict beside the pooled
    # one) keeps that distinction in the row instead of losing it in the JTL.
    return {"system_verdict": stats.get("system_verdict"),
            "oracle": {"isolation": "single-machine", "available": True,
                      "p_value": stats.get("p_value"),
                      "cliffs_delta": stats.get("cliffs_delta"),
                      "mpd_pct": stats.get("mpd_pct"),
                      "feature_exercised_by_workload":
                          stats.get("feature_exercised_by_workload", False),
                      "feature_verdict": stats.get("feature_verdict"),
                      "feature_mpd_pct": stats.get("feature_mpd_pct"),
                      "feature_cliffs_delta": stats.get("feature_cliffs_delta"),
                      "feature_p_value": stats.get("feature_p_value"),
                      "feature_samples": stats.get("feature_samples"),
                      # The share of end-to-end response time the feature's own endpoint holds
                      # under the deployed request mix. This is the quantity step C's
                      # denominator is supposed to carry, and the isolated profile sets it to 1
                      # by construction, so a row that reports a demand scale without it cannot
                      # be interpreted. Measured 2.1-7.0% across the corpus (median 3.1%),
                      # against the paper's own hardest case at 2.65%.
                      "feature_share_of_pooled_pct": _feature_share_pct(stats),
                      # Pooled deltas hide their own spread: churn reads a uniform ~+7% pooled
                      # while its three repeats are -9.4/+8.4/+5.3. Two rows in this corpus
                      # (orphans, revenuepareto) rest entirely on whether their repeats agree.
                      "per_repeat_feature_mpd_pct": [
                          r.get("feature_mpd_pct") for r in (stats.get("per_repeat") or [])],
                      "per_repeat_mpd_median_pct": stats.get("per_repeat_mpd_median_pct"),
                      "per_repeat_mpd_range_pct": stats.get("per_repeat_mpd_range_pct"),
                      # parse_jtl drops failed samples, so without this a stack that fell over
                      # under load is indistinguishable from one that served every request.
                      "error_rate_pct": {
                          "baseline": ((stats.get("error_rates") or {})
                                       .get("baseline", {}) or {}).get("error_rate_pct"),
                          "candidate": ((stats.get("error_rates") or {})
                                        .get("candidate", {}) or {}).get("error_rate_pct")},
                      "warmup_discard": stats.get("warmup_discard"),
                      # Carried through as-is; None on runs that predate it.
                      "browse_only": stats.get("browse_only"),
                      # response_nonempty gate (2026-08-23): every check above confirms the
                      # request was sent and returned success=true; none confirm the response
                      # carried data. False/None means the Feature endpoint's payload was a
                      # fixed size across every sample in an arm — the signature that found six
                      # silently-empty endpoints by hand
                      # before this became a standing field. feature_verdict on such a row
                      # should be read as unconfirmed, not a clean true negative.
                      "response_nonempty": stats.get("response_nonempty"),
                      "response_nonempty_detail": stats.get("response_nonempty_detail"),
                      # Note: dupes' junior ran at 31-32rps against a 50rps target
                      # (worst_drift_pct -38%, starved=true) and that fact lived only in the
                      # workspace's oracle_stats.json — the whitelist never carried it, so the
                      # corpus's single largest effect size (dupes, +15018% pooled) shipped
                      # with no starvation caveat attached to the row that reports it.
                      "load_generator": stats.get("load_generator"),
                      "path": stats_path}}


def _feature_share_pct(stats: dict):
    """The Feature label's request-weighted share of pooled response time, senior side.

    mean_ms x n_calls is the label's total time; divided by the pooled equivalent it is the
    fraction of everything a user waits for that this endpoint accounts for. Returns None when
    the run predates the per-label breakdown or never exercised the feature.
    """
    per_label = stats.get("per_label") or {}
    feat = per_label.get("Feature")
    if not feat or not stats.get("baseline_mean_ms") or not stats.get("baseline_n"):
        return None
    num = (feat.get("baseline_mean_ms") or 0.0) * (feat.get("baseline_n") or 0)
    den = stats["baseline_mean_ms"] * stats["baseline_n"]
    return round(num / den * 100.0, 3) if den else None


def _injected_inefficiencies(junior_out: str) -> dict | None:
    """Ground-truth label for menu-mode runs: when the junior brief is the inefficiency
    catalog (scripts/junior_inefficiency_menu.txt), the junior self-reports which anti-patterns
    it actually injected vs skipped into `injected_inefficiencies.json` at the workspace root.
    Fail-open: returns None if the file is absent (prescribed-brief or scale runs) or unparseable,
    so the row never carries a fabricated label."""
    path = os.path.join(junior_out, "injected_inefficiencies.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            report = json.load(fh)
    except Exception:
        return None
    injected = report.get("injected") or []
    skipped = report.get("skipped") or []
    return {"injected": injected, "skipped": skipped,
            "labels": [i.get("id") for i in injected if isinstance(i, dict) and i.get("id")],
            "path": path}


_EMF_OVERRIDE_CALL = "EMFManager.configureEMFWithProperties"
_JDBC_DRIVER_KEY = "jakarta.persistence.jdbc.driver"


def _declared_driver_persistence_xml(path: str) -> str | None:
    m = re.search(rf'name\s*=\s*"{re.escape(_JDBC_DRIVER_KEY)}"\s+value\s*=\s*"([^"]+)"',
                  open(path, encoding="utf-8").read())
    return m.group(1) if m else None


def _declared_driver_properties(path: str) -> str | None:
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == _JDBC_DRIVER_KEY:
            return v.strip()
    return None


def _deployability(out_dir: str) -> dict:
    """Can this workspace actually RUN, or only pass its unit tests?

    Motivated by teastore-hist-copurchase-junior (2026-07-28), which was `admitted: true`
    and could not deploy: every service came up except persistence, which died with
        Configuration error.  Class [org.h2.Driver] not found
    Its `JpaConfig` calls `EMFManager.configureEMFWithProperties(loadJpaProperties())`,
    re-pointing the shared EntityManagerFactory at whatever `application.properties`
    declares — and that file (agent-generated; present in ALL 22 workspaces, seniors
    included) sets an H2 in-memory driver, while `persistence.xml` — the config the
    deployed app is built around — declares `com.mysql.jdbc.Driver` and the container
    ships only the MySQL connector. It passes `mvn test` because H2/HSQLDB are on the
    TEST classpath. Unit-test equivalence is not deployment equivalence.

    `configureEMFWithProperties` is upstream API, not an invention: its legitimate caller
    is `src/test/java/.../EMFManagerInitializer.java`, which is exactly how a test points
    the EMF at an in-memory database. So the defect is not the call, nor the properties
    file — each is harmless alone, and nine of the ten juniors ship the file inert. The
    defect is the CONJUNCTION: a main-scope (deployed) caller applying properties that
    CONTRADICT persistence.xml's driver. Both halves are required before flagging.

    Fail-open: `deployable: True` whenever the evidence is absent or unreadable, so the
    row never carries a fabricated rejection (same convention as
    `_injected_inefficiencies`)."""
    try:
        overriders, props_path, xml_path = [], None, None
        for root, dirs, files in os.walk(out_dir):
            dirs[:] = [d for d in dirs if d not in ("target", ".git", "node_modules")]
            is_main_java = f"src{os.sep}main{os.sep}java" in root
            for fn in files:
                full = os.path.join(root, fn)
                if is_main_java and fn.endswith(".java"):
                    if _EMF_OVERRIDE_CALL in open(full, encoding="utf-8",
                                                  errors="ignore").read():
                        overriders.append(os.path.relpath(full, out_dir))
                elif fn == "application.properties" and \
                        f"src{os.sep}main{os.sep}resources" in root:
                    props_path = props_path or full
                elif fn == "persistence.xml" and \
                        f"src{os.sep}main{os.sep}resources" in root:
                    xml_path = xml_path or full
        if not overriders or not props_path or not xml_path:
            return {"deployable": True, "reason": "no main-scope datasource override"}

        prop_driver = _declared_driver_properties(props_path)
        xml_driver = _declared_driver_persistence_xml(xml_path)
        if not prop_driver or not xml_driver or prop_driver == xml_driver:
            return {"deployable": True, "reason": "main-scope override agrees with "
                                                  "persistence.xml"}
        return {
            "deployable": False,
            "reason": (f"{overriders[0]} calls {_EMF_OVERRIDE_CALL} at main scope, applying "
                       f"{_JDBC_DRIVER_KEY}={prop_driver} from "
                       f"{os.path.relpath(props_path, out_dir)} over persistence.xml's "
                       f"{xml_driver} — the deployed service loads a driver it does not ship"),
            "overriders": overriders,
            "properties_driver": prop_driver,
            "persistence_xml_driver": xml_driver,
        }
    except Exception as e:
        return {"deployable": True, "reason": f"deployability check skipped: {e}"}


def _step_timeouts(out_dir: str) -> dict | None:
    """Step-timeout taxonomy (2026-07-18): cline_runner drops one JSONL line per authoring
    step that RAN OUT ITS CLOCK (subprocess hard cap or a >=50%-budget abort) into
    `<out_dir>/step_timeouts.jsonl`. A track that fails to compile BECAUSE a step timed out
    is an infrastructure failure, not a model writing bad code — surfacing it keeps a
    build_success=False row from being mistaken for a genuine junior-vs-senior sample (see
    2026-07-16 teastore-hist-churn: DTO step timed out at 5400s → build red → looked like a
    real fail). Fail-open: None when the marker is absent (the common, healthy case)."""
    path = os.path.join(out_dir, "step_timeouts.jsonl")
    if not os.path.isfile(path):
        return None
    events = []
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return None
    if not events:
        return None
    return {"n": len(events),
            "titles": [e.get("title") for e in events if e.get("title")],
            "path": path}


def _call_graph_summary(profile_dir: str) -> dict:
    """Compact call_graph block for the dataset row — counts + the traces.json path, never
    the full edge list (same 'raw on disk, summary in row' discipline as raw_times/
    operations, per _without_ops)."""
    path = os.path.join(profile_dir, "traces.json")
    if not os.path.isfile(path):
        return {"available": False}
    try:
        with open(path, encoding="utf-8") as fh:
            graph = json.load(fh)
    except Exception:
        return {"available": False}
    return {"available": True, "n_edges": len(graph.get("edges", [])),
            "n_entry_points": len(graph.get("entry_points", [])), "path": path}


def _perf_from_ops(res: dict, top: int = 10) -> dict:
    """Build the dataset-row perf block from a track's profiled operations."""
    ops = res.get("operations") or {}
    if not ops:
        return {"profiled": False, "exec_time_ms": None, "cpu_pct": None, "max_rss_kb": None}
    ranked = sorted(ops.values(), key=lambda r: -r["mean_ns"])[:top]
    return {
        "profiled": True,
        "n_operations": len(ops),
        "profile_iterations": res.get("profile_iterations"),
        "top_methods": [{
            "signature": r["operation_signature"],
            "mean_ms": round(r["mean_ns"] / 1e6, 4),
            "stddev_ms": round(r.get("stddev_ns", 0) / 1e6, 4),
            "p50_ms": round(r.get("p50_ns", 0) / 1e6, 4),
            "p99_ms": round(r["p99_ns"] / 1e6, 4),
            "n_calls": r["n_calls"],
        } for r in ranked],
    }


# ── chain topology (longitudinal / chained-trunk mode) ───────────────────────
def _chain_block(senior_res: dict) -> dict:
    """Chain topology for the dataset row. In chained-trunk mode (the longitudinal
    history batch exports DUAL_CHAIN_MODE=chained-trunk + DUAL_CHAIN_* context) this
    records the feature's place in the senior trunk: its 1-based sequence, the last
    LANDED parent it chained off (or "genesis"), and whether senior's tests pass — the
    same gate that advances the trunk AFTER this row is written, so it is the row's
    predicted landing. Independent-pairs runs leave the env unset and get a minimal
    {"mode": "independent"} block so those scripts are unaffected."""
    mode = os.getenv("DUAL_CHAIN_MODE", "").strip() or "independent"
    if mode != "chained-trunk":
        return {"mode": "independent"}
    parent_id = os.getenv("DUAL_CHAIN_PARENT_ID", "").strip() or "genesis"
    try:
        sequence = int(os.getenv("DUAL_CHAIN_SEQUENCE", "").strip())
    except ValueError:
        sequence = None
    block = {
        "mode": "chained-trunk",
        "sequence": sequence,
        "parent_feature_id": parent_id,
        "parent_senior_out_dir": os.getenv("DUAL_CHAIN_PARENT_SENIOR", "").strip() or None,
        "trunk_dir": os.getenv("DUAL_CHAIN_TRUNK", "").strip() or None,
        "senior_tests_pass": bool(senior_res.get("tests_pass")),
        "baseline_provenance": "genesis" if parent_id == "genesis" else f"trunk@{parent_id}",
    }
    # A test point can be a valid SENIOR-VS-JUNIOR sample without being a valid link in the
    # sequential chain. The chain invariant is that feature N's baseline is feature N-1's
    # landed trunk; features built in parallel off a shared ancestor, or off an ad-hoc merged
    # composite, break it. Their local_regression is still sound (both tracks fork from the
    # same baseline), but `trunk_regression` must NOT be read as cumulative sequential drift
    # and the row must not imply its baseline contains its predecessor's code.
    # DUAL_CHAIN_OFF_CHAIN carries the reason; absent = a normal sequential test point.
    off = os.getenv("DUAL_CHAIN_OFF_CHAIN", "").strip()
    if off:
        block["off_chain"] = True
        block["off_chain_reason"] = off
    return block


def _trunk_regression(chain: dict, senior_res: dict, sen_targets: dict = None) -> dict | None:
    """Cumulative (trunk-drift) regression: does THIS feature's senior slow down code
    that already existed on the trunk? Compares the previous LANDED senior profile
    (baseline) against this feature's senior profile (candidate) over their SHARED
    (pre-existing) methods, reusing the SAME comparator + Wilcoxon/Cliff's-Delta noise
    guards as the within-feature senior-vs-junior local_regression. Methods this feature
    ADDED show as NEW and are surfaced separately, never counted as regressions. Returns
    None at genesis (no parent), when profiling is off, or when the parent profile is
    absent (e.g. a red senior that never profiled)."""
    if not _KIEKER_ENABLED or chain.get("mode") != "chained-trunk":
        return None
    parent_senior = chain.get("parent_senior_out_dir")
    cur_ops = senior_res.get("operations")
    if not parent_senior or not cur_ops:
        return None
    prev_times = parse_kieker.parse_dir(os.path.join(parent_senior, _profile_dirname()))
    if not prev_times:
        return None
    prev_ops = {r["operation_signature"]: r for r in parse_kieker.summarise(prev_times)}
    # Controls matter MORE here than for local_regression: the two profiles were captured in
    # separate sessions, often days apart, so the machine-state offset between them is
    # unbounded. Untouched-by-this-feature methods estimate it. apply_offset=False (2026-07-30,
    # matching local_regression below, for consistency across every trunk-drift number in the
    # dataset) — the estimate is still computed and reported (run_offset_ratio/pct) as a
    # diagnostic; a run that shows a large one here is a genuine candidate for `--apply-offset`
    # via compare_kieker directly when auditing a specific trunk-drift flag.
    controls = _control_signatures((prev_ops, sen_targets or {}),
                                   (cur_ops, sen_targets or {}))
    cmp = compare_kieker.compare_summaries(
        prev_ops, cur_ops,
        threshold_pct=float(os.getenv("DUAL_KIEKER_THRESHOLD_PCT", "20")),
        min_delta_ms=float(os.getenv("DUAL_KIEKER_MIN_DELTA_MS", "0.5")),
        min_calls=int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")),
        baseline_raw=prev_times, candidate_raw=senior_res.get("raw_times"),
        control_sigs=controls, apply_offset=False,
        # Trunk comparison: a "control" here is a method that predates this feature, and
        # whether those slowed down IS the cumulative-drift question. Excluding them would
        # make this block structurally unable to report anything.
        exclude_controls=False)
    block = {k: cmp[k] for k in
             ("threshold_pct", "min_delta_ms", "min_calls",
              "run_offset_ratio", "run_offset_pct", "normalized", "offset_estimated",
              "n_control_methods", "normalization_note",
              "n_methods_compared", "n_regressions", "n_suspect",
              "regressed", "suspect")}
    block["parent_feature_id"] = chain.get("parent_feature_id")
    block["new_methods"] = [r["operation_signature"] for r in cmp["rows"]
                            if r["status"] == "NEW"]
    return block


# ── shell helpers ────────────────────────────────────────────────────────────
def _run(cmd, env=None, cwd=None, capture=False):
    print(f"   $ {' '.join(cmd)}", flush=True)
    if capture:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    return subprocess.run(cmd, env=env, cwd=cwd).returncode, ""


def _write_src_snapshot(out: str) -> None:
    """Tar.gz snapshot of every *.java/pom.xml/persistence.xml under `out` (excl. target/),
    taken right after staging. Recovery for whole-source-tree deletions — the Jun-28 event
    deleted baseline PACKAGES, not just poms, and the existing pom_snapshot.json guard
    (agile_team_cline/sprint.py _restore_missing_poms) is pom-only. A multi-hour unattended
    batch needs source-level recovery too. Deliberately a plain archive, not `git init`
    inside the workspace — cline operates on this tree and its checkpoint/shadow-git
    behavior on an already-git workspace is untested; a tar snapshot has zero side effects.
    Disable with DUAL_SRC_SNAPSHOT=0."""
    import tarfile
    snap_path = os.path.join(out, "src_snapshot.tar.gz")
    try:
        with tarfile.open(snap_path, "w:gz") as tar:
            for dp, dirs, files in os.walk(out):
                dirs[:] = [d for d in dirs
                          if d not in ("target", ".git") and not d.startswith(".")]
                for fn in files:
                    if fn == "pom.xml" or fn.endswith((".java", "persistence.xml")):
                        full = os.path.join(dp, fn)
                        tar.add(full, arcname=os.path.relpath(full, out))
        size_kb = os.path.getsize(snap_path) / 1024
        print(f"   src snapshot: {snap_path} ({size_kb:.0f} KB)", flush=True)
    except Exception as e:
        print(f"   ⚠️ src snapshot failed (non-fatal): {e}", flush=True)


def _count_java(root: str) -> int:
    """.java files under `root`, excluding build output — the same shape the pipeline
    counts as the brownfield baseline (target/.git excluded to match stage_workspace)."""
    if not os.path.isdir(root):
        return 0
    total = 0
    for dp, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in ("target", ".git") and not d.startswith(".")]
        total += sum(1 for f in files if f.endswith(".java"))
    return total


def stage_workspace(baseline: str, out: str) -> None:
    """Fresh rsync of the baseline repo into `out` (mirrors run_*_v2.sh staging).

    Staging-integrity guard (recoverability fix): an EMPTY staged tree is the silent
    trigger for greenfield regeneration — the pipeline sees no existing files, sets
    milestone_file_baseline=0, and regenerates the WHOLE reactor (Product/Order/Category/
    User/OrderItem repos + a test each) for a one-method feature. That's 84% of tokens
    spent on gen, and any one non-compiling file sinks `mvn test` → non-equivalent (27/30
    pairs died this way). Root cause: `targets/TeaStore` is a bare git gitlink (mode 160000)
    with no .gitmodules, so a fresh checkout leaves it empty and rsync copies emptiness.
    We turn that SILENT greenfield into a LOUD, recoverable failure by distinguishing:
      • rsync dropped files (baseline populated, staged smaller) → always fatal
      • baseline itself empty (the gitlink case) → fatal unless DUAL_ALLOW_EMPTY_BASELINE=1
        (the explicit opt-in for a genuinely greenfield target)."""
    if os.path.exists(out):
        shutil.rmtree(out)
    _run(["rsync", "-a", "--exclude", "target", "--exclude", ".git",
          baseline.rstrip("/") + "/", out + "/"])
    # Baseline-integrity guard source of truth: the pipeline subprocess (run_track
    # inherits os.environ) restores any mid-run-deleted baseline file from here.
    os.environ.setdefault("AGILE_BASELINE_REF", os.path.abspath(baseline))
    n = _count_java(out)
    baseline_java = _count_java(baseline)
    print(f"   staged {out}: {n} java files (baseline has {baseline_java})", flush=True)

    if baseline_java > 0 and n < baseline_java:
        raise RuntimeError(
            f"Staging integrity failure: baseline {baseline!r} has {baseline_java} .java "
            f"files but only {n} landed in {out!r} — rsync dropped source. Refusing to run: "
            "a partial baseline forces greenfield regeneration of the missing modules and "
            "sinks the build. Check disk space / rsync errors above and re-stage.")
    if baseline_java == 0 and os.getenv("DUAL_ALLOW_EMPTY_BASELINE", "0") != "1":
        raise RuntimeError(
            f"Empty baseline: {baseline!r} contains ZERO .java files, so this run would "
            "regenerate the entire project greenfield (the documented token-blowup + "
            "non-equivalence failure mode). This usually means `targets/<repo>` is an "
            "uninitialized git gitlink (mode 160000, no .gitmodules) — populate it with the "
            "real upstream source before running. If you INTEND a greenfield target, set "
            "DUAL_ALLOW_EMPTY_BASELINE=1 to proceed.")

    if os.getenv("DUAL_SRC_SNAPSHOT", "1") == "1":
        _write_src_snapshot(out)


def _is_infra_abort_line(line: str) -> bool:
    """True only for the harness's OWN quota-abort sentinel lines (the ✋ prints in
    shared.step_with_retry / cline_runner.run_cline_with_retry). Free-text matching the whole
    stream would false-tag a run whose brief or generated code merely mentions quotas."""
    return "✋" in line and "quota exhausted" in line.lower()


def run_track(feature: str, stack: str, out: str, model: str, extra_env: dict):
    """Spawn the EXISTING pipeline as a subprocess for one track. Returns (rc, infra_abort),
    where infra_abort is True when the run died on model-quota exhaustion (429 weekly/monthly
    limit) — an INFRASTRUCTURE failure that must not be scored as a compile failure. Streams
    the pipeline output live while scanning it for the harness's quota-abort sentinel; also
    honors <out>/INFRA_ABORT.json (written by cline_runner), which survives buffering and
    encoding mangling."""
    env = os.environ.copy()
    env["AGILE_CLINE_MODEL"] = model      # the senior/junior fork — implementation model only
    env.update(extra_env)
    cmd = [sys.executable, "-u", "-m", "agile_team_cline.pipeline",
           feature, "--stack", stack, "--output-dir", out]
    print(f"   $ {' '.join(cmd)}", flush=True)
    infra_abort = False
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            errors="replace")
    for line in proc.stdout:
        sys.stdout.write(line)
        if not infra_abort and _is_infra_abort_line(line):
            infra_abort = True
    proc.wait()
    if os.path.exists(os.path.join(out, "INFRA_ABORT.json")):
        infra_abort = True
    if infra_abort:
        print(f"   ✋ track infra-abort (model quota exhausted): {out}", flush=True)
    return proc.returncode, infra_abort


# ── freeze + fork ────────────────────────────────────────────────────────────
def seed_frozen_spec(senior_out: str, junior_out: str) -> dict:
    """Copy senior's frozen contract.json + src/test/** into the junior workspace so
    both implement the identical spec and are judged by the identical suite."""
    seeded = {"contract": False, "test_files": 0}
    src_contract = os.path.join(senior_out, "contract.json")
    if os.path.exists(src_contract):
        shutil.copy2(src_contract, os.path.join(junior_out, "contract.json"))
        seeded["contract"] = True
    # copy every src/test/** tree (skip build output)
    for dp, _dirs, files in os.walk(senior_out):
        norm = dp.replace("\\", "/")
        if "/src/test/" not in norm + "/" or "/target/" in norm:
            continue
        rel = os.path.relpath(dp, senior_out)
        dst = os.path.join(junior_out, rel)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(dp, f), os.path.join(dst, f))
            seeded["test_files"] += 1
    print(f"   seeded junior: contract={seeded['contract']}, "
          f"test_files={seeded['test_files']}", flush=True)
    return seeded


# ── equivalence gate ─────────────────────────────────────────────────────────
# Only the per-CLASS surefire lines ("... - in <TestClass>"): the bare module-summary line
# repeats the same totals, so matching both double-counted tests_run in every dataset row
# (equivalence itself was unaffected — both tracks doubled identically).
_SUREFIRE_RE = re.compile(
    r"Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)"
    r".*\bin\s+[\w.]+")


_PERF_EXERCISE_KNOBS = ("only", "categories", "productsPerCategory", "users", "maxOrdersPerUser",
                        "priceChangesPerProduct", "stockMovementsPerProduct")

# The one test class the profiling run should execute. TeaStore's persistence pom inherits
# <reuseForks>false</reuseForks> from upstream, so surefire forks a FRESH JVM PER TEST CLASS:
# profiling the whole suite yields ~60 Kieker sessions writing into one output dir, every one
# JIT-cold. Measured 2026-08-15 on the profiles this path produced — 61, 63 and 59 sessions —
# against 1 for the scripts/kieker_profile.sh path. That breaks three things at once:
# meta.json's "in-session" claim, parse_run_tagged's block-bootstrap premise (it rebuckets a
# 61-session pool into 30 fake blocks), and build_call_graph (trace ids are unique per JVM
# only, so cross-session merging invents edges).
#
# kieker_profile.sh fixed this with -Dtest scoping; this module never got the fix and the two
# harnesses drifted. Set DUAL_KIEKER_TEST_SCOPE="" to profile the whole suite deliberately.
_KIEKER_TEST_SCOPE_DEFAULT = "FeaturePerfExerciseTest"


def _perf_repeats() -> int:
    """In-session repeat count for the perf-exercise workload. Defaults to 30 — Liao et
    al.'s protocol repeats the component test 30 times INSIDE one Kieker session (confirmed
    against their own replication package: every kieker.dat in Component_perf_res/ carries
    one Kieker metadata record and call counts that are exact multiples of 30), not across
    30 separate `mvn test` invocations. Always resolved explicitly (not left to
    FeaturePerfExerciseTest's own default) so `_write_profile_meta` records the exact value
    used, for parse_kieker.py's block-bootstrap labeling to read back."""
    try:
        return int(os.getenv("DUAL_PERF_repeats", "30"))
    except ValueError:
        return 30


def _perf_exercise_args() -> list:
    """Maven properties that switch on the profiling workload (FeaturePerfExerciseTest).

    The frozen functional tests size their fixtures for readable assertions — four products
    in one category for the price-outlier contract, and so on. Timing a catalog-wide query
    over those row counts measures EntityManager setup rather than the query: a cleanly
    injected N+1 measured 14527us senior vs 14536us junior, a 0.06% difference caused by the
    data being too small for the query's SHAPE to matter, not by the two implementations
    being equivalent. Interleaved against a representative catalog the same pair separates
    by ~32%.

    Only profiled runs get these properties. Without them the test class is a no-op, so a
    plain `mvn test` — including a re-run of the frozen suite by anything downstream — sees
    exactly the workload it saw before this existed. Set DUAL_PERF_EXERCISE=0 to profile the
    old way; DUAL_PERF_<knob> overrides an individual fixture dimension.

    DUAL_PERF_only=<methodName> restricts the sweep to ONE query — the isolated workload the
    paper's propagation step requires. Without it the full 25-query sweep's relative call counts
    describe the sweep rather than a request mix, and one expensive unrelated query dominates
    station demand: measured on reorder, getNearDuplicatePairs took 83.9% of inclusive time and
    pushed the feature under test to 0.238% of demand, so a 22x local regression predicted
    +0.03 ms system-wide. This knob was missing until 2026-08-15, which meant this module could
    not produce a valid propagation profile at all."""
    if os.getenv("DUAL_PERF_EXERCISE", "1") != "1":
        return []
    args = ["-Dteastore.perf.exercise=true", f"-Dteastore.perf.repeats={_perf_repeats()}"]
    for knob in _PERF_EXERCISE_KNOBS:
        override = os.getenv(f"DUAL_PERF_{knob}")
        if override:
            args.append(f"-Dteastore.perf.{knob}={override}")
    return args


def _kieker_test_scope_args() -> list:
    """Restrict a PROFILED run to one test class, so it is one JVM and one Kieker session.

    See _KIEKER_TEST_SCOPE_DEFAULT for why. Applied only to profiled runs: an unprofiled
    `mvn test` must keep executing the whole frozen suite, since that is the equivalence check.
    """
    scope = os.getenv("DUAL_KIEKER_TEST_SCOPE", _KIEKER_TEST_SCOPE_DEFAULT)
    if not scope:
        return []
    return [f"-Dtest={scope}", "-DfailIfNoTests=false",
            "-Dsurefire.failIfNoSpecifiedTests=false"]


def _profile_dirname() -> str:
    """Which subdirectory of a workspace this run's profile belongs in.

    Two workloads produce incompatible profiles and must not share a directory. The full
    25-query sweep goes to `profile/`; an ISOLATED run (DUAL_PERF_only set to one method) goes
    to `profile_only/`, which is what scripts/run_paper_algorithm.py and
    scripts/measurement_gate.py both read by default. Before 2026-08-16 the engine hardcoded
    `profile/` for both, so an isolated batch would have written F6-valid measurements exactly
    where the gate and the sweep never look — the A3/D2 incident class, recreated by directory
    convention rather than by code.

    DUAL_PROFILE_DIR overrides, for a caller that wants a third location."""
    override = os.getenv("DUAL_PROFILE_DIR")
    if override:
        return override
    return "profile_only" if os.getenv("DUAL_PERF_only") else "profile"


def _write_profile_meta(profile_dir: str, lead_track: str = None) -> None:
    """Record the in-session repeat count next to a profiled run, for parse_kieker.py's
    `parse_run_tagged` to rank-rebucket a single-session profile into `repeats` blocks
    instead of one undifferentiated block — see FeaturePerfExerciseTest / kieker_profile.sh.
    Skipped when the exerciser is off: without it, queries aren't repeated in-session, so
    there is nothing meaningful to rebucket by rank."""
    if os.getenv("DUAL_PERF_EXERCISE", "1") != "1":
        return
    os.makedirs(profile_dir, exist_ok=True)
    # v2 (2026-08-15): record what the run WAS, not just its repeat count, so
    # scripts/measurement_gate.py can audit it afterwards. `repeats` alone is a claim the
    # harness makes about itself — the 61-session profiles carried "repeats": 30 while being
    # 61 cold JVMs, and nothing in the artifacts contradicted them.
    meta = {"meta_version": 2, "repeats": _perf_repeats(), "protocol": "in-session",
            "only": os.getenv("DUAL_PERF_only") or None,
            "test_scope": os.getenv("DUAL_KIEKER_TEST_SCOPE", _KIEKER_TEST_SCOPE_DEFAULT) or None,
            # Which track took the COLD first slot. At the default iters=1 the within-pair
            # alternation never fires, so this per-pair hash is the entire warm-slot control —
            # and a hash is only auditable if you still have the input. Record the realized
            # assignment rather than making a reader recompute it. (Realized split across the
            # 25 chain features: 13 senior-first, 12 junior-first.)
            "lead_track": lead_track or os.getenv("DUAL_LEAD_TRACK"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(_repo, "scripts"))
        import measurement_gate
        meta["harness_fingerprint"] = measurement_gate.harness_fingerprint()
    except Exception:                                              # noqa: BLE001
        meta["harness_fingerprint"] = "unavailable"
    with open(os.path.join(profile_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)


# Test classes excluded from the FUNCTIONAL VERDICT run (never from profiling — the profiled
# runs are already scoped to the exerciser). Set DUAL_VERDICT_EXCLUDE="" to keep them.
#
# `*LargeScaleTest` classes are PERFORMANCE fixtures that live in the functional suite: they
# build a large catalog and time one query. Timing them again inside the equivalence gate
# conflates the two axes this dataset exists to separate — the performance question is answered
# by Kieker on the profiled runs, not by a functional assertion's wall clock.
#
# They also make the batch intractable, which is how this was found. `teastore-hist-dupes`
# carries an O(n^2) pairwise self-join with a `String.split` inside the loop:
# ProductRepositoryLargeScaleDupesTest takes **3,131 s (52 min) on the SENIOR** and did not
# finish on the junior in two separate runs (5 h and 2.5 h, both measured at 99.7-99.8% CPU, so
# genuinely computing rather than blocked). One feature of 25 consumed an entire night twice.
#
# This became load-bearing only when the verdict moved to the full frozen suite (the 2026-08-16
# L1 fix). Before that the verdict was parsed from the profiled run scoped to
# `-Dtest=FeaturePerfExerciseTest`, so these classes never executed here at all — the L1 fix was
# correct and this is the bill for it.
#
# The trade: a junior wrong ONLY at large scale would now pass the verdict. Accepted because
# these juniors carry induced PERFORMANCE defects, and the ~27 normal-scale tests per feature
# still check correctness. The exclusion is recorded on every row rather than left implicit
# (specification: "silently missing is never acceptable; explicitly excluded always is").
_VERDICT_EXCLUDE_DEFAULT = "*LargeScale*Test"

# Wall-clock cap per surefire FORK on the verdict run. With reuseForks=false that is one test
# class, so this bounds a single pathological class rather than the suite. Backstop only: it
# exists so an unknown second `dupes` fails in 15 minutes instead of stalling a 25-feature batch
# indefinitely. Six features still carry never-timed LargeScale fixtures.
_VERDICT_TIMEOUT_DEFAULT = "900"


def verdict_scope_args() -> list:
    """Surefire args applied to the UNPROFILED verdict run only. See _VERDICT_EXCLUDE_DEFAULT."""
    args = []
    exclude = os.getenv("DUAL_VERDICT_EXCLUDE", _VERDICT_EXCLUDE_DEFAULT)
    if exclude:
        # `!pattern` is surefire's exclusion form; failIfNoSpecifiedTests guards a module where
        # the filter happens to match everything.
        args += [f"-Dtest=!{exclude}", "-Dsurefire.failIfNoSpecifiedTests=false"]
    timeout = os.getenv("DUAL_VERDICT_TIMEOUT", _VERDICT_TIMEOUT_DEFAULT)
    if timeout and timeout != "0":
        args += [f"-Dsurefire.timeout={timeout}"]
    return args


def _mvn_once(out: str, maven_test_args: list, iter_dir: str = None) -> tuple:
    """One `mvn test` invocation, optionally weaving Kieker into `iter_dir`."""
    cmd = ["mvn", "test", *maven_test_args]
    if iter_dir:
        os.makedirs(iter_dir, exist_ok=True)
        cmd = cmd + [f"-DargLine={kieker_args.arg_line(os.path.abspath(iter_dir))}"]
        cmd = cmd + _perf_exercise_args() + _kieker_test_scope_args()
    else:
        cmd = cmd + verdict_scope_args()
    return _run(cmd, cwd=out, capture=True)


# Exactly the keys _parse_surefire produces — i.e. the functional verdict and nothing else.
# `mvn_test_interleaved(known_verdict=...)` copies this set forward rather than the whole result
# dict, which by then also carries `operations`/`raw_times`/`profile_iterations` from the
# previous pass's profile. Carrying those into a fresh pass would hand the sweep the ISOLATED
# run's timings, which `_attach_profile` happens to overwrite — a silent dependency on call
# order that this narrow copy removes.
_SUREFIRE_FIELDS = ("build_success", "mvn_rc", "tests_run", "failures", "errors",
                    "skipped", "tests_pass")


def _parse_surefire(log: str, rc: int) -> dict:
    """Functional verdict from one suite run's output (pooled per-class surefire lines)."""
    runs = fails = errs = skips = 0
    for m in _SUREFIRE_RE.finditer(log):
        runs += int(m.group(1)); fails += int(m.group(2))
        errs += int(m.group(3)); skips += int(m.group(4))
    build_success = "BUILD SUCCESS" in log
    return {
        "build_success": build_success,
        "mvn_rc": rc,
        "tests_run": runs, "failures": fails, "errors": errs, "skipped": skips,
        "tests_pass": build_success and fails == 0 and errs == 0,
    }


def _attach_profile(res: dict, out: str, kept_iters: int) -> dict:
    """Parse the pooled profile dir into the track result (same fields as mvn_test)."""
    profile_dir = os.path.join(out, _profile_dirname())
    _write_profile_meta(profile_dir)
    ops, raw_times = _profile_summary(profile_dir)
    res["operations"] = ops
    res["raw_times"] = raw_times
    res["profile_iterations"] = kept_iters
    if ops:
        print(f"   profiled {len(ops)} operation(s) over {kept_iters} iteration(s) "
              f"→ {profile_dir}", flush=True)
    else:
        print(f"   ⚠️ Kieker enabled but 0 operations recorded in {out} — check the aop.xml "
              "weave scope for this stack", flush=True)
    return res


def _leads_with_senior(senior_out: str) -> bool:
    """Deterministic per-pair coin flip for which track occupies the cold (first) slot on
    profiling iteration 1.

    With DUAL_KIEKER_ITERS now defaulting to 1 (one in-session profile per track, matching
    Liao et al.'s in-session-repetition protocol — see FeaturePerfExerciseTest), the
    within-run alternation below never fires: there is only ever an "iteration 1", and
    `i % 2` always picks the same starting order. Left as `["senior", "junior"]` always,
    that puts senior in the cold slot and junior in the warm slot on EVERY pair in the
    15-feature chain — a systematic bias aligned with track identity, exactly the kind of
    confound the ±30% whole-run drift finding was about, just moved from within-run to
    across-the-chain. Hashing the pair's own directory name gives a fixed, auditable 50/50
    split across pairs without needing external state (a counter, a config file)."""
    digest = hashlib.md5(os.path.basename(senior_out.rstrip("/")).encode()).hexdigest()
    return int(digest, 16) % 2 == 0


def mvn_test_interleaved(senior_out: str, junior_out: str,
                         maven_test_args: list,
                         known_verdict: tuple = None) -> tuple:
    """Profile both tracks with ALTERNATING iterations, returning (senior_res, junior_res).

    Running all senior iterations and then all junior iterations makes any machine drift over
    the run perfectly confounded with track identity — the defect that produced a 30% whole-run
    offset and a sign-inverted regression in the reorder pair. Alternating turns that drift into
    noise shared by both tracks; the leading track also swaps each iteration so neither one
    systematically occupies the warmer slot. The STARTING order is a per-pair coin flip
    (`_leads_with_senior`), not always senior-first, since at the now-default single
    iteration the alternation itself never executes.

    The functional verdict comes from a SEPARATE UNPROFILED run of the whole frozen suite, one
    per track, before any timing. It cannot come from a profiled iteration: those are scoped to
    `-Dtest=FeaturePerfExerciseTest` (one JVM, one Kieker session — see _kieker_test_scope_args),
    and the exerciser is a single `@Test` whose javadoc states "Never fails. A query that throws
    is counted and reported, not asserted on." Parsing equivalence out of that run yields
    tests_run=1 on both tracks with no way to fail, i.e. `equivalence_verdict` reduces to "both
    compiled" — against historical rows carrying up to 274 tests. That is what this function did
    between the test-scoping change and 2026-08-16; the extra unprofiled run per track is the
    price of the scoping, and it is not optional.

    If either track is red the pair is already non-equivalent, so timing is skipped entirely
    rather than profiling a broken suite.

    `known_verdict` — a (senior_res, junior_res) pair from an EARLIER call in the SAME session
    against the SAME two workspaces — skips phase 1 and reuses that verdict. A feature is
    profiled twice (isolated, then the sweep pass), and the second pass would otherwise re-run
    the whole frozen suite on both tracks to re-derive an answer it already has: the trees are
    untouched between the passes and an unprofiled `mvn test` is deterministic, so the only
    product of the repeat is ~82 s per feature on a 125-test suite — and it grows with the
    suite, which reaches 274 tests late in the chain. Reuse is sound ONLY under those two
    guarantees; never carry a verdict across features or across a rebuild."""
    iters = max(1, int(os.getenv("DUAL_KIEKER_ITERS", "1")))
    tracks = {"senior": senior_out, "junior": junior_out}
    profile_name = _profile_dirname()
    for out in tracks.values():
        shutil.rmtree(os.path.join(out, profile_name), ignore_errors=True)

    # ── phase 1: functional verdict, unprofiled and UNSCOPED (the whole frozen suite) ──
    results, kept = {}, {"senior": 0, "junior": 0}
    if known_verdict is not None:
        for name, prior in zip(("senior", "junior"), known_verdict):
            results[name] = {k: prior[k] for k in _SUREFIRE_FIELDS}
            print(f"   {name} frozen suite: {results[name]['tests_run']} test(s), "
                  f"pass={results[name]['tests_pass']} (reused from this session's "
                  f"earlier pass)", flush=True)
    else:
        for name, out in tracks.items():
            rc, log = _mvn_once(out, maven_test_args)      # no iter_dir → no argLine, no -Dtest
            results[name] = _parse_surefire(log, rc)
            print(f"   {name} frozen suite: {results[name]['tests_run']} test(s), "
                  f"pass={results[name]['tests_pass']}", flush=True)
    if not (results["senior"]["tests_pass"] and results["junior"]["tests_pass"]):
        print("   ⚠️ a track is red on the frozen suite — skipping timing "
              "(a broken suite's timings are meaningless)", flush=True)
        return (_attach_profile(results["senior"], senior_out, 0),
                _attach_profile(results["junior"], junior_out, 0))

    # ── phase 2: timing, profiled and scoped ──
    base_order = (["senior", "junior"] if _leads_with_senior(senior_out)
                  else ["junior", "senior"])
    # Publish the realized cold-slot assignment so `_write_profile_meta` can record it on BOTH
    # tracks' profiles. At iters=1 this hash is the whole warm-slot control, and recording it
    # beats making a reader re-derive it from a path hash months later.
    os.environ["DUAL_LEAD_TRACK"] = base_order[0]
    for i in range(1, iters + 1):
        # swap the leading track every other iteration (base / reversed(base) / base ...)
        order = base_order if i % 2 else list(reversed(base_order))
        print(f"   ── profile iteration {i}/{iters} (order: {' → '.join(order)})", flush=True)
        for name in order:
            out = tracks[name]
            iter_dir = os.path.join(out, profile_name, f"iter_{i:02d}")
            rc, log = _mvn_once(out, maven_test_args, iter_dir)
            # `results` is already final from phase 1 and is NEVER updated here — a scoped
            # run's surefire totals must not reach equivalence_verdict. A failed timing
            # iteration only costs its own samples.
            if "BUILD SUCCESS" not in log:
                print(f"   ⚠️ {name} profile iteration {i}/{iters} failed (the frozen suite "
                      "passed) — discarding its partial timing data", flush=True)
                shutil.rmtree(iter_dir, ignore_errors=True)
            else:
                kept[name] += 1
    return (_attach_profile(results["senior"], senior_out, kept["senior"]),
            _attach_profile(results["junior"], junior_out, kept["junior"]))


def mvn_test(out: str, maven_test_args: list, kieker: bool = False,
             known_verdict: dict = None) -> dict:
    """Run the frozen suite in `out` and parse the surefire totals. Deterministic,
    re-runnable equivalence measurement (does not trust the pipeline's own report).

    When `kieker` is on (Component 3), also weave the Kieker AspectJ agent into the test
    JVM and parse per-method execution time — the local performance signal. Profiling never
    affects the functional verdict, which is measured by its OWN unprofiled, unscoped run of
    the whole frozen suite: a profiled run is scoped to the single-@Test exerciser, which by
    design never fails, so parsing equivalence out of it would reduce the verdict to "it
    compiled". See mvn_test_interleaved for the full account.

    DUAL_KIEKER_ITERS (default 1) repeats the profiled suite run and POOLS the samples
    (<profile dir>/iter_NN/), averaging out JIT/GC noise the way the paper's 30-iteration
    protocol does. A profiling iteration that fails is discarded (its timing data would be
    partial) but never changes the verdict.

    `known_verdict` — a result dict from an EARLIER run of this SAME workspace in the SAME
    session — skips the unprofiled verdict run and reuses it. Used by the replicate (S') pass,
    which re-profiles a tree whose frozen suite was run minutes ago and has not been touched
    since; see mvn_test_interleaved for the full contract."""
    base_cmd = ["mvn", "test", *maven_test_args]
    profile_dir = os.path.join(out, _profile_dirname())
    iters = max(1, int(os.getenv("DUAL_KIEKER_ITERS", "1"))) if kieker else 1
    if kieker:
        shutil.rmtree(profile_dir, ignore_errors=True)

    # ── functional verdict: always its own unprofiled, unscoped full-suite run ──
    if known_verdict is not None:
        result = {k: known_verdict[k] for k in _SUREFIRE_FIELDS}
    else:
        # Same verdict scope as _mvn_once's unprofiled branch — the two verdict paths must not
        # disagree about which classes decide equivalence.
        rc, log = _run(list(base_cmd) + verdict_scope_args(), cwd=out, capture=True)
        result = _parse_surefire(log, rc)
    kept_iters = 0
    if not (kieker and result["tests_pass"]):
        # red suite (timing it would measure nothing real), or profiling is off entirely
        return result if not kieker else _attach_profile(result, out, 0)

    for i in range(1, iters + 1):
        cmd = list(base_cmd)
        iter_dir = os.path.join(profile_dir, f"iter_{i:02d}")
        os.makedirs(iter_dir, exist_ok=True)
        cmd = cmd + [f"-DargLine={kieker_args.arg_line(os.path.abspath(iter_dir))}"]
        cmd = cmd + _perf_exercise_args() + _kieker_test_scope_args()
        rc, log = _run(cmd, cwd=out, capture=True)
        if "BUILD SUCCESS" not in log:
            print(f"   ⚠️ profile iteration {i}/{iters} failed (the frozen suite passed) — "
                  f"discarding its partial timing data", flush=True)
            shutil.rmtree(iter_dir, ignore_errors=True)
        else:
            kept_iters += 1
    return _attach_profile(result, out, kept_iters)


# ── feature-exercised gate ───────────────────────────────────────────────────
# A row can pass every functional gate while the profiler never times the FEATURE: if the
# frozen suite doesn't call the feature method, Kieker only sees baseline methods — which are
# byte-identical in both tracks — and local_regression compares identical code with itself
# (proven on the teastore-cheapest-dual row: getCheapestProducts on disk in both tracks,
# never invoked, and the row's one "regression" was n=1 noise on an unmodified method).
# The gate asks a deterministic question: does at least one PROFILED operation come from a
# class the feature added/changed? Caveat: it checks the profiled (woven) surface only — a
# feature whose perf lives outside the weave scope is honestly reported as not exercised.

_METHOD_DECL_RE = re.compile(r"\b(?:public|protected|private)\b[^=;{()]*?\b(\w+)\s*\(")


def _feature_targets(out: str, baseline: str) -> dict:
    """{class_simple_name: set(added_method_names) | None} for every .java file the track
    ADDED or MODIFIED vs the staged baseline (content compare, not mtime). None means ANY
    profiled method of that class counts (new file, or a body-only modification where no
    method was added — the feature lives inside an existing method). A non-empty set means
    only those ADDED methods count: matching the whole class would let the class's untouched
    baseline methods satisfy the gate (the cheapest-dual false-True: getCheapestProducts was
    added to ProductRepository but never called, while baseline ProductRepository methods
    were profiled)."""
    import filecmp
    targets = {}
    for dp, dirs, files in os.walk(out):
        dirs[:] = [d for d in dirs
                   if d not in ("target", ".git", "profile", "profile_only")
                   and not d.startswith(".")]
        for fn in files:
            if not fn.endswith(".java"):
                continue
            p = os.path.join(dp, fn)
            bp = os.path.join(baseline, os.path.relpath(p, out))
            cls = fn[:-len(".java")]
            try:
                if not os.path.exists(bp):
                    targets[cls] = None            # new file: any method of it counts
                elif not filecmp.cmp(p, bp, shallow=False):
                    new_m = set(_METHOD_DECL_RE.findall(open(p, errors="replace").read()))
                    old_m = set(_METHOD_DECL_RE.findall(open(bp, errors="replace").read()))
                    added = new_m - old_m
                    targets[cls] = added or None   # no added methods -> body-only change
            except OSError:
                targets[cls] = None
    return targets


def _sig_class_method(sig: str) -> tuple:
    """(declaring class simple name, method name) from a Kieker operation signature, e.g.
    'public java.util.List a.b.CategoryRepository.getBasketLift()'
    -> ('CategoryRepository', 'getBasketLift'); constructors yield ('Converter', '<init>')."""
    head = sig.split("(", 1)[0].strip()
    fqcn, method = head.rsplit(".", 1)
    return fqcn.rsplit(".", 1)[-1], method


def _is_touched(sig: str, targets: dict) -> bool:
    """True when `sig` belongs to code this track added or modified vs the baseline."""
    try:
        cls, method = _sig_class_method(sig)
    except Exception:
        return True          # unparseable → treat as touched (never use it as a control)
    return cls in targets and (targets[cls] is None or method in targets[cls])


def _control_signatures(*op_maps_and_targets) -> set:
    """Signatures safe to use as run-offset CONTROLS: profiled in BOTH tracks and touched by
    NEITHER. Their source is identical across senior and junior, so any timing difference on
    them is machine state by construction — exactly the offset compare_kieker divides out.

    Takes ((ops, targets), (ops, targets)) for the two tracks. A signature qualifies only if
    it appears in both tracks' profiled operations and is untouched in both — a method the
    junior modified but the senior did not would otherwise silently absorb real regression
    into the offset estimate."""
    pairs = list(op_maps_and_targets)
    common = None
    for ops, _t in pairs:
        keys = set(ops or {})
        common = keys if common is None else (common & keys)
    controls = set()
    for sig in (common or set()):
        if not any(_is_touched(sig, t or {}) for _ops, t in pairs):
            controls.add(sig)
    return controls


def _feature_exercised(res: dict, targets: dict):
    """True/False when the track was profiled; None when there is nothing to judge
    (profiling off/empty — absence of evidence must not read as a verdict)."""
    ops = res.get("operations")
    if not ops:
        return None
    for sig in ops:
        cls, method = _sig_class_method(sig)
        if cls in targets and (targets[cls] is None or method in targets[cls]):
            return True
    return False


def _admit_row(equivalent: bool, infra_abort: bool, feature_exercised,
               require_feature: bool, deployable=None) -> bool:
    """Central admission rule. feature_exercised=None (unprofiled) never blocks —
    only a POSITIVE finding that the feature was not measured does, and only when
    DUAL_REQUIRE_FEATURE_EXERCISED=1 (default). `deployable` follows the same shape:
    only a POSITIVE False from `_deployability` blocks, never None/unknown."""
    if not equivalent or infra_abort:
        return False
    if require_feature and feature_exercised is False:
        return False
    if deployable is False:
        return False
    return True


def equivalence_verdict(senior: dict, junior: dict) -> dict:
    """Functionally equivalent iff both build green, both pass with zero fail/err,
    and both ran the SAME number of tests (same frozen suite)."""
    both_green = senior["tests_pass"] and junior["tests_pass"]
    same_suite = senior["tests_run"] == junior["tests_run"] and senior["tests_run"] > 0
    equivalent = both_green and same_suite
    if equivalent:
        reason = f"both green, {senior['tests_run']} shared tests pass on each"
    elif not both_green:
        bad = [t for t, v in (("senior", senior), ("junior", junior)) if not v["tests_pass"]]
        reason = f"not functionally equivalent — {', '.join(bad)} failed compile/tests"
    else:
        reason = (f"test-count mismatch (senior={senior['tests_run']} "
                  f"junior={junior['tests_run']}) — suites diverged")
    # `tests_run` is the shared count when the suites agree (the only case a single count is meaningful); on a
    # mismatch, both counts are still visible in `reason`'s text, and a structured pair isn't
    # added here to keep the common (equal) case a plain int rather than every caller having to
    # handle both a scalar and a two-element shape.
    return {"equivalent": equivalent, "reason": reason,
            "tests_run": senior["tests_run"] if same_suite else None}


# ── dataset row ──────────────────────────────────────────────────────────────
def emit_dataset_row(dataset_path: str, row: dict) -> None:
    with open(dataset_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"   dataset row appended → {dataset_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dual-Track Implementation Engine")
    ap.add_argument("--feature", required=True)
    ap.add_argument("--stack", default="java-jakarta")
    ap.add_argument("--baseline", required=True, help="baseline repo dir to rsync per track")
    ap.add_argument("--out-prefix", required=True, help="workspace prefix; -senior/-junior appended")
    ap.add_argument("--senior-model", required=True)
    ap.add_argument("--junior-model", required=True)
    ap.add_argument("--maven-test-args", default=os.getenv("DUAL_MAVEN_TEST_ARGS", ""),
                    help="extra mvn args for the equivalence gate, e.g. '-pl a,b -am'")
    ap.add_argument("--dataset", default=os.getenv("DUAL_DATASET", "perf_dataset.jsonl"))
    ap.add_argument("--feature-id", default=os.getenv("DUAL_FEATURE_ID", None))
    ap.add_argument("--reuse-senior", default=os.getenv("DUAL_REUSE_SENIOR", ""),
                    help="path to an ALREADY-COMPLETED senior workspace to reuse as the "
                         "senior track: skips Phase A entirely (no re-staging, no LLM cost) "
                         "and freezes this tree's contract+tests for the junior. Use when a "
                         "clean senior run already exists (e.g. a prior probe) so a "
                         "token-constrained validation only pays for the junior track.")
    ap.add_argument("--junior-brief", default=os.getenv("DUAL_JUNIOR_BRIEF", ""),
                    help="text appended to the JUNIOR track's implementation brief only "
                         "(senior gets the clean feature). Use to induce a guaranteed "
                         "performance regression when both tracks share a model — the "
                         "junior still implements the frozen contract and must pass the "
                         "frozen suite, so it stays functionally equivalent (worse-but-"
                         "working), only its internals are made inefficient.")
    args = ap.parse_args()

    reuse_senior = args.reuse_senior.strip()
    if reuse_senior and not os.path.isdir(reuse_senior):
        print(f"   ⚠️ --reuse-senior '{reuse_senior}' not a directory — falling back to a "
              "fresh senior track", flush=True)
        reuse_senior = ""
    senior_out = reuse_senior or f"{args.out_prefix}-senior"
    junior_out = f"{args.out_prefix}-junior"
    feature_id = args.feature_id or args.out_prefix
    mvn_args = args.maven_test_args.split() if args.maven_test_args else []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")

    print(f"\n{'='*60}\n  Dual-Track Engine — {feature_id}\n"
          f"  senior={args.senior_model}  junior={args.junior_model}\n{'='*60}")
    # Profiling silently off would produce perf-less rows for a PERFORMANCE dataset — be loud.
    if os.getenv("DUAL_KIEKER", "1") == "1" and not _KIEKER_AVAILABLE:
        print("   ⚠️ DUAL_KIEKER=1 but profiling/ modules failed to import — "
              "rows will carry profiled=False and no local_regression", flush=True)

    # ── Phase A: SENIOR (plans, authors tests, implements) ──
    if reuse_senior:
        print(f"\n▶ Phase A/E: SENIOR track REUSED from {reuse_senior} "
              "(skipped — no LLM cost)")
        os.environ.setdefault("AGILE_BASELINE_REF", os.path.abspath(args.baseline))
        rc_senior, senior_abort = 0, False
    else:
        print("\n▶ Phase A/E: SENIOR track (plan + author tests + implement)")
        stage_workspace(args.baseline, senior_out)
        # Senior authors the frozen test suite by default (DUAL_SENIOR_AUTHOR_TESTS=1). Set to 0
        # for a machinery-validation run that judges equivalence on the pre-existing base tests
        # only (e.g. when the stack's test-authoring brief is not yet ported — TeaStore/jakarta).
        rc_senior, senior_abort = run_track(args.feature, args.stack, senior_out, args.senior_model,
                              extra_env={"AGILE_AUTHOR_TESTS": os.getenv("DUAL_SENIOR_AUTHOR_TESTS", "1")})

    # ── Phase B: freeze senior spec/tests, fork JUNIOR ──
    print("\n▶ Phase B/E: freeze senior contract+tests, fork junior")
    stage_workspace(args.baseline, junior_out)
    seeded = seed_frozen_spec(senior_out, junior_out)

    # ── Phase C: JUNIOR (same frozen spec+tests, small model) ──
    print("\n▶ Phase C/E: JUNIOR track (implement frozen spec, tests not re-authored)")
    # The junior implements senior's already-frozen contract.json (pipeline skips the
    # architect when a detailed contract exists), so an appended brief steers only the
    # cline IMPLEMENTATION, never the spec or the shared test suite — the row stays a
    # valid worse-but-working sample. Empty by default (pure scale-variable runs).
    junior_feature = args.feature
    junior_env = {"AGILE_AUTHOR_TESTS": "0",
                  # junior runs the senior's FROZEN contract — never let it
                  # re-derive/diverge (would break the equivalence invariant).
                  "AGILE_CONTRACT_ESCALATION": "0"}
    if args.junior_brief:
        junior_feature = f"{args.feature}\n\n{args.junior_brief}"
        # The junior SKIPS the architect (frozen contract), and the coder brief
        # (_build_cline_brief) is built from the contract, not the feature text — so a
        # brief appended to the feature alone never reaches the implementer. Pass it as an
        # implementation constraint the coder brief injects verbatim so it actually steers
        # the junior's internals (still correct, still passes the frozen suite).
        junior_env["AGILE_IMPL_CONSTRAINT"] = args.junior_brief
        print(f"   ↳ junior brief augmented (+{len(args.junior_brief)} chars, "
              "induced-inefficiency mode)")
    rc_junior, junior_abort = run_track(junior_feature, args.stack, junior_out, args.junior_model,
                          extra_env=junior_env)

    # ── Phase D: equivalence gate (same suite vs both) + Kieker profiling ──
    print("\n▶ Phase D/E: equivalence gate (run frozen suite against both)")
    # INTERLEAVED profiling (2026-07-26): senior and junior iterations alternate
    # (S,J,S,J,...) instead of all-senior-then-all-junior. Machine state drifts over the
    # ~10 min a pair takes; running the tracks in separate blocks makes that drift
    # perfectly confounded with track identity (measured: a 30% whole-run offset that
    # inverted a real regression's sign). Alternating spreads the drift across both tracks
    # so it becomes shared noise, which the control-method normalization then removes.
    # DUAL_INTERLEAVE=0 restores the old sequential behaviour.
    if _KIEKER_ENABLED and os.getenv("DUAL_INTERLEAVE", "1") == "1":
        senior_res, junior_res = mvn_test_interleaved(senior_out, junior_out, mvn_args)
    else:
        senior_res = mvn_test(senior_out, mvn_args, kieker=_KIEKER_ENABLED)
        junior_res = mvn_test(junior_out, mvn_args, kieker=_KIEKER_ENABLED)
    verdict = equivalence_verdict(senior_res, junior_res)
    _without_ops = lambda r: {k: v for k, v in r.items()
                              if k not in ("operations", "raw_times")}
    print(f"   senior: {_without_ops(senior_res)}")
    print(f"   junior: {_without_ops(junior_res)}")
    print(f"   ⚖️  equivalence: {verdict['equivalent']} — {verdict['reason']}")

    # Which .java files each track added/changed vs the staged baseline. Computed BEFORE the
    # comparison (not just for the exercised gate below) because its complement is the
    # run-offset CONTROL set: methods identical in both tracks, whose timing difference can
    # only be machine drift. See compare_kieker's run-offset normalization.
    sen_targets = jun_targets = None
    control_sigs = None
    if _KIEKER_ENABLED:
        sen_targets = _feature_targets(senior_out, args.baseline)
        jun_targets = _feature_targets(junior_out, args.baseline)
        control_sigs = _control_signatures(
            (senior_res.get("operations"), sen_targets),
            (junior_res.get("operations"), jun_targets))

    # local performance regression (junior vs senior), Component 3. apply_offset=False
    # (2026-07-30): the paper's own local measurement has no run-offset correction, and
    # doesn't need one — its 30 repetitions happen inside ONE Kieker session (confirmed
    # against their replication package), so there is no cross-invocation drift to correct
    # for. This project's harnesses now default to the same in-session protocol
    # (FeaturePerfExerciseTest's repeats), which removes the offset at its source. The
    # estimate is still computed and reported (run_offset_ratio/pct/offset_estimated) as a
    # contamination diagnostic: a nonzero one here, under the new protocol, means senior and
    # junior were NOT profiled cleanly in-session and is worth investigating on its own.
    local_regression = None
    if _KIEKER_ENABLED and senior_res.get("operations") and junior_res.get("operations"):
        cmp = compare_kieker.compare_summaries(
            senior_res["operations"], junior_res["operations"],
            threshold_pct=float(os.getenv("DUAL_KIEKER_THRESHOLD_PCT", "20")),
            min_delta_ms=float(os.getenv("DUAL_KIEKER_MIN_DELTA_MS", "0.5")),
            min_calls=int(os.getenv("DUAL_KIEKER_MIN_CALLS", "5")),
            baseline_raw=senior_res.get("raw_times"),
            candidate_raw=junior_res.get("raw_times"),
            control_sigs=control_sigs, apply_offset=False)
        compare_kieker.write_compare_csv(cmp, os.path.join(junior_out, "compare.csv"))
        local_regression = {k: cmp[k] for k in
                            ("threshold_pct", "min_delta_ms", "min_calls",
                             "run_offset_ratio", "run_offset_pct", "normalized",
                             "offset_estimated", "n_control_methods", "normalization_note",
                             "n_methods_compared", "n_regressions", "n_suspect",
                             "regressed", "suspect")}
        if local_regression["offset_estimated"]:
            print(f"   🎚️  run-offset diagnostic: {local_regression['run_offset_pct']:+.1f}% "
                  f"over {local_regression['n_control_methods']} unchanged control method(s) "
                  "— NOT divided out (paper-literal); a large value here means this pair "
                  "was not profiled cleanly in-session", flush=True)
        else:
            print(f"   ℹ️ run-offset not estimated — {local_regression['normalization_note']}",
                  flush=True)
        print(f"   📈 local regression: {local_regression['n_regressions']} method(s) "
              f"slower in junior (>= +{local_regression['threshold_pct']:.0f}%, "
              f"n>={local_regression['min_calls']}, 2σ), "
              f"{local_regression['n_suspect']} suspect")

    # feature-exercised gate: the profiled ops must include the feature's own code, else
    # local_regression compared identical baseline code and carries no senior-vs-junior signal.
    # sen_ex/jun_ex are kept SEPARATELY (persisted per-track below) because the combined
    # `feature_exercised` collapses to None whenever the junior fails to compile — which
    # loses the fact that the SENIOR was genuinely exercised. The chained-trunk advance gate
    # must key on the senior signal alone (senior code is what lands on the trunk), so a
    # broken junior no longer masks whether the trunk state was actually measured.
    feature_exercised = None
    sen_ex = jun_ex = None
    if _KIEKER_ENABLED:
        sen_ex = _feature_exercised(senior_res, sen_targets)
        jun_ex = _feature_exercised(junior_res, jun_targets)
        if sen_ex is not None and jun_ex is not None:
            feature_exercised = sen_ex and jun_ex
        if feature_exercised is False:
            print("   ⚠️ feature NOT exercised by the profiled suite "
                  f"(senior={sen_ex} junior={jun_ex}) — per-method perf covers baseline "
                  "code only; row carries no feature perf signal", flush=True)
            # Name the target method(s) the profiled suite never called, so the operator sees
            # WHICH feature code went untested (typically a new repository/endpoint method with
            # no authored test that invokes it — the chained-trunk test-authoring gap). Additive.
            def _uncalled(res, targets):
                called = set()
                for sig in (res.get("operations") or {}):
                    try:
                        called.add(_sig_class_method(sig))
                    except Exception:
                        pass
                miss = []
                for cls, methods in (targets or {}).items():
                    if methods is None:
                        if not any(c == cls for c, _ in called):
                            miss.append(f"{cls}.*")
                    else:
                        miss += [f"{cls}.{m}" for m in methods if (cls, m) not in called]
                return miss
            _miss = _uncalled(senior_res, sen_targets) or _uncalled(junior_res, jun_targets)
            if _miss:
                print(f"      uncalled feature target(s): {', '.join(sorted(set(_miss))[:8])} — "
                      "no authored test invokes this code (chained-trunk test-authoring)", flush=True)

    # ── Phase E: dataset row (real per-method perf from Component 3) ──
    print("\n▶ Phase E/E: dataset row")
    # An infra-abort (model quota exhausted) produces NO feature code, so the green/red build
    # result is meaningless — tag the row so the failure taxonomy separates it from a genuine
    # compile/test failure (and never admits it).
    infra_abort = senior_abort or junior_abort
    # Step-timeout taxonomy: a track whose authoring ran out its clock is an infra failure,
    # not a bad-code sample. Surfaced per-track (timed_out/step_timeouts) so a build_success=
    # False row driven by a timeout is distinguishable from a genuine compile/test failure.
    senior_timeouts = _step_timeouts(senior_out)
    junior_timeouts = _step_timeouts(junior_out)
    if junior_timeouts or senior_timeouts:
        _who = ", ".join(w for w, t in (("senior", senior_timeouts),
                                        ("junior", junior_timeouts)) if t)
        print(f"   ⏱️  step timeout(s) recorded on {_who} — a red build here is an infra "
              "timeout, not necessarily a bad-code sample", flush=True)
    status = "infra_abort" if infra_abort else "ok"
    chain = _chain_block(senior_res)
    trunk_regression = _trunk_regression(chain, senior_res, sen_targets)
    if trunk_regression is not None:
        print(f"   🌲 trunk drift vs {trunk_regression['parent_feature_id']}: "
              f"{trunk_regression['n_regressions']} pre-existing method(s) slower, "
              f"{trunk_regression['n_suspect']} suspect, "
              f"{len(trunk_regression['new_methods'])} new", flush=True)
    # A track that passes the frozen suite can still be undeployable — checked on BOTH,
    # since either track being unable to boot invalidates the pair (the senior is the
    # reference implementation and is what the system oracle load-tests as baseline).
    _deploy = {t: _deployability(o) for t, o in (("senior", senior_out),
                                                 ("junior", junior_out))}
    deployability = {
        "deployable": all(d["deployable"] for d in _deploy.values()),
        **{t: d for t, d in _deploy.items()},
    }
    if not deployability["deployable"]:
        for _t, _d in _deploy.items():
            if not _d["deployable"]:
                print(f"   🚫 {_t} is NOT deployable — {_d['reason']}", flush=True)
    row = {
        "feature_id": feature_id,
        "feature": args.feature,
        "stack": args.stack,
        "started_at": started,
        "status": status,
        "chain_baseline": args.baseline,
        "chain": chain,
        "trunk_regression": trunk_regression,
        "contract_frozen": seeded["contract"],
        "shared_test_files": seeded["test_files"],
        "senior": {
            # None when the senior is REUSED: no senior LLM call happened in this run, so
            # args.senior_model describes nothing that occurred here (reused seniors
            # were authored by 22x glm-5.2 + 3x glm-5.3, not by the runner's default).
            # `reused_from` below points at the workspace that IS the source of truth.
            "model": None if reuse_senior else args.senior_model, "out_dir": senior_out,
            "pipeline_rc": rc_senior, "infra_abort": senior_abort, **_without_ops(senior_res),
            "perf": _perf_from_ops(senior_res),
            "feature_exercised": sen_ex,
            "step_timeouts": senior_timeouts,
            "timed_out": senior_timeouts is not None,
            "reused_from": reuse_senior or None,
            "call_graph": _call_graph_summary(os.path.join(senior_out, _profile_dirname())),
        },
        "junior": {
            "model": args.junior_model, "out_dir": junior_out,
            "pipeline_rc": rc_junior, "infra_abort": junior_abort, **_without_ops(junior_res),
            "perf": _perf_from_ops(junior_res),
            "feature_exercised": jun_ex,
            "step_timeouts": junior_timeouts,
            "timed_out": junior_timeouts is not None,
            "induced_brief": args.junior_brief or None,
            "injected_inefficiencies": _injected_inefficiencies(junior_out),
            "call_graph": _call_graph_summary(os.path.join(junior_out, _profile_dirname())),
        },
        # The junior track is always the deliberately-degraded one. `junior.induced_brief`
        # carries the brief text (None when none was given); `injected_inefficiencies.json`
        # is its self-report where a catalog brief was used.
        #
        # NOT derived from `args.junior_brief`: the three earliest canonical features
        # (copurchase, dupes, volatility) pre-date the anti-pattern catalog and carry
        # induced_brief=None, yet they WERE induced (instructed slower-but-correct in prose).
        # It is an explicit knob instead: the default keeps every canonical run
        # `induced` byte-for-byte, and a genuinely un-briefed arm sets it (see
        # scripts/run_openrouter_junior.sh, which pairs it with DUAL_JUNIOR_BRIEF="").
        "regression_source": os.getenv("DUAL_REGRESSION_SOURCE", "induced"),
        "equivalence": verdict,
        **_system_oracle_row(junior_out),
        "local_regression": local_regression,
        "feature_exercised": feature_exercised,
        "deployability": deployability,
        "admitted": _admit_row(verdict["equivalent"], infra_abort, feature_exercised,
                               os.getenv("DUAL_REQUIRE_FEATURE_EXERCISED", "1") == "1",
                               deployability["deployable"]),
    }
    # Stable identity for a single attempt: several rows can share a feature_id (retries and
    # scale-vs-induced samples both land in the same file), so a consumer needs a key that
    # distinguishes them.
    row["run_id"] = f"{feature_id}@{started}"
    emit_dataset_row(args.dataset, row)

    print(f"\n{'='*60}\n  DONE — admitted={row['admitted']} status={status} "
          f"feature_exercised={feature_exercised} "
          f"(senior_rc={rc_senior} junior_rc={rc_junior})\n{'='*60}")
    # exit 0 even on non-equivalence: a rejected pair is a valid orchestration outcome.
    return 0


if __name__ == "__main__":
    sys.exit(main())
