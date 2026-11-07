"""No-LLM regression suite for the profiling revision (2026-07-02).

Run:  .venv/bin/python -m agile_team_cline.tests.test_profiling
      (also runs under system python — no camel dependency in this suite)

P1  parse_kieker — record-type id resolved from kieker.map (not hardcoded $1);
    fallback to $1 when no map; parse_run pools across iter_NN dirs.
P2  compare_kieker — REGRESSION needs pct + abs + min_calls + 2-sigma noise guard;
    borderline cases become "suspect", never REGRESSION; NEW/REMOVED unchanged.
P3  dual_track._profile_summary — keeps the full distribution (stddev/p50/min/max),
    persists summary.csv; _perf_from_ops carries stddev/p50 into the dataset row.
P4  kieker_args — KIEKER_NO_ADD_OPENS=1 strips the Java-9+-only --add-opens flags
    (a Java 8 forked JVM refuses to start on them).
P6  feature_exercised gate — method-level diff vs baseline; admission rule.
P7  dual_track.stage_workspace source-snapshot guard (2026-07-02) — tar.gz written at
    staging, scripts/restore_workspace_sources.sh recovers MISSING files only, never
    overwrites a feature edit already on disk.
P12 deployability gate (2026-07-28) — a workspace that passes its unit tests but cannot boot
    (main-scope datasource override contradicting persistence.xml) is not admitted.
P13 chain row validity — a failed junior no longer marks a feature done (audit HIGH 5);
    chain_manifest advanced_at survives a re-upsert (audit HIGH 8).
P14 workload symmetry — a senior/junior pair profiled over different amounts of work is
    reported as invalid rather than silently normalized (audit BLOCKER 0).
P15 profiling workload (2026-07-29) — catalog-wide queries were timed against assertion-sized
    fixtures, so an injected inefficiency measured 0.06%; the exercise that fixes it is gated
    on a property so the frozen functional run is unchanged.
P16 in-session repeat blocking (2026-07-30) — local measurement aligned to Liao et al.'s
    actual protocol (30 repetitions INSIDE one Kieker session, confirmed against their own
    replication package, not 30 separate `mvn test` invocations): meta.json + rank-based
    rep_NN labeling in parse_kieker.py, repeats defaulting to 30, run-offset normalization
    off by default (compare_kieker's apply_offset), and a per-pair coin flip for which track
    leads a single-iteration profile (P11) instead of always senior-first.
"""
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiling"))
import parse_kieker
import compare_kieker
import kieker_args
from agile_team_cline.dual_track import _profile_summary, _perf_from_ops, stage_workspace

_fails = []


def check(name, cond):
    """Record a named assertion, and FAIL the enclosing test when it does not hold.

    The raise is load-bearing. Without it this helper only appended to `_fails`, which
    solely `main()` reads — so pytest, which calls the test_* functions directly and
    never reaches `main()`, saw every check pass regardless of `cond`. Two genuine
    failures (P16 here, T7 in test_trunk_chain) sat under a green `pytest -q` for as
    long as both runners existed. Script mode stays informative because the print
    happens before the raise."""
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        _fails.append(name)
        raise AssertionError(name)


def _write_dat(dirpath, rec_id, sig, samples_ns, map_id=None):
    """Write one synthetic Kieker text .dat (+ optional kieker.map) into dirpath."""
    os.makedirs(dirpath, exist_ok=True)
    if map_id is not None:
        with open(os.path.join(dirpath, "kieker.map"), "w") as f:
            f.write("$0=kieker.common.record.misc.KiekerMetadataRecord\n")
            f.write(f"{map_id}=kieker.common.record.controlflow.OperationExecutionRecord\n")
    with open(os.path.join(dirpath, "kieker-01.dat"), "w") as f:
        tin = 1000
        for ns in samples_ns:
            f.write(f"{rec_id};1751400000000000000;{sig};sess;1;{tin};{tin + ns};host;0;0\n")
            tin += ns + 10


def _write_dat_records(dirpath, records, rec_id="$1", map_id=None):
    """Write one synthetic .dat with explicit per-record traceId/eoi/ess (Upgrade A/B
    tests need real call-tree shape, not the flat rec_id=1/eoi=0/ess=0 `_write_dat` always
    emits). Each record: (sig, exec_ns, trace_id, eoi, ess)."""
    os.makedirs(dirpath, exist_ok=True)
    if map_id is not None:
        with open(os.path.join(dirpath, "kieker.map"), "w") as f:
            f.write("$0=kieker.common.record.misc.KiekerMetadataRecord\n")
            f.write(f"{map_id}=kieker.common.record.controlflow.OperationExecutionRecord\n")
        rec_id = map_id
    with open(os.path.join(dirpath, "kieker-01.dat"), "w") as f:
        tin = 1000
        for sig, ns, trace_id, eoi, ess in records:
            f.write(f"{rec_id};1751400000000000000;{sig};sess;{trace_id};"
                    f"{tin};{tin + ns};host;{eoi};{ess}\n")
            tin += ns + 10


def test_p1_parser():
    sig = "public java.util.List a.b.CRepository.getAll()"
    with tempfile.TemporaryDirectory() as td:
        # map says $2 is the OperationExecutionRecord — hardcoded $1 would parse nothing
        _write_dat(os.path.join(td, "a"), "$2", sig, [1000, 2000], map_id="$2")
        t = parse_kieker.parse_dir(os.path.join(td, "a"))
        check("P1 $2 records parsed via kieker.map", t.get(sig) == [1000, 2000])

        # no map at all -> $1 fallback (older runs)
        _write_dat(os.path.join(td, "b"), "$1", sig, [500])
        check("P1 $1 fallback without map",
              parse_kieker.parse_dir(os.path.join(td, "b")).get(sig) == [500])

        # $1 lines in a dir whose map declares $2 are ignored (metadata etc.)
        t = parse_kieker.parse_dir(os.path.join(td, "a"))
        check("P1 non-OER ids ignored", all(v > 0 for v in t.get(sig, [])))

    with tempfile.TemporaryDirectory() as td:
        _write_dat(os.path.join(td, "iter_01"), "$1", sig, [100], map_id="$1")
        _write_dat(os.path.join(td, "iter_02"), "$2", sig, [300], map_id="$2")
        pooled = parse_kieker.parse_run(td)
        check("P1 parse_run pools across iterations (mixed ids)",
              sorted(pooled.get(sig, [])) == [100, 300])


def _row(mean_ms, n, std_ms=0.01):
    return {"mean_ns": mean_ms * 1e6, "stddev_ns": std_ms * 1e6, "n_calls": n}


def test_p2_comparator():
    sig = "m()"
    # clean regression: +100%, +1ms, n=10 both sides, tight stddev
    r = compare_kieker.compare_summaries({sig: _row(1.0, 10)}, {sig: _row(2.0, 10)})
    check("P2 clean regression flagged", r["regressed"] == [sig] and r["n_suspect"] == 0)

    # single-call methods: big delta but n=1 -> suspect, never REGRESSION
    r = compare_kieker.compare_summaries({sig: _row(3.0, 1)}, {sig: _row(5.0, 1)})
    check("P2 n=1 delta is suspect not regression",
          r["n_regressions"] == 0 and r["suspect"] == [sig])

    # overlapping distributions: delta 1ms but stddev 5ms on both sides (n=5) -> suspect
    r = compare_kieker.compare_summaries({sig: _row(1.0, 5, std_ms=5.0)},
                                         {sig: _row(2.0, 5, std_ms=5.0)})
    check("P2 2-sigma noise guard blocks overlapping dists",
          r["n_regressions"] == 0 and r["suspect"] == [sig])

    # same delta with tight stddev -> real regression
    r = compare_kieker.compare_summaries({sig: _row(1.0, 5, std_ms=0.05)},
                                         {sig: _row(2.0, 5, std_ms=0.05)})
    check("P2 tight dists still flagged", r["regressed"] == [sig])

    # legacy summaries without stddev: guard is graceful (no crash, thresholds only)
    r = compare_kieker.compare_summaries({sig: {"mean_ns": 1e6, "n_calls": 10}},
                                         {sig: {"mean_ns": 2e6, "n_calls": 10}})
    check("P2 graceful without stddev", r["regressed"] == [sig])

    # below thresholds stays ok; NEW/REMOVED statuses intact
    r = compare_kieker.compare_summaries(
        {sig: _row(1.0, 10), "gone()": _row(1.0, 10)},
        {sig: _row(1.05, 10), "new()": _row(1.0, 10)})
    st = {x["operation_signature"]: x["status"] for x in r["rows"]}
    check("P2 small delta ok / NEW / REMOVED",
          st[sig] == "ok" and st["new()"] == "NEW" and st["gone()"] == "REMOVED")


def test_p3_dual_track_stats():
    sig = "public int a.b.CRepository.count()"
    with tempfile.TemporaryDirectory() as td:
        prof = os.path.join(td, "profile")
        _write_dat(os.path.join(prof, "iter_01"), "$1", sig, [1000, 2000, 3000], map_id="$1")
        ops, raw = _profile_summary(prof)
        row = ops.get(sig, {})
        check("P3 stddev/p50/min/max kept",
              all(k in row for k in ("stddev_ns", "p50_ns", "min_ns", "max_ns")))
        check("P3 pooled stats correct",
              row.get("n_calls") == 3 and row.get("mean_ns") == 2000.0
              and row.get("p50_ns") == 2000.0)
        check("P3 summary.csv persisted", os.path.isfile(os.path.join(prof, "summary.csv")))
        check("P3 raw per-call samples returned (Upgrade A)",
              sorted(raw.get(sig, [])) == [1000, 2000, 3000])
        check("P3 samples.csv persisted (Upgrade A)",
              os.path.isfile(os.path.join(prof, "samples.csv")))

        perf = _perf_from_ops({"operations": ops, "profile_iterations": 4})
        top = perf["top_methods"][0]
        check("P3 dataset row carries stddev/p50 + iteration count",
              "stddev_ms" in top and "p50_ms" in top and perf["profile_iterations"] == 4)


def test_p5_surefire_totals():
    from agile_team_cline.dual_track import _SUREFIRE_RE
    log = """
[INFO] Tests run: 0, Failures: 0, Errors: 0, Skipped: 0
[INFO] Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: 1.0 s - in tools.a.RepositoryTest
[ERROR] Tests run: 1, Failures: 0, Errors: 1, Skipped: 0, Time elapsed: 0.9 s <<< FAILURE! - in tools.a.OtherTest
[ERROR] Tests run: 2, Failures: 0, Errors: 1, Skipped: 0
"""
    runs = errs = 0
    for m in _SUREFIRE_RE.finditer(log):
        runs += int(m.group(1)); errs += int(m.group(3))
    check("P5 per-class lines only (no module-summary double count)",
          runs == 2 and errs == 1)


def test_p4_add_opens_guard():
    old = os.environ.pop("KIEKER_NO_ADD_OPENS", None)
    try:
        a = kieker_args.kieker_jvm_args("/tmp/x")
        check("P4 --add-opens present by default", "--add-opens" in a)
        os.environ["KIEKER_NO_ADD_OPENS"] = "1"
        a = kieker_args.kieker_jvm_args("/tmp/x")
        check("P4 --add-opens stripped for Java 8 forks",
              "--add-opens" not in a and any(x.startswith("-javaagent:") for x in a))
    finally:
        if old is None:
            os.environ.pop("KIEKER_NO_ADD_OPENS", None)
        else:
            os.environ["KIEKER_NO_ADD_OPENS"] = old


def test_p6_feature_exercised_gate():
    from agile_team_cline.dual_track import (_feature_targets, _sig_class_method,
                                             _feature_exercised, _admit_row)
    # signature -> (declaring class, method) (method + constructor forms)
    check("P6 _sig_class_method method",
          _sig_class_method("public java.util.List a.b.CategoryRepository.getBasketLift()")
          == ("CategoryRepository", "getBasketLift"))
    check("P6 _sig_class_method constructor",
          _sig_class_method("public a.b.LocalDateTimeAttributeConverter.<init>()")
          == ("LocalDateTimeAttributeConverter", "<init>"))

    # targets diff vs baseline: new file -> None; modified w/ added method -> {method};
    # body-only modification -> None; identical file absent
    with tempfile.TemporaryDirectory() as td:
        base, out = os.path.join(td, "base"), os.path.join(td, "out")
        body = "public class X {\n  public int oldMethod() { return 1; }\n}\n"
        for root in (base, out):
            os.makedirs(os.path.join(root, "src"))
            for fn in ("Same.java", "WithAdded.java", "BodyOnly.java"):
                with open(os.path.join(root, "src", fn), "w") as f:
                    f.write(body)
        with open(os.path.join(out, "src", "WithAdded.java"), "w") as f:
            f.write(body.replace("}\n", "  public long getCheapest() { return 2; }\n}\n", 1))
        with open(os.path.join(out, "src", "BodyOnly.java"), "w") as f:
            f.write(body.replace("return 1", "return 42"))
        with open(os.path.join(out, "src", "NewFile.java"), "w") as f:
            f.write("public class NewFile { public void go() { } }")
        t = _feature_targets(out, base)
        check("P6 identical file not a target", "Same" not in t)
        check("P6 new file -> any method counts", t.get("NewFile", "x") is None)
        check("P6 modified w/ added method -> only that method",
              t.get("WithAdded") == {"getCheapest"})
        check("P6 body-only change -> any method counts", t.get("BodyOnly", "x") is None)

    # exercised verdicts — the cheapest-dual false-True case is the key one
    base_ops = {"public int a.b.WithAdded.oldMethod()": {},
                "public int a.b.Same.n()": {}}
    check("P6 baseline method of feature class does NOT satisfy gate",
          _feature_exercised({"operations": base_ops},
                             {"WithAdded": {"getCheapest"}}) is False)
    feat_ops = dict(base_ops, **{"public long a.b.WithAdded.getCheapest()": {}})
    check("P6 added feature method satisfies gate",
          _feature_exercised({"operations": feat_ops},
                             {"WithAdded": {"getCheapest"}}) is True)
    check("P6 body-only class satisfies via any method",
          _feature_exercised({"operations": {"public int a.b.BodyOnly.oldMethod()": {}}},
                             {"BodyOnly": None}) is True)
    check("P6 None when unprofiled (no verdict from absence)",
          _feature_exercised({"operations": {}}, {"WithAdded": None}) is None)

    # admission rule
    check("P6 admit: green + exercised", _admit_row(True, False, True, True))
    check("P6 reject: green but feature unmeasured", not _admit_row(True, False, False, True))
    check("P6 admit: unprofiled never blocks", _admit_row(True, False, None, True))
    check("P6 tag-only mode admits", _admit_row(True, False, False, False))
    check("P6 infra abort still rejects", not _admit_row(True, True, True, True))


def test_p7_src_snapshot_guard():
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _restore_sh = os.path.join(_repo_root, "scripts", "restore_workspace_sources.sh")
    with tempfile.TemporaryDirectory() as td:
        baseline = os.path.join(td, "baseline")
        out = os.path.join(td, "out")
        os.makedirs(os.path.join(baseline, "src", "main", "java", "a"))
        with open(os.path.join(baseline, "pom.xml"), "w") as f:
            f.write("<project/>\n")
        with open(os.path.join(baseline, "src", "main", "java", "a", "Foo.java"), "w") as f:
            f.write("class Foo { }\n")
        with open(os.path.join(baseline, "src", "main", "java", "a", "Bar.java"), "w") as f:
            f.write("class Bar { }\n")

        stage_workspace(baseline, out)
        snap = os.path.join(out, "src_snapshot.tar.gz")
        check("P7 snapshot written at staging", os.path.isfile(snap))

        # simulate the deleter: remove Foo.java entirely, and separately EDIT Bar.java
        # (a real feature change — must survive restore untouched).
        os.remove(os.path.join(out, "src", "main", "java", "a", "Foo.java"))
        bar_path = os.path.join(out, "src", "main", "java", "a", "Bar.java")
        with open(bar_path, "w") as f:
            f.write("class Bar { /* feature edit */ int x; }\n")

        r = subprocess.run(["zsh", _restore_sh, out], capture_output=True, text=True)
        check("P7 restore script exits 0", r.returncode == 0)
        check("P7 deleted file restored",
              os.path.isfile(os.path.join(out, "src", "main", "java", "a", "Foo.java")))
        check("P7 restored file matches original baseline content",
              open(os.path.join(out, "src", "main", "java", "a", "Foo.java")).read()
              == "class Foo { }\n")
        check("P7 existing (feature-edited) file NEVER overwritten",
              open(bar_path).read() == "class Bar { /* feature edit */ int x; }\n")

    # DUAL_SRC_SNAPSHOT=0 disables the snapshot
    with tempfile.TemporaryDirectory() as td:
        baseline = os.path.join(td, "baseline")
        out = os.path.join(td, "out")
        os.makedirs(baseline)
        with open(os.path.join(baseline, "pom.xml"), "w") as f:
            f.write("<project/>\n")
        old = os.environ.get("DUAL_SRC_SNAPSHOT")
        old_allow = os.environ.get("DUAL_ALLOW_EMPTY_BASELINE")
        os.environ["DUAL_SRC_SNAPSHOT"] = "0"
        # This sub-case uses a .java-free baseline only to exercise the snapshot toggle;
        # opt into the empty-baseline staging guard so it isn't the thing under test here.
        os.environ["DUAL_ALLOW_EMPTY_BASELINE"] = "1"
        try:
            stage_workspace(baseline, out)
        finally:
            if old is None:
                os.environ.pop("DUAL_SRC_SNAPSHOT", None)
            else:
                os.environ["DUAL_SRC_SNAPSHOT"] = old
            if old_allow is None:
                os.environ.pop("DUAL_ALLOW_EMPTY_BASELINE", None)
            else:
                os.environ["DUAL_ALLOW_EMPTY_BASELINE"] = old_allow
        check("P7 DUAL_SRC_SNAPSHOT=0 disables the snapshot",
              not os.path.isfile(os.path.join(out, "src_snapshot.tar.gz")))


def test_p8_nonparametric_guard():
    """P8 (2026-07-08): raw-sample Wilcoxon/Cliff's-Delta guard. Motivated by the live
    PERF-307-dualval finding — a +181.8% mean slowdown was held 'suspect' because JVM
    outliers pushed stddev above the mean, defeating the 2σ-on-means guard. Rank stats
    must flag that shape, AND must NOT flag an equal-distribution pair whose mean delta
    is a single outlier artifact (anti-overfit case)."""
    import nonparametric_stats as nps

    # -- unit vectors --
    p = nps.wilcoxon_rank_sum([1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18])
    check("P8 wilcoxon disjoint samples significant", p < 0.05)
    check("P8 wilcoxon identical samples p=1",
          nps.wilcoxon_rank_sum([5, 5, 5], [5, 5, 5]) == 1.0)
    d, lab = nps.cliffs_delta([1, 2, 3], [10, 11, 12])
    check("P8 cliffs disjoint = 1.0/large", d == 1.0 and lab == "large")
    d, lab = nps.cliffs_delta([1, 2, 3, 4], [1, 2, 3, 4])
    check("P8 cliffs identical = 0/negligible", d == 0.0 and lab == "negligible")

    def _summ(vals):
        import math
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        return {"mean_ns": mean, "stddev_ns": std, "n_calls": n}

    sig = "m()"
    # -- the real-world shape: consistent ~2.4x shift at every rank + heavy outliers.
    # 2σ on means fails here (stddev > mean); rank stats must call it REGRESSION.
    base = [1.0e6] * 13 + [8.0e6, 8.8e6]
    cand = [2.4e6] * 13 + [23.0e6, 28.5e6]
    r = compare_kieker.compare_summaries({sig: _summ(base)}, {sig: _summ(cand)},
                                         baseline_raw={sig: base},
                                         candidate_raw={sig: cand})
    check("P8 outlier-heavy consistent shift flagged via rank guard",
          r["regressed"] == [sig])
    r2 = compare_kieker.compare_summaries({sig: _summ(base)}, {sig: _summ(cand)})
    check("P8 same data via summaries-only 2σ stays suspect (documents the gap)",
          r2["n_regressions"] == 0 and r2["suspect"] == [sig])

    # -- anti-overfit: identical distributions except ONE giant outlier lifts the
    # candidate mean past every threshold. Rank guard must NOT call it a regression.
    base = [float(v) * 1e6 for v in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)]
    cand = [float(v) * 1e6 for v in (1, 2, 3, 4, 5, 6, 7, 8, 9)] + [40.0e6]
    r = compare_kieker.compare_summaries({sig: _summ(base)}, {sig: _summ(cand)},
                                         baseline_raw={sig: base},
                                         candidate_raw={sig: cand})
    check("P8 single-outlier mean delta NOT a regression under rank guard",
          r["n_regressions"] == 0)

    # -- faster candidate is never a regression regardless of stats
    base = [10.0e6] * 10
    cand = [1.0e6] * 10
    r = compare_kieker.compare_summaries({sig: _summ(base)}, {sig: _summ(cand)},
                                         baseline_raw={sig: base},
                                         candidate_raw={sig: cand})
    check("P8 faster candidate never flagged", r["n_regressions"] == 0)

    # -- rows carry the stats for auditability
    row = r["rows"][0]
    check("P8 p_value/cliffs_delta/effect present in rows",
          all(k in row for k in ("p_value", "cliffs_delta", "effect")))


def test_p9_kieker_upgrades():
    """P9 (2026-07-08): Upgrade A completion (iteration+trace_id kept, not just pooled) and
    Upgrade B (traceId/eoi/ess -> call graph). Validated against a hand-written multi-trace
    fixture with real depth transitions (0->1->2), mirroring the shape confirmed present in
    real trending .dat data (50 depth-0, 11 depth-1, 6 depth-2 records)."""
    root = "endpoint()"
    repo = "repo.get()"
    child = "repo.helper()"

    with tempfile.TemporaryDirectory() as td:
        # Trace A: endpoint -> repo.get -> repo.helper (depth 0,1,2) — a real call chain.
        # Trace B: same iteration, a second call to the flat repo.get alone (depth 0 only,
        # tests that a lone root doesn't spuriously inherit an edge from another trace).
        _write_dat_records(os.path.join(td, "iter_01"), [
            (root, 5000, "T1", 0, 0),
            (repo, 3000, "T1", 1, 1),
            (child, 1000, "T1", 2, 2),
            (repo, 2000, "T2", 0, 0),
        ], map_id="$1")
        # A second iteration reusing trace-id "T1" for a DIFFERENT, unrelated flat call —
        # proves iteration-scoped grouping (a bug here would fold it into iter_01's T1 trace).
        _write_dat_records(os.path.join(td, "iter_02"), [
            (repo, 4000, "T1", 0, 0),
        ], map_id="$1")

        # -- Upgrade A: tagged raw retention --
        tagged = parse_kieker.parse_run_tagged(td)
        check("P9 tagged samples grouped by signature",
              set(tagged.keys()) == {root, repo, child})
        check("P9 iteration labels preserved (not pooled away)",
              sorted(lbl for lbl, _ in tagged[repo]) == ["iter_01", "iter_01", "iter_02"])
        out_csv = os.path.join(td, "samples_tagged.csv")
        parse_kieker.write_raw_tagged(tagged, out_csv)
        check("P9 samples_tagged.csv persisted", os.path.isfile(out_csv))
        with open(out_csv) as f:
            header = f.readline().strip()
        check("P9 samples_tagged.csv header carries iteration+trace columns",
              header == "iteration,operation_signature,exec_time_ns")

        # -- Upgrade B: call-graph reconstruction --
        graph = parse_kieker.build_call_graph(td)
        edges = {(e["caller"], e["callee"]): e["calls"] for e in graph["edges"]}
        check("P9 depth 0->1 edge reconstructed", edges.get((root, repo)) == 1)
        check("P9 depth 1->2 edge reconstructed", edges.get((repo, child)) == 1)
        check("P9 entry points = every root-level (ess=0) signature",
              set(graph["entry_points"]) == {root, repo})
        check("P9 flat trace (T2) contributes no edges beyond the chain",
              len(edges) == 2)
        check("P9 iter_02's reused trace-id T1 does NOT merge with iter_01's T1 chain",
              (root, repo) in edges and edges[(root, repo)] == 1)  # not doubled to 2

        out_json = os.path.join(td, "traces.json")
        parse_kieker.write_traces(graph, out_json)
        check("P9 traces.json persisted", os.path.isfile(out_json))
        import json as _json
        with open(out_json) as f:
            reloaded = _json.load(f)
        check("P9 traces.json round-trips the same edge/entry-point shape",
              reloaded == graph)


def test_p10_run_offset_normalization():
    """P10 — run-offset normalization (2026-07-26).

    Replays the live reorder failure: a -30% whole-run machine offset made every method,
    including byte-identical controls, look faster and inverted the sign of a real +24%
    regression on the feature method (reported n_regressions=0 and admitted)."""
    import math

    def _summ(vals):
        n = len(vals)
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / n) if n > 1 else 0.0
        return {"mean_ns": mean, "stddev_ns": std, "n_calls": n}

    K = 0.695                      # the measured -30.5% whole-run offset
    ctrl_sigs = [f"c{i}()" for i in range(6)]
    feat = "getReplenishmentForecast()"
    base_raw, cand_raw, base, cand = {}, {}, {}, {}
    for i, s in enumerate(ctrl_sigs):                 # unchanged code: pure offset
        b = [(1.0 + 0.1 * i) * 1e6] * 12
        base_raw[s], cand_raw[s] = b, [v * K for v in b]
        base[s], cand[s] = _summ(base_raw[s]), _summ(cand_raw[s])
    b = [10.0e6] * 12                                  # feature: truly +24%, then offset
    base_raw[feat] = b
    cand_raw[feat] = [v * 1.24 * K for v in b]
    base[feat], cand[feat] = _summ(base_raw[feat]), _summ(cand_raw[feat])

    un = compare_kieker.compare_summaries(base, cand, baseline_raw=base_raw,
                                          candidate_raw=cand_raw)
    check("P10 unnormalized reproduces the live miss (regression read as faster)",
          un["n_regressions"] == 0 and un["normalized"] is False
          and un["rows"][0]["delta_pct"] < 0)

    # apply_offset=False (2026-07-30 default, paper-literal): the offset is estimated for
    # diagnostic purposes but NOT divided out — matches Liao et al.'s local measurement,
    # which never crosses a JVM boundary between repetitions (30 in-session, not 30 separate
    # invocations) and so has nothing to correct for. Only the ESTIMATE, not the correction,
    # is exercised by default here.
    diag = compare_kieker.compare_summaries(base, cand, baseline_raw=base_raw,
                                            candidate_raw=cand_raw,
                                            control_sigs=set(ctrl_sigs))
    check("P10 default (apply_offset=False): offset ESTIMATED but not applied",
          diag["normalized"] is False and diag["offset_estimated"] is True
          and diag["n_control_methods"] == 6
          and abs(diag["run_offset_ratio"] - K) < 0.01)
    check("P10 default: uncorrected delta still reads the masked (faster-looking) sign",
          diag["n_regressions"] == 0
          and [r for r in diag["rows"]
               if r["operation_signature"] == feat][0]["delta_pct"] < 0)

    no = compare_kieker.compare_summaries(base, cand, baseline_raw=base_raw,
                                          candidate_raw=cand_raw,
                                          control_sigs=set(ctrl_sigs),
                                          apply_offset=True)
    check("P10 apply_offset=True: offset estimated from controls (~-30.5%) and applied",
          no["normalized"] is True and no["offset_estimated"] is True
          and no["n_control_methods"] == 6
          and abs(no["run_offset_ratio"] - K) < 0.01)
    check("P10 masked regression recovered after normalization",
          no["regressed"] == [feat])
    frow = [r for r in no["rows"] if r["operation_signature"] == feat][0]
    check("P10 normalized delta ~+24%, raw delta preserved for audit",
          abs(frow["delta_pct"] - 24.0) < 1.5 and frow["delta_pct_raw"] < 0)
    check("P10 controls stay 'ok' and are flagged as controls",
          all(r["status"] == "ok" and r["is_control"]
              for r in no["rows"] if r["operation_signature"] in ctrl_sigs))

    # -- controls must not manufacture regressions when there is no offset
    flat_c = {s: _summ([v for v in base_raw[s]]) for s in ctrl_sigs}
    flat_c[feat] = _summ(base_raw[feat])
    r = compare_kieker.compare_summaries(base, flat_c, baseline_raw=base_raw,
                                         candidate_raw=base_raw,
                                         control_sigs=set(ctrl_sigs))
    check("P10 identical runs → offset 1.0, no regressions",
          abs(r["run_offset_ratio"] - 1.0) < 1e-6 and r["n_regressions"] == 0)

    # -- too few usable controls: report unnormalized + flagged, never half-corrected
    r = compare_kieker.compare_summaries(base, cand, baseline_raw=base_raw,
                                         candidate_raw=cand_raw,
                                         control_sigs={ctrl_sigs[0], ctrl_sigs[1]})
    check("P10 below min_controls stays unnormalized and says why",
          r["normalized"] is False and r["run_offset_ratio"] == 1.0
          and "min_controls" in r["normalization_note"])

    # -- a low-sample control is ignored by the estimator (min_calls)
    thin = dict(cand)
    thin[ctrl_sigs[0]] = {"mean_ns": 99.0e6, "stddev_ns": 0.0, "n_calls": 2}
    r = compare_kieker.compare_summaries(base, thin, control_sigs=set(ctrl_sigs))
    check("P10 controls below min_calls excluded from the offset estimate",
          r["n_control_methods"] == 5 and abs(r["run_offset_ratio"] - K) < 0.01)

    # -- back-compat: no control_sigs => byte-identical to the pre-normalization module
    a = compare_kieker.compare_summaries(base, cand, baseline_raw=base_raw,
                                         candidate_raw=cand_raw)
    check("P10 default path unchanged (ratio 1.0, raw == normalized delta)",
          a["run_offset_ratio"] == 1.0
          and all(r["delta_pct"] == r["delta_pct_raw"]
                  for r in a["rows"] if r["delta_pct"] is not None))


def test_p11_interleaved_profiling():
    """P11 — interleaved profiling alternates tracks and swaps the leading slot.

    The STARTING order (which track leads iteration 1) is a deterministic per-pair hash
    (`_leads_with_senior`, 2026-07-30) rather than always senior-first: with
    DUAL_KIEKER_ITERS now defaulting to 1, a hardcoded senior-first start would put senior
    in the cold slot on every pair in the 15-feature chain — the same kind of systematic,
    track-identity-aligned bias the run-offset normalization work was about, just moved
    from within-run to across-the-chain. The test computes the expected order from the same
    function under test rather than hardcoding one, so it stays correct regardless of which
    way a given pair's hash happens to fall."""
    import agile_team_cline.dual_track as dt

    calls = []
    orig_once, orig_attach = dt._mvn_once, dt._attach_profile
    dt._mvn_once = lambda out, args, iter_dir=None: (
        calls.append((os.path.basename(out), iter_dir and os.path.basename(iter_dir))),
        (0, "Tests run: 3, Failures: 0, Errors: 0, Skipped: 0 - in T\nBUILD SUCCESS"))[1]
    dt._attach_profile = lambda res, out, kept: dict(res, profile_iterations=kept)
    try:
        os.environ["DUAL_KIEKER_ITERS"] = "4"
        with tempfile.TemporaryDirectory() as td:
            s, j = os.path.join(td, "x-senior"), os.path.join(td, "x-junior")
            os.makedirs(s); os.makedirs(j)
            base_order = (["x-senior", "x-junior"] if dt._leads_with_senior(s)
                          else ["x-junior", "x-senior"])
            sr, jr = dt.mvn_test_interleaved(s, j, [])
    finally:
        dt._mvn_once, dt._attach_profile = orig_once, orig_attach
        os.environ.pop("DUAL_KIEKER_ITERS", None)

    # Two unprofiled verdict runs come FIRST (one per track, iter_dir=None), then the
    # profiled iterations. The verdict cannot come from a scoped run.
    verdict_calls = [c for c in calls if c[1] is None]
    timing_calls = [c for c in calls if c[1] is not None]
    check("P11 exactly one unprofiled verdict run per track, before any timing",
          len(verdict_calls) == 2 and calls[:2] == verdict_calls)

    tracks = [c[0] for c in timing_calls]
    reversed_order = list(reversed(base_order))
    expected = base_order + reversed_order + base_order + reversed_order
    check("P11 both tracks are timed every iteration (8 profiled runs over 4 iters)",
          len(timing_calls) == 8)
    check("P11 iterations alternate, leading track swaps each round, "
          "starting order is the per-pair coin flip under test",
          tracks == expected)
    check("P11 leading track is deterministic (same pair -> same start every call)",
          dt._leads_with_senior(s) == dt._leads_with_senior(s))
    check("P11 iteration dirs align across tracks",
          [c[1] for c in timing_calls] == ["iter_01"] * 2 + ["iter_02"] * 2
                                          + ["iter_03"] * 2 + ["iter_04"] * 2)
    check("P11 verdict from the unprofiled full-suite run, all iterations pooled",
          sr["tests_pass"] and jr["tests_pass"] and sr["tests_run"] == 3
          and sr["profile_iterations"] == 4 and jr["profile_iterations"] == 4)

    # a red suite on iteration 1 stops further profiling of the pair
    calls.clear()
    dt._mvn_once = lambda out, args, iter_dir=None: (
        calls.append(os.path.basename(out)),
        (1, "Tests run: 3, Failures: 1, Errors: 0, Skipped: 0 - in T\nBUILD FAILURE")
        if out.endswith("-junior") else
        (0, "Tests run: 3, Failures: 0, Errors: 0, Skipped: 0 - in T\nBUILD SUCCESS"))[1]
    dt._attach_profile = lambda res, out, kept: dict(res, profile_iterations=kept)
    try:
        os.environ["DUAL_KIEKER_ITERS"] = "5"
        with tempfile.TemporaryDirectory() as td:
            s, j = os.path.join(td, "x-senior"), os.path.join(td, "x-junior")
            os.makedirs(s); os.makedirs(j)
            sr, jr = dt.mvn_test_interleaved(s, j, [])
    finally:
        dt._mvn_once, dt._attach_profile = orig_once, orig_attach
        os.environ.pop("DUAL_KIEKER_ITERS", None)
    check("P11 a red track short-circuits BEFORE any timing — only the two verdict runs "
          "happen, and no profiled iteration is started",
          len(calls) == 2 and jr["tests_pass"] is False and sr["tests_pass"] is True)


def _mk_persistence_module(root, jpa_config_body, props_driver, xml_driver="com.mysql.jdbc.Driver"):
    """Minimal workspace: one main-scope JpaConfig, application.properties, persistence.xml,
    plus the upstream test-scope EMFManagerInitializer that legitimately calls the same API."""
    mod = os.path.join(root, "services", "tools.descartes.teastore.persistence")
    main_java = os.path.join(mod, "src", "main", "java", "tools", "descartes", "teastore",
                             "persistence", "repository")
    test_java = os.path.join(mod, "src", "test", "java", "tools", "descartes", "teastore",
                             "persistence", "repository")
    res = os.path.join(mod, "src", "main", "resources")
    meta = os.path.join(res, "META-INF")
    for d in (main_java, test_java, meta):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(main_java, "JpaConfig.java"), "w") as f:
        f.write(jpa_config_body)
    # Upstream test helper — present in EVERY workspace and must never trip the check.
    with open(os.path.join(test_java, "EMFManagerInitializer.java"), "w") as f:
        f.write("class EMFManagerInitializer {"
                " void s(){ EMFManager.configureEMFWithProperties(p); } }")
    if props_driver:
        with open(os.path.join(res, "application.properties"), "w") as f:
            f.write("# generated\njakarta.persistence.jdbc.driver=%s\n"
                    "jakarta.persistence.jdbc.url=jdbc:h2:mem:testdb\n" % props_driver)
    with open(os.path.join(meta, "persistence.xml"), "w") as f:
        f.write('<persistence><persistence-unit name="p"><properties>'
                '<property name="jakarta.persistence.jdbc.driver" value="%s"/>'
                '</properties></persistence-unit></persistence>' % xml_driver)
    return root


_ACTIVE = ("public final class JpaConfig {\n"
           "  static void init() { EMFManager.configureEMFWithProperties(loadJpaProperties()); }\n}")
_PASSIVE = ("public final class JpaConfig {\n"
            "  static final String UNIT = \"tools.descartes.teastore.persistence\";\n}")


def test_p12_deployability_gate():
    """P12 — an undeployable junior must not be admitted (2026-07-28, copurchase-junior).

    The defect is the CONJUNCTION of a main-scope datasource override and a properties file
    that contradicts persistence.xml. Either half alone is normal and must not fire."""
    import agile_team_cline.dual_track as dt

    with tempfile.TemporaryDirectory() as td:
        both = _mk_persistence_module(os.path.join(td, "both"), _ACTIVE, "org.h2.Driver")
        call_only = _mk_persistence_module(os.path.join(td, "call"), _ACTIVE,
                                           "com.mysql.jdbc.Driver")
        props_only = _mk_persistence_module(os.path.join(td, "props"), _PASSIVE, "org.h2.Driver")
        neither = _mk_persistence_module(os.path.join(td, "none"), _PASSIVE, None)

        d_both = dt._deployability(both)
        check("P12 fires when a main-scope override contradicts persistence.xml",
              d_both["deployable"] is False)
        check("P12 reason names the offending file, both drivers",
              "JpaConfig.java" in d_both["reason"] and "org.h2.Driver" in d_both["reason"]
              and "com.mysql.jdbc.Driver" in d_both["reason"])
        check("P12 does NOT fire when the override agrees with persistence.xml",
              dt._deployability(call_only)["deployable"] is True)
        check("P12 does NOT fire on an inert properties file (passive JpaConfig)",
              dt._deployability(props_only)["deployable"] is True)
        check("P12 does NOT fire on a stock workspace",
              dt._deployability(neither)["deployable"] is True)
        # The upstream test-scope caller is written into every fixture above; if src/test were
        # scanned, props_only and neither would both have flipped.
        check("P12 ignores the test-scope EMFManagerInitializer",
              dt._deployability(props_only)["deployable"] is True)

    check("P12 admission blocks on deployable=False",
          dt._admit_row(True, False, True, True, False) is False)
    check("P12 admission unaffected when deployable=True",
          dt._admit_row(True, False, True, True, True) is True)
    check("P12 unknown deployability never blocks (fail-open)",
          dt._admit_row(True, False, True, True, None) is True)


def test_p13_chain_row_validity():
    """P13 — HIGH 5/HIGH 8 of the 2026-07-26 audit."""
    import json
    import shutil

    lib = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts", "lib_trunk_chain.sh")
    if not (shutil.which("zsh") and os.path.isfile(lib)):
        check("P13 skipped (zsh or lib_trunk_chain.sh unavailable)", True)
        return

    def done(row):
        with tempfile.TemporaryDirectory() as td:
            ds = os.path.join(td, "d.jsonl")
            with open(ds, "w") as f:
                f.write(json.dumps(row) + "\n")
            # bare `python` is what the library invokes; point it at this interpreter
            shim = os.path.join(td, "bin")
            os.makedirs(shim)
            with open(os.path.join(shim, "python"), "w") as f:
                f.write('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
            os.chmod(os.path.join(shim, "python"), 0o755)
            env = dict(os.environ, DUAL_DATASET=ds,
                       PATH=shim + os.pathsep + os.environ.get("PATH", ""))
            r = subprocess.run(["zsh", "-c",
                                f'source "{lib}"; _feature_already_done f && echo DONE || echo RERUN'],
                               capture_output=True, text=True, env=env)
            return "DONE" in r.stdout

    base = {"feature_id": "f", "started_at": "t"}
    check("P13 a green, exercised, admitted row counts as done",
          done({**base, "admitted": True, "feature_exercised": True,
                "junior": {"tests_pass": True}}))
    check("P13 a FAILED junior no longer counts as done (HIGH 5)",
          not done({**base, "admitted": False, "feature_exercised": True,
                    "junior": {"tests_pass": False}}))
    check("P13 a non-equivalent but GREEN junior still counts as done",
          done({**base, "admitted": False, "feature_exercised": True,
                "junior": {"tests_pass": True}}))
    check("P13 a pre-field legacy row still counts as done",
          done({**base, "admitted": False}))
    check("P13 an uncompiled junior (feature_exercised=null) re-runs",
          not done({**base, "admitted": False, "feature_exercised": None,
                    "junior": {"tests_pass": False}}))

    # HIGH 8 — advanced_at must survive a re-upsert of the same sequence
    with tempfile.TemporaryDirectory() as td:
        shim = os.path.join(td, "bin")
        os.makedirs(shim)
        with open(os.path.join(shim, "python"), "w") as f:
            f.write('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable)
        os.chmod(os.path.join(shim, "python"), 0o755)
        man = os.path.join(td, "m.json")
        env = dict(os.environ, CHAIN_MANIFEST=man, TRUNK_DIR=td, DUAL_DATASET=td,
                   HIST_DIR=td, WORKSPACE_ROOT="ws",
                   PATH=shim + os.pathsep + os.environ.get("PATH", ""))
        subprocess.run(["zsh", "-c",
                        f'source "{lib}"; _update_chain_manifest 1 a genesis s.tar.gz'],
                       capture_output=True, env=env)
        first = json.load(open(man))["test_points"][0]["advanced_at"]
        subprocess.run(["zsh", "-c",
                        f'source "{lib}"; sleep 1.1; _update_chain_manifest 1 a genesis s.tar.gz'],
                       capture_output=True, env=env)
        doc = json.load(open(man))
        check("P13 advanced_at preserved across re-upsert (HIGH 8)",
              doc["test_points"][0]["advanced_at"] == first)
        subprocess.run(["zsh", "-c",
                        f'source "{lib}"; _update_chain_manifest 2 b a s2.tar.gz'],
                       capture_output=True, env=env)
        doc = json.load(open(man))
        check("P13 a genuinely new test point still gets stamped",
              len(doc["test_points"]) == 2 and doc["test_points"][1]["advanced_at"])


def test_p14_workload_symmetry():
    """P14 — an asymmetric senior/junior workload is reported, not silently normalized
    (2026-07-28 volatility: 1 iteration vs 5, 2 696 samples vs 338 294)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts"))
    import remeasure_chain as rc

    with tempfile.TemporaryDirectory() as td:
        sig = "public void A.m()"
        sym_s, sym_j = os.path.join(td, "sym-senior"), os.path.join(td, "sym-junior")
        for ws in (sym_s, sym_j):
            for i in (1, 2):
                _write_dat(os.path.join(ws, "profile", f"iter_0{i}"), "$1", sig,
                           [100] * 50, map_id="$1")
        check("P14 counts iterations from disk, not from the row",
              rc.profile_iterations(sym_s) == 2)
        why, vols = rc.check_symmetry("f", sym_s, sym_j)
        check("P14 a symmetric pair reports no asymmetry", why is None)

        asy_s, asy_j = os.path.join(td, "asy-senior"), os.path.join(td, "asy-junior")
        for i in (1, 2, 3, 4, 5):
            _write_dat(os.path.join(asy_s, "profile", f"iter_0{i}"), "$1", sig,
                       [100] * 200, map_id="$1")
        _write_dat(os.path.join(asy_j, "profile", "iter_01"), "$1", sig, [100] * 5,
                   map_id="$1")
        why, vols = rc.check_symmetry("f", asy_s, asy_j)
        check("P14 a 1-vs-5 iteration pair is reported as asymmetric", why is not None)
        check("P14 the report carries both replicate counts and the call totals",
              why and "senior=5" in why and "junior=1" in why and "calls" in why)
        check("P14 an empty profile counts as zero iterations",
              rc.profile_iterations(os.path.join(td, "nonexistent")) == 0)

        # --- 2026-08-07: the gate covers every profile dir a result is computed from, and
        # separates "the tracks ran a different number of times" (blocking) from "the junior
        # executed more calls" (reported — that is what an N+1 regression looks like).
        multi_s, multi_j = os.path.join(td, "multi-senior"), os.path.join(td, "multi-junior")
        for ws in (multi_s, multi_j):
            _write_dat(os.path.join(ws, "profile", "iter_01"), "$1", sig, [100] * 40,
                       map_id="$1")
        # equal replicates, junior does 3x the calls: the lift/profile_only shape
        _write_dat(os.path.join(multi_s, "profile_only", "iter_01"), "$1", sig, [100] * 30,
                   map_id="$1")
        _write_dat(os.path.join(multi_j, "profile_only", "iter_01"), "$1", sig, [100] * 90,
                   map_id="$1")
        why, vols = rc.check_symmetry("f", multi_s, multi_j)
        check("P14 profile_only is checked, not just profile", "profile_only" in vols)
        check("P14 call-volume skew at equal replicates does NOT block", why is None)
        check("P14 but the skew is reported", vols["profile_only"]["skewed"] is True
              and abs(vols["profile_only"]["call_ratio"] - 3.0) < 1e-6)
        check("P14 a balanced dir is not flagged skewed", vols["profile"]["skewed"] is False)

        # a consumed profile dir present on only one side blocks (the volatility shape, where
        # an intersection-based check returns an empty list and reports the pair as clean)
        one_s, one_j = os.path.join(td, "one-senior"), os.path.join(td, "one-junior")
        _write_dat(os.path.join(one_j, "profile", "iter_01"), "$1", sig, [100] * 10,
                   map_id="$1")
        os.makedirs(one_s, exist_ok=True)
        why, vols = rc.check_symmetry("f", one_s, one_j)
        check("P14 a one-sided consumed profile dir blocks",
              why is not None and "absent from the senior" in why)

        # an exploratory dir on one side only (churn's senior-only profile_largescale) is not
        # a validity defect and must not disqualify the pair
        exp_s, exp_j = os.path.join(td, "exp-senior"), os.path.join(td, "exp-junior")
        for ws in (exp_s, exp_j):
            _write_dat(os.path.join(ws, "profile", "iter_01"), "$1", sig, [100] * 20,
                       map_id="$1")
        _write_dat(os.path.join(exp_s, "profile_largescale", "iter_01"), "$1", sig,
                   [100] * 900, map_id="$1")
        why, vols = rc.check_symmetry("f", exp_s, exp_j)
        check("P14 a senior-only EXPLORATORY profile dir does not block", why is None)
        # main() iterates every entry and reads ["skewed"]; the one-sided branch does not
        # set that key, so an unguarded read crashed the whole run on this shape
        check("P14 every returned entry survives main()'s skew read",
              all(v.get("skewed") in (True, False, None) for v in vols.values()))

        # no profile data at all is missing data, not a clean pair
        why, vols = rc.check_symmetry("f", os.path.join(td, "none-s"),
                                      os.path.join(td, "none-j"))
        check("P14 no profile data anywhere is reported, not passed",
              why is not None and "nothing to compare" in why)


def test_p15_perf_exercise_workload():
    """P15 profiling workload (2026-07-29) — the frozen functional tests size their fixtures for
    readable assertions (four products in one category), so a catalog-wide query was timed
    against single-digit row counts and a cleanly injected N+1 measured 14527us vs 14536us:
    a 0.06% difference caused by the data, not by the code. FeaturePerfExerciseTest supplies a
    representative catalog, but ONLY under the profiling property — an ordinary `mvn test`,
    including the frozen-contract run that decides equivalence, must be untouched."""
    import agile_team_cline.dual_track as dt

    saved = {k: os.environ.get(k) for k in
             ("DUAL_PERF_EXERCISE", "DUAL_PERF_categories", "DUAL_PERF_repeats")}
    try:
        os.environ.pop("DUAL_PERF_EXERCISE", None)
        os.environ.pop("DUAL_PERF_categories", None)
        os.environ.pop("DUAL_PERF_repeats", None)
        args = dt._perf_exercise_args()
        check("P15 profiling runs enable the exercise by default",
              "-Dteastore.perf.exercise=true" in args)
        check("P15 in-session repeats default to 30 (Liao et al.'s in-session protocol, "
              "not this project's earlier 5 separate mvn invocations)",
              "-Dteastore.perf.repeats=30" in args)

        os.environ["DUAL_PERF_repeats"] = "12"
        check("P15 the repeat count can be overridden without a code change",
              "-Dteastore.perf.repeats=12" in dt._perf_exercise_args())
        os.environ.pop("DUAL_PERF_repeats")

        os.environ["DUAL_PERF_categories"] = "250"
        check("P15 a fixture dimension can be overridden without a code change",
              "-Dteastore.perf.categories=250" in dt._perf_exercise_args())
        os.environ.pop("DUAL_PERF_categories")

        os.environ["DUAL_PERF_EXERCISE"] = "0"
        check("P15 the old profiling workload is still reachable via DUAL_PERF_EXERCISE=0",
              dt._perf_exercise_args() == [])
        os.environ.pop("DUAL_PERF_EXERCISE")

        # The functional verdict must never see the property: a plain _mvn_once (no iter_dir)
        # is the equivalence run, and changing its workload would change what "equivalent" means.
        captured = []
        orig_run = dt._run
        dt._run = lambda cmd, **kw: (captured.append(list(cmd)), (0, ""))[1]
        try:
            with tempfile.TemporaryDirectory() as td:
                dt._mvn_once(td, [])
                check("P15 a non-profiled `mvn test` carries no perf-exercise property",
                      not any("teastore.perf" in a for a in captured[-1]))
                dt._mvn_once(td, [], iter_dir=os.path.join(td, "profile", "iter_01"))
                check("P15 a profiled iteration does carry it",
                      any("teastore.perf.exercise=true" in a for a in captured[-1]))
        finally:
            dt._run = orig_run
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exerciser = os.path.join(
        repo, "teastore-dataset", "teastore-trunk", "services",
        "tools.descartes.teastore.persistence", "src", "test", "java", "tools", "descartes",
        "teastore", "persistence", "FeaturePerfExerciseTest.java")
    if os.path.isfile(exerciser):
        check("P15 the trunk carries the perf-exercise workload", True)
        src = open(exerciser, encoding="utf-8").read()
        check("P15 the exercise is gated on the profiling property",
              'Boolean.getBoolean(ENABLED_PROPERTY)' in src)
        check("P15 a query that throws is reported, not asserted on (a red profiling run "
              "would be discarded rather than explained)",
              "catch (Throwable t)" in src)
        check("P15 queries are bound reflectively so one file compiles on every trunk point",
              "ClassNotFoundException" in src)
        check("P15 the Java default repeats matches the harnesses' default (30, "
              "Liao et al.'s in-session protocol)",
              '"teastore.perf.repeats", 30' in src)


def test_p16_in_session_repeat_blocking():
    """P16 (2026-07-30) — the local measurement now matches Liao et al.'s protocol: 30
    repetitions INSIDE one Kieker session, not 30 separate `mvn test` invocations (confirmed
    directly against their own replication package — see profiling/parse_kieker.py's
    `_read_repeats`/`parse_run_tagged` docstrings). This validates the mechanics dual_track.py
    and kieker_profile.sh rely on: a `meta.json` next to a single-session profile makes
    `parse_run_tagged` rank-rebucket "run"/"iter_01" into `repeats` blocks by execution
    order, while `build_call_graph` (which needs whole traces, not per-signature blocks)
    and old profiles without meta.json are completely unaffected."""
    import json as _json
    import agile_team_cline.dual_track as dt

    sig = "public java.util.List repo.Query()"
    # 12 UNIQUE, NON-MONOTONIC durations. _write_dat always advances `tin` forward by the
    # same rule regardless of duration, so the true execution order is exactly this list's
    # index order — deliberately decoupled from sorting by duration, so a rebucketing bug
    # that ranked by exec_ns instead of tin (execution order) would be caught here instead
    # of accidentally passing on a monotonic fixture.
    durations = [5000, 100, 4000, 200, 3000, 300, 2000, 400, 1000, 500, 6000, 50]

    with tempfile.TemporaryDirectory() as td:
        # A single in-session profile: one iter_01 dir, 12 calls of the same query, in order.
        _write_dat(os.path.join(td, "iter_01"), "$1", sig, durations, map_id="$1")

        # -- no meta.json (old profile, or exerciser off): unaffected, single "iter_01" block
        tagged = parse_kieker.parse_run_tagged(td)
        labels = [lbl for lbl, _ in tagged[sig]]
        check("P16 no meta.json -> single block, label unchanged from before this feature",
              len(set(labels)) == 1 and labels[0] == "iter_01" and len(labels) == 12)

        # -- meta.json present: rank-rebucketed into `repeats` blocks by execution order
        with open(os.path.join(td, "meta.json"), "w") as f:
            _json.dump({"repeats": 4, "protocol": "in-session"}, f)
        tagged = parse_kieker.parse_run_tagged(td)
        from collections import Counter
        counts = Counter(lbl for lbl, _ in tagged[sig])
        check("P16 with meta.json: rebucketed into exactly `repeats` blocks",
              len(counts) == 4 and set(counts) == {"rep_00", "rep_01", "rep_02", "rep_03"})
        check("P16 blocks are near-equal (12 calls / 4 repeats = 3 each)",
              all(n == 3 for n in counts.values()))
        # Bucketing follows EXECUTION order (tin), not duration: durations are unique here
        # (see `durations` above), so exec_ns -> label is a clean lookup. Index order in
        # `durations` IS execution order, since _write_dat advances tin forward regardless
        # of duration magnitude.
        by_exec_ns = {ns: lbl for lbl, ns in tagged[sig]}
        check("P16 first 3 calls BY EXECUTION ORDER land in rep_00 (not by duration — "
              "durations[0]=5000 is the LARGEST value but the FIRST call)",
              {by_exec_ns[durations[i]] for i in range(3)} == {"rep_00"})
        check("P16 last 3 calls by execution order land in rep_03",
              {by_exec_ns[durations[i]] for i in range(9, 12)} == {"rep_03"})
        check("P16 middle calls land in rep_01/rep_02 in execution order, not duration order",
              by_exec_ns[durations[3]] == "rep_01" and by_exec_ns[durations[5]] == "rep_01"
              and by_exec_ns[durations[6]] == "rep_02" and by_exec_ns[durations[8]] == "rep_02")

        # -- build_call_graph is untouched: it groups by (iteration, trace_id), and a
        # per-signature rebucket must never split a trace's own records apart.
        graph_before = parse_kieker.build_call_graph(td)
        check("P16 call-graph reconstruction ignores the rep_NN rebucketing entirely",
              graph_before == parse_kieker.build_call_graph(td))

        # -- genuine multi-invocation profile (old protocol): 2 real iter_NN dirs must NOT
        # be collapsed into rep_NN, even with meta.json present (only a true single session
        # is eligible for rebucketing).
        _write_dat(os.path.join(td, "iter_02"), "$1", sig, [500, 600, 700], map_id="$1")
        tagged2 = parse_kieker.parse_run_tagged(td)
        labels2 = sorted({lbl for lbl, _ in tagged2[sig]})
        check("P16 a genuine multi-invocation profile is left alone, not rebucketed",
              labels2 == ["iter_01", "iter_02"])

    # -- dual_track's meta.json writer agrees with the harness on the repeats value
    saved = {k: os.environ.get(k) for k in ("DUAL_PERF_EXERCISE", "DUAL_PERF_repeats")}
    try:
        os.environ.pop("DUAL_PERF_EXERCISE", None)
        os.environ["DUAL_PERF_repeats"] = "17"
        with tempfile.TemporaryDirectory() as td:
            profile_dir = os.path.join(td, "profile")
            dt._write_profile_meta(profile_dir)
            with open(os.path.join(profile_dir, "meta.json")) as f:
                meta = _json.load(f)
        # Assert the STABLE SUBSET, not dict equality. meta.json is v2 as of 2026-08-15 and
        # gained fields (only/test_scope/lead_track/timestamp/harness_fingerprint) that
        # measurement_gate.py audits; exact equality made every such addition a test failure
        # in a test that is about the repeat count. Pin what P16 is actually about — the
        # effective `repeats` and the protocol — plus the version, so a v3 is a deliberate
        # decision rather than a silent drift.
        check("P16 dual_track._write_profile_meta records the effective repeat count",
              meta.get("repeats") == 17 and meta.get("protocol") == "in-session")
        check("P16 meta.json is v2 and records what the run WAS (auditable by "
              "scripts/measurement_gate.py), not just its repeat count",
              meta.get("meta_version") == 2 and "only" in meta and "test_scope" in meta)

        os.environ["DUAL_PERF_EXERCISE"] = "0"
        with tempfile.TemporaryDirectory() as td:
            profile_dir = os.path.join(td, "profile")
            dt._write_profile_meta(profile_dir)
            check("P16 no meta.json written when the exerciser is off (nothing repeated "
                  "in-session to rebucket)",
                  not os.path.exists(os.path.join(profile_dir, "meta.json")))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_p17_scoped_run_never_feeds_the_verdict():
    """P17 (2026-08-16, Review audit) — the equivalence verdict must come from an UNSCOPED,
    UNPROFILED run of the whole frozen suite, never from a profiled iteration.

    Profiled runs are scoped to `-Dtest=FeaturePerfExerciseTest` so they are one JVM and one
    Kieker session (the D1 fix). The exerciser is a single @Test whose javadoc says "Never
    fails. A query that throws is counted and reported, not asserted on." Parsing surefire
    totals out of that run gives tests_run=1 on both tracks with no way to fail, so
    `equivalence_verdict` reduces to "both compiled" — against historical rows carrying up to
    274 tests. The D1 fix introduced exactly that regression and nothing caught it, because
    every node-level test passed; only the EDGE from profiling to the functional verdict was
    broken. This test is that edge."""
    import agile_team_cline.dual_track as dt

    saved = {k: os.environ.get(k) for k in ("DUAL_KIEKER_ITERS", "DUAL_PERF_only")}
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        # A PROFILED run is the one carrying the Kieker argLine. Both kinds now carry a -Dtest
        # flag with opposite meanings (exerciser inclusion vs large-scale exclusion), so keying
        # this stub on "-Dtest=" would make the verdict run answer as the 1-test exerciser.
        profiled = any("argLine" in a for a in cmd)
        # the scoped exerciser reports 1 test; the real frozen suite reports 274
        n = 1 if profiled else 274
        cls = "FeaturePerfExerciseTest" if profiled else "FrozenSuite"
        return (0, f"Tests run: {n}, Failures: 0, Errors: 0, Skipped: 0, "
                   f"Time elapsed: 1.0 s - in {cls}\nBUILD SUCCESS\n")

    orig_run, orig_summary = dt._run, dt._profile_summary
    try:
        os.environ["DUAL_KIEKER_ITERS"] = "1"
        os.environ.pop("DUAL_PERF_only", None)
        dt._run = fake_run
        dt._profile_summary = lambda d: ({}, {})
        with tempfile.TemporaryDirectory() as td:
            sen = os.path.join(td, "s"); jun = os.path.join(td, "j")
            os.makedirs(sen); os.makedirs(jun)
            sr, jr = dt.mvn_test_interleaved(sen, jun, [])

        check("P17 the verdict comes from the FULL frozen suite, not the 1-test exerciser",
              sr["tests_run"] == 274 and jr["tests_run"] == 274)
        verdict = dt.equivalence_verdict(sr, jr)
        check("P17 equivalence is decided on the full suite", verdict["equivalent"] is True
              and "274" in verdict["reason"])
        # Check that tests_run survives as a structured field rather than only prose in reason.
        check("P17b tests_run is a structured field, not only prose inside reason",
              verdict["tests_run"] == 274)
        mismatched = dt.equivalence_verdict({**sr, "tests_run": 10}, jr)
        check("P17b a test-count mismatch reports tests_run=None, not a misleading number",
              mismatched["tests_run"] is None)
        # Classify on the Kieker argLine, not on "-Dtest=" — BOTH kinds of run now carry a
        # -Dtest flag and they mean opposite things: the timing run INCLUDES the exerciser
        # (-Dtest=FeaturePerfExerciseTest), the verdict run EXCLUDES the large-scale perf
        # fixtures (-Dtest=!*LargeScale*Test). The argLine is what actually makes a run profiled.
        unscoped = [c for c in calls if not any("argLine" in a for a in c)]
        scoped = [c for c in calls if any("argLine" in a for a in c)]
        check("P17 exactly one unprofiled verdict run per track (2 total)", len(unscoped) == 2)
        check("P17 no verdict run carries the Kieker argLine",
              not any(any("argLine" in a for a in c) for c in unscoped))
        check("P17 the timing runs are still scoped to the exerciser",
              len(scoped) == 2
              and all(any("-Dtest=FeaturePerfExerciseTest" in a for a in c) for c in scoped))
        check("P17 the verdict run excludes the large-scale PERFORMANCE fixtures",
              all(any(a.startswith("-Dtest=!") for a in c) for c in unscoped))
        check("P17 the verdict run is capped so one pathological class cannot stall a batch",
              all(any("surefire.timeout" in a for a in c) for c in unscoped))

        # and the converse: a red frozen suite must skip timing altogether
        calls.clear()
        dt._run = lambda cmd, **kw: (calls.append(list(cmd)),
                                     (1, "Tests run: 274, Failures: 3, Errors: 0, "
                                         "Skipped: 0, Time elapsed: 1.0 s - in FrozenSuite"
                                         "\nBUILD FAILURE\n"))[1]
        with tempfile.TemporaryDirectory() as td:
            sen = os.path.join(td, "s"); jun = os.path.join(td, "j")
            os.makedirs(sen); os.makedirs(jun)
            sr2, _ = dt.mvn_test_interleaved(sen, jun, [])
        check("P17 a red frozen suite is reported as red (not masked by a passing exerciser)",
              sr2["tests_pass"] is False and sr2["tests_run"] == 274)
        check("P17 a red frozen suite skips timing entirely — no profiled run happens",
              not any(any("argLine" in a for a in c) for c in calls))
    finally:
        dt._run, dt._profile_summary = orig_run, orig_summary
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_p18_profile_dir_follows_the_workload():
    """P18 (2026-08-16, Review audit) — an ISOLATED run must write `profile_only/`, the
    directory run_paper_algorithm.py and measurement_gate.py both read by default. The engine
    hardcoded `profile/` for both workloads, so an isolated batch would have written its
    F6-valid measurements exactly where the gate and the sweep never look."""
    import agile_team_cline.dual_track as dt

    saved = {k: os.environ.get(k) for k in ("DUAL_PERF_only", "DUAL_PROFILE_DIR")}
    try:
        for k in ("DUAL_PERF_only", "DUAL_PROFILE_DIR"):
            os.environ.pop(k, None)
        check("P18 a full-sweep run still writes profile/", dt._profile_dirname() == "profile")
        os.environ["DUAL_PERF_only"] = "getFrequentBundles"
        check("P18 an isolated run writes profile_only/, where the gate and the sweep look",
              dt._profile_dirname() == "profile_only")
        os.environ["DUAL_PROFILE_DIR"] = "profile_paper"
        check("P18 DUAL_PROFILE_DIR overrides both", dt._profile_dirname() == "profile_paper")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_p19_feature_methods_match_the_exerciser():
    """P19 (2026-08-16, Review audit) — scripts/feature_methods.json and
    FeaturePerfExerciseTest.QUERIES must stay 1:1.

    feature_methods.json is hand-maintained and has three consumers (measurement_gate C1/C4,
    remeasure_chain's PERF_only, the F4 mapping) but no producer. Its own header already
    states the 1:1 requirement and why — "a feature missing from QUERIES is silently never
    driven" — but nothing enforced it, so the correspondence was a verified snapshot rather
    than an invariant. This is the enforcement. A feature added to the chain without a QUERIES
    entry fails here instead of producing a profile in which the feature method never appears
    (the F23-F25 class of defect: n_calls=11 incidental calls, gate-red on C2)."""
    import json as _json
    import re as _re

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mapping_path = os.path.join(repo, "scripts", "feature_methods.json")
    exerciser = os.path.join(
        repo, "teastore-dataset", "teastore-trunk", "services",
        "tools.descartes.teastore.persistence", "src", "test", "java", "tools", "descartes",
        "teastore", "persistence", "FeaturePerfExerciseTest.java")
    if not (os.path.isfile(mapping_path) and os.path.isfile(exerciser)):
        return
    check("P19 feature_methods.json exists (measurement_gate cannot run without it)",
          os.path.isfile(mapping_path))
    check("P19 the trunk carries the exerciser", os.path.isfile(exerciser))

    with open(mapping_path, encoding="utf-8") as fh:
        features = _json.load(fh)["features"]
    src = open(exerciser, encoding="utf-8").read()
    block = src[src.index("QUERIES = {"):]
    block = block[:block.index("\n\t};")]
    # each row is {"<Repository>", "<method>", ...}
    queries = set(_re.findall(r'\{"(\w+)",\s*"(\w+)"', block))

    mapped = {(v["repository"], v["method"]) for v in features.values()}
    missing = sorted(mapped - queries)
    extra = sorted(queries - mapped)
    check(f"P19 every mapped feature method is driven by the exerciser "
          f"(missing from QUERIES: {missing})", not missing)
    check(f"P19 the exerciser drives nothing the mapping does not know about "
          f"(unmapped QUERIES rows: {extra})", not extra)
    check("P19 the mapping is 1:1 — no two features claim the same method",
          len({v["method"] for v in features.values()}) == len(features))


def test_p20_second_pass_reuses_the_verdict():
    """P20 (2026-08-16) — a same-session second profiling pass must not re-run the frozen suite,
    and reusing the verdict must not drag the first pass's TIMINGS along with it.

    Each feature is profiled twice: isolated (the paper's propagation workload) and then the
    full sweep (trunk regression + the offset diagnostic). Both passes went through
    `mvn_test_interleaved`, so the sweep re-ran phase 1 — the whole unprofiled frozen suite on
    both tracks — to re-derive a verdict established minutes earlier against untouched trees.
    That is ~82 s per feature on a 125-test suite and grows to 274 tests late in the chain.

    The second check is the trap in the fix. By the time the isolated result is handed back it
    also carries `operations`/`raw_times` from that pass's profile, and copying the dict whole
    would seed the sweep with the ISOLATED run's timings — masked today only because
    `_attach_profile` overwrites them afterwards. `_SUREFIRE_FIELDS` narrows the copy to the
    verdict, so correctness does not rest on that call order."""
    import agile_team_cline.dual_track as dt

    saved = {k: os.environ.get(k) for k in ("DUAL_KIEKER_ITERS", "DUAL_PERF_only")}
    calls = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        # keyed on the Kieker argLine, not "-Dtest=" — see P17's stub for why
        profiled = any("argLine" in a for a in cmd)
        n = 1 if profiled else 274
        cls = "FeaturePerfExerciseTest" if profiled else "FrozenSuite"
        return (0, f"Tests run: {n}, Failures: 0, Errors: 0, Skipped: 0, "
                   f"Time elapsed: 1.0 s - in {cls}\nBUILD SUCCESS\n")

    orig_run, orig_summary = dt._run, dt._profile_summary
    try:
        os.environ["DUAL_KIEKER_ITERS"] = "1"
        os.environ.pop("DUAL_PERF_only", None)
        dt._run = fake_run
        # Stand in for a parsed profile so the reused result carries timing keys, as the real
        # one does — that is the state the narrow copy has to defend against.
        dt._profile_summary = lambda d: ({"sig": {"mean_ns": 1}}, {"sig": [1, 2, 3]})
        with tempfile.TemporaryDirectory() as td:
            sen = os.path.join(td, "s"); jun = os.path.join(td, "j")
            os.makedirs(sen); os.makedirs(jun)
            # unprofiled == no Kieker argLine; see P17 for why "-Dtest=" no longer separates them
            first = dt.mvn_test_interleaved(sen, jun, [])
            n_after_first = len([c for c in calls if not any("argLine" in a for a in c)])
            second = dt.mvn_test_interleaved(sen, jun, [], known_verdict=first)
            n_after_second = len([c for c in calls if not any("argLine" in a for a in c)])

        check("P20 first pass runs the frozen suite once per track", n_after_first == 2)
        check("P20 the second pass runs NO further unprofiled suite",
              n_after_second == n_after_first)
        check("P20 the reused verdict is the full-suite one (274, not the 1-test exerciser)",
              second[0]["tests_run"] == 274 and second[1]["tests_pass"] is True)
        check("P20 the second pass still profiles both tracks",
              len([c for c in calls if any("argLine" in a for a in c)]) == 4)

        # The narrow copy is guarded by an invariant rather than by behaviour, deliberately.
        # Feeding a polluted `operations` through the reuse path proves nothing:
        # `_attach_profile` overwrites that key on the way out, so a whole-dict copy passes such
        # a check too (verified by mutation). What IS checkable is the thing the copy depends
        # on — that `_SUREFIRE_FIELDS` still names exactly the verdict `_parse_surefire`
        # produces. If a field is added there and not here, reuse silently drops it; if a
        # timing key is ever added there, reuse silently carries it.
        produced = set(dt._parse_surefire("Tests run: 1, Failures: 0, Errors: 0, Skipped: 0, "
                                          "Time elapsed: 1.0 s - in X\nBUILD SUCCESS\n", 0))
        check("P20 _SUREFIRE_FIELDS names exactly the verdict _parse_surefire produces",
              produced == set(dt._SUREFIRE_FIELDS))
    finally:
        dt._run, dt._profile_summary = orig_run, orig_summary
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_p21_validity_never_reads_unmeasured_as_clean():
    """P21 (2026-08-16) — `validity.status` must separate "nothing was measured" from
    "measured, nothing found".

    Collapsing the two is the single error this corpus has made most often: the oracle made it
    for three features and the scorer made it for one, each time by reading a null as "no
    regression". A row that was never validly measured has to say so in a field a consumer can
    filter on, rather than leaving every consumer to re-derive validity from a scatter of
    nullable fields — which is where those bugs came from.

    Also asserts the noise floor is wired in: a missing replicate must degrade the row rather
    than silently pass, because "no false positives recorded" and "false positives never looked
    for" are the same distinction one level down."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "scripts"))
    import remeasure_chain as rc

    ok_rep = {"clean": True, "n_false_positives": 0}
    # offset_estimated=True is part of a CLEAN row: without it nothing checked whether the
    # session drifted. A shrunken sweep workload starved exactly this check through a whole
    # pilot while every other field looked clean, so its absence has to degrade the row.
    base = lambda **kw: {"local": {"n_control_flagged": 0, "offset_estimated": True},
                         "feature_exercised": True, "replicate": ok_rep, **kw}

    v = rc.validity_block(base(), {"parent_feature_id": "x"}, "m")
    check("P21 a clean, fully cross-checked pair is valid",
          v["status"] == "valid" and not v["reasons"])

    v = rc.validity_block(base(feature_exercised=None, replicate=None), None, "m")
    check("P21 a feature never observed running is 'unmeasured', NOT valid/clean",
          v["status"] == "unmeasured")
    check("P21 the unmeasured row says so in words a consumer cannot misread",
          any("NOT evidence of no regression" in r for r in v["reasons"]))

    v = rc.validity_block(base(asymmetry="replicates 1 vs 5", replicate=None), None, "m")
    check("P21 an asymmetric workload is invalid, not merely degraded",
          v["status"] == "invalid")

    v = rc.validity_block(base(replicate=None), {"parent_feature_id": "x"}, "m")
    check("P21 a missing noise floor degrades the row rather than passing silently",
          v["status"] == "degraded")

    v = rc.validity_block(base(replicate={"clean": False, "n_false_positives": 3}),
                          {"parent_feature_id": "x"}, "m")
    check("P21 false positives on identical source degrade the row",
          v["status"] == "degraded" and any("noise floor" in r for r in v["reasons"]))

    v = rc.validity_block(base(local={"n_control_flagged": 2, "offset_estimated": True}),
                          {"parent_feature_id": "x"}, "m")
    check("P21 flagged source-identical controls degrade the row (session drift)",
          v["status"] == "degraded")

    v = rc.validity_block(base(local={"n_control_flagged": 0, "offset_estimated": False}),
                          {"parent_feature_id": "x"}, "m")
    check("P21 a missing run-offset estimate degrades the row rather than reading as clean",
          v["status"] == "degraded" and any("drift" in r for r in v["reasons"]))

    # The gate's full report (checks/fingerprint/timestamp), not just the status word,
    # must survive into the row when the gate actually ran this session — and stay the
    # bare word when it didn't (recompute-only, or legacy caller).
    gd = {"ok": True, "checks": [{"id": "C1", "status": "PASS", "detail": "30 == 30"}],
         "fingerprint": "abc123", "checked_at": "2026-08-25T00:00:00"}
    v = rc.validity_block(base(gate_detail=gd), {"parent_feature_id": "x"}, "m")
    check("P22 a run that executed the gate carries its full report, not just 'green'",
          v["measurement_gate"] == {"status": "green", "checks": gd["checks"],
                                    "fingerprint": "abc123", "checked_at": gd["checked_at"]})

    gd_red = {**gd, "ok": False}
    v = rc.validity_block(base(gate_detail=gd_red), {"parent_feature_id": "x"}, "m")
    check("P22 a red gate's detail records status='red', not silently 'green'",
          v["measurement_gate"]["status"] == "red")

    v = rc.validity_block(base(), {"parent_feature_id": "x"}, "m")
    check("P22 no gate_detail (recompute-only / legacy) keeps the bare status word",
          v["measurement_gate"] == "green")

    check("P21 the F18/F19 brief split is recorded per feature",
          rc.junior_brief_protocol(18) == "v1-pure-invariants"
          and rc.junior_brief_protocol(19) == "v2-impure-invariants")


def main():
    test_p1_parser()
    test_p2_comparator()
    test_p3_dual_track_stats()
    test_p4_add_opens_guard()
    test_p5_surefire_totals()
    test_p6_feature_exercised_gate()
    test_p7_src_snapshot_guard()
    test_p8_nonparametric_guard()
    test_p9_kieker_upgrades()
    test_p10_run_offset_normalization()
    test_p11_interleaved_profiling()
    test_p12_deployability_gate()
    test_p13_chain_row_validity()
    test_p14_workload_symmetry()
    test_p15_perf_exercise_workload()
    test_p16_in_session_repeat_blocking()
    test_p17_scoped_run_never_feeds_the_verdict()
    test_p18_profile_dir_follows_the_workload()
    test_p19_feature_methods_match_the_exerciser()
    test_p20_second_pass_reuses_the_verdict()
    test_p21_validity_never_reads_unmeasured_as_clean()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
