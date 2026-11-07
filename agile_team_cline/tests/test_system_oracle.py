"""Tests for the system-level oracle scaffolding (2026-07-08): oracle_compose.py (build-from-
source docker-compose generation) and profiling/system_oracle_stats.py (JTL parsing +
Wilcoxon/Cliff's-Delta verdict, reusing profiling/nonparametric_stats.py). No Docker/JMeter
required — these are all pure-Python unit tests against synthetic fixtures.
"""
import csv
import json
import os
import sys
import tempfile
import unittest

import yaml

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiling"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts"))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "profiling", "oracle"))

import system_oracle_stats as sos
import oracle_compose
import adapt_jmx


def _fake_workspace(modules):
    """A tmp dir with a Dockerfile under each given module-relative path."""
    ws = tempfile.mkdtemp()
    for rel in modules:
        d = os.path.join(ws, rel)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "Dockerfile"), "w") as fh:
            fh.write("FROM descartesresearch/teastore-base:latest\n")
    return ws


def _fake_base_compose(td):
    """Minimal 2-service compose (one buildable, one not) mirroring the shipped shape."""
    path = os.path.join(td, "base.yaml")
    compose = {
        "version": "3",
        "services": {
            "registry": {"image": "descartesresearch/teastore-registry", "expose": ["8080"]},
            "db": {"image": "descartesresearch/teastore-db", "expose": ["3306"],
                   "ports": ["3306:3306"]},
        },
    }
    with open(path, "w") as fh:
        yaml.safe_dump(compose, fh)
    return path


class TestOracleCompose(unittest.TestCase):

    def test_generates_build_context_for_app_services(self):
        with tempfile.TemporaryDirectory() as td:
            base = _fake_base_compose(td)
            ws = _fake_workspace(["services/tools.descartes.teastore.registry"])
            out = os.path.join(td, "out.yaml")
            orig_map = oracle_compose._SERVICE_MODULE
            oracle_compose._SERVICE_MODULE = {
                "registry": "services/tools.descartes.teastore.registry"}
            try:
                compose = oracle_compose.generate(ws, out, cpu_limit="0.5", base_compose=base)
            finally:
                oracle_compose._SERVICE_MODULE = orig_map
            self.assertNotIn("image", compose["services"]["registry"])
            context = compose["services"]["registry"]["build"]["context"]
            # Must be absolute — a relative path here double-prefixes once the compose file
            # is written INSIDE the workspace dir (Compose resolves relative build.context
            # against the compose FILE's dir, not the caller's cwd; confirmed by a live run).
            self.assertTrue(os.path.isabs(context))
            self.assertEqual(context, os.path.abspath(
                os.path.join(ws, "services/tools.descartes.teastore.registry")))
            self.assertEqual(
                compose["services"]["registry"]["deploy"]["resources"]["limits"]["cpus"],
                "0.5")

    def test_db_keeps_shipped_image_untouched(self):
        with tempfile.TemporaryDirectory() as td:
            base = _fake_base_compose(td)
            ws = _fake_workspace([])
            out = os.path.join(td, "out.yaml")
            orig_map = oracle_compose._SERVICE_MODULE
            oracle_compose._SERVICE_MODULE = {}
            try:
                compose = oracle_compose.generate(ws, out, base_compose=base)
            finally:
                oracle_compose._SERVICE_MODULE = orig_map
            self.assertEqual(compose["services"]["db"]["image"],
                             "descartesresearch/teastore-db")
            self.assertNotIn("build", compose["services"]["db"])

    def test_missing_dockerfile_raises_loud(self):
        with tempfile.TemporaryDirectory() as td:
            base = _fake_base_compose(td)
            ws = tempfile.mkdtemp()  # no Dockerfiles at all
            out = os.path.join(td, "out.yaml")
            orig_map = oracle_compose._SERVICE_MODULE
            oracle_compose._SERVICE_MODULE = {
                "registry": "services/tools.descartes.teastore.registry"}
            try:
                with self.assertRaises(FileNotFoundError):
                    oracle_compose.generate(ws, out, base_compose=base)
            finally:
                oracle_compose._SERVICE_MODULE = orig_map


def _write_jtl(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timeStamp", "elapsed", "label", "responseCode", "responseMessage",
                   "threadName", "dataType", "success", "failureMessage", "bytes"])
        for elapsed, label, success in rows:
            w.writerow([0, elapsed, label, "200", "OK", "t1", "text",
                       "true" if success else "false", "", "0"])


class TestSystemOracleStats(unittest.TestCase):

    def test_parse_jtl_keeps_only_successful_samples(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "r.jtl")
            _write_jtl(p, [(10, "Home", True), (20, "Home", False), (30, "Product", True)])
            parsed = sos.parse_jtl(p)
            self.assertEqual(parsed["Home"], [10])
            self.assertEqual(parsed["Product"], [30])

    def test_compute_verdict_flags_consistent_slowdown(self):
        baseline = {"Home": [100] * 20}
        candidate = {"Home": [250] * 20}
        v = sos.compute_verdict(baseline, candidate)
        self.assertEqual(v["system_verdict"], "regression")
        self.assertGreater(v["mpd_pct"], 0)

    def test_compute_verdict_no_regression_when_faster(self):
        baseline = {"Home": [250] * 20}
        candidate = {"Home": [100] * 20}
        v = sos.compute_verdict(baseline, candidate)
        self.assertEqual(v["system_verdict"], "none")

    def test_compute_verdict_insufficient_samples_is_unlabelled(self):
        v = sos.compute_verdict({}, {})
        self.assertEqual(v["system_verdict"], "unlabelled")

    def test_per_label_breakdown_present(self):
        baseline = {"Home": [100] * 10, "Product": [50] * 10}
        candidate = {"Home": [100] * 10, "Product": [200] * 10}
        v = sos.compute_verdict(baseline, candidate)
        self.assertIn("Home", v["per_label"])
        self.assertIn("Product", v["per_label"])
        self.assertEqual(v["per_label"]["Product"]["system_verdict"], "regression")
        self.assertEqual(v["per_label"]["Home"]["system_verdict"], "none")


def _write_jtl_ts(path, rows):
    """Like _write_jtl but with explicit timestamps, for the warmup-window cases."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timeStamp", "elapsed", "label", "responseCode", "responseMessage",
                   "threadName", "dataType", "success", "failureMessage", "bytes"])
        for ts, elapsed, label, success in rows:
            w.writerow([ts, elapsed, label, "200", "OK", "t1", "text",
                       "true" if success else "false", "", "0"])


class TestErrorRates(unittest.TestCase):
    """2026-08-19. parse_jtl drops failed samples, so a run where the app fell over under load
    produced a timing distribution indistinguishable from a healthy one — the failure count was
    visible only on the JMeter console and never reached the row. These pin the accounting."""

    def test_counts_failures_per_label_and_overall(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "oracle_results.jtl")
            _write_jtl(p, [(10, "Home", True), (20, "Home", False),
                           (30, "Feature", True), (40, "Feature", False),
                           (50, "Feature", False)])
            e = sos.jtl_error_stats(p)
            self.assertEqual(e["samples"], 5)
            self.assertEqual(e["failed"], 3)
            self.assertEqual(e["error_rate_pct"], 60.0)
            self.assertEqual(e["per_label"]["Feature"]["failed"], 2)
            self.assertAlmostEqual(e["per_label"]["Feature"]["error_rate_pct"], 66.667, places=2)
            self.assertEqual(e["per_label"]["Home"]["error_rate_pct"], 50.0)

    def test_clean_run_reports_zero_not_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "oracle_results.jtl")
            _write_jtl(p, [(10, "Home", True), (20, "Home", True)])
            e = sos.jtl_error_stats(p)
            self.assertEqual(e["failed"], 0)
            self.assertEqual(e["error_rate_pct"], 0.0)

    def test_warmup_window_excluded_per_repeat(self):
        """Each repeat restarts the stack, so each has its own ramp. A single cutoff over the
        merge would keep repeat 2's ramp errors and inflate the steady-state rate."""
        with tempfile.TemporaryDirectory() as td:
            base = os.path.join(td, "oracle_results")
            # repeat 1 starts at t=0, repeat 2 at t=1_000_000; each has a failed ramp sample
            # inside its own first 10 s and a clean steady sample after it.
            _write_jtl_ts(base + "_r1.jtl", [(0, 10, "Home", False), (20_000, 10, "Home", True)])
            _write_jtl_ts(base + "_r2.jtl",
                          [(1_000_000, 10, "Home", False), (1_020_000, 10, "Home", True)])
            open(base + ".jtl", "w").close()
            e = sos.jtl_error_stats(base + ".jtl", discard_warmup_s=10)
            self.assertEqual(e["samples"], 2)
            self.assertEqual(e["failed"], 0)

    def test_cli_attaches_error_rates_to_the_stats_file(self):
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl(b, [(100, "Feature", True)] * 10)
            _write_jtl(c, [(100, "Feature", True)] * 8 + [(1, "Feature", False)] * 2)
            out = os.path.join(td, "stats.json")
            argv = sys.argv
            sys.argv = ["system_oracle_stats.py", b, c, "--out", out]
            try:
                sos.main()
            finally:
                sys.argv = argv
            with open(out) as fh:
                d = json.load(fh)
            self.assertEqual(d["error_rates"]["baseline"]["failed"], 0)
            self.assertEqual(d["error_rates"]["candidate"]["failed"], 2)
            self.assertEqual(d["error_rates"]["candidate"]["error_rate_pct"], 20.0)


def _write_jtl_bytes(path, rows):
    """Like _write_jtl but with an explicit `bytes` value per row: (elapsed, label, success, nbytes)."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timeStamp", "elapsed", "label", "responseCode", "responseMessage",
                   "threadName", "dataType", "success", "failureMessage", "bytes"])
        for elapsed, label, success, nbytes in rows:
            w.writerow([0, elapsed, label, "200", "OK", "t1", "text",
                       "true" if success else "false", "", nbytes])


class TestResponseNonemptyGate(unittest.TestCase):
    """2026-08-23. Every gate before this one confirms a request was sent and returned
    success=true; none confirm the response carried data. This is how six feature endpoints
    went unnoticed for weeks, each returning an identical byte count across 5,000+ calls. These
    pin the new standing check: response_nonempty_gate flags a Feature label whose response
    size never varies at 30+ samples."""

    def test_constant_bytes_flagged_degenerate(self):
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Feature", True, 179)] * 30)
            _write_jtl_bytes(c, [(10, "Feature", True, 179)] * 30)
            gate = sos.response_nonempty_gate(sos.jtl_response_bytes(b), sos.jtl_response_bytes(c))
            self.assertFalse(gate["response_nonempty"])
            flagged = gate["response_nonempty_detail"]["flagged"]
            self.assertEqual(len(flagged), 2)  # both arms constant
            self.assertEqual(flagged[0]["bytes"], 179)

    def test_varied_bytes_pass(self):
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Feature", True, 200 + i) for i in range(30)])
            _write_jtl_bytes(c, [(10, "Feature", True, 200 + i) for i in range(30)])
            gate = sos.response_nonempty_gate(sos.jtl_response_bytes(b), sos.jtl_response_bytes(c))
            self.assertTrue(gate["response_nonempty"])
            self.assertEqual(gate["response_nonempty_detail"], {})

    def test_large_constant_payload_not_flagged(self):
        """A fixed id + fixed query params against a database that is never mutated during a
        run legitimately returns the exact same byte count every call, real data included
        (measured on this corpus up to 76,089B). Only a SMALL constant size is the empty
        signature; a large one is deterministic, not degenerate."""
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Feature", True, 38050)] * 30)
            _write_jtl_bytes(c, [(10, "Feature", True, 38050)] * 30)
            gate = sos.response_nonempty_gate(sos.jtl_response_bytes(b), sos.jtl_response_bytes(c))
            self.assertTrue(gate["response_nonempty"])

    def test_below_sample_floor_not_flagged(self):
        """A handful of coincidentally-identical sizes isn't yet evidence — matches the
        paper's own n=30 protocol used elsewhere in this pipeline."""
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Feature", True, 179)] * 5)
            _write_jtl_bytes(c, [(10, "Feature", True, 179)] * 5)
            gate = sos.response_nonempty_gate(sos.jtl_response_bytes(b), sos.jtl_response_bytes(c))
            self.assertTrue(gate["response_nonempty"])

    def test_no_feature_label_is_unscorable(self):
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Home", True, 500)] * 30)
            _write_jtl_bytes(c, [(10, "Home", True, 500)] * 30)
            gate = sos.response_nonempty_gate(sos.jtl_response_bytes(b), sos.jtl_response_bytes(c))
            self.assertIsNone(gate["response_nonempty"])

    def test_cli_attaches_response_bytes_and_gate_to_stats_file(self):
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "b.jtl")
            c = os.path.join(td, "c.jtl")
            _write_jtl_bytes(b, [(10, "Feature", True, 179)] * 30)
            _write_jtl_bytes(c, [(10, "Feature", True, 179)] * 30)
            out = os.path.join(td, "stats.json")
            argv = sys.argv
            sys.argv = ["system_oracle_stats.py", b, c, "--out", out]
            try:
                sos.main()
            finally:
                sys.argv = argv
            with open(out) as fh:
                d = json.load(fh)
            self.assertFalse(d["response_nonempty"])
            self.assertIn("response_bytes", d)
            self.assertTrue(d["response_bytes"]["baseline"]["Feature"]["constant_bytes"])


class TestOracleRowWhitelistGuard(unittest.TestCase):
    """Guards that all producer fields from system_oracle_stats reach the merged row.
    These tests pin that a NEW top-level key in system_oracle_stats output cannot ship
    unnoticed: it must be triaged into dual_track.ORACLE_STATS_KEY_DISPOSITION, and every
    "carry" key must actually reach `_system_oracle_row`'s output."""

    def _produced_stats(self, tmpdir):
        """A real stats.json, produced by the actual CLI (not hand-built), so this test fails
        the moment `main()` starts emitting a field the disposition table doesn't know about."""
        b = os.path.join(tmpdir, "b.jtl")
        c = os.path.join(tmpdir, "c.jtl")
        _write_jtl_bytes(b, [(100, "Feature", True, 210 + i) for i in range(30)])
        _write_jtl_bytes(c, [(150, "Feature", True, 210 + i) for i in range(30)])
        out = os.path.join(tmpdir, "stats.json")
        argv = sys.argv
        sys.argv = ["system_oracle_stats.py", b, c, "--out", out, "--target-rps", "50"]
        try:
            sos.main()
        finally:
            sys.argv = argv
        with open(out) as fh:
            return json.load(fh)

    def test_every_producer_key_is_triaged(self):
        from agile_team_cline import dual_track as dt
        with tempfile.TemporaryDirectory() as td:
            stats = self._produced_stats(td)
        untriaged = sorted(set(stats) - set(dt.ORACLE_STATS_KEY_DISPOSITION))
        self.assertEqual(untriaged, [],
                          f"system_oracle_stats emits key(s) {untriaged} not triaged in "
                          f"dual_track.ORACLE_STATS_KEY_DISPOSITION — decide 'carry' or "
                          f"'raw-detail' before this ships silently missing from every row.")

    def test_every_carry_key_reaches_the_row(self):
        from agile_team_cline import dual_track as dt
        with tempfile.TemporaryDirectory() as td:
            stats = self._produced_stats(td)
            ws = os.path.join(td, "ws-junior")
            os.makedirs(ws)
            with open(os.path.join(ws, "oracle_stats.json"), "w") as fh:
                json.dump(stats, fh)
            row = dt._system_oracle_row(ws)
        oracle = row["oracle"]
        missing = []
        for key, disposition in dt.ORACLE_STATS_KEY_DISPOSITION.items():
            if disposition != "carry" or key not in stats:
                continue
            row_key = dt._ORACLE_KEY_ROW_NAME.get(key, key)
            # system_verdict is deliberately hoisted to the row's top level (it's the primary
            # scorer input), not nested under "oracle" like everything else.
            container = row if key == "system_verdict" else oracle
            if row_key not in container:
                missing.append((key, row_key))
        self.assertEqual(missing, [],
                          f"producer key(s) marked 'carry' did not reach row['oracle']: {missing}")

    def test_load_generator_specifically_survives(self):
        """Checks that load_generator survives into the row."""
        from agile_team_cline import dual_track as dt
        with tempfile.TemporaryDirectory() as td:
            stats = self._produced_stats(td)
            self.assertIn("load_generator", stats)  # sanity: the producer did emit it
            ws = os.path.join(td, "ws-junior")
            os.makedirs(ws)
            with open(os.path.join(ws, "oracle_stats.json"), "w") as fh:
                json.dump(stats, fh)
            row = dt._system_oracle_row(ws)
        self.assertIn("load_generator", row["oracle"])
        self.assertIsNotNone(row["oracle"]["load_generator"])


class TestFeatureEndpointReachesTheChangedCode(unittest.TestCase):
    """2026-07-29. The workload drove TeaStore's stock browse path only, while every injected
    inefficiency lives in a NEW persistence repository method. Four rows were therefore given a
    system_verdict computed from traffic that never reached the changed code. These tests pin
    the three parts of the fix: the workload has a Feature sampler, its verdict is reported
    separately from the diluted pooled one, and a run without it is marked as such rather than
    passing for a clean null."""

    def test_feature_verdict_survives_dilution_by_browse_traffic(self):
        # Six browse samplers identical on both sides, one feature sampler 50% slower. Pooled
        # over seven labels the effect is negligible; on the feature label it is unmissable.
        baseline = {lab: [10] * 200 for lab in
                    ("Home", "Login", "LoginAction", "Category", "Product", "Logout")}
        candidate = dict(baseline)
        baseline["Feature"] = [40] * 200
        candidate = {lab: list(v) for lab, v in candidate.items()}
        candidate["Feature"] = [60] * 200
        v = sos.compute_verdict(baseline, candidate)
        self.assertEqual(v["system_verdict"], "none",
                         "pooled over the whole workload the effect is diluted — that is the "
                         "paper's question and its honest answer")
        self.assertEqual(v["feature_verdict"], "regression",
                         "the changed endpoint is unambiguously slower and must be reported")
        self.assertTrue(v["feature_exercised_by_workload"])
        self.assertEqual(v["feature_samples"], {"baseline_n": 200, "candidate_n": 200})

    def test_browse_only_run_is_marked_not_silently_none(self):
        # The exact shape of every row measured before the fix: no Feature sampler at all.
        baseline = {"Home": [10] * 50, "Product": [10] * 50}
        candidate = {"Home": [10] * 50, "Product": [10] * 50}
        v = sos.compute_verdict(baseline, candidate)
        self.assertFalse(v["feature_exercised_by_workload"],
                         "a workload that never called the feature must not read as evidence")
        self.assertIsNone(v["feature_verdict"])

    def test_workload_plan_declares_a_feature_sampler(self):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        with open(os.path.join(repo, "profiling", "oracle", "teastore_adapted.jmx"),
                  encoding="utf-8") as fh:
            plan = fh.read()
        self.assertIn('testname="Feature"', plan)
        self.assertIn("${featurepath}", plan)
        # Gated: an unset featurepath must not fire a request at the persistence root.
        self.assertIn('"${featurepath}" != ""', plan)

    def test_persistence_is_published_so_the_load_generator_can_reach_it(self):
        # The shipped compose file publishes webui and db only; the feature endpoints live on
        # persistence, which was unreachable from the host.
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "base.yaml")
            with open(path, "w") as fh:
                yaml.safe_dump({"version": "3", "services": {
                    "persistence": {"image": "descartesresearch/teastore-persistence",
                                    "expose": ["8080"]}}}, fh)
            ws = _fake_workspace(["services/tools.descartes.teastore.persistence"])
            out = os.path.join(td, "out.yaml")
            orig_map = oracle_compose._SERVICE_MODULE
            oracle_compose._SERVICE_MODULE = {
                "persistence": "services/tools.descartes.teastore.persistence"}
            try:
                compose = oracle_compose.generate(ws, out, base_compose=path)
            finally:
                oracle_compose._SERVICE_MODULE = orig_map
            ports = compose["services"]["persistence"].get("ports")
            self.assertIn(f"{oracle_compose.PERSISTENCE_HOST_PORT}:8080", ports)


class TestFeatureEndpointResolution(unittest.TestCase):
    """scripts/feature_endpoint.py — the ticket is the contract, the workspace is the truth,
    and a disagreement between them stops the run instead of producing a 404 workload."""

    def setUp(self):
        try:
            import feature_endpoint
        except ImportError:
            self.skipTest("scripts/feature_endpoint.py not present (target-specific script removed)")
        self.fe = feature_endpoint

    def test_resolves_ticket_path_to_persistence_rest_path(self):
        resolved = self.fe.resolve("teastore-hist-priceoutlier")
        self.assertEqual(resolved["ticket_path"], "/products/price-outliers")
        self.assertEqual(
            resolved["path"],
            "/tools.descartes.teastore.persistence/rest/products/price-outliers")

    def test_templated_path_gets_a_concrete_id(self):
        resolved = self.fe.resolve("teastore-hist-alsobought")
        self.assertNotIn("{id}", resolved["path"])
        self.assertTrue(resolved["path"].endswith("/products/1/also-bought"))

    def test_regex_constrained_template_still_verifies(self):
        """2026-08-19, found in oracle pre-flight. The resources declare
        `@Path("{id:[0-9][0-9]*}/substitutes")` while the ticket promises
        `/products/{id}/substitutes`; a literal comparison called that a missing endpoint and
        would have aborted the substitutes and alsobought pairs mid-batch. Shape, not spelling."""
        ws = tempfile.mkdtemp()
        d = os.path.join(ws, self.fe._ENDPOINT_DIR)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ProductEndpoint.java"), "w") as fh:
            fh.write('@Path("products")\nclass P {\n'
                     '  @GET @Path("{id:[0-9][0-9]*}/substitutes") public void s() {}\n}\n')
        resolved = self.fe.resolve("teastore-hist-substitutes", workspace=ws)
        self.assertTrue(resolved["verified_against_workspace"])
        self.assertTrue(resolved["path"].endswith("/products/1/substitutes"))

    def test_differently_named_template_still_verifies(self):
        """Same shape, different parameter name (`{product}` vs the ticket's `{id}`)."""
        ws = tempfile.mkdtemp()
        d = os.path.join(ws, self.fe._ENDPOINT_DIR)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ProductEndpoint.java"), "w") as fh:
            fh.write('@Path("{product}/substitutes")\n')
        self.assertTrue(
            self.fe.resolve("teastore-hist-substitutes", workspace=ws)["verified_against_workspace"])

    def test_normalisation_does_not_match_a_different_endpoint(self):
        """The loosened comparison must not turn every templated path into a wildcard."""
        ws = tempfile.mkdtemp()
        d = os.path.join(ws, self.fe._ENDPOINT_DIR)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "ProductEndpoint.java"), "w") as fh:
            fh.write('@Path("{id:[0-9][0-9]*}/also-bought")\n')
        with self.assertRaises(LookupError):
            self.fe.resolve("teastore-hist-substitutes", workspace=ws)

    def test_workspace_without_the_endpoint_raises(self):
        # A workspace that does not serve the promised @Path must abort the oracle: this is the
        # API-name divergence that cost feature 13 a full re-run, caught in seconds instead.
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(LookupError):
                self.fe.resolve("teastore-hist-priceoutlier", workspace=td)


class TestDualTrackOracleRow(unittest.TestCase):
    """dual_track._system_oracle_row: the additive, optional post-step merge (kept aside
    from the main dual_track flow — no Docker/JMeter run happens here, this only reads a
    stats.json IF scripts/run_oracle.sh already produced one)."""

    def test_no_stats_file_yields_honest_null_placeholder(self):
        from agile_team_cline.dual_track import _system_oracle_row
        with tempfile.TemporaryDirectory() as td:
            row = _system_oracle_row(td)
            self.assertIsNone(row["system_verdict"])
            self.assertFalse(row["oracle"]["available"])
            self.assertEqual(row["oracle"]["isolation"], "single-machine")

    def test_row_carries_whether_the_workload_reached_the_feature(self):
        import json
        from agile_team_cline.dual_track import _system_oracle_row
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "oracle_stats.json"), "w") as fh:
                json.dump({"system_verdict": "none", "mpd_pct": 1.0,
                           "feature_exercised_by_workload": True,
                           "feature_verdict": "regression", "feature_mpd_pct": 48.7,
                           "feature_samples": {"baseline_n": 600, "candidate_n": 600}}, fh)
            row = _system_oracle_row(td)
            self.assertTrue(row["oracle"]["feature_exercised_by_workload"])
            self.assertEqual(row["oracle"]["feature_verdict"], "regression")

    def test_legacy_stats_without_the_flag_default_to_not_exercised(self):
        # Rows measured before the workload fix must not read as "feature was exercised".
        import json
        from agile_team_cline.dual_track import _system_oracle_row
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "oracle_stats.json"), "w") as fh:
                json.dump({"system_verdict": "none", "mpd_pct": 7.26}, fh)
            row = _system_oracle_row(td)
            self.assertFalse(row["oracle"]["feature_exercised_by_workload"])
            self.assertIsNone(row["oracle"]["feature_verdict"])

    def test_existing_stats_file_merged_into_row(self):
        from agile_team_cline.dual_track import _system_oracle_row
        import json
        with tempfile.TemporaryDirectory() as td:
            stats = {"system_verdict": "regression", "p_value": 0.001,
                     "cliffs_delta": 0.6, "mpd_pct": 42.0}
            with open(os.path.join(td, "oracle_stats.json"), "w") as fh:
                json.dump(stats, fh)
            row = _system_oracle_row(td)
            self.assertEqual(row["system_verdict"], "regression")
            self.assertTrue(row["oracle"]["available"])
            self.assertEqual(row["oracle"]["p_value"], 0.001)


class TestAdaptJmx(unittest.TestCase):
    """profiling/oracle/adapt_jmx.py against the real shipped teastore.jmx source — regression
    guard for the 1h45m hang caused by empty HTTPSampler connect/response timeouts (2026-07-08:
    a single stalled backend call hung the whole JMeter run with zero log output)."""

    @classmethod
    def setUpClass(cls):
        oracle_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "profiling", "oracle")
        with open(os.path.join(oracle_dir, "teastore.jmx"), encoding="utf-8") as fh:
            cls.adapted = adapt_jmx.adapt(fh.read())

    def test_thread_group_swapped(self):
        self.assertNotIn("ConcurrencyThreadGroup", self.adapted)
        self.assertIn("Standard Thread Group", self.adapted)

    def test_no_sampler_left_with_empty_timeouts(self):
        self.assertNotIn('<stringProp name="HTTPSampler.connect_timeout"></stringProp>',
                         self.adapted)
        self.assertNotIn('<stringProp name="HTTPSampler.response_timeout"></stringProp>',
                         self.adapted)

    def test_every_sampler_has_bounded_timeouts(self):
        n_samplers = self.adapted.count('testclass="HTTPSamplerProxy"')
        n_connect = self.adapted.count('<stringProp name="HTTPSampler.connect_timeout">10000<')
        n_response = self.adapted.count(
            '<stringProp name="HTTPSampler.response_timeout">120000<')
        self.assertGreater(n_samplers, 0)
        self.assertEqual(n_connect, n_samplers)
        self.assertEqual(n_response, n_samplers)


if __name__ == "__main__":
    unittest.main()
