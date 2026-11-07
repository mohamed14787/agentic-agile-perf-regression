"""Tests for the generalized _vf_ceiling (2026-07-05): re-opens a PO 'accept' the initial
verdict never rejected, when a whole-tree deterministic check (duplicate-route or the new
contract-fidelity check) proves a defect. Validates against the real teastore-probe-stockledger
workspace where PersistenceConfig.java is genuine dead code the PO never flagged and that
survived to final acceptance before this fix.
"""
import os
import unittest

from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.pipeline import _vf_ceiling, _all_code_targets, _all_accepted_targets, _frozen_suite_msg

REAL_WS = "teastore-probe-stockledger"


class TestAllCodeTargets(unittest.TestCase):

    def test_selects_code_producing_types_only(self):
        backlog = [
            {"id": "ENT-Foo", "type": "entity"},
            {"id": "DTO-Bar", "type": "dto"},
            {"id": "S3-1", "type": "test"},
            {"id": "ARCH-1", "type": "architecture"},
        ]
        accepted = ["ENT-Foo", "DTO-Bar", "S3-1", "ARCH-1"]
        self.assertEqual(sorted(_all_code_targets(backlog, accepted)), ["DTO-Bar", "ENT-Foo"])


class TestVfCeilingContractFidelity(unittest.TestCase):

    def test_clean_tree_is_a_noop(self):
        # Genuinely clean: Foo is grandfathered (in baseline) so it isn't new, and Api
        # references it, so even if it weren't grandfathered it wouldn't be orphaned.
        rel = "src/main/java/com/example/Foo.java"
        with_tmp = self._tmp_ws({
            rel: "package com.example; public class Foo {}",
            "src/main/java/com/example/Api.java":
                'package com.example; @Path("api") public class Api { Foo f; }',
        }, contract={}, baseline={rel: [0, 0], "pom.xml": [0, 0]})
        bb, backlog = with_tmp
        summary = {"accepted": ["ENT-Foo"], "rejected": [], "disputed": []}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, {}, final=False)
        self.assertEqual(out["accepted"], ["ENT-Foo"])
        self.assertEqual(out["rejected"], [])

    def test_orphan_artifact_reopens_accept(self):
        with_tmp = self._tmp_ws({
            "src/main/java/com/example/Orphan.java":
                "package com.example; public class Orphan {}",
            "src/main/java/com/example/Api.java":
                'package com.example; @Path("api") public class Api {}',
        }, contract={}, baseline={"pom.xml": [0, 0]})
        bb, backlog = with_tmp
        summary = {"accepted": ["ENT-Orphan"], "rejected": [], "disputed": []}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, {}, final=False)
        self.assertNotIn("ENT-Orphan", out["accepted"])
        self.assertIn("ENT-Orphan", out["rejected"])
        self.assertIn("[VF ceiling]", out["notes"])

    def test_final_downgrades_to_disputed_not_rejected(self):
        with_tmp = self._tmp_ws({
            "src/main/java/com/example/Orphan.java":
                "package com.example; public class Orphan {}",
        }, contract={}, baseline={"pom.xml": [0, 0]})
        bb, backlog = with_tmp
        summary = {"accepted": ["ENT-Orphan"], "rejected": [], "disputed": []}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, {}, final=True)
        self.assertNotIn("ENT-Orphan", out["accepted"])
        self.assertIn("ENT-Orphan", out["disputed"])
        self.assertNotIn("ENT-Orphan", out["rejected"])

    def _tmp_ws(self, files: dict, contract: dict, baseline: dict):
        import tempfile
        td = tempfile.mkdtemp()
        open(os.path.join(td, "pom.xml"), "w").write("<project/>")
        for rel, content in files.items():
            p = os.path.join(td, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write(content)
        bb = Blackboard(td)
        bb.write("contract", contract)
        bb.write("baseline_src", baseline)
        backlog = [{"id": tid.replace("Orphan", "Orphan").replace("Api", "Api"),
                    "type": "entity", "target_file": rel}
                   for rel in files for tid in
                   [f"ENT-{os.path.basename(rel)[:-5]}"]]
        # Simplify: single-entry backlog matching the accepted id used per test.
        backlog = [{"id": f"ENT-{os.path.basename(rel)[:-5]}", "type": "entity",
                    "target_file": rel} for rel in files
                   if os.path.basename(rel) in ("Orphan.java", "Foo.java")]
        return bb, backlog


class TestFrozenSuiteMsg(unittest.TestCase):

    def test_clean_report_is_noop(self):
        self.assertEqual(_frozen_suite_msg({"tests_failed": 0, "issues": []}), "")
        self.assertEqual(_frozen_suite_msg({}), "")
        self.assertEqual(_frozen_suite_msg(None), "")

    def test_red_report_names_failing_files(self):
        report = {"tests_failed": 6, "issues": [
            {"file": "CustomerRfmSegmentRepositoryTest.java", "concern": "..."},
            {"file": "CustomerRfmSegmentEndpointTest.java", "concern": "..."},
        ]}
        msg = _frozen_suite_msg(report)
        self.assertIn("6 frozen-suite test(s)", msg)
        self.assertIn("CustomerRfmSegmentRepositoryTest.java", msg)
        self.assertIn("CustomerRfmSegmentEndpointTest.java", msg)


class TestAllAcceptedTargets(unittest.TestCase):

    def test_returns_every_accepted_id_unfiltered(self):
        # Unlike _all_code_targets, type/target_file are irrelevant: a task accepted
        # while the frozen suite is red is suspect regardless of what it produced.
        backlog = [{"id": "S1-T4", "type": "config"}, {"id": "DTO-Bar", "type": "dto"}]
        accepted = ["S1-T4", "DTO-Bar"]
        self.assertEqual(_all_accepted_targets(backlog, accepted), ["S1-T4", "DTO-Bar"])


class TestVfCeilingFrozenSuite(unittest.TestCase):
    """Regression guard for the 2026-08-02 rfm false-accept: a milestone whose only
    backlog task is a DTO must not sail through Sprint Review with 6 failing tests
    just because the failing test files aren't that task's target_file."""

    def _tmp_ws(self):
        import tempfile
        td = tempfile.mkdtemp()
        open(os.path.join(td, "pom.xml"), "w").write("<project/>")
        bb = Blackboard(td)
        bb.write("contract", {})
        bb.write("baseline_src", {"pom.xml": [0, 0]})
        return bb

    def test_red_suite_reopens_accept_with_no_owning_task(self):
        bb = self._tmp_ws()
        backlog = [{"id": "DTO-CustomerRfmSegment", "type": "dto",
                    "target_file": "CustomerRfmSegment.java"}]
        summary = {"accepted": ["DTO-CustomerRfmSegment"], "rejected": [], "disputed": []}
        test_report = {"tests_failed": 6, "issues": [
            {"file": "CustomerRfmSegmentRepositoryTest.java", "concern": "..."},
            {"file": "CustomerRfmSegmentEndpointTest.java", "concern": "..."},
        ]}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, test_report, final=False)
        self.assertNotIn("DTO-CustomerRfmSegment", out["accepted"])
        self.assertIn("DTO-CustomerRfmSegment", out["rejected"])
        self.assertIn("frozen-suite test(s) still failing", out["notes"])

    def test_final_downgrades_red_suite_accept_to_disputed(self):
        bb = self._tmp_ws()
        backlog = [{"id": "DTO-CustomerRfmSegment", "type": "dto",
                    "target_file": "CustomerRfmSegment.java"}]
        summary = {"accepted": ["DTO-CustomerRfmSegment"], "rejected": [], "disputed": []}
        test_report = {"tests_failed": 6, "issues": [
            {"file": "CustomerRfmSegmentRepositoryTest.java", "concern": "..."},
        ]}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, test_report, final=True)
        self.assertNotIn("DTO-CustomerRfmSegment", out["accepted"])
        self.assertIn("DTO-CustomerRfmSegment", out["disputed"])

    def test_green_suite_is_a_noop(self):
        bb = self._tmp_ws()
        backlog = [{"id": "DTO-CustomerRfmSegment", "type": "dto",
                    "target_file": "CustomerRfmSegment.java"}]
        summary = {"accepted": ["DTO-CustomerRfmSegment"], "rejected": [], "disputed": []}
        test_report = {"tests_failed": 0, "issues": []}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, test_report, final=False)
        self.assertEqual(out["accepted"], ["DTO-CustomerRfmSegment"])


class TestVfCeilingRealStockledgerWorkspace(unittest.TestCase):
    """Regression guard for the exact defect found in the 2026-07-05 probe run: the ceiling
    must catch PersistenceConfig.java as dead code on the real generated tree."""

    def setUp(self):
        if not os.path.isdir(REAL_WS):
            self.skipTest(f"{REAL_WS} workspace not present in this checkout")

    def test_real_workspace_orphan_caught(self):
        bb = Blackboard(REAL_WS)
        contract = bb.read("contract") or {}
        backlog = bb.read("blackboard").get("sprint_backlog", [])
        if not backlog:
            self.skipTest("no sprint_backlog recorded in this workspace")
        accepted = [t["id"] for t in backlog if t.get("type") in
                    ("entity", "dto", "controller", "endpoint", "repository", "service")]
        if not accepted:
            self.skipTest("no code-producing tasks recorded in this workspace")
        summary = {"accepted": accepted, "rejected": [], "disputed": []}
        out = _vf_ceiling(summary, backlog, bb, {"framework": "jax-rs"}, {}, final=True)
        self.assertTrue(out["disputed"] or out["rejected"],
                         "expected the known PersistenceConfig.java orphan to reopen at "
                         "least one accept on the real stockledger workspace")


if __name__ == "__main__":
    unittest.main()
