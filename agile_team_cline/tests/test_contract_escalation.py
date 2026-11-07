"""Tests for the VF→architect contract escalation (2026-07-12).

A contract-fidelity VF failure (CONTRACT DRIFT / ORPHAN ARTIFACT naming a contract type) can't be
fixed by code refinement — the loop only edits code, VF verifies code against the contract. The
escalation asks the architect for a second look and, if it drops the phantom (only-shrink), applies
a bounded correction + purges the orphans. These tests stub the architect call so no cline/network
runs, and cover: detection, directive text, no-repeat guard, only-shrink guard, the env + once-per
-run caps, and the happy path (drop + purge).
"""
import os
import tempfile
import unittest

import agile_team_cline.ceremonies.planning as planning
from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.ceremonies import contract_escalation as ce

_STACK = {"name": "Jakarta EE", "language": "Java", "file_extension": ".java"}

_CONTRACT = {
    "entities": [
        {"name": "ProductCoPurchasePair",
         "fields": {"firstProductId": "long", "secondProductId": "long"}, "dto": True},
        {"name": "FirstProduct", "fields": {"id": "long"}, "table": "first_product"},
    ],
    "endpoints": [{"method": "GET", "path": "/products/co-purchase-matrix",
                   "response": "ProductCoPurchasePair"}],
    "feature": "co-purchase matrix",
}
_VF = {"ENT-FirstProduct": [
        "ORPHAN ARTIFACT: FirstProductRepository.java was created this run but nothing references "
        "FirstProductRepository. Wire it into the code that should use it, or delete it.",
        "CONTRACT DRIFT: FirstProduct.java exists but no other source file references FirstProduct"]}


def _bb(td, contract=_CONTRACT, vf=_VF):
    bb = Blackboard(td)
    bb.write("contract", contract)
    bb.write("vf_failures", vf)
    return bb


class TestDetection(unittest.TestCase):

    def test_implicated_types_finds_the_phantom(self):
        hits = ce._implicated_types(_VF, ["ENT-FirstProduct"], _CONTRACT)
        self.assertIn("FirstProduct", hits)
        self.assertNotIn("ProductCoPurchasePair", hits)  # not named in a fidelity message

    def test_no_implicated_when_no_fidelity_message(self):
        vf = {"T1": ["some unrelated compile note about FirstProduct"]}
        self.assertEqual(ce._implicated_types(vf, ["T1"], _CONTRACT), set())

    def test_directive_quotes_evidence_and_names_type(self):
        d = ce._build_directive(_VF, ["ENT-FirstProduct"], {"FirstProduct"})
        self.assertIn("FirstProduct", d)
        self.assertIn("ORPHAN ARTIFACT", d)
        self.assertIn("SECOND LOOK", d)


class TestEscalation(unittest.TestCase):

    def setUp(self):
        self._orig_env = os.environ.get("AGILE_CONTRACT_ESCALATION")
        self._orig_derive = planning._architect_derive
        self._orig_norm = planning.normalize_contract
        os.environ["AGILE_CONTRACT_ESCALATION"] = "1"
        planning.normalize_contract = lambda c, s, output_dir=None: c  # identity, deterministic

    def tearDown(self):
        planning._architect_derive = self._orig_derive
        planning.normalize_contract = self._orig_norm
        if self._orig_env is None:
            os.environ.pop("AGILE_CONTRACT_ESCALATION", None)
        else:
            os.environ["AGILE_CONTRACT_ESCALATION"] = self._orig_env

    def test_env_off_disables(self):
        os.environ["AGILE_CONTRACT_ESCALATION"] = "0"
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(ce.escalate_contract_defect(_bb(td), _STACK, None, ["ENT-FirstProduct"]))

    def test_no_implicated_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td, vf={"T1": ["plain failure"]})
            self.assertIsNone(ce.escalate_contract_defect(bb, _STACK, None, ["T1"]))

    def test_no_repeat_guard_when_architect_re_emits(self):
        # architect ignores the directive and returns the SAME contract (still has FirstProduct)
        planning._architect_derive = lambda *a, **k: _CONTRACT
        with tempfile.TemporaryDirectory() as td:
            res = ce.escalate_contract_defect(_bb(td), _STACK, None, ["ENT-FirstProduct"])
            self.assertFalse(res["revised"])

    def test_only_shrink_guard_rejects_added_types(self):
        # architect drops FirstProduct but ADDS a brand-new entity → too risky post-loop
        revised = {"entities": [
            {"name": "ProductCoPurchasePair", "fields": {"a": "long"}, "dto": True},
            {"name": "BrandNewThing", "fields": {"id": "long"}}],
            "endpoints": _CONTRACT["endpoints"]}
        planning._architect_derive = lambda *a, **k: revised
        with tempfile.TemporaryDirectory() as td:
            res = ce.escalate_contract_defect(_bb(td), _STACK, None, ["ENT-FirstProduct"])
            self.assertFalse(res["revised"])

    def test_happy_path_drops_and_purges(self):
        revised = {"entities": [
            {"name": "ProductCoPurchasePair", "fields": {"a": "long"}, "dto": True}],
            "endpoints": _CONTRACT["endpoints"]}
        planning._architect_derive = lambda *a, **k: revised
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td)
            # orphan artifacts the phantom spawned, plus a baseline file that must NOT be purged
            base = os.path.join(td, "src", "main", "java")
            os.makedirs(base, exist_ok=True)
            for fn in ("FirstProduct.java", "FirstProductRepository.java",
                       "FirstProductRepositoryTest.java", "ProductRepository.java"):
                open(os.path.join(base, fn), "w").write("class X {}")
            bb.write("baseline_src", {"src/main/java/ProductRepository.java": "x"})

            res = ce.escalate_contract_defect(bb, _STACK, None, ["ENT-FirstProduct"])
            self.assertTrue(res["revised"])
            self.assertIn("FirstProduct", res["dropped"])
            self.assertEqual(res["purged"], 3)  # the 3 First* files, not ProductRepository
            self.assertTrue(os.path.exists(os.path.join(base, "ProductRepository.java")))
            self.assertFalse(os.path.exists(os.path.join(base, "FirstProductRepository.java")))
            # contract updated; FirstProduct gone
            names = [e["name"] for e in bb.read("contract")["entities"]]
            self.assertNotIn("FirstProduct", names)

    def test_once_per_run_cap(self):
        planning._architect_derive = lambda *a, **k: {"entities": [
            {"name": "ProductCoPurchasePair", "fields": {"a": "long"}, "dto": True}],
            "endpoints": _CONTRACT["endpoints"]}
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td)
            first = ce.escalate_contract_defect(bb, _STACK, None, ["ENT-FirstProduct"])
            self.assertTrue(first["revised"])
            # second call is capped by the _contract_escalated flag
            self.assertIsNone(ce.escalate_contract_defect(bb, _STACK, None, ["ENT-FirstProduct"]))


# The Spring-config-orphan VF failure copurchase-2026-07-11 thrashed on: a cline-authored dead
# config class in a non-Spring reactor, flagged ORPHAN, that refinement tried to wire in.
_ORPHAN_VF = {"DTO-ProductCoPurchasePair": [
    "ORPHAN ARTIFACT: JpaConfig.java was created this run but nothing references JpaConfig. "
    "Wire it into the code that should use it, or delete it."]}


class TestOrphanPurge(unittest.TestCase):

    def setUp(self):
        self._orig = os.environ.get("AGILE_ORPHAN_PURGE")

    def tearDown(self):
        if self._orig is None:
            os.environ.pop("AGILE_ORPHAN_PURGE", None)
        else:
            os.environ["AGILE_ORPHAN_PURGE"] = self._orig

    def _seed(self, td, files):
        base = os.path.join(td, "src", "main", "java")
        os.makedirs(base, exist_ok=True)
        for fn in files:
            open(os.path.join(base, fn), "w").write("class X {}")
        return base

    def test_purges_disputed_noncontract_orphan(self):
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td, vf=_ORPHAN_VF)
            base = self._seed(td, ["JpaConfig.java", "ProductRepository.java"])
            bb.write("baseline_src", {"src/main/java/ProductRepository.java": "x"})
            res = ce.purge_unresolved_orphans(bb, ["DTO-ProductCoPurchasePair"])
            self.assertEqual(res["purged_files"], ["JpaConfig.java"])
            self.assertFalse(os.path.exists(os.path.join(base, "JpaConfig.java")))
            self.assertTrue(os.path.exists(os.path.join(base, "ProductRepository.java")))  # baseline kept

    def test_skips_when_not_disputed(self):
        # same orphan, but the task is NOT in the disputed set → refinement hasn't given up → leave it
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td, vf=_ORPHAN_VF)
            base = self._seed(td, ["JpaConfig.java"])
            self.assertIsNone(ce.purge_unresolved_orphans(bb, ["SOME-OTHER-TASK"]))
            self.assertTrue(os.path.exists(os.path.join(base, "JpaConfig.java")))

    def test_never_purges_contract_type_artifact(self):
        # ORPHAN names FirstProductRepository — a contract type's artifact → escalation's job, not ours
        vf = {"REPO-FirstProduct": [
            "ORPHAN ARTIFACT: FirstProductRepository.java was created this run but nothing "
            "references FirstProductRepository. Wire it into the code that should use it, or delete it."]}
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td, vf=vf)
            base = self._seed(td, ["FirstProductRepository.java"])
            self.assertIsNone(ce.purge_unresolved_orphans(bb, ["REPO-FirstProduct"]))
            self.assertTrue(os.path.exists(os.path.join(base, "FirstProductRepository.java")))

    def test_env_off_disables(self):
        os.environ["AGILE_ORPHAN_PURGE"] = "0"
        with tempfile.TemporaryDirectory() as td:
            bb = _bb(td, vf=_ORPHAN_VF)
            base = self._seed(td, ["JpaConfig.java"])
            self.assertIsNone(ce.purge_unresolved_orphans(bb, ["DTO-ProductCoPurchasePair"]))
            self.assertTrue(os.path.exists(os.path.join(base, "JpaConfig.java")))


if __name__ == "__main__":
    unittest.main()
