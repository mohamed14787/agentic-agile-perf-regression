"""Tests for the general contract-fidelity invariants (2026-07-05):

  1. vf_contract_types_realized / CONTRACT DRIFT — every contract-declared type must
     exist and be referenced.
  2. vf_contract_types_realized / ORPHAN ARTIFACT — every new production file must be
     referenced or be an entry point.
  3. _untargeted_test_tasks — an authored test must reference its subject class.

Anti-tailoring guard: each invariant includes at least one failure case structurally
DIFFERENT from the probe bugs that motivated it (a dropped request DTO, an orphan
helper class, a webmvc test referencing the wrong controller).
"""
import os
import tempfile
import unittest

from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.verify_lib import vf_contract_types_realized, VF_REGISTRY
from agile_team_cline.sprint import _untargeted_test_tasks

PKG = "src/main/java/com/example"
TPKG = "src/test/java/com/example"


def _ws(td: str, files: dict) -> None:
    """Materialize {relpath: content} plus a module pom.xml (so _find_src_roots sees it)."""
    open(os.path.join(td, "pom.xml"), "w").write("<project/>")
    for rel, content in files.items():
        p = os.path.join(td, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)


class TestContractTypesRealized(unittest.TestCase):

    def test_registered(self):
        self.assertIn("contract_types_realized", VF_REGISTRY)

    def test_pass_declared_dto_exists_and_referenced(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/FooSummary.java": "package com.example; public class FooSummary {}",
                f"{PKG}/FooEndpoint.java":
                    'package com.example; @Path("foos") public class FooEndpoint '
                    "{ public java.util.List<FooSummary> get() { return null; } }",
            })
            bb = Blackboard(td)
            bb.write("contract", {"entities": [{"name": "FooSummary", "dto": True}]})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))

    def test_fail_dropped_request_dto(self):
        # Structurally different from the probe bugs: a REQUEST type on a POST endpoint
        # that was never created at all.
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/WidgetEndpoint.java":
                    'package com.example; @Path("widgets") public class WidgetEndpoint {}',
            })
            bb = Blackboard(td)
            bb.write("contract", {"endpoints": [
                {"method": "POST", "path": "/widgets", "request": "CreateWidgetRequest"}]})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            msg = vf_contract_types_realized(td, {}, bb)
            self.assertIsNotNone(msg)
            self.assertIn("CONTRACT DRIFT", msg)
            self.assertIn("CreateWidgetRequest", msg)

    def test_fail_declared_type_exists_but_unreferenced(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/BarStats.java": "package com.example; public class BarStats {}",
                f"{PKG}/BarEndpoint.java":
                    'package com.example; @Path("bars") public class BarEndpoint '
                    "{ public java.util.List<java.util.Map<String, Object>> get() { return null; } }",
            })
            bb = Blackboard(td)
            bb.write("contract", {
                "entities": [{"name": "BarStats", "dto": True}],
                "endpoints": [{"method": "GET", "path": "/bars", "response": "List<BarStats>"}]})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            msg = vf_contract_types_realized(td, {}, bb)
            self.assertIsNotNone(msg)
            self.assertIn("dead code", msg)

    def test_javadoc_mention_is_not_a_reference(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/BazView.java": "package com.example; public class BazView {}",
                f"{PKG}/BazEndpoint.java":
                    "package com.example; /** returns BazView entries */ "
                    '@Path("baz") public class BazEndpoint {}',
            })
            bb = Blackboard(td)
            bb.write("contract", {"entities": [{"name": "BazView", "dto": True}]})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            msg = vf_contract_types_realized(td, {}, bb)
            self.assertIsNotNone(msg)
            self.assertIn("BazView", msg)

    def test_baseline_types_grandfathered(self):
        with tempfile.TemporaryDirectory() as td:
            rel = f"{PKG}/Legacy.java"
            _ws(td, {rel: "package com.example; public class Legacy {}"})
            bb = Blackboard(td)
            bb.write("contract", {"entities": [{"name": "Legacy"}]})
            bb.write("baseline_src", {rel: [0, 0]})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))

    def test_builtin_generics_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {f"{PKG}/A.java": "package com.example; public class A { B b; }",
                     f"{PKG}/B.java": "package com.example; public class B { A a; }"})
            bb = Blackboard(td)
            bb.write("contract", {"endpoints": [
                {"method": "GET", "path": "/x", "response": "Map<String, List<Long>>"}]})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))


class TestOrphanArtifacts(unittest.TestCase):

    def test_fail_orphan_helper_class(self):
        # Structurally different from the probe bugs: a helper class the contract never
        # mentions, created this run, referenced by nothing.
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/PriceCalculator.java":
                    "package com.example; public class PriceCalculator {}",
                f"{PKG}/ShopEndpoint.java":
                    'package com.example; @Path("shop") public class ShopEndpoint {}',
            })
            bb = Blackboard(td)
            bb.write("contract", {})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            msg = vf_contract_types_realized(td, {}, bb)
            self.assertIsNotNone(msg)
            self.assertIn("ORPHAN ARTIFACT", msg)
            self.assertIn("PriceCalculator", msg)

    def test_entry_points_are_not_orphans(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/NewEndpoint.java":
                    'package com.example; @Path("new") public class NewEndpoint {}',
                f"{PKG}/Main.java":
                    "package com.example; public class Main "
                    "{ public static void main(String[] a) {} }",
            })
            bb = Blackboard(td)
            bb.write("contract", {})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))

    def test_test_reference_keeps_artifact_alive(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {
                f"{PKG}/MathUtil.java": "package com.example; public class MathUtil {}",
                f"{TPKG}/MathUtilTest.java":
                    "package com.example; public class MathUtilTest "
                    "{ MathUtil u = new MathUtil(); }",
            })
            bb = Blackboard(td)
            bb.write("contract", {})
            bb.write("baseline_src", {"pom.xml": [0, 0]})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))

    def test_no_baseline_skips_orphan_check(self):
        with tempfile.TemporaryDirectory() as td:
            _ws(td, {f"{PKG}/Lonely.java": "package com.example; public class Lonely {}"})
            bb = Blackboard(td)
            bb.write("contract", {})
            self.assertIsNone(vf_contract_types_realized(td, {}, bb))


class TestUntargetedTestTasks(unittest.TestCase):

    def _bb(self, td: str, rel: str, content: str) -> Blackboard:
        bb = Blackboard(td)
        p = os.path.join(td, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w").write(content)
        bb.record_file(rel)
        return bb

    def test_fail_webmvc_test_referencing_wrong_controller(self):
        # Structurally different from the probe bugs: a Spring @WebMvcTest that
        # exercises OrderController while its subject is InvoiceController.
        with tempfile.TemporaryDirectory() as td:
            bb = self._bb(td, f"{TPKG}/InvoiceControllerTest.java",
                          "package com.example; public class InvoiceControllerTest "
                          "{ OrderController c = new OrderController(); }")
            tasks = [{"id": "TEST-InvoiceController", "subject": "InvoiceController",
                      "target_file": "InvoiceControllerTest.java", "type": "test"}]
            self.assertEqual(_untargeted_test_tasks(tasks, bb), tasks)

    def test_pass_when_subject_invoked(self):
        with tempfile.TemporaryDirectory() as td:
            bb = self._bb(td, f"{TPKG}/FooRepositoryTest.java",
                          "package com.example; public class FooRepositoryTest "
                          "{ Object r = FooRepository.REPOSITORY; }")
            tasks = [{"id": "TEST-FooRepository", "subject": "FooRepository",
                      "target_file": "FooRepositoryTest.java", "type": "test"}]
            self.assertEqual(_untargeted_test_tasks(tasks, bb), [])

    def test_own_class_name_and_comments_do_not_count(self):
        # "BarEndpointTest" must not satisfy \bBarEndpoint\b, and neither must a comment.
        with tempfile.TemporaryDirectory() as td:
            bb = self._bb(td, f"{TPKG}/BarEndpointTest.java",
                          "package com.example; /* verifying BarEndpoint */ "
                          "public class BarEndpointTest { Object r = SomeRepo.get(); }")
            tasks = [{"id": "TEST-BarEndpoint", "subject": "BarEndpoint",
                      "target_file": "BarEndpointTest.java", "type": "test"}]
            self.assertEqual(_untargeted_test_tasks(tasks, bb), tasks)

    def test_subjectless_and_missing_files_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            tasks = [
                {"id": "TEST-X-Integration", "target_file": "XIntegrationTest.java",
                 "type": "test"},                              # no subject → skip
                {"id": "TEST-Gone", "subject": "Gone",
                 "target_file": "GoneTest.java", "type": "test"},  # file absent → skip
            ]
            self.assertEqual(_untargeted_test_tasks(tasks, bb), [])


if __name__ == "__main__":
    unittest.main()
