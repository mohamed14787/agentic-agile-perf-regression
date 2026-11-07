"""Regression tests for the contract-driven test-authoring scope filter (2026-07-10):
_build_test_tasks previously authored a dedicated test for EVERY milestone Repository/
Endpoint/Service/Controller, including baseline classes a feature only reads. On a
greenfield/regenerated milestone (TeaStore's baseline submodule is empty, so every run
regenerates the whole persistence module) this meant a single-endpoint feature like
co-purchase (contract: one endpoint, one DTO) still got CategoryRepositoryTest and
UserRepositoryTest authored (809K + 360K tokens) alongside the actual ProductRepositoryTest
— test authoring is the dominant cost driver (64% of all project tokens historically).

AGILE_TEST_SCOPE (default "contract") now scopes those four test kinds to classes whose
resource is named by the contract's endpoints, or whose source references a feature DTO/
endpoint-path token (catches a method placed on a differently-named sibling repository).
Fails OPEN (no filtering) when the contract yields no usable signal.
"""
import os
import tempfile
import unittest

from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.sprint import _build_test_tasks

JAKARTA = {"language": "Java", "framework": "Jakarta EE — JAX-RS (Jersey) on Tomcat, "
           "JPA/EclipseLink", "framework_family": "jakarta", "build_tool": "Maven",
           "file_extension": ".java"}

_COPURCHASE_CONTRACT = {
    "entities": [
        {"name": "Product", "fields": {"id": "Long", "name": "string"}, "table": "product"},
        {"name": "Order", "fields": {"id": "Long"}, "table": "order"},
        {"name": "OrderItem", "fields": {"id": "Long"}, "table": "order_item"},
        {"name": "CoPurchasePair", "fields": {"product1": "Product", "product2": "Product"},
         "dto": True},
        {"name": "Category", "fields": {"id": "Long"}, "table": "category"},
        {"name": "User", "fields": {"id": "Long"}, "table": "user"},
    ],
    "endpoints": [{"method": "GET", "path": "/products/co-purchase-matrix",
                   "response": "CoPurchasePair"}],
}


def _write(bb: Blackboard, rel: str, content: str) -> None:
    p = bb.output_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    gen = bb.read("blackboard")
    gen.setdefault("generated_files", []).append(rel)
    bb.write("blackboard", gen)


def _regenerated_module(bb: Blackboard, extra_body: str = "") -> None:
    """Mirrors a greenfield-regenerated persistence module: 5 repositories, only one
    (ProductRepository) actually carrying the new feature method."""
    base = ("services/tools.descartes.teastore.persistence/src/main/java/"
            "tools/descartes/teastore/persistence/")
    _write(bb, base + "ProductRepository.java",
           "package tools.descartes.teastore.persistence;\n"
           "public class ProductRepository {\n"
           f"  public java.util.List<Object> getCoPurchaseMatrix() {{ {extra_body} return null; }}\n"
           "}\n")
    for name in ("Category", "User", "Order", "OrderItem"):
        _write(bb, base + f"{name}Repository.java",
               f"package tools.descartes.teastore.persistence;\n"
               f"public class {name}Repository {{ }}\n")


class TestFeatureScopedTestTasks(unittest.TestCase):

    def test_scopes_to_feature_resource_only(self):
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            _regenerated_module(bb)
            bb.write("contract", _COPURCHASE_CONTRACT)
            bb.write("blackboard", dict(bb.read("blackboard"), milestone_file_baseline=0))
            tasks = _build_test_tasks(bb, JAKARTA)
            targets = {t["target_file"] for t in tasks}
            self.assertIn("ProductRepositoryTest.java", targets)
            self.assertNotIn("CategoryRepositoryTest.java", targets)
            self.assertNotIn("UserRepositoryTest.java", targets)
            self.assertNotIn("OrderRepositoryTest.java", targets)
            self.assertNotIn("OrderItemRepositoryTest.java", targets)

    def test_keeps_cross_entity_method_via_content_token(self):
        """The new method can land on a sibling repository (e.g. a join-heavy query
        placed on OrderRepository instead of the endpoint's nominal Product owner).
        The DTO-name/endpoint-token content check must still catch it."""
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            base = ("services/tools.descartes.teastore.persistence/src/main/java/"
                    "tools/descartes/teastore/persistence/")
            _write(bb, base + "ProductRepository.java",
                   "package tools.descartes.teastore.persistence;\n"
                   "public class ProductRepository { }\n")
            _write(bb, base + "OrderRepository.java",
                   "package tools.descartes.teastore.persistence;\n"
                   "public class OrderRepository {\n"
                   "  public java.util.List<Object> getCoPurchasePair() { return null; }\n"
                   "}\n")
            _write(bb, base + "CategoryRepository.java",
                   "package tools.descartes.teastore.persistence;\n"
                   "public class CategoryRepository { }\n")
            bb.write("contract", _COPURCHASE_CONTRACT)
            bb.write("blackboard", dict(bb.read("blackboard"), milestone_file_baseline=0))
            tasks = _build_test_tasks(bb, JAKARTA)
            targets = {t["target_file"] for t in tasks}
            self.assertIn("OrderRepositoryTest.java", targets,
                          "content-token match must keep the class actually holding "
                          "the feature method even off its nominal resource")
            self.assertNotIn("CategoryRepositoryTest.java", targets)

    def test_fails_open_when_contract_has_no_signal(self):
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            _regenerated_module(bb)
            bb.write("contract", {})
            bb.write("blackboard", dict(bb.read("blackboard"), milestone_file_baseline=0))
            tasks = _build_test_tasks(bb, JAKARTA)
            targets = {t["target_file"] for t in tasks}
            self.assertIn("CategoryRepositoryTest.java", targets)
            self.assertIn("UserRepositoryTest.java", targets)
            self.assertIn("OrderRepositoryTest.java", targets)
            self.assertIn("OrderItemRepositoryTest.java", targets)

    def test_scope_off_disables_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            _regenerated_module(bb)
            bb.write("contract", _COPURCHASE_CONTRACT)
            bb.write("blackboard", dict(bb.read("blackboard"), milestone_file_baseline=0))
            old = os.environ.get("AGILE_TEST_SCOPE")
            os.environ["AGILE_TEST_SCOPE"] = "off"
            try:
                tasks = _build_test_tasks(bb, JAKARTA)
            finally:
                if old is None:
                    os.environ.pop("AGILE_TEST_SCOPE", None)
                else:
                    os.environ["AGILE_TEST_SCOPE"] = old
            targets = {t["target_file"] for t in tasks}
            self.assertIn("CategoryRepositoryTest.java", targets)
            self.assertIn("UserRepositoryTest.java", targets)


if __name__ == "__main__":
    unittest.main()
