"""Regression tests for the JPA registration gate (vf_entity_registered, 2026-08-03).

Motivation — volatility (#4) shipped PersistenceProductPriceChange.java as an @Entity that
was never added to persistence.xml's <class> list. `mvn test` bootstraps the persistence
unit in Java SE mode, where EclipseLink does NOT auto-discover annotated classes, so the
code compiled and then threw "The abstract schema type 'PersistenceProductPriceChange' is
unknown" on 11 tests across 4 test classes. The sample is in the dataset as a broken row.

Why this needs its own gate rather than better failure→source attribution: a stack trace
names *executing code*. All 11 traces pointed at PersistenceProductPriceChangeRepository —
where the query ran — while the defect was one line of XML that appears in no frame.
Configuration defects have no stack frame, so the answer is a static invariant.

Frequency scan across all 36 dataset workspaces (2026-08-03): exactly one carried the
defect, and only two persisted entities are ever introduced across the 15-feature chain
(everything else is a DTO). Rare, but total per occurrence — hence a cheap detector rather
than the deterministic persistence.xml repair that was also considered.

T1  entity on disk but not listed → reported, naming the descriptor and the FQCN.
T2  everything listed → no-op.
T3  exclude-unlisted-classes=false → no-op (the provider scans; listing not required).
T4  no persistence.xml at all (Spring) → no-op.
T5  @MappedSuperclass / @Embeddable / @Converter also count; annotations in comments and
    string literals do not; test sources and target/ copies are ignored.
T6  attach_verifications binds it to entity+repository on non-Spring, never on Spring.
"""
import os
import tempfile
import unittest

from agile_team_cline.verify_lib import (
    attach_verifications, unregistered_managed_classes, vf_entity_registered,
)

JAKARTA = {"language": "Java", "framework": "Jakarta EE — JAX-RS (Jersey), JPA/EclipseLink",
           "framework_family": "jakarta", "build_tool": "Maven", "file_extension": ".java"}
SPRING = {"language": "Java", "framework": "Spring Boot", "framework_family": "spring",
          "build_tool": "Maven", "file_extension": ".java"}

_MODULE = os.path.join("services", "tools.descartes.teastore.persistence")
_PKG = "tools.descartes.teastore.persistence.domain"

_UNIT_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<persistence version="2.1" xmlns="http://xmlns.jcp.org/xml/ns/persistence">
  <persistence-unit name="tools.descartes.teastore.persistence" transaction-type="RESOURCE_LOCAL">
%s%s
  </persistence-unit>
</persistence>
"""


def _unit(listed, scans=False):
    body = "".join(f"    <class>{_PKG}.{c}</class>\n" for c in listed)
    opt = "    <exclude-unlisted-classes>false</exclude-unlisted-classes>\n" if scans else ""
    return _UNIT_HEAD % (opt, body)


def _entity(name, anno="Entity"):
    return f"package {_PKG};\n\nimport jakarta.persistence.*;\n\n@{anno}\npublic class {name} {{}}\n"


def _workspace(unit_xml, main=None, test=None, target=None):
    """Reactor-shaped workspace: one module, one persistence unit, some sources."""
    ws = tempfile.mkdtemp()
    mod = os.path.join(ws, _MODULE)
    meta = os.path.join(mod, "src", "main", "resources", "META-INF")
    os.makedirs(meta)
    with open(os.path.join(meta, "persistence.xml"), "w") as f:
        f.write(unit_xml)
    for scope, files in (("main", main), ("test", test)):
        for fn, body in (files or {}).items():
            d = os.path.join(mod, "src", scope, "java", *_PKG.split("."))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fn), "w") as f:
                f.write(body)
    for fn, body in (target or {}).items():  # stale build output must never be scanned
        d = os.path.join(mod, "target", "classes", *_PKG.split("."))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, fn), "w") as f:
            f.write(body)
    return ws


class EntityRegisteredTests(unittest.TestCase):
    def test_t1_unregistered_entity_is_reported(self):
        ws = _workspace(_unit(["PersistenceProduct"]),
                        main={"PersistenceProduct.java": _entity("PersistenceProduct"),
                              "PersistenceProductPriceChange.java":
                                  _entity("PersistenceProductPriceChange")})
        missing = unregistered_managed_classes(ws)
        self.assertEqual(len(missing), 1)
        (px, items), = missing.items()
        self.assertEqual(items, [(f"{_PKG}.PersistenceProductPriceChange", "Entity")])
        self.assertTrue(px.endswith("persistence.xml"))

        msg = vf_entity_registered(ws, {}, None)
        self.assertIsNotNone(msg)
        self.assertIn("PersistenceProductPriceChange", msg)
        self.assertIn("persistence.xml", msg)
        self.assertIn("abstract schema type", msg)
        # Must not leak any performance instruction — same contract as vf_query_executes.
        for banned in ("fast", "slow", "optimi", "performance"):
            self.assertNotIn(banned, msg.lower())

    def test_t2_fully_registered_is_noop(self):
        ws = _workspace(_unit(["PersistenceProduct", "PersistenceProductPriceChange"]),
                        main={"PersistenceProduct.java": _entity("PersistenceProduct"),
                              "PersistenceProductPriceChange.java":
                                  _entity("PersistenceProductPriceChange")})
        self.assertEqual(unregistered_managed_classes(ws), {})
        self.assertIsNone(vf_entity_registered(ws, {}, None))

    def test_t3_exclude_unlisted_false_is_noop(self):
        """The unit opts into scanning, so listing is not required and the check is moot."""
        ws = _workspace(_unit([], scans=True),
                        main={"PersistenceProductPriceChange.java":
                              _entity("PersistenceProductPriceChange")})
        self.assertEqual(unregistered_managed_classes(ws), {})
        self.assertIsNone(vf_entity_registered(ws, {}, None))

    def test_t4_no_descriptor_is_noop(self):
        """Spring configures JPA in application.properties and ships no persistence.xml."""
        ws = tempfile.mkdtemp()
        d = os.path.join(ws, "src", "main", "java", "com", "example")
        os.makedirs(d)
        with open(os.path.join(d, "Product.java"), "w") as f:
            f.write("package com.example;\n@Entity\npublic class Product {}\n")
        self.assertEqual(unregistered_managed_classes(ws), {})
        self.assertIsNone(vf_entity_registered(ws, {}, None))

    def test_t5_annotation_set_and_scope(self):
        commented = (f"package {_PKG};\n\n"
                     "// @Entity  <- historical note, not an annotation\n"
                     'String s = "@Entity";\n'
                     "public class NotManaged {}\n")
        ws = _workspace(
            _unit(["PersistenceProduct"]),
            main={"PersistenceProduct.java": _entity("PersistenceProduct"),
                  "AuditBase.java": _entity("AuditBase", "MappedSuperclass"),
                  "Money.java": _entity("Money", "Embeddable"),
                  "InstantConverter.java": _entity("InstantConverter", "Converter"),
                  "NotManaged.java": commented},
            test={"PersistenceProductTest.java": _entity("PersistenceProductTest")},
            target={"StaleEntity.java": _entity("StaleEntity")})
        (_px, items), = unregistered_managed_classes(ws).items()
        got = {c.rsplit(".", 1)[-1]: a for c, a in items}
        self.assertEqual(got, {"AuditBase": "MappedSuperclass", "Money": "Embeddable",
                              "InstantConverter": "Converter"})
        self.assertNotIn("NotManaged", got)      # comment / string literal
        self.assertNotIn("PersistenceProductTest", got)  # test scope
        self.assertNotIn("StaleEntity", got)     # target/ build output

    def test_t6_attachment_is_non_spring_entity_and_repository(self):
        def vfs(task_type, stack):
            t = {"id": "ENT-PersistenceProductPriceChange", "type": task_type,
                 "target_file": "PersistenceProductPriceChange.java"}
            attach_verifications(t, {"entities": []}, stack)
            return [v.get("vf") for v in t.get("verifications", [])]

        self.assertIn("entity_registered", vfs("entity", JAKARTA))
        self.assertIn("entity_registered", vfs("repository", JAKARTA))
        self.assertNotIn("entity_registered", vfs("entity", SPRING))
        self.assertNotIn("entity_registered", vfs("repository", SPRING))
        self.assertNotIn("entity_registered", vfs("dto", JAKARTA))


class RegistrationBriefTests(unittest.TestCase):
    """Layer 1 — prevention. The coder that CREATES the entity must be told to register it;
    the Sprint-1 architect rules mention persistence.xml but never reach Sprint 2."""

    def _entity_task(self, stack):
        from agile_team_cline.ceremonies.planning import build_coverage_tasks
        contract = {"entities": [{"name": "PersistenceProductPriceChange",
                                  "fields": {"id": "long", "productId": "long"},
                                  "table": "product_price_change"}],
                    "endpoints": []}
        tasks = build_coverage_tasks(contract, stack)
        return next(t for t in tasks if t["id"].startswith("ENT-"))

    def test_jakarta_entity_task_demands_registration(self):
        title = self._entity_task(JAKARTA)["title"]
        self.assertIn("persistence.xml", title)
        self.assertIn("<class>", title)
        self.assertIn("Java SE", title)
        self.assertIn("edit that file in place", title.lower())

    def test_spring_entity_task_is_unchanged(self):
        """Spring has no persistence.xml — mentioning it would invite the phantom-descriptor
        failure mode that the Sprint-1 prompt fix exists to prevent."""
        title = self._entity_task(SPRING)["title"]
        self.assertNotIn("persistence.xml", title)
        self.assertNotIn("<class>", title)
        self.assertEqual(
            title, "Create the PersistenceProductPriceChange JPA entity with the fields "
                   "defined in the contract.")


if __name__ == "__main__":
    unittest.main()
