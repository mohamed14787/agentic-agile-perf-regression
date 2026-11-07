"""Regression tests for _merge_ordinal_twin_entities (2026-07-12 phantom-alias fix).

The co-purchase defect: a self-comparison feature (pair a Product with another Product) made the
architect invent FirstProduct/SecondProduct entities beside the baseline's real Product. The gate
used to MERGE the pair into a surviving `FirstProduct` phantom — which VF then flagged as an
unresolvable orphan (contract declares it -> must create; nothing references it -> must delete).
The fix: when the ordinal base already exists (on disk in the staged baseline, or as a real
contract entity), DROP the phantom(s) entirely and repoint references at the base, rather than
keeping a merged twin. When the base is genuinely absent, the old identical-field pair-merge
behavior is preserved.
"""
import unittest

from agile_team_cline.ceremonies.planning import _merge_ordinal_twin_entities


def _names(contract):
    return [e["name"] for e in contract.get("entities", [])]


class TestOrdinalTwinPhantomDrop(unittest.TestCase):

    def test_drops_pair_when_base_exists_on_disk(self):
        c = {"entities": [
                {"name": "ProductCoPurchasePair",
                 "fields": {"firstProductId": "long", "secondProductId": "long"}, "dto": True},
                {"name": "FirstProduct", "fields": {"id": "long"}, "table": "first_product"},
                {"name": "SecondProduct", "fields": {"id": "long"}, "table": "second_product"}],
             "endpoints": [{"method": "GET", "path": "/products/co-purchase-matrix",
                            "response": "ProductCoPurchasePair"}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types={"Product", "Order", "OrderItem"})
        self.assertEqual(_names(out), ["ProductCoPurchasePair"])
        self.assertNotIn("FirstProduct", _names(out))
        self.assertNotIn("SecondProduct", _names(out))

    def test_drops_lone_ordinal_when_base_exists(self):
        """A lone FirstProduct (no SecondProduct twin) is still a phantom alias of the on-disk
        Product — the old gate skipped it because it required a pair."""
        c = {"entities": [
                {"name": "ProductCoPurchasePair", "fields": {"a": "long"}, "dto": True},
                {"name": "FirstProduct", "fields": {"id": "long"}, "table": "first_product"}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types={"product"})
        self.assertEqual(_names(out), ["ProductCoPurchasePair"])

    def test_repoints_typed_references_to_base(self):
        """When the DTO references the phantom by TYPE (not just an id), the reference is
        rewritten to the real base so nothing dangles."""
        c = {"entities": [
                {"name": "Pair", "fields": {"left": "FirstProduct", "right": "SecondProduct"},
                 "dto": True},
                {"name": "FirstProduct", "fields": {"id": "long"}},
                {"name": "SecondProduct", "fields": {"id": "long"}}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types={"product"})
        self.assertEqual(_names(out), ["Pair"])
        pair = out["entities"][0]
        self.assertEqual(pair["fields"]["left"], "Product")
        self.assertEqual(pair["fields"]["right"], "Product")

    def test_merges_pair_into_base_when_base_is_a_contract_entity(self):
        """Base present as a real contract entity (not on disk): twins fold into it."""
        c = {"entities": [
                {"name": "Product", "fields": {"id": "long", "name": "string"}},
                {"name": "FirstProduct", "fields": {"id": "long"}},
                {"name": "SecondProduct", "fields": {"id": "long"}}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types=set())
        self.assertEqual(sorted(_names(out)), ["Product"])

    def test_novel_twin_without_base_still_collapses_to_one(self):
        """No on-disk base and no contract base: preserve the old identical-field pair-merge."""
        c = {"entities": [
                {"name": "FirstSnapshot", "fields": {"id": "long", "v": "int"}},
                {"name": "SecondSnapshot", "fields": {"id": "long", "v": "int"}}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types=set())
        self.assertEqual(len(_names(out)), 1)
        self.assertIn(_names(out)[0], ("FirstSnapshot", "SecondSnapshot"))

    def test_different_fields_not_merged_when_base_absent(self):
        """Structurally different entities sharing an ordinal pattern are left alone."""
        c = {"entities": [
                {"name": "FirstSnapshot", "fields": {"id": "long"}},
                {"name": "SecondSnapshot", "fields": {"id": "long", "extra": "string"}}]}
        out = _merge_ordinal_twin_entities(c, on_disk_types=set())
        self.assertEqual(sorted(_names(out)), ["FirstSnapshot", "SecondSnapshot"])

    def test_no_on_disk_types_arg_is_backward_compatible(self):
        """Called without on_disk_types (default None) it must not crash and behaves like the
        base-absent path."""
        c = {"entities": [
                {"name": "FirstSnapshot", "fields": {"id": "long"}},
                {"name": "SecondSnapshot", "fields": {"id": "long"}}]}
        out = _merge_ordinal_twin_entities(c)
        self.assertEqual(len(_names(out)), 1)


if __name__ == "__main__":
    unittest.main()
