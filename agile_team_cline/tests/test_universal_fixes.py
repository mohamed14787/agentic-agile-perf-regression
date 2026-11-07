"""No-LLM regression suite for the three universal harness fixes (2026-07-01).

Run:  .venv/bin/python -m agile_team_cline.tests.test_universal_fixes

U1  planning._structural_dto_names / _dto_names — a request-only input body is demoted to a
    DTO (no @Entity/table); response-only projections stay demoted; persisted/returned entities
    stay entities (Part-16 / p2-batch hardened cases guarded, incl. ChurnRiskUser).
U2  planning._contract_missing_types — endpoint types with no entity/DTO and not on disk are
    flagged; on-disk classes and container wrappers are not.
U3  shared.is_quota_exhausted / step_with_retry — a weekly/monthly quota 429 is fatal (no
    retry); a benign transient still retries.

2026-07-02 hardening: per-minute/per-second quotas classified transient; error code 1310
marker; dual_track infra-abort detection restricted to the harness's ✋ sentinel (plus the
INFRA_ABORT.json file channel); lowercase types inside generic brackets flagged by U2.
"""
import os
import sys
import tempfile

try:
    import camel  # noqa: F401 — planning.py needs it; fail early with a usable message
except ModuleNotFoundError:
    raise SystemExit("Run with .venv/bin/python — the system python lacks the camel dependency")

from agile_team_cline.ceremonies.planning import (
    _dto_names, _contract_missing_types, normalize_contract, _backfill_relationship_entities)
from agile_team_cline import shared
from agile_team_cline.shared import is_quota_exhausted, step_with_retry
from agile_team_cline.dual_track import _is_infra_abort_line

_fails = []


def check(name, cond):
    """Record a named assertion, and FAIL the enclosing test when it does not hold.

    The raise is load-bearing — without it only `main()` ever reads `_fails`, so pytest
    (which calls test_* directly) reported green over real failures. See test_profiling."""
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        _fails.append(name)
        raise AssertionError(name)


def test_u1_reclassifier():
    # request-only input body -> DTO; returned/persisted "…Request" -> entity
    loyalty = {
        "entities": [{"name": "PointsTransaction", "fields": {"id": "String"}},
                     {"name": "EarnRequest", "fields": {"accountId": "String", "price": "double"}}],
        "endpoints": [
            {"method": "POST", "path": "/loyalty/earn", "request": "EarnRequest",
             "response": "Response<PointsAccount>"},
            {"method": "GET", "path": "/loyalty/transactions/{id}",
             "response": "Response<PointsTransaction[]>"}],
    }
    d = _dto_names(loyalty)
    check("U1 EarnRequest (request-only body) demoted", "EarnRequest" in d)

    upgrade = {
        "entities": [{"name": "UpgradeRequest", "fields": {"id": "String", "status": "String"}}],
        "endpoints": [
            {"method": "POST", "path": "/upgrade/request", "request": {"body": "UpgradeRequestInfo"},
             "response": "Response<UpgradeRequest>"},
            {"method": "GET", "path": "/upgrade/{id}", "response": "Response<UpgradeRequest>"}],
    }
    check("U1 UpgradeRequest (returned/persisted) kept", "UpgradeRequest" not in _dto_names(upgrade))

    # Hardened set: projections demote, internal/written entities kept.
    p16 = {
        "entities": [
            {"name": "PriceHistory", "fields": {"id": "Long", "productId": "Long"}},
            {"name": "StockMovement", "fields": {"id": "Long", "qty": "int"}},
            {"name": "ProductCoPurchase", "fields": {"productId": "Long", "count": "int"}},
            {"name": "ChurnRiskUser", "fields": {"userId": "Long", "risk": "double"}},
            {"name": "ProductPriceVolatility", "fields": {"productId": "Long", "stddev": "double"}},
            {"name": "NearDuplicatePair", "fields": {"aId": "Long", "sim": "double"}},
            {"name": "Product", "fields": {"id": "Long", "name": "String"}},
        ],
        "endpoints": [
            {"method": "GET", "path": "/products/copurchases", "response": "List<ProductCoPurchase>"},
            {"method": "GET", "path": "/users/churn-risk", "response": "List<ChurnRiskUser>"},
            {"method": "GET", "path": "/products/volatility", "response": "List<ProductPriceVolatility>"},
            {"method": "GET", "path": "/products/near-duplicates", "response": "List<NearDuplicatePair>"},
            {"method": "POST", "path": "/stock/movements", "request": "StockMovement",
             "response": "StockMovement"},
            {"method": "GET", "path": "/products/{id}", "response": "Product"},
        ],
    }
    dp = _dto_names(p16)
    for proj in ("ProductCoPurchase", "ChurnRiskUser", "ProductPriceVolatility", "NearDuplicatePair"):
        check(f"U1 {proj} demoted (projection)", proj in dp)
    for ent in ("PriceHistory", "StockMovement", "Product"):
        check(f"U1 {ent} kept (persisted)", ent not in dp)

    # Item-6 (2026-07-02): the response-only rule alone also matched a genuinely-persisted
    # entity exposed ONLY via GET and never referenced (real bug, trainticket-tt001
    # contract.json — PointsTransaction: id/accountId/orderId/delta/type/createdAt) ->
    # demoted -> lost its repository -> compile-fatal. Fix = corroboration guard: block
    # demotion when a lifecycle/audit field (type/status/state/...At) is present. Real
    # contract, field names verbatim from disk.
    loyalty_real = {
        "entities": [
            {"name": "PointsTransaction", "fields": {
                "id": "String", "accountId": "String", "orderId": "String",
                "delta": "long", "type": "String", "createdAt": "String"}},
            {"name": "EarnRequest", "fields": {
                "id": "String", "accountId": "String", "orderId": "String", "price": "double"}},
        ],
        "endpoints": [
            {"method": "POST", "path": "/api/v1/orderservice/loyalty/earn",
             "request": "EarnRequest", "response": "Response<PointsAccount>"},
            {"method": "GET", "path": "/api/v1/orderservice/loyalty/balance/{accountId}",
             "response": "Response<PointsAccount>"},
            {"method": "GET", "path": "/api/v1/orderservice/loyalty/transactions/{accountId}",
             "response": "Response<PointsTransaction[]>"},
        ],
    }
    dl = _dto_names(loyalty_real)
    check("U1 item-6: PointsTransaction (lifecycle fields) kept, NOT demoted",
          "PointsTransaction" not in dl)
    check("U1 item-6: EarnRequest (request-only body) still demoted",
          "EarnRequest" in dl)


def test_u1b_pipeline_simplification():
    """2026-07-04 pipeline-simplification fixes, using the real TeaStore stockledger
    probe shapes: the xhigh architect emitted a CORRECT contract (LowStockEntry
    dto:true, no phantom Product) which normalize_contract then corrupted — these
    checks pin the fix so a correct model output survives normalization."""
    # (a) _entity_list must carry the `dto` flag through normalize_contract, and the
    # id/table backfill must skip entities _dto_names classifies as DTOs.
    stockledger = {
        "entities": [
            {"name": "StockMovement", "fields": {
                "id": "long", "productId": "long", "quantityChange": "long", "timestamp": "string"}},
            {"name": "LowStockEntry", "dto": True, "fields": {
                "product": "Product", "currentStock": "long"}},
        ],
        "endpoints": [
            {"method": "POST", "path": "/products/stock-movements",
             "request": "StockMovement", "response": "StockMovement"},
            {"method": "GET", "path": "/products/low-stock", "response": "List<LowStockEntry>"},
        ],
    }
    normed = normalize_contract(dict(stockledger), {"language": "java"},
                                 output_dir=None)
    by_name = {e["name"]: e for e in normed["entities"]}
    low = by_name.get("LowStockEntry", {})
    check("U1b LowStockEntry dto flag survives normalize_contract", low.get("dto") is True)
    check("U1b LowStockEntry gets NO id injected", "id" not in low.get("fields", {}))
    check("U1b LowStockEntry gets NO table", "table" not in low)
    check("U1b LowStockEntry still classified as dto by _dto_names",
          "LowStockEntry" in _dto_names(normed))
    mov = by_name.get("StockMovement", {})
    check("U1b StockMovement (real entity) still gets a table", mov.get("table") == "stock_movement")

    # (b) explicit dto:false still overrides heuristic demotion (flag-hierarchy regression)
    forced_entity = {
        "entities": [{"name": "OrderSummary", "dto": False,
                      "fields": {"id": "long", "total": "long"}}],
        "endpoints": [{"method": "GET", "path": "/orders/summary", "response": "OrderSummary"}],
    }
    fe_normed = normalize_contract(dict(forced_entity), {"language": "java"}, output_dir=None)
    fe_by_name = {e["name"]: e for e in fe_normed["entities"]}
    check("U1b explicit dto:false still gets a table (flag overrides name/structural heuristic)",
          fe_by_name.get("OrderSummary", {}).get("table") == "order_summary")

    # (c) _backfill_relationship_entities: on-disk awareness (brownfield) vs greenfield
    # unchanged. StockMovement.productId implies a "product" parent; the real entity is
    # PersistenceProduct (TeaStore's Persistence<X> naming convention) which the model
    # correctly did NOT redefine — the backfill must NOT synthesize a phantom Product.
    brownfield = {"entities": [{"name": "StockMovement", "fields": {
        "id": "long", "productId": "long", "quantityChange": "long"}}]}
    _backfill_relationship_entities(brownfield, on_disk={"PersistenceProduct"})
    names = {e["name"] for e in brownfield["entities"]}
    check("U1b on-disk-aware: no phantom Product synthesized (PersistenceProduct exists)",
          "Product" not in names)

    # Greenfield regression: PetClinic Pet.ownerId -> Owner still synthesized when
    # on_disk is empty/absent (unchanged behavior — the guard must still fire here).
    greenfield = {"entities": [{"name": "Pet", "fields": {"id": "Integer", "ownerId": "Integer"}}]}
    _backfill_relationship_entities(greenfield)
    gf_names = {e["name"] for e in greenfield["entities"]}
    check("U1b greenfield: Owner stub still synthesized when on_disk is empty (regression)",
          "Owner" in gf_names)

    # The copurchase run's actual failure mode: productAId/productBId -> phantom
    # ProductA/ProductB, same on-disk guard applies.
    copurchase = {"entities": [{"name": "ProductCoPurchase", "fields": {
        "id": "long", "productAId": "long", "productBId": "long", "coOccurrenceCount": "long"}}]}
    _backfill_relationship_entities(copurchase, on_disk={"PersistenceProduct"})
    cp_names = {e["name"] for e in copurchase["entities"]}
    check("U1b copurchase case: no phantom ProductA/ProductB (PersistenceProduct exists)",
          "ProductA" not in cp_names and "ProductB" not in cp_names)


def test_u2_completeness():
    # PointsAccount referenced by endpoints, no entity, not on disk -> flagged.
    loyalty = {
        "entities": [{"name": "PointsTransaction", "fields": {"id": "String"}}],
        "endpoints": [{"method": "GET", "path": "/loyalty/balance/{id}",
                       "response": "Response<PointsAccount>"}],
    }
    m = _contract_missing_types(loyalty, None)
    check("U2 undefined PointsAccount flagged", "PointsAccount" in m)
    check("U2 defined PointsTransaction not flagged", "PointsTransaction" not in m)
    check("U2 Response wrapper not flagged", "Response" not in m)

    # container/collection wrappers never flagged
    coll = {"entities": [{"name": "Order", "fields": {"id": "String"}}],
            "endpoints": [{"method": "GET", "path": "/orders", "response": "ArrayList<Order>"}]}
    check("U2 ArrayList wrapper not flagged", _contract_missing_types(coll, None) == set())

    # on-disk class counts as defined (brownfield) -> not flagged
    with tempfile.TemporaryDirectory() as td:
        pkg = os.path.join(td, "src", "main", "java", "contacts")
        os.makedirs(pkg)
        with open(os.path.join(pkg, "Contacts.java"), "w") as f:
            f.write("package contacts;\npublic class Contacts { }\n")
        c = {"entities": [{"name": "Contact", "fields": {"id": "String"}}],
             "endpoints": [{"method": "GET", "path": "/contacts/{id}",
                            "response": "Response<Contacts>"}]}
        check("U2 on-disk Contacts not flagged", "Contacts" not in _contract_missing_types(c, td))
        c2 = {"entities": [{"name": "Contact", "fields": {"id": "String"}}],
              "endpoints": [{"method": "GET", "path": "/x", "response": "Response<Ghost>"}]}
        check("U2 undefined Ghost flagged even with on-disk dir", "Ghost" in _contract_missing_types(c2, td))

    # generic brackets are a type position: a lowercase payload there is still flagged...
    lc = {"entities": [{"name": "Order", "fields": {"id": "String"}}],
          "endpoints": [{"method": "GET", "path": "/v", "response": "Response<voucherInfo>"}]}
    check("U2 lowercase type inside generics flagged", "voucherInfo" in _contract_missing_types(lc, None))
    # ...but lowercase prose OUTSIDE brackets stays exempt (uppercase-only rule there)
    prose = {"entities": [{"name": "Order", "fields": {"id": "String"}}],
             "endpoints": [{"method": "GET", "path": "/o",
                            "response": "list of the current orders as Order"}]}
    check("U2 lowercase prose outside generics not flagged",
          _contract_missing_types(prose, None) == set())


def test_u3_quota():
    real = ("Error code: 429 - {'error': {'code': '1310', 'message': "
            "'Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-07-04'}}")
    check("U3 quota 429 is fatal", is_quota_exhausted(real))
    check("U3 insufficient_quota is fatal", is_quota_exhausted("429 insufficient_quota"))
    check("U3 benign rate-limit not fatal", not is_quota_exhausted("429 rate limit, try again"))
    # short-window quotas reset within the retry budget — transient, never fatal
    check("U3 per-minute quota not fatal", not is_quota_exhausted(
        "429 Quota exceeded for quota metric 'Requests' and limit 'Requests per minute'"))
    check("U3 per-second quota not fatal",
          not is_quota_exhausted("Quota exceeded: 10 requests per second"))
    # Numeric code alone identifies exhaustion even if the wording changes
    check("U3 code 1310 is fatal",
          is_quota_exhausted("Error code: 429 - {'error': {'code': '1310', 'message': '...'}}"))

    # dual_track scans the child's stream for the harness's OWN ✋ sentinel only
    check("U3 sentinel line detected", _is_infra_abort_line(
        "      ✋ Fatal: model quota exhausted, not retrying — Error code: 429"))
    check("U3 cline sentinel detected", _is_infra_abort_line(
        "   ✋ cline [dev]: model quota exhausted, not retrying — 429"))
    check("U3 free-text quota mention NOT an abort", not _is_infra_abort_line(
        "   task: implement QuotaService returning 'quota exceeded' when limit exhausted"))

    calls = {"n": 0}
    sleeps = {"n": 0}
    orig = shared.time.sleep
    shared.time.sleep = lambda s: sleeps.__setitem__("n", sleeps["n"] + 1)

    class A:
        def step(self, msg):
            calls["n"] += 1
            raise RuntimeError(real)

    try:
        try:
            step_with_retry(A(), "m")
        except Exception:
            pass
        check("U3 quota raises immediately (1 call)", calls["n"] == 1)
        check("U3 quota does not sleep/retry", sleeps["n"] == 0)

        calls["n"] = 0

        class B:
            def step(self, msg):
                calls["n"] += 1
                raise RuntimeError("429 temporarily unavailable")

        try:
            step_with_retry(B(), "m", retries=2, backoff=0.0)
        except Exception:
            pass
        check("U3 transient retried (retries+1 calls)", calls["n"] == 3)
    finally:
        shared.time.sleep = orig


def main():
    test_u1_reclassifier()
    test_u1b_pipeline_simplification()
    test_u2_completeness()
    test_u3_quota()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
