"""Regression tests for the runtime-execution VF gate (vf_query_executes, 2026-07-18).

Motivation: on DB-query features (F3 churn, F4 volatility) the naive junior kept shipping
code that COMPILES but THROWS at run time — e.g. a JPQL that selects a full entity beside
an aggregate, which EclipseLink rejects with `Invalid query key ... QueryException`. That
surfaced only at the frozen equivalence gate (terminal, un-actionable) because TeaStore's
persistence module has no boot/smoke probe (smoke.py early-returns under AGILE_BUILD_MODULE).
vf_query_executes reads the surefire reports the module's own `mvn test` already produced
and feeds the thrown exception back as task feedback, so refinement can turn broken →
worse-but-working.

The load-bearing property is the TIMING-EXCLUSION GUARANTEE: the gate must fire on code
that THROWS, and must NEVER fire on code that merely runs slowly. Two fences enforce it —
only <error> (thrown) entries are read, never <failure> (assertion, incl. timing) entries;
and any error carrying performance vocabulary is skipped.
"""
import os
import tempfile
import unittest

from agile_team_cline.verify_lib import vf_query_executes, attach_verifications

JAKARTA = {"language": "Java", "framework": "Jakarta EE — JAX-RS (Jersey), JPA/EclipseLink",
           "framework_family": "jakarta", "build_tool": "Maven", "file_extension": ".java"}

# The real churn failure, verbatim shape.
_QUERY_EXCEPTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.UserEndpointTest" tests="6" errors="2">
  <testcase name="testGetChurnRiskCustomers" classname="tools.descartes.teastore.persistence.UserEndpointTest" time="0.056">
    <error message="&#10;Exception Description: Invalid query key [totalPriceInCents] in expression.&#10;Query: ReportQuery(referenceClass=PersistenceUser jpql=&quot;SELECT u, SUM(o.totalPriceInCents) ...&quot;)" type="org.eclipse.persistence.exceptions.QueryException">stacktrace here</error>
  </testcase>
</testsuite>
"""

# A timing/perf check the way it would actually be written: an assertion → <failure>.
_TIMING_FAILURE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.PerfTest" tests="1" failures="1">
  <testcase name="testFastEnough" classname="tools.descartes.teastore.persistence.PerfTest" time="1.9">
    <failure message="expected query under 500 ms but elapsed 1900 ms" type="java.lang.AssertionError">at PerfTest.testFastEnough</failure>
  </testcase>
</testsuite>
"""

# A pathological perf check that THROWS instead of asserting — still must be excluded.
_TIMING_ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.UserEndpointTest" tests="1" errors="1">
  <testcase name="testGetChurnRiskCustomers" classname="tools.descartes.teastore.persistence.UserEndpointTest" time="12.0">
    <error message="query exceeded 5000 ms budget — too slow" type="java.lang.RuntimeException">at UserEndpointTest</error>
  </testcase>
</testsuite>
"""

# A green run: everything passes, no error/failure elements at all.
_GREEN_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.UserEndpointTest" tests="6" errors="0" failures="0">
  <testcase name="testGetChurnRiskCustomers" classname="tools.descartes.teastore.persistence.UserEndpointTest" time="0.05"/>
</testsuite>
"""


def _workspace(xml: str) -> str:
    """Minimal reactor-shaped workspace with one surefire report file."""
    ws = tempfile.mkdtemp()
    module = os.path.join(ws, "services", "tools.descartes.teastore.persistence")
    os.makedirs(os.path.join(module, "src", "main", "java"))
    reports = os.path.join(module, "target", "surefire-reports")
    os.makedirs(reports)
    with open(os.path.join(reports, "TEST-x.xml"), "w") as f:
        f.write(xml)
    return ws


class QueryExecutesVFTests(unittest.TestCase):
    def test_thrown_query_exception_is_reported(self):
        ws = _workspace(_QUERY_EXCEPTION_XML)
        msg = vf_query_executes(ws, {"class_hints": ["UserEndpoint"]}, None)
        self.assertIsNotNone(msg)
        self.assertIn("RUNTIME EXECUTION DEFECT", msg)
        self.assertIn("QueryException", msg)

    def test_timing_assertion_failure_is_ignored(self):
        # <failure> (assertion) is never read — a slow-but-working junior must not re-open.
        ws = _workspace(_TIMING_FAILURE_XML)
        self.assertIsNone(vf_query_executes(ws, {"class_hints": ["PerfTest"]}, None))

    def test_thrown_but_perf_worded_error_is_excluded(self):
        # Even a thrown exception is skipped when its text carries performance vocabulary.
        ws = _workspace(_TIMING_ERROR_XML)
        self.assertIsNone(vf_query_executes(ws, {"class_hints": ["UserEndpoint"]}, None))

    def test_green_run_passes(self):
        ws = _workspace(_GREEN_XML)
        self.assertIsNone(vf_query_executes(ws, {"class_hints": ["UserEndpoint"]}, None))

    def test_out_of_scope_error_not_attached(self):
        # A real error, but not referencing this task's classes → belongs to another task.
        ws = _workspace(_QUERY_EXCEPTION_XML)
        self.assertIsNone(vf_query_executes(ws, {"class_hints": ["CategoryEndpoint"]}, None))

    def test_no_reports_is_noop(self):
        ws = tempfile.mkdtemp()
        os.makedirs(os.path.join(ws, "services", "m", "src", "main", "java"))
        self.assertIsNone(vf_query_executes(ws, {"class_hints": ["UserEndpoint"]}, None))

    def test_no_hints_falls_open_to_any_execution_error(self):
        ws = _workspace(_QUERY_EXCEPTION_XML)
        self.assertIsNotNone(vf_query_executes(ws, {}, None))


class AttachVerificationsTests(unittest.TestCase):
    _CHURN_CONTRACT = {
        "entities": [{"name": "ChurnRiskCustomer",
                      "fields": {"userId": "long", "totalSpend": "long"}, "dto": True}],
        "endpoints": [{"method": "GET", "path": "/users/churn-risk",
                       "response": "ChurnRiskCustomer"}],
    }

    def test_endpoint_task_gets_query_executes_with_scoped_hints(self):
        task = {"id": "CTL-User", "type": "endpoint",
                "target_file": "services/.../UserEndpoint.java"}
        attach_verifications(task, self._CHURN_CONTRACT, JAKARTA)
        qe = [v for v in task["verifications"] if v.get("vf") == "query_executes"]
        self.assertEqual(len(qe), 1)
        hints = qe[0]["args"]["class_hints"]
        # scoping hints that will match the real QueryException's testcase classname/method
        self.assertIn("UserEndpoint", hints)
        self.assertIn("ChurnRiskCustomer", hints)

    def test_non_query_task_gets_no_gate(self):
        task = {"id": "DTO-ChurnRiskCustomer", "type": "dto",
                "target_file": "services/.../ChurnRiskCustomer.java"}
        attach_verifications(task, self._CHURN_CONTRACT, JAKARTA)
        self.assertFalse(any(v.get("vf") == "query_executes"
                             for v in task["verifications"]))


if __name__ == "__main__":
    unittest.main()
