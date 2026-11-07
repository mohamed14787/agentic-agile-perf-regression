"""Regression tests for frozen-suite failure routing (2026-08-02).

Motivation — rfm (#16) was UNRECOVERABLE, not merely expensive. The junior's defect was one
line in UserRepository.getRfmSegments (MAX() over a converter-mapped column, which
EclipseLink returns unconverted → ClassCastException on 6/6 CustomerRfmSegment* cases). Four
attempts, two full refinement loops, and not one round ever opened UserRepository.java.
Four filters in series, each independently sufficient to lose the defect:

  1. pipeline.py refines BACKLOG TASKS, and the backlog is contract-derived over the feature's
     NEW types. UserRepository.java is a pre-existing file edited in place → no task exists
     for it, by construction.
  2. the brief points cline at each task's target_file.
  3. _build_cline_brief filtered qa_issues to those target files → the red tests were dropped
     from the brief entirely.
  4. _parse_surefire attributed the failure to <FailingTestClass>.java — the TEST, not the
     code under test. So lifting (3) alone would only aim the fixer at a frozen test file.

This suite covers the fix for (3) and (4). The load-bearing property is the TIMING-EXCLUSION
GUARANTEE inherited from vf_query_executes: routing fires on code that THREW, never on code
that is merely SLOW. If it ever leaked a timing hint the junior could be nudged toward the
fast implementation and the experiment's regression signal collapses — t3 exists to fail
loudly if that fence is weakened.

Note on real data: surefire's default trimStackTrace deletes every frame between the test
method and the throw site — exactly the range that names the culprit. sprint._run_maven_tests
now passes -DtrimStackTrace=false; without it the rfm trace carries ONLY the test frame.
"""
import json
import os
import tempfile
import unittest

from agile_team_cline import sprint
from agile_team_cline.sprint import _build_cline_brief, _parse_surefire
from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.verify_lib import owning_source_files

JAKARTA = {"name": "Java Jakarta EE", "language": "Java",
           "framework": "Jakarta EE — JAX-RS (Jersey), JPA/EclipseLink",
           "framework_family": "jakarta", "build_tool": "Maven", "file_extension": ".java"}

# The real rfm trace, shape verbatim from the 2026-08-02 run re-taken with
# -DtrimStackTrace=false: 49 frames, of which the FIRST project frame is the defect.
_RFM_TRACE = """jakarta.persistence.PersistenceException: An exception occurred while calling convertToEntityAttribute on converter class LocalDateTimeAttributeConverter with value 2026-07-31T18:20:11
\tat org.eclipse.persistence.internal.jpa.QueryImpl.getResultList(QueryImpl.java:264)
\tat tools.descartes.teastore.persistence.domain.UserRepository.getRfmSegments(UserRepository.java:392)
\tat tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest.testMultipleOrdersAggregatedPerUser(CustomerRfmSegmentRepositoryTest.java:251)
\tat java.base/jdk.internal.reflect.NativeMethodAccessorImpl.invoke0(NativeMethodAccessorImpl.java:-2)
\tat org.junit.runners.ParentRunner.run(ParentRunner.java:413)
Caused by: java.lang.ClassCastException: class java.time.LocalDateTime cannot be cast to class java.sql.Timestamp
\tat tools.descartes.teastore.persistence.domain.converters.LocalDateTimeAttributeConverter.convertToEntityAttribute(LocalDateTimeAttributeConverter.java:38)
"""

_RFM_XML = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" tests="6" errors="1">
  <testcase name="testMultipleOrdersAggregatedPerUser" classname="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" time="0.1">
    <error message="An exception occurred while calling convertToEntityAttribute" type="jakarta.persistence.PersistenceException">{_RFM_TRACE}</error>
  </testcase>
</testsuite>
"""

# A correctness ASSERTION failure: <failure>, and its trace has no production frame at all.
_ASSERTION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" tests="1" failures="1">
  <testcase name="testGetRfmSegmentsRankingAndScores" classname="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" time="0.1">
    <failure message="expected:&lt;3&gt; but was:&lt;2&gt;" type="java.lang.AssertionError">java.lang.AssertionError: expected:&lt;3&gt; but was:&lt;2&gt;
\tat org.junit.Assert.assertEquals(Assert.java:118)
\tat tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest.testGetRfmSegmentsRankingAndScores(CustomerRfmSegmentRepositoryTest.java:148)
</failure>
  </testcase>
</testsuite>
"""

# A perf check that THROWS, with a perfectly good production frame. Must never route.
_PERF_ERROR_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" tests="1" errors="1">
  <testcase name="testFastEnough" classname="tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest" time="12.0">
    <error message="query took 812ms, exceeded 500ms budget" type="java.lang.RuntimeException">java.lang.RuntimeException: query took 812ms, exceeded 500ms budget
\tat tools.descartes.teastore.persistence.domain.UserRepository.getRfmSegments(UserRepository.java:392)
\tat tools.descartes.teastore.persistence.CustomerRfmSegmentRepositoryTest.testFastEnough(CustomerRfmSegmentRepositoryTest.java:300)
</error>
  </testcase>
</testsuite>
"""

# Every frame above the production one is library code (JDK module-prefixed, junit, EclipseLink).
_LIBRARY_FRAMES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.ProductRepositoryTest" tests="1" errors="1">
  <testcase name="testX" classname="tools.descartes.teastore.persistence.ProductRepositoryTest" time="0.1">
    <error message="boom" type="java.lang.IllegalStateException">java.lang.IllegalStateException: boom
\tat java.base/java.util.stream.ReferencePipeline.collect(ReferencePipeline.java:578)
\tat org.eclipse.persistence.internal.jpa.QueryImpl.getResultList(QueryImpl.java:264)
\tat org.apache.commons.lang3.Validate.notNull(Validate.java:222)
\tat org.hibernate.engine.Foo.bar(Foo.java:11)
\tat tools.descartes.teastore.persistence.domain.ProductRepository.getAll(ProductRepository.java:88)
\tat tools.descartes.teastore.persistence.ProductRepositoryTest.testX(ProductRepositoryTest.java:40)
</error>
  </testcase>
</testsuite>
"""

_CTOR_FRAME_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="tools.descartes.teastore.persistence.ProductRepositoryTest" tests="1" errors="1">
  <testcase name="testX" classname="tools.descartes.teastore.persistence.ProductRepositoryTest" time="0.1">
    <error message="boom" type="jakarta.persistence.PersistenceException">jakarta.persistence.PersistenceException: boom
\tat tools.descartes.teastore.persistence.domain.PriceConverter.&lt;init&gt;(PriceConverter.java:22)
\tat java.base/jdk.internal.reflect.NativeConstructorAccessorImpl.newInstance(NativeConstructorAccessorImpl.java:77)
\tat java.base/java.lang.reflect.Constructor.newInstance(Constructor.java:490)
\tat tools.descartes.teastore.persistence.domain.ProductRepository.getAll(ProductRepository.java:88)
\tat tools.descartes.teastore.persistence.ProductRepositoryTest.testX(ProductRepositoryTest.java:40)
</error>
  </testcase>
</testsuite>
"""

_STDOUT_RFM = """
[ERROR] Tests run: 6, Failures: 0, Errors: 1, Skipped: 0 -- in CustomerRfmSegmentRepositoryTest
[ERROR] CustomerRfmSegmentRepositoryTest.testMultipleOrdersAggregatedPerUser:251 » Persistence
[INFO] Results:
[ERROR] Tests run: 190, Failures: 0, Errors: 1, Skipped: 0
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


class OwningSourceAttributionTests(unittest.TestCase):
    def test_t1_error_maps_to_production_file(self):
        """The topmost non-test, non-library frame is the file that threw."""
        m = owning_source_files(_workspace(_RFM_XML))
        self.assertEqual(set(m), {"CustomerRfmSegmentRepositoryTest.java"})
        src, exc = m["CustomerRfmSegmentRepositoryTest.java"]
        self.assertEqual(src, "UserRepository.java")
        self.assertIn("PersistenceException", exc)

        report = _parse_surefire(_STDOUT_RFM, _workspace(_RFM_XML))
        issue = next(i for i in report["issues"]
                     if i.get("scope") == "frozen_suite")
        self.assertEqual(issue["file"], "UserRepository.java")
        self.assertIn("CustomerRfmSegmentRepositoryTest", issue["concern"])
        self.assertIn("THREW", issue["concern"])

    def test_t2_assertion_failure_is_not_routed(self):
        """<failure> is an assertion — possibly a TIMING assertion. Never routed, and the
        issue stays attributed to the test class exactly as before."""
        self.assertEqual(owning_source_files(_workspace(_ASSERTION_XML)), {})

        report = _parse_surefire(_STDOUT_RFM, _workspace(_ASSERTION_XML))
        self.assertTrue(report["issues"])
        for i in report["issues"]:
            self.assertNotIn("scope", i)
            self.assertTrue(i["file"].endswith("Test.java"))

    def test_t3_perf_vocabulary_never_routes(self):
        """THE THESIS-VALIDITY TEST. A thrown perf complaint with a clean production frame
        must still yield nothing: the pipeline may make the junior RUN, never make it FAST."""
        self.assertEqual(owning_source_files(_workspace(_PERF_ERROR_XML)), {})

        report = _parse_surefire(_STDOUT_RFM, _workspace(_PERF_ERROR_XML))
        self.assertFalse([i for i in report["issues"] if i.get("scope") == "frozen_suite"])

    def test_t4_library_frames_skipped(self):
        """JDK (module-prefixed), junit, EclipseLink, commons and hibernate frames are all
        stepped over to reach project code."""
        m = owning_source_files(_workspace(_LIBRARY_FRAMES_XML))
        self.assertEqual(m["ProductRepositoryTest.java"][0], "ProductRepository.java")

    def test_t4b_constructor_frame_is_the_throw_site(self):
        """A defect thrown from a constructor must attribute to the constructor's file, not
        to the caller below it — `<init>` frames only match once the regex admits `<>`."""
        m = owning_source_files(_workspace(_CTOR_FRAME_XML))
        self.assertEqual(m["ProductRepositoryTest.java"][0], "PriceConverter.java")

    def test_t5_no_output_dir_is_byte_identical(self):
        """Omitting output_dir must reproduce the pre-change behaviour exactly."""
        report = _parse_surefire(_STDOUT_RFM)
        self.assertEqual(report["tests_run"], 190)
        self.assertEqual(report["tests_failed"], 1)
        self.assertTrue(report["issues"])
        for i in report["issues"]:
            self.assertNotIn("scope", i)
            self.assertTrue(i["file"].endswith("Test.java"), i["file"])


class BriefRoutingTests(unittest.TestCase):
    """(3) — the ownership filter in _build_cline_brief."""

    _TASKS = [{"id": "DTO-CustomerRfmSegment", "title": "Create the CustomerRfmSegment DTO",
               "type": "entity", "target_file": "CustomerRfmSegment.java"}]

    def _brief(self, issue, files=None):
        with tempfile.TemporaryDirectory() as td:
            bb = Blackboard(td)
            for rel, content in (files or {}).items():
                p = bb.output_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content)
                gen = bb.read("blackboard")
                gen.setdefault("generated_files", []).append(rel)
                bb.write("blackboard", gen)
            return _build_cline_brief(self._TASKS, bb, JAKARTA,
                                      {"qa_issues": [issue], "po_notes": "", "vf_failures": {}})

    def test_t6_unowned_marked_issue_reaches_brief(self):
        brief = self._brief({
            "file": "UserRepository.java", "scope": "frozen_suite",
            "concern": "Test failure: CustomerRfmSegmentRepositoryTest.testX did not pass. "
                       "UserRepository.java THREW at run time — PersistenceException: boom.",
        })
        self.assertIn("UserRepository.java", brief)
        self.assertIn("outside this batch's target files", brief)
        self.assertIn("EDIT it in place", brief)
        self.assertIn("do NOT modify any test file", brief)

    def test_t7_unowned_unmarked_issue_still_filtered(self):
        """The filter is widened by MARKER, not by file — an ordinary unowned issue is
        still dropped, exactly as before."""
        brief = self._brief({
            "file": "UserRepository.java",
            "concern": "Compilation Error at line 12: cannot find symbol",
        })
        self.assertNotIn("UserRepository.java", brief)
        self.assertNotIn("outside this batch's target files", brief)

    def test_t8_oversized_file_is_not_truncated_into_the_brief(self):
        """The embed header tells cline NOT to re-read the embedded files. rfm's
        UserRepository.java is 400+ lines with its defect at :392, so a 250-line window
        would hide the exact line the round exists to fix. Embed whole files only."""
        issue = {"file": "UserRepository.java", "scope": "frozen_suite",
                 "concern": "UserRepository.java THREW at run time — PersistenceException: boom."}
        rel = ("services/tools.descartes.teastore.persistence/src/main/java/tools/"
               "descartes/teastore/persistence/domain/UserRepository.java")

        big = "\n".join(f"// line {n}" for n in range(1, 401)) + "\nMAX_OVER_CONVERTED_COLUMN\n"
        brief = self._brief(issue, {rel: big})
        self.assertNotIn("MAX_OVER_CONVERTED_COLUMN", brief)
        self.assertNotIn("CURRENT CONTENT OF FILES TO FIX", brief)
        # The issue itself must still be routed — only the embed is skipped.
        self.assertIn("outside this batch's target files", brief)

        small = "\n".join(f"// line {n}" for n in range(1, 40)) + "\nSMALL_FILE_MARKER\n"
        brief = self._brief(issue, {rel: small})
        self.assertIn("SMALL_FILE_MARKER", brief)
        self.assertIn("CURRENT CONTENT OF FILES TO FIX", brief)


class TrimStackTraceFlagTest(unittest.TestCase):
    def test_test_gate_disables_stack_trimming(self):
        """Without -DtrimStackTrace=false the XML carries only the test frame, and every
        attribution above degrades to the old behaviour. Guard the flag itself."""
        import inspect
        src = inspect.getsource(sprint._run_maven_tests)
        self.assertIn("-DtrimStackTrace=false", src)


if __name__ == "__main__":
    unittest.main()
