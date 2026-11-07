"""No-LLM regression suite for the jakarta/JUnit-4 test-authoring brief port (2026-07-02).

Run:  .venv/bin/python -m agile_team_cline.tests.test_test_briefs

Context: the dual-track feature_exercised gate (agile_team_cline/dual_track.py) only admits
a row when the profiled suite actually calls the feature method. On TeaStore (stack
java-jakarta) that suite comes from _build_simple_test_brief / _build_test_tasks
(agile_team_cline/sprint.py), which previously emitted ONLY Spring/JUnit 5 instructions
(@WebMvcTest/@DataJpaTest/@SpringBootTest, MockMvc, Mockito-jupiter) — meaningless on a
plain JAX-RS + JPA/EclipseLink + JUnit 4 stack, and Repository.java files were skipped
outright even though that's exactly where TeaStore's perf features live.

T1  _build_test_tasks: jakarta emits a 'repository'-kind task for *Repository.java files
    (previously skipped); spring behavior (skip Repository.java, run datajpa/integration
    loops) is unchanged (regression).
T2  _build_simple_test_brief: jakarta 'repository' brief uses JUnit 4 idiom + the
    must-invoke-the-method instruction and contains ZERO Spring/JUnit5 tokens; spring
    briefs (unit/webmvc/datajpa/integration) are byte-for-byte unaffected (regression).
T3  Package/path derivation: the jakarta test lands in the on-disk RepositoryTest.java
    exemplar's package (project convention), not the source class's own subpackage.
T4  stacks/java-jakarta.json declares JUnit 4 (was JUnit 5, unread until this port).
T5  _is_spring: the exact boolean the repair.py starter-test-pom-injector gate now uses.
"""
import os
import re
import sys
import json
import tempfile

from agile_team_cline.shared.blackboard import Blackboard
from agile_team_cline.sprint import (
    _build_simple_test_brief, _build_test_tasks, _is_spring, _run_source_hygiene,
)

_fails = []


def check(name, cond):
    """Record a named assertion, and FAIL the enclosing test when it does not hold.

    The raise is load-bearing — without it only `main()` ever reads `_fails`, so pytest
    (which calls test_* directly) reported green over real failures. See test_profiling."""
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        _fails.append(name)
        raise AssertionError(name)


JAKARTA = {"language": "Java", "framework": "Jakarta EE — JAX-RS (Jersey) on Tomcat, "
           "JPA/EclipseLink", "framework_family": "jakarta", "build_tool": "Maven",
           "file_extension": ".java"}
SPRING = {"language": "Java", "framework": "Spring Boot", "framework_family": "spring",
          "build_tool": "Maven", "file_extension": ".java"}

_SPRING_TOKENS = ("@SpringBootTest", "@WebMvcTest", "@DataJpaTest", "MockMvc", "@MockBean",
                  "jupiter", "@ExtendWith", "Mockito", "@Autowired")


def _write(bb: Blackboard, rel: str, content: str) -> None:
    p = bb.output_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    gen = bb.read("blackboard")
    gen.setdefault("generated_files", []).append(rel)
    bb.write("blackboard", gen)


def _teastore_workspace(td: str) -> Blackboard:
    """A minimal on-disk fixture mirroring TeaStore's real layout: a repository class
    under .../domain/, and the project's own RepositoryTest.java under the module ROOT
    test package (the real convention — see teastore-lift's checked-in exemplar)."""
    bb = Blackboard(td)
    mod = ("services/tools.descartes.teastore.persistence/src/main/java/"
           "tools/descartes/teastore/persistence/domain/CategoryRepository.java")
    _write(bb, mod, (
        "package tools.descartes.teastore.persistence.domain;\n\n"
        "public class CategoryRepository {\n"
        "  public static final CategoryRepository REPOSITORY = new CategoryRepository();\n"
        "  public java.util.List<Object> getBasketLift() { return null; }\n"
        "}\n"))
    exemplar = ("services/tools.descartes.teastore.persistence/src/test/java/"
                "tools/descartes/teastore/persistence/RepositoryTest.java")
    _write(bb, exemplar, (
        "package tools.descartes.teastore.persistence;\n\n"
        "import org.junit.Before;\nimport org.junit.Test;\n\n"
        "public class RepositoryTest {\n"
        "  @Before public void setup() throws Throwable { "
        "tools.descartes.teastore.persistence.repository.EMFManagerInitializer"
        ".initializeEMF(); }\n"
        "  @Test public void testRepos() { }\n"
        "}\n"))
    emf = ("services/tools.descartes.teastore.persistence/src/test/java/"
           "tools/descartes/teastore/persistence/repository/EMFManagerInitializer.java")
    _write(bb, emf, (
        "package tools.descartes.teastore.persistence.repository;\n"
        "public class EMFManagerInitializer { public static void initializeEMF() { } }\n"))
    return bb


def test_t1_build_test_tasks_jakarta_vs_spring():
    with tempfile.TemporaryDirectory() as td:
        bb = _teastore_workspace(td)
        bb.write("blackboard", dict(bb.read("blackboard"),
                                    milestone_file_baseline=0))
        tasks = _build_test_tasks(bb, JAKARTA)
        repo_tasks = [t for t in tasks if t["target_file"] == "CategoryRepositoryTest.java"]
        check("T1 jakarta emits a 'repository'-kind task for *Repository.java",
              len(repo_tasks) == 1 and repo_tasks[0]["kind"] == "repository")

    with tempfile.TemporaryDirectory() as td:
        bb = _teastore_workspace(td)
        bb.write("blackboard", dict(bb.read("blackboard"),
                                    milestone_file_baseline=0))
        tasks = _build_test_tasks(bb, SPRING)
        repo_tasks = [t for t in tasks if "CategoryRepository" in t["target_file"]]
        check("T1 spring still skips Repository.java (regression)", repo_tasks == [])


def test_t2_brief_content_jakarta_vs_spring():
    with tempfile.TemporaryDirectory() as td:
        bb = _teastore_workspace(td)
        task = {"id": "TEST-CategoryRepository",
                "title": "Create CategoryRepositoryTest verifying CategoryRepository: "
                          "a plain JUnit 4 test that directly invokes getBasketLift()",
                "target_file": "CategoryRepositoryTest.java", "type": "test",
                "kind": "repository"}
        brief = _build_simple_test_brief(task, bb, JAKARTA)
        check("T2 jakarta brief uses org.junit.Test", "org.junit.Test" in brief)
        check("T2 jakarta brief requires calling the repository method",
              "MUST directly invoke" in brief)
        check("T2 jakarta brief references EMFManagerInitializer.initializeEMF",
              "EMFManagerInitializer" in brief and "initializeEMF" in brief)
        # A line naming a Spring/JUnit5 token as a NEGATIVE instruction ("Do NOT use
        # @SpringBootTest") is correct and desired — same pattern the Spring branches use
        # ("Do NOT use @SpringBootTest" appears in the real 'unit'-kind Spring brief too).
        # Only a POSITIVE (non-negated) mention would mean the port leaked Spring guidance.
        _positive_hits = [tok for tok in _SPRING_TOKENS if tok in brief
                          and not any("not" in ln.lower() or "instead" in ln.lower()
                                      for ln in brief.splitlines() if tok in ln)]
        check("T2 jakarta brief has ZERO POSITIVE Spring/JUnit5 instructions "
              f"(only negated mentions allowed; found: {_positive_hits})",
              _positive_hits == [])
        check("T2 jakarta brief embeds CategoryRepository.java source",
              "public class CategoryRepository" in brief)
        check("T2 jakarta brief embeds the RepositoryTest.java exemplar",
              "structural exemplar" in brief and "testRepos" in brief)

    # Spring regression: unit/webmvc/datajpa/integration briefs unchanged.
    with tempfile.TemporaryDirectory() as td:
        bb = Blackboard(td)
        _write(bb, "src/main/java/org/springframework/samples/petclinic/vet/VetService.java",
               "package org.springframework.samples.petclinic.vet;\n"
               "public class VetService { }\n")
        task = {"id": "TEST-VetService", "title": "t", "target_file": "VetServiceTest.java",
                "type": "test", "kind": "unit"}
        brief = _build_simple_test_brief(task, bb, SPRING)
        check("T2 spring unit brief still uses Mockito/@ExtendWith (regression)",
              "@ExtendWith(MockitoExtension.class)" in brief and "@Mock" in brief)
        check("T2 spring unit brief still forbids @SpringBootTest (regression)",
              "Do NOT use @SpringBootTest" in brief)

        task_mvc = {"id": "TEST-VetController", "title": "t",
                    "target_file": "VetControllerTest.java", "type": "test", "kind": "webmvc"}
        _write(bb, "src/main/java/org/springframework/samples/petclinic/vet/VetController.java",
               "package org.springframework.samples.petclinic.vet;\n"
               "public class VetController { }\n")
        brief_mvc = _build_simple_test_brief(task_mvc, bb, SPRING)
        check("T2 spring webmvc brief still uses @WebMvcTest/MockMvc (regression)",
              "@WebMvcTest" in brief_mvc and "MockMvc" in brief_mvc)


def test_t3_package_follows_exemplar_not_source_subpackage():
    with tempfile.TemporaryDirectory() as td:
        bb = _teastore_workspace(td)
        task = {"id": "TEST-CategoryRepository", "title": "t",
                "target_file": "CategoryRepositoryTest.java", "type": "test",
                "kind": "repository"}
        brief = _build_simple_test_brief(task, bb, JAKARTA)
        m = re.search(r"Package:\s*(\S+)", brief)
        check("T3 package matches the RepositoryTest.java exemplar's ROOT package "
              "(not the source class's .domain subpackage)",
              m is not None and m.group(1) == "tools.descartes.teastore.persistence")
        check("T3 write path matches exemplar's test-tree location",
              "src/test/java/tools/descartes/teastore/persistence/"
              "CategoryRepositoryTest.java" in brief)


def test_t4_stack_declares_junit4():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "agile_team_cline", "stacks", "java-jakarta.json")
    with open(path) as f:
        cfg = json.load(f)
    check("T4 java-jakarta.json declares JUnit 4", cfg.get("test_framework") == "JUnit 4")


def test_t5_is_spring_gate():
    check("T5 _is_spring(JAKARTA) is False (repair.py starter-test injector must skip)",
          _is_spring(JAKARTA) is False)
    check("T5 _is_spring(SPRING) is True (repair.py starter-test injector must run)",
          _is_spring(SPRING) is True)


def test_t6_hygiene_scoped_to_build_module():
    """2026-07-04: the javax->jakarta rewrite must stay inside AGILE_BUILD_MODULE (+deps)
    when it's set — `generated_files` is brownfield-pre-seeded with the WHOLE tree, so an
    unscoped rewrite touches every javax hit in the repo (the real TeaStore runs rewrote 5
    unrelated files in modules outside the build scope, riding along into the trunk in
    chained-history mode)."""
    with tempfile.TemporaryDirectory() as td:
        bb = Blackboard(td)
        in_scope = ("services/tools.descartes.teastore.persistence/src/main/java/"
                    "tools/descartes/teastore/persistence/domain/Foo.java")
        out_scope = ("services/tools.descartes.teastore.webui/src/main/java/"
                     "tools/descartes/teastore/webui/servlet/ErrorServlet.java")
        javax_src = "package p;\nimport javax.persistence.Entity;\npublic class X {}\n"
        _write(bb, in_scope, javax_src)
        _write(bb, out_scope, javax_src)

        old = os.environ.get("AGILE_BUILD_MODULE")
        os.environ["AGILE_BUILD_MODULE"] = "services/tools.descartes.teastore.persistence"
        try:
            _run_source_hygiene(bb, JAKARTA)
        finally:
            if old is None:
                os.environ.pop("AGILE_BUILD_MODULE", None)
            else:
                os.environ["AGILE_BUILD_MODULE"] = old

        in_content = (bb.output_dir / in_scope).read_text()
        out_content = (bb.output_dir / out_scope).read_text()
        check("T6 in-scope javax file IS rewritten to jakarta",
              "jakarta.persistence" in in_content)
        check("T6 out-of-scope javax file is NOT rewritten",
              "javax.persistence" in out_content)


def test_t6b_hygiene_unscoped_when_module_unset():
    """AGILE_BUILD_MODULE unset -> unscoped, unchanged (single-module stacks like
    PetClinic never set it; behavior must stay byte-identical)."""
    with tempfile.TemporaryDirectory() as td:
        bb = Blackboard(td)
        rel = "src/main/java/p/Foo.java"
        _write(bb, rel, "package p;\nimport javax.persistence.Entity;\npublic class X {}\n")
        old = os.environ.pop("AGILE_BUILD_MODULE", None)
        try:
            _run_source_hygiene(bb, JAKARTA)
        finally:
            if old is not None:
                os.environ["AGILE_BUILD_MODULE"] = old
        check("T6b unscoped rewrite still fires when AGILE_BUILD_MODULE is unset",
              "jakarta.persistence" in (bb.output_dir / rel).read_text())


def main():
    test_t1_build_test_tasks_jakarta_vs_spring()
    test_t2_brief_content_jakarta_vs_spring()
    test_t3_package_follows_exemplar_not_source_subpackage()
    test_t4_stack_declares_junit4()
    test_t5_is_spring_gate()
    test_t6_hygiene_scoped_to_build_module()
    test_t6b_hygiene_unscoped_when_module_unset()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
