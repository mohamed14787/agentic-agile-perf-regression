"""No-LLM regression suite for the stack-aware Sprint-1 architect prompt (2026-08-02).

Run:  .venv/bin/python -m agile_team_cline.tests.test_sprint1_config_prompt

Context: _ARCHITECT_SPRINT1_PROMPT sent Spring Boot instructions ("configure the datasource
ONLY in application.properties", "Any JPA/Hibernate configuration must be a Java
@Configuration class") to EVERY stack unconditionally, plus a WebSocketConfig.java filename
example. On TeaStore — JAX-RS (Jersey) + JPA/EclipseLink, no Spring on the classpath —
12 of the first 15 trunk features duly hallucinated a phantom Spring config class
(PersistenceConfig, JerseyConfig, JpaConfig, RestApplicationConfig, DataInitializerConfig,
ApplicationConfig, Application.java): 171 ORPHAN ARTIFACT / CONTRACT DRIFT hits that
_vf_ceiling could only catch after the fact, at >=2 cline refinement rounds each.

The fix routes the stack-specific fragments through _SPRINT1_CONFIG. The overriding
constraint is that SPRING output must not move by a single byte — the existing 15-feature
dataset was generated with the old prompt, so any drift invalidates cross-feature
comparison.

T1  Spring rendering is frozen (equals the literal the dataset was generated with).
T2  An under-specified stack (no framework_family, no framework) still renders Spring.
T3  Jakarta carries no Spring directive and no Spring-flavoured filename examples.
T4  Jakarta names the descriptors that actually exist, and forbids all seven observed
    phantom *Config classes by name.
T5  Family gate: spring / jakarta / generic, and FastAPI gets no Java EE descriptors.
T6  ZH twin parity under AGILE_CHINESE=1.
T7  Placeholder contract: EN and ZH declare identical fields, and nothing renders with a
    literal brace left in it (the likeliest silent break is dropping {rules} from the twin).
"""
import os
import string
import sys

from agile_team_cline.config import MAX_TASKS_PER_SPRINT
from agile_team_cline.ceremonies.planning import (
    _ARCHITECT_SPRINT1_PROMPT, _ARCHITECT_SPRINT1_PROMPT_ZH, _SPRINT1_CONFIG,
    _build_sprint1_prompt, _sprint1_config_family,
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


FEATURE = "PERF-311 RFM customer segmentation"

SPRING = {"name": "Java / Spring Boot", "language": "Java", "framework": "Spring Boot",
          "framework_family": "spring", "build_tool": "Maven", "file_extension": ".java"}
JAKARTA = {"name": "Java Jakarta EE (JAX-RS + JPA/EclipseLink)", "language": "Java",
           "framework": "Jakarta EE — JAX-RS (Jersey) on Tomcat, JPA/EclipseLink",
           "framework_family": "jakarta", "build_tool": "Maven", "file_extension": ".java"}
FASTAPI = {"name": "Python FastAPI", "language": "Python", "framework": "FastAPI",
           "build_tool": "pip", "file_extension": ".py"}
# Mirrors test_contract_escalation.py's _STACK: declares neither framework_family nor
# framework, so _is_spring's default-True is what decides it.
UNDERSPEC = {"name": "Jakarta EE", "language": "Java", "file_extension": ".java"}

# Every phantom class name observed across the 15-feature diagnostic.
_OBSERVED_PHANTOMS = ("PersistenceConfig", "JerseyConfig", "JpaConfig",
                      "RestApplicationConfig", "DataInitializerConfig", "ApplicationConfig",
                      "Application.java")


def _render(stack, chinese=False):
    old = os.environ.get("AGILE_CHINESE")
    if chinese:
        os.environ["AGILE_CHINESE"] = "1"
    else:
        os.environ.pop("AGILE_CHINESE", None)
    try:
        return _build_sprint1_prompt(FEATURE, stack)
    finally:
        if old is None:
            os.environ.pop("AGILE_CHINESE", None)
        else:
            os.environ["AGILE_CHINESE"] = old


def test_t1_spring_rendering_is_frozen():
    """The pre-refactor prompt, transcribed. MAX_TASKS_PER_SPRINT is interpolated rather
    than hardcoded because it is env-overridable (config.py)."""
    frozen = (
        "⚡ List the Sprint 1 architecture files to build for: " + FEATURE + "\n"
        "Stack: " + SPRING["name"] + ".\n"
        "Sprint 1 = Architecture only: project config, build files, database config.\n"
        "Output ONLY lines in this exact format (no headers, no extra text):\n"
        "ID|title|type|sprint|target_file\n"
        "Where type is one of: config\n"
        "Where sprint must be 1 for ALL tasks.\n"
        "Where target_file is the exact filename (e.g. pom.xml, application.properties, WebSocketConfig.java).\n"
        "Spring Boot rules: configure the datasource ONLY in application.properties (target the H2 in-memory database). "
        "NEVER generate a persistence.xml — Spring Boot ignores it and it conflicts with application.properties. "
        "Any JPA/Hibernate configuration must be a Java @Configuration class, not persistence.xml.\n"
        f"Max {MAX_TASKS_PER_SPRINT} tasks. Example:\n"
        "T1|Initialize Maven project|config|1|pom.xml\n"
        "T2|Configure database and WebSocket|config|1|application.properties\n"
        "Now output Sprint 1 lines only for " + FEATURE + ":"
    )
    got = _render(SPRING)
    check("T1 spring rendering is byte-for-byte the pre-refactor prompt", got == frozen)
    if got != frozen:
        import difflib
        for line in difflib.unified_diff(frozen.splitlines(), got.splitlines(),
                                         "frozen", "rendered", lineterm=""):
            print("      " + line)


def test_t2_underspecified_stack_renders_spring():
    """_is_spring defaults to True when a stack declares neither framework_family nor
    framework. That default is load-bearing for existing fixtures — pin it here rather
    than discovering it via a surprise Jakarta rendering."""
    check("T2 under-specified stack resolves to the spring family",
          _sprint1_config_family(UNDERSPEC) == "spring")
    check("T2 under-specified stack renders the frozen spring rules",
          _SPRINT1_CONFIG["spring"]["rules"] in _render(UNDERSPEC))


def test_t3_jakarta_has_no_spring_directive():
    """Assert absence of the DIRECTIVE, not of the substring 'Spring' — the Jakarta block
    deliberately says 'there is NO Spring on this classpath'."""
    out = _render(JAKARTA)
    check("T3 jakarta drops the Spring Boot datasource directive",
          _SPRINT1_CONFIG["spring"]["rules"] not in out)
    check("T3 jakarta does not name application.properties as a target_file example",
          "|application.properties" not in out
          and "e.g. pom.xml, application.properties" not in out)
    check("T3 jakarta does not offer WebSocketConfig.java as a filename exemplar",
          "e.g. pom.xml, persistence.xml, web.xml" in out)
    check("T3 jakarta forbids rather than mandates a @Configuration class",
          "must be a Java @Configuration class" not in out)


def test_t4_jakarta_names_real_descriptors():
    out = _render(JAKARTA)
    for token in ("META-INF/persistence.xml", "WEB-INF/web.xml", "JAX-RS"):
        check(f"T4 jakarta names the real descriptor {token}", token in out)
    check("T4 jakarta marks persistence.xml REQUIRED (sprint.py refuses to delete it)",
          "REQUIRED" in out)
    for cls in _OBSERVED_PHANTOMS:
        check(f"T4 jakarta forbids the observed phantom {cls}", cls in out)
    check("T4 jakarta forbids the application.properties that fails the deployability gate",
          "application.properties" in out and "NEVER emit" in out)


def test_t5_family_gate():
    check("T5 SPRING -> spring", _sprint1_config_family(SPRING) == "spring")
    check("T5 JAKARTA -> jakarta", _sprint1_config_family(JAKARTA) == "jakarta")
    check("T5 FASTAPI -> generic", _sprint1_config_family(FASTAPI) == "generic")
    out = _render(FASTAPI)
    check("T5 FastAPI is not handed Java EE descriptors",
          "persistence.xml" not in out.split("Do NOT emit")[0]
          and "META-INF" not in out and "WEB-INF" not in out)
    check("T5 FastAPI is not handed the Spring datasource directive",
          _SPRINT1_CONFIG["spring"]["rules"] not in out)


def test_t6_zh_twin_parity():
    spring_zh = _render(SPRING, chinese=True)
    check("T6 ZH spring carries the frozen ZH rules",
          _SPRINT1_CONFIG["spring"]["rules_zh"] in spring_zh)
    check("T6 ZH spring uses the ZH template", "现在仅为" in spring_zh)

    jakarta_zh = _render(JAKARTA, chinese=True)
    check("T6 ZH jakarta carries the jakarta ZH rules",
          _SPRINT1_CONFIG["jakarta"]["rules_zh"] in jakarta_zh)
    check("T6 ZH jakarta does not carry the spring ZH rules",
          _SPRINT1_CONFIG["spring"]["rules_zh"] not in jakarta_zh)
    check("T6 ZH never mixes in the EN rules block",
          _SPRINT1_CONFIG["jakarta"]["rules"] not in jakarta_zh)
    for fam in _SPRINT1_CONFIG:
        check(f"T6 {fam} examples are shared EN/ZH (pipe-format rows are language-neutral)",
              _SPRINT1_CONFIG[fam]["examples"] in _render(
                  {"spring": SPRING, "jakarta": JAKARTA, "generic": FASTAPI}[fam],
                  chinese=True))


def test_t7_placeholder_contract():
    expected = {"feature", "stack_name", "max_tasks", "file_examples", "rules", "examples"}

    def fields(t):
        return {f for _lit, f, _spec, _conv in string.Formatter().parse(t) if f}

    en, zh = fields(_ARCHITECT_SPRINT1_PROMPT), fields(_ARCHITECT_SPRINT1_PROMPT_ZH)
    check("T7 EN template declares exactly the expected placeholders", en == expected)
    check("T7 ZH template declares the identical placeholder set", zh == en)

    for fam, stack in (("spring", SPRING), ("jakarta", JAKARTA), ("generic", FASTAPI)):
        for lang, zh_on in (("EN", False), ("ZH", True)):
            out = _render(stack, chinese=zh_on)
            check(f"T7 {fam}/{lang} renders with no unsubstituted placeholder",
                  "{" not in out and "}" not in out)
            check(f"T7 {fam}/{lang} actually carries its family's rules",
                  _SPRINT1_CONFIG[fam]["rules_zh" if zh_on else "rules"] in out)


def main():
    test_t1_spring_rendering_is_frozen()
    test_t2_underspecified_stack_renders_spring()
    test_t3_jakarta_has_no_spring_directive()
    test_t4_jakarta_names_real_descriptors()
    test_t5_family_gate()
    test_t6_zh_twin_parity()
    test_t7_placeholder_contract()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
