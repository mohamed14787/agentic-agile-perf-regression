#!/usr/bin/env python3
"""
kieker_args.py — single source of truth for the Kieker AspectJ load-time-weaving
JVM arguments, shared by BOTH the standalone profiler (scripts/kieker_profile.sh)
and the dual-track engine (agile_team_cline/dual_track.py) so the two never drift.

The argLine injects, into the Surefire test JVM:
  • the Kieker AspectJ agent (kieker-2.0.0-aspectj.jar — ajc-compiled aspects),
  • --add-opens for the Java 9+ module system (so AspectJ can define $AjcClosure aux classes),
  • the aop.xml weave config + the local Kieker FileWriter config,
  • the per-run output directory (customStoragePath).

Env overrides (honoured by both callers): AJ_WEAVER, AOP_XML, KIEKER_PROPS.
Set KIEKER_NO_ADD_OPENS=1 when the forked test JVM is Java 8 — `--add-opens` is a Java 9+
flag and an 8 JVM refuses to start on it, which would corrupt the equivalence verdict
(build red from the harness, not the code).

CLI (used by the shell harness):
  python3 profiling/kieker_args.py <storage_path>   # prints the single -DargLine value
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_WEAVER = os.path.join(
    os.path.expanduser("~"),
    ".m2/repository/net/kieker-monitoring/kieker/2.0.0/kieker-2.0.0-aspectj.jar")


def kieker_jvm_args(storage_path: str, root: str = ROOT,
                    aj_weaver: str | None = None, aop_xml: str | None = None,
                    kieker_props: str | None = None) -> list[str]:
    """Return the Kieker LTW JVM args as a list (suitable for a -DargLine value)."""
    aj_weaver = aj_weaver or os.environ.get("AJ_WEAVER") or _DEFAULT_WEAVER
    aop_xml = aop_xml or os.environ.get("AOP_XML") or os.path.join(root, "profiling/aop.xml")
    kieker_props = (kieker_props or os.environ.get("KIEKER_PROPS")
                    or os.path.join(root, "profiling/kieker.local.properties"))
    args = [f"-javaagent:{aj_weaver}"]
    if os.environ.get("KIEKER_NO_ADD_OPENS") != "1":  # Java 9+ module system only
        args += ["--add-opens", "java.base/java.lang=ALL-UNNAMED",
                 "--add-opens", "java.base/java.lang.reflect=ALL-UNNAMED"]
    args += [
        f"-Dorg.aspectj.weaver.loadtime.configuration=file:{aop_xml}",
        f"-Dkieker.monitoring.configuration={kieker_props}",
        f"-Dkieker.monitoring.writer.filesystem.FileWriter.customStoragePath={storage_path}",
    ]
    return args


def arg_line(storage_path: str, **kw) -> str:
    """The single string value to pass as Surefire's -DargLine."""
    return " ".join(kieker_jvm_args(storage_path, **kw))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: kieker_args.py <storage_path>")
    print(arg_line(sys.argv[1]))
