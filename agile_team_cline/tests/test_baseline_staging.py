"""Tests for the stage_workspace staging-integrity guard (2026-07-12).

An EMPTY staged tree is the silent trigger for greenfield regeneration: the pipeline sees
no existing files, sets milestone_file_baseline=0, and regenerates the whole reactor for a
one-method feature — 84% of tokens on gen, and any non-compiling file sinks the build (27/30
dual-track pairs died this way). Root cause: `targets/TeaStore` is a bare gitlink (mode
160000, no .gitmodules) so a fresh checkout leaves it empty and rsync copies emptiness.

The guard turns that SILENT greenfield into a LOUD failure, distinguishing:
  • rsync dropped files (baseline populated, staged smaller) → always fatal
  • baseline itself empty → fatal unless DUAL_ALLOW_EMPTY_BASELINE=1
These tests use a real rsync (present on macOS/Linux CI) against tiny synthetic baselines.
"""
import os
import shutil
import tempfile
import unittest

from agile_team_cline import dual_track


def _mk_baseline(root: str, n_java: int) -> None:
    """A minimal reactor-shaped baseline: n_java .java files plus a pom and a target/ dir
    that must be excluded from the count (mirrors the real exclude list)."""
    src = os.path.join(root, "services", "mod", "src", "main", "java", "pkg")
    os.makedirs(src, exist_ok=True)
    for i in range(n_java):
        open(os.path.join(src, f"Repo{i}.java"), "w").write("class Repo%d {}" % i)
    open(os.path.join(root, "pom.xml"), "w").write("<project/>")
    # build output that must NOT be counted
    tgt = os.path.join(root, "services", "mod", "target", "classes")
    os.makedirs(tgt, exist_ok=True)
    open(os.path.join(tgt, "Stale.java"), "w").write("class Stale {}")


class TestStagingGuard(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self._orig_allow = os.environ.get("DUAL_ALLOW_EMPTY_BASELINE")
        self._orig_snap = os.environ.get("DUAL_SRC_SNAPSHOT")
        os.environ["DUAL_SRC_SNAPSHOT"] = "0"  # no tar.gz side effect in tests

    def tearDown(self):
        shutil.rmtree(self._tmp, ignore_errors=True)
        for k, v in (("DUAL_ALLOW_EMPTY_BASELINE", self._orig_allow),
                     ("DUAL_SRC_SNAPSHOT", self._orig_snap)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _paths(self):
        return os.path.join(self._tmp, "baseline"), os.path.join(self._tmp, "out")

    def test_populated_baseline_stages_cleanly(self):
        base, out = self._paths()
        _mk_baseline(base, 5)
        dual_track.stage_workspace(base, out)  # must not raise
        self.assertEqual(dual_track._count_java(out), 5)  # target/ excluded

    def test_empty_baseline_is_fatal_by_default(self):
        base, out = self._paths()
        os.makedirs(base, exist_ok=True)
        open(os.path.join(base, "pom.xml"), "w").write("<project/>")  # no .java
        with self.assertRaises(RuntimeError) as ctx:
            dual_track.stage_workspace(base, out)
        self.assertIn("Empty baseline", str(ctx.exception))

    def test_empty_baseline_allowed_with_optin(self):
        base, out = self._paths()
        os.makedirs(base, exist_ok=True)
        open(os.path.join(base, "pom.xml"), "w").write("<project/>")
        os.environ["DUAL_ALLOW_EMPTY_BASELINE"] = "1"
        dual_track.stage_workspace(base, out)  # must not raise
        self.assertEqual(dual_track._count_java(out), 0)

    def test_count_java_excludes_build_output(self):
        base, _ = self._paths()
        _mk_baseline(base, 3)
        self.assertEqual(dual_track._count_java(base), 3)  # the target/ Stale.java not counted

    def test_rsync_dropped_files_is_fatal(self):
        # Populated baseline, but staging lands fewer files (simulate an rsync that dropped
        # source) → integrity failure even with the empty-baseline opt-in set.
        base, out = self._paths()
        _mk_baseline(base, 6)
        os.makedirs(out, exist_ok=True)  # exists so stage_workspace's rmtree is exercised
        os.environ["DUAL_ALLOW_EMPTY_BASELINE"] = "1"  # prove THIS guard is independent
        orig = dual_track._run
        dual_track._run = lambda *a, **k: (0, "")  # no-op rsync → out stays empty
        try:
            with self.assertRaises(RuntimeError) as ctx:
                dual_track.stage_workspace(base, out)
            self.assertIn("Staging integrity failure", str(ctx.exception))
        finally:
            dual_track._run = orig


if __name__ == "__main__":
    unittest.main()
