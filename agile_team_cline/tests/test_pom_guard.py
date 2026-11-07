"""Regression tests for the pom-snapshot guard (2026-07-07): the revenue probe rerun proved
_restore_missing_poms was only wired into sprint._run_maven_compile — a path the live pipeline
never executes — so a mid-run pom deletion aborted every real compile (repair._compile_raw /
_compile_main and sprint._run_maven_tests) at POM-parse. The guard now lives in repair.py and
runs on every one of those paths.
"""
import json
import os
import tempfile
import unittest
from unittest import mock

from agile_team_cline import repair


def _make_ws():
    td = tempfile.mkdtemp()
    pom = "<project>root</project>"
    child_rel = "modules/child/pom.xml"
    child_pom = "<project>child</project>"
    with open(os.path.join(td, "pom.xml"), "w") as fh:
        fh.write(pom)
    snap = {"pom.xml": pom, child_rel: child_pom}
    with open(os.path.join(td, "pom_snapshot.json"), "w") as fh:
        json.dump(snap, fh)
    return td, child_rel, child_pom


class TestRestoreMissingPoms(unittest.TestCase):

    def test_restores_deleted_child_pom(self):
        td, child_rel, child_pom = _make_ws()
        restored = repair._restore_missing_poms(td)
        self.assertEqual(restored, [child_rel])
        with open(os.path.join(td, child_rel)) as fh:
            self.assertEqual(fh.read(), child_pom)

    def test_noop_when_all_present(self):
        td, child_rel, _ = _make_ws()
        repair._restore_missing_poms(td)
        self.assertEqual(repair._restore_missing_poms(td), [])

    def test_noop_without_snapshot(self):
        td = tempfile.mkdtemp()
        self.assertEqual(repair._restore_missing_poms(td), [])


class TestGuardOnLiveCompilePaths(unittest.TestCase):
    """The guard must run before the mvn invocation on every path that gates the pipeline."""

    def _fake_run(self, *a, **k):
        return mock.Mock(returncode=0, stdout="", stderr="")

    def test_compile_raw_restores_first(self):
        td, child_rel, _ = _make_ws()
        with mock.patch.object(repair.subprocess, "run", side_effect=self._fake_run):
            repair._compile_raw(td)
        self.assertTrue(os.path.exists(os.path.join(td, child_rel)))

    def test_compile_main_restores_first(self):
        td, child_rel, _ = _make_ws()
        with mock.patch.object(repair.subprocess, "run", side_effect=self._fake_run):
            repair._compile_main(td)
        self.assertTrue(os.path.exists(os.path.join(td, child_rel)))

    def test_run_maven_tests_restores_first(self):
        from agile_team_cline import sprint
        td, child_rel, _ = _make_ws()
        with mock.patch("subprocess.run", side_effect=self._fake_run):
            sprint._run_maven_tests(td)
        self.assertTrue(os.path.exists(os.path.join(td, child_rel)))

    def test_deleted_root_pom_also_restored(self):
        td, child_rel, _ = _make_ws()
        os.remove(os.path.join(td, "pom.xml"))
        with mock.patch.object(repair.subprocess, "run", side_effect=self._fake_run):
            code, _ = repair._compile_raw(td)
        self.assertTrue(os.path.exists(os.path.join(td, "pom.xml")))
        self.assertEqual(code, 0)


class TestBaselineIntegrityGuard(unittest.TestCase):
    """The pom guard generalized: any manifest-listed (baseline_src.json) file missing on
    disk is restored from the pristine tree named by AGILE_BASELINE_REF; existing files
    (feature/hygiene-modified) are never overwritten."""

    def _make_ref_and_ws(self):
        ref = tempfile.mkdtemp()
        ws = tempfile.mkdtemp()
        rel = "modules/child/src/main/java/A.java"
        for root in (ref, ws):
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write("class A {}")
        with open(os.path.join(ws, "baseline_src.json"), "w") as fh:
            json.dump({rel: [0, 10]}, fh)
        return ref, ws, rel

    def test_restores_deleted_baseline_file(self):
        ref, ws, rel = self._make_ref_and_ws()
        os.remove(os.path.join(ws, rel))
        with mock.patch.dict(os.environ, {"AGILE_BASELINE_REF": ref}):
            restored = repair._restore_missing_baseline_files(ws)
        self.assertEqual(restored, [rel])
        with open(os.path.join(ws, rel)) as fh:
            self.assertEqual(fh.read(), "class A {}")

    def test_never_overwrites_modified_file(self):
        ref, ws, rel = self._make_ref_and_ws()
        with open(os.path.join(ws, rel), "w") as fh:
            fh.write("class A { int feature; }")
        with mock.patch.dict(os.environ, {"AGILE_BASELINE_REF": ref}):
            self.assertEqual(repair._restore_missing_baseline_files(ws), [])
        with open(os.path.join(ws, rel)) as fh:
            self.assertIn("feature", fh.read())

    def test_noop_without_env(self):
        _, ws, rel = self._make_ref_and_ws()
        os.remove(os.path.join(ws, rel))
        env = {k: v for k, v in os.environ.items() if k != "AGILE_BASELINE_REF"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(repair._restore_missing_baseline_files(ws), [])
        self.assertFalse(os.path.exists(os.path.join(ws, rel)))

    def test_skips_files_absent_from_ref(self):
        ref, ws, rel = self._make_ref_and_ws()
        os.remove(os.path.join(ws, rel))
        os.remove(os.path.join(ref, rel))
        with mock.patch.dict(os.environ, {"AGILE_BASELINE_REF": ref}):
            self.assertEqual(repair._restore_missing_baseline_files(ws), [])

    def test_ensure_tree_integrity_runs_both_guards(self):
        ref, ws, rel = self._make_ref_and_ws()
        os.remove(os.path.join(ws, rel))
        pom = "<project>child</project>"
        with open(os.path.join(ws, "pom_snapshot.json"), "w") as fh:
            json.dump({"modules/child/pom.xml": pom}, fh)
        with mock.patch.dict(os.environ, {"AGILE_BASELINE_REF": ref}):
            repair._ensure_tree_integrity(ws)
        self.assertTrue(os.path.exists(os.path.join(ws, rel)))
        self.assertTrue(os.path.exists(os.path.join(ws, "modules/child/pom.xml")))


class TestTaskEvidence(unittest.TestCase):
    """pipeline._task_evidence: deterministic disk facts assembled at verdict time so the
    PO can never reject on-disk code as 'entirely missing' (the recurring ordering bug)."""

    def _bb(self):
        from agile_team_cline.shared.blackboard import Blackboard
        td = tempfile.mkdtemp()
        rel = "src/main/java/com/example/Thing.java"
        p = os.path.join(td, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as fh:
            fh.write("package com.example; public class Thing {}")
        bb = Blackboard(td)
        bb.write("blackboard", {"generated_files": [rel]})
        return bb, rel

    def test_existing_file_reported_with_facts(self):
        from agile_team_cline.pipeline import _task_evidence
        bb, rel = self._bb()
        ev = _task_evidence([{"id": "ENT-Thing", "target_file": "Thing.java"}], bb)
        self.assertEqual(ev[0]["id"], "ENT-Thing")
        self.assertTrue(ev[0]["exists"])
        self.assertEqual(ev[0]["path"], rel)
        self.assertGreater(ev[0]["bytes"], 0)
        self.assertIn("modified", ev[0])

    def test_missing_file_reported_absent(self):
        from agile_team_cline.pipeline import _task_evidence
        bb, _ = self._bb()
        ev = _task_evidence([{"id": "ENT-Ghost", "target_file": "Ghost.java"}], bb)
        self.assertFalse(ev[0]["exists"])

    def test_task_without_target_file_still_listed(self):
        from agile_team_cline.pipeline import _task_evidence
        bb, _ = self._bb()
        ev = _task_evidence([{"id": "ARCH-1"}], bb)
        self.assertEqual(ev[0]["id"], "ARCH-1")
        self.assertNotIn("exists", ev[0])


if __name__ == "__main__":
    unittest.main()
