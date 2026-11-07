"""No-LLM regression suite for the chained-trunk longitudinal history batch (2026-07-03).

Run:  .venv/bin/python -m agile_team_cline.tests.test_trunk_chain
      (also runs under system python — pure subprocess/filesystem, no camel dependency)

Covers scripts/advance_trunk.sh and scripts/lib_trunk_chain.sh — the new, additive
"trunk + one-step fork" mechanism that turns the independent senior/junior pairs batch
(scripts/run_teastore_all_features.sh, untouched) into a chained longitudinal history
(scripts/run_teastore_chained_history.sh): one senior-authored trunk advances feature to
feature, each junior fork is a discarded one-step counterfactual measured against the
trunk state before that feature (so junior regressions never compound).

T1  advance_trunk.sh — added/modified files land in the trunk; harness bookkeeping
    (contract.json, profile/, etc.) is excluded; a file present ONLY in the trunk (not in
    the senior workspace) is removed (--delete, no orphan accumulation).
T2  advance_trunk.sh deletion guard — a "gutted" senior workspace (most trunk files
    missing) is REFUSED without touching the trunk; ADVANCE_FORCE=1 overrides.
T3  lib_trunk_chain.sh decision functions — _feature_already_done / _infra_aborted /
    _senior_tests_pass read the dataset JSONL correctly (green tests_pass vs red vs
    missing row); _senior_tests_pass is independent of junior/equivalence fields.
T4  _ensure_trunk_advanced — idempotent: a first call with a green row advances the
    trunk and writes the test-point snapshot; a second call (simulating a resumed batch
    where the row is already written) is a no-op, proving no double-advance. A red-tests
    row leaves the trunk untouched and writes no snapshot (so a later real advance is
    still possible once the state improves).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ADVANCE_SH = os.path.join(_REPO, "scripts", "advance_trunk.sh")
_LIB_SH = os.path.join(_REPO, "scripts", "lib_trunk_chain.sh")

_fails = []


def check(name, cond):
    """Record a named assertion, and FAIL the enclosing test when it does not hold.

    The raise is load-bearing — without it only `main()` ever reads `_fails`, so pytest
    (which calls test_* directly) reported green over real failures. See test_profiling."""
    print(("PASS" if cond else "FAIL"), name)
    if not cond:
        _fails.append(name)
        raise AssertionError(name)


_VENV_BIN = os.path.join(_REPO, ".venv", "bin")


def _run(cmd, env=None, cwd=None, venv_on_path=True):
    """Run a command; by default with the venv prepended to PATH.

    That default USED TO HIDE A PRODUCTION BUG. lib_trunk_chain.sh called bare `python`, which
    does not exist on a Homebrew/macOS PATH, so every decision function in it exited non-zero
    with "command not found" — and each caller reads non-zero as a routine "no" (not done / not
    landable / nothing pruned). Prepending the venv here made the tests the ONLY context in
    which the library worked, so a green suite said nothing about the batch. The library now
    resolves its own interpreter (`_py`), and T3d below asserts that with venv_on_path=False.
    """
    full_env = dict(os.environ)
    if venv_on_path:
        full_env["PATH"] = _VENV_BIN + os.pathsep + full_env.get("PATH", "")
    if env:
        full_env.update(env)
    return subprocess.run(cmd, env=full_env, cwd=cwd, capture_output=True, text=True)


def _write_row(dataset_path, feature_id, senior_tests_pass=None, status="ok",
               senior_feature_exercised="__absent__", feature_exercised="__absent__",
               admitted="__absent__", junior_tests_pass=None):
    row = {"feature_id": feature_id, "status": status}
    if senior_tests_pass is not None:
        row["senior"] = {"tests_pass": senior_tests_pass, "build_success": True}
    # An EXPLICIT junior verdict. Omitting the junior key makes `tests_pass` false by
    # absence, which reads the same to the predicates but models nothing real: every row
    # of this shape in the corpus carries junior.tests_pass=False outright.
    if junior_tests_pass is not None:
        row["junior"] = {"tests_pass": junior_tests_pass, "build_success": True}
    if senior_feature_exercised != "__absent__":
        row.setdefault("senior", {})["feature_exercised"] = senior_feature_exercised
    if feature_exercised != "__absent__":
        row["feature_exercised"] = feature_exercised
    if admitted != "__absent__":
        row["admitted"] = admitted
    with open(dataset_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def test_t1_advance_basic():
    with tempfile.TemporaryDirectory() as td:
        trunk = os.path.join(td, "trunk")
        senior = os.path.join(td, "senior")
        # Real feature source lives under a MODULE path (services/<mod>/src/…), never a
        # bare repo-root src/ — advance_trunk.sh anchors an `--exclude /src` to drop the
        # junk root-src tree a coder agent sometimes creates. The fixture mirrors the real
        # layout so it exercises a genuine advance rather than colliding with that exclude.
        mod = os.path.join("services", "m", "src")
        os.makedirs(os.path.join(trunk, mod))
        os.makedirs(os.path.join(senior, mod))

        with open(os.path.join(trunk, mod, "A.java"), "w") as f:
            f.write("class A {}\n")
        with open(os.path.join(trunk, mod, "B.java"), "w") as f:
            f.write("class B {}\n")

        shutil.copy(os.path.join(trunk, mod, "A.java"), os.path.join(senior, mod, "A.java"))
        with open(os.path.join(senior, mod, "B.java"), "w") as f:
            f.write("class B { void x(){} }\n")  # modified
        with open(os.path.join(senior, mod, "C.java"), "w") as f:
            f.write("class C {}\n")  # added
        with open(os.path.join(senior, "contract.json"), "w") as f:
            f.write('{"fake":true}\n')  # bookkeeping — must NOT land in trunk

        r = _run(["zsh", _ADVANCE_SH, trunk, senior])
        check("T1 advance_trunk.sh exits 0 on normal advance", r.returncode == 0)
        check("T1 added file C.java lands in trunk",
              os.path.isfile(os.path.join(trunk, mod, "C.java")))
        with open(os.path.join(trunk, mod, "B.java")) as f:
            check("T1 modified file B.java content updated", "void x()" in f.read())
        check("T1 A.java unchanged", os.path.isfile(os.path.join(trunk, mod, "A.java")))
        check("T1 contract.json bookkeeping excluded from trunk",
              not os.path.exists(os.path.join(trunk, "contract.json")))


def test_t1_delete_orphan():
    with tempfile.TemporaryDirectory() as td:
        trunk = os.path.join(td, "trunk")
        senior = os.path.join(td, "senior")
        os.makedirs(trunk)
        os.makedirs(senior)
        with open(os.path.join(trunk, "stray.java"), "w") as f:
            f.write("class Stray {}\n")  # only in trunk, not in senior
        with open(os.path.join(senior, "kept.java"), "w") as f:
            f.write("class Kept {}\n")

        r = _run(["zsh", _ADVANCE_SH, trunk, senior])
        check("T1 delete-orphan advance exits 0", r.returncode == 0)
        check("T1 stray.java (orphan) removed from trunk",
              not os.path.exists(os.path.join(trunk, "stray.java")))
        check("T1 kept.java present in trunk", os.path.isfile(os.path.join(trunk, "kept.java")))


def test_t2_deletion_guard():
    with tempfile.TemporaryDirectory() as td:
        trunk = os.path.join(td, "trunk")
        gutted = os.path.join(td, "gutted")
        os.makedirs(trunk)
        os.makedirs(gutted)
        for i in range(15):
            with open(os.path.join(trunk, f"F{i}.java"), "w") as f:
                f.write(f"class F{i} {{}}\n")
        # gutted senior workspace: only 1 of 15 files survives -> 14 deletions, over default guard (10)
        with open(os.path.join(gutted, "F0.java"), "w") as f:
            f.write("class F0 {}\n")

        r = _run(["zsh", _ADVANCE_SH, trunk, gutted])
        check("T2 guard REFUSES a mass-deletion advance (non-zero exit)", r.returncode != 0)
        remaining = len([n for n in os.listdir(trunk) if n.endswith(".java")])
        check("T2 trunk untouched after refusal (still 15 files)", remaining == 15)

        r2 = _run(["zsh", _ADVANCE_SH, trunk, gutted], env={"ADVANCE_FORCE": "1"})
        check("T2 ADVANCE_FORCE=1 overrides the guard (exit 0)", r2.returncode == 0)
        remaining2 = len([n for n in os.listdir(trunk) if n.endswith(".java")])
        check("T2 forced advance actually deletes down to 1 file", remaining2 == 1)


def _lib_call(func_and_args, env, cwd=None, venv_on_path=True):
    """Source lib_trunk_chain.sh and invoke one function; return CompletedProcess."""
    script = f"source {_LIB_SH!r} && {func_and_args}"
    return _run(["zsh", "-c", script], env=env, cwd=cwd, venv_on_path=venv_on_path)


def test_t3d_decisions_hold_without_the_venv_on_path():
    """The chain's decision layer must work on the PATH the batch actually runs with.

    The batch is launched from a plain shell where `python` does not exist — only `python3`.
    Every function in lib_trunk_chain.sh is a Python heredoc, so on that PATH they all failed
    with "command not found" and returned non-zero, which callers read as "not done" / "not
    landable" / "nothing to prune". The trunk silently stops advancing and the pruner silently
    keeps stale rows; nothing reports that the decision layer is inert.

    This asserts the POSITIVE answer specifically. A missing interpreter can only produce
    non-zero, so any test whose expectation is "false" passes just as well when the library is
    completely broken — which is how this survived. Run without the venv on PATH, the way
    production does.
    """
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        _write_row(dataset, "feat-green", senior_tests_pass=True, admitted=True)
        env = {"DUAL_DATASET": dataset, "TRUNK_DIR": td, "HIST_DIR": td}

        res = _lib_call("_feature_already_done feat-green", env, venv_on_path=False)
        check("T3d done-check answers TRUE with no venv on PATH", res.returncode == 0)
        check("T3d no 'command not found' from the decision layer",
              "command not found" not in (res.stderr or ""))
        check("T3d missing row still answers FALSE",
              _lib_call("_feature_already_done feat-none", env,
                        venv_on_path=False).returncode != 0)


def test_t3_decision_functions():
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        _write_row(dataset, "feat-green", senior_tests_pass=True)
        _write_row(dataset, "feat-red", senior_tests_pass=False)
        _write_row(dataset, "feat-abort", status="infra_abort")

        env = {"DUAL_DATASET": dataset}

        check("T3 _feature_already_done true for existing row",
              _lib_call("_feature_already_done feat-green", env).returncode == 0)
        check("T3 _feature_already_done false for missing row",
              _lib_call("_feature_already_done feat-nonexistent", env).returncode != 0)

        check("T3 _senior_tests_pass true when tests_pass=True",
              _lib_call("_senior_tests_pass feat-green", env).returncode == 0)
        check("T3 _senior_tests_pass false when tests_pass=False",
              _lib_call("_senior_tests_pass feat-red", env).returncode != 0)
        check("T3 _senior_tests_pass false when no row at all",
              _lib_call("_senior_tests_pass feat-nonexistent", env).returncode != 0)

        check("T3 _infra_aborted true for status=infra_abort row",
              _lib_call("_infra_aborted feat-abort", env).returncode == 0)
        check("T3 _infra_aborted false for a normal ok row",
              _lib_call("_infra_aborted feat-green", env).returncode != 0)


def test_t4_ensure_trunk_advanced_idempotent():
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        trunk = os.path.join(td, "trunk")
        hist = os.path.join(td, "hist")
        senior_ws = os.path.join(td, "feat1-senior")
        os.makedirs(trunk)
        os.makedirs(hist)
        os.makedirs(senior_ws)
        with open(os.path.join(senior_ws, "New.java"), "w") as f:
            f.write("class New {}\n")

        _write_row(dataset, "feat1", senior_tests_pass=True)
        env = {
            "DUAL_DATASET": dataset, "TRUNK_DIR": trunk, "HIST_DIR": hist,
            "TRUNK_CHAIN_LIB_DIR": os.path.join(_REPO, "scripts"),
        }

        r1 = _lib_call("_ensure_trunk_advanced 1 feat1", env, cwd=td)
        check("T4 first call succeeds", r1.returncode == 0)
        check("T4 trunk received New.java after first advance",
              os.path.isfile(os.path.join(trunk, "New.java")))
        snap = os.path.join(hist, "tp-01-feat1.tar.gz")
        check("T4 test-point snapshot written", os.path.isfile(snap))
        mtime_before = os.path.getmtime(snap)

        # simulate a resumed batch calling the reconcile again for the same test point
        r2 = _lib_call("_ensure_trunk_advanced 1 feat1", env, cwd=td)
        check("T4 second (resume) call succeeds", r2.returncode == 0)
        check("T4 second call is a no-op (snapshot mtime unchanged)",
              os.path.getmtime(snap) == mtime_before)

        # a red-tests feature must NOT advance and must NOT write a snapshot
        senior_ws2 = os.path.join(td, "feat2-senior")
        os.makedirs(senior_ws2)
        with open(os.path.join(senior_ws2, "ShouldNotLand.java"), "w") as f:
            f.write("class ShouldNotLand {}\n")
        _write_row(dataset, "feat2", senior_tests_pass=False)
        r3 = _lib_call("_ensure_trunk_advanced 2 feat2", env, cwd=td)
        check("T4 red-tests call still exits 0 (hold is not an error)", r3.returncode == 0)
        check("T4 red-tests feature's file NOT merged into trunk",
              not os.path.exists(os.path.join(trunk, "ShouldNotLand.java")))
        check("T4 red-tests feature writes no snapshot",
              not os.path.isfile(os.path.join(hist, "tp-02-feat2.tar.gz")))


def test_t5_chain_manifest():
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        trunk = os.path.join(td, "trunk")
        hist = os.path.join(td, "hist")
        manifest = os.path.join(td, "chain_manifest.json")
        senior_ws = os.path.join(td, "feat1-senior")
        os.makedirs(trunk)
        os.makedirs(hist)
        os.makedirs(senior_ws)
        with open(os.path.join(senior_ws, "New.java"), "w") as f:
            f.write("class New {}\n")

        _write_row(dataset, "feat1", senior_tests_pass=True)
        env = {
            "DUAL_DATASET": dataset, "TRUNK_DIR": trunk, "HIST_DIR": hist,
            "CHAIN_MANIFEST": manifest,
            "TRUNK_CHAIN_LIB_DIR": os.path.join(_REPO, "scripts"),
        }

        # green advance (parent = genesis) → manifest gains one landed test point
        r1 = _lib_call("_ensure_trunk_advanced 1 feat1 genesis", env, cwd=td)
        check("T5 advance-with-parent call succeeds", r1.returncode == 0)
        check("T5 manifest file written", os.path.isfile(manifest))
        with open(manifest) as f:
            doc = json.load(f)
        tps = doc.get("test_points", [])
        check("T5 manifest has exactly one test point", len(tps) == 1)
        tp = tps[0] if tps else {}
        check("T5 test point sequence recorded", tp.get("sequence") == 1)
        check("T5 test point feature_id recorded", tp.get("feature_id") == "feat1")
        check("T5 test point parent recorded", tp.get("parent") == "genesis")
        check("T5 test point landed_on_trunk true", tp.get("landed_on_trunk") is True)
        check("T5 test point snapshot path recorded",
              tp.get("snapshot", "").endswith("tp-01-feat1.tar.gz"))

        # resume-reconcile the SAME test point → idempotent upsert, no duplicate row
        r2 = _lib_call("_ensure_trunk_advanced 1 feat1 genesis", env, cwd=td)
        check("T5 resume reconcile succeeds", r2.returncode == 0)
        with open(manifest) as f:
            doc2 = json.load(f)
        check("T5 manifest still has exactly one test point (idempotent upsert)",
              len(doc2.get("test_points", [])) == 1)


def test_t6_senior_feature_exercised_gate():
    """2026-07-18 gate fix: the trunk-advance gate keys on the SENIOR-specific
    `senior.feature_exercised`, not the conflated top-level `feature_exercised` (which
    collapses to None whenever the junior fails to compile). The core case: a junior that
    broke (top-level feature_exercised=None) must STILL let a genuinely-exercised senior
    land — while a senior that was NOT exercised must be BLOCKED even if the old conflated
    field would have passed it."""
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        # senior exercised, junior broke → top-level None, senior.feature_exercised True
        _write_row(dataset, "feat-senior-ok", senior_tests_pass=True,
                   senior_feature_exercised=True, feature_exercised=None)
        # senior NOT exercised, junior broke → BOTH None at top level, but senior.False
        _write_row(dataset, "feat-senior-bad", senior_tests_pass=True,
                   senior_feature_exercised=False, feature_exercised=None)
        # old pre-fix row (no senior.feature_exercised field) → must not block (back-compat)
        _write_row(dataset, "feat-legacy", senior_tests_pass=True, feature_exercised=True)
        env = {"DUAL_DATASET": dataset}

        check("T6 senior-exercised passes even when junior broke (top-level None)",
              _lib_call("_senior_feature_exercised_ok feat-senior-ok", env).returncode == 0)
        check("T6 senior-NOT-exercised is BLOCKED (senior.feature_exercised False)",
              _lib_call("_senior_feature_exercised_ok feat-senior-bad", env).returncode != 0)
        check("T6 legacy row without the field does not block (back-compat)",
              _lib_call("_senior_feature_exercised_ok feat-legacy", env).returncode == 0)

        # end-to-end landability: green senior + broken junior still lands; unexercised blocks
        check("T6 _feature_landable TRUE for senior-exercised + broken junior",
              _lib_call("_feature_landable feat-senior-ok", env).returncode == 0)
        check("T6 _feature_landable FALSE for unexercised senior",
              _lib_call("_feature_landable feat-senior-bad", env).returncode != 0)


def test_t7_incomplete_junior_row_is_redoable():
    """Regression for the 2026-07-21 reorder trap: when the JUNIOR fails to compile, the
    profiler records nothing for it, so dual_track.py writes the row with the TOP-LEVEL
    `feature_exercised` collapsed to null (None) — NOT False. The resume gate must treat
    that incomplete measurement as NOT-done (so the plain batch re-runs it) and the pruner
    must be able to drop it, exactly as it does an explicit-False row. Previously
    `_feature_already_done` keyed on `is not False`, so a null (junior-broke) row read as
    'done' and the feature was skipped forever; and `_prune_unexercised_row` only dropped
    explicit False, so the real (null) invalid rows were never cleaned.

    Also pins the 2026-08-16 SPLIT of the two predicates. They used to be exact negations,
    which made a re-run delete its own measured history — see `_prune_unexercised_row`'s
    header in scripts/lib_trunk_chain.sh. Done-ness and data-bearing-ness are now asked
    separately, and a measured row with a red junior is not-done AND kept."""
    with tempfile.TemporaryDirectory() as td:
        dataset = os.path.join(td, "ds.jsonl")
        # feat-jbroke: senior fine, junior failed to COMPILE → top-level feature_exercised
        # is null (present, None), admitted False. This is the reorder-attempt-1/2 shape.
        _write_row(dataset, "feat-jbroke", senior_tests_pass=True,
                   feature_exercised=None, admitted=False)
        # feat-good: a complete, admitted pair (both tracks profiled).
        _write_row(dataset, "feat-good", senior_tests_pass=True,
                   feature_exercised=True, admitted=True)
        # feat-nonequiv: a fully-MEASURED row whose junior compiled and ran but FAILED its
        # tests (broke equivalence at runtime) → both tracks profiled → feature_exercised
        # True, junior.tests_pass False, admitted False. This is the volatility shape, and
        # the shape of 6 real corpus rows (churn x2, reorder, rfm x3). The two predicates
        # split on it: it is NOT done (a red junior is not a valid pair, so re-run it) but
        # it IS kept (the measurement is real, and it is the only record of the attempt).
        _write_row(dataset, "feat-nonequiv", senior_tests_pass=True,
                   feature_exercised=True, admitted=False, junior_tests_pass=False)
        # feat-legacy: pre-field row (no feature_exercised key) — back-compat, still done.
        _write_row(dataset, "feat-legacy", senior_tests_pass=True)
        env = {"DUAL_DATASET": dataset}

        check("T7 junior-compile-fail (feature_exercised=null) is NOT done → re-runnable",
              _lib_call("_feature_already_done feat-jbroke", env).returncode != 0)
        check("T7 complete admitted pair IS done",
              _lib_call("_feature_already_done feat-good", env).returncode == 0)
        check("T7 measured row with a RED junior is NOT done → re-runnable (HIGH 5: a red "
              "junior is not a valid pair, however complete the measurement)",
              _lib_call("_feature_already_done feat-nonequiv", env).returncode != 0)
        check("T7 legacy row without the field stays done (back-compat)",
              _lib_call("_feature_already_done feat-legacy", env).returncode == 0)

        # pruner must drop the null junior-broke row, keep everything valid
        _lib_call("_prune_unexercised_row feat-jbroke", env)
        remaining = [json.loads(l) for l in open(dataset) if l.strip()]
        ids = [r["feature_id"] for r in remaining]
        check("T7 pruner drops the null junior-broke row",
              "feat-jbroke" not in ids)
        check("T7 pruner keeps the admitted row", "feat-good" in ids)
        check("T7 pruner keeps the legacy row", "feat-legacy" in ids)

        # ── the split, asserted on the row the pruner is actually asked about ──
        # Until 2026-08-16 the pruner's predicate was the exact negation of the
        # done-check's, so pruning feat-nonequiv (which a re-run does, since it is
        # not-done) DELETED the only record of that attempt. The two questions are
        # different: "is this feature finished?" vs "does this row hold data?".
        _lib_call("_prune_unexercised_row feat-nonequiv", env)
        ids_after = [json.loads(l)["feature_id"] for l in open(dataset) if l.strip()]
        check("T7 pruner KEEPS the measured red-junior row even when pruning that very "
              "feature_id — a re-run must not delete its own measured history",
              "feat-nonequiv" in ids_after)
        check("T7 pruning feat-nonequiv leaves every other row untouched",
              sorted(ids_after) == ["feat-good", "feat-legacy", "feat-nonequiv"])


def main():
    test_t1_advance_basic()
    test_t1_delete_orphan()
    test_t2_deletion_guard()
    test_t3_decision_functions()
    test_t4_ensure_trunk_advanced_idempotent()
    test_t5_chain_manifest()
    test_t6_senior_feature_exercised_gate()
    test_t7_incomplete_junior_row_is_redoable()
    print()
    print("RESULT:", "ALL PASS" if not _fails else f"{len(_fails)} FAIL: {_fails}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
