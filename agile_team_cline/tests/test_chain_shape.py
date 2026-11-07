"""No-LLM unit tests for the dataset feature-chain SHAPE helpers in dual_track.py
(2026-07-14): _chain_block (env → per-row chain topology) and _trunk_regression
(cumulative senior-N vs previous-landed-senior drift over shared methods).

Run:  .venv/bin/python -m agile_team_cline.tests.test_chain_shape
      (or under pytest — assertions are real, so pytest catches failures)

C1  _chain_block independent mode — no DUAL_CHAIN_* env → {"mode": "independent"}.
C2  _chain_block chained mode — env populated → sequence/parent/provenance/senior_tests_pass.
C3  _trunk_regression genesis — no parent_senior_out_dir → None.
C4  _trunk_regression drift — a pre-existing method slowed >20% lands in `regressed`;
    a method only in the candidate lands in `new_methods` (never `regressed`).
"""
import os
import sys
import tempfile

import agile_team_cline.dual_track as dt

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "profiling"))
import parse_kieker  # noqa: E402

_MShared = "public java.util.List demo.Repo.getAll(long,int,int)"
_MNew = "public java.util.List demo.Repo.getBasketLift()"

_CHAIN_ENV = ("DUAL_CHAIN_MODE", "DUAL_CHAIN_SEQUENCE", "DUAL_CHAIN_PARENT_ID",
              "DUAL_CHAIN_PARENT_SENIOR", "DUAL_CHAIN_TRUNK")


def _clear_chain_env():
    for k in _CHAIN_ENV:
        os.environ.pop(k, None)


def _write_dat(profile_dir, times_by_sig):
    """Write one Kieker OperationExecutionRecord .dat ($1 lines, no .map needed —
    parse_kieker falls back to ('$1',)). times_by_sig: {sig: [exec_ns, ...]}."""
    os.makedirs(profile_dir, exist_ok=True)
    with open(os.path.join(profile_dir, "kieker.dat"), "w") as f:
        eoi = 0
        for sig, times in times_by_sig.items():
            for t in times:
                # $1;writeTs;sig;session;trace;tin;tout;host;eoi;ess
                f.write(f"$1;0;{sig};ses;{eoi};0;{t};host;{eoi};0\n")
                eoi += 1


def _ops_from_times(times_by_sig):
    """Candidate side is passed in-memory (like senior_res): {sig: summary}, {sig: [ns]}."""
    rows = parse_kieker.summarise(times_by_sig)
    return {r["operation_signature"]: r for r in rows}, times_by_sig


def test_c1_chain_block_independent():
    _clear_chain_env()
    block = dt._chain_block({"tests_pass": True})
    assert block == {"mode": "independent"}, block


def test_c2_chain_block_chained():
    _clear_chain_env()
    os.environ.update({
        "DUAL_CHAIN_MODE": "chained-trunk",
        "DUAL_CHAIN_SEQUENCE": "3",
        "DUAL_CHAIN_PARENT_ID": "teastore-hist-dupes",
        "DUAL_CHAIN_PARENT_SENIOR": "teastore-hist-dupes-senior",
        "DUAL_CHAIN_TRUNK": "teastore-trunk",
    })
    try:
        block = dt._chain_block({"tests_pass": True})
    finally:
        _clear_chain_env()
    assert block["mode"] == "chained-trunk", block
    assert block["sequence"] == 3, block
    assert block["parent_feature_id"] == "teastore-hist-dupes", block
    assert block["parent_senior_out_dir"] == "teastore-hist-dupes-senior", block
    assert block["trunk_dir"] == "teastore-trunk", block
    assert block["senior_tests_pass"] is True, block
    assert block["baseline_provenance"] == "trunk@teastore-hist-dupes", block


def test_c3_trunk_regression_genesis():
    # genesis: chained mode but no parent senior → no cumulative comparison
    chain = {"mode": "chained-trunk", "parent_feature_id": "genesis",
             "parent_senior_out_dir": None}
    assert dt._trunk_regression(chain, {"operations": {"x": {}}}) is None


def test_c4_trunk_regression_drift():
    kieker_enabled = dt._KIEKER_ENABLED
    dt._KIEKER_ENABLED = True  # exercise the compare path regardless of ambient env
    try:
        with tempfile.TemporaryDirectory() as td:
            parent_senior = os.path.join(td, "parent-senior")
            # parent (baseline): shared method fast (~1ms), 6 calls
            _write_dat(os.path.join(parent_senior, "profile"),
                       {_MShared: [1_000_000] * 6})
            # candidate (this feature's senior): shared method slow (~5ms) + a NEW method
            cur_ops, cur_raw = _ops_from_times(
                {_MShared: [5_000_000] * 6, _MNew: [2_000_000] * 6})
            senior_res = {"operations": cur_ops, "raw_times": cur_raw, "tests_pass": True}
            chain = {"mode": "chained-trunk", "parent_feature_id": "feat-prev",
                     "parent_senior_out_dir": parent_senior}
            block = dt._trunk_regression(chain, senior_res)
    finally:
        dt._KIEKER_ENABLED = kieker_enabled

    assert block is not None, "expected a trunk_regression block"
    assert block["parent_feature_id"] == "feat-prev", block
    assert _MShared in block["regressed"], block
    assert _MNew in block["new_methods"], block
    assert _MNew not in block["regressed"], block
    # shared method is the only pre-existing one compared
    assert block["n_methods_compared"] == 1, block
    assert block["n_regressions"] == 1, block


def main():
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as e:
                print("FAIL", name, "-", e)
                fails.append(name)
    print()
    print("RESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
