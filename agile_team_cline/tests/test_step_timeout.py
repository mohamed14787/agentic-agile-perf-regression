"""Regression test for the camel LLM wall-clock backstop (2026-07-13).

Bug: the copurchase probe run #1 hung ~19h at architect planning despite AGILE_LLM_TIMEOUT=1800.
step_with_retry called `agent.step(msg)` with NO independent wall-clock bound — its docstring
assumed the OpenAI client's own request timeout would fire, but a stalled mid-stream TLS read is
not reclaimed by that timeout (the documented TLS-hang class). So a single hung call stalled the
whole run forever.

Seam: step_with_retry is the shared call site for EVERY camel agent (architect/QA/review), so a
wall-clock deadline here fixes all of them. These tests stub `agent.step` with a FINITE-but-long
sleep so that pre-fix the assertion fails fast (elapsed too high) instead of hanging the suite.
"""
import os
import time
import unittest

from agile_team_cline.shared import step_with_retry


class _FakeResp:
    info = {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _SleepAgent:
    """Stands in for a CAMEL ChatAgent whose .step() blocks (simulating a hung TLS read)."""
    def __init__(self, sleep_s):
        self.sleep_s = sleep_s
        self.calls = 0
        self.system_message = None

    def step(self, msg):
        self.calls += 1
        time.sleep(self.sleep_s)
        return _FakeResp()


class TestStepWallClock(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in
                       ("AGILE_LLM_TIMEOUT", "AGILE_LLM_HARD_BUFFER", "AGILE_LLM_BACKOFF",
                        "AGILE_TOKEN_LOG", "AGILE_LLM_WALLCLOCK")}
        os.environ["AGILE_TOKEN_LOG"] = "0"        # no usage-log side effects
        os.environ["AGILE_LLM_BACKOFF"] = "0"      # no sleep between retries
        os.environ["AGILE_LLM_TIMEOUT"] = "0.2"    # wall-clock bound = 0.2 + buffer
        os.environ["AGILE_LLM_HARD_BUFFER"] = "0"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_hung_step_is_reclaimed_within_deadline(self):
        # step() blocks 3s but the wall-clock bound is 0.2s; with retries=0 the call must
        # RAISE within ~the deadline, not block for the full 3s.
        agent = _SleepAgent(sleep_s=3.0)
        t0 = time.time()
        with self.assertRaises(Exception) as ctx:
            step_with_retry(agent, "msg", retries=0)
        elapsed = time.time() - t0
        self.assertLess(elapsed, 1.5, f"step_with_retry did not honor the wall-clock bound (took {elapsed:.1f}s)")
        self.assertIn("Timeout", type(ctx.exception).__name__)

    def test_timeout_is_transient_and_retried(self):
        # The wall-clock breach must classify as transient so the call is reissued.
        agent = _SleepAgent(sleep_s=3.0)
        with self.assertRaises(Exception):
            step_with_retry(agent, "msg", retries=1)
        self.assertEqual(agent.calls, 2, "wall-clock timeout should be retried once (2 attempts)")

    def test_fast_step_returns_normally(self):
        # A well-behaved fast call is unaffected and returns its response.
        agent = _SleepAgent(sleep_s=0.0)
        resp = step_with_retry(agent, "msg", retries=0)
        self.assertIs(resp, _FakeResp) if False else self.assertTrue(hasattr(resp, "info"))
        self.assertEqual(agent.calls, 1)

    def test_wallclock_disabled_passthrough(self):
        # Escape hatch: AGILE_LLM_WALLCLOCK=0 restores the old bare-call behavior.
        os.environ["AGILE_LLM_WALLCLOCK"] = "0"
        agent = _SleepAgent(sleep_s=0.3)
        t0 = time.time()
        resp = step_with_retry(agent, "msg", retries=0)
        self.assertTrue(hasattr(resp, "info"))
        self.assertGreaterEqual(time.time() - t0, 0.3)  # it actually waited (no bound)


if __name__ == "__main__":
    unittest.main()
