"""Owns all interaction with the Cline CLI (https://docs.cline.bot).

The agile_team_cline fork delegates code authoring/editing to cline instead of
the in-process CAMEL model: this module is the single seam. It configures the
provider once, then runs headless act-mode sessions in a workspace directory and
parses cline's JSONL output.

Confirmed against cline 3.0.15:
  - provider id "openai-compatible" with --baseurl/--apikey/--modelid
  - bare prompt == act mode; auto-approve defaults to true (no -y flag)
  - -t/--timeout defaults to 0 (no timeout) so it MUST be passed
  - --json emits JSONL; the final {"type":"run_result", ...} line carries
    finishReason / text / usage / iterations / durationMs
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

# Ensure standard binary paths (Homebrew, /usr/local/bin) are in the environment PATH
# so that both node and the cline CLI binary can be resolved by all subprocess calls.
for _p in ("/opt/homebrew/bin", "/usr/local/bin"):
    if _p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{_p}:{os.environ.get('PATH', '')}"



def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _provider() -> str:
    return _env("AGILE_CLINE_PROVIDER", "openai-compatible")


def _model() -> str:
    return _env("AGILE_CLINE_MODEL") or _env("TECHTEAM_MODEL_TYPE", "qwen-35-35b-coding")


def _timeout() -> int:
    # Per-SESSION wall clock: a whole-sprint agentic session (read + write several
    # files + run mvn + self-correct) is far longer than one model call, so this
    # defaults much higher than the camel path's AGILE_LLM_TIMEOUT (600).
    return int(_env("AGILE_CLINE_TIMEOUT", "1800"))


def _compaction() -> str:
    # cline's default is "basic", which lets each iteration's input tokens grow
    # roughly linearly with iteration count (confirmed live: a single test-authoring
    # task ramped from ~6K to ~78K input tokens over 33 iterations, summing to 1.7M
    # tokens for ONE file). "agentic" compaction summarizes/prunes stale history instead of
    # resending it whole every iteration, directly targeting that growth. Overridable
    # per-run via AGILE_CLINE_COMPACTION if a specific stack/task needs "basic"/"off".
    return _env("AGILE_CLINE_COMPACTION", "agentic")


def _context_window() -> int:
    # cline's agentic compaction (above) is threshold-triggered on a % of the MODEL's
    # context window. For an openai-compatible model cline cannot detect the window
    # and falls back to openAiModelInfoSaneDefaults (~128K) — above our worst task's ~78K
    # peak, so compaction never fires. Registering a DELIBERATELY MODEST window (default
    # 60K) makes cline's derived maxAllowedSize trip a few times across a long 20-33
    # iteration task, condensing stale history instead of resending it whole. "0"/empty
    # disables the whole mechanism (cline keeps its own default). Keep ≥ ~45K so a single
    # large tool result (file embed / mvn error dump) can't exceed maxAllowedSize.
    # Defaulted OFF (2026-07-10): unverified against cline bug #2073 (UI doesn't
    # propagate a custom openai-compatible context window) and can lossily summarize
    # a type-precision task (test authoring). Superseded for now by the contract-driven
    # test-scoping filter in sprint.py's _build_test_tasks. Re-enable via
    # AGILE_CLINE_CONTEXT_WINDOW=60000 once verified live with quota.
    try:
        return int(_env("AGILE_CLINE_CONTEXT_WINDOW", "0"))
    except ValueError:
        return 0


def _cline_data_dir() -> str:
    # cline persists provider/model config + openAiModelInfo under this dir (default
    # ~/.cline). Honor AGILE_CLINE_DATA_DIR for an isolated/sandbox state dir if set.
    return os.path.expanduser(_env("AGILE_CLINE_DATA_DIR") or "~/.cline")


def _write_model_info() -> None:
    """Patch cline's global state so the openai-compatible model carries a modest
    contextWindow, enabling `--compaction agentic` to actually fire mid-task, AND so
    the CONFIGURED provider is actually the ACTIVE one cline uses on the next task.

    Two separate cline bugs land in the same fix here:
      1. `cline auth` can't set model info (no CLI flag) and the UI path is buggy
         (cline issue #2073), so we write the documented `openAiModelInfo` globalState
         key directly (added in cline PR #1540; read by OpenAiHandler when present,
         else openAiModelInfoSaneDefaults).
      2. `cline auth --provider openai-compatible ...` writes the settings correctly
         into data/settings/providers.json, but does NOT update globalState.json's
         planModeApiProvider/actModeApiProvider (+ the matching ...ApiModelId keys) —
         those select which provider/model is actually ACTIVE, and a stale value here
         silently overrides the just-configured provider on the next task (confirmed
         live 2026-08-25: an openai-compatible/OpenRouter auth call left globalState
         pinned on a leftover "zai" provider from account migration, so every task ran
         against https://api.z.ai/... with our modelid string spliced in, 400ing with
         "modelCode: does not exist" before any real generation happened — matches
         upstream cline#10520 "stale provider causes invalid API key errors" and
         cline#11357 "Act Mode silently falls back to Plan model"). Forcing both
         Plan/Act provider+model keys here is the documented community workaround.
    Runs for every openai-compatible auth (not just when a context window is
    configured) so the provider-pin fix always applies. Never raises — a failed
    state patch must not break a run.
    """
    if _provider() != "openai-compatible":
        return
    try:
        state_path = os.path.join(_cline_data_dir(), "data", "globalState.json")
        if not os.path.isfile(state_path):
            return
        with open(state_path) as f:
            state = json.load(f)

        # (2) force this provider+model to be the ACTIVE one for both Plan and Act
        # mode — see docstring above.
        model_id = _model()
        state["planModeApiProvider"] = "openai-compatible"
        state["actModeApiProvider"] = "openai-compatible"
        state["planModeApiModelId"] = model_id
        state["actModeApiModelId"] = model_id

        # (1) modest contextWindow so --compaction agentic can fire mid-task.
        window = _context_window()
        if window > 0:
            # maxTokens = max OUTPUT tokens per response. Kept generous (16K) and <
            # window: too low truncates a large test-file generation under
            # thinking=high; too high (or > window) breaks cline's input/output
            # budget math. Tune via env if needed.
            state["openAiModelInfo"] = {
                "contextWindow": window,
                "maxTokens": int(_env("AGILE_CLINE_MAX_TOKENS", "16384")),
                "supportsImages": False,
                "supportsPromptCache": False,
                "inputPrice": 0,
                "outputPrice": 0,
            }

        tmp = f"{state_path}.tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, state_path)
    except Exception:
        # never let a state-patch failure abort the pipeline
        pass


def _cline_bin() -> str:
    override = os.getenv("AGILE_CLINE_PATH")
    if override:
        return override
    # Try standard Homebrew path on Apple Silicon macOS first
    standard_mac = "/opt/homebrew/bin/cline"
    try:
        if os.path.exists(standard_mac):
            return standard_mac
    except OSError:
        pass
    # Try /usr/local/bin (Intel or other installations)
    standard_intel = "/usr/local/bin/cline"
    try:
        if os.path.exists(standard_intel):
            return standard_intel
    except OSError:
        pass
    # Fallback to bare name, which will search PATH
    return "cline"


@dataclass
class ClineResult:
    success: bool
    text: str = ""
    finish_reason: str = ""
    iterations: int = 0
    duration_ms: int = 0
    usage: dict = field(default_factory=dict)
    returncode: int = 0
    error: str = ""
    stdout: str = ""  # Raw cline JSONL stream for iteration visibility


_auth_done = False


def ensure_cline_auth() -> None:
    """Idempotently configure the cline provider from the pipeline's .env.

    Persists globally under ~/.cline; we still re-run once per process (cheap) so a
    fresh machine / changed endpoint is always reconciled before the first task.

    Prefers AGILE_CLINE_API_KEY/AGILE_CLINE_API_BASE_URL over OPENAI_API_KEY/
    OPENAI_API_BASE_URL when set, falling back to the latter otherwise (unchanged
    default behavior for every existing caller). This split exists because
    OPENAI_API_KEY/BASE_URL are ALSO read directly by agile_team_cline/config.py for
    the QA/Architect/Product-Owner CAMEL roles (TECHTEAM_MODEL_TYPE, independent of
    AGILE_CLINE_MODEL) — those roles must keep hitting their own endpoint even when
    ONLY cline's code-authoring step is being redirected elsewhere. Confirmed live
    (2026-08-26): scripts/run_openrouter_junior.sh overriding OPENAI_API_BASE_URL to
    route the junior's cline code step to OpenRouter/DeepSeek ALSO silently swept the
    QA/PO/Architect roles' glm-5.3 calls onto OpenRouter (real, unintended token spend
    — ~61K tokens on one feature alone) since they share the same two env vars but a
    completely separate model selector. The AGILE_CLINE_* pair lets a caller redirect
    ONLY cline's own endpoint, leaving OPENAI_API_KEY/BASE_URL (and therefore the CAMEL
    roles) untouched.
    """
    global _auth_done
    if _auth_done:
        return
    api_key = _env("AGILE_CLINE_API_KEY") or _env("OPENAI_API_KEY")
    base_url = (_env("AGILE_CLINE_API_BASE_URL") or _env("OPENAI_API_BASE_URL")
                or _env("OPENAI_BASE_URL"))
    if not api_key or not base_url:
        raise RuntimeError(
            "cline auth needs AGILE_CLINE_API_KEY/AGILE_CLINE_API_BASE_URL or "
            "OPENAI_API_KEY and OPENAI_API_BASE_URL/OPENAI_BASE_URL in env"
        )
    cmd = [
        _cline_bin(), "auth",
        "--provider", _provider(),
        "--apikey", api_key,
        "--baseurl", base_url,
        "--modelid", _model(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise RuntimeError(f"cline auth failed: {res.stderr.strip() or res.stdout.strip()}")
    # AFTER auth (which writes provider/model into globalState): patch in openAiModelInfo
    # so a modest contextWindow makes --compaction agentic fire mid-task. Must run after
    # auth so the auth write can't clobber it.
    _write_model_info()
    _auth_done = True


def _usage_tokens(usage: dict) -> tuple:
    """Best-effort (tokens_in, tokens_out) from cline's usage dict, tolerant of key
    naming (tokensIn/inputTokens/prompt_tokens, etc.)."""
    def pick(*names):
        for n in names:
            v = usage.get(n)
            if isinstance(v, (int, float)):
                return int(v)
        return 0
    tin = pick("tokensIn", "inputTokens", "input_tokens", "prompt_tokens", "promptTokens")
    tout = pick("tokensOut", "outputTokens", "output_tokens", "completion_tokens", "completionTokens")
    return tin, tout


def _log_usage(cwd: str, title: str, res: "ClineResult") -> None:
    """Append one JSONL record of this cline session's token usage to
    <cwd>/token_usage.jsonl. Enabled by default; set AGILE_TOKEN_LOG=0 to disable.
    Never raises — token accounting must not break a run."""
    if _env("AGILE_TOKEN_LOG", "1") == "0":
        return
    try:
        tin, tout = _usage_tokens(res.usage or {})
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "title": title,
            "finish_reason": res.finish_reason,
            "iterations": res.iterations,
            "duration_ms": res.duration_ms,
            "tokens_in": tin,
            "tokens_out": tout,
            "usage": res.usage or {},
        }
        with open(os.path.join(os.path.abspath(cwd), "token_usage.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _parse_run_result(stdout: str) -> dict:
    """Pull the final type=='run_result' object out of cline's JSONL stream."""
    result = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "run_result":
            result = obj
    return result


def _extract_plan_mode_output(stdout: str) -> str:
    """Extract analysis/reasoning from Cline plan-mode JSONL output.

    Cline outputs messages with type='say' containing text.
    Collects all 'say' type messages to reconstruct the analysis.
    """
    output_parts = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        # Capture all 'say' messages (the actual analysis/output from Cline)
        if obj.get("type") == "say":
            text = obj.get("text", "").strip()
            if text:
                output_parts.append(text)

    return "\n\n".join(output_parts) if output_parts else ""


def run_cline(cwd: str, brief: str, *, timeout: int = None, plan: bool = False,
              thinking: str = None, system: str = None) -> ClineResult:
    """Run one headless cline session in `cwd` and parse the result.

    Defaults to act mode (writes files). `plan=True` adds -p (plan mode: reads the
    workspace and reasons, does not edit) — used by the workspace-aware architect.
    `thinking` overrides AGILE_CLINE_THINKING for this session; `system` overrides
    the system prompt (-s). The outer subprocess timeout is a hard belt-and-suspenders
    cap on top of cline's own -t so a wedged session can never hang the run.
    """
    ensure_cline_auth()
    if timeout is None:
        timeout = _timeout()
    # Thinking: True → "high" (enable for qwen); False → "none" (off).
    # Valid levels: none, low, medium, high, xhigh.
    # String is honored as-is. None falls back to AGILE_CLINE_THINKING env default.
    if thinking is True:
        thinking = "high"
    elif thinking is False:
        thinking = "none"
    elif thinking is None:
        thinking = _env("AGILE_CLINE_THINKING", "medium")
    # search_codebase needs an absolute -c to index the workspace; with a relative
    # path it silently reports "Searched 0 files" for every query.
    cwd = os.path.abspath(cwd)
    cmd = [
        _cline_bin(),
        "--provider", _provider(),
        "--model", _model(),
        "--json",
        "--auto-approve", "true",
        "-t", str(timeout),
        "-c", cwd,
    ]
    cmd += ["--thinking", thinking]
    compaction = _compaction()
    if compaction:
        cmd += ["--compaction", compaction]
    if plan:
        cmd.append("-p")
    if system:
        cmd += ["-s", system]
    cmd.append(brief)
    hard_cap = timeout + int(_env("AGILE_CLINE_TIMEOUT_BUFFER", "120"))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=hard_cap, env=os.environ.copy())
    except subprocess.TimeoutExpired:
        return ClineResult(success=False, finish_reason="subprocess_timeout",
                           error=f"cline exceeded hard cap {hard_cap}s")

    run_result = _parse_run_result(res.stdout)
    finish = run_result.get("finishReason", "")
    success = res.returncode == 0 and finish == "completed"

    # For plan mode, extract analysis from JSONL stream if text field is empty
    text = run_result.get("text", "")
    if plan and not text:
        text = _extract_plan_mode_output(res.stdout)

    return ClineResult(
        success=success,
        text=text,
        finish_reason=finish or (f"exit {res.returncode}"),
        iterations=run_result.get("iterations", 0),
        duration_ms=run_result.get("durationMs", 0),
        usage=run_result.get("usage", {}),
        returncode=res.returncode,
        # On failure prefer stderr, but a morpheus mid-stream abort reports its message
        # in the stdout JSONL (run_result.error / .text) with empty stderr — fall back to
        # that so transient classification has something to match on.
        error="" if success else (
            (res.stderr.strip()[-2000:] if res.stderr else "")
            or str(run_result.get("error") or run_result.get("errorMessage") or "")[-2000:]
            or (text[-2000:] if (finish == "error" and text) else "")
        ),
        stdout=res.stdout,  # Raw JSONL for iteration analysis
    )


# Transient classification mirrors shared.step_with_retry so cline sessions ride
# out the same gateway degradation (504 bursts, rate limits) the camel path does.
_TRANSIENT = (
    " 429", " 500", " 502", " 503", " 504",
    "rate limit", "Rate limit", "rate_limit",
    "overloaded", "Overloaded", "temporarily unavailable", "try again",
    "timeout", "Timeout", "ECONNRESET", "ETIMEDOUT",
)


def _save_cline_jsonl(cwd: str, title: str, stdout: str) -> None:
    """Save raw cline JSONL stream to <cwd>/<title>_cline.jsonl for iteration visibility.
    Never raises — logging must not break a run."""
    if not title or not stdout:
        return
    try:
        import re
        # Sanitize title to a safe filename (letters, numbers, underscore, hyphen)
        safe_title = re.sub(r"[^a-zA-Z0-9_-]", "_", title[:60])
        path = os.path.join(os.path.abspath(cwd), f"{safe_title}_{time.strftime('%H%M%S')}_cline.jsonl")
        with open(path, "w") as f:
            for line in stdout.splitlines():
                if line.strip().startswith("{"):
                    f.write(line + "\n")
    except Exception:
        pass


def run_cline_with_retry(cwd: str, brief: str, *, timeout: int = None, title: str = "",
                         plan: bool = False, thinking: str = None, system: str = None) -> ClineResult:
    """run_cline with linear backoff on transient failures (env-tunable)."""
    retries = int(_env("AGILE_CLINE_RETRIES", "3"))
    backoff = float(_env("AGILE_CLINE_BACKOFF", "10.0"))
    backoff_max = float(_env("AGILE_CLINE_BACKOFF_MAX", "120.0"))
    eff_timeout = timeout if timeout is not None else _timeout()
    label = f" [{title}]" if title else ""
    from agile_team_cline.shared import is_quota_exhausted
    last = None
    for attempt in range(retries + 1):
        res = run_cline(cwd, brief, timeout=timeout, plan=plan, thinking=thinking, system=system)
        last = res
        _log_usage(cwd, title, res)
        if res.success and title and res.stdout:
            _save_cline_jsonl(cwd, title, res.stdout)
        if res.success:
            if res.iterations:
                tin, tout = _usage_tokens(res.usage or {})
                tok = f", {tin}+{tout} tok" if (tin or tout) else ""
                print(f"   ✅ cline{label}: {res.finish_reason} in {res.iterations} iters, "
                      f"{res.duration_ms/1000:.0f}s{tok}")
            return res
        # A weekly/monthly quota-exhausted 429 is fatal, never transient — stop now instead
        # of burning the whole retry budget and aborting anyway.
        if is_quota_exhausted(res.error or ""):
            print(f"   ✋ cline{label}: model quota exhausted, not retrying — "
                  f"{(res.error or '')[:160]}")
            # Machine-readable abort marker for orchestrators (dual_track) — the workspace
            # cwd IS the track's output dir, so <out>/INFRA_ABORT.json is checkable after
            # the pipeline exits regardless of stdout buffering/encoding.
            try:
                with open(os.path.join(cwd, "INFRA_ABORT.json"), "w") as f:
                    json.dump({"reason": "quota_exhausted", "title": title,
                               "error": (res.error or "")[:500],
                               "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=1)
            except Exception:
                pass
            break
        # An EARLY abort (model gives up after a tool-format slip, well before the
        # session timeout) is worth retrying; a slow abort that ran out the clock is
        # a genuine timeout and retrying just burns another full session.
        early_abort = (res.finish_reason == "aborted"
                       and res.duration_ms < eff_timeout * 1000 * 0.5)
        # finishReason=="error" is a provider/stream failure (morpheus dropped the
        # connection mid-session) — distinct from a clean "completed" with missing
        # files. Worth a retry rather than instantly halting a multi-hour run.
        stream_error = (res.finish_reason == "error")
        is_transient = (res.finish_reason == "subprocess_timeout" or early_abort
                        or stream_error
                        or any(s in (res.error or "") for s in _TRANSIENT))
        if is_transient and attempt < retries:
            wait = min(backoff * (attempt + 1), backoff_max)
            print(f"   [cline retry {attempt+1}/{retries}]{label} transient "
                  f"({res.finish_reason}); retrying in {wait:.0f}s...")
            time.sleep(wait)
            continue
        break
    print(f"   ⚠️ cline{label} failed: {last.finish_reason} — {last.error[:200] if last.error else ''}")
    # Step-timeout taxonomy (2026-07-18): a step that RAN OUT ITS CLOCK (subprocess hard
    # cap, or an abort that consumed >=50% of the budget) is an INFRASTRUCTURE failure, not
    # a model writing bad code — yet unlike quota-exhaustion it does NOT abort the pipeline,
    # so the run limps on with an incomplete file and the dataset row looks like a genuine
    # "junior failed to compile" sample (see 2026-07-16 teastore-hist-churn: the DTO step
    # timed out at 5400s, the retry lost its cline session, and the row landed status=ok /
    # build_success=False / infra_abort=False). Drop a durable marker in the track workspace
    # (cwd) so dual_track can tag the row `timed_out` and keep the failure taxonomy honest.
    # Additive + best-effort: never raises, mirroring the INFRA_ABORT.json marker.
    _timed_out = (last.finish_reason == "subprocess_timeout"
                  or (last.finish_reason == "aborted"
                      and (last.duration_ms or 0) >= eff_timeout * 1000 * 0.5))
    if _timed_out:
        try:
            with open(os.path.join(os.path.abspath(cwd), "step_timeouts.jsonl"), "a") as f:
                json.dump({"title": title, "finish_reason": last.finish_reason,
                           "duration_ms": last.duration_ms,
                           "timeout_s": eff_timeout,
                           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
                f.write("\n")
        except Exception:
            pass
    return last
