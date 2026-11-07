import json
import os
import threading
import time


def _step_wallclock(agent, msg):
    """Run agent.step(msg) under an independent WALL-CLOCK deadline.

    The OpenAI client's own request timeout (config._build_model timeout=) should fire first
    in the normal case, but a stalled mid-stream TLS read is not reliably reclaimed by it —
    the copurchase probe run #1 hung ~19h at architect planning with AGILE_LLM_TIMEOUT=1800.
    This is the belt-and-suspenders wall-clock bound the old bare call lacked: a step() that
    blocks past AGILE_LLM_TIMEOUT + AGILE_LLM_HARD_BUFFER raises TimeoutError (name contains
    'Timeout' → step_with_retry treats it as transient and reissues). The abandoned worker
    thread is left to unwind when the socket finally errors — a bounded leak, vastly better
    than an unbounded stall. Escape hatch: AGILE_LLM_WALLCLOCK=0 restores the bare call.
    """
    if os.getenv("AGILE_LLM_WALLCLOCK", "1") == "0":
        return agent.step(msg)
    hard = (float(os.getenv("AGILE_LLM_TIMEOUT", "600"))
            + float(os.getenv("AGILE_LLM_HARD_BUFFER", "60")))
    if hard <= 0:
        return agent.step(msg)
    # A DAEMON worker so a truly-wedged step() never blocks interpreter exit (a
    # ThreadPoolExecutor's non-daemon threads would join at atexit and re-hang the process).
    box = {}
    def _run():
        try:
            box["r"] = agent.step(msg)
        except BaseException as e:   # re-raised on the caller thread → transient/quota logic intact
            box["e"] = e
    t = threading.Thread(target=_run, name="camel-step", daemon=True)
    t.start()
    t.join(hard)
    if t.is_alive():
        raise TimeoutError(
            f"agent.step exceeded wall-clock deadline of {hard:.0f}s — LLM hang backstop "
            "(the client request timeout did not reclaim a stalled read)")
    if "e" in box:
        raise box["e"]
    return box["r"]


# A weekly/monthly quota-exhausted 429 (e.g. error code 1310, "Weekly/Monthly Limit
# Exhausted") is NOT a transient rate-limit — the allotment does not reset for days. It must
# be raised immediately instead of consuming the whole retry/backoff budget and aborting
# anyway. Shared by the CAMEL and cline retry paths so both classify it the same way.
_QUOTA_EXHAUSTED_MARKERS = (
    "limit exhausted", "weekly limit", "monthly limit",
    "quota exceeded", "quota exhausted", "insufficient_quota", "out of quota",
    "'code': '1310'",  # Quota-exhausted error code
)

# A short-window quota ("Quota exceeded for quota metric ... per minute", Google-style) DOES
# reset within the retry budget — it is transient, never fatal, whatever marker it matches.
_QUOTA_TRANSIENT_WINDOWS = ("per minute", "per-minute", "per second", "per-second")


def is_quota_exhausted(msg: str) -> bool:
    m = (msg or "").lower()
    if any(w in m for w in _QUOTA_TRANSIENT_WINDOWS):
        return False
    return any(s in m for s in _QUOTA_EXHAUSTED_MARKERS)


def step_with_retry(agent, msg, retries: int = None, backoff: float = None):
    """Call agent.step() with retry on transient connection errors.

    Each attempt runs under an independent wall-clock deadline (_step_wallclock): the
    OpenAI client's own request timeout should fire first, but a stalled TLS read is not
    reliably reclaimed by it (observed 19h architect hang), so the wall-clock bound is the
    backstop. A breach raises TimeoutError, which the transient check below treats as
    retryable — a hung call fails fast and the run keeps moving instead of stalling forever.

    Defaults are env-tunable so a prolonged gateway degradation (e.g. a burst of
    504s where each attempt itself hangs for minutes) can be ridden out without
    tearing down the run: AGILE_LLM_RETRIES (count) and AGILE_LLM_BACKOFF (base
    seconds). Wait grows linearly with the attempt but is capped at
    AGILE_LLM_BACKOFF_MAX so it never sleeps absurdly long.
    """
    if retries is None:
        retries = int(os.getenv("AGILE_LLM_RETRIES", "6"))
    if backoff is None:
        backoff = float(os.getenv("AGILE_LLM_BACKOFF", "10.0"))
    backoff_max = float(os.getenv("AGILE_LLM_BACKOFF_MAX", "120.0"))
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = _step_wallclock(agent, msg)
            _log_camel_usage(agent, response)
            return response
        except Exception as e:
            last_exc = e
            name = type(e).__name__
            msg_text = str(e)

            # Quota-exhausted 429 is fatal, never retryable — raise now for a clean abort.
            if is_quota_exhausted(msg_text):
                print(f"      ✋ Fatal: model quota exhausted, not retrying — {msg_text[:160]}")
                raise

            transient_name = any(x in name for x in (
                "Connection", "Protocol", "Timeout", "APIConnection",
                "RateLimit", "ServiceUnavailable", "InternalServer",
                "BadGateway", "GatewayTimeout", "Overloaded",
            ))
            transient_msg = any(s in msg_text for s in (
                " 429", " 500", " 502", " 503", " 504",
                "rate limit", "Rate limit", "rate_limit",
                "overloaded", "Overloaded",
                "temporarily unavailable", "try again",
            ))

            if (transient_name or transient_msg) and attempt < retries:
                wait = min(backoff * (attempt + 1), backoff_max)
                print(f"      [Retry {attempt+1}/{retries}] Transient error ({name}), retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            raise
    raise last_exc


def _log_camel_usage(agent, response) -> None:
    """Append one JSONL record of this CAMEL agent.step() call's token usage to
    $AGILE_OUTPUT_DIR/camel_token_usage.jsonl. Mirrors cline_runner._log_usage
    semantics (AGILE_TOKEN_LOG=0 disables; never raises; no-op if AGILE_OUTPUT_DIR
    unset) so cline + camel spend can be totalled from one place."""
    if os.getenv("AGILE_TOKEN_LOG", "1") == "0":
        return
    output_dir = os.getenv("AGILE_OUTPUT_DIR")
    if not output_dir:
        return
    try:
        usage = (getattr(response, "info", None) or {}).get("usage") or {}
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "role": getattr(getattr(agent, "system_message", None), "role_name", ""),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
        with open(os.path.join(os.path.abspath(output_dir), "camel_token_usage.jsonl"), "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Search for `{"` to skip URL path-variable braces like {ownerId} that appear in
    # architect prose before the actual JSON object.
    start = text.find('{"')
    if start < 0:
        start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {}
