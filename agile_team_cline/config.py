import os

# Reasoning models bill reasoning tokens against
# max_completion_tokens, so a tight budget gets eaten by reasoning → empty output
# ("No messages returned in step()"). Give the model lots of room, with the most on
# the reasoning-heavy CAMEL calls (architect planning / QA / review / default).
TASK_TOKEN_LIMITS = {
    "entity": 49152,
    "repository": 32768,
    "service": 98304,
    "controller": 98304,
    "config": 32768,
    "test": 65536,
    "integration": 32768,
    "default": 65536,
    "architect_planning": 98304,
    "qa_report": 49152,
    "review": 49152,
}

MAX_TASKS_PER_SPRINT = int(os.getenv("AGILE_MAX_TASKS_PER_SPRINT", "6"))
TOTAL_SPRINTS = int(os.getenv("AGILE_TOTAL_SPRINTS", "5"))


def _build_model(max_tokens: int, *, model_type: str, thinking: bool):
    from camel.models import OpenAICompatibleModel
    api_key = os.getenv("OPENAI_API_KEY", "")
    url = os.getenv("OPENAI_API_BASE_URL") or os.getenv("OPENAI_BASE_URL", "")
    temperature = float(os.getenv("TECHTEAM_TEMPERATURE", "0.2"))
    # Wall-clock bound per request. Without it a single call can block on a deep
    # gateway read for hours (observed 1–3h hangs). When it fires the client raises
    # APITimeoutError, which step_with_retry treats as transient and reissues — so a
    # hung call fails fast and the run keeps moving instead of stalling indefinitely.
    timeout = float(os.getenv("AGILE_LLM_TIMEOUT", "600"))
    # Real OpenAI (gpt-*) rejects the qwen-tuned params: max_tokens (wants
    # max_completion_tokens), non-default temperature/top_p, and the enable_thinking
    # extra_body. Emit an OpenAI-safe config there; keep the qwen config otherwise.
    if "api.openai.com" in url:
        config = {
            "frequency_penalty": 0.0,
            "max_completion_tokens": max_tokens,
        }
    else:
        config = {
            "temperature": temperature,
            "top_p": 0.3,
            "frequency_penalty": 0.0,
            "max_tokens": max_tokens,
            "extra_body": {
                "thinking": {
                    "type": "enabled" if thinking else "disabled",
                    "clear_thinking": True,
                }
            },
        }
    return OpenAICompatibleModel(
        model_type=model_type,
        api_key=api_key,
        url=url,
        timeout=timeout,
        model_config_dict=config,
    )


def make_model(max_tokens: int):
    return _build_model(
        max_tokens,
        model_type=os.getenv("TECHTEAM_MODEL_TYPE", "qwen-max"),
        thinking=False,
    )


def make_architect_model(max_tokens: int):
    # AGILE_ARCHITECT_THINKING=1 turns on extended thinking for the architect/
    # planner/tech-debt agents. Reasoning tokens share max_tokens with the JSON
    # contract/plan, and the architect budgets (≤6144) would let thinking truncate
    # the answer — so bump to 16384 when thinking is on (mirrors the QA toggle).
    thinking = os.getenv("AGILE_ARCHITECT_THINKING") == "1"
    if thinking:
        max_tokens = max(max_tokens, 16384)
    return _build_model(
        max_tokens,
        model_type=os.getenv("TECHTEAM_ARCHITECT_MODEL_TYPE", "google/gemma-4-31B-it"),
        thinking=thinking,
    )


def make_thinking_model(max_tokens: int):
    return _build_model(
        max_tokens,
        model_type=os.getenv("TECHTEAM_MODEL_TYPE", "qwen-35-35b-coding"),
        thinking=True,
    )
