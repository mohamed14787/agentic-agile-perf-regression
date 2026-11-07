import os

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from agile_team_cline.config import make_model


_MODEL_CACHE = {}

# Wenyan-ultra inter-agent compression (caveman skill, JuliusBrussee/caveman).
# Ceremony agents only — cline coding briefs and code/JSON payloads never pass
# through make_agent, so they stay uncompressed by construction.
_WENYAN_SUFFIX = (
    "\n\nINTER-AGENT DIALECT: you operate inside a multi-agent loop where every "
    "message you write is read by another agent, not a human. To minimize tokens, "
    "write all prose in extreme Classical Chinese (文言文) compression — 'wenyan-ultra'. "
    "Incoming messages from other agents use the same dialect; read them as such.\n"
    "Rules:\n"
    "- No greetings, hedging, filler, or descriptions of intent. Omit subjects, "
    "pronouns, and redundant verbs where context allows. Use classical particles "
    "(之, 乃, 為, 其) for logical relations.\n"
    "- Code keywords, identifiers, function names, file paths, task IDs, JSON keys, "
    "and exact error strings stay verbatim in Western alphanumerics — never "
    "translate or abbreviate them.\n"
    "- Semantic markers: causality →, error 🔴/敗, success 🟢/畢, "
    "connection/consumer pool 池, re-render/recompute 重繪/重算, memory leak 內存洩.\n"
    "- When a task demands a JSON object, output the JSON exactly per the requested "
    "schema; compress only prose string values (e.g. a notes field), never keys, "
    "IDs, or structure.\n"
    "Examples:\n"
    "- 'The component re-renders because a new object reference is created each "
    "render; wrap it in useMemo.' → 新參照→重繪。useMemo Wrap。\n"
    "- 'A connection pool reuses open connections instead of a new handshake per "
    "request, avoiding handshake latency under load.' → 池reuse conn。skip handshake → fast。\n"
    "AUTO-CLARITY — drop the dialect and use plain prose when: confirming a "
    "destructive/irreversible action; reporting a security or credential failure; a "
    "multi-step sequence whose order would be ambiguous without conjunctions; the "
    "receiving agent signals it could not parse your message."
)

# Modern-Chinese (现代汉语) inter-agent dialect — alternative to wenyan-ultra above.
# Same scope/guarantees: ceremony agents only, code/JSON/identifiers stay verbatim.
_MODERN_CHINESE_SUFFIX = (
    "\n\n语言设定：你在一个多智能体系统中工作，你写的每一条消息都由另一个智能体读取，"
    "而不是人类。为了减少token消耗，请用简洁的现代汉语撰写所有说明性文字（非文言文）。\n"
    "规则：\n"
    "- 不要寒暄、客套或解释你的意图，直接给出结论。\n"
    "- 代码关键字、标识符、函数名、文件路径、任务ID、JSON键名和准确的错误信息，"
    "必须保持西文原文，不要翻译或缩写。\n"
    "- 如果任务要求输出JSON对象，严格按照指定的schema输出；只用现代汉语压缩"
    "说明性的字符串字段（例如notes/concern字段的值），不要改动键名、ID或结构。\n"
    "特殊情况：在确认破坏性/不可逆操作、报告安全或凭证故障、多步骤顺序在缺少连接词"
    "时会产生歧义、或对方智能体表示无法解析你的消息时，放弃以上压缩规则，"
    "改用清晰的普通说明文字。"
)


def zh_prompt(en: str, zh: str) -> str:
    """Pick the Modern-Chinese prompt template when AGILE_CHINESE=1, else English.
    Used for camel ceremony-agent prompt templates (Architect/QA/PO/tech-debt) —
    never for cline briefs or JSON schemas, which must stay in English."""
    return zh if os.getenv("AGILE_CHINESE") == "1" else en


def make_agent(role: str, goal: str, max_tokens: int, model=None) -> ChatAgent:
    content = f"You are a {role}. {goal}"
    if os.getenv("AGILE_WENYAN") == "1":
        content += _WENYAN_SUFFIX
    if os.getenv("AGILE_CHINESE") == "1":
        content += _MODERN_CHINESE_SUFFIX
    system_msg = BaseMessage.make_assistant_message(
        role_name=role,
        content=content,
    )
    if model is None:
        if max_tokens not in _MODEL_CACHE:
            _MODEL_CACHE[max_tokens] = make_model(max_tokens)
        model = _MODEL_CACHE[max_tokens]
    return ChatAgent(
        system_message=system_msg,
        model=model,
    )
