"""Conversation compaction (build plan 6.3 stage 3)."""

from __future__ import annotations

import json

from agent.types import Message, ModelAdapter

COMPACTION_PROMPT = """Summarize the conversation so far for an AI coding agent that will continue the work.
Preserve, as concise bullet lists: (1) the user's goal and constraints; (2) decisions made and why;
(3) files created/modified with a one-line description of each change; (4) commands run and key results;
(5) errors encountered and their fixes; (6) open TODOs and the next step.
Keep exact file paths, function names, and error messages. Do not invent details. Max 600 words."""

SUMMARY_TAG = "[Conversation summary]"
SUMMARIZER_SYSTEM = "You write faithful, compact summaries of conversations between a user and an AI coding agent."


def render_transcript(messages: list[Message]) -> str:
    """Plain-text rendering of messages for the summarizer (no tool-role messages are sent)."""
    lines: list[str] = []
    for m in messages:
        if m.role == "tool":
            lines.append(f"[tool result: {m.name or '?'} id={m.tool_call_id}]\n{m.content or ''}")
            continue
        head = f"[{m.role}]"
        body = m.content or ""
        for c in m.tool_calls:
            body += f"\n[tool call: {c.name} {json.dumps(c.arguments, ensure_ascii=False)}]"
        lines.append(f"{head}\n{body}".rstrip())
    return "\n\n".join(lines)


def build_compaction_request(messages: list[Message]) -> list[Message]:
    """The exact messages sent to the model to compact ``messages``."""
    return [Message(role="system", content=SUMMARIZER_SYSTEM),
            Message(role="user", content=f"<conversation>\n{render_transcript(messages)}\n</conversation>\n\n"
                                         f"{COMPACTION_PROMPT}")]


async def summarize(adapter: ModelAdapter, messages: list[Message], max_tokens: int = 1200) -> str:
    """Ask ``adapter`` (temperature 0, no tools) for a summary of ``messages``."""
    resp = await adapter.complete(build_compaction_request(messages), None, temperature=0.0, max_tokens=max_tokens)
    text = (resp.message.content or "").strip()
    if not text:
        raise ValueError("compaction returned an empty summary")
    return text


def summary_message(text: str) -> Message:
    """The user message that replaces a summarized range."""
    return Message(role="user", content=f"{SUMMARY_TAG}\n{text}")


def is_summary(m: Message) -> bool:
    """True for a summary message produced by :func:`summary_message`."""
    return m.role == "user" and (m.content or "").startswith(SUMMARY_TAG)
