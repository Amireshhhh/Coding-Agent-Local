"""Canonical data contract (build plan section 2).

Everything inside the agent uses ONLY these types. Adapters translate to/from
provider formats; no provider-specific type may leak outside ``agent.adapters``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

TOOL_NAME_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


def new_call_id() -> str:
    """Return a fresh tool-call id: ``uuid4().hex[:12]``."""
    return uuid.uuid4().hex[:12]


class ToolSpec(BaseModel):
    """A tool as advertised to the model."""

    name: str = Field(pattern=TOOL_NAME_PATTERN)
    description: str
    parameters: dict[str, Any]


class ToolCall(BaseModel):
    """A tool invocation requested by the model. ``arguments`` is always a dict."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw: str | None = None
    parse_error: str | None = None


class Message(BaseModel):
    """One transcript entry."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class Usage(BaseModel):
    """Token usage reported by a provider (either field may be missing)."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None


FinishReason = Literal["stop", "tool_calls", "length", "error"]


class ModelResponse(BaseModel):
    """A complete model turn."""

    message: Message
    finish_reason: FinishReason
    usage: Usage | None = None


class Capabilities(BaseModel):
    """Static description of what an adapter/model supports."""

    native_tools: bool
    parallel_tool_calls: bool
    streaming: bool
    supports_system_role: bool = True
    supports_tool_role: bool = True
    context_window: int
    max_output_tokens: int = 4096
    prompted_tool_format: Literal["hermes", "json_fence", "custom"] = "hermes"


class StreamEvent(BaseModel):
    """Streaming event. ``tool_call`` is emitted only for complete, parsed calls."""

    type: Literal["text_delta", "tool_call", "done"]
    text: str | None = None
    tool_call: ToolCall | None = None
    response: ModelResponse | None = None


class ModelAdapter(Protocol):
    """The single internal contract every model backend implements."""

    capabilities: Capabilities

    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None,
        *,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> AsyncIterator[StreamEvent]: ...

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec] | None) -> int: ...


class TranscriptError(ValueError):
    """Raised when a transcript violates the tool-call/tool-result invariant."""


def validate_transcript(messages: list[Message], *, allow_pending_tail: bool = False) -> None:
    """Check the invariant: every assistant message with tool_calls is followed by
    exactly one ``tool`` message per call id before the next non-tool message.

    Also rejects orphan tool messages and tool_calls on non-assistant messages.

    Args:
        messages: transcript to check.
        allow_pending_tail: if True, the final assistant tool_calls group may be
            incomplete (results not yet appended).

    Raises:
        TranscriptError: describing the first violation found.
    """
    pending: list[str] = []
    for i, m in enumerate(messages):
        if m.tool_calls and m.role != "assistant":
            raise TranscriptError(f"message {i}: role {m.role!r} may not carry tool_calls")
        if m.role == "tool":
            if m.tool_call_id is None or m.tool_call_id not in pending:
                raise TranscriptError(f"message {i}: tool result {m.tool_call_id!r} has no pending call")
            pending.remove(m.tool_call_id)
            continue
        if pending:
            raise TranscriptError(f"message {i}: calls {pending} not answered before a {m.role} message")
        if m.role == "assistant" and m.tool_calls:
            ids = [c.id for c in m.tool_calls]
            if len(set(ids)) != len(ids):
                raise TranscriptError(f"message {i}: duplicate tool call ids {ids}")
            pending = list(ids)
    if pending and not allow_pending_tail:
        raise TranscriptError(f"end of transcript: calls {pending} have no results")
