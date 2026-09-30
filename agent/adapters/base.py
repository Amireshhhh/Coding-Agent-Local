"""Shared adapter helpers: retry/backoff, id normalization, response finalization."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from agent.adapters.parsers import extract_tool_calls
from agent.errors import AdapterError
from agent.types import FinishReason, Message, ModelResponse, ToolCall, Usage, new_call_id

log = logging.getLogger(__name__)

BACKOFF = (0.5, 1.0, 2.0)
TRUNCATED_CALL_ERROR = ("output was cut off at max_tokens; the tool call may be incomplete. "
                        "Re-send the complete call")
Sleep = Callable[[float], Awaitable[None]]


def is_retryable_status(code: int) -> bool:
    """429 and 5xx are retryable; other 4xx never are."""
    return code == 429 or 500 <= code < 600


def backoff_delay(attempt: int) -> float:
    """Delay before retry ``attempt`` (0-based): 0.5s, 1s, 2s, then 2s."""
    return BACKOFF[min(attempt, len(BACKOFF) - 1)]


async def send_with_retry(client: httpx.AsyncClient, request: httpx.Request, *, max_retries: int,
                          stream: bool = False, sleep: Sleep = asyncio.sleep,
                          what: str = "model request") -> httpx.Response:
    """Send ``request`` retrying on 429/5xx/connect errors with exponential backoff.

    Non-retryable responses (e.g. 400) are returned to the caller. For ``stream=True`` the
    returned response is open and must be closed by the caller.

    Raises:
        AdapterError: retries exhausted (message names the URL and last error).
    """
    last = ""
    for attempt in range(max_retries + 1):
        try:
            resp = await client.send(request, stream=stream)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError, httpx.ReadError) as e:
            last = f"{type(e).__name__}: {e}"
        else:
            if not is_retryable_status(resp.status_code):
                return resp
            body = (await resp.aread()).decode("utf-8", "replace")[:300]
            await resp.aclose()
            last = f"HTTP {resp.status_code}: {body}"
        if attempt < max_retries:
            delay = backoff_delay(attempt)
            log.warning("%s to %s failed (%s); retry %d/%d in %.1fs", what, request.url, last, attempt + 1,
                        max_retries, delay)
            await sleep(delay)
    raise AdapterError(f"{what} to {request.url} failed after {max_retries + 1} attempts: {last}. "
                       "Check that the server is running and reachable (base_url / http.endpoint).")


def ensure_unique_ids(calls: list[ToolCall]) -> list[ToolCall]:
    """Generate ids for missing/empty ones and regenerate duplicates (first occurrence kept)."""
    seen: set[str] = set()
    out: list[ToolCall] = []
    for c in calls:
        if not c.id or c.id in seen:
            c = c.model_copy(update={"id": new_call_id()})
        seen.add(c.id)
        out.append(c)
    return out


def map_finish_reason(raw: str | None) -> FinishReason:
    """Map a provider finish reason to the canonical set."""
    r = (raw or "stop").lower()
    if r in ("length", "max_tokens", "max_length", "model_length"):
        return "length"
    if r in ("tool_calls", "function_call", "tool_use"):
        return "tool_calls"
    if r in ("error",):
        return "error"
    return "stop"


def finalize_response(content: str | None, calls: list[ToolCall], raw_finish: str | None,
                      usage: Usage | None, known_tools: set[str], *,
                      parse_content: bool = True) -> ModelResponse:
    """Apply quirk handling common to all adapters and build a canonical ModelResponse.

    * If there are no structured calls, run the parser chain on ``content`` (quirk 4).
    * Ids are made unique (quirk 3); content ``""`` becomes ``None`` (quirk 6).
    * ``finish_reason`` is ``tool_calls`` whenever calls exist (quirk 5).
    * If the provider stopped for length, the last call is flagged as possibly truncated.
    """
    text = content if isinstance(content, str) else None
    if not calls and text and parse_content and known_tools:
        remaining, parsed = extract_tool_calls(text, known_tools)
        if parsed:
            calls, text = parsed, remaining
    calls = ensure_unique_ids(calls)
    reason = map_finish_reason(raw_finish)
    if calls and reason == "length" and calls[-1].parse_error is None:
        calls[-1] = calls[-1].model_copy(update={"parse_error": TRUNCATED_CALL_ERROR})
    if calls:
        reason = "tool_calls"
    elif reason == "tool_calls":
        reason = "stop"
    msg = Message(role="assistant", content=text if text else None, tool_calls=calls)
    return ModelResponse(message=msg, finish_reason=reason, usage=usage)


def json_snippet(data: Any, n: int = 300) -> str:
    """First ``n`` chars of a JSON rendering, for error messages."""
    try:
        s = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        s = repr(data)
    return s[:n]
