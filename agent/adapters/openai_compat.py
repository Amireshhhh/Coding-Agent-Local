"""OpenAI-compatible ``/chat/completions`` adapter with native tool calling (build plan 5.2).

Works with Ollama, vLLM, llama.cpp server and LM Studio. Also provides text-only backends
(chat and legacy completions) used by :class:`agent.adapters.prompted.PromptedAdapter`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from agent.adapters.base import finalize_response, json_snippet, send_with_retry
from agent.adapters.parsers import normalize_arguments
from agent.config import ModelConfig
from agent.context.tokens import MessageCounter, build_counter
from agent.errors import AdapterError, NativeToolsUnsupported
from agent.types import Capabilities, Message, ModelResponse, StreamEvent, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)


# ---- lenient validation models for provider responses (never leave this module) ----------
class _Lenient(BaseModel):
    model_config = ConfigDict(extra="allow")


class _OAFunction(_Lenient):
    name: str | None = None
    arguments: Any = None


class _OAToolCall(_Lenient):
    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: _OAFunction | None = None


class _OAMessage(_Lenient):
    role: str | None = None
    content: Any = None
    tool_calls: list[_OAToolCall] | None = None


class _OAChoice(_Lenient):
    message: _OAMessage | None = None
    delta: _OAMessage | None = None
    text: str | None = None
    finish_reason: str | None = None


class _OAUsage(_Lenient):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _OAResponse(_Lenient):
    choices: list[_OAChoice] = []
    usage: _OAUsage | None = None


def _content_to_str(content: Any) -> str | None:
    """Content may be a string, null, or a list of parts ``[{"type":"text","text":...}]``."""
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Canonical -> OpenAI chat messages."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append({"role": "assistant", "content": m.content, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": json.dumps(c.arguments, ensure_ascii=False)}}
                for c in m.tool_calls]})
        elif m.role == "tool":
            d: dict[str, Any] = {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""}
            if m.name:
                d["name"] = m.name
            out.append(d)
        else:
            out.append({"role": m.role, "content": m.content or ""})
    return out


def to_openai_tools(tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
    """Canonical tool specs -> OpenAI ``tools`` array."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.parameters}} for t in tools or []]


def _usage(u: _OAUsage | None) -> Usage | None:
    if u is None or (u.prompt_tokens is None and u.completion_tokens is None):
        return None
    return Usage(prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens)


def _convert_calls(raw_calls: list[_OAToolCall]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for tc in raw_calls:
        fn = tc.function or _OAFunction()
        args, err = normalize_arguments(fn.arguments)
        raw = fn.arguments if isinstance(fn.arguments, str) else json.dumps(fn.arguments)
        calls.append(ToolCall(id=tc.id or "", name=fn.name or "unknown", arguments=args, raw=raw,
                              parse_error=err if fn.name else (err or "tool call has no function name")))
    return calls


def _mentions_tools(body: str) -> bool:
    return "tool" in body.lower()


class _HTTPBase:
    """Shared client/config plumbing."""

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s, connect=30))
        self._headers = {"Content-Type": "application/json"}
        if cfg.api_key and cfg.api_key != "none":
            self._headers["Authorization"] = f"Bearer {cfg.api_key}"
        self.url_base = cfg.base_url.rstrip("/")

    def _request(self, path: str, payload: dict[str, Any]) -> httpx.Request:
        return self._client.build_request("POST", f"{self.url_base}{path}", json=payload, headers=self._headers)

    async def _post_json(self, path: str, payload: dict[str, Any], *, tools_sent: bool) -> Any:
        resp = await send_with_retry(self._client, self._request(path, payload), max_retries=self.cfg.max_retries)
        return await self._check(resp, tools_sent=tools_sent, path=path)

    async def _check(self, resp: httpx.Response, *, tools_sent: bool, path: str) -> Any:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            if resp.status_code == 400 and tools_sent and _mentions_tools(body):
                raise NativeToolsUnsupported(f"{self.url_base}{path} rejected native tools: {body[:300]}")
            raise AdapterError(f"POST {self.url_base}{path} returned HTTP {resp.status_code}: {body[:300]}. "
                               "Check model name, base_url and api_key in agent.yaml.")
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            raise AdapterError(f"POST {self.url_base}{path} returned non-JSON body: {resp.text[:300]}") from e

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class OpenAICompatAdapter(_HTTPBase):
    """Native tool calling via ``POST {base_url}/chat/completions``."""

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None,
                 counter: MessageCounter | None = None) -> None:
        super().__init__(cfg, client)
        self.capabilities = Capabilities(
            native_tools=True, parallel_tool_calls=cfg.parallel_tool_calls, streaming=cfg.streaming,
            supports_system_role=True, supports_tool_role=True, context_window=cfg.context_window,
            max_output_tokens=cfg.max_output_tokens, prompted_tool_format=cfg.prompted_tool_format)
        self.counter = counter or MessageCounter(build_counter(cfg.effective_tokenizer()))

    def payload(self, messages: list[Message], tools: list[ToolSpec] | None, *, temperature: float,
                max_tokens: int | None, stop: list[str] | None, stream: bool) -> dict[str, Any]:
        """Build the request body. An empty tools list is omitted (quirk 9)."""
        p: dict[str, Any] = {"model": self.cfg.model, "messages": to_openai_messages(messages),
                             "temperature": temperature, "max_tokens": max_tokens or self.cfg.max_output_tokens,
                             "stream": stream}
        if tools:
            p["tools"] = to_openai_tools(tools)
            p["tool_choice"] = self.cfg.tool_choice
        if stop:
            p["stop"] = stop
        return p

    async def complete(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                       temperature: float = 0.2, max_tokens: int | None = None,
                       stop: list[str] | None = None) -> ModelResponse:
        """Non-streaming completion."""
        body = self.payload(messages, tools, temperature=temperature, max_tokens=max_tokens, stop=stop,
                            stream=False)
        data = await self._post_json("/chat/completions", body, tools_sent=bool(tools))
        try:
            resp = _OAResponse.model_validate(data)
        except ValidationError as e:
            raise AdapterError(f"unexpected /chat/completions response shape: {e.errors()[0]['msg']}; "
                               f"body: {json_snippet(data)}") from e
        if not resp.choices or resp.choices[0].message is None:
            raise AdapterError(f"/chat/completions response has no choices[0].message: {json_snippet(data)}")
        ch = resp.choices[0]
        msg = ch.message
        assert msg is not None
        calls = _convert_calls(msg.tool_calls or [])
        known = {t.name for t in tools or []}
        return finalize_response(_content_to_str(msg.content), calls, ch.finish_reason, _usage(resp.usage), known)

    async def stream(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                     temperature: float = 0.2, max_tokens: int | None = None,
                     stop: list[str] | None = None) -> AsyncIterator[StreamEvent]:
        """Streaming completion: text deltas immediately; tool calls when complete (quirk 7)."""
        body = self.payload(messages, tools, temperature=temperature, max_tokens=max_tokens, stop=stop, stream=True)
        req = self._request("/chat/completions", body)
        resp = await send_with_retry(self._client, req, max_retries=self.cfg.max_retries, stream=True)
        if resp.status_code >= 400:
            await self._check(resp, tools_sent=bool(tools), path="/chat/completions")
        text_parts: list[str] = []
        acc: dict[int, dict[str, Any]] = {}
        order: list[int] = []
        emitted: set[int] = set()
        finish: str | None = None
        usage: Usage | None = None

        def build(idx: int) -> ToolCall:
            a = acc[idx]
            args, err = normalize_arguments(a["args"])
            return ToolCall(id=a["id"] or "", name=a["name"] or "unknown", arguments=args, raw=a["args"],
                            parse_error=err if a["name"] else "tool call has no function name")

        try:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = _OAResponse.model_validate_json(data_str)
                except ValidationError:
                    log.warning("skipping malformed stream chunk: %s", data_str[:200])
                    continue
                if chunk.usage is not None:
                    usage = _usage(chunk.usage) or usage
                for ch in chunk.choices:
                    if ch.finish_reason:
                        finish = ch.finish_reason
                    d = ch.delta
                    if d is None:
                        continue
                    t = _content_to_str(d.content)
                    if t:
                        text_parts.append(t)
                        yield StreamEvent(type="text_delta", text=t)
                    for tc in d.tool_calls or []:
                        idx = tc.index if tc.index is not None else (order[-1] if order and not tc.id else len(order))
                        if idx not in acc:
                            # a new index closes all previous ones
                            for prev in order:
                                if prev not in emitted:
                                    call = build(prev)
                                    if call.parse_error is None and call.id:
                                        emitted.add(prev)
                                        yield StreamEvent(type="tool_call", tool_call=call)
                            acc[idx] = {"id": "", "name": "", "args": ""}
                            order.append(idx)
                        a = acc[idx]
                        if tc.id:
                            a["id"] = tc.id
                        if tc.function is not None:
                            if tc.function.name:
                                a["name"] += tc.function.name if not a["name"] else ""
                            if isinstance(tc.function.arguments, str):
                                a["args"] += tc.function.arguments
                            elif isinstance(tc.function.arguments, dict):
                                a["args"] = json.dumps(tc.function.arguments)
        finally:
            await resp.aclose()
        calls = [build(i) for i in order]
        known = {t.name for t in tools or []}
        final = finalize_response("".join(text_parts), calls, finish, usage, known)
        if final.usage is None or final.usage.prompt_tokens is None:  # quirk 8
            final.usage = Usage(prompt_tokens=await self.count_tokens(messages, tools),
                                completion_tokens=await self.counter.text("".join(text_parts)
                                                                          + "".join(a["args"] for a in acc.values())))
        emitted_ids = {acc[i]["id"] for i in emitted}
        for c in final.message.tool_calls:
            if c.parse_error is None and c.id not in emitted_ids:
                yield StreamEvent(type="tool_call", tool_call=c)
        yield StreamEvent(type="done", response=final)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec] | None) -> int:
        """Estimated prompt tokens including tool schemas."""
        return await self.counter.total(messages, tools)


class OpenAICompatTextBackend(_HTTPBase):
    """Text-only backend over an OpenAI-compatible server (no ``tools`` field).

    ``kind="chat"`` uses ``/chat/completions`` with role messages; ``kind="completion"``
    uses ``/completions`` with a single rendered prompt string.
    """

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None) -> None:
        super().__init__(cfg, client)
        self.kind: Literal["chat", "completion"] = cfg.backend_kind
        self.supports_system_role: bool = cfg.supports_system_role
        self.supports_tool_role: bool = cfg.supports_tool_role
        self.strict_alternation: bool = cfg.strict_alternation
        self.prompt_format: str = cfg.prompt_format
        self.prompt_template_file: str | None = cfg.prompt_template_file
        self.extra_body: dict[str, Any] = {}

    def _stops(self, stop: list[str]) -> list[str]:
        """OpenAI-style APIs accept at most 4 stop sequences; keep the first 4 and log the rest."""
        if len(stop) > 4:
            log.warning("server accepts at most 4 stop sequences; dropping %s", stop[4:])
        return stop[:4]

    async def generate(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float,
                       max_tokens: int, stop: list[str] | None,
                       extra: dict[str, Any] | None = None) -> tuple[str, Usage | None, str]:
        """Return ``(text, usage, raw_finish_reason)``."""
        body: dict[str, Any] = {"model": self.cfg.model, "temperature": temperature, "max_tokens": max_tokens}
        if stop:
            body["stop"] = self._stops(stop)
        body.update(extra or {})
        if self.kind == "chat":
            body["messages"] = prompt_or_messages
            data = await self._post_json("/chat/completions", body, tools_sent=False)
        else:
            body["prompt"] = prompt_or_messages
            data = await self._post_json("/completions", body, tools_sent=False)
        try:
            resp = _OAResponse.model_validate(data)
        except ValidationError as e:
            raise AdapterError(f"unexpected response shape: {json_snippet(data)}") from e
        if not resp.choices:
            raise AdapterError(f"response has no choices: {json_snippet(data)}")
        ch = resp.choices[0]
        text = ch.text if self.kind == "completion" else (_content_to_str(ch.message.content) if ch.message else None)
        return text or "", _usage(resp.usage), ch.finish_reason or "stop"

    async def generate_stream(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float,
                              max_tokens: int, stop: list[str] | None,
                              extra: dict[str, Any] | None = None) -> AsyncIterator[tuple[str, str | None]]:
        """Yield ``(text_delta, finish_reason_or_None)``."""
        body: dict[str, Any] = {"model": self.cfg.model, "temperature": temperature, "max_tokens": max_tokens,
                                "stream": True}
        if stop:
            body["stop"] = self._stops(stop)
        body.update(extra or {})
        path = "/chat/completions" if self.kind == "chat" else "/completions"
        body["messages" if self.kind == "chat" else "prompt"] = prompt_or_messages
        resp = await send_with_retry(self._client, self._request(path, body), max_retries=self.cfg.max_retries,
                                     stream=True)
        if resp.status_code >= 400:
            await self._check(resp, tools_sent=False, path=path)
        try:
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s == "[DONE]":
                    break
                try:
                    chunk = _OAResponse.model_validate_json(s)
                except ValidationError:
                    continue
                for ch in chunk.choices:
                    t = ch.text if self.kind == "completion" else (
                        _content_to_str(ch.delta.content) if ch.delta else None)
                    yield (t or "", ch.finish_reason)
        finally:
            await resp.aclose()
