"""Config-mapped in-house HTTP LLM (build plan 5.6). Onboarding a model needs only agent.yaml."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from jinja2 import TemplateError
from jinja2.sandbox import SandboxedEnvironment
from jsonpath_ng import parse as jp_parse

from agent.adapters.base import finalize_response, json_snippet, send_with_retry
from agent.adapters.parsers import normalize_arguments
from agent.config import HTTPConfig, ModelConfig
from agent.context.tokens import MessageCounter
from agent.errors import AdapterError, MappingError
from agent.types import Capabilities, Message, ModelResponse, StreamEvent, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)
_ENV = SandboxedEnvironment()
_ENV.policies["json.dumps_kwargs"] = {"ensure_ascii": False, "sort_keys": False}


class JSONPathCache:
    """Compiled JSONPath expressions keyed by (config key, expression)."""

    def __init__(self) -> None:
        self._c: dict[str, Any] = {}

    def find(self, key: str, expr: str, data: Any) -> list[Any]:
        """All values matched by ``expr``. Raises MappingError naming ``key`` for a bad expression."""
        if expr not in self._c:
            try:
                self._c[expr] = jp_parse(expr)
            except Exception as e:
                raise MappingError(f"{key} '{expr}' is not a valid JSONPath: {e}") from e
        return [m.value for m in self._c[expr].find(data)]

    def first(self, key: str, expr: str, data: Any, *, required: bool) -> Any:
        """First match; if ``required`` and nothing matches, raise a friendly MappingError."""
        vals = self.find(key, expr, data)
        if not vals:
            if required:
                raise MappingError(f"{key} '{expr}' matched nothing in: {json_snippet(data)}")
            return None
        return vals[0]


def render_json_template(key: str, template: str, **variables: Any) -> Any:
    """Render a jinja2 template in a sandbox and parse the result as JSON.

    Raises:
        MappingError: template error or invalid JSON (message names ``key``).
    """
    try:
        rendered = _ENV.from_string(template).render(**variables)
    except TemplateError as e:
        raise MappingError(f"{key}: template error: {e}") from e
    try:
        return json.loads(rendered)
    except json.JSONDecodeError as e:
        raise MappingError(f"{key} rendered invalid JSON ({e.msg} at pos {e.pos}): {rendered[:300]}") from e


def canonical_message_dicts(messages: list[Message]) -> list[dict[str, Any]]:
    """Messages as plain dicts for templates (``input_mode: messages`` with native tools)."""
    return [m.model_dump(exclude_none=True) for m in messages]


class CustomHTTPBackend:
    """TextBackend for an arbitrary HTTP LLM mapped entirely by config."""

    def __init__(self, cfg: ModelConfig, client: httpx.AsyncClient | None = None) -> None:
        assert cfg.http is not None
        self.cfg = cfg
        self.http: HTTPConfig = cfg.http
        self.kind: Literal["chat", "completion"] = "chat" if self.http.input_mode == "messages" else "completion"
        self.supports_system_role: bool = cfg.supports_system_role
        self.supports_tool_role: bool = cfg.supports_tool_role
        self.strict_alternation: bool = cfg.strict_alternation
        self.prompt_format: str = self.http.prompt_format
        self.prompt_template_file: str | None = self.http.prompt_template_file
        self.paths = JSONPathCache()
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(cfg.timeout_s, connect=30))
        self.last_request: Any = None
        self.last_response: Any = None

    def build_body(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float, max_tokens: int,
                   stop: list[str] | None, tools: list[dict[str, Any]] | None = None) -> Any:
        """Render ``http.request_template`` to a JSON body."""
        prompt = prompt_or_messages if isinstance(prompt_or_messages, str) else ""
        messages = prompt_or_messages if isinstance(prompt_or_messages, list) else []
        body = render_json_template("http.request_template", self.http.request_template, prompt=prompt,
                                    messages=messages, temperature=temperature, max_tokens=max_tokens,
                                    stop=stop or [], tools=tools or [])
        if tools and self.http.tool_mapping is not None:
            tool_part = render_json_template("http.tool_mapping.request_tools_template",
                                             self.http.tool_mapping.request_tools_template, tools=tools)
            if not isinstance(tool_part, dict) or not isinstance(body, dict):
                raise MappingError("http.tool_mapping.request_tools_template must render a JSON object and "
                                   "http.request_template must render a JSON object to merge into")
            body.update(tool_part)
        return body

    async def post(self, body: Any, *, stream: bool = False) -> httpx.Response:
        """Send the rendered body with retries; raise on HTTP errors."""
        self.last_request = body
        req = self._client.build_request(self.http.method.upper(), self.http.endpoint, json=body,
                                         headers=self.http.headers)
        resp = await send_with_retry(self._client, req, max_retries=self.cfg.max_retries, stream=stream,
                                     what="http.endpoint request")
        if resp.status_code >= 400:
            text = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            raise AdapterError(f"http.endpoint {self.http.endpoint} returned HTTP {resp.status_code}: {text[:300]}")
        return resp

    def parse_json(self, resp: httpx.Response) -> Any:
        """Response JSON; errors name ``http.endpoint``."""
        try:
            data = resp.json()
        except json.JSONDecodeError as e:
            raise MappingError(f"http.endpoint returned non-JSON body: {resp.text[:300]}") from e
        self.last_response = data
        err_path = self.http.response.error_path
        if err_path:
            err = self.paths.first("response.error_path", err_path, data, required=False)
            if err is not None:
                raise AdapterError(f"model server reported an error (response.error_path '{err_path}'): "
                                   f"{json_snippet(err)}")
        return data

    def extract_text(self, data: Any) -> str:
        """Apply ``response.text_path``."""
        r = self.http.response
        v = self.paths.first("response.text_path", r.text_path, data, required=True)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            v = "".join(v)
        if not isinstance(v, str):
            raise MappingError(f"response.text_path '{r.text_path}' selected {type(v).__name__}, expected a "
                               f"string, in: {json_snippet(data)}")
        return v

    def extract_meta(self, data: Any) -> tuple[Usage | None, str]:
        """Usage and finish reason via optional paths."""
        r = self.http.response
        fin = (self.paths.first("response.finish_reason_path", r.finish_reason_path, data, required=False)
               if r.finish_reason_path else None)
        pt = (self.paths.first("response.usage_prompt_path", r.usage_prompt_path, data, required=False)
              if r.usage_prompt_path else None)
        ct = (self.paths.first("response.usage_completion_path", r.usage_completion_path, data, required=False)
              if r.usage_completion_path else None)
        usage = None
        if isinstance(pt, int) or isinstance(ct, int):
            usage = Usage(prompt_tokens=pt if isinstance(pt, int) else None,
                          completion_tokens=ct if isinstance(ct, int) else None)
        return usage, str(fin) if fin is not None else "stop"

    async def generate(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float,
                       max_tokens: int, stop: list[str] | None,
                       extra: dict[str, Any] | None = None) -> tuple[str, Usage | None, str]:
        """Non-streaming generation (or aggregated streaming if ``stream.enabled``)."""
        if self.http.stream.enabled:
            parts: list[str] = []
            finish = "stop"
            async for delta, fr in self.generate_stream(prompt_or_messages, temperature=temperature,
                                                        max_tokens=max_tokens, stop=stop, extra=extra):
                parts.append(delta)
                finish = fr or finish
            return "".join(parts), None, finish
        body = self.build_body(prompt_or_messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
        if extra and isinstance(body, dict):
            body.update(extra)
        data = self.parse_json(await self.post(body))
        usage, finish = self.extract_meta(data)
        return self.extract_text(data), usage, finish

    async def generate_stream(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float,
                              max_tokens: int, stop: list[str] | None,
                              extra: dict[str, Any] | None = None) -> AsyncIterator[tuple[str, str | None]]:
        """Yield ``(delta, finish_or_None)`` from an SSE or NDJSON stream."""
        st = self.http.stream
        if not st.enabled:
            text, _, finish = await self.generate(prompt_or_messages, temperature=temperature,
                                                  max_tokens=max_tokens, stop=stop, extra=extra)
            yield text, finish
            return
        body = self.build_body(prompt_or_messages, temperature=temperature, max_tokens=max_tokens, stop=stop)
        if extra and isinstance(body, dict):
            body.update(extra)
        resp = await self.post(body, stream=True)
        try:
            async for raw_line in resp.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue
                if st.format == "sse":
                    if not line.startswith("data:"):
                        continue
                    line = line[5:].strip()
                if line == st.done_marker:
                    break
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as e:
                    raise MappingError(f"stream.format '{st.format}': line is not JSON: {line[:200]}") from e
                err_path = self.http.response.error_path
                if err_path and self.paths.first("response.error_path", err_path, chunk, required=False) is not None:
                    raise AdapterError(f"model server reported an error mid-stream: {json_snippet(chunk)}")
                deltas = self.paths.find("stream.delta_path", st.delta_path, chunk)
                fin = (self.paths.first("stream.finish_reason_path", st.finish_reason_path, chunk, required=False)
                       if st.finish_reason_path else None)
                text = "".join(d for d in deltas if isinstance(d, str))
                yield text, str(fin) if fin else None
        finally:
            await resp.aclose()

    async def aclose(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()


class CustomHTTPNativeAdapter:
    """Native tool calling for an in-house LLM with a config-described tool shape."""

    def __init__(self, cfg: ModelConfig, counter: MessageCounter, client: httpx.AsyncClient | None = None) -> None:
        assert cfg.http is not None and cfg.http.tool_mapping is not None
        self.backend = CustomHTTPBackend(cfg, client)
        self.cfg = cfg
        self.counter = counter
        self.capabilities = Capabilities(
            native_tools=True, parallel_tool_calls=cfg.parallel_tool_calls, streaming=False,
            supports_system_role=True, supports_tool_role=True, context_window=cfg.context_window,
            max_output_tokens=cfg.max_output_tokens, prompted_tool_format=cfg.prompted_tool_format)

    def _input(self, messages: list[Message]) -> str | list[dict[str, Any]]:
        from agent.adapters.prompt_formats import render_prompt

        dicts = canonical_message_dicts(messages)
        if self.backend.kind == "chat":
            return dicts
        return render_prompt([{"role": d["role"], "content": d.get("content", "")} for d in dicts],
                             self.backend.prompt_format, self.backend.prompt_template_file)

    def extract_calls(self, data: Any) -> list[ToolCall]:
        """Apply ``response.tool_calls_path`` and the per-call relative paths."""
        r = self.backend.http.response
        if not r.tool_calls_path:
            return []
        found = self.backend.paths.find("response.tool_calls_path", r.tool_calls_path, data)
        items: list[Any] = []
        for f in found:
            items.extend(f if isinstance(f, list) else [f])
        calls: list[ToolCall] = []
        for obj in items:
            name = self.backend.paths.first("response.tool_call_name_path", r.tool_call_name_path, obj, required=True)
            raw_args = self.backend.paths.first("response.tool_call_args_path", r.tool_call_args_path, obj,
                                                required=False)
            cid = (self.backend.paths.first("response.tool_call_id_path", r.tool_call_id_path, obj, required=False)
                   if r.tool_call_id_path else None)
            args, err = normalize_arguments(raw_args)
            calls.append(ToolCall(id=str(cid) if cid else "", name=str(name), arguments=args,
                                  raw=json_snippet(obj, 2000), parse_error=err))
        return calls

    async def complete(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                       temperature: float = 0.2, max_tokens: int | None = None,
                       stop: list[str] | None = None) -> ModelResponse:
        """One request; tool calls read via the configured JSONPaths."""
        tool_dicts = [t.model_dump() for t in tools or []]
        body = self.backend.build_body(self._input(messages), temperature=temperature,
                                       max_tokens=max_tokens or self.cfg.max_output_tokens, stop=stop,
                                       tools=tool_dicts or None)
        data = self.backend.parse_json(await self.backend.post(body))
        calls = self.extract_calls(data)
        text = self.backend.paths.first("response.text_path", self.backend.http.response.text_path, data,
                                        required=not calls)
        usage, finish = self.backend.extract_meta(data)
        known = {t.name for t in tools or []}
        return finalize_response(text if isinstance(text, str) else None, calls, finish, usage, known)

    async def stream(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                     temperature: float = 0.2, max_tokens: int | None = None,
                     stop: list[str] | None = None) -> AsyncIterator[StreamEvent]:
        """Non-streaming fallback presented as a stream."""
        resp = await self.complete(messages, tools, temperature=temperature, max_tokens=max_tokens, stop=stop)
        if resp.message.content:
            yield StreamEvent(type="text_delta", text=resp.message.content)
        for c in resp.message.tool_calls:
            if c.parse_error is None:
                yield StreamEvent(type="tool_call", tool_call=c)
        yield StreamEvent(type="done", response=resp)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec] | None) -> int:
        """Estimated prompt tokens including tool schemas."""
        return await self.counter.total(messages, tools)
