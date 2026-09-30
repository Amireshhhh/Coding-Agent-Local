"""Prompted tool calling: gives ANY text-in/text-out model the ability to call tools (5.5)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from jinja2.sandbox import SandboxedEnvironment

from agent.adapters.base import finalize_response
from agent.adapters.parsers import extract_tool_calls, has_tool_markers
from agent.adapters.prompt_formats import STOP_TOKENS, render_prompt
from agent.context.tokens import MessageCounter
from agent.types import Capabilities, Message, ModelResponse, StreamEvent, ToolCall, ToolSpec, Usage

log = logging.getLogger(__name__)

DEFAULT_STOP = ["</tool_response>", "<tool_response>"]

HERMES_BLOCK = """You can call tools. Available tools (JSON Schema):
<tools>
{tools}
</tools>

To call a tool, output EXACTLY this and nothing else after it:
<tool_call>
{{"name": "<tool-name>", "arguments": {{<args-matching-schema>}}}}
</tool_call>

Rules:
- Emit one <tool_call> block per call. You may emit several blocks to call tools in parallel.
- Arguments MUST be valid JSON matching the tool's schema. Use double quotes. No comments. No trailing commas.
- After emitting tool calls, STOP. Tool results will be sent to you in <tool_response> blocks.
- Never write <tool_response> yourself.
- If no tool is needed, answer normally in plain text."""

JSON_FENCE_BLOCK = """You can call tools. Available tools (JSON Schema):
<tools>
{tools}
</tools>

To call a tool, output EXACTLY this and nothing else after it:
```tool_call
{{"name": "<tool-name>", "arguments": {{<args-matching-schema>}}}}
```

Rules:
- Emit one ```tool_call block per call. You may emit several blocks to call tools in parallel.
- Arguments MUST be valid JSON matching the tool's schema. Use double quotes. No comments. No trailing commas.
- After emitting tool calls, STOP. Tool results will be sent to you in <tool_response> blocks.
- Never write <tool_response> yourself.
- If no tool is needed, answer normally in plain text."""

GUIDED_BLOCK = """You can call tools. Available tools (JSON Schema):
<tools>
{tools}
</tools>

Reply with ONE JSON object and nothing else:
- to call tools: {{"tool_calls": [{{"name": "<tool-name>", "arguments": {{...}}}}]}}
- to answer without tools: {{"answer": "<your answer>"}}
Tool results will be sent to you in <tool_response> blocks."""


@runtime_checkable
class TextBackend(Protocol):
    """A text-in/text-out model.

    ``kind="chat"`` receives role messages; ``kind="completion"`` receives a single prompt
    string rendered with ``prompt_format``.
    """

    kind: Literal["chat", "completion"]
    supports_system_role: bool
    supports_tool_role: bool
    strict_alternation: bool
    prompt_format: str
    prompt_template_file: str | None

    async def generate(self, prompt_or_messages: str | list[dict[str, Any]], *, temperature: float,
                       max_tokens: int, stop: list[str] | None,
                       extra: dict[str, Any] | None = None) -> tuple[str, Usage | None, str]: ...


class GuidedDecoder:
    """Constrained decoding for the tool-call JSON (default OFF).

    ``kind="vllm"`` sends ``guided_json`` in the request body (vLLM ``extra_body``);
    ``kind="llamacpp"`` sends ``json_schema``. Both use :func:`build_tool_call_schema`.
    """

    def __init__(self, kind: Literal["vllm", "llamacpp"]) -> None:
        self.kind = kind

    def request_extras(self, tools: list[ToolSpec]) -> dict[str, Any]:
        """Extra request-body fields that constrain generation."""
        schema = build_tool_call_schema(tools)
        return {"guided_json": schema} if self.kind == "vllm" else {"json_schema": schema}

    @staticmethod
    def postprocess(text: str) -> str:
        """Turn ``{"answer": "..."}`` into plain text; leave tool-call JSON for the parser."""
        try:
            v = json.loads(text)
        except (ValueError, RecursionError):
            return text
        if isinstance(v, dict) and set(v) == {"answer"} and isinstance(v["answer"], str):
            return str(v["answer"])
        return text


def build_tool_call_schema(tools: list[ToolSpec]) -> dict[str, Any]:
    """JSON Schema: either ``{"answer": str}`` or ``{"tool_calls": [oneOf over tools]}``."""
    branches = [{"type": "object", "properties": {"name": {"const": t.name}, "arguments": t.parameters},
                 "required": ["name", "arguments"], "additionalProperties": False} for t in tools]
    return {"oneOf": [
        {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"],
         "additionalProperties": False},
        {"type": "object", "properties": {"tool_calls": {"type": "array", "minItems": 1,
                                                         "items": {"oneOf": branches}}},
         "required": ["tool_calls"], "additionalProperties": False},
    ]}


_ENV = SandboxedEnvironment()


def tools_lines(tools: list[ToolSpec]) -> str:
    """One JSON object per line: ``{"name","description","parameters"}``."""
    return "\n".join(json.dumps({"name": t.name, "description": t.description, "parameters": t.parameters},
                                ensure_ascii=False) for t in tools)


def render_call(c: ToolCall, fmt: str) -> str:
    """Render a call in the format the model is asked to produce."""
    body = json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False)
    if fmt == "json_fence":
        return f"```tool_call\n{body}\n```"
    return f"<tool_call>\n{body}\n</tool_call>"


class PromptedAdapter:
    """Wrap a :class:`TextBackend` and add tool calling via prompting + parsing."""

    def __init__(self, backend: TextBackend, *, context_window: int, max_output_tokens: int,
                 counter: MessageCounter, tool_format: Literal["hermes", "json_fence", "custom"] = "hermes",
                 tool_template: str | None = None, repair_attempts: int = 2,
                 guided: GuidedDecoder | None = None, streaming: bool = True) -> None:
        self.backend = backend
        self.counter = counter
        self.tool_format = tool_format
        self.tool_template = tool_template
        self.repair_attempts = repair_attempts
        self.guided = guided
        self.capabilities = Capabilities(
            native_tools=False, parallel_tool_calls=True, streaming=streaming,
            supports_system_role=backend.supports_system_role, supports_tool_role=backend.supports_tool_role,
            context_window=context_window, max_output_tokens=max_output_tokens, prompted_tool_format=tool_format)

    # ---------------- prompt construction ----------------
    def tool_block(self, tools: list[ToolSpec] | None) -> str:
        """The tool-instruction block injected into the system message ("" without tools)."""
        if not tools:
            return ""
        if self.guided is not None:
            return GUIDED_BLOCK.format(tools=tools_lines(tools))
        if self.tool_format == "custom":
            assert self.tool_template is not None
            tpl = _ENV.from_string(Path(self.tool_template).expanduser().read_text(encoding="utf-8"))
            return tpl.render(tools=[t.model_dump() for t in tools], tools_json=tools_lines(tools))
        block = JSON_FENCE_BLOCK if self.tool_format == "json_fence" else HERMES_BLOCK
        return block.format(tools=tools_lines(tools))

    def _call_format(self) -> str:
        return "json_fence" if self.tool_format == "json_fence" else "hermes"

    def translate(self, messages: list[Message], tools: list[ToolSpec] | None) -> list[dict[str, Any]]:
        """Canonical messages -> role/content dicts for a text-only model."""
        block = self.tool_block(tools)
        out: list[dict[str, Any]] = []
        sys_done = False
        i = 0
        call_order: dict[str, int] = {}
        while i < len(messages):
            m = messages[i]
            if m.role == "system":
                content = m.content or ""
                if block and not sys_done:
                    content = f"{content}\n\n{block}" if content else block
                sys_done = True
                out.append({"role": "system", "content": content})
            elif m.role == "assistant":
                parts = [m.content] if m.content else []
                for n, c in enumerate(m.tool_calls):
                    call_order[c.id] = n
                    parts.append(render_call(c, self._call_format()))
                out.append({"role": "assistant", "content": "\n".join(parts)})
            elif m.role == "tool":
                group: list[Message] = []
                while i < len(messages) and messages[i].role == "tool":
                    group.append(messages[i])
                    i += 1
                group.sort(key=lambda t: call_order.get(t.tool_call_id or "", 1 << 30))
                if self.backend.supports_tool_role:
                    for t in group:
                        out.append({"role": "tool", "content": t.content or "", "tool_call_id": t.tool_call_id,
                                    "name": t.name})
                else:
                    out.append({"role": "user", "content": "\n".join(
                        f'<tool_response id="{t.tool_call_id}" name="{t.name}">\n{t.content or ""}\n</tool_response>'
                        for t in group)})
                continue
            else:
                out.append({"role": "user", "content": m.content or ""})
            i += 1
        if block and not sys_done:
            out.insert(0, {"role": "system", "content": block})
        if not self.backend.supports_system_role:
            out = self._fold_system(out)
        if self.backend.strict_alternation:
            out = self._merge_same_role(out)
        return out

    @staticmethod
    def _fold_system(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system" and m["content"])
        rest = [dict(m) for m in msgs if m["role"] != "system"]
        if not system:
            return rest
        for m in rest:
            if m["role"] == "user":
                m["content"] = f"{system}\n\n{m['content']}"
                return rest
        return [{"role": "user", "content": system}, *rest]

    @staticmethod
    def _merge_same_role(msgs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in msgs:
            if out and out[-1]["role"] == m["role"]:
                out[-1] = {**out[-1], "content": f"{out[-1]['content']}\n\n{m['content']}"}
            else:
                out.append(dict(m))
        return out

    def _backend_input(self, msgs: list[dict[str, Any]]) -> tuple[str | list[dict[str, Any]], list[str]]:
        if self.backend.kind == "completion":
            prompt = render_prompt(msgs, self.backend.prompt_format, self.backend.prompt_template_file)
            return prompt, list(STOP_TOKENS.get(self.backend.prompt_format, []))
        return msgs, []

    # ---------------- generation ----------------
    async def _generate(self, msgs: list[dict[str, Any]], tools: list[ToolSpec] | None, temperature: float,
                        max_tokens: int, stop: list[str] | None) -> tuple[str, Usage | None, str]:
        inp, fmt_stop = self._backend_input(msgs)
        stops = list(dict.fromkeys([*DEFAULT_STOP, *fmt_stop, *(stop or [])]))
        extra = self.guided.request_extras(tools) if (self.guided and tools) else None
        text, usage, finish = await self.backend.generate(inp, temperature=temperature, max_tokens=max_tokens,
                                                          stop=stops, extra=extra)
        if self.guided and tools:
            text = self.guided.postprocess(text)
        return text, usage, finish

    @staticmethod
    def _needs_repair(text: str, calls: list[ToolCall]) -> str | None:
        errs = [f"{c.name}: {c.parse_error}" for c in calls if c.parse_error]
        if errs:
            return "; ".join(errs)
        if not calls and has_tool_markers(text):
            return "no valid tool call could be parsed from your output"
        return None

    async def complete(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                       temperature: float = 0.2, max_tokens: int | None = None,
                       stop: list[str] | None = None) -> ModelResponse:
        """Generate, extract tool calls, and run the bounded repair loop."""
        mt = max_tokens or self.capabilities.max_output_tokens
        known = {t.name for t in tools or []}
        msgs = self.translate(messages, tools)
        text, usage, finish = await self._generate(msgs, tools, temperature, mt, stop)
        return await self._finish(msgs, tools, known, text, usage, finish, temperature, mt, stop)

    async def _finish(self, msgs: list[dict[str, Any]], tools: list[ToolSpec] | None, known: set[str], text: str,
                      usage: Usage | None, finish: str, temperature: float, mt: int,
                      stop: list[str] | None) -> ModelResponse:
        remaining, calls = extract_tool_calls(text, known) if known else (text, [])
        problem = self._needs_repair(text, calls) if known else None
        attempt_text = text
        for attempt in range(self.repair_attempts):
            if problem is None:
                break
            log.info("prompted tool call invalid (%s); repair attempt %d/%d", problem, attempt + 1,
                     self.repair_attempts)
            marker = "```tool_call block" if self.tool_format == "json_fence" else "<tool_call> block"
            repair_msgs = [*msgs, {"role": "assistant", "content": attempt_text},
                           {"role": "user", "content": f"Your previous tool call was invalid: {problem}. "
                                                       f"Output ONLY a corrected {marker}."}]
            if not self.backend.supports_system_role:
                repair_msgs = self._fold_system(repair_msgs)
            if self.backend.strict_alternation:
                repair_msgs = self._merge_same_role(repair_msgs)
            attempt_text, r_usage, r_finish = await self._generate(repair_msgs, tools, 0.0, mt, stop)
            _, r_calls = extract_tool_calls(attempt_text, known)
            r_problem = self._needs_repair(attempt_text, r_calls)
            if r_problem is None and r_calls:
                calls, finish, problem = r_calls, r_finish, None
                usage = r_usage or usage
                break
            problem = r_problem or "no tool call in corrected output"
        return finalize_response(remaining, calls, finish, usage, known, parse_content=False)

    async def stream(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                     temperature: float = 0.2, max_tokens: int | None = None,
                     stop: list[str] | None = None) -> AsyncIterator[StreamEvent]:
        """Stream text deltas (holding back anything that may be tool-call markup), then calls."""
        mt = max_tokens or self.capabilities.max_output_tokens
        known = {t.name for t in tools or []}
        msgs = self.translate(messages, tools)
        gen_stream = getattr(self.backend, "generate_stream", None)
        if gen_stream is None or (self.guided and tools):
            resp = await self.complete(messages, tools, temperature=temperature, max_tokens=max_tokens, stop=stop)
            if resp.message.content:
                yield StreamEvent(type="text_delta", text=resp.message.content)
            for c in resp.message.tool_calls:
                if c.parse_error is None:
                    yield StreamEvent(type="tool_call", tool_call=c)
            yield StreamEvent(type="done", response=resp)
            return
        inp, fmt_stop = self._backend_input(msgs)
        stops = list(dict.fromkeys([*DEFAULT_STOP, *fmt_stop, *(stop or [])]))
        buf = ""
        sent = 0
        halted = False
        finish = "stop"
        markers = ("<tool_call>", "```", "[TOOL_CALLS]", "<|python_tag|>", '{"tool_calls"', "<think>")
        async for delta, fr in gen_stream(inp, temperature=temperature, max_tokens=mt, stop=stops):
            if fr:
                finish = fr
            if not delta:
                continue
            buf += delta
            if halted:
                continue
            hit = min((buf.find(mk, sent) for mk in markers if buf.find(mk, sent) >= 0), default=-1)
            if hit >= 0:
                if hit > sent:
                    yield StreamEvent(type="text_delta", text=buf[sent:hit])
                sent, halted = hit, True
                continue
            safe = len(buf)
            for mk in markers:  # hold back a suffix that could be the start of a marker
                for k in range(min(len(mk) - 1, len(buf) - sent), 0, -1):
                    if buf.endswith(mk[:k]):
                        safe = min(safe, len(buf) - k)
                        break
            if safe > sent:
                yield StreamEvent(type="text_delta", text=buf[sent:safe])
                sent = safe
        resp = await self._finish(msgs, tools, known, buf, None, finish, temperature, mt, stop)
        if resp.usage is None:
            resp.usage = Usage(prompt_tokens=await self.count_tokens(messages, tools),
                               completion_tokens=await self.counter.text(buf))
        if sent < len(buf) and not resp.message.tool_calls:  # e.g. a plain code fence was held back
            yield StreamEvent(type="text_delta", text=buf[sent:])
        for c in resp.message.tool_calls:
            if c.parse_error is None:
                yield StreamEvent(type="tool_call", tool_call=c)
        yield StreamEvent(type="done", response=resp)

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec] | None) -> int:
        """Prompt tokens including the rendered tool-instruction block (not the raw tools JSON)."""
        return await self.counter.total(messages, None, extra_text=self.tool_block(tools))
