"""``agent doctor``: connectivity/mapping check and the 6-check tool-calling probe (5.6, 5.8)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from agent.adapters import AutoFallbackAdapter, build_adapter
from agent.adapters.custom_http import CustomHTTPBackend
from agent.adapters.prompted import PromptedAdapter
from agent.config import AgentConfig
from agent.trace import TraceRecorder, redact_text
from agent.types import Message, ModelAdapter, ModelResponse, ToolCall, ToolSpec

WEATHER = ToolSpec(name="get_weather", description="Get the current weather for a city.",
                   parameters={"type": "object", "properties": {
                       "city": {"type": "string", "description": "City name, e.g. Paris"}},
                       "required": ["city"], "additionalProperties": False})
CONFIGURE = ToolSpec(name="configure", description="Apply a configuration.", parameters={
    "type": "object", "properties": {
        "count": {"type": "integer", "description": "How many workers"},
        "enabled": {"type": "boolean", "description": "Whether the feature is on"},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tag list"},
        "options": {"type": "object", "description": "Options", "properties": {
            "level": {"type": "string", "enum": ["low", "high"], "description": "Level"},
            "depth": {"type": "integer", "description": "Depth"}},
            "required": ["level", "depth"], "additionalProperties": False}},
    "required": ["count", "enabled", "tags", "options"], "additionalProperties": False})
WRITE = ToolSpec(name="write_file", description="Write text content to a file.", parameters={
    "type": "object", "properties": {"path": {"type": "string", "description": "File path"},
                                     "content": {"type": "string", "description": "Full file content"}},
    "required": ["path", "content"], "additionalProperties": False})
SYSTEM = "You are a helpful assistant. Use the provided tools when they are needed."


@dataclass
class CheckResult:
    """Outcome of one probe check."""

    name: str
    passed: bool
    detail: str
    response: dict[str, Any] | None = field(default=None, repr=False)


def _valid(call: ToolCall, spec: ToolSpec) -> str | None:
    if call.parse_error:
        return f"parse error: {call.parse_error}"
    errs = list(Draft202012Validator(spec.parameters).iter_errors(call.arguments))
    return errs[0].message if errs else None


def _calls(r: ModelResponse, name: str) -> list[ToolCall]:
    return [c for c in r.message.tool_calls if c.name == name]


async def _run(adapter: ModelAdapter, msgs: list[Message], tools: list[ToolSpec] | None,
               max_tokens: int | None = None) -> ModelResponse:
    return await adapter.complete(msgs, tools, temperature=0.0, max_tokens=max_tokens)


async def check_single(a: ModelAdapter) -> CheckResult:
    """1. Single tool call must produce a valid call."""
    r = await _run(a, [Message(role="system", content=SYSTEM),
                       Message(role="user", content="What is the weather in Paris right now?")], [WEATHER])
    calls = _calls(r, "get_weather")
    if len(calls) < 1:
        return CheckResult("single_call", False, f"no get_weather call; got text: {(r.message.content or '')[:120]!r}")
    err = _valid(calls[0], WEATHER)
    ok = err is None and "paris" in str(calls[0].arguments.get("city", "")).lower()
    return CheckResult("single_call", ok, err or f"arguments={calls[0].arguments}")


async def check_parallel(a: ModelAdapter) -> CheckResult:
    """2. Two calls in one response for two cities."""
    r = await _run(a, [Message(role="system", content=SYSTEM),
                       Message(role="user", content="Get the current weather in Paris and in Tokyo. "
                                                    "Call the tool once for each city, in parallel.")], [WEATHER])
    calls = _calls(r, "get_weather")
    cities = {str(c.arguments.get("city", "")).lower() for c in calls if _valid(c, WEATHER) is None}
    ok = any("paris" in c for c in cities) and any("tokyo" in c for c in cities)
    return CheckResult("parallel_calls", ok, f"{len(calls)} call(s), cities={sorted(cities)}")


async def check_roundtrip(a: ModelAdapter) -> CheckResult:
    """3. Feed a tool result back; expect a final text answer using it."""
    call = ToolCall(id="call_probe_1", name="get_weather", arguments={"city": "Paris"})
    msgs = [Message(role="system", content=SYSTEM),
            Message(role="user", content="What is the weather in Paris right now?"),
            Message(role="assistant", content=None, tool_calls=[call]),
            Message(role="tool", tool_call_id=call.id, name="get_weather",
                    content='{"city": "Paris", "temperature_c": 17, "condition": "light rain"}')]
    r = await _run(a, msgs, [WEATHER])
    text = r.message.content or ""
    ok = not r.message.tool_calls and "17" in text
    return CheckResult("tool_result_roundtrip", ok, f"text={text[:120]!r}, calls={len(r.message.tool_calls)}")


async def check_types(a: ModelAdapter) -> CheckResult:
    """4. int, bool, array, nested object, enum arguments."""
    r = await _run(a, [Message(role="system", content=SYSTEM), Message(role="user", content=(
        "Call configure with: count 3, enabled true, tags ['alpha', 'beta'], options level 'high' and depth 2."))],
        [CONFIGURE])
    calls = _calls(r, "configure")
    if not calls:
        return CheckResult("argument_types", False, "no configure call")
    c = calls[0]
    err = _valid(c, CONFIGURE)
    want = {"count": 3, "enabled": True, "tags": ["alpha", "beta"], "options": {"level": "high", "depth": 2}}
    ok = err is None and c.arguments == want
    return CheckResult("argument_types", ok, err or f"arguments={json.dumps(c.arguments)}")


async def check_no_tool(a: ModelAdapter) -> CheckResult:
    """5. A question that needs no tool must not call one."""
    r = await _run(a, [Message(role="system", content=SYSTEM),
                       Message(role="user", content="What is 2 + 2? Answer with just the number.")], [WEATHER])
    ok = not r.message.tool_calls and "4" in (r.message.content or "")
    return CheckResult("no_tool_question", ok, f"calls={len(r.message.tool_calls)}, "
                                               f"text={(r.message.content or '')[:60]!r}")


async def check_long_argument(a: ModelAdapter, max_tokens: int) -> CheckResult:
    """6. A 200-line string argument."""
    r = await _run(a, [Message(role="system", content=SYSTEM), Message(role="user", content=(
        "Use write_file to create numbers.txt whose content is 200 lines: 'line 1', 'line 2', ... up to "
        "'line 200', one per line. Write all 200 lines."))], [WRITE], max_tokens=max_tokens)
    calls = _calls(r, "write_file")
    if not calls:
        return CheckResult("long_argument", False, f"no write_file call (finish={r.finish_reason})")
    err = _valid(calls[0], WRITE)
    lines = str(calls[0].arguments.get("content", "")).strip().splitlines()
    ok = err is None and len(lines) >= 200 and lines[0].strip() == "line 1" and lines[199].strip() == "line 200"
    return CheckResult("long_argument", ok, err or f"{len(lines)} lines")


async def probe_tools(adapter: ModelAdapter, max_output_tokens: int = 4096) -> list[CheckResult]:
    """Run all six checks; an exception in one check fails only that check."""
    checks = [check_single, check_parallel, check_roundtrip, check_types, check_no_tool]
    results: list[CheckResult] = []
    for fn in checks:
        try:
            results.append(await fn(adapter))
        except Exception as e:
            results.append(CheckResult(fn.__name__.removeprefix("check_"), False, f"{type(e).__name__}: {e}"))
    try:
        results.append(await check_long_argument(adapter, max(2048, min(max_output_tokens, 4096))))
    except Exception as e:
        results.append(CheckResult("long_argument", False, f"{type(e).__name__}: {e}"))
    return results


def recommend(results: list[CheckResult], cfg: AgentConfig, adapter: ModelAdapter) -> dict[str, Any]:
    """Recommended ``native_tools`` / ``schema_level`` / ``prompted_tool_format``."""
    by = {r.name: r.passed for r in results}
    native_now = adapter.capabilities.native_tools
    rec: dict[str, Any] = {"native_tools": cfg.model.native_tools, "schema_level": cfg.model.schema_level,
                           "prompted_tool_format": cfg.model.prompted_tool_format}
    if isinstance(adapter, AutoFallbackAdapter) and adapter.fell_back:
        rec["native_tools"] = False
    elif native_now and not (by.get("single_call") and by.get("tool_result_roundtrip")):
        rec["native_tools"] = False
    if not by.get("argument_types"):
        rec["schema_level"] = "simple"
    if not native_now and not by.get("single_call"):
        rec["prompted_tool_format"] = "json_fence" if cfg.model.prompted_tool_format == "hermes" else "hermes"
    return rec


def render_probe(results: list[CheckResult], rec: dict[str, Any]) -> str:
    """Human-readable probe report."""
    lines = ["Tool-calling probe:"]
    for r in results:
        lines.append(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    lines.append(f"Passed {sum(r.passed for r in results)}/{len(results)}")
    lines.append("Recommended config:")
    lines += [f"  {k}: {str(v).lower() if isinstance(v, bool) else v}" for k, v in rec.items()]
    return "\n".join(lines)


async def doctor(cfg: AgentConfig, *, probe: bool = False) -> tuple[str, bool]:
    """Send a test prompt, print raw request/response and mapping results; optionally probe tools.

    Returns ``(report, ok)``.
    """
    secrets = [cfg.model.api_key] + ([v for v in cfg.model.http.headers.values()] if cfg.model.http else [])
    rec = TraceRecorder(secrets=secrets)
    client = rec.client(cfg.model.timeout_s)
    out: list[str] = [f"provider: {cfg.model.provider}  model: {cfg.model.model or '(n/a)'}  "
                      f"context_window: {cfg.model.context_window}"]
    ok = True
    try:
        adapter = build_adapter(cfg.model, client)
        out.append(f"adapter: {type(adapter).__name__}"
                   + (f" (inner: {type(adapter.inner).__name__})" if isinstance(adapter, AutoFallbackAdapter) else ""))
        try:
            resp = await adapter.complete([Message(role="user", content="Reply with exactly the word: OK")], None,
                                          temperature=0.0, max_tokens=16)
            out.append(f"test prompt -> finish={resp.finish_reason} text={resp.message.content!r} usage={resp.usage}")
        except Exception as e:
            ok = False
            out.append(f"test prompt FAILED: {type(e).__name__}: {e}")
        for i, entry in enumerate(rec.entries, 1):
            out.append(f"--- request {i}: {entry['method']} {entry['url']} -> {entry.get('status')}")
            out.append(redact_text(json.dumps(entry.get("request"), ensure_ascii=False, indent=1)[:3000], secrets))
            out.append(f"--- response {i}:")
            out.append(redact_text(json.dumps(entry.get("response"), ensure_ascii=False, indent=1)[:3000], secrets))
        inner = adapter.inner if isinstance(adapter, AutoFallbackAdapter) else adapter
        backend = inner.backend if isinstance(inner, PromptedAdapter) else getattr(inner, "backend", None)
        if isinstance(backend, CustomHTTPBackend) and backend.last_response is not None:
            out.append("mapping results:")
            out.append(_mapping_report(backend))
        if probe and ok:
            results = await probe_tools(adapter, cfg.model.max_output_tokens)
            out.append(render_probe(results, recommend(results, cfg, adapter)))
            ok = all(r.passed for r in results)
    finally:
        await client.aclose()
    return "\n".join(out), ok


def _mapping_report(b: CustomHTTPBackend) -> str:
    r = b.http.response
    lines = []
    for key in ("text_path", "finish_reason_path", "usage_prompt_path", "usage_completion_path", "error_path",
                "tool_calls_path"):
        expr = getattr(r, key)
        if not expr:
            continue
        try:
            vals = b.paths.find(f"response.{key}", expr, b.last_response)
            lines.append(f"  response.{key} '{expr}' -> {json.dumps(vals[:3], ensure_ascii=False)[:300]}")
        except Exception as e:
            lines.append(f"  response.{key} '{expr}' -> ERROR {e}")
    return "\n".join(lines)
