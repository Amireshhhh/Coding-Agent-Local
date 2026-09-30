"""Adapter contract suite (5.9): every adapter must produce identical canonical output for the
same scenario. Provider responses are mocked with respx (HTTP adapters) or a mock TextBackend."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from agent.adapters import build_adapter
from agent.adapters.prompted import PromptedAdapter
from agent.config import load_config_dict
from agent.context.tokens import CharsDiv4Counter, MessageCounter
from agent.types import Message, ModelAdapter, ModelResponse, ToolSpec, Usage

BASE = "http://contract.test/v1"
EP = "http://contract.test/gen"
TOOLS = [ToolSpec(name="get_weather", description="Get weather",
                  parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})]
MSGS = [Message(role="system", content="S"), Message(role="user", content="Weather in Paris and Rome?")]

SCENARIOS: dict[str, dict[str, Any]] = {
    "single": {"calls": [("get_weather", {"city": "Paris"})], "content": None, "finish": "tool_calls"},
    "parallel": {"calls": [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Rome"})],
                 "content": None, "finish": "tool_calls"},
    "plain": {"calls": [], "content": "It is sunny.", "finish": "stop"},
    "length": {"calls": [], "content": "It is sun", "finish": "length"},
}


def hermes_text(s: dict[str, Any]) -> str:
    if s["calls"]:
        return "\n".join(f'<tool_call>\n{json.dumps({"name": n, "arguments": a})}\n</tool_call>' for n, a in s["calls"])
    return str(s["content"])


def raw_finish(s: dict[str, Any], native: bool) -> str:
    if s["finish"] == "length":
        return "length"
    return "tool_calls" if (s["calls"] and native) else "stop"


# ---------------- adapter factories (each installs its HTTP mock for the scenario) --------------
def openai_native(s: dict[str, Any]) -> ModelAdapter:
    msg: dict[str, Any] = {"role": "assistant", "content": s["content"]}
    if s["calls"]:
        msg["tool_calls"] = [{"id": f"id{i}", "type": "function",
                              "function": {"name": n, "arguments": json.dumps(a)}}
                             for i, (n, a) in enumerate(s["calls"])]
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"message": msg, "finish_reason": raw_finish(s, True)}]}))
    cfg = load_config_dict({"model": {"provider": "openai_compat", "base_url": BASE, "model": "m",
                                      "context_window": 4096, "max_output_tokens": 256}}, env={}).model
    return build_adapter(cfg)


def openai_prompted(s: dict[str, Any]) -> ModelAdapter:
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"role": "assistant", "content": hermes_text(s)},
                     "finish_reason": raw_finish(s, False)}]}))
    cfg = load_config_dict({"model": {"provider": "openai_compat", "base_url": BASE, "model": "m",
                                      "native_tools": False, "context_window": 4096, "max_output_tokens": 256}},
                           env={}).model
    return build_adapter(cfg)


def custom_prompted(s: dict[str, Any]) -> ModelAdapter:
    respx.post(EP).mock(return_value=httpx.Response(200, json={
        "data": {"text": hermes_text(s), "why": raw_finish(s, False)}}))
    cfg = load_config_dict({"model": {"provider": "custom_http", "context_window": 4096, "max_output_tokens": 256,
                                      "http": {"endpoint": EP, "request_template": '{"p": {{ prompt | tojson }}}',
                                               "response": {"text_path": "$.data.text",
                                                            "finish_reason_path": "$.data.why"}}}}, env={}).model
    return build_adapter(cfg)


def custom_native(s: dict[str, Any]) -> ModelAdapter:
    respx.post(EP).mock(return_value=httpx.Response(200, json={
        "out": s["content"] or "", "calls": [{"n": n, "a": a} for n, a in s["calls"]],
        "why": raw_finish(s, True)}))
    cfg = load_config_dict({"model": {
        "provider": "custom_http", "native_tools": True, "context_window": 4096, "max_output_tokens": 256,
        "http": {"endpoint": EP, "input_mode": "messages", "request_template": '{"m": {{ messages | tojson }}}',
                 "tool_mapping": {"request_tools_template": '{"tools": {{ tools | tojson }}}'},
                 "response": {"text_path": "$.out", "finish_reason_path": "$.why", "tool_calls_path": "$.calls[*]",
                              "tool_call_name_path": "$.n", "tool_call_args_path": "$.a"}}}}, env={}).model
    return build_adapter(cfg)


class _Backend:
    kind = "chat"
    supports_system_role = True
    supports_tool_role = False
    strict_alternation = False
    prompt_format = "chatml"
    prompt_template_file = None

    def __init__(self, s: dict[str, Any]) -> None:
        self.s = s

    async def generate(self, prompt_or_messages: Any, *, temperature: float, max_tokens: int,
                       stop: list[str] | None, extra: dict[str, Any] | None = None) -> tuple[str, Usage | None, str]:
        return hermes_text(self.s), None, raw_finish(self.s, False)


def prompted_mock(s: dict[str, Any]) -> ModelAdapter:
    return PromptedAdapter(_Backend(s), context_window=4096, max_output_tokens=256,  # type: ignore[arg-type]
                           counter=MessageCounter(CharsDiv4Counter()))


ADAPTERS: dict[str, Callable[[dict[str, Any]], ModelAdapter]] = {
    "openai_native": openai_native, "openai_prompted": openai_prompted, "custom_prompted": custom_prompted,
    "custom_native": custom_native, "prompted_mock": prompted_mock,
}


def canonical(r: ModelResponse) -> dict[str, Any]:
    return {"content": r.message.content, "finish": r.finish_reason, "role": r.message.role,
            "calls": [(c.name, c.arguments, c.parse_error) for c in r.message.tool_calls]}


def expected(s: dict[str, Any]) -> dict[str, Any]:
    return {"content": s["content"], "finish": s["finish"], "role": "assistant",
            "calls": [(n, a, None) for n, a in s["calls"]]}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("name", list(ADAPTERS))
@respx.mock
async def test_contract_complete(name: str, scenario: str) -> None:
    s = SCENARIOS[scenario]
    r = await ADAPTERS[name](s).complete(MSGS, TOOLS)
    assert canonical(r) == expected(s)
    ids = [c.id for c in r.message.tool_calls]
    assert len(set(ids)) == len(ids) and all(ids)


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("name", list(ADAPTERS))
@respx.mock
async def test_contract_same_output_across_adapters(name: str, scenario: str) -> None:
    s = SCENARIOS[scenario]
    ref = canonical(await openai_native(s).complete(MSGS, TOOLS))
    respx.reset()
    assert canonical(await ADAPTERS[name](s).complete(MSGS, TOOLS)) == ref


@pytest.mark.parametrize("name", list(ADAPTERS))
@respx.mock
async def test_contract_count_tokens_positive(name: str) -> None:
    a = ADAPTERS[name](SCENARIOS["plain"])
    assert await a.count_tokens(MSGS, TOOLS) > await a.count_tokens(MSGS, None) > 0
