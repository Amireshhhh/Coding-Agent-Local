"""Phase 1 demo harness: ``python -m agent.demo``.

Runs three scripted scenarios end-to-end through the real stack (config -> adapter -> HTTP ->
parser -> registry -> MCP -> context manager -> loop) against local fake model servers:

1. native tools: the model calls an MCP tool and a local tool in parallel, gets both results, answers.
2. native-tools rejection: the server answers HTTP 400 for ``tools``; the adapter falls back to prompted
   (Hermes) tool calling; the model reads then edits a file.
3. in-house model with a nonstandard JSON shape, onboarded purely through an ``agent.yaml`` file.

With ``--config agent.yaml --live`` it additionally runs scenario 1's prompt against a real model.
Exit code 0 only if every scenario passes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agent.app import build_session
from agent.config import AgentConfig, load_config, load_config_dict
from agent.mcp.config import StdioServer
from agent.tools.base import FunctionTool
from agent.types import validate_transcript


# ------------------------------------------------------------------ fake model servers
def _tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in reversed(messages):
        if m.get("role") != "tool":
            break
        out.append(m)
    return list(reversed(out))


async def native_chat(request: Request) -> JSONResponse:
    """OpenAI-compatible server WITH native tool calling (scenario 1)."""
    body = await request.json()
    msgs = body["messages"]
    names = {t["function"]["name"] for t in body.get("tools", [])}
    last = msgs[-1]
    if last["role"] == "user" and "DEMO-PARALLEL" in (last.get("content") or ""):
        missing = {"mcp__demo__add", "clock"} - names
        if missing:
            return JSONResponse({"error": {"message": f"demo model: tools {sorted(missing)} not offered"}}, 422)
        return JSONResponse({"choices": [{"finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "tool_calls": [
                {"id": "call_add", "type": "function",
                 "function": {"name": "mcp__demo__add", "arguments": json.dumps({"a": 2, "b": 3})}},
                {"id": "call_clock", "type": "function", "function": {"name": "clock", "arguments": "{}"}}]}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 30}})
    if last["role"] == "tool":
        res = {m["tool_call_id"]: m["content"] for m in _tool_results(msgs)}
        text = f"2 + 3 = {res.get('call_add')}; the clock reads {res.get('call_clock')}."
        return JSONResponse({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                             "usage": {"prompt_tokens": 260, "completion_tokens": 20}})
    return JSONResponse({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}]})


async def text_only_chat(request: Request) -> JSONResponse:
    """OpenAI-compatible server WITHOUT tool support: 400 on ``tools`` (scenario 2)."""
    body = await request.json()
    if "tools" in body:
        return JSONResponse({"error": {"message": "this model does not support tools"}}, 400)
    msgs = body["messages"]
    last = msgs[-1].get("content") or ""
    if "DEMO-EDIT" in last:
        out = '<tool_call>\n{"name": "read_file", "arguments": {"path": "greeting.txt"}}\n</tool_call>'
    elif 'name="read_file"' in last:
        out = ('I will update it.\n<tool_call>\n{"name": "edit_file", "arguments": {"path": "greeting.txt", '
               '"old_str": "Hello", "new_str": "Goodbye"}}\n</tool_call>\n<tool_response>fake</tool_response>')
    elif 'name="edit_file"' in last:
        out = "Done: greeting.txt now says Goodbye."
    else:
        out = "OK"
    return JSONResponse({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": out}}]})


async def inhouse_generate(request: Request) -> JSONResponse:
    """In-house model with a nonstandard JSON shape (scenario 3)."""
    body = await request.json()
    prompt = body["conversation"]
    turn = prompt.rsplit("<|im_start|>user\n", 1)[-1]
    if "DEMO-INHOUSE" in turn:
        out = '<tool_call>\n{"name": "ls", "arguments": {}}\n</tool_call>'
    elif '<tool_response id="' in turn and 'name="ls"' in turn:
        entries = re.findall(r"^(\S+)", turn.split("\n", 1)[1], re.M)
        out = f"The project contains: {', '.join(e for e in entries if not e.startswith('<'))}."
    else:
        out = "OK"
    return JSONResponse({"payload": {"candidates": [{"output": out}]},
                         "meta": {"why": "eos", "tok": {"in": len(prompt) // 4, "out": len(out) // 4}}})


def fake_app() -> Starlette:
    """All three fake model endpoints."""
    return Starlette(routes=[Route("/native/v1/chat/completions", native_chat, methods=["POST"]),
                             Route("/textonly/v1/chat/completions", text_only_chat, methods=["POST"]),
                             Route("/inhouse/generate", inhouse_generate, methods=["POST"])])


def free_port() -> int:
    """An unused localhost TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class FakeServer:
    """Runs :func:`fake_app` with uvicorn on 127.0.0.1 inside the current event loop."""

    def __init__(self) -> None:
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(fake_app(), host="127.0.0.1", port=self.port, log_level="error"))
        self.task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> FakeServer:
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(200):
            if self.server.started:
                return self
            await asyncio.sleep(0.02)
        raise RuntimeError("fake model server did not start")

    async def __aexit__(self, *exc: object) -> None:
        self.server.should_exit = True
        if self.task is not None:
            await self.task

    @property
    def base(self) -> str:
        """Base URL."""
        return f"http://127.0.0.1:{self.port}"


# ------------------------------------------------------------------ scenarios
def _cfg(model: dict[str, Any], sessions: str) -> AgentConfig:
    return load_config_dict({"model": model, "context": {"session_dir": sessions},
                             "permissions": {"mode": "accept_edits", "allow": ["mcp__demo__*"]}}, env={})


async def scenario_parallel(base: str, work: Path, cfg: AgentConfig | None = None) -> tuple[bool, str]:
    """Scenario 1: MCP tool + local tool in parallel."""
    cfg = cfg or _cfg({"provider": "openai_compat", "base_url": f"{base}/native/v1", "model": "demo",
                       "context_window": 8192, "max_output_tokens": 512}, str(work / "sessions"))
    servers = {"demo": StdioServer(command=sys.executable, args=["-m", "agent.demo_mcp_server"])}
    s = await build_session(cfg, work, mcp_servers=servers, mcp_log_dir=str(work / "logs"))

    async def clock(args: dict[str, Any]) -> str:
        return "12:00"
    s.registry.register(FunctionTool("clock", "Current time (fake local tool).",
                                     {"type": "object", "properties": {}}, clock, read_only=True))
    try:
        res = await s.agent.run("DEMO-PARALLEL: what is 2+3 and what time is it?")
        validate_transcript(s.agent.messages)
        tool_msgs = {m.tool_call_id: m.content for m in s.agent.messages if m.role == "tool"}
        ok = (res.text == "2 + 3 = 5; the clock reads 12:00." and tool_msgs == {"call_add": "5", "call_clock": "12:00"}
              and res.tool_calls == 2 and res.iterations == 2)
        return ok, f"answer={res.text!r} tool_results={tool_msgs} iterations={res.iterations}"
    finally:
        await s.aclose()


async def scenario_fallback_edit(base: str, work: Path) -> tuple[bool, str]:
    """Scenario 2: automatic fallback to prompted tool calling; read + edit a file."""
    (work / "greeting.txt").write_text("Hello, world\n")
    cfg = _cfg({"provider": "openai_compat", "base_url": f"{base}/textonly/v1", "model": "demo",
                "native_tools": True, "context_window": 8192, "max_output_tokens": 512}, str(work / "sessions"))
    s = await build_session(cfg, work, use_mcp=False)
    try:
        res = await s.agent.run("DEMO-EDIT: change the greeting")
        validate_transcript(s.agent.messages)
        content = (work / "greeting.txt").read_text()
        fell_back = getattr(s.adapter, "fell_back", False)
        ok = content == "Goodbye, world\n" and fell_back and res.text == "Done: greeting.txt now says Goodbye."
        return ok, f"fell_back={fell_back} file={content!r} answer={res.text!r} iterations={res.iterations}"
    finally:
        await s.aclose()


INHOUSE_YAML = """\
# Onboarding a never-seen in-house model: this file is the ONLY thing that changes.
model:
  provider: custom_http
  context_window: 8192
  max_output_tokens: 512
  schema_level: simple
  http:
    endpoint: {endpoint}
    headers:
      X-Team-Key: ${{ENV:DEMO_TEAM_KEY:-demo-key}}
    input_mode: prompt
    prompt_format: chatml
    request_template: |
      {{"conversation": {{{{ prompt | tojson }}}},
       "knobs": {{"len": {{{{ max_tokens }}}}, "temp": {{{{ temperature }}}}, "halt": {{{{ stop | tojson }}}}}}}}
    response:
      text_path: "$.payload.candidates[0].output"
      finish_reason_path: "$.meta.why"
      usage_prompt_path: "$.meta.tok.in"
      usage_completion_path: "$.meta.tok.out"
context:
  session_dir: {sessions}
"""


async def scenario_inhouse(base: str, work: Path) -> tuple[bool, str]:
    """Scenario 3: in-house nonstandard JSON model configured only via agent.yaml."""
    (work / "alpha.txt").write_text("a")
    (work / "beta.py").write_text("b")
    yaml_path = work / "agent.yaml"
    yaml_path.write_text(INHOUSE_YAML.format(endpoint=f"{base}/inhouse/generate", sessions=work / "sessions"))
    cfg = load_config(yaml_path)
    s = await build_session(cfg, work, use_mcp=False)
    try:
        res = await s.agent.run("DEMO-INHOUSE: what files are in the project?")
        validate_transcript(s.agent.messages)
        ok = all(name in res.text for name in ("alpha.txt", "beta.py", "agent.yaml")) and res.tool_calls == 1
        return ok, f"answer={res.text!r} iterations={res.iterations}"
    finally:
        await s.aclose()


async def scenario_live(config: Path, work: Path) -> tuple[bool, str]:
    """Scenario 1 prompt against a real model (MCP add + local clock in parallel)."""
    cfg = load_config(config)
    cfg = cfg.model_copy(update={"context": cfg.context.model_copy(update={"session_dir": str(work / "sessions")}),
                                 "permissions": cfg.permissions.model_copy(update={
                                     "mode": "accept_edits", "allow": [*cfg.permissions.allow, "mcp__demo__*"]})})
    servers = {"demo": StdioServer(command=sys.executable, args=["-m", "agent.demo_mcp_server"])}
    s = await build_session(cfg, work, mcp_servers=servers, mcp_log_dir=str(work / "logs"))

    async def clock(args: dict[str, Any]) -> str:
        return "12:00"
    s.registry.register(FunctionTool("clock", "Return the current time.", {"type": "object", "properties": {}},
                                     clock, read_only=True))
    try:
        res = await s.agent.run("Use the tools: add 2 and 3 with mcp__demo__add, and get the time with clock. "
                                "Call both tools, then tell me both results.")
        validate_transcript(s.agent.messages)
        names = {c.name for m in s.agent.messages for c in m.tool_calls}
        ok = {"mcp__demo__add", "clock"} <= names and "5" in res.text
        return ok, f"answer={res.text!r} tools_called={sorted(names)}"
    finally:
        await s.aclose()


async def main_async(args: argparse.Namespace) -> int:
    """Run all scenarios and print a report."""
    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="agent-demo-") as tmp:
        root = Path(tmp)
        async with FakeServer() as srv:
            for name, fn in (("1 parallel MCP + local tool", scenario_parallel),
                             ("2 native-tools fallback + edit", scenario_fallback_edit),
                             ("3 in-house model via agent.yaml only", scenario_inhouse)):
                work = root / name.split()[0]
                work.mkdir()
                try:
                    ok, detail = await fn(srv.base, work)
                except Exception as e:
                    ok, detail = False, f"{type(e).__name__}: {e}"
                results.append((name, ok, detail))
        if args.live:
            if not args.config:
                print("--live requires --config", file=sys.stderr)
                return 2
            work = root / "live"
            work.mkdir()
            try:
                ok, detail = await scenario_live(Path(args.config), work)
            except Exception as e:
                ok, detail = False, f"{type(e).__name__}: {e}"
            results.append(("live model", ok, detail))
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] scenario {name}: {detail}")
    passed = all(ok for _, ok, _ in results)
    print("ALL SCENARIOS PASSED" if passed else "SOME SCENARIOS FAILED")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    p = argparse.ArgumentParser(prog="python -m agent.demo", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="agent.yaml of a live model (used with --live)")
    p.add_argument("--live", action="store_true", help="also run against the configured live model")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
