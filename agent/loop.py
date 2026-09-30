"""The agent loop (Phase 2): model -> tools -> results -> repeat."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from agent.checkpoints import Checkpoints
from agent.context.manager import ContextManager, SessionStore
from agent.hooks import Hooks
from agent.permissions import Permissions
from agent.tools.base import Tool
from agent.tools.builtin.files import EditFile, WriteFile, unified_diff
from agent.tools.registry import ToolRegistry
from agent.types import Message, ModelAdapter, ModelResponse, ToolCall, Usage, validate_transcript

log = logging.getLogger(__name__)

EventFn = Callable[[str, dict[str, Any]], Awaitable[None] | None]
LOOP_REPEAT = 3
LOOP_REMINDER = ("You have called {name} with the same arguments {n} times in a row and got the same kind of "
                 "result. Stop repeating it: read the last result carefully, then try a different approach "
                 "or explain what is blocking you.")


def load_system_prompt(root: Path, family: str | None = None, instructions_file: str = "AGENT.md",
                       prompts_dir: Path | None = None) -> str:
    """``prompts/system.md`` (+ ``prompts/<family>.md`` override) + the project instructions file."""
    def read(name: str) -> str | None:
        if prompts_dir is not None:
            p = prompts_dir / name
            return p.read_text(encoding="utf-8") if p.is_file() else None
        try:
            return resources.files("agent").joinpath("prompts", name).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return None

    base = (family and read(f"{family}.md")) or read("system.md") or "You are a coding agent."
    text = base.replace("{project_root}", str(root))
    proj = root / instructions_file
    if proj.is_file():
        text += f"\n\n# Project instructions ({instructions_file})\n{proj.read_text(encoding='utf-8')}"
    return text


@dataclass
class TurnResult:
    """Outcome of one :meth:`Agent.run`."""

    text: str
    iterations: int
    stopped: str  # "final" | "max_iterations" | "length" | "error"
    tool_calls: int = 0
    parse_errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class Agent:
    """Runs the agent loop over a transcript."""

    def __init__(self, adapter: ModelAdapter, registry: ToolRegistry, context: ContextManager, *,
                 system_prompt: str, schema_level: str = "standard", max_iterations: int = 50,
                 permissions: Permissions | None = None, hooks: Hooks | None = None,
                 checkpoints: Checkpoints | None = None, store: SessionStore | None = None,
                 on_event: EventFn | None = None, stream: bool = False, temperature: float = 0.2,
                 messages: list[Message] | None = None) -> None:
        self.adapter = adapter
        self.registry = registry
        self.context = context
        self.schema_level = schema_level
        self.max_iterations = max_iterations
        self.permissions = permissions or Permissions()
        self.hooks = hooks
        self.checkpoints = checkpoints
        self.store = store
        self.on_event = on_event
        self.stream = stream
        self.temperature = temperature
        self.messages: list[Message] = messages or [Message(role="system", content=system_prompt)]
        self._checkpointed_turn = False

    # ------------------------------------------------------------------ helpers
    async def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event is not None:
            r = self.on_event(kind, data)
            if asyncio.iscoroutine(r):
                await r

    async def _append(self, m: Message) -> None:
        self.messages.append(m)
        if self.store is not None:
            await self.store.append(m)

    async def _call_model(self) -> ModelResponse:
        tools = self.registry.specs(self.schema_level)
        self.messages = await self.context.prepare(self.messages, tools)
        if self.stream and self.adapter.capabilities.streaming:
            final: ModelResponse | None = None
            async for ev in self.adapter.stream(self.messages, tools, temperature=self.temperature):
                if ev.type == "text_delta" and ev.text:
                    await self._emit("text_delta", text=ev.text)
                elif ev.type == "done":
                    final = ev.response
            if final is None:
                raise RuntimeError("model stream ended without a final response")
            resp = final
        else:
            resp = await self.adapter.complete(self.messages, tools, temperature=self.temperature)
        if resp.usage is not None and resp.usage.prompt_tokens is not None:
            self.context.observe_usage(resp.usage.prompt_tokens)
        else:  # provider reported no usage: fall back to our own estimates
            est_out = await self.context.counter.text((resp.message.content or "") + "".join(
                json.dumps(c.arguments) for c in resp.message.tool_calls))
            resp = resp.model_copy(update={"usage": Usage(prompt_tokens=self.context.last_estimate or 0,
                                                          completion_tokens=est_out)})
        return resp

    def _preview(self, tool: Tool, args: dict[str, Any]) -> str:
        """Diff shown in approval prompts for edit/write."""
        try:
            if isinstance(tool, EditFile):
                p, old, new = tool.plan(args)
                return unified_diff(old, new, tool.ws.rel(p))
            if isinstance(tool, WriteFile):
                p, old = tool.plan(args)
                return unified_diff(old, args.get("content", ""), tool.ws.rel(p))
        except Exception as e:
            return f"(no preview: {e})"
        return ""

    async def _execute_one(self, call: ToolCall) -> Message:
        tool = self.registry.get(call.name)
        if tool is not None and call.parse_error is None:
            args, err = self.registry.validate(call)
            if err is None and args is not None:
                ok, reason = await self.permissions.check(tool, args, self._preview(tool, args))
                if not ok:
                    return Message(role="tool", tool_call_id=call.id, name=call.name,
                                   content=f"ERROR: permission denied: {reason}")
                if self.hooks is not None:
                    blocked = await self.hooks.pre(call.name, args)
                    if blocked:
                        return Message(role="tool", tool_call_id=call.id, name=call.name, content=f"ERROR: {blocked}")
                if not tool.read_only and self.checkpoints is not None and not self._checkpointed_turn:
                    self._checkpointed_turn = True
                    try:
                        if await self.checkpoints.available():
                            ref = await self.checkpoints.create(f"before {call.name}")
                            await self._emit("notice", text=f"checkpoint {ref}")
                    except Exception as e:
                        await self._emit("notice", text=f"checkpoint failed: {e}")
                call = call.model_copy(update={"arguments": args})
        result = await self.registry.execute(call)
        if self.hooks is not None and tool is not None and not (result.content or "").startswith("ERROR:"):
            notes = await self.hooks.post(call.name, call.arguments, result.content or "")
            if notes:
                result = result.model_copy(update={"content": (result.content or "") + "\n" + "\n".join(notes)})
        return result

    async def _execute_all(self, calls: list[ToolCall]) -> list[Message]:
        """Consecutive read-only calls run concurrently; everything else sequentially, in order."""
        results: dict[str, Message] = {}
        i = 0
        while i < len(calls):
            tool = self.registry.get(calls[i].name)
            if tool is not None and tool.read_only:
                j = i
                while j < len(calls) and (t := self.registry.get(calls[j].name)) is not None and t.read_only:
                    j += 1
                batch = calls[i:j]
                for c in batch:
                    await self._emit("tool_call", call=c)
                outs = await asyncio.gather(*(self._execute_one(c) for c in batch))
                for c, m in zip(batch, outs, strict=True):
                    results[c.id] = m
                    await self._emit("tool_result", call=c, result=m)
                i = j
            else:
                await self._emit("tool_call", call=calls[i])
                m = await self._execute_one(calls[i])
                results[calls[i].id] = m
                await self._emit("tool_result", call=calls[i], result=m)
                i += 1
        return [results[c.id] for c in calls]

    def _loop_detected(self) -> tuple[str, int] | None:
        sigs: list[str] = []
        for m in reversed(self.messages):
            if m.role == "assistant":
                if not m.tool_calls:
                    break
                sigs.extend(f"{c.name}:{json.dumps(c.arguments, sort_keys=True)}" for c in reversed(m.tool_calls))
            if m.role == "user":
                break
        if len(sigs) >= LOOP_REPEAT and len(set(sigs[:LOOP_REPEAT])) == 1:
            n = 0
            for s in sigs:
                if s != sigs[0]:
                    break
                n += 1
            return sigs[0].split(":", 1)[0], n
        return None

    # ------------------------------------------------------------------ public
    async def run(self, user_input: str) -> TurnResult:
        """Process one user message until the model gives a final answer or a stop condition hits."""
        await self._append(Message(role="user", content=user_input))
        self._checkpointed_turn = False
        res = TurnResult(text="", iterations=0, stopped="final")
        reminded: set[str] = set()
        for it in range(1, self.max_iterations + 1):
            res.iterations = it
            resp = await self._call_model()
            if resp.usage:
                res.prompt_tokens += resp.usage.prompt_tokens or 0
                res.completion_tokens += resp.usage.completion_tokens or 0
            msg = resp.message
            await self._append(msg)
            await self._emit("assistant", message=msg, finish=resp.finish_reason)
            if not msg.tool_calls:
                res.text = msg.content or ""
                res.stopped = "length" if resp.finish_reason == "length" else "final"
                return res
            res.tool_calls += len(msg.tool_calls)
            res.parse_errors += sum(1 for c in msg.tool_calls if c.parse_error)
            try:
                results = await self._execute_all(msg.tool_calls)
            except asyncio.CancelledError:
                done = {m.tool_call_id for m in self.messages if m.role == "tool"}
                for c in msg.tool_calls:
                    if c.id not in done:
                        await self._append(Message(role="tool", tool_call_id=c.id, name=c.name,
                                                   content="ERROR: cancelled by the user"))
                raise
            for m in results:
                await self._append(m)
            loop = self._loop_detected()
            if loop is not None and f"{loop[0]}:{loop[1]}" not in reminded:
                reminded.add(f"{loop[0]}:{loop[1]}")
                await self._append(Message(role="user", content=LOOP_REMINDER.format(name=loop[0], n=loop[1])))
                await self._emit("notice", text=f"loop detected: {loop[0]} x{loop[1]}")
        res.stopped = "max_iterations"
        res.text = f"Stopped after {self.max_iterations} iterations without a final answer (max_iterations)."
        validate_transcript(self.messages)
        return res
