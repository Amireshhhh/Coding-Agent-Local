"""ContextManager: keeps every request within the context budget (build plan 6.1-6.4)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.config import ContextConfig
from agent.context.compact import is_summary, summarize, summary_message
from agent.context.tokens import MessageCounter
from agent.context.truncate import truncate_output
from agent.errors import ContextOverflow
from agent.types import Message, ModelAdapter, ToolSpec, validate_transcript

log = logging.getLogger(__name__)

CLEARED_RE = re.compile(r"^\[tool result cleared: ")
MARKER_ALLOWANCE = 64  # tokens allowed for the truncation marker itself


def compute_budget(context_window: int, max_output_tokens: int, cfg: ContextConfig) -> int:
    """``context_window - max_output_tokens - max(5% of window, 512)``."""
    margin = max(math.ceil(context_window * cfg.safety_margin_pct), cfg.safety_margin_min)
    return context_window - max_output_tokens - margin


@dataclass
class Unit:
    """An atomic slice of the transcript: a single message, or an assistant tool-call group."""

    start: int
    end: int
    pinned: bool = False
    group: bool = False  # assistant message with tool_calls + its tool results


def build_units(msgs: list[Message]) -> list[Unit]:
    """Partition into atomic units and mark pinned ones.

    Pinned: system messages, the first user message, the most recent user message (summary
    messages excluded from both), and the final unit if it is an assistant tool-call group.
    """
    units: list[Unit] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        if m.role == "assistant" and m.tool_calls:
            j = i + 1
            while j < len(msgs) and msgs[j].role == "tool":
                j += 1
            units.append(Unit(i, j, group=True))
            i = j
            continue
        units.append(Unit(i, i + 1))
        i += 1
    real_users = [k for k, u in enumerate(units)
                  if msgs[u.start].role == "user" and not u.group and not is_summary(msgs[u.start])]
    for u in units:
        if msgs[u.start].role == "system":
            u.pinned = True
    if real_users:
        units[real_users[0]].pinned = True
        units[real_users[-1]].pinned = True
    if units and units[-1].group:
        units[-1].pinned = True
    return units


class SessionStore:
    """Append-only JSONL log of every message plus compaction events (6.4)."""

    def __init__(self, session_dir: str, session_id: str | None = None) -> None:
        self.dir = Path(session_dir).expanduser()
        self.session_id = session_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = self.dir / f"{self.session_id}.jsonl"

    def _append_sync(self, line: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    async def append(self, message: Message) -> None:
        """Append one canonical message."""
        await asyncio.to_thread(self._append_sync, message.model_dump_json())

    async def event(self, kind: str, **data: Any) -> None:
        """Append an event record (e.g. ``compaction``)."""
        await asyncio.to_thread(self._append_sync, json.dumps({"event": kind, "ts": time.time(), **data},
                                                              ensure_ascii=False))

    @staticmethod
    def load(session_dir: str, session_id: str) -> tuple[list[Message], list[dict[str, Any]]]:
        """Return ``(messages, events)`` of a stored session.

        Raises:
            FileNotFoundError: unknown session id.
            ValueError: corrupt line (message names the line number).
        """
        path = Path(session_dir).expanduser() / f"{session_id}.jsonl"
        msgs: list[Message] = []
        events: list[dict[str, Any]] = []
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                if "event" in d:
                    events.append(d)
                else:
                    msgs.append(Message.model_validate(d))
            except Exception as e:
                raise ValueError(f"{path}:{n}: corrupt session record: {e}") from e
        return msgs, events

    @staticmethod
    def list_sessions(session_dir: str) -> list[str]:
        """Session ids, newest first."""
        d = Path(session_dir).expanduser()
        if not d.is_dir():
            return []
        files = sorted(d.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in files]


class ContextManager:
    """Reduces a transcript to fit ``budget`` using the 4-stage pipeline of 6.3."""

    def __init__(self, adapter: ModelAdapter, counter: MessageCounter, cfg: ContextConfig, *,
                 context_window: int, max_output_tokens: int, store: SessionStore | None = None) -> None:
        self.adapter = adapter
        self.counter = counter
        self.cfg = cfg
        self.budget = compute_budget(context_window, max_output_tokens, cfg)
        if self.budget <= 0:
            raise ContextOverflow(f"context_window {context_window} leaves no room after max_output_tokens "
                                  f"{max_output_tokens} and the safety margin; raise context_window")
        self.store = store
        self.stats: dict[str, int] = {"truncated": 0, "cleared": 0, "compactions": 0, "dropped": 0}
        self.last_estimate: int | None = None

    async def total(self, msgs: list[Message], tools: list[ToolSpec] | None) -> int:
        """Estimated request size (messages + tool schemas / prompted block)."""
        return await self.adapter.count_tokens(msgs, tools)

    def observe_usage(self, prompt_tokens: int | None) -> None:
        """Feed provider-reported prompt tokens back into the fudge factor."""
        if prompt_tokens and self.last_estimate:
            self.counter.observe(self.last_estimate, prompt_tokens)

    # ---- stages --------------------------------------------------------------------------
    async def _truncate_outputs(self, msgs: list[Message]) -> list[Message]:
        out: list[Message] = []
        limit = self.cfg.max_tool_output_tokens
        for m in msgs:
            if m.role == "tool" and m.content:
                n = await self.counter.counter.count_text(m.content)
                if n > limit + MARKER_ALLOWANCE:
                    m = m.model_copy(update={"content": await truncate_output(m.content, limit,
                                                                              self.counter.counter)})
                    self.stats["truncated"] += 1
            out.append(m)
        return out

    async def _clear_stale(self, msgs: list[Message]) -> list[Message]:
        units = build_units(msgs)
        groups = [u for u in units if u.group]
        stale = groups[: max(0, len(groups) - self.cfg.keep_recent_tool_results)]
        out = list(msgs)
        for u in stale:
            if u.pinned:
                continue
            for k in range(u.start + 1, u.end):
                m = out[k]
                if m.role == "tool" and m.content and not CLEARED_RE.match(m.content):
                    n = await self.counter.counter.count_text(m.content)
                    out[k] = m.model_copy(update={"content": f"[tool result cleared: {m.name or 'tool'}, {n} tokens]"})
                    self.stats["cleared"] += 1
        return out

    async def _compact(self, msgs: list[Message], tools: list[ToolSpec] | None) -> list[Message]:
        units = build_units(msgs)
        free = [u for u in units if not u.pinned]
        if len(free) < 2:
            return msgs
        sizes = [sum([await self.counter.raw_message(m) for m in msgs[u.start:u.end]]) for u in free]
        target = 0.6 * sum(sizes)
        cap = int(self.budget * 0.8)
        chosen: list[Unit] = []
        acc = 0
        for u, s in zip(free, sizes, strict=True):
            if chosen and u.start != chosen[-1].end:
                break  # keep the range contiguous
            if chosen and (acc >= target or acc + s > cap):
                break
            chosen.append(u)
            acc += s
        if len(chosen) < 2 and acc < 256:
            return msgs
        start, end = chosen[0].start, chosen[-1].end
        try:
            text = await summarize(self.adapter, msgs[start:end])
        except Exception as e:
            log.warning("compaction failed (%s: %s); falling back to dropping old turns", type(e).__name__, e)
            return msgs
        self.stats["compactions"] += 1
        if self.store is not None:
            await self.store.event("compaction", removed=end - start, removed_tokens=acc, summary=text)
        return [*msgs[:start], summary_message(text), *msgs[end:]]

    async def _hard_drop(self, msgs: list[Message], tools: list[ToolSpec] | None) -> list[Message]:
        cur = list(msgs)
        while await self.total(cur, tools) > self.budget:
            units = build_units(cur)
            victim = next((u for u in units if not u.pinned), None)
            if victim is None:
                break
            del cur[victim.start:victim.end]
            self.stats["dropped"] += 1
        return cur

    async def _overflow(self, msgs: list[Message], tools: list[ToolSpec] | None, total: int) -> ContextOverflow:
        parts: list[tuple[int, str]] = []
        tools_tokens = await self.total([], tools) - await self.total([], None)
        if tools_tokens:
            parts.append((tools_tokens, f"tool schemas ({len(tools or [])} tools)"))
        for i, m in enumerate(msgs):
            label = "system prompt" if m.role == "system" else f"message #{i} ({m.role})"
            parts.append((await self.counter.message(m), label))
        parts.sort(reverse=True)
        top = ", ".join(f"{label}: {n} tokens" for n, label in parts[:3])
        return ContextOverflow(
            f"context overflow: request needs ~{total} tokens but the budget is {self.budget} "
            f"(context_window - max_output_tokens - safety margin). Largest contributors: {top}. "
            "Try: fewer tools (mcp.max_tools, fewer MCP servers), a shorter system prompt/AGENT.md, "
            "a larger context_window, or a smaller max_output_tokens.")

    # ---- public ----------------------------------------------------------------------------
    async def compact_now(self, messages: list[Message], tools: list[ToolSpec] | None) -> list[Message]:
        """Force stages 2+3 (``/compact``) regardless of the threshold."""
        validate_transcript(messages)
        msgs = await self._clear_stale(await self._truncate_outputs(messages))
        msgs = await self._compact(msgs, tools)
        validate_transcript(msgs)
        return msgs

    async def prepare(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                      allow_pending_tail: bool = False) -> list[Message]:
        """Return messages that fit the budget (pinned ones preserved) or raise ContextOverflow."""
        validate_transcript(messages, allow_pending_tail=allow_pending_tail)
        system_only = [m for m in messages if m.role == "system"]
        base = await self.total(system_only, tools)
        if base > self.budget:
            raise await self._overflow(system_only, tools, base)
        msgs = await self._truncate_outputs(messages)
        threshold = self.cfg.compact_threshold * self.budget
        total = await self.total(msgs, tools)
        if total > threshold:
            msgs = await self._clear_stale(msgs)
            total = await self.total(msgs, tools)
        if total > threshold:
            msgs = await self._compact(msgs, tools)
            total = await self.total(msgs, tools)
        if total > self.budget:
            msgs = await self._hard_drop(msgs, tools)
            total = await self.total(msgs, tools)
        if total > self.budget:
            raise await self._overflow(msgs, tools, total)
        validate_transcript(msgs, allow_pending_tail=allow_pending_tail)  # a violation here is a bug
        self.last_estimate = total
        return msgs
