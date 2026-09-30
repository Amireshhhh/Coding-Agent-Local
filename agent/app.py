"""Wire everything from config into a running agent session."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent.adapters import build_adapter
from agent.checkpoints import Checkpoints
from agent.config import AgentConfig
from agent.context.manager import ContextManager, SessionStore
from agent.context.tokens import MessageCounter, build_counter
from agent.hooks import Hooks
from agent.loop import Agent, EventFn, load_system_prompt
from agent.mcp.client import MCPManager
from agent.mcp.config import ServerConfig, load_mcp_config
from agent.permissions import Approver, Permissions
from agent.subagent import SUBAGENT_PROMPT, TaskTool
from agent.tools.builtin import builtin_tools
from agent.tools.registry import ToolRegistry
from agent.types import Message, ModelAdapter
from agent.workspace import Workspace

log = logging.getLogger(__name__)


@dataclass
class Session:
    """Everything a running agent needs; close with :meth:`aclose`."""

    cfg: AgentConfig
    agent: Agent
    adapter: ModelAdapter
    registry: ToolRegistry
    workspace: Workspace
    mcp: MCPManager | None
    store: SessionStore
    counter: MessageCounter
    client: httpx.AsyncClient | None = None

    async def aclose(self) -> None:
        """Shut down MCP servers and HTTP clients."""
        if self.mcp is not None:
            await self.mcp.shutdown()
        if self.client is not None:
            await self.client.aclose()


def default_max_tools(cfg: AgentConfig) -> int:
    """40 for ``simple`` schema level, 100 otherwise (unless configured)."""
    if cfg.mcp.max_tools is not None:
        return cfg.mcp.max_tools
    return 40 if cfg.model.schema_level == "simple" else 100


async def build_session(cfg: AgentConfig, root: Path, *, adapter: ModelAdapter | None = None,
                        mcp_servers: Mapping[str, ServerConfig] | None = None, approver: Approver | None = None,
                        on_event: EventFn | None = None, stream: bool = False, resume: str | None = None,
                        client: httpx.AsyncClient | None = None, use_mcp: bool = True,
                        mcp_log_dir: str | None = "~/.agent/logs") -> Session:
    """Build adapter, registry (builtins + task + MCP), context manager, permissions, hooks, loop."""
    ws = Workspace(root, sandbox=cfg.sandbox, sandbox_network=cfg.sandbox_network)
    counter = MessageCounter(build_counter(cfg.model.effective_tokenizer(), client),
                             overhead=cfg.context.message_overhead, fudge=cfg.context.count_fudge)
    adapter = adapter or build_adapter(cfg.model, client, counter)
    registry = ToolRegistry(timeout_s=cfg.tool_timeout_s, max_output_tokens=cfg.context.max_tool_output_tokens,
                            counter=counter.counter)
    for t in builtin_tools(ws, cfg):
        registry.register(t)
    permissions = Permissions(cfg.permissions, ws, approver)
    system = load_system_prompt(ws.root, cfg.model.family, cfg.context.project_instructions_file)

    async def run_child(description: str, tools: list[str] | None) -> str:
        child_reg = ToolRegistry(timeout_s=cfg.tool_timeout_s, max_output_tokens=cfg.context.max_tool_output_tokens,
                                 counter=counter.counter)
        for t in registry.tools():
            if t.name == "task":
                continue
            if (tools is None and t.read_only and not t.requires_approval) or (tools is not None and t.name in tools):
                child_reg.register(t)
        child_ctx = ContextManager(adapter, counter, cfg.context, context_window=cfg.model.context_window,
                                   max_output_tokens=cfg.model.max_output_tokens)
        child = Agent(adapter, child_reg, child_ctx, system_prompt=SUBAGENT_PROMPT, schema_level=cfg.model.schema_level,
                      max_iterations=cfg.subagent_max_iterations, permissions=permissions,
                      temperature=cfg.model.temperature)
        result = await child.run(description)
        return result.text

    registry.register(TaskTool(run_child))
    mcp: MCPManager | None = None
    if use_mcp:
        servers = (dict(mcp_servers) if mcp_servers is not None
                   else load_mcp_config(cfg.mcp.config_files, cwd=str(ws.root)))
        if servers:
            settings = cfg.mcp.model_copy(update={"max_tools": default_max_tools(cfg)})
            mcp = MCPManager(servers, settings, registry=registry, log_dir=mcp_log_dir)
            await mcp.start(builtin_count=len(registry.names()))
    store = SessionStore(cfg.context.session_dir, resume)
    messages: list[Message] | None = None
    if resume:
        messages, _ = SessionStore.load(cfg.context.session_dir, resume)
        if messages and messages[0].role == "system":
            messages[0] = Message(role="system", content=system)
        messages = close_pending_calls(messages)
    ctx = ContextManager(adapter, counter, cfg.context, context_window=cfg.model.context_window,
                         max_output_tokens=cfg.model.max_output_tokens, store=store)
    hooks = Hooks(cfg.hooks, ws.root) if (cfg.hooks.pre_tool_use or cfg.hooks.post_tool_use) else None
    checkpoints = Checkpoints(ws.root) if cfg.git_checkpoints else None
    agent = Agent(adapter, registry, ctx, system_prompt=system, schema_level=cfg.model.schema_level,
                  max_iterations=cfg.max_iterations, permissions=permissions, hooks=hooks, checkpoints=checkpoints,
                  store=store, on_event=on_event, stream=stream, temperature=cfg.model.temperature,
                  messages=messages)
    if not resume:
        await store.append(agent.messages[0])
    return Session(cfg, agent, adapter, registry, ws, mcp, store, counter, client)


def close_pending_calls(messages: list[Message]) -> list[Message]:
    """Answer tool calls left without results (e.g. the process died mid-turn) so the transcript is valid.

    Missing results are inserted directly after the existing results of their group.
    """
    out: list[Message] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        if m.role == "assistant" and m.tool_calls:
            answered: set[str | None] = set()
            while i < len(messages) and messages[i].role == "tool":
                answered.add(messages[i].tool_call_id)
                out.append(messages[i])
                i += 1
            for c in m.tool_calls:
                if c.id not in answered:
                    out.append(Message(role="tool", tool_call_id=c.id, name=c.name,
                                       content="ERROR: interrupted before this tool call completed"))
    return out


def describe_session(s: Session) -> dict[str, Any]:
    """Summary for status displays."""
    return {"session_id": s.store.session_id, "tools": len(s.registry.names()), "root": str(s.workspace.root),
            "adapter": type(s.adapter).__name__, "mode": s.agent.permissions.mode}
