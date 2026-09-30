"""Builtin tools (Phase 2) and web tools (Phase 3)."""

from __future__ import annotations

from agent.config import AgentConfig
from agent.tools.base import Tool
from agent.tools.builtin.files import EditFile, Ls, ReadFile, WriteFile
from agent.tools.builtin.search import Glob, Grep, TodoWrite
from agent.tools.builtin.shell import Bash
from agent.tools.builtin.web import WebFetch, WebSearch
from agent.workspace import Workspace

__all__ = ["builtin_tools", "ReadFile", "EditFile", "WriteFile", "Ls", "Bash", "Grep", "Glob", "TodoWrite",
           "WebFetch", "WebSearch", "MUTATING_TOOLS", "EDIT_TOOLS"]

EDIT_TOOLS = {"edit_file", "write_file"}
MUTATING_TOOLS = EDIT_TOOLS | {"bash"}


def builtin_tools(ws: Workspace, cfg: AgentConfig | None = None, *, web: bool = True) -> list[Tool]:
    """All builtin tools bound to ``ws``. Web search is included only if a backend is configured."""
    tools: list[Tool] = [ReadFile(ws), EditFile(ws), WriteFile(ws), Bash(ws), Grep(ws), Glob(ws), Ls(ws),
                         TodoWrite(ws)]
    if web:
        tools.append(WebFetch())
        if cfg is not None and cfg.search.backend != "none":
            tools.append(WebSearch(cfg.search))
    return tools
