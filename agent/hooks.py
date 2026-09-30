"""Pre/post tool-use shell hooks (Phase 3).

A hook's ``matcher`` is a glob on the tool name. Hooks receive:
``AGENT_TOOL_NAME``, ``AGENT_TOOL_ARGS`` (JSON), ``AGENT_PROJECT_ROOT`` and, for post hooks,
``AGENT_TOOL_OUTPUT``. A pre hook exiting non-zero blocks the call; its stderr/stdout becomes
the reason shown to the model. Post hook failures are logged and reported, never fatal.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any

from agent.config import HookConfig, HooksConfig

log = logging.getLogger(__name__)


async def _run(hook: HookConfig, env: dict[str, str], cwd: Path) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_shell(hook.command, cwd=str(cwd), env={**os.environ, **env},
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                                                 start_new_session=True)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=hook.timeout_s)
    except TimeoutError:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await proc.wait()
        return 124, f"hook timed out after {hook.timeout_s:g}s: {hook.command}"
    return proc.returncode or 0, out.decode("utf-8", errors="replace").strip()


class Hooks:
    """Runs configured hooks around tool execution."""

    def __init__(self, cfg: HooksConfig, root: Path) -> None:
        self.cfg = cfg
        self.root = root

    def _env(self, tool: str, args: dict[str, Any], output: str | None = None) -> dict[str, str]:
        env = {"AGENT_TOOL_NAME": tool, "AGENT_TOOL_ARGS": json.dumps(args, ensure_ascii=False),
               "AGENT_PROJECT_ROOT": str(self.root)}
        if output is not None:
            env["AGENT_TOOL_OUTPUT"] = output[:100_000]
        return env

    async def pre(self, tool: str, args: dict[str, Any]) -> str | None:
        """Run matching pre hooks; return a block reason or None."""
        for h in self.cfg.pre_tool_use:
            if fnmatch.fnmatchcase(tool, h.matcher):
                rc, out = await _run(h, self._env(tool, args), self.root)
                if rc != 0:
                    return f"blocked by pre_tool_use hook '{h.command}' (exit {rc}): {out[:500]}"
        return None

    async def post(self, tool: str, args: dict[str, Any], output: str) -> list[str]:
        """Run matching post hooks; return messages from failing hooks."""
        notes: list[str] = []
        for h in self.cfg.post_tool_use:
            if fnmatch.fnmatchcase(tool, h.matcher):
                rc, out = await _run(h, self._env(tool, args, output), self.root)
                if rc != 0:
                    log.warning("post_tool_use hook %s failed (exit %d): %s", h.command, rc, out[:300])
                    notes.append(f"[post_tool_use hook '{h.command}' failed (exit {rc}): {out[:300]}]")
        return notes
