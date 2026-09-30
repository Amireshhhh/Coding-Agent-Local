"""Shell tool: bash with timeout, cwd persistence, optional bubblewrap/firejail sandbox (Phase 2/4)."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from agent.workspace import Workspace

CWD_MARKER = "__AGENT_CWD__"
MAX_TIMEOUT = 600


def sandbox_prefix(ws: Workspace) -> list[str]:
    """Command prefix for the configured sandbox.

    Raises:
        RuntimeError: sandbox configured but the binary is not installed.
    """
    if ws.sandbox == "none":
        return []
    binary = shutil.which(ws.sandbox)
    if binary is None:
        raise RuntimeError(f"sandbox '{ws.sandbox}' is configured but not installed; install it or set "
                           "sandbox: none")
    root = str(ws.root)
    if ws.sandbox == "bwrap":
        # order matters: the fresh /tmp must be mounted BEFORE the project bind (the root may live under /tmp)
        pre = [binary, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
               "--bind", root, root, "--die-with-parent", "--chdir", str(ws.cwd)]
        if not ws.sandbox_network:
            pre.append("--unshare-net")
        return pre
    pre = [binary, "--quiet", f"--whitelist={root}"]
    if not ws.sandbox_network:
        pre.append("--net=none")
    return pre


class Bash(Tool):
    name = "bash"
    description = ("Run a shell command in the project directory and return its output and exit code. "
                   "The working directory persists between calls (cd works). "
                   'Example: {"command": "pytest -q tests/test_x.py", "timeout": 120}')
    parameters = {"type": "object", "properties": {
        "command": {"type": "string", "description": "The shell command"},
        "timeout": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT,
                    "description": "Seconds before the command is killed. Default 120"}},
        "required": ["command"], "additionalProperties": False}

    def __init__(self, ws: Workspace, default_timeout: int = 120) -> None:
        self.ws = ws
        self.default_timeout = default_timeout
        self.timeout_s = MAX_TIMEOUT + 10

    def describe_call(self, args: dict[str, Any]) -> str:
        """``bash(<command>)``."""
        return f"bash({args.get('command', '')})"

    async def run(self, args: dict[str, Any]) -> str:
        """Execute; kill the whole process group on timeout."""
        command: str = args["command"]
        timeout = min(int(args.get("timeout", self.default_timeout)), MAX_TIMEOUT)
        wrapped = f'{command}\n__agent_rc=$?\nprintf "\\n{CWD_MARKER}%s\\n" "$(pwd)"\nexit $__agent_rc'
        argv = [*sandbox_prefix(self.ws), "bash", "-c", wrapped]
        env = {**os.environ, "AGENT": "1", "PAGER": "cat", "GIT_PAGER": "cat"}
        proc = await asyncio.create_subprocess_exec(
            *argv, cwd=str(self.ws.cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL, start_new_session=True, env=env)
        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except (TimeoutError, asyncio.CancelledError) as e:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            if isinstance(e, asyncio.CancelledError):
                raise
            return f"ERROR: command timed out after {timeout}s and was killed: {command}"
        out = out_b.decode("utf-8", errors="replace")
        idx = out.rfind(f"\n{CWD_MARKER}")
        if idx >= 0:
            new_cwd = out[idx + len(CWD_MARKER) + 1:].strip()
            out = out[:idx]
            p = Path(new_cwd)
            if p.is_dir() and (p.resolve() == self.ws.root or self.ws.root in p.resolve().parents):
                self.ws.cwd = p.resolve()
        rc = proc.returncode
        body = out.rstrip("\n") or "(no output)"
        return f"{body}\n[exit code: {rc}]"
