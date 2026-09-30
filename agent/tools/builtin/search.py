"""Search tools: grep (ripgrep if available, else Python) and glob; plus todo_write."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from agent.tools.builtin.files import is_binary
from agent.workspace import Workspace

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


class Grep(Tool):
    name = "grep"
    description = ("Search file contents with a regular expression. Returns path:line:text. "
                   'Example: {"pattern": "def main", "path": "src", "glob": "*.py"}')
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Regular expression"},
        "path": {"type": "string", "description": "File or directory. Default: project root"},
        "glob": {"type": "string", "description": "Only files matching this glob, e.g. *.py"},
        "ignore_case": {"type": "boolean", "description": "Case-insensitive. Default false"},
        "max_results": {"type": "integer", "minimum": 1, "description": "Max matching lines. Default 200"}},
        "required": ["pattern"], "additionalProperties": False}
    read_only = True

    def __init__(self, ws: Workspace, use_rg: bool | None = None) -> None:
        self.ws = ws
        self.rg = shutil.which("rg") if use_rg is None or use_rg else None

    async def run(self, args: dict[str, Any]) -> str:
        """Search."""
        root = self.ws.resolve(args.get("path", "."))
        limit = int(args.get("max_results", 200))
        try:
            re.compile(args["pattern"])
        except re.error as e:
            raise ValueError(f"invalid regular expression: {e}") from e
        lines = await (self._rg(args, root, limit) if self.rg else asyncio.to_thread(self._py, args, root, limit))
        if not lines:
            return "No matches."
        more = f"\n[showing first {limit} matches]" if len(lines) >= limit else ""
        return "\n".join(lines[:limit]) + more

    async def _rg(self, args: dict[str, Any], root: Path, limit: int) -> list[str]:
        assert self.rg is not None
        cmd = [self.rg, "--line-number", "--no-heading", "--color", "never", "--max-columns", "400"]
        if args.get("ignore_case"):
            cmd.append("-i")
        if args.get("glob"):
            cmd += ["--glob", args["glob"]]
        cmd += ["-e", args["pattern"], str(root)]
        proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(self.ws.root), stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"ripgrep failed: {err.decode(errors='replace')[:300]}")
        res = []
        for line in out.decode(errors="replace").splitlines():
            res.append(self._relativize(line))
            if len(res) >= limit:
                break
        return res

    def _relativize(self, line: str) -> str:
        prefix = str(self.ws.root) + os.sep
        return line[len(prefix):] if line.startswith(prefix) else line

    def _py(self, args: dict[str, Any], root: Path, limit: int) -> list[str]:
        rx = re.compile(args["pattern"], re.IGNORECASE if args.get("ignore_case") else 0)
        glob = args.get("glob")
        files = [root] if root.is_file() else self._walk(root)
        res: list[str] = []
        for f in files:
            if glob and not f.match(glob):
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if is_binary(data):
                continue
            for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    res.append(f"{self.ws.rel(f)}:{n}:{line[:400]}")
                    if len(res) >= limit:
                        return res
        return res

    @staticmethod
    def _walk(root: Path) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            out += [Path(dirpath) / f for f in sorted(filenames)]
        return out


class Glob(Tool):
    name = "glob"
    description = 'Find files by glob pattern, newest first. Example: {"pattern": "**/*.py"}'
    parameters = {"type": "object", "properties": {
        "pattern": {"type": "string", "description": "Glob pattern, e.g. src/**/*.ts"},
        "path": {"type": "string", "description": "Directory to search. Default: project root"}},
        "required": ["pattern"], "additionalProperties": False}
    read_only = True

    def __init__(self, ws: Workspace, limit: int = 500) -> None:
        self.ws = ws
        self.limit = limit

    def _glob(self, args: dict[str, Any]) -> str:
        base = self.ws.resolve(args.get("path", "."))
        matches = [p for p in base.glob(args["pattern"])
                   if p.is_file() and not (set(p.relative_to(base).parts) & SKIP_DIRS)]
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            return "No files found."
        shown = [self.ws.rel(p) for p in matches[: self.limit]]
        more = f"\n[{len(matches) - self.limit} more not shown]" if len(matches) > self.limit else ""
        return "\n".join(shown) + more

    async def run(self, args: dict[str, Any]) -> str:
        """Glob."""
        return await asyncio.to_thread(self._glob, args)


class TodoWrite(Tool):
    name = "todo_write"
    description = ("Replace the task list for multi-step work. Keep exactly one item in_progress. "
                   'Example: {"todos": [{"content": "Run tests", "status": "in_progress"}]}')
    parameters = {"type": "object", "properties": {
        "todos": {"type": "array", "description": "The full, updated list", "items": {
            "type": "object", "properties": {
                "content": {"type": "string", "description": "What to do"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed"],
                           "description": "Item state"}},
            "required": ["content", "status"], "additionalProperties": False}}},
        "required": ["todos"], "additionalProperties": False}
    read_only = True

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    async def run(self, args: dict[str, Any]) -> str:
        """Store and render."""
        todos = args["todos"]
        in_progress = sum(1 for t in todos if t["status"] == "in_progress")
        self.ws.todos = [dict(t) for t in todos]
        mark = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        body = "\n".join(f"{mark[t['status']]} {t['content']}" for t in todos) or "(empty)"
        warn = "\nNote: keep exactly one item in_progress." if todos and in_progress != 1 and \
            any(t["status"] != "completed" for t in todos) else ""
        return f"Todo list updated:\n{body}{warn}"
