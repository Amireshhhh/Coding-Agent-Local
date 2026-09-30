"""File tools: read_file, edit_file, write_file, ls (Phase 2)."""

from __future__ import annotations

import asyncio
import difflib
from pathlib import Path
from typing import Any

from agent.tools.base import Tool
from agent.workspace import Workspace

MAX_READ_BYTES = 10 * 1024 * 1024
DEFAULT_LIMIT = 2000
MAX_LINE = 2000


def is_binary(data: bytes) -> bool:
    """Heuristic: a NUL byte in the first 8 KiB."""
    return b"\x00" in data[:8192]


def unified_diff(old: str, new: str, name: str) -> str:
    """Unified diff text for display."""
    return "".join(difflib.unified_diff(old.splitlines(keepends=True), new.splitlines(keepends=True),
                                        fromfile=f"a/{name}", tofile=f"b/{name}"))


class ReadFile(Tool):
    name = "read_file"
    description = ("Read a text file and return it with line numbers. Use offset/limit for large files. "
                   'Example: {"path": "src/main.py", "offset": 1, "limit": 200}')
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path, relative to the project root"},
        "offset": {"type": "integer", "minimum": 1, "description": "First line to read (1-based). Default 1"},
        "limit": {"type": "integer", "minimum": 1, "description": "Max number of lines. Default 2000"}},
        "required": ["path"], "additionalProperties": False}
    read_only = True

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    def _read(self, args: dict[str, Any]) -> str:
        p = self.ws.resolve(args["path"])
        if not p.exists():
            raise FileNotFoundError(f"{args['path']} does not exist")
        if p.is_dir():
            raise IsADirectoryError(f"{args['path']} is a directory; use ls")
        size = p.stat().st_size
        with p.open("rb") as f:
            head = f.read(8192)
        if is_binary(head):
            return f"[binary file: {self.ws.rel(p)}, {size} bytes; not shown]"
        offset = int(args.get("offset", 1))
        limit = int(args.get("limit", DEFAULT_LIMIT))
        if size > MAX_READ_BYTES and "limit" not in args:
            limit = min(limit, 500)
        out: list[str] = []
        total = 0
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                total = n
                if n < offset:
                    continue
                if len(out) >= limit:
                    continue
                line = line.rstrip("\n")
                if len(line) > MAX_LINE:
                    line = line[:MAX_LINE] + "...[line truncated]"
                out.append(f"{n:>6}\t{line}")
        self.ws.mark_read(p)
        if not out:
            return f"[{self.ws.rel(p)} has {total} lines; offset {offset} is past the end]" if total else \
                f"[{self.ws.rel(p)} is empty]"
        last = offset + len(out) - 1
        footer = f"\n[lines {offset}-{last} of {total}]" if (offset > 1 or last < total) else ""
        return "\n".join(out) + footer

    async def run(self, args: dict[str, Any]) -> str:
        """Read with line numbers."""
        return await asyncio.to_thread(self._read, args)


class EditFile(Tool):
    name = "edit_file"
    description = ("Replace an exact, unique string in a file. Read the file first. old_str must match exactly "
                   "(including indentation) and occur once, unless replace_all is true. "
                   'Example: {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"}')
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path"},
        "old_str": {"type": "string", "description": "Exact text to replace"},
        "new_str": {"type": "string", "description": "Replacement text"},
        "replace_all": {"type": "boolean", "description": "Replace every occurrence. Default false"}},
        "required": ["path", "old_str", "new_str"], "additionalProperties": False}

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.last_diff = ""

    def plan(self, args: dict[str, Any]) -> tuple[Path, str, str]:
        """Validate and compute ``(path, old_text, new_text)`` without writing."""
        p = self.ws.resolve(args["path"])
        if not p.is_file():
            raise FileNotFoundError(f"{args['path']} does not exist; use write_file to create it")
        if p not in self.ws.read_files:
            raise PermissionError(f"read {args['path']} with read_file before editing it")
        if p.stat().st_mtime != self.ws.read_files[p]:
            raise PermissionError(f"{args['path']} changed since it was read; read it again before editing")
        old_str, new_str = args["old_str"], args["new_str"]
        if old_str == new_str:
            raise ValueError("old_str and new_str are identical; nothing to change")
        if old_str == "":
            raise ValueError("old_str is empty; use write_file to create or overwrite a file")
        text = p.read_text(encoding="utf-8")
        n = text.count(old_str)
        if n == 0:
            raise ValueError("old_str not found in the file. Copy it exactly from read_file output "
                             "(without the line-number prefix)")
        if n > 1 and not args.get("replace_all", False):
            raise ValueError(f"old_str occurs {n} times; add surrounding lines to make it unique or set "
                             "replace_all: true")
        new = text.replace(old_str, new_str) if args.get("replace_all") else text.replace(old_str, new_str, 1)
        return p, text, new

    def _edit(self, args: dict[str, Any]) -> str:
        p, old, new = self.plan(args)
        p.write_text(new, encoding="utf-8")
        self.ws.mark_read(p)
        self.last_diff = unified_diff(old, new, self.ws.rel(p))
        count = old.count(args["old_str"]) if args.get("replace_all") else 1
        return f"Edited {self.ws.rel(p)} ({count} replacement{'s' if count != 1 else ''}).\n{self.last_diff}"

    async def run(self, args: dict[str, Any]) -> str:
        """Apply the edit."""
        return await asyncio.to_thread(self._edit, args)


class WriteFile(Tool):
    name = "write_file"
    description = ("Create or overwrite a file with the given content. To overwrite an existing file, read it "
                   'first. Example: {"path": "notes/todo.md", "content": "# TODO\\n"}')
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path"},
        "content": {"type": "string", "description": "Complete file content"}},
        "required": ["path", "content"], "additionalProperties": False}

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.last_diff = ""

    def plan(self, args: dict[str, Any]) -> tuple[Path, str]:
        """Validate and return ``(path, previous_content)``."""
        p = self.ws.resolve(args["path"])
        if p.is_dir():
            raise IsADirectoryError(f"{args['path']} is a directory")
        old = ""
        if p.exists():
            if p not in self.ws.read_files:
                raise PermissionError(f"{args['path']} exists; read it with read_file before overwriting")
            old = p.read_text(encoding="utf-8", errors="replace")
        return p, old

    def _write(self, args: dict[str, Any]) -> str:
        p, old = self.plan(args)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"], encoding="utf-8")
        self.ws.mark_read(p)
        self.last_diff = unified_diff(old, args["content"], self.ws.rel(p))
        lines = args["content"].count("\n") + (0 if args["content"].endswith("\n") or not args["content"] else 1)
        return f"Wrote {self.ws.rel(p)} ({lines} lines, {len(args['content'].encode())} bytes)."

    async def run(self, args: dict[str, Any]) -> str:
        """Write the file."""
        return await asyncio.to_thread(self._write, args)


class Ls(Tool):
    name = "ls"
    description = 'List a directory (directories end with /). Example: {"path": "src"}'
    parameters = {"type": "object", "properties": {
        "path": {"type": "string", "description": "Directory path. Default: project root"}},
        "additionalProperties": False}
    read_only = True

    def __init__(self, ws: Workspace) -> None:
        self.ws = ws

    def _ls(self, args: dict[str, Any]) -> str:
        p = self.ws.resolve(args.get("path", "."))
        if not p.is_dir():
            raise NotADirectoryError(f"{args.get('path', '.')} is not a directory")
        rows = []
        for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
            if child.name == ".git":
                continue
            if child.is_dir():
                rows.append(f"{child.name}/")
            else:
                try:
                    rows.append(f"{child.name}  ({child.stat().st_size} bytes)")
                except OSError:
                    rows.append(child.name)
        return "\n".join(rows) if rows else "[empty directory]"

    async def run(self, args: dict[str, Any]) -> str:
        """List entries."""
        return await asyncio.to_thread(self._ls, args)
