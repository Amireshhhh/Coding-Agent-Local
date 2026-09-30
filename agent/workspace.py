"""Per-session workspace state shared by builtin tools."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PathOutsideWorkspace(ValueError):
    """A path resolves outside the project root."""


@dataclass
class Workspace:
    """Project root, current directory, files read this session, todos, and last diff."""

    root: Path
    cwd: Path = field(default=Path())
    read_files: dict[Path, float] = field(default_factory=dict)  # path -> mtime when read
    todos: list[dict[str, Any]] = field(default_factory=list)
    sandbox: str = "none"  # none | bwrap | firejail
    sandbox_network: bool = False

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.cwd = self.root if self.cwd == Path() else Path(self.cwd).resolve()

    def resolve(self, p: str) -> Path:
        """Resolve ``p`` (relative to cwd) and ensure it stays inside the root (symlinks resolved).

        Raises:
            PathOutsideWorkspace: if the resolved path escapes the project root.
        """
        path = Path(os.path.expanduser(p))
        if not path.is_absolute():
            path = self.cwd / path
        resolved = path.resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise PathOutsideWorkspace(f"path '{p}' resolves to {resolved}, outside the project root {self.root}")
        return resolved

    def rel(self, p: Path) -> str:
        """Path relative to root for display."""
        try:
            return str(p.relative_to(self.root)) or "."
        except ValueError:
            return str(p)

    def mark_read(self, p: Path) -> None:
        """Record that ``p`` was read (with its current mtime)."""
        try:
            self.read_files[p] = p.stat().st_mtime
        except OSError:
            self.read_files[p] = 0.0
