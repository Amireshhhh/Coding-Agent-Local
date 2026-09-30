"""Custom slash commands from markdown files (Phase 3).

``.agent/commands/<name>.md`` (project) and ``~/.agent/commands/<name>.md`` (user; project wins).
Optional frontmatter ``description:`` line. ``$ARGUMENTS`` is replaced by the text after the command.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
BUILTIN = {"help", "clear", "compact", "model", "mcp", "permissions", "resume", "doctor", "undo", "exit", "quit",
           "status", "todos"}


@dataclass
class CustomCommand:
    """A markdown-defined command."""

    name: str
    description: str
    body: str
    path: Path

    def expand(self, arguments: str) -> str:
        """Body with ``$ARGUMENTS`` substituted (appended if the placeholder is absent and args given)."""
        if "$ARGUMENTS" in self.body:
            return self.body.replace("$ARGUMENTS", arguments)
        return f"{self.body}\n\n{arguments}".strip() if arguments else self.body


def _parse(path: Path) -> CustomCommand:
    text = path.read_text(encoding="utf-8")
    desc = ""
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                if line.startswith("description:"):
                    desc = line.split(":", 1)[1].strip()
            text = text[end + 4:].lstrip("\n")
    return CustomCommand(path.stem, desc or text.strip().splitlines()[0][:80] if text.strip() else "", text.strip(),
                         path)


def load_commands(project_dir: Path, user_dir: Path | None = None) -> dict[str, CustomCommand]:
    """Load user then project commands; built-in names are never shadowed."""
    out: dict[str, CustomCommand] = {}
    for d in [user_dir or Path("~/.agent/commands").expanduser(), project_dir]:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            if _NAME_RE.match(f.stem) and f.stem not in BUILTIN:
                out[f.stem] = _parse(f)
    return out


def split_command(line: str) -> tuple[str, str] | None:
    """``"/name args"`` -> ``("name", "args")``; None if not a slash command."""
    if not line.startswith("/") or line.startswith("//"):
        return None
    head, _, rest = line[1:].partition(" ")
    return head.strip(), rest.strip()
