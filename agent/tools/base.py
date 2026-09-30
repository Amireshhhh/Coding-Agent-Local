"""Tool base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any


class Tool(ABC):
    """A tool the model can call.

    Attributes:
        name: registry name, ``^[a-zA-Z0-9_-]{1,64}$``.
        description: text shown to the model.
        parameters: JSON Schema (type=object) for arguments; the ORIGINAL schema.
        requires_approval: permission layer must ask before first use.
        read_only: no side effects; safe to run concurrently with other read-only tools.
        timeout_s: per-tool override of the registry timeout.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    requires_approval: bool = False
    read_only: bool = False
    timeout_s: float | None = None

    @abstractmethod
    async def run(self, args: dict[str, Any]) -> str:
        """Execute with validated arguments and return text output. May raise."""

    def describe_call(self, args: dict[str, Any]) -> str:
        """Short human-readable summary used by permissions and the UI."""
        return f"{self.name}({', '.join(f'{k}={v!r}'[:60] for k, v in args.items())})"


class FunctionTool(Tool):
    """Wrap an async function as a tool."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any],
                 fn: Callable[[dict[str, Any]], Awaitable[str]], *, read_only: bool = False,
                 requires_approval: bool = False, timeout_s: float | None = None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters
        self._fn = fn
        self.read_only = read_only
        self.requires_approval = requires_approval
        self.timeout_s = timeout_s

    async def run(self, args: dict[str, Any]) -> str:
        """Call the wrapped function."""
        return await self._fn(args)
