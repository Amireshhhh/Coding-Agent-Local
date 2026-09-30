"""Tool registry: registration, normalized specs, validation, coercion, safe execution (4.3)."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError, best_match

from agent.context.tokens import CharsDiv4Counter, TokenCounter
from agent.context.truncate import truncate_output
from agent.tools.base import Tool
from agent.tools.schema import compact_schema, normalize_schema
from agent.types import TOOL_NAME_PATTERN, Message, ToolCall, ToolSpec

log = logging.getLogger(__name__)
_NAME_RE = re.compile(TOOL_NAME_PATTERN)
_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _types(schema: dict[str, Any]) -> list[str]:
    t = schema.get("type")
    if isinstance(t, str):
        return [t]
    if isinstance(t, list):
        return [x for x in t if isinstance(x, str)]
    return []


def coerce(value: Any, schema: Any) -> Any:
    """Coerce safe type mismatches: numeric strings, "true"/"false", JSON-string arrays/objects.

    Recurses into object properties and array items. Never raises; returns the value
    unchanged when no safe coercion applies.
    """
    if not isinstance(schema, dict):
        return value
    types = _types(schema)
    if not types:
        for comb in ("anyOf", "oneOf"):
            for branch in schema.get(comb, []) or []:
                if isinstance(branch, dict) and branch.get("type") not in (None, "null"):
                    return coerce(value, branch)
        return value
    if isinstance(value, str):
        if "string" in types:
            return value
        s = value.strip()
        if "integer" in types and _INT_RE.match(s):
            return int(s)
        if "number" in types and _NUM_RE.match(s):
            f = float(s)
            return int(f) if f.is_integer() and "integer" in types else f
        if "boolean" in types and s.lower() in ("true", "false"):
            return s.lower() == "true"
        if ("array" in types or "object" in types) and s[:1] in "[{":
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return value
            if ("array" in types and isinstance(parsed, list)) or (
                    "object" in types and isinstance(parsed, dict)):
                return coerce(parsed, schema)
        if "null" in types and s.lower() in ("null", "none"):
            return None
        return value
    if "integer" in types and isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        props = schema["properties"]
        return {k: coerce(v, props.get(k)) for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [coerce(v, schema["items"]) for v in value]
    return value


class _Entry:
    def __init__(self, tool: Tool) -> None:
        self.tool = tool
        self.validator: Draft202012Validator | None = None
        self.fallback: Draft202012Validator | None = None
        try:
            Draft202012Validator.check_schema(tool.parameters)
            self.validator = Draft202012Validator(tool.parameters)
        except SchemaError as e:
            log.warning("tool %s: original schema invalid (%s); validating against normalized",
                        tool.name, e.message)
        self.fallback = Draft202012Validator(normalize_schema(tool.parameters, "standard",
                                                              tool_name=tool.name))

    def errors(self, args: dict[str, Any]) -> list[ValidationError]:
        """Validation errors; falls back to the normalized schema if the original can't be used."""
        if self.validator is not None:
            try:
                return list(self.validator.iter_errors(args))
            except Exception as e:  # unresolvable remote $ref etc.
                log.warning("tool %s: original schema unusable at validation (%s); using normalized",
                            self.tool.name, e)
                self.validator = None
        assert self.fallback is not None
        return list(self.fallback.iter_errors(args))

    def schema_for_errors(self) -> dict[str, Any]:
        return self.tool.parameters if self.validator is not None else dict(self.fallback.schema)  # type: ignore[union-attr]


class ToolRegistry:
    """Holds tools, produces model-facing specs, and executes calls without ever raising."""

    def __init__(self, *, timeout_s: float = 120.0, max_output_tokens: int = 4000,
                 counter: TokenCounter | None = None) -> None:
        self._tools: dict[str, _Entry] = {}
        self._spec_cache: dict[str, list[ToolSpec]] = {}
        self.timeout_s = timeout_s
        self.max_output_tokens = max_output_tokens
        self.counter = counter or CharsDiv4Counter()

    # -- registration -------------------------------------------------------
    def register(self, tool: Tool) -> None:
        """Add a tool.

        Raises:
            ValueError: duplicate or invalid name.
        """
        if not _NAME_RE.match(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}; must match {TOOL_NAME_PATTERN}")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = _Entry(tool)
        self._spec_cache.clear()

    def unregister(self, name: str) -> bool:
        """Remove a tool; returns True if it existed."""
        existed = self._tools.pop(name, None) is not None
        self._spec_cache.clear()
        return existed

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name."""
        e = self._tools.get(name)
        return e.tool if e else None

    def names(self) -> list[str]:
        """Registered names in registration order."""
        return list(self._tools)

    def tools(self) -> list[Tool]:
        """Registered tools in registration order."""
        return [e.tool for e in self._tools.values()]

    def specs(self, level: str) -> list[ToolSpec]:
        """Normalized specs for the model at ``level`` (cached)."""
        if level not in self._spec_cache:
            self._spec_cache[level] = [
                ToolSpec(name=e.tool.name, description=e.tool.description,
                         parameters=normalize_schema(e.tool.parameters, level, tool_name=e.tool.name))
                for e in self._tools.values()
            ]
        return list(self._spec_cache[level])

    # -- execution ----------------------------------------------------------
    def _msg(self, call: ToolCall, content: str) -> Message:
        return Message(role="tool", content=content, tool_call_id=call.id, name=call.name)

    def validate(self, call: ToolCall) -> tuple[dict[str, Any] | None, str | None]:
        """Return ``(coerced_args, None)`` or ``(None, error_text)``. Never raises."""
        entry = self._tools.get(call.name)
        if entry is None:
            avail = ", ".join(self._tools) or "(none)"
            close = difflib.get_close_matches(call.name, list(self._tools), n=1, cutoff=0.5)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            return None, f"ERROR: unknown tool '{call.name}'. Available tools: {avail}.{hint}"
        if call.parse_error:
            return None, (f"ERROR: could not parse arguments: {call.parse_error}. "
                          "Re-send the call with valid JSON matching the schema.")
        args: Any = call.arguments
        errs = entry.errors(args)
        if errs:
            args = coerce(args, entry.schema_for_errors())
            errs = entry.errors(args) if isinstance(args, dict) else errs
        if errs:
            err = best_match(errs)
            path = err.json_path if hasattr(err, "json_path") else "$"
            return None, (f"ERROR: invalid arguments: {path}: {err.message}. "
                          f"Expected schema: {compact_schema(entry.schema_for_errors())}")
        return args, None

    async def execute(self, call: ToolCall) -> Message:
        """Validate and run ``call``; always returns a ``role="tool"`` Message."""
        try:
            args, error = self.validate(call)
            if error is not None or args is None:
                return self._msg(call, error or "ERROR: invalid arguments")
            tool = self._tools[call.name].tool
            timeout = tool.timeout_s or self.timeout_s
            try:
                out = await asyncio.wait_for(tool.run(args), timeout=timeout)
            except TimeoutError:
                return self._msg(call, f"ERROR: TimeoutError: tool '{call.name}' exceeded {timeout:g}s")
            if not isinstance(out, str):
                out = json.dumps(out, ensure_ascii=False, default=str)
            out = await truncate_output(out, self.max_output_tokens, self.counter)
            return self._msg(call, out)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return self._msg(call, f"ERROR: {type(e).__name__}: {e}")
