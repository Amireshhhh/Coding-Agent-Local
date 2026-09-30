"""JSON Schema normalizer / simplifier for tool parameters (build plan 4.2).

Levels:
  * ``full``     - unchanged (deep copy).
  * ``standard`` - inline local ``$ref``/``$defs``; drop ``$schema``/``title``/``examples``
                   everywhere and ``default`` on non-required properties; ``anyOf``/``oneOf``
                   of ``[T, null]`` and ``type: [T, "null"]`` become ``T``; force
                   ``additionalProperties: false`` on objects that declare ``properties``;
                   cap nesting depth at 3.
  * ``simple``   - ``standard`` plus: collapse ``oneOf``/``anyOf``/``allOf`` to the first
                   branch; flatten nested objects to ``{"type": "object", "description"}``;
                   integer enums become string enums; descriptions truncated to 300 chars.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Literal

log = logging.getLogger(__name__)

SchemaLevel = Literal["full", "standard", "simple"]
MAX_DEPTH = 3
MAX_DESC = 300
_DROP_KEYS = ("$schema", "title", "examples", "$id", "$comment")
_COMBINATORS = ("oneOf", "anyOf", "allOf")


class _Ctx:
    def __init__(self, root: dict[str, Any], level: SchemaLevel) -> None:
        self.root = root
        self.level = level
        self.losses: list[str] = []

    def lose(self, path: str, what: str) -> None:
        self.losses.append(f"{path or '<root>'}: {what}")


def _resolve_ref(ref: str, ctx: _Ctx) -> Any:
    """Resolve a local JSON pointer (``#/...``). Returns None if unresolvable."""
    if not ref.startswith("#"):
        return None
    node: Any = ctx.root
    for part in ref.lstrip("#").strip("/").split("/"):
        if part == "":
            continue
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _is_null(s: Any) -> bool:
    return isinstance(s, dict) and s.get("type") == "null"


def _schema_type(s: dict[str, Any]) -> str:
    t = s.get("type")
    if isinstance(t, str):
        return t
    if "properties" in s:
        return "object"
    if "items" in s:
        return "array"
    return "string"


def _walk(s: Any, ctx: _Ctx, path: str, depth: int, seen: tuple[str, ...]) -> Any:
    """Normalize one schema node. ``depth`` counts enclosing object/array levels."""
    if not isinstance(s, dict):
        return s
    s = dict(s)

    # --- $ref inlining ------------------------------------------------------
    if "$ref" in s:
        ref = s.pop("$ref")
        if ref in seen:
            ctx.lose(path, f"recursive $ref {ref} replaced by generic object")
            target: Any = {"type": "object", "description": "recursive structure"}
        else:
            resolved = _resolve_ref(ref, ctx)
            if resolved is None:
                ctx.lose(path, f"unresolvable $ref {ref} replaced by generic object")
                target = {"type": "object"}
            else:
                target = _walk(copy.deepcopy(resolved), ctx, path, depth, (*seen, ref))
        merged = dict(target) if isinstance(target, dict) else {}
        merged.update({k: v for k, v in s.items()})  # sibling keywords override
        s = merged
    s.pop("$defs", None)
    s.pop("definitions", None)
    for k in _DROP_KEYS:
        s.pop(k, None)

    # --- nullable collapse --------------------------------------------------
    t = s.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        if len(non_null) < len(t):
            ctx.lose(path, "nullable dropped")
        if len(non_null) == 1:
            s["type"] = non_null[0]
        elif ctx.level == "simple" and non_null:
            ctx.lose(path, f"type union {non_null} collapsed to {non_null[0]}")
            s["type"] = non_null[0]
        else:
            s["type"] = non_null
    for comb in ("anyOf", "oneOf"):
        branches = s.get(comb)
        if isinstance(branches, list):
            non_null = [b for b in branches if not _is_null(b)]
            if len(non_null) < len(branches):
                ctx.lose(path, "nullable dropped")
            if len(non_null) == 1:
                s.pop(comb)
                inner = _walk(non_null[0], ctx, path, depth, seen)
                base = {k: v for k, v in s.items()}
                s = dict(inner) if isinstance(inner, dict) else {}
                s.update(base)
            else:
                s[comb] = non_null

    # --- simple: collapse combinators ---------------------------------------
    if ctx.level == "simple":
        for comb in _COMBINATORS:
            branches = s.get(comb)
            if isinstance(branches, list) and branches:
                s.pop(comb)
                if len(branches) > 1:
                    ctx.lose(path, f"{comb} collapsed to first branch")
                first = _walk(branches[0], ctx, path, depth, seen)
                base = s
                s = dict(first) if isinstance(first, dict) else {}
                s.update(base)  # the parent's keywords (e.g. its description) win
    else:
        for comb in _COMBINATORS:
            if isinstance(s.get(comb), list):
                s[comb] = [_walk(b, ctx, f"{path}/{comb}[{i}]", depth, seen) for i, b in enumerate(s[comb])]

    stype = _schema_type(s) if ("type" in s or "properties" in s or "items" in s) else None
    is_container = stype in ("object", "array")

    # --- depth cap / simple flattening -------------------------------------
    if is_container and depth > MAX_DEPTH:
        ctx.lose(path, f"nesting deeper than {MAX_DEPTH} replaced by generic {stype}")
        desc = s.get("description", "")
        return {"type": stype, "description": (desc + " (nested structure omitted)").strip()}
    if ctx.level == "simple" and stype == "object" and depth >= 1 and "properties" in s:
        keys = ", ".join(s["properties"].keys())
        ctx.lose(path, "nested object flattened")
        desc = s.get("description", "")
        return {"type": "object", "description": (f"{desc} JSON object with keys: {keys}").strip()[:MAX_DESC]}

    # --- recurse into children ----------------------------------------------
    if isinstance(s.get("properties"), dict):
        required = set(s.get("required", []) or [])
        props: dict[str, Any] = {}
        for name, sub in s["properties"].items():
            child = _walk(sub, ctx, f"{path}/{name}", depth + 1, seen)
            if isinstance(child, dict) and name not in required:
                child.pop("default", None)
            props[name] = child
        s["properties"] = props
    if isinstance(s.get("items"), dict):
        s["items"] = _walk(s["items"], ctx, f"{path}[]", depth + 1, seen)
    if isinstance(s.get("additionalProperties"), dict):
        s["additionalProperties"] = _walk(s["additionalProperties"], ctx, f"{path}{{*}}", depth + 1, seen)

    if stype == "object" and "properties" in s:
        if s.get("additionalProperties", True) is not False:
            if s.get("additionalProperties") not in (None, True):
                ctx.lose(path, "additionalProperties schema replaced by false")
            s["additionalProperties"] = False

    # --- simple: enums and descriptions -------------------------------------
    if ctx.level == "simple":
        enum = s.get("enum")
        if isinstance(enum, list) and enum and all(isinstance(x, int) and not isinstance(x, bool)
                                                    for x in enum):
            s["enum"] = [str(x) for x in enum]
            s["type"] = "string"
        d = s.get("description")
        if isinstance(d, str) and len(d) > MAX_DESC:
            ctx.lose(path, f"description truncated to {MAX_DESC} chars")
            s["description"] = d[: MAX_DESC - 3] + "..."
    return s


def normalize_schema_report(schema: dict[str, Any], level: str) -> tuple[dict[str, Any], list[str]]:
    """Normalize ``schema`` at ``level``; return ``(normalized, losses)``.

    ``losses`` lists every place where information was dropped. The input is never mutated.

    Raises:
        ValueError: unknown level.
    """
    if level not in ("full", "standard", "simple"):
        raise ValueError(f"unknown schema_level {level!r}; expected full|standard|simple")
    src = copy.deepcopy(schema) if isinstance(schema, dict) else {}
    if level == "full":
        if "type" not in src:
            src["type"] = "object"
        return src, []
    ctx = _Ctx(src, level)  # type: ignore[arg-type]
    out = _walk(src, ctx, "", 0, ())
    if not isinstance(out, dict):
        out = {}
    out.setdefault("type", "object")
    if out["type"] == "object":
        out.setdefault("properties", {})
        if "required" in out and not out["required"]:
            out.pop("required")
        out["additionalProperties"] = False
    return out, ctx.losses


def normalize_schema(schema: dict[str, Any], level: str, *, tool_name: str = "?") -> dict[str, Any]:
    """Normalize and log a warning if information was lost."""
    out, losses = normalize_schema_report(schema, level)
    if losses:
        log.warning("schema normalization for tool %s (level=%s) lost information: %s",
                    tool_name, level, "; ".join(losses))
    return out


def compact_schema(schema: dict[str, Any], limit: int = 600) -> str:
    """Compact one-line JSON rendering of a schema for error messages."""
    import json

    text = json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."
