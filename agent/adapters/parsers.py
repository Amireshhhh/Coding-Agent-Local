"""Tool-call extraction chain and JSON repair (build plan 5.3, 5.4).

Pure functions, no I/O. Nothing here uses ``eval``/``exec``; Python-literal fallbacks use
``ast.literal_eval`` and pythonic calls are parsed with ``ast.parse`` only.
"""

from __future__ import annotations

import ast
import json
import re
import warnings
from collections.abc import Callable
from typing import Any

from agent.types import ToolCall, new_call_id

NAME_KEYS = ("name", "tool", "function", "tool_name")
ARG_KEYS = ("arguments", "args", "parameters", "input")
_FENCE_RE = re.compile(r"^\s*```[\w-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_SMART = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "«": '"',
                        "»": '"'})
_THINK_RE = re.compile(r"<think>.*?(</think>|\Z)", re.DOTALL)
_FABRICATED_RE = re.compile(r"<tool_response|^\s*Observation:", re.MULTILINE)

# --------------------------------------------------------------------------------------
# JSON repair
# --------------------------------------------------------------------------------------


def _loads(s: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(s)
    except (ValueError, RecursionError):
        return False, None


def _strip_fences(s: str) -> str:
    m = _FENCE_RE.match(s)
    return m.group(1) if m else s


def _remove_trailing_commas(s: str) -> str:
    """Remove commas directly before ``}``/``]`` (outside strings)."""
    out: list[str] = []
    in_str = esc = False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
            out.append(c)
        elif c == ",":
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                i += 1
                continue
            out.append(c)
        else:
            out.append(c)
        i += 1
    return "".join(out)


def _escape_control_in_strings(s: str) -> str:
    """Escape raw newlines / tabs / CRs that appear inside JSON strings."""
    out: list[str] = []
    in_str = esc = False
    for c in s:
        if in_str:
            if esc:
                esc = False
                out.append(c)
            elif c == "\\":
                esc = True
                out.append(c)
            elif c == '"':
                in_str = False
                out.append(c)
            elif c == "\n":
                out.append("\\n")
            elif c == "\t":
                out.append("\\t")
            elif c == "\r":
                out.append("\\r")
            else:
                out.append(c)
        else:
            if c == '"':
                in_str = True
            out.append(c)
    return "".join(out)


def _close_unbalanced(s: str) -> str:
    """Close an unterminated string and unbalanced ``{``/``[`` (truncated output)."""
    stack: list[str] = []
    in_str = esc = False
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if stack and stack[-1] == c:
                stack.pop()
            else:
                return s  # structurally broken, not merely truncated
    if not stack and not in_str:
        return s
    out = s
    if in_str:
        if esc:
            out = out[:-1]
        out += '"'
    out = out.rstrip()
    while out.endswith(","):
        out = out[:-1].rstrip()
    if out.endswith(":"):
        out += " null"
    elif stack and stack[-1] == "}":
        # a dangling key without colon: {"a": 1, "b"
        m = re.search(r'[{,]\s*"(?:[^"\\]|\\.)*"$', out)
        if m:
            out = out[: m.start() + 1].rstrip()
            if out.endswith(","):
                out = out[:-1]
    return out + "".join(reversed(stack))


def _py_to_json(v: Any) -> Any:
    """Convert a literal_eval result to JSON-compatible data; raise ValueError if impossible."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [_py_to_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _py_to_json(x) for k, x in v.items()}
    raise ValueError(f"non-JSON literal {type(v).__name__}")


def _literal(s: str) -> tuple[bool, Any]:
    if len(s) > 200_000:
        return False, None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return True, _py_to_json(ast.literal_eval(s.strip()))
    except Exception:  # SyntaxError, ValueError, RecursionError, MemoryError...
        return False, None


def repair_json_value(s: str) -> tuple[bool, Any]:
    """Parse ``s`` as JSON, applying the ordered repairs of 5.4. Returns ``(ok, value)``.

    Never raises.
    """
    try:
        if not isinstance(s, str):
            return False, None
        ok, v = _loads(s)
        if ok:
            return True, v
        cur = _strip_fences(s.strip())
        steps: list[Callable[[str], str]] = [
            lambda x: x.translate(_SMART),
            _remove_trailing_commas,
            lambda x: x.replace("'", '"') if '"' not in x else x,
            _escape_control_in_strings,
            _close_unbalanced,
            _remove_trailing_commas,
        ]
        ok, v = _loads(cur)
        if ok:
            return True, v
        for step in steps:
            cur = step(cur)
            ok, v = _loads(cur)
            if ok:
                return True, v
        for cand in (_strip_fences(s.strip()).translate(_SMART), cur):
            ok, v = _literal(cand)
            if ok:
                return True, v
        return False, None
    except Exception:
        return False, None


def repair_json(s: str) -> dict[str, Any] | None:
    """Repair and parse ``s``; return a dict or ``None``. Never raises."""
    ok, v = repair_json_value(s)
    return v if ok and isinstance(v, dict) else None


def normalize_arguments(args: Any) -> tuple[dict[str, Any], str | None]:
    """Normalize tool arguments (dict, JSON string, or missing) to ``(dict, parse_error)``."""
    if args is None:
        return {}, None
    if isinstance(args, dict):
        return args, None
    if isinstance(args, str):
        if not args.strip():
            return {}, None
        ok, v = repair_json_value(args)
        if not ok:
            return {}, f"arguments are not valid JSON: {args[:200]!r}"
        if isinstance(v, str):  # double-encoded JSON string
            ok, v = repair_json_value(v)
        if isinstance(v, dict):
            return v, None
        return {}, f"arguments must be a JSON object, got {type(v).__name__}"
    return {}, f"arguments must be a JSON object, got {type(args).__name__}"


# --------------------------------------------------------------------------------------
# Scanning helpers
# --------------------------------------------------------------------------------------


def scan_json_end(text: str, start: int) -> int | None:
    """Given ``text[start]`` in ``{[``, return the index after the matching close, or None."""
    stack: list[str] = []
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]":
            if not stack or stack.pop() != c:
                return None
            if not stack:
                return i + 1
    return None


def _scan_paren_end(text: str, start: int) -> int | None:
    """``text[start] == '('``; return index after the matching ``)`` (Python string aware)."""
    depth = 0
    quote: str | None = None
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


# --------------------------------------------------------------------------------------
# Candidate conversion
# --------------------------------------------------------------------------------------

Span = tuple[int, int, ToolCall]


def _resolve_name(name: str, known: set[str]) -> str | None:
    if name in known:
        return name
    low = {k.lower(): k for k in known}
    return low.get(name.lower())


def _get_name_args(obj: dict[str, Any]) -> tuple[str | None, bool, Any]:
    """Return ``(name, has_args_key, args)`` from a candidate object."""
    fn = obj.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        for k in ARG_KEYS:
            if k in fn:
                return (name if isinstance(name, str) else None), True, fn[k]
        return (name if isinstance(name, str) else None), False, None
    name = None
    for k in NAME_KEYS:
        v = obj.get(k)
        if isinstance(v, str) and v:
            name = v
            break
    for k in ARG_KEYS:
        if k in obj:
            return name, True, obj[k]
    return name, False, None


def candidate_to_call(obj: Any, known: set[str], raw: str, *, strong: bool = False) -> ToolCall | None:
    """Convert a parsed candidate into a ToolCall, or None if it is not a tool call.

    ``strong`` marks explicit tool-call syntax (e.g. inside ``<tool_call>``): unknown tools
    then always produce a ``parse_error`` call instead of being ignored.
    """
    if not isinstance(obj, dict):
        return None
    name, has_args, args = _get_name_args(obj)
    if name is None:
        if strong:
            return ToolCall(id=new_call_id(), name="unknown", arguments={}, raw=raw,
                            parse_error="tool call has no 'name'")
        return None
    cid = obj.get("id") if isinstance(obj.get("id"), str) and obj.get("id") else new_call_id()
    canonical = _resolve_name(name, known)
    if canonical is None:
        if has_args or strong:
            return ToolCall(id=cid, name=name, arguments={}, raw=raw, parse_error="unknown tool")
        return None
    arguments, err = normalize_arguments(args)
    return ToolCall(id=cid, name=canonical, arguments=arguments, raw=raw, parse_error=err)


def _calls_from_value(v: Any, known: set[str], raw: str, strong: bool) -> list[ToolCall]:
    items = v if isinstance(v, list) else [v]
    calls = [candidate_to_call(x, known, raw, strong=strong) for x in items]
    return [c for c in calls if c is not None]


def _broken_call(body: str) -> ToolCall:
    m = re.search(r'"(?:name|tool|function)"\s*:\s*"([^"]+)"', body)
    return ToolCall(id=new_call_id(), name=m.group(1) if m else "unknown", arguments={}, raw=body,
                    parse_error=f"invalid JSON in tool call: {body[:200]!r}")


# --------------------------------------------------------------------------------------
# Strategies: each returns [(start, end, call)] over the (think-masked) text
# --------------------------------------------------------------------------------------

_HERMES_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>)|\Z)", re.DOTALL)


def _hermes(text: str, known: set[str]) -> list[Span]:
    out: list[Span] = []
    for m in _HERMES_RE.finditer(text):
        body = _strip_fences(m.group(1).strip())
        if not body:
            continue
        ok, v = repair_json_value(body)
        calls = _calls_from_value(v, known, body, True) if ok else [_broken_call(body)]
        if not calls:
            calls = [_broken_call(body)]
        for c in calls:
            out.append((m.start(), m.end(), c))
    return out


_FENCE_BLOCK_RE = re.compile(r"```(json|tool_call|tool|tool_code|)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _json_fence(text: str, known: set[str]) -> list[Span]:
    out: list[Span] = []
    for m in _FENCE_BLOCK_RE.finditer(text):
        lang, body = m.group(1), m.group(2).strip()
        strong = lang in ("tool_call", "tool")
        ok, v = repair_json_value(body)
        if not ok:
            if strong:
                out.append((m.start(), m.end(), _broken_call(body)))
            continue
        for c in _calls_from_value(v, known, body, strong):
            out.append((m.start(), m.end(), c))
    return out


_INLINE_RE = re.compile(r'\{\s*"tool_calls"\s*:')


def _openai_inline(text: str, known: set[str]) -> list[Span]:
    out: list[Span] = []
    for m in _INLINE_RE.finditer(text):
        end = scan_json_end(text, m.start())
        chunk = text[m.start(): end] if end else text[m.start():]
        ok, v = repair_json_value(chunk)
        if not ok or not isinstance(v, dict) or not isinstance(v.get("tool_calls"), list):
            continue
        for c in _calls_from_value(v["tool_calls"], known, chunk, True):
            out.append((m.start(), end or len(text), c))
    return out


_PY_TAG = "<|python_tag|>"


def _llama(text: str, known: set[str]) -> list[Span]:
    out: list[Span] = []
    idx = text.find(_PY_TAG)
    if idx >= 0:
        pos = idx + len(_PY_TAG)
        found: list[ToolCall] = []
        last_end = pos
        while True:
            while pos < len(text) and text[pos] in " \t\r\n;":
                pos += 1
            if pos >= len(text) or text[pos] not in "{[":
                break
            end = scan_json_end(text, pos)
            chunk = text[pos: end] if end else text[pos:]
            ok, v = repair_json_value(chunk)
            calls = _calls_from_value(v, known, chunk, True) if ok else [_broken_call(chunk)]
            found.extend(calls)
            last_end = end or len(text)
            if end is None:
                break
            pos = end
        return [(idx, last_end, c) for c in found]
    stripped = text.strip()
    if stripped.startswith("{") and '"parameters"' in stripped:
        ok, v = repair_json_value(stripped)
        if ok and isinstance(v, dict) and "parameters" in v and "name" in v:
            c = candidate_to_call(v, known, stripped)
            if c:
                out.append((0, len(text), c))
    return out


_MISTRAL_RE = re.compile(r"\[TOOL_CALLS\]")
_MISTRAL_V11_RE = re.compile(r"([A-Za-z0-9_-]+)\s*(?:\[CALL_ID\]\s*([A-Za-z0-9]+)\s*)?\[ARGS\]")


def _mistral(text: str, known: set[str]) -> list[Span]:
    out: list[Span] = []
    for m in _MISTRAL_RE.finditer(text):
        pos = m.end()
        while pos < len(text) and text[pos] in " \t\r\n":
            pos += 1
        if pos < len(text) and text[pos] in "[{":
            end = scan_json_end(text, pos)
            chunk = text[pos: end] if end else text[pos:]
            ok, v = repair_json_value(chunk)
            calls = _calls_from_value(v, known, chunk, True) if ok else [_broken_call(chunk)]
            for c in calls:
                out.append((m.start(), end or len(text), c))
            continue
        v11 = _MISTRAL_V11_RE.match(text, pos)
        if v11:
            apos = v11.end()
            while apos < len(text) and text[apos] in " \t":
                apos += 1
            end = scan_json_end(text, apos) if apos < len(text) and text[apos] == "{" else None
            chunk = text[apos: end] if end else text[apos:]
            obj: dict[str, Any] = {"name": v11.group(1), "arguments": chunk}
            if v11.group(2):
                obj["id"] = v11.group(2)
            v11_call = candidate_to_call(obj, known, text[m.start(): end or len(text)], strong=True)
            if v11_call:
                out.append((m.start(), end or len(text), v11_call))
    return out


def _bare_json(text: str, known: set[str]) -> list[Span]:
    stripped = _strip_fences(text.strip())
    if not stripped or stripped[0] not in "[{":
        return []
    ok, v = repair_json_value(stripped)
    if not ok:
        return []
    return [(0, len(text), c) for c in _calls_from_value(v, known, stripped, False)]


_PY_LINE_RE = re.compile(r"^[ \t]*\[?[ \t]*([A-Za-z_][\w-]*)\(", re.MULTILINE)


def _eval_call_node(node: ast.expr, name_map: dict[str, str], raw: str) -> ToolCall | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    canonical = name_map.get(node.func.id.lower())
    if canonical is None:
        return None
    if node.args:
        return ToolCall(id=new_call_id(), name=canonical, arguments={}, raw=raw,
                        parse_error="positional arguments are not supported; use name=value")
    args: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            return ToolCall(id=new_call_id(), name=canonical, arguments={}, raw=raw,
                            parse_error="**kwargs is not supported")
        try:
            args[kw.arg] = _py_to_json(ast.literal_eval(kw.value))
        except Exception:
            return ToolCall(id=new_call_id(), name=canonical, arguments={}, raw=raw,
                            parse_error=f"argument {kw.arg!r} is not a literal value")
    return ToolCall(id=new_call_id(), name=canonical, arguments=args, raw=raw)


def _pythonic(text: str, known: set[str]) -> list[Span]:
    name_map = {k.replace("-", "_").lower(): k for k in known}
    out: list[Span] = []
    for m in _PY_LINE_RE.finditer(text):
        if m.group(1).replace("-", "_").lower() not in name_map:
            continue
        if out and m.start() < out[-1][1]:
            continue
        line_start = m.start()
        bracket = text.find("[", line_start, m.start(1))
        if bracket >= 0:
            # list form: [a(x=1), b(y=2)]
            close = _find_list_end(text, bracket)
            if close is None:
                continue
            src = text[bracket: close]
            span_end = close
        else:
            paren = m.end() - 1
            end = _scan_paren_end(text, paren)
            if end is None:
                continue
            src = text[m.start(1): end]
            span_end = end
        src_py = re.sub(r"\b([A-Za-z_][\w]*(?:-[\w]+)+)\(", lambda mm: mm.group(1).replace("-", "_") + "(", src)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree = ast.parse(src_py.strip(), mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        nodes = tree.body.elts if isinstance(tree.body, ast.List) else [tree.body]
        for node in nodes:
            c = _eval_call_node(node, name_map, src)
            if c:
                out.append((line_start, span_end, c))
    return out


def _find_list_end(text: str, start: int) -> int | None:
    """``text[start] == '['``; matching ``]`` index + 1 (Python string aware)."""
    depth = 0
    quote: str | None = None
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                quote = None
            continue
        if c in "\"'":
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


STRATEGIES: list[tuple[str, Callable[[str, set[str]], list[Span]]]] = [
    ("hermes", _hermes),
    ("json_fence", _json_fence),
    ("openai_inline", _openai_inline),
    ("llama_python_tag", _llama),
    ("mistral", _mistral),
    ("bare_json", _bare_json),
    ("pythonic", _pythonic),
]


def _mask_think(text: str) -> str:
    return _THINK_RE.sub(lambda m: " " * len(m.group(0)), text)


def extract_tool_calls_detailed(text: str, known_tools: set[str]) -> tuple[str, list[ToolCall], str | None]:
    """Like :func:`extract_tool_calls` but also returns the winning strategy name."""
    try:
        if not text:
            return text or "", [], None
        masked = _mask_think(text)
        for sname, strategy in STRATEGIES:
            spans = strategy(masked, known_tools)
            if not spans:
                continue
            first_end = min(e for _, e, _ in spans)
            cut = len(text)
            fab = _FABRICATED_RE.search(masked, first_end)
            if fab:
                cut = fab.start()
            kept = [(s, e, c) for s, e, c in spans if e <= cut]  # non-empty: cut >= first_end
            # dedupe calls sharing identical ids from one span
            seen: set[str] = set()
            calls: list[ToolCall] = []
            for _, _, c in kept:
                if c.id in seen:
                    c = c.model_copy(update={"id": new_call_id()})
                seen.add(c.id)
                calls.append(c)
            pieces: list[str] = []
            pos = 0
            for s, e in sorted({(s, e) for s, e, _ in kept}):
                if s >= pos:
                    pieces.append(text[pos:s])
                    pos = max(pos, e)
            pieces.append(text[pos:cut] if pos < cut else "")
            remaining = re.sub(r"\n{3,}", "\n\n", "".join(pieces)).strip()
            return remaining, calls, sname
        return text, [], None
    except Exception as e:  # defensive: parser must never raise
        return text, [ToolCall(id=new_call_id(), name="unknown", arguments={}, raw=text[:500],
                               parse_error=f"parser failure: {type(e).__name__}: {e}")], "error"


def extract_tool_calls(text: str, known_tools: set[str]) -> tuple[str, list[ToolCall]]:
    """Return ``(remaining_text_without_calls, calls)``.

    Tries strategies in order (hermes, json_fence, openai_inline, llama_python_tag, mistral,
    bare_json, pythonic); the first that yields at least one call wins. Text from the first
    fabricated tool response (``<tool_response`` or a line starting ``Observation:``) that
    follows a call is dropped, together with any calls after it. Never raises.
    """
    remaining, calls, _ = extract_tool_calls_detailed(text, known_tools)
    return remaining, calls


def has_tool_markers(text: str) -> bool:
    """True if ``text`` contains explicit tool-call syntax (used to trigger the repair loop)."""
    return any(tok in text for tok in ("<tool_call>", "```tool_call", "[TOOL_CALLS]", _PY_TAG,
                                        '"tool_calls"'))
