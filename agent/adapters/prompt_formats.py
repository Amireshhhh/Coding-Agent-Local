"""Render role messages into a single prompt string for completion-style backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2.sandbox import SandboxedEnvironment

STOP_TOKENS: dict[str, list[str]] = {
    "chatml": ["<|im_end|>"],
    "llama3": ["<|eot_id|>"],
    "mistral": ["</s>"],
    "alpaca": ["### Instruction:"],
    "jinja": [],
}


def _chatml(msgs: list[dict[str, Any]]) -> str:
    out = [f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs]
    return "".join(out) + "<|im_start|>assistant\n"


def _llama3(msgs: list[dict[str, Any]]) -> str:
    role_map = {"tool": "ipython"}
    out = ["<|begin_of_text|>"]
    for m in msgs:
        role = role_map.get(m["role"], m["role"])
        out.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{m.get('content') or ''}<|eot_id|>")
    out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(out)


def _mistral(msgs: list[dict[str, Any]]) -> str:
    system = "\n\n".join(m.get("content") or "" for m in msgs if m["role"] == "system")
    out = ["<s>"]
    first_user = True
    for m in msgs:
        if m["role"] == "system":
            continue
        content = m.get("content") or ""
        if m["role"] == "assistant":
            out.append(f" {content}</s>")
        else:
            if first_user and system:
                content = f"{system}\n\n{content}"
            first_user = False
            out.append(f"[INST] {content} [/INST]")
    if first_user and system:
        out.append(f"[INST] {system} [/INST]")
    return "".join(out)


def _alpaca(msgs: list[dict[str, Any]]) -> str:
    out: list[str] = []
    for m in msgs:
        c = m.get("content") or ""
        if m["role"] == "system":
            out.append(f"{c}\n\n")
        elif m["role"] == "assistant":
            out.append(f"### Response:\n{c}\n\n")
        else:
            out.append(f"### Instruction:\n{c}\n\n")
    out.append("### Response:\n")
    return "".join(out)


_RENDERERS = {"chatml": _chatml, "llama3": _llama3, "mistral": _mistral, "alpaca": _alpaca}
_ENV = SandboxedEnvironment(keep_trailing_newline=True)


def render_prompt(msgs: list[dict[str, Any]], fmt: str, template_file: str | None = None) -> str:
    """Render ``msgs`` (``{"role","content"}`` dicts) into one prompt ending with the assistant cue.

    Raises:
        ValueError: unknown format, or ``jinja`` without ``template_file``.
    """
    if fmt == "jinja":
        if not template_file:
            raise ValueError("prompt_format 'jinja' requires prompt_template_file")
        tpl = _ENV.from_string(Path(template_file).expanduser().read_text(encoding="utf-8"))
        return tpl.render(messages=msgs, add_generation_prompt=True)
    try:
        return _RENDERERS[fmt](msgs)
    except KeyError:
        raise ValueError(f"unknown prompt_format {fmt!r}; expected chatml|llama3|mistral|alpaca|jinja") from None
