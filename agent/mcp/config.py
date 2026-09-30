"""``mcp.json`` loading: user (``~/.agent/mcp.json``) and project (``./.mcp.json``) scopes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.config import ConfigError, interpolate_env

USER_MCP = "~/.agent/mcp.json"
PROJECT_MCP = ".mcp.json"


class StdioServer(BaseModel):
    """A server launched as a subprocess speaking MCP over stdio."""

    model_config = ConfigDict(extra="ignore")
    type: Literal["stdio"] = "stdio"
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False


class HTTPServer(BaseModel):
    """A remote server over streamable HTTP (``type: http``) or legacy SSE (``type: sse``)."""

    model_config = ConfigDict(extra="ignore")
    type: Literal["http", "streamable-http", "sse"]
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    disabled: bool = False


ServerConfig = StdioServer | HTTPServer


def parse_servers(data: Any, source: str, env: dict[str, str] | None = None) -> dict[str, ServerConfig]:
    """Validate an ``{"mcpServers": {...}}`` document.

    Raises:
        ConfigError: naming the file and server key.
    """
    if not isinstance(data, dict) or not isinstance(data.get("mcpServers", {}), dict):
        raise ConfigError(f"{source}: expected an object with an 'mcpServers' object")
    out: dict[str, ServerConfig] = {}
    for name, raw in data.get("mcpServers", {}).items():
        raw = interpolate_env(raw, f"mcpServers.{name}", env)
        try:
            if isinstance(raw, dict) and raw.get("type") in ("http", "streamable-http", "sse"):
                out[name] = HTTPServer.model_validate(raw)
            elif isinstance(raw, dict) and "url" in raw and "command" not in raw:
                out[name] = HTTPServer.model_validate({**raw, "type": "http"})
            else:
                out[name] = StdioServer.model_validate(raw)
        except ValidationError as e:
            err = e.errors()[0]
            loc = ".".join(str(p) for p in err["loc"])
            raise ConfigError(f"{source}: mcpServers.{name}.{loc}: {err['msg']}") from None
    return out


def load_mcp_config(paths: list[str] | None = None, env: dict[str, str] | None = None,
                    cwd: str | None = None) -> dict[str, ServerConfig]:
    """Merge configs in order (later files override earlier ones by server name).

    Default order: user scope, then project scope. Missing files are skipped. Disabled servers
    are removed after merging.
    """
    if paths is None:
        base = Path(cwd) if cwd else Path.cwd()
        paths = [USER_MCP, str(base / PROJECT_MCP)]
    merged: dict[str, ServerConfig] = {}
    for p in paths:
        path = Path(p).expanduser()
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: invalid JSON: {e}") from None
        merged.update(parse_servers(data, str(path), env))
    return {k: v for k, v in merged.items() if not v.disabled}
