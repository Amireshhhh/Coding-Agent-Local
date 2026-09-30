"""Stdio MCP server used by ``python -m agent.demo`` (tools: add, echo)."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

server = FastMCP("demo")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def echo(text: str) -> str:
    """Echo the text back."""
    return text


if __name__ == "__main__":
    server.run("stdio")
