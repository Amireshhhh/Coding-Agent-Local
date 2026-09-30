"""Web tools (Phase 3): web_fetch and web_search with a pluggable backend."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlparse

import httpx
from jsonpath_ng import parse as jp_parse

from agent.config import SearchConfig
from agent.tools.base import Tool


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def html_to_text(doc: str) -> str:
    """Visible text of an HTML document, whitespace-collapsed."""
    p = _TextExtractor()
    p.feed(doc)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


class WebFetch(Tool):
    name = "web_fetch"
    description = ('Fetch a web page (http/https) and return its text. Example: {"url": "https://example.com"}')
    parameters = {"type": "object", "properties": {
        "url": {"type": "string", "description": "Absolute http(s) URL"},
        "max_chars": {"type": "integer", "minimum": 100, "description": "Truncate text. Default 20000"}},
        "required": ["url"], "additionalProperties": False}
    read_only = True
    requires_approval = True

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self.client = client

    async def run(self, args: dict[str, Any]) -> str:
        """GET the URL; HTML is converted to text."""
        url = args["url"]
        if urlparse(url).scheme not in ("http", "https"):
            raise ValueError("only http and https URLs are allowed")
        client = self.client or httpx.AsyncClient(timeout=30, follow_redirects=True)
        try:
            r = await client.get(url, headers={"User-Agent": "agent/0.1"})
        finally:
            if self.client is None:
                await client.aclose()
        ctype = r.headers.get("content-type", "")
        body = html_to_text(r.text) if "html" in ctype else r.text
        limit = int(args.get("max_chars", 20000))
        if len(body) > limit:
            body = body[:limit] + f"\n...[truncated {len(body) - limit} chars]"
        return f"[{r.status_code} {url}]\n{body}"


class WebSearch(Tool):
    name = "web_search"
    description = 'Search the web. Returns title, URL and snippet per result. Example: {"query": "httpx timeout"}'
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "Search query"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Default 5"}},
        "required": ["query"], "additionalProperties": False}
    read_only = True
    requires_approval = True

    def __init__(self, cfg: SearchConfig, client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self.client = client

    def _request(self, query: str) -> tuple[str, dict[str, str], dict[str, str], str, tuple[str, str, str]]:
        c = self.cfg
        if c.backend == "searxng":
            if not c.url:
                raise ValueError("search.url is required for backend 'searxng'")
            return (c.url.rstrip("/") + "/search", {"q": query, "format": "json"}, {}, "$.results[*]",
                    ("title", "url", "content"))
        if c.backend == "brave":
            if not c.api_key:
                raise ValueError("search.api_key is required for backend 'brave'")
            return ("https://api.search.brave.com/res/v1/web/search", {"q": query},
                    {"X-Subscription-Token": c.api_key, "Accept": "application/json"}, "$.web.results[*]",
                    ("title", "url", "description"))
        if c.backend == "custom":
            if not c.url:
                raise ValueError("search.url is required for backend 'custom'")
            headers = {"Authorization": f"Bearer {c.api_key}"} if c.api_key else {}
            return c.url, {"q": query}, headers, c.results_path, (c.title_key, c.url_key, c.snippet_key)
        raise ValueError("web search is not configured; set search.backend in agent.yaml")

    async def run(self, args: dict[str, Any]) -> str:
        """Query the backend and format results."""
        url, params, headers, path, (tk, uk, sk) = self._request(args["query"])
        client = self.client or httpx.AsyncClient(timeout=30)
        try:
            r = await client.get(url, params=params, headers=headers)
            r.raise_for_status()
            data = r.json()
        finally:
            if self.client is None:
                await client.aclose()
        items = [m.value for m in jp_parse(path).find(data)][: int(args.get("max_results", 5))]
        if not items:
            return "No results."
        lines = []
        for i, it in enumerate(items, 1):
            if isinstance(it, dict):
                lines.append(f"{i}. {it.get(tk, '')}\n   {it.get(uk, '')}\n   {str(it.get(sk, ''))[:300]}")
        return "\n".join(lines)
