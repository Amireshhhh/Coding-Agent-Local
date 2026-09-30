"""HTTP trace recording for ``agent doctor`` and ``--trace-llm`` (secrets redacted)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx

SECRET_HEADERS = {"authorization", "x-api-key", "api-key", "proxy-authorization", "cookie"}
_SECRET_VALUE_RE = re.compile(r"(Bearer\s+)[^\s\"']+", re.IGNORECASE)


def redact_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    """Copy of headers with secret values replaced by ``***``."""
    return {k: ("***" if k.lower() in SECRET_HEADERS else v) for k, v in dict(headers).items()}


def redact_text(text: str, secrets: list[str] | None = None) -> str:
    """Mask bearer tokens and any explicitly listed secret values."""
    out = _SECRET_VALUE_RE.sub(r"\1***", text)
    for s in secrets or []:
        if s and len(s) >= 4:
            out = out.replace(s, "***")
    return out


class TraceRecorder:
    """httpx event hooks capturing request/response pairs (non-streaming bodies only)."""

    def __init__(self, dump_path: str | None = None, secrets: list[str] | None = None) -> None:
        self.entries: list[dict[str, Any]] = []
        self.dump_path = Path(dump_path).expanduser() if dump_path else None
        self.secrets = secrets or []

    def _body(self, raw: bytes) -> Any:
        text = raw.decode("utf-8", "replace")
        try:
            return json.loads(text)
        except ValueError:
            return redact_text(text[:20000], self.secrets)

    async def on_request(self, request: httpx.Request) -> None:
        """Record the outgoing request."""
        self.entries.append({"method": request.method, "url": str(request.url),
                             "headers": redact_headers(request.headers), "request": self._body(request.content)})

    async def on_response(self, response: httpx.Response) -> None:
        """Record the response (reads the body unless it is a stream)."""
        entry = self.entries[-1] if self.entries else {}
        entry["status"] = response.status_code
        ctype = response.headers.get("content-type", "")
        if "text/event-stream" in ctype or "ndjson" in ctype:
            entry["response"] = "<streamed>"
        else:
            await response.aread()
            entry["response"] = self._body(response.content)
        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            with self.dump_path.open("a", encoding="utf-8") as f:
                f.write(redact_text(json.dumps(entry, ensure_ascii=False), self.secrets) + "\n")

    def client(self, timeout: float = 300) -> httpx.AsyncClient:
        """An AsyncClient wired to this recorder."""
        return httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=30),
                                 event_hooks={"request": [self.on_request], "response": [self.on_response]})
