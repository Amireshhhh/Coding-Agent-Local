"""Structured JSON logging with secret redaction (Phase 4 observability)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from agent.config import AgentConfig
from agent.trace import redact_text

SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "AUTH")


def collect_secrets(cfg: AgentConfig | None) -> list[str]:
    """Values that must never appear in logs: api keys, header values, secret-looking env vars."""
    out: list[str] = []
    if cfg is not None:
        if cfg.model.api_key and cfg.model.api_key not in ("none", ""):
            out.append(cfg.model.api_key)
        if cfg.model.http is not None:
            out += [v.split(" ", 1)[-1] for v in cfg.model.http.headers.values()]
        if cfg.search.api_key:
            out.append(cfg.search.api_key)
    out += [v for k, v in os.environ.items() if any(h in k.upper() for h in SECRET_ENV_HINTS) and len(v) >= 6]
    return sorted({s for s in out if len(s) >= 4}, key=len, reverse=True)


class JSONFormatter(logging.Formatter):
    """One JSON object per line; message text redacted."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = secrets

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a record."""
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) +
                           f".{int(record.msecs):03d}Z", "level": record.levelname, "logger": record.name,
                           "msg": redact_text(msg, self.secrets)}, ensure_ascii=False)


def setup_logging(log_file: str | None, cfg: AgentConfig | None, *, verbose: bool = False) -> None:
    """Send agent logs to ``log_file`` as JSON lines (or WARNING+ to stderr if no file)."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    secrets = collect_secrets(cfg)
    if log_file:
        p = Path(log_file).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(JSONFormatter(secrets))
        root.addHandler(fh)
        root.setLevel(logging.DEBUG if verbose else logging.INFO)
    else:
        sh = logging.StreamHandler()
        sh.setFormatter(JSONFormatter(secrets))
        sh.setLevel(logging.WARNING)
        root.addHandler(sh)
        root.setLevel(logging.WARNING)
    for noisy in ("httpx", "httpcore", "mcp", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
