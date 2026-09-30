"""Pluggable token counting (build plan 6.2)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import httpx

from agent.config import TokenizerConfig
from agent.errors import AdapterError
from agent.types import Message, ToolSpec

log = logging.getLogger(__name__)

FUDGE_MIN = 1.0
FUDGE_MAX = 1.5


class TokenCounter(ABC):
    """Counts tokens in a text. ``exact`` is True only for the model's own tokenizer."""

    exact: bool = False

    def __init__(self, cache_size: int = 4096) -> None:
        self._cache: OrderedDict[str, int] = OrderedDict()
        self._cache_size = cache_size

    @abstractmethod
    async def _count(self, text: str) -> int:
        """Count without caching."""

    async def count_text(self, text: str) -> int:
        """Count tokens in ``text`` (memoized)."""
        if not text:
            return 0
        hit = self._cache.get(text)
        if hit is not None:
            self._cache.move_to_end(text)
            return hit
        n = await self._count(text)
        self._cache[text] = n
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return n


class CharsDiv4Counter(TokenCounter):
    """``ceil(len(text) / 4)``."""

    async def _count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class TiktokenCounter(TokenCounter):
    """tiktoken encoding (approximate for non-OpenAI models)."""

    def __init__(self, encoding: str) -> None:
        super().__init__()
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding)

    async def _count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class HFCounter(TokenCounter):
    """HuggingFace ``AutoTokenizer`` (optional dependency ``transformers``)."""

    exact = True

    def __init__(self, path: str) -> None:
        super().__init__()
        from transformers import AutoTokenizer

        self._tok: Any = AutoTokenizer.from_pretrained(path)

    async def _count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))


class ServerCounter(TokenCounter):
    """POST ``{request_field: text}`` to a tokenize endpoint and read a JSONPath.

    The JSONPath may select an integer (the count) or a list (its length).
    """

    exact = True

    def __init__(self, url: str, request_field: str, count_path: str,
                 client: httpx.AsyncClient | None = None) -> None:
        super().__init__()
        from jsonpath_ng import parse as jp_parse

        self._url = url
        self._field = request_field
        self._path_str = count_path
        self._path = jp_parse(count_path)
        self._client = client

    async def _count(self, text: str) -> int:
        client = self._client or httpx.AsyncClient(timeout=30)
        try:
            r = await client.post(self._url, json={self._field: text})
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, json.JSONDecodeError) as e:
            raise AdapterError(f"tokenizer.url {self._url}: {type(e).__name__}: {e}") from e
        finally:
            if self._client is None:
                await client.aclose()
        found = [m.value for m in self._path.find(data)]
        if not found:
            raise AdapterError(f"tokenizer.count_path '{self._path_str}' matched nothing in: "
                               f"{json.dumps(data)[:300]}")
        v = found[0]
        if isinstance(v, list):
            return len(v)
        if isinstance(v, int):
            return v
        raise AdapterError(f"tokenizer.count_path '{self._path_str}' selected {type(v).__name__}, "
                           "expected int or list")


def build_counter(cfg: TokenizerConfig, client: httpx.AsyncClient | None = None) -> TokenCounter:
    """Instantiate the counter named by ``cfg.kind``.

    Raises:
        AdapterError: missing parameters or optional dependency unavailable.
    """
    try:
        if cfg.kind == "chars_div_4":
            return CharsDiv4Counter()
        if cfg.kind == "tiktoken":
            return TiktokenCounter(cfg.encoding)
        if cfg.kind == "hf":
            if not cfg.path:
                raise AdapterError("tokenizer.path is required for kind 'hf'")
            return HFCounter(cfg.path)
        if not cfg.url:
            raise AdapterError("tokenizer.url is required for kind 'server'")
        return ServerCounter(cfg.url, cfg.request_field, cfg.count_path, client)
    except ImportError as e:
        raise AdapterError(f"tokenizer.kind '{cfg.kind}' needs an optional dependency: {e}") from e
    except AdapterError:
        raise
    except Exception as e:  # e.g. tiktoken cannot download its encoding file
        raise AdapterError(f"tokenizer.kind '{cfg.kind}' could not be initialized: "
                           f"{type(e).__name__}: {e}") from e


def message_text(m: Message) -> str:
    """Text that represents a message for counting purposes."""
    parts = [m.role, m.content or ""]
    for c in m.tool_calls:
        parts.append(c.name)
        parts.append(json.dumps(c.arguments, ensure_ascii=False))
    if m.name:
        parts.append(m.name)
    return "\n".join(parts)


def tools_text(tools: list[ToolSpec] | None) -> str:
    """Text that represents tool specs for counting purposes."""
    if not tools:
        return ""
    return "\n".join(t.model_dump_json() for t in tools)


class MessageCounter:
    """Counts messages + tools with per-message overhead and an adaptive fudge factor."""

    def __init__(self, counter: TokenCounter, overhead: int = 4, fudge: float | None = None) -> None:
        self.counter = counter
        self.overhead = overhead
        self.fudge = fudge if fudge is not None else (1.0 if counter.exact else 1.1)
        self.fudge = min(FUDGE_MAX, max(FUDGE_MIN, self.fudge))

    async def raw_message(self, m: Message) -> int:
        """Unfudged tokens for one message including overhead."""
        return await self.counter.count_text(message_text(m)) + self.overhead

    async def message(self, m: Message) -> int:
        """Fudged tokens for one message."""
        return math.ceil(await self.raw_message(m) * self.fudge)

    async def text(self, text: str) -> int:
        """Fudged tokens for a raw text."""
        return math.ceil(await self.counter.count_text(text) * self.fudge)

    async def total(self, messages: list[Message], tools: list[ToolSpec] | None,
                    extra_text: str = "") -> int:
        """Fudged total for a request."""
        counts = await asyncio.gather(*(self.raw_message(m) for m in messages))
        raw = sum(counts)
        raw += await self.counter.count_text(tools_text(tools))
        raw += await self.counter.count_text(extra_text)
        return math.ceil(raw * self.fudge)

    def observe(self, estimate: int, actual: int) -> float:
        """Record estimate vs provider-reported prompt tokens; adjust fudge within [1.0, 1.5].

        Returns the new fudge value.
        """
        if estimate <= 0 or actual <= 0:
            return self.fudge
        drift = (actual - estimate) / estimate
        log.info("token estimate drift: estimate=%d actual=%d drift=%+.1f%%", estimate, actual, drift * 100)
        target = self.fudge * actual / estimate
        # move halfway toward target to damp noise, then clamp
        new = self.fudge + 0.5 * (target - self.fudge)
        self.fudge = min(FUDGE_MAX, max(FUDGE_MIN, new))
        return self.fudge
