"""Adapter registry: ``build_adapter(config) -> ModelAdapter`` (5.1, task 1.7)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable

import httpx

from agent.adapters.custom_http import CustomHTTPBackend, CustomHTTPNativeAdapter
from agent.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatTextBackend
from agent.adapters.prompted import GuidedDecoder, PromptedAdapter, TextBackend
from agent.config import ModelConfig
from agent.context.tokens import MessageCounter, build_counter
from agent.errors import NativeToolsUnsupported
from agent.types import Capabilities, Message, ModelAdapter, ModelResponse, StreamEvent, ToolSpec

log = logging.getLogger(__name__)

__all__ = ["build_adapter", "build_prompted", "AutoFallbackAdapter"]


def build_prompted(cfg: ModelConfig, backend: TextBackend, counter: MessageCounter) -> PromptedAdapter:
    """Wrap ``backend`` in a PromptedAdapter configured from ``cfg``."""
    guided = GuidedDecoder(cfg.guided_decoding) if cfg.guided_decoding != "off" else None
    return PromptedAdapter(backend, context_window=cfg.context_window, max_output_tokens=cfg.max_output_tokens,
                           counter=counter, tool_format=cfg.prompted_tool_format,
                           tool_template=cfg.prompted_tool_template, repair_attempts=cfg.repair_attempts,
                           guided=guided, streaming=cfg.streaming)


class AutoFallbackAdapter:
    """Delegates to a native adapter; on the first ``NativeToolsUnsupported`` it logs,
    rebuilds itself as a PromptedAdapter (once) and retries the call."""

    def __init__(self, native: ModelAdapter, make_prompted: Callable[[], ModelAdapter]) -> None:
        self.inner: ModelAdapter = native
        self._make_prompted = make_prompted
        self.fell_back = False

    @property
    def capabilities(self) -> Capabilities:
        """Capabilities of the current inner adapter."""
        return self.inner.capabilities

    @capabilities.setter
    def capabilities(self, value: Capabilities) -> None:
        self.inner.capabilities = value

    def _fallback(self, e: NativeToolsUnsupported) -> None:
        if self.fell_back:
            raise e
        log.warning("native tool calling unsupported (%s); switching to prompted tool calling. "
                    "Set native_tools: false in agent.yaml to skip this probe.", e)
        self.inner = self._make_prompted()
        self.fell_back = True

    async def complete(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                       temperature: float = 0.2, max_tokens: int | None = None,
                       stop: list[str] | None = None) -> ModelResponse:
        """Complete, falling back once if native tools are rejected."""
        try:
            return await self.inner.complete(messages, tools, temperature=temperature, max_tokens=max_tokens,
                                             stop=stop)
        except NativeToolsUnsupported as e:
            self._fallback(e)
            return await self.inner.complete(messages, tools, temperature=temperature, max_tokens=max_tokens,
                                             stop=stop)

    async def stream(self, messages: list[Message], tools: list[ToolSpec] | None, *,
                     temperature: float = 0.2, max_tokens: int | None = None,
                     stop: list[str] | None = None) -> AsyncIterator[StreamEvent]:
        """Stream, falling back once if native tools are rejected before any event."""
        try:
            async for ev in self.inner.stream(messages, tools, temperature=temperature, max_tokens=max_tokens,
                                              stop=stop):
                yield ev
            return
        except NativeToolsUnsupported as e:
            self._fallback(e)
        async for ev in self.inner.stream(messages, tools, temperature=temperature, max_tokens=max_tokens,
                                          stop=stop):
            yield ev

    async def count_tokens(self, messages: list[Message], tools: list[ToolSpec] | None) -> int:
        """Delegate."""
        return await self.inner.count_tokens(messages, tools)


def build_adapter(cfg: ModelConfig, client: httpx.AsyncClient | None = None,
                  counter: MessageCounter | None = None) -> ModelAdapter:
    """Build the adapter described by ``cfg`` (see 5.1 rules).

    * ``openai_compat`` + ``native_tools: true``  -> OpenAICompatAdapter with automatic prompted fallback
    * ``openai_compat`` + ``native_tools: false`` -> PromptedAdapter(OpenAICompatTextBackend)
    * ``prompted``                                -> PromptedAdapter over custom_http (if ``http``) or OpenAI text
    * ``custom_http``                             -> PromptedAdapter(CustomHTTPBackend) unless
      ``native_tools: true`` and ``http.tool_mapping`` is set -> CustomHTTPNativeAdapter
    """
    mc = counter or MessageCounter(build_counter(cfg.effective_tokenizer(), client))
    if cfg.provider == "openai_compat":
        if cfg.native_tools:
            native = OpenAICompatAdapter(cfg, client, mc)
            return AutoFallbackAdapter(native, lambda: build_prompted(cfg, OpenAICompatTextBackend(cfg, client), mc))
        return build_prompted(cfg, OpenAICompatTextBackend(cfg, client), mc)
    if cfg.provider == "prompted":
        backend: TextBackend = CustomHTTPBackend(cfg, client) if cfg.http else OpenAICompatTextBackend(cfg, client)
        return build_prompted(cfg, backend, mc)
    assert cfg.http is not None
    if cfg.native_tools and cfg.http.tool_mapping is not None:
        return CustomHTTPNativeAdapter(cfg, mc, client)
    return build_prompted(cfg, CustomHTTPBackend(cfg, client), mc)
