"""Typed exceptions shared across the agent."""


class AgentError(Exception):
    """Base class for all agent errors."""


class AdapterError(AgentError):
    """A model backend call failed (HTTP, mapping, or protocol error)."""


class NativeToolsUnsupported(AdapterError):
    """The server rejected the ``tools`` request field (HTTP 400 mentioning tools)."""


class MappingError(AdapterError):
    """A custom_http config mapping failed. Message names the config key."""


class ContextOverflow(AgentError):
    """Messages cannot be made to fit the context budget."""
