"""Configuration loading (``agent.yaml``) with ``${ENV:NAME}`` / ``${ENV:NAME:-default}`` interpolation."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_ENV_RE = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


class ConfigError(Exception):
    """Invalid configuration. The message names the offending key."""


def interpolate_env(value: Any, path: str = "", env: dict[str, str] | None = None) -> Any:
    """Recursively replace ``${ENV:NAME}`` and ``${ENV:NAME:-default}`` in all strings.

    Raises:
        ConfigError: if a variable without default is unset (message names key path).
    """
    env_map = os.environ if env is None else env
    if isinstance(value, dict):
        return {k: interpolate_env(v, f"{path}.{k}" if path else str(k), env) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env(v, f"{path}[{i}]", env) for i, v in enumerate(value)]
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name, default = m.group(1), m.group(2)
            if name in env_map:
                return env_map[name]
            if default is not None:
                return default
            raise ConfigError(f"config key '{path}': environment variable {name} is not set "
                              f"(set it or use ${{ENV:{name}:-default}})")

        return _ENV_RE.sub(repl, value)
    return value


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TokenizerConfig(_Strict):
    """Token counter selection (section 6.2)."""

    kind: Literal["tiktoken", "hf", "server", "chars_div_4"] = "chars_div_4"
    encoding: str = "cl100k_base"
    path: str | None = None
    url: str | None = None
    request_field: str = "content"
    count_path: str = "$.tokens"


class ResponseMapping(_Strict):
    """JSONPath mapping of an in-house LLM response."""

    text_path: str = "$.generated_text"
    finish_reason_path: str | None = None
    usage_prompt_path: str | None = None
    usage_completion_path: str | None = None
    error_path: str | None = None
    tool_calls_path: str | None = None
    tool_call_name_path: str = "$.name"
    tool_call_args_path: str = "$.arguments"
    tool_call_id_path: str | None = None


class StreamMapping(_Strict):
    """Streaming mapping for an in-house LLM."""

    enabled: bool = False
    format: Literal["sse", "ndjson"] = "sse"
    delta_path: str = "$.token.text"
    done_marker: str = "[DONE]"
    finish_reason_path: str | None = None


class ToolMapping(_Strict):
    """Native tool mapping for an in-house LLM (5.6)."""

    request_tools_template: str


class HTTPConfig(_Strict):
    """Config-only mapping for ``custom_http``."""

    endpoint: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    input_mode: Literal["prompt", "messages"] = "prompt"
    prompt_format: Literal["chatml", "llama3", "mistral", "alpaca", "jinja"] = "chatml"
    prompt_template_file: str | None = None
    request_template: str
    response: ResponseMapping = Field(default_factory=ResponseMapping)
    stream: StreamMapping = Field(default_factory=StreamMapping)
    tool_mapping: ToolMapping | None = None
    tokenizer: TokenizerConfig | None = None


class ModelConfig(_Strict):
    """``model:`` section of agent.yaml (5.1)."""

    provider: Literal["openai_compat", "prompted", "custom_http"] = "openai_compat"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "none"
    model: str = ""
    native_tools: bool = True
    context_window: int = Field(default=32768, gt=0)
    max_output_tokens: int = Field(default=4096, gt=0)
    temperature: float = 0.2
    schema_level: Literal["full", "standard", "simple"] = "standard"
    prompted_tool_format: Literal["hermes", "json_fence", "custom"] = "hermes"
    prompted_tool_template: str | None = None
    tool_choice: str = "auto"
    timeout_s: float = 300
    max_retries: int = 3
    parallel_tool_calls: bool = True
    streaming: bool = True
    supports_system_role: bool = True
    supports_tool_role: bool = False
    strict_alternation: bool = False
    repair_attempts: int = 2
    guided_decoding: Literal["off", "vllm", "llamacpp"] = "off"
    backend_kind: Literal["chat", "completion"] = "chat"
    prompt_format: Literal["chatml", "llama3", "mistral", "alpaca", "jinja"] = "chatml"
    prompt_template_file: str | None = None
    http: HTTPConfig | None = None
    tokenizer: TokenizerConfig | None = None
    family: str | None = None

    @field_validator("guided_decoding", mode="before")
    @classmethod
    def _yaml_off(cls, v: Any) -> Any:
        # YAML 1.1 parses a bare `off` as the boolean False.
        return "off" if v is False else v

    @model_validator(mode="after")
    def _check(self) -> ModelConfig:
        if self.provider == "custom_http" and self.http is None:
            raise ValueError("provider 'custom_http' requires an 'http' section")
        if self.prompted_tool_format == "custom" and not self.prompted_tool_template:
            raise ValueError("prompted_tool_format 'custom' requires 'prompted_tool_template'")
        if self.max_output_tokens >= self.context_window:
            raise ValueError("max_output_tokens must be smaller than context_window")
        return self

    def effective_tokenizer(self) -> TokenizerConfig:
        """Tokenizer config: ``model.tokenizer`` then ``model.http.tokenizer`` then default."""
        if self.tokenizer is not None:
            return self.tokenizer
        if self.http is not None and self.http.tokenizer is not None:
            return self.http.tokenizer
        return TokenizerConfig()


class ContextConfig(_Strict):
    """Context window management settings (section 6)."""

    safety_margin_pct: float = 0.05
    safety_margin_min: int = 512
    compact_threshold: float = 0.8
    max_tool_output_tokens: int = 4000
    keep_recent_tool_results: int = 6
    message_overhead: int = 4
    count_fudge: float | None = None
    project_instructions_file: str = "AGENT.md"
    session_dir: str = "~/.agent/sessions"


class MCPSettings(_Strict):
    """MCP client settings (section 7)."""

    config_files: list[str] | None = None
    max_tools: int | None = None
    allowed_servers: list[str] = Field(default_factory=list)
    call_timeout_s: float = 120
    connect_timeout_s: float = 30
    degraded_after_failures: int = 3


class PermissionsConfig(_Strict):
    """Permission settings (Phase 2)."""

    mode: Literal["ask", "accept_edits", "plan", "bypass"] = "ask"
    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class HookConfig(_Strict):
    """A pre/post tool-use shell hook (Phase 3)."""

    matcher: str = "*"
    command: str
    timeout_s: float = 30


class HooksConfig(_Strict):
    """Hook lists."""

    pre_tool_use: list[HookConfig] = Field(default_factory=list)
    post_tool_use: list[HookConfig] = Field(default_factory=list)


class SearchConfig(_Strict):
    """Pluggable web search backend (Phase 3)."""

    backend: Literal["none", "searxng", "brave", "custom"] = "none"
    url: str | None = None
    api_key: str | None = None
    results_path: str = "$.results[*]"
    title_key: str = "title"
    url_key: str = "url"
    snippet_key: str = "content"


class AgentConfig(_Strict):
    """Root of agent.yaml."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    permissions: PermissionsConfig = Field(default_factory=PermissionsConfig)
    hooks: HooksConfig = Field(default_factory=HooksConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    max_iterations: int = 50
    tool_timeout_s: float = 120
    git_checkpoints: bool = False
    commands_dir: str = ".agent/commands"
    sandbox: Literal["none", "bwrap", "firejail"] = "none"
    sandbox_network: bool = False
    subagent_max_iterations: int = 20


def load_config_dict(data: dict[str, Any], env: dict[str, str] | None = None) -> AgentConfig:
    """Validate an already-parsed config mapping (after env interpolation).

    Raises:
        ConfigError: with dotted key paths for every validation failure.
    """
    resolved = interpolate_env(data, env=env)
    try:
        return AgentConfig.model_validate(resolved)
    except ValidationError as e:
        lines = []
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"config key '{loc}': {err['msg']}")
        raise ConfigError("invalid configuration:\n  " + "\n  ".join(lines)) from None


def load_config(path: str | Path, env: dict[str, str] | None = None) -> AgentConfig:
    """Load and validate ``agent.yaml``.

    Raises:
        ConfigError: file missing, YAML invalid, env var missing, or schema invalid.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"config file {p}: invalid YAML: {e}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"config file {p}: top level must be a mapping")
    return load_config_dict(raw, env=env)
