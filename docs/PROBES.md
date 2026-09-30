# PROBES

Rule 3: uncertain library / server behavior is probed and recorded here. Each entry says whether it was
**verified** in this build environment (with the command) or is **unverified** (no access) together with the
exact command to verify it.

Environment of this build: Python 3.11.15, mcp 1.27.0, httpx 0.28.1, pydantic 2.13.3, jinja2 3.1.6,
jsonpath-ng 1.8.0, jsonschema 4.26.0. Outbound internet was restricted (package registries allowed; Docker Hub,
GitHub releases, ollama.com, huggingface.co and openaipublic.blob.core.windows.net were blocked).

## Verified

| # | Question | Result | How |
|---|---|---|---|
| 1 | `mcp` SDK: timeout on `ClientSession.call_tool(read_timeout_seconds=...)` | raises `McpError`, `error.code == 408` | scratch probe against `tests/mcp_test_server.py`; covered by `tests/test_mcp.py::test_timeout` |
| 2 | `mcp` SDK: stdio server dies mid-call | pending call raises `McpError("Connection closed")`; later calls raise `anyio.ClosedResourceError` | same probe; `test_server_crash_mid_call` |
| 3 | `mcp` SDK: tool list change notification | arrives at `message_handler` as `ServerNotification` whose `.root` is `ToolListChangedNotification` | same probe; `test_list_changed_refresh` |
| 4 | `mcp` SDK: tool error | `CallToolResult.isError == True`, text `Error executing tool fail: boom` | `test_is_error_mapping` |
| 5 | `mcp` SDK: pagination API | `ClientSession.list_tools(params=PaginatedRequestParams(cursor=...))`, `ListToolsResult.nextCursor` | `inspect.signature`; `test_pagination_cursor` |
| 6 | `mcp` SDK 1.27: `streamablehttp_client` | deprecated (DeprecationWarning) in favour of `streamable_http_client(url, http_client=...)`; both are supported, the new one preferred | pytest warning; `test_streamable_http_transport` |
| 7 | `mcp` SDK: anyio cancel scopes | contexts must be exited in the task that entered them, hence one runner task per server | design + `test_no_zombies_after_shutdown` (child processes gone after shutdown) |
| 8 | FastMCP input schemas contain `title` keys | yes (e.g. `{"title": "A", "type": "integer"}`); the `standard` normalizer drops them | `test_start_lists_and_registers` |
| 9 | PyYAML parses bare `off` | as boolean `False` | `tests/test_config.py::test_example_configs_parse` failed before the fix |
| 10 | tiktoken encoding download | fails here (proxy blocks `openaipublic.blob.core.windows.net`); `build_counter` raises `AdapterError` naming `tokenizer.kind 'tiktoken'` | `python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"`; `test_tiktoken_counter_or_clear_error` |
| 11 | bubblewrap sandbox | with `--ro-bind / / --tmpfs /tmp --bind ROOT ROOT --unshare-net`: writes outside ROOT fail (Read-only file system), writes inside succeed, network is unreachable. `--tmpfs /tmp` must come before the ROOT bind when ROOT is under /tmp | `test_bash_bwrap_sandbox` (bwrap 0.9.0) |
| 12 | Debian system setuptools + `pip wheel --no-build-isolation` | fails (`AttributeError: install_layout`); isolated builds work | `python -m pip wheel --no-deps -w dist .` |
| 13 | jinja2 `tojson` filter in `SandboxedEnvironment` | available; attribute access like `"".__class__` raises `SecurityError` | `test_template_sandboxed` |

## Unverified (no access to these servers from the build environment)

The adapters were tested against mocked HTTP (respx) and local fake servers that follow the OpenAI chat
completions shape. The following behaviors of real servers are **documented expectations, not verified here**.
Run the commands below and record the results.

| # | Server | Expectation | Verify with |
|---|---|---|---|
| U1 | Ollama | `/v1/chat/completions` accepts `tools`; models without tool support answer HTTP 400 mentioning tools (triggers the prompted fallback) | `agent doctor --config examples/ollama-qwen.yaml --probe-tools` |
| U2 | vLLM | native tools need `--enable-auto-tool-choice --tool-call-parser <parser>`; without them tool calls may arrive as text in `content` (handled by quirk 4) | `vllm serve <model> --enable-auto-tool-choice --tool-call-parser hermes` then `agent doctor --probe-tools` |
| U3 | llama.cpp `llama-server` | tool calling needs `--jinja`; `POST /tokenize {"content": ...}` returns `{"tokens": [...]}` | `agent doctor --config examples/llamacpp.yaml --probe-tools` |
| U4 | vLLM `POST /tokenize` | returns a JSON object with `count` | `curl -s localhost:8000/tokenize -d '{"prompt":"hi","model":"<m>"}'` |
| U5 | LM Studio | OpenAI-compatible endpoint at `:1234/v1` | `agent doctor --config examples/lmstudio.yaml --probe-tools` |
| U6 | vLLM guided decoding | request field `guided_json` (older) — newer vLLM versions may expect `structured_outputs` | set `guided_decoding: vllm`, run `agent doctor --probe-tools`, inspect with `--trace-llm` |
| U7 | llama.cpp guided decoding | request field `json_schema` on `/v1/chat/completions` | same as U6 with `guided_decoding: llamacpp` |
| U8 | Brave / SearXNG search | response paths `$.web.results[*]` / `$.results[*]` | configure `search:` and call the `web_search` tool |
| U9 | firejail sandbox | `firejail --quiet --whitelist=ROOT --net=none` | set `sandbox: firejail` (firejail was not installable here) |
| U10 | Docker image build | `python:3.11-slim` base; Docker Hub was blocked (HTTP 403), so the image was not built. `docker compose config` validated all three profiles | `docker build -t agent .` |

## Phase 1 exit criterion 2 (live model)

**Not verified here**: no local model could be downloaded or run (ollama.com / huggingface.co blocked). To
complete it:

```
ollama pull qwen2.5-coder:32b         # or any tool-capable model
agent doctor --config examples/ollama-qwen.yaml --probe-tools
python -m agent.demo --config examples/ollama-qwen.yaml --live
python -m evals.run --config examples/ollama-qwen.yaml --out results.json
```

Paste the reports below.
