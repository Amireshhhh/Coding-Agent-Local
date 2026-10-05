# Coding Agent Local

A terminal coding agent for models you host yourself. It reads, searches, edits and runs code in a project directory, and it talks to the model through a small adapter layer, so the same agent works with Ollama, vLLM, llama.cpp, LM Studio, or an internal HTTP endpoint with its own request format.

Local models are inconsistent about tool calling. Some support it natively, some emit calls as text in one of several formats, and some return malformed JSON. Most of this project is the machinery that makes the agent usable anyway: a parser for the common tool-call formats, JSON repair, a bounded retry when a call cannot be parsed, schema simplification for weaker models, and automatic fallback from native to prompted tool calling.

## Status

Version 0.1.0. The test suite (439 tests) passes, along with `ruff` and `mypy --strict`. All model behaviour is tested against mocked HTTP and local stand-in servers. The agent has not yet been run against a live model, so real-world tool-calling reliability is unmeasured. `docs/PROBES.md` lists which server behaviours are verified and which are assumptions, and `docs/BUILD_REPORT.md` has the full verification record.

## Requirements

- Python 3.11 or newer
- A model server: any OpenAI-compatible endpoint, or an HTTP API you can describe in a config file
- Optional: `git` (checkpoints and undo), `ripgrep` (faster search), `bubblewrap` (shell sandbox)

## Installation

```bash
git clone https://github.com/Amireshhhh/Coding-Agent-Local.git
cd Coding-Agent-Local
pip install .
```

This installs the `agent` command.

## Getting started

Copy the example config and point it at your server:

```bash
cp agent.yaml.example agent.yaml
```

```yaml
model:
  provider: openai_compat
  base_url: http://localhost:11434/v1   # Ollama
  model: qwen2.5-coder:32b
  context_window: 32768
  max_output_tokens: 4096
```

Check the connection and the model's tool calling:

```bash
agent doctor --probe-tools
```

The probe runs six checks (a single call, parallel calls, a result round trip, argument types, a question that needs no tool, and a long argument) and prints recommended settings for the model.

Then start the agent in your project directory:

```bash
agent                                             # interactive session
agent -p "fix the failing test" --output-format json   # one task, then exit
```

Ready-made configs for Ollama, vLLM, llama.cpp, LM Studio and a custom HTTP endpoint are in `examples/`.

## Configuration

The agent looks for `agent.yaml` in the current directory, then `~/.agent/agent.yaml`, or takes a path from `--config`. Values can reference environment variables with `${ENV:NAME}` or `${ENV:NAME:-default}`.

| Provider | Use it for |
|---|---|
| `openai_compat` | Servers with an OpenAI-style `/chat/completions` endpoint. Uses native tool calling and falls back to prompted tool calling if the server rejects the `tools` field. |
| `prompted` | Text-only models. Tools are described in the system prompt and calls are parsed from the output. |
| `custom_http` | Any other HTTP API. The request body is a Jinja template and the response is read with JSONPath, so onboarding a model needs no code changes. |

`docs/ADDING_A_MODEL.md` walks through each case and the tuning options.

Project-specific instructions go in `AGENT.md` at the project root. The agent adds them to its system prompt.

## Tools

Built in: `read_file`, `edit_file`, `write_file`, `bash`, `grep`, `glob`, `ls`, `todo_write`, `web_fetch`, `web_search`, and `task`, which hands a subtask to a sub-agent with its own context.

Tools from MCP servers are loaded from `.mcp.json` in the project or `~/.agent/mcp.json`. See `mcp.json.example`. The stdio, streamable HTTP and SSE transports are supported.

## Permissions

| Mode | Behaviour |
|---|---|
| `ask` (default) | Read-only tools run freely. Edits, shell commands and MCP tools need approval. |
| `accept_edits` | File edits run without asking. Shell commands still need approval. |
| `plan` | Read-only. |
| `bypass` | Nothing needs approval. |

Allow and deny rules refine this, for example `bash(git status)`, `edit_file(src/**)` or `mcp__github__*`.

File tools cannot reach outside the project root in any mode. A built-in list blocks some destructive shell commands, but it is pattern-based and is not a security boundary. For real isolation, set `sandbox: bwrap`, which runs shell commands with a read-only filesystem outside the project and no network, or use the Docker setup.

## Interactive commands

| Command | Effect |
|---|---|
| `/help` | List commands |
| `/clear` | Start a new conversation |
| `/compact` | Summarise older history to free context |
| `/model [name]` | Show or change the model |
| `/mcp` | MCP server status |
| `/permissions [mode]` | Show or change the permission mode |
| `/resume [id]` | List stored sessions or resume one |
| `/undo` | Restore files to the last checkpoint (needs `git_checkpoints: true`) |
| `/doctor` | Check the model configuration |

Markdown files in `.agent/commands/` become custom commands.

## Docker

`docker-compose.yml` defines three profiles: `ollama` and `vllm` start a model server next to the agent on a network with no internet access, and `custom` runs the agent alone against an external endpoint.

```bash
docker compose --profile ollama run --rm agent-ollama
```

The image has not been built or tested yet.

## Project layout

```
agent/
  adapters/    model backends, tool-call parsing, JSON repair
  tools/       tool registry, schema normalisation, built-in tools
  mcp/         MCP client
  context/     token budgeting, truncation, compaction, session storage
  loop.py      the agent loop
  cli.py       command-line interface
evals/         25 small coding tasks with pass/fail checks
tests/
docs/
examples/      sample configs
```

## Development

```bash
pip install -e '.[dev]'
./scripts/ci.sh                    # lint, type check, tests, coverage
python -m agent.demo               # three end-to-end scenarios against local stand-in servers
python -m evals.run --self-check   # confirm every eval task is well formed
```

To measure a model on the eval tasks:

```bash
python -m evals.run --config agent.yaml --out results.json
```

This reports success rate, tool-call parse-error rate, average iterations and token use.

## Documentation

- `docs/ADDING_A_MODEL.md`: configuring a new model or server
- `docs/PROBES.md`: verified and unverified server behaviour
- `docs/DECISIONS.md`: design decisions
- `docs/BUILD_REPORT.md`: test and verification results
