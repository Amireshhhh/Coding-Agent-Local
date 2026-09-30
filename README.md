# agent — a model-agnostic terminal coding agent

A Claude-Code-style coding agent that runs on local open-source models (Ollama, vLLM, llama.cpp,
LM Studio) and on arbitrary in-house HTTP LLMs through one internal contract (`agent/types.py`).

## Quickstart

```bash
pip install .                         # installs the `agent` command (Python 3.11+)
cp agent.yaml.example agent.yaml      # point it at your model server
agent doctor --probe-tools            # check connectivity + tool calling, get recommended settings
agent                                 # interactive REPL in the current directory
agent -p "fix the failing test" --permission-mode accept_edits --output-format json
```

MCP servers: copy `mcp.json.example` to `.mcp.json` (project) or `~/.agent/mcp.json` (user).
Project instructions: put them in `AGENT.md` at the project root.

## What is inside

| Area | Where |
|---|---|
| Canonical types + transcript invariant | `agent/types.py` |
| Config + `${ENV:NAME:-default}` | `agent/config.py`, `agent.yaml.example`, `examples/` |
| Adapters: native OpenAI-compatible, prompted (any text model), config-mapped in-house HTTP, auto-fallback | `agent/adapters/` |
| Tool-call parser chain (hermes, json_fence, openai_inline, llama python_tag, mistral, bare JSON, pythonic) + JSON repair | `agent/adapters/parsers.py` |
| Schema normalizer (full / standard / simple), registry with validation, coercion, timeouts, truncation | `agent/tools/` |
| MCP client (stdio + streamable HTTP + SSE), namespacing, reconnect, list_changed, tool limits | `agent/mcp/` |
| Context budget, truncation, stale-result clearing, compaction, hard drop, JSONL sessions | `agent/context/` |
| Builtin tools: read_file, edit_file, write_file, bash, grep, glob, ls, todo_write, web_fetch, web_search | `agent/tools/builtin/` |
| Agent loop, permissions, hooks, sub-agents, git checkpoints | `agent/loop.py`, `agent/permissions.py`, `agent/hooks.py`, `agent/subagent.py`, `agent/checkpoints.py` |
| CLI / REPL (`/clear /compact /model /mcp /permissions /resume /doctor /undo`, custom `.agent/commands/*.md`) | `agent/cli.py`, `agent/commands.py` |
| Doctor + 6-check tool probe | `agent/doctor.py` |
| Demo harness (3 end-to-end scenarios) | `python -m agent.demo` |
| Evaluation harness (25 tasks) | `python -m evals.run` |
| Logging (JSON, secrets redacted), `--trace-llm` | `agent/logsetup.py`, `agent/trace.py` |
| Docker | `Dockerfile`, `docker-compose.yml` (profiles `ollama`, `vllm`, `custom`) |

## Permission modes

`ask` (default: read-only tools run, others ask), `accept_edits` (file edits allowed, bash asks),
`plan` (read-only), `bypass` (everything except the dangerous-command denylist and the path sandbox).
Rules: `permissions.allow` / `deny` with patterns like `bash(git status)`, `edit_file(src/**)`, `mcp__github__*`.
The denylist is a guard against accidents, not a security boundary; use `sandbox: bwrap` (bash runs with a
read-only filesystem outside the project and no network) or the Docker profiles for isolation.

## Development

```bash
pip install -e '.[dev]'
./scripts/ci.sh          # ruff, mypy --strict agent/, pytest + coverage gate (>= 85% on core packages)
python -m agent.demo     # end-to-end scenarios against local fake model servers
python -m evals.run --self-check   # every eval task fails at baseline and passes with its reference solution
```

Docs: `docs/ADDING_A_MODEL.md`, `docs/PROBES.md` (verified vs. unverified server behavior),
`docs/DECISIONS.md`.
