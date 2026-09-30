# BUILD REPORT

Everything below was produced by running the commands shown, in this repository, on
2026-09-30 (Python 3.11.15, Linux). "Verified" means a command ran and its output is quoted.
"Not verified" means it could not be run here, with the reason.

## Gate results

```
$ ./scripts/ci.sh            # exit code 0
ruff check agent tests evals          -> All checks passed!
mypy --strict agent                   -> Success: no issues found in 44 source files
pytest --cov=agent                    -> 439 passed
coverage (adapters/tools/context/mcp) -> 94% combined; per package: adapters 95%, tools 94%, context 96%, mcp 90%
```

The full suite was also run 3 more times in a row: 437/437 passed each time (before the 2 last MCP tests were
added); no flaky tests observed.

## Phase 1 tasks

| ID | Task | Files | Tests (count) |
|---|---|---|---|
| 0.1 | scaffold, tooling, CI, types | `pyproject.toml`, `scripts/ci.sh`, `agent/types.py`, `agent/errors.py` | `test_types.py` (10) |
| 0.2 | config + env interpolation | `agent/config.py`, `agent.yaml.example`, `examples/*.yaml` | `test_config.py` (9) |
| 1.1 | schema normalizer (3 levels) | `agent/tools/schema.py` | `test_schema.py` (31): 6 required schemas x levels + ref/recursion/depth/enum/truncation |
| 1.2 | registry, validation, coercion, errors | `agent/tools/registry.py`, `agent/tools/base.py` | `test_registry.py` (33): every `execute` error path + 18 coercion cases |
| 1.3 | parsers + repair_json + fixtures + fuzz | `agent/adapters/parsers.py`, `tests/fixtures/` | `test_parsers.py` (117): 76 fixture cases, 500-mutation fuzz |
| 1.4 | OpenAI-compatible adapter, 10 quirks | `agent/adapters/openai_compat.py`, `agent/adapters/base.py` | `test_openai_compat.py` (24): one respx test per quirk + retry policy |
| 1.5 | prompt formats + prompted adapter + repair loop | `agent/adapters/prompt_formats.py`, `agent/adapters/prompted.py` | `test_prompted.py` (28) |
| 1.6 | custom HTTP adapter | `agent/adapters/custom_http.py` | `test_custom_http.py` (16): flat text, nested list, NDJSON stream, SSE, native tool mapping |
| 1.7 | factory + fallback + contract suite | `agent/adapters/__init__.py` | `adapter_contract.py` (45): 5 adapters x 4 scenarios, identical canonical output |
| 1.8 | context manager | `agent/context/*` | `test_context.py` (21): hypothesis property test (300 examples in CI; 3000 run once, passed) |
| 1.9 | doctor + probe | `agent/doctor.py`, `agent/trace.py` | `test_doctor.py` (6) |
| 1.10 | MCP client | `agent/mcp/*`, `tests/mcp_test_server.py` | `test_mcp.py` (22): stdio, streamable HTTP, SSE, crash, timeout, list_changed, zombies |
| 1.11 | demo harness | `agent/demo.py`, `agent/demo_mcp_server.py` | `test_demo.py` (2) |
| 1.12 | docs | `README.md`, `docs/ADDING_A_MODEL.md`, `docs/PROBES.md`, `docs/DECISIONS.md` | - |

Fixture counts per format (>= 8 required): hermes 11, json_fence 9, openai_inline 8, llama_python_tag 8,
mistral 8, bare_json 10, pythonic 10, malformed 12. **They are hand-written to the documented syntax of each
model family, not recorded from live servers** (no model server was reachable).

Verbatim-text checks (script run against the build plan file): the compaction prompt and the Hermes
tool-instruction block are byte-identical to the plan (`True`, `True`).

## Phase 1 exit criteria

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | pytest green; mypy --strict clean; coverage >= 85% on adapters/tools/context/mcp | **Met** | CI output above |
| 2 | `agent doctor --probe-tools` passes 6 checks on a local model via Ollama/vLLM | **Not verified** | No model could be downloaded or served here (ollama.com, huggingface.co, GitHub releases blocked). The probe logic is tested with scripted models (`test_doctor.py`). Commands to finish this are in `docs/PROBES.md`. |
| 3 | model calls an MCP tool + a local tool in parallel, gets results, answers; transcript valid | **Met** | `python -m agent.demo` scenario 1: `answer='2 + 3 = 5; the clock reads 12:00.' tool_results={'call_add': '5', 'call_clock': '12:00'} iterations=2` (real stdio MCP server, real HTTP to a local fake model) |
| 4 | never-seen in-house model onboarded by editing ONLY agent.yaml | **Met** | demo scenario 3: a local HTTP server with a nonstandard shape (`{"conversation": ...}` in, `{"payload": {"candidates": [{"output": ...}]}}` out) driven by a YAML file loaded through `load_config`; the model called `ls` and answered |
| 5 | 100-turn synthetic session stays under budget, pinned items preserved | **Met** | `test_context.py::test_100_turn_session_stays_under_budget` asserts budget + pins + valid transcript after every one of 100 turns |

## Phase 2 (core agent)

Implemented: builtin tools (`read_file`, `edit_file`, `write_file`, `bash`, `grep`, `glob`, `ls`,
`todo_write`), loop (`agent/loop.py`), permissions (`agent/permissions.py`), system prompt + `AGENT.md` +
per-family overrides (`agent/prompts/`). Tests: `test_builtin_tools.py` (14), `test_permissions.py` (29),
`test_loop.py` (19).

Exit criterion "fix a failing test in a sample repo with a mock LLM": **met** (`test_loop.py::
test_benchmark_fix_failing_test` runs the real loop, real tools and real `pytest` in a temp repo). The
"(target) local 30B-class model" part: **not verified** (no model available).

## Phase 3 (UX & extensibility)

Implemented and tested: REPL with slash commands, streaming rendering, tool-call display, diff preview in
approval prompts, `-p` with `text|json|stream-json`, sub-agents (`task` tool), hooks, custom markdown
commands, git checkpoints + `/undo`, `web_fetch` / `web_search`. `test_cli.py` (13) drives the REPL with
scripted input and a real prompt_toolkit session on a pipe input; `test_loop.py` covers hooks, checkpoints
and sub-agents.

Not verified: interactive use on a real TTY by a person; `web_search` against real Brave/SearXNG APIs
(mocked only).

## Phase 4 (hardening & deployment)

| Item | Status | Evidence |
|---|---|---|
| Eval harness, 25 tasks (12 fix-bug, 7 add-function, 4 refactor, 2 write-tests) | **Verified mechanics** | `python -m evals.run --self-check` -> `25/25 tasks well-formed` (each fails at baseline, passes with its reference solution); `python -m evals.run --oracle` -> `success_rate 1.0, parse_error_rate 0.0, baseline_all_failed true` |
| Eval results for real models | **Not verified** | needs a live model: `python -m evals.run --config agent.yaml` |
| `pip install .` -> `agent` command | **Verified** | wheel built, installed into a fresh venv, `agent --version` -> `agent 0.1.0`; installed CLI ran `-p` end-to-end against a local fake server and edited a file |
| Dockerfile | **Written, not built** | Docker Hub returned 403 for `python:3.11-slim` |
| docker-compose profiles `ollama`, `vllm`, `custom` | **Config validated only** | `docker compose --profile <p> config -q` succeeded for all three; services not started |
| bash sandbox: bubblewrap | **Verified** | `test_bash_bwrap_sandbox`: writes outside the project fail, network unreachable |
| bash sandbox: firejail | **Not verified** | firejail not installable here |
| Config examples (5) | **Verified parse** | `test_config.py::test_example_configs_parse` |
| JSON logs with redaction, `--trace-llm` | **Verified** | `test_cli.py::test_redaction_and_logging`, `test_trace_recorder_dump`, `test_main_async_print_and_doctor` |

## Known limitations (stated plainly)

- No live LLM was available, so tool-calling reliability of any real model is unmeasured. All adapter
  behavior is verified against mocked HTTP, local fake servers and hand-written model outputs.
- Server-specific behaviors listed as U1-U10 in `docs/PROBES.md` are unverified.
- `tiktoken` cannot download its encoding here; the code reports this as a clear configuration error.
  The `hf` tokenizer path requires `transformers`, which is not installed and not tested.
- The dangerous-command denylist is pattern-based and can be evaded by obfuscation; it is a guard, not a
  security boundary. Use `sandbox: bwrap` or the Docker profiles for isolation.
