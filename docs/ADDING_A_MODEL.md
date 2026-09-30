# Adding a model

Swapping or adding a model changes only `agent.yaml`. Code changes are needed only for a brand-new tool-call
syntax (step 3), and then only in `agent/adapters/parsers.py` plus a fixture.

## Decision tree

1. **Does the server speak the OpenAI chat-completions API?** (Ollama, vLLM, llama.cpp `llama-server`,
   LM Studio, most gateways.)
   Use `provider: openai_compat` and run the probe:

   ```yaml
   model:
     provider: openai_compat
     base_url: http://localhost:11434/v1
     model: qwen2.5-coder:32b
     native_tools: true
     context_window: 32768
     max_output_tokens: 4096
   ```

   ```
   agent doctor --config agent.yaml --probe-tools
   ```

   The probe runs six checks (single call, parallel calls, result round-trip, argument types, no-tool
   question, 200-line argument) and prints a recommended `native_tools`, `schema_level` and
   `prompted_tool_format`. If the server rejects the `tools` field (HTTP 400 mentioning tools) the adapter
   falls back to prompted tool calling automatically; set `native_tools: false` to skip that round trip.

2. **Otherwise** use `provider: custom_http` and describe the request/response shape:

   ```yaml
   model:
     provider: custom_http
     context_window: 16384
     max_output_tokens: 2048
     http:
       endpoint: https://llm.internal/v1/generate
       headers: {Authorization: "Bearer ${ENV:INHOUSE_KEY}"}
       input_mode: prompt          # prompt: one rendered string | messages: role list
       prompt_format: chatml       # chatml | llama3 | mistral | alpaca | jinja (+ prompt_template_file)
       request_template: |
         {"inputs": {{ prompt | tojson }},
          "parameters": {"max_new_tokens": {{ max_tokens }}, "temperature": {{ temperature }},
                         "stop": {{ stop | tojson }}}}
       response:
         text_path: "$.generated_text"         # JSONPath, required
         finish_reason_path: "$.finish_reason" # optional
         usage_prompt_path: "$.usage.input"    # optional
         usage_completion_path: "$.usage.output"
         error_path: "$.error"                 # non-null -> error
       stream: {enabled: false, format: sse, delta_path: "$.token.text", done_marker: "[DONE]"}
   ```

   Template variables: `prompt`, `messages`, `temperature`, `max_tokens`, `stop` (and `tools` for native
   mapping). Templates render in a sandbox and must produce valid JSON. Run `agent doctor --config agent.yaml`:
   it prints the raw request, the raw response and what every JSONPath matched. Mapping errors name the
   config key, e.g. `response.text_path '$.generated_text' matched nothing in: {...}`.

   If the in-house model has **native tool calls**, add:

   ```yaml
       tool_mapping:
         request_tools_template: '{"functions": {{ tools | tojson }}}'   # merged into the request body
       response:
         tool_calls_path: "$.result.calls[*]"
         tool_call_name_path: "$.fn"         # relative to each call
         tool_call_args_path: "$.params"     # object or JSON string
         tool_call_id_path: "$.cid"          # optional
   ```
   and set `native_tools: true`.

3. **The model emits an unusual tool-call syntax.** Add a strategy function to `agent/adapters/parsers.py`
   (append it to `STRATEGIES`), add at least 8 raw outputs to a new file under
   `tests/fixtures/model_outputs/` (edit `tests/fixtures/make_fixtures.py`), and run
   `pytest tests/test_parsers.py`. Do not touch other layers.

## Tuning knobs

| Symptom | Setting |
|---|---|
| Model breaks on nested/complex schemas | `schema_level: simple` (flattens objects, first `oneOf` branch, int enums as strings) |
| Too many tools confuse the model | `mcp.max_tools` and `mcp.allowed_servers` |
| Prompted calls often malformed | `prompted_tool_format: json_fence`, `repair_attempts: 3`, or `guided_decoding: vllm|llamacpp` |
| No system role in the chat template | `supports_system_role: false` |
| Template requires user/assistant alternation | `strict_alternation: true` |
| Token estimates drift | `tokenizer: {kind: server, url: .../tokenize, count_path: ...}` or `kind: hf` |
| Model-specific instructions | `family: qwen` loads `agent/prompts/qwen.md` instead of `system.md` |

## Measure

```
python -m evals.run --config agent.yaml --out results.json
```

Reports success rate, tool-call parse-error rate (target < 2% after repair), average iterations and tokens
over 25 small coding tasks.
