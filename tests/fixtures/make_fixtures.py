"""Generate tests/fixtures/model_outputs/*.json.

Fixtures are hand-written to match the documented output syntax of each model family
(Hermes/Qwen, Llama 3.x, Mistral, pythonic Llama 3.2 style, OpenAI-shaped JSON). They are NOT
recordings from live model servers. Each entry: text, expected calls, expected strategy,
expected remaining text (or null to skip that check).

Run: python tests/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).parent / "model_outputs"
KNOWN = ["get_weather", "read_file", "edit_file", "bash", "search-code", "mcp__fs__list"]


def c(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "arguments": args}


def e(text: str, calls: list[dict[str, Any]], strategy: str | None, remaining: str | None = None,
      errors: list[str | None] | None = None) -> dict[str, Any]:
    return {"text": text, "calls": calls, "strategy": strategy, "remaining": remaining,
            "errors": errors or [None] * len(calls)}


W = c("get_weather", city="Paris")
FIX: dict[str, list[dict[str, Any]]] = {}

FIX["hermes"] = [
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>', [W], "hermes", ""),
    e('Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
      [W], "hermes", "Let me check."),
    e('<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>\n'
      '<tool_call>{"name": "get_weather", "arguments": {"city": "Rome"}}</tool_call>',
      [W, c("get_weather", city="Rome")], "hermes", ""),
    e('<tool_call>\n{"name": "read_file", "parameters": {"path": "src/a.py"}}\n</tool_call>',
      [c("read_file", path="src/a.py")], "hermes", ""),
    e('<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}\n</tool_call>',
      [W], "hermes", ""),
    e('<tool_call>\n{"name": "Get_Weather", "arguments": {"city": "Paris"}}\n</tool_call>', [W], "hermes", ""),
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}', [W], "hermes", ""),
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
      '<tool_response>\n{"temp": 99}\n</tool_response>\nIt is 99 degrees.', [W], "hermes", ""),
    e('<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>\n'
      'Observation: total 0\nThe directory is empty.', [c("bash", command="ls -la")], "hermes", ""),
    e('<think>I should call <tool_call>{"name":"bash","arguments":{"command":"rm"}}</tool_call></think>\n'
      '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>', [W], "hermes", None),
    e('<tool_call>\n{"name": "edit_file", "arguments": {"path": "a.py", "old_str": "x = 1\\n", '
      '"new_str": "x = 2\\n"}}\n</tool_call>', [c("edit_file", path="a.py", old_str="x = 1\n", new_str="x = 2\n")],
      "hermes", ""),
]

FIX["json_fence"] = [
    e('```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```', [W], "json_fence", ""),
    e('I will call the tool:\n```json\n{"tool": "get_weather", "args": {"city": "Paris"}}\n```',
      [W], "json_fence", "I will call the tool:"),
    e('```tool_call\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```', [W], "json_fence", ""),
    e('```json\n{"function": "read_file", "input": {"path": "x"}}\n```', [c("read_file", path="x")], "json_fence", ""),
    e('```json\n[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
      '{"name": "get_weather", "arguments": {"city": "Oslo"}}]\n```', [W, c("get_weather", city="Oslo")],
      "json_fence", ""),
    e('```json\n{"name": "get_weather", "parameters": {"city": "Paris"}}\n```\nDone.', [W], "json_fence", "Done."),
    e('```tool_call\n{"name": "bash", "arguments": {"command": "pytest -q"}}\n```\n'
      '```tool_call\n{"name": "read_file", "arguments": {"path": "log.txt"}}\n```',
      [c("bash", command="pytest -q"), c("read_file", path="log.txt")], "json_fence", ""),
    e('```\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```', [W], "json_fence", ""),
    e('Here is the config:\n```json\n{"host": "x", "port": 80}\n```\nThat is all.', [], None,
      'Here is the config:\n```json\n{"host": "x", "port": 80}\n```\nThat is all.'),
]

FIX["openai_inline"] = [
    e('{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", '
      '"arguments": "{\\"city\\": \\"Paris\\"}"}}]}', [W], "openai_inline", ""),
    e('Calling now {"tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"}}}]}',
      [W], "openai_inline", "Calling now"),
    e('{"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}}, '
      '{"function": {"name": "read_file", "arguments": "{\\"path\\": \\"b\\"}"}}]}',
      [W, c("read_file", path="b")], "openai_inline", ""),
    e('{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}', [W], "openai_inline", ""),
    e('{"tool_calls": [{"function": {"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}}]} trailing',
      [c("bash", command="ls")], "openai_inline", "trailing"),
    e('{"tool_calls": [{"function": {"name": "get_weather", "arguments": "{\'city\': \'Paris\'}"}}]}',
      [W], "openai_inline", ""),
    e('{"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}], "content": null}',
      [W], "openai_inline", ""),
    e('{"tool_calls": [{"function": {"name": "mcp__fs__list", "arguments": "{}"}}]}', [c("mcp__fs__list")],
      "openai_inline", ""),
]

FIX["llama_python_tag"] = [
    e('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}', [W], "llama_python_tag", ""),
    e('{"name": "get_weather", "parameters": {"city": "Paris"}}', [W], "llama_python_tag", ""),
    e('{"type": "function", "name": "get_weather", "parameters": {"city": "Paris"}}', [W], "llama_python_tag", ""),
    e('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}; '
      '{"name": "get_weather", "parameters": {"city": "Rome"}}', [W, c("get_weather", city="Rome")],
      "llama_python_tag", ""),
    e('<|python_tag|>{"name": "read_file", "parameters": {"path": "main.py"}}<|eom_id|>',
      [c("read_file", path="main.py")], "llama_python_tag", "<|eom_id|>"),
    e('  {"name": "bash", "parameters": {"command": "git status"}}  \n', [c("bash", command="git status")],
      "llama_python_tag", ""),
    e('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"', [W], "llama_python_tag", ""),
    e('{"name": "search-code", "parameters": {"query": "def main"}}', [c("search-code", query="def main")],
      "llama_python_tag", ""),
]

FIX["mistral"] = [
    e('[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]', [W], "mistral", ""),
    e('[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Paris"}, "id": "abc123def"}]', [W], "mistral", ""),
    e('[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, '
      '{"name": "get_weather", "arguments": {"city": "Lima"}}]', [W, c("get_weather", city="Lima")], "mistral", ""),
    e('[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}', [W], "mistral", ""),
    e('[TOOL_CALLS]get_weather[CALL_ID]a1b2c3d4e[ARGS]{"city": "Paris"}', [W], "mistral", ""),
    e('Checking.[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "x.md"}}]',
      [c("read_file", path="x.md")], "mistral", "Checking."),
    e('[TOOL_CALLS] [{"name": "bash", "arguments": "{\\"command\\": \\"make\\"}"}]', [c("bash", command="make")],
      "mistral", ""),
    e('[TOOL_CALLS] {"name": "get_weather", "arguments": {"city": "Paris"}}', [W], "mistral", ""),
]

FIX["bare_json"] = [
    e('{"name": "get_weather", "arguments": {"city": "Paris"}}', [W], "bare_json", ""),
    e('  {"tool": "get_weather", "args": {"city": "Paris"}}\n', [W], "bare_json", ""),
    e('[{"name": "get_weather", "arguments": {"city": "Paris"}}, {"name": "read_file", "arguments": {"path": "a"}}]',
      [W, c("read_file", path="a")], "bare_json", ""),
    e('{"function": {"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}}', [W], "bare_json", ""),
    e('{"name": "get_weather", "input": {"city": "Paris"}}', [W], "bare_json", ""),
    e("{'name': 'get_weather', 'arguments': {'city': 'Paris'}}", [W], "bare_json", ""),
    e('{"name": "get_weather", "arguments": {"city": "Paris",}}', [W], "bare_json", ""),
    e('{"name": "bash", "arguments": {"command": "echo hi"}}', [c("bash", command="echo hi")], "bare_json", ""),
    e('{"name": "nonexistent_tool", "arguments": {"x": 1}}', [{"name": "nonexistent_tool", "arguments": {}}],
      "bare_json", "", ["unknown tool"]),
    e('{"answer": 42}', [], None, '{"answer": 42}'),
]

FIX["pythonic"] = [
    e('get_weather(city="Paris")', [W], "pythonic", ""),
    e("get_weather(city='Paris')", [W], "pythonic", ""),
    e('I will check.\nget_weather(city="Paris")', [W], "pythonic", "I will check."),
    e('[get_weather(city="Paris"), read_file(path="a.py")]', [W, c("read_file", path="a.py")], "pythonic", ""),
    e('get_weather(city="Paris")\nget_weather(city="Rome")', [W, c("get_weather", city="Rome")], "pythonic", ""),
    e('search-code(query="TODO", max_results=5)', [c("search-code", query="TODO", max_results=5)], "pythonic", ""),
    e('bash(command="ls", timeout=30, verbose=True, env=None, tags=["a", "b"])',
      [c("bash", command="ls", timeout=30, verbose=True, env=None, tags=["a", "b"])], "pythonic", ""),
    e('edit_file(path="a.py",\n          old_str="x",\n          new_str="y")',
      [c("edit_file", path="a.py", old_str="x", new_str="y")], "pythonic", ""),
    e('print(city="Paris")', [], None, 'print(city="Paris")'),
    e('The function get_weather(city="Paris") is useful.', [], None,
      'The function get_weather(city="Paris") is useful.'),
]

FIX["malformed"] = [
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris",}}\n</tool_call>', [W], "hermes", ""),
    e("<tool_call>\n{'name': 'get_weather', 'arguments': {'city': 'Paris'}}\n</tool_call>", [W], "hermes", ""),
    e('<tool_call>\n{“name”: “get_weather”, “arguments”: {“city”: '
      '“Paris”}}\n</tool_call>', [W], "hermes", ""),
    e('<tool_call>\n{"name": "bash", "arguments": {"command": "echo a\nb"}}\n</tool_call>',
      [c("bash", command="echo a\nb")], "hermes", ""),
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}\n</tool_call>', [W], "hermes", ""),
    e('<tool_call>\n{"name": "get_weather", "arguments": {"city": }}\n</tool_call>',
      [{"name": "get_weather", "arguments": {}}], "hermes", "", ["invalid JSON"]),
    e('<tool_call>\n{"name": "made_up", "arguments": {"a": 1}}\n</tool_call>',
      [{"name": "made_up", "arguments": {}}], "hermes", "", ["unknown tool"]),
    e('<tool_call>\n{"arguments": {"a": 1}}\n</tool_call>', [{"name": "unknown", "arguments": {}}], "hermes", "",
      ["no 'name'"]),
    e('<tool_call>\n{"name": "get_weather", "arguments": [1, 2]}\n</tool_call>',
      [{"name": "get_weather", "arguments": {}}], "hermes", "", ["must be a JSON object"]),
    e('```tool_call\nnot json at all\n```', [{"name": "unknown", "arguments": {}}], "json_fence", "",
      ["invalid JSON"]),
    e('get_weather("Paris")', [{"name": "get_weather", "arguments": {}}], "pythonic", "", ["positional"]),
    e('{"name": "get_weather", "arguments": {"city": "Paris"}', [W], "bare_json", ""),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fmt, entries in FIX.items():
        (OUT / f"{fmt}.json").write_text(json.dumps({"known_tools": KNOWN, "cases": entries}, indent=1,
                                                    ensure_ascii=False) + "\n", encoding="utf-8")
        print(fmt, len(entries))


if __name__ == "__main__":
    main()
