"""Evaluation harness (Phase 4).

    python -m evals.run --self-check                 # every task fails at baseline and passes with its solution
    python -m evals.run --oracle                     # run the loop with a scripted oracle model (harness check)
    python -m evals.run --config agent.yaml          # run a real model on all tasks
    python -m evals.run --config agent.yaml --only fix_add,add_chunk --out results.json

Records per task: success, iterations, tool calls, tool-call parse errors, prompt/completion tokens.
Summary: success rate, parse-error rate (parse errors / tool calls), average iterations, total tokens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.app import build_session
from agent.config import AgentConfig, load_config, load_config_dict
from agent.scripted import ScriptedAdapter, call, calls, say, steps_taken
from agent.types import Message, ModelResponse, ToolSpec
from evals.tasks import TASKS, Task


@dataclass
class TaskResult:
    """Outcome of one task run."""

    id: str
    category: str
    success: bool
    baseline_failed: bool
    iterations: int
    tool_calls: int
    parse_errors: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    stopped: str
    error: str | None = None


def materialize(task: Task, root: Path) -> None:
    """Write the task's files under ``root``."""
    for rel, content in task.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def run_check(task: Task, root: Path, timeout: int = 120) -> tuple[bool, str]:
    """Run the task's check command; True if exit code 0."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    cmd = task.check.replace("python ", f"{sys.executable} ", 1) if task.check.startswith("python ") else task.check
    p = subprocess.run(cmd, shell=True, cwd=root, capture_output=True, text=True, timeout=timeout, env=env)  # noqa: S602
    return p.returncode == 0, (p.stdout + p.stderr)[-2000:]


def apply_solution(task: Task, root: Path) -> None:
    """Apply reference ops directly (used by --self-check)."""
    for op in task.solution:
        p = root / op["path"]
        if op["op"] == "write":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(op["content"], encoding="utf-8")
        else:
            text = p.read_text(encoding="utf-8")
            if op["old_str"] not in text:
                raise AssertionError(f"{task.id}: reference edit old_str not found in {op['path']}")
            p.write_text(text.replace(op["old_str"], op["new_str"]) if op.get("replace_all")
                         else text.replace(op["old_str"], op["new_str"], 1), encoding="utf-8")


def oracle_policy(task: Task) -> Any:
    """Scripted model: read touched files, apply the reference ops via tools, run the check, answer."""
    existing = [op["path"] for op in task.solution if op["path"] in task.files]
    plan: list[ModelResponse] = []
    if existing:
        plan.append(calls([("read_file", {"path": p}) for p in dict.fromkeys(existing)], "Reading the files."))
    for op in task.solution:
        if op["op"] == "write":
            plan.append(call("write_file", path=op["path"], content=op["content"]))
        else:
            args = {"path": op["path"], "old_str": op["old_str"], "new_str": op["new_str"]}
            if op.get("replace_all"):
                args["replace_all"] = True
            plan.append(call("edit_file", **args))
    plan.append(call("bash", command=task.check))
    plan.append(say("Done. The check passes."))

    def policy(messages: list[Message], tools: list[ToolSpec] | None) -> ModelResponse:
        i = steps_taken(messages)
        return plan[min(i, len(plan) - 1)]
    return policy


def eval_config(base: AgentConfig | None, work: Path) -> AgentConfig:
    """Config for an eval run: isolated session dir, bypass permissions (the workspace is a temp copy)."""
    if base is None:
        return load_config_dict({"context": {"session_dir": str(work / ".sessions")},
                                 "permissions": {"mode": "bypass"}, "max_iterations": 30}, env={})
    return base.model_copy(update={
        "context": base.context.model_copy(update={"session_dir": str(work / ".sessions"),
                                                   "project_instructions_file": "AGENT.md"}),
        "permissions": base.permissions.model_copy(update={"mode": "bypass"})})


async def run_task(task: Task, base_cfg: AgentConfig | None, *, oracle: bool) -> TaskResult:
    """Materialize, verify baseline fails, run the agent, run the check."""
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"eval-{task.id}-") as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        materialize(task, root)
        baseline_ok, _ = run_check(task, root)
        cfg = eval_config(base_cfg, Path(tmp))
        adapter = ScriptedAdapter(oracle_policy(task)) if oracle else None
        session = await build_session(cfg, root, adapter=adapter, use_mcp=False)
        err = None
        try:
            res = await session.agent.run(task.prompt)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            res = None
        finally:
            await session.aclose()
        ok, _ = run_check(task, root)
        return TaskResult(task.id, task.category, ok and err is None, not baseline_ok,
                          res.iterations if res else 0, res.tool_calls if res else 0, res.parse_errors if res else 0,
                          res.prompt_tokens if res else 0, res.completion_tokens if res else 0,
                          round(time.monotonic() - t0, 2), res.stopped if res else "error", err)


def self_check(tasks: list[Task]) -> list[tuple[str, bool, bool]]:
    """For each task: (id, baseline_fails, solution_passes)."""
    out = []
    for task in tasks:
        with tempfile.TemporaryDirectory(prefix=f"selfcheck-{task.id}-") as tmp:
            root = Path(tmp)
            materialize(task, root)
            base_ok, _ = run_check(task, root)
            apply_solution(task, root)
            sol_ok, log = run_check(task, root)
            if not sol_ok:
                print(f"{task.id}: solution check output:\n{log}", file=sys.stderr)
            out.append((task.id, not base_ok, sol_ok))
    return out


def summarize(results: list[TaskResult]) -> dict[str, Any]:
    """Aggregate metrics."""
    n = len(results)
    calls_ = sum(r.tool_calls for r in results)
    return {"tasks": n, "success_rate": round(sum(r.success for r in results) / n, 4) if n else 0.0,
            "parse_error_rate": round(sum(r.parse_errors for r in results) / calls_, 4) if calls_ else 0.0,
            "avg_iterations": round(sum(r.iterations for r in results) / n, 2) if n else 0.0,
            "prompt_tokens": sum(r.prompt_tokens for r in results),
            "completion_tokens": sum(r.completion_tokens for r in results),
            "baseline_all_failed": all(r.baseline_failed for r in results)}


async def main_async(args: argparse.Namespace) -> int:
    """CLI implementation."""
    tasks = [t for t in TASKS if not args.only or t.id in args.only.split(",")]
    if args.self_check:
        rows = self_check(tasks)
        for tid, bf, sp in rows:
            print(f"[{'OK' if bf and sp else 'BAD'}] {tid}: baseline_fails={bf} solution_passes={sp}")
        good = all(bf and sp for _, bf, sp in rows)
        print(f"{sum(bf and sp for _, bf, sp in rows)}/{len(rows)} tasks well-formed")
        return 0 if good else 1
    if not args.oracle and not args.config:
        print("give --config agent.yaml (real model), --oracle, or --self-check", file=sys.stderr)
        return 2
    base = load_config(args.config) if args.config else None
    results = []
    for t in tasks:
        r = await run_task(t, base, oracle=args.oracle)
        results.append(r)
        print(f"[{'PASS' if r.success else 'FAIL'}] {r.id} iterations={r.iterations} calls={r.tool_calls} "
              f"parse_errors={r.parse_errors} {r.seconds}s" + (f" error={r.error}" if r.error else ""))
    summary = summarize(results)
    print(json.dumps(summary, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "results": [asdict(r) for r in results],
                                              "model": base.model.model_dump() if base else "oracle"}, indent=1,
                                             default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    p = argparse.ArgumentParser(prog="python -m evals.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--only")
    p.add_argument("--out")
    return asyncio.run(main_async(p.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
