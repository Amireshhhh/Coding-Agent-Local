"""Command-line interface (Phase 3).

    agent                         interactive REPL
    agent -p "task"               one task, non-interactive (--output-format text|json|stream-json)
    agent --resume <id>           continue a stored session
    agent doctor [--probe-tools]  check the model configuration
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from rich.console import Console
from rich.markup import escape
from rich.syntax import Syntax
from rich.table import Table

from agent import __version__
from agent.adapters import build_adapter
from agent.app import Session, build_session, describe_session
from agent.checkpoints import Checkpoints, GitError
from agent.commands import CustomCommand, load_commands, split_command
from agent.config import AgentConfig, ConfigError, load_config, load_config_dict
from agent.context.manager import SessionStore
from agent.doctor import doctor
from agent.errors import AgentError
from agent.logsetup import collect_secrets, setup_logging
from agent.tools.base import Tool
from agent.trace import TraceRecorder
from agent.types import Message, ToolCall

ReadLine = Callable[[str], Awaitable[str]]
MODES = ("ask", "accept_edits", "plan", "bypass")
HELP = """Commands:
  /help                 this help
  /clear                start a fresh conversation (same session settings)
  /compact              summarize older history now
  /model [name]         show or switch the model name
  /mcp                  MCP server status
  /permissions [mode]   show or set the permission mode (ask|accept_edits|plan|bypass)
  /resume [id]          list sessions, or resume one
  /doctor               check the model configuration
  /undo                 restore the working tree to the last checkpoint (git_checkpoints: true)
  /status  /todos       session info / current todo list
  /exit                 quit
Custom commands: put markdown files in .agent/commands/<name>.md ($ARGUMENTS is replaced)."""


def find_config(explicit: str | None) -> AgentConfig:
    """``--config`` or ``./agent.yaml`` or ``~/.agent/agent.yaml``.

    Raises:
        ConfigError: none found.
    """
    if explicit:
        return load_config(explicit)
    for cand in (Path("agent.yaml"), Path("~/.agent/agent.yaml").expanduser()):
        if cand.is_file():
            return load_config(cand)
    raise ConfigError("no agent.yaml found. Copy agent.yaml.example to ./agent.yaml (or ~/.agent/agent.yaml) "
                      "or pass --config")


def short_args(call: ToolCall, limit: int = 100) -> str:
    """Compact one-line rendering of call arguments."""
    s = json.dumps(call.arguments, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 3] + "..."


class Renderer:
    """Prints agent events to a rich console."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self.streamed = False

    def __call__(self, kind: str, data: dict[str, Any]) -> None:
        c = self.console
        if kind == "text_delta":
            c.print(escape(data["text"]), end="", soft_wrap=True, highlight=False)
            self.streamed = True
        elif kind == "assistant":
            msg: Message = data["message"]
            if self.streamed:
                c.print()
                self.streamed = False
            elif msg.content:
                c.print(escape(msg.content), highlight=False)
        elif kind == "tool_call":
            call: ToolCall = data["call"]
            c.print(f"[bold cyan]> {escape(call.name)}[/] [dim]{escape(short_args(call))}[/]")
        elif kind == "tool_result":
            res: Message = data["result"]
            text = res.content or ""
            first = text.splitlines()[0] if text else "(empty)"
            n = text.count("\n") + 1 if text else 0
            style = "red" if text.startswith("ERROR") else "dim"
            c.print(f"  [{style}]⎿ {escape(first[:160])}{'' if n <= 1 else f'  (+{n - 1} lines)'}[/]")
        elif kind == "notice":
            c.print(f"[dim]{escape(data['text'])}[/]")


def make_approver(console: Console, read_line: ReadLine) -> Callable[..., Awaitable[Literal["yes", "always", "no"]]]:
    """Interactive approval prompt showing a diff preview for edits."""
    async def approve(tool: Tool, args: dict[str, Any], preview: str) -> Literal["yes", "always", "no"]:
        console.print(f"[bold yellow]Permission needed:[/] {escape(tool.describe_call(args))}")
        if preview:
            console.print(Syntax(preview, "diff", theme="ansi_dark", word_wrap=True))
        while True:
            ans = (await read_line("Allow? [y]es / [a]lways for this tool / [n]o: ")).strip().lower()
            if ans in ("y", "yes"):
                return "yes"
            if ans in ("a", "always"):
                return "always"
            if ans in ("n", "no", ""):
                return "no"
    return approve


class Repl:
    """Interactive loop; input and output are injectable for testing."""

    def __init__(self, cfg: AgentConfig, root: Path, console: Console, read_line: ReadLine, *,
                 resume: str | None = None, trace: TraceRecorder | None = None) -> None:
        self.cfg = cfg
        self.root = root
        self.console = console
        self.read_line = read_line
        self.resume = resume
        self.trace = trace
        self.session: Session | None = None
        self.commands: dict[str, CustomCommand] = {}

    async def start(self) -> None:
        """Build the session."""
        self.commands = load_commands(self.root / self.cfg.commands_dir)
        client = self.trace.client(self.cfg.model.timeout_s) if self.trace else None
        self.session = await build_session(self.cfg, self.root, approver=make_approver(self.console, self.read_line),
                                           on_event=Renderer(self.console), stream=self.cfg.model.streaming,
                                           resume=self.resume, client=client)
        info = describe_session(self.session)
        model = escape(self.cfg.model.model or self.cfg.model.provider)
        self.console.print(f"[bold]agent {__version__}[/] model={model} tools={info['tools']} mode={info['mode']} "
                           f"session={info['session_id']}")
        self.console.print("[dim]Type /help for commands.[/]")

    async def close(self) -> None:
        """Close the session."""
        if self.session is not None:
            await self.session.aclose()

    async def run_turn(self, text: str) -> None:
        """Run one user turn; Ctrl-C cancels it."""
        assert self.session is not None
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(self.session.agent.run(text))
        try:
            loop.add_signal_handler(signal.SIGINT, task.cancel)
            installed = True
        except (NotImplementedError, RuntimeError, ValueError):
            installed = False
        try:
            res = await task
            if res.stopped != "final":
                self.console.print(f"[yellow]({res.stopped})[/]")
        except asyncio.CancelledError:
            self.console.print("[yellow]Interrupted.[/]")
        except AgentError as e:
            self.console.print(f"[red]Error: {escape(str(e))}[/]")
        finally:
            if installed:
                loop.remove_signal_handler(signal.SIGINT)

    async def command(self, name: str, arg: str) -> bool:
        """Handle a slash command. Returns False to exit."""
        s = self.session
        assert s is not None
        c = self.console
        if name in ("exit", "quit"):
            return False
        if name == "help":
            c.print(HELP)
            for cc in self.commands.values():
                c.print(f"  /{cc.name:<20} {escape(cc.description)}")
        elif name == "clear":
            s.agent.messages = s.agent.messages[:1]
            await s.store.event("clear")
            c.print("Conversation cleared.")
        elif name == "compact":
            before = await s.agent.context.total(s.agent.messages, None)
            specs = s.registry.specs(s.cfg.model.schema_level)
            s.agent.messages = await s.agent.context.compact_now(s.agent.messages, specs)
            after = await s.agent.context.total(s.agent.messages, None)
            c.print(f"Compacted: ~{before} -> ~{after} tokens.")
        elif name == "model":
            if not arg:
                c.print(f"provider={s.cfg.model.provider} model={s.cfg.model.model} base_url={s.cfg.model.base_url}")
            else:
                new_model = s.cfg.model.model_copy(update={"model": arg})
                s.cfg.model = new_model
                adapter = build_adapter(new_model, s.client, s.counter)
                s.adapter = adapter
                s.agent.adapter = adapter
                s.agent.context.adapter = adapter
                c.print(f"Model set to {escape(arg)}.")
        elif name == "mcp":
            if s.mcp is None:
                c.print("No MCP servers configured.")
            else:
                t = Table("server", "transport", "status", "tools", "failures", "error")
                for n, st in s.mcp.status().items():
                    t.add_row(n, st["transport"], st["status"], str(st["tools"]), str(st["failures"]),
                              st["error"] or "")
                c.print(t)
        elif name == "permissions":
            if not arg:
                p = s.agent.permissions
                c.print(f"mode={p.mode} allow={[r.tool + (f'({r.arg})' if r.arg else '') for r in p.allow]} "
                        f"deny={[r.tool + (f'({r.arg})' if r.arg else '') for r in p.deny]}")
            elif arg in MODES:
                s.agent.permissions.mode = arg  # type: ignore[assignment]
                c.print(f"Permission mode: {arg}")
            else:
                c.print(f"[red]Unknown mode {escape(arg)}; use one of {', '.join(MODES)}[/]")
        elif name == "resume":
            if not arg:
                ids = SessionStore.list_sessions(s.cfg.context.session_dir)[:10]
                c.print("\n".join(ids) if ids else "No stored sessions.")
            else:
                await self.close()
                self.resume = arg
                await self.start()
                c.print(f"Resumed {escape(arg)} ({len(self.session.agent.messages) if self.session else 0} messages).")
        elif name == "doctor":
            report, ok = await doctor(s.cfg)
            c.print(escape(report))
        elif name == "undo":
            try:
                ref, written, deleted = await Checkpoints(s.workspace.root).restore()
                c.print(f"Restored {ref}: {written} files written, {deleted} removed.")
            except (GitError, FileNotFoundError) as e:
                c.print(f"[red]{escape(str(e))}[/]")
        elif name == "status":
            c.print(describe_session(s))
        elif name == "todos":
            c.print("\n".join(f"[{t['status']}] {t['content']}" for t in s.workspace.todos) or "(no todos)")
        elif name in self.commands:
            await self.run_turn(self.commands[name].expand(arg))
        else:
            c.print(f"[red]Unknown command /{escape(name)}. Type /help.[/]")
        return True

    async def handle_line(self, line: str) -> bool:
        """Dispatch one input line. Returns False to exit."""
        line = line.strip()
        if not line:
            return True
        cmd = split_command(line)
        if cmd is not None:
            return await self.command(*cmd)
        await self.run_turn(line)
        return True

    async def run(self) -> None:
        """Read-eval loop until /exit or EOF."""
        await self.start()
        try:
            while True:
                try:
                    line = await self.read_line("> ")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
                if not await self.handle_line(line):
                    break
        finally:
            await self.close()


def prompt_toolkit_reader(history: str = "~/.agent/history", pt_input: Any = None, pt_output: Any = None) -> ReadLine:
    """Async line reader backed by prompt_toolkit (persistent history)."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory

    hist = Path(history).expanduser()
    hist.parent.mkdir(parents=True, exist_ok=True)
    ps: PromptSession[str] = PromptSession(history=FileHistory(str(hist)), input=pt_input, output=pt_output)

    async def read(prompt: str) -> str:
        return await ps.prompt_async(prompt)
    return read


async def run_print(cfg: AgentConfig, root: Path, prompt: str, fmt: str, resume: str | None,
                    trace: TraceRecorder | None, out: Any = None) -> int:
    """Non-interactive mode."""
    out = out or sys.stdout

    def emit(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        out.flush()

    def on_event(kind: str, data: dict[str, Any]) -> None:
        if fmt != "stream-json":
            return
        if kind == "text_delta":
            emit({"type": "text_delta", "text": data["text"]})
        elif kind == "assistant":
            emit({"type": "assistant", "message": data["message"].model_dump(exclude_none=True)})
        elif kind == "tool_call":
            emit({"type": "tool_call", "call": data["call"].model_dump(exclude_none=True)})
        elif kind == "tool_result":
            emit({"type": "tool_result", "result": data["result"].model_dump(exclude_none=True)})
        elif kind == "notice":
            emit({"type": "notice", "text": data["text"]})

    client = trace.client(cfg.model.timeout_s) if trace else None
    session = await build_session(cfg, root, on_event=on_event, stream=fmt == "stream-json", resume=resume,
                                  client=client)
    try:
        res = await session.agent.run(prompt)
    finally:
        await session.aclose()
    result = {"type": "result", "result": res.text, "session_id": session.store.session_id, "stopped": res.stopped,
              "iterations": res.iterations, "tool_calls": res.tool_calls, "parse_errors": res.parse_errors,
              "usage": {"prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens}}
    if fmt == "text":
        out.write(res.text + "\n")
    else:
        emit(result)
    return 0 if res.stopped == "final" else 1


def build_parser() -> argparse.ArgumentParser:
    """Argument parser."""
    p = argparse.ArgumentParser(prog="agent", description="Model-agnostic terminal coding agent.")
    p.add_argument("--version", action="version", version=f"agent {__version__}")
    p.add_argument("--config", help="path to agent.yaml")
    p.add_argument("--cwd", default=".", help="project root (default: current directory)")
    p.add_argument("-p", "--print", dest="prompt", help="run one task non-interactively and print the result")
    p.add_argument("--output-format", choices=["text", "json", "stream-json"], default="text")
    p.add_argument("--resume", help="resume a stored session id")
    p.add_argument("--permission-mode", choices=MODES)
    p.add_argument("--log-file", help="write JSON logs here")
    p.add_argument("--trace-llm", metavar="FILE", help="dump redacted model HTTP requests/responses (JSONL)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    d = sub.add_parser("doctor", help="check the model configuration")
    d.add_argument("--config", dest="doctor_config")
    d.add_argument("--probe-tools", action="store_true", help="run the 6-check tool-calling probe")
    ss = sub.add_parser("sessions", help="list stored sessions")
    ss.add_argument("--config", dest="sessions_config")
    return p


async def main_async(argv: list[str] | None = None) -> int:
    """CLI implementation."""
    args = build_parser().parse_args(argv)
    try:
        cfg = find_config(getattr(args, "doctor_config", None) or getattr(args, "sessions_config", None)
                          or args.config)
    except ConfigError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.permission_mode:
        cfg = cfg.model_copy(update={"permissions": cfg.permissions.model_copy(update={"mode": args.permission_mode})})
    setup_logging(args.log_file, cfg, verbose=args.verbose)
    trace = TraceRecorder(args.trace_llm, collect_secrets(cfg)) if args.trace_llm else None
    root = Path(args.cwd).resolve()
    if args.cmd == "doctor":
        report, ok = await doctor(cfg, probe=args.probe_tools)
        print(report)
        return 0 if ok else 1
    if args.cmd == "sessions":
        print("\n".join(SessionStore.list_sessions(cfg.context.session_dir)) or "No stored sessions.")
        return 0
    try:
        if args.prompt is not None:
            return await run_print(cfg, root, args.prompt, args.output_format, args.resume, trace)
        repl = Repl(cfg, root, Console(), prompt_toolkit_reader(), resume=args.resume, trace=trace)
        await repl.run()
        return 0
    except (ConfigError, AgentError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """Console-script entry point."""
    try:
        code = asyncio.run(main_async())
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code)


__all__ = ["main", "main_async", "Repl", "run_print", "load_config_dict"]
