"""
bluebox/scripts/run_guide_agent_tui.py

Multi-pane terminal UI for the Guide Agent using Textual.

Layout:
  ┌─────────────────────────────┬──────────────────────┐
  │                             │  Tool Calls History   │
  │       Chat (scrolling)      │                       │
  │                             ├──────────────────────┤
  │  ┌────────────────────────┐ │  Status / Branding    │
  │  │ Input                  │ │                       │
  │  └────────────────────────┘ │                       │
  └─────────────────────────────┴──────────────────────┘

Usage:
    bluebox-guide-tui
    bluebox-guide-tui --model gpt-5.1
    bluebox-guide-tui --cdp-captures-dir ./cdp_captures -q
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from openai import OpenAI
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.widgets import Input, RichLog

from bluebox.agents.guide_agent import GuideAgent
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    ChatRole,
    BaseEmittedMessage,
    ToolInvocationRequestEmittedMessage,
    SuggestedEditEmittedMessage,
    BrowserRecordingRequestEmittedMessage,
    RoutineDiscoveryRequestEmittedMessage,
    RoutineCreationRequestEmittedMessage,
    PendingToolInvocation,
    SuggestedEditRoutine,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.tools.guide_agent_tools import validate_routine
from bluebox.llms.infra.data_store import LocalDiscoveryDataStore
from bluebox.utils.chrome_utils import ensure_chrome_running
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent

# Package root for code_paths
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Browser monitoring constants
PORT = 9222
DEFAULT_CDP_CAPTURES_DIR = Path("./cdp_captures")

# ─── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS: dict[str, str] = {
    "/load": "Load a routine file",
    "/unload": "Unload the current routine",
    "/show": "Show current routine details",
    "/validate": "Validate the current routine",
    "/execute": "Execute the loaded routine",
    "/monitor": "Start browser monitoring",
    "/diff": "Show pending edit diff",
    "/accept": "Accept pending edit",
    "/reject": "Reject pending edit",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/load <file.json>[/cyan]     Load a routine file
  [cyan]/unload[/cyan]               Unload the current routine
  [cyan]/show[/cyan]                 Show current routine details
  [cyan]/validate[/cyan]             Validate the current routine
  [cyan]/execute[/cyan]              Execute the loaded routine
  [cyan]/monitor[/cyan]              Start browser monitoring
  [cyan]/diff[/cyan]                 Show pending edit diff
  [cyan]/accept[/cyan]               Accept pending edit
  [cyan]/reject[/cyan]               Reject pending edit
  [cyan]/status[/cyan]               Show current state
  [cyan]/chats[/cyan]                Show message history
  [cyan]/clear[/cyan]                Clear the chat display
  [cyan]/reset[/cyan]                Start new conversation
  [cyan]/help[/cyan]                 Show this help
  [cyan]/quit[/cyan]                 Exit
"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def safe_parse_routine(routine_str: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """Parse a routine string to dict. Returns (dict, None) or (None, error)."""
    if routine_str is None:
        return None, None
    try:
        return json.loads(routine_str), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


# ─── Textual App ─────────────────────────────────────────────────────────────

class GuideAgentTUI(AbstractAgentTUI):
    """Multi-pane TUI for the Guide Agent."""

    TITLE = "Guide Agent"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: OpenAIModel,
        data_store: LocalDiscoveryDataStore | None = None,
        cdp_captures_dir: Path = DEFAULT_CDP_CAPTURES_DIR,
    ) -> None:
        super().__init__(llm_model)
        self._data_store = data_store
        self._cdp_captures_dir = cdp_captures_dir

        # Guide-specific state
        self._pending_invocation: PendingToolInvocation | None = None
        self._pending_suggested_edit: SuggestedEditRoutine | None = None
        self._loaded_routine_path: Path | None = None

    # ── Abstract implementations ─────────────────────────────────────────

    def _create_agent(self) -> AbstractAgent:
        return GuideAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            data_store=self._data_store,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold magenta]Guide Agent[/bold magenta]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")
        chat.write(Text.from_markup(
            "Welcome! I'll help you create web automation routines.\n"
            "Type [cyan]/help[/cyan] for commands, or just describe what you need."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)

        return (
            f"[bold magenta]VECTORLY[/bold magenta]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]    {self._llm_model.value}\n"
            f"[dim]Messages:[/dim] {msg_count}\n"
            f"[dim]Tools:[/dim]    {self._tool_call_count}\n"
            f"[dim]Context:[/dim]  {ctx_bar}\n"
            f"[dim](est.)     ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]Time:[/dim]     {now}\n"
        )

    # ── Hooks ────────────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        self._pending_invocation = None
        self._pending_suggested_edit = None
        self._loaded_routine_path = None

    def _pre_process_input(self, user_input: str) -> bool:
        """Intercept input when waiting for tool confirmation."""
        if self._pending_invocation:
            self._handle_tool_confirmation(user_input)
            return True
        return False

    def _format_user_label(self) -> str:
        routine_label = ""
        if self._agent and hasattr(self._agent, "routine_state"):
            routine_str = self._agent.routine_state.current_routine_str
            if routine_str:
                rd, _ = safe_parse_routine(routine_str)
                if rd:
                    name = rd.get("name", "")[:20]
                    routine_label = f" [dim]({name})[/dim]"
        return f"[bold green]You{routine_label}>[/bold green]"

    def _handle_additional_message(self, message: BaseEmittedMessage) -> bool:
        chat = self.query_one("#chat-log", RichLog)

        if isinstance(message, ToolInvocationRequestEmittedMessage):
            inv = message.tool_invocation
            self._pending_invocation = inv
            self._tool_call_count += 1

            chat.write(Text.from_markup(
                f"[yellow]\u25b6 Tool:[/yellow] [bold]{inv.tool_name}[/bold] "
                f"[dim]\u2014 y/n to approve[/dim]"
            ))

            ts = datetime.now().strftime("%H:%M:%S")
            self._add_tool_node(
                Text.assemble(
                    (ts, "dim"), " ", ("REQUEST", "yellow"), " ", (inv.tool_name, "bold"),
                ),
                json.dumps(inv.tool_arguments, indent=2).split("\n"),
            )

            inp = self.query_one("#user-input", Input)
            inp.placeholder = "Approve tool call? (y/n)"
            return True

        if isinstance(message, SuggestedEditEmittedMessage):
            self._pending_suggested_edit = message.suggested_edit
            chat.write(Text.from_markup(
                "[bold yellow]Agent suggested a routine edit[/bold yellow]\n"
                "[dim]Use /diff, /accept, or /reject[/dim]"
            ))
            return True

        if isinstance(message, BrowserRecordingRequestEmittedMessage):
            chat.write(Text.from_markup(
                "[yellow]Agent requests browser recording. Use /monitor to start.[/yellow]"
            ))
            return True

        if isinstance(message, RoutineDiscoveryRequestEmittedMessage):
            chat.write(Text.from_markup(
                f"[yellow]Agent requests routine discovery:[/yellow] {message.routine_discovery_task}\n"
                "[dim]Use bluebox-guide for discovery features.[/dim]"
            ))
            return True

        if isinstance(message, RoutineCreationRequestEmittedMessage):
            self._handle_routine_creation(message.created_routine)
            return True

        return False

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        chat = self.query_one("#chat-log", RichLog)

        if cmd == "/show":
            self._show_routine()
            return True
        if cmd == "/validate":
            self._validate_routine()
            return True
        if cmd == "/diff":
            self._show_diff()
            return True
        if cmd == "/accept":
            self._accept_edit()
            return True
        if cmd == "/reject":
            self._reject_edit()
            return True
        if cmd == "/unload":
            self._unload_routine()
            return True
        if cmd == "/monitor":
            chat.write(Text.from_markup(
                "[yellow]Browser monitoring requires the scrolling terminal.[/yellow]\n"
                "[dim]Use bluebox-guide for monitor/discovery features.[/dim]"
            ))
            return True
        if raw_input.startswith("/load "):
            self._load_routine(raw_input[6:].strip())
            return True
        if raw_input.startswith("/execute"):
            params_path = raw_input[8:].strip() or None
            self._execute_routine(params_path)
            return True

        return False

    # ── Send to agent (override to reload routine file) ──────────────────

    @work(thread=True)
    def _send_to_agent(self, user_input: str) -> None:
        self._reload_routine_from_file()
        self._agent.process_new_message(user_input, ChatRole.USER)
        self.call_from_thread(self._update_status)

    # ── Tool confirmation ────────────────────────────────────────────────

    def _handle_tool_confirmation(self, user_input: str) -> None:
        normalized = user_input.strip().lower()
        chat = self.query_one("#chat-log", RichLog)
        inp = self.query_one("#user-input", Input)

        if normalized in ("y", "yes"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            inp.placeholder = "Type a message or /help ..."
            self._confirm_tool(invocation_id)
        elif normalized in ("n", "no"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            inp.placeholder = "Type a message or /help ..."
            self._deny_tool(invocation_id)
        else:
            chat.write(Text.from_markup("[yellow]Please enter 'y' or 'n'[/yellow]"))

    @work(thread=True)
    def _confirm_tool(self, invocation_id: str) -> None:
        self._agent.confirm_tool_invocation(invocation_id)
        self.call_from_thread(self._update_status)

    @work(thread=True)
    def _deny_tool(self, invocation_id: str) -> None:
        self._agent.deny_tool_invocation(invocation_id, reason="User declined")
        self.call_from_thread(self._update_status)

    # ── Routine management ───────────────────────────────────────────────

    def _load_routine(self, file_path: str) -> None:
        chat = self.query_one("#chat-log", RichLog)
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                chat.write(Text.from_markup(f"[red]\u2717 File not found: {file_path}[/red]"))
                return

            with open(path, encoding="utf-8") as f:
                routine_str = f.read()

            self._loaded_routine_path = path
            self._agent.routine_state.update_current_routine(routine_str)

            routine_dict, parse_error = safe_parse_routine(routine_str)
            if routine_dict is not None:
                name = routine_dict.get("name", "N/A")
                ops = len(routine_dict.get("operations", []))
                params = len(routine_dict.get("parameters", []))
                validation = validate_routine(routine_dict)
                valid_str = "[green]\u2713 valid[/green]" if validation.get("valid") else f"[red]\u2717 {validation.get('error', '')}[/red]"
                chat.write(Text.from_markup(
                    f"[green]\u2713 Loaded:[/green] {name}\n"
                    f"  Operations: {ops} | Parameters: {params}\n"
                    f"  {valid_str}\n"
                    f"  [dim]File: {path}[/dim]"
                ))
            else:
                chat.write(Text.from_markup(
                    f"[yellow]\u26a0 Loaded file with invalid JSON: {parse_error}[/yellow]\n"
                    "[dim]Ask the agent for help fixing the JSON.[/dim]"
                ))
            self._update_status()
        except Exception as e:
            chat.write(Text.from_markup(f"[red]\u2717 Error loading routine: {e}[/red]"))

    def _unload_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if self._agent.routine_state.current_routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return
        self._loaded_routine_path = None
        self._agent.routine_state.update_current_routine(None)
        chat.write(Text.from_markup("[yellow]\u2713 Routine unloaded[/yellow]"))
        self._update_status()

    def _show_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return

        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]\u2717 {parse_error}[/red]"))
            return

        name = routine_dict.get("name", "N/A")
        desc = routine_dict.get("description", "N/A")
        ops = routine_dict.get("operations", [])
        params = routine_dict.get("parameters", [])

        lines = [f"[bold cyan]Routine: {name}[/bold cyan]", f"  {desc}", ""]
        if params:
            lines.append("[dim]Parameters:[/dim]")
            for p in params:
                req = "[red]*[/red]" if p.get("required") else " "
                lines.append(f"  {req}{p.get('name', '?')} ({p.get('type', '?')}): {p.get('description', '')}")
            lines.append("")
        if ops:
            lines.append(f"[dim]Operations ({len(ops)}):[/dim]")
            for i, op in enumerate(ops[:10], 1):
                lines.append(f"  {i}. {op.get('type', '?')}")
            if len(ops) > 10:
                lines.append(f"  ... and {len(ops) - 10} more")
        if self._loaded_routine_path:
            lines.append(f"\n[dim]File: {self._loaded_routine_path}[/dim]")
        chat.write(Text.from_markup("\n".join(lines)))

    def _validate_routine(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[yellow]No routine loaded.[/yellow]"))
            return
        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]\u2717 {parse_error}[/red]"))
            return
        result = validate_routine(routine_dict)
        if result.get("valid"):
            chat.write(Text.from_markup(f"[green]\u2713 Valid:[/green] {result.get('message', 'OK')}"))
        else:
            chat.write(Text.from_markup(f"[red]\u2717 Invalid:[/red] {result.get('error', 'Unknown')}"))

    def _show_status_in_chat(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        routine_name = "None"
        if routine_str:
            rd, _ = safe_parse_routine(routine_str)
            if rd:
                routine_name = rd.get("name", "Unnamed")
        msg_count = len(self._agent.get_chats())
        thread_id = self._agent.chat_thread_id[:8] + "..."
        tokens_used, ctx_pct = self._estimate_context_usage()
        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Routine: {routine_name}\n"
            f"  Messages: {msg_count}\n"
            f"  Thread: {thread_id}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  File: {self._loaded_routine_path or 'N/A'}"
        ))

    def _show_diff(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        current_str = self._agent.routine_state.current_routine_str or "{}"
        new_str = json.dumps(self._pending_suggested_edit.routine.model_dump(), indent=2)
        diff = difflib.unified_diff(
            current_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile="current", tofile="suggested", lineterm="",
        )
        diff_lines = list(diff)
        if diff_lines:
            output_lines = []
            for line in diff_lines:
                line = line.rstrip("\n")
                if line.startswith("+"):
                    output_lines.append(f"[green]{escape(line)}[/green]")
                elif line.startswith("-"):
                    output_lines.append(f"[red]{escape(line)}[/red]")
                elif line.startswith("@@"):
                    output_lines.append(f"[cyan]{escape(line)}[/cyan]")
                else:
                    output_lines.append(escape(line))
            chat.write(Text.from_markup("[bold cyan]Suggested Edit Diff:[/bold cyan]\n" + "\n".join(output_lines)))
        else:
            chat.write(Text.from_markup("[yellow]No differences found.[/yellow]"))

    def _accept_edit(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        routine = self._pending_suggested_edit.routine
        routine_dict = routine.model_dump()
        routine_str = json.dumps(routine_dict)
        self._agent.routine_state.update_current_routine(routine_str)
        self._persist_routine(routine_dict)
        chat.write(Text.from_markup("[green]\u2713 Edit accepted and applied[/green]"))
        self._pending_suggested_edit = None
        self._update_status()

    def _reject_edit(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        if not self._pending_suggested_edit:
            chat.write(Text.from_markup("[yellow]No pending suggested edit.[/yellow]"))
            return
        chat.write(Text.from_markup("[yellow]\u2717 Edit rejected[/yellow]"))
        self._pending_suggested_edit = None

    def _persist_routine(self, routine_dict: dict[str, Any]) -> None:
        """Save routine to loaded file path."""
        chat = self.query_one("#chat-log", RichLog)
        if self._loaded_routine_path is None:
            chat.write(Text.from_markup("[yellow]\u26a0 No file loaded \u2014 routine updated in memory only[/yellow]"))
            return
        try:
            new_content = json.dumps(routine_dict, indent=2)
            with open(self._loaded_routine_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            chat.write(Text.from_markup(f"[green]\u2713 Saved to {self._loaded_routine_path}[/green]"))
        except Exception as e:
            chat.write(Text.from_markup(f"[red]\u2717 Failed to save: {e}[/red]"))

    def _execute_routine(self, params_path: str | None) -> None:
        """Execute the loaded routine."""
        chat = self.query_one("#chat-log", RichLog)
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            chat.write(Text.from_markup("[red]\u2717 No routine loaded.[/red]"))
            return
        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            chat.write(Text.from_markup(f"[red]\u2717 Cannot execute: {parse_error}[/red]"))
            return

        params: dict[str, Any] = {}
        if params_path:
            try:
                path = Path(params_path)
                if not path.exists():
                    chat.write(Text.from_markup(f"[red]\u2717 Params file not found: {params_path}[/red]"))
                    return
                with open(path, encoding="utf-8") as f:
                    params = json.load(f)
            except json.JSONDecodeError as e:
                chat.write(Text.from_markup(f"[red]\u2717 Invalid params JSON: {e}[/red]"))
                return

        if not ensure_chrome_running(PORT):
            chat.write(Text.from_markup(f"[red]\u2717 Chrome not running on port {PORT}[/red]"))
            return
        self._run_execution(routine_dict, params)

    @work(thread=True)
    def _run_execution(self, routine_dict: dict[str, Any], params: dict[str, Any]) -> None:
        """Execute routine in background thread."""
        chat = self.query_one("#chat-log", RichLog)
        try:
            routine = Routine(**routine_dict)
            result = routine.execute(params)
            result_dict = result.model_dump()
            self._agent.routine_state.update_last_execution(
                routine=routine_dict, parameters=params, result=result_dict,
            )
            if result.ok:
                self.call_from_thread(
                    lambda: chat.write(Text.from_markup("[green]\u2713 Execution succeeded[/green]"))
                )
            else:
                self.call_from_thread(
                    lambda: chat.write(Text.from_markup(
                        f"[red]\u2717 Execution failed: {result.error or 'Unknown'}[/red]"
                    ))
                )
        except Exception as e:
            self.call_from_thread(
                lambda: chat.write(Text.from_markup(f"[red]\u2717 Execution error: {e}[/red]"))
            )
        self.call_from_thread(self._update_status)

    def _handle_routine_creation(self, routine: Routine | None) -> None:
        """Handle routine creation from agent."""
        chat = self.query_one("#chat-log", RichLog)
        if not routine:
            return
        try:
            safe_name = routine.name.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            routines_dir = Path("./example_data/example_routines")
            routines_dir.mkdir(parents=True, exist_ok=True)
            routine_path = routines_dir / f"{safe_name}.json"
            routine_path.write_text(json.dumps(routine.model_dump(), indent=2))

            routine_str = routine_path.read_text()
            self._loaded_routine_path = routine_path
            self._agent.routine_state.update_current_routine(routine_str)

            chat.write(Text.from_markup(
                f"[green]\u2713 Routine created:[/green] {routine.name}\n"
                f"  Ops: {len(routine.operations)} | Params: {len(routine.parameters)}\n"
                f"  [dim]Saved: {routine_path}[/dim]"
            ))

            system_message = (
                f"[ACTION REQUIRED] Routine '{routine.name}' has been created and saved to {routine_path}. "
                f"It has {len(routine.operations)} operations and {len(routine.parameters)} parameters. "
                "The routine is now loaded into context. Review it using get_current_routine and explain "
                "to the user what it does, what parameters it needs, and how to use it."
            )
            self._agent.process_new_message(system_message, ChatRole.SYSTEM)
            self._update_status()
        except Exception as e:
            chat.write(Text.from_markup(f"[red]\u2717 Failed to create routine: {e}[/red]"))

    def _reload_routine_from_file(self) -> None:
        """Re-read routine from file (picks up external edits)."""
        if self._loaded_routine_path is None:
            return
        try:
            with open(self._loaded_routine_path, encoding="utf-8") as f:
                routine_str = f.read()
            self._agent.routine_state.update_current_routine(routine_str)
        except Exception:
            pass


# ─── Entry point ─────────────────────────────────────────────────────────────

def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the guide agent TUI."""
    parser = argparse.ArgumentParser(description="Guide Agent \u2014 Multi-pane TUI")
    parser.add_argument(
        "--model", type=str, default=OpenAIModel.GPT_5_1.value,
        help=f"LLM model to use (default: {OpenAIModel.GPT_5_1.value})",
    )
    parser.add_argument(
        "--cdp-captures-dir", type=str, default=None,
        help="Path to CDP captures directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./guide_agent_output",
        help="Output directory for temporary files",
    )
    parser.add_argument(
        "--docs-dir", type=str,
        default=str(BLUEBOX_PACKAGE_ROOT / "agent_docs"),
        help="Documentation directory",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

    if Config.OPENAI_API_KEY is None:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable is not set[/bold red]")
        sys.exit(1)

    data_store: LocalDiscoveryDataStore | None = None

    try:
        llm_model = parse_model(args.model)
        openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        data_store_kwargs: dict[str, Any] = {
            "client": openai_client,
            "documentation_paths": [args.docs_dir],
            "code_paths": [
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
                str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
                str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
                str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
                "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
            ],
        }

        if args.cdp_captures_dir:
            data_store_kwargs.update({
                "cdp_captures_dir": args.cdp_captures_dir,
                "tmp_dir": str(Path(args.output_dir) / "tmp"),
            })

        with console.status("[bold blue]Initializing...[/bold blue]") as status:
            status.update("[bold blue]Creating data store...[/bold blue]")
            data_store = LocalDiscoveryDataStore(**data_store_kwargs)

            if args.cdp_captures_dir:
                status.update("[bold blue]Creating CDP captures vectorstore...[/bold blue]")
                data_store.make_cdp_captures_vectorstore()

            status.update("[bold blue]Creating documentation vectorstore...[/bold blue]")
            data_store.make_documentation_vectorstore()

        console.print("[green]\u2713 Vectorstores ready![/green]")
        console.print()

        cdp_captures_dir = Path(args.cdp_captures_dir) if args.cdp_captures_dir else DEFAULT_CDP_CAPTURES_DIR

        # Redirect logging + stderr AFTER all console output, right before TUI takes over.
        enable_tui_logging(log_file=args.log_file or ".bluebox_tui.log", quiet=args.quiet)

        app = GuideAgentTUI(
            llm_model=llm_model,
            data_store=data_store,
            cdp_captures_dir=cdp_captures_dir,
        )
        app.run()

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        sys.exit(1)
    finally:
        if data_store is not None:
            console.print()
            with console.status("[dim]Cleaning up vectorstores...[/dim]"):
                try:
                    data_store.clean_up()
                except KeyboardInterrupt:
                    console.print("[yellow]Cleanup interrupted[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Cleanup failed: {e}[/yellow]")
            console.print("[green]\u2713 Cleanup complete![/green]")


if __name__ == "__main__":
    main()
