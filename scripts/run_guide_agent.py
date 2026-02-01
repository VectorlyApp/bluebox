#!/usr/bin/env python3
"""
scripts/run_guide_agent.py

Interactive terminal interface for the Guide Agent.
Guides users through creating web automation routines.

Commands:
  /load <routine.json>     Load a routine file (auto-reloads on edits)
  /unload                  Unload the current routine
  /show                    Show current routine details
  /validate                Validate the current routine
  /execute [params.json]   Execute the loaded routine
  /monitor                 Start browser monitoring session
  /diff                    Show pending suggested edit diff
  /accept                  Accept pending suggested edit
  /reject                  Reject pending suggested edit
  /status                  Show current state
  /chats                   Show all messages in the thread
  /reset                   Start a new conversation
  /help                    Show help
  /quit                    Exit
"""

import argparse
import asyncio
import difflib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

# Package root for code_paths (scripts/ is sibling to bluebox/)
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "bluebox"
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bluebox.agents.guide_agent import GuideAgent
from bluebox.config import Config
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.llms.interaction import (
    EmittedMessageType,
    ChatRole,
    EmittedMessage,
    BaseEmittedMessage,
    ChatResponseEmittedMessage,
    ToolInvocationRequestEmittedMessage,
    ToolInvocationResultEmittedMessage,
    SuggestedEditEmittedMessage,
    BrowserRecordingRequestEmittedMessage,
    RoutineDiscoveryRequestEmittedMessage,
    RoutineCreationRequestEmittedMessage,
    ErrorEmittedMessage,
    PendingToolInvocation,
    SuggestedEditRoutine,
    ToolInvocationStatus,
)
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.tools.guide_agent_tools import validate_routine
from bluebox.llms.infra.data_store import DiscoveryDataStore, LocalDiscoveryDataStore
from bluebox.data_models.routine_discovery.message import RoutineDiscoveryMessage, RoutineDiscoveryMessageType
from bluebox.sdk import BrowserMonitor
from bluebox.sdk.discovery import RoutineDiscovery
from bluebox.utils.chrome_utils import ensure_chrome_running
from bluebox.utils.terminal_utils import ask_yes_no


console = Console()

# Browser monitoring constants
PORT = 9222
REMOTE_DEBUGGING_ADDRESS = f"http://127.0.0.1:{PORT}"
DEFAULT_CDP_CAPTURES_DIR = Path("./cdp_captures")


def configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """
    Configure logging for all bluebox modules.

    Args:
        quiet: If True, suppress all logs to console.
        log_file: If provided, write logs to this file instead of console.
    """
    # Get the parent logger for all bluebox modules
    wh_logger = logging.getLogger("bluebox")

    if quiet:
        # Suppress all logs
        wh_logger.setLevel(logging.CRITICAL + 1)
        return

    if log_file:
        # Remove existing handlers and add file handler
        wh_logger.handlers.clear()
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s:%(name)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        wh_logger.addHandler(file_handler)
        wh_logger.propagate = False


BANNER = """
[bold magenta]╔══════════════════════════════════════════════════════════════════════════════════════════╗
║   ██████╗ ██╗   ██╗██╗██████╗ ███████╗     █████╗  ██████╗ ███████╗███╗   ██╗████████╗   ║
║  ██╔════╝ ██║   ██║██║██╔══██╗██╔════╝    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝   ║
║  ██║  ███╗██║   ██║██║██║  ██║█████╗      ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║      ║
║  ██║   ██║██║   ██║██║██║  ██║██╔══╝      ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║      ║
║  ╚██████╔╝╚██████╔╝██║██████╔╝███████╗    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║      ║
║   ╚═════╝  ╚═════╝ ╚═╝╚═════╝ ╚══════╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝      ║
║                                                                                          ║
║[/bold magenta][dim]                                  powered by vectorly                                     [/dim][bold magenta]║
╚══════════════════════════════════════════════════════════════════════════════════════════╝[/bold magenta]
"""


def print_welcome(model: str) -> None:
    """Print welcome message and help."""
    console.print(BANNER)
    console.print()
    console.print(Panel(
        r"""[bold]Welcome![/bold] I'll help you create web automation routines from your
CDP (Chrome DevTools Protocol) captures.

I'll analyze your network transactions to identify relevant API
endpoints, required cookies, headers, and request patterns that
can be turned into a reusable routine.

[bold]Commands:[/bold]
  [cyan]/load <routine.json>[/cyan]     Load a routine file (auto-reloads on edits)
  [cyan]/unload[/cyan]                  Unload the current routine
  [cyan]/show[/cyan]                    Show current routine details
  [cyan]/validate[/cyan]                Validate the current routine
  [cyan]/execute \[params.json][/cyan]   Execute the loaded routine
  [cyan]/monitor[/cyan]                 Start browser monitoring session
  [cyan]/diff[/cyan]                    Show pending suggested edit diff
  [cyan]/accept[/cyan]                  Accept pending suggested edit
  [cyan]/reject[/cyan]                  Reject pending suggested edit
  [cyan]/status[/cyan]                  Show current state
  [cyan]/chats[/cyan]                   Show all messages in the thread
  [cyan]/reset[/cyan]                   Start a new conversation
  [cyan]/help[/cyan]                    Show all commands
  [cyan]/quit[/cyan]                    Exit

[bold]Links:[/bold]
  [link=https://vectorly.app/docs]https://vectorly.app/docs[/link]
  [link=https://console.vectorly.app]https://console.vectorly.app[/link]""",
        title="[bold magenta]Guide Agent[/bold magenta]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()


def print_tool_request(invocation: PendingToolInvocation) -> None:
    """Print a tool invocation request with formatted arguments."""
    args_formatted = json.dumps(invocation.tool_arguments, indent=2)

    content = Text()
    content.append("Tool: ", style="dim")
    content.append(invocation.tool_name, style="bold white")
    content.append("\n\n")
    content.append("Arguments:\n", style="dim")
    content.append(args_formatted, style="white")

    console.print()
    console.print(Panel(
        content,
        title="[bold yellow]TOOL INVOCATION REQUEST[/bold yellow]",
        style="yellow",
        box=box.ROUNDED,
    ))
    console.print()
    console.print("[bold yellow]Do you want to proceed?[/bold yellow] [dim][y/n][/dim] ", end="")


def print_tool_result(
    invocation: PendingToolInvocation,
    result: dict[str, Any] | None,
    error: str | None,
) -> None:
    """Print a tool invocation result."""
    console.print()

    if invocation.status == ToolInvocationStatus.DENIED:
        console.print("[yellow]✗ Tool invocation denied[/yellow]")

    elif invocation.status == ToolInvocationStatus.EXECUTED:
        console.print("[bold green]✓ Tool executed successfully[/bold green]")
        if result:
            result_json = json.dumps(result, indent=2)
            # Limit display
            lines = result_json.split("\n")
            if len(lines) > 15:
                display = "\n".join(lines[:15]) + f"\n... ({len(lines) - 15} more lines)"
            else:
                display = result_json
            console.print()
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]✗ Tool execution failed[/bold red]")
        if error:
            console.print()
            console.print(Panel(error, title="Error", style="red", box=box.ROUNDED))

    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold cyan]Assistant[/bold cyan]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str) -> None:
    """Print an error message."""
    console.print()
    console.print(f"[bold red]⚠ Error:[/bold red] [red]{escape(error)}[/red]")
    console.print()


def safe_parse_routine(routine_str: str | None) -> tuple[dict[str, Any] | None, str | None]:
    """
    Safely parse a routine string to dict.

    Returns:
        Tuple of (parsed_dict, error_message).
        If parsing succeeds: (dict, None)
        If parsing fails: (None, error_message)
        If routine_str is None: (None, None)
    """
    if routine_str is None:
        return None, None
    try:
        return json.loads(routine_str), None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def print_routine_info(routine_json: dict[str, Any], validation_result: dict | None = None) -> None:
    """Print routine info in a nice table."""
    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Field", style="dim")
    table.add_column("Value", style="white")

    table.add_row("Name", routine_json.get("name", "N/A"))
    table.add_row("Description", routine_json.get("description", "N/A"))
    table.add_row("Operations", str(len(routine_json.get("operations", []))))
    table.add_row("Parameters", str(len(routine_json.get("parameters", []))))

    # Add parameter names if any
    params = routine_json.get("parameters", [])
    if params:
        param_names = ", ".join(p.get("name", "?") for p in params)
        table.add_row("Param Names", param_names)

    console.print()
    console.print("[bold green]✓ Routine loaded[/bold green]")
    console.print()
    console.print(table)

    # Show validation result
    if validation_result:
        if validation_result.get("valid"):
            console.print(f"[green]✓ Valid[/green]")
        else:
            console.print(f"[red]✗ Invalid: {validation_result.get('error', 'Unknown error')}[/red]")

    console.print()


def print_execution_result(result_dict: dict[str, Any], ok: bool, error: str | None) -> None:
    """Print routine execution result."""
    console.print()

    if ok:
        console.print("[bold green]✓ Execution succeeded[/bold green]")
    else:
        console.print("[bold red]✗ Execution failed[/bold red]")
        if error:
            console.print(f"[red]  Error: {escape(error)}[/red]")

    # Show output preview
    output_preview = json.dumps(result_dict, indent=2)
    lines = output_preview.split("\n")

    if len(lines) > 20:
        display = "\n".join(lines[:20]) + f"\n[dim]... ({len(lines) - 20} more lines)[/dim]"
    else:
        display = output_preview

    console.print()
    console.print(Panel(
        display,
        title="Output Preview",
        style="green" if ok else "red",
        box=box.ROUNDED,
    ))
    console.print()


class TerminalGuideChat:
    """Interactive terminal chat interface for the Guide Agent."""

    def __init__(
        self,
        llm_model: OpenAIModel | None = None,
        data_store: DiscoveryDataStore | None = None,
        cdp_captures_dir: Path | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._pending_invocation: PendingToolInvocation | None = None
        self._pending_suggested_edit: SuggestedEditRoutine | None = None
        self._streaming_started: bool = False
        self._data_store = data_store
        self._loaded_routine_path: Path | None = None
        self._last_execution_ok: bool | None = None
        self._browser_recording_requested: bool = False
        self._routine_discovery_requested: bool = False
        self._routine_discovery_task: str | None = None
        self._routine_creation_requested: bool = False
        self._created_routine: Routine | None = None
        self._cdp_captures_dir: Path = cdp_captures_dir or DEFAULT_CDP_CAPTURES_DIR
        self._agent = GuideAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model if llm_model else OpenAIModel.GPT_5_1,
            data_store=data_store,
        )

    def _persist_routine(self, routine_dict: dict[str, Any]) -> None:
        """
        Persist the routine to the loaded file path.

        Called when user approves the update_routine tool call.
        Shows a diff and overwrites the file with the new routine JSON.
        """
        if self._loaded_routine_path is None:
            console.print()
            console.print("[yellow]⚠ No file loaded - routine updated in memory only[/yellow]")
            console.print()
            return

        try:
            # Read old content for diff
            old_content = ""
            if self._loaded_routine_path.exists():
                with open(self._loaded_routine_path, encoding="utf-8") as f:
                    old_content = f.read()

            # Format new content
            new_content = json.dumps(routine_dict, indent=2)

            # Generate and display diff
            old_lines = old_content.splitlines(keepends=True)
            new_lines = new_content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=str(self._loaded_routine_path),
                tofile=str(self._loaded_routine_path),
                lineterm="",
            )

            diff_lines = list(diff)
            if diff_lines:
                console.print()
                console.print("[bold cyan]Diff:[/bold cyan]")
                for line in diff_lines:
                    line = line.rstrip("\n")
                    if line.startswith("+++") or line.startswith("---"):
                        console.print(f"[bold]{line}[/bold]")
                    elif line.startswith("@@"):
                        console.print(f"[cyan]{line}[/cyan]")
                    elif line.startswith("+"):
                        console.print(f"[green]{line}[/green]")
                    elif line.startswith("-"):
                        console.print(f"[red]{line}[/red]")
                    else:
                        console.print(line)
                console.print()

            # Write new content
            with open(self._loaded_routine_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            console.print(f"[green]✓ Routine saved to {self._loaded_routine_path}[/green]")
            console.print()

        except Exception as e:
            console.print()
            console.print(f"[red]✗ Failed to save routine: {e}[/red]")
            console.print()

    def _get_prompt(self) -> str:
        """Get the input prompt with routine name if loaded."""
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str:
            routine_dict, _ = safe_parse_routine(routine_str)
            if routine_dict:
                name = routine_dict.get("name", "routine")
                # Truncate long names
                if len(name) > 20:
                    name = name[:17] + "..."
                return f"[bold green]You[/bold green] [dim]({name})[/dim][bold green]>[/bold green] "
            else:
                return f"[bold green]You[/bold green] [dim](invalid json)[/dim][bold green]>[/bold green] "
        return "[bold green]You>[/bold green] "

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold cyan]Assistant[/bold cyan]")
            console.print()
            self._streaming_started = True

        # Use plain print for streaming - Rich console.print breaks on char-by-char output
        print(chunk, end="", flush=True)

    def _handle_message(self, message: BaseEmittedMessage) -> None:
        """Handle messages emitted by the Guide Agent."""
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()  # End the streamed line
                print()  # Add spacing
                self._streaming_started = False
            else:
                print_assistant_message(message.content)

        elif isinstance(message, ToolInvocationRequestEmittedMessage):
            self._pending_invocation = message.tool_invocation
            print_tool_request(message.tool_invocation)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            # Check if result contains an error
            error = message.tool_result.get("error") if isinstance(message.tool_result, dict) else None
            print_tool_result(
                message.tool_invocation,
                message.tool_result,
                error,
            )

        elif isinstance(message, SuggestedEditEmittedMessage):
            self._pending_suggested_edit = message.suggested_edit
            console.print()
            console.print("[bold yellow]📝 Agent suggested a routine edit[/bold yellow]")
            console.print("[dim]Use /diff to see changes, /accept to apply, /reject to discard[/dim]")
            console.print()

        elif isinstance(message, BrowserRecordingRequestEmittedMessage):
            self._browser_recording_requested = True

        elif isinstance(message, RoutineDiscoveryRequestEmittedMessage):
            self._routine_discovery_requested = True
            self._routine_discovery_task = message.routine_discovery_task

        elif isinstance(message, RoutineCreationRequestEmittedMessage):
            self._routine_creation_requested = True
            self._created_routine = message.created_routine

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)

    def _handle_tool_confirmation(self, user_input: str) -> bool:
        """Handle yes/no confirmation for pending tool invocation."""
        if not self._pending_invocation:
            return False

        normalized = user_input.strip().lower()

        if normalized in ("y", "yes"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            self._agent.confirm_tool_invocation(invocation_id)
            return True

        elif normalized in ("n", "no"):
            invocation_id = self._pending_invocation.invocation_id
            self._pending_invocation = None
            self._agent.deny_tool_invocation(invocation_id, reason="User declined")
            return True

        else:
            console.print("[yellow]Please enter 'y' or 'n': [/yellow]", end="")
            return True  # Still in confirmation mode

    def _handle_load_command(self, file_path: str) -> None:
        """Handle /load command to load a routine JSON file."""
        try:
            path = Path(file_path).resolve()
            if not path.exists():
                console.print()
                console.print(f"[red]✗ File not found: {file_path}[/red]")
                console.print()
                return

            # Load as raw string (preserves content even if JSON is invalid)
            with open(path, encoding="utf-8") as f:
                routine_str = f.read()

            self._loaded_routine_path = path
            self._agent.routine_state.update_current_routine(routine_str)

            # Try to parse and validate
            routine_dict, parse_error = safe_parse_routine(routine_str)
            if routine_dict is not None:
                validation_result = validate_routine(routine_dict)
                print_routine_info(routine_dict, validation_result)
            else:
                console.print()
                console.print(f"[yellow]⚠ Loaded file with invalid JSON: {parse_error}[/yellow]")
                console.print("[dim]Ask the agent for help fixing the JSON.[/dim]")

            console.print(f"[dim]Watching: {path}[/dim]")
            console.print()

        except Exception as e:
            console.print()
            console.print(f"[red]✗ Error loading routine: {e}[/red]")
            console.print()

    def _handle_unload_command(self) -> None:
        """Handle /unload command to clear the current routine."""
        if self._agent.routine_state.current_routine_str is None:
            console.print()
            console.print("[yellow]No routine loaded.[/yellow]")
            console.print()
            return

        self._loaded_routine_path = None
        self._agent.routine_state.update_current_routine(None)
        console.print()
        console.print("[yellow]✓ Routine unloaded[/yellow]")
        console.print()

    def _handle_show_command(self) -> None:
        """Handle /show command to display current routine details."""
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            console.print()
            console.print("[yellow]No routine loaded. Use /load <file.json> first.[/yellow]")
            console.print()
            return

        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            console.print()
            console.print(f"[red]✗ {parse_error}[/red]")
            console.print("[dim]Raw content:[/dim]")
            console.print(routine_str[:500] + ("..." if len(routine_str) > 500 else ""))
            console.print()
            return

        # Build detailed view
        table = Table(box=box.ROUNDED, show_header=False, expand=True)
        table.add_column("Field", style="dim", width=15)
        table.add_column("Value", style="white")

        table.add_row("Name", routine_dict.get("name", "N/A"))
        table.add_row("Description", routine_dict.get("description", "N/A"))

        # Parameters
        params = routine_dict.get("parameters", [])
        if params:
            param_lines = []
            for p in params:
                req = "[red]*[/red]" if p.get("required", False) else ""
                param_lines.append(f"  {req}{p.get('name', '?')} [dim]({p.get('type', '?')})[/dim]: {p.get('description', '')}")
            table.add_row("Parameters", "\n".join(param_lines))
        else:
            table.add_row("Parameters", "[dim]None[/dim]")

        # Operations summary
        ops = routine_dict.get("operations", [])
        if ops:
            op_lines = []
            for i, op in enumerate(ops, 1):
                op_type = op.get("type", "?")
                op_lines.append(f"  {i}. {op_type}")
            table.add_row("Operations", "\n".join(op_lines[:10]))
            if len(ops) > 10:
                table.add_row("", f"[dim]... and {len(ops) - 10} more[/dim]")

        console.print()
        console.print(Panel(table, title="[bold cyan]Current Routine[/bold cyan]", border_style="cyan", box=box.ROUNDED))

        if self._loaded_routine_path:
            console.print(f"[dim]File: {self._loaded_routine_path}[/dim]")
        console.print()

    def _handle_validate_command(self) -> None:
        """Handle /validate command to validate the current routine."""
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            console.print()
            console.print("[yellow]No routine loaded. Use /load <file.json> first.[/yellow]")
            console.print()
            return

        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            console.print()
            console.print(f"[bold red]✗ {parse_error}[/bold red]")
            console.print()
            return

        result = validate_routine(routine_dict)
        console.print()
        if result.get("valid"):
            console.print(f"[bold green]✓ Valid:[/bold green] {result.get('message', 'Routine is valid')}")
        else:
            console.print(f"[bold red]✗ Invalid[/bold red]")
            console.print()
            error = result.get("error", "Unknown error")
            # Format error nicely
            console.print(Panel(error, title="Validation Error", style="red", box=box.ROUNDED))
        console.print()

    def _handle_chats_command(self) -> None:
        """Handle /chats command to show all messages in the thread."""
        chats = self._agent.get_chats()
        if not chats:
            console.print()
            console.print("[yellow]No messages in thread yet.[/yellow]")
            console.print()
            return

        console.print()
        console.print(f"[bold cyan]Chat History ({len(chats)} messages)[/bold cyan]")
        console.print()

        role_styles = {"user": "green", "assistant": "cyan", "system": "yellow", "tool": "magenta"}

        for i, chat in enumerate(chats, 1):
            role_style = role_styles.get(chat.role.value, "white")

            # TOOL role = tool result
            if chat.role.value == "tool":
                content = chat.content.replace("\n", " ")[:50]
                tool_id = chat.tool_call_id[:8] + "..." if chat.tool_call_id else "?"
                console.print(f"[dim]{i}.[/dim] [{role_style}]TOOL_RESULT[/{role_style}] [dim]({tool_id})[/dim] {escape(content)}...")
                continue

            # ASSISTANT with tool_calls = tool request
            if chat.role.value == "assistant" and chat.tool_calls:
                tool_names = ", ".join(tc.tool_name for tc in chat.tool_calls)
                content = chat.content.replace("\n", " ")[:30] if chat.content else ""
                prefix = f"{escape(content)}... " if content else ""
                console.print(f"[dim]{i}.[/dim] [{role_style}]ASSISTANT[/{role_style}] {prefix}[yellow]→ {tool_names}[/yellow]")
                continue

            # Regular message - single line, ~50 chars
            content = chat.content.replace("\n", " ")[:50]
            suffix = "..." if len(chat.content) > 50 else ""
            console.print(f"[dim]{i}.[/dim] [{role_style}]{chat.role.value.upper()}[/{role_style}] {escape(content)}{suffix}")

        console.print()

    def _handle_status_command(self) -> None:
        """Handle /status command to show current state."""
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Field", style="dim")
        table.add_column("Value", style="white")

        # Routine status
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str:
            routine_dict, parse_error = safe_parse_routine(routine_str)
            if routine_dict:
                table.add_row("Routine", f"[green]{routine_dict.get('name', 'Unnamed')}[/green]")
                table.add_row("Operations", str(len(routine_dict.get("operations", []))))
                table.add_row("Parameters", str(len(routine_dict.get("parameters", []))))
            else:
                table.add_row("Routine", f"[yellow]Invalid JSON[/yellow]")
        else:
            table.add_row("Routine", "[dim]None loaded[/dim]")

        # File path
        if self._loaded_routine_path:
            table.add_row("File", str(self._loaded_routine_path))
        else:
            table.add_row("File", "[dim]N/A[/dim]")

        # Last execution
        last_result = self._agent.routine_state.last_execution_result
        if last_result:
            ok = last_result.get("ok", False)
            status = "[green]Success[/green]" if ok else "[red]Failed[/red]"
            table.add_row("Last Execution", status)
            if not ok and last_result.get("error"):
                table.add_row("Last Error", f"[red]{last_result.get('error')[:50]}...[/red]")
        else:
            table.add_row("Last Execution", "[dim]None[/dim]")

        # Thread info
        table.add_row("Thread ID", self._agent.chat_thread_id[:8] + "...")
        table.add_row("Messages", str(len(self._agent.get_chats())))

        console.print()
        console.print(Panel(table, title="[bold cyan]Status[/bold cyan]", border_style="cyan", box=box.ROUNDED))
        console.print()

    def _reload_routine_from_file(self) -> None:
        """Re-read the routine from the loaded file path if set."""
        if self._loaded_routine_path is None:
            return
        try:
            with open(self._loaded_routine_path, encoding="utf-8") as f:
                routine_str = f.read()
            self._agent.routine_state.update_current_routine(routine_str)
        except Exception:
            pass  # Silently ignore reload errors

    def _prompt_for_parameters(self, routine_json: dict[str, Any]) -> dict[str, Any] | None:
        """Prompt user for required parameters interactively."""
        params = routine_json.get("parameters", [])
        required_params = [p for p in params if p.get("required", False)]

        if not required_params:
            return {}

        console.print()
        console.print("[bold cyan]Enter parameter values[/bold cyan] [dim](press Enter to skip optional)[/dim]")
        console.print()

        values: dict[str, Any] = {}

        for param in params:
            name = param.get("name", "unknown")
            desc = param.get("description", "")
            param_type = param.get("type", "string")
            required = param.get("required", False)

            req_marker = "[red]*[/red]" if required else ""
            prompt = f"  {req_marker}[cyan]{name}[/cyan] [dim]({param_type})[/dim]"
            if desc:
                prompt += f" - {desc}"
            prompt += ": "

            try:
                value = console.input(prompt)
                if value.strip():
                    # Try to parse JSON for complex types
                    if param_type in ("object", "array"):
                        try:
                            values[name] = json.loads(value)
                        except json.JSONDecodeError:
                            values[name] = value
                    elif param_type == "number":
                        try:
                            values[name] = float(value)
                        except ValueError:
                            values[name] = value
                    elif param_type == "boolean":
                        values[name] = value.lower() in ("true", "1", "yes")
                    else:
                        values[name] = value
                elif required:
                    console.print(f"[red]  ✗ {name} is required[/red]")
                    return None
            except KeyboardInterrupt:
                console.print()
                console.print("[yellow]Cancelled[/yellow]")
                return None

        console.print()
        return values

    def _handle_execute_command(self, params_path: str | None) -> None:
        """Handle /execute command to execute the loaded routine."""
        routine_str = self._agent.routine_state.current_routine_str
        if routine_str is None:
            console.print()
            console.print("[red]✗ No routine loaded. Use /load <file.json> first.[/red]")
            console.print()
            return

        # Parse routine
        routine_dict, parse_error = safe_parse_routine(routine_str)
        if routine_dict is None:
            console.print()
            console.print(f"[red]✗ Cannot execute: {parse_error}[/red]")
            console.print()
            return

        # Load parameters
        params: dict[str, Any] = {}
        if params_path:
            try:
                path = Path(params_path)
                if not path.exists():
                    console.print()
                    console.print(f"[red]✗ Parameters file not found: {params_path}[/red]")
                    console.print()
                    return
                with open(path, encoding="utf-8") as f:
                    params = json.load(f)
            except json.JSONDecodeError as e:
                console.print()
                console.print(f"[red]✗ Invalid parameters JSON: {e}[/red]")
                console.print()
                return
        else:
            # Check if routine has required parameters
            routine_params = routine_dict.get("parameters", [])
            has_required = any(p.get("required", False) for p in routine_params)

            if has_required:
                params = self._prompt_for_parameters(routine_dict)
                if params is None:
                    return  # User cancelled

        # Ensure Chrome is running in debug mode
        if not ensure_chrome_running(PORT):
            console.print()
            console.print("[red]✗ Could not start Chrome in debug mode.[/red]")
            console.print(f"[dim]Launch Chrome manually with: --remote-debugging-port={PORT}[/dim]")
            console.print()
            return

        try:
            console.print()
            with console.status("[bold yellow]Executing routine...[/bold yellow]"):
                # Create and execute routine
                routine = Routine(**routine_dict)
                result = routine.execute(params)
                result_dict = result.model_dump()

            # Update agent state with execution result
            self._agent.routine_state.update_last_execution(
                routine=routine_dict,
                parameters=params,
                result=result_dict,
            )
            self._last_execution_ok = result.ok

            print_execution_result(result_dict, result.ok, result.error)

        except Exception as e:
            console.print()
            console.print(f"[bold red]✗ Execution error: {e}[/bold red]")
            console.print()

    def _handle_diff_command(self) -> None:
        """Handle /diff command to show pending suggested edit."""
        if not self._pending_suggested_edit:
            console.print()
            console.print("[yellow]No pending suggested edit.[/yellow]")
            console.print()
            return

        # Get current routine
        current_str = self._agent.routine_state.current_routine_str or "{}"
        # Serialize Routine object to JSON for diff
        new_str = json.dumps(self._pending_suggested_edit.routine.model_dump(), indent=2)

        # Generate unified diff
        current_lines = current_str.splitlines(keepends=True)
        new_lines = new_str.splitlines(keepends=True)
        diff = difflib.unified_diff(
            current_lines,
            new_lines,
            fromfile="current",
            tofile="suggested",
            lineterm="",
        )

        diff_lines = list(diff)
        if diff_lines:
            console.print()
            console.print("[bold cyan]Suggested Edit Diff:[/bold cyan]")
            for line in diff_lines:
                line = line.rstrip("\n")
                if line.startswith("+++") or line.startswith("---"):
                    console.print(f"[bold]{line}[/bold]")
                elif line.startswith("@@"):
                    console.print(f"[cyan]{line}[/cyan]")
                elif line.startswith("+"):
                    console.print(f"[green]{line}[/green]")
                elif line.startswith("-"):
                    console.print(f"[red]{line}[/red]")
                else:
                    console.print(line)
            console.print()
        else:
            console.print()
            console.print("[yellow]No differences found.[/yellow]")
            console.print()

    def _handle_accept_command(self) -> None:
        """Handle /accept command to approve pending edit."""
        if not self._pending_suggested_edit:
            console.print()
            console.print("[yellow]No pending suggested edit to accept.[/yellow]")
            console.print()
            return

        # Get routine from pending edit
        routine = self._pending_suggested_edit.routine
        routine_dict = routine.model_dump()
        routine_str = json.dumps(routine_dict)

        # Update agent's routine state
        self._agent.routine_state.update_current_routine(routine_str)

        # Persist to file (reuses existing _persist_routine method)
        self._persist_routine(routine_dict)

        console.print()
        console.print("[bold green]✓ Edit accepted and applied[/bold green]")
        console.print()
        self._pending_suggested_edit = None

    def _handle_reject_command(self) -> None:
        """Handle /reject command to reject pending edit."""
        if not self._pending_suggested_edit:
            console.print()
            console.print("[yellow]No pending suggested edit to reject.[/yellow]")
            console.print()
            return

        console.print()
        console.print("[yellow]✗ Edit rejected[/yellow]")
        console.print()
        self._pending_suggested_edit = None

    def _handle_browser_recording(self, skip_prompt: bool = False) -> None:
        """Handle a browser recording request."""
        if not skip_prompt and not ask_yes_no("Start browser monitoring?"):
            self._agent.notify_browser_recording_result(accepted=False)
            return

        # Ensure Chrome is running in debug mode (launch if needed)
        if not ensure_chrome_running(PORT):
            console.print()
            console.print("[red]✗ Could not start Chrome in debug mode.[/red]")
            console.print(f"[dim]Launch Chrome manually with: --remote-debugging-port={PORT}[/dim]")
            console.print()
            self._agent.notify_browser_recording_result(
                accepted=True,
                error="Could not start Chrome in debug mode"
            )
            return

        cdp_captures_dir = self._cdp_captures_dir
        cdp_captures_dir.mkdir(parents=True, exist_ok=True)

        console.print()
        console.print("[bold blue]Starting browser monitor...[/bold blue]")
        console.print(f"[dim]Output directory: {cdp_captures_dir}[/dim]")
        console.print()

        monitor = BrowserMonitor(
            remote_debugging_address=REMOTE_DEBUGGING_ADDRESS,
            output_dir=str(cdp_captures_dir),
            create_tab=False,
        )

        # Run monitor in async context so the event loop stays active
        async def run_monitor() -> dict:
            await monitor.astart()
            console.print("[green]Monitoring started! Perform your actions in the browser.[/green]")
            console.print("[yellow]Press Ctrl+C when done...[/yellow]")
            console.print()

            try:
                while monitor.is_alive:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass

            return await monitor.astop()

        summary: dict = {}
        try:
            summary = asyncio.run(run_monitor())
        except KeyboardInterrupt:
            console.print()
            console.print("Stopping monitor...")
            # Need to stop in a new event loop since the previous was interrupted
            try:
                summary = asyncio.run(monitor.astop())
            except Exception:
                pass

        console.print()
        console.print("[bold green]✓ Monitoring complete![/bold green]")
        transaction_count = summary.get('network', {}).get('completed_transactions', 0) if summary else 0
        if summary:
            console.print(f"[dim]Duration: {summary.get('duration', 0):.1f}s[/dim]")
            console.print(f"[dim]Transactions captured: {transaction_count}[/dim]")
        console.print()

        # Create vectorstore from captured data if we have transactions
        if transaction_count == 0:
            console.print("[yellow]⚠ No transactions captured. Skipping vectorstore creation.[/yellow]")
            console.print()
            self._agent.notify_browser_recording_result(
                accepted=True,
                error="No network transactions were captured during recording"
            )
            return

        if not isinstance(self._data_store, LocalDiscoveryDataStore):
            console.print("[yellow]⚠ No data store available. Skipping vectorstore creation.[/yellow]")
            console.print()
            self._agent.notify_browser_recording_result(
                accepted=True,
                error="No data store available"
            )
            return

        # Update data store with CDP captures directory (events.jsonl format)
        # The model validator will set up the processed output paths automatically
        self._data_store.cdp_captures_dir = str(cdp_captures_dir)
        self._data_store.tmp_dir = str(cdp_captures_dir / "tmp")
        # Trigger the model validator to set up derived paths
        self._data_store.setup_cdp_captures_paths()

        # Delete old CDP vectorstore if it exists
        if self._data_store.cdp_captures_vectorstore_id is not None:
            console.print("[dim]Cleaning up previous CDP vectorstore...[/dim]")
            try:
                self._data_store.client.vector_stores.delete(
                    vector_store_id=self._data_store.cdp_captures_vectorstore_id
                )
            except Exception:
                pass  # Ignore errors - vectorstore may have expired
            self._data_store.cdp_captures_vectorstore_id = None

        # Create new vectorstore
        with console.status("[bold blue]Creating CDP captures vectorstore...[/bold blue]"):
            self._data_store.make_cdp_captures_vectorstore()

        # Refresh agent's vectorstore access so file_search can find CDP captures
        self._agent.refresh_vectorstores()

        console.print("[green]✓ CDP captures vectorstore ready![/green]")
        console.print()

        # Notify the agent about the new data
        self._agent.notify_browser_recording_result(accepted=True)

    def _handle_routine_discovery(self, task: str | None) -> None:
        """Handle routine discovery request."""
        if not task:
            console.print("[yellow]⚠ No task description provided.[/yellow]")
            return

        console.print()
        console.print("[bold cyan]Routine Discovery Task:[/bold cyan]")
        console.print(f"  {task}")
        console.print()

        # Allow user to accept, reject, or modify the task
        while True:
            response = console.input("[yellow]Start routine discovery? (y/n/m to modify): [/yellow]").strip().lower()
            if response == "y":
                break
            if response == "n":
                self._agent.notify_routine_discovery_response(accepted=False)
                return
            if response == "m":
                modified_task = console.input("[yellow]Enter new task description: [/yellow]").strip()
                if modified_task:
                    task = modified_task
                    console.print()
                    console.print("[bold cyan]Updated Task:[/bold cyan]")
                    console.print(f"  {task}")
                    console.print()
                else:
                    console.print("[yellow]⚠ Empty input, task unchanged.[/yellow]")
            else:
                console.print("[yellow]⚠ Please enter 'y', 'n', or 'm'[/yellow]")

        # User accepted - log that discovery is starting (no agent response yet)
        self._agent.notify_routine_discovery_response(accepted=True, task_description=task)

        # Verify we have data store with CDP captures
        if not isinstance(self._data_store, LocalDiscoveryDataStore):
            console.print("[red]✗ No data store available.[/red]")
            self._agent.notify_routine_discovery_result(error="No data store available")
            return

        if self._data_store.cdp_captures_vectorstore_id is None:
            console.print("[red]✗ No CDP captures available. Run /monitor first.[/red]")
            self._agent.notify_routine_discovery_result(error="No CDP captures available")
            return

        # Create output directory
        output_dir = Path("./routine_discovery_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print()
        console.print("[bold blue]Starting routine discovery...[/bold blue]")

        # Progress callback for discovery messages
        def handle_discovery_message(msg: RoutineDiscoveryMessage) -> None:
            if msg.type == RoutineDiscoveryMessageType.PROGRESS_THINKING:
                console.print(f"[dim]🤔 {msg.content}[/dim]")
            elif msg.type == RoutineDiscoveryMessageType.PROGRESS_RESULT:
                console.print(f"[green]✓ {msg.content}[/green]")
            elif msg.type == RoutineDiscoveryMessageType.ERROR:
                console.print(f"[red]✗ {msg.content}[/red]")

        try:
            # Run discovery using existing data store's vectorstore
            discovery = RoutineDiscovery(
                client=self._data_store.client,
                task=task,
                cdp_captures_dir=str(self._cdp_captures_dir),
                output_dir=str(output_dir),
                llm_model=str(self._agent.llm_model.value),
                message_callback=handle_discovery_message,
            )
            result = discovery.run()
            routine = result.routine

            console.print()
            console.print("[bold green]✓ Routine discovered successfully![/bold green]")
            console.print(f"  Name: {routine.name}")
            console.print(f"  Operations: {len(routine.operations)}")
            console.print(f"  Parameters: {len(routine.parameters)}")

            # Save routine to file (name -> lowercase_underscores)
            safe_name = routine.name.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            routines_dir = Path("./example_data/example_routines")
            routines_dir.mkdir(parents=True, exist_ok=True)
            routine_path = routines_dir / f"{safe_name}.json"
            routine_path.write_text(json.dumps(routine.model_dump(), indent=2))
            console.print(f"  Saved to: {routine_path}")

            # Load routine into context (like /load)
            routine_str = routine_path.read_text()
            self._loaded_routine_path = routine_path
            self._agent.routine_state.update_current_routine(routine_str)

            console.print()
            console.print("[green]✓ Routine loaded into context![/green]")
            console.print()

            # Notify agent to review the routine
            self._agent.notify_routine_discovery_result(routine=routine)

        except Exception as e:
            console.print()
            console.print(f"[bold red]✗ Discovery failed: {e}[/bold red]")
            console.print()
            self._agent.notify_routine_discovery_result(error=str(e))

    def _handle_routine_creation(self, routine: Routine | None) -> None:
        """Handle routine creation request - save and load the routine."""
        if not routine:
            console.print("[yellow]⚠ No routine provided.[/yellow]")
            return

        console.print()
        console.print("[bold cyan]Creating new routine:[/bold cyan]")
        console.print(f"  Name: {routine.name}")
        console.print(f"  Operations: {len(routine.operations)}")
        console.print(f"  Parameters: {len(routine.parameters)}")
        console.print()

        try:
            # Save routine to file (name -> lowercase_underscores)
            safe_name = routine.name.lower().replace(" ", "_").replace("-", "_")
            safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
            routines_dir = Path("./example_data/example_routines")
            routines_dir.mkdir(parents=True, exist_ok=True)
            routine_path = routines_dir / f"{safe_name}.json"
            routine_path.write_text(json.dumps(routine.model_dump(), indent=2))
            console.print(f"[green]✓ Saved to: {routine_path}[/green]")

            # Load routine into context (like /load)
            routine_str = routine_path.read_text()
            self._loaded_routine_path = routine_path
            self._agent.routine_state.update_current_routine(routine_str)

            console.print("[green]✓ Routine loaded into context![/green]")
            console.print()

            # Notify agent about the created routine
            system_message = (
                f"[ACTION REQUIRED] Routine '{routine.name}' has been created and saved to {routine_path}. "
                f"It has {len(routine.operations)} operations and {len(routine.parameters)} parameters. "
                "The routine is now loaded into context. Review it using get_current_routine and explain "
                "to the user what it does, what parameters it needs, and how to use it."
            )
            self._agent.process_new_message(system_message, ChatRole.SYSTEM)

        except Exception as e:
            console.print()
            console.print(f"[bold red]✗ Failed to create routine: {e}[/bold red]")
            console.print()

    def run(self) -> None:
        """Run the interactive chat loop."""
        print_welcome(str(self._agent.llm_model))

        while True:
            try:
                # Handle pending tool confirmation
                if self._pending_invocation:
                    user_input = console.input()
                    if self._handle_tool_confirmation(user_input):
                        if not self._pending_invocation:
                            continue
                        else:
                            continue
                else:
                    user_input = console.input(self._get_prompt())

                # Skip empty input
                if not user_input.strip():
                    continue

                # Check for commands (all start with /)
                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold cyan]Goodbye![/bold cyan]")
                    console.print()
                    break

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        r"""[bold]Commands:[/bold]
  [cyan]/load <routine.json>[/cyan]     Load a routine file (auto-reloads on edits)
  [cyan]/unload[/cyan]                  Unload the current routine
  [cyan]/show[/cyan]                    Show current routine details
  [cyan]/validate[/cyan]                Validate the current routine
  [cyan]/execute \[params.json][/cyan]   Execute the loaded routine
  [cyan]/monitor[/cyan]                 Start browser monitoring session
  [cyan]/diff[/cyan]                    Show pending suggested edit diff
  [cyan]/accept[/cyan]                  Accept pending suggested edit
  [cyan]/reject[/cyan]                  Reject pending suggested edit
  [cyan]/status[/cyan]                  Show current state
  [cyan]/chats[/cyan]                   Show all messages in the thread
  [cyan]/reset[/cyan]                   Start a new conversation
  [cyan]/help[/cyan]                    Show this help message
  [cyan]/quit[/cyan]                    Exit

[bold]Tips:[/bold]
  - After /load, edit the file externally and changes are picked up automatically
  - The prompt shows the loaded routine name: [dim]You (routine_name)>[/dim]
  - Use /execute without params to enter values interactively
  - Ask the agent to validate, explain, or debug your routine
  - When agent suggests edits, use /diff, /accept, or /reject to review them""",
                        title="[bold magenta]Help[/bold magenta]",
                        border_style="magenta",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                if cmd == "/reset":
                    self._agent.reset()
                    self._pending_invocation = None
                    self._pending_suggested_edit = None
                    self._loaded_routine_path = None
                    self._last_execution_ok = None
                    console.print()
                    console.print("[yellow]↺ Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd == "/status":
                    self._handle_status_command()
                    continue

                if cmd == "/chats":
                    self._handle_chats_command()
                    continue

                if cmd == "/show":
                    self._handle_show_command()
                    continue

                if cmd == "/validate":
                    self._handle_validate_command()
                    continue

                if cmd == "/diff":
                    self._handle_diff_command()
                    continue

                if cmd == "/accept":
                    self._handle_accept_command()
                    continue

                if cmd == "/reject":
                    self._handle_reject_command()
                    continue

                if cmd == "/unload":
                    self._handle_unload_command()
                    continue

                if cmd == "/monitor":
                    self._handle_browser_recording(skip_prompt=True)
                    continue

                if user_input.strip().startswith("/load "):
                    self._handle_load_command(user_input.strip()[6:].strip())
                    continue

                if user_input.strip().startswith("/execute"):
                    params_path = user_input.strip()[8:].strip() or None
                    self._handle_execute_command(params_path)
                    continue

                # Re-read routine from file before each message (picks up external edits)
                self._reload_routine_from_file()

                # Process the message (no spinner - conflicts with streaming output)
                self._agent.process_new_message(user_input, ChatRole.USER)

                # Check if agent requested a browser recording
                if self._browser_recording_requested:
                    self._browser_recording_requested = False
                    self._handle_browser_recording()

                # Check if agent requested routine discovery
                if self._routine_discovery_requested:
                    self._routine_discovery_requested = False
                    task = self._routine_discovery_task
                    self._routine_discovery_task = None
                    self._handle_routine_discovery(task)

                # Check if agent requested routine creation
                if self._routine_creation_requested:
                    self._routine_creation_requested = False
                    routine = self._created_routine
                    self._created_routine = None
                    self._handle_routine_creation(routine)

            except KeyboardInterrupt:
                console.print()
                console.print("[cyan]Interrupted. Goodbye![/cyan]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[cyan]Goodbye![/cyan]")
                console.print()
                break


def parse_model(model_str: str) -> OpenAIModel:
    """Parse a model string into an OpenAIModel enum value."""
    for model in OpenAIModel:
        if model.value == model_str or model.name == model_str:
            return model
    raise ValueError(f"Unknown model: {model_str}")


def main() -> None:
    """Entry point for the guide agent terminal."""
    parser = argparse.ArgumentParser(description="Interactive Guide Agent terminal")
    parser.add_argument(
        "--model",
        type=str,
        default=OpenAIModel.GPT_5_1.value,
        help=f"LLM model to use (default: {OpenAIModel.GPT_5_1.value})",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=str,
        default=None,
        help="Path to CDP captures directory (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./guide_agent_output",
        help="Output directory for temporary files (default: ./guide_agent_output)",
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default=str(BLUEBOX_PACKAGE_ROOT / "agent_docs"),
        help="Documentation directory (default: bluebox/agent_docs)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress all log output",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Write logs to file instead of console",
    )
    args = parser.parse_args()

    # Configure logging before anything else
    configure_logging(quiet=args.quiet, log_file=args.log_file)

    # Validate API key
    if Config.OPENAI_API_KEY is None:
        console.print("[bold red]Error: OPENAI_API_KEY environment variable is not set[/bold red]")
        sys.exit(1)

    data_store: LocalDiscoveryDataStore | None = None

    try:
        llm_model = parse_model(args.model)

        # Initialize OpenAI client for data store
        openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

        # Build data store kwargs - CDP paths are optional
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

        # Add CDP captures paths if provided
        if args.cdp_captures_dir:
            data_store_kwargs.update({
                "cdp_captures_dir": args.cdp_captures_dir,
                "tmp_dir": str(Path(args.output_dir) / "tmp"),
            })

        # Create data store with status
        with console.status("[bold blue]Initializing...[/bold blue]") as status:
            status.update("[bold blue]Creating data store...[/bold blue]")
            data_store = LocalDiscoveryDataStore(**data_store_kwargs)

            if args.cdp_captures_dir:
                status.update("[bold blue]Creating CDP captures vectorstore...[/bold blue]")
                data_store.make_cdp_captures_vectorstore()

            status.update("[bold blue]Creating documentation vectorstore...[/bold blue]")
            data_store.make_documentation_vectorstore()

        console.print("[green]✓ Vectorstores ready![/green]")
        console.print()

        cdp_captures_dir = Path(args.cdp_captures_dir) if args.cdp_captures_dir else DEFAULT_CDP_CAPTURES_DIR
        chat = TerminalGuideChat(llm_model=llm_model, data_store=data_store, cdp_captures_dir=cdp_captures_dir)
        chat.run()

    except ValueError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        sys.exit(1)
    finally:
        # Clean up vectorstores
        if data_store is not None:
            console.print()
            with console.status("[dim]Cleaning up vectorstores...[/dim]"):
                try:
                    data_store.clean_up()
                except KeyboardInterrupt:
                    console.print("[yellow]Cleanup interrupted[/yellow]")
                except Exception as e:
                    console.print(f"[yellow]Warning: Cleanup failed: {e}[/yellow]")
            console.print("[green]✓ Cleanup complete![/green]")


if __name__ == "__main__":
    main()
