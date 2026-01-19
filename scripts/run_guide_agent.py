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
  /execute [params.json]    Execute the loaded routine
  /status                  Show current state
  /reset                   Start a new conversation
  /help                    Show help
  /quit                    Exit
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from openai import OpenAI

# Package root for code_paths (scripts/ is sibling to web_hacker/)
WEB_HACKER_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "web_hacker"
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from web_hacker.agents.guide_agent import GuideAgent
from web_hacker.config import Config
from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.data_models.llms.interaction import (
    ChatMessageType,
    EmittedChatMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from web_hacker.data_models.routine.routine import Routine
from web_hacker.llms.tools.guide_agent_tools import validate_routine
from web_hacker.routine_discovery.data_store import DiscoveryDataStore, LocalDiscoveryDataStore


console = Console()


def configure_logging(quiet: bool = False, log_file: str | None = None) -> None:
    """
    Configure logging for all web_hacker modules.

    Args:
        quiet: If True, suppress all logs to console.
        log_file: If provided, write logs to this file instead of console.
    """
    # Get the parent logger for all web_hacker modules
    wh_logger = logging.getLogger("web_hacker")

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
  [cyan]/show[/cyan]                    Show current routine details
  [cyan]/validate[/cyan]                Validate the current routine
  [cyan]/execute \[params.json][/cyan]   Execute the loaded routine
  [cyan]/status[/cyan]                  Show current state
  [cyan]/help[/cyan]                    Show all commands

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
            console.print(f"[red]  Error: {error}[/red]")

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
    ) -> None:
        """Initialize the terminal chat interface."""
        self._pending_invocation: PendingToolInvocation | None = None
        self._streaming_started: bool = False
        self._data_store = data_store
        self._loaded_routine_path: Path | None = None
        self._last_execution_ok: bool | None = None
        self._agent = GuideAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model if llm_model else OpenAIModel.GPT_5_1,
            data_store=data_store,
        )

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

    def _handle_message(self, message: EmittedChatMessage) -> None:
        """Handle messages emitted by the Guide Agent."""
        if message.type == ChatMessageType.CHAT_RESPONSE:
            if self._streaming_started:
                print()  # End the streamed line
                print()  # Add spacing
                self._streaming_started = False
            else:
                print_assistant_message(message.content or "")

        elif message.type == ChatMessageType.TOOL_INVOCATION_REQUEST:
            if message.tool_invocation:
                self._pending_invocation = message.tool_invocation
                print_tool_request(message.tool_invocation)

        elif message.type == ChatMessageType.TOOL_INVOCATION_RESULT:
            if message.tool_invocation:
                print_tool_result(
                    message.tool_invocation,
                    message.tool_result,
                    message.error,
                )

        elif message.type == ChatMessageType.ERROR:
            print_error(message.error or "Unknown error")

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
        last_result = self._agent.routine_state.last_executed_routine_result
        if last_result:
            ok = last_result.get("ok", False)
            status = "[green]Success[/green]" if ok else "[red]Failed[/red]"
            table.add_row("Last Execution", status)
            if not ok and last_result.get("error"):
                table.add_row("Last Error", f"[red]{last_result.get('error')[:50]}...[/red]")
        else:
            table.add_row("Last Execution", "[dim]None[/dim]")

        # Thread info
        table.add_row("Thread ID", self._agent.thread_id[:8] + "...")
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

        try:
            console.print()
            with console.status("[bold yellow]Executing routine...[/bold yellow]"):
                # Create and execute routine
                routine = Routine(**routine_dict)
                result = routine.execute(params)
                result_dict = result.model_dump()

            # Update agent state with execution result
            self._agent.routine_state.update_last_execution(
                routine_json=routine_dict,
                routine_params=params,
                routine_result=result_dict,
            )
            self._last_execution_ok = result.ok

            print_execution_result(result_dict, result.ok, result.error)

        except Exception as e:
            console.print()
            console.print(f"[bold red]✗ Execution error: {e}[/bold red]")
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
  [cyan]/status[/cyan]                  Show current state
  [cyan]/reset[/cyan]                   Start a new conversation
  [cyan]/help[/cyan]                    Show this help message
  [cyan]/quit[/cyan]                    Exit

[bold]Tips:[/bold]
  - After /load, edit the file externally and changes are picked up automatically
  - The prompt shows the loaded routine name: [dim]You (routine_name)>[/dim]
  - Use /execute without params to enter values interactively
  - Ask the agent to validate, explain, or debug your routine""",
                        title="[bold magenta]Help[/bold magenta]",
                        border_style="magenta",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                if cmd == "/reset":
                    self._agent.reset()
                    self._pending_invocation = None
                    self._loaded_routine_path = None
                    self._last_execution_ok = None
                    console.print()
                    console.print("[yellow]↺ Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd == "/status":
                    self._handle_status_command()
                    continue

                if cmd == "/show":
                    self._handle_show_command()
                    continue

                if cmd == "/validate":
                    self._handle_validate_command()
                    continue

                if cmd == "/unload":
                    self._handle_unload_command()
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
                self._agent.process_user_message(user_input)

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
        default=str(WEB_HACKER_PACKAGE_ROOT / "agent_docs"),
        help="Documentation directory (default: web_hacker/agent_docs)",
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
                str(WEB_HACKER_PACKAGE_ROOT / "data_models" / "routine"),
                str(WEB_HACKER_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
                str(WEB_HACKER_PACKAGE_ROOT / "routine_discovery"),
                str(WEB_HACKER_PACKAGE_ROOT / "utils" / "js_utils.py"),
                str(WEB_HACKER_PACKAGE_ROOT / "utils" / "data_utils.py"),
            ],
        }

        # Add CDP captures paths if provided
        if args.cdp_captures_dir:
            data_store_kwargs.update({
                "tmp_dir": str(Path(args.output_dir) / "tmp"),
                "transactions_dir": str(Path(args.cdp_captures_dir) / "network" / "transactions"),
                "consolidated_transactions_path": str(Path(args.cdp_captures_dir) / "network" / "consolidated_transactions.json"),
                "storage_jsonl_path": str(Path(args.cdp_captures_dir) / "storage" / "events.jsonl"),
                "window_properties_path": str(Path(args.cdp_captures_dir) / "window_properties" / "window_properties.json"),
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

        chat = TerminalGuideChat(llm_model=llm_model, data_store=data_store)
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
                except Exception as e:
                    console.print(f"[yellow]Warning: Cleanup failed: {e}[/yellow]")
            console.print("[green]✓ Cleanup complete![/green]")


if __name__ == "__main__":
    main()
