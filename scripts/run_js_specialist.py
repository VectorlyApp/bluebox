#!/usr/bin/env python3
"""
scripts/run_js_specialist.py

Interactive CLI for the JS Specialist agent.

Usage:
    python scripts/run_js_specialist.py

    python scripts/run_js_specialist.py \
        --dom-snapshots-dir ./cdp_captures/dom/ \
        --remote-debugging-address 127.0.0.1:9222

"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bluebox.utils.terminal_utils import SlashCommandCompleter
from bluebox.agents.specialists.js_specialist import (
    JSSpecialist,
    JSCodeResult,
    JSCodeFailureResult,
)
from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.data_models.llms.interaction import (
    ChatRole,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

SLASH_COMMANDS = [
    ("/autonomous", "Run autonomous JS code generation — /autonomous <task>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]



BANNER = """\
[bold green]╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║       ██╗███████╗    ███████╗██████╗ ███████╗ ██████╗██╗ █████╗ ██╗     ██╗███████╗████████╗ ║
║       ██║██╔════╝    ██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔══██╗██║     ██║██╔════╝╚══██╔══╝ ║
║       ██║███████╗    ███████╗██████╔╝█████╗  ██║     ██║███████║██║     ██║███████╗   ██║    ║
║  ██   ██║╚════██║    ╚════██║██╔═══╝ ██╔══╝  ██║     ██║██╔══██║██║     ██║╚════██║   ██║    ║
║  ╚█████╔╝███████║    ███████║██║     ███████╗╚██████╗██║██║  ██║███████╗██║███████║   ██║    ║
║   ╚════╝ ╚══════╝    ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝    ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝[/bold green]
"""


def print_welcome(model: str, dom_count: int = 0) -> None:
    """Print welcome message."""
    console.print(BANNER)
    console.print()

    if dom_count > 0:
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("DOM Snapshots", str(dom_count))

        console.print(Panel(
            stats_table,
            title="[bold green]Context[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        console.print()

    console.print(Panel(
        """[bold]Commands:[/bold]
  [cyan]/autonomous <task>[/cyan]  Run autonomous JS code generation
  [cyan]/reset[/cyan]              Start a new conversation
  [cyan]/help[/cyan]               Show help
  [cyan]/quit[/cyan]               Exit

Just ask questions about the JavaScript files!""",
        title="[bold green]JS Specialist[/bold green]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))
    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold green]Assistant[/bold green]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str) -> None:
    """Print an error message."""
    console.print()
    console.print(f"[bold red]Error:[/bold red] [red]{escape(error)}[/red]")
    console.print()


def print_tool_call(invocation: PendingToolInvocation) -> None:
    """Print a tool call with formatted arguments."""
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
        title="[bold yellow]TOOL CALL[/bold yellow]",
        style="yellow",
        box=box.ROUNDED,
    ))


def print_tool_result(
    invocation: PendingToolInvocation,
    result: dict[str, Any] | None,
) -> None:
    """Print a tool invocation result."""
    if invocation.status == ToolInvocationStatus.EXECUTED:
        console.print("[bold green]Tool executed[/bold green]")
        if result:
            result_json = json.dumps(result, indent=2)
            lines = result_json.split("\n")
            if len(lines) > 150:
                display = "\n".join(lines[:150]) + f"\n... ({len(lines) - 150} more lines)"
            else:
                display = result_json
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]Tool execution failed[/bold red]")
        error = result.get("error") if result else None
        if error:
            console.print(Panel(str(error), title="Error", style="red", box=box.ROUNDED))

    console.print()


class TerminalJSSpecialistChat:
    """Interactive terminal chat interface for the JS Specialist Agent."""

    def __init__(
        self,
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        remote_debugging_address: str | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._streaming_started: bool = False
        self._dom_snapshots = dom_snapshots
        self._llm_model = llm_model
        self._remote_debugging_address = remote_debugging_address
        self._agent = self._create_agent()

    def _create_agent(self) -> JSSpecialist:
        """Create a fresh JSSpecialist agent."""
        return JSSpecialist(
            emit_message_callable=self._handle_message,
            dom_snapshots=self._dom_snapshots,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            remote_debugging_address=self._remote_debugging_address,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold green]Assistant[/bold green]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """Handle messages emitted by the JS Specialist Agent."""
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()
                print()
                self._streaming_started = False
            else:
                print_assistant_message(message.content)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            print_tool_call(message.tool_invocation)
            print_tool_result(message.tool_invocation, message.tool_result)

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)

    def _run_autonomous(self, task: str) -> None:
        """Run autonomous JS code generation for a given task."""
        console.print()
        console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]Starting Autonomous JS Analysis[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        console.print()

        self._agent.reset()

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        console.print()

        if isinstance(result, JSCodeResult):
            result_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            result_table.add_column("Field", style="bold green")
            result_table.add_column("Value", style="white")

            result_table.add_row("Description", result.description)
            if result.session_storage_key:
                result_table.add_row("Session Storage Key", result.session_storage_key)
            result_table.add_row("Timeout", f"{result.timeout_seconds}s")
            result_table.add_row("JS Code", result.js_code[:500] + ("..." if len(result.js_code) > 500 else ""))

            console.print(Panel(
                result_table,
                title=f"[bold green]JS Code Generated[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))

        elif isinstance(result, JSCodeFailureResult):
            failure_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            failure_table.add_column("Field", style="bold red")
            failure_table.add_column("Value", style="white")

            failure_table.add_row("Reason", result.reason)
            if result.attempted_approaches:
                failure_table.add_row("Attempted", "\n".join(result.attempted_approaches))

            console.print(Panel(
                failure_table,
                title=f"[bold red]JS Code Generation Failed[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            console.print(Panel(
                "[yellow]Could not finalize JS code generation. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]Analysis Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        console.print()

    def run(self) -> None:
        """Run the interactive chat loop."""
        while True:
            try:
                user_input = pt_prompt(
                    HTML("<b><ansigreen>You&gt;</ansigreen></b> "),
                    completer=SlashCommandCompleter(SLASH_COMMANDS),
                    complete_while_typing=True,
                )

                if not user_input.strip():
                    continue

                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold green]Goodbye![/bold green]")
                    console.print()
                    break

                if cmd == "/reset":
                    self._agent = self._create_agent()
                    console.print()
                    console.print("[yellow]Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        """[bold]Commands:[/bold]
  [cyan]/autonomous <task>[/cyan]  Run autonomous JS code generation
                        Example: /autonomous extract the search results from the page
  [cyan]/reset[/cyan]              Start a new conversation
  [cyan]/help[/cyan]               Show this help message
  [cyan]/quit[/cyan]               Exit

[bold]Tips:[/bold]
  - Ask about specific JS files or patterns
  - Request code to extract data from pages
  - Ask about DOM structure and element selectors""",
                        title="[bold green]Help[/bold green]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                if user_input.strip().lower().startswith("/autonomous"):
                    task = user_input.strip()[len("/autonomous"):].strip()
                    if not task:
                        console.print()
                        console.print("[bold yellow]Usage:[/bold yellow] /autonomous <task description>")
                        console.print("[dim]Example: /autonomous extract the search results from the page[/dim]")
                        console.print()
                        continue

                    self._run_autonomous(task)
                    continue

                self._agent.process_new_message(user_input, ChatRole.USER)

            except KeyboardInterrupt:
                console.print()
                console.print("[green]Interrupted. Goodbye![/green]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[green]Goodbye![/green]")
                console.print()
                break


def main() -> None:
    """Run the JS Specialist agent interactively."""
    parser = argparse.ArgumentParser(
        description="JS Specialist - Interactive JavaScript code generation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default=None,
        help="Chrome remote debugging address (e.g. 127.0.0.1:9222) for execute_js_in_browser tool",
    )
    parser.add_argument(
        "--dom-snapshots-dir",
        type=str,
        default=None,
        help="Directory containing DOM snapshot JSON files",
    )
    args = parser.parse_args()

    # Load DOM snapshots if provided
    dom_snapshots: list[DOMSnapshotEvent] | None = None
    if args.dom_snapshots_dir:
        dom_dir = Path(args.dom_snapshots_dir)
        if not dom_dir.is_dir():
            console.print(f"[bold red]Error: DOM snapshots directory not found: {dom_dir}[/bold red]")
            sys.exit(1)

        dom_snapshots = []
        for snap_file in sorted(dom_dir.glob("*.json")):
            try:
                data = json.loads(snap_file.read_text())
                dom_snapshots.append(DOMSnapshotEvent(**data))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not parse {snap_file.name}: {e}[/yellow]")

        console.print(f"[dim]Loaded {len(dom_snapshots)} DOM snapshots from {dom_dir}[/dim]")

    # Map model string to enum
    model_map = {
        "gpt-5.1": OpenAIModel.GPT_5_1,
    }
    llm_model = model_map.get(args.model, OpenAIModel.GPT_5_1)

    print_welcome(args.model, dom_count=len(dom_snapshots) if dom_snapshots else 0)

    chat = TerminalJSSpecialistChat(
        dom_snapshots=dom_snapshots,
        llm_model=llm_model,
        remote_debugging_address=args.remote_debugging_address,
    )
    chat.run()


if __name__ == "__main__":
    main()
