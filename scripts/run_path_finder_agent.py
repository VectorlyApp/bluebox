#!/usr/bin/env python3
"""
scripts/run_path_finder_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Interactive CLI for the Path Finder agent - traces where tokens/values originated from.

Usage:
    python scripts/run_path_finder_agent.py \
        --network-jsonl ./cdp_captures/network/events.jsonl \
        --storage-jsonl ./cdp_captures/storage/events.jsonl \
        --window-props-jsonl ./cdp_captures/window_properties/events.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bluebox.agents.path_finder_agent import (
    PathFinderAgent,
    TokenOriginResult,
    TokenOriginFailure,
)
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.llms.infra.storage_data_store import StorageDataStore
from bluebox.llms.infra.window_property_data_store import WindowPropertyDataStore
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


BANNER = """\
[bold magenta]╔═════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                         ║
║ ▄▄▄▄▄▄              ██    ██       ▄▄▄▄▄▄▄▄ ██                 ██                       ║
║ ██▀▀▀▀██            ██    ██       ██▀▀▀▀██ ▀▀                 ██                       ║
║ ██    ██  ▄█████▄ ███████ ██████   ██       ██  ██▄████▄  ▄███▄██   ▄████▄    ██▄████   ║
║ ███████▀  ▀ ▄▄▄██   ██    ██   ██  █████    ██  ██▀   ██  ██▀  ██  ██▄▄▄▄██   ██▀       ║
║ ██        ▄██▀▀▀██  ██    ██   ██  ██       ██  ██    ██  ██   ██  ██▀▀▀▀▀▀   ██        ║
║ ██        ██▄▄▄███  ██▄▄▄ ██   ██  ██       ██  ██    ██  ▀█▄▄███  ▀██▄▄▄▄█   ██        ║
║ ▀▀         ▀▀▀▀ ▀▀   ▀▀▀▀ ▀▀   ▀▀  ▀▀       ▀▀  ▀▀    ▀▀   ▀▀▀ ▀▀    ▀▀▀▀▀    ▀▀        ║
║                                                                                         ║
║                    Token Origin Tracer - Find where values come from                    ║
║                                                                                         ║
╚══════════════════════════════════════(BETA)═════════════════════════════════════════════╝[/bold magenta]
"""


def print_welcome(
    model: str,
    network_store: NetworkDataStore | None,
    storage_store: StorageDataStore | None,
    window_store: WindowPropertyDataStore | None,
) -> None:
    """Print welcome message with data store stats."""
    console.print(BANNER)
    console.print()

    # Build stats table
    stats_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
    stats_table.add_column("Data Store", style="bold cyan")
    stats_table.add_column("Status", style="white")
    stats_table.add_column("Details", style="dim")

    # Network stats
    if network_store:
        stats = network_store.stats
        stats_table.add_row(
            "Network Traffic",
            f"[green]Loaded[/green]",
            f"{stats.total_requests} requests, {stats.unique_urls} URLs, {stats.unique_hosts} hosts",
        )
    else:
        stats_table.add_row("Network Traffic", "[yellow]Not loaded[/yellow]", "")

    # Storage stats
    if storage_store:
        stats = storage_store.stats
        stats_table.add_row(
            "Browser Storage",
            f"[green]Loaded[/green]",
            f"{stats.total_events} events (cookies: {stats.cookie_events}, "
            f"localStorage: {stats.local_storage_events}, sessionStorage: {stats.session_storage_events})",
        )
    else:
        stats_table.add_row("Browser Storage", "[yellow]Not loaded[/yellow]", "")

    # Window properties stats
    if window_store:
        stats = window_store.stats
        stats_table.add_row(
            "Window Properties",
            f"[green]Loaded[/green]",
            f"{stats.total_events} events, {stats.total_changes} changes, "
            f"{stats.unique_property_paths} unique paths",
        )
    else:
        stats_table.add_row("Window Properties", "[yellow]Not loaded[/yellow]", "")

    console.print(Panel(
        stats_table,
        title="[bold magenta]Data Sources[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()

    # Show summary stats if we have storage
    if storage_store:
        console.print(Panel(
            storage_store.stats.to_summary(),
            title="[bold cyan]Storage Summary[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()

    # Show window props summary if we have it
    if window_store:
        console.print(Panel(
            window_store.stats.to_summary(),
            title="[bold yellow]Window Properties Summary[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        console.print()

    console.print(Panel(
        """[bold]Commands:[/bold]
  [magenta]/trace <value>[/magenta]    Autonomously trace where a token/value came from
  [magenta]/reset[/magenta]            Start a new conversation
  [magenta]/help[/magenta]             Show help
  [magenta]/quit[/magenta]             Exit

Just ask questions about where values came from!

[bold]Examples:[/bold]
  "Where did the token abc123xyz come from?"
  "/trace eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
  "Find the origin of this session ID: sess_12345" """,
        title="[bold magenta]Path Finder[/bold magenta]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold magenta]Assistant[/bold magenta]")
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
            if len(lines) > 100:
                display = "\n".join(lines[:100]) + f"\n... ({len(lines) - 100} more lines)"
            else:
                display = result_json
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]Tool execution failed[/bold red]")
        error = result.get("error") if result else None
        if error:
            console.print(Panel(str(error), title="Error", style="red", box=box.ROUNDED))

    console.print()


class TerminalPathFinderChat:
    """Interactive terminal chat interface for the Path Finder Agent."""

    def __init__(
        self,
        network_store: NetworkDataStore | None = None,
        storage_store: StorageDataStore | None = None,
        window_store: WindowPropertyDataStore | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._streaming_started: bool = False
        self._agent = PathFinderAgent(
            emit_message_callable=self._handle_message,
            network_data_store=network_store,
            storage_data_store=storage_store,
            window_property_data_store=window_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold magenta]Assistant[/bold magenta]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """Handle messages emitted by the Path Finder Agent."""
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

    def _run_trace(self, value: str) -> None:
        """Run autonomous token tracing for a given value."""
        console.print()
        console.print(Panel(
            f"[bold]Value:[/bold] {value[:100]}{'...' if len(value) > 100 else ''}",
            title="[bold magenta]Starting Token Trace[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous tracing with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(value)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        console.print()

        if isinstance(result, TokenOriginResult):
            # Success - show discovered origins
            origins_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            origins_table.add_column("#", style="dim", width=3)
            origins_table.add_column("Source", style="cyan")
            origins_table.add_column("Location", style="white")
            origins_table.add_column("Context", style="dim")

            for i, origin in enumerate(result.origins):
                is_likely = result.likely_source and origin == result.likely_source
                marker = "[bold green]*[/bold green]" if is_likely else ""
                origins_table.add_row(
                    f"{i+1}{marker}",
                    origin.source_type,
                    origin.location[:60] + "..." if len(origin.location) > 60 else origin.location,
                    origin.context[:40] + "..." if len(origin.context) > 40 else origin.context,
                )

            console.print(Panel(
                origins_table,
                title=f"[bold green]Token Origins Found[/bold green] [dim]({len(result.origins)} locations)[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))

            if result.likely_source:
                console.print()
                console.print(Panel(
                    f"[bold]Source Type:[/bold] {result.likely_source.source_type}\n"
                    f"[bold]Location:[/bold] {result.likely_source.location}\n"
                    f"[bold]Context:[/bold] {result.likely_source.context}",
                    title="[bold green]Most Likely Original Source[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                ))

            console.print()
            console.print(Panel(
                result.explanation,
                title="[bold cyan]Explanation[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))

            console.print()
            console.print(f"[bold green]Trace completed[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")

        elif isinstance(result, TokenOriginFailure):
            # Failure - value not found
            content = f"[bold]Reason:[/bold] {result.reason}"
            if result.suggestions:
                content += "\n\n[bold]Suggestions:[/bold]\n" + "\n".join(f"  - {s}" for s in result.suggestions)

            console.print(Panel(
                content,
                title=f"[bold red]Value Not Found[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            # None - max iterations without finalization
            console.print(Panel(
                "[yellow]Could not complete token tracing. "
                "The agent reached max iterations without finding a conclusion.[/yellow]",
                title=f"[bold yellow]Trace Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        console.print()

    def run(self) -> None:
        """Run the interactive chat loop."""
        while True:
            try:
                user_input = console.input("[bold green]You>[/bold green] ")

                if not user_input.strip():
                    continue

                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold magenta]Goodbye![/bold magenta]")
                    console.print()
                    break

                if cmd == "/reset":
                    self._agent.reset()
                    console.print()
                    console.print("[yellow]Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        """[bold]Commands:[/bold]
  [magenta]/trace <value>[/magenta]    Autonomously trace where a token/value came from
                       Example: /trace abc123xyz
                       Example: /trace eyJhbGciOiJIUzI1NiJ9
  [magenta]/reset[/magenta]            Start a new conversation
  [magenta]/help[/magenta]             Show this help message
  [magenta]/quit[/magenta]             Exit

[bold]Tips:[/bold]
  - Use /trace for quick autonomous token origin discovery
  - Ask conversational questions for deeper analysis
  - The agent searches network traffic, browser storage, and window properties""",
                        title="[bold magenta]Help[/bold magenta]",
                        border_style="magenta",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                # Handle /trace command
                if user_input.strip().lower().startswith("/trace"):
                    value = user_input.strip()[len("/trace"):].strip()
                    if not value:
                        console.print()
                        console.print("[bold yellow]Usage:[/bold yellow] /trace <value>")
                        console.print("[dim]Example: /trace abc123xyz[/dim]")
                        console.print()
                        continue

                    self._run_trace(value)
                    continue

                self._agent.process_new_message(user_input, ChatRole.USER)

            except KeyboardInterrupt:
                console.print()
                console.print("[magenta]Interrupted. Goodbye![/magenta]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[magenta]Goodbye![/magenta]")
                console.print()
                break


def main() -> None:
    """Run the Path Finder agent interactively."""
    parser = argparse.ArgumentParser(
        description="Path Finder - Token origin tracer across network, storage, and window properties"
    )
    parser.add_argument(
        "--network-jsonl",
        type=str,
        help="Path to JSONL file containing NetworkTransactionEvent entries",
    )
    parser.add_argument(
        "--storage-jsonl",
        type=str,
        help="Path to JSONL file containing StorageEvent entries",
    )
    parser.add_argument(
        "--window-props-jsonl",
        type=str,
        help="Path to JSONL file containing WindowPropertyEvent entries",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    args = parser.parse_args()

    # Validate that at least one data source is provided
    if not any([args.network_jsonl, args.storage_jsonl, args.window_props_jsonl]):
        console.print("[bold red]Error: At least one data source must be provided[/bold red]")
        console.print("[dim]Use --network-jsonl, --storage-jsonl, or --window-props-jsonl[/dim]")
        sys.exit(1)

    # Load data stores
    network_store: NetworkDataStore | None = None
    storage_store: StorageDataStore | None = None
    window_store: WindowPropertyDataStore | None = None

    if args.network_jsonl:
        network_path = Path(args.network_jsonl)
        if not network_path.exists():
            console.print(f"[bold red]Error: Network JSONL file not found: {network_path}[/bold red]")
            sys.exit(1)
        console.print(f"[dim]Loading network data: {network_path}[/dim]")
        try:
            network_store = NetworkDataStore(str(network_path))
        except ValueError as e:
            console.print(f"[bold red]Error parsing network JSONL: {e}[/bold red]")
            sys.exit(1)

    if args.storage_jsonl:
        storage_path = Path(args.storage_jsonl)
        if not storage_path.exists():
            console.print(f"[bold red]Error: Storage JSONL file not found: {storage_path}[/bold red]")
            sys.exit(1)
        console.print(f"[dim]Loading storage data: {storage_path}[/dim]")
        try:
            storage_store = StorageDataStore(str(storage_path))
        except ValueError as e:
            console.print(f"[bold red]Error parsing storage JSONL: {e}[/bold red]")
            sys.exit(1)

    if args.window_props_jsonl:
        window_path = Path(args.window_props_jsonl)
        if not window_path.exists():
            console.print(f"[bold red]Error: Window props JSONL file not found: {window_path}[/bold red]")
            sys.exit(1)
        console.print(f"[dim]Loading window properties data: {window_path}[/dim]")
        try:
            window_store = WindowPropertyDataStore(str(window_path))
        except ValueError as e:
            console.print(f"[bold red]Error parsing window props JSONL: {e}[/bold red]")
            sys.exit(1)

    # Map model string to enum
    model_map = {
        "gpt-5.1": OpenAIModel.GPT_5_1,
    }
    llm_model = model_map.get(args.model, OpenAIModel.GPT_5_1)

    print_welcome(args.model, network_store, storage_store, window_store)

    chat = TerminalPathFinderChat(
        network_store=network_store,
        storage_store=storage_store,
        window_store=window_store,
        llm_model=llm_model,
    )
    chat.run()


if __name__ == "__main__":
    main()
