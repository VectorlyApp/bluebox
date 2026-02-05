#!/usr/bin/env python3
"""
bluebox/scripts/specialists/run_value_trace_resolver_specialist.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Interactive CLI for the ValueTraceResolverSpecialist - traces where tokens/values originated from.

Usage:
    bluebox-value-trace-resolver --network-jsonl ./cdp_captures/network/events.jsonl
    bluebox-value-trace-resolver --storage-jsonl ./cdp_captures/storage/events.jsonl
    bluebox-value-trace-resolver \
        --network-jsonl ./cdp_captures/network/events.jsonl \
        --storage-jsonl ./cdp_captures/storage/events.jsonl \
        --window-props-jsonl ./cdp_captures/window_properties/events.jsonl
"""

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.specialists.value_trace_resolver_specialist import (
    ValueTraceResolverSpecialist,
    TokenOriginResult,
    TokenOriginFailure,
)
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.agents.terminal_agent_base import AbstractTerminalAgentChat
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

SLASH_COMMANDS = [
    ("/trace", "Trace where a token/value came from — /trace <value>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]


BANNER = """\
[bold magenta]╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                     ║
║ ████████╗███████╗ █████╗  ██████╗███████╗    ██╗  ██╗ ██████╗ ██╗   ██╗███╗   ██╗██████╗            ║
║ ╚══██╔══╝██╔══██╗██╔══██╗██╔════╝██╔════╝    ██║  ██║██╔═══██╗██║   ██║████╗  ██║██╔══██╗           ║
║    ██║   ██████╔╝███████║██║     █████╗      ███████║██║   ██║██║   ██║██╔██╗ ██║██║  ██║           ║
║    ██║   ██╔══██╗██╔══██║██║     ██╔══╝      ██╔══██║██║   ██║██║   ██║██║╚██╗██║██║  ██║           ║
║    ██║   ██║  ██║██║  ██║╚██████╗███████╗    ██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║██████╔╝           ║
║    ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚══════╝    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═════╝            ║
║                                                                                                     ║
║                         Token Origin Tracer - Find where values come from                           ║
║                                                                                                     ║
╚════════════════════════════════════════════════(BETA)═══════════════════════════════════════════════════╝[/bold magenta]
"""


class TerminalTraceHoundChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the Trace Hound Agent."""

    autonomous_command_name = "trace"

    def __init__(
        self,
        network_store: NetworkDataLoader | None = None,
        storage_store: StorageDataLoader | None = None,
        window_store: WindowPropertyDataLoader | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self.network_store = network_store
        self.storage_store = storage_store
        self.window_store = window_store
        self.llm_model = llm_model
        super().__init__(console=console, agent_color="magenta")

    def _create_agent(self) -> ValueTraceResolverSpecialist:
        """Create the Trace Hound agent instance."""
        return ValueTraceResolverSpecialist(
            emit_message_callable=self._handle_message,
            network_data_store=self.network_store,
            storage_data_store=self.storage_store,
            window_property_data_store=self.window_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    def print_welcome(self) -> None:
        """Print welcome message with data store stats."""
        self.console.print(BANNER)
        self.console.print()

        # Build stats table
        stats_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 2))
        stats_table.add_column("Data Store", style="bold cyan")
        stats_table.add_column("Status", style="white")
        stats_table.add_column("Details", style="dim")

        # Network stats
        if self.network_store:
            stats = self.network_store.stats
            stats_table.add_row(
                "Network Traffic",
                f"[green]Loaded[/green]",
                f"{stats.total_requests} requests, {stats.unique_urls} URLs, {stats.unique_hosts} hosts",
            )
        else:
            stats_table.add_row("Network Traffic", "[yellow]Not loaded[/yellow]", "")

        # Storage stats
        if self.storage_store:
            stats = self.storage_store.stats
            stats_table.add_row(
                "Browser Storage",
                f"[green]Loaded[/green]",
                f"{stats.total_events} events (cookies: {stats.cookie_events}, "
                f"localStorage: {stats.local_storage_events}, sessionStorage: {stats.session_storage_events})",
            )
        else:
            stats_table.add_row("Browser Storage", "[yellow]Not loaded[/yellow]", "")

        # Window properties stats
        if self.window_store:
            stats = self.window_store.stats
            stats_table.add_row(
                "Window Properties",
                f"[green]Loaded[/green]",
                f"{stats.total_events} events, {stats.total_changes} changes, "
                f"{stats.unique_property_paths} unique paths",
            )
        else:
            stats_table.add_row("Window Properties", "[yellow]Not loaded[/yellow]", "")

        self.console.print(Panel(
            stats_table,
            title="[bold magenta]Data Sources[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Show summary stats if we have storage
        if self.storage_store:
            self.console.print(Panel(
                self.storage_store.stats.to_summary(),
                title="[bold cyan]Storage Summary[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))
            self.console.print()

        # Show window props summary if we have it
        if self.window_store:
            self.console.print(Panel(
                self.window_store.stats.to_summary(),
                title="[bold yellow]Window Properties Summary[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            ))
            self.console.print()

        self.console.print(Panel(
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
            title="[bold magenta]Trace Hound[/bold magenta]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, value: str) -> None:
        """Run autonomous token tracing for a given value."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Value:[/bold] {value[:100]}{'...' if len(value) > 100 else ''}",
            title="[bold magenta]Starting Token Trace[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous tracing with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(value)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        self.console.print()

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

            self.console.print(Panel(
                origins_table,
                title=f"[bold green]Token Origins Found[/bold green] [dim]({len(result.origins)} locations)[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))

            if result.likely_source:
                self.console.print()
                self.console.print(Panel(
                    f"[bold]Source Type:[/bold] {result.likely_source.source_type}\n"
                    f"[bold]Location:[/bold] {result.likely_source.location}\n"
                    f"[bold]Context:[/bold] {result.likely_source.context}",
                    title="[bold green]Most Likely Original Source[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                ))

            self.console.print()
            self.console.print(Panel(
                result.explanation,
                title="[bold cyan]Explanation[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))

            self.console.print()
            self.console.print(f"[bold green]Trace completed[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")

        elif isinstance(result, TokenOriginFailure):
            # Failure - value not found
            content = f"[bold]Reason:[/bold] {result.reason}"
            if result.suggestions:
                content += "\n\n[bold]Suggestions:[/bold]\n" + "\n".join(f"  - {s}" for s in result.suggestions)

            self.console.print(Panel(
                content,
                title=f"[bold red]Value Not Found[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            # None - max iterations without finalization
            self.console.print(Panel(
                "[yellow]Could not complete token tracing. "
                "The agent reached max iterations without finding a conclusion.[/yellow]",
                title=f"[bold yellow]Trace Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        self.console.print()


def main() -> None:
    """Run the Trace Hound agent interactively."""
    parser = argparse.ArgumentParser(
        description="Trace Hound - Token origin tracer across network, storage, and window properties"
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
    add_model_argument(parser)
    args = parser.parse_args()

    # Validate that at least one data source is provided
    if not any([args.network_jsonl, args.storage_jsonl, args.window_props_jsonl]):
        console.print("[bold red]Error: At least one data source must be provided[/bold red]")
        console.print("[dim]Use --network-jsonl, --storage-jsonl, or --window-props-jsonl[/dim]")
        sys.exit(1)

    # Load data stores
    network_store: NetworkDataLoader | None = None
    storage_store: StorageDataLoader | None = None
    window_store: WindowPropertyDataLoader | None = None

    if args.network_jsonl:
        network_path = Path(args.network_jsonl)
        if not network_path.exists():
            console.print(f"[bold red]Error: Network JSONL file not found: {network_path}[/bold red]")
            sys.exit(1)
        console.print(f"[dim]Loading network data: {network_path}[/dim]")
        try:
            network_store = NetworkDataLoader(str(network_path))
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
            storage_store = StorageDataLoader(str(storage_path))
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
            window_store = WindowPropertyDataLoader(str(window_path))
        except ValueError as e:
            console.print(f"[bold red]Error parsing window props JSONL: {e}[/bold red]")
            sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalTraceHoundChat(
        network_store=network_store,
        storage_store=storage_store,
        window_store=window_store,
        llm_model=llm_model,
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
