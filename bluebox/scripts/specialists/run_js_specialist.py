#!/usr/bin/env python3
"""
bluebox/scripts/specialists/run_js_specialist.py

Interactive CLI for the JS Specialist agent.

Usage:
    bluebox-js-specialist
    bluebox-js-specialist --dom-snapshots-dir ./cdp_captures/dom/
    bluebox-js-specialist \
        --dom-snapshots-dir ./cdp_captures/dom/ \
        --javascript-events-jsonl-path ./cdp_captures/network/javascript_events.jsonl \
        --network-events-jsonl-path ./cdp_captures/network/events.jsonl \
        --remote-debugging-address 127.0.0.1:9222

"""

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import dedent

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.specialists.abstract_specialist import RunMode
from bluebox.agents.specialists.js_specialist import (
    JSSpecialist,
    JSCodeResult,
    JSCodeFailureResult,
)
from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.infra.js_data_store import JSDataStore
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.agents.terminal_agent_base import AbstractTerminalAgentChat
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
[bold green]╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                              ║
║       ██╗███████╗    ███████╗██████╗ ███████╗ ██████╗██╗ █████╗ ██╗     ██╗███████╗████████╗ ║
║       ██║██╔════╝    ██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔══██╗██║     ██║██╔════╝╚══██╔══╝ ║
║       ██║███████╗    ███████╗██████╔╝█████╗  ██║     ██║███████║██║     ██║███████╗   ██║    ║
║  ██   ██║╚════██║    ╚════██║██╔═══╝ ██╔══╝  ██║     ██║██╔══██║██║     ██║╚════██║   ██║    ║
║  ╚█████╔╝███████║    ███████║██║     ███████╗╚██████╗██║██║  ██║███████╗██║███████║   ██║    ║
║   ╚════╝ ╚══════╝    ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝    ║
║                                                                                              ║
╚═══════════════════════════════════════════(BETA)═════════════════════════════════════════════╝[/bold green]
"""


class TerminalJSSpecialistChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the JS Specialist Agent."""

    autonomous_command_name = "autonomous"

    def __init__(
        self,
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        js_data_store: JSDataStore | None = None,
        network_data_store: NetworkDataStore | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        remote_debugging_address: str | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self.dom_snapshots = dom_snapshots
        self.js_data_store = js_data_store
        self.network_data_store = network_data_store
        self.llm_model = llm_model
        self.remote_debugging_address = remote_debugging_address
        super().__init__(console=console, agent_color="green")

    def _create_agent(self) -> JSSpecialist:
        """Create the JS Specialist agent instance."""
        return JSSpecialist(
            emit_message_callable=self._handle_message,
            dom_snapshots=self.dom_snapshots,
            js_data_store=self.js_data_store,
            network_data_store=self.network_data_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
            run_mode=RunMode.CONVERSATIONAL,
            remote_debugging_address=self.remote_debugging_address,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    def print_welcome(self) -> None:
        """Print welcome message."""
        self.console.print(BANNER)
        self.console.print()

        dom_count = len(self.dom_snapshots) if self.dom_snapshots else 0
        js_files_count = self.js_data_store.stats.total_files if self.js_data_store else 0
        network_entries_count = self.network_data_store.stats.total_requests if self.network_data_store else 0

        if dom_count > 0 or js_files_count > 0 or network_entries_count > 0:
            stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            stats_table.add_column("Label", style="dim")
            stats_table.add_column("Value", style="white")
            if dom_count > 0:
                stats_table.add_row("DOM Snapshots", str(dom_count))
            if js_files_count > 0:
                stats_table.add_row("JS Files", str(js_files_count))
            if network_entries_count > 0:
                stats_table.add_row("Network Requests", str(network_entries_count))

            self.console.print(Panel(
                stats_table,
                title="[bold green]Context[/bold green]",
                border_style="green",
                box=box.ROUNDED,
            ))
            self.console.print()

        self.console.print(Panel(
            dedent("""\
                [bold]Commands:[/bold]
                  [cyan]/autonomous <task>[/cyan]  Run autonomous JS code generation
                  [cyan]/reset[/cyan]              Start a new conversation
                  [cyan]/help[/cyan]               Show help
                  [cyan]/quit[/cyan]               Exit

                Just ask questions about JavaScript!"""),
            title="[bold green]JS Specialist[/bold green]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, task: str) -> None:
        """Run autonomous JS code generation for a given task."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]Starting Autonomous JS Analysis[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        self._agent.reset()

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        self.console.print()

        if isinstance(result, JSCodeResult):
            result_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            result_table.add_column("Field", style="bold green")
            result_table.add_column("Value", style="white")

            result_table.add_row("Description", result.description)
            if result.session_storage_key:
                result_table.add_row("Session Storage Key", result.session_storage_key)
            result_table.add_row("Timeout", f"{result.timeout_seconds}s")
            result_table.add_row("JS Code", result.js_code[:500] + ("..." if len(result.js_code) > 500 else ""))

            self.console.print(Panel(
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

            self.console.print(Panel(
                failure_table,
                title=f"[bold red]JS Code Generation Failed[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            self.console.print(Panel(
                "[yellow]Could not finalize JS code generation. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]Analysis Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        self.console.print()


def main() -> None:
    """Run the JS Specialist agent interactively."""
    parser = argparse.ArgumentParser(
        description="JS Specialist - Interactive JavaScript code generation"
    )
    add_model_argument(parser)
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
    parser.add_argument(
        "--javascript-events-jsonl-path",
        type=str,
        default=None,
        help="Path to javascript_events.jsonl file for JS file analysis tools",
    )
    parser.add_argument(
        "--network-events-jsonl-path",
        type=str,
        default=None,
        help="Path to network_events.jsonl file for network traffic analysis tools",
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

    # Load JS data store if provided
    js_data_store: JSDataStore | None = None
    if args.javascript_events_jsonl_path:
        js_path = Path(args.javascript_events_jsonl_path)
        if not js_path.exists():
            console.print(f"[bold red]Error: JS data store file not found: {js_path}[/bold red]")
            sys.exit(1)

        try:
            js_data_store = JSDataStore(str(js_path))
            console.print(f"[dim]Loaded {js_data_store.stats.total_files} JS files from {js_path}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error loading JS data store: {e}[/bold red]")
            sys.exit(1)

    # Load network data store if provided
    network_data_store: NetworkDataStore | None = None
    if args.network_events_jsonl_path:
        network_path = Path(args.network_events_jsonl_path)
        if not network_path.exists():
            console.print(f"[bold red]Error: Network data store file not found: {network_path}[/bold red]")
            sys.exit(1)

        try:
            network_data_store = NetworkDataStore(str(network_path))
            console.print(f"[dim]Loaded {network_data_store.stats.total_requests} network requests from {network_path}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error loading network data store: {e}[/bold red]")
            sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalJSSpecialistChat(
        dom_snapshots=dom_snapshots,
        js_data_store=js_data_store,
        network_data_store=network_data_store,
        llm_model=llm_model,
        remote_debugging_address=args.remote_debugging_address,
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
