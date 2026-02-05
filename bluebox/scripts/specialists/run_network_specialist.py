#!/usr/bin/env python3
"""
bluebox/scripts/specialists/run_network_specialist.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Interactive CLI for the NetworkSpecialist.

Usage:
    bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl
    bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl --model gpt-5.1
"""

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.specialists.network_specialist import (
    NetworkSpecialist,
    EndpointDiscoveryResult,
    DiscoveryFailureResult,
)
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.agents.terminal_agent_base import AbstractTerminalAgentChat
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

SLASH_COMMANDS = [
    ("/discover", "Discover API endpoints for a task — /discover <task>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]


BANNER = """\
[bold cyan]╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                              ║
║ ▄▄▄   ▄▄                                                    ▄▄                    ▄▄▄▄                       ║
║ ███   ██              ██                                    ██                  ▄█▀▀▀▀█                      ║
║ ██▀█  ██   ▄████▄   ███████  ██      ██  ▄████▄    ██▄████  ██ ▄██▀             ██▄       ██▄███▄   ▀██  ███ ║
║ ██ ██ ██  ██▄▄▄▄██    ██     ▀█  ██  █▀ ██▀  ▀██   ██▀      ██▄██                ▀████▄   ██▀  ▀██   ██▄ ██  ║
║ ██  █▄██  ██▀▀▀▀▀▀    ██      ██▄██▄██  ██    ██   ██       ██▀██▄                   ▀██  ██    ██    ████▀  ║
║ ██   ███  ▀██▄▄▄▄█    ██▄▄▄   ▀██  ██▀  ▀██▄▄██▀   ██       ██  ▀█▄             █▄▄▄▄▄█▀  ███▄▄██▀     ███   ║
║ ▀▀   ▀▀▀    ▀▀▀▀▀      ▀▀▀▀    ▀▀  ▀▀     ▀▀▀▀     ▀▀       ▀▀   ▀▀▀             ▀▀▀▀▀    ██ ▀▀▀       ██    ║
║                                                                                           ██         ███     ║
║                                                                                                              ║
╚═════════════════════════════════════════════════(beta)═══════════════════════════════════════════════════════╝[/bold cyan]
"""


class TerminalNetworkSpyChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the Network Spy Agent."""

    autonomous_command_name = "discover"

    def __init__(
        self,
        network_store: NetworkDataLoader,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        data_path: str = "",
    ) -> None:
        """Initialize the terminal chat interface."""
        self.network_store = network_store
        self.llm_model = llm_model
        self.data_path = data_path
        super().__init__(console=console, agent_color="cyan")

    def _create_agent(self) -> NetworkSpecialist:
        """Create the Network Spy agent instance."""
        return NetworkSpecialist(
            emit_message_callable=self._handle_message,
            network_data_store=self.network_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    def print_welcome(self) -> None:
        """Print welcome message with network stats."""
        self.console.print(BANNER)
        self.console.print()

        stats = self.network_store.stats

        # Build stats table
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Total Requests", str(stats.total_requests))
        stats_table.add_row("Unique URLs", str(stats.unique_urls))
        stats_table.add_row("Unique Hosts", str(stats.unique_hosts))

        # Methods breakdown
        methods_str = ", ".join(f"{m}: {c}" for m, c in sorted(stats.methods.items(), key=lambda x: -x[1]))
        stats_table.add_row("Methods", methods_str)

        # Status codes breakdown
        status_str = ", ".join(f"{s}: {c}" for s, c in sorted(stats.status_codes.items()))
        stats_table.add_row("Status Codes", status_str)

        # Features
        features = []
        if stats.has_cookies:
            features.append("🍪 Cookies")
        if stats.has_auth_headers:
            features.append("🔐 Auth Headers")
        if stats.has_json_requests:
            features.append("📦 JSON")
        if stats.has_form_data:
            features.append("📝 Form Data")
        if features:
            stats_table.add_row("Features", " ".join(features))

        # Top hosts
        top_hosts = sorted(stats.hosts.items(), key=lambda x: -x[1])[:5]
        if top_hosts:
            hosts_str = ", ".join(f"{h} ({c})" for h, c in top_hosts)
            stats_table.add_row("Top Hosts", hosts_str)

        self.console.print(Panel(
            stats_table,
            title=f"[bold cyan]Network Stats[/bold cyan] [dim]({self.data_path})[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Show host stats
        host_stats = self.network_store.get_host_stats()
        if host_stats:
            host_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            host_table.add_column("Host", style="white")
            host_table.add_column("Reqs", style="cyan", justify="right")
            host_table.add_column("Methods", style="dim")

            for hs in host_stats[:10]:  # Top 10 hosts
                methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(hs["methods"].items()))
                host_table.add_row(
                    hs["host"][:50] + "..." if len(hs["host"]) > 50 else hs["host"],
                    str(hs["request_count"]),
                    methods_str,
                )

            if len(host_stats) > 10:
                host_table.add_row(f"[dim]... and {len(host_stats) - 10} more hosts[/dim]", "", "")

            self.console.print(Panel(
                host_table,
                title=f"[bold magenta]📊 Host Statistics[/bold magenta] [dim]({len(host_stats)} hosts)[/dim]",
                border_style="magenta",
                box=box.ROUNDED,
            ))
            self.console.print()

        # Show likely API endpoints
        likely_urls = self.network_store.api_urls
        if likely_urls:
            urls_table = Table(box=None, show_header=False, padding=(0, 1))
            urls_table.add_column("URL", style="white")

            # Show up to 20 URLs
            for url in likely_urls[:20]:
                urls_table.add_row(f"• {url}")

            if len(likely_urls) > 20:
                urls_table.add_row(f"[dim]... and {len(likely_urls) - 20} more[/dim]")

            self.console.print(Panel(
                urls_table,
                title=f"[bold yellow]⚡ Likely API Endpoints[/bold yellow] [dim]({len(likely_urls)} found)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))
            self.console.print()

        self.console.print(Panel(
            """[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Discover API endpoints for a task
  [cyan]/reset[/cyan]             Start a new conversation
  [cyan]/help[/cyan]              Show help
  [cyan]/quit[/cyan]              Exit

Just ask questions about the network traffic!""",
            title="[bold cyan]Network Spy[/bold cyan]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, task: str) -> None:
        """Run autonomous endpoint discovery for a given task."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]🤖 Starting Autonomous Discovery[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous discovery with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        self.console.print()

        if isinstance(result, EndpointDiscoveryResult):
            # Success - build result tables for each endpoint
            endpoint_count = len(result.endpoints)

            for i, ep in enumerate(result.endpoints, 1):
                ep_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                ep_table.add_column("Field", style="bold cyan")
                ep_table.add_column("Value", style="white")

                ep_table.add_row("Request IDs", str(ep.request_ids))
                ep_table.add_row("URL", ep.url)
                ep_table.add_row("Inputs", ep.endpoint_inputs)
                ep_table.add_row("Outputs", ep.endpoint_outputs)

                if endpoint_count > 1:
                    self.console.print(Panel(
                        ep_table,
                        title=f"[bold green]Endpoint {i}/{endpoint_count}[/bold green]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))
                else:
                    self.console.print(Panel(
                        ep_table,
                        title=f"[bold green]✓ Endpoint Discovery Complete[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))

            if endpoint_count > 1:
                self.console.print(f"[bold green]✓ Found {endpoint_count} endpoints[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")

        elif isinstance(result, DiscoveryFailureResult):
            # Explicit failure - agent determined endpoint doesn't exist
            failure_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            failure_table.add_column("Field", style="bold red")
            failure_table.add_column("Value", style="white")

            failure_table.add_row("Reason", result.reason)
            if result.searched_terms:
                failure_table.add_row("Terms Searched", ", ".join(result.searched_terms[:15]))
            if result.closest_matches:
                failure_table.add_row("Closest Matches", "\n".join(result.closest_matches[:5]))

            self.console.print(Panel(
                failure_table,
                title=f"[bold red]✗ Endpoint Not Found[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            # None - max iterations without finalization
            self.console.print(Panel(
                "[yellow]Could not finalize endpoint discovery. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]⚠ Discovery Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        self.console.print()


def main() -> None:
    """Run the Network Spy agent interactively."""
    parser = argparse.ArgumentParser(
        description="Network Spy - Interactive network traffic analyzer"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing NetworkTransactionEvent entries",
    )
    add_model_argument(parser)
    args = parser.parse_args()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    # Parse JSONL into data store
    try:
        network_store = NetworkDataLoader(jsonl_path)
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalNetworkSpyChat(
        network_store=network_store,
        llm_model=llm_model,
        data_path=str(jsonl_path),
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
