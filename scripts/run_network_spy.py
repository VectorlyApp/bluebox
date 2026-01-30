#!/usr/bin/env python3
"""
scripts/run_network_spy.py

Interactive CLI for the Network Spy agent.

Usage:
    python scripts/run_network_spy.py --jsonl-path ./cdp_captures/network/events.jsonl
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

from bluebox.agents.network_spy import NetworkSpyAgent, EndpointDiscoveryResult, DiscoveredEndpoint
from bluebox.llms.infra.network_data_store import NetworkDataStore
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
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝[/bold cyan]
"""


def print_welcome(model: str, data_path: str, network_store: NetworkDataStore) -> None:
    """Print welcome message with network stats."""
    console.print(BANNER)
    console.print()

    stats = network_store.stats

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

    console.print(Panel(
        stats_table,
        title=f"[bold cyan]Network Stats[/bold cyan] [dim]({data_path})[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))
    console.print()

    # Show host stats
    host_stats = network_store.get_host_stats()
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

        console.print(Panel(
            host_table,
            title=f"[bold magenta]📊 Host Statistics[/bold magenta] [dim]({len(host_stats)} hosts)[/dim]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        console.print()

    # Show likely API endpoints
    likely_urls = network_store.likely_api_urls()
    if likely_urls:
        urls_table = Table(box=None, show_header=False, padding=(0, 1))
        urls_table.add_column("URL", style="white")

        # Show up to 20 URLs
        for url in likely_urls[:20]:
            urls_table.add_row(f"• {url}")

        if len(likely_urls) > 20:
            urls_table.add_row(f"[dim]... and {len(likely_urls) - 20} more[/dim]")

        console.print(Panel(
            urls_table,
            title=f"[bold yellow]⚡ Likely API Endpoints[/bold yellow] [dim]({len(likely_urls)} found)[/dim]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        console.print()

    console.print(Panel(
        """[bold]Commands:[/bold]
  [cyan]/autonomous <task>[/cyan]  Run autonomous endpoint discovery
  [cyan]/reset[/cyan]              Start a new conversation
  [cyan]/help[/cyan]               Show help
  [cyan]/quit[/cyan]               Exit

Just ask questions about the network traffic!""",
        title="[bold cyan]Network Spy[/bold cyan]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
    ))
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
        title="[bold yellow]⚙ TOOL CALL[/bold yellow]",
        style="yellow",
        box=box.ROUNDED,
    ))


def print_tool_result(
    invocation: PendingToolInvocation,
    result: dict[str, Any] | None,
) -> None:
    """Print a tool invocation result."""
    if invocation.status == ToolInvocationStatus.EXECUTED:
        console.print("[bold green]✓ Tool executed[/bold green]")
        if result:
            result_json = json.dumps(result, indent=2)
            # Limit display to 150 lines
            lines = result_json.split("\n")
            if len(lines) > 150:
                display = "\n".join(lines[:150]) + f"\n... ({len(lines) - 150} more lines)"
            else:
                display = result_json
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]✗ Tool execution failed[/bold red]")
        error = result.get("error") if result else None
        if error:
            console.print(Panel(str(error), title="Error", style="red", box=box.ROUNDED))

    console.print()


class TerminalNetworkSpyChat:
    """Interactive terminal chat interface for the Network Spy Agent."""

    def __init__(
        self,
        network_store: NetworkDataStore,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._streaming_started: bool = False
        self._agent = NetworkSpyAgent(
            emit_message_callable=self._handle_message,
            network_data_store=network_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold cyan]Assistant[/bold cyan]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """Handle messages emitted by the Network Spy Agent."""
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()
                print()
                self._streaming_started = False
            else:
                print_assistant_message(message.content)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            # Show tool call and result
            print_tool_call(message.tool_invocation)
            print_tool_result(message.tool_invocation, message.tool_result)

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)

    def _run_autonomous(self, task: str) -> None:
        """Run autonomous endpoint discovery for a given task."""
        console.print()
        console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]🤖 Starting Autonomous Discovery[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous discovery with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        console.print()

        if result:
            # Build result tables for each endpoint
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
                    console.print(Panel(
                        ep_table,
                        title=f"[bold green]Endpoint {i}/{endpoint_count}[/bold green]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))
                else:
                    console.print(Panel(
                        ep_table,
                        title=f"[bold green]✓ Endpoint Discovery Complete[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))

            if endpoint_count > 1:
                console.print(f"[bold green]✓ Found {endpoint_count} endpoints[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")
        else:
            console.print(Panel(
                "[yellow]Could not finalize endpoint discovery. "
                "The agent reached max iterations without calling finalize_result.[/yellow]",
                title=f"[bold yellow]⚠ Discovery Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
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
                    console.print("[bold cyan]Goodbye![/bold cyan]")
                    console.print()
                    break

                if cmd == "/reset":
                    self._agent.reset()
                    console.print()
                    console.print("[yellow]↺ Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        """[bold]Commands:[/bold]
  [cyan]/autonomous <task>[/cyan]  Run autonomous endpoint discovery
                        Example: /autonomous find train prices from NYC to Boston
  [cyan]/reset[/cyan]              Start a new conversation
  [cyan]/help[/cyan]               Show this help message
  [cyan]/quit[/cyan]               Exit

[bold]Tips:[/bold]
  - Ask about specific endpoints, headers, or cookies
  - Request a summary of API calls
  - Ask about authentication patterns""",
                        title="[bold cyan]Help[/bold cyan]",
                        border_style="cyan",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                # Handle /autonomous command
                if user_input.strip().lower().startswith("/autonomous"):
                    task = user_input.strip()[len("/autonomous"):].strip()
                    if not task:
                        console.print()
                        console.print("[bold yellow]Usage:[/bold yellow] /autonomous <task description>")
                        console.print("[dim]Example: /autonomous find train prices from NYC to Boston[/dim]")
                        console.print()
                        continue

                    self._run_autonomous(task)
                    continue

                self._agent.process_new_message(user_input, ChatRole.USER)

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
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    args = parser.parse_args()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    # Parse JSONL into data store
    try:
        network_store = NetworkDataStore.from_jsonl(str(jsonl_path))
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    # Map model string to enum
    model_map = {
        "gpt-5.1": OpenAIModel.GPT_5_1,
    }
    llm_model = model_map.get(args.model, OpenAIModel.GPT_5_1)

    print_welcome(args.model, str(jsonl_path), network_store)

    chat = TerminalNetworkSpyChat(
        network_store=network_store,
        llm_model=llm_model,
    )
    chat.run()


if __name__ == "__main__":
    main()
