"""
bluebox/scripts/specialists/run_network_specialist.py

Multi-pane terminal UI for the NetworkSpecialist using Textual.

Layout:
  ┌─────────────────────────────┬──────────────────────┐
  │                             │  Tool Calls History   │
  │       Chat (scrolling)      │                       │
  │                             ├──────────────────────┤
  │  ┌────────────────────────┐ │  Status / Stats       │
  │  │ Input                  │ │                       │
  │  └────────────────────────┘ │                       │
  └─────────────────────────────┴──────────────────────┘

Usage:
    bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl
    bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl --model gpt-5.1
    bluebox-network-specialist --jsonl-path ./cdp_captures/network/events.jsonl --model claude-sonnet-4-5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.widgets import RichLog

from bluebox.agents.specialists.network_specialist import NetworkSpecialist
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# ─── Slash commands ──────────────────────────────────────────────────────────

SLASH_COMMANDS: dict[str, str] = {
    "/discover": "Discover API endpoints for a task",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Discover API endpoints for a task
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# ─── Textual App ─────────────────────────────────────────────────────────────

class NetworkSpecialistTUI(AbstractAgentTUI):
    """Multi-pane TUI for the Network Specialist."""

    TITLE = "Network specialist"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: LLMModel,
        network_store: NetworkDataLoader,
        data_path: str = "",
    ) -> None:
        super().__init__(llm_model)
        self._network_store = network_store
        self._data_path = data_path

    # ── Abstract implementations ─────────────────────────────────────────

    def _create_agent(self) -> AbstractAgent:
        return NetworkSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            network_data_loader=self._network_store,
            llm_model=self._llm_model,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold cyan]Network Spy[/bold cyan]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        stats = self._network_store.stats

        # Stats summary
        lines = [
            f"[dim]Total Requests:[/dim] {stats.total_requests}",
            f"[dim]Unique URLs:[/dim]    {stats.unique_urls}",
            f"[dim]Unique Hosts:[/dim]   {stats.unique_hosts}",
        ]

        # Methods
        if stats.methods:
            methods_str = ", ".join(f"{m}: {c}" for m, c in sorted(stats.methods.items(), key=lambda x: -x[1]))
            lines.append(f"[dim]Methods:[/dim]        {methods_str}")

        # Features
        features = []
        if stats.has_cookies:
            features.append("Cookies")
        if stats.has_auth_headers:
            features.append("Auth Headers")
        if stats.has_json_requests:
            features.append("JSON")
        if stats.has_form_data:
            features.append("Form Data")
        if features:
            lines.append(f"[dim]Features:[/dim]       {', '.join(features)}")

        if self._data_path:
            lines.append(f"[dim]File:[/dim]           {self._data_path}")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

        # Show top hosts
        host_stats = self._network_store.get_host_stats()
        if host_stats:
            host_lines = ["[dim]Top Hosts:[/dim]"]
            for hs in host_stats[:8]:
                methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(hs["methods"].items()))
                host = hs["host"][:45] + "..." if len(hs["host"]) > 45 else hs["host"]
                host_lines.append(f"  {host} ({hs['request_count']} reqs, {methods_str})")
            if len(host_stats) > 8:
                host_lines.append(f"  [dim]... and {len(host_stats) - 8} more hosts[/dim]")
            chat.write(Text.from_markup("\n".join(host_lines)))
            chat.write("")

        # Show likely API endpoints
        likely_urls = self._network_store.api_urls
        if likely_urls:
            url_lines = [f"[dim]Likely API Endpoints ({len(likely_urls)}):[/dim]"]
            for url in likely_urls[:15]:
                url_lines.append(f"  [yellow]{escape(url)}[/yellow]")
            if len(likely_urls) > 15:
                url_lines.append(f"  [dim]... and {len(likely_urls) - 15} more[/dim]")
            chat.write(Text.from_markup("\n".join(url_lines)))
            chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask questions about the network traffic."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        stats = self._network_store.stats

        return (
            f"[bold cyan]NETWORK SPY[/bold cyan]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Requests:[/dim]  {stats.total_requests}\n"
            f"[dim]URLs:[/dim]      {stats.unique_urls}\n"
            f"[dim]Hosts:[/dim]     {stats.unique_hosts}\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    # ── Custom commands ──────────────────────────────────────────────────

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if raw_input.lower().startswith("/discover"):
            task = raw_input[9:].strip()
            chat = self.query_one("#chat-log", RichLog)
            if not task:
                chat.write(Text.from_markup("[yellow]Usage: /discover <task>[/yellow]"))
            else:
                self._run_discovery(task)
            return True
        return False

    # ── Autonomous discovery ─────────────────────────────────────────────

    @work(thread=True)
    def _run_discovery(self, task: str) -> None:
        """Run autonomous endpoint discovery in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Autonomous Discovery[/bold magenta]\n"
                f"[dim]Task:[/dim] {escape(task)}"
            ))
        )

        self._agent.reset()
        self._last_seen_chat_count = 0

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        def _show_result() -> None:
            chat.write("")

            if isinstance(result, SpecialistResultWrapper) and result.success and result.output:
                output = result.output
                chat.write(Text.from_markup(
                    f"[bold green]\u2713 Discovery Complete[/bold green] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]"
                ))
                output_str = json.dumps(output, indent=2)
                output_lines = output_str.split("\n")
                if len(output_lines) > 40:
                    output_str = "\n".join(output_lines[:40]) + f"\n... ({len(output_lines) - 40} more lines)"
                chat.write(output_str)

                self._add_tool_node(
                    Text.assemble(
                        ("DISCOVERY RESULT", "green"),
                        " ",
                        (f"({iterations} iter, {elapsed:.1f}s)", "dim"),
                    ),
                    output_str.split("\n"),
                )

            elif isinstance(result, SpecialistResultWrapper) and not result.success:
                reason = result.failure_reason or "Unknown"
                chat.write(Text.from_markup(
                    f"[bold red]\u2717 Endpoint Not Found[/bold red] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    f"[red]Reason:[/red] {escape(reason)}"
                ))
                if result.notes:
                    notes_str = "\n".join(f"  - {n}" for n in result.notes[:10])
                    chat.write(Text.from_markup(f"[dim]Notes:[/dim]\n{notes_str}"))

            else:
                chat.write(Text.from_markup(
                    f"[bold yellow]\u26a0 Discovery Incomplete[/bold yellow] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    "[yellow]Agent reached max iterations without finalizing.[/yellow]"
                ))

            chat.write("")
            self._update_status()

        self.call_from_thread(_show_result)

    # ── Overrides ────────────────────────────────────────────────────────

    def _show_status_in_chat(self) -> None:
        """Show a compact status summary in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        stats = self._network_store.stats
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()

        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Requests: {stats.total_requests}\n"
            f"  URLs: {stats.unique_urls}\n"
            f"  Hosts: {stats.unique_hosts}\n"
            f"  File: {self._data_path or 'N/A'}"
        ))


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the network specialist TUI."""
    parser = argparse.ArgumentParser(description="Network Spy \u2014 Multi-pane TUI")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing NetworkTransactionEvent entries",
    )
    add_model_argument(parser)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    try:
        network_store = NetworkDataLoader(jsonl_path)
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    console.print(f"[green]\u2713 Loaded {network_store.stats.total_requests} requests[/green]")
    console.print()

    # Redirect logging + stderr AFTER all console output, right before TUI takes over.
    enable_tui_logging(log_file=args.log_file or ".bluebox_network_tui.log", quiet=args.quiet)

    app = NetworkSpecialistTUI(
        llm_model=llm_model,
        network_store=network_store,
        data_path=str(jsonl_path),
    )
    app.run()


if __name__ == "__main__":
    main()
