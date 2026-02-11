"""
bluebox/scripts/specialists/run_value_trace_resolver_specialist.py

Multi-pane terminal UI for the ValueTraceResolverSpecialist using Textual.

Layout:
  +-----------------------------+----------------------+
  |                             |  Tool Calls History   |
  |       Chat (scrolling)      |                       |
  |                             +----------------------+
  |  +------------------------+ |  Status / Stats       |
  |  | Input                  | |                       |
  |  +------------------------+ |                       |
  +-----------------------------+----------------------+

Usage:
    bluebox-value-trace-resolver-specialist --network-jsonl ./cdp_captures/network/events.jsonl
    bluebox-value-trace-resolver-specialist --storage-jsonl ./cdp_captures/storage/events.jsonl
    bluebox-value-trace-resolver-specialist \
        --network-jsonl ./cdp_captures/network/events.jsonl \
        --storage-jsonl ./cdp_captures/storage/events.jsonl \
        --window-props-jsonl ./cdp_captures/window_properties/events.jsonl
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

from bluebox.agents.specialists.value_trace_resolver_specialist import ValueTraceResolverSpecialist
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# --- Slash commands -----------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/trace": "Trace where a token/value came from",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/trace <value>[/cyan]    Trace where a token/value came from
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# --- Textual App --------------------------------------------------------------

class ValueTraceResolverTUI(AbstractAgentTUI):
    """Multi-pane TUI for the Value Trace Resolver."""

    TITLE = "Trace Hound"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: LLMModel,
        network_store: NetworkDataLoader | None = None,
        storage_store: StorageDataLoader | None = None,
        window_store: WindowPropertyDataLoader | None = None,
    ) -> None:
        super().__init__(llm_model)
        self._network_store = network_store
        self._storage_store = storage_store
        self._window_store = window_store

    # -- Abstract implementations ----------------------------------------------

    def _create_agent(self) -> AbstractAgent:
        return ValueTraceResolverSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            network_data_loader=self._network_store,
            storage_data_loader=self._storage_store,
            window_property_data_loader=self._window_store,
            llm_model=self._llm_model,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold magenta]Trace Hound[/bold magenta]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        lines: list[str] = []

        # Network stats
        if self._network_store:
            stats = self._network_store.stats
            lines.append(
                f"[dim]Network:[/dim]    [green]Loaded[/green]  "
                f"{stats.total_requests} requests, {stats.unique_urls} URLs, {stats.unique_hosts} hosts"
            )
        else:
            lines.append("[dim]Network:[/dim]    [yellow]Not loaded[/yellow]")

        # Storage stats
        if self._storage_store:
            stats = self._storage_store.stats
            lines.append(
                f"[dim]Storage:[/dim]    [green]Loaded[/green]  "
                f"{stats.total_events} events (cookies: {stats.cookie_events}, "
                f"localStorage: {stats.local_storage_events}, "
                f"sessionStorage: {stats.session_storage_events})"
            )
        else:
            lines.append("[dim]Storage:[/dim]    [yellow]Not loaded[/yellow]")

        # Window properties stats
        if self._window_store:
            stats = self._window_store.stats
            lines.append(
                f"[dim]Window:[/dim]     [green]Loaded[/green]  "
                f"{stats.total_events} events, {stats.total_changes} changes, "
                f"{stats.unique_property_paths} unique paths"
            )
        else:
            lines.append("[dim]Window:[/dim]     [yellow]Not loaded[/yellow]")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask questions about where values came from."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)

        net_reqs = self._network_store.stats.total_requests if self._network_store else 0
        stor_evts = self._storage_store.stats.total_events if self._storage_store else 0
        win_evts = self._window_store.stats.total_events if self._window_store else 0

        return (
            f"[bold magenta]TRACE HOUND[/bold magenta]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Net reqs:[/dim]  {net_reqs}\n"
            f"[dim]Storage:[/dim]   {stor_evts} events\n"
            f"[dim]Window:[/dim]    {win_evts} events\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    # -- Custom commands -------------------------------------------------------

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if raw_input.lower().startswith("/trace"):
            value = raw_input[6:].strip()
            chat = self.query_one("#chat-log", RichLog)
            if not value:
                chat.write(Text.from_markup("[yellow]Usage: /trace <value>[/yellow]"))
            else:
                self._run_trace(value)
            return True
        return False

    # -- Autonomous trace ------------------------------------------------------

    @work(thread=True)
    def _run_trace(self, value: str) -> None:
        """Run autonomous token tracing in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        display_value = value[:100] + ("..." if len(value) > 100 else "")
        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Token Trace[/bold magenta]\n"
                f"[dim]Value:[/dim] {escape(display_value)}"
            ))
        )

        self._agent.reset()
        self._last_seen_chat_count = 0

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(value)
        elapsed = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        def _show_result() -> None:
            chat.write("")

            if isinstance(result, SpecialistResultWrapper) and result.success and result.output:
                output_str = json.dumps(result.output, indent=2)
                chat.write(Text.from_markup(
                    f"[bold green]\u2713 Token Trace Complete[/bold green] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]"
                ))
                output_lines = output_str.split("\n")
                if len(output_lines) > 40:
                    output_str = "\n".join(output_lines[:40]) + f"\n... ({len(output_lines) - 40} more lines)"
                chat.write(output_str)

                self._add_tool_node(
                    Text.assemble(
                        ("TRACE RESULT", "green"),
                        " ",
                        (f"({iterations} iter, {elapsed:.1f}s)", "dim"),
                    ),
                    output_str.split("\n"),
                )

            elif isinstance(result, SpecialistResultWrapper) and not result.success:
                reason = result.failure_reason or "Unknown"
                chat.write(Text.from_markup(
                    f"[bold red]\u2717 Value Not Found[/bold red] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    f"[red]Reason:[/red] {escape(reason)}"
                ))
                if result.notes:
                    notes_str = "\n".join(f"  - {n}" for n in result.notes[:10])
                    chat.write(Text.from_markup(f"[dim]Notes:[/dim]\n{notes_str}"))

            else:
                chat.write(Text.from_markup(
                    f"[bold yellow]\u26a0 Trace Incomplete[/bold yellow] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    "[yellow]Agent reached max iterations without finalizing.[/yellow]"
                ))

            chat.write("")
            self._update_status()

        self.call_from_thread(_show_result)

    # -- Overrides -------------------------------------------------------------

    def _show_status_in_chat(self) -> None:
        """Show a compact status summary in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()

        net_reqs = self._network_store.stats.total_requests if self._network_store else 0
        stor_evts = self._storage_store.stats.total_events if self._storage_store else 0
        win_evts = self._window_store.stats.total_events if self._window_store else 0

        chat.write(Text.from_markup(
            f"[bold magenta]Status[/bold magenta]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Network Requests: {net_reqs}\n"
            f"  Storage Events: {stor_evts}\n"
            f"  Window Events: {win_evts}"
        ))


# --- Entry point --------------------------------------------------------------

def main() -> None:
    """Entry point for the value trace resolver TUI."""
    parser = argparse.ArgumentParser(description="Trace Hound \u2014 Multi-pane TUI")
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
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

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
            console.print(f"[green]\u2713 Loaded {network_store.stats.total_requests} network requests[/green]")
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
            console.print(f"[green]\u2713 Loaded {storage_store.stats.total_events} storage events[/green]")
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
            console.print(f"[green]\u2713 Loaded {window_store.stats.total_events} window property events[/green]")
        except ValueError as e:
            console.print(f"[bold red]Error parsing window props JSONL: {e}[/bold red]")
            sys.exit(1)

    llm_model = resolve_model(args.model, console)
    console.print()

    # Redirect logging + stderr right before TUI takes over
    enable_tui_logging(log_file=args.log_file or ".bluebox_trace_tui.log", quiet=args.quiet)

    app = ValueTraceResolverTUI(
        llm_model=llm_model,
        network_store=network_store,
        storage_store=storage_store,
        window_store=window_store,
    )
    app.run()


if __name__ == "__main__":
    main()
