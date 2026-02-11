"""
bluebox/scripts/specialists/run_js_specialist.py

Multi-pane terminal UI for the JSSpecialist using Textual.

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
    bluebox-js-specialist
    bluebox-js-specialist --dom-snapshots-path ./cdp_captures/dom/events.jsonl
    bluebox-js-specialist \
        --dom-snapshots-path ./cdp_captures/dom/events.jsonl \
        --javascript-events-jsonl-path ./cdp_captures/network/javascript_events.jsonl \
        --network-events-jsonl-path ./cdp_captures/network/events.jsonl \
        --remote-debugging-address 127.0.0.1:9222
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

from bluebox.agents.specialists.js_specialist import JSSpecialist
from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# --- Slash commands -----------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/autonomous": "Run autonomous JS code generation for a task",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/autonomous <task>[/cyan]  Run autonomous JS code generation
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# --- Textual App --------------------------------------------------------------

class JSSpecialistTUI(AbstractAgentTUI):
    """Multi-pane TUI for the JS Specialist."""

    TITLE = "JS Specialist"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: LLMModel,
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        js_data_loader: JSDataLoader | None = None,
        network_data_loader: NetworkDataLoader | None = None,
        remote_debugging_address: str | None = None,
    ) -> None:
        super().__init__(llm_model)
        self._dom_snapshots = dom_snapshots
        self._js_data_loader = js_data_loader
        self._network_data_loader = network_data_loader
        self._remote_debugging_address = remote_debugging_address

    # -- Abstract implementations ----------------------------------------------

    def _create_agent(self) -> AbstractAgent:
        return JSSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            dom_snapshots=self._dom_snapshots,
            js_data_loader=self._js_data_loader,
            network_data_loader=self._network_data_loader,
            llm_model=self._llm_model,
            remote_debugging_address=self._remote_debugging_address,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold green]JS Specialist[/bold green]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        dom_count = len(self._dom_snapshots) if self._dom_snapshots else 0
        js_files_count = self._js_data_loader.stats.total_files if self._js_data_loader else 0
        network_count = self._network_data_loader.stats.total_requests if self._network_data_loader else 0

        lines: list[str] = []
        if dom_count > 0:
            lines.append(f"[dim]DOM Snapshots:[/dim]    {dom_count}")
        if js_files_count > 0:
            lines.append(f"[dim]JS Files:[/dim]         {js_files_count}")
        if network_count > 0:
            lines.append(f"[dim]Network Requests:[/dim] {network_count}")
        if self._remote_debugging_address:
            lines.append(f"[dim]Remote Debug:[/dim]     {self._remote_debugging_address}")

        if lines:
            chat.write(Text.from_markup("\n".join(lines)))
            chat.write("")

        if not any([dom_count, js_files_count, network_count, self._remote_debugging_address]):
            chat.write(Text.from_markup(
                "[yellow]No data sources loaded. Use --dom-snapshots-path, "
                "--javascript-events-jsonl-path, --network-events-jsonl-path, "
                "or --remote-debugging-address to provide context.[/yellow]"
            ))
            chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask questions about JavaScript."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)

        dom_count = len(self._dom_snapshots) if self._dom_snapshots else 0
        js_count = self._js_data_loader.stats.total_files if self._js_data_loader else 0
        net_count = self._network_data_loader.stats.total_requests if self._network_data_loader else 0

        return (
            f"[bold green]JS SPECIALIST[/bold green]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]DOM snaps:[/dim] {dom_count}\n"
            f"[dim]JS files:[/dim]  {js_count}\n"
            f"[dim]Net reqs:[/dim]  {net_count}\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    # -- Custom commands -------------------------------------------------------

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if raw_input.lower().startswith("/autonomous"):
            task = raw_input[11:].strip()
            chat = self.query_one("#chat-log", RichLog)
            if not task:
                chat.write(Text.from_markup("[yellow]Usage: /autonomous <task>[/yellow]"))
            else:
                self._run_autonomous(task)
            return True
        return False

    # -- Autonomous run --------------------------------------------------------

    @work(thread=True)
    def _run_autonomous(self, task: str) -> None:
        """Run autonomous JS code generation in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Autonomous JS Analysis[/bold magenta]\n"
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
                output_str = json.dumps(result.output, indent=2)
                chat.write(Text.from_markup(
                    f"[bold green]\u2713 JS Code Generated[/bold green] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]"
                ))
                output_lines = output_str.split("\n")
                if len(output_lines) > 40:
                    output_str = "\n".join(output_lines[:40]) + f"\n... ({len(output_lines) - 40} more lines)"
                chat.write(output_str)

                self._add_tool_node(
                    Text.assemble(
                        ("JS RESULT", "green"),
                        " ",
                        (f"({iterations} iter, {elapsed:.1f}s)", "dim"),
                    ),
                    output_str.split("\n"),
                )

            elif isinstance(result, SpecialistResultWrapper) and not result.success:
                reason = result.failure_reason or "Unknown"
                chat.write(Text.from_markup(
                    f"[bold red]\u2717 JS Code Generation Failed[/bold red] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]\n"
                    f"[red]Reason:[/red] {escape(reason)}"
                ))
                if result.notes:
                    notes_str = "\n".join(f"  - {n}" for n in result.notes[:10])
                    chat.write(Text.from_markup(f"[dim]Notes:[/dim]\n{notes_str}"))

            else:
                chat.write(Text.from_markup(
                    f"[bold yellow]\u26a0 Analysis Incomplete[/bold yellow] "
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

        dom_count = len(self._dom_snapshots) if self._dom_snapshots else 0
        js_count = self._js_data_loader.stats.total_files if self._js_data_loader else 0
        net_count = self._network_data_loader.stats.total_requests if self._network_data_loader else 0

        chat.write(Text.from_markup(
            f"[bold green]Status[/bold green]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  DOM Snapshots: {dom_count}\n"
            f"  JS Files: {js_count}\n"
            f"  Network Requests: {net_count}"
        ))


# --- Entry point --------------------------------------------------------------

def main() -> None:
    """Entry point for the JS specialist TUI."""
    parser = argparse.ArgumentParser(description="JS Specialist \u2014 Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default=None,
        help="Chrome remote debugging address (e.g. 127.0.0.1:9222) for execute_js_in_browser tool",
    )
    parser.add_argument(
        "--dom-snapshots-path",
        type=str,
        default=None,
        help="Path to DOM snapshots events.jsonl file",
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
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

    # Load DOM snapshots if provided
    dom_snapshots: list[DOMSnapshotEvent] | None = None
    if args.dom_snapshots_path:
        dom_path = Path(args.dom_snapshots_path)
        if not dom_path.exists():
            console.print(f"[bold red]Error: DOM snapshots file not found: {dom_path}[/bold red]")
            sys.exit(1)

        dom_snapshots = []
        for line_num, line in enumerate(dom_path.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                dom_snapshots.append(DOMSnapshotEvent(**data))
            except Exception as e:
                console.print(f"[yellow]Warning: Could not parse line {line_num}: {e}[/yellow]")

        console.print(f"[green]\u2713 Loaded {len(dom_snapshots)} DOM snapshots from {dom_path}[/green]")

    # Load JS data store if provided
    js_data_loader: JSDataLoader | None = None
    if args.javascript_events_jsonl_path:
        js_path = Path(args.javascript_events_jsonl_path)
        if not js_path.exists():
            console.print(f"[bold red]Error: JS data store file not found: {js_path}[/bold red]")
            sys.exit(1)

        try:
            js_data_loader = JSDataLoader(str(js_path))
            console.print(f"[green]\u2713 Loaded {js_data_loader.stats.total_files} JS files from {js_path}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading JS data store: {e}[/bold red]")
            sys.exit(1)

    # Load network data store if provided
    network_data_loader: NetworkDataLoader | None = None
    if args.network_events_jsonl_path:
        network_path = Path(args.network_events_jsonl_path)
        if not network_path.exists():
            console.print(f"[bold red]Error: Network data store file not found: {network_path}[/bold red]")
            sys.exit(1)

        try:
            network_data_loader = NetworkDataLoader(str(network_path))
            console.print(f"[green]\u2713 Loaded {network_data_loader.stats.total_requests} network requests from {network_path}[/green]")
        except Exception as e:
            console.print(f"[bold red]Error loading network data store: {e}[/bold red]")
            sys.exit(1)

    llm_model = resolve_model(args.model, console)
    console.print()

    # Redirect logging + stderr right before TUI takes over
    enable_tui_logging(log_file=args.log_file or ".bluebox_js_tui.log", quiet=args.quiet)

    app = JSSpecialistTUI(
        llm_model=llm_model,
        dom_snapshots=dom_snapshots,
        js_data_loader=js_data_loader,
        network_data_loader=network_data_loader,
        remote_debugging_address=args.remote_debugging_address,
    )
    app.run()


if __name__ == "__main__":
    main()
