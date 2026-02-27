"""
bluebox/scripts/specialists/run_dom_specialist.py

Multi-pane terminal UI for the DOMSpecialist using Textual.

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
    bluebox-dom-specialist --jsonl-path ./cdp_captures/dom/events.jsonl
    bluebox-dom-specialist --jsonl-path ./cdp_captures/dom/events.jsonl --model gpt-5.1
    bluebox-dom-specialist --jsonl-path ./cdp_captures/dom/events.jsonl --workspace-dir ./agent_workspace/dom_specialist
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

from bluebox.agents.specialists.dom_specialist import DOMSpecialist
from bluebox.workspace import LocalAgentWorkspace
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# --- Slash commands -----------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/discover": "Run autonomous DOM structure discovery for a task",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Run autonomous DOM structure discovery
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# --- Textual App --------------------------------------------------------------

class DOMSpecialistTUI(AbstractAgentTUI):
    """Multi-pane TUI for the DOM Specialist."""

    TITLE = "DOM Specialist"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: LLMModel,
        dom_data_loader: DOMDataLoader,
        data_path: str = "",
        workspace_dir: str | None = None,
    ) -> None:
        super().__init__(llm_model, working_dir=workspace_dir)
        self._dom_data_loader = dom_data_loader
        self._data_path = data_path
        self._workspace = LocalAgentWorkspace.from_directory_path(
            workspace_dir or "./agent_workspace/dom_specialist",
        )
        if self._data_path:
            self._workspace.attach_input_file("dom_events", self._data_path)

    # -- Abstract implementations ----------------------------------------------

    def _create_agent(self) -> AbstractAgent:
        return DOMSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            dom_data_loader=self._dom_data_loader,
            llm_model=self._llm_model,
            workspace=self._workspace,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold cyan]DOM Specialist[/bold cyan]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        stats = self._dom_data_loader.stats

        lines = [
            f"[dim]Total Snapshots:[/dim]  {stats.total_snapshots}",
            f"[dim]Unique URLs:[/dim]      {stats.unique_urls}",
            f"[dim]Unique Titles:[/dim]    {stats.unique_titles}",
            f"[dim]Total Strings:[/dim]    {stats.total_strings}",
        ]

        if stats.hosts:
            hosts_str = ", ".join(
                f"{h}: {c}" for h, c in sorted(stats.hosts.items(), key=lambda x: -x[1])
            )
            lines.append(f"[dim]Hosts:[/dim]           {hosts_str}")

        if self._data_path:
            lines.append(f"[dim]File:[/dim]            {self._data_path}")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        stats = self._dom_data_loader.stats

        return (
            f"[bold cyan]DOM[/bold cyan]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Snapshots:[/dim] {stats.total_snapshots}\n"
            f"[dim]URLs:[/dim]      {stats.unique_urls}\n"
            f"[dim]Strings:[/dim]   {stats.total_strings}\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    # -- Custom commands -------------------------------------------------------

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

    # -- Autonomous discovery --------------------------------------------------

    @work(thread=True)
    def _run_discovery(self, task: str) -> None:
        """Run autonomous DOM structure discovery in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold cyan]Starting Autonomous DOM Discovery[/bold cyan]\n"
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
                    f"[bold green]\u2713 DOM Discovery Complete[/bold green] "
                    f"[dim]({iterations} iterations, {elapsed:.1f}s)[/dim]"
                ))
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
                    f"[bold red]\u2717 DOM Discovery Failed[/bold red] "
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

    # -- Overrides -------------------------------------------------------------

    def _show_status_in_chat(self) -> None:
        """Show a compact status summary in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        stats = self._dom_data_loader.stats
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()

        chat.write(Text.from_markup(
            f"[bold cyan]Status[/bold cyan]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Snapshots: {stats.total_snapshots}\n"
            f"  URLs: {stats.unique_urls}\n"
            f"  Strings: {stats.total_strings}\n"
            f"  File: {self._data_path or 'N/A'}"
        ))


# --- Entry point --------------------------------------------------------------

def main() -> None:
    """Entry point for the DOM specialist TUI."""
    parser = argparse.ArgumentParser(description="DOM Specialist \u2014 Multi-pane TUI")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing DOM snapshot events",
    )
    add_model_argument(parser)
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./agent_workspace/dom_specialist",
        help="Workspace directory for tool results, artifacts, and code execution files.",
    )
    args = parser.parse_args()

    console = Console()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    try:
        dom_data_loader = DOMDataLoader(str(jsonl_path))
    except (ValueError, FileNotFoundError) as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    console.print(f"[green]\u2713 Loaded {dom_data_loader.stats.total_snapshots} DOM snapshots[/green]")
    console.print()

    # Redirect logging + stderr right before TUI takes over
    enable_tui_logging(log_file=args.log_file or ".bluebox_dom_tui.log", quiet=args.quiet)

    app = DOMSpecialistTUI(
        llm_model=llm_model,
        dom_data_loader=dom_data_loader,
        data_path=str(jsonl_path),
        workspace_dir=args.workspace_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
