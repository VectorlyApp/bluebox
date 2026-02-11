"""
bluebox/scripts/specialists/run_interaction_specialist.py

Multi-pane terminal UI for the InteractionSpecialist using Textual.

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
    bluebox-interaction-specialist --jsonl-path ./cdp_captures/interaction/events.jsonl
    bluebox-interaction-specialist --jsonl-path ./cdp_captures/interaction/events.jsonl --model gpt-5.1
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

from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.data_models.orchestration.result import SpecialistResultWrapper
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


# --- Slash commands -----------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/discover": "Run autonomous parameter discovery for a task",
    **BASE_SLASH_COMMANDS,
}

HELP_TEXT = """\
[bold]Commands:[/bold]
  [cyan]/discover <task>[/cyan]  Run autonomous parameter discovery
  [cyan]/status[/cyan]           Show current state
  [cyan]/chats[/cyan]            Show message history
  [cyan]/clear[/cyan]            Clear the chat display
  [cyan]/reset[/cyan]            Start new conversation
  [cyan]/help[/cyan]             Show this help
  [cyan]/quit[/cyan]             Exit
"""


# --- Textual App --------------------------------------------------------------

class InteractionSpecialistTUI(AbstractAgentTUI):
    """Multi-pane TUI for the Interaction Specialist."""

    TITLE = "Interaction Specialist"
    SLASH_COMMANDS = SLASH_COMMANDS
    HELP_TEXT = HELP_TEXT

    def __init__(
        self,
        llm_model: LLMModel,
        interaction_store: InteractionsDataLoader,
        data_path: str = "",
    ) -> None:
        super().__init__(llm_model)
        self._interaction_store = interaction_store
        self._data_path = data_path

    # -- Abstract implementations ----------------------------------------------

    def _create_agent(self) -> AbstractAgent:
        return InteractionSpecialist(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            interaction_data_loader=self._interaction_store,
            llm_model=self._llm_model,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold magenta]Interaction Specialist[/bold magenta]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        stats = self._interaction_store.stats

        lines = [
            f"[dim]Total Events:[/dim]     {stats.total_events}",
            f"[dim]Unique URLs:[/dim]      {stats.unique_urls}",
            f"[dim]Unique Elements:[/dim]  {stats.unique_elements}",
        ]

        if stats.events_by_type:
            types_str = ", ".join(
                f"{t}: {c}" for t, c in sorted(stats.events_by_type.items(), key=lambda x: -x[1])
            )
            lines.append(f"[dim]Events by Type:[/dim]  {types_str}")

        if self._data_path:
            lines.append(f"[dim]File:[/dim]            {self._data_path}")

        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask questions about the user interactions."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        stats = self._interaction_store.stats

        return (
            f"[bold magenta]INTERACTION[/bold magenta]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Events:[/dim]    {stats.total_events}\n"
            f"[dim]URLs:[/dim]      {stats.unique_urls}\n"
            f"[dim]Elements:[/dim]  {stats.unique_elements}\n"
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
        """Run autonomous parameter discovery in a background thread."""
        chat = self.query_one("#chat-log", RichLog)

        self.call_from_thread(
            lambda: chat.write(Text.from_markup(
                f"\n[bold magenta]Starting Autonomous Parameter Discovery[/bold magenta]\n"
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
                    f"[bold green]\u2713 Parameter Discovery Complete[/bold green] "
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
                    f"[bold red]\u2717 Parameter Discovery Failed[/bold red] "
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
        stats = self._interaction_store.stats
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()

        chat.write(Text.from_markup(
            f"[bold magenta]Status[/bold magenta]\n"
            f"  Model: {self._llm_model.value}\n"
            f"  Messages: {msg_count}\n"
            f"  Context: ~{tokens_used:,}t ({ctx_pct:.0f}%)\n"
            f"  Events: {stats.total_events}\n"
            f"  URLs: {stats.unique_urls}\n"
            f"  Elements: {stats.unique_elements}\n"
            f"  File: {self._data_path or 'N/A'}"
        ))


# --- Entry point --------------------------------------------------------------

def main() -> None:
    """Entry point for the interaction specialist TUI."""
    parser = argparse.ArgumentParser(description="Interaction Specialist \u2014 Multi-pane TUI")
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing interaction events",
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
        interaction_store = InteractionsDataLoader.from_jsonl(str(jsonl_path))
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    console.print(f"[green]\u2713 Loaded {interaction_store.stats.total_events} interaction events[/green]")
    console.print()

    # Redirect logging + stderr right before TUI takes over
    enable_tui_logging(log_file=args.log_file or ".bluebox_interaction_tui.log", quiet=args.quiet)

    app = InteractionSpecialistTUI(
        llm_model=llm_model,
        interaction_store=interaction_store,
        data_path=str(jsonl_path),
    )
    app.run()


if __name__ == "__main__":
    main()
