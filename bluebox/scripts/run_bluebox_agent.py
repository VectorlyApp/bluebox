"""
bluebox/scripts/run_bluebox_agent.py

Multi-pane terminal UI for the BlueBoxAgent using Textual.

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
    bluebox-agent
    bluebox-agent --model gpt-5.1
"""

from __future__ import annotations

import sys
print("\n  \033[1;31mBlueBox Agent is still a work in progress. Please check back soon.\033[0m\n")
sys.exit(0)

import argparse
from datetime import datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.text import Text
from textual.widgets import RichLog

from bluebox.agents.bluebox_agent import BlueBoxAgent
from bluebox.config import Config
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI, BASE_SLASH_COMMANDS, BASE_HELP_TEXT

if TYPE_CHECKING:
    from bluebox.agents.abstract_agent import AbstractAgent


class BlueBoxAgentTUI(AbstractAgentTUI):
    """Multi-pane TUI for the BlueBox Agent."""

    TITLE = "BlueBox Agent"
    SLASH_COMMANDS = BASE_SLASH_COMMANDS
    HELP_TEXT = BASE_HELP_TEXT
    SHOW_SAVED_FILES_PANE = True

    def __init__(
        self,
        llm_model: LLMModel,
        routine_output_dir: str = "./routine_output",
    ) -> None:
        super().__init__(llm_model, working_dir=routine_output_dir)
        self._routine_output_dir = routine_output_dir

    # ── Abstract implementations ─────────────────────────────────────────

    def _create_agent(self) -> AbstractAgent:
        return BlueBoxAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            routine_output_dir=self._routine_output_dir,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            "[bold green]BlueBox Agent[/bold green]  "
            "[dim]powered by Vectorly[/dim]"
        ))
        chat.write("")

        lines = [
            f"[dim]Model:[/dim]       {self._llm_model.value}",
        ]
        chat.write(Text.from_markup("\n".join(lines)))
        chat.write("")

        chat.write(Text.from_markup(
            "Type [cyan]/help[/cyan] for commands, or ask me to browse the web "
            "or execute a routine."
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        return (
            f"[bold green]BlueBox Agent[/bold green]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500[/dim]\n"
            f"[dim]Time:[/dim]      {now}\n"
        )

    def _extract_tool_result_prefix_lines(self, tool_name: str, tool_result: Any) -> list[str]:
        """Surface output_file paths at the top of RESULT nodes."""
        if not isinstance(tool_result, dict):
            return []
        paths: list[str] = []
        if tool_result.get("output_file"):
            paths.append(f"📄 {tool_result['output_file']}")
        for r in tool_result.get("results", []):
            if isinstance(r, dict) and r.get("output_file"):
                paths.append(f"📄 {r['output_file']}")
        return paths


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the BlueBox Agent TUI."""
    parser = argparse.ArgumentParser(description="BlueBox Agent \u2014 Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--routine-output-dir",
        type=str,
        default="./routine_output",
        help="Directory to save routine execution results as JSON files (default: ./routine_output)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

    # Validate required config
    if not Config.VECTORLY_API_KEY:
        console.print("[bold red]Error: VECTORLY_API_KEY is not set[/bold red]")
        sys.exit(1)
    if not Config.VECTORLY_API_BASE:
        console.print("[bold red]Error: VECTORLY_API_BASE is not set[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    console.print(f"[dim]Model: {llm_model.value}[/dim]")
    console.print()

    # Redirect logging + stderr AFTER all console output, right before TUI takes over.
    enable_tui_logging(log_file=args.log_file or ".bluebox_browser_tui.log", quiet=args.quiet)

    app = BlueBoxAgentTUI(
        llm_model=llm_model,
        routine_output_dir=args.routine_output_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
