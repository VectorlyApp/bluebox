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
    bluebox-agent --model gpt-5.2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bluebox.utils.code_execution_sandbox import is_docker_available
from bluebox.utils.terminal_utils import ask_yes_no, print_colored, YELLOW

from rich.console import Console
from rich.text import Text
from textual import work
from textual.widgets import RichLog

from bluebox.agents.bluebox_agent import BlueBoxAgent, GenerateContextResult
from bluebox.agents.workspace import LocalWorkspace
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
    SLASH_COMMANDS = {
        **BASE_SLASH_COMMANDS,
        "/generate_context": "Save a reusable context file (optionally with a focus prompt)",
    }
    HELP_TEXT = BASE_HELP_TEXT + (
        "\n    [cyan]/generate_context[/cyan]  Save a reusable context file from this session"
        "\n                       Optionally add a focus: [cyan]/generate_context focus on the flight search part[/cyan]\n"
    )
    SHOW_SAVED_FILES_PANE = True

    def __init__(
        self,
        llm_model: LLMModel,
        workspace_dir: str = "./bluebox_workspace",
        context_file: str | None = None,
    ) -> None:
        super().__init__(llm_model, working_dir=workspace_dir)
        self._workspace_dir = workspace_dir
        self._context_file = context_file

    # ── Abstract implementations ─────────────────────────────────────────

    def _create_agent(self) -> AbstractAgent:
        return BlueBoxAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=LocalWorkspace(self._workspace_dir),
            context_file=self._context_file,
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
        if isinstance(self._agent, BlueBoxAgent) and self._agent.loaded_context:
            ctx = self._agent.loaded_context
            lines.append(f"[dim]Context:[/dim]     [green]loaded[/green] — {ctx.goal[:60]}")
            lines.append(f"[dim]             {len(ctx.routines_used)} routine(s), {len(ctx.output_files)} output file(s)[/dim]")
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
        seen: set[str] = set()
        paths: list[str] = []

        def _add(p: str) -> None:
            if p and p not in seen:
                seen.add(p)
                paths.append(f"📄 {p}")

        _add(tool_result.get("output_file", ""))
        for f in tool_result.get("files_created", []):
            _add(f)
        for r in tool_result.get("results", []):
            if isinstance(r, dict):
                _add(r.get("output_file", ""))
        return paths

    # ── Custom slash commands ─────────────────────────────────────────

    def _handle_custom_command(self, cmd: str, raw_input: str) -> bool:
        if raw_input.lower().startswith("/generate_context"):
            chat = self.query_one("#chat-log", RichLog)
            if not self._agent:
                chat.write(Text.from_markup("[red]Agent not initialized.[/red]"))
                return True

            user_focus = raw_input[len("/generate_context"):].strip() or None
            chat.write(Text.from_markup(
                "[yellow]Generating context from this session...[/yellow]"
            ))
            self._processing = True
            self._generate_context_async(user_focus)
            return True
        return False

    @work(thread=True)
    def _generate_context_async(self, focus: str | None) -> None:
        """Run generate_context in a background thread via structured output."""
        try:
            if not isinstance(self._agent, BlueBoxAgent):
                raise TypeError(f"Expected BlueBoxAgent, got {type(self._agent).__name__}")
            result = self._agent.generate_context(focus=focus)
            self.call_from_thread(self._show_context_success, result)
        except Exception as e:
            self.call_from_thread(self._show_context_error, str(e))

    def _show_context_success(self, result: GenerateContextResult) -> None:
        """Display context generation success in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(
            f"[bold green]Context saved![/bold green]\n"
            f"[dim]Goal:[/dim]      {result.context.goal}\n"
            f"[dim]Summary:[/dim]   {result.context.summary}\n"
            f"[dim]Routines:[/dim]  {len(result.context.routines_used)}\n"
            f"[dim]JSON:[/dim]      {result.json_path}\n"
            f"[dim]Markdown:[/dim]  {result.md_path}"
        ))
        self._processing = False
        self._update_status()

    def _show_context_error(self, error: str) -> None:
        """Display context generation error in the chat pane."""
        chat = self.query_one("#chat-log", RichLog)
        chat.write(Text.from_markup(f"[bold red]Context generation failed:[/bold red] {error}"))
        self._processing = False
        self._update_status()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the BlueBox Agent TUI."""
    parser = argparse.ArgumentParser(description="BlueBox Agent \u2014 Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default="./bluebox_workspace",
        help="Workspace directory. Raw results in raw/, output files in outputs/ (default: ./bluebox_workspace)",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        default=None,
        help="Path to a context file (.json or .md) from a previous session to guide the agent",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()

    # Validate required config
    if not Config.VECTORLY_SERVICE_TOKEN:
        console.print("[bold red]Error: VECTORLY_SERVICE_TOKEN is not set[/bold red]")
        sys.exit(1)

    llm_model = resolve_model(args.model, console)

    # Check if workspace directory exists and offer to clear it
    workspace_path = Path(args.workspace_dir)
    if workspace_path.exists() and any(workspace_path.iterdir()):
        print_colored(f"Workspace directory already exists: {workspace_path.resolve()}", YELLOW)
        if ask_yes_no("Clear workspace?"):
            shutil.rmtree(workspace_path)
            print_colored("Workspace cleared.", YELLOW)
        else:
            print_colored("Keeping existing workspace.", YELLOW)
        print()

    # Warn if Docker is not available (code execution will fall back to blocklist sandbox)
    if not is_docker_available():
        print_colored(
            "Warning: Docker is not available. Code execution will use the blocklist sandbox,\n"
            "which is less secure and has limited isolation.",
            YELLOW,
        )
        if not ask_yes_no("Continue without Docker?"):
            sys.exit(0)
        print()

    console.print(f"[dim]Model: {llm_model.value}[/dim]")
    console.print()

    # Redirect logging + stderr AFTER all console output, right before TUI takes over.
    enable_tui_logging(log_file=args.log_file or ".bluebox_browser_tui.log", quiet=args.quiet)

    app = BlueBoxAgentTUI(
        llm_model=llm_model,
        workspace_dir=args.workspace_dir,
        context_file=args.context_file,
    )
    app.run()


if __name__ == "__main__":
    main()
