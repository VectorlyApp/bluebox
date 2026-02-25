"""
bluebox/scripts/run_dumb_agent.py

Multi-pane terminal UI for DumbAgent using Textual.

Usage:
    bluebox-dumb-agent
    bluebox-dumb-agent --workspace-dir ./agent_workspace/dumb_agent
"""

from __future__ import annotations

import argparse
from datetime import datetime

from rich.console import Console
from rich.text import Text
from textual.widgets import RichLog

from bluebox.agents.dumb_agent import DumbAgent
from bluebox.agents.workspace import LocalWorkspace
from bluebox.data_models.llms.vendors import LLMModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.logger import enable_tui_logging
from bluebox.utils.tui_base import AbstractAgentTUI


class DumbAgentTUI(AbstractAgentTUI):
    """Minimal TUI for DumbAgent."""

    TITLE = "Dumb Agent"

    def __init__(self, llm_model: LLMModel, workspace_dir: str | None = None) -> None:
        super().__init__(llm_model=llm_model, working_dir=workspace_dir)
        self._workspace_dir = workspace_dir
        self._workspace = (
            LocalWorkspace.from_directory_path(workspace_dir)
            if workspace_dir
            else None
        )

    def _create_agent(self) -> DumbAgent:
        return DumbAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            workspace=self._workspace,
            allow_code_execution=True,
        )

    def _print_welcome(self) -> None:
        chat = self.query_one("#chat-log", RichLog)
        mode = "workspace-scoped file I/O enabled" if self._workspace else "compute-only (no file I/O)"
        chat.write(Text.from_markup(
            "[bold green]Dumb Agent[/bold green]  [dim]minimal tool surface[/dim]"
        ))
        chat.write("")
        chat.write(Text.from_markup(
            f"[dim]Model:[/dim]      {self._llm_model.value}\n"
            f"[dim]Mode:[/dim]       {mode}\n"
            f"[dim]Workspace:[/dim]  {self._workspace_dir or '(none)'}"
        ))
        chat.write("")

    def _build_status_text(self) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        msg_count = len(self._agent.get_chats()) if self._agent else 0
        tokens_used, ctx_pct = self._estimate_context_usage()
        ctx_bar = self._context_bar(ctx_pct)
        mode = "workspace" if self._workspace else "compute-only"
        return (
            f"[bold green]DUMB AGENT[/bold green]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Model:[/dim]     {self._llm_model.value}\n"
            f"[dim]Mode:[/dim]      {mode}\n"
            f"[dim]Messages:[/dim]  {msg_count}\n"
            f"[dim]Tools:[/dim]     {self._tool_call_count}\n"
            f"[dim]Context:[/dim]   {ctx_bar}\n"
            f"[dim](est.)      ~{tokens_used:,} / {self._context_window_size:,}[/dim]\n"
            f"[dim]──────────────────[/dim]\n"
            f"[dim]Time:[/dim]      {now}\n"
        )


def main() -> None:
    """Entry point for the DumbAgent TUI."""
    parser = argparse.ArgumentParser(description="Dumb Agent — Multi-pane TUI")
    add_model_argument(parser)
    parser.add_argument(
        "--workspace-dir",
        type=str,
        default=None,
        help="Optional workspace directory. If omitted, agent runs compute-only with no file I/O.",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress logs")
    parser.add_argument("--log-file", type=str, default=None, help="Log to file")
    args = parser.parse_args()

    console = Console()
    llm_model = resolve_model(args.model, console)
    console.print(f"[dim]Model: {llm_model.value}[/dim]")
    if args.workspace_dir:
        console.print(f"[dim]Workspace: {args.workspace_dir}[/dim]")
    else:
        console.print("[dim]Workspace: (none, compute-only)[/dim]")
    console.print()

    enable_tui_logging(log_file=args.log_file or ".bluebox_dumb_tui.log", quiet=args.quiet)

    app = DumbAgentTUI(
        llm_model=llm_model,
        workspace_dir=args.workspace_dir,
    )
    app.run()


if __name__ == "__main__":
    main()
