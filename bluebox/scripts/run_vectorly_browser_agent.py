#!/usr/bin/env python3
"""
bluebox/scripts/run_vectorly_browser_agent.py

Interactive CLI for the VectorlyBrowserAgent.

Usage:
    bluebox-vectorly-browser
    bluebox-vectorly-browser --model gpt-5.1
    bluebox-vectorly-browser --remote-debugging-address http://127.0.0.1:9222
"""

import argparse
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.vectorly_browser_agent import VectorlyBrowserAgent
from bluebox.config import Config
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.agents.terminal_agent_base import AbstractTerminalAgentChat
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

SLASH_COMMANDS = [
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]


BANNER = """\
[bold green]
██╗   ██╗███████╗ ██████╗████████╗ ██████╗ ██████╗ ██╗  ██╗   ██╗
██║   ██║██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗██║  ╚██╗ ██╔╝
██║   ██║█████╗  ██║        ██║   ██║   ██║██████╔╝██║   ╚████╔╝
╚██╗ ██╔╝██╔══╝  ██║        ██║   ██║   ██║██╔══██╗██║    ╚██╔╝
 ╚████╔╝ ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║███████╗██║
  ╚═══╝  ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝

██████╗ ██████╗  ██████╗ ██╗    ██╗███████╗███████╗██████╗      █████╗  ██████╗ ███████╗███╗   ██╗████████╗
██╔══██╗██╔══██╗██╔═══██╗██║    ██║██╔════╝██╔════╝██╔══██╗    ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
██████╔╝██████╔╝██║   ██║██║ █╗ ██║███████╗█████╗  ██████╔╝    ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
██╔══██╗██╔══██╗██║   ██║██║███╗██║╚════██║██╔══╝  ██╔══██╗    ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
██████╔╝██║  ██║╚██████╔╝╚███╔███╔╝███████║███████╗██║  ██║    ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═╝  ╚═╝    ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝
[/bold green]"""


class TerminalVectorlyBrowserChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the Vectorly Browser Agent."""

    autonomous_command_name = "run"  # Not used for now, but required by base class

    def __init__(
        self,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        routine_output_dir: str | None = None,
    ) -> None:
        """Initialize the terminal chat interface."""
        self.llm_model = llm_model
        self.remote_debugging_address = remote_debugging_address
        self.routine_output_dir = routine_output_dir
        super().__init__(console=console, agent_color="green")

    def _create_agent(self) -> VectorlyBrowserAgent:
        """Create the Vectorly Browser agent instance."""
        return VectorlyBrowserAgent(
            emit_message_callable=self._handle_message,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
            remote_debugging_address=self.remote_debugging_address,
            routine_output_dir=self.routine_output_dir,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    def print_welcome(self) -> None:
        """Print welcome message with connection info."""
        self.console.print(BANNER)
        self.console.print()

        # Build config table
        config_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        config_table.add_column("Label", style="dim")
        config_table.add_column("Value", style="white")

        config_table.add_row("Remote Debug", self.remote_debugging_address)
        config_table.add_row("Model", self.llm_model.value)

        self.console.print(Panel(
            config_table,
            title="[bold green]Configuration[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

        self.console.print(Panel(
            """[bold]Commands:[/bold]
  [green]/reset[/green]  Start a new conversation
  [green]/help[/green]   Show help
  [green]/quit[/green]   Exit

Ask me to execute a routine or help you find the right one!""",
            title="[bold green]Vectorly Browser Agent[/bold green]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, task: str) -> None:
        """Handle autonomous command (not implemented for this agent)."""
        self.console.print()
        self.console.print("[yellow]Autonomous mode is not available for this agent.[/yellow]")
        self.console.print("[dim]Use natural language to ask about or execute routines.[/dim]")
        self.console.print()


def main() -> None:
    """Run the Vectorly Browser agent interactively."""
    parser = argparse.ArgumentParser(
        description="Vectorly Browser Agent - Execute web automation routines"
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default="http://127.0.0.1:9222",
        help="Chrome remote debugging address (default: http://127.0.0.1:9222)",
    )
    parser.add_argument(
        "--routine-output-dir",
        type=str,
        default=None,
        help="Directory to save routine execution results as JSON files",
    )
    add_model_argument(parser)
    args = parser.parse_args()

    # Check required API configuration
    if not Config.VECTORLY_API_KEY:
        console.print("[bold red]Error:[/bold red] VECTORLY_API_KEY is not set")
        sys.exit(1)
    if not Config.VECTORLY_API_BASE:
        console.print("[bold red]Error:[/bold red] VECTORLY_API_BASE is not set")
        sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalVectorlyBrowserChat(
        llm_model=llm_model,
        remote_debugging_address=args.remote_debugging_address,
        routine_output_dir=args.routine_output_dir,
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
