#!/usr/bin/env python3
"""
scripts/specialists/interaction_specialist.py

Interactive CLI for the Interaction Specialist agent.

Usage:
    python scripts/specialists/interaction_specialist.py --jsonl-path ./cdp_captures/interaction/events.jsonl
"""

import argparse
import sys
import time
from pathlib import Path
from textwrap import dedent

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.specialists.abstract_specialist import RunMode
from bluebox.agents.specialists.interaction_specialist import (
    InteractionSpecialist,
    ParameterDiscoveryResult,
    ParameterDiscoveryFailureResult,
)
from bluebox.llms.infra.interactions_data_store import InteractionsDataStore
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.terminal_agent_base import AbstractTerminalAgentChat
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

SLASH_COMMANDS = [
    ("/autonomous", "Run autonomous parameter discovery; /autonomous <task>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]


BANNER = """\
[bold magenta]╔════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                        ║
║  ██╗███╗   ██╗████████╗███████╗██████╗  █████╗  ██████╗████████╗██╗ ██████╗ ███╗   ██╗ ║
║  ██║████╗  ██║╚══██╔══╝██╔════╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║ ║
║  ██║██╔██╗ ██║   ██║   █████╗  ██████╔╝███████║██║        ██║   ██║██║   ██║██╔██╗ ██║ ║
║  ██║██║╚██╗██║   ██║   ██╔══╝  ██╔══██╗██╔══██║██║        ██║   ██║██║   ██║██║╚██╗██║ ║
║  ██║██║ ╚████║   ██║   ███████╗██║  ██║██║  ██║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║ ║
║  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ║
║  ███████╗██████╗ ███████╗ ██████╗██╗ █████╗ ██╗     ██╗███████╗████████╗               ║
║  ██╔════╝██╔══██╗██╔════╝██╔════╝██║██╔══██╗██║     ██║██╔════╝╚══██╔══╝               ║
║  ███████╗██████╔╝█████╗  ██║     ██║███████║██║     ██║███████╗   ██║                  ║
║  ╚════██║██╔═══╝ ██╔══╝  ██║     ██║██╔══██║██║     ██║╚════██║   ██║                  ║
║  ███████║██║     ███████╗╚██████╗██║██║  ██║███████╗██║███████║   ██║                  ║
║  ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝                  ║
║                                                                                        ║
╚════════════════════════════════════════(BETA)══════════════════════════════════════════╝[/bold magenta]
"""


class TerminalInteractionSpecialistChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the Interaction Specialist Agent."""

    def __init__(
        self,
        interaction_store: InteractionsDataStore,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        data_path: str = "",
    ) -> None:
        """Initialize the terminal chat interface."""
        self.interaction_store = interaction_store
        self.llm_model = llm_model
        self.data_path = data_path
        super().__init__(console=console, agent_color="magenta")

    def _create_agent(self) -> InteractionSpecialist:
        """Create the Interaction Specialist agent instance."""
        return InteractionSpecialist(
            emit_message_callable=self._handle_message,
            interaction_data_store=self.interaction_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
            run_mode=RunMode.CONVERSATIONAL,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    @property
    def autonomous_command_name(self) -> str:
        """Return the autonomous command name."""
        return "autonomous"

    def print_welcome(self) -> None:
        """Print welcome message with interaction stats."""
        self.console.print(BANNER)
        self.console.print()

        stats = self.interaction_store.stats

        # Build stats table
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Total Events", str(stats.total_events))
        stats_table.add_row("Unique URLs", str(stats.unique_urls))
        stats_table.add_row("Unique Elements", str(stats.unique_elements))

        # Events by type breakdown
        if stats.events_by_type:
            types_str = ", ".join(f"{t}: {c}" for t, c in sorted(stats.events_by_type.items(), key=lambda x: -x[1]))
            stats_table.add_row("Events by Type", types_str)

        self.console.print(Panel(
            stats_table,
            title=f"[bold magenta]Interaction Stats[/bold magenta] [dim]({self.data_path})[/dim]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        self.console.print(Panel(
            dedent("""\
                [bold]Commands:[/bold]
                  [cyan]/autonomous <task>[/cyan]  Run autonomous parameter discovery
                  [cyan]/reset[/cyan]              Start a new conversation
                  [cyan]/help[/cyan]               Show help
                  [cyan]/quit[/cyan]               Exit

                Just ask questions about the user interactions!"""),
            title="[bold magenta]Interaction Specialist[/bold magenta]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, task: str) -> None:
        """Run autonomous parameter discovery for a given task."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]Starting Autonomous Parameter Discovery[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        self.console.print()

        self._agent.reset()

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        self.console.print()

        if isinstance(result, ParameterDiscoveryResult):
            param_count = len(result.parameters)

            for i, param in enumerate(result.parameters, 1):
                param_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
                param_table.add_column("Field", style="bold green")
                param_table.add_column("Value", style="white")

                param_table.add_row("Name", param.name)
                param_table.add_row("Type", param.type)
                param_table.add_row("Description", param.description)
                if param.examples:
                    param_table.add_row("Examples", ", ".join(param.examples[:5]))
                if param.source_element_css_path:
                    param_table.add_row("CSS Path", param.source_element_css_path)
                if param.source_element_tag:
                    param_table.add_row("Element Tag", param.source_element_tag)

                self.console.print(Panel(
                    param_table,
                    title=f"[bold green]Parameter {i}/{param_count}[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                ))

            self.console.print(f"[bold green]Found {param_count} parameters[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")

        elif isinstance(result, ParameterDiscoveryFailureResult):
            failure_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            failure_table.add_column("Field", style="bold red")
            failure_table.add_column("Value", style="white")

            failure_table.add_row("Reason", result.reason)
            failure_table.add_row("Interaction Summary", result.interaction_summary)

            self.console.print(Panel(
                failure_table,
                title=f"[bold red]Parameter Discovery Failed[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            self.console.print(Panel(
                "[yellow]Could not finalize parameter discovery. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]Discovery Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        self.console.print()


def main() -> None:
    """Run the Interaction Specialist agent interactively."""
    parser = argparse.ArgumentParser(
        description="Interaction Specialist - Interactive user interaction analyzer"
    )
    parser.add_argument(
        "--jsonl-path",
        type=str,
        required=True,
        help="Path to the JSONL file containing interaction events",
    )
    add_model_argument(parser)
    args = parser.parse_args()

    # Load JSONL file
    jsonl_path = Path(args.jsonl_path)
    if not jsonl_path.exists():
        console.print(f"[bold red]Error: JSONL file not found: {jsonl_path}[/bold red]")
        sys.exit(1)

    console.print(f"[dim]Loading JSONL file: {jsonl_path}[/dim]")

    try:
        interaction_store = InteractionsDataStore.from_jsonl(str(jsonl_path))
    except ValueError as e:
        console.print(f"[bold red]Error parsing JSONL file: {e}[/bold red]")
        sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalInteractionSpecialistChat(
        interaction_store=interaction_store,
        llm_model=llm_model,
        data_path=str(jsonl_path),
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
