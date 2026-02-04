"""
scripts/specialists/interaction_specialist.py

Interactive CLI for the Interaction Specialist agent.

Usage:
    python scripts/specialists/interaction_specialist.py --jsonl-path ./cdp_captures/interaction/events.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path
from textwrap import dedent
from typing import Any

from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.formatted_text import HTML
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bluebox.utils.terminal_utils import SlashCommandCompleter, SlashCommandLexer
from bluebox.agents.specialists.abstract_specialist import RunMode
from bluebox.agents.specialists.interaction_specialist import (
    InteractionSpecialist,
    ParameterDiscoveryResult,
    ParameterDiscoveryFailureResult,
)
from bluebox.llms.infra.interactions_data_store import InteractionsDataStore
from bluebox.data_models.llms.interaction import (
    ChatRole,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import (
    LLMModel,
    OpenAIModel,
    get_all_model_values,
    get_model_by_value,
)
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


def print_welcome(model: str, data_path: str, interaction_store: InteractionsDataStore) -> None:
    """Print welcome message with interaction stats."""
    console.print(BANNER)
    console.print()

    stats = interaction_store.stats

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

    console.print(Panel(
        stats_table,
        title=f"[bold magenta]Interaction Stats[/bold magenta] [dim]({data_path})[/dim]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()

    console.print(Panel(
        dedent("""\
            [bold]Commands:[/bold]
              [cyan]/autonomous <task>[/cyan]  Run autonomous parameter discovery
              [cyan]/reset[/cyan]              Start a new conversation
              [cyan]/help[/cyan]               Show help
              [cyan]/quit[/cyan]               Exit

            Just ask questions about the user interactions!"""),
        title="[bold magenta]Interaction Specialist[/bold magenta]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="magenta",
        box=box.ROUNDED,
    ))
    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold magenta]Assistant[/bold magenta]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str) -> None:
    """Print an error message."""
    console.print()
    console.print(f"[bold red]Error:[/bold red] [red]{escape(error)}[/red]")
    console.print()


def print_tool_call(invocation: PendingToolInvocation) -> None:
    """Print a tool call with formatted arguments."""
    args_formatted = json.dumps(invocation.tool_arguments, indent=2)

    content = Text()
    content.append("Tool: ", style="dim")
    content.append(invocation.tool_name, style="bold white")
    content.append("\n\n")
    content.append("Arguments:\n", style="dim")
    content.append(args_formatted, style="white")

    console.print()
    console.print(Panel(
        content,
        title="[bold yellow]TOOL CALL[/bold yellow]",
        style="yellow",
        box=box.ROUNDED,
    ))


def print_tool_result(
    invocation: PendingToolInvocation,
    result: dict[str, Any] | None,
) -> None:
    """Print a tool invocation result."""
    if invocation.status == ToolInvocationStatus.EXECUTED:
        console.print("[bold green]Tool executed[/bold green]")
        if result:
            result_json = json.dumps(result, indent=2)
            lines = result_json.split("\n")
            if len(lines) > 150:
                display = "\n".join(lines[:150]) + f"\n... ({len(lines) - 150} more lines)"
            else:
                display = result_json
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]Tool execution failed[/bold red]")
        error = result.get("error") if result else None
        if error:
            console.print(Panel(str(error), title="Error", style="red", box=box.ROUNDED))

    console.print()


class TerminalInteractionSpecialistChat:
    """Interactive terminal chat interface for the Interaction Specialist Agent."""

    def __init__(
        self,
        interaction_store: InteractionsDataStore,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._streaming_started: bool = False
        self._interaction_store = interaction_store
        self._llm_model = llm_model
        self._agent = self._create_agent()

    def _create_agent(self) -> InteractionSpecialist:
        """Create a fresh InteractionSpecialist agent."""
        return InteractionSpecialist(
            emit_message_callable=self._handle_message,
            interaction_data_store=self._interaction_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self._llm_model,
            run_mode=RunMode.CONVERSATIONAL,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold magenta]Assistant[/bold magenta]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """Handle messages emitted by the Interaction Specialist Agent."""
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()
                print()
                self._streaming_started = False
            else:
                print_assistant_message(message.content)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            print_tool_call(message.tool_invocation)
            print_tool_result(message.tool_invocation, message.tool_result)

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)

    def _run_autonomous(self, task: str) -> None:
        """Run autonomous parameter discovery for a given task."""
        console.print()
        console.print(Panel(
            f"[bold]Task:[/bold] {task}",
            title="[bold magenta]Starting Autonomous Parameter Discovery[/bold magenta]",
            border_style="magenta",
            box=box.ROUNDED,
        ))
        console.print()

        self._agent.reset()

        start_time = time.perf_counter()
        result = self._agent.run_autonomous(task)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        console.print()

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

                console.print(Panel(
                    param_table,
                    title=f"[bold green]Parameter {i}/{param_count}[/bold green]",
                    border_style="green",
                    box=box.ROUNDED,
                ))

            console.print(f"[bold green]Found {param_count} parameters[/bold green] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]")

        elif isinstance(result, ParameterDiscoveryFailureResult):
            failure_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            failure_table.add_column("Field", style="bold red")
            failure_table.add_column("Value", style="white")

            failure_table.add_row("Reason", result.reason)
            failure_table.add_row("Interaction Summary", result.interaction_summary)

            console.print(Panel(
                failure_table,
                title=f"[bold red]Parameter Discovery Failed[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            console.print(Panel(
                "[yellow]Could not finalize parameter discovery. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]Discovery Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        console.print()

    def run(self) -> None:
        """Run the interactive chat loop."""
        while True:
            try:
                user_input = pt_prompt(
                    HTML("<b><ansimagenta>You&gt;</ansimagenta></b> "),
                    completer=SlashCommandCompleter(SLASH_COMMANDS),
                    lexer=SlashCommandLexer(),
                    complete_while_typing=True,
                )

                if not user_input.strip():
                    continue

                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold magenta]Goodbye![/bold magenta]")
                    console.print()
                    break

                if cmd == "/reset":
                    self._agent = self._create_agent()
                    console.print()
                    console.print("[yellow]Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        dedent("""\
                            [bold]Commands:[/bold]
                              [cyan]/autonomous <task>[/cyan]  Run autonomous parameter discovery
                                                    Example: /autonomous discover parameters for train search
                              [cyan]/reset[/cyan]              Start a new conversation
                              [cyan]/help[/cyan]               Show this help message
                              [cyan]/quit[/cyan]               Exit

                            [bold]Tips:[/bold]
                              - Ask about specific user interactions or form elements
                              - Request analysis of click patterns or input fields
                              - Ask about form submission workflows"""),
                        title="[bold magenta]Help[/bold magenta]",
                        border_style="magenta",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                if user_input.strip().lower().startswith("/autonomous"):
                    task = user_input.strip()[len("/autonomous"):].strip()
                    if not task:
                        console.print()
                        console.print("[bold yellow]Usage:[/bold yellow] /autonomous <task description>")
                        console.print("[dim]Example: /autonomous discover parameters for train search[/dim]")
                        console.print()
                        continue

                    self._run_autonomous(task)
                    continue

                self._agent.process_new_message(user_input, ChatRole.USER)

            except KeyboardInterrupt:
                console.print()
                console.print("[magenta]Interrupted. Goodbye![/magenta]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[magenta]Goodbye![/magenta]")
                console.print()
                break


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
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help=f"LLM model to use (default: gpt-5.1). Options: {', '.join(get_all_model_values())}",
    )
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

    # Resolve model string to enum
    model_result = get_model_by_value(args.model)
    if model_result is None:
        console.print(f"[bold red]Error: Unknown model '{args.model}'[/bold red]")
        console.print(f"[dim]Available models: {', '.join(get_all_model_values())}[/dim]")
        sys.exit(1)
    llm_model = model_result

    print_welcome(args.model, str(jsonl_path), interaction_store)

    chat = TerminalInteractionSpecialistChat(
        interaction_store=interaction_store,
        llm_model=llm_model,
    )
    chat.run()


if __name__ == "__main__":
    main()
