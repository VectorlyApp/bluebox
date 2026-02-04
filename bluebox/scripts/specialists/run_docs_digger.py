#!/usr/bin/env python3
"""
bluebox/scripts/specialists/run_docs_digger.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Interactive CLI for the Docs Digger agent - searches documentation and code.

Usage:
    # Run with defaults (bluebox agent_docs and core modules)
    bluebox-docs-digger

    # Or specify custom paths
    bluebox-docs-digger --docs-paths "docs/**/*.md" --code-paths "src/**/*.py"
    bluebox-docs-digger --model gpt-5.1
"""

import argparse
import sys
import time
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from bluebox.agents.specialists.docs_digger_agent import (
    DocsDiggerAgent,
    DocumentSearchResult,
    DocumentSearchFailureResult,
)
from bluebox.llms.infra.documentation_data_store import DocumentationDataStore
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.cli_utils import add_model_argument, resolve_model
from bluebox.utils.data_utils import format_bytes
from bluebox.utils.terminal_agent_base import AbstractTerminalAgentChat
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()

# Package root for default paths (scripts/ is now inside bluebox/)
BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

# Default documentation and code paths (same as run_guide_agent.py)
DEFAULT_DOCS_DIR = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
DEFAULT_CODE_PATHS = [
    str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
    str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
    str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
    str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
    str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
    str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
    "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
]

SLASH_COMMANDS = [
    ("/search", "Autonomously search for documentation — /search <query>"),
    ("/reset", "Start a new conversation"),
    ("/help", "Show help"),
    ("/quit", "Exit"),
]


BANNER = """\
[bold green]╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                              ║
║ ▄▄▄▄▄▄                                 ▄▄▄▄▄▄   ██                                                           ║
║ ██▀▀▀▀██                               ██▀▀▀▀██ ▀▀                                                           ║
║ ██    ██   ▄████▄    ▄█████▄  ▄▄█████▄ ██    ██ ██   ▄███▄██  ▄███▄██   ▄████▄    ██▄████                    ║
║ ██    ██  ██▀  ▀██  ██▀    ▀  ██▄▄▄▄ ▀ ██    ██ ██  ██▀  ▀██ ██▀  ▀██  ██▄▄▄▄██   ██▀                        ║
║ ██    ██  ██    ██  ██        ▀▀▀▀██▄  ██    ██ ██  ██    ██ ██    ██  ██▀▀▀▀▀▀   ██                         ║
║ ██▄▄▄▄██  ▀██▄▄██▀  ▀██▄▄▄▄█  █▄▄▄▄▄██ ██▄▄▄▄██ ██  ▀██▄▄███ ▀██▄▄███  ▀██▄▄▄▄█   ██                         ║
║ ▀▀▀▀▀▀      ▀▀▀▀       ▀▀▀▀▀   ▀▀▀▀▀▀  ▀▀▀▀▀▀   ▀▀   ▄▀▀▀ ██  ▄▀▀▀ ██    ▀▀▀▀▀    ▀▀                         ║
║                                                      ▀████▀▀  ▀████▀▀                                        ║
║                                                                                                              ║
║                         Documentation & Code Search Agent - Find answers fast                                ║
║                                                                                                              ║
╚═══════════════════════════════════════════(BETA)═════════════════════════════════════════════════════════════╝[/bold green]
"""


class TerminalDocsDiggerChat(AbstractTerminalAgentChat):
    """Interactive terminal chat interface for the Docs Digger Agent."""

    def __init__(
        self,
        docs_store: DocumentationDataStore,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self.docs_store = docs_store
        self.llm_model = llm_model
        super().__init__(console=console, agent_color="green")

    def _create_agent(self) -> DocsDiggerAgent:
        """Create the Docs Digger agent instance."""
        return DocsDiggerAgent(
            emit_message_callable=self._handle_message,
            documentation_data_store=self.docs_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=self.llm_model,
        )

    def get_slash_commands(self) -> list[tuple[str, str]]:
        """Return list of slash commands."""
        return SLASH_COMMANDS

    @property
    def autonomous_command_name(self) -> str:
        """Return the autonomous command name."""
        return "search"

    def print_welcome(self) -> None:
        """Print welcome message with documentation stats."""
        self.console.print(BANNER)
        self.console.print()

        stats = self.docs_store.stats

        # Build stats table
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        stats_table.add_column("Label", style="dim")
        stats_table.add_column("Value", style="white")

        stats_table.add_row("Total Files", str(stats.total_files))
        stats_table.add_row("Documentation Files", str(stats.total_docs))
        stats_table.add_row("Code Files", str(stats.total_code))
        stats_table.add_row("Total Size", format_bytes(stats.total_bytes))

        # Extensions breakdown
        if stats.extensions:
            ext_str = ", ".join(
                f"{ext}: {count}"
                for ext, count in sorted(stats.extensions.items(), key=lambda x: -x[1])[:8]
            )
            stats_table.add_row("Extensions", ext_str)

        # Quality stats
        if stats.total_docs > 0:
            stats_table.add_row(
                "Docs with Title",
                f"{stats.docs_with_title}/{stats.total_docs}"
            )
            stats_table.add_row(
                "Docs with Summary",
                f"{stats.docs_with_summary}/{stats.total_docs}"
            )

        if stats.total_code > 0:
            stats_table.add_row(
                "Code with Docstring",
                f"{stats.code_with_docstring}/{stats.total_code}"
            )

        self.console.print(Panel(
            stats_table,
            title="[bold green]Documentation Stats[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Show documentation files index
        doc_index = self.docs_store.get_documentation_index()
        if doc_index:
            doc_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            doc_table.add_column("File", style="white")
            doc_table.add_column("Title", style="cyan")

            for doc in doc_index[:15]:  # Top 15 docs
                title = doc["title"] or "[dim]No title[/dim]"
                if len(title) > 50:
                    title = title[:47] + "..."
                doc_table.add_row(
                    doc["filename"],
                    title,
                )

            if len(doc_index) > 15:
                doc_table.add_row(
                    f"[dim]... and {len(doc_index) - 15} more[/dim]",
                    "",
                )

            self.console.print(Panel(
                doc_table,
                title=f"[bold cyan]📚 Documentation Files[/bold cyan] [dim]({len(doc_index)} files)[/dim]",
                border_style="cyan",
                box=box.ROUNDED,
            ))
            self.console.print()

        # Show code files index
        code_index = self.docs_store.get_code_index()
        if code_index:
            code_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            code_table.add_column("File", style="white")
            code_table.add_column("Docstring", style="dim")

            for code in code_index[:15]:  # Top 15 code files
                docstring = code["docstring"] or ""
                if len(docstring) > 60:
                    docstring = docstring[:57] + "..."
                code_table.add_row(
                    code["filename"],
                    docstring or "[dim]-[/dim]",
                )

            if len(code_index) > 15:
                code_table.add_row(
                    f"[dim]... and {len(code_index) - 15} more[/dim]",
                    "",
                )

            self.console.print(Panel(
                code_table,
                title=f"[bold yellow]💻 Code Files[/bold yellow] [dim]({len(code_index)} files)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))
            self.console.print()

        self.console.print(Panel(
            """[bold]Commands:[/bold]
  [green]/search <query>[/green]   Autonomously search for documentation
  [green]/reset[/green]            Start a new conversation
  [green]/help[/green]             Show help
  [green]/quit[/green]             Exit

Just ask questions about the documentation and code!""",
            title="[bold green]Docs Digger[/bold green]",
            subtitle=f"[dim]Model: {self.llm_model.value}[/dim]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

    def handle_autonomous_command(self, query: str) -> None:
        """Run autonomous documentation search for a given query."""
        self.console.print()
        self.console.print(Panel(
            f"[bold]Query:[/bold] {query}",
            title="[bold green]🔍 Starting Documentation Search[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        self.console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous search with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(query)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        self.console.print()

        if isinstance(result, DocumentSearchResult):
            # Success - show discovered documents
            doc_count = len(result.documents)

            docs_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
            docs_table.add_column("#", style="dim", width=3)
            docs_table.add_column("Type", style="cyan", width=8)
            docs_table.add_column("Path", style="white")
            docs_table.add_column("Reason", style="dim")

            for i, doc in enumerate(result.documents, 1):
                path_display = doc.path
                if len(path_display) > 50:
                    path_display = "..." + path_display[-47:]
                reason_display = doc.relevance_reason
                if len(reason_display) > 40:
                    reason_display = reason_display[:37] + "..."
                docs_table.add_row(
                    str(i),
                    doc.file_type[:4].upper(),
                    path_display,
                    reason_display,
                )

            self.console.print(Panel(
                docs_table,
                title=f"[bold green]✓ Documents Found[/bold green] [dim]({doc_count} files)[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))

            # Show summary
            self.console.print()
            self.console.print(Panel(
                Markdown(result.summary),
                title="[bold cyan]📋 Summary[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))

            # Show key content from each document
            for i, doc in enumerate(result.documents, 1):
                if doc.key_content:
                    self.console.print()
                    path_short = doc.path.split("/")[-1] if "/" in doc.path else doc.path
                    self.console.print(Panel(
                        doc.key_content[:500] + ("..." if len(doc.key_content) > 500 else ""),
                        title=f"[bold yellow]📄 {path_short}[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                    ))

            self.console.print()
            self.console.print(
                f"[bold green]✓ Search completed[/bold green] "
                f"[dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]"
            )

        elif isinstance(result, DocumentSearchFailureResult):
            # Explicit failure - documentation not found
            failure_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
            failure_table.add_column("Field", style="bold red")
            failure_table.add_column("Value", style="white")

            failure_table.add_row("Reason", result.reason)
            if result.searched_terms:
                failure_table.add_row("Terms Searched", ", ".join(result.searched_terms[:15]))
            if result.closest_matches:
                failure_table.add_row("Closest Matches", "\n".join(result.closest_matches[:5]))

            self.console.print(Panel(
                failure_table,
                title=f"[bold red]✗ Documentation Not Found[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            # None - max iterations without finalization
            self.console.print(Panel(
                "[yellow]Could not finalize documentation search. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]⚠ Search Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        self.console.print()


def main() -> None:
    """Run the Docs Digger agent interactively."""
    parser = argparse.ArgumentParser(
        description="Docs Digger - Interactive documentation and code search"
    )
    parser.add_argument(
        "--docs-paths",
        type=str,
        nargs="+",
        default=None,
        help="Glob patterns for documentation files (default: bluebox/agent_docs)",
    )
    parser.add_argument(
        "--code-paths",
        type=str,
        nargs="+",
        default=None,
        help="Glob patterns for code files (default: bluebox core modules)",
    )
    add_model_argument(parser)
    args = parser.parse_args()

    # Use defaults if not provided
    docs_paths = args.docs_paths if args.docs_paths else [DEFAULT_DOCS_DIR]
    code_paths = args.code_paths if args.code_paths else DEFAULT_CODE_PATHS

    console.print("[dim]Loading documentation data store...[/dim]")

    # Create documentation data store
    try:
        docs_store = DocumentationDataStore(
            documentation_paths=docs_paths,
            code_paths=code_paths,
        )
    except Exception as e:
        console.print(f"[bold red]Error loading files: {e}[/bold red]")
        sys.exit(1)

    if len(docs_store.entries) == 0:
        console.print("[bold red]Error: No files found matching the provided patterns[/bold red]")
        console.print("[dim]Check your glob patterns and ensure files exist[/dim]")
        sys.exit(1)

    # Resolve model
    llm_model = resolve_model(args.model, console)

    # Create and run chat
    chat = TerminalDocsDiggerChat(
        docs_store=docs_store,
        llm_model=llm_model,
    )
    chat.print_welcome()
    chat.run()


if __name__ == "__main__":
    main()
