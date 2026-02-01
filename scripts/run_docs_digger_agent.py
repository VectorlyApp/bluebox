#!/usr/bin/env python3
"""
scripts/run_docs_digger_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Interactive CLI for the Docs Digger agent - searches documentation and code.

Usage:
    python scripts/run_docs_digger_agent.py \
        --docs-paths "docs/**/*.md" \
        --code-paths "src/**/*.py"
"""

import argparse
import json
import sys
import time
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from bluebox.agents.docs_digger_agent import (
    DocsDiggerAgent,
    DocumentSearchResult,
    DocumentSearchFailureResult,
)
from bluebox.llms.infra.documentation_data_store import DocumentationDataStore
from bluebox.data_models.llms.interaction import (
    ChatRole,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)
console = Console()


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


def print_welcome(
    model: str,
    docs_store: DocumentationDataStore,
) -> None:
    """Print welcome message with documentation stats."""
    console.print(BANNER)
    console.print()

    stats = docs_store.stats

    # Build stats table
    stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    stats_table.add_column("Label", style="dim")
    stats_table.add_column("Value", style="white")

    stats_table.add_row("Total Files", str(stats.total_files))
    stats_table.add_row("Documentation Files", str(stats.total_docs))
    stats_table.add_row("Code Files", str(stats.total_code))
    stats_table.add_row("Total Size", stats._format_bytes(stats.total_bytes))

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

    console.print(Panel(
        stats_table,
        title="[bold green]Documentation Stats[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    ))
    console.print()

    # Show documentation files index
    doc_index = docs_store.get_documentation_index()
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

        console.print(Panel(
            doc_table,
            title=f"[bold cyan]📚 Documentation Files[/bold cyan] [dim]({len(doc_index)} files)[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        ))
        console.print()

    # Show code files index
    code_index = docs_store.get_code_index()
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

        console.print(Panel(
            code_table,
            title=f"[bold yellow]💻 Code Files[/bold yellow] [dim]({len(code_index)} files)[/dim]",
            border_style="yellow",
            box=box.ROUNDED,
        ))
        console.print()

    console.print(Panel(
        """[bold]Commands:[/bold]
  [green]/search <query>[/green]   Autonomously search for documentation
  [green]/reset[/green]            Start a new conversation
  [green]/help[/green]             Show help
  [green]/quit[/green]             Exit

Just ask questions about the documentation and code!""",
        title="[bold green]Docs Digger[/bold green]",
        subtitle=f"[dim]Model: {model}[/dim]",
        border_style="green",
        box=box.ROUNDED,
    ))
    console.print()


def print_assistant_message(content: str) -> None:
    """Print an assistant response using markdown rendering."""
    console.print()
    console.print("[bold green]Assistant[/bold green]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str) -> None:
    """Print an error message."""
    console.print()
    console.print(f"[bold red]⚠ Error:[/bold red] [red]{escape(error)}[/red]")
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
        title="[bold yellow]⚙ TOOL CALL[/bold yellow]",
        style="yellow",
        box=box.ROUNDED,
    ))


def print_tool_result(
    invocation: PendingToolInvocation,
    result: dict[str, Any] | None,
) -> None:
    """Print a tool invocation result."""
    if invocation.status == ToolInvocationStatus.EXECUTED:
        console.print("[bold green]✓ Tool executed[/bold green]")
        if result:
            result_json = json.dumps(result, indent=2)
            # Limit display to 150 lines
            lines = result_json.split("\n")
            if len(lines) > 150:
                display = "\n".join(lines[:150]) + f"\n... ({len(lines) - 150} more lines)"
            else:
                display = result_json
            console.print(Panel(display, title="Result", style="green", box=box.ROUNDED))

    elif invocation.status == ToolInvocationStatus.FAILED:
        console.print("[bold red]✗ Tool execution failed[/bold red]")
        error = result.get("error") if result else None
        if error:
            console.print(Panel(str(error), title="Error", style="red", box=box.ROUNDED))

    console.print()


class TerminalDocsDiggerChat:
    """Interactive terminal chat interface for the Docs Digger Agent."""

    def __init__(
        self,
        docs_store: DocumentationDataStore,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
    ) -> None:
        """Initialize the terminal chat interface."""
        self._streaming_started: bool = False
        self._agent = DocsDiggerAgent(
            emit_message_callable=self._handle_message,
            documentation_data_store=docs_store,
            stream_chunk_callable=self._handle_stream_chunk,
            llm_model=llm_model,
        )

    def _handle_stream_chunk(self, chunk: str) -> None:
        """Handle a streaming text chunk from the LLM."""
        if not self._streaming_started:
            console.print()
            console.print("[bold green]Assistant[/bold green]")
            console.print()
            self._streaming_started = True

        print(chunk, end="", flush=True)

    def _handle_message(self, message: EmittedMessage) -> None:
        """Handle messages emitted by the Docs Digger Agent."""
        if isinstance(message, ChatResponseEmittedMessage):
            if self._streaming_started:
                print()
                print()
                self._streaming_started = False
            else:
                print_assistant_message(message.content)

        elif isinstance(message, ToolInvocationResultEmittedMessage):
            # Show tool call and result
            print_tool_call(message.tool_invocation)
            print_tool_result(message.tool_invocation, message.tool_result)

        elif isinstance(message, ErrorEmittedMessage):
            print_error(message.error)

    def _run_search(self, query: str) -> None:
        """Run autonomous documentation search for a given query."""
        console.print()
        console.print(Panel(
            f"[bold]Query:[/bold] {query}",
            title="[bold green]🔍 Starting Documentation Search[/bold green]",
            border_style="green",
            box=box.ROUNDED,
        ))
        console.print()

        # Reset agent state for fresh autonomous run
        self._agent.reset()

        # Run autonomous search with timing
        start_time = time.perf_counter()
        result = self._agent.run_autonomous(query)
        elapsed_time = time.perf_counter() - start_time
        iterations = self._agent.autonomous_iteration

        console.print()

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

            console.print(Panel(
                docs_table,
                title=f"[bold green]✓ Documents Found[/bold green] [dim]({doc_count} files)[/dim]",
                border_style="green",
                box=box.ROUNDED,
            ))

            # Show summary
            console.print()
            console.print(Panel(
                Markdown(result.summary),
                title="[bold cyan]📋 Summary[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            ))

            # Show key content from each document
            for i, doc in enumerate(result.documents, 1):
                if doc.key_content:
                    console.print()
                    path_short = doc.path.split("/")[-1] if "/" in doc.path else doc.path
                    console.print(Panel(
                        doc.key_content[:500] + ("..." if len(doc.key_content) > 500 else ""),
                        title=f"[bold yellow]📄 {path_short}[/bold yellow]",
                        border_style="yellow",
                        box=box.ROUNDED,
                    ))

            console.print()
            console.print(
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

            console.print(Panel(
                failure_table,
                title=f"[bold red]✗ Documentation Not Found[/bold red] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="red",
                box=box.ROUNDED,
            ))

        else:
            # None - max iterations without finalization
            console.print(Panel(
                "[yellow]Could not finalize documentation search. "
                "The agent reached max iterations without calling finalize_result or finalize_failure.[/yellow]",
                title=f"[bold yellow]⚠ Search Incomplete[/bold yellow] [dim]({iterations} iterations, {elapsed_time:.1f}s)[/dim]",
                border_style="yellow",
                box=box.ROUNDED,
            ))

        console.print()

    def run(self) -> None:
        """Run the interactive chat loop."""
        while True:
            try:
                user_input = console.input("[bold green]You>[/bold green] ")

                if not user_input.strip():
                    continue

                cmd = user_input.strip().lower()

                if cmd in ("/quit", "/exit", "/q"):
                    console.print()
                    console.print("[bold green]Goodbye![/bold green]")
                    console.print()
                    break

                if cmd == "/reset":
                    self._agent.reset()
                    console.print()
                    console.print("[yellow]↺ Conversation reset[/yellow]")
                    console.print()
                    continue

                if cmd in ("/help", "/h", "/?"):
                    console.print()
                    console.print(Panel(
                        """[bold]Commands:[/bold]
  [green]/search <query>[/green]   Autonomously search documentation
                       Example: /search How do I configure authentication?
                       Example: /search What does the SDK client do?
  [green]/reset[/green]            Start a new conversation
  [green]/help[/green]             Show this help message
  [green]/quit[/green]             Exit

[bold]Tips:[/bold]
  - Ask about specific functions, classes, or concepts
  - Request explanations of how code works
  - Search for configuration options or API usage""",
                        title="[bold green]Help[/bold green]",
                        border_style="green",
                        box=box.ROUNDED,
                    ))
                    console.print()
                    continue

                # Handle /search command
                if user_input.strip().lower().startswith("/search"):
                    query = user_input.strip()[len("/search"):].strip()
                    if not query:
                        console.print()
                        console.print("[bold yellow]Usage:[/bold yellow] /search <query>")
                        console.print("[dim]Example: /search How do I configure the SDK?[/dim]")
                        console.print()
                        continue

                    self._run_search(query)
                    continue

                self._agent.process_new_message(user_input, ChatRole.USER)

            except KeyboardInterrupt:
                console.print()
                console.print("[green]Interrupted. Goodbye![/green]")
                console.print()
                break

            except EOFError:
                console.print()
                console.print("[green]Goodbye![/green]")
                console.print()
                break


def main() -> None:
    """Run the Docs Digger agent interactively."""
    parser = argparse.ArgumentParser(
        description="Docs Digger - Interactive documentation and code search"
    )
    parser.add_argument(
        "--docs-paths",
        type=str,
        nargs="+",
        help="Glob patterns for documentation files (e.g., 'docs/**/*.md')",
    )
    parser.add_argument(
        "--code-paths",
        type=str,
        nargs="+",
        help="Glob patterns for code files (e.g., 'src/**/*.py')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    args = parser.parse_args()

    # Validate that at least one path is provided
    if not args.docs_paths and not args.code_paths:
        console.print("[bold red]Error: At least one path must be provided[/bold red]")
        console.print("[dim]Use --docs-paths and/or --code-paths[/dim]")
        console.print()
        console.print("[bold]Examples:[/bold]")
        console.print("  python scripts/run_docs_digger_agent.py --docs-paths 'docs/**/*.md'")
        console.print("  python scripts/run_docs_digger_agent.py --code-paths 'src/**/*.py'")
        console.print("  python scripts/run_docs_digger_agent.py --docs-paths 'docs/**/*.md' --code-paths 'src/**/*.py'")
        sys.exit(1)

    console.print("[dim]Loading documentation data store...[/dim]")

    # Create documentation data store
    try:
        docs_store = DocumentationDataStore(
            documentation_paths=args.docs_paths,
            code_paths=args.code_paths,
        )
    except Exception as e:
        console.print(f"[bold red]Error loading files: {e}[/bold red]")
        sys.exit(1)

    if len(docs_store.entries) == 0:
        console.print("[bold red]Error: No files found matching the provided patterns[/bold red]")
        console.print("[dim]Check your glob patterns and ensure files exist[/dim]")
        sys.exit(1)

    # Map model string to enum
    model_map = {
        "gpt-5.1": OpenAIModel.GPT_5_1,
    }
    llm_model = model_map.get(args.model, OpenAIModel.GPT_5_1)

    print_welcome(args.model, docs_store)

    chat = TerminalDocsDiggerChat(
        docs_store=docs_store,
        llm_model=llm_model,
    )
    chat.run()


if __name__ == "__main__":
    main()
