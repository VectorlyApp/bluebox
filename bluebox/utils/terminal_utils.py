"""
bluebox/utils/terminal_utils.py

Utility functions for terminal input/output.
"""

import json
from typing import Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text import StyleAndTextTuples
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from bluebox.data_models.llms.interaction import (
    PendingToolInvocation,
    ToolInvocationStatus,
)

# Colors for output (ANSI codes)
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color


class SlashCommandLexer(Lexer):
    """
    Highlight slash commands (e.g., /help, /quit) in bold.

    The slash command is considered to be the text from '/' until the first space
    (or end of line if no space).
    """

    def lex_document(self, document: Document) -> Any:
        """Return a callable that returns styled tokens for a given line."""
        def get_line(lineno: int) -> StyleAndTextTuples:
            line = document.lines[lineno]
            if line.startswith("/"):
                # find end of command (first space or end of line)
                space_idx = line.find(" ")
                if space_idx == -1:
                    # entire line is the command
                    return [("bold", line)]
                else:
                    # command portion + rest
                    return [("bold", line[:space_idx]), ("", line[space_idx:])]
            return [("", line)]

        return get_line


class SlashCommandCompleter(Completer):
    """
    Show slash command suggestions when the input starts with '/'.

    Args:
        commands: List of (command, description) tuples.
    """

    def __init__(self, commands: list[tuple[str, str]]) -> None:
        self._commands = commands

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        for cmd, desc in self._commands:
            if cmd.startswith(text):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )


def print_colored(text: str, color: str = NC) -> None:
    """Print colored text."""
    print(f"{color}{text}{NC}")


def print_header(title: str) -> None:
    """Print a styled header."""
    print()
    print_colored(f"{'─' * 60}", CYAN)
    print_colored(f"  {title}", CYAN)
    print_colored(f"{'─' * 60}", CYAN)
    print()


def ask_yes_no(prompt: str) -> bool:
    """
    Ask a yes/no question and return True for 'y', False for 'n'.
    Keeps asking until valid input is provided.
    """
    while True:
        response = input(f"{YELLOW}{prompt} (y/n): {NC}").strip().lower()
        if response in ('y', 'n'):
            return response == 'y'
        print_colored("   ⚠️  Please enter 'y' or 'n'", YELLOW)


# ============================================================================
# Terminal Agent Display Functions
# ============================================================================


def print_assistant_message(content: str, console: Console | None = None) -> None:
    """
    Print an assistant response using markdown rendering.

    Args:
        content: The message content to display
        console: Optional Rich Console instance (creates one if not provided)
    """
    if console is None:
        console = Console()

    console.print()
    console.print("[bold cyan]Assistant[/bold cyan]")
    console.print()
    console.print(Markdown(content))
    console.print()


def print_error(error: str, console: Console | None = None) -> None:
    """
    Print an error message.

    Args:
        error: The error message to display
        console: Optional Rich Console instance (creates one if not provided)
    """
    if console is None:
        console = Console()

    console.print()
    console.print(f"[bold red]Error:[/bold red] [red]{escape(error)}[/red]")
    console.print()


def print_tool_call(invocation: PendingToolInvocation, console: Console | None = None) -> None:
    """
    Print a tool call with formatted arguments.

    Args:
        invocation: The tool invocation to display
        console: Optional Rich Console instance (creates one if not provided)
    """
    if console is None:
        console = Console()

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
    console: Console | None = None,
) -> None:
    """
    Print a tool invocation result.

    Args:
        invocation: The tool invocation
        result: The result dictionary from the tool execution
        console: Optional Rich Console instance (creates one if not provided)
    """
    if console is None:
        console = Console()

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
