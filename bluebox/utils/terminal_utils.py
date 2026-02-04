"""
bluebox/utils/terminal_utils.py

Utility functions for terminal input/output.
"""

from typing import Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.formatted_text import StyleAndTextTuples

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
