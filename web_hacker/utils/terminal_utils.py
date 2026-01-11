"""
web_hacker/utils/terminal_utils.py

Utility functions for terminal input/output.
"""

# Colors for output (ANSI codes)
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color


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
        response = input(f"{prompt} (y/n): ").strip().lower()
        if response in ('y', 'n'):
            return response == 'y'
        print_colored("   ⚠️  Please enter 'y' or 'n'", YELLOW)

