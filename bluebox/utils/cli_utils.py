"""
bluebox/utils/cli_utils.py

Utility functions for CLI argument parsing.
"""

import sys
from argparse import ArgumentParser

from rich.console import Console

from bluebox.data_models.llms.vendors import (
    LLMModel,
    get_all_model_values,
    get_model_by_value,
)


def add_model_argument(parser: ArgumentParser) -> None:
    """
    Add the --model argument to an ArgumentParser.

    Args:
        parser: The ArgumentParser to add the argument to
    """
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help=f"LLM model to use (default: gpt-5.1). Options: {', '.join(get_all_model_values())}",
    )


def resolve_model(model_str: str, console: Console) -> LLMModel:
    """
    Resolve a model string to an LLMModel enum value.

    Args:
        model_str: The model string to resolve (e.g., "gpt-5.1")
        console: Rich Console instance for error output

    Returns:
        The resolved LLMModel enum value

    Exits with error if the model string is invalid.
    """
    model_result = get_model_by_value(model_str)
    if model_result is None:
        console.print(f"[bold red]Error: Unknown model '{model_str}'[/bold red]")
        console.print(f"[dim]Available models: {', '.join(get_all_model_values())}[/dim]")
        sys.exit(1)
    return model_result
