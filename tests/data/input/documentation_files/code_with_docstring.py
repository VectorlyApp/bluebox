"""
code_with_docstring.py

This is a sample Python file with a module-level docstring.
Used for testing docstring extraction.
"""

from typing import Any


def sample_function(param: str) -> str:
    """
    A sample function.

    Args:
        param: The input parameter.

    Returns:
        The processed result.
    """
    return f"processed: {param}"


class SampleClass:
    """A sample class for testing."""

    def __init__(self) -> None:
        self.value = 0

    def get_value(self) -> int:
        """Get the current value."""
        return self.value
