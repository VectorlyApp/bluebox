"""
bluebox/llms/tools/tool_utils.py

Utilities for converting Python functions to LLM tool definitions.
"""

import inspect
from typing import Any, Callable, get_type_hints

from pydantic import TypeAdapter


def extract_description_from_docstring(docstring: str | None) -> str:
    """
    Extract the full description from a docstring (everything before Args/Returns/etc).

    Args:
        docstring: The function's docstring (func.__doc__)

    Returns:
        The description portion of the docstring, or empty string if none.
    """
    if not docstring:
        return ""

    # Find where the Args/Returns/Raises/Yields section starts
    section_markers = ("Args:", "Returns:", "Raises:", "Yields:", "Example:", "Examples:", "Note:", "Notes:")
    lines = docstring.strip().split("\n")

    description_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Stop if we hit a section marker
        if any(stripped.startswith(marker) for marker in section_markers):
            break
        description_lines.append(stripped)

    # Join lines, collapse multiple spaces, and strip trailing whitespace
    description = " ".join(description_lines)
    # Collapse multiple spaces into single space
    while "  " in description:
        description = description.replace("  ", " ")
    return description.strip()


def _parse_args_from_docstring(docstring: str | None) -> dict[str, str]:
    """Extract param descriptions from docstring Args section."""
    if not docstring:
        return {}
    result = {}
    in_args = False
    for line in docstring.split("\n"):
        s = line.strip()
        if s.startswith("Args:"):
            in_args = True
            continue
        if in_args and s.endswith(":") and s in ("Returns:", "Raises:", "Yields:"):
            break
        if in_args and ":" in s:
            name, desc = s.split(":", 1)
            name = name.split("(")[0].strip()  # handle "param (type):" format
            if name and " " not in name:
                result[name] = desc.strip()
    return result


def generate_parameters_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Generate JSON Schema for function parameters using pydantic.

    Args:
        func: The function to generate schema for. Must have type hints.

    Returns:
        JSON Schema dict with 'type', 'properties', and 'required' keys.
    """
    sig = inspect.signature(obj=func)
    hints = get_type_hints(obj=func)
    param_descs = _parse_args_from_docstring(docstring=func.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, Any)
        # use pydantic TypeAdapter to generate schema for this type
        schema = TypeAdapter(param_type).json_schema()

        # remove pydantic metadata that's not needed for tool schemas
        schema.pop("title", None)

        # add description from docstring if available
        if param_name in param_descs:
            schema["description"] = param_descs[param_name]

        properties[param_name] = schema

        # parameter is required if it has no default value
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
