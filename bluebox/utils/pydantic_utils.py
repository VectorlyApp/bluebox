"""
bluebox/utils/pydantic_utils.py

Utilities for working with Pydantic models.

Contains:
- format_field_type: Compact type annotation formatting (e.g. "str?", '"GET" | "POST"')
- format_default: Compact default value formatting (e.g. "null", "true", "3.0")
- format_model_fields: Render a model's fields as markdown schema lines
"""

import types as types_module
from enum import StrEnum
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel


def format_field_type(annotation: type) -> str:
    """
    Format a Pydantic field type annotation as a compact string.

    Args:
        annotation: The type annotation to format.

    Returns:
        A compact string representation of the type annotation.

    Examples:
        str → "str"
        str | None → "str?"
        Literal["GET", "POST"] → '"GET" | "POST"'
        StrEnum subclass → '"value1" | "value2"'
    """
    if annotation is None:
        return "any"

    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional (typing.Union or X | Y syntax)
    if origin is Union or isinstance(annotation, types_module.UnionType):
        if isinstance(annotation, types_module.UnionType):
            args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        has_none = type(None) in args
        inner = " | ".join(format_field_type(a) for a in non_none)
        return f"{inner}?" if has_none else inner

    # Literal
    if origin is Literal:
        vals = []
        for a in args:
            val = a.value if hasattr(a, "value") else a
            vals.append(f'"{val}"' if isinstance(val, str) else str(val))
        return " | ".join(vals)

    # dict / list
    if origin is dict:
        return "dict"
    if origin is list:
        return f"list[{format_field_type(args[0])}]" if args else "list"

    # Concrete classes
    if isinstance(annotation, type):
        if issubclass(annotation, StrEnum):
            return " | ".join(f'"{m.value}"' for m in annotation)
        return annotation.__name__

    return str(annotation)


def format_default(default: object) -> str:
    """
    Format a field default value compactly for markdown schema output.

    Args:
        default: The default value to format.

    Returns:
        A compact string representation of the default value.
    """
    if default is None:
        return "null"
    if isinstance(default, bool):
        return "true" if default else "false"
    if isinstance(default, (int, float)):
        return str(default)
    if hasattr(default, "value"):  # enum member
        return f'"{default.value}"'
    if isinstance(default, str):
        return f'"{default}"'
    if isinstance(default, list):
        return "[]"
    return repr(default)


def format_model_fields(
    model_cls: type[BaseModel],
    skip_fields: set[str] | None = None,
) -> list[str]:
    """
    Format a Pydantic model's fields as compact markdown schema lines.

    Args:
        model_cls: The Pydantic model class to format.
        skip_fields: Field names to omit from the output.

    Returns:
        List of markdown lines like '- name: str (required)' or '- count: int = 0'.
    """
    skip = skip_fields or set()
    lines: list[str] = []
    for name, info in model_cls.model_fields.items():
        if name in skip:
            continue
        type_str = format_field_type(info.annotation)
        if info.is_required():
            lines.append(
                f"- {name}: {type_str} (required)"
            )
        else:
            default = info.get_default(call_default_factory=True)
            lines.append(
                f"- {name}: {type_str} = {format_default(default)}"
            )
    return lines
