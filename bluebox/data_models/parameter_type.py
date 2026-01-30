"""
bluebox/data_models/routine/parameter_type.py

Shared enums and constants for parameter definitions.
Extracted to avoid circular imports between data_utils and routine models.
"""

from enum import StrEnum


# Valid prefixes for storage/meta/window placeholders
VALID_PLACEHOLDER_PREFIXES = frozenset([
    "sessionStorage", "localStorage", "cookie", "meta", "windowProperty"
])


class ParameterType(StrEnum):
    """Supported parameter types for MCP tools."""
    # python primitives
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"

    # non-python primitives
    DATE = "date"
    DATETIME = "datetime"
    EMAIL = "email"
    URL = "url"
    ENUM = "enum"
