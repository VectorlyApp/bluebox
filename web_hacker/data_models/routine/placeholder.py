"""
web_hacker/data_models/routine/placeholder.py

Placeholder extraction and validation utilities for routines.
"""

import re
from dataclasses import dataclass
from enum import StrEnum


class PlaceholderQuoteType(StrEnum):
    """Type of quoting around a placeholder."""
    QUOTED = "quoted"              # "{{param}}" - regular JSON string quotes
    ESCAPE_QUOTED = "escape_quoted"  # \"{{param}}\" - backslash-escaped quotes


@dataclass
class ExtractedPlaceholder:
    """Represents an extracted placeholder from a JSON string."""
    content: str                    # The content inside {{...}}
    quote_type: PlaceholderQuoteType  # How it's quoted


def extract_placeholders_from_json_str(json_string: str) -> list[ExtractedPlaceholder]:
    """
    Extract all placeholders from a JSON string, identifying their quote type.
    
    Finds two types of placeholders (unquoted placeholders are ignored):
    - Escape-quoted: \\"{{param}}\\" - backslash-escaped quotes (valid for all param types)
    - Quoted: "{{param}}" - regular JSON string quotes (valid for int/number/bool/storage/builtins)
    
    Args:
        json_string: The JSON string to search
        
    Returns:
        List of ExtractedPlaceholder objects with content and quote type
    """
    placeholders: list[ExtractedPlaceholder] = []
    
    # Track which positions have been matched to avoid double-counting
    matched_positions: set[int] = set()
    
    # Pattern for escape-quoted placeholders: \"{{...}}\"
    escape_quoted_pattern = r'\\"\{\{([^}"]*)\}\}\\"'
    
    # Pattern for regular quoted placeholders: "{{...}}"
    quoted_pattern = r'(?<!\\)"\{\{([^}"]*)\}\}"'
    
    # Find escape-quoted placeholders first (highest priority)
    for match in re.finditer(escape_quoted_pattern, json_string):
        content = match.group(1).strip()
        placeholders.append(ExtractedPlaceholder(
            content=content,
            quote_type=PlaceholderQuoteType.ESCAPE_QUOTED
        ))
        # Mark the inner {{...}} position as matched
        inner_start = match.start() + 2  # skip \"
        matched_positions.add(inner_start)
    
    # Find regular quoted placeholders
    for match in re.finditer(quoted_pattern, json_string):
        content = match.group(1).strip()
        inner_start = match.start() + 1  # skip "
        if inner_start not in matched_positions:
            placeholders.append(ExtractedPlaceholder(
                content=content,
                quote_type=PlaceholderQuoteType.QUOTED
            ))
            matched_positions.add(inner_start)
    
    # Unquoted placeholders are ignored - they won't be detected
    
    return placeholders

