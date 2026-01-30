"""
bluebox/data_models/routine/placeholder.py

Placeholder extraction and parsing for template strings.

Contains:
- extract_placeholders_from_json_str(): Find all {{...}} patterns in text
- Supports: user params, sessionStorage, localStorage, cookies, windowProperty
"""

import re


def extract_placeholders_from_json_str(json_string: str) -> list[str]:
    """
    Extract all placeholder contents from a JSON string.

    Finds all {{...}} patterns and returns their inner content (stripped).

    Args:
        json_string: The JSON string to search

    Returns:
        List of placeholder content strings (deduplicated, preserving order).
    """
    pattern = r"\{\{\s*([^}]+?)\s*\}\}"
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(pattern=pattern, string=json_string):
        content = match.group(1)
        if content not in seen:
            seen.add(content)
            result.append(content)
    return result
