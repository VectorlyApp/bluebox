"""
bluebox/data_models/routine/placeholder.py

Placeholder extraction for template strings.

Contains:
- extract_placeholders_from_json_str(): Find all {{...}} patterns in text, returns deduplicated list[str]
- Supports: user params, sessionStorage, localStorage, cookies, windowProperty, builtins
"""

import re


def extract_placeholders_from_json_str(json_string: str) -> list[str]:
    """
    Extract all placeholders from a JSON string.

    Finds all {{...}} patterns uniformly — no quote-type tracking.
    Returns deduplicated list preserving first-seen order.

    Args:
        json_string: The JSON string to search.

    Returns:
        Deduplicated list of placeholder content strings (trimmed).
    """
    pattern = r'\{\{\s*([^}]+?)\s*\}\}'
    seen: set[str] = set()
    result: list[str] = []
    for match in re.finditer(pattern, json_string):
        content = match.group(1).strip()
        if content not in seen:
            seen.add(content)
            result.append(content)
    return result
