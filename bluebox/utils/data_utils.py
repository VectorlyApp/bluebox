"""
bluebox/utils/data_utils.py

Data loading, writing, and transformation utilities.

Contains:
- write_json_file(), write_jsonl(), read_jsonl(): Save/load data to/from files
- get_text_from_html(): Extract text from HTML
- resolve_dotted_path(): Access nested dict values by dot notation
- apply_params(): Substitute {{placeholders}} in text
- assert_balanced_js_delimiters(): Validate JS code structure
- sanitize_filename(): Clean filenames for filesystem
"""

import base64
import copy
import datetime
import json
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import tldextract
from bs4 import BeautifulSoup

from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


def convert_floats_to_decimals(obj: Any) -> Any:
    """
    Convert all float values in a JSON-like object to Decimal values.
    Useful when putting or updating data into a DynamoDB table.
    Parameters:
        obj (Any): The object to convert.
    Returns:
        Any: The converted object.
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: convert_floats_to_decimals(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats_to_decimals(i) for i in obj]
    return obj


def convert_decimals_to_floats(obj: Any) -> Any:
    """
    Convert all Decimal values in a JSON-like object to float values.
    Useful when getting data from a DynamoDB table.
    Parameters:
        obj (Any): The object to convert.
    Returns:
        Any: The converted object.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_decimals_to_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_floats(i) for i in obj]
    return obj


def serialize_datetime(obj: Any) -> Any:
    """
    Recursively convert datetime.datetime instances to ISO-8601 strings.
    DynamoDB/Boto3 cannot accept raw datetimes.
    """
    if isinstance(obj, dict):
        return {k: serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize_datetime(v) for v in obj]
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return obj


def get_text_from_html(html: str) -> str:
    """
    Sanitize the HTML data.
    """
    # use the built-in html parser for robustness
    soup = BeautifulSoup(html, "html.parser")

    # remove non-visible elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # get visible text
    text = soup.get_text(separator="\n")

    # normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = "\n".join(chunk for chunk in chunks if chunk)
    
    # remove ALL consecutive newlines - replace any sequence of 2+ newlines with single newline
    # handle both \n and \r\n line endings
    clean_text = re.sub(
        pattern=r'[\r\n]+',
        repl='\n',
        string=clean_text,
    )

    # remove leading and trailing whitespace
    clean_text = clean_text.strip()

    return clean_text


def write_jsonl(path: str, obj: Any) -> None:
    """
    Write object as JSON line to file.
    Args:
        path (str): The path to the file to write to.
        obj (Any): The object to write to the file.
    """
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """
    Read a JSONL file and yield parsed JSON objects one line at a time.

    Args:
        path: Path to the JSONL file.

    Yields:
        Tuple of (line_number, parsed_dict) for each valid JSON line.
        Line numbers are 0-indexed.
        Skips empty lines and logs warnings for invalid JSON.
    """
    file_path = Path(path) if isinstance(path, str) else path

    if not file_path.exists():
        raise ValueError(f"JSONL file does not exist: {path}")

    with open(file_path, mode="r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield line_num, data
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse line %d: %s", line_num + 1, e)
                continue


def write_json_file(path: str, obj: Any) -> None:
    """
    Write pretty JSON file.
    Args:
        path (str): The path to the file to write to.
        obj (Any): The object to write to the file.
    """
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_data_to_file(
    data: Any,
    file_path: str,
    is_base64: bool = False,
) -> None:
    """
    Save data to file, creating directories as needed.
    
    Args:
        data: The data to save (dict, list, str, or base64-encoded string).
        file_path: Path to save the file to.
        is_base64: If True, decode data as base64 and write as binary.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    
    if data is None:
        logger.warning("Data is None. Skipping file save.")
        return
    
    if is_base64 and isinstance(data, str):
        raw_data = base64.b64decode(data)
        with open(file_path, mode="wb") as f:
            f.write(raw_data)
        logger.info(f"Saved data to: {file_path} ({len(raw_data)} bytes)")
    elif isinstance(data, (dict, list)):
        with open(file_path, mode="w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved data to: {file_path}")
    else:
        with open(file_path, mode="w", encoding="utf-8", errors="replace") as f:
            f.write(str(data))
        logger.info(f"Saved data to: {file_path}")


def get_set_cookie_values(headers: dict) -> list:
    """
    Extract Set-Cookie values from headers.
    Args:
        headers (dict): The headers to extract Set-Cookie values from.
    Returns:
        list: The list of Set-Cookie values.
    """
    values = []
    if not headers:
        return values
    try:
        for k, v in headers.items():
            if str(k).lower() == "set-cookie":
                if isinstance(v, str):
                    parts = v.split("\n") if "\n" in v else [v]
                    for line in parts:
                        line = line.strip()
                        if line:
                            values.append(line)
                elif isinstance(v, (list, tuple)):
                    for line in v:
                        if line:
                            values.append(str(line))
    except Exception:
        pass
    return values


def cookie_names_from_set_cookie(values: list) -> list:
    """
    Extract cookie names from Set-Cookie values.
    Args:
        values (list): The list of Set-Cookie values to extract cookie names from.
    Returns:
        list: The list of cookie names.
    """
    names = []
    for sc in values:
        first = sc.split(";", 1)[0]
        name = first.split("=", 1)[0].strip()
        if name:
            names.append(name)
    return names


def blocked_by_regex(url: str, block_regexes: list) -> bool:
    """
    Check if URL should be blocked by regex patterns.
    Args:
        url (str): The URL to check.
        block_regexes (list): The list of regex patterns to check against the URL.
    Returns:
        bool: True if the URL should be blocked, False otherwise.
    """
    u = url or ""
    for rx in block_regexes:
        if re.search(rx, u, flags=re.IGNORECASE):
            return True
    return False


def resolve_dotted_path(
    logger: logging.Logger,
    obj: str | dict | list,
    path: list[str] | str,
) -> str | None:
    """
    Resolve a dotted path through a nested data structure. Extracts values from complex nested objects using dot notation paths.
    Args:
        logger (logging.Logger): Logger instance for error reporting
        obj (str | dict | list): The object to traverse (string will be parsed as JSON)
        path (list[str] | str): Dotted path as string or list of path components
            E.g., "user.profile.name" or ["user", "profile", "name"].
    Returns:
        str | None: The value at the specified path as a string, or None if not found
    Example:
        >>> dict_data = {"a": {"b": {"c": 123}}}
        >>> resolve_dotted_path(logger, dict_data, "a.b.c")
        "123"
    """    
    # make a copy of the path to avoid modifying the original
    path_copy = copy.deepcopy(path)
    obj_copy = copy.deepcopy(obj)
    
    # convert string path to list
    if isinstance(path_copy, str):
        path_copy = path_copy.split(".")
    
    # handle empty path case
    if not path_copy:
        return str(obj_copy) if obj_copy is not None else None
    
    try:
        
        # loop until we have no more steps in the path
        for step_idx, path_step in enumerate(path_copy):
            
            # if the object is a string, parse it as JSON
            if isinstance(obj_copy, str):
                obj_copy = json.loads(obj_copy)
                
            # if the object is a list, get the next step as an integer
            if isinstance(obj_copy, list):
                obj_copy = obj_copy[int(path_step)]
                
            # if the object is a dictionary, get the next step as a key
            elif isinstance(obj_copy, dict):
                obj_copy = obj_copy.get(path_step, None)
            else:
                logger.error(f"Step {path_step} (index {step_idx}) in path {path_copy} is not a string, list, or dictionary")
                return None
                
        return str(obj_copy) if obj_copy is not None else None
    
    except Exception as e:
        logger.error(f"Error resolving dotted path {path}: for object {obj}: \nException: {e}")
        return None


def apply_params(text: str, parameters_dict: dict | None) -> str:
    """
    Replace parameter placeholders in text with actual values.

    Only replaces {{param}} where 'param' is in parameters_dict.
    Leaves other placeholders like {{sessionStorage:...}} untouched.
    
    Follows the pattern from test.py:
    - For string values in quoted placeholders: insert raw string (no quotes)
    - For non-string values in quoted placeholders: use json.dumps(value)
    - All placeholders must be quoted: "{{param}}" or \"{{param}}\"

    Args:
        text: Text containing parameter placeholders.
        parameters_dict: Dictionary of parameter values.

    Returns:
        str: Text with parameters replaced.
    """
    
    logger.info(f"Applying params to text: {text} with parameters_dict: {parameters_dict}")
    
    if not text or not parameters_dict:
        return text

    for key, value in parameters_dict.items():
        # Compute replacement based on value type (following test.py pattern)
        if isinstance(value, str):
            literal = value  # For strings, insert raw string (no quotes)
        else:
            literal = json.dumps(value)  # For numbers/bools/null, use JSON encoding
        
        escaped_key = re.escape(key)
        
        # Pattern 1: Simple quoted placeholder "{{key}}" in JSON string
        # Matches: "{{key}}" (when the JSON value itself is the string "{{key}}")
        simple_quoted = '"' + r'\{\{' + r'\s*' + escaped_key + r'\s*' + r'\}\}' + '"'
        text = re.sub(simple_quoted, literal, text)
        
        # Pattern 2: Escaped quote variant \"{{key}}\"
        # In JSON string this appears as: \\"{{key}}\\" 
        double_escaped = r'\\"' + r'\{\{' + r'\s*' + escaped_key + r'\s*' + r'\}\}' + r'\\"'
        text = re.sub(double_escaped, literal, text)
    
    logger.info(f"Applied params to text: {text}")
    return text


def extract_base_url_from_url(url: str) -> str | None:
    """
    Extracts a base URL (root domain) from a URL string.
    Returns just the root domain (e.g., "amtrak.com" from "https://www.amtrak.com" or "https://api.amtrak.com").
    Handles special TLDs correctly (e.g., "example.co.uk" -> "example.co.uk").
    Returns None if the URL is invalid.
    
    Args:
        url: The URL string to extract base URL from.
    
    Returns:
        The root domain (without protocol, subdomains, or port) or None if invalid.
    """
    try:
        # Try to parse the URL to extract the hostname
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        
        if not hostname:
            return None
        
        # Use tldextract to properly extract the root domain, handling special TLDs
        extracted = tldextract.extract(hostname)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        # Fallback: if domain extraction fails, return the hostname as-is
        return hostname
        
    except Exception:
        # If URL parsing fails (e.g., contains placeholders), try to extract hostname manually
        # Match protocol://hostname pattern
        match = re.match(r'^[^:]+://([^/\?\:]+)', url)
        if match and match.group(1):
            hostname = match.group(1)
            # Remove port if present
            hostname = hostname.split(':')[0]
            
            # Use tldextract to parse the hostname even if URL parsing failed
            extracted = tldextract.extract(hostname)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}"
            # Fallback: return hostname as-is if parsing fails
            return hostname
    
    return None


def assert_balanced_js_delimiters(js: str) -> None:
    """
    Perform basic sanity check on JavaScript code to detect syntax errors.
    
    Checks for:
    - Balanced brackets (parentheses, braces, square brackets)
    - Properly terminated string literals (single, double, template literals)
    - Proper escape sequences in strings
    
    Args:
        js: JavaScript code to validate.
    
    Raises:
        ValueError: If syntax errors are detected (unbalanced brackets, unterminated strings).
    """
    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    in_string = None
    escape = False

    for ch in js:
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == in_string:
                in_string = None
            continue

        if ch in ('"', "'", '`'):
            in_string = ch
        elif ch in '({[':
            stack.append(ch)
        elif ch in ')}]':
            if not stack or stack.pop() != pairs[ch]:
                raise ValueError("Unbalanced brackets in JavaScript")

    if in_string:
        raise ValueError("Unterminated string literal in JavaScript")

    if stack:
        raise ValueError("Unbalanced brackets in JavaScript")


def sanitize_filename(s: str, default: str = "file") -> str:
    """
    Sanitize string for use as filename.
    Args:
        s (str): The string to sanitize.
        default (str): The default string to return if the string is empty.
    Returns:
        str: The sanitized string.
    """
    s = "".join(c for c in s if c.isalnum() or c in ("-", "_", "."))
    return s or default


def format_bytes(num_bytes: int | float) -> str:
    """
    Format bytes as human-readable string.

    Args:
        num_bytes: Number of bytes to format.

    Returns:
        Human-readable string with appropriate unit (B, KB, MB, GB, TB).

    Example:
        >>> format_bytes(1024)
        "1.0 KB"
        >>> format_bytes(1536)
        "1.5 KB"
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def build_transaction_dir(url: str, ts_ms: int, output_dir: str) -> str:
    """
    Build per-transaction directory: date_timestamp_url.
    Args:
        url (str): The URL to build the directory for.
        ts_ms (int): The timestamp in milliseconds.
        output_dir (str): The output directory to build the directory in.
    Returns:
        str: The path to the directory.
    """
    date_str = time.strftime("%Y%m%d", time.localtime(ts_ms / 1000))
    url_core = (url or "").split("://", 1)[-1].split("?", 1)[0].strip("/")
    url_core = url_core.replace("/", "_")
    safe_url = sanitize_filename(url_core)[:120] or "url"
    dir_name = f"{date_str}_{ts_ms}_{safe_url}"
    dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def extract_object_schema(data: Any) -> dict[str, Any]:
    """
    Extract schema/structure from nested JSON data.

    Recursively processes nested data and returns a schema with explicit type
    information at every level. Useful for understanding the shape of large
    JSON responses.

    Args:
        data: Any data structure (dict, list, or primitive).

    Returns:
        Schema dict with _type, _count, and nested structure.

    Example:
        >>> data = [{"id": 1, "name": "a"}, {"id": 2}]
        >>> extract_object_schema(data)
        {"_type": "list", "_count": 2, "_items": {"_type": "dict", "id": ..., "name": ...}}
    """

    def walk(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            node: dict[str, Any] = {"_type": "dict"}
            for k, v in value.items():
                node[k] = walk(v)
            return node

        if isinstance(value, list):
            node = {"_type": "list", "_count": len(value)}
            if not value:
                node["_items"] = {}
                return node

            merged: dict[str, list[Any]] = defaultdict(list)
            for item in value:
                if isinstance(item, dict):
                    for k, v in item.items():
                        merged[k].append(v)

            if not merged:
                node["_items"] = {"_type": "scalar"}
                return node

            items_schema: dict[str, Any] = {"_type": "dict"}
            for k, vals in merged.items():
                items_schema[k] = merge(vals)
            node["_items"] = items_schema
            return node

        return {"_type": "scalar", "_count": 1}

    def merge(values: list[Any]) -> dict[str, Any]:
        if all(isinstance(v, dict) for v in values):
            merged: dict[str, list[Any]] = defaultdict(list)
            for d in values:
                for k, v in d.items():
                    merged[k].append(v)

            node: dict[str, Any] = {"_type": "dict", "_count": len(values)}
            for k, vals in merged.items():
                node[k] = merge(vals)
            return node

        if all(isinstance(v, list) for v in values):
            flattened = [x for lst in values for x in lst]
            return walk(flattened)

        return {"_type": "scalar", "_count": len(values)}

    return walk(data)