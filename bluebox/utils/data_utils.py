"""
bluebox/utils/data_utils.py

Data loading, writing, and transformation utilities.

Contains:
- load_data(): Load JSON/JSONL files
- write_json_file(), write_jsonl(): Save data to files
- get_text_from_html(): Extract text from HTML
- resolve_dotted_path(): Access nested dict values by dot notation
- apply_params_to_str(): Substitute {{placeholders}} in strings
- apply_params_to_dict(): Substitute {{placeholders}} in dicts with type-aware replacement
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
from decimal import Decimal
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

import tldextract
from bs4 import BeautifulSoup

from bluebox.data_models.routine.parameter import ParameterType
from bluebox.utils.exceptions import UnsupportedFileFormat
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# regex matching a standalone placeholder: the entire string is "{{key}}" with optional whitespace
_STANDALONE_PLACEHOLDER_RE = re.compile(
    pattern=r"^\{\{\s*([^}]+?)\s*\}\}$"
)

# regex matching any {{key}} pattern (for substring replacement)
_PLACEHOLDER_RE = re.compile(
    pattern=r"\{\{\s*([^}]+?)\s*\}\}"
)


def load_data(file_path: Path) -> Union[dict, list]:
    """
    Load data from a file.
    Raises:
        UnsupportedFileFormat: If the file is of an unsupported type.
    Args:
        file_path (str): Path to the JSON file.
    Returns:
        Union[dict, list]: Data contained in file.
    """
    file_path_str = str(file_path)
    if file_path_str.endswith(".json"):
        with open(file_path_str, mode="r", encoding="utf-8") as data_file:
            json_data = json.load(data_file)
            return json_data

    raise UnsupportedFileFormat(f"No support for provided file type: {file_path_str}.")


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
    # use built-in html parser for robustness
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-visible elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Get visible text
    text = soup.get_text(separator="\n")

    # Normalize whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = "\n".join(chunk for chunk in chunks if chunk)

    # Remove ALL consecutive newlines - replace any sequence of 2+ newlines with single newline
    # Handle both \n and \r\n line endings
    clean_text = re.sub(r'[\r\n]+', '\n', clean_text)
    # Remove leading and trailing whitespace
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
        if re.search(
            pattern=rx,
            string=u,
            flags=re.IGNORECASE,
        ):
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


def _coerce_value(value: Any, param_type: str) -> Any:
    """
    Coerce a parameter value to the correct Python type based on the parameter schema type.
    Args:
        value: The value to coerce.
        param_type: The parameter type to coerce to.
    Returns:
        The coerced value.
    """
    if param_type in (ParameterType.INTEGER,):
        return int(value)
    elif param_type in (ParameterType.NUMBER,):
        return float(value)
    elif param_type in (ParameterType.BOOLEAN,):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    else:
        # string, date, datetime, email, url, enum; all string types
        return str(value)


def apply_params_to_dict(
    d: dict | list,
    parameters_dict: dict,
    param_type_map: dict[str, str],
) -> dict | list:
    """
    Walk a dict/list and replace {{param}} placeholders with typed values.

    For standalone placeholders (the entire string value is "{{key}}"):
      - Uses param_type_map to determine the output Python type
      - e.g., integer → int, number → float, boolean → bool, string → str

    For substring placeholders ("prefix {{key}} suffix"):
      - Always str(value) substitution; result stays a string

    Non-parameter placeholders ({{sessionStorage:...}}, {{uuid}}, etc.) are left untouched.

    Args:
        d: The dict or list to process (not mutated; returns a new copy).
        parameters_dict: Mapping of parameter names to their values.
        param_type_map: Mapping of parameter names to their ParameterType string.

    Returns:
        A new dict/list with placeholders replaced.
    """
    if not parameters_dict:
        return d

    def _resolve_value(v: Any) -> Any:
        if isinstance(v, str):
            # check for standalone placeholder
            m = _STANDALONE_PLACEHOLDER_RE.match(v)
            if m:
                key = m.group(1)
                if key in parameters_dict:
                    param_type = param_type_map.get(key, "string")
                    return _coerce_value(parameters_dict[key], param_type)
                # not a user param (e.g. sessionStorage:...), leave as-is
                return v

            # substring replacement: replace all {{key}} occurrences within the string
            def _sub(match: re.Match) -> str:
                key = match.group(1)
                if key in parameters_dict:
                    return str(parameters_dict[key])
                return match.group(0)  # leave non-parameter placeholders untouched

            return _PLACEHOLDER_RE.sub(
                repl=_sub,
                string=v,
            )
        elif isinstance(v, dict):
            return {k: _resolve_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [_resolve_value(item) for item in v]
        return v

    return _resolve_value(d)


def apply_params_to_str(text: str, parameters_dict: dict | None) -> str:
    """
    Replace {{param}} placeholders in a plain string with str(value).

    Use this for URLs, CSS selectors, text fields, JS code, filenames — contexts
    where the result is always a string.

    Non-parameter placeholders ({{sessionStorage:...}}, etc.) are left untouched.

    Args:
        text: Text containing parameter placeholders.
        parameters_dict: Dictionary of parameter values.

    Returns:
        Text with parameter placeholders replaced.
    """
    if not text or not parameters_dict:
        return text

    def _sub(match: re.Match) -> str:
        key = match.group(1)
        if key in parameters_dict:
            return str(parameters_dict[key])
        return match.group(0)

    result = _PLACEHOLDER_RE.sub(
        repl=_sub,
        string=text,
    )
    return result


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
        # try to parse the URL to extract the hostname
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        
        if not hostname:
            return None
        
        # use tldextract to properly extract the root domain, handling special TLDs
        extracted = tldextract.extract(hostname)
        if extracted.domain and extracted.suffix:
            return f"{extracted.domain}.{extracted.suffix}"
        # fallback: if domain extraction fails, return the hostname as-is
        return hostname
        
    except Exception:
        # if URL parsing fails (e.g., contains placeholders), try to extract hostname manually
        # match protocol://hostname pattern
        match = re.match(
            pattern=r'^[^:]+://([^/\?\:]+)',
            string=url
        )
        if match and match.group(1):
            hostname = match.group(1)
            # remove port if present
            hostname = hostname.split(':')[0]
            
            # use tldextract to parse the hostname even if URL parsing failed
            extracted = tldextract.extract(hostname)
            if extracted.domain and extracted.suffix:
                return f"{extracted.domain}.{extracted.suffix}"
            # fallback: return hostname as-is if parsing fails
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
    date_str = time.strftime(
        format="%Y%m%d",
        time_tuple=time.localtime(ts_ms / 1000)
    )
    url_core = (url or "").split("://", 1)[-1].split("?", 1)[0].strip("/")
    url_core = url_core.replace("/", "_")
    safe_url = sanitize_filename(url_core)[:120] or "url"
    dir_name = f"{date_str}_{ts_ms}_{safe_url}"
    dir_path = os.path.join(output_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path
