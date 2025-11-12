"""
src/utils/data_utils.py

Utility functions for loading data.
"""

import datetime
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Union
from bs4 import BeautifulSoup

from src.utils.exceptions import UnsupportedFileFormat


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
    
    # Use the built-in html parser for robustness
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