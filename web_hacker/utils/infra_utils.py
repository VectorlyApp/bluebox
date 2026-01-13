"""
web_hacker/utils/infra_utils.py

Infrastructure utility functions for directory management and file operations.
"""

import shutil
import zipfile
from pathlib import Path

import requests

from web_hacker.utils.terminal_utils import YELLOW, print_colored


def clear_directory(path: Path) -> None:
    """Clear all files and subdirectories in a directory."""
    if path.exists():
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def remove_directory(path: Path) -> None:
    """Remove a directory and all its contents."""
    if path.exists():
        shutil.rmtree(path)


def download_zip(url: str, dest_path: Path) -> bool:
    """
    Download a zip file from URL to destination path.

    Args:
        url: URL to download from
        dest_path: Destination path for the downloaded file

    Returns:
        bool: True if download succeeded, False otherwise
    """
    try:
        print(f"  Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    print(f"\r  Downloaded: {downloaded / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="")

        print()  # newline after progress
        return True

    except requests.RequestException as e:
        print_colored(f"  Download failed: {e}", YELLOW)
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a zip file to a directory.

    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to

    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    try:
        print(f"  Extracting to {extract_to}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_to)
        return True
    except zipfile.BadZipFile as e:
        print_colored(f"  Extraction failed: {e}", YELLOW)
        return False
