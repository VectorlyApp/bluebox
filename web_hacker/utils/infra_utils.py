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


def resolve_glob_patterns(
    patterns: list[str],
    extensions: set[str] | None = None,
    recursive: bool = True,
    raise_on_missing: bool = False,
) -> list[Path]:
    """
    Resolve glob patterns to file paths.

    Supports gitignore-style patterns:
    - "path/to/file.py" - single file
    - "path/to/dir/" - directory (recursive if recursive=True)
    - "path/to/dir/**/*.py" - explicit recursive glob
    - "!pattern" - exclude files matching pattern

    Args:
        patterns: List of paths/globs, with optional ! prefix for exclusions
        extensions: Optional set of allowed extensions (e.g., {".py", ".md"})
        recursive: Whether to scan directories recursively (default True)
        raise_on_missing: Whether to raise ValueError for non-existent paths (default False)

    Returns:
        List of resolved file Paths

    Raises:
        ValueError: If raise_on_missing=True and a path doesn't exist
    """
    include_files: set[Path] = set()
    exclude_patterns: list[str] = []

    for pattern in patterns:
        if pattern.startswith("!"):
            exclude_patterns.append(pattern[1:])
            continue

        path = Path(pattern)

        if path.is_file():
            # Single file
            if extensions is None or path.suffix.lower() in extensions:
                include_files.add(path.resolve())
        elif path.is_dir():
            # Directory - scan for files
            iter_func = path.rglob("*") if recursive else path.iterdir()
            for file in iter_func:
                if file.is_file():
                    if extensions is None or file.suffix.lower() in extensions:
                        include_files.add(file.resolve())
        elif "*" in pattern or "?" in pattern:
            # Glob pattern - find base directory
            parts = Path(pattern).parts
            base_idx = 0
            for i, part in enumerate(parts):
                if "*" in part or "?" in part:
                    break
                base_idx = i + 1
            base_path = Path(*parts[:base_idx]) if base_idx > 0 else Path(".")
            glob_pattern = str(Path(*parts[base_idx:])) if base_idx < len(parts) else "*"

            if base_path.exists():
                for file in base_path.glob(glob_pattern):
                    if file.is_file():
                        if extensions is None or file.suffix.lower() in extensions:
                            include_files.add(file.resolve())
            elif raise_on_missing:
                raise ValueError(f"Base path does not exist: {base_path}")
        else:
            # Path doesn't exist
            if raise_on_missing:
                raise ValueError(f"Path does not exist: {pattern}")
            continue

    # Apply exclusions
    for exc_pattern in exclude_patterns:
        exc_path = Path(exc_pattern)
        if exc_path.is_file():
            include_files.discard(exc_path.resolve())
        elif exc_path.is_dir():
            # Exclude entire directory
            exc_resolved = exc_path.resolve()
            include_files = {f for f in include_files if not str(f).startswith(str(exc_resolved))}
        else:
            # Glob-based exclusion
            parts = Path(exc_pattern).parts
            base_idx = 0
            for i, part in enumerate(parts):
                if "*" in part or "?" in part:
                    break
                base_idx = i + 1
            base_path = Path(*parts[:base_idx]) if base_idx > 0 else Path(".")
            glob_pattern = str(Path(*parts[base_idx:])) if base_idx < len(parts) else "*"

            if base_path.exists():
                for file in base_path.glob(glob_pattern):
                    include_files.discard(file.resolve())

    return sorted(include_files)
