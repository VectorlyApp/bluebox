"""
bluebox/agents/workspace.py

Abstract workspace interface and local filesystem implementation.

Contains:
- AgentWorkspace: ABC defining how agents interact with file storage
- LocalWorkspace: Local filesystem implementation
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

from bluebox.utils.infra_utils import read_file_lines
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class AgentWorkspace(ABC):
    """
    Abstract workspace that agents use for file I/O.

    A workspace has two logical subdirectories:
    - raw/    : input data (e.g., routine results saved automatically)
    - outputs/: agent-generated output files (e.g., CSVs, processed JSON)
    """

    @property
    @abstractmethod
    def root_path(self) -> Path:
        """Workspace root directory."""

    @abstractmethod
    def save_file(
        self,
        subdirectory: str,
        filename_prefix: str,
        content: str,
        extension: str = ".json",
    ) -> dict[str, str]:
        """Save content with a unique timestamped filename.

        Args:
            subdirectory: Logical subdirectory ("raw" or "outputs").
            filename_prefix: Prefix for the generated filename.
            content: File content to write.
            extension: File extension including the dot.

        Returns:
            Dict with at least "output_file" key (the saved path).
        """

    @abstractmethod
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Read a file by relative path with optional line range.

        Must enforce path traversal protection.

        Args:
            path: Relative path within the workspace.
            start_line: Optional 1-based start line.
            end_line: Optional 1-based end line (inclusive).

        Returns:
            Dict with "content" and "line_range", or "error".
        """

    @abstractmethod
    def list_files(self) -> dict[str, Any]:
        """List all files in the workspace as a directory tree.

        Returns:
            Dict with "tree" (str) and "total_files" (int).
        """

    @abstractmethod
    def load_raw_json(self) -> list[dict[str, Any]]:
        """Load all JSON files from the raw/ subdirectory.

        Returns:
            List of parsed dicts, one per JSON file (sorted by name).
        """

    @abstractmethod
    def snapshot_outputs(self) -> dict[str, float]:
        """Take a snapshot of files in outputs/ (path -> mtime).

        Used before code execution to detect new/modified files afterward.
        """

    @abstractmethod
    def diff_outputs(self, before: dict[str, float]) -> list[str]:
        """Compare current outputs/ against a prior snapshot.

        Args:
            before: Snapshot from a previous snapshot_outputs() call.

        Returns:
            List of file paths that are new or modified since the snapshot.
        """

    @abstractmethod
    def ensure_dirs(self) -> None:
        """Ensure the workspace directory structure exists (raw/, outputs/)."""


class LocalWorkspace(AgentWorkspace):
    """Workspace backed by the local filesystem."""

    def __init__(self, workspace_dir: str = "./bluebox_workspace") -> None:
        self._workspace_dir = Path(workspace_dir)
        self._raw_dir = self._workspace_dir / "raw"
        self._outputs_dir = self._workspace_dir / "outputs"
        self._execution_counter: int = 0
        self._counter_lock = threading.Lock()

    @property
    def root_path(self) -> Path:
        return self._workspace_dir

    def save_file(
        self,
        subdirectory: str,
        filename_prefix: str,
        content: str,
        extension: str = ".json",
    ) -> dict[str, str]:
        directory = self._workspace_dir / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        with self._counter_lock:
            self._execution_counter += 1
            idx = self._execution_counter
        timestamp = datetime.now().strftime("%y-%m-%d-%H%M%S")
        filename = f"{timestamp}-{filename_prefix}_{idx}{extension}"
        output_path = directory / filename
        output_path.write_text(content)
        logger.info("Result saved to %s", output_path)
        return {"output_file": str(output_path)}

    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        resolved = (self._workspace_dir / path).resolve()
        workspace_resolved = self._workspace_dir.resolve()
        try:
            resolved.relative_to(workspace_resolved)
        except ValueError:
            return {"error": f"Access denied: '{path}' is outside the workspace directory"}

        result = read_file_lines(resolved, start_line=start_line, end_line=end_line)
        result["path"] = path
        return result

    def list_files(self) -> dict[str, Any]:
        self._workspace_dir.mkdir(parents=True, exist_ok=True)

        tree_lines: list[str] = []
        total_files = 0

        for dirpath, dirnames, filenames in sorted(self._workspace_dir.walk()):
            rel_dir = dirpath.relative_to(self._workspace_dir)
            depth = len(rel_dir.parts)
            indent = "  " * depth
            dir_name = rel_dir.name or str(self._workspace_dir.name)
            tree_lines.append(f"{indent}{dir_name}/")

            dirnames.sort()
            for filename in sorted(filenames):
                filepath = dirpath / filename
                size = filepath.stat().st_size
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                tree_lines.append(f"{indent}  {filename}  ({size_str})")
                total_files += 1

        return {
            "tree": "\n".join(tree_lines),
            "total_files": total_files,
        }

    def load_raw_json(self) -> list[dict[str, Any]]:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        results: list[dict[str, Any]] = []
        for json_file in sorted(self._raw_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text())
                results.append(data)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load %s: %s", json_file, e)
        return results

    def snapshot_outputs(self) -> dict[str, float]:
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
        snapshot: dict[str, float] = {}
        for p in self._outputs_dir.iterdir():
            if p.is_file():
                snapshot[str(p)] = p.stat().st_mtime
        return snapshot

    def diff_outputs(self, before: dict[str, float]) -> list[str]:
        changed: list[str] = []
        for p in self._outputs_dir.iterdir():
            if not p.is_file():
                continue
            path_str = str(p)
            mtime = p.stat().st_mtime
            if path_str not in before or before[path_str] < mtime:
                changed.append(path_str)
        return changed

    def ensure_dirs(self) -> None:
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._outputs_dir.mkdir(parents=True, exist_ok=True)
