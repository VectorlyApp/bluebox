"""
tests/unit/test_read_workspace_file.py

Unit tests for the path traversal fix in BlueBoxAgent._read_workspace_file.
Tests the method directly using a minimal stub with only _workspace_dir set.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from bluebox.agents.bluebox_agent import BlueBoxAgent


def _make_stub(workspace_dir: Path) -> SimpleNamespace:
    """Create a minimal stub with just _workspace_dir for testing."""
    return SimpleNamespace(_workspace_dir=workspace_dir)


def _call(stub: SimpleNamespace, path: str, **kwargs: object) -> dict:
    """Call the unbound _read_workspace_file with our stub as self."""
    return BlueBoxAgent._read_workspace_file(stub, path=path, **kwargs)


class TestPathTraversalPrevention:
    """Edge cases for the path validation in _read_workspace_file."""

    def test_parent_traversal_blocked(self, tmp_path: Path) -> None:
        """../  should be denied."""
        stub = _make_stub(tmp_path / "workspace")
        stub._workspace_dir.mkdir()
        result = _call(stub, "../../../etc/passwd")
        assert "error" in result
        assert "Access denied" in result["error"]

    def test_absolute_path_outside_blocked(self, tmp_path: Path) -> None:
        """/etc/passwd should be denied."""
        stub = _make_stub(tmp_path / "workspace")
        stub._workspace_dir.mkdir()
        result = _call(stub, "/etc/passwd")
        assert "error" in result
        assert "Access denied" in result["error"]

    def test_sibling_dir_with_shared_prefix_blocked(self, tmp_path: Path) -> None:
        """A sibling directory whose name starts with the workspace name must be blocked.

        This was the original bug: string-based startswith("/workspace") would
        incorrectly allow "/workspace-evil/secret.txt".
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        evil = tmp_path / "workspace-evil"
        evil.mkdir()
        (evil / "secret.txt").write_text("stolen")

        stub = _make_stub(workspace)
        result = _call(stub, "../workspace-evil/secret.txt")
        assert "error" in result
        assert "Access denied" in result["error"]

    def test_symlink_escape_blocked(self, tmp_path: Path) -> None:
        """A symlink pointing outside the workspace should be denied."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("stolen")

        link = workspace / "escape"
        link.symlink_to(outside)

        stub = _make_stub(workspace)
        result = _call(stub, "escape/secret.txt")
        assert "error" in result
        assert "Access denied" in result["error"]

    def test_dot_dot_in_middle_of_path_blocked(self, tmp_path: Path) -> None:
        """Paths like subdir/../../outside should be denied."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "subdir").mkdir()

        stub = _make_stub(workspace)
        result = _call(stub, "subdir/../../etc/passwd")
        assert "error" in result
        assert "Access denied" in result["error"]


class TestLegitimateAccess:
    """Verify that valid paths still work after the fix."""

    def test_file_in_workspace_root(self, tmp_path: Path) -> None:
        """Reading a file directly in the workspace should succeed."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "data.txt").write_text("line1\nline2")

        stub = _make_stub(workspace)
        result = _call(stub, "data.txt")
        assert "error" not in result
        assert result["content"] == "line1\nline2"

    def test_file_in_subdirectory(self, tmp_path: Path) -> None:
        """Reading a file in a subdirectory should succeed."""
        workspace = tmp_path / "workspace"
        (workspace / "raw").mkdir(parents=True)
        (workspace / "raw" / "results.json").write_text('{"ok": true}')

        stub = _make_stub(workspace)
        result = _call(stub, "raw/results.json")
        assert "error" not in result
        assert '{"ok": true}' in result["content"]

    def test_workspace_dir_itself_rejected_as_not_file(self, tmp_path: Path) -> None:
        """Passing '.' resolves to workspace itself — should fail as 'not a file'."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        stub = _make_stub(workspace)
        result = _call(stub, ".")
        assert "error" in result
        assert "Not a file" in result["error"]

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """A path inside the workspace that doesn't exist should return file-not-found."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        stub = _make_stub(workspace)
        result = _call(stub, "nope.txt")
        assert "error" in result
        assert "File not found" in result["error"]

    def test_line_range(self, tmp_path: Path) -> None:
        """Line range parameters should still work."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "big.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))

        stub = _make_stub(workspace)
        result = _call(stub, "big.txt", start_line=3, end_line=5)
        assert "error" not in result
        assert result["content"] == "line3\nline4\nline5"
