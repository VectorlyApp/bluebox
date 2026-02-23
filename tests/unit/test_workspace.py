"""
tests/unit/test_workspace.py

Unit tests for AgentWorkspace / LocalWorkspace.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bluebox.agents.workspace import LocalWorkspace


class TestSaveFile:
    """Tests for LocalWorkspace.save_file."""

    def test_saves_file_with_content(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        result = ws.save_file("raw", "routine_result.json", '{"data": 1}')
        assert "output_file" in result
        saved = Path(result["output_file"])
        assert saved.exists()
        assert saved.read_text() == '{"data": 1}'
        assert saved.name == "routine_result.json"

    def test_creates_subdirectory(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        ws.save_file("custom_subdir", "test.json", "content")
        assert (tmp_path / "custom_subdir").is_dir()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        ws.save_file("raw", "test.json", "old")
        ws.save_file("raw", "test.json", "new")
        assert (tmp_path / "raw" / "test.json").read_text() == "new"

    def test_different_extensions(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        result = ws.save_file("outputs", "result.md", "# Result")
        assert result["output_file"].endswith(".md")

    def test_no_s3_key_in_result(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        result = ws.save_file("raw", "test.json", "data")
        assert "output_file_s3_key" not in result


class TestReadFile:
    """Tests for LocalWorkspace.read_file."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        (tmp_path / "test.txt").write_text("hello\nworld")
        result = ws.read_file("test.txt")
        assert "error" not in result
        assert result["content"] == "hello\nworld"
        assert result["path"] == "test.txt"

    def test_read_with_line_range(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        (tmp_path / "data.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = ws.read_file("data.txt", start_line=3, end_line=5)
        assert result["content"] == "line3\nline4\nline5"

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        result = ws.read_file("missing.txt")
        assert "error" in result

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        ws = LocalWorkspace(str(workspace))
        result = ws.read_file("../../../etc/passwd")
        assert "error" in result
        assert "Access denied" in result["error"]


class TestListFiles:
    """Tests for LocalWorkspace.list_files."""

    def test_empty_workspace(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        result = ws.list_files()
        assert result["total_files"] == 0
        assert "tree" in result

    def test_lists_files_in_subdirs(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        (tmp_path / "raw" / "result.json").write_text("{}")
        (tmp_path / "outputs" / "out.csv").write_text("a,b")
        result = ws.list_files()
        assert result["total_files"] == 2
        assert "result.json" in result["tree"]
        assert "out.csv" in result["tree"]


class TestLoadRawJson:
    """Tests for LocalWorkspace.load_raw_json."""

    def test_loads_json_files(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        raw = tmp_path / "raw"
        (raw / "a.json").write_text('{"key": "a"}')
        (raw / "b.json").write_text('{"key": "b"}')
        results = ws.load_raw_json()
        assert len(results) == 2
        assert results[0]["key"] == "a"
        assert results[1]["key"] == "b"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        raw = tmp_path / "raw"
        (raw / "good.json").write_text('{"ok": true}')
        (raw / "bad.json").write_text("not json")
        results = ws.load_raw_json()
        assert len(results) == 1

    def test_empty_raw_dir(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        results = ws.load_raw_json()
        assert results == []


class TestSnapshotAndDiffOutputs:
    """Tests for LocalWorkspace.snapshot_outputs and diff_outputs."""

    def test_detects_new_file(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        before = ws.snapshot_outputs()
        (tmp_path / "outputs" / "new.csv").write_text("data")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1
        assert "new.csv" in changed[0]

    def test_detects_modified_file(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        outputs = tmp_path / "outputs"
        f = outputs / "existing.csv"
        f.write_text("old")
        before = ws.snapshot_outputs()
        time.sleep(0.05)  # Ensure mtime changes
        f.write_text("new")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1

    def test_no_changes(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        outputs = tmp_path / "outputs"
        (outputs / "stable.csv").write_text("data")
        before = ws.snapshot_outputs()
        changed = ws.diff_outputs(before)
        assert changed == []


class TestEnsureDirs:
    """Tests for LocalWorkspace.ensure_dirs."""

    def test_creates_raw_outputs_and_context(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path / "new_workspace"))
        assert (tmp_path / "new_workspace" / "raw").is_dir()
        assert (tmp_path / "new_workspace" / "outputs").is_dir()
        assert (tmp_path / "new_workspace" / "context").is_dir()

    def test_idempotent(self, tmp_path: Path) -> None:
        ws = LocalWorkspace(str(tmp_path))
        ws.ensure_dirs()
        ws.ensure_dirs()  # Should not raise
        assert (tmp_path / "raw").is_dir()
