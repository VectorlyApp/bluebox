"""
tests/unit/test_workspace.py

Unit tests for AgentWorkspace / LocalAgentWorkspace.
"""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path

import pytest

from bluebox.agents.workspace import (
    LocalAgentWorkspace,
    LocalWorkspace,
)


class TestBackwardCompatAlias:
    """LocalWorkspace is an alias for LocalAgentWorkspace."""

    def test_alias_is_same_class(self) -> None:
        assert LocalWorkspace is LocalAgentWorkspace

    def test_from_directory_path_constructor(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "resumed_workspace"
        ws_dir.mkdir()

        ws = LocalAgentWorkspace.from_directory_path(ws_dir)

        assert ws.root_path == ws_dir
        assert (ws_dir / "raw").is_dir()
        assert (ws_dir / "output").is_dir()
        assert (ws_dir / "context").is_dir()
        assert (ws_dir / "scratch").is_dir()
        assert (ws_dir / "meta").is_dir()


class TestSaveFile:
    """Tests for the backward-compatible save_file method."""

    def test_saves_file_with_content(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.save_file("raw", "routine_result.json", '{"data": 1}')
        assert "output_file" in result
        assert "artifact_id" in result
        saved = Path(result["output_file"])
        assert saved.exists()
        assert saved.read_text() == '{"data": 1}'
        assert saved.name == "routine_result.json"

    def test_creates_subdirectory_for_unknown_source(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.save_file("custom_subdir", "test.json", "content")
        assert (tmp_path / "custom_subdir").is_dir()

    def test_dedupes_filename_on_collision(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        r1 = ws.save_file("raw", "test.json", "first")
        r2 = ws.save_file("raw", "test.json", "second")
        assert Path(r1["output_file"]).read_text() == "first"
        assert Path(r2["output_file"]).exists()
        assert Path(r2["output_file"]).read_text() == "second"
        assert r1["artifact_id"] != r2["artifact_id"]

    def test_different_extensions(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.save_file("output", "result.md", "# Result")
        assert result["output_file"].endswith(".md")

    def test_no_s3_key_in_result(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.save_file("raw", "test.json", "data")
        assert "output_file_s3_key" not in result

    def test_rejects_filename_with_path_separator(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid filename"):
            ws.save_file("raw", "../escape.json", "data")

    def test_rejects_filename_with_windows_separator(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid filename"):
            ws.save_file("raw", "..\\escape.json", "data")

    def test_rejects_subdirectory_path_traversal(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid subdirectory"):
            ws.save_file("../evil", "x.txt", "data")

    def test_rejects_absolute_subdirectory(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid subdirectory"):
            ws.save_file("/tmp/evil", "x.txt", "data")


class TestSaveArtifact:
    """Tests for the v2 save_artifact method."""

    def test_creates_artifact_with_ref(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ref = ws.save_artifact("raw", "data.json", '{"x": 1}')
        assert ref.artifact_id == "a_000001"
        assert ref.source == "raw"
        assert ref.content_type == "json"
        assert ref.size_bytes > 0
        assert ref.sha256 is not None

    def test_monotonic_ids(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        r1 = ws.save_artifact("raw", "a.json", "{}")
        r2 = ws.save_artifact("output", "b.csv", "a,b")
        r3 = ws.save_artifact("context", "c.md", "# C")
        assert r1.index == 1
        assert r2.index == 2
        assert r3.index == 3

    def test_manifest_written(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.save_artifact("raw", "data.json", '{"x": 1}', tool_name="my_tool")
        manifest = (tmp_path / "meta" / "manifest.jsonl").read_text()
        lines = [l for l in manifest.strip().splitlines() if l.strip()]
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["tool_name"] == "my_tool"
        assert entry["artifact"]["artifact_id"] == "a_000001"

    def test_source_routing(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        r1 = ws.save_artifact("raw", "a.json", "{}")
        r2 = ws.save_artifact("output", "b.csv", "a,b")
        r3 = ws.save_artifact("context", "c.md", "# C")
        assert r1.relative_path.startswith("raw/")
        assert r2.relative_path.startswith("output/")
        assert r3.relative_path.startswith("context/")

    def test_binary_content(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ref = ws.save_artifact("raw", "data.bin", b"\x00\x01\x02")
        assert ref.content_type == "binary"
        assert ref.size_bytes == 3

    def test_custom_content_type(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ref = ws.save_artifact("raw", "data.txt", "hello", content_type="csv")
        assert ref.content_type == "csv"

    def test_metadata_stored(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ref = ws.save_artifact("raw", "d.json", "{}", metadata={"key": "val"})
        assert ref.metadata == {"key": "val"}

    def test_rejects_artifact_filename_with_path_separator(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid filename"):
            ws.save_artifact("raw", "../escape.json", "{}")

    def test_rejects_artifact_filename_with_windows_separator(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(ValueError, match="Invalid filename"):
            ws.save_artifact("raw", "..\\escape.json", "{}")


class TestListArtifacts:
    """Tests for list_artifacts."""

    def test_empty(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        assert ws.list_artifacts() == []

    def test_filter_by_source(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.save_artifact("raw", "a.json", "{}")
        ws.save_artifact("output", "b.csv", "a,b")
        ws.save_artifact("raw", "c.json", "{}")
        assert len(ws.list_artifacts("raw")) == 2
        assert len(ws.list_artifacts("output")) == 1
        assert len(ws.list_artifacts("context")) == 0
        assert len(ws.list_artifacts()) == 3


class TestMountedInputs:
    """Tests for attach_input_file/list_mounted_inputs."""

    def test_attach_input_file_creates_hardlink(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        source = tmp_path / "source.jsonl"
        source.write_text('{"id": 1}\n')

        ref = ws.attach_input_file("network_events", source)
        target = tmp_path / ref.relative_path

        assert target.exists()
        assert ref.relative_path == "raw/network_events.jsonl"
        assert target.read_text() == source.read_text()

        src_stat = source.stat()
        tgt_stat = target.stat()
        assert src_stat.st_ino == tgt_stat.st_ino
        assert src_stat.st_dev == tgt_stat.st_dev

    def test_attach_input_file_rejects_invalid_name(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        source = tmp_path / "source.jsonl"
        source.write_text("{}\n")

        with pytest.raises(ValueError, match="Invalid mount name"):
            ws.attach_input_file("../escape", source)

    def test_attach_input_file_rejects_missing_source(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            ws.attach_input_file("network_events", tmp_path / "missing.jsonl")

    def test_attach_input_file_rejects_cross_filesystem(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        source = tmp_path / "source.jsonl"
        source.write_text("{}\n")

        def _raise_exdev(src: str | Path, dst: str | Path) -> None:
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        monkeypatch.setattr(os, "link", _raise_exdev)

        with pytest.raises(ValueError, match="Cannot hardlink across filesystems"):
            ws.attach_input_file("network_events", source)

    def test_attach_input_file_rejects_conflicting_existing_target(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        source = tmp_path / "source.jsonl"
        source.write_text('{"id": 1}\n')

        target = tmp_path / "raw" / "network_events.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"id": 999}\n')

        with pytest.raises(ValueError, match="already exists with different inode"):
            ws.attach_input_file("network_events", source)

    def test_list_mounted_inputs_reads_manifest(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        source = tmp_path / "source.jsonl"
        source.write_text('{"id": 1}\n')
        created = ws.attach_input_file("network_events", source)

        mounted = ws.list_mounted_inputs()
        assert len(mounted) == 1
        assert mounted[0].mount_id == created.mount_id
        assert mounted[0].relative_path == "raw/network_events.jsonl"


class TestReadArtifact:
    """Tests for read_artifact."""

    def test_read_existing(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ref = ws.save_artifact("raw", "data.json", '{"x": 1}')
        result = ws.read_artifact(ref.artifact_id)
        assert "error" not in result
        assert '{"x": 1}' in result["content"]

    def test_read_missing(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.read_artifact("a_999999")
        assert "error" in result


class TestReadFile:
    """Tests for LocalAgentWorkspace.read_file."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        (tmp_path / "test.txt").write_text("hello\nworld")
        result = ws.read_file("test.txt")
        assert "error" not in result
        assert result["content"] == "hello\nworld"
        assert result["path"] == "test.txt"

    def test_read_with_line_range(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        (tmp_path / "data.txt").write_text("\n".join(f"line{i}" for i in range(1, 11)))
        result = ws.read_file("data.txt", start_line=3, end_line=5)
        assert result["content"] == "line3\nline4\nline5"

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.read_file("missing.txt")
        assert "error" in result

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        ws = LocalAgentWorkspace(str(workspace))
        result = ws.read_file("../../../etc/passwd")
        assert "error" in result
        assert "Access denied" in result["error"]


class TestListFiles:
    """Tests for LocalAgentWorkspace.list_files."""

    def test_workspace_has_manifest_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.list_files()
        # meta/ contains manifest.jsonl (empty but touched)
        assert result["total_files"] >= 1
        assert "tree" in result

    def test_lists_files_in_subdirs(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        (tmp_path / "raw" / "result.json").write_text("{}")
        (tmp_path / "output" / "out.csv").write_text("a,b")
        result = ws.list_files()
        assert "result.json" in result["tree"]
        assert "out.csv" in result["tree"]
        # 2 user files + manifest.jsonl
        assert result["total_files"] >= 3


class TestSummarizeForPrompt:
    """Tests for generate_summary."""

    def test_empty_workspace_summary(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        summary = ws.generate_summary()
        assert "## Workspace State" in summary
        assert "Artifacts: 0 total" in summary
        assert "Recent artifacts: none" in summary

    def test_summary_includes_counts_and_recent(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.save_artifact("raw", "a.json", '{"k": "v"}')
        ws.save_artifact("output", "b.csv", "a,b")
        ws.save_artifact("context", "c.md", "# context")

        summary = ws.generate_summary(max_artifacts=3)
        assert "Artifacts: 3 total (raw: 1, output: 1, context: 1)" in summary
        assert "a_000001 [raw] raw/a.json" in summary
        assert "a_000002 [output] output/b.csv" in summary
        assert "a_000003 [context] context/c.md" in summary

    def test_summary_respects_max_artifacts(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.save_artifact("raw", "a.json", "{}")
        ws.save_artifact("raw", "b.json", "{}")
        ws.save_artifact("raw", "c.json", "{}")

        summary = ws.generate_summary(max_artifacts=2)
        assert "a_000003 [raw] raw/c.json" in summary
        assert "a_000002 [raw] raw/b.json" in summary
        assert "a_000001 [raw] raw/a.json" not in summary
        assert "... and 1 more artifact(s)" in summary

    def test_summary_truncates_long_preview(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        long_text = "x" * 200
        ws.save_artifact("raw", "long.txt", long_text)

        summary = ws.generate_summary(max_summary_chars=50)
        assert ("x" * 50) + "..." in summary


class TestLoadRawJson:
    """Tests for LocalAgentWorkspace.load_raw_json."""

    def test_loads_json_files(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        raw = tmp_path / "raw"
        (raw / "a.json").write_text('{"key": "a"}')
        (raw / "b.json").write_text('{"key": "b"}')
        results = ws.load_raw_json()
        assert len(results) == 2
        assert results[0]["key"] == "a"
        assert results[1]["key"] == "b"

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        raw = tmp_path / "raw"
        (raw / "good.json").write_text('{"ok": true}')
        (raw / "bad.json").write_text("not json")
        results = ws.load_raw_json()
        assert len(results) == 1

    def test_empty_raw_dir(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        results = ws.load_raw_json()
        assert results == []


class TestSnapshotAndDiffOutputs:
    """Tests for backward-compatible snapshot_outputs and diff_outputs."""

    def test_detects_new_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        before = ws.snapshot_outputs()
        (tmp_path / "output" / "new.csv").write_text("data")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1
        assert "new.csv" in changed[0]

    def test_detects_modified_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        output = tmp_path / "output"
        f = output / "existing.csv"
        f.write_text("old")
        before = ws.snapshot_outputs()
        time.sleep(0.05)
        f.write_text("new")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1

    def test_no_changes(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        output = tmp_path / "output"
        (output / "stable.csv").write_text("data")
        before = ws.snapshot_outputs()
        changed = ws.diff_outputs(before)
        assert changed == []


class TestSnapshotPaths:
    """Tests for v2 snapshot_paths / diff_snapshot."""

    def test_snapshot_captures_files(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        (tmp_path / "raw" / "a.json").write_text("{}")
        snap = ws.snapshot_paths(["raw"])
        assert len(snap.files) == 1
        assert "raw/a.json" in snap.files

    def test_diff_detects_created(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        before = ws.snapshot_paths(["output"])
        (tmp_path / "output" / "new.csv").write_text("x")
        after = ws.snapshot_paths(["output"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.created) == 1
        assert delta.created[0].relative_path == "output/new.csv"

    def test_diff_detects_deleted(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        f = tmp_path / "output" / "old.csv"
        f.write_text("x")
        before = ws.snapshot_paths(["output"])
        f.unlink()
        after = ws.snapshot_paths(["output"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.deleted) == 1

    def test_diff_detects_modified(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        f = tmp_path / "output" / "data.csv"
        f.write_text("old")
        before = ws.snapshot_paths(["output"])
        time.sleep(0.05)
        f.write_text("new-longer")
        after = ws.snapshot_paths(["output"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.modified) == 1


class TestEnsureDirs:
    """Tests for LocalAgentWorkspace.ensure_dirs."""

    def test_creates_all_directories(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path / "new_workspace"))
        assert (tmp_path / "new_workspace" / "raw").is_dir()
        assert (tmp_path / "new_workspace" / "output").is_dir()
        assert (tmp_path / "new_workspace" / "context").is_dir()
        assert (tmp_path / "new_workspace" / "scratch").is_dir()
        assert (tmp_path / "new_workspace" / "meta").is_dir()
        assert (tmp_path / "new_workspace" / "meta" / "manifest.jsonl").is_file()

    def test_idempotent(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        ws.ensure_dirs()
        ws.ensure_dirs()  # Should not raise
        assert (tmp_path / "raw").is_dir()


class TestCleanup:
    """Tests for cleanup."""

    def test_cleanup_removes_root(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "ws"
        ws = LocalAgentWorkspace(str(ws_dir))
        ws.save_artifact("raw", "a.json", "{}")
        assert ws_dir.exists()
        ws.cleanup(remove_root=True)
        assert not ws_dir.exists()

    def test_cleanup_noop_without_flag(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "ws"
        ws = LocalAgentWorkspace(str(ws_dir))
        ws.cleanup(remove_root=False)
        assert ws_dir.exists()


class TestArtifactIndexResume:
    """Tests for manifest-based artifact index resume across instances."""

    def test_index_survives_restart(self, tmp_path: Path) -> None:
        ws1 = LocalAgentWorkspace(str(tmp_path))
        ws1.save_artifact("raw", "a.json", "{}")
        ws1.save_artifact("raw", "b.json", "{}")
        # New instance picks up where the old one left off
        ws2 = LocalAgentWorkspace(str(tmp_path))
        ref = ws2.save_artifact("raw", "c.json", "{}")
        assert ref.index == 3
