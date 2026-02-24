"""
tests/unit/test_workspace.py

Unit tests for AgentWorkspace / LocalAgentWorkspace.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from bluebox.agents.workspace import (
    ArtifactRef,
    LocalAgentWorkspace,
    LocalWorkspace,
)


# ---------------------------------------------------------------------------
# Helper: count user files (exclude meta/ directory)
# ---------------------------------------------------------------------------

def _user_file_count(result: dict) -> int:
    """Count files reported by list_files, excluding meta/ internals."""
    count = 0
    for line in result["tree"].splitlines():
        stripped = line.strip()
        # Lines with a size suffix like "(123B)" are file entries
        if "(" in stripped and ")" in stripped and "/" not in stripped.split("(")[0].strip():
            # Skip meta/ files
            count += 1
    return result["total_files"]


class TestBackwardCompatAlias:
    """LocalWorkspace is an alias for LocalAgentWorkspace."""

    def test_alias_is_same_class(self) -> None:
        assert LocalWorkspace is LocalAgentWorkspace

    def test_from_directory_path_constructor(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / 'resumed_workspace'
        ws_dir.mkdir()

        ws = LocalAgentWorkspace.from_directory_path(ws_dir)

        assert ws.root_path == ws_dir
        assert (ws_dir / 'raw').is_dir()
        assert (ws_dir / 'outputs').is_dir()
        assert (ws_dir / 'context').is_dir()
        assert (ws_dir / 'scratch').is_dir()
        assert (ws_dir / 'meta').is_dir()


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
        # First write lands at test.json, second gets deduplicated
        assert Path(r1["output_file"]).read_text() == "first"
        assert Path(r2["output_file"]).exists()
        assert Path(r2["output_file"]).read_text() == "second"
        assert r1["artifact_id"] != r2["artifact_id"]

    def test_different_extensions(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.save_file("outputs", "result.md", "# Result")
        assert result["output_file"].endswith(".md")

    def test_no_s3_key_in_result(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.save_file("raw", "test.json", "data")
        assert "output_file_s3_key" not in result


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
        assert r2.relative_path.startswith("outputs/")
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

    def test_workspace_has_meta_files(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        result = ws.list_files()
        # meta/ contains run.json and counters.json
        assert result["total_files"] >= 2
        assert "tree" in result

    def test_lists_files_in_subdirs(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        (tmp_path / "raw" / "result.json").write_text("{}")
        (tmp_path / "outputs" / "out.csv").write_text("a,b")
        result = ws.list_files()
        assert "result.json" in result["tree"]
        assert "out.csv" in result["tree"]
        # 2 user files + 2 meta files
        assert result["total_files"] >= 4


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
        assert "a_000002 [output] outputs/b.csv" in summary
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
        (tmp_path / "outputs" / "new.csv").write_text("data")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1
        assert "new.csv" in changed[0]

    def test_detects_modified_file(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        outputs = tmp_path / "outputs"
        f = outputs / "existing.csv"
        f.write_text("old")
        before = ws.snapshot_outputs()
        time.sleep(0.05)  # Ensure mtime changes
        f.write_text("new")
        changed = ws.diff_outputs(before)
        assert len(changed) == 1

    def test_no_changes(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        outputs = tmp_path / "outputs"
        (outputs / "stable.csv").write_text("data")
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
        before = ws.snapshot_paths(["outputs"])
        (tmp_path / "outputs" / "new.csv").write_text("x")
        after = ws.snapshot_paths(["outputs"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.created) == 1
        assert delta.created[0].relative_path == "outputs/new.csv"

    def test_diff_detects_deleted(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        f = tmp_path / "outputs" / "old.csv"
        f.write_text("x")
        before = ws.snapshot_paths(["outputs"])
        f.unlink()
        after = ws.snapshot_paths(["outputs"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.deleted) == 1

    def test_diff_detects_modified(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path))
        f = tmp_path / "outputs" / "data.csv"
        f.write_text("old")
        before = ws.snapshot_paths(["outputs"])
        time.sleep(0.05)
        f.write_text("new-longer")
        after = ws.snapshot_paths(["outputs"])
        delta = ws.diff_snapshot(before, after)
        assert len(delta.modified) == 1


class TestEnsureDirs:
    """Tests for LocalAgentWorkspace.ensure_dirs."""

    def test_creates_all_directories(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path / "new_workspace"))
        assert (tmp_path / "new_workspace" / "raw").is_dir()
        assert (tmp_path / "new_workspace" / "outputs").is_dir()
        assert (tmp_path / "new_workspace" / "context").is_dir()
        assert (tmp_path / "new_workspace" / "scratch").is_dir()
        assert (tmp_path / "new_workspace" / "meta").is_dir()

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


class TestRunMetadata:
    """Tests for run metadata initialization."""

    def test_run_json_created(self, tmp_path: Path) -> None:
        ws = LocalAgentWorkspace(str(tmp_path), agent_id="test-agent")
        run_path = tmp_path / "meta" / "run.json"
        assert run_path.exists()
        data = json.loads(run_path.read_text())
        assert data["agent_id"] == "test-agent"
        assert data["run_id"].startswith("run_")

    def test_run_json_not_overwritten(self, tmp_path: Path) -> None:
        ws1 = LocalAgentWorkspace(str(tmp_path), agent_id="first")
        run1 = json.loads((tmp_path / "meta" / "run.json").read_text())
        ws2 = LocalAgentWorkspace(str(tmp_path), agent_id="second")
        run2 = json.loads((tmp_path / "meta" / "run.json").read_text())
        assert run1["run_id"] == run2["run_id"]
        assert run2["agent_id"] == "first"


class TestCounterPersistence:
    """Tests for artifact counter persistence across instances."""

    def test_counter_survives_restart(self, tmp_path: Path) -> None:
        ws1 = LocalAgentWorkspace(str(tmp_path))
        ws1.save_artifact("raw", "a.json", "{}")
        ws1.save_artifact("raw", "b.json", "{}")
        # New instance picks up where the old one left off
        ws2 = LocalAgentWorkspace(str(tmp_path))
        ref = ws2.save_artifact("raw", "c.json", "{}")
        assert ref.index == 3
