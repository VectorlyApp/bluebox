"""
bluebox/workspace/local_workspace.py

Local filesystem implementation of AgentWorkspace.
"""

from __future__ import annotations

import errno
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bluebox.data_models.agents.workspace import (
    ArtifactManifestEntry,
    ArtifactRef,
    ArtifactSource,
    MountedInputRef,
    WorkspaceDelta,
    WorkspaceFileState,
    WorkspaceSnapshot,
)
from bluebox.utils.infra_utils import read_file_lines
from bluebox.utils.logger import get_logger
from bluebox.workspace.abstract_workspace import AgentWorkspace

logger = get_logger(name=__name__)


class LocalAgentWorkspace(AgentWorkspace):
    def __init__(
        self,
        workspace_dir: str,
        *,
        agent_id: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        _ = agent_id
        _ = thread_id
        self._workspace_dir = Path(workspace_dir)
        self._scratch_dir = self._workspace_dir / "scratch"
        self._raw_dir = self._workspace_dir / "raw"
        self._output_dir = self._workspace_dir / "output"
        self._context_dir = self._workspace_dir / "context"
        self._meta_dir = self._workspace_dir / "meta"

        self._manifest_path = self._meta_dir / "manifest.jsonl"
        self._input_mounts_manifest_path = self._meta_dir / "input_mounts.jsonl"

        self.ensure_dirs()
        self._artifact_index = self._load_last_index_from_manifest()

    @classmethod
    def from_directory_path(
        cls,
        directory_path: str | Path,
        *,
        agent_id: str | None = None,
        thread_id: str | None = None,
    ) -> LocalAgentWorkspace:
        """
        Construct a workspace from an existing directory path.

        Useful for resume flows where the caller already has a concrete workspace
        directory (e.g., persisted from a previous agent run).
        """
        return cls(
            workspace_dir=str(directory_path),
            agent_id=agent_id,
            thread_id=thread_id,
        )

    @property
    def root_path(self) -> Path:
        return self._workspace_dir

    def ensure_dirs(self) -> None:
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._context_dir.mkdir(parents=True, exist_ok=True)
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path.touch(exist_ok=True)
        self._input_mounts_manifest_path.touch(exist_ok=True)

    def cleanup(self, remove_root: bool = False) -> None:
        if remove_root and self._workspace_dir.exists():
            for p in sorted(self._workspace_dir.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink(missing_ok=True)
                elif p.is_dir():
                    try:
                        p.rmdir()
                    except OSError:
                        pass
            try:
                self._workspace_dir.rmdir()
            except OSError:
                pass

    def _load_last_index_from_manifest(self) -> int:
        if not self._manifest_path.exists():
            return 0

        max_index = 0
        for line in self._manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = ArtifactManifestEntry.model_validate_json(line)
                if entry.index > max_index:
                    max_index = entry.index
            except Exception as e:
                logger.warning("Bad manifest entry skipped when loading index: %s", e)
        return max_index

    def _next_index(self) -> int:
        self._artifact_index += 1
        return self._artifact_index

    def _dir_for_source(self, source: ArtifactSource) -> Path:
        if source == "raw":
            return self._raw_dir
        if source == "output":
            return self._output_dir
        return self._context_dir

    def _coerce_bytes(self, content: str | bytes) -> tuple[bytes, str]:
        if isinstance(content, bytes):
            return content, "binary"
        return content.encode("utf-8"), "text"

    def _infer_content_type(self, filename: str, fallback: str) -> str:
        ext = Path(filename).suffix.lower()
        mapping = {
            ".json": "json",
            ".txt": "text",
            ".md": "markdown",
            ".csv": "csv",
            ".html": "html",
            ".htm": "html",
        }
        return mapping.get(ext, fallback)

    def _make_summary(self, content: str | bytes, max_chars: int = 300) -> str:
        if isinstance(content, bytes):
            return f"<binary {len(content)} bytes>"
        c = content.strip()
        return c[:max_chars] + ("..." if len(c) > max_chars else "")

    def _sha256(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def _dedupe_filename(self, directory: Path, filename: str, index: int) -> str:
        candidate = directory / filename
        if not candidate.exists():
            return filename
        stem = candidate.stem
        suffix = candidate.suffix
        return f"{stem}-{index}{suffix}"

    def _validate_artifact_filename(self, filename: str) -> str:
        normalized_filename = filename.replace("\\", "/")
        path = Path(normalized_filename)
        if (
            not normalized_filename
            or normalized_filename in {".", ".."}
            or "/" in normalized_filename
            or path.name != normalized_filename
        ):
            raise ValueError(
                f"Invalid filename '{filename}'. Filenames must not include path separators.",
            )
        return normalized_filename

    def _validate_mount_name(self, name: str) -> str:
        normalized_name = name.replace("\\", "/")
        path = Path(normalized_name)
        if (
            not normalized_name
            or normalized_name in {".", ".."}
            or "/" in normalized_name
            or path.name != normalized_name
        ):
            raise ValueError(
                f"Invalid mount name '{name}'. Names must not include path separators.",
            )
        return normalized_name

    def save_artifact(
        self,
        source: ArtifactSource,
        filename: str,
        content: str | bytes,
        *,
        tool_name: str | None = None,
        code_run_id: str | None = None,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        safe_input_filename = self._validate_artifact_filename(filename)
        directory = self._dir_for_source(source)
        index = self._next_index()
        artifact_id = f"a_{index:06d}"

        safe_filename = self._dedupe_filename(directory, safe_input_filename, index)
        path = directory / safe_filename

        raw_bytes, fallback_ct = self._coerce_bytes(content)
        path.write_bytes(raw_bytes)

        rel = str(path.relative_to(self._workspace_dir))
        ct = content_type or self._infer_content_type(safe_filename, fallback_ct)
        created_at = datetime.now(timezone.utc).isoformat()

        ref = ArtifactRef(
            artifact_id=artifact_id,
            index=index,
            source=source,
            relative_path=rel,
            size_bytes=len(raw_bytes),
            content_type=ct,
            summary=self._make_summary(content),
            created_at=created_at,
            sha256=self._sha256(raw_bytes),
            metadata=metadata or {},
        )
        entry = ArtifactManifestEntry(
            index=index,
            artifact=ref,
            tool_name=tool_name,
            code_run_id=code_run_id,
        )
        with self._manifest_path.open("a", encoding="utf-8") as f:
            f.write(entry.model_dump_json())
            f.write("\n")

        logger.info("Saved artifact %s -> %s", artifact_id, path)
        return ref

    def _iter_manifest(self) -> list[ArtifactManifestEntry]:
        if not self._manifest_path.exists():
            return []
        entries: list[ArtifactManifestEntry] = []
        for line in self._manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(ArtifactManifestEntry.model_validate_json(line))
            except Exception as e:
                logger.warning("Bad manifest entry skipped: %s", e)
        return entries

    def _iter_input_mount_manifest(self) -> list[MountedInputRef]:
        if not self._input_mounts_manifest_path.exists():
            return []
        entries: list[MountedInputRef] = []
        for line in self._input_mounts_manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entries.append(MountedInputRef.model_validate_json(line))
            except Exception as e:
                logger.warning("Bad input mount entry skipped: %s", e)
        return entries

    def list_artifacts(self, source: ArtifactSource | None = None) -> list[ArtifactRef]:
        refs = [e.artifact for e in self._iter_manifest()]
        if source is None:
            return refs
        return [r for r in refs if r.source == source]

    def read_artifact(
        self,
        artifact_id: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        for ref in self.list_artifacts():
            if ref.artifact_id == artifact_id:
                return self.read_file(ref.relative_path, start_line=start_line, end_line=end_line)
        return {"error": f"Artifact not found: {artifact_id}"}

    def attach_input_file(self, name: str, source_path: str | Path) -> MountedInputRef:
        self.ensure_dirs()

        safe_name = self._validate_mount_name(name)
        source = Path(source_path).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError(f"Input source is not a file: {source}")

        suffix = source.suffix if source.suffix else ""
        filename = safe_name if Path(safe_name).suffix else f"{safe_name}{suffix}"
        safe_filename = self._validate_artifact_filename(filename)
        target = self._raw_dir / safe_filename

        try:
            os.link(source, target)
        except FileExistsError:
            source_stat = source.stat()
            target_stat = target.stat()
            if source_stat.st_ino != target_stat.st_ino or source_stat.st_dev != target_stat.st_dev:
                raise ValueError(
                    f"Input target already exists with different inode: {target}",
                ) from None
        except OSError as e:
            if e.errno == errno.EXDEV:
                raise ValueError(
                    f"Cannot hardlink across filesystems: source={source} target={target}",
                ) from e
            raise

        st = target.stat()
        ref = MountedInputRef(
            mount_id=f"m_{uuid.uuid4().hex}",
            name=safe_name,
            source_path=str(source),
            relative_path=str(target.relative_to(self._workspace_dir)),
            mode="hardlink",
            size_bytes=st.st_size,
            mtime_ns=st.st_mtime_ns,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata={},
        )
        with self._input_mounts_manifest_path.open("a", encoding="utf-8") as f:
            f.write(ref.model_dump_json())
            f.write("\n")

        logger.info("Attached input file %s -> %s", source, target)
        return ref

    def list_mounted_inputs(self) -> list[MountedInputRef]:
        return self._iter_input_mount_manifest()

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

        return {"tree": "\n".join(tree_lines), "total_files": total_files}

    def generate_summary(
        self,
        max_artifacts: int = 10,
        max_summary_chars: int = 160,
    ) -> str:
        refs = self.list_artifacts()

        raw_count = sum(1 for r in refs if r.source == "raw")
        output_count = sum(1 for r in refs if r.source == "output")
        context_count = sum(1 for r in refs if r.source == "context")

        max_artifacts = max(0, int(max_artifacts))
        max_summary_chars = max(20, int(max_summary_chars))

        lines: list[str] = [
            "## Workspace State",
            f"- Root: {self._workspace_dir}",
            (
                f"- Artifacts: {len(refs)} total "
                f"(raw: {raw_count}, output: {output_count}, context: {context_count})"
            ),
        ]

        if not refs:
            lines.append("- Recent artifacts: none")
            return "\n".join(lines)

        refs_sorted = sorted(refs, key=lambda r: r.index, reverse=True)
        recent = refs_sorted[:max_artifacts]

        lines.append("- Recent artifacts (newest first):")
        for r in recent:
            summary = (r.summary or "").replace("\n", " ").strip()
            if len(summary) > max_summary_chars:
                summary = summary[:max_summary_chars] + "..."
            if not summary:
                summary = "(no summary)"

            lines.append(
                f"  - {r.artifact_id} [{r.source}] {r.relative_path} "
                f"({r.size_bytes} bytes) :: {summary}"
            )

        if len(refs_sorted) > len(recent):
            lines.append(f"- ... and {len(refs_sorted) - len(recent)} more artifact(s)")

        return "\n".join(lines)

    def snapshot_paths(self, roots: list[str]) -> WorkspaceSnapshot:
        out: dict[str, WorkspaceFileState] = {}
        for root in roots:
            base = self._workspace_dir / root
            if not base.exists():
                continue
            for p in base.rglob("*"):
                if not p.is_file():
                    continue
                rel = str(p.relative_to(self._workspace_dir))
                st = p.stat()
                out[rel] = WorkspaceFileState(
                    relative_path=rel,
                    size_bytes=st.st_size,
                    mtime_ns=st.st_mtime_ns,
                )
        return WorkspaceSnapshot(roots=roots, files=out)

    def diff_snapshot(
        self,
        before: WorkspaceSnapshot,
        after: WorkspaceSnapshot,
    ) -> WorkspaceDelta:
        created: list[WorkspaceFileState] = []
        modified: list[WorkspaceFileState] = []
        deleted: list[str] = []

        for rel, state in after.files.items():
            prev = before.files.get(rel)
            if prev is None:
                created.append(state)
            elif prev.size_bytes != state.size_bytes or prev.mtime_ns != state.mtime_ns:
                modified.append(state)

        for rel in before.files:
            if rel not in after.files:
                deleted.append(rel)

        return WorkspaceDelta(created=created, modified=modified, deleted=deleted)
