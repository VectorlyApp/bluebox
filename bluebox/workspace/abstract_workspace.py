"""
bluebox/workspace/abstract_workspace.py

Abstract workspace interface for agent file operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from bluebox.data_models.agents.workspace import (
    ArtifactRef,
    ArtifactSource,
    MountedInputRef,
    WorkspaceDelta,
    WorkspaceSnapshot,
)


class AgentWorkspace(ABC):
    @property
    @abstractmethod
    def root_path(self) -> Path:
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def list_artifacts(self, source: ArtifactSource | None = None) -> list[ArtifactRef]:
        pass

    @abstractmethod
    def read_artifact(
        self,
        artifact_id: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def attach_input_file(self, name: str, source_path: str | Path) -> MountedInputRef:
        """
        Attach an external input file into workspace raw/ via hardlink.

        Args:
            name: Logical input name, used for target filename.
            source_path: Source file path on host filesystem.

        Returns:
            MountedInputRef describing the attached file.
        """
        pass

    @abstractmethod
    def list_mounted_inputs(self) -> list[MountedInputRef]:
        """List mounted input files from the input-mount manifest."""
        pass

    @abstractmethod
    def snapshot_paths(self, roots: list[str]) -> WorkspaceSnapshot:
        pass

    @abstractmethod
    def diff_snapshot(
        self,
        before: WorkspaceSnapshot,
        after: WorkspaceSnapshot,
    ) -> WorkspaceDelta:
        pass

    @abstractmethod
    def read_file(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        pass

    @abstractmethod
    def list_files(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def generate_summary(
        self,
        max_artifacts: int = 10,
        max_summary_chars: int = 160,
    ) -> str:
        """
        Return a compact, prompt-ready summary of workspace state.

        This is intended for system prompt injection, so it should stay concise
        and focus on artifact inventory and recent output/context files.
        """
        pass

    @abstractmethod
    def ensure_dirs(self) -> None:
        pass

    @abstractmethod
    def cleanup(self, remove_root: bool = False) -> None:
        pass
