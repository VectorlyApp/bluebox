"""
bluebox/data_models/agents/workspace.py

Pydantic models for the v2 artifact-oriented workspace.

Contains:
- ArtifactRef / ArtifactManifestEntry: immutable artifact identity and provenance
- WorkspaceRunMetadata: minimal run metadata persisted as run.json
- WorkspaceFileState / WorkspaceSnapshot / WorkspaceDelta: snapshot/diff primitives
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

ArtifactSource = Literal["raw", "output", "context"]


class ArtifactRef(BaseModel):
    artifact_id: str
    index: int
    source: ArtifactSource
    relative_path: str
    size_bytes: int
    content_type: str
    summary: str = ""
    created_at: str
    sha256: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ArtifactManifestEntry(BaseModel):
    index: int
    artifact: ArtifactRef
    tool_name: str | None = None
    code_run_id: str | None = None


class WorkspaceRunMetadata(BaseModel):
    run_id: str
    started_at: str
    agent_id: str | None = None
    thread_id: str | None = None
    ended_at: str | None = None


class WorkspaceFileState(BaseModel):
    relative_path: str
    size_bytes: int
    mtime_ns: int


class WorkspaceSnapshot(BaseModel):
    roots: list[str] = Field(default_factory=list)
    files: dict[str, WorkspaceFileState] = Field(default_factory=dict)
    captured_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class WorkspaceDelta(BaseModel):
    created: list[WorkspaceFileState] = Field(default_factory=list)
    modified: list[WorkspaceFileState] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
