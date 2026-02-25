# Agent Workspace Guide

This document defines the workspace contract used by `AbstractAgent` and specialist agents.

## Workspace layout

- `raw/` (read-only during `execute_python`)
  - Tool-call artifacts and captured inputs/results.
  - Not for final deliverables.
- `output/` (writable)
  - Generated deliverables and transformed outputs.
- `context/` (writable)
  - Reusable notes and session context artifacts.
- `meta/` (read-only during `execute_python`)
  - System-managed metadata (artifact manifest, internal bookkeeping).
  - Not user-editable.

## Prompt-level workspace guidance

`AbstractAgent` exposes `WORKSPACE_USAGE_SECTION` as a class-level string and injects it into the system prompt for all agents. Subclasses can override this section.

Default section:

```md
## Workspace
- Use `raw/` (read-only) for tool-call artifacts (inputs/results), not deliverables.
- Write generated deliverables to `output/`.
- Store reusable notes/context in `context/`.
- `meta/` (read-only) is system-managed and not editable.
```

## Runtime file safety model

### 1) Workspace API validation

- `save_file(...)` rejects:
  - filenames with path separators
  - invalid subdirectories (`..`, absolute paths, escapes outside workspace root)
- `save_artifact(...)` validates artifact filenames with the same no-separator/no-traversal rules.
- `read_file(...)` resolves and rejects paths outside workspace root.

These checks prevent path traversal writes/reads at the workspace API layer.

### 2) Python sandbox scoping

When agents run `execute_python`, the sandbox is invoked with:

- `work_dir = workspace root`
- `read_only_paths = [workspace/raw, workspace/meta]`

This enforces:

- file I/O scoped to workspace root
- write protection for `raw/` and `meta/`

In blocklist mode, `open()` and preloaded `Path` are scoped to `work_dir` and use `realpath` checks to prevent `..` and symlink escape bypasses.

In docker mode, the workspace is mounted at `/data`, and read-only mounts are only accepted if they are under the mounted workspace.

### 3) Lambda mode caveat

`work_dir` is not supported in Lambda mode. If workspace-scoped file I/O is required, use docker or blocklist mode.

## Artifact model notes

- Artifacts are stored under `raw/`, `output/`, and `context/`.
- Each artifact gets an incrementing ID (`a_000001`, etc.) and an entry in `meta/manifest.jsonl`.
- Tool results can be auto-persisted (depending on tool persistence mode) to `raw/`.

## Network Specialist notes

- `NetworkSpecialist` embeds workspace guidance directly in both conversational and autonomous system prompts.
- It passes `entries` as a preloaded Python global for `execute_python`.
- It disables duplicate workspace prompt injection by overriding `_get_workspace_usage_prompt_section()` to return an empty string.

## Recommended data pattern for large inputs

Prefer ingesting large datasets into workspace files (typically under `raw/`) and reading them through workspace-scoped paths, rather than injecting very large in-memory globals or mounting arbitrary external host paths.

Rationale:

- preserves workspace boundary guarantees
- avoids broad host filesystem exposure
- improves reproducibility (all inputs live inside workspace)

