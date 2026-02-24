# Plan: Agent Workspace — Per-Agent Working Directory for Tool Result Persistence

## Context

Every tool result in the agent system currently flows through one path:
`tool executes → dict → json.dumps() → Chat message → LLM context`. Large results are truncated at 5-15K chars and lost. There's no way for agents to store intermediate results, run code over them, or reference past tool outputs by ID.

This PR introduces `AgentWorkspace` — a per-agent directory with three zones:
- **`artifacts/`** — immutable, append-only. Tool results and promoted code outputs. Each entry has an `artifact_id`, provenance (`kind`), and preview. Never modified in place.
- **`scratch/`** — mutable scratchpad. `execute_python` runs here. The agent can write, overwrite, iterate freely. Files are promoted to artifacts via snapshot/diff after each code run.
- **`outputs/`** — final deliverables for user/downstream consumption. Files are copied here when an artifact is marked as a final output. External consumers (pipeline runner, CLI) can `ls outputs/` without parsing the manifest.

## Directory Layout

```
/tmp/bluebox_workspaces/{agent_id}/
├── artifacts/            # immutable evidence — all kinds
│   ├── art_001.json      # kind=tool_result (from capture_get_transaction)
│   ├── art_002.json      # kind=tool_result (from browser_eval_js)
│   ├── art_003.csv       # kind=python_output (promoted from scratch/)
│   ├── art_007.json      # kind=final_output (the routine JSON)
│   └── ...
├── scratch/              # mutable — execute_python work_dir
│   ├── analysis.py       # agent-written script (may be overwritten)
│   ├── filtered.csv      # intermediate output
│   └── ...
├── outputs/              # final deliverables — copies of kind=final_output artifacts
│   ├── routine.json      # copy of art_007.json, user-friendly filename
│   └── ...
├── manifest.jsonl        # ordered append-only log of all artifact registrations
└── (no other top-level files)
```

**Key invariants**:
- Code execution (`execute_python`) NEVER runs in `artifacts/` or `outputs/`. Its `work_dir` points to `scratch/`.
- Code CAN read artifact files (exposed as read-only paths in scratch), but cannot overwrite them.
- `outputs/` contains copies of artifacts with `kind="final_output"`. The source of truth remains `artifacts/` + `manifest.jsonl`.

## Files to Create

### 1. `bluebox/agents/workspace.py` — AgentWorkspace + ArtifactRef

**`ArtifactRef`** (Pydantic BaseModel):
```
- artifact_id: str          # monotonic: "art_001", "art_002", ...
- name: str                 # display label (e.g. "capture_get_transaction")
- tool_name: str            # which tool produced this
- kind: str                 # "tool_result" | "python_output" | "final_output"
- path: Path                # absolute path to file in artifacts/
- size_bytes: int
- content_type: str         # "json" | "text" | "error"
- preview: str              # first ~500 chars for inline display
- created_at: float         # unix timestamp
- code_run_id: str | None   # set when kind="python_output", traces which execute_python call
```

Not a `ResourceBase` subclass — artifact IDs are monotonic counters scoped to the workspace, not global UUIDs. Keeps IDs short and LLM-friendly (`art_001` not `ArtifactRef_550e8400-...`).

**`AgentWorkspace`**:
```
- workspace_dir: Path                          # /tmp/bluebox_workspaces/{agent_id}/
- artifacts_dir: Path                          # workspace_dir/artifacts/
- scratch_dir: Path                            # workspace_dir/scratch/
- outputs_dir: Path                            # workspace_dir/outputs/
- _artifacts: dict[str, ArtifactRef]           # artifact_id → ArtifactRef
- _lock: threading.Lock                        # protects parallel writes
- _counter: int                                # monotonic ID generator
- _manifest_path: Path                         # workspace_dir/manifest.jsonl
```

Methods:
- `__init__(agent_id: str)` — creates workspace_dir, artifacts/, scratch/, outputs/, initializes manifest
- `save_artifact(tool_name: str, data: Any, kind: str = "tool_result") -> ArtifactRef` — serialize data to JSON file **in artifacts/**, register in dict, append to manifest.jsonl. Under lock. Uses `_counter` for monotonic `art_NNN` IDs.
- `get_artifact(artifact_id: str) -> ArtifactRef | None` — dict lookup
- `list_artifacts() -> list[ArtifactRef]` — `list(self._artifacts.values())`
- `read_artifact_content(artifact_id: str) -> dict | str | None` — load file from disk (from artifacts/)
- `mark_as_output(artifact_id: str, filename: str | None = None) -> Path` — copies artifact to `outputs/` with a user-friendly filename (defaults to artifact's name + extension). Updates the artifact's `kind` to `"final_output"`. Returns the output path. Idempotent — calling again with the same artifact_id overwrites the previous copy.
- `cleanup()` — `shutil.rmtree(workspace_dir)` — explicit, NOT called on close by default
- `_next_id() -> str` — returns next monotonic ID under lock, e.g. `"art_003"`

Snapshot/diff methods (for execute_python integration):
- `snapshot_scratch() -> dict[str, _FileState]` — recursive scan of **scratch/** only. Returns `{relative_path: _FileState(size, mtime)}`. Ignores `__pycache__/`, `*.pyc`, `*.tmp`.
- `diff_and_promote(before: dict, after: dict, code_run_id: str) -> list[ArtifactRef]` — compares two scratch snapshots. For created/modified files: **copies** file from scratch/ to artifacts/, registers as artifact with `kind="python_output"`. Returns new artifact refs.
- `run_python_with_tracking(code: str, extra_globals: dict | None) -> dict` — orchestrates: snapshot → execute_python_sandboxed(work_dir=scratch_dir) → snapshot → diff_and_promote → return result with created_artifacts

**`_FileState`** (NamedTuple, internal):
```
- size: int
- mtime: float
```

**manifest.jsonl format** — one JSON line per artifact, appended on each `save_artifact()`:
```json
{"timestamp": 1708790400.0, "tool_name": "capture_get_transaction", "artifact_id": "art_001", "kind": "tool_result", "path": "artifacts/art_001.json", "size_bytes": 47200, "content_type": "json", "code_run_id": null, "preview": "{\"url\": \"/api/..."}
{"timestamp": 1708790401.5, "tool_name": "execute_python", "artifact_id": "art_004", "kind": "python_output", "path": "artifacts/art_004.csv", "size_bytes": 1230, "content_type": "text", "code_run_id": "pyrun_003", "preview": "host,count\napi.example.com,45\n..."}
```

**How scratch/ accesses artifacts for reading:**
Before each `execute_python` run, symlink (or copy) artifact files into `scratch/artifacts_readonly/` so sandboxed code can read them via `open("artifacts_readonly/art_001.json")`. This preserves immutability — the sandbox can read artifacts but its `work_dir` is `scratch/`, so writes go to scratch, not artifacts.

### 2. Tests: `tests/unit/test_workspace.py`

- `test_save_and_retrieve_artifact` — save dict, get by ID, verify content lives in artifacts/
- `test_monotonic_ids` — save 3 artifacts, verify art_001/002/003
- `test_list_artifacts_order` — insertion order preserved
- `test_manifest_written` — verify manifest.jsonl has correct lines
- `test_thread_safety` — parallel saves from ThreadPoolExecutor, verify no lost artifacts and IDs are contiguous
- `test_cleanup` — verify rmtree removes entire workspace
- `test_preview_truncation` — large data gets truncated preview
- `test_save_error_result` — `{"error": "..."}` gets content_type "error"
- `test_snapshot_scratch` — create files in scratch/, verify snapshot captures them
- `test_snapshot_ignores_internal` — `__pycache__`, `.tmp` files excluded from snapshot
- `test_diff_and_promote` — create file in scratch after snapshot, verify it's copied to artifacts/ and registered
- `test_promoted_artifact_is_copy` — after promotion, modifying scratch file doesn't affect artifact
- `test_artifacts_dir_not_in_scratch_snapshot` — artifacts/ dir excluded from scratch snapshot
- `test_mark_as_output` — mark artifact as output, verify file copied to outputs/ with correct filename
- `test_mark_as_output_idempotent` — calling twice overwrites, no duplicates
- `test_mark_as_output_updates_kind` — artifact kind changes to "final_output"

## Files to Modify

### 3. `bluebox/agents/abstract_agent.py` — Wire workspace + code execution into base agent

**Constructor changes** (`__init__`):
- Add `workspace: AgentWorkspace | None = None` parameter (default `None`)
- Store as `self._workspace`
- If not provided, create one: `AgentWorkspace(agent_id=self._thread.id)`
- Every agent gets a workspace — no opt-in flag needed
- Add `supports_code_execution: bool = False` parameter
- Store as `self._supports_code_execution`

**`_auto_execute_tool()` changes** (line ~813):
- After `result = self._execute_tool(...)`, save to workspace:
  ```python
  artifact_ref = self._workspace.save_artifact(tool_name, result)
  ```
- Build chat message with artifact reference + inline preview instead of raw JSON:
  ```
  Tool 'capture_get_transaction' result:
  [artifact: art_001 | 47.2 KB | json]
  Preview: {"url": "/api/v2/standings", "method": "GET", ...}
  Use read_artifact("art_001") to access full result.
  ```
- The `ToolInvocationResultEmittedMessage` still gets the full `result` dict (for the TUI/host — not constrained by context windows)

**New base tools** (3 tools):

`list_artifacts()`:
- Returns list of `{artifact_id, name, tool_name, kind, size_bytes, content_type, preview}` for all artifacts
- Availability: always

`read_artifact(artifact_id: str, max_chars: int = 10000)`:
- Loads artifact content from disk (from artifacts/), truncates to `max_chars`
- Returns `{artifact_id, content_type, content, truncated: bool, total_size_bytes}`
- Availability: always

`execute_python(code: str)` — **defined once on AbstractAgent**, gated by constructor flag:
- Availability: `lambda self: self._supports_code_execution`
- Calls `self._workspace.run_python_with_tracking(code, self._get_python_extra_globals())`
- Sandbox runs in `scratch/`, can read artifacts via `scratch/artifacts_readonly/`
- New/modified files in scratch/ are auto-promoted to artifacts
- Subclasses override `_get_python_extra_globals() -> dict[str, Any]` to inject domain data

**`_get_python_extra_globals()`** — new overridable method on AbstractAgent:
- Default: returns `{}`
- ExperimentWorker overrides → `network_entries`, `storage_entries`, `window_prop_entries`
- NetworkSpecialist overrides → `network_entries`
- ValueTraceResolverSpecialist overrides → `network_entries`, `storage_entries`, `window_prop_entries`
- This is the ONLY thing subclasses need to do — no more copy-pasted `_execute_python` tools

**System prompt injection** — workspace context for LLM:
```
## Workspace
Tool results are saved as immutable artifacts (e.g. "art_001"). The chat shows
a preview of each result. Use read_artifact(artifact_id) for full content.
Use list_artifacts() to see all artifacts.

If execute_python is available: code runs in a scratch directory. Artifact files
from previous tools are readable at artifacts_readonly/. Any files you create
in the scratch directory are automatically promoted to artifacts.
```

### 4. Downstream agents — Delete duplicate `_execute_python`, add `_get_python_extra_globals`

**`bluebox/agents/workers/experiment_worker.py`**:
- DELETE `_execute_python` tool method (line ~976-1035)
- ADD `supports_code_execution=True` to `super().__init__()` call
- ADD `_get_python_extra_globals()` override:
  ```python
  def _get_python_extra_globals(self) -> dict[str, Any]:
      extra = {}
      if self._network_data_loader:
          extra["network_entries"] = [e.model_dump() for e in self._network_data_loader.entries]
      if self._storage_data_loader:
          extra["storage_entries"] = [e.model_dump() for e in self._storage_data_loader.entries]
      if self._window_property_data_loader:
          extra["window_prop_entries"] = [e.model_dump() for e in self._window_property_data_loader.entries]
      return extra
  ```
- `close()` — do NOT call `workspace.cleanup()`

**`bluebox/agents/specialists/network_specialist.py`**:
- DELETE `_execute_python` tool method (line ~348)
- ADD `supports_code_execution=True` to `super().__init__()` call
- ADD `_get_python_extra_globals()` override with `network_entries` only

**`bluebox/agents/specialists/value_trace_resolver_specialist.py`**:
- DELETE `_execute_python` tool method (line ~509)
- ADD `supports_code_execution=True` to `super().__init__()` call
- ADD `_get_python_extra_globals()` override with `network_entries`, `storage_entries`, `window_prop_entries`

### 5. `bluebox/agents/specialists/abstract_specialist.py` — Forward new params

Add `supports_code_execution` to constructor and forward to `super().__init__()`. The `workspace` is auto-created in `AbstractAgent.__init__`, so no explicit handling needed here.

## What's NOT in This PR

- **TTL cleanup job** — workspaces accumulate in `/tmp/bluebox_workspaces/` until explicit `cleanup()` or OS tmpdir cleanup
- **Inline-if-small optimization** — always persist, always show preview. Add size-based inline later
- **Workspace promotion across agents** — worker workspace doesn't auto-flow to PI. Results flow through `SpecialistResultWrapper` → ledger
- **PI output dir integration** — no auto-copy of workspace to pipeline output dir

## Implementation Order

1. Create `bluebox/agents/workspace.py` (AgentWorkspace + ArtifactRef + snapshot/diff/promote + run_python_with_tracking)
2. Create `tests/unit/test_workspace.py` and verify
3. Modify `abstract_agent.py` — workspace init, `supports_code_execution` flag, `_auto_execute_tool` artifact spill, base tools (`list_artifacts`, `read_artifact`, `execute_python`), `_get_python_extra_globals()` hook, prompt section
4. Modify `abstract_specialist.py` — forward `supports_code_execution` param
5. Modify downstream agents — delete duplicate `_execute_python`, add `supports_code_execution=True`, add `_get_python_extra_globals()` overrides:
   - `bluebox/agents/workers/experiment_worker.py`
   - `bluebox/agents/specialists/network_specialist.py`
   - `bluebox/agents/specialists/value_trace_resolver_specialist.py`
6. Run full test suite: `pytest tests/ -v`

## Verification

1. **Unit tests**: `pytest tests/unit/test_workspace.py -v`
2. **Full test suite**: `pytest tests/ -v` — ensure no regressions
3. **Manual smoke test**: Run `bluebox-agent-adapter --agent NetworkSpecialist --cdp-captures-dir ./cdp_captures` and verify:
   - workspace created under `/tmp/bluebox_workspaces/`
   - artifacts/ contains tool result files
   - manifest.jsonl has correct entries
4. **Verify immutability**: After execute_python writes to scratch/, verify artifact files in artifacts/ are untouched copies
5. **Verify thread safety**: Unit test with concurrent saves covers this
