# Context Management & Data Flow

How information flows from raw CDP captures through exploration, experimentation, and routine assembly. Every agent sees different context — this doc explains who sees what, when, and why.

## The Data Pipeline

```
CDP Captures (.jsonl)
    │
    ▼
DataLoaders (parse + index)
    │
    ├──→ Exploration Specialists (Phase 1)
    │         │
    │         ▼
    │    Exploration Summaries (JSON)
    │         │
    │         ▼
    ├──→ PI System Prompt (summaries injected as context)
    │         │
    │         ▼
    │    PI plans routines, dispatches experiments
    │         │
    │         ├──→ ExperimentWorker (gets: task prompt + capture loaders + live browser)
    │         │         │
    │         │         ▼
    │         │    Experiment result (structured output)
    │         │         │
    │         ▼         ▼
    │    PI reviews results, records findings
    │         │
    │         ▼
    │    PI assembles routine JSON
    │         │
    │         ▼
    │    submit_routine → execute → inspect
    │         │
    │         ├──→ Routine Execution (live browser, real HTTP calls)
    │         │         │
    │         │         ▼
    │         │    RoutineExecutionResult (status codes, response data)
    │         │         │
    │         ├──→ RoutineInspector (gets: routine + execution result + exploration summaries)
    │         │         │
    │         │         ▼
    │         │    InspectionResult (6 scores, pass/fail, blocking issues)
    │         │         │
    │         ▼         ▼
    │    PI decides: ship, iterate, or fail
    │         │
    │         ▼
    └──→ RoutineCatalog (final output)
```

## What Each Agent Sees

### PrincipalInvestigator

`_get_system_prompt()` is called **every iteration**, so the system prompt is fully rebuilt on each LLM call with fresh ledger state.

| Context Source | How It's Provided | Contents |
|---------------|-------------------|----------|
| Exploration summaries | Injected in system prompt every iteration | All 4 domain summaries (network, DOM, storage, UI) |
| Discovery Ledger | Rendered via `to_summary()` and injected in system prompt every iteration — only included once there is activity | Routine plan, experiment history, proven artifacts, attempt results |
| Task queue status | Injected in system prompt every iteration — only included when queue is non-empty | Pending/running/done experiment counts |
| Worker capabilities | Hardcoded string in system prompt | List of worker tools (browser + capture) |
| Routine schema | Auto-generated via `Routine.model_schema_markdown()` in system prompt | JSON Schema for valid Routine objects |
| Agent docs | Via `search_docs` / `get_doc_file` tools (on demand, not pre-loaded) | Markdown files from `bluebox/agent_docs/` — operation types, naming conventions, placeholder syntax, auth strategies, common errors, examples |

**What it does NOT see:**
- Raw CDP captures (no direct access — holds data loader references only to pass to workers)
- Live browser (no browser tools)
- Individual network requests (must dispatch workers to look these up)

**Enforcement:** The PI is gated from dispatching experiments until it calls a doc review tool first (`_docs_reviewed` flag). `dispatch_experiment` and `dispatch_experiments_batch` both return an error if docs haven't been reviewed.

### ExperimentWorker

| Context Source | How It's Provided | Contents |
|---------------|-------------------|----------|
| Experiment prompt | Task message from PI | Hypothesis, instructions, expected behavior |
| Capture data | Via capture lookup tools | Network traffic, storage events, DOM snapshots, window properties |
| Live browser | Via browser tools | Navigate, JS eval, CDP commands, DOM inspection |
| Exploration summaries | Included in system prompt | Same 4 summaries the PI sees |

**What it does NOT see:**
- The Discovery Ledger (doesn't know about other experiments or the routine plan)
- Other workers' results (workers are fully independent)
- The routine being assembled (only tests specific hypotheses)

### RoutineInspector

| Context Source | How It's Provided | Contents |
|---------------|-------------------|----------|
| Routine JSON | In task prompt | The complete routine being inspected |
| Execution result | In task prompt | HTTP status codes, response data, unresolved placeholders, operation metadata |
| Exploration summaries | In task prompt | Cross-reference what the site actually has |
| Test parameters | In task prompt | What values were used for testing |

**What it does NOT see:**
- Raw captures (no data loader tools)
- Live browser (zero tools — pure judgment)
- Experiment history (judges the OUTPUT, not the PROCESS)

## The Discovery Ledger

The ledger is the PI's working memory — a single `DiscoveryLedger` object that tracks everything about the pipeline run. It's rendered as a compact summary and injected into the PI's context each iteration.

### Ledger Contents

```
DiscoveryLedger
├── user_task: str              # Original task description
├── routine_specs: list         # Planned routines with statuses
│   └── RoutineSpec
│       ├── name, description, priority (1-3)
│       ├── status: PLANNED → EXPERIMENTING → ASSEMBLING → VALIDATING → SHIPPED/FAILED
│       ├── experiment_ids: list  # Linked experiments
│       └── attempt_ids: list     # Linked routine attempts
│
├── active_spec_id: str         # Currently focused routine
│
├── experiments: list           # All experiments across all routines
│   └── ExperimentEntry
│       ├── hypothesis, rationale, prompt
│       ├── status: PENDING → RUNNING → DONE/FAILED
│       ├── verdict: CONFIRMED / REFUTED / PARTIAL / NEEDS_FOLLOWUP
│       ├── summary: what was learned
│       └── output: raw worker result
│
├── proven: ProvenArtifacts     # Accumulated knowledge
│   ├── fetches: list           # Proven API calls (url, method, headers, response shape)
│   ├── navigations: list       # Proven page navigations (url, what it sets up)
│   ├── tokens: list            # Proven auth tokens (name, source, storage type)
│   └── parameters: list        # Proven user parameters (name, type, description, example)
│
├── unresolved: list            # Open questions
│
├── attempts: list              # All routine attempts across all specs
│   └── RoutineAttempt
│       ├── routine_json: dict  # The routine that was submitted
│       ├── test_parameters     # Values used for testing
│       ├── execution_result    # HTTP status, response data, warnings
│       ├── inspection_result   # 6 dimension scores, blocking issues
│       ├── overall_pass: bool  # Did it pass inspection?
│       └── blocking_issues     # What must be fixed
│
└── catalog: RoutineCatalog     # Final output (built on mark_complete)
```

### Ledger Summary Rendering

Each PI iteration, `to_summary()` renders a compact text view:

```
=== ROUTINE CATALOG PLAN ===
[SHIPPED] get_spirit_stations (1 experiments, 1 attempts) — score: 85
[EXPERIMENTING] search_spirit_flights (3 experiments, 2 attempts)
  → Active: attempt #2 FAILED — blocking: "HTTP 401 on availability endpoint"
[PLANNED] get_spirit_lowfare_calendar (0 experiments)

=== PROVEN ARTIFACTS ===
Fetches: POST /api/prod-token → 200, GET /api/nk/stations → 200
Tokens: subscription_key from network header (static), JWT from token endpoint (dynamic)
Parameters: origin_station_code (string, "BOS"), departure_date (date, "2026-04-06")

=== RECENT EXPERIMENTS ===
[CONFIRMED] exp_abc: "Token endpoint returns JWT with subscription key" — confirmed
[REFUTED] exp_def: "Availability endpoint works without auth" — refuted (401)
```

## Data Loader Architecture

Data loaders parse JSONL capture files and provide structured, searchable access to captured data. They serve two roles:

1. **Exploration** — specialists use them directly via tools to analyze captured data
2. **Experimentation** — workers use them via capture lookup tools to reference the recorded session

### Common Base: `AbstractDataLoader`

All loaders inherit from `AbstractDataLoader` which provides:
- `search_by_terms(terms, top_n)` — relevance-ranked search across entries
- `search_by_regex(pattern, top_n)` — regex search with timeout protection
- `search_content(value)` — substring search with context snippets
- `.entries` — list of parsed entries
- `.stats` — domain-specific statistics

### Loader Types

| Loader | JSONL Source | Entry Type | Key Fields |
|--------|-------------|-----------|------------|
| `NetworkDataLoader` | network_events.jsonl | `NetworkTransactionEvent` | url, method, status, headers, body |
| `DOMDataLoader` | dom_events.jsonl | `DOMSnapshotEvent` | DOM tree, string table, elements |
| `StorageDataLoader` | storage_events.jsonl | `StorageEvent` | type, origin, key, value |
| `WindowPropertyDataLoader` | window_prop_events.jsonl | `WindowPropertyEvent` | path, value, change_type |
| `JSDataLoader` | javascript_events.jsonl | `NetworkTransactionEvent` (JS) | url, content (JS source) |
| `InteractionsDataLoader` | interactions.jsonl | `UIInteractionEvent` | element, type, value |
| `DocumentationDataLoader` | local filesystem | `FileEntry` | path, content, title |

## Context Propagation

### Exploration → PI

```python
# run_api_indexing.py
summaries = run_explorations(cdp_captures_dir, output_dir, llm_model)
# summaries = {"network": "...", "dom": "...", "storage": "...", "ui": "..."}

pi = PrincipalInvestigator(
    exploration_summaries=summaries,  # injected into system prompt
    network_data_loader=network_dl,   # passed through to workers
    storage_data_loader=storage_dl,
    dom_data_loader=dom_dl,
    ...
)
```

### PI → Worker

Workers receive context through TWO channels:

1. **Experiment prompt** (task-specific instructions from PI):
   ```
   "Navigate to spirit.com, find the subscription key in the network captures
   using capture_search_transactions('subscription'), then try calling the
   token endpoint at /api/prod-token/api/v1/token with that key..."
   ```

2. **Data loaders** (capture lookup tools):
   ```python
   worker = ExperimentWorker(
       network_data_loader=self._network_data_loader,
       storage_data_loader=self._storage_data_loader,
       dom_data_loader=self._dom_data_loader,
       remote_debugging_address=self._remote_debugging_address,
   )
   ```

### PI → Inspector

The inspector receives everything in a single task prompt:

```python
# All available exploration summaries are appended (network, dom, storage, ui)
# Prompt is truncated at 50,000 chars if too large
prompt_parts = [
    f"## Routine JSON\n{routine_json}\n",
    f"## Execution Result\n{execution_result_json}\n",
    f"## Exploration Summaries\n",
]
for domain, summary in exploration_summaries.items():
    prompt_parts.append(f"### {domain}\n{summary}\n")
```

> **Note:** All 4 summaries are passed, not just network. See [potential improvements](../api_indexing_spec_v2/potential_improvements.md#4-trim-inspector-context-to-what-it-actually-needs) for why this may be wasteful.

## Incremental Persistence

`PipelinePersistence` writes state to disk after every significant event:

| Event | What's Written | File(s) |
|-------|---------------|---------|
| Ledger change | Full ledger state | `ledger.json` |
| Experiment created/completed | Experiment entry | `experiments/exp_*.json` |
| Routine attempt | Attempt record | `attempts/attempt_*.json` |
| Routine executed + inspected | Unified record | `attempt_records/routine_attempt_N_*.json` |
| Agent conversation | Full message history | `agent_threads/*.json` |
| Routine shipped | Routine JSON | `routines/routine_name.json` |
| Pipeline complete | Final catalog | `catalog.json` |

This enables:
- **Crash recovery** — PI can resume from the last ledger state
- **Debugging** — every experiment, attempt, and inspection is inspectable
- **Skip exploration** — reuse existing summaries with `--skip-exploration`

## Recovery Flow

If the PI crashes or hits a timeout, `run_pi_with_recovery()` retries up to 3 times:

```python
for attempt in range(MAX_PI_ATTEMPTS):
    try:
        pi = PrincipalInvestigator(ledger=existing_ledger, ...)
        catalog = pi.run()
        return catalog
    except Exception:
        existing_ledger = pi._ledger  # preserve state
        # retry with same ledger → PI picks up where it left off
```
