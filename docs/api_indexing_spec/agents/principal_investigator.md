# PrincipalInvestigator (PI)

The orchestrator agent for Phase 2. Reads exploration summaries, plans a routine catalog, dispatches experiments to workers, reviews results, assembles routines, and ships a final catalog.

**Has NO browser. Has NO domain tools.** It only strategizes, delegates, and reviews.

## What It Sees

| Context | Source | Updated |
|---------|--------|---------|
| Exploration summaries (all 4 domains) | System prompt | Once (at start) |
| Discovery Ledger summary | System prompt (re-rendered each iteration) | Every iteration |
| Worker capabilities list | System prompt | Once (at start) |
| Routine JSON Schema | System prompt | Once (at start) |
| Agent documentation | Via `search_docs` / `get_doc_file` tools | On demand |

## What It Returns

A `RoutineCatalog` containing:
- All shipped routines (name, description, parameters, operations, inspection score)
- Failed routines (name, reason)
- Usage guide (how routines relate to each other)
- Pipeline statistics (total experiments, attempts)

## Tools

### Planning

| Tool | Purpose |
|------|---------|
| `plan_routines(specs)` | Declare the catalog — list of routines to build with names, descriptions, priorities |
| `set_active_routine(spec_id)` | Switch focus to a different routine |

### Experimentation

| Tool | Purpose |
|------|---------|
| `dispatch_experiments_batch(experiments)` | **Primary.** Run N experiments in parallel on separate workers. Each gets its own browser tab. |
| `dispatch_experiment(...)` | Run a single experiment. Use only for `follow_up` scenarios. |
| `follow_up(experiment_id, message)` | Send follow-up to the SAME worker — preserves browser state and context. Cheaper than a new experiment. |
| `get_experiment_result(experiment_id)` | Read the result of a completed experiment. |

### Recording

| Tool | Purpose |
|------|---------|
| `record_finding(experiment_id, verdict, summary)` | Record what was learned. Verdict: `confirmed`, `refuted`, `partial`, `needs_followup`. |
| `record_proven_artifact(artifact_type, details)` | Add a proven artifact: `fetch`, `navigation`, `token`, or `parameter`. |

### Routine Submission

| Tool | Purpose |
|------|---------|
| `submit_routine(spec_id, routine_json, test_parameters)` | Submit for validation → execution → inspection. Routine runs in a live browser. |
| `mark_routine_shipped(spec_id, attempt_id, when_to_use, parameters_summary)` | Ship a routine that passed inspection. |
| `mark_routine_failed(spec_id, reason)` | Give up on a routine. Requires >= 2 experiments first. |

### Dashboard

| Tool | Purpose |
|------|---------|
| `get_ledger()` | Read the full Discovery Ledger summary. |

### Termination

| Tool | Purpose |
|------|---------|
| `mark_complete(usage_guide)` | Pipeline done. All routines must be shipped or failed. Builds RoutineCatalog. |
| `mark_failed(reason)` | Pipeline failed entirely. Guardrail: requires minimum experiments first. |

### Documentation

| Tool | Purpose |
|------|---------|
| `search_docs(query)` | Search agent documentation by keyword. |
| `get_doc_file(path)` | Read a specific documentation file. |
| `search_docs_by_terms(terms)` | Multi-term relevance search. |

## Key Behaviors

### Auth-First Ordering

**Prompt-only convention — not enforced in code.** The PI's system prompt instructs it to follow this order:

1. **Phase A** — dispatch experiments for auth/token endpoints only
2. Prove auth works (get a valid token, find the subscription key)
3. Record proven auth artifacts
4. **Phase B** — dispatch data endpoint experiments WITH auth instructions included in every prompt

Workers don't share browser state — the PI must pass full auth instructions (token URL, headers, key values) explicitly in each experiment prompt. Nothing in the code prevents the PI from skipping this order; it relies entirely on the LLM following the system prompt instructions.

### Duplicate Routine Detection

Before executing and inspecting a routine, `submit_routine` hashes the operations list and compares against all previous attempts for the same spec. Identical operations are rejected immediately with the previous attempt's blocking issues, saving inspector tokens.

### Quality Gates in submit_routine

Before a routine reaches the inspector, `submit_routine` runs a sequence of **pure Python static checks** (no LLM involved). Each gate returns an error immediately if it fails — the routine never reaches execution or inspection until all pass:

1. **Documentation quality** (`_check_routine_documentation_quality`) — static regex/word-count checks:
   - Name must be `snake_case`, ≥ 3 underscore-separated segments, no generic-only nouns
   - Description must be ≥ 8 words and contain a return/fetch keyword
   - Each parameter description must be ≥ 3 words; opaque params (ending in `_id`, `_slug`, `_code`, etc.) must explain where to get valid values
2. **Credential parameter check** — rejects parameters whose names match a hardcoded set of site-credential patterns (`api_key`, `subscription_key`, `client_secret`, etc.); these should be hardcoded from captures, not exposed as user parameters
3. **Attempt limit** — max 5 `submit_routine` calls per routine spec
4. **Duplicate detection** — hashes the operations list and rejects if identical to any previous failed attempt for the same spec
5. **Pydantic validation** — routine JSON must parse cleanly against the `Routine` model schema

### Loop Detection

The PI's main loop tracks tool call patterns and detects:
- Any tool called 3+ times in a row
- Alternating A-B-A-B patterns

When detected, the PI gets a nudge listing unaddressed routines.

### Resilience

- Never calls `mark_failed` (pipeline-level) after fewer than N experiments
- Never calls `mark_routine_failed` after fewer than 2 experiments for that spec
- Must try multiple auth resolution strategies before giving up
- Uses `follow_up` to preserve worker context instead of dispatching new experiments

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iterations` | 200 | Max PI loop iterations |
| `num_workers` | 3 | Parallel experiment workers |
| `num_inspectors` | 1 | Parallel inspectors |
| `max_attempts_per_routine` | 5 | Max submit_routine calls per spec |
| `min_experiments_before_fail` | 10 | Min experiments before mark_failed allowed |
| `WORKER_TIMEOUT_SECONDS` | 180 | Timeout per worker/inspector dispatch (3 min) |

## File

`bluebox/agents/principal_investigator.py`
