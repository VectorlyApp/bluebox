# API Indexing Pipeline — Overview

The API Indexing Pipeline transforms raw CDP (Chrome DevTools Protocol) browser captures into a catalog of executable, documented routines. It is a two-phase, multi-agent system that discovers, tests, and ships API routines automatically.

## What It Does

1. A human browses a website while the CDP monitor records everything: network requests, DOM snapshots, storage mutations, window property changes, and UI interactions.
2. The pipeline reads those captures and produces a **RoutineCatalog** — a set of JSON routines that can replay specific API workflows (search flights, get standings, fetch prices) with parameterized inputs.

## Two Phases

### Phase 1: Exploration

Four specialist agents run **in parallel**, each analyzing a different domain of the captured data:

| Specialist | Data Source | What It Finds |
|-----------|------------|---------------|
| **NetworkSpecialist** | HTTP request/response traffic | API endpoints, auth patterns, request shapes |
| **DOMSpecialist** | Page DOM snapshots | Forms, embedded tokens, meta tags, framework data blobs |
| **StorageSpecialist** | Cookies, localStorage, sessionStorage | Auth tokens, session data, cached keys |
| **InteractionSpecialist** | User clicks, inputs, navigation (+ DOM snapshots for structural context) | User intent, interaction flow, form submissions |

Each produces a structured **exploration summary** (JSON) saved to `exploration/`. These summaries become the PI's understanding of the site.

### Phase 2: Experiment-Driven Routine Construction

The **PrincipalInvestigator (PI)** — an orchestrator agent with no browser — reads the exploration summaries and drives the construction loop:

```
PI reads summaries
  → plans routine catalog (plan_routines)
  → dispatches experiments to workers (dispatch_experiments_batch)
  → workers test hypotheses in a live browser
  → PI reviews results (record_finding)
  → PI accumulates proven artifacts (record_proven_artifact)
  → PI assembles routines (submit_routine)
  → RoutineInspector scores the routine (6 dimensions)
  → PI ships or iterates
  → PI calls mark_complete → RoutineCatalog
```

## Pipeline Architecture

```
CDP Captures (.jsonl files)
    │
    ├─ network_events.jsonl ──→ NetworkDataLoader ──→ NetworkSpecialist ──→ network.json
    ├─ dom_events.jsonl ───────→ DOMDataLoader ─────→ DOMSpecialist ─────→ dom.json
    ├─ storage_events.jsonl ──→ StorageDataLoader ──→ StorageSpecialist ──→ storage.json
    └─ interactions.jsonl ────→ InteractionsLoader ─→ InteractionSpec. ───→ ui.json
       dom_events.jsonl ────────────────────────────↗ (optional context)
                                                                             │
                                    ┌────────────────────────────────────────┘
                                    ▼
                        PrincipalInvestigator
                        (reads all 4 summaries)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ExperimentWorker ExperimentWorker ExperimentWorker
        (live browser)   (live browser)   (live browser)
              │               │               │
              └───────────────┴───────────────┘
                                    │
                                    │ results
                                    ▼
                        PrincipalInvestigator
                        (reviews results, assembles routine)
                                    │
                                    │ submit_routine
                                    ▼
                            Routine Execution
                            (live browser)
                                    │
                                    ▼
                          RoutineInspector
                          (quality gate)
                                    │
                                    ▼
                        PrincipalInvestigator
                        (ship or iterate)
                                    │
                                    ▼
                          RoutineCatalog
                          (shipped routines)
```

## Output Structure

```
output_dir/
├── exploration/
│   ├── network.json          # NetworkExplorationSummary
│   ├── dom.json              # DOMExplorationSummary
│   ├── storage.json          # StorageExplorationSummary
│   └── ui.json               # UIExplorationSummary
├── experiments/
│   └── exp_*.json            # Individual experiment results
├── attempts/
│   └── attempt_*.json        # Routine attempt records
├── attempt_records/
│   └── routine_attempt_N.json # Unified: routine + params + execution + inspection
├── routines/
│   └── routine_name.json     # Shipped routine files
├── agent_threads/
│   ├── principal_investigator.json
│   └── worker_*.json         # Agent conversation histories
├── ledger.json               # Full DiscoveryLedger state
└── catalog.json              # Final RoutineCatalog
```

## Entry Point

```bash
python -m bluebox.scripts.run_api_indexing \
    --cdp-captures-dir ./cdp_captures \
    --task "Make routines for useful APIs from the Premier League website" \
    --output-dir ./api_indexing_output \
    --llm-model gpt-5.1
```

Key flags:
- `--skip-exploration` — reuse existing `exploration/` summaries (skip Phase 1)
- `--max-pi-iterations N` — cap PI loop iterations per session (default 200)
- `--num-workers N` — parallel experiment workers (default 3)
- `--num-inspectors N` — parallel inspectors (default 1)
- `--max-pi-attempts N` — max PI recovery attempts on context exhaustion or failure (default 3); each attempt spins up a fresh PI with the preserved ledger

## Key Design Principles

1. **Separation of concerns** — exploration agents analyze captures, PI strategizes, workers execute, inspector judges. No agent does two jobs.
2. **Parallel by default** — exploration runs 4 specialists in parallel; PI batches experiments to run N workers simultaneously.
3. **Incremental persistence** — every ledger change, experiment result, and agent thread is written to disk immediately. Pipeline can recover from crashes.
4. **Quality gates** — before execution, routines pass 5 static Python checks in `submit_routine` (name format, description length, parameter descriptions, credential detection, duplicate detection, Pydantic validation). After execution, a `RoutineInspector` scores on 6 dimensions. Both must pass before shipping.
5. **Auth-first ordering** — prompt-only convention (not enforced in code). The PI's system prompt instructs it to solve auth before data endpoints and to include full auth instructions in every experiment prompt, since workers are stateless and don't share browser sessions.

## Related Docs

- [Exploration Phase](./exploration.md)
- [Context & Data Flow](./context_and_data_flow.md)
- **Agents:**
  - [PrincipalInvestigator](./agents/principal_investigator.md)
  - [ExperimentWorker](./agents/experiment_worker.md)
  - [RoutineInspector](./agents/routine_inspector.md)
  - [Exploration Specialists](./agents/exploration_specialists.md)
