# Worker Agent Design — Phase 2: Experiment-Driven Routine Construction

## Context

Phase 1 (Exploration) is done. Four specialists run in parallel and produce structured summaries:
- **Network** → endpoints, auth patterns, request IDs (~10K tokens)
- **Storage** → tokens, data blocks, auth lifecycle
- **DOM** → pages, forms, embedded tokens, data blobs, tables, framework
- **UI/Interaction** → clicks, inputs, navigation flow, inferred intent

These summaries tell us **what exists**. Phase 2 must figure out **how to get it** — empirically, not analytically.

## The Core Insight (from v3 Proposal #7)

> "The current system is purely analytical — it reads captured traffic, reasons about what's needed, builds an entire routine, and only discovers it's wrong during validation at the very end. This is like writing a whole program without running it once."

The worker agent should be an **experimentalist**, not an analyst. It has a persistent browser tab and tests hypotheses by actually running fetches, navigating pages, and checking results.

---

## Architecture

```
                    ┌─────────────────────┐
                    │    PRINCIPAL INVESTIGATOR      │
                    │  (no browser, no     │
                    │   domain tools)      │
                    │                      │
                    │  Reads: exploration  │
                    │  summaries, user     │
                    │  task, experiment    │
                    │  log                 │
                    │                      │
                    │  Decides: what to    │
                    │  try next            │
                    │                      │
                    │  Writes: experiment  │
                    │  tasks with schemas  │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  WORKER 1  │  │  WORKER 2  │  │  WORKER N  │
     │            │  │            │  │            │
     │  Browser + │  │  Browser + │  │  Browser + │
     │  Lookup    │  │  Lookup    │  │  Lookup    │
     │  tools     │  │  tools     │  │  tools     │
     └────────────┘  └────────────┘  └────────────┘
```

### Two Actors

**1. Orchestrator** — The strategist. Has NO browser, NO domain tools. It only:
- Reads exploration summaries (in its system prompt)
- Reads the experiment log (what's been tried, what worked)
- Creates experiment tasks with specific hypotheses
- Updates the experiment log with results
- Decides when enough is proven to assemble the routine
- Assembles the final Routine from proven experiments

**2. Worker** — The experimentalist. Has a persistent browser tab AND curated lookup tools. It:
- Looks up captured data (transactions, storage events, DOM structure) for reference
- Navigates pages, runs fetches, evaluates JS, checks state in the live browser
- Reports back: "this worked / this failed / here's what I found"
- Does NOT decide strategy — just executes experiments and reports

Why combined: A browser worker without lookup is blind. It needs to see the captured request to know what headers to send. It needs to check where a token lives in storage to know what to resolve. Splitting these into separate agents means the browser worker constantly fails because it's guessing. One agent sees the reference data AND acts in the browser.

---

## Worker Tool Set (~12 tools)

One unified worker with browser primitives + curated lookup tools. The worker does NOT construct routines — it executes experiments and reports findings.

### CRITICAL: Two Sources of Truth

The worker has access to TWO different worlds and MUST NOT confuse them:

```
┌─────────────────────────────────────┐  ┌─────────────────────────────────────┐
│     RECORDED CAPTURE (old data)     │  │      LIVE BROWSER (current)         │
│                                     │  │                                     │
│  What happened during the user's    │  │  The actual Chrome tab the worker   │
│  original session. STATIC. Tokens   │  │  is experimenting in. DYNAMIC.     │
│  may be expired. URLs may have      │  │  Tokens are fresh. State changes   │
│  changed. This is REFERENCE data    │  │  with each action.                 │
│  — use it to understand what to     │  │                                     │
│  reproduce, NOT as current truth.   │  │  This is reality.                  │
│                                     │  │                                     │
│  Tools: capture_*                   │  │  Tools: browser_*                  │
└─────────────────────────────────────┘  └─────────────────────────────────────┘
```

**Tool naming enforces this.** All lookup tools are prefixed `capture_*` to make it
impossible to confuse "search captured storage" with "read current browser storage."
The system prompt will also hammer this home.

### Browser Tools (4 tools — live browser, current state)

These act in the **real, live browser tab**. Results reflect the current page state.

```
browser_navigate(url) → {final_url, status}
  Navigate the persistent tab to a URL and wait for load.

browser_eval_js(expression) → {result, error}
  Execute arbitrary JavaScript in the CURRENT page context.
  This is the Swiss army knife — covers fetch(), reading DOM,
  checking storage, clicking elements, typing, everything.
  The worker writes JS to do what it needs.

browser_cdp_command(method, params) → {result}
  Send a raw CDP command. Escape hatch for anything JS can't do:
  get cookies (including HttpOnly), intercept network, etc.

browser_get_dom(selector?, max_depth?, include_tags?) → {html, truncated}
  Get a filtered view of the CURRENT page's DOM so the worker can
  "see" what's on screen without blowing up context.

  Filtering options (all optional, applied in order):
  - selector: CSS selector to scope to a subtree (e.g. "form#search", "main")
  - max_depth: max nesting depth from root/selector (default: 5)
  - include_tags: only include these tag types (e.g. ["form", "input", "button", "a", "table"])

  Safety: output is HARD CAPPED at 15,000 chars. If the DOM exceeds this
  after filtering, it's truncated and {truncated: true} is returned.
  The worker should then either narrow its selector, reduce depth,
  or use browser_eval_js to extract specific data programmatically.
```

**Why `browser_eval_js` instead of separate `click`, `type`, `press`, `fetch` tools:**
Fewer tools = better tool selection. The LLM writes JS:
`document.querySelector('#search').click()` or `fetch('/api/data', {...})`.
Dedicated tools for each UI action would bloat the count for no benefit.

### Capture Lookup Tools (6 tools — recorded session, OLD data)

These search the **recorded capture from the user's original session**. This data
is STATIC and potentially STALE. Tokens found here may be expired. Use these to
understand what the original requests looked like, then reproduce them in the
live browser with fresh tokens.

```
capture_search_transactions(query) → matching requests with summaries
  Search the RECORDED network traffic. Returns request summaries with IDs.
  e.g. "standings", "search", "auth" → finds captured HTTP requests.

capture_get_transaction(request_id) → full request headers, body, response
  Get full details of a RECORDED request — the exact URL, method, headers,
  body, and response from the original session. Use this as a REFERENCE
  for what to reproduce in the browser, NOT as current data.

capture_search_storage(query) → matching storage events across all types
  Search RECORDED storage events (cookies, localStorage, sessionStorage).
  Shows what was stored during the original session. Token VALUES here
  are likely expired — use this to learn KEY NAMES and STORAGE LOCATIONS,
  then read current values from the live browser via browser_eval_js.

capture_trace_value(value) → everywhere this value appears in the capture
  Cross-domain search across RECORDED data: where does this literal
  string appear in captured network requests, responses, storage, DOM?
  Useful for tracing token origins in the original session.

capture_get_page_structure(snapshot_index?) → forms, inputs, meta tags, scripts
  Get forms, inputs, and page structure from a RECORDED DOM snapshot.
  Shows what the page looked like during the original session.

capture_get_element(selector) → specific element info from RECORDED DOM
  Get details about a specific element from RECORDED DOM snapshots.
```

**Why these 6**: Each answers one common question about the original session:
- "What did the original request look like?" → `capture_search_transactions` + `capture_get_transaction`
- "Where was this token stored originally?" → `capture_search_storage` + `capture_trace_value`
- "What forms/inputs were on this page?" → `capture_get_page_structure` + `capture_get_element`

### Built-in Tools (from AbstractSpecialist, 2-3 tools)

```
finalize_with_output(output) → end experiment with structured result
finalize_with_failure(reason) → experiment failed
add_note(note) → record a warning or observation
```

### Total: ~13 tools

Well within the sweet spot. The `browser_*` / `capture_*` prefix makes it
unambiguous which world the worker is operating in.

---

## PrincipalInvestigator Tool Set (~8 tools)

The PI is the brain. It has NO browser, NO domain tools. It plans, dispatches,
reviews, follows up, and assembles. Everything it knows about the session comes from
exploration summaries (in its system prompt) and experiment results (from workers).

### What the PI knows (system prompt context)

The PI's system prompt includes:

1. **Exploration summaries** — all 4 domain summaries from Phase 1 (~10K tokens)
2. **Documentation** — routine format, operation types, parameter syntax, placeholder
   resolution rules. The PI MUST know how routines work to construct them
   correctly. This comes from a DocumentationDataLoader or embedded reference docs.
3. **Worker capabilities** — the full list of worker tools and their descriptions,
   so the PI knows exactly what workers can and can't do. This prevents
   the PI from writing vague prompts or asking workers to do things they
   have no tools for. Example system prompt section:

```
## Worker Capabilities

Workers have access to the following tools. When writing experiment prompts,
reference these tools by name so the worker knows exactly what to use.

BROWSER TOOLS (act in the live browser):
  browser_navigate(url) — go to a URL
  browser_eval_js(expression) — run JavaScript in the page
  browser_cdp_command(method, params) — raw CDP command
  browser_get_dom(selector?, max_depth?, include_tags?) — see page structure

CAPTURE LOOKUP TOOLS (search RECORDED session data — old, potentially stale):
  capture_search_transactions(query) — find requests in the recorded capture
  capture_get_transaction(request_id) — get full recorded request/response details
  capture_search_storage(query) — find recorded storage events
  capture_trace_value(value) — find where a value appears across the recorded capture
  capture_get_page_structure(snapshot_index?) — get recorded DOM structure
  capture_get_element(selector) — get recorded element details

Workers also receive the exploration summaries as shared context.
```

4. **Experiment log** — the running history of experiments, verdicts, and proven
   artifacts. Grows over time. See "Experiment Log" section below.

### PI Tool Set (~13 tools)

See "PI Working Memory — Discovery Ledger" and "The Loop" sections below
for full tool descriptions.

```
EXPERIMENT  (4)  dispatch_experiment, get_experiment_result, follow_up, record_finding
RECORDING   (1)  record_proven_artifact
CATALOG     (3)  plan_routines, set_active_routine, submit_routine
SHIPPING    (2)  mark_routine_shipped, mark_routine_failed
TERMINATION (2)  mark_complete(usage_guide), mark_failed(reason)
DASHBOARD   (1)  get_ledger
DOCS        (*)  inherited from DocumentationDataLoader
```

The PI works through routines one at a time, shipping each as it passes
inspection. When all are done (shipped or failed), it calls `mark_complete`
with a usage guide explaining the catalog. External hard limits act as safety net.

The PI is lean. It doesn't touch data or browsers — it thinks, delegates,
reviews, and assembles. It knows what workers can do (from its system prompt) and
how routines work (from docs).

---

## PI Working Memory — Discovery Ledger

The pipeline-level tracker for the entire API indexing lifecycle. Tracks:
- What routines the PI plans to build (the catalog plan)
- All experiments across all routines
- All routine attempts, executions, and inspections
- The final deliverable: a list of shipped routines with usage docs

The PI reads a compact view of this in its system prompt every iteration.

### Key concept: the PI builds a CATALOG, not a single routine

The exploration summaries may reveal multiple distinct capabilities:
- "Get league standings" (data fetch)
- "Search for a team" (parameterized search)
- "Get team details" (navigation + data extraction)

The PI's FIRST job is to identify which routines to build from the exploration
data. Then it works through them one at a time (or identifies shared
dependencies like auth that apply to multiple routines).

```python
# ---------------------------------------------------------------------------
# Routine planning — what to build
# ---------------------------------------------------------------------------

class RoutineSpec(BaseModel):
    id: str                           # short ID
    name: str                         # "get_league_standings"
    description: str                  # what this routine does
    status: str                       # "planned" | "experimenting" | "assembling" | "validating" | "shipped" | "failed"
    priority: int                     # 1=must-have, 2=should-have, 3=nice-to-have
    depends_on_specs: list[str]       # other RoutineSpec IDs that share deps (e.g. same auth)
    experiment_ids: list[str]         # experiments run for this routine
    attempt_ids: list[str]           # routine attempts for this routine
    shipped_attempt_id: str | None    # the attempt that passed inspection

# ---------------------------------------------------------------------------
# Experiments (shared across all routines)
# ---------------------------------------------------------------------------

class ExperimentEntry(BaseModel):
    id: str
    routine_spec_id: str | None   # which routine this experiment is for (None = shared/auth)
    hypothesis: str               # "GET /api/v2/tables returns standings"
    rationale: str                # WHY — evidence, reasoning, expectations
    prompt: str                   # what the worker was told to do
    priority: int                 # 1=critical, 2=important, 3=nice-to-have
    task_id: str                  # reference to the dispatched Task
    status: str                   # "running" | "done" | "failed"
    verdict: str | None           # "confirmed" | "refuted" | "partial" | "needs_followup"
    summary: str | None           # what we learned (recorded by PI)
    output: dict | None           # the raw specialist output

class ProvenArtifacts(BaseModel):
    """Shared across all routines — proven building blocks."""
    fetches: list[dict]           # {url, method, headers, body, response_preview}
    navigations: list[dict]       # {url, sets_up: [cookies, storage_keys]}
    tokens: list[dict]            # {name, source, storage_type, key_name}
    parameters: list[dict]        # {name, type, description, example_value}

# ---------------------------------------------------------------------------
# Routine attempts (per routine spec)
# ---------------------------------------------------------------------------

class RoutineAttempt(BaseModel):
    id: str                           # short ID, e.g. "standings_v1", "standings_v2"
    routine_spec_id: str              # which RoutineSpec this attempt is for
    routine_json: dict                # the full routine as dict
    status: str                       # "draft" | "executing" | "inspecting" | "passed" | "failed"

    # Execution
    test_parameters: dict | None
    execution_result: dict | None
    execution_error: str | None

    # Inspection
    inspection_result: dict | None
    overall_pass: bool | None
    blocking_issues: list[str]
    recommendations: list[str]

    # Lineage
    parent_attempt_id: str | None
    changes_from_parent: str | None

# ---------------------------------------------------------------------------
# Final deliverable
# ---------------------------------------------------------------------------

class ShippedRoutine(BaseModel):
    routine_spec_id: str              # which spec this fulfills
    routine_json: dict                # the final routine
    name: str                         # routine name
    description: str                  # what it does
    when_to_use: str                  # guidance for the user
    parameters_summary: list[str]     # human-readable parameter descriptions
    inspection_score: int             # the inspector's final score

class RoutineCatalog(BaseModel):
    """The final output of the entire pipeline."""
    site: str                         # e.g. "premierleague.com"
    user_task: str                    # the original task
    routines: list[ShippedRoutine]    # all shipped routines
    usage_guide: str                  # how to use these routines together
    failed_routines: list[dict]       # routines we couldn't build + why
    total_experiments: int
    total_attempts: int

# ---------------------------------------------------------------------------
# The Ledger — everything the PI needs to see
# ---------------------------------------------------------------------------

class DiscoveryLedger(BaseModel):
    # Context
    user_task: str

    # Planning — what routines to build
    routine_specs: list[RoutineSpec]
    active_spec_id: str | None        # which routine the PI is currently working on

    # Experiments (shared pool)
    experiments: list[ExperimentEntry]
    proven: ProvenArtifacts
    unresolved: list[str]

    # Routine attempts (across all specs)
    attempts: list[RoutineAttempt]

    # Final output
    catalog: RoutineCatalog | None    # built when PI calls mark_complete
```

### What the PI sees in its system prompt

```
ROUTINE CATALOG PLAN:
  1. [shipped]       get_league_standings — "Fetch Premier League standings table"
  2. [experimenting] get_team_details — "Get details for a specific team"
  3. [planned]       search_players — "Search for players by name"

ACTIVE: get_team_details
  Experiments: 3 run (2 confirmed, 1 refuted)
  Proven: 1 fetch, 1 navigation, 1 token (shared with #1)
  Unresolved: ["team detail endpoint requires team slug — where does the slug come from?"]
  Attempts: none yet

SHIPPED ROUTINES:
  get_league_standings (v2, score: 85/100)
    "GET /api/v2/tables — returns 20 teams with position, points, GD"

EXPERIMENT HISTORY:
  [+] exp_1 (shared): "Bearer JWT from localStorage authenticates all /api/ calls" → Confirmed
  [+] exp_2 (#1): "GET /api/v2/tables returns standings" → Confirmed
  [x] exp_3 (#2): "GET /api/v2/teams/{id} returns team details" → Refuted (needs slug not id)
  [+] exp_4 (#2): "GET /api/v2/teams/{slug} returns team details" → Confirmed
  ...
```

### PI tools for catalog management

```
plan_routines(specs: list[{name, description, priority}]) → {spec_ids}
  Called early in the process after analyzing exploration summaries.
  Creates RoutineSpecs on the ledger. The PI decides what routines
  to build based on the exploration data + user task.

  Can be called again to add/update specs as the PI learns more.
  e.g. "I discovered a search endpoint during auth experiments — adding a search routine."

set_active_routine(spec_id) → {ok}
  Switch focus to a different routine. The PI works on one routine
  at a time but can switch when blocked or when dependencies are shared.

submit_routine(spec_id, routine_json, test_parameters) → {attempt_id}
  Submit a routine attempt for a specific spec.
  Step 1: Validates routine JSON against Routine model.
  Step 2: Executes routine end-to-end in a fresh browser tab.
  Step 3: Sends routine + result to RoutineInspector.
  Step 4: Records RoutineAttempt on the ledger.

mark_routine_shipped(spec_id, attempt_id) → {ok}
  Mark a routine as shipped after it passes inspection.
  Moves the spec status to "shipped".

mark_routine_failed(spec_id, reason) → {ok}
  Give up on a specific routine. Records why.

get_ledger() → {full DiscoveryLedger state}

mark_complete(usage_guide) → signals pipeline done
  Called when the PI is done with ALL routines (shipped or failed).
  Provides a usage_guide string explaining how to use the routines
  together and when to use each one.
  Builds the final RoutineCatalog.

mark_failed(reason) → signals pipeline failed
  Called when the PI can't build ANY routines at all.
```

---

## The Loop

```
1. PLAN (PI reads exploration summaries, decides what routines to build)
   PI calls plan_routines([
     {name: "get_league_standings", description: "...", priority: 1},
     {name: "get_team_details", description: "...", priority: 2},
   ])

   Also identifies shared dependencies:
   "All /api/ endpoints need a Bearer JWT from localStorage — experiment with auth first."

2. SHARED EXPERIMENTS (auth, navigation, tokens used across routines)
   PI dispatches experiments for shared dependencies first.
   e.g. "Can we navigate to the site and get a fresh JWT?"
   These experiments have routine_spec_id = None (shared).

3. PER-ROUTINE LOOP (for each routine spec, by priority)
   PI calls set_active_routine(spec_id)

   a. EXPERIMENT — dispatch experiments specific to this routine
      - "Does GET /api/v2/tables return standings data?"
      - "What fields are in the response?"
      - Record findings, build up proven artifacts

   b. ASSEMBLE — PI constructs routine JSON from proven artifacts
      - PI calls submit_routine(spec_id, routine_json, test_parameters)
      - Validates → Executes → Inspects

   c. INSPECT RESULT
      - PI calls get_ledger to see inspection result
      - If passed → mark_routine_shipped(spec_id, attempt_id) → move to next routine
      - If failed → read blocking_issues → fix → resubmit
      - If stuck → mark_routine_failed(spec_id, reason) → move to next routine

   Repeat for each routine in the catalog plan.

4. FINALIZE (all routines addressed)
   PI calls mark_complete(usage_guide) with a guide explaining:
   - What each routine does
   - When to use each one
   - How they relate to each other
   - What parameters each expects

   This builds the final RoutineCatalog — the deliverable of the entire pipeline.

   If the PI can't build ANY routines → mark_failed(reason).
```

### Termination — the lifecycle endpoint

Two levels of completion:

**Per-routine:**
```
mark_routine_shipped(spec_id, attempt_id) → this routine is done and good
mark_routine_failed(spec_id, reason) → gave up on this routine
```

**Pipeline-level:**
```
mark_complete(usage_guide) → ALL routines addressed, here's the catalog
  Builds the RoutineCatalog with:
  - All shipped routines + their inspection scores
  - A usage_guide string explaining how to use them
  - Failed routines + why they failed

  The external loop returns this catalog as the pipeline output.

mark_failed(reason) → can't build anything at all
  Total failure. The external loop returns the failure reason.
```

**External hard limits** (set by the runner script):
- Max total iterations (e.g. 100)
- Max routine attempts per spec (e.g. 5)
- Max time (e.g. 30 minutes)
If any hard limit is hit, the external loop force-terminates and
returns whatever has been shipped so far + warnings for unfinished routines.

---

## RoutineInspector — Independent Quality Gate

The PI built the routine. Now someone else has to judge it. The PI has sunk-cost
bias after 10+ experiments — it WANTS the routine to pass. So we bring in an
independent inspector that has zero knowledge of the discovery process.

### The Metaphor

The PI is the researcher who ran the experiments and wrote the paper. The
RoutineInspector is the **peer reviewer** — reads the paper cold, checks if
the claims hold up, and decides: publish, revise, or reject.

### What It Receives (all as context, no tools needed)

The inspector is a **zero-tool agent**. Everything it needs comes in the prompt:

1. **User task** — what was the user trying to accomplish?
2. **Routine JSON** — the full routine the PI constructed
3. **Test parameters** — what values were used for the test run
4. **Execution result** — the data returned + any errors/warnings/metadata
5. **Exploration summaries** — what the site looks like (so it knows what
   "correct" data should look like — e.g. if the task is "get standings"
   and the network exploration shows 20 teams, the result should have ~20 teams)

### What It Does NOT Receive

- The experiment log (no knowledge of HOW the routine was built)
- The PI's reasoning or rationale
- Any conversation history from the discovery process

This is intentional. The inspector judges the OUTPUT, not the PROCESS.

### Scoring Rubric (5 dimensions, 0-10 each)

| Dimension | What It Checks |
|-----------|---------------|
| **Task Completion** | Does the returned data actually accomplish the user's task? Not "did it 200 OK" but "does this contain the right information?" |
| **Data Quality** | Is the response complete and meaningful? Not an error page wrapped in 200, not truncated, not missing critical fields? |
| **Parameter Coverage** | Are the right values parameterized? Any hardcoded values that should be params? Any unnecessary params? |
| **Routine Robustness** | Would this work in a fresh session? Are dynamic tokens properly resolved via placeholders (not hardcoded)? Is operation ordering correct? |
| **Structural Correctness** | Navigate before fetch? Dependencies before dependents? Consistent session_storage_key usage? Valid placeholder types? |

### Output Schema

```python
class DimensionScore(BaseModel):
    score: int              # 0-10
    reasoning: str          # why this score

class RoutineInspectionResult(BaseModel):
    overall_pass: bool              # ship it or not
    overall_score: int              # 0-100 (sum of dimensions × 2)
    dimensions: dict[str, DimensionScore]  # score per dimension
    blocking_issues: list[str]      # MUST fix before shipping
    recommendations: list[str]      # SHOULD fix (non-blocking)
    summary: str                    # 2-3 sentence overall assessment
```

### How It Fits the Loop

```
PI calls construct_routine → routine JSON
PI dispatches Worker to run routine end-to-end → execution result
PI sends routine + result to RoutineInspector → inspection result

If overall_pass = True:
  → Done. Ship the routine.

If overall_pass = False:
  → PI reads blocking_issues
  → PI designs new experiments to fix those specific issues
  → Back to experiment loop
  → Re-inspect after fixes
```

The inspector's `blocking_issues` become the PI's next TODO list. This is the
feedback loop that prevents the PI from shipping broken routines.

### Calibration

Start LENIENT. An overly strict inspector causes infinite retry loops on
routines that are actually fine. Tighten based on real failure data.

A good starting threshold: overall_pass = True if no blocking_issues AND
overall_score >= 60.

### Implementation

New file: `bluebox/agents/routine_inspector.py`
- Extends `AbstractSpecialist` (uses autonomous mode with output_schema)
- **Zero domain tools** — pure judgment, no data access
- Constructor takes: nothing special (just LLM model)
- Single `run_autonomous()` call with the full context as the task prompt
- Returns `RoutineInspectionResult` via `finalize_with_output`

This is the simplest agent in the system. ~50 lines of actual code.

---

## What We're NOT Building

- **No new data models for experiments** — the existing `Task` + `SpecialistResultWrapper` is the contract. The experiment log is just PI-internal state (a list of dicts, not a complex model hierarchy).
- **No experiment dependency graph** — the PI decides order. It's an LLM, it can reason about "I need auth before I can test the API call."
- **No `simplify_fetch` tool** — premature. The PI can run simplification as a series of experiments ("try without header X"). Build the tool later if the pattern proves useful.
- **No separate worker types** — one worker with both browser + lookup tools. Splitting them forces the browser worker to guess blindly.
- **No dedicated click/type/press tools** — `eval_js` covers all UI actions. The LLM writes JS.
- **No structured "DataRequirement" model** — the PI tracks what it needs in natural language in the experiment log. Adding formal requirement tracking is bureaucracy the LLM doesn't need.

---

## Implementation Order

### Step 1: Worker Agent
New file: `bluebox/agents/workers/experiment_worker.py`
- Extends `AbstractSpecialist`
- ~12 tools: 4 browser primitives + 6 lookup tools + finalize/notes
- Constructor takes:
  - CDP connection context (persistent tab)
  - `NetworkDataLoader` (for transaction lookup)
  - `StorageDataLoader` (for storage search)
  - `DOMDataLoader` (for page structure, optional)
- Single agent does both lookup and browser execution

### Step 2: PrincipalInvestigator Agent
New file: `bluebox/agents/principal_investigator.py`
- Extends `AbstractAgent` (not specialist — it's the coordinator)
- Tools for: creating experiments, dispatching to worker, reading results, assembling routine
- Internal state: ExperimentLog
- System prompt includes exploration summaries + experiment history

### Step 3: RoutineInspector Agent
New file: `bluebox/agents/routine_inspector.py`
- Extends `AbstractSpecialist`
- Zero tools — pure judgment
- Receives routine + execution result + exploration context as task prompt
- Returns RoutineInspectionResult via finalize_with_output
- Simplest agent in the system

### Step 4: Runner Script
New file: `bluebox/scripts/run_routine_construction.py`
- Takes: cdp_captures_dir, user_task, exploration_output_dir
- Sets up: browser tab, data loaders, worker, PI
- Runs the loop
- Outputs: routine.json

### Step 5: Integration
Wire into `bluebox-discover` CLI:
- Phase 1: run explorations (existing scripts)
- Phase 2: run routine construction (new PI)
- Output: complete routine

---

## Context Window Recovery

The PI may exhaust its context window during a long session. When this happens,
the runner spins up a **fresh PI instance** with:
- The same exploration summaries
- The same data loaders
- The **same DiscoveryLedger** (passed via the `ledger` constructor parameter)
- A fresh, empty chat history

The new PI receives a resume prompt:

```
You are RESUMING a previous session that ran out of context.
Your Discovery Ledger has been preserved with all prior work.

FIRST: Call get_ledger to see exactly where things stand — what routines
are planned, what experiments have been run, what's shipped, and what
still needs work.

Then pick up where the previous session left off. Do NOT repeat experiments
that already have verdicts.
```

This works because:
1. The DiscoveryLedger is the single source of truth — not the chat history
2. `get_ledger` returns a full summary of all prior work
3. The ledger summary is also rendered in the system prompt every iteration
4. Worker instances are lost (can't follow_up on old experiments), but new
   workers can be dispatched for fresh experiments

The runner script should catch context-length errors and auto-resume.

---

## Open Questions

1. **When to give up.** Max experiments? Max time? The PI should have a hard cap
   (e.g., 30 experiments) and a soft signal ("3 consecutive failures on the same
   requirement → mark as unresolvable and move on").
