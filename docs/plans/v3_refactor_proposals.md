# RoutineDiscoveryAgent v3 — Refactor Proposals

> Living document. Each proposal is a discrete, actionable change for the next version.

---

## 1. Routine Execution Validation by a Separate Evaluator Agent

### Problem

The orchestrator currently grades its own work. After constructing and executing a routine, it calls `analyze_validation` — asking the same LLM that built the routine to judge whether the result is correct. This leads to premature success claims because the LLM has sunk-cost bias after 20+ iterations of building.

### Proposal

Replace `analyze_validation` with a standalone **RoutineEvaluatorSpecialist** that runs as an independent agent with no knowledge of the discovery process. It receives only:

- The original task description
- The routine JSON
- The execution result (data + metadata)
- The test parameters used
- Optionally: the original CDP capture response for comparison

### Scoring Rubric

The evaluator scores across 5 dimensions (0-10 each):

| Dimension | What It Checks |
|-----------|---------------|
| **Task Completion** | Does the returned data actually accomplish the user's task? Not "did it 200" but "does this contain the right data?" |
| **Data Quality** | Is the response complete, meaningful, and not an error/login page wrapped in a 200? |
| **Parameter Coverage** | Are the right values parameterized? Are there hardcoded values that should be params, or unnecessary params? |
| **Routine Robustness** | Would this work in a fresh session? Are dynamic tokens properly resolved? Is operation ordering correct? |
| **Structural Correctness** | Navigate before fetch, dependencies before dependents, consistent session storage keys, correct placeholder types? |

### Output Schema

```python
class RoutineEvaluationResult(BaseModel):
    overall_pass: bool
    overall_score: float  # 0-100
    dimensions: dict[str, DimensionScore]
    recommendations: list[str]
    blocking_issues: list[str]
```

### Integration Point

Replace the current flow:
```
construct_routine → validate_routine → analyze_validation (self-grade) → done
```

With:
```
construct_routine → validate_routine → evaluate_routine (separate agent) → done / retry
```

The evaluator's `blocking_issues` become feedback to the orchestrator for the retry loop. This is strictly better than the orchestrator's own analysis.

### Calibration Note

Start lenient and tighten based on real failure data. An overly strict evaluator causes infinite retry loops on correct routines.

---

## 2. Rethink the Routine Data Model: Conditional Logic, Branching, Looping

### Problem

The current `Routine` model is a flat, strictly sequential list of operations. This cannot express:

- **Conditional logic:** "If the auth token is expired, refresh it first." Or "If the API returns a redirect, follow it."
- **Branching:** "Call endpoint A. If field X is present, call endpoint B; otherwise call endpoint C."
- **Looping / Pagination:** "Keep calling `/search?page=N` until `has_next_page` is false, accumulating results."
- **Error recovery:** "If fetch returns 401, re-navigate to refresh cookies, then retry."

Real-world APIs frequently require these patterns. The current model forces everything into a straight line, which means the agent either produces broken routines for complex flows or silently ignores the complexity.

### What to Explore

- What primitives are needed? Likely: `if/else` (conditional on a prior operation's result), `loop` (with a condition or count), `try/catch` (error handling with fallback operations).
- How does this affect the execution engine? `RoutineExecutionContext` currently iterates a flat list. It would need a tree/graph walker.
- How does this affect discovery? The BFS algorithm discovers linear dependency chains. Branching requires the agent to understand *when* different paths are taken, not just *what* they contain.
- How does this affect the LLM's ability to construct routines? More expressive models are harder for LLMs to produce correctly. There's a tradeoff between expressiveness and reliability.
- Should we look at existing workflow DSLs (Temporal, AWS Step Functions, Prefect) for inspiration?

### Impact

High. This changes the core data model, the execution engine, and the construction pipeline. Needs careful design before implementation.

---

## 3. Summarize Context on Phase Transition

### Problem

The orchestrator rebuilds full message history every iteration. By iteration 30, the context is mostly noise — old tool results from PLANNING aren't relevant during CONSTRUCTING. This wastes tokens, increases latency, and can confuse the LLM with stale information.

### Proposal

When transitioning between phases, compress the prior phase's messages into a structured summary. For example, when moving from DISCOVERING → CONSTRUCTING:

**Instead of:** 40 messages of create_task / run_pending_tasks / get_task_result / record_extracted_variable / record_resolved_variable / mark_transaction_processed...

**Replace with one summary message:**
```
Phase DISCOVERING complete. Summary:
- Root transaction: POST /api/search (request_id: abc123)
- Processed 3 transactions in dependency order: [tx3, tx2, abc123]
- Parameters: origin (observed: "NYC"), destination (observed: "BOS")
- Dynamic tokens: auth_token (source: sessionStorage:auth.access_token), csrf (source: transaction tx2, path: data.token)
- Static values: client_version="3.2.1", language="en"
```

### Implementation Options

- **Hard summarization:** Deterministic — build the summary from `RoutineDiscoveryState` fields. No LLM call needed. Fast, reliable, but may miss nuance.
- **LLM summarization:** Ask the LLM to summarize before clearing. More flexible but adds latency and cost.
- **Hybrid:** Deterministic summary of structured data + LLM summary of any unstructured findings/notes.

Hard summarization is probably sufficient since all the important data is already in `RoutineDiscoveryState`.

### Impact

Medium. Improves token efficiency and LLM focus. Straightforward to implement since the state objects already contain everything needed for the summary.

---

## 4. Typed Communication Protocol Between Orchestrator and Specialists

### Problem

Currently, the orchestrator communicates with specialists via:
- **Input:** Free-text `prompt` string on a `Task` object
- **Output:** Untyped `task.result` dict (whatever the specialist returned)

This means:
- The orchestrator has to *interpret* specialist results by reading free-text
- There's no contract enforcement — a specialist can return anything
- Failures are opaque ("Max loops reached" tells the orchestrator nothing)
- The orchestrator can't validate results without understanding the specialist's domain

### Design Principle: Flexible Contracts, Not Rigid Schemas

Hardcoding a single response schema per specialist kills the value of using an LLM — it turns a flexible reasoning agent into a glorified function call. The orchestrator needs to handle both well-understood queries ("find the endpoint") and open-ended exploration ("this auth flow looks weird, figure out what's going on").

The solution is **typed schemas as optional, not mandatory**, with a universal response envelope.

### Layer 1: Universal Response Envelope

Every specialist response wraps in a minimal envelope, regardless of whether a schema was provided:

```python
class SpecialistResponse(BaseModel):
    success: bool
    confidence: float          # 0-1, how confident the specialist is
    summary: str               # one-paragraph plain English summary
    structured_output: Any     # typed payload if schema was provided, None otherwise
    raw_findings: dict         # freeform findings the orchestrator can read
    follow_up_suggestions: list[str]  # "you might also want to check X"
```

This is always present. The orchestrator can always read `summary` and `confidence` without understanding the specialist's domain. That alone solves the "failures are opaque" problem — a `confidence: 0.3` with `summary: "Found no clear match, closest candidate is..."` is infinitely more useful than an untyped dict.

### Layer 2: The Orchestrator Chooses the Contract Per Task

The orchestrator decides on a per-task basis whether to enforce a typed schema or leave it open:

```python
# Structured — I know exactly what I want
create_task(
    agent_type="network_specialist",
    prompt="Find the API endpoint for: search trains",
    output_schema=FindEndpointResult.model_json_schema(),
)
# specialist MUST return data matching this schema in structured_output
# envelope fields (success, confidence, summary) are still always present

# Open-ended — go explore and tell me what you find
create_task(
    agent_type="value_trace_resolver",
    prompt="This token appears in 3 different formats across requests. "
           "Figure out the relationship between them.",
)
# specialist returns findings in raw_findings + summary, no schema constraint
# orchestrator reads the summary and decides what to do next
```

### Layer 3: Escalation from Open-Ended to Structured

The orchestrator can send an exploratory task first, read the summary, then send a follow-up task *to the same specialist* (reusing `agent_id` to preserve context) with a schema now that it knows what to ask for:

```python
# Step 1: Open exploration
task1 = create_task(
    agent_type="value_trace_resolver",
    prompt="Trace where this JWT comes from: eyJhbG...",
)
# Result: summary="Found in both sessionStorage and a prior /auth/token response.
#          The sessionStorage copy is set by JavaScript after the /auth/token call."
#         confidence=0.85

# Step 2: Now I know what I want — ask with schema
task2 = create_task(
    agent_type="value_trace_resolver",
    agent_id=task1.agent_id,  # reuse same specialist (has full context from step 1)
    prompt="Confirm: the primary source is the /auth/token response. Give me the exact path.",
    output_schema=TraceTokenResult.model_json_schema(),
)
# Result: structured_output={found: true, source_type: "transaction", ...}
```

### Pre-Defined Schemas for Common Interactions

For well-understood interaction patterns, define reusable schemas:

```python
# NetworkSpecialist: Find endpoint
class FindEndpointResult(BaseModel):
    candidates: list[CandidateEndpoint]  # ranked by confidence
    search_terms_used: list[str]
    reasoning: str

# ValueTraceResolver: Trace token origin
class TraceTokenResult(BaseModel):
    found: bool
    source_type: str | None  # "transaction", "storage", "window_property"
    source_detail: dict | None
    alternatives: list[dict]  # other possible sources
```

These are the **preferred** schemas for common tasks. But the orchestrator is never forced to use them — it can always fall back to open-ended mode when the situation doesn't fit a known pattern.

### Impact

Medium-high. The envelope (`SpecialistResponse`) is easy to implement and immediately improves reliability. Per-task schemas and escalation patterns can be added incrementally for the most common interaction types.

---

## 5. Structured Error Diagnosis for Validation Failures

### Problem

When routine execution fails, the orchestrator gets raw error text and has to figure out what went wrong. Common failure patterns have known fixes, but the LLM has to rediscover them every time:

| Error Pattern | Root Cause | Fix |
|--------------|-----------|-----|
| CORS / network error on fetch | No navigate before fetch | Add navigate to origin URL |
| 401 / 403 | Token resolution failed | Re-trace auth tokens, check source transaction is included |
| Empty session storage | session_storage_key mismatch | Align keys between fetch and return operations |
| HTML in response (login page) | Auth cookies missing | Add navigate + wait for cookies, or add get_cookies operation |
| Timeout | Page didn't load, or JS took too long | Add sleep, or increase timeout |

### Proposal

Add a deterministic diagnostic layer that runs before handing failure info back to the LLM. Pattern-match on error type and produce structured remediation suggestions:

```python
class ValidationDiagnostic(BaseModel):
    error_category: str  # "cors", "auth_failure", "storage_mismatch", etc.
    probable_cause: str
    suggested_fix: str  # actionable instruction for the orchestrator
    affected_operations: list[int]  # operation indices involved
```

This isn't an agent — it's a function. Fast, deterministic, and doesn't burn tokens.

### Impact

Medium. Reduces wasted retry iterations where the LLM flails trying to interpret raw errors. Easy to implement incrementally — start with the top 5 error patterns and expand.

---

## 6. Replace Specialist Classes with Agent Profiles + Shared Tool Pool

### Problem

The current system has a class hierarchy for specialists:

```
AbstractAgent → AbstractSpecialist → NetworkSpecialist
                                   → ValueTraceResolverSpecialist
                                   → JSSpecialist
                                   → InteractionSpecialist
```

But strip away the class names and look at what actually differs between them:

| Specialist | What's Unique |
|-----------|--------------|
| `NetworkSpecialist` | Has `NetworkDataLoader`, exposes network search tools |
| `ValueTraceResolverSpecialist` | Has `NetworkDataLoader` + `StorageDataLoader` + `WindowPropertyDataLoader`, exposes cross-source search tools |
| `JSSpecialist` | Has `JSDataLoader` + live browser, exposes JS validation/execution tools |
| `InteractionSpecialist` | Has `InteractionsDataLoader`, exposes interaction filter tools |

They're not doing different *reasoning*. They're the same LLM doing the same thing (search data, analyze, report) with different *tool sets* and *data loaders* plugged in. The "specialization" is a configuration problem, not an inheritance problem.

This creates real costs:
- Adding a new "specialist" means a new file, new class, new enum value in `SpecialistAgentType`, new branch in the `_create_specialist` factory
- The orchestrator can't compose agents on the fly (e.g., "search both network AND JS files" requires a new specialist class)
- The specialist types are fixed at compile time — the orchestrator LLM can only choose from a hardcoded set

### Proposal: Agent Profiles

Replace the class hierarchy with **agent profiles** — configuration objects that define what a subagent looks like:

```python
class AgentProfile(BaseModel):
    name: str
    system_prompt: str                    # domain knowledge lives here
    tools: list[str]                      # which tools to enable from shared pool
    required_data: list[str]              # which data loaders it needs
    default_output_schema: dict | None    # optional default response schema
    max_iterations: int = 10
```

Predefined profiles replace the current specialist classes:

```python
PROFILES = {
    "network_analyst": AgentProfile(
        name="network_analyst",
        system_prompt="You analyze HTTP network traffic to find API endpoints...",
        tools=["search_by_terms", "get_entry", "search_by_regex", "get_unique_urls"],
        required_data=["network"],
    ),
    "value_tracer": AgentProfile(
        name="value_tracer",
        system_prompt="You trace where dynamic values originate across data sources...",
        tools=["search_network", "search_storage", "search_window_props", "get_entry"],
        required_data=["network", "storage", "window_properties"],
    ),
    "js_analyst": AgentProfile(
        name="js_analyst",
        system_prompt="You write and validate JavaScript for browser execution...",
        tools=["search_js", "validate_js", "execute_js", "get_dom"],
        required_data=["network", "js"],
    ),
    "interaction_analyst": AgentProfile(
        name="interaction_analyst",
        system_prompt="You analyze recorded UI interactions to discover parameters...",
        tools=["search_interactions", "get_form_inputs", "get_unique_elements"],
        required_data=["interactions"],
    ),
}
```

### The Orchestrator's Interface

```python
# Use a predefined profile (replaces current specialist delegation)
create_task(profile="network_analyst", prompt="Find the endpoint for: search trains")

# Compose on the fly — no new class needed
create_task(
    profile="custom",
    system_prompt="You analyze both network traffic and JS files to find...",
    tools=["search_by_terms", "search_js", "get_entry"],
    data=["network", "js"],
    prompt="Find where the auth token is generated client-side",
)
```

The second form is the key advantage — the orchestrator (or the orchestrator's LLM) can create agents for tasks we didn't anticipate at design time.

### Shared Tool Pool

Tools become a flat, shared registry instead of being owned by specialist classes:

```python
# Data search tools (parameterized by which data loader is injected)
search_by_terms(loader, terms, top_n)        # works for network, JS, etc.
get_entry(loader, entry_id)                   # works for any indexed data
search_values(loaders, value)                 # cross-source value search

# Analysis tools
execute_python(code, context)                 # sandboxed Python with data access
get_response_schema(loader, entry_id)         # JSON schema extraction

# Browser tools (only available if remote_debugging_address provided)
execute_js(code)
get_dom_snapshot()
```

A subagent gets a subset of these based on its profile. The tool implementations don't change — they just get different data loaders injected at runtime.

### What This Preserves

- **Domain knowledge** — lives in `AgentProfile.system_prompt`, not class methods
- **Tool scoping** — the profile defines which tools are available, same as today's specialist tool sets
- **Data isolation** — `required_data` ensures agents only see the data they need
- **The autonomous loop** — `AbstractSpecialist.run_autonomous()` becomes a generic `SubagentRunner.run()` that takes a profile

### What This Eliminates

- The class hierarchy (`AbstractSpecialist` → 4 subclasses, ~4 files)
- The factory method (`_create_specialist` with its 4 if/elif branches)
- The enum (`SpecialistAgentType`)
- The `_agent_instances` dict keyed by subagent ID
- The assumption that agent types are known at compile time

### What This Enables

- **Ad-hoc composition** — combine any tools + any data + any prompt without new code
- **Runtime profile creation** — the orchestrator LLM could theoretically define a custom profile when the predefined ones don't fit
- **Profile iteration** — tuning a specialist's behavior is editing a config object, not modifying a class
- **Easier testing** — test tools independently, test the runner independently, test profiles as configuration

### Relationship to Proposal #4

This pairs naturally with the `SpecialistResponse` envelope. Every subagent (regardless of profile) returns the universal envelope. If the profile specifies a `default_output_schema`, the subagent's finalize tools enforce it in `structured_output`. If not, the orchestrator reads `summary` + `raw_findings`.

### Impact

High. This is a significant structural change but simplifies the codebase overall — fewer classes, fewer files, more flexibility. The migration path is incremental: convert one specialist at a time to a profile, verify behavior is identical, then delete the class.

---

## 7. Persistent Browser Tab for Experimentation-Driven Discovery

### Problem

The current system is purely analytical — it reads captured traffic, reasons about what's needed, builds an entire routine, and only discovers it's wrong during validation at the very end. This is like writing a whole program without running it once.

When building routines manually, the process is empirical: copy a fetch into the console, see if it works, strip unnecessary headers, trace missing tokens, test again. Each experiment informs the next step. The system has no equivalent of this — it reasons in a vacuum until validation.

### Core Idea

Give the orchestrator a **persistent Chrome tab** that lives for the entire discovery session. The orchestrator uses it to run experiments during discovery — not just at the end for validation.

### How Navigation Works

The agent already has all the information it needs for navigation from the captured traffic:

1. **Every captured request has an origin.** The network events include the `documentURL` or page URL the request was made from. If `POST https://api.amtrak.com/search` was captured from `https://www.amtrak.com/tickets`, the agent navigates there.
2. **The base URL is in the data.** The network captures contain all hosts. The agent can see which one serves HTML pages vs API responses.
3. **Experimentation self-corrects.** If the agent navigates to `https://example.com` and a fetch fails with CORS, it tries `https://www.example.com`. The error is immediate feedback, not a mystery to debug after 30 iterations.

### Why a Persistent Tab

A persistent tab gives the agent what a fresh tab can't:

- **Cookies and storage accumulate naturally** — navigate to the site, the page loads, JS runs, auth tokens get set, cookies populate. Just like a real user session.
- **The right CORS origin** — fetches work because the tab is already on the correct domain.
- **JS context** — window properties, service workers, auth interceptors are all loaded and live.
- **No cold-start per experiment** — navigate once, test many fetches from the same context.

The current system creates and destroys tabs for validation. That's wasteful and loses context between attempts.

### Orchestrator Tools for Experimentation

```python
@agent_tool
def test_fetch(
    url: str,
    method: str,
    headers: dict | None = None,
    body: dict | None = None,
    credentials: str = "same-origin",
) -> dict:
    """Execute a fetch in the persistent tab and return the result.
    Use to test hypotheses about what a request needs."""
    # Returns: {status, response_headers, body_preview, success}

@agent_tool
def simplify_fetch(
    working_fetch: dict,
    remove: str,  # "header:X-Request-ID", "body_field:tracking_id", etc.
) -> dict:
    """Re-run a working fetch without a specific component.
    Use to find the minimal request that still works."""
    # Returns: {still_works: bool, status, diff}

@agent_tool
def check_tab_state() -> dict:
    """Read the current browser tab state."""
    # Returns: {url, cookies, sessionStorage_keys, localStorage_keys}

@agent_tool
def navigate_tab(url: str) -> dict:
    """Navigate the persistent tab to a URL and wait for load."""
    # Returns: {final_url, cookies_set, storage_populated}
```

### Experiment Log

Every experiment is recorded as structured data:

```python
class Experiment(BaseModel):
    id: str                     # auto-generated short ID
    fetch_config: dict          # what was sent
    result_status: int          # HTTP status
    result_preview: str         # first 500 chars of response
    success: bool               # did it return the expected data?
    learnings: str              # what the agent concluded
    timestamp: datetime
```

The experiment log becomes the primary source of truth for routine construction — not the captured traffic analysis. Each operation in the final routine maps back to a proven experiment.

### Discovery Workspace

The experiment log tracks individual tests, but the agent also needs a **working state** that tracks what's been proven and what's still needed. Without this, the agent can go in circles — trying the same failing fetch repeatedly or forgetting which headers it already proved unnecessary.

The `DiscoveryWorkspace` is the agent's workbench — structured artifacts built up from experiments, not a plan of what to do:

```python
class TokenDependency(BaseModel):
    """A dynamic value needed by a fetch, with its proven source."""
    name: str                    # human-readable name (e.g., "csrf_token")
    value_observed: str          # the value seen in captures
    source: str                  # proven source: "fetch:blueprint_id:path.to.value"
                                 # or "cookie:name", "sessionStorage:key",
                                 # "windowProperty:path", "page_load" (set by navigation)
    verified: bool               # was the source actually tested?
    experiment_ids: list[str]    # which experiments established this

class FetchBlueprint(BaseModel):
    """A single fetch that has been tested and proven to work."""
    id: str                              # auto-generated short ID
    url: str
    method: str
    required_headers: dict[str, str]     # headers proven necessary by simplification
    removed_headers: list[str]           # headers proven unnecessary (for the record)
    required_body: dict | None           # body fields proven necessary
    removed_body_fields: list[str]       # body fields proven unnecessary
    required_tokens: list[TokenDependency]  # dynamic values this fetch needs
    response_preview: str                # what a successful response looks like
    experiment_ids: list[str]            # experiments that established this blueprint
    minimal: bool                        # has simplification been completed?
    parameterizable_values: dict[str, str]  # values that should become routine parameters
                                            # e.g., {"NYC": "origin_city", "BOS": "destination_city"}

class DiscoveryWorkspace(BaseModel):
    """The orchestrator's working state during experimentation."""
    # The target
    task: str                                    # what the user asked for
    navigation_origin: str | None                # where the tab is / should be

    # Proven artifacts
    target_fetch: FetchBlueprint | None          # the main API call we're building
    dependency_fetches: list[FetchBlueprint]      # supporting fetches (auth, tokens, etc.)

    # Experiment history
    experiment_log: list[Experiment]               # everything tried, in order

    # Current focus
    blocked_on: str | None                         # what the agent is currently stuck on
                                                   # e.g., "need source for csrf_token"
```

#### Why Structure the Artifacts, Not the Reasoning

The workspace structures **what the agent has built** (blueprints, dependencies, experiment results) — not **how the agent thinks** (hypotheses, plans, conclusions). The LLM reasons freely in natural language. But when it proves something works, the result gets recorded in a structured artifact it can reference later.

This prevents circular work without imposing bureaucratic overhead:
- Before running `test_fetch`, the agent can check: "did I already try this exact config?" → it's in the experiment log
- Before tracing a token, the agent can check: "is this already a verified `TokenDependency`?" → skip it
- When assembling the routine, the agent reads `FetchBlueprint` objects directly — each one is a proven, minimal fetch config ready to become a routine operation

#### Reflection Prompts (Lightweight, Not Structured)

Instead of forcing the agent to fill out scientific-method forms, inject a reflection prompt every N experiments:

```
You've run {n} experiments so far. Take stock:
- What fetches are proven and minimal? (check your blueprints)
- What tokens are still unresolved? (check blocked_on)
- What haven't you tried yet?
- Are you going in circles on anything?
```

This is a system message, not a data model. It nudges the LLM to consolidate without the overhead of structured output. The reflection lives in the conversation history where the LLM can reference it naturally.

#### How the Workspace Feeds Routine Construction

When all blueprints are minimal and all token dependencies are verified, routine assembly is mechanical:

```
1. Navigation operation → workspace.navigation_origin
2. For each dependency_fetch (in dependency order):
   → Fetch operation from FetchBlueprint (url, method, required_headers, required_body)
   → session_storage_key to capture response for downstream token resolution
3. Target fetch → from target_fetch blueprint
   → Replace parameterizable_values with {{placeholder}} syntax
   → Replace verified token sources with {{sessionStorage:...}} or {{cookie:...}} syntax
4. Return operation → read from session storage
```

Each routine operation maps 1:1 to a proven `FetchBlueprint`. No guessing.

### The Revised Discovery Flow

```
1. EXPLORE (delegate analysis to subagents)
   ├── Subagent: "Search network traffic for endpoints matching: search trains"
   ├── Subagent: "What auth patterns does this site use?"
   └── Results: candidate endpoint + auth context

2. REPRODUCE (orchestrator with persistent tab)
   ├── navigate_tab() to the site origin (from captured data)
   ├── Wait for page load — cookies, storage, JS populate naturally
   ├── test_fetch() with the exact captured request
   ├── Did it work?
   │   ├── YES → move to SIMPLIFY
   │   └── NO → what failed?
   │       ├── 403 → ask subagent: "trace this token's origin"
   │       ├── CORS → navigate_tab() to correct origin
   │       └── 500 → compare request with capture, fix format
   └── Iterate until the fetch reproduces

3. SIMPLIFY (orchestrator with persistent tab)
   ├── simplify_fetch(remove="header:X-Request-ID") → still works? → not needed
   ├── simplify_fetch(remove="body_field:client_version") → broke? → required
   └── Result: minimal working fetch config

4. RESOLVE DEPENDENCIES (subagents + orchestrator experiments)
   ├── For each required dynamic value:
   │   ├── Subagent: "Where does this value come from?"
   │   ├── test_fetch() on the source endpoint
   │   ├── Simplify that fetch too
   │   └── Chain them together
   └── Result: ordered list of proven, minimal fetches

5. ASSEMBLE (orchestrator)
   ├── Build routine from proven experiments
   ├── Each operation was individually tested — agent knows it works
   ├── Run end-to-end in a FRESH tab to confirm the full chain
   └── Result: routine
```

### What This Replaces

- **The BFS algorithm goes away.** You don't need to algorithmically trace dependency chains when you can try the fetch and see what's missing. A 403 tells you more than static analysis ever could.
- **The rigid phase machine softens.** The agent naturally moves from exploration to reproduction to simplification based on what it learns, not based on a hardcoded state machine.
- **Validation is no longer a separate phase.** Every experiment is a mini-validation. The final end-to-end run is a formality — each piece was already proven.
- **The variable classification step becomes optional.** Instead of classifying variables upfront (PARAMETER / DYNAMIC_TOKEN / STATIC_VALUE), the simplification loop discovers what matters empirically: if removing a value breaks the fetch, it's needed; if it doesn't, it's not.

### Division of Labor

```
Orchestrator (has the browser, runs experiments)
├── navigate_tab, test_fetch, simplify_fetch, check_tab_state
├── Owns the experiment log
├── Assembles the routine from proven experiments
│
└── Delegates ANALYSIS to subagents (no browser access):
    ├── "Search network data for endpoints matching X"
    ├── "Find where this token value appears in captures"
    ├── "Analyze these JS files for auth logic"
    └── Return structured findings via SpecialistResponse envelope
```

Analysis can be parallelized (multiple subagents searching different data sources simultaneously). Experimentation is sequential and owned by the orchestrator — each step depends on the previous result.

### Requirements

- **`remote_debugging_address` becomes strongly recommended**, not optional. Without a browser, the agent falls back to the current static analysis approach. With a browser, it uses the empirical experimentation loop.
- **Tab lifecycle management.** The persistent tab needs to be created at discovery start and cleaned up at the end. If the tab crashes or navigates away unexpectedly, the orchestrator needs to detect and recover.
- **Existing CDP infrastructure is sufficient.** The `bluebox/cdp/connection.py` and `RoutineFetchOperation` already contain the fetch-in-browser logic. The `test_fetch` tool extracts and exposes what's already there.

### Impact

Very high. This is the single biggest upgrade to discovery quality. It changes the agent from an analyst that guesses to an experimentalist that verifies. Every operation in the final routine is backed by a successful experiment, not a hypothesis.

Effort is medium — the CDP connection code exists, the fetch execution code exists in `RoutineFetchOperation`, and the tab management is straightforward. The main work is restructuring the orchestrator's flow to be experiment-driven rather than analysis-driven.

---

## 8. API Index Mode: Map All Endpoints, Not Just One Task

### Problem

The current system is **task-driven** — the user says "find me the train search," and the agent hunts for that one flow. But this misses the bigger opportunity: a single capture session contains dozens of API endpoints, and the user might want more than one of them. Worse, re-running discovery for each task means re-analyzing the same capture data, re-tracing the same auth flows, and re-discovering shared dependencies (cookies, tokens) that are common across endpoints.

### Core Idea

Instead of "find the one routine for task X," run a **bottom-up mapping pass** that discovers *all* interesting endpoints in the session and produces a catalog of routines. The user's task then becomes a query over the catalog, not a discovery process.

### The Shift

| Current (task-driven) | API index (data-driven) |
|---|---|
| User gives a task → agent finds one flow | Agent maps all interesting endpoints in the session |
| One routine out | A catalog of routines out |
| Top-down: "find the train search" | Bottom-up: "here are 14 things this site can do" |
| Re-runs discovery per task | Discovery runs once, catalog is reusable |
| Shared dependencies (auth) are rediscovered each time | Shared dependencies are identified once and linked |

### What's "Interesting"?

A session capture contains hundreds of requests. The agent needs to classify them:

| Category | Examples | Action |
|---|---|---|
| **Core API calls** | Search, submit, fetch data, CRUD operations | Always map — these are the catalog entries |
| **Auth / session setup** | Login, token refresh, OAuth flows | Map as shared dependencies |
| **Navigation** | Page loads, redirects, HTML documents | Note as context (origin URLs for CORS) |
| **Infrastructure noise** | Analytics pings, CDN assets, ads, preflight CORS | Skip entirely |

The current `NetworkSpecialist` already does basic filtering (ignoring static assets). For API index mode, this classification becomes the **primary job** of the first phase.

### Data Model

```python
class EndpointEntry(BaseModel):
    """A single discovered API endpoint with its proven routine."""
    name: str                       # e.g., "one_way_train_search"
    description: str                # "Search for one-way train routes between two stations"
    category: str                   # "search", "auth", "data_retrieval", "mutation"
    routine: Routine                # the actual executable routine
    depends_on: list[str]           # names of other entries (e.g., auth endpoints)
    parameters: list[Parameter]     # what the caller can vary
    sample_response: dict           # what a successful call returns
    source_request_ids: list[str]   # which captured requests this maps to

class RoutineCatalog(BaseModel):
    """The complete API surface discovered from a capture session."""
    site: str                       # e.g., "amtrak.com"
    discovered_at: datetime
    shared_dependencies: list[EndpointEntry]  # auth flows, token refreshes
    endpoints: list[EndpointEntry]             # the actual API catalog
    navigation_origins: list[str]              # URLs needed for CORS context
```

### Discovery Flow for Index Mode

```
1. CLASSIFY
   ├── Analyze all captured transactions
   ├── Group by purpose: API calls, auth, navigation, noise
   ├── Cluster related requests (e.g., search + pagination are one "endpoint")
   └── Output: list of endpoint candidates + shared auth patterns

2. MAP SHARED DEPENDENCIES
   ├── Identify auth/session flows that multiple endpoints need
   ├── Use experimentation loop (Proposal #7) to prove and minimize them
   └── Output: shared dependency blueprints (FetchBlueprints)

3. MAP EACH ENDPOINT (parallelizable)
   ├── For each candidate endpoint:
   │   ├── Reproduce the fetch (test_fetch in persistent tab)
   │   ├── Simplify (strip unnecessary headers/body)
   │   ├── Identify parameters (what would a caller want to vary?)
   │   ├── Link to shared dependencies
   │   └── Build minimal routine
   └── Output: EndpointEntry per candidate

4. ASSEMBLE CATALOG
   ├── Combine all entries into RoutineCatalog
   ├── Cross-reference dependencies
   ├── Generate descriptions from observed request/response patterns
   └── Output: complete RoutineCatalog
```

### Relationship to Task-Driven Discovery

API index mode doesn't replace task-driven mode — it's a different entry point:

- **Task-driven** (current): "Find me the train search" → focused discovery → one routine
- **Index mode** (new): "Map this site" → broad discovery → catalog of routines
- **Hybrid**: "Map this site" → catalog → user asks "which one searches trains?" → filter catalog

The hybrid is the most interesting — discovery happens once, and multiple user tasks are answered from the same catalog without re-running the agent.

### Why This Matters for the Product

- **Reusability**: One capture session → many routines. Users don't re-run discovery per task.
- **Discoverability**: Users see what's *possible*, not just what they asked for. "I didn't know this site had a price alert API."
- **Shared dependencies**: Auth flows are mapped once and linked, not rediscovered per routine.
- **Composability** (ties to Proposal #2): Once you have a catalog of atomic endpoints, branching/looping routines become compositions of catalog entries.

### Impact

High. This is a product-level shift — from a single-routine tool to an API mapping platform. Implementation builds directly on top of Proposal #7 (experimentation loop) — the per-endpoint mapping uses the same `test_fetch` / `simplify_fetch` / `FetchBlueprint` machinery. The main new work is the classification/clustering phase and the catalog data model.

---

## Exploration Priority

| # | Proposal | Impact | Effort | Priority |
|---|----------|--------|--------|----------|
| 7 | Persistent tab + experimentation loop | Very High | Medium | **Do first** — changes the core discovery paradigm |
| 8 | API index mode (endpoint catalog) | High | Medium | **Do second** — builds on #7, multiplies value per session |
| 1 | Evaluator agent | High | Medium | **Do third** — validates the final output |
| 4 | Typed communication protocol (SpecialistResponse envelope) | High | Medium | **Do fourth** |
| 6 | Agent profiles replacing specialist classes | High | Medium | **Do fifth** (builds on #4) |
| 3 | Phase transition summarization | Medium | Low | **Do sixth** |
| 5 | Structured error diagnosis | Medium | Low | **Do seventh** |
| 2 | Routine data model (branching/loops) | High | Very High | **Explore / design only** |
