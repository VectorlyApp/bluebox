# Potential Improvements

Organized by theme. Cross-references between related improvements are noted inline.

| # | Improvement | Impact | Effort | Theme |
|---|-----------|--------|--------|-------|
| 1 | Pipeline Resumability & Agent Thread Replay | Medium | High | Infrastructure |
| 2 | Tool Usage Observability | Medium | Low | Infrastructure |
| 3 | Anti-Bot Detection as Exploration Output | Medium | Low | Exploration |
| 4 | Deferred Routine Planning | High | Medium | Exploration |
| 5 | Structured Experiment Output Schemas | High | Medium | Knowledge Sharing |
| 6 | Proven Artifacts Redesign | High | High | Knowledge Sharing |
| 7 | Smarter Inspector Context Management | Medium | Low | Inspector |
| 8 | PI Execution Visibility | High | Low | PI Visibility |
| 9 | Visual Page Understanding via Screenshots | Medium | Medium | Worker Capabilities |

---

## Pipeline Infrastructure

### 1. True Pipeline Resumability & Agent Thread Replay

**Impact: Medium** · **Effort: High**

Saves time on crash recovery and enables post-hoc debugging, but doesn't improve routine quality directly. High effort because it requires serialization/deserialization of full agent state, OpenAI response chain replay, and new CLI flags with partial-state detection.

**Gap:** Recovery only works within the same process run. There is no way to resume from a previous run's output directory. Agent threads are written to disk as JSON but never loaded back — they're append-only logs, not resumable states.

**Why it matters:**
- Long pipeline runs (30+ min) that crash have no recovery path
- Useful context is locked inside agent thread files that no one can query
- Post-hoc debugging requires manually reading raw JSON conversation logs

**Proposed fixes:**
- **Pipeline resume:** Detect partial runs (ledger exists, catalog does not) and resume the PI loop from saved state. Add a `--resume` CLI flag.
- **Agent thread reload:** Load saved agent threads back into live agent instances for continued conversation.
- **Interactive agent replay:** CLI or adapter endpoint to ask follow-up questions of past agents with full prior context intact.
- **Inspector re-run:** Re-run `RoutineInspector` on saved attempt records without re-executing the routine.

### 2. Tool Usage Observability

**Impact: Medium** · **Effort: Low**

Doesn't fix anything directly, but provides the data needed to prioritize every other improvement. Low effort — a counter dict in `_execute_tool` and a JSON dump at pipeline end.

**Gap:** No visibility into tool usage patterns across the pipeline. Nothing tracks which tools were available, which were called, how many times, or which were never touched. The only way to understand this today is manually reading raw agent thread JSON.

**Why it matters:**
- Dead tools burn token budget without providing value (e.g. if workers never call `capture_get_page_structure`)
- Behavior drift is invisible — the PI silently stopping `record_proven_artifact` calls (see #6) went undetected
- Prompt tuning is guesswork without knowing which tools agents lean on vs. ignore
- Cross-agent patterns (PI dispatch rate vs. artifact recording rate, browser vs. capture tool usage) reveal whether architecture matches behavior

**Proposed fixes:**
- **Per-agent tracking:** Log every tool invocation in `AbstractAgent._execute_tool` as `{tool_name: call_count}`. Emit summary at agent lifecycle end: tools registered, called, never called, call counts.
- **Pipeline-level report:** Aggregate tool usage across all agents into `{agent_type: {tool_name: {registered, call_count, avg_per_run}}}`. Write to output directory alongside `catalog.json`.
- **Availability vs. usage diff:** Flag tools registered but never called (removal candidates) and tools called every time (auto-invocation candidates).
- **Dashboard data:** Structured JSON with metrics: utilization rate, tool concentration, unused tool token cost estimate.

---

## Exploration & Planning

### 3. Anti-Bot Detection as a First-Class Exploration Output

**Impact: Medium** · **Effort: Low**

Prevents wasted experiments on bot-protected sites — a common failure mode. Low effort: add a field to the network specialist's output schema and a few lines to the PI's system prompt injection.

**Gap:** Anti-bot defenses (PerimeterX, Akamai, CAPTCHA, bot challenges) are only mentioned in the network exploration prompt as "mark low-interest and move on." No specialist produces a structured summary, and the PI receives no explicit warning.

**Why it matters:** Anti-bot defenses are often the primary reason experiments fail. The PI may waste multiple attempts before realizing the issue is bot detection, not auth.

**Proposed fix:** Have the `NetworkSpecialist` produce a dedicated `anti_bot_observations` field in `NetworkExplorationSummary` — a list of observed defenses (name, evidence, likely impact). Inject prominently into the PI's system prompt.

### 4. Deferred Routine Planning with Exploratory Pre-Phase

**Impact: High** · **Effort: Medium**

Directly improves routine coverage and quality by letting the PI plan from live interaction, not just static captures. Unlocks non-API routine types that are currently systematically missed. Medium effort: primarily prompt engineering and workflow changes, not new tooling.

**Gap:** The PI commits to a full routine catalog (`plan_routines`) before dispatching any experiments, based solely on static exploration summaries. Two problems:

1. **Premature commitment:** Summaries are biased toward what was captured. Endpoints not exercised, pages not visited, and capabilities requiring multi-step interaction are invisible.
2. **API-endpoint tunnel vision:** The PI overwhelmingly plans routines that map 1:1 to API endpoints. Routines can do much more — DOM table extraction, multi-page navigation, JS evaluation, hybrid fetch+DOM — but these aren't obvious from HTTP transaction lists.

**Why it matters:**
- Wasted experiments validating a plan that may be wrong
- Site capabilities that don't surface as REST endpoints are systematically missed
- The rigid plan-then-execute structure prevents pivoting on live findings

**Proposed fixes:**
- **Exploratory pre-phase:** Before `plan_routines`, dispatch lightweight "scouting" experiments that navigate the site and report what's actually available.
- **Progressive planning:** Incremental spec declaration instead of one upfront lock-in. The ledger already supports calling `plan_routines` multiple times — the improvement is in prompt strategy and workflow guidance.
- **Broaden routine archetypes:** Update PI system prompt and doc examples to highlight non-API patterns: DOM table extraction, multi-page flows, JS evaluation, hybrid routines. Current examples are heavily fetch-oriented.

---

## Knowledge Sharing & Proven Artifacts

### 5. Structured Experiment Output Schemas

**Impact: High** · **Effort: Medium**

Upstream fix that unblocks #6 (proven artifacts) and improves PI decision-making across the board. Medium effort: define schemas, update PI prompt guidance, wire schema selection into dispatch.

**Gap:** `dispatch_experiment` accepts an optional `output_schema` that flows to the worker's `run_autonomous()`. The plumbing is fully wired. **The PI never uses it.** Every `output_schema` across all observed runs is `null`. Workers return wildly inconsistent structures — two experiments doing the same task return different key names and nesting.

**Why it matters:**
- The PI must parse prose to extract facts — error-prone, context-burning
- No programmatic extraction possible (key names are unpredictable)
- Blocks proven artifact auto-extraction (see #6)
- Cross-experiment comparison is impossible

**Proposed fixes:**
- **3-5 canonical output schemas** for common experiment types (endpoint discovery, auth testing, token tracing, page navigation, DOM inspection).
- **Universal base schema** as minimum: `{confirmed, endpoint_url, method, headers, auth_details, response_preview, key_findings, notes}`.
- **PI prompt guidance** to select schemas when dispatching.
- **Feed structured output into proven artifacts** — once results have predictable fields, auto-extraction becomes mechanical.

### 6. Proven Artifacts Are Broken in Practice — Redesign Knowledge Sharing

**Impact: High** · **Effort: High**

The single biggest systemic issue. Fixing this means workers share knowledge, the PI stops re-explaining auth, and routine assembly draws from a reliable registry. High effort: touches worker output, PI loop, ledger models, and dispatch injection — multiple coordinated changes.

**Gap:** `ProvenArtifacts` is almost completely unused. Real data:

| Run | Shipped | Proven Fetches | Proven Navs | Proven Tokens | Proven Params |
|-----|---------|----------------|-------------|---------------|---------------|
| Nasdaq | 10/10 | 0 | 0 | 0 | 0 |
| Spirit | 6/6 | 0 | 0 | 1 | 0 |

**Root causes:**
1. Recording is optional and PI-initiated — the PI skips it because nothing breaks when it does
2. Workers can't contribute artifacts directly — they output free-text, PI must manually extract and record
3. Artifacts are unstructured `list[dict[str, Any]]` with no schema enforcement
4. Artifacts aren't injected anywhere useful — workers never see them, inspector never sees them
5. No automatic extraction from experiment results

**Why it matters:**
- Workers rediscover auth from scratch on every experiment
- PI burns context re-explaining proven auth in every dispatch prompt
- No accumulated institutional knowledge across the pipeline run

**Proposed fixes:**
- **Structured worker output for artifacts:** Extend `finalize_with_output` schema with optional `discovered_artifacts` field (connects to #5).
- **Auto-extract from confirmed experiments:** Parse worker output for artifact-shaped data and upsert into `ProvenArtifacts`.
- **Inject into worker context at dispatch time:** Workers see proven fetches, tokens, navigations in their system prompt — no PI prompt-engineering required.
- **Enforce recording in the PI loop:** Gate the PI from moving on after a CONFIRMED experiment until artifacts are recorded or explicitly skipped.
- **Schema enforcement:** Replace `list[dict[str, Any]]` with typed Pydantic models (`ProvenFetch`, `ProvenToken`, etc.).

---

## Inspector Quality Gate

### 7. Smarter Inspector Context Management

**Impact: Medium** · **Effort: Low**

Reduces token waste, prevents truncation-induced quality loss, and gives the inspector continuity across attempts. Low effort: conditional summary filtering, targeted data truncation, and a few extra lines in the inspection prompt builder.

**Gap (combines three related issues):**

**Too much irrelevant context:** Every inspector call receives all 4 exploration summaries (network, DOM, storage, UI). The inspector mainly needs the network summary to verify endpoints and auth. DOM/storage/UI add 10-20k chars of token cost but are rarely relevant.

**Dumb truncation:** The 50k-char hard cap is a blind character cut. The prompt is built in order: routine → execution result → exploration summaries. When the execution result's `data` field is huge (e.g. 500 search results pretty-printed with `indent=2`), it can consume most of the budget. Truncation then cuts mid-JSON or sacrifices the exploration summaries entirely.

**No previous attempt context:** Each inspection is fully stateless. Attempt #3 has zero knowledge of what failed in attempts #1 and #2. The inspector can't identify recurring issues, verify fixes, or escalate persistent problems.

**Proposed fixes:**
- **Network-only summaries:** Pass only the network summary by default, or pre-compute a compact "site facts" digest (1-2k chars: endpoints, auth mechanism, anti-bot defenses, key storage items).
- **Smart data truncation:** Before building the prompt, truncate the execution result's `data` field specifically — keep first N array items or first 2000 chars with remaining count. Never blind-cut the entire prompt.
- **Previous attempt summary:** Include compact history in the inspection prompt: "Attempt #1: FAILED (score 35) — blocking: HTTP 401 on /api/token." Gives continuity without dumping full results.

---

## PI Visibility & Decision-Making

### 8. PI Is Blind to Execution Details

**Impact: High** · **Effort: Low**

Directly improves the PI's ability to diagnose and fix failed routines — the core iteration loop. Low effort: the data already exists in `OperationExecutionMetadata`, just needs to be included in the `submit_routine` response dict.

**Gap:** When `submit_routine` returns, the PI sees: `ok`, `error`, `content_type`, `warnings`, and `data_preview` (500-char truncation). The full response payload and per-operation metadata (`OperationExecutionMetadata`: type, duration, details, errors) are written to disk but never returned to the PI.

The PI makes iteration decisions — what to fix, retry, rearchitect — based on a 500-char snippet and the inspector's prose.

**Why it matters:**
- **Guesses at root causes:** Sees "HTTP 401" but can't tell which operation, what headers it sent, or what the error body said
- **Can't judge data quality:** 200 OK with garbage data looks "successful" from a 500-char preview
- **Operation timing invisible:** 45-second operations (timeout risk) or 0ms (cached/stale) go unnoticed
- **Less info than the inspector:** The inspector saw the full execution result; the PI sees a summary of the inspector's summary

**Proposed fixes:**
- **Return `operations_metadata` always.** Compact per-operation breakdown: type, duration, error, key details (resolved URL, status code, response size). Highest-priority fix.
- **`get_attempt_details(attempt_id)` tool.** Let the PI read full persisted attempt records from disk on demand — execution results, inspection scores, operation metadata for any past attempt.
- **PI filesystem access.** Scope a file-read tool to the output directory, turning `attempt_records/` into queryable working memory.
- **Compact `operations_summary` in every response.** `[{"op": 1, "type": "navigate", "ok": true, "duration_s": 1.2}, {"op": 2, "type": "fetch", "ok": false, "error": "401", "duration_s": 0.3}]` — immediate low-effort fix.

---

## Worker Capabilities

### 9. Visual Page Understanding via Screenshots

**Impact: Medium** · **Effort: Medium**

Improves worker accuracy on SPAs, bot-protected sites, and visually complex pages. Medium effort: CDP screenshot call is trivial, but vision message injection into the agent loop and OCR integration require plumbing.

**Gap:** Workers interact with live browser tabs but are completely blind — no use of `Page.captureScreenshot` anywhere. Workers rely on `browser_get_dom` and `browser_eval_js` only. They can't visually confirm navigation results, read rendered text in complex layouts, assess scroll position, or spot visual cues (modals, error banners, CAPTCHA challenges).

**Why it matters:**
- "Successful navigation" might show an error modal or bot challenge invisible in the DOM
- Client-rendered content (SPAs, canvas, dynamically injected text) may not appear in DOM snapshots
- Scroll position is unknown — unnecessary or missed scroll interactions
- Debugging failed experiments lacks visual evidence

**Proposed fixes:**
- **`browser_screenshot` tool:** CDP `Page.captureScreenshot` → vision input to the LLM. On-demand visual inspection.
- **Auto-screenshot after key actions:** Configurable: `screenshot_mode: "manual" | "after_navigation" | "every_action"`.
- **Screenshot-derived context:** OCR for visible text, scroll position estimation, viewport dimensions, overlay/modal detection.
- **Local OCR via EasyOCR:** Extract visible text without burning vision tokens. Outperforms Tesseract on modern UI text. Inject as structured string alongside or instead of raw image:
  ```python
  import easyocr
  import cv2

  def ocr_easyocr(image_path: str) -> str:
      reader = easyocr.Reader(["en"], gpu=False)  # gpu=True if CUDA available
      img = cv2.imread(image_path)
      if img is None:
          raise FileNotFoundError(image_path)
      lines = reader.readtext(img, detail=0, paragraph=True)
      return "\n".join(line.strip() for line in lines if line.strip())
  ```
