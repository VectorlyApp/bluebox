# Potential Improvements

## 1. True Pipeline Resumability & Agent Thread Replay

**Gap:** The doc claims "pipeline can recover from crashes" but this is only partially true. The current recovery mechanism (`run_pi_with_recovery`) catches PI failures and spins up a fresh PI with the preserved `DiscoveryLedger` — but this only works within the same process run. There is no way to resume a pipeline from a previous run's output directory. Agent threads (PI, workers, inspectors) are written to disk as JSON but are never loaded back — they're append-only logs, not resumable states. You also cannot go back and interrogate a past worker or inspector about what it found.

**Why it matters:**
- Long pipeline runs (30+ min) that crash or are killed have no recovery path — you restart from scratch or `--skip-exploration` at best
- Useful context is locked inside agent thread files that no one can query after the fact
- Post-hoc debugging requires manually reading raw JSON conversation logs

**Proposed improvements:**
- **Pipeline resume:** Given an existing `output_dir`, detect a partial run (ledger exists, catalog does not) and resume the PI loop from the saved ledger state without re-running exploration or cleaning up prior Phase 2 artifacts. Add a `--resume` CLI flag.
- **Agent thread reload:** Load a saved agent thread (PI or worker) back into a live agent instance so you can send new messages to it — effectively "continuing the conversation" with a past agent that has full context of what it did.
- **Interactive agent replay:** A lightweight CLI or adapter endpoint that lets you point at an `agent_threads/worker_*.json` file and ask follow-up questions (e.g. "what headers did you see on that token endpoint?") — the agent resumes with its full prior context intact.
- **Inspector re-run:** Re-run the `RoutineInspector` on any saved `attempt_records/routine_attempt_N.json` without re-executing the routine, useful for tuning inspector prompts or debugging scoring.

## 2. Anti-Bot Detection as a First-Class Exploration Output

**Gap:** Anti-bot/security defenses (PerimeterX, Akamai, Dynatrace, CAPTCHA, bot challenge pages) are currently only mentioned in the network exploration prompt as a "mark low-interest and move on" hint. No specialist produces a structured summary of what defenses were observed, and the PI receives no explicit warning about them.

**Why it matters:** Anti-bot defenses are often the primary reason experiments fail — workers get 403s, CAPTCHA walls, or silent request drops that look like broken endpoints. If the PI doesn't know a site uses PerimeterX, it may waste multiple experiment attempts before realizing the issue is bot detection, not auth. Currently this insight is buried in the `narrative` free-text field at best.

**Proposed fix:** Have the `NetworkSpecialist` produce a dedicated `anti_bot_observations` field in `NetworkExplorationSummary` — a list of observed defenses (name, evidence, likely impact). Inject this prominently into the PI's system prompt so it can warn workers upfront and factor it into experiment strategy (e.g. "this site uses Akamai — navigate first and avoid direct fetch calls").

## 3. Trim Inspector Context to What It Actually Needs

**Gap:** Every `RoutineInspector` call receives ALL exploration summaries — network, DOM, storage, and UI — injected into the task prompt. The stated reason is "cross-reference." In practice, the inspector is judging a routine's correctness and robustness: whether endpoints are real, whether auth is handled, whether the execution result is meaningful. For that, it mainly needs the **network summary** (to verify endpoints and auth patterns). DOM, storage, and UI summaries add significant token cost but are rarely relevant to inspection decisions. The prompt is also hard-capped at 50,000 chars and truncated silently if exceeded — meaning on complex sites the summaries may crowd out the actual routine and execution result.

**Why it matters:** The inspector runs on every `submit_routine` call, potentially many times per pipeline run. Passing 4 full exploration summaries each time is wasteful and may actually hurt quality if truncation cuts off the execution result.

**Proposed fix:** Pass only the network summary to the inspector by default (the most relevant for verifying endpoints and auth). Make DOM/storage/UI summaries opt-in or omit them entirely. Alternatively, pre-summarize exploration context into a compact "site facts" block specifically for inspection use, rather than dumping raw full summaries.

## 4. Give Workers Access to Proven Artifacts

**Gap:** Workers are fully stateless — they receive only a task prompt from the PI and a data availability summary (raw counts). They have no access to exploration summaries, no access to prior experiment results, and no access to the ledger's `ProvenArtifacts`. The PI is expected to copy all relevant proven context (token endpoints, subscription keys, auth headers) into every experiment prompt manually. This is fragile — if the PI's prompt omits a detail, the worker has no way to recover it.

**Why it matters:** The PI writing complete auth instructions into every experiment prompt is verbose, error-prone, and burns PI context. Workers that need to authenticate before testing a data endpoint must rediscover auth from scratch if the PI's prompt is incomplete, even though the ledger already has proven fetches, tokens, and parameters sitting in `ProvenArtifacts`.

**Proposed fix:** Inject the ledger's `ProvenArtifacts` (proven fetches, tokens, navigations, parameters) into the worker's system prompt at dispatch time — not the full ledger, just the proven facts. This gives workers a reliable source of truth for auth and known-good values without requiring the PI to manually copy everything into each prompt. The injection should be compact and structured (not the full ledger summary) so it doesn't bloat worker context.

## 5. Deferred Routine Planning with Exploratory Pre-Phase

**Gap:** The PI currently commits to a full routine catalog (`plan_routines`) before dispatching any experiments. This plan is based solely on the exploration summaries — static analysis of captured network traffic, DOM snapshots, and storage events. The PI has never interacted with the live site at this point. Two problems follow:

1. **Premature commitment:** The exploration summaries are biased toward what was captured during the manual recording session. Endpoints that weren't exercised, pages that weren't visited, or capabilities that require multi-step interaction to discover are invisible. The PI locks in a routine list based on incomplete information and then spends the rest of the pipeline trying to make that list work.

2. **API-endpoint tunnel vision:** Because the network exploration summary is the richest input, the PI overwhelmingly plans routines that map 1:1 to API endpoints (e.g. "get_stations", "search_flights"). But routines can do much more — scrape and structure DOM tables, extract formatted data from rendered pages, chain navigations with JS evaluation, combine multiple fetches into a single higher-level capability. These composite routines are rarely planned because they aren't obvious from a list of HTTP transactions.

**Why it matters:**
- The PI wastes experiments validating a routine plan that may be wrong — discovering mid-pipeline that an endpoint requires an undocumented prerequisite or that a better approach exists via DOM scraping
- Valuable site capabilities that don't surface as clean REST endpoints are systematically missed
- The rigid plan-then-execute structure means the PI can't pivot based on what workers actually find in the live environment

**Proposed fix:**
- **Exploratory pre-phase:** Before `plan_routines`, allow the PI to dispatch a small number of "scouting" experiments — lightweight workers that navigate the site, poke at key pages, and report back what's actually available (not just what was captured). These scouts would focus on questions like: "What data is on this page?", "Does this endpoint require a session cookie from a prior navigation?", "Is there a table here we could scrape?"
- **Progressive planning:** Instead of one upfront `plan_routines` call, support incremental planning. The PI declares an initial set of high-confidence routines, then adds or revises specs as experiments reveal new capabilities or invalidate assumptions. The ledger already supports calling `plan_routines` multiple times — the improvement is in the prompt strategy and workflow guidance, not the tooling.
- **Broaden routine archetypes:** Update the PI's system prompt and documentation examples to explicitly highlight non-API routine patterns: DOM table extraction, multi-page navigation flows, JS evaluation for client-rendered data, hybrid fetch+DOM routines. The current examples and schema docs are heavily fetch-oriented, which anchors the PI's planning toward API-only routines.

## 6. Visual Page Understanding via Screenshots

**Gap:** Experiment workers interact with live browser tabs but are completely blind — they never see what the page actually looks like. Workers rely entirely on `browser_get_dom` (raw DOM tree) and `browser_eval_js` (programmatic queries) to understand page state. There is no use of `Page.captureScreenshot` anywhere in the pipeline. This means workers can't visually confirm what happened after a navigation, can't read rendered text that lives in complex CSS/SVG/canvas layouts, can't assess scroll position, and can't spot visual cues (modals, error banners, loading spinners, CAPTCHA challenges) that are obvious in a screenshot but invisible in the DOM tree.

**Why it matters:**
- Workers frequently misjudge page state — a "successful navigation" might actually show an error modal, a cookie consent overlay, or a bot challenge page that the DOM alone doesn't make obvious
- Client-rendered content (React/Vue SPAs, canvas-based charts, dynamically injected text) may not appear in a DOM snapshot but is clearly visible in a screenshot
- Scroll position matters — workers don't know if the data they need is above or below the fold, leading to unnecessary or missed scroll interactions
- Debugging failed experiments is harder without visual evidence of what the worker was actually looking at

**Proposed fix:**
- **`browser_screenshot` tool:** Add a new worker tool that calls CDP `Page.captureScreenshot`, returns the image to the LLM as a vision input. Workers can call it on demand to visually inspect the current page state — confirming navigations succeeded, reading rendered text, identifying UI elements, spotting error states.
- **Auto-screenshot in the loop:** Optionally capture a screenshot automatically after key actions (navigation, click, form submission) and inject it into the worker's context as a vision message. This gives workers continuous visual feedback without requiring them to explicitly request it. Could be configurable (e.g. `screenshot_mode: "manual" | "after_navigation" | "every_action"`).
- **Screenshot-derived context:** Beyond raw vision, screenshots can be processed to extract useful structured context — OCR for visible text, scroll position estimation, viewport dimensions, detection of overlay/modal states. This could be a lightweight post-processing step that annotates the screenshot with metadata before passing it to the worker.
- **Local OCR via EasyOCR:** For extracting visible text without burning vision tokens, run EasyOCR locally on captured screenshots. EasyOCR outperforms Tesseract on modern UI text (odd fonts, non-white backgrounds, overlays). The extracted text can be injected as a structured string alongside or instead of the raw image, keeping token cost low while giving workers full visibility into rendered page content. Example:
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
  This could be used as a fallback when vision models aren't available, or as a complement — send both the OCR text (cheap, searchable) and the screenshot (rich, visual) to the worker.

## 7. Smarter Inspector Context: Truncate Response Data, Not Summaries

**Gap:** The inspection prompt is built by concatenating: routine JSON → execution result → exploration summaries. The execution result includes the full `data` field — the entire API response payload (e.g. hundreds of flight results, full standings tables) pretty-printed with `indent=2`. When the prompt exceeds the 50,000-char hard cap, it's truncated with a dumb character cut that has no awareness of what it's slicing. Because exploration summaries are appended last, they're the first to get cut. But the real bloat is usually the `data` field inside the execution result — a massive JSON blob that the inspector doesn't need in full to judge correctness.

**Why it matters:**
- A routine returning 500 search results dumps the entire payload into the inspection prompt, potentially 30k+ chars of raw data that the inspector only needs to glance at (e.g. "did it return flight objects with the right fields?")
- When truncation kicks in, it can cut mid-JSON, leaving a broken block that's useless
- Exploration summaries (useful for cross-referencing endpoints and auth) get sacrificed for response data the inspector barely needs

**Proposed fix:** Before building the inspection prompt, truncate or summarize the execution result's `data` field — e.g. keep the first N items of an array, or the first 2000 chars with a count of remaining items. Apply truncation to the data payload specifically, not to the entire prompt blindly. Ensure exploration summaries (or at minimum the network summary) always survive.

## 8. Give the Inspector Previous Attempt Context

**Gap:** Each inspection is fully stateless. If this is attempt #3 on the same routine, the inspector has zero knowledge of what failed in attempts #1 and #2. It can't identify recurring issues ("this is the same auth 401 as the last two attempts"), can't verify that a specific blocking issue was actually addressed, and can't escalate persistent problems. The PI sees the attempt history via the ledger, but the inspector — the actual quality gate — is blind to it.

**Why it matters:**
- The inspector may pass a routine that "looks fixed" but actually has the same underlying issue that caused previous failures, just manifesting differently
- Without history, the inspector can't say "this blocking issue was flagged twice before and still isn't resolved — escalate priority"
- The PI has to manually summarize previous failures in the spec description or hope the inspector catches the same issues independently each time

**Proposed fix:** Include a compact summary of previous attempts for the same routine spec in the inspection prompt — e.g. "Attempt #1: FAILED (score 35) — blocking: HTTP 401 on /api/token. Attempt #2: FAILED (score 52) — blocking: unresolved placeholder {{authToken}}." This gives the inspector continuity without dumping full previous results. The inspector can then verify that specific blocking issues from prior attempts were addressed and flag regressions.

## 9. Trim Exploration Summaries to Network-Only for Inspection

**Gap:** Every inspection call receives all 4 exploration summaries (network, DOM, storage, UI). The inspector's job is to judge whether endpoints are real, auth is handled correctly, the execution result is meaningful, and the routine is structurally sound. For all of these, the **network summary** is the primary reference — it lists the endpoints, auth patterns, and API structure. DOM, storage, and UI summaries add token cost but are rarely relevant to inspection decisions. This is especially wasteful because the inspector runs on every `submit_routine` call, potentially many times per pipeline run.

**Why it matters:**
- 4 summaries can easily consume 10-20k chars of the 50k budget, crowding out the routine and execution result that actually matter
- DOM/storage/UI summaries are most useful for *building* routines (the PI's job), not *judging* them (the inspector's job)
- On complex sites, the summaries alone can trigger truncation before the inspector even sees the execution result

**Proposed fix:** Pass only the network summary to the inspector by default. Alternatively, pre-compute a compact "site facts" block for inspection use — a 1-2k char digest of: known endpoints, auth mechanism, anti-bot defenses, key storage items. This replaces 4 full summaries with a focused reference that fits the inspector's actual needs.

## 10. Proven Artifacts Are Broken in Practice — Redesign Knowledge Sharing

**Gap:** The `ProvenArtifacts` system is the pipeline's intended mechanism for accumulating knowledge across experiments — proven fetches, navigations, tokens, and parameters. In theory, the PI records artifacts as experiments confirm them, and this growing knowledge base informs subsequent experiments and routine assembly. In practice, **it's almost completely unused.** Real pipeline runs show the evidence:

| Run | Shipped Routines | Proven Fetches | Proven Navs | Proven Tokens | Proven Params |
|-----|-----------------|----------------|-------------|---------------|---------------|
| Nasdaq | 10/10 | 0 | 0 | 0 | 0 |
| Spirit | 6/6 | 0 | 0 | 1 | 0 |

The PI ships entire catalogs of working routines without recording a single proven artifact. This means the knowledge sharing mechanism is dead — workers rediscover auth from scratch on every experiment, the PI re-explains token endpoints in every dispatch prompt, and there is no accumulated institutional knowledge across the pipeline run.

**Root causes:**

1. **Recording is entirely optional and PI-initiated.** The `record_proven_artifact` tool exists but the PI rarely calls it. The system prompt mentions it in passing ("accumulate proven artifacts") but doesn't enforce recording after confirmed experiments. The PI is busy dispatching and reviewing — it skips the bookkeeping step because nothing breaks when it does.

2. **Workers can't contribute artifacts directly.** Workers discover the most valuable information (exact headers, working token endpoints, required cookies) but have no way to record it. Their only output is a free-text summary via `finalize_with_output`. The PI must parse this summary, extract the proven facts, and call `record_proven_artifact` manually — a lossy, error-prone handoff.

3. **Artifacts are unstructured dicts.** `ProvenArtifacts` stores `list[dict[str, Any]]` with no schema enforcement. A "proven fetch" could contain anything — there's no guarantee it has the fields needed to actually reproduce the call. This makes artifacts unreliable as a source of truth.

4. **Artifacts aren't injected anywhere useful.** Even when recorded, proven artifacts only appear in the ledger summary (rendered into the PI's system prompt). Workers never see them (improvement #4). The inspector never sees them. The only consumer is the PI itself, which already has the information in its conversation history.

5. **No automatic extraction from experiment results.** When a worker confirms "POST /api/token with header X returns a JWT", nobody automatically extracts that into a proven fetch + proven token. The structured knowledge dies in the experiment's free-text summary.

**Why it matters:**
- Workers waste time and tokens rediscovering auth patterns that earlier workers already solved
- The PI burns context re-explaining proven auth details in every experiment prompt because it can't rely on artifacts being recorded
- Later experiments can't build on earlier findings — each worker starts from zero
- Routine assembly is harder because the PI must reconstruct proven patterns from experiment summaries instead of reading a clean artifact registry

**Proposed fixes:**

- **Structured worker output for artifacts:** Extend the worker's `finalize_with_output` schema to include an optional `discovered_artifacts` field — a structured list of fetches, tokens, navigations, and parameters the worker confirmed. This removes the PI from the extraction loop. When a worker confirms an endpoint works, it records the exact URL, method, headers, and response shape in a typed format, not prose.

- **Auto-extract artifacts from experiment results:** When the PI processes a CONFIRMED experiment, automatically parse the worker's output for artifact-shaped data (URLs, tokens, auth headers) and upsert them into `ProvenArtifacts`. This acts as a safety net — even if the worker or PI forgets to explicitly record, confirmed findings get captured.

- **Inject proven artifacts into worker context (per #4):** Once artifacts are reliably recorded, inject them into every worker's system prompt at dispatch time. A worker testing a data endpoint immediately sees: "Proven: POST /api/token with Ocp-Apim-Subscription-Key: abc123 returns JWT. Use this for auth." No PI prompt-engineering required.

- **Enforce artifact recording in the PI loop:** After processing a CONFIRMED experiment verdict, gate the PI from moving on until it has either recorded artifacts or explicitly noted "no new artifacts from this experiment." Make the bookkeeping step non-optional.

- **Schema enforcement for artifacts:** Replace `list[dict[str, Any]]` with typed Pydantic models — `ProvenFetch(url, method, headers, body_template, response_shape)`, `ProvenToken(name, source_type, source_url, header_name, storage_key)`, etc. This ensures artifacts are complete and usable, not arbitrary dicts that may be missing critical fields.

## 11. Tool Usage Observability — Track What Each Agent Can Use vs. Actually Uses

**Gap:** There is no visibility into tool usage patterns across the pipeline. Each agent (PI, workers, inspector) has a set of registered tools gated by `availability` lambdas, and the LLM decides which to call at runtime. But nothing tracks: which tools were available to an agent, which it actually called, how many times, in what order, or which it never touched. The only way to understand tool usage today is to manually read raw agent thread JSON files and count function_call entries.

**Why it matters:**
- **Dead tools go unnoticed.** If workers never call `capture_get_page_structure` or `capture_trace_value`, those tools are burning token budget in the tool list without providing value. We can't identify these without instrumentation.
- **Behavior drift is invisible.** If the PI stops calling `record_proven_artifact` (as the data shows — improvement #10), there's no alert or metric that flags it. The pipeline silently degrades.
- **Prompt tuning is guesswork.** When optimizing agent prompts, we need to know which tools agents lean on vs. ignore. A worker that only ever uses `browser_eval_js` and `capture_search_transactions` doesn't need 12 tools in its context — trimming the tool list saves tokens and reduces decision complexity for the LLM.
- **Cross-agent patterns are hidden.** Understanding the system as a whole requires knowing: does the PI dispatch more experiments than it records artifacts? Do workers use browser tools more than capture tools? Do inspectors ever use their doc search tools? These aggregate patterns reveal whether the pipeline architecture matches actual behavior.

**Proposed fix:**
- **Per-agent tool usage tracking:** In `AbstractAgent._execute_tool`, log every tool invocation to a lightweight counter: `{tool_name: call_count}`. Store this on the agent instance alongside the registered tool set. At the end of an agent's lifecycle (or on thread dump), emit a summary: tools registered, tools called, tools never called, call counts.
- **Pipeline-level tool usage report:** After a pipeline run completes, aggregate tool usage across all agents into a single report. Structure it as: `{agent_type: {tool_name: {registered: bool, call_count: int, avg_per_run: float}}}`. Write this to the output directory alongside `catalog.json`.
- **Tool availability vs. usage diff:** Flag tools that are registered but never called across multiple runs — candidates for removal or demotion from the default tool set. Conversely, flag tools that are called on nearly every invocation — candidates for deeper integration (e.g. auto-invocation rather than LLM-initiated).
- **Per-run dashboard data:** Emit tool usage as structured JSON so it can feed into a dashboard or analysis notebook. Key metrics: tool utilization rate (called/registered), tool concentration (what % of calls go to the top 3 tools), unused tool token cost estimate (tokens spent listing tools the LLM never picks).

## 12. Experiment Workers Return Freeform Output — PI Never Specifies Output Schemas

**Gap:** The `dispatch_experiment` and `dispatch_experiments_batch` tools both accept an optional `output_schema` parameter that flows all the way through to the worker's `run_autonomous()` call. When provided, the schema is injected into the worker's system prompt and the worker gets `finalize_with_output` (schema-validated) instead of `finalize_result` (freeform dict). The plumbing is fully wired. **The PI never uses it.** Across all observed pipeline runs (Spirit, Nasdaq, Premier League, Mass Corp — 27 batch dispatch calls, ~31 experiments total), every single `output_schema` is `null`.

**What workers actually return:** Since there's no schema, experiment outputs vary wildly even for the same type of task:

| Experiment | Task Type | Output Structure |
|-----------|-----------|-----------------|
| Spirit `exp_5ray7v` | Auth token test | `{summary, reference_capture, live_request}` |
| Nasdaq `exp_2ycchs` | Endpoint discovery | `{targets: {url_templates, headers, live_test_results}}` |
| Nasdaq `exp_3e4nzj` | Endpoint discovery | `{url_template, http_method, observed_query_parameters, request_headers_observed}` |
| Prem `exp_n9sfsn` | Endpoint discovery | `{summary, endpoints: [{url, method, live_result}]}` |
| Prem `exp_4x7iiu` | Failed search | `output: null, success: false` |

Two Nasdaq experiments doing the *same kind of work* (find an endpoint) return completely different key names and nesting structures.

**The irony:** The one agent that *does* get a schema is the `RoutineInspector` — the PI passes `INSPECTION_OUTPUT_SCHEMA` for inspection tasks. So the quality gate has structured output, but the experiments that feed all discovery work are freeform.

**Why it matters:**
- **The PI must parse prose to extract facts.** Every experiment result is a unique snowflake. The PI LLM reads free-text summaries to find URLs, headers, status codes, and token paths — an error-prone interpretation step that burns context and can miss details buried in unusual output structures.
- **No programmatic extraction is possible.** You can't write code that says "get the confirmed endpoint URL from this experiment result" because the key might be `url_template`, `request_url`, `endpoint`, `targets.url`, or any other name the worker chose.
- **Proven artifact auto-extraction is blocked (compounds #10).** Even if we wanted to auto-extract proven artifacts from confirmed experiment results, there's no consistent structure to parse. The freeform output is the upstream bottleneck for the entire knowledge-sharing problem.
- **Cross-experiment comparison is impossible.** When the PI reviews multiple experiment results to assemble a routine, it can't diff structured fields across experiments — it must re-read and re-interpret each one from scratch.

**Proposed fixes:**

- **Define 3-5 canonical output schemas for common experiment types.** Most experiments fall into recognizable categories: endpoint discovery, auth testing, token tracing, page navigation, DOM inspection. Define a typed schema for each (e.g. `EndpointDiscoveryResult(url, method, required_headers, optional_headers, body_template, response_preview, status_code, auth_required)`). The PI selects the appropriate schema when dispatching, or falls back to freeform for truly novel experiments.
- **Default schema with optional structured fields.** Even without per-experiment-type schemas, a universal base schema would help: `{confirmed: bool, endpoint_url: str | null, method: str | null, headers: dict | null, auth_details: dict | null, response_preview: str | null, key_findings: list[str], notes: str}`. This guarantees that the most commonly needed fields are always present and extractable, while `key_findings` and `notes` preserve freeform flexibility.
- **PI prompt guidance to use schemas.** The PI's system prompt currently says nothing about output schemas. Add explicit guidance: "When dispatching endpoint discovery experiments, provide the EndpointDiscoveryResult schema. When dispatching auth experiments, provide the AuthTestResult schema." Make schema selection part of the dispatching workflow, not an afterthought.
- **Feed structured output into proven artifacts (connects to #10).** Once experiment results have predictable fields, auto-extraction becomes mechanical: a CONFIRMED experiment with `endpoint_url`, `method`, `required_headers` can be upserted into `ProvenArtifacts.fetches` without the PI manually calling `record_proven_artifact`.

## 13. PI Is Blind to Execution Details — Needs Full Results + Operation Metadata

**Gap:** When `submit_routine` runs the routine and sends it through inspection, the PI receives a **lossy summary** of what happened. The execution section in the tool response contains: `ok` (bool), `error` (string), `content_type`, `warnings`, and `data_preview` — a **500-character truncation** of `str(execution_result.data)`. The full response payload and per-operation metadata (`OperationExecutionMetadata`: type, duration, details, errors for each operation) are written to disk in the attempt record but **never returned to the PI**.

This means the PI is making routine iteration decisions — what to fix, what to retry, what to rearchitect — based on a 500-char data snippet and the inspector's prose summary. It cannot:

- See which specific operation failed or took abnormally long (no `operations_metadata`)
- Inspect the actual response body to understand what the API returned (truncated at 500 chars)
- See per-operation details like resolved placeholder values, actual request URLs after substitution, or redirect chains
- Diff the response data against what experiments proved should come back

**Why it matters:**
- **The PI guesses at root causes.** When a routine fails, the PI sees "HTTP 401" in the inspector's blocking issues but can't see *which* operation 401'd, what headers it sent, or what the error response body said. It may fix the wrong operation or add auth to an operation that already has it.
- **Data quality judgment is impossible.** A routine that returns 200 OK but with garbage data looks "successful" from the 500-char preview. The PI can't verify that the response contains the expected fields, has the right shape, or matches what experiments discovered.
- **Operation timing is invisible.** If one operation takes 45 seconds (timeout risk) or 0ms (likely cached/stale), the PI has no signal. These are critical for routine robustness but only exist in `operations_metadata`.
- **Iteration is blind.** The PI is expected to fix failed routines and resubmit, but it's iterating with less information than the inspector had. The inspector saw the full execution result; the PI sees a summary of the inspector's summary.

**Proposed fixes:**

- **Return full `operations_metadata` to the PI.** Always include the per-operation breakdown in the `submit_routine` response: operation type, duration, error (if any), and key details (resolved URL, status code, response size). This is compact — one line per operation — and gives the PI precise failure localization. This is the highest-priority fix.
- **Expand data preview or make it queryable.** Increase the 500-char truncation to at least 2000 chars, or better: write the full execution result to a file in the output directory and give the PI a tool to read it. A `get_attempt_details(attempt_id)` tool that loads the persisted attempt record from disk would let the PI drill into any past attempt on demand without bloating every `submit_routine` response.
- **PI filesystem access for attempt records.** The pipeline already writes detailed attempt records to `attempt_records/routine_attempt_N_*.json`. Give the PI a tool to read these files — either a dedicated `read_attempt_record` tool or a general-purpose file read scoped to the output directory. This turns the output directory into the PI's working memory, allowing it to review full execution results, past inspection scores, and operation metadata for any attempt without carrying it all in context.
- **Structured operation summary in the response.** At minimum, add a compact `operations_summary` field to the `submit_routine` response: `[{"op": 1, "type": "navigate", "ok": true, "duration_s": 1.2}, {"op": 2, "type": "fetch", "ok": false, "error": "401 Unauthorized", "duration_s": 0.3}]`. This gives the PI immediate visibility into the operation-level execution flow without requiring a separate tool call.
