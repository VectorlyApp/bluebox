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
