# Potential Improvements

## 1. WindowProperty Exploration Specialist

**Gap:** During Phase 1 exploration, no specialist analyzes `window_prop_events.jsonl`. The `WindowPropertyDataLoader` exists and is even pre-loaded in the worker's `execute_python` sandbox (`window_prop_entries`), but there's no `WindowPropertySpecialist` to analyze it during exploration.

**Why it matters:** Window properties are a rich source of:
- Embedded config objects (API keys, feature flags, site metadata)
- Auth tokens set directly on `window` by third-party SDKs
- `dataLayer` (Google Tag Manager) events that reveal user intent and parameters
- Framework state (e.g., `__NEXT_DATA__`, `window.__NUXT__`) that may duplicate or extend DOM blobs

**Proposed fix:** Add a `WindowPropertySpecialist` as a 5th parallel explorer in Phase 1, producing a `WindowPropertyExplorationSummary` (unique paths, interesting values, auth-relevant properties, narrative). Inject its summary into the PI's system prompt alongside the existing 4.

## 2. True Pipeline Resumability & Agent Thread Replay

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

## 3. Anti-Bot Detection as a First-Class Exploration Output

**Gap:** Anti-bot/security defenses (PerimeterX, Akamai, Dynatrace, CAPTCHA, bot challenge pages) are currently only mentioned in the network exploration prompt as a "mark low-interest and move on" hint. No specialist produces a structured summary of what defenses were observed, and the PI receives no explicit warning about them.

**Why it matters:** Anti-bot defenses are often the primary reason experiments fail — workers get 403s, CAPTCHA walls, or silent request drops that look like broken endpoints. If the PI doesn't know a site uses PerimeterX, it may waste multiple experiment attempts before realizing the issue is bot detection, not auth. Currently this insight is buried in the `narrative` free-text field at best.

**Proposed fix:** Have the `NetworkSpecialist` produce a dedicated `anti_bot_observations` field in `NetworkExplorationSummary` — a list of observed defenses (name, evidence, likely impact). Inject this prominently into the PI's system prompt so it can warn workers upfront and factor it into experiment strategy (e.g. "this site uses Akamai — navigate first and avoid direct fetch calls").

## 4. Trim Inspector Context to What It Actually Needs

**Gap:** Every `RoutineInspector` call receives ALL exploration summaries — network, DOM, storage, and UI — injected into the task prompt. The stated reason is "cross-reference." In practice, the inspector is judging a routine's correctness and robustness: whether endpoints are real, whether auth is handled, whether the execution result is meaningful. For that, it mainly needs the **network summary** (to verify endpoints and auth patterns). DOM, storage, and UI summaries add significant token cost but are rarely relevant to inspection decisions. The prompt is also hard-capped at 50,000 chars and truncated silently if exceeded — meaning on complex sites the summaries may crowd out the actual routine and execution result.

**Why it matters:** The inspector runs on every `submit_routine` call, potentially many times per pipeline run. Passing 4 full exploration summaries each time is wasteful and may actually hurt quality if truncation cuts off the execution result.

**Proposed fix:** Pass only the network summary to the inspector by default (the most relevant for verifying endpoints and auth). Make DOM/storage/UI summaries opt-in or omit them entirely. Alternatively, pre-summarize exploration context into a compact "site facts" block specifically for inspection use, rather than dumping raw full summaries.

## 5. Give Workers Access to Proven Artifacts

**Gap:** Workers are fully stateless — they receive only a task prompt from the PI and a data availability summary (raw counts). They have no access to exploration summaries, no access to prior experiment results, and no access to the ledger's `ProvenArtifacts`. The PI is expected to copy all relevant proven context (token endpoints, subscription keys, auth headers) into every experiment prompt manually. This is fragile — if the PI's prompt omits a detail, the worker has no way to recover it.

**Why it matters:** The PI writing complete auth instructions into every experiment prompt is verbose, error-prone, and burns PI context. Workers that need to authenticate before testing a data endpoint must rediscover auth from scratch if the PI's prompt is incomplete, even though the ledger already has proven fetches, tokens, and parameters sitting in `ProvenArtifacts`.

**Proposed fix:** Inject the ledger's `ProvenArtifacts` (proven fetches, tokens, navigations, parameters) into the worker's system prompt at dispatch time — not the full ledger, just the proven facts. This gives workers a reliable source of truth for auth and known-good values without requiring the PI to manually copy everything into each prompt. The injection should be compact and structured (not the full ledger summary) so it doesn't bloat worker context.
