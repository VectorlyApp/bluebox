"""
bluebox/agents/principal_investigator.py

PrincipalInvestigator (PI) agent — the orchestrator for Phase 2: Experiment-Driven
Routine Construction.

The PI has NO browser and NO domain tools. It only:
- Reads exploration summaries (in its system prompt)
- Reads the Discovery Ledger (routines planned, experiments, proven artifacts)
- Plans what routines to build from the exploration data
- Creates experiment tasks with specific hypotheses
- Records findings and proven artifacts
- Assembles routines and submits them for inspection
- Ships a catalog of routines when done
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime
from textwrap import dedent
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel, ValidationError
from toon import encode as toon_encode

from bluebox.agents.abstract_agent import (
    AbstractAgent,
    AgentCard,
    AutonomousRunConfig,
    ToolResultPersistMode,
    agent_tool,
)
from bluebox.agents.routine_inspector import RoutineInspector
from bluebox.data_models.orchestration.inspection import RoutineInspectionResult
from bluebox.agents.workspace import AgentWorkspace
from bluebox.agents.workers.experiment_worker import ExperimentWorker
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.experiment import (
    ArtifactType,
    ExperimentEntry,
    ExperimentStatus,
    ExperimentTakeaway,
    ExperimentVerdict,
)
from bluebox.data_models.orchestration.ledger import (
    DiscoveryLedger,
    RoutineAttempt,
    RoutineAttemptStatus,
    RoutineCatalog,
    RoutineSpec,
    RoutineSpecStatus,
    ShippedRoutine,
)
from bluebox.data_models.orchestration.task import (
    SubAgent,
    Task,
    TaskStatus,
    SpecialistAgentType,
)
from bluebox.data_models.orchestration.state import AgentOrchestrationState
from bluebox.data_models.routine.execution import RoutineExecutionResultWithMetadata
from bluebox.data_models.routine.routine import Routine
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
    from websocket import WebSocket

logger = get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Worker tool descriptions — injected into PI system prompt so the PI knows
# what workers can do and references tools by name in experiment methodologies.
# ---------------------------------------------------------------------------

WORKER_CAPABILITIES = dedent("""\
    ## Worker Capabilities

    Workers have access to the following tools. When writing experiment methodologies,
    reference these tools by name so the worker knows exactly what to use.

    BROWSER TOOLS (act in the live browser):
      browser_navigate(url) — go to a URL and wait for page load.
        TIP: Navigating directly to an API URL (e.g. https://api.example.com/data)
        bypasses CORS restrictions since it's a top-level navigation, not a fetch.
        The worker can then read the page body to get the JSON response.
      browser_eval_js(expression) — run JavaScript in the page context.
        Use for fetch() calls, DOM reads, clicks, storage access.
        If fetch() fails with CORS, try: mode 'no-cors', or navigate to the URL first.
      browser_cdp_command(method, params) — raw Chrome DevTools Protocol command.
        POWERFUL: Can intercept/modify network requests below the browser security layer.
        Key CDP methods for bypassing CORS/auth issues:
          - Fetch.enable + Fetch.continueRequest: intercept and modify requests
          - Network.enable + Network.getResponseBody: capture responses at protocol level
          - Network.setExtraHTTPHeaders: add headers to all requests
          - Page.navigate: navigate and capture response at CDP level
        Workers should use this when browser_eval_js fetch() fails due to CORS.
      browser_get_dom(selector?, max_depth?, include_tags?) — filtered view of current DOM

    CAPTURE LOOKUP TOOLS (search RECORDED session data — old, potentially stale):
      capture_search_transactions(query) — find requests in the recorded capture
      capture_get_transaction(request_id) — get full recorded request/response details
        USE THIS FIRST when an API call fails — it shows the exact headers, cookies,
        and parameters that worked during the original recorded session.
      capture_search_storage(query) — find recorded storage events
      capture_trace_value(value) — find where a value appears across the recorded capture
      capture_get_page_structure(snapshot_index?) — get recorded DOM structure
      capture_get_element(element_type, snapshot_index?) — get recorded element details

    Workers also receive the exploration summaries as shared context.
""")


class PrincipalInvestigator(AbstractAgent):
    """
    Orchestrator agent for experiment-driven routine construction.

    The PI reads exploration summaries, plans a catalog of routines,
    dispatches experiments to ExperimentWorker agents, reviews results,
    and assembles proven artifacts into shipped routines.

    The PI is self-organizing — no strict phases. It decides what to work on,
    when to switch routines, and when to call it done.
    """

    # Maximum time (seconds) a single worker or inspector can run before being killed.
    # Covers both LLM call hangs and browser/CDP hangs.
    WORKER_TIMEOUT_SECONDS: int = 180  # 3 minutes
    # If execution payload exceeds this size, persist it to inspector workspace raw/
    # and have the inspector analyze from file instead of inline prompt JSON.
    INSPECTOR_INLINE_EXECUTION_MAX_CHARS: int = 20_000
    # Default worker experiment output schema when PI omits one.
    # This enforces finalize_with_output(output={...}) instead of finalize_result.
    DEFAULT_WORKER_OUTPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "description": "Structured experiment findings object.",
        "additionalProperties": True,
    }
    # Error patterns → common-issues doc paths.
    # When an experiment result contains these keywords, the matching doc is
    # auto-injected into the get_experiment_result response so the PI sees
    # remediation guidance without having to search for it.
    _ERROR_DOC_PATTERNS: list[tuple[list[str], str]] = [
        (["failed to fetch", "typeerror: failed", "cors"], "common-issues/cors-failed-to-fetch.md"),
        (["401", "403", "unauthorized", "forbidden"], "common-issues/unauthenticated.md"),
        (["<html", "<!doctype", "cloudflare"], "common-issues/fetch-returns-html.md"),
        (["element not found", "selector"], "common-issues/element-not-found.md"),
        (["timeout", "timed out"], "common-issues/execution-timeout.md"),
        (["placeholder", "unresolved"], "common-issues/placeholder-not-resolved.md"),
    ]

    AGENT_CARD = AgentCard(
        description=(
            "Orchestrates experiment-driven routine construction. Reads exploration "
            "summaries, plans a routine catalog, dispatches experiments to workers, "
            "reviews results, and assembles proven artifacts into shipped routines."
        ),
    )

    # -----------------------------------------------------------------------
    # System prompts
    # -----------------------------------------------------------------------

    SYSTEM_PROMPT_CORE: str = dedent("""\
        You are a Principal Investigator (PI) for automated routine construction.

        ## Your Role

        You are the strategist. You have NO browser and NO domain tools. You:
        1. Read exploration summaries to understand the captured session
        2. Plan what routines to build (a catalog — often multiple routines)
        3. Design experiments with specific, falsifiable hypotheses
        4. Dispatch experiments to workers who have browser + capture lookup tools
        5. Review results and record verdicts
        6. Accumulate proven artifacts (fetches, navigations, tokens, parameters)
        7. Assemble routines and submit them for validation + inspection
        8. Ship routines that pass, iterate on ones that fail
        9. Call mark_complete when all routines are addressed

        ## Data + Python Access (PI and Workers)

        You and your workers both have Python execution tools and workspace access.
        Capture files are mounted under each agent workspace `raw/` directory.

        Use this intentionally:
        - For quick, precise analysis, call `execute_python` on the PI or delegate
          Python analysis to workers.
        - In Python, read mounted capture files from `raw/` when needed (for example,
          to inspect exact payloads or run focused filtering).
        - Prefer targeted queries over broad dumps; summarize findings back into
          concise experiment methodologies and routine designs.

        ## MANDATORY: Review Documentation First

        BEFORE planning or dispatching ANY experiments, you MUST review the routine
        documentation to understand the full capabilities available to you. Call:

        1. search_files(scope="docs", query="operation", mode="exact") — to see all operation types (navigate, fetch,
           click, input_text, js_evaluate, get_cookies, download, etc.)
        2. read_file(scope="docs", path="...") on the Routine and operation model files to understand
           what each operation can do and its required fields

        This is NOT optional. You cannot dispatch experiments until you have reviewed
        the documentation. Routines are much more powerful than simple fetch calls —
        they support UI automation (click, type, scroll), JavaScript evaluation,
        cookie extraction, file downloads, and more. Understanding these capabilities
        BEFORE you plan will lead to better routine designs.

        ## Catalog-First Thinking

        AFTER reviewing docs, call plan_routines to declare what routines you intend
        to build. The exploration data reveals the API surface — each distinct
        capability can be a routine.

        Then work through routines by priority, starting with shared dependencies
        (auth, navigation) that apply across routines.

        ## How Experiments Work

        Each experiment has:
        - **hypothesis**: What you're testing (specific and falsifiable)
        - **rationale**: WHY — what evidence led here, what you expect to learn
        - **methodology**: Instructions for the worker (reference worker tools by name!)

        The worker executes the experiment and returns structured output.
        You review the output, record a verdict, and decide what to try next.

        ## Experiment Pattern — Exact Replay First

        For API endpoint validation, your DEFAULT first experiment should be:
        "Perform the exact fetch request as observed in capture, then report what changed."

        Your experiment methodology should explicitly instruct the worker to:
        1. Call `capture_search_transactions` to find the candidate request
        2. Call `capture_get_transaction` for the exact captured request
        3. Re-run the same request in live browser using `browser_navigate` + `browser_eval_js`
        4. Report a field-by-field diff:
           - URL/path/query differences
           - Header differences
           - Body shape/value differences
           - Status code and response shape differences
        5. If the replay fails, test one minimal delta at a time (never many at once)

        Include this sentence in methodologies when relevant:
        "Replay the exact captured fetch first; do not generalize until exact replay is tested."

        ## Strategy

        1. Review documentation with search_files(scope="docs", ...) and read_file(scope="docs", ...) (MANDATORY first step)
        2. Call plan_routines to declare your catalog plan
        3. IDENTIFY AUTH DEPENDENCIES FIRST (see below) — solve auth before data endpoints
        4. Use dispatch_experiments_batch to test routines in parallel (respecting dependencies)
        5. Record findings, then batch the next round of experiments
        6. For each routine: experiment → prove → assemble → submit_routine → ship or fix
        7. Use follow_up to ask the SAME worker clarifying questions (preserves context)
        8. submit_routine REQUIRES test_parameters — provide realistic values for EVERY parameter!
           The routine will be executed in a live browser and reviewed by an independent inspector.
           If the routine has 0 parameters, pass test_parameters: {}
        9. When a routine passes inspection: mark_routine_shipped
        10. When a routine is hopeless: mark_routine_failed
        11. When all routines are addressed: mark_complete with a usage guide

        ## Auth-First Dependency Ordering — CRITICAL

        Most APIs require authentication (tokens, API keys, session cookies). If the
        exploration summaries mention ANY of these patterns:
        - JWT / Bearer tokens
        - API keys / subscription keys (e.g. Ocp-Apim-Subscription-Key)
        - Session tokens / CSRF tokens
        - OAuth flows
        - Login/cookie-based auth

        Then you MUST follow this order:

        **Phase A — Solve auth FIRST (before any data endpoints):**
        1. Dispatch experiments ONLY for the auth/token routine
        2. The worker must: find the token endpoint in captures, extract the exact
           headers/body/key needed, call it live, and prove it returns a valid token
        3. Record the proven auth artifact (token URL, required headers, subscription key)
        4. Only proceed to Phase B after auth is CONFIRMED working

        **Phase B — Test data endpoints WITH proven auth:**
        1. NOW batch experiments for data endpoints
        2. Each experiment methodology must include the proven auth details from Phase A
           so the worker can authenticate before calling the data endpoint
        3. Workers are independent — they do NOT share state. You must pass the
           auth instructions (token URL, headers, key) in every experiment methodology.

        **NEVER do this:**
        - Batch auth + data experiments in parallel (data will fail without auth)
        - Skip auth and hope data endpoints are public (check exploration first)
        - Give up on data endpoints because they 401'd — solve auth first, then retry

        **For public endpoints** (no auth required): batch freely in parallel.

        ## Auth Token Resolution — Where Tokens Come From

        Tokens and API keys can live in MANY places. When designing auth experiments,
        you MUST explore MULTIPLE resolution strategies, not just one. If one fails,
        try the next. Include observed values from captures so workers know what to
        look for.

        **Source 1: Network captures (token endpoints)**
        The most common pattern — a dedicated API endpoint returns a token.
        - Search captures for URLs containing "token", "auth", "login", "oauth"
        - Get the EXACT headers and body from the capture with capture_get_transaction
        - Tell the worker: "Call POST {token_url} with these headers: {headers} and
          body: {body}. In the capture the response had a field called {field} with
          a token that looked like '{first_20_chars}...'"
        - The routine chains this: fetch token → store → use in subsequent fetches

        **Source 2: DOM (inline scripts, meta tags, data attributes)**
        Sites embed tokens/keys directly in the HTML page.
        - Meta tags: `<meta name="csrf-token" content="...">`
        - Inline config: `window.__CONFIG__ = { apiKey: "..." }`
        - Data attributes: `<div data-api-key="...">`
        - Experiment methodology: "Navigate to {url}, then run JS to check:
          document.querySelector('meta[name=csrf-token]'), window.__CONFIG__,
          window.__INITIAL_STATE__, window.ENV. We saw a key like '{observed_value}'
          in the captured session — is it still the same or has it changed?"
        - Routine uses: `{{meta:csrf-token}}` or `{{windowProperty:__CONFIG__.apiKey}}`

        **Source 3: Browser storage (localStorage / sessionStorage)**
        Sites store tokens after their JS authenticates on page load.
        - Experiment methodology: "Navigate to {url}, wait 3 seconds for JS to execute,
          then dump sessionStorage and localStorage. Look for keys containing
          'token', 'auth', 'jwt', 'session'. In the capture we saw '{key_name}'
          with value starting '{prefix}...'"
        - Routine uses: `{{localStorage:auth.access_token}}` or
          `{{sessionStorage:token.jwt}}`

        **Source 4: Cookies**
        Some sites use cookie-based auth — navigation establishes the session.
        - Experiment methodology: "Navigate to {url}, then try calling {data_endpoint}
          with credentials:'include'. If it works, auth is cookie-based and the
          routine just needs navigate + fetch with credentials:'include'. If it
          fails, dump cookies with get_cookies to see what exists."
        - Routine uses: `credentials: "include"` or `{{cookie:XSRF-TOKEN}}`

        **Source 5: Window properties (JS globals)**
        Sites set global variables with config and auth.
        - Experiment methodology: "Navigate to {url}, run JS to check window.__CONFIG__,
          window.__INITIAL_STATE__, window.ENV, window.__NEXT_DATA__"
        - Routine uses: `{{windowProperty:__CONFIG__.apiKey}}`

        **Source 6: JS evaluation (compute from page state)**
        When tokens are derived/computed by the site's JS and stored in non-obvious places.
        - Experiment methodology: "Navigate to {url}, wait for page load. The site's JS
          likely stores auth state somewhere. Try: JSON.parse(sessionStorage.getItem(
          'persist:root')).auth, or look through all sessionStorage keys for anything
          containing 'token'. Extract the value and try using it."
        - Routine uses: js_evaluate operation to extract + store in sessionStorage

        **CRITICAL: When dispatching auth experiments, ALWAYS include:**
        1. The observed token/key value (or first 20 chars) from the captured session
        2. Where you found it in captures (which header, which response field)
        3. Whether it appears static (same across captures) or dynamic (different each time)
        4. Multiple strategies to try — "First try X, if that fails try Y, then Z"

        ## Hardcoding Site-Level Credentials — CRITICAL

        Many sites use API keys, subscription keys, or client IDs that are NOT user
        secrets — they are site-wide constants baked into the website's JavaScript,
        HTML meta tags, or network requests. Examples:
        - Ocp-Apim-Subscription-Key
        - x-api-key / apiKey / client_id
        - Firebase API keys
        - Public OAuth client IDs

        These MUST be resolved from captures (network headers, DOM, storage) and
        HARDCODED directly into the routine. They must NEVER be exposed as user
        parameters — no user would know where to find them.

        **Resolution order for static keys:**
        1. Network captures: check request headers from capture_get_transaction
        2. DOM: check inline scripts, meta tags, window.* config objects
        3. Storage: check localStorage/sessionStorage for cached keys
        4. If found in captures, hardcode the value directly in routine headers/body

        **JWT/Bearer tokens are DIFFERENT** — they expire and must be fetched at
        runtime via a fetch operation within the routine. But the API key USED TO
        fetch the token should itself be hardcoded.

        **When building routines:** only parameterize values that a USER would
        naturally provide (search terms, dates, IDs, locations). Everything else
        should be hardcoded from captures.

        ## Parallel Experiments — ALWAYS PREFER BATCH (within dependency order)

        ALWAYS use dispatch_experiments_batch instead of dispatch_experiment when you
        have 2+ INDEPENDENT experiments to run. This runs them IN PARALLEL on separate
        workers — N experiments complete in the time of 1.

        But NEVER batch experiments that have unresolved dependencies on each other.
        Auth must be solved before data endpoints. Reference data (e.g. station lists)
        should be solved before parameterized endpoints that depend on those IDs.

        dispatch_experiment (singular) should ONLY be used when you need follow_up
        on a specific worker's prior context. For all new experiments, batch them.

        Batch aggressively:
        - After plan_routines, immediately batch experiments for all priority-1 routines
        - When testing multiple API endpoints, batch them all at once
        - When probing auth + multiple data endpoints, batch everything together

        ## Routine Naming & Documentation Standards

        These routines will be VECTORIZED and stored in databases for other agents to
        discover via semantic search. Poor names and vague descriptions make routines
        invisible and unusable. Follow these rules strictly:

        **Routine name** — snake_case, verb_noun pattern, 3+ segments, MUST include site context:
          The name must make sense in isolation — another agent reading ONLY the name
          should know what site/service this targets and what it does. Include a short
          site identifier as a prefix or qualifier.

          GOOD: get_premierleague_standings, search_premierleague_matches_by_season,
                fetch_amtrak_train_schedules, download_arxiv_paper_pdf,
                list_espn_upcoming_fixtures, get_github_repo_stars
          BAD:  get_standings (standings from where?), get_content_item (what content?
                what site?), fetch_data (completely generic), search_matches (which sport?
                which site?), get_league_standings (which league? which site?)

        **Routine description** — ≥8 words, must explain:
          1. What it does (the action)
          2. What inputs it accepts (parameters)
          3. What data it returns (the output)
          GOOD: "Fetches Premier League standings for a given competition ID and
                 season ID, returning team names, positions, wins, draws, losses,
                 goals scored, goals conceded, and total points."
          BAD:  "Get standings" (too short, no input/output info)
          BAD:  "A routine for the Premier League" (doesn't say what it does or returns)

        **Parameter names** — snake_case, descriptive:
          GOOD: competition_id, season_year, team_name, departure_date
          BAD:  id (ambiguous), param1 (meaningless), x (cryptic)

        **Parameter descriptions** — ≥3 words, explain what the value represents:
          GOOD: "The unique competition identifier (e.g. 1 for Premier League)"
          GOOD: "Season year in YYYY format (e.g. 2024)"
          BAD:  "ID" (too terse)
          BAD:  "The season" (doesn't explain format or expected values)

        **Non-obvious parameter sourcing** — CRITICAL for opaque IDs and codes:
          If a parameter is NOT something a human would naturally know (e.g. an internal
          numeric ID, a slug, an encoded token, a UUID), the description MUST explain
          WHERE to get that value. The user calling this routine has no idea what
          "competition_id: 1" means unless you tell them how to find it.

          GOOD: "Internal competition ID. Obtain from the get_competitions routine or
                 the /competitions API endpoint. Example: 1 = Premier League, 2 = Championship."
          GOOD: "Season ID as used by the Premier League API. Use the get_seasons routine
                 to list valid season IDs for a competition. Example: 418 = 2023-24 season."
          GOOD: "Team slug as it appears in the site URL path (e.g. 'arsenal', 'manchester-united').
                 Find by calling get_teams or navigating to the team page."
          BAD:  "The competition ID" (where do I get it?)
          BAD:  "Season identifier" (what values are valid? how do I look them up?)

          Rule of thumb: if you can't google the value, the description must say how to get it.

        ## CRITICAL RULES

        - NEVER guess at request details. Always dispatch experiments to verify.
        - Write experiment methodologies that reference worker tools by name.
        - Record a verdict for EVERY completed experiment via record_finding.
        - Always include reusable takeaways in record_finding so future workers
          receive concrete lessons (claim + how_to_apply_next + evidence).
        - If an experiment is ambiguous, use follow_up — don't dispatch a new one.
        - ALWAYS provide test_parameters when calling submit_routine — the routine
          WILL be executed and inspected. Use realistic values the experiments proved work.
          If the routine has 0 parameters, pass test_parameters: {}
        - DEPENDENCY ORDER IS SACRED: auth → reference data → data endpoints → assembly.
          NEVER dispatch data endpoint experiments until auth is CONFIRMED working.
          NEVER give up on data endpoints just because they returned 401 — that means
          you need to solve auth first, not that the endpoint is broken.
        - Workers do NOT share browser state. When an endpoint requires auth, your
          experiment methodology must include FULL auth instructions (token URL, headers,
          subscription key) so the worker can authenticate within its own session.
        - mark_routine_failed is globally gated: you cannot fail any routine until at
          least 5 routine attempts have failed across the pipeline. Keep iterating and
          submitting improved routines before giving up on individual specs.

        ## Resilience — NEVER Give Up Early

        - NEVER call mark_failed after fewer than 5 experiments per routine.
          CORS failures, 400 errors, and network issues are NORMAL obstacles, not
          reasons to quit. They mean you need a different approach, not that the
          pipeline is hopeless.
        - When a fetch fails (CORS, 400, timeout), iterate with alternative approaches:
          1. Use capture_search_transactions / capture_get_transaction to see the EXACT
             request headers and patterns that worked in the recorded session, then
             replicate them in the worker's experiment.
          2. Use browser_cdp_command with Fetch.enable to intercept requests at the CDP
             level — this bypasses CORS entirely since it operates below the browser
             security layer.
          3. Try navigating directly to the API URL with browser_navigate — GET requests
             via top-level navigation don't have CORS restrictions.
          4. Try fetch with mode: 'no-cors' or from a different origin context.
          5. Check if the site's JS uses a proxy path (e.g. /api/* proxied to the API
             domain) — search the captured network data for path patterns.
        - If ALL alternative approaches fail for a routine, mark_routine_failed for THAT
          routine and move on to the next one. Do NOT call mark_failed (pipeline-level)
          unless every single routine has been individually addressed.
        - Use follow_up liberally — it's cheaper than dispatching a new experiment and
          preserves the worker's browser state and context.

        ## Common Execution Failures — MUST READ

        ### TypeError: Failed to fetch (CORS)
        If a routine's fetch operation fails with "TypeError: Failed to fetch", this
        almost always means the browser's current origin doesn't match the API's
        CORS Access-Control-Allow-Origin header. Routines start from about:blank
        (origin = null), so ANY cross-origin fetch will fail without navigation.

        The fix is to add a `navigate` operation BEFORE the first fetch to set the
        browser origin to the allowed domain.

        Example: If the API is at https://api.example.com but CORS only allows
        https://www.example.com, the routine MUST start with:
          {"type": "navigate", "url": "https://www.example.com"}
        before any fetch to https://api.example.com/...

        RULE: Every routine that calls an external API MUST start with a navigate
        operation. This is cheap (one page load) and prevents CORS issues. If you
        see "Failed to fetch" in an inspection blocking issue, ADD A NAVIGATE OP.

        For more details: search_files(scope="docs", query="cors-failed-to-fetch", mode="exact")

        ### HTTP 401/403 (Authentication)
        If a fetch returns 401/403, the routine is missing authentication. Check
        experiment findings for auth token endpoints and subscription keys. The
        routine must obtain a token (via fetch + js_evaluate) before calling
        protected endpoints. For more details: search_files(scope="docs", query="unauthenticated", mode="exact")
    """)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        task: str,
        # Exploration summaries — injected into system prompt
        exploration_summaries: dict[str, str] | None = None,
        # Data loaders — passed through to workers
        network_data_loader: NetworkDataLoader | None = None,
        storage_data_loader: StorageDataLoader | None = None,
        dom_data_loader: DOMDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        # Browser context — passed through to workers
        remote_debugging_address: str | None = None,
        # Resume support — pass an existing ledger to pick up where a previous PI left off
        ledger: DiscoveryLedger | None = None,
        # LLM config
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        worker_llm_model: LLMModel | None = None,
        max_iterations: int = 200,
        worker_max_loops: int = 30,
        max_attempts_per_routine: int = 5,
        min_experiments_before_fail: int = 10,
        min_global_failed_attempts_before_routine_failure: int = 5,
        # Agent pool sizes
        num_workers: int = 3,
        num_inspectors: int = 1,
        # Persistence callbacks
        on_ledger_change: Callable[[DiscoveryLedger, str], None] | None = None,
        on_agent_thread: Callable[[str, str, list[dict[str, Any]]], None] | None = None,
        on_attempt_record: Callable[[dict[str, Any]], None] | None = None,
        # Standard agent args
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        workspace: AgentWorkspace | None = None,
        worker_workspace_factory: Callable[[], AgentWorkspace] | None = None,
        inspector_workspace_factory: Callable[[], AgentWorkspace] | None = None,
    ) -> None:
        # Task
        self._task = task
        self._max_iterations = max_iterations
        self._worker_max_loops = worker_max_loops
        self._max_attempts_per_routine = max_attempts_per_routine
        self._min_experiments_before_fail = min_experiments_before_fail
        self._min_global_failed_attempts_before_routine_failure = max(
            0,
            int(min_global_failed_attempts_before_routine_failure),
        )

        # Exploration context
        raw_summaries = exploration_summaries or {}
        self._exploration_summaries_raw = dict(raw_summaries)
        self._exploration_summaries = self._toonify_exploration_summaries(raw_summaries)

        # Data loaders (passed through to workers)
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._dom_data_loader = dom_data_loader
        self._window_property_data_loader = window_property_data_loader
        self._documentation_data_loader = documentation_data_loader

        # Browser context (passed through to workers)
        self._remote_debugging_address = remote_debugging_address

        # LLM
        self._worker_llm_model = worker_llm_model or llm_model

        # Agent pools
        self._num_workers = num_workers
        self._num_inspectors = num_inspectors
        self._worker_counter = 0  # Round-robin counter for workers
        self._inspector_counter = 0  # Round-robin counter for inspectors

        # Persistence callbacks
        self._on_ledger_change = on_ledger_change
        self._on_agent_thread = on_agent_thread
        self._on_attempt_record = on_attempt_record
        self._worker_workspace_factory = worker_workspace_factory
        self._inspector_workspace_factory = inspector_workspace_factory

        # Internal state — the Discovery Ledger tracks everything
        # Accept an existing ledger for resume after context exhaustion
        self._ledger = ledger or DiscoveryLedger(user_task=task)
        self._orchestration_state = AgentOrchestrationState()
        self._agent_instances: dict[str, AbstractAgent] = {}
        self._is_done = False
        self._pipeline_result: RoutineCatalog | None = None
        self._recent_tool_calls: list[str] = []  # Track recent tool names for loop detection
        self._docs_reviewed: bool = False  # Gate: must review docs before dispatching experiments

        super().__init__(
            emit_message_callable=emit_message_callable,
            workspace=workspace,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=documentation_data_loader,
            allow_code_execution=True,
        )

        logger.debug(
            "PrincipalInvestigator initialized: task=%s, explorations=%s",
            task[:80],
            list(self._exploration_summaries.keys()),
        )

    @staticmethod
    def _toonify_exploration_summaries(
        summaries: dict[str, str],
    ) -> dict[str, str]:
        """
        TOON-encode exploration summaries for prompt efficiency.

        If a summary is JSON text, parse and encode the underlying object.
        Non-JSON text is passed through unchanged.
        """
        toonified: dict[str, str] = {}
        for domain, summary in summaries.items():
            if not isinstance(summary, str):
                toonified[domain] = toon_encode(summary)
                continue

            stripped = summary.strip()
            if not stripped:
                toonified[domain] = summary
                continue

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                toonified[domain] = summary
                continue

            toonified[domain] = toon_encode(parsed)
        return toonified

    @staticmethod
    def _encode_prompt_payload(payload: Any) -> str:
        """TOON-encode prompt payloads with a JSON fallback."""
        try:
            return toon_encode(payload)
        except Exception:
            return json.dumps(payload, ensure_ascii=False, default=str)

    # -----------------------------------------------------------------------
    # System prompt
    # -----------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        parts: list[str] = [self.SYSTEM_PROMPT_CORE]
        if self.has_workspace:
            try:
                summary = self._require_workspace().generate_summary()
                parts.append(f"\n\n## Workspace Summary\n{summary}")
            except Exception as e:
                logger.warning("Failed to generate PI workspace summary: %s", e)

        # Routine JSON schema — auto-generated from the Pydantic models
        parts.append("\n## Routine JSON Schema\n\n")
        parts.append("When calling submit_routine, the routine_json MUST conform to this schema.\n")
        parts.append("Every operation needs a 'type' field as discriminator.\n\n")
        parts.append(Routine.model_schema_markdown())
        parts.append("\n\n### Example routine_json (fetch a parameterized API)\n\n")
        parts.append(dedent("""\
            ```json
            {
              "name": "get_premierleague_standings",
              "description": "Fetches Premier League standings for a given competition and season, returning team names, positions, wins, draws, losses, goals scored, goals conceded, and total points.",
              "parameters": [
                {"name": "competition_id", "type": "integer", "description": "Internal competition identifier. Obtain from the get_premierleague_competitions routine. Example: 1 = Premier League."},
                {"name": "season_id", "type": "integer", "description": "Season identifier as used by the Premier League API. Obtain from the get_premierleague_seasons routine. Example: 418 = 2023-24 season."}
              ],
              "operations": [
                {
                  "type": "navigate",
                  "url": "https://www.example.com"
                },
                {
                  "type": "fetch",
                  "endpoint": {
                    "url": "https://api.example.com/competitions/{{competitionId}}/seasons/{{seasonId}}/standings",
                    "method": "GET"
                  },
                  "session_storage_key": "standings_result"
                },
                {
                  "type": "return",
                  "session_storage_key": "standings_result"
                }
              ]
            }
            ```
        """))

        # Worker capabilities
        parts.append(WORKER_CAPABILITIES)
        parts.append(self._generate_code_execution_prompt())

        # Exploration summaries
        if self._exploration_summaries:
            parts.append("\n## Exploration Summaries\n")
            for domain, summary in self._exploration_summaries.items():
                parts.append(f"### {domain}\n{summary}\n")

        # Discovery Ledger
        if self._ledger.routine_specs or self._ledger.experiments or self._ledger.attempts:
            ledger_payload = self._ledger.model_dump(
                mode="json",
                exclude={
                    "experiments": {
                        "__all__": {
                            "prompt": True,
                            "output": True,
                        }
                    },
                    "attempts": {
                        "__all__": {
                            "routine_json": True,
                            "execution_result": True,
                            "inspection_result": True,
                        }
                    },
                },
            )
            parts.append("\n## Discovery Ledger\n")
            parts.append(self._encode_prompt_payload(ledger_payload))

        # Task queue status
        queue = self._orchestration_state.get_queue_status()
        if any(v > 0 for v in queue.values()):
            parts.append("\n## Task Queue\n")
            parts.append(self._encode_prompt_payload(queue))

        return "".join(parts)

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> RoutineCatalog | None:
        """
        Run the PI loop to completion.

        Returns:
            A RoutineCatalog of shipped routines, or None if construction failed.
        """
        # Seed the conversation — detect resume vs fresh start
        is_resume = bool(self._ledger.experiments or self._ledger.routine_specs)

        if is_resume:
            initial_message = (
                f"TASK: {self._task}\n\n"
                "You are RESUMING a previous session that ran out of context. "
                "Your Discovery Ledger has been preserved with all prior work.\n\n"
                "FIRST: Call get_ledger to see exactly where things stand — "
                "what routines are planned, what experiments have been run, "
                "what's shipped, and what still needs work.\n\n"
                "Then pick up where the previous session left off. Do NOT repeat "
                "experiments that already have verdicts."
            )
            logger.info(
                "PI resuming: %d specs, %d experiments, %d attempts",
                len(self._ledger.routine_specs),
                len(self._ledger.experiments),
                len(self._ledger.attempts),
            )
        else:
            initial_message = (
                f"TASK: {self._task}\n\n"
                "MANDATORY FIRST STEP: Review the routine documentation before doing anything else.\n"
                "Call search_files(scope='docs', query='operation', mode='exact') and read_file(scope='docs', path='...') on the Routine/operation model files\n"
                "to understand ALL operation types (fetch, click, input_text, js_evaluate, download, etc.).\n"
                "You CANNOT dispatch experiments until you understand the full routine capabilities.\n\n"
                "After reviewing docs:\n"
                "1. Analyze the exploration summaries\n"
                "2. Call plan_routines to declare what routines to build\n"
                "3. Use dispatch_experiments_batch to test ALL priority-1 routines IN PARALLEL\n"
                "4. Record findings for each, then batch the next round\n"
                "5. Build and submit_routine for each proven routine\n"
                "6. Call mark_complete when all routines are shipped or failed\n\n"
                "IMPORTANT: Always use dispatch_experiments_batch (not dispatch_experiment) "
                "to run multiple experiments in parallel. This is much faster."
            )
        self._add_chat(ChatRole.USER, initial_message)

        for iteration in range(self._max_iterations):
            if self._is_done:
                logger.info("PI completed after %d iterations", iteration)
                self._dump_agent_thread("principal_investigator", self)
                return self._pipeline_result

            messages = self._build_messages_for_llm()
            response = self._call_llm(
                messages,
                self._get_system_prompt(),
                tool_choice="required",
            )

            # Add assistant response to chat
            self._add_chat(
                ChatRole.ASSISTANT,
                content=response.content or "",
                tool_calls=response.tool_calls,
                llm_provider_response_id=response.response_id,
            )

            # Persist PI thread after every iteration
            self._dump_agent_thread("principal_investigator", self)

            if response.tool_calls:
                self._process_tool_calls(response.tool_calls)

                # Track docs review — any docs tool call satisfies the gate
                _DOCS_TOOLS = {"search_files", "read_file", "list_files"}
                for tc in response.tool_calls:
                    if tc.tool_name in _DOCS_TOOLS:
                        scope = (tc.tool_arguments or {}).get("scope")
                        if scope == "docs":
                            self._docs_reviewed = True

                # Loop detection: track recent tool calls (name + whether it errored)
                for tc in response.tool_calls:
                    self._recent_tool_calls.append(tc.tool_name)
                # Keep only last 6
                self._recent_tool_calls = self._recent_tool_calls[-6:]

                # Detect stuck patterns:
                # 1. Same tool called 3+ times in a row (any tool)
                # 2. Alternating pair (e.g. mark_routine_failed + get_ledger)
                _is_stuck = False
                _stuck_msg = ""

                if len(self._recent_tool_calls) >= 3:
                    last3 = self._recent_tool_calls[-3:]
                    # Same tool 3x in a row
                    if len(set(last3)) == 1:
                        _is_stuck = True
                        _stuck_msg = f"calling {last3[0]} repeatedly"

                if not _is_stuck and len(self._recent_tool_calls) >= 4:
                    last4 = self._recent_tool_calls[-4:]
                    # Alternating pair: A B A B
                    if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                        _is_stuck = True
                        _stuck_msg = f"alternating between {last4[0]} and {last4[1]}"

                if _is_stuck:
                    self._recent_tool_calls.clear()
                    shipped = sum(1 for s in self._ledger.routine_specs if s.status == RoutineSpecStatus.SHIPPED)
                    failed = sum(1 for s in self._ledger.routine_specs if s.status == RoutineSpecStatus.FAILED)
                    unaddressed = [
                        s for s in self._ledger.routine_specs
                        if s.status not in (RoutineSpecStatus.SHIPPED, RoutineSpecStatus.FAILED)
                    ]
                    unaddressed_names = [f"{s.name} ({s.status.value})" for s in unaddressed]
                    self._add_chat(
                        ChatRole.USER,
                        f"STOP — you are stuck {_stuck_msg}. This is wasting iterations.\n\n"
                        f"Current progress: {shipped} shipped, {failed} failed, "
                        f"{len(unaddressed)} unaddressed.\n"
                        f"Unaddressed routines: {', '.join(unaddressed_names) or 'none'}\n\n"
                        "If a tool keeps returning an error, do NOT retry the same call. "
                        "Read the error message and change your approach.\n\n"
                        "To make progress you MUST do ONE of these:\n"
                        "1. dispatch_experiments_batch for unaddressed routines\n"
                        "2. submit_routine with VALID routine_json AND test_parameters "
                        "(if routine has 0 params, pass test_parameters: {})\n"
                        "3. mark_routine_failed for routines that truly can't work\n"
                        "4. mark_complete if all routines are shipped or failed\n\n"
                        "Pick ONE action for a DIFFERENT routine than the one you're stuck on.",
                    )
            else:
                # Nudge the PI to act
                self._add_chat(
                    ChatRole.USER,
                    "You must use a tool. Dispatch an experiment, record a finding, "
                    "submit a routine, or mark_complete if done.",
                )

        logger.warning("PI exhausted %d iterations without completing", self._max_iterations)
        # Dump PI thread before returning partial results
        self._dump_agent_thread("principal_investigator", self)
        # Return whatever we've shipped so far
        return self._build_partial_catalog()

    # ===================================================================
    # Persistence — notify external listener after ledger mutations
    # ===================================================================

    def _persist(self, reason: str) -> None:
        """Fire the on_ledger_change callback if registered."""
        if self._on_ledger_change is not None:
            try:
                self._on_ledger_change(self._ledger, reason)
            except Exception as e:
                logger.warning("on_ledger_change callback failed: %s", e)

    def _record_attempt(
        self,
        spec: RoutineSpec,
        attempt: RoutineAttempt,
        routine_json: dict[str, Any],
        test_parameters: dict[str, Any],
        execution_result: RoutineExecutionResultWithMetadata | None,
        inspection_result: dict[str, Any] | None,
    ) -> None:
        """Persist a unified attempt record via the on_attempt_record callback."""
        if self._on_attempt_record is None:
            return
        try:
            # Count which attempt number this is for the spec
            spec_attempts = self._ledger.get_attempts_for_spec(spec.id)
            attempt_number = len(spec_attempts)

            record: dict[str, Any] = {
                "attempt_id": attempt.id,
                "spec_id": spec.id,
                "spec_name": spec.name,
                "spec_description": spec.description,
                "attempt_number": attempt_number,
                "timestamp": str(datetime.now()),
                "verdict": attempt.status.value,
                "routine_json": routine_json,
                "test_parameters": test_parameters,
                "execution_result": (
                    execution_result.model_dump() if execution_result is not None else None
                ),
                "inspection_result": inspection_result,
            }
            self._on_attempt_record(record)
        except Exception as e:
            logger.warning("on_attempt_record callback failed: %s", e)

    def _dump_agent_thread(self, agent_label: str, agent: AbstractAgent) -> None:
        """Dump an agent's full message history via the on_agent_thread callback."""
        if self._on_agent_thread is None:
            return
        try:
            chats = agent.get_chats()
            messages = []
            for c in chats:
                try:
                    msg = {
                        "id": c.id,
                        "role": c.role.value if c.role else "unknown",
                        "content": c.content or "",
                        "tool_calls": [
                            {"tool_name": tc.tool_name, "arguments": tc.tool_arguments, "call_id": tc.call_id}
                            for tc in (c.tool_calls or [])
                        ],
                        "tool_call_id": c.tool_call_id,
                        "created_at": str(c.created_at) if c.created_at else None,
                    }
                    messages.append(msg)
                except Exception as chat_err:
                    logger.warning("Failed to serialize chat %s for %s: %s", c.id, agent_label, chat_err)
                    messages.append({"id": c.id, "role": "unknown", "content": f"[serialization error: {chat_err}]"})
            thread_id = agent.get_thread().id if agent.get_thread() else "unknown"
            self._on_agent_thread(agent_label, thread_id, messages)
        except Exception as e:
            logger.warning("on_agent_thread callback failed for %s: %s", agent_label, e, exc_info=True)

    # ===================================================================
    # CATALOG PLANNING TOOLS
    # ===================================================================

    @agent_tool()
    def _plan_routines(
        self,
        specs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Declare what routines to build from the exploration data.

        Call this early after analyzing exploration summaries. Each spec
        represents a distinct capability to extract from the site.
        Can be called again to add new specs discovered during experimentation.

        Args:
            specs: List of routine specs. Each dict has:
                - name: Short name (e.g. "get_league_standings")
                - description: What the routine does
                - priority: 1=must-have, 2=should-have, 3=nice-to-have (default 1)
        """
        created_ids: list[str] = []
        for spec_dict in specs:
            spec = RoutineSpec(
                name=spec_dict.get("name") or spec_dict.get("id", "unnamed"),
                description=spec_dict.get("description", ""),
                priority=spec_dict.get("priority", 1),
            )
            self._ledger.add_spec(spec)
            created_ids.append(spec.id)

        # Auto-set the first one as active if none is active
        if self._ledger.active_spec_id is None and self._ledger.routine_specs:
            self._ledger.active_spec_id = self._ledger.routine_specs[0].id

        self._persist("plan_routines")
        return {
            "created": len(created_ids),
            "spec_ids": created_ids,
            "total_specs": len(self._ledger.routine_specs),
        }

    @agent_tool()
    def _set_active_routine(self, spec_id: str) -> dict[str, Any]:
        """
        Switch focus to a different routine. The PI works on one routine at a
        time but can switch when blocked or when dependencies are shared.

        Args:
            spec_id: ID of the RoutineSpec to focus on.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        self._ledger.active_spec_id = spec_id
        return {"active": spec.name, "status": spec.status.value}

    # ===================================================================
    # EXPERIMENT TOOLS
    # ===================================================================

    def _truncate_text_for_briefing(self, value: Any, max_chars: int = 220) -> str:
        """Normalize arbitrary values to a compact, single-line string."""
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        text = text.replace("\n", " ").strip()
        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    def _build_worker_briefing(self, routine_spec_id: str | None) -> str:
        """
        Build compact reusable context for workers from ledger state.

        Includes proven artifacts, reusable experiment takeaways, and known blockers
        from latest failed routine attempts.
        """
        lines: list[str] = [
            "## Worker Briefing (Reusable Context)",
            "Use this as prior context. Verify assumptions against live/browser evidence.",
        ]

        # Proven artifacts (compact, capped)
        proven_lines: list[str] = []
        for fetch in self._ledger.proven.fetches[-5:]:
            method = self._truncate_text_for_briefing(fetch.get("method", "?"), 24)
            url = self._truncate_text_for_briefing(fetch.get("url", "?"), 140)
            proven_lines.append(f"- FETCH: {method} {url}")
        for nav in self._ledger.proven.navigations[-4:]:
            url = self._truncate_text_for_briefing(nav.get("url", "?"), 160)
            proven_lines.append(f"- NAV: {url}")
        for token in self._ledger.proven.tokens[-4:]:
            name = self._truncate_text_for_briefing(token.get("name", "?"), 40)
            source = self._truncate_text_for_briefing(token.get("source", "?"), 120)
            proven_lines.append(f"- TOKEN: {name} (source: {source})")
        for param in self._ledger.proven.parameters[-5:]:
            name = self._truncate_text_for_briefing(param.get("name", "?"), 40)
            ptype = self._truncate_text_for_briefing(param.get("type", "?"), 24)
            example = self._truncate_text_for_briefing(param.get("example_value", ""), 90)
            proven_lines.append(f"- PARAM: {name} ({ptype}) example={example}")
        if proven_lines:
            lines.append("\n### Proven Artifacts")
            lines.extend(proven_lines[:12])

        # Reusable takeaways from confirmed/partial experiments
        relevant_takeaways: list[tuple[ExperimentEntry, ExperimentTakeaway]] = []
        for exp in reversed(self._ledger.experiments):
            if exp.verdict not in {ExperimentVerdict.CONFIRMED, ExperimentVerdict.PARTIAL}:
                continue
            if exp.routine_spec_id is not None and routine_spec_id is not None:
                if exp.routine_spec_id != routine_spec_id:
                    continue
            elif exp.routine_spec_id is not None and routine_spec_id is None:
                # Shared experiments should not inherit routine-specific assumptions by default.
                continue

            for takeaway in exp.takeaways:
                relevant_takeaways.append((exp, takeaway))
                if len(relevant_takeaways) >= 8:
                    break
            if len(relevant_takeaways) >= 8:
                break

        if relevant_takeaways:
            lines.append("\n### Prior Experiment Takeaways")
            for exp, takeaway in relevant_takeaways:
                claim = self._truncate_text_for_briefing(takeaway.claim, 190)
                tags = ", ".join(takeaway.tags[:4]) if takeaway.tags else ""
                prefix = f"[{exp.id}]"
                if tags:
                    prefix += f" [{tags}]"
                lines.append(f"- {prefix} {claim}")
                if takeaway.how_to_apply_next:
                    how = self._truncate_text_for_briefing(takeaway.how_to_apply_next, 180)
                    lines.append(f"- apply: {how}")
                if takeaway.evidence:
                    ev = self._truncate_text_for_briefing(takeaway.evidence, 180)
                    lines.append(f"- evidence: {ev}")

        # Latest blockers for this routine spec
        if routine_spec_id:
            attempts = self._ledger.get_attempts_for_spec(routine_spec_id)
            latest_failed = next(
                (attempt for attempt in reversed(attempts) if attempt.status == RoutineAttemptStatus.FAILED),
                None,
            )
            if latest_failed and latest_failed.blocking_issues:
                lines.append("\n### Known Blockers (Latest Failed Attempt)")
                for issue in latest_failed.blocking_issues[:4]:
                    lines.append(f"- {self._truncate_text_for_briefing(issue, 220)}")

        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    def _compose_worker_methodology(self, methodology: str, routine_spec_id: str | None) -> str:
        """Compose final worker task methodology with briefing + assigned experiment."""
        briefing = self._build_worker_briefing(routine_spec_id)
        if not briefing:
            return methodology
        return f"{briefing}\n\n## Assigned Experiment Methodology\n{methodology}"

    @agent_tool(token_optimized=True)
    def _dispatch_experiment(
        self,
        hypothesis: str,
        rationale: str,
        methodology: str,
        output_description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create and dispatch an experiment to a worker.

        The worker has browser tools and capture lookup tools. Write the methodology
        so the worker knows exactly what to do — reference tools by name.

        Args:
            hypothesis: What we're testing. Specific and falsifiable.
            rationale: WHY we're testing this — evidence, reasoning, expectations.
            methodology: Instructions for the worker. Reference worker tools by name.
            output_description: Description of expected output.
        """
        # Gate: must review docs first
        if not self._docs_reviewed and self._documentation_data_loader is not None:
            return {
                "error": (
                    "You must review the routine documentation BEFORE dispatching experiments. "
                    "Call search_files(scope='docs', query='operation', mode='exact') and read_file(scope='docs', path='...') to understand all available "
                    "operation types (fetch, click, input_text, js_evaluate, get_cookies, "
                    "download, etc.). This ensures you design experiments that leverage the "
                    "full routine capabilities."
                )
            }

        worker_methodology = self._compose_worker_methodology(
            methodology=methodology,
            routine_spec_id=None,
        )
        resolved_output_description = output_description or "Structured experiment findings."

        # Create experiment entry
        experiment = ExperimentEntry(
            hypothesis=hypothesis,
            rationale=rationale,
            methodology=worker_methodology,
            routine_spec_id=None,
            status=ExperimentStatus.RUNNING,
        )
        self._ledger.add_experiment(experiment)

        # Create and dispatch task
        task = Task(
            agent_type=SpecialistAgentType.EXPERIMENT_WORKER,
            prompt=worker_methodology,
            max_loops=self._worker_max_loops,
            output_schema=self.DEFAULT_WORKER_OUTPUT_SCHEMA,
            output_description=resolved_output_description,
        )
        self._orchestration_state.add_task(task)
        experiment.task_id = task.id

        # Execute immediately
        result = self._execute_task(task)

        # Update experiment status from task
        if task.status == TaskStatus.COMPLETED:
            experiment.status = ExperimentStatus.DONE
            experiment.output = task.result
        elif task.status == TaskStatus.FAILED:
            experiment.status = ExperimentStatus.FAILED
            experiment.output = {"error": task.error}
        elif task.status == TaskStatus.PAUSED:
            experiment.status = ExperimentStatus.RUNNING

        self._persist(f"experiment_{experiment.id}")
        return {
            "experiment_id": experiment.id,
            "task_id": task.id,
            "status": experiment.status.value,
            "result": result,
        }

    @agent_tool(token_optimized=True)
    def _dispatch_experiments_batch(
        self,
        experiments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Dispatch multiple experiments IN PARALLEL to separate workers.

        Each experiment runs on its own worker with its own browser tab,
        all executing concurrently. Use this when you have independent
        experiments that don't depend on each other's results.

        Much faster than calling dispatch_experiment sequentially — N experiments
        run in roughly the time of 1.

        Args:
            experiments: List of experiment dicts, each with:
                - hypothesis: What we're testing (specific and falsifiable)
                - rationale: WHY we're testing this
                - methodology: Instructions for the worker (reference tools by name!)
                - output_description: (optional) Description of expected output
        """
        if not experiments:
            return {"error": "No experiments provided"}

        # Gate: must review docs first
        if not self._docs_reviewed and self._documentation_data_loader is not None:
            return {
                "error": (
                    "You must review the routine documentation BEFORE dispatching experiments. "
                    "Call search_files(scope='docs', query='operation', mode='exact') and read_file(scope='docs', path='...') to understand all available "
                    "operation types (fetch, click, input_text, js_evaluate, get_cookies, "
                    "download, etc.). This ensures you design experiments that leverage the "
                    "full routine capabilities."
                )
            }

        # Cap at num_workers to avoid overwhelming the system
        max_parallel = self._num_workers
        if len(experiments) > max_parallel:
            logger.warning(
                "Batch of %d experiments exceeds worker pool (%d), running first %d",
                len(experiments), max_parallel, max_parallel,
            )
            experiments = experiments[:max_parallel]

        # Phase 1: Create all experiment entries and tasks (sequential — fast, no I/O)
        task_experiment_pairs: list[tuple[Task, ExperimentEntry]] = []
        for exp_dict in experiments:
            worker_methodology = self._compose_worker_methodology(
                methodology=exp_dict.get("methodology", ""),
                routine_spec_id=None,
            )
            resolved_output_description = (
                exp_dict.get("output_description") or "Structured experiment findings."
            )

            experiment = ExperimentEntry(
                hypothesis=exp_dict.get("hypothesis", ""),
                rationale=exp_dict.get("rationale", ""),
                methodology=worker_methodology,
                routine_spec_id=None,
                status=ExperimentStatus.RUNNING,
            )
            self._ledger.add_experiment(experiment)

            task = Task(
                agent_type=SpecialistAgentType.EXPERIMENT_WORKER,
                prompt=worker_methodology,
                max_loops=self._worker_max_loops,
                output_schema=self.DEFAULT_WORKER_OUTPUT_SCHEMA,
                output_description=resolved_output_description,
            )
            self._orchestration_state.add_task(task)
            experiment.task_id = task.id
            task_experiment_pairs.append((task, experiment))

        self._persist("batch_dispatched")

        # Phase 2: Execute all tasks in parallel using ThreadPoolExecutor
        results: list[dict[str, Any]] = []

        def _run_one(pair: tuple[Task, ExperimentEntry]) -> dict[str, Any]:
            task, experiment = pair
            # Create a dedicated worker for this parallel task
            worker = self._create_worker()
            subagent = SubAgent(
                type=task.agent_type,
                llm_model=self._worker_llm_model.value,
            )
            self._orchestration_state.subagents[subagent.id] = subagent
            self._agent_instances[subagent.id] = worker
            task.agent_id = subagent.id
            subagent.task_ids.append(task.id)

            # Wire up real-time thread persistence
            agent_label = f"worker_{subagent.id}"
            worker._on_chat_added = lambda: self._dump_agent_thread(agent_label, worker)

            try:
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now()

                config = AutonomousRunConfig(
                    min_iterations=1,
                    max_iterations=task.max_loops,
                )
                result = worker.run_autonomous(
                    task=task.prompt,
                    config=config,
                    output_schema=task.output_schema,
                    output_description=task.output_description,
                )
                task.loops_used += worker.autonomous_iteration
                self._dump_agent_thread(f"worker_{subagent.id}", worker)

                if result is not None:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = datetime.now()
                    task.result = result.model_dump() if isinstance(result, BaseModel) else result
                    experiment.status = ExperimentStatus.DONE
                    experiment.output = task.result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Max loops reached without result"
                    experiment.status = ExperimentStatus.FAILED
                    experiment.output = {"error": task.error}
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                experiment.status = ExperimentStatus.FAILED
                experiment.output = {"error": str(e)}
                logger.error("Parallel task %s failed: %s", task.id, e)
            finally:
                worker.close()

            return {
                "experiment_id": experiment.id,
                "hypothesis": experiment.hypothesis[:100],
                "status": experiment.status.value,
                "result_preview": str(experiment.output)[:300] if experiment.output else None,
            }

        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {
                pool.submit(_run_one, pair): pair
                for pair in task_experiment_pairs
            }
            for future in as_completed(futures, timeout=self.WORKER_TIMEOUT_SECONDS + 30):
                pair = futures[future]
                try:
                    results.append(future.result(timeout=self.WORKER_TIMEOUT_SECONDS))
                except FuturesTimeoutError:
                    _, experiment = pair
                    logger.error(
                        "Batch experiment %s timed out after %ds",
                        experiment.id, self.WORKER_TIMEOUT_SECONDS,
                    )
                    experiment.status = ExperimentStatus.FAILED
                    experiment.output = {"error": f"Worker timed out after {self.WORKER_TIMEOUT_SECONDS}s"}
                    results.append({
                        "experiment_id": experiment.id,
                        "status": "failed",
                        "error": f"Worker timed out after {self.WORKER_TIMEOUT_SECONDS}s",
                    })
                except Exception as e:
                    logger.error("Batch experiment failed: %s", e)
                    results.append({
                        "experiment_id": pair[1].id,
                        "status": "failed",
                        "error": str(e),
                    })

        self._persist("batch_completed")

        completed = sum(1 for r in results if r.get("status") == "done")
        failed = sum(1 for r in results if r.get("status") == "failed")

        return {
            "total": len(results),
            "completed": completed,
            "failed": failed,
            "experiments": results,
        }

    def _get_remediation_docs_for_experiment(
        self, experiment: ExperimentEntry,
    ) -> str | None:
        """Scan experiment output for known error patterns and return relevant doc content."""
        if self._documentation_data_loader is None:
            return None

        # Build searchable text from output + summary
        output_text = json.dumps(experiment.output, default=str).lower() if experiment.output else ""
        summary_text = (experiment.summary or "").lower()
        searchable = output_text + " " + summary_text

        matched_paths: list[str] = []
        for keywords, doc_path in self._ERROR_DOC_PATTERNS:
            if any(kw in searchable for kw in keywords):
                matched_paths.append(doc_path)

        if not matched_paths:
            return None

        doc_sections: list[str] = []
        for doc_path in matched_paths:
            content = self._documentation_data_loader.get_file_content(doc_path)
            if content:
                doc_sections.append(f"--- {doc_path} ---\n{content}")

        if not doc_sections:
            return None

        return (
            "IMPORTANT: The following documentation covers known fixes for the "
            "errors observed in this experiment. Read carefully before deciding "
            "to give up on this routine spec.\n\n" + "\n\n".join(doc_sections)
        )

    @agent_tool(token_optimized=True)
    def _get_experiment_result(self, experiment_id: str) -> dict[str, Any]:
        """
        Read the result of a completed experiment.

        Args:
            experiment_id: ID of the experiment.
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        result: dict[str, Any] = {
            "experiment_id": experiment.id,
            "hypothesis": experiment.hypothesis,
            "routine_spec_id": experiment.routine_spec_id,
            "status": experiment.status.value,
            "verdict": experiment.verdict.value if experiment.verdict else None,
            "summary": experiment.summary,
            "takeaways": [t.model_dump(mode="json") for t in experiment.takeaways],
            "output": experiment.output,
        }

        # Auto-inject relevant common-issues docs when experiment has errors
        remediation_docs = self._get_remediation_docs_for_experiment(experiment)
        if remediation_docs:
            result["remediation_docs"] = remediation_docs

        return result

    @agent_tool()
    def _follow_up(
        self,
        experiment_id: str,
        message: str,
    ) -> dict[str, Any]:
        """
        Send a follow-up message to the SAME worker that ran an experiment.
        The worker retains its full context — no cold start.

        Use this when:
        - The result is ambiguous and you need clarification
        - You want the worker to try a variation
        - You need more detail about the findings

        Args:
            experiment_id: ID of the experiment to follow up on.
            message: Follow-up instructions for the worker.
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        if experiment.task_id is None:
            return {"error": "Experiment has no associated task"}

        task = self._orchestration_state.tasks.get(experiment.task_id)
        if task is None:
            return {"error": f"Task {experiment.task_id} not found"}

        if task.agent_id is None:
            return {"error": "No agent instance associated with this task"}

        agent = self._agent_instances.get(task.agent_id)
        if agent is None:
            return {"error": f"Agent instance {task.agent_id} no longer exists"}

        # Send follow-up via the agent's conversational interface
        agent.process_new_message(message)

        # Collect the last assistant message as the response
        last_chat = None
        for chat_id in reversed(agent._thread.chat_ids):
            chat = agent._chats.get(chat_id)
            if chat and chat.role == ChatRole.ASSISTANT:
                last_chat = chat
                break

        response_text = last_chat.content if last_chat else "(no response)"
        return {
            "experiment_id": experiment.id,
            "follow_up_response": response_text,
        }

    # ===================================================================
    # RECORDING TOOLS
    # ===================================================================

    @agent_tool()
    def _record_finding(
        self,
        experiment_id: str,
        verdict: str,
        summary: str,
        takeaways: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Record a verdict after reviewing an experiment result.

        MUST be called for every completed experiment. This builds the
        experiment log that drives your next decisions.

        Args:
            experiment_id: ID of the experiment.
            verdict: One of 'confirmed', 'refuted', 'partial', 'needs_followup'.
            summary: What we learned, in one or two sentences.
            takeaways: Optional reusable lessons for future workers.
                Each item should include:
                - claim (required): concrete fact to reuse
                - evidence (optional): supporting detail
                - how_to_apply_next (optional): instruction for later experiments
                - confidence (optional): float in [0, 1]
                - tags (optional): short labels like auth/pagination/endpoint
        """
        experiment = self._ledger.get_experiment(experiment_id)
        if experiment is None:
            return {"error": f"No experiment found with ID: {experiment_id}"}

        try:
            experiment.verdict = ExperimentVerdict(verdict)
        except ValueError:
            return {
                "error": f"Invalid verdict: {verdict}. "
                f"Must be one of: {[v.value for v in ExperimentVerdict]}"
            }

        experiment.summary = summary
        if takeaways is not None:
            parsed_takeaways: list[ExperimentTakeaway] = []
            for idx, raw_takeaway in enumerate(takeaways):
                if not isinstance(raw_takeaway, dict):
                    return {"error": f"takeaways[{idx}] must be an object"}
                claim = raw_takeaway.get("claim")
                if not isinstance(claim, str) or not claim.strip():
                    return {"error": f"takeaways[{idx}].claim is required and must be a non-empty string"}

                confidence = raw_takeaway.get("confidence")
                if confidence is not None:
                    try:
                        confidence = float(confidence)
                    except (TypeError, ValueError):
                        return {"error": f"takeaways[{idx}].confidence must be a float in [0, 1]"}
                    if confidence < 0 or confidence > 1:
                        return {"error": f"takeaways[{idx}].confidence must be between 0 and 1"}

                tags_raw = raw_takeaway.get("tags", [])
                if tags_raw is None:
                    tags_raw = []
                if not isinstance(tags_raw, list):
                    return {"error": f"takeaways[{idx}].tags must be a list of strings"}

                tags: list[str] = []
                for t in tags_raw:
                    if isinstance(t, str):
                        tag = t.strip()
                        if tag:
                            tags.append(tag)
                parsed_takeaways.append(
                    ExperimentTakeaway(
                        claim=claim.strip(),
                        evidence=raw_takeaway.get("evidence"),
                        how_to_apply_next=raw_takeaway.get("how_to_apply_next"),
                        confidence=confidence,
                        tags=tags,
                    )
                )
            experiment.takeaways = parsed_takeaways

        self._persist(f"finding_{experiment.id}")
        return {
            "experiment_id": experiment.id,
            "verdict": experiment.verdict.value,
            "summary": summary,
            "takeaway_count": len(experiment.takeaways),
        }

    @agent_tool()
    def _record_proven_artifact(
        self,
        artifact_type: str,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Add a proven artifact to the ledger. Call this when an experiment confirms
        a fetch, navigation, token, or parameter that will be part of a routine.

        IMPORTANT: The 'details' parameter MUST be a JSON object (dict), NOT a string.

        Args:
            artifact_type: One of 'fetch', 'navigation', 'token', 'parameter'.
            details: A JSON object with artifact-specific info. NOT a string.

        Example calls:
            record_proven_artifact({
                "artifact_type": "fetch",
                "details": {"url": "https://api.example.com/data", "method": "GET",
                            "headers": {}, "response_preview": "200 OK with JSON"}
            })
            record_proven_artifact({
                "artifact_type": "navigation",
                "details": {"url": "https://example.com", "sets_up": ["session_cookie"]}
            })
            record_proven_artifact({
                "artifact_type": "parameter",
                "details": {"name": "seasonId", "type": "number",
                            "description": "Season identifier", "example_value": 2025}
            })
        """
        try:
            atype = ArtifactType(artifact_type)
        except ValueError:
            return {
                "error": f"Invalid artifact_type: {artifact_type}. "
                f"Must be one of: {[t.value for t in ArtifactType]}"
            }

        proven = self._ledger.proven
        if atype == ArtifactType.FETCH:
            proven.fetches.append(details)
        elif atype == ArtifactType.NAVIGATION:
            proven.navigations.append(details)
        elif atype == ArtifactType.TOKEN:
            proven.tokens.append(details)
        elif atype == ArtifactType.PARAMETER:
            proven.parameters.append(details)

        self._persist(f"artifact_{artifact_type}")
        return {"ok": True, "artifact_type": artifact_type, "details": details}

    # ===================================================================
    # ROUTINE SUBMISSION TOOLS
    # ===================================================================

    @agent_tool(persist=ToolResultPersistMode.ALWAYS, token_optimized=True)
    def _submit_routine(
        self,
        spec_id: str,
        routine_json: dict[str, Any],
        test_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Submit a routine attempt for validation, execution, and inspection.

        Pipeline: validate → execute with test_parameters → inspect → verdict.

        IMPORTANT: You MUST provide test_parameters with realistic values for
        every parameter defined in the routine. The routine will be executed
        in a live browser and the result sent to an independent inspector.

        The routine_json MUST match the Routine schema (see system prompt).
        Key rules:
        - Use "operations" (not "steps"), each with a "type" discriminator
        - fetch operations need "endpoint": {"url": "...", "method": "GET"}
        - Last operation must be "return", "return_html", or "download"
        - Must have ≥2 operations (navigate + fetch + return is typical)
        - Use {{paramName}} placeholders in URLs/headers/body for parameters
        - fetch needs session_storage_key to save results for the return op

        Example:
            submit_routine({
                "spec_id": "abc123",
                "routine_json": {
                    "name": "search_examplesite_products",
                    "description": "Searches the ExampleSite product catalog by query string, returning product names, prices, ratings, and availability.",
                    "parameters": [
                        {"name": "search_query", "type": "string", "description": "Free-text search query to find products (e.g. 'wireless headphones', 'running shoes')"}
                    ],
                    "operations": [
                        {"type": "navigate", "url": "https://www.example.com"},
                        {"type": "fetch", "endpoint": {"url": "https://api.example.com/search?q={{search_query}}", "method": "GET"}, "session_storage_key": "result"},
                        {"type": "return", "session_storage_key": "result"}
                    ]
                },
                "test_parameters": {"search_query": "wireless headphones"}
            })

        Args:
            spec_id: Which RoutineSpec this routine fulfills.
            routine_json: The complete routine dict matching the Routine schema.
            test_parameters: Parameter values for test execution. REQUIRED —
                must include a value for every parameter in the routine.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        if test_parameters is None:
            return {
                "error": "test_parameters is required. Provide realistic values "
                "for every parameter so the routine can be executed and inspected. "
                "If the routine has no parameters, pass an empty object: {}"
            }

        # ----- Documentation quality gate (before Pydantic validation) -----
        doc_issues = self._check_routine_documentation_quality(routine_json)
        if doc_issues:
            return {
                "success": False,
                "stage": "documentation_quality",
                "issues": doc_issues,
                "hint": (
                    "These routines will be vectorized and stored in databases for other "
                    "agents to discover and use. Names and descriptions must be precise "
                    "enough for semantic search and unambiguous enough for autonomous use. "
                    "Fix the issues above and resubmit."
                ),
            }

        # ----- Site-credential parameter gate -----
        # Reject parameters that look like site-level API keys / subscription keys.
        # These should be hardcoded from captures, not exposed as user parameters.
        _CREDENTIAL_PATTERNS = {
            "api_key", "apikey", "api-key", "subscription_key", "subscriptionkey",
            "subscription-key", "client_secret", "client_id", "app_key", "appkey",
            "app_secret", "secret_key", "secretkey", "access_key", "accesskey",
        }
        params = routine_json.get("parameters", [])
        suspect_params = [
            p["name"] for p in params
            if isinstance(p, dict) and p.get("name", "").lower().replace("-", "_") in _CREDENTIAL_PATTERNS
        ]
        if suspect_params:
            return {
                "success": False,
                "stage": "credential_parameter_check",
                "suspect_parameters": suspect_params,
                "error": (
                    f"Parameter(s) {suspect_params} look like site-level API keys or "
                    "subscription keys. These are NOT user secrets — they are constants "
                    "baked into the website's JavaScript or network requests. "
                    "You MUST: (1) find the actual value from captures using "
                    "capture_get_transaction to inspect request headers, "
                    "(2) hardcode it directly in the routine's headers/body, "
                    "(3) remove it from parameters. "
                    "Only parameterize values a USER would naturally provide "
                    "(search terms, dates, IDs, locations)."
                ),
            }

        # Check attempt limit
        existing_attempts = self._ledger.get_attempts_for_spec(spec_id)
        if len(existing_attempts) >= self._max_attempts_per_routine:
            return {
                "error": f"Max attempts ({self._max_attempts_per_routine}) reached for {spec.name}. "
                "Consider mark_routine_failed if this routine can't be built."
            }

        # ----- Duplicate routine check -----
        # Hash the operations list to detect identical resubmissions.
        # This prevents wasting inspector tokens on the exact same broken routine.
        new_ops_hash = hashlib.sha256(
            json.dumps(routine_json.get("operations", []), sort_keys=True).encode()
        ).hexdigest()
        for prev_attempt in existing_attempts:
            prev_ops_hash = hashlib.sha256(
                json.dumps(prev_attempt.routine_json.get("operations", []), sort_keys=True).encode()
            ).hexdigest()
            if new_ops_hash == prev_ops_hash:
                prev_issues = prev_attempt.blocking_issues or []
                return {
                    "error": (
                        f"This routine has IDENTICAL operations to attempt #{prev_attempt.id} "
                        f"which already FAILED inspection. Resubmitting the same routine wastes "
                        f"tokens and will produce the same result. You MUST change the operations "
                        f"to address the previous blocking issues before resubmitting."
                    ),
                    "previous_blocking_issues": prev_issues,
                    "hint": (
                        "Review the blocking issues above. Common fixes: add auth/token "
                        "operations, add missing headers, fix endpoint URLs, resolve "
                        "placeholder issues. The routine must be STRUCTURALLY DIFFERENT "
                        "from previous failed attempts."
                    ),
                }

        # Step 1: Validate routine JSON against the Routine model
        try:
            routine = Routine.model_validate(routine_json)
        except ValidationError as e:
            issues: list[str] = []
            for err in e.errors():
                loc = ".".join(str(part) for part in err.get("loc", [])) or "root"
                msg = err.get("msg", "Invalid value")
                issues.append(f"{loc}: {msg}")
            return {
                "success": False,
                "stage": "validation",
                "validation_errors": issues or [str(e)],
                "issues": issues or [str(e)],
                "hint": (
                    "Routine validation failed. BEFORE retrying, use your documentation "
                    "tools to review the correct schema: call search_files(scope='docs', query='Routine operation "
                    "endpoint fetch', mode='exact') or read_file(scope='docs', path='...') to read the Routine and operation "
                    "model source code. Key reminders: each operation needs 'type' "
                    "(e.g. 'fetch'), fetch operations need 'endpoint': {'url': '...', "
                    "'method': 'GET'}, last operation must be type 'return' or "
                    "'return_html' or 'download', and the routine needs at least 2 "
                    "operations. Check the schema in your system prompt carefully."
                ),
            }
        except Exception as e:
            return {
                "success": False,
                "stage": "validation",
                "validation_errors": [str(e)],
                "issues": [str(e)],
                "hint": (
                    "Routine validation failed. BEFORE retrying, use your documentation "
                    "tools to review the correct schema and operation requirements."
                ),
            }

        # Create attempt record
        parent_id = existing_attempts[-1].id if existing_attempts else None
        attempt = RoutineAttempt(
            routine_spec_id=spec_id,
            routine_json=json.loads(routine.model_dump_json()),
            status=RoutineAttemptStatus.VALIDATING,
            test_parameters=test_parameters,
            parent_attempt_id=parent_id,
        )
        self._ledger.add_attempt(attempt)
        spec.status = RoutineSpecStatus.VALIDATING
        self._persist(f"attempt_{attempt.id}_validated")

        # Step 2: Execute the routine with test parameters
        attempt.status = RoutineAttemptStatus.EXECUTING
        self._persist(f"attempt_{attempt.id}_executing")

        execution_result = self._execute_routine_with_params(routine, test_parameters)

        if execution_result is not None:
            attempt.execution_result = execution_result.model_dump()
            if not execution_result.ok:
                attempt.execution_error = execution_result.error
                logger.warning(
                    "Routine %s execution failed: %s", spec.name, execution_result.error,
                )
        else:
            attempt.execution_error = "Execution unavailable (no browser or execution crashed)"

        self._persist(f"attempt_{attempt.id}_executed")

        # Step 3: Send to inspector for quality review
        attempt.status = RoutineAttemptStatus.INSPECTING
        self._persist(f"attempt_{attempt.id}_inspecting")

        inspection_result = self._run_inspection(routine, execution_result, spec)

        if inspection_result is not None:
            # run_autonomous returns a SpecialistResultWrapper dict with the actual
            # inspection data nested under "output". Unwrap it so downstream code
            # can read overall_pass / blocking_issues / recommendations directly.
            inner = inspection_result.get("output", inspection_result)
            attempt.inspection_result = inner
            attempt.overall_pass = inner.get("overall_pass", False)
            attempt.blocking_issues = inner.get("blocking_issues", [])
            attempt.recommendations = inner.get("recommendations", [])

            if attempt.overall_pass:
                attempt.status = RoutineAttemptStatus.PASSED
            else:
                attempt.status = RoutineAttemptStatus.FAILED
        else:
            # Inspector failed — treat as passed with warning (let PI decide)
            attempt.status = RoutineAttemptStatus.PASSED
            attempt.recommendations = ["Inspector was unavailable — manual review recommended"]

        self._persist(f"attempt_{attempt.id}_inspected")

        # Build response
        response: dict[str, Any] = {
            "success": True,
            "attempt_id": attempt.id,
            "spec_id": spec_id,
            "operations_count": len(routine.operations),
            "parameters_count": len(routine.parameters),
        }

        # Prepare execution payload (appended as the LAST response key below)
        if execution_result is not None:
            execution_payload: dict[str, Any] = execution_result.model_dump(mode="json")
        else:
            execution_payload = {"ok": False, "error": attempt.execution_error}

        # Inspection summary
        if inspection_result is not None:
            response["inspection"] = {
                "overall_pass": attempt.overall_pass,
                "overall_score": inner.get("overall_score"),
                "blocking_issues": attempt.blocking_issues,
                "recommendations": attempt.recommendations,
                "summary": inner.get("summary"),
            }
        else:
            response["inspection"] = {"overall_pass": None, "note": "Inspector unavailable"}

        response["verdict"] = attempt.status.value

        # ----- Remediation hints for failed inspections -----
        if not attempt.overall_pass and attempt.blocking_issues:
            hints: list[str] = []
            issues_text = " ".join(attempt.blocking_issues).lower()
            if "failed to fetch" in issues_text or "typeerror" in issues_text:
                hints.append(
                    "CORS FIX: Add a 'navigate' operation to the API's allowed origin "
                    "BEFORE any fetch. Routines start from about:blank (origin=null) so "
                    "all cross-origin fetches fail. Example: if API is at api.example.com "
                    "but CORS allows www.example.com, add {\"type\": \"navigate\", "
                    "\"url\": \"https://www.example.com\"} as the FIRST operation. "
                    "Review docs: search_files(scope='docs', query='cors-failed-to-fetch', mode='exact')."
                )
            if "401" in issues_text or "403" in issues_text or "unauthorized" in issues_text or "access denied" in issues_text:
                hints.append(
                    "AUTH FIX: The routine is missing authentication. Add a fetch "
                    "operation to obtain a token/key, then a js_evaluate to extract "
                    "it, then include it in subsequent fetch headers via a "
                    "sessionStorage placeholder. Review docs: "
                    "search_files(scope='docs', query='unauthenticated', mode='exact')."
                )
            if "documentation quality" in issues_text:
                hints.append(
                    "DOCS FIX: Improve routine name (verb_site_noun, 3+ segments), "
                    "description (>=8 words, explain action+inputs+outputs), and "
                    "parameter descriptions (>=3 words, explain where to get values)."
                )
            if hints:
                response["remediation_hints"] = hints

        # ----- Persist unified attempt record -----
        self._record_attempt(
            spec=spec,
            attempt=attempt,
            routine_json=routine_json,
            test_parameters=test_parameters,
            execution_result=execution_result,
            inspection_result=inspection_result,
        )

        # Keep execution as the final key for readability in tool responses.
        response["execution"] = execution_payload

        return response

    @agent_tool()
    def _mark_routine_shipped(
        self,
        spec_id: str,
        attempt_id: str,
        when_to_use: str,
        parameters_summary: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Mark a routine as shipped after it passes inspection/validation.
        Moves the spec status to "shipped".

        Args:
            spec_id: ID of the RoutineSpec.
            attempt_id: ID of the RoutineAttempt to ship.
            when_to_use: Guidance for the user on when to use this routine.
            parameters_summary: Human-readable parameter descriptions.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        attempt = self._ledger.get_attempt(attempt_id)
        if attempt is None:
            return {"error": f"No attempt found with ID: {attempt_id}"}

        spec.status = RoutineSpecStatus.SHIPPED
        spec.shipped_attempt_id = attempt_id

        self._persist(f"shipped_{spec.name}")
        return {
            "ok": True,
            "shipped": spec.name,
            "attempt_id": attempt_id,
        }

    @agent_tool()
    def _mark_routine_failed(
        self,
        spec_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """
        Give up on a specific routine. Records why it failed.

        Args:
            spec_id: ID of the RoutineSpec.
            reason: Why this routine can't be built.
        """
        spec = self._ledger.get_spec(spec_id)
        if spec is None:
            return {"error": f"No spec found with ID: {spec_id}"}

        # Reject if already failed — stop the PI from looping on the same spec
        if spec.status == RoutineSpecStatus.FAILED:
            return {
                "error": (
                    f"Routine '{spec.name}' is ALREADY marked as failed. "
                    "Do not call mark_routine_failed again. Move on to the next "
                    "unaddressed routine — call get_ledger to see which routines "
                    "still need work, then dispatch_experiment or submit_routine for those."
                )
            }

        # Reject if already shipped
        if spec.status == RoutineSpecStatus.SHIPPED:
            return {"error": f"Routine '{spec.name}' is already shipped. Cannot mark as failed."}

        # Global guardrail: require enough failed routine attempts across the pipeline
        failed_attempts_global = sum(
            1 for attempt in self._ledger.attempts
            if attempt.status == RoutineAttemptStatus.FAILED
        )
        if failed_attempts_global < self._min_global_failed_attempts_before_routine_failure:
            return {
                "error": (
                    f"Cannot mark routine '{spec.name}' as failed yet. "
                    f"Global failed routine attempts: {failed_attempts_global}/"
                    f"{self._min_global_failed_attempts_before_routine_failure} required. "
                    "Keep iterating: submit improved routine attempts, inspect failures, "
                    "run experiments to delegate exploration to workers when needed, "
                    "and only mark routines failed after enough global evidence exists."
                )
            }

        # Guardrail: require minimum experimentation before giving up
        spec_experiments = self._ledger.get_experiments_for_spec(spec_id)
        if len(spec_experiments) < 2:
            return {
                "error": (
                    f"Cannot mark routine '{spec.name}' as failed after only "
                    f"{len(spec_experiments)} experiment(s). Try at least 2 experiments "
                    "with different approaches before giving up. Consider: CDP-level "
                    "intercepts, direct navigation to API URLs, or checking the "
                    "captured session data for working request patterns."
                )
            }

        spec.status = RoutineSpecStatus.FAILED
        spec.failure_reason = reason

        self._persist(f"failed_{spec.name}")
        return {"ok": True, "failed": spec.name, "reason": reason}

    # ===================================================================
    # DASHBOARD TOOL
    # ===================================================================

    @agent_tool(token_optimized=True)
    def _get_ledger(self) -> dict[str, Any]:
        """
        Read the full Discovery Ledger — routine specs, experiments, proven
        artifacts, attempts, and unresolved questions. Use this to review
        progress and decide what to work on next.
        """
        return {
            "summary": self._ledger.to_summary(),
            "total_specs": len(self._ledger.routine_specs),
            "shipped": sum(
                1 for s in self._ledger.routine_specs
                if s.status == RoutineSpecStatus.SHIPPED
            ),
            "failed": sum(
                1 for s in self._ledger.routine_specs
                if s.status == RoutineSpecStatus.FAILED
            ),
            "total_experiments": len(self._ledger.experiments),
            "confirmed": len(self._ledger.get_confirmed_experiments()),
            "total_attempts": len(self._ledger.attempts),
            "proven_fetches": len(self._ledger.proven.fetches),
            "proven_navigations": len(self._ledger.proven.navigations),
            "proven_tokens": len(self._ledger.proven.tokens),
            "proven_parameters": len(self._ledger.proven.parameters),
            "unresolved": self._ledger.unresolved,
        }

    # ===================================================================
    # TERMINATION TOOLS
    # ===================================================================

    @agent_tool()
    def _mark_complete(self, usage_guide: str) -> dict[str, Any]:
        """
        Signal that the pipeline is done. Call this when ALL routines
        have been addressed (shipped or failed).

        Provides a usage_guide string explaining how to use the routines
        together and when to use each one. Builds the final RoutineCatalog.

        Args:
            usage_guide: How to use these routines together. Include:
                - What each routine does
                - When to use each one
                - How they relate to each other
                - What parameters each expects
        """
        # Guardrail: reject if routines are still unaddressed
        unaddressed = [
            s for s in self._ledger.routine_specs
            if s.status not in (RoutineSpecStatus.SHIPPED, RoutineSpecStatus.FAILED)
        ]
        shipped_count = sum(
            1 for s in self._ledger.routine_specs
            if s.status == RoutineSpecStatus.SHIPPED
        )
        if unaddressed:
            unaddressed_names = [f"{s.name} ({s.status.value})" for s in unaddressed]
            return {
                "error": (
                    f"Cannot mark complete — {len(unaddressed)} routine(s) are still unaddressed: "
                    f"{', '.join(unaddressed_names)}. "
                    "Each routine must be either shipped (via submit_routine → mark_routine_shipped) "
                    "or explicitly failed (via mark_routine_failed) before calling mark_complete. "
                    "You must build and submit actual routine JSON with test_parameters for each routine."
                )
            }

        # Guardrail: reject if nothing was shipped at all
        if shipped_count == 0:
            return {
                "error": (
                    "Cannot mark complete with 0 shipped routines. At least one routine "
                    "must be successfully built, submitted, and shipped. Use submit_routine "
                    "with a complete routine_json and test_parameters to create routine attempts."
                )
            }

        catalog = self._build_catalog(usage_guide)
        self._ledger.catalog = catalog
        self._pipeline_result = catalog
        self._is_done = True

        self._persist("complete")
        return {
            "status": "complete",
            "routines_shipped": len(catalog.routines),
            "routines_failed": len(catalog.failed_routines),
            "total_experiments": catalog.total_experiments,
            "total_attempts": catalog.total_attempts,
        }

    @agent_tool()
    def _mark_failed(self, reason: str) -> dict[str, Any]:
        """
        Signal that the pipeline has failed — can't build ANY routines at all.

        Args:
            reason: Why construction failed entirely.
        """
        # Guardrail: prevent premature pipeline abandonment
        total_experiments = len(self._ledger.experiments)
        unaddressed_specs = [
            s for s in self._ledger.routine_specs
            if s.status not in (RoutineSpecStatus.SHIPPED, RoutineSpecStatus.FAILED)
        ]
        if total_experiments < self._min_experiments_before_fail and unaddressed_specs:
            return {
                "error": (
                    f"Cannot mark pipeline as failed after only {total_experiments} experiment(s). "
                    f"You have {len(unaddressed_specs)} unaddressed routine(s). "
                    "Try alternative approaches: use capture_search_transactions to find working "
                    "request patterns, use browser_cdp_command for CDP-level intercepts, or "
                    "navigate directly to API URLs. Mark individual routines as failed with "
                    "mark_routine_failed if they truly can't be built, then call mark_complete."
                )
            }

        self._is_done = True
        self._pipeline_result = None
        logger.warning("PI marked pipeline as failed: %s", reason)

        self._persist("failed")
        return {"status": "failed", "reason": reason}

    # ===================================================================
    # Internal — catalog building
    # ===================================================================

    def _build_catalog(self, usage_guide: str) -> RoutineCatalog:
        """Build a RoutineCatalog from the current ledger state."""
        shipped_routines: list[ShippedRoutine] = []
        failed_routines: list[dict[str, Any]] = []

        for spec in self._ledger.routine_specs:
            if spec.status == RoutineSpecStatus.SHIPPED and spec.shipped_attempt_id:
                attempt = self._ledger.get_attempt(spec.shipped_attempt_id)
                if attempt:
                    routine_name = attempt.routine_json.get("name") or spec.name
                    routine_description = attempt.routine_json.get("description") or spec.description
                    shipped_routines.append(ShippedRoutine(
                        routine_spec_id=spec.id,
                        routine_json=attempt.routine_json,
                        name=routine_name,
                        description=routine_description,
                        when_to_use=f"Use to {routine_description.lower()}",
                        parameters_summary=[],
                        inspection_score=attempt.inspection_result.get("overall_score", 0)
                        if attempt.inspection_result else 0,
                    ))
            elif spec.status == RoutineSpecStatus.FAILED:
                failed_routines.append({
                    "name": spec.name,
                    "description": spec.description,
                    "reason": spec.failure_reason or "Unknown",
                })

        # Infer site from exploration summaries or first URL
        site = "unknown"
        for summary_text in self._exploration_summaries_raw.values():
            if "://" in summary_text:
                # Try to extract domain
                match = re.search(r'https?://([^/\s]+)', summary_text)
                if match:
                    site = match.group(1)
                    break

        return RoutineCatalog(
            site=site,
            user_task=self._task,
            routines=shipped_routines,
            usage_guide=usage_guide,
            failed_routines=failed_routines,
            total_experiments=len(self._ledger.experiments),
            total_attempts=len(self._ledger.attempts),
        )

    def _build_partial_catalog(self) -> RoutineCatalog | None:
        """Build a partial catalog from whatever has been shipped so far."""
        shipped = [
            s for s in self._ledger.routine_specs
            if s.status == RoutineSpecStatus.SHIPPED
        ]
        if not shipped:
            return None
        return self._build_catalog(
            "Pipeline hit iteration limit. These routines were completed."
        )

    # ===================================================================
    # Internal — worker management
    # ===================================================================

    def _create_worker(self) -> ExperimentWorker:
        """Create a new ExperimentWorker instance with all available context."""
        worker_workspace = (
            self._worker_workspace_factory()
            if self._worker_workspace_factory is not None
            else None
        )
        return ExperimentWorker(
            emit_message_callable=self._emit_message_callable,
            # Browser context
            remote_debugging_address=self._remote_debugging_address,
            # Capture data loaders
            network_data_loader=self._network_data_loader,
            storage_data_loader=self._storage_data_loader,
            dom_data_loader=self._dom_data_loader,
            window_property_data_loader=self._window_property_data_loader,
            # Config
            llm_model=self._worker_llm_model,
            workspace=worker_workspace,
        )

    def _create_inspector(self) -> RoutineInspector:
        """Create a new RoutineInspector instance."""
        inspector_workspace = (
            self._inspector_workspace_factory()
            if self._inspector_workspace_factory is not None
            else None
        )
        return RoutineInspector(
            emit_message_callable=self._emit_message_callable,
            llm_model=self._worker_llm_model,
            documentation_data_loader=self._documentation_data_loader,
            workspace=inspector_workspace,
        )

    def _get_or_create_agent(self, task: Task) -> AbstractAgent:
        """
        Get existing agent instance or create/reuse one for the task.

        Workers are capped at num_workers. Once the pool is full, new tasks
        are assigned round-robin to existing workers (each gets a fresh
        autonomous run but the PI can still follow_up on the same worker).
        """
        if task.agent_id and task.agent_id in self._agent_instances:
            return self._agent_instances[task.agent_id]

        # Check if we can reuse an existing worker (pool is full)
        worker_ids = [
            sid for sid, agent in self._agent_instances.items()
            if isinstance(agent, ExperimentWorker)
        ]

        if len(worker_ids) >= self._num_workers:
            # Round-robin to existing workers
            reuse_id = worker_ids[self._worker_counter % len(worker_ids)]
            self._worker_counter += 1
            task.agent_id = reuse_id
            subagent = self._orchestration_state.subagents.get(reuse_id)
            if subagent:
                subagent.task_ids.append(task.id)
            # Close old browser tab — _ensure_browser will create a fresh one
            worker = self._agent_instances[reuse_id]
            if isinstance(worker, ExperimentWorker):
                worker.close()
            return worker

        # Create new worker
        agent = self._create_worker()

        subagent = SubAgent(
            type=task.agent_type,
            llm_model=self._worker_llm_model.value,
        )
        self._orchestration_state.subagents[subagent.id] = subagent
        self._agent_instances[subagent.id] = agent

        task.agent_id = subagent.id
        subagent.task_ids.append(task.id)

        # Wire up real-time thread persistence
        agent_label = f"worker_{subagent.id}"
        agent._on_chat_added = lambda: self._dump_agent_thread(agent_label, agent)

        return agent

    def _get_or_create_inspector(self) -> RoutineInspector:
        """
        Get an existing inspector instance or create one.

        Inspectors are capped at num_inspectors. Once the pool is full,
        existing inspectors are reused round-robin (each gets a fresh
        autonomous run via reset).
        """
        inspector_ids = [
            sid for sid, agent in self._agent_instances.items()
            if isinstance(agent, RoutineInspector)
        ]

        if len(inspector_ids) < self._num_inspectors:
            # Pool not full — create a new inspector
            inspector = self._create_inspector()
            subagent = SubAgent(
                type=SpecialistAgentType.ROUTINE_INSPECTOR,
                llm_model=self._worker_llm_model.value,
            )
            self._orchestration_state.subagents[subagent.id] = subagent
            self._agent_instances[subagent.id] = inspector
            # Wire up real-time thread persistence
            inspector_label = f"inspector_{subagent.id}"
            inspector._on_chat_added = lambda: self._dump_agent_thread(inspector_label, inspector)
            return inspector

        # Pool full — reuse round-robin with fresh conversation
        reuse_id = inspector_ids[self._inspector_counter % len(inspector_ids)]
        self._inspector_counter += 1
        inspector = self._agent_instances[reuse_id]
        assert isinstance(inspector, RoutineInspector)
        inspector.reset()
        return inspector

    # ===================================================================
    # Internal — routine execution and inspection
    # ===================================================================

    def _execute_routine_with_params(
        self,
        routine: Routine,
        test_parameters: dict[str, Any] | None,
    ) -> RoutineExecutionResultWithMetadata | None:
        """Execute a routine with test parameters in a live browser."""
        if not self._remote_debugging_address:
            logger.warning("No remote_debugging_address — skipping routine execution")
            return None

        try:
            result = routine.execute(
                parameters_dict=test_parameters,
                remote_debugging_address=self._remote_debugging_address,
                timeout=120.0,
                close_tab_when_done=True,
                incognito=True,
            )
            return result
        except Exception as e:
            logger.error("Routine execution failed: %s", e)
            return None

    def _run_inspection(
        self,
        routine: Routine,
        execution_result: RoutineExecutionResultWithMetadata | None,
        spec: RoutineSpec,
    ) -> dict[str, Any] | None:
        """Run a RoutineInspector on a routine + execution result."""
        inspector = self._get_or_create_inspector()

        # Build inspection prompt with all context
        # NOTE: User task is intentionally excluded — the inspector should judge
        # the routine on its own merits (correctness, robustness, data quality),
        # not whether it fulfills the user's high-level goal.
        prompt_parts: list[str] = [
            f"## Routine Name\n{routine.name}\n",
            f"## Routine Description\n{routine.description}\n",
            f"## Routine JSON\n```json\n{json.dumps(routine.model_dump(), indent=2, default=str)}\n```\n",
        ]

        if execution_result is not None:
            exec_data = execution_result.model_dump(mode="json")
            exec_json = json.dumps(exec_data, indent=2, default=str)

            persisted_for_inspector = False
            if (
                len(exec_json) > self.INSPECTOR_INLINE_EXECUTION_MAX_CHARS
                and inspector.has_workspace
            ):
                try:
                    inspector_workspace = inspector._require_workspace()
                    inspector_workspace.ensure_dirs()
                    safe_spec = re.sub(r"[^a-zA-Z0-9_.-]+", "_", spec.name).strip("_")
                    if not safe_spec:
                        safe_spec = spec.id
                    artifact_ref = inspector_workspace.save_artifact(
                        source="raw",
                        filename=f"{safe_spec}_execution_result.json",
                        content=exec_json,
                        tool_name="pi_run_inspection",
                        content_type="json",
                        metadata={
                            "spec_id": spec.id,
                            "spec_name": spec.name,
                            "char_count": len(exec_json),
                        },
                    )
                    prompt_parts.append(
                        "## Execution Result\n"
                        f"Execution payload is large ({len(exec_json)} chars) and was saved to:\n"
                        f"- workspace path: `{artifact_ref.relative_path}`\n"
                        f"- artifact_id: `{artifact_ref.artifact_id}`\n\n"
                        "Use `execute_python` or `read_file(scope=\"workspace\", path=\"...\")` to inspect "
                        "this file directly and base your judgment on the full payload.\n"
                        "Do not claim truncation — the full execution result is available in workspace raw/.\n"
                    )
                    persisted_for_inspector = True
                except Exception as e:
                    logger.warning(
                        "Failed to persist large execution payload for inspector; falling back to inline JSON: %s",
                        e,
                    )

            if not persisted_for_inspector:
                prompt_parts.append(
                    f"## Execution Result\n```json\n{exec_json}\n```\n"
                )
        else:
            prompt_parts.append("## Execution Result\nNot available (no browser or execution failed).\n")

        base_prompt = "\n".join(prompt_parts)

        # Add exploration summaries for cross-reference. If the combined prompt gets too
        # large, drop exploration summaries first. Do NOT truncate execution payload.
        exploration_section = ""
        if self._exploration_summaries:
            exploration_parts: list[str] = ["## Exploration Summaries\n"]
            for domain, summary in self._exploration_summaries.items():
                exploration_parts.append(f"### {domain}\n{summary}\n")
            exploration_section = "\n".join(exploration_parts)

        inspection_prompt = (
            f"{base_prompt}\n\n{exploration_section}"
            if exploration_section
            else base_prompt
        )

        max_chars = 120_000
        if len(inspection_prompt) > max_chars and exploration_section:
            logger.warning(
                "Inspection prompt too large (%d chars); omitting exploration summaries to preserve full execution payload",
                len(inspection_prompt),
            )
            inspection_prompt = (
                f"{base_prompt}\n\n"
                "## Exploration Summaries\n"
                "[omitted due prompt size; consult persisted exploration artifacts if needed]\n"
            )

        if len(inspection_prompt) > max_chars:
            logger.warning(
                "Inspection prompt still large after omitting summaries (%d chars); sending full routine + execution payload without truncation",
                len(inspection_prompt),
            )

        try:
            config = AutonomousRunConfig(min_iterations=1, max_iterations=10)

            # Run inspector with timeout to prevent indefinite hangs
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    inspector.run_autonomous,
                    task=inspection_prompt,
                    config=config,
                    output_schema=RoutineInspectionResult.model_json_schema(),
                    output_description="RoutineInspectionResult with scores, blocking issues, and verdict",
                )
                try:
                    result = future.result(timeout=self.WORKER_TIMEOUT_SECONDS)
                except FuturesTimeoutError:
                    logger.error(
                        "Inspector timed out for %s after %ds", spec.name, self.WORKER_TIMEOUT_SECONDS,
                    )
                    self._dump_agent_thread(f"inspector_{spec.name}", inspector)
                    return None

            # Dump inspector thread
            self._dump_agent_thread(f"inspector_{spec.name}", inspector)

            if result is not None:
                return result.model_dump() if isinstance(result, BaseModel) else result
            return None
        except Exception as e:
            logger.error("Inspection failed for %s: %s", spec.name, e)
            return None

    def _execute_task(self, task: Task) -> dict[str, Any]:
        """Execute a task using an ExperimentWorker with a timeout guard."""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()

        try:
            agent = self._get_or_create_agent(task)

            remaining_loops = task.max_loops - task.loops_used
            if remaining_loops <= 0:
                task.status = TaskStatus.FAILED
                task.error = "No loops remaining"
                return {"success": False, "error": "No loops remaining"}

            config = AutonomousRunConfig(
                min_iterations=1,
                max_iterations=remaining_loops,
            )

            # Run with timeout to prevent indefinite hangs (LLM or browser)
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    agent.run_autonomous,
                    task=task.prompt,
                    config=config,
                    output_schema=task.output_schema,
                    output_description=task.output_description,
                )
                try:
                    result = future.result(timeout=self.WORKER_TIMEOUT_SECONDS)
                except FuturesTimeoutError:
                    logger.error(
                        "Task %s timed out after %ds", task.id, self.WORKER_TIMEOUT_SECONDS,
                    )
                    task.status = TaskStatus.FAILED
                    task.error = f"Worker timed out after {self.WORKER_TIMEOUT_SECONDS}s"
                    task.completed_at = datetime.now()
                    self._dump_agent_thread(f"worker_{task.agent_id}", agent)
                    return {"success": False, "error": task.error}

            task.loops_used += agent.autonomous_iteration

            # Dump the worker's full message history for debugging
            self._dump_agent_thread(f"worker_{task.agent_id}", agent)

            if result is not None:
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                task.result = result.model_dump() if isinstance(result, BaseModel) else result
                return {"success": True, "result": task.result}
            else:
                if task.loops_used < task.max_loops:
                    task.status = TaskStatus.PAUSED
                    return {"success": False, "status": "paused", "loops_used": task.loops_used}
                else:
                    task.status = TaskStatus.FAILED
                    task.error = "Max loops reached without result"
                    return {"success": False, "error": task.error}

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            logger.error("Task %s failed: %s", task.id, e)
            return {"success": False, "error": str(e)}

    # ===================================================================
    # Internal — documentation quality checks
    # ===================================================================

    @staticmethod
    def _check_routine_documentation_quality(routine_json: dict[str, Any]) -> list[str]:
        """
        Validate that routine metadata is detailed enough for vectorized storage
        and discovery by other agents. Returns a list of issues (empty = pass).
        """
        issues: list[str] = []

        # --- Routine name ---
        name = routine_json.get("name", "")
        if not name:
            issues.append("Routine name is missing.")
        else:
            # Must be snake_case (lowercase + underscores)
            if not re.match(r'^[a-z][a-z0-9]*(_[a-z0-9]+)*$', name):
                issues.append(
                    f"Routine name '{name}' must be snake_case (e.g. 'get_premierleague_standings', "
                    "'search_amtrak_trains'). No camelCase, no spaces, no uppercase."
                )
            # Must be descriptive — at least 3 underscore-separated segments
            # (verb + site/context + noun, e.g. get_premierleague_standings)
            segments = name.split("_")
            if len(segments) < 3:
                issues.append(
                    f"Routine name '{name}' needs more context ({len(segments)} segments, need ≥3). "
                    "The name must include the site/service so it makes sense in isolation. "
                    "Pattern: verb_site_noun (e.g. 'get_premierleague_standings', "
                    "'search_espn_scores', 'fetch_amtrak_schedules'). "
                    "Another agent reading ONLY the name should know what site this targets."
                )
            # Reject overly generic names that lack site context
            _GENERIC_NOUNS = {
                "data", "items", "item", "content", "results", "result",
                "info", "details", "list", "response", "output", "records",
            }
            # Check if the non-verb segments are all generic
            non_verb_segments = segments[1:] if len(segments) > 1 else []
            if non_verb_segments and all(seg in _GENERIC_NOUNS for seg in non_verb_segments):
                issues.append(
                    f"Routine name '{name}' uses only generic nouns ({non_verb_segments}). "
                    "Include the site/domain name and a specific noun. "
                    "Example: 'get_content_item' → 'get_premierleague_article', "
                    "'fetch_data' → 'fetch_espn_game_scores'."
                )

        # --- Routine description ---
        desc = routine_json.get("description", "")
        if not desc:
            issues.append("Routine description is missing.")
        else:
            word_count = len(desc.split())
            if word_count < 8:
                issues.append(
                    f"Routine description is too short ({word_count} words). Must be ≥8 words. "
                    "Describe: what the routine does, what inputs it takes, and what data it returns. "
                    "Example: 'Fetches Premier League standings for a given competition and season, "
                    "returning team names, positions, wins, draws, losses, and points.'"
                )
            # Should mention what it returns
            desc_lower = desc.lower()
            return_keywords = ("return", "fetch", "retriev", "get", "extract", "download", "output", "produc", "yield")
            if not any(kw in desc_lower for kw in return_keywords):
                issues.append(
                    "Routine description should explain what data it returns. "
                    "Include words like 'returns', 'fetches', 'retrieves', or 'extracts'."
                )

        # --- Parameter descriptions ---
        # Suffixes/keywords that signal an opaque, non-obvious parameter value
        _OPAQUE_SIGNALS = ("_id", "_ids", "_slug", "_code", "_token", "_key", "_hash", "_uuid")
        _SOURCE_KEYWORDS = (
            "obtain", "from the", "get from", "found in", "returned by",
            "use the", "listed by", "provided by", "available via", "see the",
            "look up", "call the", "via the", "endpoint", "routine",
        )

        params = routine_json.get("parameters", [])
        for param in params:
            if not isinstance(param, dict):
                continue
            pname = param.get("name", "unknown")
            pdesc = param.get("description", "")
            if not pdesc:
                issues.append(f"Parameter '{pname}' is missing a description.")
                continue

            if len(pdesc.split()) < 3:
                issues.append(
                    f"Parameter '{pname}' description is too terse: '{pdesc}'. "
                    "Descriptions must be ≥3 words and explain what the value represents "
                    "and its expected format (e.g. 'The unique season identifier, typically a 4-digit year like 2024')."
                )
                continue

            # Check if this looks like an opaque/non-obvious parameter
            pname_lower = pname.lower()
            ptype = param.get("type", "string")
            is_opaque = (
                any(pname_lower.endswith(sig) for sig in _OPAQUE_SIGNALS)
                or ptype in ("integer", "number") and pname_lower.endswith("id")
            )

            if is_opaque:
                pdesc_lower = pdesc.lower()
                has_source = any(kw in pdesc_lower for kw in _SOURCE_KEYWORDS)
                if not has_source:
                    issues.append(
                        f"Parameter '{pname}' looks like an opaque/internal identifier but its "
                        f"description doesn't explain WHERE to get valid values. "
                        f"Current description: '{pdesc}'. "
                        "For non-obvious IDs, slugs, and codes, the description MUST say how "
                        "to obtain valid values — e.g. 'Obtain from the get_competitions routine' "
                        "or 'Found in the /api/seasons endpoint response'."
                    )

        return issues

    def close(self) -> None:
        """Clean up all worker agent instances."""
        for agent in self._agent_instances.values():
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception:
                    pass
        self._agent_instances.clear()
