"""
bluebox/agents/routine_inspector.py

RoutineInspector — independent quality gate for constructed routines.

The inspector receives ALL context in the task prompt and returns a structured
RoutineInspectionResult. It has no knowledge of the discovery process — it
judges the OUTPUT, not the PROCESS. When equipped with documentation tools,
it can search common-issues docs to provide specific remediation advice.

Think of it as a peer reviewer: reads the routine cold, checks if the claims
hold up, and decides: publish, revise, or reject.
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable, TYPE_CHECKING

from bluebox.agents.abstract_agent import AgentCard
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


# ---------------------------------------------------------------------------
# Output schema — derived from RoutineInspectionResult
# ---------------------------------------------------------------------------

INSPECTION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_pass": {
            "type": "boolean",
            "description": "Whether the routine should ship (true) or needs fixes (false).",
        },
        "overall_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Sum of all 6 dimension scores, scaled to 0-100. Formula: round(sum / 60 * 100).",
        },
        "dimensions": {
            "type": "object",
            "properties": {
                "task_completion": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "data_quality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "parameter_coverage": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "routine_robustness": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "structural_correctness": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
                "documentation_quality": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer", "minimum": 0, "maximum": 10},
                        "reasoning": {"type": "string"},
                    },
                    "required": ["score", "reasoning"],
                },
            },
            "required": [
                "task_completion",
                "data_quality",
                "parameter_coverage",
                "routine_robustness",
                "structural_correctness",
                "documentation_quality",
            ],
        },
        "blocking_issues": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Issues that MUST be fixed before shipping.",
        },
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Issues that SHOULD be fixed but are non-blocking.",
        },
        "summary": {
            "type": "string",
            "description": "2-3 sentence overall assessment.",
        },
    },
    "required": [
        "overall_pass",
        "overall_score",
        "dimensions",
        "blocking_issues",
        "recommendations",
        "summary",
    ],
}


class RoutineInspector(AbstractSpecialist):
    """
    Independent quality gate for constructed routines.

    Receives routine + execution result + exploration context as the task prompt,
    scores on 6 dimensions, and returns a RoutineInspectionResult via
    finalize_with_output. Has optional access to documentation tools to provide
    specific, actionable remediation advice in recommendations.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Independent quality gate that judges constructed routines on 6 dimensions: "
            "task completion, data quality, parameter coverage, routine robustness, "
            "structural correctness, and documentation quality. Can reference routine "
            "documentation to provide actionable fix recommendations."
        ),
    )

    SYSTEM_PROMPT: str = dedent("""\
        You are a routine quality inspector. You judge routines objectively.

        You have NO knowledge of how the routine was built. You only see:
        - The user's task
        - The routine JSON
        - The execution result (if available)
        - Exploration summaries (what the site looks like)

        Your job: score the routine and decide if it ships.
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are an independent routine quality inspector. You receive a routine
        and must judge whether it correctly accomplishes its own stated purpose
        (name + description). Do NOT judge it against any broader project goal —
        only against what the routine itself claims to do.

        ## CRITICAL: Judge ACTUAL Results, Not Hypotheticals

        You score based on WHAT ACTUALLY HAPPENED, not what "would work if...".
        If the execution returned a 401, the routine FAILED. Period. You do not
        get to say "it would return rich data with valid credentials" — that is
        speculation, not inspection. A routine that doesn't work doesn't ship.

        **Automatic failure signals (ANY of these → task_completion ≤ 2, data_quality ≤ 2):**
        - HTTP 4xx or 5xx status codes in ANY operation response
        - Unresolved placeholders (e.g. "Could not resolve placeholder: ...")
        - Error messages in the response body (e.g. "Access denied", "Unauthorized",
          "Invalid", "Forbidden", "Not found")
        - Test parameters containing obvious placeholder values like "REPLACE_WITH_...",
          "YOUR_..._HERE", "TODO", "FIXME" — this means the routine can't be tested
        - Empty or null response data when the routine promises to return something
        - The execution_result.data containing an error object instead of real data

        **You are a quality gate, not a cheerleader.** Your job is to BLOCK bad routines
        from shipping. If you let a broken routine through, it pollutes the database
        and wastes other agents' time. When in doubt, FAIL it.

        ## CRITICAL: Spec Description Downgrade Detection

        When the inspection prompt includes a "Spec vs Routine Description Comparison"
        section, you MUST check whether the routine's own description has been watered
        down from the original spec. If the spec promises rich, detailed data but the
        routine description claims to return only minimal fields, this is a BLOCKING issue:

        - Add blocking issue: "Routine description is significantly weaker than the spec
          description. Spec promises: '<spec desc>'. Routine claims: '<routine desc>'.
          The routine must deliver on the original spec or the spec should be updated."
        - Cap task_completion at 4 — the routine may work for what it claims, but it
          does NOT fulfill the originally planned capability.
        - Cap data_quality at 4 — returning 2 fields when 15 were promised is not
          quality data.

        ## Scoring Rubric (6 dimensions, 0-10 each)

        1. **Task Completion** — Does the returned data ACTUALLY accomplish what
           the routine's name and description promise? Check the REAL execution result.
           - Did the routine return the data it claims to return? Not "could it" — DID IT?
           - A flight search that returned a 401 error did NOT return flights → score 0-2
           - A standings routine that returned an HTML error page did NOT return standings → score 0-2
           - ONLY score above 5 if the execution result contains ACTUAL meaningful data
             that matches what the routine promises

        2. **Data Quality** — Is the ACTUAL response complete and meaningful?
           - Check the REAL response data, not what you imagine it could contain
           - A 401/403/500 response has ZERO data quality regardless of how "correct"
             the request structure looks → score 0-2
           - An error message body is not "data" → score 0
           - Truncated, empty, or missing data → score 0-3
           - ONLY score above 5 if the response contains REAL, COMPLETE, MEANINGFUL data

        3. **Parameter Coverage** — Are the right values parameterized? Any hardcoded
           values that should be params (dates, search terms, IDs)? Any unnecessary
           params that could be hardcoded?

        4. **Routine Robustness** — Would this work in a fresh session? Are dynamic
           tokens properly resolved via placeholders (not hardcoded expired values)?
           Does it handle auth correctly (navigate first to establish cookies/tokens
           before making API calls)?
           - If any placeholder failed to resolve → score ≤ 4
           - If auth tokens are not properly obtained → score ≤ 3

        5. **Structural Correctness** — Navigate before fetch? Dependencies before
           dependents? Consistent session_storage_key usage (write before read)?
           Valid placeholder types? Operations in correct order?

        6. **Documentation Quality** — CRITICAL: These routines will be vectorized and
           stored in databases for other agents to discover via semantic search.
           Score strictly:

           **Routine name** (0-3 points):
           - Must be snake_case with verb_site_noun pattern, ≥3 segments
           - MUST include the site/service name so the name makes sense in isolation
             to an agent that has never seen this routine before
           - GOOD: get_premierleague_standings, search_amtrak_trains, fetch_espn_scores
           - BAD: get_standings (from where?), get_content_item (what content? what site?),
             fetch_data (completely generic), search_matches (which sport? which site?)
           - 0 = missing/generic/no site context, 1 = has site but vague noun,
             2 = decent with site + noun, 3 = precise verb_site_noun with clear specificity

           **Routine description** (0-4 points):
           - Must be ≥8 words
           - Must explain: (a) what it does, (b) what inputs it takes, (c) what data it returns
           - Example of 4/4: "Fetches Premier League standings for a given competition ID
             and season ID, returning team names, positions, points, and goal difference."
           - 0 = missing/useless, 1 = says what it does only, 2 = adds inputs, 3 = adds outputs, 4 = complete

           **Parameter descriptions** (0-3 points):
           - Every parameter must have a description of ≥3 words
           - Should explain what the value represents AND its expected format/range
           - CRITICAL for non-obvious parameters (opaque IDs, slugs, codes, UUIDs):
             The description MUST explain WHERE to get the value. If the user can't
             google it, the description must say how to obtain it — e.g. which other
             routine or API endpoint provides valid values.
           - Example of 3/3: "Internal competition ID. Obtain from the get_competitions
             routine or the /competitions endpoint. Example: 1 = Premier League."
           - Example of 2/3: "The unique competition identifier (e.g. 1 for Premier League)"
             (good but doesn't say where to get other valid IDs)
           - Example of 0/3: "ID" or "the season"
           - 0 = missing descriptions, 1 = all present but terse, 2 = mostly good, 3 = all
             excellent with sourcing info for non-obvious params

           A score ≤4 in documentation_quality is a BLOCKING issue — the routine cannot
           ship with poor metadata because it will be invisible to other agents.

        ## Verdict Rules

        - overall_pass = True if: no blocking_issues AND overall_score >= 60
        - overall_score = round(sum of all 6 dimension scores / 60 × 100) (max 100)
        - documentation_quality ≤ 4 → add blocking issue: "Documentation quality too low
          for vectorized storage — fix routine name, description, or parameter descriptions"
        - ANY HTTP 4xx/5xx in execution → add blocking issue describing the failure
        - ANY unresolved placeholder → add blocking issue describing which placeholder failed
        - Be STRICT on all dimensions. A broken routine is WORSE than no routine — it
          wastes database space and misleads other agents. Only pass routines that
          ACTUALLY WORK with REAL DATA in the execution result.

        ## Documentation-Backed Recommendations

        When you have access to documentation tools (search_docs, get_doc_file),
        use them to provide SPECIFIC, actionable remediation advice in your
        recommendations. Don't just say "fix the auth" — search for the relevant
        doc and cite the exact fix pattern.

        Common patterns to search for:
        - "TypeError: Failed to fetch" → search_docs("cors-failed-to-fetch") →
          the fix is adding a navigate operation to the allowed origin
        - 401/403 errors → search_docs("unauthenticated") → the fix is adding
          auth token fetch + js_evaluate extraction before data fetches
        - Placeholder issues → search_docs("placeholder-not-resolved") →
          check placeholder syntax and resolution types
        - HTML instead of JSON → search_docs("fetch-returns-html") → wrong URL
          or CORS redirect

        Your recommendations should include: (1) what's wrong, (2) the specific
        fix from documentation with example operations if applicable.

        IMPORTANT: Only search docs when you identify a blocking issue that has
        a known fix pattern. Do NOT search docs for every inspection — only when
        you can provide actionable remediation. Keep doc searches to 1-2 max per
        inspection to stay within iteration limits.

        ## Process

        1. Read the routine name and description — this is what you're scoring against
        2. Read the routine JSON — understand each operation's purpose
        3. Read the execution result — **DID IT ACTUALLY WORK?** Check EVERY operation's
           HTTP status code. Check for unresolved placeholders. Check for error messages.
           This is the MOST IMPORTANT step. If the execution failed, the routine fails.
        4. Cross-reference with exploration summaries — does the data match?
        5. Score each dimension with specific reasoning based on ACTUAL results
        6. List blocking issues (MUST fix) and recommendations (SHOULD fix)
           - If docs are available and you identified a fixable issue, search the
             common-issues docs to include a specific fix in recommendations
        7. Write a 2-3 sentence summary
        8. Call finalize_with_output with the complete inspection result
    """)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.AUTONOMOUS,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
    ) -> None:
        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=documentation_data_loader,
        )
        logger.debug("RoutineInspector initialized")

    # -----------------------------------------------------------------------
    # Abstract method implementations
    # -----------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPT

    def _get_autonomous_system_prompt(self) -> str:
        return (
            self.AUTONOMOUS_SYSTEM_PROMPT
            + self._get_output_schema_prompt_section()
            + self._get_documentation_prompt_section()
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"INSPECTION REQUEST:\n\n{task}\n\n"
            "Score this routine on all 6 dimensions (including documentation_quality), "
            "identify blocking issues vs. recommendations, and call finalize_with_output "
            "with the complete RoutineInspectionResult.\n\n"
            "CRITICAL REMINDERS:\n"
            "1. CHECK THE EXECUTION RESULT FIRST. If ANY operation returned HTTP 4xx/5xx, "
            "the routine FAILED. Score task_completion and data_quality ≤ 2. Do NOT "
            "speculate about what 'would work' — judge what ACTUALLY happened.\n"
            "2. Check for unresolved placeholders in warnings — these are automatic failures.\n"
            "3. Check test_parameters for placeholder values like 'REPLACE_WITH_...' — "
            "if the routine wasn't tested with real inputs, it cannot pass.\n"
            "4. Documentation quality: score name, description, and parameter descriptions "
            "strictly. documentation_quality ≤ 4 is a blocking issue."
        )
