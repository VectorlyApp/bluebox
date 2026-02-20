"""
bluebox/agents/routine_inspector.py

RoutineInspector — independent quality gate for constructed routines.

The inspector is a zero-tool specialist that receives ALL context in the task
prompt and returns a structured RoutineInspectionResult. It has no knowledge
of the discovery process — it judges the OUTPUT, not the PROCESS.

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
            "description": "Sum of dimension scores × 2 (0-100 scale).",
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
            },
            "required": [
                "task_completion",
                "data_quality",
                "parameter_coverage",
                "routine_robustness",
                "structural_correctness",
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

    Zero domain tools — pure judgment. Receives routine + execution result +
    exploration context as the task prompt, scores on 5 dimensions, and returns
    a RoutineInspectionResult via finalize_with_output.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Independent quality gate that judges constructed routines on 5 dimensions: "
            "task completion, data quality, parameter coverage, routine robustness, "
            "and structural correctness. Zero tools — pure judgment."
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
        and must judge whether it correctly accomplishes the user's task.

        ## Scoring Rubric (5 dimensions, 0-10 each)

        1. **Task Completion** — Does the returned data actually accomplish the user's
           task? Not "did it 200 OK" but "does this contain the right information?"
           A flight search routine should return flights. A standings routine should
           return team standings with scores/points.

        2. **Data Quality** — Is the response complete and meaningful? Not an error
           page wrapped in 200, not truncated, not missing critical fields? Check
           against what the exploration summaries describe the site as having.

        3. **Parameter Coverage** — Are the right values parameterized? Any hardcoded
           values that should be params (dates, search terms, IDs)? Any unnecessary
           params that could be hardcoded?

        4. **Routine Robustness** — Would this work in a fresh session? Are dynamic
           tokens properly resolved via placeholders (not hardcoded expired values)?
           Does it handle auth correctly (navigate first to establish cookies/tokens
           before making API calls)?

        5. **Structural Correctness** — Navigate before fetch? Dependencies before
           dependents? Consistent session_storage_key usage (write before read)?
           Valid placeholder types? Operations in correct order?

        ## Verdict Rules

        - overall_pass = True if: no blocking_issues AND overall_score >= 60
        - overall_score = sum of all 5 dimension scores × 2 (max 100)
        - Be LENIENT on first pass — a routine that mostly works is better than
          infinite retry loops. Flag clear bugs as blocking, minor issues as recommendations.

        ## Process

        1. Read the user task
        2. Read the routine JSON — understand each operation's purpose
        3. Read the execution result (if available) — did it actually work?
        4. Cross-reference with exploration summaries — does the data match?
        5. Score each dimension with specific reasoning
        6. List blocking issues (MUST fix) and recommendations (SHOULD fix)
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
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"INSPECTION REQUEST:\n\n{task}\n\n"
            "Score this routine on all 5 dimensions, identify blocking issues vs. "
            "recommendations, and call finalize_with_output with the complete "
            "RoutineInspectionResult."
        )
