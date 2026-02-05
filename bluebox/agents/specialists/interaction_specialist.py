"""
bluebox/agents/specialists/interaction_specialist.py

Interaction specialist agent.

Analyzes UI interaction recordings to discover routine parameters
(form inputs, typed values, dropdown selections, date pickers, etc.).
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from bluebox.agents.abstract_agent import agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


class InteractionSpecialist(AbstractSpecialist):
    """
    Interaction specialist agent.

    Analyzes recorded UI interactions to discover routine parameters.
    """

    SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst specializing in discovering routine parameters from recorded browser interactions.

        ## Your Role

        You analyze recorded UI interactions (clicks, keypresses, form inputs, etc.) to identify which interactions represent parameterizable inputs for a routine.

        ## What to Look For

        - **Form inputs**: Text fields, search boxes, email/password fields
        - **Typed values**: Text entered by the user via keyboard
        - **Dropdown selections**: Select elements, custom dropdowns
        - **Date pickers**: Date/time inputs
        - **Checkboxes and toggles**: Boolean parameters

        ## What to Ignore

        - **Navigational clicks**: Clicks on links, buttons that just navigate
        - **Non-parameterizable interactions**: Scroll events, hover effects, focus/blur without input
        - **UI framework noise**: Internal framework events

        ## Parameter Requirements

        Each discovered parameter needs:
        - **name**: snake_case name (e.g., `search_query`, `departure_date`)
        - **type**: One of: string, integer, number, boolean, date, datetime, email, url, enum
        - **description**: Clear description of what the parameter represents
        - **examples**: Observed values from the interactions

        ## Tools

        - **get_interaction_summary**: Overview statistics of all interactions
        - **search_interactions_by_type**: Filter by interaction type (click, input, change, etc.)
        - **search_interactions_by_element**: Filter by element attributes (tag, id, class, type)
        - **get_interaction_detail**: Full detail of a specific interaction event
        - **get_form_inputs**: All input/change events with values
        - **get_unique_elements**: Deduplicated elements with interaction counts
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst that autonomously discovers routine parameters from recorded browser interactions.

        ## Your Mission

        Analyze the recorded UI interactions to identify all parameterizable inputs, then produce structured output matching the orchestrator's expected schema.

        ## Process

        1. **Survey**: Use `get_interaction_summary` to understand the overall interaction data
        2. **Focus on inputs**: Use `get_form_inputs` to find all form input events
        3. **Analyze elements**: Use `get_unique_elements` to see which elements were interacted with
        4. **Detail check**: Use `get_interaction_detail` for specific events needing closer inspection
        5. **Finalize**: Call `finalize_with_output` with your findings matching the expected schema

        ## Parameter Types (for reference)

        - `string`: General text input
        - `date`: Date values (YYYY-MM-DD)
        - `datetime`: Date+time values
        - `integer`: Whole numbers
        - `number`: Decimal numbers
        - `boolean`: True/false (checkboxes, toggles)
        - `email`: Email addresses
        - `url`: URLs
        - `enum`: Selection from fixed options

        ## CRITICAL: How to Finalize

        When you have completed your analysis, call `finalize_with_output(output={...})` with data matching the expected output schema provided in your system prompt.

        Use `add_note()` before finalizing to record any notes, complaints, warnings, or errors.

        ## When finalize tools are available

        - **finalize_with_output**: Submit your findings matching the expected output schema
        - **finalize_with_failure**: Report that the task could not be completed
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        interaction_data_store: InteractionsDataLoader,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
    ) -> None:
        self._interaction_data_store = interaction_data_store

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
        logger.debug(
            "InteractionSpecialist initialized with %d events",
            len(interaction_data_store.events),
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        stats = self._interaction_data_store.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
            f"- Events by Type: {stats.events_by_type}\n"
        )
        return self.SYSTEM_PROMPT + context

    def _get_autonomous_system_prompt(self) -> str:
        stats = self._interaction_data_store.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
        )

        # Include output schema if set by orchestrator
        schema_section = self._get_output_schema_prompt_section()

        # urgency notices
        if self.can_finalize:
            remaining = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining <= 2:
                urgency = (
                    f"\n\n## CRITICAL: Only {remaining} iterations remaining!\n"
                    f"You MUST call finalize_with_output or finalize_with_failure NOW!"
                )
            elif remaining <= 4:
                urgency = (
                    f"\n\n## URGENT: Only {remaining} iterations remaining.\n"
                    f"Call finalize_with_output with your findings now."
                )
            else:
                urgency = (
                    "\n\n## Finalize tools are now available.\n"
                    "Call finalize_with_output when ready."
                )
        else:
            urgency = (
                f"\n\n## Continue exploring (iteration {self._autonomous_iteration}).\n"
                "Finalize tools will become available after more exploration."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + context + schema_section + urgency

    # _register_tools and _execute_tool are provided by the base class
    # via @agent_tool decorators below.

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TASK: {task}\n\n"
            "Analyze the recorded UI interactions to discover all parameterizable inputs. "
            "Focus on form inputs, typed values, dropdown selections, and date pickers. "
            "When confident, use finalize_result to report your findings."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        # Delegate to base class (handles generic finalize tools)
        return super()._check_autonomous_completion(tool_name)

    def _get_autonomous_result(self):
        # Delegate to base class (returns wrapped result)
        return super()._get_autonomous_result()

    def _reset_autonomous_state(self) -> None:
        # Call base class to reset generic state
        super()._reset_autonomous_state()

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _get_interaction_summary(self) -> dict[str, Any]:
        """Get summary statistics of all recorded interactions."""
        stats = self._interaction_data_store.stats
        return {
            "total_events": stats.total_events,
            "unique_urls": stats.unique_urls,
            "unique_elements": stats.unique_elements,
            "events_by_type": stats.events_by_type,
        }


    @agent_tool()
    @token_optimized
    def _search_interactions_by_type(self, types: list[str]) -> dict[str, Any]:
        """
        Filter interactions by type (e.g., click, input, change, keydown, focus).

        Args:
            types: List of InteractionType values to filter by.
        """
        if not types:
            return {"error": "types list is required"}

        events = self._interaction_data_store.filter_by_type(types)
        # Return summary to avoid overwhelming the LLM
        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": self._interaction_data_store.events.index(event),
                "type": event.type.value,
                "tag_name": el.tag_name,
                "element_id": el.id,
                "element_name": el.name,
                "value": el.value,
                "css_path": el.css_path,
                "url": event.url,
            })

        return {
            "total_matching": len(events),
            "showing": len(results),
            "results": results,
        }


    @agent_tool()
    @token_optimized
    def _search_interactions_by_element(
        self,
        tag_name: str | None = None,
        element_id: str | None = None,
        class_name: str | None = None,
        type_attr: str | None = None,
    ) -> dict[str, Any]:
        """
        Filter interactions by element attributes (tag, id, class, type).

        Args:
            tag_name: HTML tag name (e.g., input, select, button).
            element_id: Element ID attribute.
            class_name: CSS class name (substring match).
            type_attr: Input type attribute (e.g., text, email, date).
        """
        events = self._interaction_data_store.filter_by_element(
            tag_name=tag_name,
            element_id=element_id,
            class_name=class_name,
            type_attr=type_attr,
        )

        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": self._interaction_data_store.events.index(event),
                "type": event.type.value,
                "tag_name": el.tag_name,
                "element_id": el.id,
                "element_name": el.name,
                "type_attr": el.type_attr,
                "value": el.value,
                "css_path": el.css_path,
                "url": event.url,
            })

        return {
            "total_matching": len(events),
            "showing": len(results),
            "results": results,
        }


    @agent_tool()
    @token_optimized
    def _get_interaction_detail(self, index: int) -> dict[str, Any]:
        """
        Get full details of a specific interaction event by index.

        Args:
            index: Zero-based index of the interaction event.
        """
        detail = self._interaction_data_store.get_event_detail(index)
        if detail is None:
            return {"error": f"Event index {index} out of range (0-{len(self._interaction_data_store.events) - 1})"}

        return detail


    @agent_tool()
    @token_optimized
    def _get_form_inputs(self) -> dict[str, Any]:
        """Get all input/change events with their values and element info."""
        inputs = self._interaction_data_store.get_form_inputs()
        return {
            "total_inputs": len(inputs),
            "inputs": inputs[:100],
        }


    @agent_tool()
    @token_optimized
    def _get_unique_elements(self) -> dict[str, Any]:
        """Get deduplicated elements with interaction counts and types."""
        elements = self._interaction_data_store.get_unique_elements()
        return {
            "total_unique_elements": len(elements),
            "elements": elements[:50],
        }
