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

        ## What to Look For

        - **Form inputs**: Text fields, search boxes, email/password fields
        - **Typed values**: Text entered by the user via keyboard
        - **Dropdown selections**: Select elements, custom dropdowns
        - **Date pickers**: Date/time inputs
        - **Checkboxes and toggles**: Boolean parameters

        ## What to Ignore

        - Navigational clicks, scroll events, hover effects, focus/blur without input
        - UI framework noise / internal framework events

        ## Parameter Requirements

        Each discovered parameter needs:
        - **name**: snake_case (e.g., `search_query`, `departure_date`)
        - **type**: One of: string, integer, number, boolean, date, datetime, email, url, enum
        - **description**: What the parameter represents
        - **examples**: Observed values from the interactions

        ## Tools

        - `get_interaction_summary` — overview stats
        - `search_interactions_by_type` — filter by type (click, input, change, etc.)
        - `search_interactions_by_element` — filter by element attributes
        - `get_interaction_detail` — full detail for a specific event
        - `get_form_inputs` — all input/change events with values
        - `get_unique_elements` — deduplicated elements with interaction counts
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst that autonomously discovers routine parameters from recorded browser interactions.

        ## Your Mission

        Analyze recorded UI interactions to identify all parameterizable inputs.

        ## Process

        1. **Survey**: Use `get_interaction_summary` for an overview
        2. **Focus on inputs**: Use `get_form_inputs` to find form input events
        3. **Analyze elements**: Use `get_unique_elements` to see interacted elements
        4. **Detail check**: Use `get_interaction_detail` for events needing closer inspection
        5. **Finalize**: Call the appropriate finalize tool with your findings

        ## Parameter Types

        string, date (YYYY-MM-DD), datetime, integer, number, boolean, email, url, enum
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

        return (
            self.AUTONOMOUS_SYSTEM_PROMPT
            + context
            + self._get_output_schema_prompt_section()
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        # Use correct tool names based on whether output schema is set
        finalize_success = "finalize_with_output" if self.has_output_schema else "finalize_result"

        return (
            f"TASK: {task}\n\n"
            f"Analyze the recorded UI interactions to discover all parameterizable inputs. "
            f"Focus on form inputs, typed values, dropdown selections, and date pickers. "
            f"When confident, use {finalize_success} to report your findings."
        )

    ## Tool handlers

    @agent_tool
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


    @agent_tool
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


    @agent_tool
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


    @agent_tool
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


    @agent_tool
    @token_optimized
    def _get_form_inputs(self) -> dict[str, Any]:
        """Get all input/change events with their values and element info."""
        inputs = self._interaction_data_store.get_form_inputs()
        return {
            "total_inputs": len(inputs),
            "inputs": inputs[:100],
        }


    @agent_tool
    @token_optimized
    def _get_unique_elements(self) -> dict[str, Any]:
        """Get deduplicated elements with interaction counts and types."""
        elements = self._interaction_data_store.get_unique_elements()
        return {
            "total_unique_elements": len(elements),
            "elements": elements[:50],
        }
