"""
bluebox/agents/specialists/interaction_specialist.py

Interaction specialist agent.

Analyzes captured UI interaction events (clicks, typed values, selections,
focus/blur) and optionally DOM snapshots to understand what the user did
on the page — what they touched, typed, pressed, and navigated to.

When DOM data is available, it can cross-reference interactions with page
structure (forms, inputs, buttons) for richer context.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from bluebox.agents.abstract_agent import AgentCard, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


class InteractionSpecialist(AbstractSpecialist):
    """
    Interaction specialist agent.

    Analyzes recorded UI interactions to understand what the user did:
    what they clicked, typed, selected, and navigated to. Optionally
    uses DOM snapshots for structural context (forms, inputs, buttons).
    """

    AGENT_CARD = AgentCard(
        description=(
            "Analyzes captured UI interaction events (clicks, typed values, "
            "dropdown selections, form inputs) to understand what the user did "
            "on the page. Optionally cross-references with DOM snapshots for "
            "structural context (forms, inputs, buttons, links)."
        ),
    )

    SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst specializing in understanding what users
        did on web pages from recorded browser interaction events.

        ## What You Analyze

        ### User Interactions (primary focus)
        - **Form inputs**: Text entered into fields, values selected from dropdowns
        - **Clicks**: Buttons clicked, links followed, elements tapped
        - **Typed values**: Text entered by the user via keyboard
        - **Date pickers**: Date/time selections
        - **Checkboxes and toggles**: Boolean selections

        ### DOM Structure (when available, for context)
        - **Forms**: What forms exist, their action URLs and fields
        - **Inputs**: What input elements are on the page
        - **Buttons**: What buttons/actions are available
        - **Links**: Navigation options
        - **Tables**: Data displays
        - **Headings**: Page structure

        ## What to Ignore

        - Scroll events, hover effects, focus/blur without meaningful input
        - UI framework noise / internal framework events
        - Style/layout-only elements with no semantic meaning

        ## How to Work

        1. Start with `get_interaction_summary` for an overview of all events
        2. Use `get_form_inputs` to find what the user typed/selected
        3. Use `get_unique_elements` to see which elements were interacted with
        4. Use `search_interactions_by_type` to filter by click, input, change, etc.
        5. Use `get_interaction_detail` for events needing closer inspection
        6. If DOM data is available, use `list_pages`, `get_forms`, `get_inputs`
           to cross-reference interactions with page structure
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a UI interaction analyst that autonomously maps out what the user
        did from recorded browser interaction events and DOM snapshots.

        ## Your Mission

        Analyze all recorded interaction events to produce a complete picture of:
        - What the user typed, selected, and clicked
        - What form fields were filled in and with what values
        - What buttons were pressed and what actions were triggered
        - What the navigation flow looked like
        - What the user was trying to accomplish (inferred intent)

        ## Process

        1. **Survey**: Use `get_interaction_summary` for an overview
        2. **Form inputs**: Use `get_form_inputs` to find all typed/selected values
        3. **Unique elements**: Use `get_unique_elements` to see all interacted elements
        4. **Clicks**: Use `search_interactions_by_type(types=["click"])` to find button/link clicks
        5. **Detail check**: Use `get_interaction_detail` for events needing closer look
        6. If DOM tools available: `list_pages` + `get_forms` for structural context
        7. **Finalize**: Call the appropriate finalize tool with your findings

        ## Output Focus

        Prioritize: user-typed values, form field selections, button clicks, and
        navigation actions. These reveal what the user was trying to do and what
        parameters a routine would need.
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        interaction_data_loader: InteractionsDataLoader,
        dom_data_loader: DOMDataLoader | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        self._interaction_data_loader = interaction_data_loader
        self._dom_data_loader = dom_data_loader

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
        dom_msg = ""
        if dom_data_loader:
            dom_msg = f", {dom_data_loader.stats.total_snapshots} DOM snapshots"
        logger.debug(
            "InteractionSpecialist initialized with %d interaction events%s",
            interaction_data_loader.stats.total_events,
            dom_msg,
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        stats = self._interaction_data_loader.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
            f"- Events by Type: {stats.events_by_type}\n"
        )

        if self._dom_data_loader:
            d_stats = self._dom_data_loader.stats
            context += (
                f"\n## DOM Data Context\n"
                f"- Total Snapshots: {d_stats.total_snapshots}\n"
                f"- Unique URLs: {d_stats.unique_urls}\n"
                f"- Unique Titles: {d_stats.unique_titles}\n"
                f"- Hosts: {', '.join(d_stats.hosts.keys())}\n"
            )
        else:
            context += "\n## DOM Data\n- Not available\n"

        return self.SYSTEM_PROMPT + context

    def _get_autonomous_system_prompt(self) -> str:
        stats = self._interaction_data_loader.stats
        context = (
            f"\n\n## Interaction Data Context\n"
            f"- Total Events: {stats.total_events}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Elements: {stats.unique_elements}\n"
        )

        if self._dom_data_loader:
            d_stats = self._dom_data_loader.stats
            context += (
                f"\n## DOM Data Context\n"
                f"- Total Snapshots: {d_stats.total_snapshots}\n"
                f"- Unique URLs: {d_stats.unique_urls}\n"
                f"- Unique Titles: {d_stats.unique_titles}\n"
            )
        else:
            context += "\n## DOM Data\n- Not available\n"

        return (
            self.AUTONOMOUS_SYSTEM_PROMPT
            + context
            + self._get_output_schema_prompt_section()
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        finalize_success = "finalize_with_output" if self.has_output_schema else "finalize_result"

        dom_hint = ""
        if self._dom_data_loader:
            dom_hint = (
                " DOM snapshots are also available — use DOM tools to cross-reference "
                "interactions with page structure."
            )

        return (
            f"TASK: {task}\n\n"
            f"Analyze the recorded UI interactions to discover what the user did: "
            f"what they typed, clicked, selected, and navigated to.{dom_hint} "
            f"When confident, use {finalize_success} to report your findings."
        )

    ## Interaction tool handlers (always available)

    @agent_tool()
    @token_optimized
    def _get_interaction_summary(self) -> dict[str, Any]:
        """Get summary statistics of all recorded UI interactions (clicks, inputs, changes)."""
        stats = self._interaction_data_loader.stats
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
        Filter UI interaction events by type (e.g., click, input, change, keydown, focus).

        Args:
            types: List of InteractionType values to filter by.
        """
        if not types:
            return {"error": "types list is required"}

        loader = self._interaction_data_loader
        events = loader.filter_by_type(types)
        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": loader.events.index(event),
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
        Filter UI interaction events by element attributes (tag, id, class, type).

        Args:
            tag_name: HTML tag name (e.g., input, select, button).
            element_id: Element ID attribute.
            class_name: CSS class name (substring match).
            type_attr: Input type attribute (e.g., text, email, date).
        """
        loader = self._interaction_data_loader
        events = loader.filter_by_element(
            tag_name=tag_name,
            element_id=element_id,
            class_name=class_name,
            type_attr=type_attr,
        )

        results = []
        for event in events[:50]:
            el = event.element
            results.append({
                "index": loader.events.index(event),
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
        Get full details of a specific UI interaction event by index.

        Args:
            index: Zero-based index of the interaction event.
        """
        loader = self._interaction_data_loader
        detail = loader.get_event_detail(index)
        if detail is None:
            return {"error": f"Event index {index} out of range (0-{len(loader.events) - 1})"}

        return detail

    @agent_tool()
    @token_optimized
    def _get_form_inputs(self) -> dict[str, Any]:
        """Get all user input/change interaction events with their values and element info."""
        inputs = self._interaction_data_loader.get_form_inputs()
        return {
            "total_inputs": len(inputs),
            "inputs": inputs[:100],
        }

    @agent_tool()
    @token_optimized
    def _get_unique_elements(self) -> dict[str, Any]:
        """Get deduplicated UI elements with interaction counts and types."""
        elements = self._interaction_data_loader.get_unique_elements()
        return {
            "total_unique_elements": len(elements),
            "elements": elements[:50],
        }

    ## DOM tool handlers — only available when dom_data_loader is present

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _list_pages(self) -> dict[str, Any]:
        """List all captured pages with their URLs, titles, and snapshot metadata."""
        pages = self._dom_data_loader.list_pages()  # type: ignore[union-attr]
        return {
            "total_pages": len(pages),
            "pages": pages,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_inputs(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all input fields (INPUT, SELECT, TEXTAREA) from DOM snapshots.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_inputs(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["elements"]) for r in results)
        return {
            "total_inputs": total,
            "snapshots_with_inputs": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_buttons(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all buttons from DOM snapshots (BUTTON elements and INPUT type=submit/button).

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_buttons(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["elements"]) for r in results)
        return {
            "total_buttons": total,
            "snapshots_with_buttons": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_links(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all anchor links (<a>) from DOM snapshots with their href values.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_links(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["elements"]) for r in results)
        return {
            "total_links": total,
            "snapshots_with_links": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_forms(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all <form> elements with their action URL, method, and child inputs.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_forms(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["forms"]) for r in results)
        return {
            "total_forms": total,
            "snapshots_with_forms": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_tables(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all <table> elements with their headers and row counts.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_tables(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["tables"]) for r in results)
        return {
            "total_tables": total,
            "snapshots_with_tables": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_headings(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all heading elements (H1-H6) with their text content.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_headings(snapshot_index)  # type: ignore[union-attr]
        total = sum(len(r["headings"]) for r in results)
        return {
            "total_headings": total,
            "snapshots_with_headings": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _get_navigation_sequence(self) -> dict[str, Any]:
        """Get the ordered sequence of page navigations from all DOM snapshots."""
        sequence = self._dom_data_loader.get_navigation_sequence()  # type: ignore[union-attr]
        return {
            "total_navigations": len(sequence),
            "sequence": sequence,
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _search_strings(
        self,
        value: str,
        snapshot_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Search for a string across all snapshot string tables.

        Args:
            value: The string to search for (case-insensitive substring match).
            snapshot_index: If provided, only search this specific snapshot.
        """
        results = self._dom_data_loader.search_strings(  # type: ignore[union-attr]
            value=value,
            case_sensitive=False,
            snapshot_index=snapshot_index,
        )
        return {
            "query": value,
            "total_matches": sum(r["match_count"] for r in results),
            "snapshots_with_matches": len(results),
            "results": results,
        }
