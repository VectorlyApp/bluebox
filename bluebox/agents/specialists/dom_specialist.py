"""
bluebox/agents/specialists/dom_specialist.py

DOM specialist agent.

Analyzes captured DOM snapshots to discover page structure, interactive elements,
forms, tables, links, and navigation patterns. Used during the exploration phase
to understand what the browser rendered and what UI elements are available.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from bluebox.agents.abstract_agent import AgentCard, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.agents.workspace import AgentWorkspace
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


class DOMSpecialist(AbstractSpecialist):
    """
    DOM specialist agent.

    Analyzes captured DOM snapshots to discover page structure,
    interactive elements, forms, and navigation patterns.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Analyzes captured DOM snapshots (page structure, forms, inputs, buttons, "
            "links, tables, headings). Useful for understanding what the browser rendered "
            "and what interactive elements exist on each page."
        ),
    )

    SYSTEM_PROMPT: str = dedent("""\
        You are a DOM structure analyst specializing in understanding web page layouts from captured browser snapshots.

        ## What You Analyze

        - **Forms**: Login forms, search forms, checkout forms — with their inputs, actions, and methods
        - **Elements**: Inputs, buttons, links, headings, meta tags, hidden inputs, clickable elements
        - **Tables**: Data tables with headers and row counts
        - **Script tags**: Server-side data blobs (__NEXT_DATA__, __NUXT__), inline JSON config, structured data (ld+json)

        ## What to Ignore

        - Internal framework nodes, shadow DOM internals
        - Style/layout-only elements with no semantic meaning

        ## How to Work

        1. Start with `list_pages` to see all captured pages
        2. Use `get_elements(element_type=...)` to scan for inputs, buttons, links, headings, meta_tags, hidden_inputs, or clickable elements
        3. Use `get_forms` for forms with their child inputs
        4. Use `get_tables` for data tables
        5. Use `get_scripts` to find server-side data blobs and inline configuration
        6. Use `get_snapshot_diff` to understand what changed between pages
        7. Use `search_strings` to find specific content across snapshots

    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a DOM structure analyst that autonomously maps out page structure from captured browser snapshots.

        ## Your Mission

        Analyze all captured DOM snapshots to produce a complete picture of:
        - What pages were visited and in what order
        - What forms exist and what they do (action URLs, input fields)
        - What interactive elements are available (buttons, links, inputs)
        - What data is displayed (tables, headings, text content)
        - What tokens/keys are embedded in the page (CSRF, session IDs, API keys)
        - What server-side data is rendered into the DOM (__NEXT_DATA__, inline JSON, ld+json)

        ## Process

        1. **Survey**: Use `list_pages` to see all captured pages
        2. **Scan forms**: Use `get_forms` to find all forms with their inputs
        3. **Scan elements**: Use `get_elements(element_type=...)` for each type:
           - `inputs` — text fields, dropdowns, checkboxes, date pickers
           - `buttons` — submit buttons, action buttons
           - `links` — anchor links with href values
           - `headings` — H1-H6 page structure
           - `meta_tags` — CSRF tokens, API configs, verification keys
           - `hidden_inputs` — CSRF tokens, session IDs, form tokens
           - `clickable` — anything the browser marked as interactive
        4. **Scan tables**: Use `get_tables` for data displays
        5. **Scan scripts**: Use `get_scripts` to find __NEXT_DATA__, inline JSON, framework state blobs
        6. **Check diffs**: Use `get_snapshot_diff` between consecutive pages to see what changed
        7. **Finalize**: Call the appropriate finalize tool with your findings

        ## Output Focus

        Prioritize: forms and their endpoints, parameterizable inputs, action buttons, data tables,
        embedded tokens/keys, and server-side data blobs. These are what matter for routine construction.
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        dom_data_loader: DOMDataLoader,
        documentation_data_loader: DocumentationDataLoader | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        workspace: AgentWorkspace | None = None,
    ) -> None:
        self._dom_data_loader = dom_data_loader

        super().__init__(
            emit_message_callable=emit_message_callable,
            workspace=workspace,
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
            "DOMSpecialist initialized with %d snapshots",
            self._dom_data_loader.stats.total_snapshots,
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        stats = self._dom_data_loader.stats
        context = (
            f"\n\n## DOM Data Context\n"
            f"- Total Snapshots: {stats.total_snapshots}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Titles: {stats.unique_titles}\n"
            f"- Hosts: {', '.join(stats.hosts.keys())}\n"
        )
        return self.SYSTEM_PROMPT + context

    def _get_autonomous_system_prompt(self) -> str:
        stats = self._dom_data_loader.stats
        context = (
            f"\n\n## DOM Data Context\n"
            f"- Total Snapshots: {stats.total_snapshots}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Titles: {stats.unique_titles}\n"
            f"- Hosts: {', '.join(stats.hosts.keys())}\n"
        )

        return (
            self.AUTONOMOUS_SYSTEM_PROMPT
            + context
            + self._get_output_schema_prompt_section()
            + self._get_urgency_notice()
        )

    def _get_autonomous_initial_message(self, task: str) -> str:
        finalize_success = "finalize_with_output" if self.has_output_schema else "finalize_result"

        return (
            f"TASK: {task}\n\n"
            f"Analyze the captured DOM snapshots to map out page structure, forms, "
            f"inputs, buttons, links, tables, and navigation patterns. "
            f"When confident, use {finalize_success} to report your findings."
        )

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _list_pages(self) -> dict[str, Any]:
        """List all captured pages with their URLs, titles, and snapshot metadata."""
        pages = self._dom_data_loader.list_pages()
        return {
            "total_pages": len(pages),
            "pages": pages,
        }

    @agent_tool()
    @token_optimized
    def _get_elements(self, element_type: str, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get elements of a specific type from DOM snapshots.

        A single tool that replaces individual per-type tools. Supports:
        - 'inputs' — INPUT, SELECT, TEXTAREA fields with their attributes and values
        - 'buttons' — BUTTON elements and INPUT type=submit/button
        - 'links' — anchor links (<a>) with href values
        - 'headings' — H1-H6 elements with their text content
        - 'meta_tags' — META elements (CSRF tokens, API endpoints, OG tags, page config)
        - 'hidden_inputs' — INPUT type=hidden (CSRF tokens, session IDs, form tokens)
        - 'clickable' — all elements marked as clickable by the browser

        Args:
            element_type: One of 'inputs', 'buttons', 'links', 'headings', 'meta_tags', 'hidden_inputs', 'clickable'.
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        try:
            results = self._dom_data_loader.get_elements(element_type, snapshot_index)
        except ValueError as e:
            return {"error": str(e)}

        total = sum(len(r["elements"]) for r in results)
        return {
            "element_type": element_type,
            "total_elements": total,
            "snapshots_with_elements": len(results),
            "results": results,
        }

    @agent_tool()
    @token_optimized
    def _get_forms(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all <form> elements with their action URL, method, and child inputs.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_forms(snapshot_index)
        total = sum(len(r["forms"]) for r in results)
        return {
            "total_forms": total,
            "snapshots_with_forms": len(results),
            "results": results,
        }

    @agent_tool()
    @token_optimized
    def _get_tables(self, snapshot_index: int | None = None) -> dict[str, Any]:
        """
        Get all <table> elements with their headers and row counts.

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
        """
        results = self._dom_data_loader.get_tables(snapshot_index)
        total = sum(len(r["tables"]) for r in results)
        return {
            "total_tables": total,
            "snapshots_with_tables": len(results),
            "results": results,
        }

    @agent_tool()
    @token_optimized
    def _get_scripts(self, snapshot_index: int | None = None, max_inline_chars: int = 2000) -> dict[str, Any]:
        """
        Get all <script> elements with their attributes and inline content.

        Finds framework data blobs (__NEXT_DATA__, __NUXT__), inline JSON config,
        structured data (ld+json), and embedded configuration (GTM, analytics).

        Args:
            snapshot_index: If provided, only search this specific snapshot. Otherwise searches all.
            max_inline_chars: Max characters for inline script content (default 2000).
        """
        results = self._dom_data_loader.get_scripts(snapshot_index, max_inline_chars)
        total = sum(len(r["scripts"]) for r in results)
        return {
            "total_scripts": total,
            "snapshots_with_scripts": len(results),
            "results": results,
        }

    @agent_tool()
    @token_optimized
    def _get_text_content(self, snapshot_index: int, max_chars: int = 5000) -> dict[str, Any]:
        """
        Get visible text content from a snapshot's string table.

        Args:
            snapshot_index: Zero-based snapshot index.
            max_chars: Maximum characters to return (default 5000).
        """
        content = self._dom_data_loader.get_text_content(snapshot_index, max_chars)
        if content is None:
            return {"error": f"Snapshot index {snapshot_index} out of range"}
        return {
            "snapshot_index": snapshot_index,
            "content": content,
        }

    @agent_tool()
    @token_optimized
    def _search_strings(
        self,
        value: str,
        snapshot_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Search for a string across all snapshot string tables.

        Useful for finding specific content, attribute values, or text.

        Args:
            value: The string to search for (case-insensitive substring match).
            snapshot_index: If provided, only search this specific snapshot.
        """
        results = self._dom_data_loader.search_strings(
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

    @agent_tool()
    @token_optimized
    def _get_snapshot_diff(self, index_a: int, index_b: int) -> dict[str, Any]:
        """
        Compare two snapshots to see what strings were added or removed.

        Useful for understanding what changed between page navigations.

        Args:
            index_a: Index of the first (earlier) snapshot.
            index_b: Index of the second (later) snapshot.
        """
        diff = self._dom_data_loader.get_snapshot_diff(index_a, index_b)
        if diff is None:
            return {"error": f"Invalid snapshot indices: {index_a}, {index_b}"}
        return diff

    @agent_tool()
    @token_optimized
    def _get_navigation_sequence(self) -> dict[str, Any]:
        """Get the ordered sequence of page navigations from all snapshots."""
        sequence = self._dom_data_loader.get_navigation_sequence()
        return {
            "total_navigations": len(sequence),
            "sequence": sequence,
        }
