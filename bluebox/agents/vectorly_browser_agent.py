"""
bluebox/agents/vectorly_browser_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- VectorlyBrowserAgent: Agent for searching and executing Vectorly routines
- Uses: AbstractAgent base class for all agent plumbing
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable

import requests

from bluebox.agents.abstract_agent import AbstractAgent, agent_tool
from bluebox.cdp.connection import cdp_new_tab
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatResponseEmittedMessage,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class VectorlyBrowserAgent(AbstractAgent):
    """
    Vectorly browser agent that searches and executes web automation routines.

    The agent uses AbstractAgent as its base and provides tools to search,
    inspect, and execute Vectorly routines.
    """

    AGENT_LOOP_MAX_ITERATIONS: int = 100

    SYSTEM_PROMPT: str = dedent("""
        You are a routine execution agent. Your job is to find and run pre-built Vectorly routines to fulfill user requests.

        ## Available Tools
        - `search_routines(keywords: list[str])` — Keyword search over routines. Each keyword is matched individually against routine names and descriptions. Pass SHORT, SINGLE-WORD keywords like `["train", "search", "amtrak"]` — NOT long phrases like `["search for amtrak trains"]`. More individual words = broader coverage.
        - `get_routine_details(routine_id: str)` — Get routine parameters before execution.
        - `execute_routines_parallel(routine_requests: list[dict], max_concurrency: int = 5)` — Execute routines in parallel, each on its own browser tab. Each dict needs 'routine_id' and optionally 'parameters'. Works for single or multiple routines.

        ## Workflow
        1. **Search broadly**: When the user makes a request, use `search_routines` with many single-word keywords to find ALL potentially relevant routines. Think of synonyms and related terms. Example: for "book a flight" use `["flight", "book", "airline", "travel", "booking"]`.
        2. **Inspect matches**: Use `get_routine_details` on each promising match to understand its parameters and what it does.
        3. **Execute all relevant routines**: Run ALL routines that could plausibly fulfill the user's request via `execute_routines_parallel`. When in doubt, include the routine — running an extra routine is cheap, missing a relevant one is costly.
        4. **Report results**: Summarize what was executed and the results to the user.

        ## Important Rules
        - You ONLY have routine tools. Do not tell the user you can browse, click, type, or interact with web pages directly.
        - If no routines match, tell the user clearly that no matching routines were found.
        - Keywords MUST be short single words. Never pass multi-word phrases as a single keyword. If your first search returns no results, try different synonyms and related single-word keywords before giving up.
        - Be concise in responses.
    """).strip()

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        routine_output_dir: str = "./routine_output",
    ) -> None:
        """
        Initialize the Vectorly browser agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            remote_debugging_address: Chrome remote debugging address for routine execution.
            routine_output_dir: Directory to save routine execution results as JSON files.
        """
        # Validate required config
        if not Config.VECTORLY_API_KEY:
            raise ValueError("VECTORLY_API_KEY is not set")
        if not Config.VECTORLY_API_BASE:
            raise ValueError("VECTORLY_API_BASE is not set")

        self._remote_debugging_address = remote_debugging_address
        self._routines_cache: dict[str, Routine] | None = None
        self._routine_output_dir = Path(routine_output_dir)

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=None,
        )
        # Pre-load routines on boot
        self._get_all_routines()

        logger.debug(
            "VectorlyBrowserAgent initialized with model: %s, chat_thread_id: %s",
            llm_model,
            self._thread.id,
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with current time."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"
        return self.SYSTEM_PROMPT + time_info

    ## Routine fetching (instance-level cache)

    def _get_all_routines(self) -> dict[str, Routine]:
        """
        Get all routines from Vectorly (organization + public).

        Returns:
            Dictionary mapping routine IDs to Routine objects.
        """
        if self._routines_cache is not None:
            return self._routines_cache

        routines: list[dict[str, Any]] = []

        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }

        # Get organization routines
        response = requests.get(
            f"{Config.VECTORLY_API_BASE}/routines/organization_routines",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        routines.extend(response.json())

        # Get public routines
        response = requests.get(
            f"{Config.VECTORLY_API_BASE}/routines/public",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        routines.extend(response.json())

        # Build dictionary of routine IDs to Routine objects
        routine_dict = {}
        for routine_data in routines:
            try:
                routine = Routine(**routine_data)
                routine_dict[routine_data["id"]] = routine
            except Exception:
                continue

        self._routines_cache = routine_dict
        return routine_dict

    def _clear_routine_cache(self) -> None:
        """Clear the routine cache to force a refresh."""
        self._routines_cache = None

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _search_routines(self, keywords: list[str]) -> dict[str, Any]:
        """
        Search for routines by keywords. Matches against routine name and description.

        Args:
            keywords: List of keywords to search for (case-insensitive, matches if ANY keyword is found).
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if not routines:
            return {"message": "No routines available", "matches": []}

        # Normalize keywords: lowercase, split multi-word entries into individual words, deduplicate
        split_keywords: list[str] = []
        for kw in keywords:
            for word in kw.lower().split():
                word = word.strip()
                if word:
                    split_keywords.append(word)
        keywords_lower = list(dict.fromkeys(split_keywords))  # deduplicate, preserve order
        if not keywords_lower:
            return {"error": "No valid keywords provided. Please retry with specific, non-empty single-word keywords describing the routine you are looking for."}

        matches = []
        for routine_id, routine in routines.items():
            # Build searchable text from name and description
            searchable = f"{routine.name} {routine.description or ''}".lower()

            # Check if any keyword matches
            matched_keywords = [kw for kw in keywords_lower if kw in searchable]
            if matched_keywords:
                matches.append({
                    "id": routine_id,
                    "name": routine.name,
                    "description": routine.description,
                    "parameter_count": len(routine.parameters) if routine.parameters else 0,
                    "matched_keywords": matched_keywords,
                })

        # Sort by number of matched keywords (most matches first)
        matches.sort(key=lambda x: len(x["matched_keywords"]), reverse=True)

        return {
            "keywords": keywords_lower,
            "match_count": len(matches),
            "matches": matches[:20],  # Limit to top 20 matches
        }

    @agent_tool()
    @token_optimized
    def _get_routine_details(self, routine_id: str) -> dict[str, Any]:
        """
        Get full details of a specific routine by ID.

        Returns the routine's name, description, parameters, and operations.
        Use this to understand what inputs are required before execution.

        Args:
            routine_id: The ID of the routine to retrieve.
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if routine_id not in routines:
            return {"error": f"Routine ID '{routine_id}' not found"}

        routine = routines[routine_id]
        return routine.model_dump()

    @agent_tool()
    def _execute_routines_parallel(
        self,
        routine_requests: list[dict[str, Any]],
        max_concurrency: int = 5,
    ) -> dict[str, Any]:
        """
        Execute one or more routines in parallel, each on its own browser tab.

        Each routine runs in an isolated tab that remains open after execution.
        The tab_id is returned in each result so you can interact with the page afterwards
        using browser tools (after switching to that tab).
        The agent's own browser tab is not affected.

        Args:
            routine_requests: List of routine execution requests. Each dict must have 'routine_id' (str) and optionally 'parameters' (dict).
            max_concurrency: Maximum number of routines to run simultaneously. Defaults to 5.
        """
        if not routine_requests:
            return {"error": "No routine requests provided"}

        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        max_concurrency = max(1, min(max_concurrency, 10))

        # Validate each request
        validated: list[tuple[str, Any, dict]] = []
        validation_errors: list[dict[str, Any]] = []

        for i, req in enumerate(routine_requests):
            routine_id = req.get("routine_id")
            parameters = req.get("parameters", {})

            if not routine_id:
                validation_errors.append({"index": i, "error": "Missing 'routine_id'"})
                continue

            if routine_id not in routines:
                validation_errors.append({"index": i, "routine_id": routine_id, "error": f"Routine ID '{routine_id}' not found"})
                continue

            validated.append((routine_id, routines[routine_id], parameters))

        if not validated:
            return {
                "success": False,
                "error": "All routine requests failed validation",
                "validation_errors": validation_errors,
                "results": [],
            }

        # Execute validated routines in parallel, each on its own new tab
        def save_result(result: dict[str, Any]) -> dict[str, Any]:
            """Save a single routine result to a JSON file."""
            try:
                self._routine_output_dir.mkdir(parents=True, exist_ok=True)
                rid = result.get("routine_id", "unknown")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                output_path = self._routine_output_dir / f"routine_results_{timestamp}_{rid}.json"
                output_path.write_text(json.dumps(result, indent=2, default=str))
                result["output_file"] = str(output_path)
                logger.info("Routine result saved to %s", output_path)
            except Exception as e:
                logger.exception("Failed to save routine result to file: %s", e)
                result["output_file_error"] = str(e)
            return result

        def execute_one(routine_id: str, routine: Any, parameters: dict) -> dict[str, Any]:
            # Create a dedicated tab for this routine
            target_id = None
            browser_context_id = None
            try:
                target_id, browser_context_id, browser_ws = cdp_new_tab(
                    remote_debugging_address=self._remote_debugging_address,
                    incognito=routine.incognito,
                )
                browser_ws.close()  # Close creation ws; routine.execute() creates its own
            except Exception as e:
                logger.exception("Failed to create tab for routine %s: %s", routine_id, e)
                return save_result({
                    "success": False,
                    "routine_id": routine_id,
                    "routine_name": routine.name,
                    "error": f"Failed to create tab: {e}",
                })

            try:
                result = routine.execute(
                    parameters_dict=parameters,
                    remote_debugging_address=self._remote_debugging_address,
                    tab_id=target_id,
                    close_tab_when_done=True,
                )
                return save_result({
                    "success": result.ok,
                    "routine_id": routine_id,
                    "routine_name": routine.name,
                    "tab_id": target_id,
                    "data": result.data,
                })
            except Exception as e:
                logger.exception("Parallel routine execution failed for %s: %s", routine_id, e)
                return save_result({
                    "success": False,
                    "routine_id": routine_id,
                    "routine_name": routine.name,
                    "tab_id": target_id,
                    "error": str(e),
                })

        results: list[dict[str, Any]] = []
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(execute_one, rid, routine, params): rid
                for rid, routine, params in validated
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed_count += 1

                # Stream each result as it completes
                status = "succeeded" if result.get("success") else "FAILED"
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"[{completed_count}/{len(validated)}] Routine '{result.get('routine_name')}' {status}.",
                ))

        results.sort(key=lambda r: r.get("routine_id", ""))

        succeeded = sum(1 for r in results if r.get("success"))
        failed = len(results) - succeeded

        return {
            "success": failed == 0 and not validation_errors,
            "total_requested": len(routine_requests),
            "total_executed": len(validated),
            "succeeded": succeeded,
            "failed": failed,
            "validation_errors": validation_errors or None,
            "results": results,
        }
