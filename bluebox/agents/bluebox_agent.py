"""
bluebox/agents/bluebox_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- BlueBoxAgent: Agent for searching and executing Vectorly routines
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

from bluebox.agents.abstract_agent import AbstractAgent, AgentCard, agent_tool
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatResponseEmittedMessage,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import RoutineExecutionRequest, RoutineInfo
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class BlueBoxAgent(AbstractAgent):
    """
    BlueBoxAgent that searches and executes web automation routines.

    The agent uses AbstractAgent as its base and provides tools to search,
    inspect, and execute Vectorly routines.
    """

    AGENT_CARD = AgentCard(
        description="Searches and executes pre-built Vectorly routines to fulfill user requests.",
    )

    AGENT_LOOP_MAX_ITERATIONS: int = 100

    SYSTEM_PROMPT: str = dedent("""
        You are a routine execution agent. Your job is to find and run pre-built Vectorly routines to fulfill user requests.

        ## Workflow
        1. **Search broadly**: When the user makes a request, use `search_routines` with a task description that describes what the user wants to do. This runs semantic search, so add some detail. You can run this multiple times if needed to get more results.
        2. **Execute all relevant routines**: Run ALL routines that could plausibly fulfill the user's request via `execute_routines_in_parallel`. When in doubt, include the routine — running an extra routine is cheap, missing a relevant one is costly.
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
        routine_output_dir: str = "./routine_output",
    ) -> None:
        """
        Initialize the BlueBox Agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            routine_output_dir: Directory to save routine execution results as JSON files.
        """
        # Validate required config
        if not Config.VECTORLY_API_KEY:
            raise ValueError("VECTORLY_API_KEY is not set")
        if not Config.VECTORLY_API_BASE:
            raise ValueError("VECTORLY_API_BASE is not set")

        self._routine_output_dir = Path(routine_output_dir)
        self._routine_cache: dict[str, RoutineInfo] = {}

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

        logger.debug(
            "BlueBoxAgent initialized with model: %s, chat_thread_id: %s",
            llm_model,
            self._thread.id,
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with current time."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"
        return self.SYSTEM_PROMPT + time_info

    ## Routine cache

    def _cache_routines_from_response(self, response: dict[str, Any] | list[Any]) -> None:
        """Parse search response and cache RoutineInfo objects for later validation."""
        if isinstance(response, list):
            items = response
        else:
            items = response.get("results", response.get("routines", []))
        if not isinstance(items, list):
            return
        for item in items:
            try:
                if not isinstance(item, dict) or "routine_id" not in item:
                    continue
                info = RoutineInfo.model_validate(item)
                self._routine_cache[info.routine_id] = info
                logger.debug("Cached routine: %s (%s)", info.name, info.routine_id)
            except Exception:
                logger.debug("Skipped caching item: %s", item.get("routine_id", "unknown"))

    def _validate_routine_params(self, routine_id: str, params: dict[str, Any]) -> str | None:
        """Validate params against cached routine info. Returns error string or None."""
        info = self._routine_cache.get(routine_id)
        if not info:
            return None  # Not cached, skip validation

        required = {p.name for p in info.parameters if p.required}
        provided = set(params.keys())
        missing = required - provided
        if missing:
            param_summary = [
                {"name": p.name, "type": p.type.value, "required": p.required, "description": p.description}
                for p in info.parameters
            ]
            return (
                f"Routine '{info.name}' ({routine_id}): missing required parameter(s) {sorted(missing)}. "
                f"Expected parameters: {param_summary}"
            )
        return None

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _search_routines(self, task: str) -> dict[str, Any]:
        """
        Search for routines by keywords. Matches against routine name and description.

        Args:
            task: Task description to search for.
        """
        url = f"{Config.VECTORLY_API_BASE}/routines/semantic-search"
        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }
        payload = {
            "query": task,
            "top_n": 5,
            "threshold": 0.0,
            "keywords": [],
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        self._cache_routines_from_response(data)
        return data

    @agent_tool()
    def _execute_routines_in_parallel(
        self,
        routine_executions: list[RoutineExecutionRequest],
    ) -> dict[str, Any]:
        """
        Execute one or more routines in parallel via the Vectorly API.

        Args:
            routine_executions: List of routines to execute. Each item needs routine_id and parameters.
        """
        if not routine_executions:
            return {"error": "No routine executions provided"}

        # Pre-flight validation against cached routine metadata
        validation_errors: list[str] = []
        for req in routine_executions:
            error = self._validate_routine_params(req.routine_id, req.parameters)
            if error:
                validation_errors.append(error)

        if validation_errors:
            return {"error": "Parameter validation failed. Fix and retry.\n" + "\n".join(validation_errors)}

        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }

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

        def execute_one(req: RoutineExecutionRequest) -> dict[str, Any]:
            url = f"{Config.VECTORLY_API_BASE}/routines/{req.routine_id}/execute"
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json={"parameters": req.parameters},
                    timeout=300,
                )
                response.raise_for_status()
                return save_result({"success": True, "routine_id": req.routine_id, "data": response.json()})
            except requests.RequestException as e:
                logger.error("Routine execution failed for %s: %s", req.routine_id, e)
                return save_result({"success": False, "routine_id": req.routine_id, "error": str(e)})

        total = len(routine_executions)
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(execute_one, req): req for req in routine_executions}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                status = "succeeded" if result.get("success") else "FAILED"
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"[{len(results)}/{total}] Routine '{result.get('routine_id')}' {status}.",
                ))

        succeeded = sum(1 for r in results if r.get("success"))
        return {
            "success": succeeded == total,
            "total_requested": total,
            "succeeded": succeeded,
            "failed": total - succeeded,
            "results": results,
        }
