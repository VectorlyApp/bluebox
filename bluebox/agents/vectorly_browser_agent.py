"""
bluebox/agents/vectorly_browser_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- VectorlyBrowserAgent: Specialist for executing browser automation via Vectorly
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

from datetime import datetime
from textwrap import dedent
from typing import Any, Callable

import requests

from bluebox.agents.abstract_agent import agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class VectorlyBrowserAgent(AbstractSpecialist):
    """
    Vectorly browser agent that helps execute web automation routines.

    The agent uses AbstractSpecialist as its base and provides tools to list,
    inspect, and execute Vectorly routines for browser automation.
    """

    SYSTEM_PROMPT: str = dedent("""
        You execute browser automation routines for users.

        ## Tools
        - `list_routines` — list available routines
        - `get_routine_details` — get routine parameters and details
        - `execute_routine` — run a routine with parameters

        ## Rules
        - Be concise
        - Get routine details before executing
        - Confirm parameters with user before execution
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = ""  # Not used for now

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
    ) -> None:
        """
        Initialize the Vectorly browser agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            remote_debugging_address: Chrome remote debugging address for routine execution.
        """
        # Validate required config
        if not Config.VECTORLY_API_KEY:
            raise ValueError("VECTORLY_API_KEY is not set")
        if not Config.VECTORLY_API_BASE:
            raise ValueError("VECTORLY_API_BASE is not set")

        self._remote_debugging_address = remote_debugging_address
        self._routines_cache: dict[str, Routine] | None = None

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
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
        """Get system prompt with current time and available routines."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"

        try:
            routines = self._get_all_routines()
            routine_count = len(routines)
            routine_summary = f"\n\n## Available Routines: {routine_count}\n"

            if routines:
                routine_lines = []
                for routine_id, routine in list(routines.items())[:20]:
                    routine_lines.append(f"- **{routine.name}** (ID: `{routine_id}`)")
                routine_summary += "\n".join(routine_lines)
                if routine_count > 20:
                    routine_summary += f"\n... and {routine_count - 20} more."
            else:
                routine_summary += "No routines available."

        except Exception as e:
            routine_summary = f"\n\n## Routines: Error - {e}"

        return self.SYSTEM_PROMPT + time_info + routine_summary

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode (not implemented)."""
        return self.AUTONOMOUS_SYSTEM_PROMPT

    def _get_autonomous_initial_message(self, task: str) -> str:
        """Build initial message for autonomous mode (not implemented)."""
        return f"TASK: {task}"

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
    def _list_routines(self) -> dict[str, Any]:
        """
        List all available routines from Vectorly (organization + public).

        Returns a list of routines with their IDs, names, and descriptions.
        Use `get_routine_details` to see full details including parameters.
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if not routines:
            return {
                "message": "No routines available",
                "total_count": 0,
            }

        routine_list = [
            {
                "id": routine_id,
                "name": routine.name,
                "description": routine.description,
                "parameter_count": len(routine.parameters) if routine.parameters else 0,
            }
            for routine_id, routine in routines.items()
        ]

        return {
            "total_count": len(routine_list),
            "routines": routine_list,
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
    def _execute_routine(
        self,
        routine_id: str,
        parameters: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Execute a routine with the given parameters.

        This will run the routine against the connected Chrome browser.
        Make sure Chrome is running with remote debugging enabled.

        Args:
            routine_id: The ID of the routine to execute.
            parameters: Dictionary of parameter values required by the routine.
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if routine_id not in routines:
            return {"error": f"Routine ID '{routine_id}' not found"}

        routine = routines[routine_id]

        try:
            result = routine.execute(
                parameters_dict=parameters,
                remote_debugging_address=self._remote_debugging_address,
            )
            return {
                "success": True,
                "routine_id": routine_id,
                "routine_name": routine.name,
                "result": result.model_dump() if hasattr(result, "model_dump") else result,
            }
        except Exception as e:
            logger.exception("Routine execution failed: %s", e)
            return {
                "success": False,
                "routine_id": routine_id,
                "routine_name": routine.name,
                "error": str(e),
            }

    @agent_tool()
    def _refresh_routines(self) -> dict[str, Any]:
        """
        Refresh the cached list of routines from Vectorly.

        Use this if you expect new routines to be available or if the
        routine list seems stale.
        """
        self._clear_routine_cache()
        try:
            routines = self._get_all_routines()
            return {
                "success": True,
                "message": "Routine cache refreshed",
                "total_count": len(routines),
            }
        except requests.RequestException as e:
            return {"error": f"Failed to refresh routines: {e}"}
