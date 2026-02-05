"""
bluebox/agents/specialists/abstract_specialist.py

Abstract base class for specialist agents.

Specialists are domain-expert agents that an orchestrator deploys for specific tasks.
Each specialist owns:
  - A system prompt (conversational + autonomous variants)
  - A set of LLM tools and their execution logic
  - Finalize tools for autonomous mode (registered after min_iterations)

This class extends AbstractAgent to add:
  - Autonomous mode with iteration tracking and finalize gating
  - Conversational mode for interactive chat

Tools are defined declaratively via the @specialist_tool decorator (alias for @agent_tool).
"""

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import Any, Callable, NamedTuple

from pydantic import BaseModel

from bluebox.agents.abstract_agent import AbstractAgent, agent_tool, _ToolMeta
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# Re-export agent_tool as specialist_tool for backwards compatibility
specialist_tool = agent_tool


class RunMode(StrEnum):
    """How the specialist is being run."""
    CONVERSATIONAL = "conversational"  # interactive chat with a user
    AUTONOMOUS = "autonomous"          # autonomous loop (exploration + finalization)


class AutonomousConfig(NamedTuple):
    """
    Configuration for autonomous specialist runs. Helps manage their "lifecycles."
    """
    min_iterations: int = 3   # Minimum iterations before finalize tools become available
    max_iterations: int = 10  # Maximum iterations before loop exits (returns None if not finalized)


class AbstractSpecialist(AbstractAgent):
    """
    Abstract base class for specialist agents.

    Subclasses implement domain-specific logic by overriding:
      - _get_system_prompt()
      - _get_autonomous_system_prompt()
      - _get_autonomous_initial_message()
      - _check_autonomous_completion() — inspect tool results for finalize signals

    Tools are defined declaratively via the @specialist_tool decorator on handler
    methods. Each tool's ``availability`` controls when it is registered: True
    (always), or a callable evaluated before each LLM call.

    This class extends AbstractAgent with:
      - Autonomous mode with iteration tracking and finalize gating
      - Conversational mode for interactive chat
    """

    ## Additional abstract methods for autonomous mode

    @abstractmethod
    def _get_autonomous_system_prompt(self) -> str:
        """
        Return the system prompt for autonomous mode.

        Called every iteration, so it can include dynamic context
        (e.g., iteration count, urgency notices).
        """

    @abstractmethod
    def _get_autonomous_initial_message(self, task: str) -> str:
        """
        Build the initial USER message for autonomous mode.

        Args:
            task: The user's task description.

        Returns:
            Message string to seed the autonomous conversation.
        """

    @abstractmethod
    def _check_autonomous_completion(self, tool_name: str) -> bool:
        """
        Check whether a tool call signals autonomous completion.

        Called after each tool execution in the autonomous loop.
        Return True to stop the loop (e.g., finalize_result was called
        and self._autonomous_result is now set).

        Args:
            tool_name: Name of the tool that was just executed.

        Returns:
            True if the autonomous loop should stop.
        """

    @abstractmethod
    def _get_autonomous_result(self) -> BaseModel | None:
        """
        Return the autonomous mode result after the loop completes.

        Returns:
            A Pydantic model with the specialist's result,
            or None if max iterations were reached without finalization.
        """

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
    ) -> None:
        """
        Initialize the specialist.

        Args:
            emit_message_callable: Callback to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        # Lifecycle state (must be set before parent __init__, which calls _sync_tools)
        self.run_mode: RunMode = run_mode
        self._autonomous_iteration: int = 0
        self._autonomous_config: AutonomousConfig = AutonomousConfig()

        # Call parent init
        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
        )

    ## Properties

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    @property
    def can_finalize(self) -> bool:
        """
        Whether "finalize tools" should be available (autonomous mode, past min_iterations).

        Returns:
            True if the specialist is in autonomous mode and has exceeded the min_iterations threshold, False otherwise.
        """
        return (
            self.run_mode == RunMode.AUTONOMOUS
            and self._autonomous_iteration >= self._autonomous_config.min_iterations
        )

    ## Public API

    def run_autonomous(
        self,
        task: str,
        config: AutonomousConfig | None = None,
    ) -> BaseModel | None:
        """
        Run the specialist autonomously to completion.

        The specialist will:
        1. Use its tools to explore and analyze data
        2. After min_iterations, finalize tools become available (via can_finalize)
        3. Return a typed result when finalize is called, or None on timeout

        Args:
            task: User task description.
            config: Autonomous run configuration (iterations limits). Uses defaults if None.

        Returns:
            Specialist-specific result model, or None if max iterations reached.
        """
        self.run_mode = RunMode.AUTONOMOUS
        self._autonomous_iteration = 0
        self._autonomous_config = config or AutonomousConfig()

        # Subclass should reset its own result fields in _reset_autonomous_state()
        self._reset_autonomous_state()

        # Seed the conversation
        initial_message = self._get_autonomous_initial_message(task)
        self._add_chat(ChatRole.USER, initial_message)

        logger.info("Starting autonomous run for task: %s", task)

        self._run_autonomous_loop()

        self.run_mode = RunMode.CONVERSATIONAL

        return self._get_autonomous_result()

    def _reset_autonomous_state(self) -> None:
        """
        Reset autonomous-mode state before a new run.

        Override in subclasses to clear specialist-specific result fields
        (e.g., self._discovery_result = None). Call super() first.

        NOTE: Method is not abstract; it is intentionally a no-op by default. Not every specialist
        has extra autonomous state to reset; those that don't simply inherit this.
        """
        pass

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        # Reset autonomous state
        self.run_mode = RunMode.CONVERSATIONAL
        self._autonomous_iteration = 0
        self._reset_autonomous_state()

        # Call parent reset
        super().reset()

    ## Agent loops

    def _run_autonomous_loop(self) -> None:
        """Run the autonomous agent loop with iteration tracking and finalize gating."""
        max_iterations = self._autonomous_config.max_iterations
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug("Autonomous loop iteration %d/%d", self._autonomous_iteration, max_iterations)

            messages = self._build_messages_for_llm()
            try:
                response = self._call_llm(messages, self._get_autonomous_system_prompt())

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        role=ChatRole.ASSISTANT,
                        content=response.content or "",
                        tool_calls=response.tool_calls if response.tool_calls else None,
                        llm_provider_response_id=response.response_id,
                    )
                    if response.content:
                        self._emit_message(
                            ChatResponseEmittedMessage(
                                content=response.content,
                                chat_id=chat.id,
                                chat_thread_id=self._thread.id,
                            )
                        )

                if not response.tool_calls:
                    logger.warning("Autonomous loop: no tool calls in iteration %d", self._autonomous_iteration)
                    return

                # Process tool calls and check for completion
                for tool_call in response.tool_calls:
                    result_str = self._auto_execute_tool(tool_call.tool_name, tool_call.tool_arguments)

                    self._add_chat(
                        role=ChatRole.TOOL,
                        content=f"Tool '{tool_call.tool_name}' result: {result_str}",
                        tool_call_id=tool_call.call_id,
                    )

                    if self._check_autonomous_completion(tool_call.tool_name):
                        logger.debug("Autonomous run completed at iteration %d", self._autonomous_iteration)
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Autonomous loop hit max iterations (%d) without finalization", max_iterations)
