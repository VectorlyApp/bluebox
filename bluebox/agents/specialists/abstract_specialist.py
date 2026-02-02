"""
bluebox/agents/specialists/abstract_specialist.py

Abstract base class for specialist agents.

Specialists are domain-expert agents that an orchestrator deploys for specific tasks.
Each specialist owns:
  - A system prompt (conversational + autonomous variants)
  - A set of LLM tools and their execution logic
  - Finalize tools for autonomous mode (registered after min_iterations)

The base class provides all shared LLM conversation plumbing:
  - Chat/thread management and persistence
  - Conversational agent loop (process_new_message)
  - Autonomous agent loop with finalize gating (run_autonomous)
  - Streaming support
  - Message emission
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Callable

from pydantic import BaseModel

from bluebox.data_models.llms.interaction import (
    Chat,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    LLMChatResponse,
    LLMToolCall,
    ToolInvocationResultEmittedMessage,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.llm_client import LLMClient
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class SpecialistMode(StrEnum):
    """Lifecycle mode of a specialist agent."""
    CONVERSATIONAL = "conversational"  # interactive chat with a user
    AUTONOMOUS = "autonomous"          # autonomous loop, exploring/analyzing
    FINALIZING = "finalizing"          # autonomous loop, finalize tools available


@dataclass(frozen=True)
class _ToolMeta:
    """Metadata attached to a handler method by @specialist_tool."""
    name: str                                           # tool name registered with the LLM client
    description: str                                    # tool description shown to the LLM
    parameters: dict[str, Any]                          # JSON Schema for tool parameters
    availability: bool | Callable[..., bool]            # whether the tool should be registered right now


def specialist_tool(
    description: str,
    parameters: dict[str, Any],
    *,
    availability: bool | Callable[..., bool] = True,
) -> Callable:
    """
    Decorator that marks a method as a specialist tool handler.

    The tool name is derived from the method name by stripping leading
    underscores. For example, ``get_dom_snapshot`` becomes ``get_dom_snapshot_tool``.

    Args:
        description: Tool description for the LLM.
        parameters: JSON Schema for tool parameters.
        availability: Controls when the tool is registered with the LLM.
            - True (default): always registered at init.
            - Callable[[self], bool]: evaluated each time _register_tools() is
              called; registered only when it returns True. Use this for tools
              gated behind optional dependencies or lifecycle state (e.g.
              ``availability=lambda self: self.mode == SpecialistMode.FINALIZING``).
    """
    def decorator(method: Callable) -> Callable:
        tool_name = method.__name__
        method._tool_meta = _ToolMeta(
            name=tool_name,
            description=description,
            parameters=parameters,
            availability=availability,
        )
        return method
    return decorator


class AbstractSpecialist(ABC):
    """
    Abstract base class for specialist agents.

    Subclasses implement domain-specific logic by overriding:
      - _get_system_prompt()
      - _get_autonomous_system_prompt()
      - _get_autonomous_initial_message()
      - _check_autonomous_completion() — inspect tool results for finalize signals

    Tools are defined declaratively via the @specialist_tool decorator on handler
    methods. Each tool's ``availability`` controls when it is registered: True
    (always), or a callable evaluated each time _register_tools() is called.
    The base class provides default _register_tools() and _execute_tool() that
    use decorated methods. Subclasses may still override these for custom logic.

    The base class handles all LLM conversation mechanics:
      - Chat history, threading, persistence
      - Conversational and autonomous agent loops
      - Streaming
      - Tool auto-execution and message emission
    """

    ## Abstract methods

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for conversational (interactive) mode."""

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

    ## Class methods

    @classmethod
    def _collect_tools(cls) -> list[tuple[_ToolMeta, Callable]]:
        """Collect all methods decorated with @specialist_tool."""
        results: list[tuple[_ToolMeta, Callable]] = []
        for attr_name in dir(cls):
            method = getattr(cls, attr_name, None)
            tool_meta = getattr(method, "_tool_meta", None)
            if isinstance(tool_meta, _ToolMeta):
                results.append((tool_meta, method))
        return results

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
    ) -> None:
        """
        Initialize the specialist.

        Args:
            emit_message_callable: Callback to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use.
            chat_thread: Existing ChatThread to continue, or None for new.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._previous_response_id: str | None = None
        self._response_id_to_chat_index: dict[str, int] = {}

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Track which tools have been registered to avoid double-registration
        self._registered_tool_names: set[str] = set()

        # Register tools whose availability is satisfied at init time
        self._register_tools()

        # Conversation state
        self._thread = chat_thread or ChatThread()
        self._chats: dict[str, Chat] = {}
        if existing_chats:
            for chat in existing_chats:
                self._chats[chat.id] = chat

        # Persist initial thread if callback provided
        if self._persist_chat_thread_callable and chat_thread is None:
            self._thread = self._persist_chat_thread_callable(self._thread)

        # Lifecycle state
        self.mode: SpecialistMode = SpecialistMode.CONVERSATIONAL
        self._autonomous_iteration: int = 0

    ## Tools

    def _register_tools(self) -> None:
        """
        Register specialist-specific tools on self.llm_client.

        Iterates all @specialist_tool methods, evaluates each tool's
        ``availability``, and registers any newly-available tools that
        haven't been registered yet. Safe to call multiple times (e.g.
        at init and again when finalize gate opens).
        """
        for tool_meta, _ in self._collect_tools():
            if tool_meta.name in self._registered_tool_names:
                continue
            available = tool_meta.availability(self) if callable(tool_meta.availability) else tool_meta.availability
            if not available:
                continue
            self.llm_client.register_tool(
                name=tool_meta.name,
                description=tool_meta.description,
                parameters=tool_meta.parameters,
            )
            self._registered_tool_names.add(tool_meta.name)

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Dispatch a tool call to the appropriate decorated handler.

        Default implementation builds a dispatch dict from @specialist_tool
        methods. Subclasses may override for custom dispatch logic.
        """
        dispatch = {tool_meta.name: method for tool_meta, method in self._collect_tools()}
        handler = dispatch.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        return handler(self, tool_arguments)

    ## Properties

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    ## Chat management (concrete)

    def _emit_message(self, message: EmittedMessage) -> None:
        """Emit a message via the callback."""
        self._emit_message_callable(message)

    def _add_chat(
        self,
        role: ChatRole,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[LLMToolCall] | None = None,
        llm_provider_response_id: str | None = None,
    ) -> Chat:
        """Create and store a new Chat, update thread, persist if callbacks set."""
        chat = Chat(
            chat_thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
            llm_provider_response_id=llm_provider_response_id,
        )

        if self._persist_chat_callable:
            chat = self._persist_chat_callable(chat)

        self._chats[chat.id] = chat
        self._thread.chat_ids.append(chat.id)
        self._thread.updated_at = int(datetime.now().timestamp())

        # Track response_id → chat index for O(1) lookup (ASSISTANT messages only)
        if llm_provider_response_id and role == ChatRole.ASSISTANT:
            self._response_id_to_chat_index[llm_provider_response_id] = len(self._thread.chat_ids) - 1

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Build messages list for LLM from Chat objects."""
        messages: list[dict[str, Any]] = []

        # Only include chats after the last response_id (for response chaining)
        chats_to_include = self._thread.chat_ids
        if self._previous_response_id is not None:
            index = self._response_id_to_chat_index.get(self._previous_response_id)
            if index is not None:
                chats_to_include = self._thread.chat_ids[index + 1:]

        for chat_id in chats_to_include:
            chat = self._chats.get(chat_id)
            if not chat:
                continue
            msg: dict[str, Any] = {
                "role": chat.role.value,
                "content": chat.content,
            }
            if chat.tool_call_id:
                msg["tool_call_id"] = chat.tool_call_id
            if chat.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.tool_arguments) if isinstance(tc.tool_arguments, dict) else tc.tool_arguments,
                        },
                    }
                    for tc in chat.tool_calls
                ]
            messages.append(msg)
        return messages

    ## Tools

    def _auto_execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Auto-execute a tool, emit result message, return JSON string."""
        invocation = PendingToolInvocation(
            invocation_id="",
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            status=ToolInvocationStatus.CONFIRMED,
        )

        try:
            result = self._execute_tool(tool_name, tool_arguments)
            invocation.status = ToolInvocationStatus.EXECUTED

            self._emit_message(
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result=result,
                )
            )

            logger.debug("Auto-executed tool %s successfully", tool_name)
            return json.dumps(result)

        except Exception as e:
            invocation.status = ToolInvocationStatus.FAILED

            self._emit_message(
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result={"error": str(e)},
                )
            )

            logger.error("Auto-executed tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    ## Conversational mode (concrete)

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        Args:
            content: The message content.
            role: The role of the message sender.
        """
        self._add_chat(role, content)
        self._run_agent_loop()

    def _run_agent_loop(self) -> None:
        """Run the conversational agent loop: call LLM → execute tools → repeat."""
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

            messages = self._build_messages_for_llm()

            try:
                response = self._call_llm(messages, self._get_system_prompt())

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        ChatRole.ASSISTANT,
                        response.content or "",
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
                    logger.debug("Agent loop complete - no more tool calls")
                    return

                self._process_tool_calls(response.tool_calls)

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)

    ## Autonomous mode (concrete)

    def run_autonomous(
        self,
        task: str,
        min_iterations: int = 3,
        max_iterations: int = 10,
    ) -> BaseModel | None:
        """
        Run the specialist autonomously to completion.

        The specialist will:
        1. Use its tools to explore and analyze data
        2. After min_iterations, finalize tools become available
        3. Return a typed result when finalize is called, or None on timeout

        Args:
            task: User task description.
            min_iterations: Minimum iterations before finalize tools are available.
            max_iterations: Maximum iterations before stopping.

        Returns:
            Specialist-specific result model, or None if max iterations reached.
        """
        self.mode = SpecialistMode.AUTONOMOUS
        self._autonomous_iteration = 0

        # Subclass should reset its own result fields in _reset_autonomous_state()
        self._reset_autonomous_state()

        # Seed the conversation
        initial_message = self._get_autonomous_initial_message(task)
        self._add_chat(ChatRole.USER, initial_message)

        logger.info("Starting autonomous run for task: %s", task)

        self._run_autonomous_loop(min_iterations, max_iterations)

        self.mode = SpecialistMode.CONVERSATIONAL

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

    def _run_autonomous_loop(self, min_iterations: int, max_iterations: int) -> None:
        """Run the autonomous agent loop with iteration tracking and finalize gating."""
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug("Autonomous loop iteration %d/%d", self._autonomous_iteration, max_iterations)

            # Gate finalize tools behind min_iterations
            if self._autonomous_iteration >= min_iterations - 1 and self.mode != SpecialistMode.FINALIZING:
                self.mode = SpecialistMode.FINALIZING
                self._register_tools()  # re-evaluate availability lambdas
                logger.info("Finalize tools now available (iteration %d)", self._autonomous_iteration)

            messages = self._build_messages_for_llm()

            try:
                response = self._call_llm(messages, self._get_autonomous_system_prompt())

                if response.response_id:
                    self._previous_response_id = response.response_id

                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        ChatRole.ASSISTANT,
                        response.content or "",
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
                        ChatRole.TOOL,
                        f"Tool '{tool_call.tool_name}' result: {result_str}",
                        tool_call_id=tool_call.call_id,
                    )

                    if self._check_autonomous_completion(tool_call.tool_name):
                        logger.info("Autonomous run completed at iteration %d", self._autonomous_iteration)
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Autonomous loop hit max iterations (%d) without finalization", max_iterations)

    ## Streaming helpers (concrete)

    def _call_llm(self, messages: list[dict[str, Any]], system_prompt: str) -> LLMChatResponse:
        """Call the LLM, using streaming if a chunk callback is configured."""
        if self._stream_chunk_callable:
            return self._process_streaming_response(messages, system_prompt)

        return self.llm_client.call_sync(
            messages=messages,
            system_prompt=system_prompt,
            previous_response_id=self._previous_response_id,
        )

    def _process_streaming_response(
        self,
        messages: list[dict[str, Any]],
        system_prompt: str,
    ) -> LLMChatResponse:
        """Process LLM response with streaming, calling chunk callback for each chunk."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=system_prompt,
            previous_response_id=self._previous_response_id,
        ):
            if isinstance(item, str):
                if self._stream_chunk_callable:
                    self._stream_chunk_callable(item)
            elif isinstance(item, LLMChatResponse):
                response = item

        if response is None:
            raise ValueError("No final response received from streaming LLM")

        return response

    ## Shared helpers (concrete)

    def _process_tool_calls(self, tool_calls: list[LLMToolCall]) -> None:
        """Execute a list of tool calls and add results to chat history."""
        for tool_call in tool_calls:
            logger.debug("Auto-executing tool %s", tool_call.tool_name)
            result_str = self._auto_execute_tool(tool_call.tool_name, tool_call.tool_arguments)

            self._add_chat(
                ChatRole.TOOL,
                f"Tool '{tool_call.tool_name}' result: {result_str}",
                tool_call_id=tool_call.call_id,
            )

    def get_thread(self) -> ChatThread:
        """Get the current conversation thread."""
        return self._thread

    def get_chats(self) -> list[Chat]:
        """Get all Chat messages in order."""
        return [self._chats[chat_id] for chat_id in self._thread.chat_ids if chat_id in self._chats]

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._previous_response_id = None
        self._response_id_to_chat_index = {}

        self.mode = SpecialistMode.CONVERSATIONAL
        self._autonomous_iteration = 0
        self._registered_tool_names = set()
        self._reset_autonomous_state()

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug("Reset conversation from %s to %s", old_chat_thread_id, self._thread.id)
