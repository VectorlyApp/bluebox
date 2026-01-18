"""
web_hacker/agents/guide_agent.py

Interactive agent for routine creation and editing.

Contains:
- GuideAgent: Conversational interface for building/modifying routines
- Uses: LLMClient with guide-specific tools
- Maintains: ChatThread for multi-turn conversation
"""

import json
from datetime import datetime
from typing import Any, Callable
from uuid import uuid4

from web_hacker.data_models.llms.interaction import (
    Chat,
    ChatMessageType,
    ChatRole,
    ChatThread,
    EmittedChatMessage,
    LLMChatResponse,
    LLMToolCall,
    PendingToolInvocation,
    ToolInvocationStatus,
)
from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.llms.llm_client import LLMClient
from web_hacker.llms.tools.guide_agent_tools import validate_routine
from web_hacker.routine_discovery.data_store import DiscoveryDataStore
from web_hacker.utils.exceptions import UnknownToolError
from web_hacker.utils.logger import get_logger


logger = get_logger(name=__name__)


class GuideAgentRoutineState:
    """
    Manages routine state for the Guide Agent.

    Tracks current routine, last execution, and queues update messages
    when state changes. Provides diff-checking so hosts can safely call
    update methods on every message without duplicate notifications.
    """

    def __init__(self) -> None:
        """Initialize empty routine state."""
        self.current_routine_json: dict | None = None
        self.last_executed_routine_json: dict | None = None
        self.last_executed_routine_params: dict | None = None
        self.last_executed_routine_result: dict | None = None
        self.update_messages: list[dict[str, Any]] = []

    def update_current_routine(self, routine_json: dict | None) -> None:
        """
        Update current routine if changed. No-op if unchanged.
        """
        if routine_json == self.current_routine_json:
            return
        was_none = self.current_routine_json is None
        self.current_routine_json = routine_json

        if routine_json is None:
            message = "Routine has been removed from context."
        elif was_none:
            message = "Routine added to context. Use get_current_routine_json to see the routine."
        else:
            message = "Routine has been updated. Use get_current_routine_json to see the changes."
            
        print(f"Routine state update: {message}")

        self.update_messages.append({
            "timestamp": int(datetime.now().timestamp()),
            "message": message,
        })

    def update_last_execution(
        self,
        routine_json: dict,
        routine_params: dict,
        routine_result: dict,
    ) -> None:
        """
        Update last execution state.
        """
        self.last_executed_routine_json = routine_json
        self.last_executed_routine_params = routine_params
        self.last_executed_routine_result = routine_result
        self.update_messages.append({
            "timestamp": int(datetime.now().timestamp()),
            "message": "Executed routine. To see the executed routine and parameters use the get_last_routine_execution tool. To see the result use the get_last_routine_execution_result tool."
        })

    def flush_update_messages(self) -> str | None:
        """
        Get formatted update messages and clear the queue.
        Returns None if no pending updates.
        """
        if not self.update_messages:
            return None
        updates = "\n".join(
            f"[System Update] {msg['message']}" for msg in self.update_messages
        )
        self.update_messages.clear()
        return updates

    def reset(self) -> None:
        """Reset all state."""
        self.current_routine_json = None
        self.last_executed_routine_json = None
        self.last_executed_routine_params = None
        self.last_executed_routine_result = None
        self.update_messages.clear()


class GuideAgent:
    """
    Guide agent that guides the user through the process of creating or editing a routine.

    The agent maintains a ChatThread with Chat messages and uses LLM tool-calling to determine
    when to initiate routine discovery. Tool invocations require user confirmation
    via callback before execution.

    Usage:
        def handle_message(message: EmittedChatMessage) -> None:
            print(f"[{message.type}] {message.content}")

        agent = GuideAgent(emit_message_callable=handle_message)
        agent.process_user_message("I want to search for flights")
    """

    # Class constants ______________________________________________________________________________________________________
    
    DATA_STORE_PROMPT: str = """
    You have access to the following data and you must refer to it when answering questions or helping debug!
    It is essecntial that you use this data, documentation, and code:
    {data_store_prompt}
    """

    SYSTEM_PROMPT: str = """You are a helpful assistant that guides users through creating \
web automation routines using the Web Hacker tool.

## What is Web Hacker?

https://github.com/vectorlyapp/web-hacker

Web Hacker is a tool that creates reusable web automation routines by learning from \
user demonstrations. Users record themselves performing a task on a website, and \
Web Hacker generates a parameterized routine that can be executed programmatically.

## What is Vectorly?

Vectorly (https://vectorly.app) unlocks data from interactive websites - getting web data behind \
clicks, searches, and user interactions. Define a routine once, then access it anywhere via API or MCP.

## Your Role

You help users in two ways:

### 1. Creating New Routines
Help users define their automation needs by gathering:
- **TASK**: What task to automate (e.g., "Search for train tickets")
- **OUTPUT**: What data to return (e.g., "List of trains with prices")
- **PARAMETERS**: Input parameters needed (name + description)
- **CONSTRAINTS**: Any filters (e.g., "Only direct trains")
- **WEBSITE**: Target website (e.g., "amtrak.com")

### 2. Debugging Existing Routines
Help users troubleshoot by reviewing:
- Routine JSON structure and operations
- Execution run logs and errors
- Parameter values and placeholder resolution
- Network transactions and responses

## Routine State Tools - USE THESE WHEN DEBUGGING
When a user asks for help debugging a routine or wants you to review their routine, use these tools:
- **`get_current_routine_json`**: Call this FIRST when the user asks about their routine or wants help editing it. Shows the routine JSON they're currently working on.
- **`get_last_routine_execution`**: Call this when the user says they ran a routine and it failed. Returns the routine JSON and parameters that were used in the last execution.
- **`get_last_routine_execution_result`**: Call this to see what actually happened during execution - success/failure status, output data, operation metadata, and error messages. Essential for diagnosing why a routine failed.

**Debugging workflow:**
1. User says "my routine failed" or "help me debug" → call `get_last_routine_execution` and `get_last_routine_execution_result`
2. User says "review my routine" or "what's wrong with my routine" → call `get_current_routine_json`
3. Analyze the results and cross-reference with documentation via file_search
4. Suggest specific fixes based on the error patterns

## Routine Validation - CRITICAL

**IMPORTANT**: Whenever you propose changes to an existing routine or generate a new routine, \
you MUST use the `validate_routine` tool to validate the complete routine JSON.

- Always validate before presenting a routine to the user
- If validation fails, carefully read the error message, fix the issues, and retry
- Keep retrying until validation passes (make at least 3 attempts if needed)
- Only present routines to the user that have passed validation

## Guidelines

- Be conversational and helpful
- Ask clarifying questions if needed
- Use the file_search tool to look up relevant documentation, code, and captured data
- When debugging, analyze the specific error and suggest concrete fixes (refer to debug docs and common issues)
- If the user asks what this tool does, explain it clearly"""

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedChatMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        data_store: DiscoveryDataStore | None = None,
        tools_requiring_approval: set[str] | None = None,
    ) -> None:
        """
        Initialize the guide agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects (for DynamoDB).
                Returns the Chat with the final ID assigned by the persistence layer.
            persist_chat_thread_callable: Optional callback to persist ChatThread (for DynamoDB).
                Returns the ChatThread with the final ID assigned by the persistence layer.
            stream_chunk_callable: Optional callback for streaming text chunks as they arrive.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            data_store: Optional data store for accessing CDP captures and documentation.
            tools_requiring_approval: Set of tool names that require user approval before execution.
                If empty or None, all tools auto-execute without approval.
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._data_store = data_store
        self._tools_requiring_approval = tools_requiring_approval or set()

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Register tools
        self._register_tools()

        # Configure file_search vectorstores if data store is provided
        if data_store:
            vector_store_ids = data_store.get_vectorstore_ids()
            if vector_store_ids:
                self.llm_client.set_file_search_vectorstores(vector_store_ids)

        # Initialize or load conversation state
        self._thread = chat_thread or ChatThread()
        self._chats: dict[str, Chat] = {}
        if existing_chats:
            for chat in existing_chats:
                self._chats[chat.id] = chat

        # Persist initial thread if callback provided
        if self._persist_chat_thread_callable and chat_thread is None:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.info(
            "Instantiated GuideAgent with model: %s, thread_id: %s",
            llm_model,
            self._thread.id,
        )

        # Initialize routine state
        self._routine_state = GuideAgentRoutineState()

    # Properties ___________________________________________________________________________________________________________

    @property
    def thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def has_pending_tool_invocation(self) -> bool:
        """Check if there's a pending tool invocation awaiting confirmation."""
        return self._thread.pending_tool_invocation is not None

    @property
    def data_store(self) -> DiscoveryDataStore | None:
        """Return the data store if available."""
        return self._data_store

    @property
    def tools_requiring_approval(self) -> set[str]:
        """Return the set of tool names that require user approval."""
        return self._tools_requiring_approval

    @property
    def routine_state(self) -> GuideAgentRoutineState:
        """Return the routine state manager."""
        return self._routine_state

    # Private methods ______________________________________________________________________________________________________

    def _get_system_prompt(self) -> str:
        """Get system prompt with data store context if available."""
        system_prompt = self.SYSTEM_PROMPT
        if self._data_store:
            data_store_prompt = self.DATA_STORE_PROMPT.format(data_store_prompt=self._data_store.generate_data_store_prompt())
            if data_store_prompt:
                system_prompt = f"{system_prompt}\n\n{data_store_prompt}"
        return system_prompt

    def _register_tools(self) -> None:
        """Register all tools with the LLM client."""
        # Register validate_routine with explicit schema
        self.llm_client.register_tool(
            name="validate_routine",
            description=(
                "Validates a routine JSON object against the Routine schema. "
                "You MUST pass the COMPLETE routine JSON object as routine_dict. "
                "If you have a routine from get_current_routine_json, pass that exact routine_json here."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routine_dict": {
                        "type": "object",
                        "description": (
                            "The complete routine JSON object to validate. Must contain: "
                            "name (string), description (string), parameters (array of parameter objects), "
                            "and operations (array of operation objects). "
                            "Pass the ENTIRE routine object, not individual fields."
                        ),
                    }
                },
                "required": ["routine_dict"],
            },
        )

        # Register routine state tools directly (no parameters needed, auto-execute)
        self.llm_client.register_tool(
            name="get_current_routine_json",
            description="Get the current routine JSON that the user is working on. Use this to see the routine before making edits or suggestions.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        self.llm_client.register_tool(
            name="get_last_routine_execution",
            description="Get the last executed routine JSON and the parameters that were used. Use this to understand what was run.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        self.llm_client.register_tool(
            name="get_last_routine_execution_result",
            description="Get the result of the last routine execution including success/failure status, output data, and any errors. Use this to debug execution issues.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

    def _emit_message(self, message: EmittedChatMessage) -> None:
        """Emit a message via the callback."""
        self._emit_message_callable(message)

    def _add_chat(
        self,
        role: ChatRole,
        content: str,
        tool_call_id: str | None = None,
        tool_calls: list[LLMToolCall] | None = None,
    ) -> Chat:
        """
        Create and store a new Chat, update thread, persist if callbacks set.

        Args:
            role: The role of the message sender.
            content: The content of the message.
            tool_call_id: For TOOL role, the call_id this is a response to.
            tool_calls: For ASSISTANT role, any tool calls made.

        Returns:
            The created Chat object (with final ID from persistence layer if callback provided).
        """
        chat = Chat(
            thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
        )

        # Persist chat first if callback provided (may assign new ID)
        if self._persist_chat_callable:
            chat = self._persist_chat_callable(chat)

        # Store with final ID
        self._chats[chat.id] = chat
        self._thread.message_ids.append(chat.id)
        self._thread.updated_at = int(datetime.now().timestamp())

        # Persist thread if callback provided
        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """
        Build messages list for LLM from Chat objects.

        Returns:
            List of message dicts with 'role', 'content', and optionally 'tool_call_id' or 'tool_calls' keys.
        """
        messages: list[dict[str, Any]] = []
        for chat_id in self._thread.message_ids:
            chat = self._chats.get(chat_id)
            if chat:
                msg: dict[str, Any] = {
                    "role": chat.role.value,
                    "content": chat.content,
                }
                # Include tool_call_id for TOOL role messages
                if chat.tool_call_id:
                    msg["tool_call_id"] = chat.tool_call_id
                # Include tool_calls for ASSISTANT role messages
                if chat.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "name": tc.tool_name,
                            "arguments": tc.tool_arguments,
                            "call_id": tc.call_id,
                        }
                        for tc in chat.tool_calls
                    ]
                messages.append(msg)
        return messages

    def _create_tool_invocation_request(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        call_id: str | None = None,
    ) -> PendingToolInvocation:
        """
        Create a tool invocation request for user confirmation.

        Args:
            tool_name: Name of the tool to invoke
            tool_arguments: Arguments for the tool
            call_id: LLM's call ID for this tool invocation (for Responses API)

        Returns:
            PendingToolInvocation stored in state and ready to emit
        """
        invocation_id = str(uuid4())
        pending = PendingToolInvocation(
            invocation_id=invocation_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            call_id=call_id,
        )

        # Store in thread
        self._thread.pending_tool_invocation = pending
        self._thread.updated_at = int(datetime.now().timestamp())

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.info(
            "Created tool invocation request: %s (tool: %s)",
            invocation_id,
            tool_name,
        )

        return pending

    def _execute_tool(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a confirmed tool invocation.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments for the tool

        Returns:
            Tool execution result with thread_id and params

        Raises:
            UnknownToolError: If tool_name is unknown
        """
        if tool_name == "validate_routine":
            logger.info("Executing tool %s with arguments: %s", tool_name, tool_arguments)
            routine_dict = tool_arguments.get("routine_dict", {})
            return validate_routine(routine_dict)

        if tool_name == "get_current_routine_json":
            logger.info("Executing tool %s", tool_name)
            if self._routine_state.current_routine_json is None:
                return {"error": "No current routine set. The user hasn't loaded or created a routine yet."}
            return {"routine_json": self._routine_state.current_routine_json}

        if tool_name == "get_last_routine_execution":
            logger.info("Executing tool %s", tool_name)
            if self._routine_state.last_executed_routine_json is None:
                return {"error": "No routine has been executed yet."}
            return {
                "routine_json": self._routine_state.last_executed_routine_json,
                "parameters": self._routine_state.last_executed_routine_params,
            }

        if tool_name == "get_last_routine_execution_result":
            logger.info("Executing tool %s", tool_name)
            if self._routine_state.last_executed_routine_result is None:
                return {"error": "No routine execution result available. No routine has been executed yet."}
            return {"result": self._routine_state.last_executed_routine_result}

        logger.error("Unknown tool \"%s\" with arguments: %s", tool_name, tool_arguments)
        raise UnknownToolError(f"Unknown tool \"{tool_name}\" with arguments: {tool_arguments}")

    def _auto_execute_tool(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> str:
        """
        Auto-execute a tool without user approval and emit the result.

        Args:
            tool_name: Name of the tool to execute
            tool_arguments: Arguments for the tool

        Returns:
            JSON string of the result (for feeding back to conversation)
        """
        # Create a transient invocation for tracking (not persisted as pending)
        invocation = PendingToolInvocation(
            invocation_id=str(uuid4()),
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            status=ToolInvocationStatus.CONFIRMED,
        )

        try:
            result = self._execute_tool(tool_name, tool_arguments)
            invocation.status = ToolInvocationStatus.EXECUTED

            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=invocation,
                    tool_result=result,
                )
            )

            logger.info(
                "Auto-executed tool %s successfully",
                tool_name,
            )

            return json.dumps(result)

        except Exception as e:
            invocation.status = ToolInvocationStatus.FAILED

            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=invocation,
                    error=str(e),
                )
            )

            logger.exception(
                "Auto-executed tool %s failed: %s",
                tool_name,
                e,
            )

            return json.dumps({"error": str(e)})

    # Public methods _______________________________________________________________________________________________________

    def process_user_message(self, content: str) -> None:
        """
        Process a user message and emit responses via callback.

        This method handles the agentic conversation loop:
        1. Adds user message to history
        2. Calls LLM to generate response
        3. If tool calls: execute tools, add results to history, call LLM again
        4. Repeat until LLM responds with text only (or tool needs approval)

        Args:
            content: The user's message content
        """
        # Block new messages if there's a pending tool invocation
        if self._thread.pending_tool_invocation:
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error="Please confirm or deny the pending tool invocation before sending new messages",
                )
            )
            return

        # Prepend any pending update messages to user content, then clear them
        updates = self._routine_state.flush_update_messages()
        if updates:
            content = f"{updates}\n\n{content}"

        # Add user message to history
        self._add_chat(ChatRole.USER, content)

        # Run the agentic loop
        self._run_agent_loop()

    def _run_agent_loop(self) -> None:
        """
        Run the agentic loop: call LLM, execute tools, feed results back, repeat.

        Continues until:
        - LLM responds with text only (no tool calls)
        - A tool requires user approval (pauses for confirmation)
        - An error occurs
        """
        max_iterations = 10  # Safety limit to prevent infinite loops

        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

            # Build messages and call LLM
            messages = self._build_messages_for_llm()

            try:
                # Use streaming if chunk callback is set
                if self._stream_chunk_callable:
                    response = self._process_streaming_response(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_system_prompt(),
                    )

                # Handle response - add assistant message if there's content or tool calls
                if response.content or response.tool_calls:
                    chat = self._add_chat(
                        ChatRole.ASSISTANT,
                        response.content or "",
                        tool_calls=response.tool_calls if response.tool_calls else None,
                    )
                    if response.content:
                        self._emit_message(
                            EmittedChatMessage(
                                type=ChatMessageType.CHAT_RESPONSE,
                                content=response.content,
                                chat_id=chat.id,
                                chat_thread_id=self._thread.id,
                            )
                        )

                # If no tool calls, we're done
                if not response.tool_calls:
                    logger.debug("Agent loop complete - no more tool calls")
                    return

                # Process tool calls
                tools_executed = False
                for tool_call in response.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments
                    call_id = tool_call.call_id

                    if tool_name in self._tools_requiring_approval:
                        # Tool requires user approval - pause the loop
                        pending = self._create_tool_invocation_request(
                            tool_name, tool_arguments, call_id
                        )
                        self._emit_message(
                            EmittedChatMessage(
                                type=ChatMessageType.TOOL_INVOCATION_REQUEST,
                                tool_invocation=pending,
                            )
                        )
                        logger.debug("Agent loop paused - awaiting tool approval")
                        return  # Pause loop until user confirms/denies

                    # Auto-execute tool
                    logger.info(
                        "Auto-executing tool %s with arguments: %s",
                        tool_name,
                        tool_arguments,
                    )
                    result_str = self._auto_execute_tool(tool_name, tool_arguments)

                    # Add tool result to conversation history for LLM to analyze
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_name}' result: {result_str}",
                        tool_call_id=call_id,
                    )
                    tools_executed = True

                # If we executed tools, continue the loop to let LLM analyze results
                if not tools_executed:
                    return

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(
                    EmittedChatMessage(
                        type=ChatMessageType.ERROR,
                        error=str(e),
                    )
                )
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)
        self._emit_message(
            EmittedChatMessage(
                type=ChatMessageType.ERROR,
                error=f"Agent loop exceeded maximum iterations ({max_iterations})",
            )
        )

    def _process_streaming_response(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """
        Process LLM response with streaming, calling chunk callback for each chunk.

        Args:
            messages: The messages to send to the LLM.

        Returns:
            The final LLMChatResponse with complete content.
        """
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=self._get_system_prompt(),
        ):
            if isinstance(item, str):
                # Text chunk - call the callback
                if self._stream_chunk_callable:
                    self._stream_chunk_callable(item)
            elif isinstance(item, LLMChatResponse):
                # Final response
                response = item

        if response is None:
            raise ValueError("No final response received from streaming LLM")

        return response

    def confirm_tool_invocation(self, invocation_id: str) -> None:
        """
        Confirm a pending tool invocation, execute it, and continue the agent loop.

        Args:
            invocation_id: ID of the tool invocation to confirm

        Emits:
            - TOOL_INVOCATION_RESULT with status "executed" and result on success
            - ERROR if no pending invocation or ID mismatch
        """
        pending = self._thread.pending_tool_invocation

        if not pending:
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error="No pending tool invocation to confirm",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error=f"Invocation ID mismatch: expected {pending.invocation_id}",
                )
            )
            return

        # Update status
        pending.status = ToolInvocationStatus.CONFIRMED
        tool_name = pending.tool_name

        try:
            result = self._execute_tool(pending.tool_name, pending.tool_arguments)
            pending.status = ToolInvocationStatus.EXECUTED
            self._thread.pending_tool_invocation = None
            self._thread.updated_at = int(datetime.now().timestamp())

            if self._persist_chat_thread_callable:
                self._thread = self._persist_chat_thread_callable(self._thread)

            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=pending,
                    tool_result=result,
                )
            )

            logger.info(
                "Tool invocation %s executed successfully",
                invocation_id,
            )

            # Add tool result to conversation and continue the agent loop
            result_str = json.dumps(result)
            self._add_chat(
                ChatRole.TOOL,
                f"Tool '{tool_name}' result: {result_str}",
                tool_call_id=pending.call_id,
            )
            self._run_agent_loop()

        except Exception as e:
            pending.status = ToolInvocationStatus.FAILED
            self._thread.pending_tool_invocation = None

            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.TOOL_INVOCATION_RESULT,
                    tool_invocation=pending,
                    error=str(e),
                )
            )

            logger.exception(
                "Tool invocation %s failed: %s",
                invocation_id,
                e,
            )

            # Add error to conversation and continue the agent loop
            self._add_chat(
                ChatRole.TOOL,
                f"Tool '{tool_name}' failed: {str(e)}",
                tool_call_id=pending.call_id,
            )
            self._run_agent_loop()

    def deny_tool_invocation(
        self,
        invocation_id: str,
        reason: str | None = None,
    ) -> None:
        """
        Deny a pending tool invocation.

        Args:
            invocation_id: ID of the tool invocation to deny
            reason: Optional reason for denial

        Emits:
            - TOOL_INVOCATION_RESULT with status "denied"
            - ERROR if no pending invocation or ID mismatch
        """
        pending = self._thread.pending_tool_invocation

        if not pending:
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error="No pending tool invocation to deny",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error=f"Invocation ID mismatch: expected {pending.invocation_id}",
                )
            )
            return

        # Update status and clear pending
        pending.status = ToolInvocationStatus.DENIED
        self._thread.pending_tool_invocation = None
        self._thread.updated_at = int(datetime.now().timestamp())

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        # Add denial to conversation history as tool response
        denial_message = f"User rejected the tool execution for '{pending.tool_name}'"
        if reason:
            denial_message += f". Reason: {reason}"
        self._add_chat(ChatRole.TOOL, denial_message, tool_call_id=pending.call_id)

        self._emit_message(
            EmittedChatMessage(
                type=ChatMessageType.TOOL_INVOCATION_RESULT,
                tool_invocation=pending,
                content=denial_message,
            )
        )

        logger.info(
            "Tool invocation %s denied: %s",
            invocation_id,
            reason or "no reason provided",
        )

        # Continue the agent loop so LLM can respond to the rejection
        self._run_agent_loop()

    def get_thread(self) -> ChatThread:
        """
        Get the current conversation thread.

        Returns:
            Current ChatThread
        """
        return self._thread

    def get_chats(self) -> list[Chat]:
        """
        Get all Chat messages in order.

        Returns:
            List of Chat objects in conversation order.
        """
        return [self._chats[chat_id] for chat_id in self._thread.message_ids if chat_id in self._chats]

    def reset(self) -> None:
        """
        Reset the conversation to a fresh state.

        Generates a new thread and clears all messages and routine state.
        """
        old_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._routine_state.reset()

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.info(
            "Reset conversation from %s to %s",
            old_thread_id,
            self._thread.id,
        )
