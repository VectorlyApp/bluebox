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
from enum import StrEnum
from typing import Any, Callable, Literal
from uuid import uuid4

from web_hacker.data_models.llms.interaction import (
    Chat,
    EmittedMessageType,
    ChatRole,
    ChatThread,
    EmittedMessage,
    ChatResponseEmittedMessage,
    ToolInvocationRequestEmittedMessage,
    ToolInvocationResultEmittedMessage,
    SuggestedEditEmittedMessage,
    BrowserRecordingRequestEmittedMessage,
    RoutineDiscoveryRequestEmittedMessage,
    RoutineCreationRequestEmittedMessage,
    ErrorEmittedMessage,
    LLMChatResponse,
    LLMToolCall,
    PendingToolInvocation,
    SuggestedEditRoutine,
    SuggestedEditUnion,
    ToolInvocationStatus,
)
from web_hacker.data_models.routine import Routine
from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.llms.llm_client import LLMClient
from web_hacker.llms.tools.guide_agent_tools import validate_routine
from web_hacker.routine_discovery.data_store import DiscoveryDataStore
from web_hacker.utils.exceptions import UnknownToolError
from web_hacker.utils.logger import get_logger


logger = get_logger(name=__name__)


class GuideAgentMode(StrEnum):
    """Operating mode for the Guide Agent."""
    CREATION = "creation"  # No routine loaded - help create one
    EDITING = "editing"    # Routine loaded - help debug/edit


class GuideAgentRoutineState:
    """
    Manages routine state for the Guide Agent.

    Tracks current routine and last execution using timestamps to record
    when state changes. On flush, generates a single message per category
    (routine change, execution) based on which timestamps exist.
    """

    def __init__(self) -> None:
        """Initialize empty routine state."""
        # Store as string to preserve raw content even if JSON is invalid
        self.current_routine_str: str | None = None
        # Execution-related fields stay as dict (only set after successful execution)
        self.last_execution_routine: dict | None = None
        self.last_execution_parameters: dict | None = None
        self.last_execution_result: dict | None = None
        # Timestamps for tracking state changes (None = no pending update)
        self._routine_change_at: int | None = None
        self._routine_change_type: Literal["added", "updated", "removed"] | None = None
        self._execution_at: int | None = None

    def update_current_routine(self, routine_str: str | None) -> None:
        """
        Update current routine string if changed. No-op if unchanged.
        Stores raw string to preserve content even if JSON is invalid.
        Records timestamp and change type for later message generation.
        """
        if routine_str == self.current_routine_str:
            return
        was_none = self.current_routine_str is None
        self.current_routine_str = routine_str

        # Determine change type and record timestamp
        if routine_str is None:
            change_type = "removed"
        elif was_none:
            change_type = "added"
        else:
            change_type = "updated"

        self._routine_change_at = int(datetime.now().timestamp())
        self._routine_change_type = change_type

    def update_last_execution(
        self,
        routine: dict,
        parameters: dict,
        result: dict,
    ) -> None:
        """
        Update last execution state and record timestamp.
        """
        self.last_execution_routine = routine
        self.last_execution_parameters = parameters
        self.last_execution_result = result
        self._execution_at = int(datetime.now().timestamp())

    def flush_update_messages(self) -> str | None:
        """
        Generate update messages based on which timestamps exist, then clear them.
        Returns None if no pending updates.
        Only generates ONE message per category (routine change, execution).
        """
        messages: list[str] = []

        # Check for routine change
        if self._routine_change_at is not None and self._routine_change_type is not None:
            if self._routine_change_type == "removed":
                messages.append("[System Update] Routine has been removed from context.")
            elif self._routine_change_type == "added":
                messages.append("[System Update] Routine added to context. Use get_current_routine to see the routine.")
            else:  # updated
                messages.append("[System Update] Routine has been updated. Use get_current_routine to see the changes.")
            # Reset timestamp
            self._routine_change_at = None
            self._routine_change_type = None

        # Check for execution
        if self._execution_at is not None:
            messages.append("[System Update] Executed routine. To see the executed routine and parameters use the get_last_routine_execution tool. To see the result use the get_last_routine_execution_result tool.")
            # Reset timestamp
            self._execution_at = None

        if not messages:
            return None
        return "\n".join(messages)

    def reset(self) -> None:
        """Reset all state."""
        self.current_routine_str = None
        self.last_execution_routine = None
        self.last_execution_parameters = None
        self.last_execution_result = None
        self._routine_change_at = None
        self._routine_change_type = None
        self._execution_at = None


class GuideAgent:
    """
    Guide agent that guides the user through the process of creating or editing a routine.

    The agent maintains a ChatThread with Chat messages and uses LLM tool-calling to determine
    when to initiate routine discovery. Tool invocations require user confirmation
    via callback before execution.

    Usage:
        def handle_message(message: EmittedMessage) -> None:
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

    CREATION_MODE_SYSTEM_PROMPT: str = """You are a helpful assistant that helps users create \
web automation routines.

## What are Routines?

Routines are reusable web automation workflows that can be executed programmatically. \
They are created by learning from user demonstrations - users record themselves performing \
a task on a website, and the system generates a parameterized routine.

## What is Vectorly?

Vectorly (https://vectorly.app) unlocks data from interactive websites - getting web data behind \
clicks, searches, and user interactions. Define a routine once, then access it anywhere via API or MCP.

## Your Role

You are in CREATION MODE. Help users create new routines by:
- Understanding what task they want to automate
- Guiding them through browser recording to capture their workflow
- Running routine discovery to generate the routine from captured data
- Creating routines directly when appropriate

## Available Tools

- **`request_user_browser_recording`**: Ask the user to demonstrate a task in the browser. \
Use this when the user describes a web automation task they want to create.
- **`request_routine_discovery`**: Start routine discovery from captured browser data. \
Use this after recording is complete.
- **`create_new_routine`**: Create a routine directly without discovery. Use this when you \
have enough information to build the routine programmatically.
- **`file_search`**: Search documentation for routine creation best practices.

## Workflow for Creating Routines

1. **Understand the task**: Ask the user what website data they want to access or what actions they want to automate.
2. **Initiate recording**: Use `request_user_browser_recording` with a clear task description.
3. **Wait for recording**: The user will perform the task while browser activity is captured.
4. **Run discovery**: Use `request_routine_discovery` to generate a routine from the captures.
5. **Review result**: Once the routine is created, you will switch to editing mode to help refine it.

## Guidelines

- Be conversational and helpful
- Ask clarifying questions to understand exactly what the user wants to automate (VERY CONSICE AND TO THE POINT!)
- Provide clear, bulleted instructions when requesting browser recordings
- If the user asks about an existing routine, inform them no routine is currently loaded
- BE VERY CONCISE AND TO THE POINT. We DONT NEED LONG CONVERSATIONS!
- IMPORTANT: When you decide to use a tool, JUST CALL IT. Do NOT announce "I will now call X" or \
"Let me use X tool" - just invoke the tool directly. The user can always decline the request.

## NOTES:
- Quotes or escaped quotes are ESSENTIAL AROUND {{{{parameter_name}}}} ALL parameters in routines!
- Before saying ANYTHING ABOUT QUOTES OR ESCAPED QUOTES, you MUST look through the docs!

## System Action Messages
When you receive a system message with the prefix "[ACTION REQUIRED]", you MUST immediately \
execute the requested action using the appropriate tools.
"""

    EDITING_MODE_SYSTEM_PROMPT: str = """You are a helpful assistant that helps users debug \
and improve web automation routines.

## What are Routines?

Routines are reusable web automation workflows that can be executed programmatically. \
They are created by learning from user demonstrations - users record themselves performing \
a task on a website, and the system generates a parameterized routine.

## What is Vectorly?

Vectorly (https://vectorly.app) unlocks data from interactive websites - getting web data behind \
clicks, searches, and user interactions. Define a routine once, then access it anywhere via API or MCP.

## Your Role

You are in EDITING MODE. A routine is currently loaded. Help users by:
- Reviewing the routine structure and operations
- Debugging execution failures
- Suggesting improvements and fixes
- Validating routine changes

## Available Tools - USE THESE WHEN DEBUGGING

- **`get_current_routine`**: Get the currently loaded routine JSON. Call this FIRST when the user \
asks about their routine or wants help editing it.
- **`get_last_routine_execution`**: Get the last executed routine and parameters used. Call when \
the user says they ran a routine and it failed.
- **`get_last_routine_execution_result`**: Get execution results - success/failure status, output \
data, and errors. Essential for debugging.
- **`validate_routine`**: Validate a routine object against the schema. REQUIRED KEY: 'routine'.
- **`suggest_routine_edit`**: Propose changes to the routine for user approval. REQUIRED KEY: 'routine' \
with the COMPLETE routine object.
- **`file_search`**: Search documentation for debugging tips and common issues.

## Debugging Workflow

1. User says "my routine failed" or "help me debug" -> call `get_last_routine_execution` and \
`get_last_routine_execution_result`
2. User says "review my routine" or "what's wrong" -> call `get_current_routine`
3. Analyze the results and cross-reference with documentation via file_search
4. Suggest specific fixes using `suggest_routine_edit`

## Suggesting Routine Edits

When proposing changes, use the `suggest_routine_edit` tool:
- **REQUIRED KEY: `routine`** - Pass the COMPLETE routine object under this key
- Example: {{"routine": {{"name": "...", "description": "...", "parameters": [...], "operations": [...]}}}}
- The tool validates automatically - you do NOT need to call `validate_routine` first
- If validation fails, read the error, fix the routine, and try again (make at least 3 attempts)

## Guidelines

- Be conversational and helpful
- Ask clarifying questions if needed (VERY CONCISE AND TO THE POINT!)
- When debugging, analyze the specific error and suggest concrete fixes
- Use file_search to reference documentation for complex issues
- BE VERY CONCISE AND TO THE POINT. We DONT NEED LONG CONVERSATIONS!
- IMPORTANT: When you decide to use a tool, JUST CALL IT. Do NOT announce "I will now call X" or \
"Let me use X tool" - just invoke the tool directly.

## NOTES:
- Quotes or escaped quotes are ESSENTIAL AROUND {{{{parameter_name}}}} ALL parameters in routines!
- Before saying ANYTHING ABOUT QUOTES OR ESCAPED QUOTES, you MUST look through the docs!

## System Action Messages
When you receive a system message with the prefix "[ACTION REQUIRED]", you MUST immediately \
execute the requested action using the appropriate tools.
"""

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        persist_suggested_edit_callable: Callable[[SuggestedEditUnion], SuggestedEditUnion] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        data_store: DiscoveryDataStore | None = None,
        tools_requiring_approval: set[str] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize the guide agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects (for DynamoDB).
                Returns the Chat with the final ID assigned by the persistence layer.
            persist_chat_thread_callable: Optional callback to persist ChatThread (for DynamoDB).
                Returns the ChatThread with the final ID assigned by the persistence layer.
            persist_suggested_edit_callable: Optional callback to persist suggested edit objects.
                Returns the SuggestedEditUnion with the final ID assigned by the persistence layer.
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
        self._persist_suggested_edit_callable = persist_suggested_edit_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._data_store = data_store
        self._tools_requiring_approval = tools_requiring_approval or set()
        self._custom_system_prompt = system_prompt  # None means use mode-based prompts
        self._previous_response_id: str | None = None

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Initialize routine state first (needed for mode determination)
        self._routine_state = GuideAgentRoutineState()

        # Initialize mode and register appropriate tools
        self._current_mode: GuideAgentMode = self._determine_mode()
        self._register_tools(self._current_mode)

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
            "Instantiated GuideAgent with model: %s, chat_thread_id: %s, mode: %s",
            llm_model,
            self._thread.id,
            self._current_mode.value,
        )

    # Properties ___________________________________________________________________________________________________________

    @property
    def chat_thread_id(self) -> str:
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

    @property
    def current_mode(self) -> GuideAgentMode:
        """Return the current operating mode."""
        return self._current_mode

    # Private methods ______________________________________________________________________________________________________

    def _get_system_prompt(self) -> str:
        """Get system prompt based on current mode with data store context."""
        # Use custom prompt if provided, otherwise select based on mode
        if self._custom_system_prompt is not None:
            system_prompt = self._custom_system_prompt
        elif self._current_mode == GuideAgentMode.CREATION:
            system_prompt = self.CREATION_MODE_SYSTEM_PROMPT
        else:
            system_prompt = self.EDITING_MODE_SYSTEM_PROMPT

        # Append data store context
        if self._data_store:
            data_store_prompt = self.DATA_STORE_PROMPT.format(
                data_store_prompt=self._data_store.generate_data_store_prompt()
            )
            if data_store_prompt:
                system_prompt = f"{system_prompt}\n\n{data_store_prompt}"
        return system_prompt

    def _determine_mode(self) -> GuideAgentMode:
        """Determine the appropriate mode based on routine state."""
        if self._routine_state.current_routine_str is not None:
            return GuideAgentMode.EDITING
        return GuideAgentMode.CREATION

    def _check_and_switch_mode(self) -> bool:
        """
        Check if mode should switch and perform switch if needed.

        Returns:
            True if mode was switched, False otherwise.
        """
        new_mode = self._determine_mode()
        if new_mode == self._current_mode:
            return False

        old_mode = self._current_mode
        self._current_mode = new_mode

        # Re-register tools for new mode
        self._register_tools(new_mode)

        logger.info(
            "GuideAgent mode switched from %s to %s",
            old_mode.value,
            new_mode.value,
        )

        return True

    def _register_tools(self, mode: GuideAgentMode) -> None:
        """Register tools appropriate for the given mode."""
        # Clear existing tools
        self.llm_client.clear_tools()

        if mode == GuideAgentMode.CREATION:
            self._register_creation_mode_tools()
        else:
            self._register_editing_mode_tools()

        logger.debug("Registered tools for %s mode", mode.value)

    def _register_creation_mode_tools(self) -> None:
        """Register tools for creation mode (no routine loaded)."""
        # request_user_browser_recording
        self.llm_client.register_tool(
            name="request_user_browser_recording",
            description=(
                "Request the user to perform a browser recording session. "
                "Use this when you need the user to demonstrate a web task so it can be captured "
                "and turned into a routine. The user will navigate to a website and perform "
                "actions while browser activity (navigation, network requests, cookies, etc.) is recorded."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Description of what the user should do during the recording. "
                            "E.g. 'Search for one-way flights from NYC to LA on March 15'"
                        ),
                    }
                },
                "required": ["task_description"],
            },
        )

        # request_routine_discovery
        self.llm_client.register_tool(
            name="request_routine_discovery",
            description=(
                "Request to start routine discovery from the captured browser data. "
                "Use this after browser recording is complete and you've verified the captures contain the needed data. "
                "This will analyze the network transactions and create a reusable routine."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "data_output": {
                        "type": "string",
                        "description": "What data the routine should return (e.g., 'flight prices', 'search results')",
                    },
                    "parameters": {
                        "type": "string",
                        "description": "What parameters/inputs the routine needs (e.g., 'departure city, arrival city, date')",
                    },
                    "website": {
                        "type": "string",
                        "description": "The website URL where the data was captured",
                    },
                },
                "required": ["data_output"],
            },
        )

        # create_new_routine
        self.llm_client.register_tool(
            name="create_new_routine",
            description=(
                "Create and save a new routine directly. Use this when you want to create a routine "
                "from scratch without going through the discovery process. The routine will be "
                "saved to a file and loaded into the current context. "
                "REQUIRED KEY: 'routine' - the COMPLETE routine object. "
                "Example: {\"routine\": {\"name\": \"...\", \"description\": \"...\", \"parameters\": [...], \"operations\": [...]}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routine": {
                        "type": "object",
                        "description": (
                            "REQUIRED. The complete routine object to create. "
                            "Must contain keys: name (string), description (string), parameters (array), operations (array)."
                        ),
                    }
                },
                "required": ["routine"],
            },
        )

    def _register_editing_mode_tools(self) -> None:
        """Register tools for editing mode (routine is loaded)."""
        # get_current_routine
        self.llm_client.register_tool(
            name="get_current_routine",
            description="Get the current routine JSON that the user is working on. No arguments required.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        # get_last_routine_execution
        self.llm_client.register_tool(
            name="get_last_routine_execution",
            description="Get the last executed routine JSON and the parameters that were used. No arguments required.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        # get_last_routine_execution_result
        self.llm_client.register_tool(
            name="get_last_routine_execution_result",
            description="Get the result of the last routine execution including success/failure status, output data, and any errors. No arguments required.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        # validate_routine
        self.llm_client.register_tool(
            name="validate_routine",
            description=(
                "Validates a routine JSON object against the Routine schema. "
                "REQUIRED KEY: 'routine' - the COMPLETE routine JSON object. "
                "Example: {\"routine\": {\"name\": \"...\", \"description\": \"...\", \"parameters\": [...], \"operations\": [...]}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routine": {
                        "type": "object",
                        "description": (
                            "REQUIRED. The complete routine JSON object to validate. "
                            "Must contain keys: name (string), description (string), parameters (array), operations (array)."
                        ),
                    }
                },
                "required": ["routine"],
            },
        )

        # suggest_routine_edit
        self.llm_client.register_tool(
            name="suggest_routine_edit",
            description=(
                "Suggest an edited/improved routine for user approval. "
                "REQUIRED KEY: 'routine' - the COMPLETE routine object. "
                "Example: {\"routine\": {\"name\": \"...\", \"description\": \"...\", \"parameters\": [...], \"operations\": [...]}}"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "routine": {
                        "type": "object",
                        "description": (
                            "REQUIRED. The complete routine object to suggest. "
                            "Must contain keys: name (string), description (string), parameters (array), operations (array)."
                        ),
                    }
                },
                "required": ["routine"],
            },
        )

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
            chat_thread_id=self._thread.id,
            role=role,
            content=content,
            tool_call_id=tool_call_id,
            tool_calls=tool_calls or [],
            llm_provider_response_id=llm_provider_response_id,
        )

        # Persist chat first if callback provided (may assign new ID)
        if self._persist_chat_callable:
            chat = self._persist_chat_callable(chat)

        # Store with final ID
        self._chats[chat.id] = chat
        self._thread.chat_ids.append(chat.id)
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
        
        # determine which chats to include based on the previous response id
        chats_to_include = self._thread.chat_ids
        if self._previous_response_id is not None:
            
            # get index of the previous response id
            index = None
            for idx, chat_id in enumerate(self._thread.chat_ids):
                if self._chats.get(chat_id).llm_provider_response_id == self._previous_response_id:
                    index = idx
                    break
                
            # include all chats after the previous response id
            if index is not None:
                chats_to_include = self._thread.chat_ids[index + 1:]
        
        
        for chat_id in chats_to_include:
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

    def _tool_validate_routine(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute validate_routine tool."""
        # Accept both "routine" and "routine_dict" keys for flexibility
        routine_dict = tool_arguments.get("routine") or tool_arguments.get("routine_dict")
        # Fallback: if no nested key, try using tool_arguments directly as the routine
        if not routine_dict and "name" in tool_arguments and "operations" in tool_arguments:
            routine_dict = tool_arguments
        if not routine_dict:
            raise ValueError("routine was empty. Pass the COMPLETE routine JSON object under the 'routine' key.")
        result = validate_routine(routine_dict)
        if not result.get("valid"):
            raise ValueError(result.get("error", "Validation failed"))
        return result

    def _tool_get_current_routine(self) -> dict[str, Any]:
        """Execute get_current_routine tool."""
        if self._routine_state.current_routine_str is None:
            return {"error": "No current routine set. The user hasn't loaded or created a routine yet."}
        # Try to parse string as JSON, fallback to raw content if invalid
        try:
            parsed = json.loads(self._routine_state.current_routine_str)
            return parsed  # Return the routine directly, not wrapped
        except json.JSONDecodeError as e:
            return {
                "error": f"Invalid JSON: {e}",
                "raw_content": self._routine_state.current_routine_str
            }

    def _tool_get_last_routine_execution(self) -> dict[str, Any]:
        """Execute get_last_routine_execution tool."""
        if self._routine_state.last_execution_routine is None:
            return {"error": "No routine has been executed yet."}
        return {
            "routine_json": self._routine_state.last_execution_routine,
            "parameters": self._routine_state.last_execution_parameters,
        }

    def _tool_get_last_routine_execution_result(self) -> dict[str, Any]:
        """Execute get_last_routine_execution_result tool."""
        if self._routine_state.last_execution_result is None:
            return {"error": "No routine execution result available. No routine has been executed yet."}
        return {"result": self._routine_state.last_execution_result}

    def _tool_request_user_browser_recording(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute request_user_browser_recording tool."""
        task_description = tool_arguments.get("task_description", "")
        if not task_description:
            raise ValueError("task_description is required.")

        self._emit_message(
            BrowserRecordingRequestEmittedMessage(
                browser_recording_task=task_description,
                chat_thread_id=self._thread.id,
            )
        )

        return {
            "success": True,
            "message": (
                "Browser recording request sent to user. "
                "Give the user brief bulleted instructions on what to do:\n"
                "- A new browser tab will open\n"
                "- Navigate to <WEBSITE>\n"
                "- Perform the following: <STEPS>\n"
                "- Ensure requested data is located in the browser tab."
            ),
        }

    def _tool_suggest_routine_edit(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute suggest_routine_edit tool."""
        # Accept both "routine" and "routine_dict" keys for flexibility
        routine_dict = tool_arguments.get("routine") or tool_arguments.get("routine_dict")
        # Fallback: if no nested key, try using tool_arguments directly as the routine
        if not routine_dict and "name" in tool_arguments and "operations" in tool_arguments:
            routine_dict = tool_arguments
        if not routine_dict:
            raise ValueError("routine was empty. Pass the COMPLETE routine object and try again.")

        # Create Routine object and SuggestedEditRoutine
        try:
            routine = Routine(**routine_dict)
        except Exception as e:
            raise ValueError(f"Invalid routine object: {e}. Fix the routine object and try again.")

        suggested_edit = SuggestedEditRoutine(
            chat_thread_id=self._thread.id,
            routine=routine,
        )

        # Persist suggested edit if callback provided (may assign new ID)
        if self._persist_suggested_edit_callable:
            suggested_edit = self._persist_suggested_edit_callable(suggested_edit)

        # Add to thread's suggested_edit_ids and persist thread
        self._thread.suggested_edit_ids.append(suggested_edit.id)
        self._thread.updated_at = int(datetime.now().timestamp())
        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        # Emit the suggested edit for host to handle
        self._emit_message(
            SuggestedEditEmittedMessage(
                suggested_edit=suggested_edit,
                chat_thread_id=self._thread.id,
            )
        )

        return {
            "success": True,
            "message": (
                "Edit suggested and sent to user for approval. "
                "Pay attention to changes in the routine to see if the user accepted the edits or not. "
                "Give the user a very brief summary (diffs) of the changes."
            ),
            "edit_id": suggested_edit.id,
        }

    def _tool_request_routine_discovery(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute request_routine_discovery tool."""
        data_output = tool_arguments.get("data_output", "")
        parameters = tool_arguments.get("parameters", "")
        website = tool_arguments.get("website", "")

        if not data_output:
            raise ValueError("data_output is required - what should the routine return?")

        # Build task description
        task_parts = [f"Create a web routine that returns {data_output}"]
        if parameters:
            task_parts.append(f"given {parameters}")
        if website:
            task_parts.append(f"from {website}")
        task = " ".join(task_parts) + "."

        self._emit_message(
            RoutineDiscoveryRequestEmittedMessage(
                routine_discovery_task=task,
                chat_thread_id=self._thread.id,
            )
        )

        return {
            "success": True,
            "message": (
                "Routine discovery request sent. "
                "The user will confirm to start the discovery process. "
                "Give the user a very brief summary of the discovery process: "
                "find network transactions relevant to the request, "
                "extract and resolve parameters, cookies, tokens, etc., "
                "and construct the routine."
            ),
            "task": task,
        }

    def _tool_create_new_routine(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute create_new_routine tool."""
        # Accept both "routine" and "routine_dict" keys for flexibility
        routine_dict = tool_arguments.get("routine") or tool_arguments.get("routine_dict")
        # Fallback: if no nested key, try using tool_arguments directly as the routine
        if not routine_dict and "name" in tool_arguments and "operations" in tool_arguments:
            routine_dict = tool_arguments
        if not routine_dict:
            raise ValueError("routine was empty. Pass the COMPLETE routine object under the 'routine' key.")

        # Create Routine object to validate
        try:
            routine = Routine(**routine_dict)
        except Exception as e:
            raise ValueError(f"Invalid routine object: {e}. Fix the routine object and try again.")

        # Emit the routine creation request for host to handle
        self._emit_message(
            RoutineCreationRequestEmittedMessage(
                created_routine=routine,
                chat_thread_id=self._thread.id,
            )
        )

        return {
            "success": True,
            "message": (
                f"Routine '{routine.name}' creation request sent. "
                "The user will now be asked to confirm the routine creation. "
                "At this point give the user a very brief summary of the routine."
            ),
            "routine_name": routine.name,
        }

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
            Tool execution result with chat_thread_id and params

        Raises:
            UnknownToolError: If tool_name is unknown
        """
        logger.info("Executing tool %s with arguments: %s", tool_name, tool_arguments)

        if tool_name == "validate_routine":
            return self._tool_validate_routine(tool_arguments)

        if tool_name == "get_current_routine":
            return self._tool_get_current_routine()

        if tool_name == "get_last_routine_execution":
            return self._tool_get_last_routine_execution()

        if tool_name == "get_last_routine_execution_result":
            return self._tool_get_last_routine_execution_result()

        if tool_name == "suggest_routine_edit":
            return self._tool_suggest_routine_edit(tool_arguments)

        if tool_name == "request_user_browser_recording":
            return self._tool_request_user_browser_recording(tool_arguments)

        if tool_name == "request_routine_discovery":
            return self._tool_request_routine_discovery(tool_arguments)

        if tool_name == "create_new_routine":
            return self._tool_create_new_routine(tool_arguments)

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
                ToolInvocationResultEmittedMessage(
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
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result={"error": str(e)},
                )
            )

            logger.error(
                "Auto-executed tool %s failed: %s",
                tool_name,
                e,
            )

            return json.dumps({"error": str(e)})

    # Public methods _______________________________________________________________________________________________________

    def process_user_message(self, content: str) -> None:
        """Process a user message. Convenience wrapper for process_new_message."""
        self.process_new_message(content, ChatRole.USER)

    def notify_browser_recording_complete(self, transaction_count: int) -> None:
        """
        Notify the agent that browser recording has completed.

        Sends a system message instructing the agent to scan the captured data
        and initiate routine discovery if appropriate.

        Args:
            transaction_count: Number of network transactions captured.
        """
        system_message = (
            "[ACTION REQUIRED] Browser recording completed. "
            "New CDP captures are now available in the vectorstore. "
            f"{transaction_count} network transactions were captured. "
            "Scan the captured data NOW using file_search to verify it contains "
            "the information needed for the user's requested automation. "
            "Examine the consolidated_transactions.json and confirm the relevant API endpoints, "
            "request/response payloads, and any authentication data are present. "
            "If you see the data related to the requested task, initiate a routine discovery process. "
            "Otherwise you can ask the user for clarifying questions or re-request the browser recording."
        )
        self.process_new_message(system_message, ChatRole.SYSTEM)

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        This method handles the agentic conversation loop:
        1. Adds the message to history
        2. Calls LLM to generate response
        3. If tool calls: execute tools, add results to history, call LLM again
        4. Repeat until LLM responds with text only (or tool needs approval)

        Args:
            content: The message content
            role: The role of the message sender (USER or SYSTEM)
        """
        # Block new messages if there's a pending tool invocation
        if self._thread.pending_tool_invocation:
            self._emit_message(
                ErrorEmittedMessage(
                    error="Please confirm or deny the pending tool invocation before sending new messages",
                )
            )
            return

        # Add any pending update messages as a system message
        system_update = self._routine_state.flush_update_messages()
        if system_update:
            self._add_chat(ChatRole.SYSTEM, system_update)

        # Add message to history
        self._add_chat(role, content)

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
        # Check for mode switch before running
        self._check_and_switch_mode()

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
                        previous_response_id=self._previous_response_id,
                    )

                # Update previous_response_id for response chaining
                if response.response_id:
                    self._previous_response_id = response.response_id

                # Handle response - add assistant message if there's content or tool calls
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
                            ToolInvocationRequestEmittedMessage(
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
                        llm_provider_response_id=response.response_id,
                    )
                    tools_executed = True

                # If we executed tools, continue the loop to let LLM analyze results
                if not tools_executed:
                    return

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(
                    ErrorEmittedMessage(
                        error=str(e),
                    )
                )
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)
        self._emit_message(
            ErrorEmittedMessage(
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
            previous_response_id=self._previous_response_id,
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
                ErrorEmittedMessage(
                    error="No pending tool invocation to confirm",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                ErrorEmittedMessage(
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
                ToolInvocationResultEmittedMessage(
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
                ToolInvocationResultEmittedMessage(
                    tool_invocation=pending,
                    tool_result={"error": str(e)},
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
                ErrorEmittedMessage(
                    error="No pending tool invocation to deny",
                )
            )
            return

        if pending.invocation_id != invocation_id:
            self._emit_message(
                ErrorEmittedMessage(
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
        self._add_chat(
            ChatRole.TOOL,
            denial_message,
            tool_call_id=pending.call_id,
        )

        self._emit_message(
            ToolInvocationResultEmittedMessage(
                tool_invocation=pending,
                tool_result={"denied": True, "message": denial_message},
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
        return [self._chats[chat_id] for chat_id in self._thread.chat_ids if chat_id in self._chats]

    def reset(self) -> None:
        """
        Reset the conversation to a fresh state.

        Generates a new thread and clears all messages and routine state.
        """
        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._routine_state.reset()

        # Reset mode to CREATION and re-register tools
        self._current_mode = GuideAgentMode.CREATION
        self._register_tools(self._current_mode)

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.info(
            "Reset conversation from %s to %s",
            old_chat_thread_id,
            self._thread.id,
        )

    def refresh_vectorstores(self) -> None:
        """
        Refresh the file_search vectorstores from the data store.

        Call this after adding new vectorstores to the data store (e.g., after
        creating a CDP captures vectorstore from browser recording).
        """
        if self._data_store:
            vector_store_ids = self._data_store.get_vectorstore_ids()
            if vector_store_ids:
                self.llm_client.set_file_search_vectorstores(vector_store_ids)
                logger.info("Refreshed vectorstores: %s", vector_store_ids)
            else:
                self.llm_client.set_file_search_vectorstores(None)
                logger.info("Cleared vectorstores (none available)")
