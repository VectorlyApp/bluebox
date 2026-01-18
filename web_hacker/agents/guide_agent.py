"""
web_hacker/agents/guide_agent.py

Guide agent that guides the user through the process of creating or editing a routine.
"""

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
    PendingToolInvocation,
    ToolInvocationStatus,
)
from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.llms.llm_client import LLMClient
from web_hacker.routine_discovery.data_store import DiscoveryDataStore
from web_hacker.utils.exceptions import UnknownToolError
from web_hacker.utils.logger import get_logger


logger = get_logger(name=__name__)


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

    SYSTEM_PROMPT: str = """You are a helpful assistant that guides users through creating \
web automation routines using the Web Hacker tool.

## What is Web Hacker?

Web Hacker is a tool that creates reusable web automation routines by learning from \
user demonstrations. Users record themselves performing a task on a website, and \
Web Hacker generates a parameterized routine that can be executed programmatically.

## Your Role

Your job is to help users define their automation needs by gathering:

1. **TASK**: What task do they want to automate?
   - Examples: "Search for train tickets", "Download a research paper", "Look up company info"

2. **OUTPUT**: What data/output should the routine return?
   - Examples: "List of available trains with prices", "PDF file of the paper", "Company registration details"

3. **PARAMETERS**: What input parameters will the routine need? For each:
   - Name (e.g., "origin", "destination", "departure_date")
   - Description (e.g., "The departure station name")

4. **CONSTRAINTS**: Any filters or constraints?
   - Examples: "Only direct trains", "Papers from 2024", "Only active companies"

5. **WEBSITE**: What website should be used?
   - Examples: "amtrak.com", "arxiv.org", "sec.gov"

## Guidelines

- Be conversational and helpful
- Ask clarifying questions if needed
- Don't overwhelm the user with too many questions at once
- When you have enough information, use the start_routine_discovery_job_creation tool
- If the user asks what this tool does, explain it clearly"""

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedChatMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_MINI,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        data_store: DiscoveryDataStore | None = None,
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
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._data_store = data_store

        self.llm_model = llm_model
        self.llm_client = LLMClient(llm_model)

        # Register tools
        self._register_tools()

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

    # Private methods ______________________________________________________________________________________________________

    def _build_file_search_tools(self) -> list[dict[str, Any]]:
        """Build file_search tools from data store vectorstores."""
        if not self._data_store:
            return []

        vector_store_ids = self._data_store.get_vectorstore_ids()
        if not vector_store_ids:
            return []

        return [{
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        }]

    def _get_system_prompt(self) -> str:
        """Get system prompt with data store context if available."""
        system_prompt = self.SYSTEM_PROMPT
        if self._data_store:
            context = self._data_store.generate_data_store_prompt()
            if context:
                system_prompt = f"{system_prompt}\n\n{context}"
        return system_prompt

    def _register_tools(self) -> None:
        """Register all tools with the LLM client."""
        # TODO: Register tools when implemented
        pass

    def _emit_message(self, message: EmittedChatMessage) -> None:
        """Emit a message via the callback."""
        self._emit_message_callable(message)

    def _add_chat(self, role: ChatRole, content: str) -> Chat:
        """
        Create and store a new Chat, update thread, persist if callbacks set.

        Args:
            role: The role of the message sender.
            content: The content of the message.

        Returns:
            The created Chat object (with final ID from persistence layer if callback provided).
        """
        chat = Chat(
            thread_id=self._thread.id,
            role=role,
            content=content,
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

    def _build_messages_for_llm(self) -> list[dict[str, str]]:
        """
        Build messages list for LLM from Chat objects.

        Returns:
            List of message dicts with 'role' and 'content' keys.
        """
        messages: list[dict[str, str]] = []
        for chat_id in self._thread.message_ids:
            chat = self._chats.get(chat_id)
            if chat:
                messages.append({
                    "role": chat.role.value,
                    "content": chat.content,
                })
        return messages

    def _create_tool_invocation_request(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
    ) -> PendingToolInvocation:
        """
        Create a tool invocation request for user confirmation.

        Args:
            tool_name: Name of the tool to invoke
            tool_arguments: Arguments for the tool

        Returns:
            PendingToolInvocation stored in state and ready to emit
        """
        invocation_id = str(uuid4())
        pending = PendingToolInvocation(
            invocation_id=invocation_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
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
        if tool_name == start_routine_discovery_job_creation.__name__:
            logger.info(
                "Executing tool %s with args: %s",
                tool_name,
                tool_arguments,
            )
            result = start_routine_discovery_job_creation(**tool_arguments)
            return {
                "thread_id": self._thread.id,
                **result,
            }

        logger.error("Unknown tool \"%s\" with arguments: %s", tool_name, tool_arguments)
        raise UnknownToolError(f"Unknown tool \"{tool_name}\" with arguments: {tool_arguments}")

    # Public methods _______________________________________________________________________________________________________

    def process_user_message(self, content: str) -> None:
        """
        Process a user message and emit responses via callback.

        This method handles the conversation loop:
        1. Adds user message to history
        2. Calls LLM to generate response
        3. Emits chat response or tool invocation request

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

        # Add user message to history
        self._add_chat(ChatRole.USER, content)

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
                    additional_tools=self._build_file_search_tools(),
                )

            # Handle text response
            if response.content:
                chat = self._add_chat(ChatRole.ASSISTANT, response.content)
                # Always emit CHAT_RESPONSE (handler checks if streaming occurred)
                self._emit_message(
                    EmittedChatMessage(
                        type=ChatMessageType.CHAT_RESPONSE,
                        content=response.content,
                        chat_id=chat.id,
                        chat_thread_id=self._thread.id,
                    )
                )

            # Handle tool call if present
            if response.tool_call:
                pending = self._create_tool_invocation_request(
                    response.tool_call.tool_name,
                    response.tool_call.tool_arguments,
                )
                self._emit_message(
                    EmittedChatMessage(
                        type=ChatMessageType.TOOL_INVOCATION_REQUEST,
                        tool_invocation=pending,
                    )
                )

        except Exception as e:
            logger.exception("Error processing user message: %s", e)
            self._emit_message(
                EmittedChatMessage(
                    type=ChatMessageType.ERROR,
                    error=str(e),
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
            additional_tools=self._build_file_search_tools(),
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
        Confirm a pending tool invocation and execute it.

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

        except Exception as e:
            pending.status = ToolInvocationStatus.FAILED

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

        # Add denial to conversation history
        denial_message = "Tool invocation denied"
        if reason:
            denial_message += f": {reason}"
        self._add_chat(ChatRole.SYSTEM, denial_message)

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

        Generates a new thread and clears all messages.
        """
        old_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.info(
            "Reset conversation from %s to %s",
            old_thread_id,
            self._thread.id,
        )
