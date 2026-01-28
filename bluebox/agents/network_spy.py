"""
bluebox/agents/network_spy.py

Agent specialized in searching through network HAR files.

Contains:
- NetworkSpyAgent: Conversational interface for HAR file analysis
- Uses: LLMClient with tools for HAR searching
- Maintains: ChatThread for multi-turn conversation
"""

import json
from datetime import datetime
from typing import Any, Callable

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
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.llm_client import LLMClient
from bluebox.llms.infra.har_data_store import HarDataStore
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class NetworkSpyAgent:
    """
    Network spy agent that helps analyze HAR (HTTP Archive) files.

    The agent maintains a ChatThread with Chat messages and uses LLM with tools
    to search and analyze network traffic captured in HAR format.

    Usage:
        def handle_message(message: EmittedMessage) -> None:
            print(f"[{message.type}] {message.content}")

        har_store = HarDataStore(open("network.har").read())
        agent = NetworkSpyAgent(
            emit_message_callable=handle_message,
            har_data_store=har_store,
        )
        agent.process_new_message("Find entries related to train prices", ChatRole.USER)
    """

    SYSTEM_PROMPT: str = """You are a network traffic analyst specializing in HAR (HTTP Archive) file analysis.

## Your Role

You help users find and analyze specific network requests in HAR files. Your main job is to:
- Find the HAR entry containing the data the user is looking for
- Identify API endpoints and their purposes
- Analyze request/response patterns

## Finding Relevant Entries

When the user asks about specific data (e.g., "train prices", "search results", "user data"):

1. Generate 20-30 relevant search terms that might appear in the response body
   - Include variations: singular/plural, different casings, related terms
   - Include data field names: "price", "amount", "cost", "fare", "total"
   - Include domain-specific terms: "departure", "arrival", "origin", "destination"

2. Use the `search_har_by_terms` tool with your terms

3. Analyze the top results - the entry with the highest score is most likely to contain the data

## Available Tools

- **`search_har_by_terms`**: Search HAR entries by a list of terms. Returns top 10 entries ranked by relevance.
  - Pass 20-30 search terms for best results
  - Only searches HTML/JSON response bodies (excludes JS, images, media)
  - Returns: id, url, unique_terms_found, total_hits, score

- **`get_entry_detail`**: Get full details of a specific HAR entry by ID.
  - Use this after finding a relevant entry to see headers, request body, response body

- **`get_entry_key_structure`**: Get only the key structure of a JSON response (no values).
  - Use this to understand the shape of large JSON responses without retrieving all the data
  - Shows all dict keys recursively, replaces values with null

## Guidelines

- Be concise and direct in your responses
- When you find a relevant entry, report its ID and URL
- Always use search_har_by_terms first when looking for specific data
"""

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        har_data_store: HarDataStore,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the network spy agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            har_data_store: The HarDataStore containing parsed HAR data.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._emit_message_callable = emit_message_callable
        self._persist_chat_callable = persist_chat_callable
        self._persist_chat_thread_callable = persist_chat_thread_callable
        self._stream_chunk_callable = stream_chunk_callable
        self._har_data_store = har_data_store
        self._previous_response_id: str | None = None
        self._response_id_to_chat_index: dict[str, int] = {}

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

        logger.debug(
            "Instantiated NetworkSpyAgent with model: %s, chat_thread_id: %s, entries: %d",
            llm_model,
            self._thread.id,
            len(har_data_store.entries),
        )

    def _register_tools(self) -> None:
        """Register tools for HAR analysis."""
        # search_har_by_terms
        self.llm_client.register_tool(
            name="search_har_by_terms",
            description=(
                "Search HAR entries by a list of terms. Searches HTML/JSON response bodies "
                "(excludes JS, images, media) and returns top 10 entries ranked by relevance score. "
                "Pass 20-30 search terms for best results."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of 20-30 search terms to look for in response bodies. "
                            "Include variations, related terms, and field names."
                        ),
                    }
                },
                "required": ["terms"],
            },
        )

        # get_entry_detail
        self.llm_client.register_tool(
            name="get_entry_detail",
            description=(
                "Get full details of a specific HAR entry by ID. "
                "Returns method, URL, headers, request body, and response body."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "integer",
                        "description": "The ID of the HAR entry to retrieve.",
                    }
                },
                "required": ["entry_id"],
            },
        )

        # get_entry_key_structure
        self.llm_client.register_tool(
            name="get_entry_key_structure",
            description=(
                "Get only the key structure of a HAR entry's JSON response. "
                "Shows all dict keys recursively without the actual values. "
                "Useful for understanding the shape of large JSON responses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "entry_id": {
                        "type": "integer",
                        "description": "The ID of the HAR entry to get key structure for.",
                    }
                },
                "required": ["entry_id"],
            },
        )

        # get_unique_urls
        self.llm_client.register_tool(
            name="get_unique_urls",
            description=(
                "Get all unique URLs from the HAR file. "
                "Returns a sorted list of all unique URLs observed in the traffic."
            ),
            parameters={
                "type": "object",
                "properties": {},
            },
        )

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    def _get_system_prompt(self) -> str:
        """Get system prompt with HAR stats context."""
        stats = self._har_data_store.stats
        stats_context = (
            f"\n\n## HAR File Context\n"
            f"- Total Requests: {stats.total_requests}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Hosts: {stats.unique_hosts}\n"
        )
        return self.SYSTEM_PROMPT + stats_context

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

        # Track response_id to chat index for O(1) lookup (only for ASSISTANT messages)
        if llm_provider_response_id and role == ChatRole.ASSISTANT:
            self._response_id_to_chat_index[llm_provider_response_id] = len(self._thread.chat_ids) - 1

        # Persist thread if callback provided
        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Build messages list for LLM from Chat objects."""
        messages: list[dict[str, Any]] = []

        # Determine which chats to include based on the previous response id
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
            # Include tool_call_id for TOOL role messages
            if chat.tool_call_id:
                msg["tool_call_id"] = chat.tool_call_id
            # Include tool_calls for ASSISTANT role messages
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

    def _tool_search_har_by_terms(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_har_by_terms tool."""
        terms = tool_arguments.get("terms", [])
        if not terms:
            return {"error": "No search terms provided"}

        results = self._har_data_store.search_entries_by_terms(terms, top_n=10)

        if not results:
            return {
                "message": "No matching entries found",
                "terms_searched": len(terms),
            }

        return {
            "terms_searched": len(terms),
            "results_found": len(results),
            "results": results,
        }

    def _tool_get_entry_detail(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_entry_detail tool."""
        entry_id = tool_arguments.get("entry_id")
        if entry_id is None:
            return {"error": "entry_id is required"}

        entry = self._har_data_store.get_entry(entry_id)
        if entry is None:
            return {"error": f"Entry {entry_id} not found"}

        # Truncate large response content
        response_content = entry.response_content
        if response_content and len(response_content) > 5000:
            response_content = response_content[:5000] + f"\n... (truncated, {len(entry.response_content)} total chars)"

        # Get key structure for JSON responses
        key_structure = self._har_data_store.get_entry_key_structure(entry_id)

        return {
            "id": entry.id,
            "method": entry.method,
            "url": entry.url,
            "status": entry.status,
            "status_text": entry.status_text,
            "mime_type": entry.mime_type,
            "request_headers": entry.request_headers,
            "response_headers": entry.response_headers,
            "query_params": entry.query_params,
            "post_data": entry.post_data,
            "response_content": response_content,
            "response_key_structure": key_structure,
        }

    def _tool_get_entry_key_structure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_entry_key_structure tool."""
        entry_id = tool_arguments.get("entry_id")
        if entry_id is None:
            return {"error": "entry_id is required"}

        key_structure = self._har_data_store.get_entry_key_structure(entry_id)
        if key_structure is None:
            entry = self._har_data_store.get_entry(entry_id)
            if entry is None:
                return {"error": f"Entry {entry_id} not found"}
            return {"error": f"Entry {entry_id} does not have valid JSON response content"}

        return {
            "entry_id": entry_id,
            "key_structure": key_structure,
        }

    def _tool_get_unique_urls(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_unique_urls tool."""
        url_counts = self._har_data_store.get_url_counts()
        return {
            "total_unique_urls": len(url_counts),
            "url_counts": url_counts,
        }

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        logger.debug("Executing tool %s with arguments: %s", tool_name, tool_arguments)

        if tool_name == "search_har_by_terms":
            return self._tool_search_har_by_terms(tool_arguments)

        if tool_name == "get_entry_detail":
            return self._tool_get_entry_detail(tool_arguments)

        if tool_name == "get_entry_key_structure":
            return self._tool_get_entry_key_structure(tool_arguments)

        if tool_name == "get_unique_urls":
            return self._tool_get_unique_urls(tool_arguments)

        return {"error": f"Unknown tool: {tool_name}"}

    def _auto_execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> str:
        """Auto-execute a tool and emit the result."""
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

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        Args:
            content: The message content
            role: The role of the message sender (USER or SYSTEM)
        """
        # Add message to history
        self._add_chat(role, content)

        # Run the agent loop
        self._run_agent_loop()

    def _run_agent_loop(self) -> None:
        """Run the agent loop: call LLM, execute tools, feed results back, repeat."""
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

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
                for tool_call in response.tool_calls:
                    tool_name = tool_call.tool_name
                    tool_arguments = tool_call.tool_arguments
                    call_id = tool_call.call_id

                    # Auto-execute tool
                    logger.debug("Auto-executing tool %s", tool_name)
                    result_str = self._auto_execute_tool(tool_name, tool_arguments)

                    # Add tool result to conversation history
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_name}' result: {result_str}",
                        tool_call_id=call_id,
                    )

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(
                    ErrorEmittedMessage(
                        error=str(e),
                    )
                )
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)

    def _process_streaming_response(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process LLM response with streaming, calling chunk callback for each chunk."""
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

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug(
            "Reset conversation from %s to %s",
            old_chat_thread_id,
            self._thread.id,
        )
