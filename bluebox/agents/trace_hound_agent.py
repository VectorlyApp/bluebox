"""
bluebox/agents/trace_hound_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in tracing where tokens/values originated from.

Contains:
- TraceHoundAgent: Traces values across network, storage, and window property data
- TokenOriginResult: Result model for autonomous token tracing
- Uses: LLMClient with tools for cross-store value searching
- Maintains: ChatThread for multi-turn conversation
"""

import json
import textwrap
from datetime import datetime
from typing import Any, Callable

from pydantic import BaseModel, Field

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
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.llms.infra.storage_data_store import StorageDataStore
from bluebox.llms.infra.window_property_data_store import WindowPropertyDataStore
from bluebox.utils.code_execution_sandbox import execute_python_sandboxed
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class TokenOrigin(BaseModel):
    """A single discovered origin for a token/value."""

    source_type: str = Field(
        description="Where the value was found: 'network', 'storage', or 'window_property'"
    )
    location: str = Field(
        description="Specific location (URL for network, origin+key for storage, path for window props)"
    )
    context: str = Field(
        description="Brief context about how the value appears (e.g., 'in response body', 'cookie value')"
    )
    entry_id: str | int = Field(
        description="ID to retrieve the full entry (request_id for network, index for others)"
    )


class TokenOriginResult(BaseModel):
    """Result of autonomous token origin tracing."""

    value_searched: str = Field(
        description="The token/value that was searched for"
    )
    origins: list[TokenOrigin] = Field(
        description="List of discovered origins for the value"
    )
    likely_source: TokenOrigin | None = Field(
        default=None,
        description="The most likely original source of the value (earliest/most authoritative)"
    )
    explanation: str = Field(
        description="Explanation of how the value flows through the system"
    )


class TokenOriginFailure(BaseModel):
    """Result when token origin tracing fails to find the value."""

    value_searched: str = Field(
        description="The token/value that was searched for"
    )
    reason: str = Field(
        description="Explanation of why the value could not be found"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for alternative searches or next steps"
    )


class TraceHoundAgent:
    """
    Trace hound agent that traces where tokens/values originated from.

    Searches across network traffic, browser storage (cookies, localStorage,
    sessionStorage, IndexedDB), and window object properties to find where
    a specific value first appeared and how it propagates.
    """

    SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a token origin specialist that traces where values come from in web traffic.

        ## Your Role

        You help users find where specific tokens, IDs, or values originated from by searching:
        - **Network traffic**: HTTP requests/responses (headers, bodies, URLs)
        - **Browser storage**: Cookies, localStorage, sessionStorage, IndexedDB
        - **Window properties**: JavaScript window object properties

        ## Strategy

        When tracing a value:
        1. Search across ALL data sources using `search_everywhere`
        2. If found, examine the entries to understand the context
        3. Determine the ORIGINAL source (where it first appeared)
        4. Trace how it propagates (e.g., response -> cookie -> request header)

        ## Available Tools

        - **`search_everywhere`**: Search for a value across ALL data stores at once. Start here.
        - **`search_in_network`**: Search network traffic response bodies for a value.
        - **`search_in_storage`**: Search browser storage (cookies, localStorage, etc.) for a value.
        - **`search_in_window_props`**: Search window object properties for a value.
        - **`get_network_entry`**: Get full details of a network entry by request_id.
        - **`get_storage_entry`**: Get full details of a storage entry by index.
        - **`get_window_prop_changes`**: Get changes for a specific window property path.
        - **`get_storage_by_key`**: Get storage entries by key name (to find what value is stored under a key).
        - **`execute_python`**: Run custom Python code to query data flexibly. Pre-loaded: network_entries, storage_entries, window_prop_entries.

        ## Guidelines

        - Always start with `search_everywhere` to get a complete picture
        - Look at timestamps to determine the order of events
        - Consider that values often flow: API response -> storage -> subsequent requests
        - Be concise and direct when reporting findings
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a token origin specialist that autonomously traces where values come from.

        ## Your Mission

        Given a specific token/value, find its ORIGINAL source and trace how it propagates
        through the system (network -> storage -> subsequent usage).

        ## Process

        1. **Search**: Use `search_everywhere` to find all occurrences of the value
        2. **Analyze**: Examine entries to understand context and timestamps
        3. **Trace**: Determine the flow (e.g., API response -> cookie -> request header)
        4. **Finalize**: Call `finalize_result` with your findings

        ## What to Look For

        - First occurrence (by timestamp) is often the original source
        - Network responses often set values that end up in storage
        - Storage values (cookies) are often sent in subsequent request headers
        - Window properties may be set by scripts after network responses

        ## When finalize tools are available

        After sufficient exploration, call `finalize_result` with:
        - The value searched
        - All discovered origins (source_type, location, context, entry_id)
        - The likely original source
        - An explanation of the value flow

        If the value cannot be found anywhere, call `finalize_failure` with an explanation.
    """).strip()

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        network_data_store: NetworkDataStore | None = None,
        storage_data_store: StorageDataStore | None = None,
        window_property_data_store: WindowPropertyDataStore | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the trace hound agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            network_data_store: NetworkDataStore for network traffic analysis.
            storage_data_store: StorageDataStore for browser storage analysis.
            window_property_data_store: WindowPropertyDataStore for window props analysis.
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

        # Data stores
        self._network_data_store = network_data_store
        self._storage_data_store = storage_data_store
        self._window_property_data_store = window_property_data_store

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

        # Autonomous mode state
        self._autonomous_mode: bool = False
        self._autonomous_iteration: int = 0
        self._autonomous_max_iterations: int = 10
        self._origin_result: TokenOriginResult | None = None
        self._origin_failure: TokenOriginFailure | None = None
        self._finalize_tool_registered: bool = False

        logger.debug(
            "Instantiated TraceHoundAgent with model: %s, chat_thread_id: %s",
            llm_model,
            self._thread.id,
        )

    def _register_tools(self) -> None:
        """Register tools for cross-store value searching."""

        # search_everywhere
        self.llm_client.register_tool(
            name="search_everywhere",
            description=(
                "Search for a value across ALL data stores (network, storage, window properties). "
                "This is the best starting point when tracing where a value came from. "
                "Returns matches from each store with context about where the value was found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The token/value to search for.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive. Defaults to false.",
                    }
                },
                "required": ["value"],
            },
        )

        # search_in_network
        self.llm_client.register_tool(
            name="search_in_network",
            description=(
                "Search network traffic response bodies for a specific value. "
                "Returns matches with context showing where in the response the value appears."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to search for in response bodies.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive. Defaults to false.",
                    }
                },
                "required": ["value"],
            },
        )

        # search_in_storage
        self.llm_client.register_tool(
            name="search_in_storage",
            description=(
                "Search browser storage (cookies, localStorage, sessionStorage, IndexedDB) for a value. "
                "Returns matches showing which storage type and key contains the value."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to search for in storage.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive. Defaults to false.",
                    }
                },
                "required": ["value"],
            },
        )

        # search_in_window_props
        self.llm_client.register_tool(
            name="search_in_window_props",
            description=(
                "Search window object property values for a specific value. "
                "Returns matches showing which property path contains the value."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The value to search for in window properties.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive. Defaults to false.",
                    }
                },
                "required": ["value"],
            },
        )

        # get_network_entry
        self.llm_client.register_tool(
            name="get_network_entry",
            description=(
                "Get full details of a network entry by request_id. "
                "Returns method, URL, headers, request body, and response body."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "The request_id of the network entry to retrieve.",
                    }
                },
                "required": ["request_id"],
            },
        )

        # get_storage_entry
        self.llm_client.register_tool(
            name="get_storage_entry",
            description=(
                "Get full details of a storage entry by index. "
                "Returns the storage event with all its fields."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "The index of the storage entry to retrieve.",
                    }
                },
                "required": ["index"],
            },
        )

        # get_window_prop_changes
        self.llm_client.register_tool(
            name="get_window_prop_changes",
            description=(
                "Get all changes for a specific window property path. "
                "Returns the history of changes (added, changed, deleted) for that property."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The property path to get changes for (e.g., 'dataLayer.0.userId').",
                    },
                    "exact": {
                        "type": "boolean",
                        "description": "If true, match path exactly. If false, match paths containing the substring. Defaults to false.",
                    }
                },
                "required": ["path"],
            },
        )

        # get_storage_by_key
        self.llm_client.register_tool(
            name="get_storage_by_key",
            description=(
                "Get all storage entries for a specific key name. "
                "Use this when you want to find the VALUE stored under a given KEY "
                "(e.g., 'get the value of _cb_svref_expires'). "
                "Returns all events where this key was set, modified, or deleted."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "The storage key name to look up.",
                    }
                },
                "required": ["key"],
            },
        )

        # execute_python
        self.llm_client.register_tool(
            name="execute_python",
            description=(
                "Execute Python code in a sandboxed environment to analyze data. "
                "Pre-loaded variables: "
                "`network_entries` (list of NetworkTransactionEvent dicts with request_id, url, method, status, "
                "request_headers, response_headers, post_data, response_body), "
                "`storage_entries` (list of StorageEvent dicts with type, origin, key, value, etc.), "
                "`window_prop_entries` (list of WindowPropertyEvent dicts with url, timestamp, changes). "
                "Use print() to output results. Example: "
                "for e in storage_entries: if e['key'] == 'token': print(e)"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. Variables available: network_entries, storage_entries, "
                            "window_prop_entries. The `json` module is available. Use print() for output. "
                            "Note: imports are disabled for security."
                        ),
                    }
                },
                "required": ["code"],
            },
        )

    def _register_finalize_tools(self) -> None:
        """Register finalize tools for autonomous mode."""
        if self._finalize_tool_registered:
            return

        self.llm_client.register_tool(
            name="finalize_result",
            description=(
                "Finalize the token origin tracing with your findings. "
                "Call this when you have traced where the value came from."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value_searched": {
                        "type": "string",
                        "description": "The token/value that was searched for.",
                    },
                    "origins": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_type": {
                                    "type": "string",
                                    "enum": ["network", "storage", "window_property"],
                                    "description": "Where the value was found.",
                                },
                                "location": {
                                    "type": "string",
                                    "description": "Specific location (URL, origin+key, or path).",
                                },
                                "context": {
                                    "type": "string",
                                    "description": "Brief context about how the value appears.",
                                },
                                "entry_id": {
                                    "type": "string",
                                    "description": "ID to retrieve the full entry.",
                                },
                            },
                            "required": ["source_type", "location", "context", "entry_id"],
                        },
                        "description": "List of all discovered origins.",
                    },
                    "likely_source_index": {
                        "type": "integer",
                        "description": "Index in origins array of the most likely original source (or -1 if unclear).",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "Explanation of how the value flows through the system.",
                    },
                },
                "required": ["value_searched", "origins", "explanation"],
            },
        )

        self.llm_client.register_tool(
            name="finalize_failure",
            description=(
                "Signal that the token origin tracing failed to find the value. "
                "Call this when you have searched thoroughly but cannot find the value."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value_searched": {
                        "type": "string",
                        "description": "The token/value that was searched for.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explanation of why the value could not be found.",
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Suggestions for alternative searches.",
                    },
                },
                "required": ["value_searched", "reason"],
            },
        )

        self._finalize_tool_registered = True
        logger.debug("Registered finalize_result and finalize_failure tools")

    def _sync_tools(self, include_finalize: bool = False) -> None:
        """Sync registered tools to match current mode.

        Clears all tools and re-registers the appropriate set:
        - Always: base search/get tools
        - If include_finalize is True: also registers finalize tools

        Args:
            include_finalize: Whether to include finalize_result and finalize_failure tools.
        """
        self.llm_client.clear_tools()
        self._finalize_tool_registered = False
        self._register_tools()

        if include_finalize:
            self._register_finalize_tools()

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    def _get_system_prompt(self) -> str:
        """Get system prompt with data store context."""
        context_parts = [self.SYSTEM_PROMPT, "\n\n## Data Store Context"]

        if self._network_data_store:
            stats = self._network_data_store.stats
            context_parts.append(
                f"\n- Network: {stats.total_requests} requests, {stats.unique_urls} unique URLs"
            )
        else:
            context_parts.append("\n- Network: Not available")

        if self._storage_data_store:
            stats = self._storage_data_store.stats
            context_parts.append(
                f"\n- Storage: {stats.total_events} events "
                f"(cookies: {stats.cookie_events}, localStorage: {stats.local_storage_events}, "
                f"sessionStorage: {stats.session_storage_events})"
            )
        else:
            context_parts.append("\n- Storage: Not available")

        if self._window_property_data_store:
            stats = self._window_property_data_store.stats
            context_parts.append(
                f"\n- Window Props: {stats.total_events} events, "
                f"{stats.unique_property_paths} unique paths"
            )
        else:
            context_parts.append("\n- Window Props: Not available")

        return "".join(context_parts)

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

        if llm_provider_response_id and role == ChatRole.ASSISTANT:
            self._response_id_to_chat_index[llm_provider_response_id] = len(self._thread.chat_ids) - 1

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        return chat

    def _build_messages_for_llm(self) -> list[dict[str, Any]]:
        """Build messages list for LLM from Chat objects."""
        messages: list[dict[str, Any]] = []

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
                        "call_id": tc.call_id if tc.call_id else f"call_{idx}_{chat_id[:8]}",
                        "name": tc.tool_name,
                        "arguments": tc.tool_arguments,
                    }
                    for idx, tc in enumerate(chat.tool_calls)
                ]
            messages.append(msg)
        return messages

    # ==================== Tool Implementations ====================

    @token_optimized
    def _tool_search_everywhere(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search for a value across all data stores."""
        value = tool_arguments.get("value", "")
        if not value:
            return {"error": "value is required"}

        case_sensitive = tool_arguments.get("case_sensitive", False)
        results: dict[str, Any] = {"value_searched": value, "case_sensitive": case_sensitive}

        # Search network
        if self._network_data_store:
            network_results = self._network_data_store.search_response_bodies(
                value=value, case_sensitive=case_sensitive
            )
            results["network"] = {
                "found": len(network_results) > 0,
                "count": len(network_results),
                "matches": network_results[:10],
            }
        else:
            results["network"] = {"available": False}

        # Search storage
        if self._storage_data_store:
            storage_results = self._storage_data_store.search_values(
                value=value, case_sensitive=case_sensitive
            )
            results["storage"] = {
                "found": len(storage_results) > 0,
                "count": len(storage_results),
                "matches": storage_results[:10],
            }
        else:
            results["storage"] = {"available": False}

        # Search window properties
        if self._window_property_data_store:
            window_results = self._window_property_data_store.search_values(
                value=value, case_sensitive=case_sensitive
            )
            results["window_properties"] = {
                "found": len(window_results) > 0,
                "count": len(window_results),
                "matches": window_results[:10],
            }
        else:
            results["window_properties"] = {"available": False}

        # Summary
        total_found = sum(
            results.get(k, {}).get("count", 0)
            for k in ["network", "storage", "window_properties"]
        )
        results["summary"] = {
            "total_matches": total_found,
            "found_in": [
                k for k in ["network", "storage", "window_properties"]
                if results.get(k, {}).get("found", False)
            ],
        }

        return results

    @token_optimized
    def _tool_search_in_network(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search network response bodies for a value."""
        if not self._network_data_store:
            return {"error": "Network data store not available"}

        value = tool_arguments.get("value", "")
        if not value:
            return {"error": "value is required"}

        case_sensitive = tool_arguments.get("case_sensitive", False)
        results = self._network_data_store.search_response_bodies(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @token_optimized
    def _tool_search_in_storage(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search browser storage for a value."""
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        value = tool_arguments.get("value", "")
        if not value:
            return {"error": "value is required"}

        case_sensitive = tool_arguments.get("case_sensitive", False)
        results = self._storage_data_store.search_values(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @token_optimized
    def _tool_search_in_window_props(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search window property values for a value."""
        if not self._window_property_data_store:
            return {"error": "Window property data store not available"}

        value = tool_arguments.get("value", "")
        if not value:
            return {"error": "value is required"}

        case_sensitive = tool_arguments.get("case_sensitive", False)
        results = self._window_property_data_store.search_values(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @token_optimized
    def _tool_get_network_entry(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get full details of a network entry."""
        if not self._network_data_store:
            return {"error": "Network data store not available"}

        request_id = tool_arguments.get("request_id")
        if not request_id:
            return {"error": "request_id is required"}

        entry = self._network_data_store.get_entry(request_id)
        if not entry:
            return {"error": f"Entry {request_id} not found"}

        # Truncate large response content
        response_content = entry.response_body
        if response_content and len(response_content) > 5000:
            response_content = response_content[:5000] + f"\n... (truncated, {len(entry.response_body)} total chars)"

        return {
            "request_id": request_id,
            "timestamp": entry.timestamp,
            "method": entry.method,
            "url": entry.url,
            "status": entry.status,
            "mime_type": entry.mime_type,
            "request_headers": entry.request_headers,
            "response_headers": entry.response_headers,
            "post_data": entry.post_data,
            "response_content": response_content,
        }

    @token_optimized
    def _tool_get_storage_entry(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get full details of a storage entry."""
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        index = tool_arguments.get("index")
        if index is None:
            return {"error": "index is required"}

        entry = self._storage_data_store.get_entry(index)
        if not entry:
            return {"error": f"Entry at index {index} not found"}

        return {
            "index": index,
            "entry": entry.model_dump(),
        }

    @token_optimized
    def _tool_get_window_prop_changes(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get changes for a specific window property path."""
        if not self._window_property_data_store:
            return {"error": "Window property data store not available"}

        path = tool_arguments.get("path")
        if not path:
            return {"error": "path is required"}

        exact = tool_arguments.get("exact", False)
        results = self._window_property_data_store.get_changes_by_path(path, exact=exact)

        return {
            "path": path,
            "exact_match": exact,
            "changes_found": len(results),
            "changes": results[:20],
        }

    @token_optimized
    def _tool_get_storage_by_key(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get all storage entries for a specific key name."""
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        key = tool_arguments.get("key")
        if not key:
            return {"error": "key is required"}

        entries = self._storage_data_store.get_entries_by_key(key)

        return {
            "key": key,
            "entries_found": len(entries),
            "entries": [e.model_dump() for e in entries[:20]],
        }

    def _tool_execute_python(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute Python code in a sandboxed environment with all data stores pre-loaded."""
        code = tool_arguments.get("code", "")

        # Build extra globals with all available data stores
        extra_globals: dict[str, Any] = {}

        if self._network_data_store:
            extra_globals["network_entries"] = [
                e.model_dump() for e in self._network_data_store.entries
            ]
        else:
            extra_globals["network_entries"] = []

        if self._storage_data_store:
            extra_globals["storage_entries"] = [
                e.model_dump() for e in self._storage_data_store.entries
            ]
        else:
            extra_globals["storage_entries"] = []

        if self._window_property_data_store:
            extra_globals["window_prop_entries"] = [
                e.model_dump() for e in self._window_property_data_store.entries
            ]
        else:
            extra_globals["window_prop_entries"] = []

        return execute_python_sandboxed(code, extra_globals=extra_globals)

    @token_optimized
    def _tool_finalize_result(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_result tool call."""
        value_searched = tool_arguments.get("value_searched", "")
        origins_data = tool_arguments.get("origins", [])
        likely_source_index = tool_arguments.get("likely_source_index", -1)
        explanation = tool_arguments.get("explanation", "")

        if not value_searched:
            return {"error": "value_searched is required"}
        if not origins_data:
            return {"error": "origins list is required and cannot be empty"}
        if not explanation:
            return {"error": "explanation is required"}

        # Build origin objects
        origins: list[TokenOrigin] = []
        for origin_data in origins_data:
            origins.append(TokenOrigin(
                source_type=origin_data.get("source_type"),
                location=origin_data.get("location"),
                context=origin_data.get("context"),
                entry_id=origin_data.get("entry_id"),
            ))

        likely_source = None
        if 0 <= likely_source_index < len(origins):
            likely_source = origins[likely_source_index]

        self._origin_result = TokenOriginResult(
            value_searched=value_searched,
            origins=origins,
            likely_source=likely_source,
            explanation=explanation,
        )

        logger.info("Token origin tracing completed: %d origin(s) found", len(origins))

        return {
            "status": "success",
            "message": f"Token origin tracing completed with {len(origins)} origin(s)",
            "result": self._origin_result.model_dump(),
        }

    @token_optimized
    def _tool_finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_failure tool call."""
        # On first iteration where finalize is available, reject and suggest trying other methods
        if self._autonomous_iteration == 2:
            return {
                "error": "Too early to give up! You must try other methods first.",
                "suggestions": [
                    "Use `get_storage_by_key` to look up the value by key name",
                    "Use `execute_python` to write custom queries across all data stores",
                    "Try searching with different variations of the value",
                    "Check if the value appears in network request/response bodies",
                    "Look for related keys or paths that might contain the value",
                ],
                "iterations_remaining": self._autonomous_max_iterations - self._autonomous_iteration,
            }

        value_searched = tool_arguments.get("value_searched", "")
        reason = tool_arguments.get("reason", "")
        suggestions = tool_arguments.get("suggestions", [])

        if not value_searched:
            return {"error": "value_searched is required"}
        if not reason:
            return {"error": "reason is required"}

        self._origin_failure = TokenOriginFailure(
            value_searched=value_searched,
            reason=reason,
            suggestions=suggestions,
        )

        logger.info("Token origin tracing failed: %s", reason)

        return {
            "status": "failure",
            "message": "Token origin tracing failed",
            "result": self._origin_failure.model_dump(),
        }

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        logger.debug("Executing tool %s", tool_name)

        tool_map = {
            "search_everywhere": self._tool_search_everywhere,
            "search_in_network": self._tool_search_in_network,
            "search_in_storage": self._tool_search_in_storage,
            "search_in_window_props": self._tool_search_in_window_props,
            "get_network_entry": self._tool_get_network_entry,
            "get_storage_entry": self._tool_get_storage_entry,
            "get_window_prop_changes": self._tool_get_window_prop_changes,
            "get_storage_by_key": self._tool_get_storage_by_key,
            "execute_python": self._tool_execute_python,
            "finalize_result": self._tool_finalize_result,
            "finalize_failure": self._tool_finalize_failure,
        }

        if tool_name in tool_map:
            return tool_map[tool_name](tool_arguments)

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

            return json.dumps(result)

        except Exception as e:
            invocation.status = ToolInvocationStatus.FAILED

            self._emit_message(
                ToolInvocationResultEmittedMessage(
                    tool_invocation=invocation,
                    tool_result={"error": str(e)},
                )
            )

            logger.error("Tool %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    def process_new_message(self, content: str, role: ChatRole = ChatRole.USER) -> None:
        """
        Process a new message and emit responses via callback.

        Args:
            content: The message content
            role: The role of the message sender (USER or SYSTEM)
        """
        self._add_chat(role, content)
        self._run_agent_loop()

    def _run_agent_loop(self) -> None:
        """Run the agent loop: call LLM, execute tools, feed results back, repeat."""
        max_iterations = 10

        for iteration in range(max_iterations):
            logger.debug("Agent loop iteration %d", iteration + 1)

            # Sync tools for conversational mode (no finalize tools)
            self._sync_tools(include_finalize=False)

            messages = self._build_messages_for_llm()

            try:
                if self._stream_chunk_callable:
                    response = self._process_streaming_response(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_system_prompt(),
                        previous_response_id=self._previous_response_id,
                    )

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

                for tool_call in response.tool_calls:
                    result_str = self._auto_execute_tool(
                        tool_call.tool_name, tool_call.tool_arguments
                    )
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_call.tool_name}' result: {result_str}",
                        tool_call_id=tool_call.call_id,
                    )

            except Exception as e:
                logger.exception("Error in agent loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Agent loop hit max iterations (%d)", max_iterations)

    def _process_streaming_response(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process LLM response with streaming."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=self._get_system_prompt(),
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

    def run_autonomous(
        self,
        value: str,
        min_iterations: int = 2,
        max_iterations: int = 5,
    ) -> TokenOriginResult | TokenOriginFailure | None:
        """
        Run the agent autonomously to trace where a value came from.

        Args:
            value: The token/value to trace
            min_iterations: Minimum iterations before allowing finalize (default 2)
            max_iterations: Maximum iterations before stopping (default 5)

        Returns:
            TokenOriginResult if origins were found,
            TokenOriginFailure if value could not be found,
            None if max iterations reached without finalization.
        """
        self._autonomous_mode = True
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = max_iterations
        self._origin_result = None
        self._origin_failure = None
        self._finalize_tool_registered = False

        initial_message = (
            f"TRACE VALUE: {value}\n\n"
            "Find where this value originated from. Search across all data stores, "
            "analyze the results, and call finalize_result with your findings."
        )
        self._add_chat(ChatRole.USER, initial_message)

        logger.info("Starting autonomous token tracing for value: %s", value[:50])

        self._run_autonomous_loop(min_iterations, max_iterations)

        self._autonomous_mode = False

        if self._origin_result is not None:
            return self._origin_result
        if self._origin_failure is not None:
            return self._origin_failure
        return None

    def _run_autonomous_loop(self, min_iterations: int, max_iterations: int) -> None:
        """Run the autonomous agent loop."""
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug("Autonomous loop iteration %d/%d", self._autonomous_iteration, max_iterations)

            # Sync tools: include finalize tools only after min_iterations
            include_finalize = self._autonomous_iteration >= min_iterations
            was_finalize_registered = self._finalize_tool_registered
            self._sync_tools(include_finalize=include_finalize)

            if include_finalize and not was_finalize_registered:
                logger.info("Finalize tools now available (iteration %d)", self._autonomous_iteration)

            messages = self._build_messages_for_llm()

            try:
                if self._stream_chunk_callable:
                    response = self._process_streaming_response_autonomous(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_autonomous_system_prompt(),
                        previous_response_id=self._previous_response_id,
                    )

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

                for tool_call in response.tool_calls:
                    result_str = self._auto_execute_tool(
                        tool_call.tool_name, tool_call.tool_arguments
                    )
                    self._add_chat(
                        ChatRole.TOOL,
                        f"Tool '{tool_call.tool_name}' result: {result_str}",
                        tool_call_id=tool_call.call_id,
                    )

                    if tool_call.tool_name == "finalize_result" and self._origin_result is not None:
                        logger.info("Token tracing completed at iteration %d", self._autonomous_iteration)
                        return

                    if tool_call.tool_name == "finalize_failure" and self._origin_failure is not None:
                        logger.info("Token tracing failed at iteration %d", self._autonomous_iteration)
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(ErrorEmittedMessage(error=str(e)))
                return

        logger.warning("Autonomous loop hit max iterations (%d)", max_iterations)

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode."""
        base_prompt = self._get_system_prompt().replace(self.SYSTEM_PROMPT, self.AUTONOMOUS_SYSTEM_PROMPT)

        if self._finalize_tool_registered:
            remaining = self._autonomous_max_iterations - self._autonomous_iteration
            if remaining <= 2:
                notice = (
                    f"\n\n## CRITICAL: Call finalize_result NOW!\n"
                    f"Only {remaining} iterations remaining."
                )
            else:
                notice = "\n\n## finalize_result is now available. Call it when ready."
        else:
            notice = f"\n\n## Continue exploring (iteration {self._autonomous_iteration})."

        return base_prompt + notice

    def _process_streaming_response_autonomous(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process streaming response for autonomous mode."""
        response: LLMChatResponse | None = None

        for item in self.llm_client.call_stream_sync(
            messages=messages,
            system_prompt=self._get_autonomous_system_prompt(),
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

        self._autonomous_mode = False
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = 10
        self._origin_result = None
        self._origin_failure = None
        self._finalize_tool_registered = False

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug("Reset conversation from %s to %s", old_chat_thread_id, self._thread.id)


# Backwards compatibility alias
PathFinderAgent = TraceHoundAgent
