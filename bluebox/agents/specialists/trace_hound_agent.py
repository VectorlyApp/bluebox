"""
bluebox/agents/specialists/trace_hound_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in tracing where tokens/values originated from.

Contains:
- TraceHoundAgent: Specialist for tracing values across network, storage, and window property data
- TokenOriginResult: Result model for autonomous token tracing
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

import textwrap
from typing import Any, Callable

from pydantic import BaseModel, Field

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode, specialist_tool
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
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


class TraceHoundAgent(AbstractSpecialist):
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

        After sufficient exploration, call `finalize_result` with ALL required parameters:

        **CRITICAL: You MUST provide ALL parameters - do NOT call with missing arguments!**

        **CORRECT USAGE EXAMPLE:**
        ```
        finalize_result(
            value_searched="x-trace-id",
            origins=[
                {
                    "source_type": "network_response",
                    "location": "response headers",
                    "context": "x-trace-id: abc123",
                    "entry_id": "req_abc123"
                }
            ],
            explanation="The x-trace-id originates from the search API response headers and is used in subsequent requests",
            likely_source_index=0
        )
        ```

        **WRONG - DO NOT DO THIS:**
        ```
        finalize_result()  # ❌ WRONG - Missing all required parameters!
        ```

        Required parameters:
        - value_searched: The token/value that was searched for (string)
        - origins: List of all discovered origins (list of dicts with keys: source_type, location, context, entry_id)
        - explanation: Explanation of how the value flows through the system (string)
        - likely_source_index: Index in origins array of the most likely original source (int, or -1 if unclear)

        If the value cannot be found anywhere, call `finalize_failure` with an explanation.
    """).strip()

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        network_data_store: NetworkDataLoader | None = None,
        storage_data_store: StorageDataLoader | None = None,
        window_property_data_store: WindowPropertyDataLoader | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the trace hound agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            network_data_store: NetworkDataLoader for network traffic analysis.
            storage_data_store: StorageDataLoader for browser storage analysis.
            window_property_data_store: WindowPropertyDataLoader for window props analysis.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        # Data stores
        self._network_data_store = network_data_store
        self._storage_data_store = storage_data_store
        self._window_property_data_store = window_property_data_store

        # Autonomous result state
        self._origin_result: TokenOriginResult | None = None
        self._origin_failure: TokenOriginFailure | None = None

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
        )

        logger.debug(
            "TraceHoundAgent initialized with model: %s, chat_thread_id: %s",
            llm_model,
            self._thread.id,
        )

    ## Abstract method implementations

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

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode with urgency notices."""
        # Replace base prompt with autonomous variant
        base_prompt = self._get_system_prompt().replace(self.SYSTEM_PROMPT, self.AUTONOMOUS_SYSTEM_PROMPT)

        if self.can_finalize:
            remaining = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining <= 2:
                notice = (
                    f"\n\n## CRITICAL: Call finalize_result NOW!\n"
                    f"Only {remaining} iterations remaining. "
                    f"You MUST call finalize_result with your findings immediately."
                )
            elif remaining <= 4:
                notice = (
                    f"\n\n## URGENT: Call finalize_result soon!\n"
                    f"Only {remaining} iterations remaining. "
                    f"Call finalize_result when ready."
                )
            else:
                notice = "\n\n## finalize_result is now available. Call it when ready."
        else:
            notice = f"\n\n## Continue exploring (iteration {self._autonomous_iteration})."

        return base_prompt + notice

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TRACE VALUE: {task}\n\n"
            "Find where this value originated from. Search across all data stores, "
            "analyze the results, and call finalize_result with your findings."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        if tool_name == "finalize_result" and self._origin_result is not None:
            return True
        if tool_name == "finalize_failure" and self._origin_failure is not None:
            return True
        return False

    def _get_autonomous_result(self) -> BaseModel | None:
        return self._origin_result or self._origin_failure

    def _reset_autonomous_state(self) -> None:
        self._origin_result = None
        self._origin_failure = None

    ## Tool handlers

    @specialist_tool()
    @token_optimized
    def _search_everywhere(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search for a value across ALL data stores (network, storage, window properties). This is the best starting point when tracing where a value came from. Returns matches from each store with context about where the value was found.

        Args:
            value: The token/value to search for.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not value:
            return {"error": "value is required"}

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

    @specialist_tool()
    @token_optimized
    def _search_in_network(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search network traffic response bodies for a specific value. Returns matches with context showing where in the response the value appears.

        Args:
            value: The value to search for in response bodies.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not self._network_data_store:
            return {"error": "Network data store not available"}

        if not value:
            return {"error": "value is required"}

        results = self._network_data_store.search_response_bodies(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @specialist_tool()
    @token_optimized
    def _search_in_storage(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search browser storage (cookies, localStorage, sessionStorage, IndexedDB) for a value. Returns matches showing which storage type and key contains the value.

        Args:
            value: The value to search for in storage.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        if not value:
            return {"error": "value is required"}

        results = self._storage_data_store.search_values(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @specialist_tool()
    @token_optimized
    def _search_in_window_props(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search window object property values for a specific value. Returns matches showing which property path contains the value.

        Args:
            value: The value to search for in window properties.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not self._window_property_data_store:
            return {"error": "Window property data store not available"}

        if not value:
            return {"error": "value is required"}

        results = self._window_property_data_store.search_values(
            value=value, case_sensitive=case_sensitive
        )

        return {
            "value_searched": value,
            "results_found": len(results),
            "results": results[:20],
        }

    @specialist_tool()
    @token_optimized
    def _get_network_entry(self, request_id: str) -> dict[str, Any]:
        """
        Get full details of a network entry by request_id. Returns method, URL, headers, request body, and response body.

        Args:
            request_id: The request_id of the network entry to retrieve.
        """
        if not self._network_data_store:
            return {"error": "Network data store not available"}

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

    @specialist_tool()
    @token_optimized
    def _get_storage_entry(self, index: int) -> dict[str, Any]:
        """
        Get full details of a storage entry by index. Returns the storage event with all its fields.

        Args:
            index: The index of the storage entry to retrieve.
        """
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        entry = self._storage_data_store.get_entry(index)
        if not entry:
            return {"error": f"Entry at index {index} not found"}

        return {
            "index": index,
            "entry": entry.model_dump(),
        }

    @specialist_tool()
    @token_optimized
    def _get_window_prop_changes(
        self,
        path: str,
        exact: bool = False,
    ) -> dict[str, Any]:
        """
        Get all changes for a specific window property path. Returns the history of changes (added, changed, deleted) for that property.

        Args:
            path: The property path to get changes for (e.g., 'dataLayer.0.userId').
            exact: If true, match path exactly. If false, match paths containing the substring. Defaults to false.
        """
        if not self._window_property_data_store:
            return {"error": "Window property data store not available"}

        if not path:
            return {"error": "path is required"}

        results = self._window_property_data_store.get_changes_by_path(path, exact=exact)

        return {
            "path": path,
            "exact_match": exact,
            "changes_found": len(results),
            "changes": results[:20],
        }

    @specialist_tool()
    @token_optimized
    def _get_storage_by_key(self, key: str) -> dict[str, Any]:
        """
        Get all storage entries for a specific key name. Use this when you want to find the VALUE stored under a given KEY (e.g., 'get the value of _cb_svref_expires'). Returns all events where this key was set, modified, or deleted.

        Args:
            key: The storage key name to look up.
        """
        if not self._storage_data_store:
            return {"error": "Storage data store not available"}

        if not key:
            return {"error": "key is required"}

        entries = self._storage_data_store.get_entries_by_key(key)

        return {
            "key": key,
            "entries_found": len(entries),
            "entries": [e.model_dump() for e in entries[:20]],
        }

    @specialist_tool()
    def _execute_python(self, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment to analyze data. Pre-loaded variables: `network_entries` (list of NetworkTransactionEvent dicts with request_id, url, method, status, request_headers, response_headers, post_data, response_body), `storage_entries` (list of StorageEvent dicts with type, origin, key, value, etc.), `window_prop_entries` (list of WindowPropertyEvent dicts with url, timestamp, changes). Use print() to output results. Example: for e in storage_entries: if e['key'] == 'token': print(e)

        Args:
            code: Python code to execute. Variables available: network_entries, storage_entries, window_prop_entries. The `json` module is available. Use print() for output. Note: imports are disabled for security.
        """
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

    @specialist_tool(
        availability=lambda self: self.can_finalize,
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
                "explanation": {
                    "type": "string",
                    "description": "Explanation of how the value flows through the system.",
                },
                "likely_source_index": {
                    "type": "integer",
                    "description": "Index in origins array of the most likely original source (or -1 if unclear).",
                },
            },
            "required": ["value_searched", "origins", "explanation"],
        }
    )
    @token_optimized
    def _finalize_result(
        self,
        value_searched: str,
        origins: list[dict[str, Any]],
        explanation: str,
        likely_source_index: int = -1,
    ) -> dict[str, Any]:
        """
        Finalize the token origin tracing with your findings. Call this when you have traced where the value came from.

        Args:
            value_searched: The token/value that was searched for.
            origins: List of all discovered origins. Each origin should have: source_type, location, context, entry_id.
            explanation: Explanation of how the value flows through the system.
            likely_source_index: Index in origins array of the most likely original source (or -1 if unclear).
        """
        if not value_searched:
            return {"error": "value_searched is required"}
        if not origins:
            return {"error": "origins list is required and cannot be empty"}
        if not explanation:
            return {"error": "explanation is required"}

        # Build origin objects
        origin_objects: list[TokenOrigin] = []
        for origin_data in origins:
            origin_objects.append(TokenOrigin(
                source_type=origin_data.get("source_type"),
                location=origin_data.get("location"),
                context=origin_data.get("context"),
                entry_id=origin_data.get("entry_id"),
            ))

        likely_source = None
        if 0 <= likely_source_index < len(origin_objects):
            likely_source = origin_objects[likely_source_index]

        self._origin_result = TokenOriginResult(
            value_searched=value_searched,
            origins=origin_objects,
            likely_source=likely_source,
            explanation=explanation,
        )

        logger.info("Token origin tracing completed: %d origin(s) found", len(origin_objects))

        return {
            "status": "success",
            "message": f"Token origin tracing completed with {len(origin_objects)} origin(s)",
            "result": self._origin_result.model_dump(),
        }

    @specialist_tool(availability=lambda self: self.can_finalize)
    @token_optimized
    def _finalize_failure(
        self,
        value_searched: str,
        reason: str,
        suggestions: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Signal that the token origin tracing failed to find the value. Call this when you have searched thoroughly but cannot find the value.

        Args:
            value_searched: The token/value that was searched for.
            reason: Explanation of why the value could not be found.
            suggestions: Suggestions for alternative searches.
        """
        # On early iterations, reject and suggest trying other methods
        if self._autonomous_iteration <= 3:
            return {
                "error": "Too early to give up! You must try other methods first.",
                "suggestions": [
                    "Use `get_storage_by_key` to look up the value by key name",
                    "Use `execute_python` to write custom queries across all data stores",
                    "Try searching with different variations of the value",
                    "Check if the value appears in network request/response bodies",
                    "Look for related keys or paths that might contain the value",
                ],
                "iterations_remaining": self._autonomous_config.max_iterations - self._autonomous_iteration,
            }

        if not value_searched:
            return {"error": "value_searched is required"}
        if not reason:
            return {"error": "reason is required"}

        self._origin_failure = TokenOriginFailure(
            value_searched=value_searched,
            reason=reason,
            suggestions=suggestions or [],
        )

        logger.info("Token origin tracing failed: %s", reason)

        return {
            "status": "failure",
            "message": "Token origin tracing failed",
            "result": self._origin_failure.model_dump(),
        }
