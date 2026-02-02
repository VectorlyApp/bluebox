"""
bluebox/agents/network_spy_agent.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in searching through network traffic data.

Contains:
- NetworkSpyAgent: Conversational interface for network traffic analysis
- EndpointDiscoveryResult: Result model for autonomous endpoint discovery
- Uses: LLMClient with tools for network data searching
- Maintains: ChatThread for multi-turn conversation
"""

import json
import textwrap
from datetime import datetime
from typing import Any, Callable
from urllib.parse import urlparse, parse_qs

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
from bluebox.utils.code_execution_sandbox import execute_python_sandboxed
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class DiscoveredEndpoint(BaseModel):
    """A single discovered API endpoint."""

    request_ids: list[str] = Field(
        description="HAR entry request_ids for this endpoint"
    )
    url: str = Field(
        description="The API endpoint URL"
    )
    endpoint_inputs: str = Field(
        description="Brief description of what the endpoint takes as input (parameters, body fields)"
    )
    endpoint_outputs: str = Field(
        description="Brief description of what data the endpoint returns"
    )


class EndpointDiscoveryResult(BaseModel):
    """
    Result of autonomous endpoint discovery.

    Contains one or more discovered endpoints needed to complete the user's task.
    Multiple endpoints may be needed for multi-step flows (e.g., auth -> search -> details).
    """

    endpoints: list[DiscoveredEndpoint] = Field(
        description="List of discovered endpoints needed for the task"
    )


class DiscoveryFailureResult(BaseModel):
    """
    Result when autonomous endpoint discovery fails.

    Returned when the agent cannot find the appropriate endpoints after exhaustive search.
    """

    reason: str = Field(
        description="Explanation of why the endpoint could not be found"
    )
    searched_terms: list[str] = Field(
        default_factory=list,
        description="List of search terms that were tried"
    )
    closest_matches: list[str] = Field(
        default_factory=list,
        description="URLs of entries that came closest to matching (if any)"
    )


class NetworkSpyAgent:
    """
    Network spy agent that helps analyze HAR (HTTP Archive) files.

    The agent maintains a ChatThread with Chat messages and uses LLM with tools
    to search and analyze network traffic.
    """

    SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a network traffic analyst specializing in HAR (HTTP Archive) file analysis.

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

        - **`get_response_body_schema`**: Get the schema of a JSON response body.
          - Use this to understand the shape of large JSON responses without retrieving all the data
          - Shows structure with types at every level

        ## Guidelines

        - Be concise and direct in your responses
        - When you find a relevant entry, report its ID and URL
        - Always use search_har_by_terms first when looking for specific data
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = textwrap.dedent("""
        You are a network traffic analyst that autonomously identifies API endpoints.

        ## Your Mission

        Given a user task, find the API endpoint(s) that return the data needed for that task.
        Some tasks require multiple endpoints (e.g., auth -> search -> details).

        ## Process

        1. **Search**: Use `search_har_responses_by_terms` with 20-30 relevant terms for the task
        2. **Analyze**: Look at top results, examine their structure with `get_response_body_schema`
        3. **Verify**: Use `get_entry_detail` to confirm the endpoint has the right data
        4. **Finalize**: Once confident, call `finalize_result` with your findings

        ## Strategy

        - Identify ALL endpoints needed to complete the task
        - Look for API/XHR calls (not HTML pages, JS files, or images)
        - Prefer endpoints with structured JSON responses
        - Consider multi-step flows: authentication, search, pagination, detail fetches

        ## When finalize tools are available

        After sufficient exploration, the `finalize_result` and `finalize_failure` tools become available.

        ### finalize_result - Use when endpoint IS found
        Call it with a list of endpoints, each containing:
        - request_ids: The HAR entry request_id(s) for this endpoint (MUST be valid IDs from the data store)
        - url: The API URL
        - endpoint_inputs: Brief description of inputs (e.g., "from_city, to_city, date as query params")
        - endpoint_outputs: Brief description of outputs (e.g., "JSON array of train options with prices")

        Order endpoints by execution sequence if they form a multi-step flow.
        Be concise with inputs/outputs - just the key fields and types, not full schema.

        ### finalize_failure - Use when endpoint is NOT found
        If after exhaustive search you determine the required endpoint does NOT exist in the traffic:
        - Call `finalize_failure` with a clear reason explaining what was searched and why no match was found
        - Include the search terms you tried and any URLs that came close but didn't match
        - Only use this after thoroughly searching - don't give up too early!
    """).strip()

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        network_data_store: NetworkDataStore,
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
            network_data_store: The NetworkDataStore containing parsed HAR data.
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
        self._network_data_store = network_data_store
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
        self._discovery_result: EndpointDiscoveryResult | None = None
        self._discovery_failure: DiscoveryFailureResult | None = None
        self._finalize_tool_registered: bool = False

        logger.debug(
            "Instantiated NetworkSpyAgent with model: %s, chat_thread_id: %s, entries: %d",
            llm_model,
            self._thread.id,
            len(network_data_store.entries),
        )

    def _register_tools(self) -> None:
        """Register tools for HAR analysis."""
        # search_har_responses_by_terms
        self.llm_client.register_tool(
            name="search_har_responses_by_terms",
            description=(
                "Search RESPONSE bodies by a list of terms. Searches HTML/JSON response bodies "
                "(excludes JS, images, media) and returns top 10-20 entries ranked by relevance score. "
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
                "Get full details of a specific HAR entry by request_id. "
                "Returns method, URL, headers, request body, and response body."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "The request_id of the HAR entry to retrieve.",
                    }
                },
                "required": ["request_id"],
            },
        )

        # get_response_body_schema
        self.llm_client.register_tool(
            name="get_response_body_schema",
            description=(
                "Get the schema of a HAR entry's JSON response body. "
                "Shows structure with types at every level. "
                "Useful for understanding the shape of large JSON responses."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "request_id": {
                        "type": "string",
                        "description": "The request_id of the HAR entry to get schema for.",
                    }
                },
                "required": ["request_id"],
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

        # execute_python
        self.llm_client.register_tool(
            name="execute_python",
            description=(
                "Execute Python code in a sandboxed environment to analyze network entries. "
                "The variable `entries` is pre-loaded as a list of NetworkTransactionEvent dicts. "
                "Each entry has: request_id, url, method, status, mime_type, request_headers, "
                "response_headers, post_data, response_body. "
                "Use print() to output results. "
                "Example: for e in entries[:5]: print(e['url'])"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. `entries` is a list of network entry dicts. "
                            "`json` module is available. Use print() for output. "
                            "Note: imports are disabled for security."
                        ),
                    }
                },
                "required": ["code"],
            },
        )

        # search_har_by_request
        self.llm_client.register_tool(
            name="search_har_by_request",
            description=(
                "Search the REQUEST side of HAR entries (URL, headers, body) for terms. "
                "Useful for finding where sensitive data or parameters are sent. "
                "Returns entries ranked by relevance."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of search terms to look for in requests.",
                    },
                    "search_in": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["url", "headers", "body"]},
                        "description": "Where to search: 'url', 'headers', 'body'. Defaults to all.",
                    }
                },
                "required": ["terms"],
            },
        )

        # search_response_bodies
        self.llm_client.register_tool(
            name="search_response_bodies",
            description=(
                "Search response bodies for a specific value and return matches with context. "
                "Unlike search_har_responses_by_terms which ranks by relevance across many terms, "
                "this tool finds exact matches for a single value and shows surrounding context. "
                "Useful for finding where specific data (IDs, tokens, values) appears."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The exact value to search for in response bodies.",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether the search should be case-sensitive. Defaults to false.",
                    }
                },
                "required": ["value"],
            },
        )

    def _register_finalize_tool(self) -> None:
        """Register the finalize_result tool for autonomous mode (available after iteration 2)."""
        if self._finalize_tool_registered:
            return

        self.llm_client.register_tool(
            name="finalize_result",
            description=(
                "Finalize the endpoint discovery with your findings. "
                "Call this when you have identified the API endpoint(s) needed for the user's task. "
                "You can specify multiple endpoints if the task requires a multi-step flow "
                "(e.g., authenticate -> search -> get details). "
                "NOTE: All request_ids must be valid IDs from the data store."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "endpoints": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "request_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "HAR entry request_id(s) for this endpoint.",
                                },
                                "url": {
                                    "type": "string",
                                    "description": "The API endpoint URL.",
                                },
                                "endpoint_inputs": {
                                    "type": "string",
                                    "description": (
                                        "Brief description of inputs. "
                                        "E.g., 'origin, destination, date as query params'."
                                    ),
                                },
                                "endpoint_outputs": {
                                    "type": "string",
                                    "description": (
                                        "Brief description of outputs. "
                                        "E.g., 'JSON array of train options with price, times'."
                                    ),
                                },
                            },
                            "required": ["request_ids", "url", "endpoint_inputs", "endpoint_outputs"],
                        },
                        "description": "List of discovered endpoints. Order by execution sequence if multi-step.",
                    },
                },
                "required": ["endpoints"],
            },
        )

        # Also register the failure tool for when no endpoint can be found
        self.llm_client.register_tool(
            name="finalize_failure",
            description=(
                "Signal that the endpoint discovery has failed. "
                "Call this ONLY when you have exhaustively searched and are confident "
                "that the required endpoint does NOT exist in the captured traffic. "
                "Provide a clear explanation of what was searched and why no match was found."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Detailed explanation of why the endpoint could not be found. "
                            "E.g., 'No API endpoints found that return train pricing data. "
                            "The traffic only contains static HTML pages and image assets.'"
                        ),
                    },
                    "searched_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key search terms that were tried.",
                    },
                    "closest_matches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs of entries that came closest to matching (if any).",
                    },
                },
                "required": ["reason"],
            },
        )

        self._finalize_tool_registered = True
        logger.debug("Registered finalize_result and finalize_failure tools")

    @property
    def chat_thread_id(self) -> str:
        """Return the current thread ID."""
        return self._thread.id

    @property
    def autonomous_iteration(self) -> int:
        """Return the current/final autonomous iteration count."""
        return self._autonomous_iteration

    def _get_system_prompt(self) -> str:
        """Get system prompt with HAR stats context, host stats, and likely API URLs."""
        stats = self._network_data_store.stats
        stats_context = (
            f"\n\n## HAR File Context\n"
            f"- Total Requests: {stats.total_requests}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Hosts: {stats.unique_hosts}\n"
        )

        # Add likely API URLs
        likely_urls = self._network_data_store.api_urls
        if likely_urls:
            urls_list = "\n".join(f"- {url}" for url in likely_urls[:50])  # Limit to 50
            urls_context = (
                f"\n\n## Likely Important API Endpoints\n"
                f"The following URLs are likely important API endpoints:\n\n"
                f"{urls_list}\n\n"
                f"Use the `get_unique_urls` tool to see all other URLs in the HAR file."
            )
        else:
            urls_context = (
                f"\n\n## API Endpoints\n"
                f"No obvious API endpoints detected. Use the `get_unique_urls` tool to see all URLs."
            )

        # Add per-host stats
        host_stats = self._network_data_store.get_host_stats()
        if host_stats:
            host_lines = []
            for hs in host_stats[:15]:  # Top 15 hosts
                methods_str = ", ".join(f"{m}:{c}" for m, c in sorted(hs["methods"].items()))
                host_lines.append(
                    f"- {hs['host']}: {hs['request_count']} reqs ({methods_str})"
                )
            host_context = (
                f"\n\n## Host Statistics\n"
                f"{chr(10).join(host_lines)}"
            )
        else:
            host_context = ""

        return self.SYSTEM_PROMPT + stats_context + host_context + urls_context

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
            # Include tool_calls for ASSISTANT role messages (generic format for openai_client)
            if chat.tool_calls:
                msg["tool_calls"] = [
                    {
                        # call_id is required by OpenAI API; generate fallback if None
                        "call_id": tc.call_id if tc.call_id else f"call_{idx}_{chat_id[:8]}",
                        "name": tc.tool_name,
                        "arguments": tc.tool_arguments,
                    }
                    for idx, tc in enumerate(chat.tool_calls)
                ]
            messages.append(msg)
        return messages

    @token_optimized
    def _tool_search_har_responses_by_terms(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_har_responses_by_terms tool."""
        terms = tool_arguments.get("terms", [])
        if not terms:
            return {"error": "No search terms provided"}

        results = self._network_data_store.search_entries_by_terms(terms, top_n=20)

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

    @token_optimized
    def _tool_get_entry_detail(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_entry_detail tool."""
        request_id = tool_arguments.get("request_id")
        if request_id is None:
            return {"error": "request_id is required"}

        entry = self._network_data_store.get_entry(request_id)
        if entry is None:
            return {"error": f"Entry {request_id} not found"}

        # Truncate large response content
        response_content = entry.response_body
        if response_content and len(response_content) > 5000:
            response_content = response_content[:5000] + f"\n... (truncated, {len(entry.response_body)} total chars)"

        # Get schema for JSON responses
        key_structure = self._network_data_store.get_response_body_schema(request_id)

        # Parse query params from URL
        parsed_url = urlparse(entry.url)
        query_params = parse_qs(parsed_url.query)

        return {
            "request_id": request_id,
            "method": entry.method,
            "url": entry.url,
            "status": entry.status,
            "status_text": entry.status_text,
            "mime_type": entry.mime_type,
            "request_headers": entry.request_headers,
            "response_headers": entry.response_headers,
            "query_params": query_params,
            "post_data": entry.post_data,
            "response_content": response_content,
            "response_key_structure": key_structure,
        }

    @token_optimized
    def _tool_get_response_body_schema(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_response_body_schema tool."""
        request_id = tool_arguments.get("request_id")
        if request_id is None:
            return {"error": "request_id is required"}

        key_structure = self._network_data_store.get_response_body_schema(request_id)
        if key_structure is None:
            entry = self._network_data_store.get_entry(request_id)
            if entry is None:
                return {"error": f"Entry {request_id} not found"}
            return {"error": f"Entry {request_id} does not have valid JSON response content"}

        return {
            "request_id": request_id,
            "key_structure": key_structure,
        }

    @token_optimized
    def _tool_get_unique_urls(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute get_unique_urls tool."""
        url_counts = self._network_data_store.url_counts
        return {
            "total_unique_urls": len(url_counts),
            "url_counts": url_counts,
        }

    def _tool_execute_python(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute Python code in a sandboxed environment with network entries pre-loaded."""
        code = tool_arguments.get("code", "")
        entries = [e.model_dump() for e in self._network_data_store.entries]
        return execute_python_sandboxed(code, extra_globals={"entries": entries})

    @token_optimized
    def _tool_search_har_by_request(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search request side (URL, headers, body) for terms."""
        terms = tool_arguments.get("terms", [])
        if not terms:
            return {"error": "No search terms provided"}

        search_in = tool_arguments.get("search_in", ["url", "headers", "body"])
        if not search_in:
            search_in = ["url", "headers", "body"]

        terms_lower = [t.lower() for t in terms]
        results: list[dict[str, Any]] = []

        for entry in self._network_data_store.entries:
            found_terms: set[str] = set()
            total_hits = 0
            matched_in: list[str] = []

            # Search URL
            if "url" in search_in:
                url_lower = entry.url.lower()
                for term in terms_lower:
                    count = url_lower.count(term)
                    if count > 0:
                        found_terms.add(term)
                        total_hits += count
                        if "url" not in matched_in:
                            matched_in.append("url")

            # Search headers
            if "headers" in search_in:
                headers_str = json.dumps(entry.request_headers).lower()
                for term in terms_lower:
                    count = headers_str.count(term)
                    if count > 0:
                        found_terms.add(term)
                        total_hits += count
                        if "headers" not in matched_in:
                            matched_in.append("headers")

            # Search body
            if "body" in search_in and entry.post_data:
                post_data_str = json.dumps(entry.post_data) if isinstance(entry.post_data, (dict, list)) else str(entry.post_data)
                body_lower = post_data_str.lower()
                for term in terms_lower:
                    count = body_lower.count(term)
                    if count > 0:
                        found_terms.add(term)
                        total_hits += count
                        if "body" not in matched_in:
                            matched_in.append("body")

            unique_terms_found = len(found_terms)
            if unique_terms_found > 0:
                score = (total_hits / len(terms_lower)) * unique_terms_found
                results.append({
                    "id": entry.request_id,
                    "method": entry.method,
                    "url": entry.url,
                    "matched_in": matched_in,
                    "unique_terms_found": unique_terms_found,
                    "total_hits": total_hits,
                    "score": round(score, 2),
                })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "terms_searched": len(terms),
            "results_found": len(results),
            "results": results[:20],  # Top 20
        }

    @token_optimized
    def _tool_search_response_bodies(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute search_response_bodies tool."""
        value = tool_arguments.get("value", "")
        if not value:
            return {"error": "value is required"}

        case_sensitive = tool_arguments.get("case_sensitive", False)

        results = self._network_data_store.search_response_bodies(
            value=value,
            case_sensitive=case_sensitive,
        )

        if not results:
            return {
                "message": f"No matches found for '{value}'",
                "case_sensitive": case_sensitive,
            }

        return {
            "value_searched": value,
            "case_sensitive": case_sensitive,
            "results_found": len(results),
            "results": results[:20],  # Top 20
        }

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute a tool and return the result."""
        logger.debug("Executing tool %s with arguments: %s", tool_name, tool_arguments)

        if tool_name == "search_har_responses_by_terms":
            return self._tool_search_har_responses_by_terms(tool_arguments)

        if tool_name == "get_entry_detail":
            return self._tool_get_entry_detail(tool_arguments)

        if tool_name == "get_response_body_schema":
            return self._tool_get_response_body_schema(tool_arguments)

        if tool_name == "get_unique_urls":
            return self._tool_get_unique_urls(tool_arguments)

        if tool_name == "execute_python":
            return self._tool_execute_python(tool_arguments)

        if tool_name == "search_har_by_request":
            return self._tool_search_har_by_request(tool_arguments)

        if tool_name == "search_response_bodies":
            return self._tool_search_response_bodies(tool_arguments)

        if tool_name == "finalize_result":
            return self._tool_finalize_result(tool_arguments)

        if tool_name == "finalize_failure":
            return self._tool_finalize_failure(tool_arguments)

        return {"error": f"Unknown tool: {tool_name}"}

    @token_optimized
    def _tool_finalize_result(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_result tool call in autonomous mode."""
        endpoints_data = tool_arguments.get("endpoints", [])

        if not endpoints_data:
            return {"error": "endpoints list is required and cannot be empty"}

        # Validate and build endpoint objects
        discovered_endpoints: list[DiscoveredEndpoint] = []
        for i, ep in enumerate(endpoints_data):
            request_ids = ep.get("request_ids", [])
            url = ep.get("url", "")
            endpoint_inputs = ep.get("endpoint_inputs", "")
            endpoint_outputs = ep.get("endpoint_outputs", "")

            if not request_ids:
                return {"error": f"endpoints[{i}].request_ids is required"}
            if not url:
                return {"error": f"endpoints[{i}].url is required"}
            if not endpoint_inputs:
                return {"error": f"endpoints[{i}].endpoint_inputs is required"}
            if not endpoint_outputs:
                return {"error": f"endpoints[{i}].endpoint_outputs is required"}

            # Validate that all request_ids actually exist in the data store
            invalid_ids = []
            for rid in request_ids:
                if self._network_data_store.get_entry(rid) is None:
                    invalid_ids.append(rid)

            if invalid_ids:
                # Get some valid request_ids to help the agent
                valid_ids_sample = [e.request_id for e in self._network_data_store.entries[:10]]
                return {
                    "error": f"endpoints[{i}].request_ids contains invalid IDs: {invalid_ids}",
                    "hint": "These request_ids do not exist in the data store. Use get_entry_detail or search tools to find valid request_ids.",
                    "sample_valid_ids": valid_ids_sample,
                }

            discovered_endpoints.append(DiscoveredEndpoint(
                request_ids=request_ids,
                url=url,
                endpoint_inputs=endpoint_inputs,
                endpoint_outputs=endpoint_outputs,
            ))

        # Store the result
        self._discovery_result = EndpointDiscoveryResult(endpoints=discovered_endpoints)

        logger.info(
            "Finalized endpoint discovery: %d endpoint(s) found",
            len(discovered_endpoints),
        )
        for ep in discovered_endpoints:
            logger.info("  - %s (request_ids: %s)", ep.url, ep.request_ids)

        return {
            "status": "success",
            "message": f"Endpoint discovery finalized with {len(discovered_endpoints)} endpoint(s)",
            "result": self._discovery_result.model_dump(),
        }

    @token_optimized
    def _tool_finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle finalize_failure tool call when endpoint discovery fails."""
        reason = tool_arguments.get("reason", "")
        searched_terms = tool_arguments.get("searched_terms", [])
        closest_matches = tool_arguments.get("closest_matches", [])

        if not reason:
            return {"error": "reason is required - explain why the endpoint could not be found"}

        # Store the failure result
        self._discovery_failure = DiscoveryFailureResult(
            reason=reason,
            searched_terms=searched_terms,
            closest_matches=closest_matches,
        )

        logger.info("Endpoint discovery failed: %s", reason)
        if searched_terms:
            logger.info("  Searched terms: %s", searched_terms[:10])
        if closest_matches:
            logger.info("  Closest matches: %s", closest_matches[:5])

        return {
            "status": "failure",
            "message": "Endpoint discovery marked as failed",
            "result": self._discovery_failure.model_dump(),
        }

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

    def run_autonomous(
        self,
        task: str,
        min_iterations: int = 3,
        max_iterations: int = 10,
    ) -> EndpointDiscoveryResult | DiscoveryFailureResult | None:
        """
        Run the agent autonomously to discover the main API endpoint for a task.

        The agent will:
        1. Search through HAR data to find relevant endpoints
        2. Analyze and verify the endpoint structure
        3. After iteration 2, the finalize tools become available
        4. Return when finalize_result/finalize_failure is called or max_iterations reached

        Args:
            task: User task description (e.g., "train prices and schedules from NYC to Boston")
            min_iterations: Minimum iterations before allowing finalize (default 3)
            max_iterations: Maximum iterations before stopping (default 10)

        Returns:
            EndpointDiscoveryResult if endpoint was found,
            DiscoveryFailureResult if agent determined endpoint doesn't exist,
            None if max iterations reached without finalization.

        Example:
            result = agent.run_autonomous(
                task="Find train prices and options from NYC to Chicago on March 15"
            )
            if isinstance(result, EndpointDiscoveryResult):
                print(f"Found {len(result.endpoints)} endpoint(s)")
            elif isinstance(result, DiscoveryFailureResult):
                print(f"Failed: {result.reason}")
            else:
                print("Reached max iterations without conclusion")
        """
        # Enable autonomous mode
        self._autonomous_mode = True
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = max_iterations
        self._discovery_result = None
        self._discovery_failure = None
        self._finalize_tool_registered = False

        # Add the task as initial message
        initial_message = (
            f"TASK: {task}\n\n"
            "Find the main API endpoint that returns the data needed for this task. "
            "Search, analyze, and when confident, use finalize_result to report your findings. "
            "If after thorough search you determine the endpoint does not exist in the traffic, "
            "use finalize_failure to report why."
        )
        self._add_chat(ChatRole.USER, initial_message)

        logger.info("Starting autonomous discovery for task: %s", task)

        # Run the autonomous agent loop
        self._run_autonomous_loop(min_iterations, max_iterations)

        # Reset autonomous mode
        self._autonomous_mode = False

        # Return result or failure
        if self._discovery_result is not None:
            return self._discovery_result
        if self._discovery_failure is not None:
            return self._discovery_failure
        return None

    def _run_autonomous_loop(self, min_iterations: int, max_iterations: int) -> None:
        """
        Run the autonomous agent loop with iteration tracking and finalize tool gating.

        Args:
            min_iterations: Minimum iterations before finalize_result is available
            max_iterations: Maximum iterations before stopping
        """
        for iteration in range(max_iterations):
            self._autonomous_iteration = iteration + 1
            logger.debug("Autonomous loop iteration %d/%d", self._autonomous_iteration, max_iterations)

            # After min_iterations-1, register the finalize_result tool
            if self._autonomous_iteration >= min_iterations - 1 and not self._finalize_tool_registered:
                self._register_finalize_tool()
                logger.info(
                    "finalize_result tool now available (iteration %d)",
                    self._autonomous_iteration,
                )

            messages = self._build_messages_for_llm()

            try:
                # Use streaming if chunk callback is set
                if self._stream_chunk_callable:
                    response = self._process_streaming_response_autonomous(messages)
                else:
                    response = self.llm_client.call_sync(
                        messages=messages,
                        system_prompt=self._get_autonomous_system_prompt(),
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

                # If no tool calls, we're done (shouldn't happen in autonomous mode)
                if not response.tool_calls:
                    logger.warning("Autonomous loop: no tool calls in iteration %d", self._autonomous_iteration)
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

                    # Check if finalize_result was called successfully
                    if tool_name == "finalize_result" and self._discovery_result is not None:
                        logger.info(
                            "Autonomous discovery completed at iteration %d",
                            self._autonomous_iteration,
                        )
                        return

                    # Check if finalize_failure was called
                    if tool_name == "finalize_failure" and self._discovery_failure is not None:
                        logger.info(
                            "Autonomous discovery failed at iteration %d: %s",
                            self._autonomous_iteration,
                            self._discovery_failure.reason,
                        )
                        return

            except Exception as e:
                logger.exception("Error in autonomous loop: %s", e)
                self._emit_message(
                    ErrorEmittedMessage(
                        error=str(e),
                    )
                )
                return

        logger.warning(
            "Autonomous loop hit max iterations (%d) without finalize_result",
            max_iterations,
        )

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode with HAR context."""
        stats = self._network_data_store.stats
        stats_context = (
            f"\n\n## HAR File Context\n"
            f"- Total Requests: {stats.total_requests}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Unique Hosts: {stats.unique_hosts}\n"
        )

        # Add likely API URLs
        likely_urls = self._network_data_store.api_urls
        if likely_urls:
            urls_list = "\n".join(f"- {url}" for url in likely_urls[:30])
            urls_context = (
                f"\n\n## Likely API Endpoints\n"
                f"{urls_list}"
            )
        else:
            urls_context = ""

        # Add finalize tool availability notice
        if self._finalize_tool_registered:
            # Get urgency based on iteration count
            remaining_iterations = self._autonomous_max_iterations - self._autonomous_iteration
            if remaining_iterations <= 2:
                finalize_notice = (
                    f"\n\n## CRITICAL: YOU MUST CALL finalize_result NOW!\n"
                    f"Only {remaining_iterations} iterations remaining. "
                    f"You MUST call `finalize_result` with your best findings immediately. "
                    f"Do NOT call any other tool - call finalize_result right now!"
                )
            elif remaining_iterations <= 4:
                finalize_notice = (
                    f"\n\n## URGENT: Call finalize_result soon!\n"
                    f"Only {remaining_iterations} iterations remaining. "
                    f"You should call `finalize_result` to complete the discovery. "
                    f"If you have identified the endpoint, finalize now."
                )
            else:
                finalize_notice = (
                    "\n\n## IMPORTANT: finalize_result is now available!\n"
                    "You can now call `finalize_result` to complete the discovery. "
                    "Do this when you have confidently identified the main API endpoint."
                )
        else:
            finalize_notice = (
                f"\n\n## Note: Continue exploring\n"
                f"The `finalize_result` tool will become available after more exploration. "
                f"Currently on iteration {self._autonomous_iteration}."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + stats_context + urls_context + finalize_notice

    def _process_streaming_response_autonomous(self, messages: list[dict[str, str]]) -> LLMChatResponse:
        """Process LLM response with streaming for autonomous mode."""
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

    def reset(self) -> None:
        """Reset the conversation to a fresh state."""
        old_chat_thread_id = self._thread.id
        self._thread = ChatThread()
        self._chats = {}
        self._previous_response_id = None
        self._response_id_to_chat_index = {}

        # Reset autonomous mode state
        self._autonomous_mode = False
        self._autonomous_iteration = 0
        self._autonomous_max_iterations = 10
        self._discovery_result = None
        self._discovery_failure = None
        self._finalize_tool_registered = False

        if self._persist_chat_thread_callable:
            self._thread = self._persist_chat_thread_callable(self._thread)

        logger.debug(
            "Reset conversation from %s to %s",
            old_chat_thread_id,
            self._thread.id,
        )
