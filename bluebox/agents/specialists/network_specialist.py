"""
bluebox/agents/specialists/network_specialist.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in searching through network traffic data.

Contains:
- NetworkSpecialist: Specialist for network traffic analysis
- EndpointDiscoveryResult: Result model for autonomous endpoint discovery
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable
from urllib.parse import urlparse, parse_qs

from pydantic import BaseModel, Field

from bluebox.agents.abstract_agent import agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.code_execution_sandbox import execute_python_sandboxed
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class DiscoveredEndpoint(BaseModel):
    """A single discovered API endpoint."""
    request_ids: list[str] = Field(
        description="Network entry request_ids for this endpoint"
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


class NetworkSpecialist(AbstractSpecialist):
    """
    Network specialist agent that helps analyze captured network traffic.

    The agent uses AbstractSpecialist as its base and provides tools to search
    and analyze network traffic data from JSONL captures.
    """

    SYSTEM_PROMPT: str = dedent("""
        You are a network traffic analyst specializing in captured browser network data.

        ## Your Role

        You help users find and analyze specific network requests in captured traffic. Your main job is to:
        - Find the network entry containing the data the user is looking for
        - Identify API endpoints and their purposes
        - Analyze request/response patterns

        ## Finding Relevant Entries

        When the user asks about specific data (e.g., "train prices", "search results", "user data"):

        1. Generate 20-30 relevant search terms that might appear in the response body
           - Include variations: singular/plural, different casings, related terms
           - Include data field names: "price", "amount", "cost", "fare", "total"
           - Include domain-specific terms: "departure", "arrival", "origin", "destination"

        2. Use the `search_responses_by_terms` tool with your terms

        3. Analyze the top results - the entry with the highest score is most likely to contain the data

        ## Available Tools

        - **`search_responses_by_terms`**: Search network entries by a list of terms. Returns top 10 entries ranked by relevance.
          - Pass 20-30 search terms for best results
          - Only searches HTML/JSON response bodies (excludes JS, images, media)
          - Returns: id, url, unique_terms_found, total_hits, score

        - **`get_entry_detail`**: Get full details of a specific network entry by ID.
          - Use this after finding a relevant entry to see headers, request body, response body

        - **`get_response_body_schema`**: Get the schema of a JSON response body.
          - Use this to understand the shape of large JSON responses without retrieving all the data
          - Shows structure with types at every level

        ## Guidelines

        - Be concise and direct in your responses
        - When you find a relevant entry, report its ID and URL
        - Always use search_responses_by_terms first when looking for specific data
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""
        You are a network traffic analyst that autonomously identifies API endpoints.

        ## Your Mission

        Given a user task, find the API endpoint(s) that return the data needed for that task.
        Some tasks require multiple endpoints (e.g., auth -> search -> details).

        ## Process

        1. **Search**: Use `search_responses_by_terms` with 20-30 relevant terms for the task
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
        - request_ids: The network entry request_id(s) for this endpoint (MUST be valid IDs from the data store)
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

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        network_data_store: NetworkDataLoader,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
    ) -> None:
        """
        Initialize the network specialist agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            network_data_store: NetworkDataLoader containing parsed network traffic data (NetworkTransactionEvent objects).
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
        """
        self._network_data_store = network_data_store

        # Autonomous result state
        self._discovery_result: EndpointDiscoveryResult | None = None
        self._discovery_failure: DiscoveryFailureResult | None = None

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
            "NetworkSpecialist initialized with model: %s, chat_thread_id: %s, entries: %d",
            llm_model,
            self._thread.id,
            len(network_data_store.entries),
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with traffic stats context, host stats, and likely API URLs."""
        stats = self._network_data_store.stats
        stats_context = (
            f"\n\n## Network Traffic Context\n"
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
                f"Use the `get_unique_urls` tool to see all other URLs in the captured traffic."
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

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode with traffic context."""
        stats = self._network_data_store.stats
        stats_context = (
            f"\n\n## Network Traffic Context\n"
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
        if self.can_finalize:
            # Get urgency based on iteration count
            remaining_iterations = self._autonomous_config.max_iterations - self._autonomous_iteration
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

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TASK: {task}\n\n"
            "Find the main API endpoint that returns the data needed for this task. "
            "Search, analyze, and when confident, use finalize_result to report your findings. "
            "If after thorough search you determine the endpoint does not exist in the traffic, "
            "use finalize_failure to report why."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        if tool_name == "finalize_result" and self._discovery_result is not None:
            return True
        if tool_name == "finalize_failure" and self._discovery_failure is not None:
            return True
        return False

    def _get_autonomous_result(self) -> BaseModel | None:
        return self._discovery_result or self._discovery_failure

    def _reset_autonomous_state(self) -> None:
        self._discovery_result = None
        self._discovery_failure = None

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _search_responses_by_terms(self, terms: list[str]) -> dict[str, Any]:
        """
        Search RESPONSE bodies by a list of terms.

        Searches HTML/JSON response bodies (excludes JS, images, media) and returns
        top 10-20 entries ranked by relevance score. Pass 20-30 search terms for best results.

        Args:
            terms: List of 20-30 search terms to look for in response bodies.
                Include variations, related terms, and field names.
        """
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

    @agent_tool()
    @token_optimized
    def _get_entry_detail(self, request_id: str) -> dict[str, Any]:
        """
        Get full details of a specific network entry by request_id.

        Returns method, URL, headers, request body, and response body.

        Args:
            request_id: The request_id of the network entry to retrieve.
        """
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

    @agent_tool()
    @token_optimized
    def _get_response_body_schema(self, request_id: str) -> dict[str, Any]:
        """
        Get the schema of a network entry's JSON response body.

        Shows structure with types at every level. Useful for understanding the
        shape of large JSON responses.

        Args:
            request_id: The request_id of the network entry to get schema for.
        """
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

    @agent_tool()
    @token_optimized
    def _get_unique_urls(self) -> dict[str, Any]:
        """
        Get all unique URLs from the captured network traffic.

        Returns a sorted list of all unique URLs observed in the traffic.
        """
        url_counts = self._network_data_store.url_counts
        return {
            "total_unique_urls": len(url_counts),
            "url_counts": url_counts,
        }

    @agent_tool()
    def _execute_python(self, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment to analyze network entries.

        The variable `entries` is pre-loaded as a list of NetworkTransactionEvent dicts.
        Each entry has: request_id, url, method, status, mime_type, request_headers,
        response_headers, post_data, response_body. Use print() to output results.
        Example: for e in entries[:5]: print(e['url'])

        Args:
            code: Python code to execute. `entries` is a list of network entry dicts.
                `json` module is available. Use print() for output. Imports are disabled.
        """
        entries = [e.model_dump() for e in self._network_data_store.entries]
        return execute_python_sandboxed(code, extra_globals={"entries": entries})

    @agent_tool()
    @token_optimized
    def _search_requests_by_terms(
        self,
        terms: list[str],
        search_in: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search the REQUEST side of network entries (URL, headers, body) for terms.

        Useful for finding where sensitive data or parameters are sent.
        Returns entries ranked by relevance.

        Args:
            terms: List of search terms to look for in requests.
            search_in: Where to search: 'url', 'headers', 'body'. Defaults to all.
        """
        if not terms:
            return {"error": "No search terms provided"}

        if search_in is None:
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
                import json as json_module
                headers_str = json_module.dumps(entry.request_headers).lower()
                for term in terms_lower:
                    count = headers_str.count(term)
                    if count > 0:
                        found_terms.add(term)
                        total_hits += count
                        if "headers" not in matched_in:
                            matched_in.append("headers")

            # Search body
            if "body" in search_in and entry.post_data:
                import json as json_module
                if isinstance(entry.post_data, (dict, list)):
                    post_data_str = json_module.dumps(entry.post_data)
                else:
                    post_data_str = str(entry.post_data)
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

    @agent_tool()
    @token_optimized
    def _search_response_bodies(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search response bodies for a specific value and return matches with context.

        Unlike search_responses_by_terms which ranks by relevance across many terms,
        this tool finds exact matches for a single value and shows surrounding context.
        Useful for finding where specific data (IDs, tokens, values) appears.

        Args:
            value: The exact value to search for in response bodies.
            case_sensitive: Whether the search should be case-sensitive. Defaults to false.
        """
        if not value:
            return {"error": "value is required"}

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

    @agent_tool(availability=lambda self: self.can_finalize)
    @token_optimized
    def _finalize_result(self, endpoints: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Finalize the endpoint discovery with your findings.

        Call this when you have identified the API endpoint(s) needed for the user's task.
        You can specify multiple endpoints if the task requires a multi-step flow
        (e.g., authenticate -> search -> get details).
        NOTE: All request_ids must be valid IDs from the data store.

        Args:
            endpoints: List of discovered endpoints. Order by execution sequence if multi-step.
        """
        if not endpoints:
            return {"error": "endpoints list is required and cannot be empty"}

        # Validate and build endpoint objects
        discovered_endpoints: list[DiscoveredEndpoint] = []
        for i, ep in enumerate(endpoints):
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
                    "hint": (
                        "These request_ids do not exist in the data store. "
                        "Use get_entry_detail or search tools to find valid request_ids."
                    ),
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

        return {
            "status": "success",
            "message": f"Endpoint discovery finalized with {len(discovered_endpoints)} endpoint(s)",
            "result": self._discovery_result.model_dump(),
        }

    @agent_tool(availability=lambda self: self.can_finalize)
    @token_optimized
    def _finalize_failure(
        self,
        reason: str,
        searched_terms: list[str] | None = None,
        closest_matches: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Signal that the endpoint discovery has failed.

        Call this ONLY when you have exhaustively searched and are confident that
        the required endpoint does NOT exist in the captured traffic. Provide a
        clear explanation of what was searched and why no match was found.

        Args:
            reason: Detailed explanation of why the endpoint could not be found.
            searched_terms: List of key search terms that were tried.
            closest_matches: URLs of entries that came closest to matching (if any).
        """
        if not reason:
            return {"error": "reason is required - explain why the endpoint could not be found"}

        # Store the failure result
        self._discovery_failure = DiscoveryFailureResult(
            reason=reason,
            searched_terms=searched_terms or [],
            closest_matches=closest_matches or [],
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
