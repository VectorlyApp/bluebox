"""
bluebox/agents/specialists/network_specialist.py

# NOTE: THIS AGENT IS IN BETA AND NOT READY FOR PRODUCTION YET

Agent specialized in searching through network traffic data.

Contains:
- NetworkSpecialist: Specialist for network traffic analysis
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable
from urllib.parse import urlparse, parse_qs

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
        Return structured output matching the orchestrator's expected schema.

        ## Process

        1. **Search**: Use `search_responses_by_terms` with 20-30 relevant terms for the task
        2. **Analyze**: Look at top results, examine their structure with `get_response_body_schema`
        3. **Verify**: Use `get_entry_detail` to confirm the endpoint has the right data
        4. **Finalize**: Call `finalize_with_output` with your findings matching the expected schema

        ## Strategy

        - Identify ALL endpoints needed to complete the task
        - Look for API/XHR calls (not HTML pages, JS files, or images)
        - Prefer endpoints with structured JSON responses
        - Consider multi-step flows: authentication, search, pagination, detail fetches

        ## When finalize tools are available

        After sufficient exploration, `finalize_with_output` and `finalize_with_failure` become available.

        - **finalize_with_output**: Submit your findings matching the expected output schema
        - **finalize_with_failure**: Report that the task could not be completed

        Use `add_note()` before finalizing to record any notes, complaints, warnings, or errors.
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

        # Include output schema if set by orchestrator
        schema_section = self._get_output_schema_prompt_section()

        # Add finalize tool availability notice
        if self.can_finalize:
            remaining_iterations = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining_iterations <= 2:
                finalize_notice = (
                    f"\n\n## CRITICAL: YOU MUST CALL finalize_with_output NOW!\n"
                    f"Only {remaining_iterations} iterations remaining."
                )
            elif remaining_iterations <= 4:
                finalize_notice = (
                    f"\n\n## URGENT: Call finalize_with_output soon!\n"
                    f"Only {remaining_iterations} iterations remaining."
                )
            else:
                finalize_notice = (
                    "\n\n## IMPORTANT: finalize_with_output is now available!\n"
                    "Call it when you have identified the endpoint."
                )
        else:
            finalize_notice = (
                f"\n\n## Note: Continue exploring\n"
                f"Finalize tools will become available after more exploration. "
                f"Currently on iteration {self._autonomous_iteration}."
            )

        return self.AUTONOMOUS_SYSTEM_PROMPT + stats_context + urls_context + schema_section + finalize_notice

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TASK: {task}\n\n"
            "Find the main API endpoint that returns the data needed for this task. "
            "Search, analyze, and when confident, use finalize_result to report your findings. "
            "If after thorough search you determine the endpoint does not exist in the traffic, "
            "use finalize_failure to report why."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        # Delegate to base class (handles generic finalize tools)
        return super()._check_autonomous_completion(tool_name)

    def _get_autonomous_result(self):
        # Delegate to base class (returns wrapped result)
        return super()._get_autonomous_result()

    def _reset_autonomous_state(self) -> None:
        # Call base class to reset generic state
        super()._reset_autonomous_state()

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
