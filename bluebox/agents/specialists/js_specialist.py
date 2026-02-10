"""
bluebox/agents/specialists/js_specialist.py

JavaScript specialist agent.

Writes IIFE JavaScript for RoutineJsEvaluateOperation execution in routines.
"""

from __future__ import annotations

import time
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Callable

from bluebox.agents.abstract_agent import AgentCard, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.cdp.connection import (
    cdp_new_tab,
    create_cdp_helpers,
    dispose_context,
)
from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.utils.js_utils import generate_js_evaluate_wrapper_js, validate_js
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


class JSSpecialist(AbstractSpecialist):
    """
    JavaScript specialist agent.

    Writes IIFE JavaScript for browser execution.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Writes and validates IIFE JavaScript for browser execution. Use for cookie/token "
            "extraction, DOM scraping, and page state manipulation."
        ),
    )

    _BASE_CONTEXT: str = dedent("""\
        ## Context

        Your code executes inside a live browser session as part of a web automation routine.
        Typical tasks include:
        - Extracting cookies, auth tokens, or CSRF tokens from the page
        - Reading values from `document.cookie`, `localStorage`, or `sessionStorage`
        - Scraping rendered DOM content (text, attributes, hidden fields)
        - Setting up page state (e.g. filling inputs, clicking elements) for downstream operations
        - Computing or transforming values needed by subsequent fetch operations

        Your JavaScript is one step in a larger routine — other steps handle network requests
        (via RoutineFetchOperation). Your job is to interact with the browser page itself.

        ## JavaScript Code Requirements

        All JavaScript code you write MUST:
        - Be wrapped in an IIFE: `(function() { ... })()` or `(() => { ... })()`
        - Return a value using `return` (the return value is captured)
        - Optionally store results in sessionStorage via `session_storage_key`

        ## Code Formatting

        - Write readable, well-formatted JavaScript. Never write extremely long single-line IIFEs.
        - Use proper indentation (2 spaces), line breaks between statements, and descriptive variable names.
        - Each statement should be on its own line. Complex expressions should be broken across lines.

        ## Blocked Patterns

        The following are NOT allowed in your JavaScript code:
        - `eval()`, `Function()` — no dynamic code generation
        - `fetch()`, `XMLHttpRequest`, `WebSocket`, `sendBeacon` — no network requests (use RoutineFetchOperation instead)
        - `addEventListener()`, `MutationObserver`, `IntersectionObserver` — no persistent event hooks
        - `window.close()` — no navigation/lifecycle control
    """)

    SYSTEM_PROMPT: str = dedent("""\
        You are a JavaScript expert specializing in browser DOM manipulation.

        ## Guidelines

        - Validate before submitting
        - Keep code concise and focused
        - Use `get_dom_snapshot` to understand page structure before writing code
    """)

    # These conditional prompt sections are no longer needed — AbstractAgent._call_llm()
    # auto-injects a "## Tools" section listing all available/unavailable tools.
    _NETWORK_TRAFFIC_PROMPT_SECTION: str = ""
    _JS_FILES_PROMPT_SECTION: str = ""

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a JavaScript expert that autonomously writes browser DOM manipulation code.

        ## Your Mission

        Given a task, write IIFE JavaScript code that accomplishes it in the browser context.

        ## Process

        1. **Understand**: Analyze the task requirements
        2. **Check DOM**: Use `get_dom_snapshot` to understand page structure
        3. **Write**: Write and validate the JavaScript code
        4. **Test** (optional): Use `execute_js_in_browser` if code depends on live page state
        5. **Finalize**: Call the appropriate finalize tool with your code
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        network_data_store: NetworkDataLoader | None = None,
        js_data_store: JSDataLoader | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
    ) -> None:
        self._dom_snapshots = dom_snapshots or []
        self._remote_debugging_address = remote_debugging_address
        self._network_data_store = network_data_store
        self._js_data_store = js_data_store

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=documentation_data_loader,
        )
        logger.debug(
            "JSSpecialist initialized: dom_snapshots=%d, network_data_store=%s, js_data_store=%s, browser=%s",
            len(self._dom_snapshots),
            "yes" if self._network_data_store is not None else "no",
            "yes" if self._js_data_store is not None else "no",
            "yes" if remote_debugging_address else "no",
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        context_parts = [self.SYSTEM_PROMPT, "\n", self._BASE_CONTEXT]

        if self._dom_snapshots:
            latest = self._dom_snapshots[-1]
            context_parts.append(
                f"\n\n## DOM Context\n"
                f"- {len(self._dom_snapshots)} snapshot(s) available\n"
                f"- Latest page: {latest.url}\n"
                f"- Latest title: {latest.title or 'N/A'}\n"
            )

        if self._network_data_store is not None:
            context_parts.append(self._NETWORK_TRAFFIC_PROMPT_SECTION)

        if self._js_data_store is not None:
            context_parts.append(self._JS_FILES_PROMPT_SECTION)

        return "".join(context_parts)

    def _get_autonomous_system_prompt(self) -> str:
        context_parts = [self.AUTONOMOUS_SYSTEM_PROMPT, "\n", self._BASE_CONTEXT]

        if self._dom_snapshots:
            latest = self._dom_snapshots[-1]
            context_parts.append(
                f"\n\n## DOM Context\n"
                f"- {len(self._dom_snapshots)} snapshot(s) available\n"
                f"- Latest page: {latest.url}\n"
            )

        if self._network_data_store is not None:
            context_parts.append(self._NETWORK_TRAFFIC_PROMPT_SECTION)

        if self._js_data_store is not None:
            context_parts.append(self._JS_FILES_PROMPT_SECTION)

        context_parts.append(self._get_output_schema_prompt_section())
        context_parts.append(self._get_urgency_notice())

        return "".join(context_parts)

    def _get_autonomous_initial_message(self, task: str) -> str:
        # Use correct tool names based on whether output schema is set
        if self.has_output_schema:
            finalize_success = "finalize_with_output"
        else:
            finalize_success = "finalize_result"

        return (
            f"TASK: {task}\n\n"
            f"Write IIFE JavaScript code to accomplish this task in the browser. "
            f"Research the available JS files and DOM structure if needed, then validate "
            f"and call {finalize_success} with your code."
        )

    ## Tools

    @agent_tool()
    @token_optimized
    def _validate_js_code(self, js_code: str) -> dict[str, Any]:
        """
        Dry-run validation of JavaScript code. Checks IIFE format and blocked patterns without submitting.

        Args:
            js_code: JavaScript code to validate.
        """
        result = validate_js(js_code)
        if result.errors:
            return {
                "valid": False,
                "errors": result.errors,
                "warnings": result.warnings,
            }
        if result.warnings:
            return {
                "valid": True,
                "warnings": result.warnings,
                "message": "Code passes validation but has formatting warnings. Consider reformatting.",
            }
        return {
            "valid": True,
            "message": "Code passes all validation checks.",
        }


    @agent_tool(availability=lambda self: bool(self._dom_snapshots))
    @token_optimized
    def _get_dom_snapshot(self, index: int = -1) -> dict[str, Any]:
        """
        Get a DOM snapshot. Returns the document structure and truncated strings table. Defaults to latest snapshot.

        Args:
            index: Snapshot index (0-based). Defaults to latest (-1).
        """
        if not self._dom_snapshots:
            return {"error": "No DOM snapshots available"}

        resolved_index = index
        if resolved_index < 0:
            resolved_index = len(self._dom_snapshots) + resolved_index
        if resolved_index < 0 or resolved_index >= len(self._dom_snapshots):
            return {
                "error": f"Snapshot index {resolved_index} out of range (0-{len(self._dom_snapshots) - 1})",
            }

        snapshot = self._dom_snapshots[resolved_index]

        # Truncate strings table to keep token usage reasonable
        strings = snapshot.strings
        truncate_length = 1_000  # chars
        truncated_strings = strings[:truncate_length] if len(strings) > truncate_length else strings

        return {
            "url": snapshot.url,
            "title": snapshot.title,
            "strings_count": len(strings),
            "strings_sample": truncated_strings,
            "documents_count": len(snapshot.documents),
            "documents": snapshot.documents,
        }


    @agent_tool(availability=lambda self: self._network_data_store is not None)
    @token_optimized
    def _search_network_traffic(
        self,
        method: str | None = None,
        host_contains: str | None = None,
        path_contains: str | None = None,
        status_code: int | None = None,
        content_type_contains: str | None = None,
        response_body_contains: str | None = None,
    ) -> dict[str, Any]:
        """
        Search/filter captured HTTP requests. Returns abbreviated results (no bodies). All parameters are optional filters.

        Args:
            method: HTTP method filter (e.g. GET, POST).
            host_contains: Substring match on the request host.
            path_contains: Substring match on the request path.
            status_code: Exact HTTP status code filter.
            content_type_contains: Substring match on response content type.
            response_body_contains: Search for text within response bodies.
        """
        if self._network_data_store is None:
            return {"error": "No network data store available"}

        # Use search_entries for structured filters
        entries = self._network_data_store.search_entries(
            method=method,
            host_contains=host_contains,
            path_contains=path_contains,
            status_code=status_code,
            content_type_contains=content_type_contains,
        )

        # If body text search requested, intersect with body search results
        if response_body_contains:
            body_results = self._network_data_store.search_response_bodies(response_body_contains)
            body_ids = {r["id"] for r in body_results}
            entries = [e for e in entries if e.request_id in body_ids]

        # Return abbreviated results
        results = [
            {
                "request_id": e.request_id,
                "url": e.url,
                "method": e.method,
                "status": e.status,
                "mime_type": e.mime_type,
            }
            for e in entries
        ]
        return {
            "count": len(results),
            "entries": results,
        }


    @agent_tool(availability=lambda self: self._network_data_store is not None)
    @token_optimized
    def _get_network_entry(
        self,
        request_id: str,
        include_response_body: bool = True,
        max_body_length: int = 5000,
    ) -> dict[str, Any]:
        """
        Get full details of a single captured HTTP request by request_id, including headers and response body.

        Args:
            request_id: The request_id from search_network_traffic results.
            include_response_body: Whether to include the response body (default true).
            max_body_length: Max characters for the response body (default 5000).
        """
        if self._network_data_store is None:
            return {"error": "No network data store available"}

        entry = self._network_data_store.get_entry(request_id)
        if entry is None:
            return {"error": f"No entry found for request_id: {request_id}"}

        result: dict[str, Any] = {
            "request_id": entry.request_id,
            "url": entry.url,
            "method": entry.method,
            "status": entry.status,
            "mime_type": entry.mime_type,
            "request_headers": entry.request_headers,
            "response_headers": entry.response_headers,
            "post_data": entry.post_data,
        }

        if include_response_body:
            body = entry.response_body
            if len(body) > max_body_length:
                result["response_body"] = body[:max_body_length]
                result["response_body_truncated"] = True
                result["response_body_full_length"] = len(body)
            else:
                result["response_body"] = body
                result["response_body_truncated"] = False

        return result


    @agent_tool(availability=lambda self: self._js_data_store is not None)
    @token_optimized
    def _search_js_files(self, terms: list[str], top_n: int = 10) -> dict[str, Any]:
        """
        Search captured JS files by keywords.

        Returns ranked results by relevance. Use this to find JS files that
        reference specific tokens, variables, or API endpoints.

        Args:
            terms: Search terms (case-insensitive). Ranked by match count and hits.
            top_n: Max results to return (default 10).
        """
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        if not terms:
            return {"error": "At least one search term is required"}

        results = self._js_data_store.search_by_terms(terms, top_n=top_n)

        return {
            "count": len(results),
            "terms_searched": terms,
            "results": results,
        }


    @agent_tool(availability=lambda self: self._js_data_store is not None)
    @token_optimized
    def _search_js_files_regex(
        self,
        pattern: str,
        top_n: int = 20,
        max_matches_per_file: int = 10,
        snippet_padding_chars: int = 80,
    ) -> dict[str, Any]:
        """
        Search captured JS files by regex pattern.

        Returns matches with surrounding context snippets. WARNING: Regex searches
        can be expensive on large minified JS files. There is a 15-second timeout.
        Prefer search_js_files (keyword search) for simple lookups.

        Args:
            pattern: Regex pattern to search for (case-insensitive).
            top_n: Max files to return (default 20).
            max_matches_per_file: Max matches per file (default 10).
            snippet_padding_chars: Characters of context around each match (default 80).
        """
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        if not pattern:
            return {"error": "pattern is required"}

        result = self._js_data_store.search_by_regex(
            pattern=pattern,
            top_n=top_n,
            max_matches_per_file=max_matches_per_file,
            snippet_padding_chars=snippet_padding_chars,
        )

        if result["error"]:
            return {"error": result["error"]}

        return {
            "pattern": pattern,
            "files_found": len(result["files"]),
            "timed_out": result["timed_out"],
            "files": result["files"],
        }


    @agent_tool(availability=lambda self: self._js_data_store is not None)
    @token_optimized
    def _get_js_file_content(self, request_id: str, max_chars: int = 10_000) -> dict[str, Any]:
        """
        Get the content of a specific JS file by request_id.

        Content is truncated for large files. Use search_js_files first to find
        relevant files.

        Args:
            request_id: The request_id from search_js_files or list_js_files results.
            max_chars: Max characters to return (default 10000). Large files truncated.
        """
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        entry = self._js_data_store.get_file(request_id)
        if entry is None:
            return {"error": f"No JS file found for request_id: {request_id}"}

        content = self._js_data_store.get_file_content(request_id, max_chars=max_chars)

        return {
            "request_id": request_id,
            "url": entry.url,
            "content": content,
            "full_size": len(entry.response_body) if entry.response_body else 0,
        }


    @agent_tool(availability=lambda self: self._js_data_store is not None)
    @token_optimized
    def _list_js_files(self) -> dict[str, Any]:
        """
        List all captured JS files with URLs and sizes.

        Use this to see what JS files are available before searching.
        """
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        files = self._js_data_store.list_files()
        stats = self._js_data_store.stats

        return {
            "total_files": stats.total_files,
            "total_bytes": stats.total_bytes,
            "files": files,
        }


    @agent_tool(availability=lambda self: bool(self._remote_debugging_address))
    @token_optimized
    def _execute_js_in_browser(
        self,
        url: str,
        js_code: str,
        timeout_seconds: float = 5.0,
        keep_open: bool = False,
    ) -> dict[str, Any]:
        """
        Test JavaScript code against the live website.

        Navigates to the URL and executes your IIFE, returning the result and any
        console output. Use this to verify your code works before submitting.

        Args:
            url: URL to navigate to first (or empty string to skip navigation).
            js_code: IIFE JavaScript code to execute.
            timeout_seconds: Max execution time in seconds (default 5.0).
            keep_open: If true, keep the browser tab open after execution.
                Useful for visual changes. Default false.
        """
        if not self._remote_debugging_address:
            return {"error": "No browser connection configured"}

        # Validate JS first
        result = validate_js(js_code)
        if result.errors:
            return {
                "error": "Validation failed",
                "errors": result.errors,
                "warnings": result.warnings,
            }

        overall_timeout = timeout_seconds + 5.0
        deadline = time.time() + overall_timeout

        target_id = None
        browser_context_id = None
        browser_ws = None

        try:
            # Open new incognito tab
            target_id, browser_context_id, browser_ws = cdp_new_tab(
                self._remote_debugging_address,
                incognito=True,
                url="about:blank",
            )

            send_cmd, _, recv_until = create_cdp_helpers(browser_ws)

            # Attach to target with flattened session
            attach_id = send_cmd(
                "Target.attachToTarget",
                {"targetId": target_id, "flatten": True},
            )
            attach_reply = recv_until(lambda m: m.get("id") == attach_id, deadline)
            if "error" in attach_reply:
                return {"error": f"Failed to attach: {attach_reply['error']}"}
            session_id = attach_reply["result"]["sessionId"]

            # Enable Page and Runtime domains
            page_id = send_cmd("Page.enable", session_id=session_id)
            recv_until(lambda m: m.get("id") == page_id, deadline)
            runtime_id = send_cmd("Runtime.enable", session_id=session_id)
            recv_until(lambda m: m.get("id") == runtime_id, deadline)

            # Navigate if URL provided
            if url:
                nav_id = send_cmd(
                    "Page.navigate",
                    {"url": url},
                    session_id=session_id,
                )
                recv_until(lambda m: m.get("id") == nav_id, deadline)

                # Wait for page load
                try:
                    recv_until(
                        lambda m: m.get("method") == "Page.loadEventFired",
                        deadline,
                    )
                except TimeoutError:
                    return {"error": "Page load timed out", "url": url}

            # Wrap JS in evaluate wrapper (captures console.log, etc.)
            wrapped_js = generate_js_evaluate_wrapper_js(js_code)

            # Execute via Runtime.evaluate
            eval_id = send_cmd(
                "Runtime.evaluate",
                {
                    "expression": wrapped_js,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
                session_id=session_id,
            )
            eval_reply = recv_until(lambda m: m.get("id") == eval_id, deadline)

            if "error" in eval_reply:
                return {"error": f"Runtime.evaluate error: {eval_reply['error']}"}

            # Parse the result
            result_obj = eval_reply.get("result", {}).get("result", {})
            value = result_obj.get("value", {})

            if isinstance(value, dict):
                return {
                    "result": value.get("result"),
                    "console_logs": value.get("console_logs", []),
                    "error": value.get("execution_error"),
                    "storage_error": value.get("storage_error"),
                }

            return {"result": value, "console_logs": [], "error": None}

        except TimeoutError:
            return {"error": "Operation timed out"}
        except Exception as e:
            logger.error("execute_js_in_browser failed: %s", e)
            return {"error": f"Browser execution failed: {e}"}
        finally:
            # Cleanup: close target and dispose context (skip if keep_open)
            if keep_open:
                if browser_ws:
                    try:
                        browser_ws.close()
                    except Exception:
                        pass
            else:
                if browser_ws:
                    try:
                        if target_id:
                            send_cmd_cleanup, _, _ = create_cdp_helpers(browser_ws)
                            send_cmd_cleanup("Target.closeTarget", {"targetId": target_id})
                    except Exception:
                        pass
                    try:
                        browser_ws.close()
                    except Exception:
                        pass
                if browser_context_id and self._remote_debugging_address:
                    try:
                        dispose_context(self._remote_debugging_address, browser_context_id)
                    except Exception:
                        pass
