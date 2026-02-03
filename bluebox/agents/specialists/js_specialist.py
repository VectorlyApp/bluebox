"""
bluebox/agents/specialists/js_specialist.py

JavaScript specialist agent.

Writes IIFE JavaScript for RoutineJsEvaluateOperation execution in routines.
"""

from __future__ import annotations

import time
from textwrap import dedent
from typing import Any, Callable

from pydantic import BaseModel, Field

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode, specialist_tool
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
from bluebox.llms.infra.js_data_store import JSDataStore
from bluebox.llms.infra.network_data_store import NetworkDataStore
from bluebox.utils.js_utils import generate_js_evaluate_wrapper_js, validate_js
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class JSCodeResult(BaseModel):
    """Successful JS code submission result."""
    js_code: str = Field(description="IIFE-wrapped JavaScript code")
    session_storage_key: str | None = Field(
        default=None,
        description="Key for sessionStorage result",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="Max execution time",
    )
    description: str = Field(description="What the code does")


class JSCodeFailureResult(BaseModel):
    """Failure result when JS code cannot be produced."""
    reason: str = Field(description="Why code could not be produced")
    attempted_approaches: list[str] = Field(
        default_factory=list,
        description="Approaches that were tried",
    )


class JSSpecialist(AbstractSpecialist):
    """
    JavaScript specialist agent.

    Writes IIFE JavaScript for browser execution.
    """

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

        ## Your Capabilities

        1. **Write IIFE JavaScript**: Write JavaScript code for browser execution in routines.
        2. **Inspect DOM**: Analyze page structure via DOM snapshots to inform your code.

        ## Tools

        - **get_dom_snapshot**: Get DOM snapshot (latest by default)
        - **validate_js_code**: Dry-run validation of JS code
        - **submit_js_code**: Submit final validated JS code
        - **execute_js_in_browser**: Test your JavaScript code against the live website. Navigates to the URL and executes your IIFE, returning the result and any console output. Use this selectively — it's most valuable when your code depends on live page state (e.g. reading cookies, sessionStorage, or dynamically rendered DOM). Simple, deterministic code can go straight to submit.

        ## Guidelines

        - Use `validate_js_code` before `submit_js_code` to catch errors early
        - Keep code concise and focused on the specific task
        - Use DOM APIs (querySelector, getElementById, etc.) for element interaction
        - Use sessionStorage for passing data between operations
        - Use `get_dom_snapshot` to understand page structure before writing code
    """)

    _NETWORK_TRAFFIC_PROMPT_SECTION: str = dedent("""
        ## Network Traffic Data

        You have access to captured network traffic from the browser session:
        - **search_network_traffic**: Search/filter captured HTTP requests by method, host, path, status code, content type, or response body text. Returns abbreviated results (no bodies).
        - **get_network_entry**: Get full details of a specific request by its request_id, including headers and response body.

        Use these to understand API endpoints, response formats, and data available on the page.
    """)

    _JS_FILES_PROMPT_SECTION: str = dedent("""
        ## JavaScript Files (Use Sparingly)

        You have access to captured JavaScript files from the browser session. These are the actual
        JS bundles served by the web app — often large and minified.

        **IMPORTANT**: Use these tools when the task specifically requires understanding the
        web app's JavaScript implementation. For example:
        - "Does this site's JS reference a specific token or variable name?"
        - "How does the client-side code handle authentication?"
        - "What API endpoints are hardcoded in the JavaScript?"

        For most DOM manipulation tasks, you likly do NOT need to inspect the site's JS files.

        Tools:
        - **search_js_files**: Search JS file contents by keywords. Returns ranked results by relevance.
        - **get_js_file_content**: Get the content of a specific JS file (truncated for large files).
        - **list_js_files**: List all captured JS files with URLs and sizes.
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = dedent("""\
        You are a JavaScript expert that autonomously writes browser DOM manipulation code.

        ## Your Mission

        Given a task, write IIFE JavaScript code that accomplishes it in the browser context.

        ## Process

        1. **Understand**: Analyze the task and determine what DOM manipulation is needed
        2. **Check DOM**: Use `get_dom_snapshot` to understand the current page structure
        3. **Write**: Write the JavaScript code, validate it, then submit
        4. **Finalize**: Call `submit_js_code` with your validated code

        ## When finalize tools are available

        - **submit_js_code**: Submit your final validated JavaScript code
        - **finalize_failure**: Report that the task cannot be accomplished with JS
        - **execute_js_in_browser**: Test your code against the live website. Most useful when code depends on live page state (cookies, storage, dynamic DOM). Simple, deterministic code can skip this.
    """)

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        network_data_store: NetworkDataStore | None = None,
        js_data_store: JSDataStore | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str | None = None,
    ) -> None:
        self._dom_snapshots = dom_snapshots or []
        self._remote_debugging_address = remote_debugging_address
        self._network_data_store = network_data_store
        self._js_data_store = js_data_store

        # autonomous result state
        self._js_result: JSCodeResult | None = None
        self._js_failure: JSCodeFailureResult | None = None

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

        # Urgency notices
        if self.can_finalize:
            remaining = self._autonomous_config.max_iterations - self._autonomous_iteration
            if remaining <= 2:
                context_parts.append(
                    f"\n\n## CRITICAL: Only {remaining} iterations remaining!\n"
                    f"You MUST call submit_js_code or finalize_failure NOW!"
                )
            elif remaining <= 4:
                context_parts.append(
                    f"\n\n## URGENT: Only {remaining} iterations remaining.\n"
                    f"Finalize your code soon."
                )
            else:
                context_parts.append(
                    "\n\n## Finalize tools are now available.\n"
                    "Call submit_js_code when your code is ready."
                )
        else:
            context_parts.append(
                f"\n\n## Continue exploring (iteration {self._autonomous_iteration}).\n"
                "Finalize tools will become available after more exploration."
            )

        return "".join(context_parts)

    # _register_tools and _execute_tool are provided by the base class
    # via @specialist_tool decorators below.

    def _get_autonomous_initial_message(self, task: str) -> str:
        return (
            f"TASK: {task}\n\n"
            "Write IIFE JavaScript code to accomplish this task in the browser. "
            "Research the available JS files and DOM structure if needed, then validate and submit your code."
        )

    def _check_autonomous_completion(self, tool_name: str) -> bool:
        if tool_name == "submit_js_code" and self._js_result is not None:
            return True
        if tool_name == "finalize_failure" and self._js_failure is not None:
            return True
        return False

    def _get_autonomous_result(self) -> BaseModel | None:
        return self._js_result or self._js_failure

    def _reset_autonomous_state(self) -> None:
        self._js_result = None
        self._js_failure = None

    ## Tools

    @specialist_tool(
        description="Dry-run validation of JavaScript code. Checks IIFE format and blocked patterns without submitting.",
        parameters={
            "type": "object",
            "properties": {
                "js_code": {
                    "type": "string",
                    "description": "JavaScript code to validate.",
                }
            },
            "required": ["js_code"],
        },
    )
    @token_optimized
    def _validate_js_code(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        js_code = tool_arguments.get("js_code", "")
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


    @specialist_tool(
        description=(
            "Get a DOM snapshot. Returns the document structure and truncated strings table. "
            "Defaults to latest snapshot."
        ),
        availability=lambda self: bool(self._dom_snapshots),
        parameters={
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Snapshot index (0-based). Defaults to latest (-1).",
                }
            },
        },
    )
    @token_optimized
    def _get_dom_snapshot(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._dom_snapshots:
            return {"error": "No DOM snapshots available"}

        index = tool_arguments.get("index", -1)
        if index < 0:
            index = len(self._dom_snapshots) + index
        if index < 0 or index >= len(self._dom_snapshots):
            return {
                "error": f"Snapshot index {index} out of range (0-{len(self._dom_snapshots) - 1})",
            }

        snapshot = self._dom_snapshots[index]

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


    @specialist_tool(
        description=(
            "Submit validated JavaScript code as the final result. "
            "The code must be IIFE-wrapped and pass all validation checks."
        ),
        availability=lambda self: self.can_finalize,
        parameters={
            "type": "object",
            "properties": {
                "js_code": {
                    "type": "string",
                    "description": "IIFE-wrapped JavaScript code.",
                },
                "description": {
                    "type": "string",
                    "description": "Brief description of what the code does.",
                },
                "session_storage_key": {
                    "type": "string",
                    "description": "Optional sessionStorage key to store the result.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max execution time in seconds (default 5.0).",
                },
            },
            "required": ["js_code", "description"],
        },
    )
    @token_optimized
    def _submit_js_code(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        js_code = tool_arguments.get("js_code", "")
        description = tool_arguments.get("description", "")
        session_storage_key = tool_arguments.get("session_storage_key")
        timeout_seconds = tool_arguments.get("timeout_seconds", 5.0)

        if not description:
            return {"error": "description is required"}

        # Validate the code
        result = validate_js(js_code)
        if result.errors:
            return {
                "error": "Validation failed",
                "errors": result.errors,
                "warnings": result.warnings,
            }

        # Store result
        self._js_result = JSCodeResult(
            js_code=js_code,
            session_storage_key=session_storage_key,
            timeout_seconds=timeout_seconds,
            description=description,
        )

        logger.info("JS code submitted: %s", description)

        return {
            "status": "success",
            "message": "JavaScript code submitted successfully",
            "result": self._js_result.model_dump(),
        }


    @specialist_tool(
        description="Report that the JavaScript task cannot be accomplished.",
        availability=lambda self: self.can_finalize,
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the task cannot be accomplished with JavaScript.",
                },
                "attempted_approaches": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of approaches that were tried.",
                },
            },
            "required": ["reason"],
        },
    )
    @token_optimized
    def _finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        reason = tool_arguments.get("reason", "")
        attempted_approaches = tool_arguments.get("attempted_approaches", [])

        if not reason:
            return {"error": "reason is required"}

        self._js_failure = JSCodeFailureResult(
            reason=reason,
            attempted_approaches=attempted_approaches,
        )

        logger.info("JS specialist failed: %s", reason)

        return {
            "status": "failure",
            "message": "JavaScript task marked as failed",
            "result": self._js_failure.model_dump(),
        }


    @specialist_tool(
        description=(
            "Search/filter captured HTTP requests. Returns abbreviated results (no bodies). "
            "All parameters are optional filters."
        ),
        availability=lambda self: self._network_data_store is not None,
        parameters={
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": "HTTP method filter (e.g. GET, POST).",
                },
                "host_contains": {
                    "type": "string",
                    "description": "Substring match on the request host.",
                },
                "path_contains": {
                    "type": "string",
                    "description": "Substring match on the request path.",
                },
                "status_code": {
                    "type": "integer",
                    "description": "Exact HTTP status code filter.",
                },
                "content_type_contains": {
                    "type": "string",
                    "description": "Substring match on response content type.",
                },
                "response_body_contains": {
                    "type": "string",
                    "description": "Search for text within response bodies.",
                },
            },
        },
    )
    @token_optimized
    def _search_network_traffic(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search captured network traffic with optional filters."""
        if self._network_data_store is None:
            return {"error": "No network data store available"}

        response_body_contains = tool_arguments.pop("response_body_contains", None)

        # Use search_entries for structured filters
        entries = self._network_data_store.search_entries(
            method=tool_arguments.get("method"),
            host_contains=tool_arguments.get("host_contains"),
            path_contains=tool_arguments.get("path_contains"),
            status_code=tool_arguments.get("status_code"),
            content_type_contains=tool_arguments.get("content_type_contains"),
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

        return {"count": len(results), "entries": results}


    @specialist_tool(
        description="Get full details of a single captured HTTP request by request_id, including headers and response body.",
        availability=lambda self: self._network_data_store is not None,
        parameters={
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "The request_id from search_network_traffic results.",
                },
                "include_response_body": {
                    "type": "boolean",
                    "description": "Whether to include the response body (default true).",
                },
                "max_body_length": {
                    "type": "integer",
                    "description": "Max characters for the response body (default 5000).",
                },
            },
            "required": ["request_id"],
        },
    )
    @token_optimized
    def _get_network_entry(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get full details of a single network entry by request_id."""
        if self._network_data_store is None:
            return {"error": "No network data store available"}

        request_id = tool_arguments.get("request_id", "")
        include_response_body = tool_arguments.get("include_response_body", True)
        max_body_length = tool_arguments.get("max_body_length", 5000)

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


    @specialist_tool(
        description=(
            "Search captured JS files by keywords. Returns ranked results by relevance. "
            "Use this to find JS files that reference specific tokens, variables, or API endpoints."
        ),
        availability=lambda self: self._js_data_store is not None,
        parameters={
            "type": "object",
            "properties": {
                "terms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Search terms (case-insensitive). Files are ranked by how many terms match and total hits.",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                },
            },
            "required": ["terms"],
        },
    )
    @token_optimized
    def _search_js_files(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Search JS file contents by keywords."""
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        terms = tool_arguments.get("terms", [])
        top_n = tool_arguments.get("top_n", 10)

        if not terms:
            return {"error": "At least one search term is required"}

        results = self._js_data_store.search_by_terms(terms, top_n=top_n)

        return {
            "count": len(results),
            "terms_searched": terms,
            "results": results,
        }


    @specialist_tool(
        description=(
            "Get the content of a specific JS file by request_id. "
            "Content is truncated for large files. Use search_js_files first to find relevant files."
        ),
        availability=lambda self: self._js_data_store is not None,
        parameters={
            "type": "object",
            "properties": {
                "request_id": {
                    "type": "string",
                    "description": "The request_id from search_js_files or list_js_files results.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters to return (default 10000). Large JS files are truncated.",
                },
            },
            "required": ["request_id"],
        },
    )
    @token_optimized
    def _get_js_file_content(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Get the content of a JS file."""
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        request_id = tool_arguments.get("request_id", "")
        max_chars = tool_arguments.get("max_chars", 10000)

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


    @specialist_tool(
        description=(
            "List all captured JS files with URLs and sizes. "
            "Use this to see what JS files are available before searching."
        ),
        availability=lambda self: self._js_data_store is not None,
        parameters={
            "type": "object",
            "properties": {},
        },
    )
    @token_optimized
    def _list_js_files(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """List all captured JS files."""
        if self._js_data_store is None:
            return {"error": "No JS data store available"}

        files = self._js_data_store.list_files()
        stats = self._js_data_store.stats

        return {
            "total_files": stats.total_files,
            "total_bytes": stats.total_bytes,
            "files": files,
        }


    @specialist_tool(
        description=(
            "Test JavaScript code against the live website. "
            "Navigates to the URL and executes your IIFE, returning the result and any console output. "
            "Use this to verify your code works before submitting. "
            "Set keep_open=true to keep the browser tab open after execution (useful for visual changes)."
        ),
        availability=lambda self: bool(self._remote_debugging_address),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to navigate to first (or empty string to skip navigation).",
                },
                "js_code": {
                    "type": "string",
                    "description": "IIFE JavaScript code to execute.",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max execution time in seconds (default 5.0).",
                },
                "keep_open": {
                    "type": "boolean",
                    "description": "If true, keep the browser tab open after execution instead of closing it. Useful for visual changes. Default false.",
                },
            },
            "required": ["url", "js_code"],
        },
    )
    @token_optimized
    def _execute_js_in_browser(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Navigate to a URL and execute JS code via CDP, returning the result."""
        if not self._remote_debugging_address:
            return {"error": "No browser connection configured"}

        url = tool_arguments.get("url", "")
        js_code = tool_arguments.get("js_code", "")
        timeout_seconds = tool_arguments.get("timeout_seconds", 5.0)
        keep_open = tool_arguments.get("keep_open", False)

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
