"""
bluebox/agents/specialists/js_specialist.py

JavaScript specialist agent.

Two roles:
1. Interpret JavaScript files served by the web server
2. Write IIFE JavaScript for RoutineJsEvaluateOperation execution in routines
"""

from __future__ import annotations

import time
from typing import Any, Callable

from pydantic import BaseModel, Field

from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist
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
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.infra.js_data_store import JSDataStore
from bluebox.utils.js_utils import generate_js_evaluate_wrapper_js, validate_js
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


# Result models

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

    Analyzes served JS files and writes IIFE JavaScript for browser execution.
    """

    SYSTEM_PROMPT: str = """You are a JavaScript expert specializing in browser DOM manipulation.

## Your Capabilities

1. **Analyze served JS files**: Search and read JavaScript files from the web server to understand client-side logic, APIs, and data structures.
2. **Write IIFE JavaScript**: Write new JavaScript code for browser execution in routines.

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

## Tools

- **search_js_files**: Search JS file response bodies by terms
- **get_js_file_detail**: Get full JS file content by request_id
- **get_dom_snapshot**: Get DOM snapshot (latest by default)
- **validate_js_code**: Dry-run validation of JS code
- **submit_js_code**: Submit final validated JS code
- **execute_js_in_browser**: Test your JavaScript code against the live website. Navigates to the URL and executes your IIFE, returning the result and any console output. Use this to verify your code works before submitting.

## Guidelines

- Use `validate_js_code` before `submit_js_code` to catch errors early
- Keep code concise and focused on the specific task
- Use DOM APIs (querySelector, getElementById, etc.) for element interaction
- Use sessionStorage for passing data between operations
"""

    AUTONOMOUS_SYSTEM_PROMPT: str = """You are a JavaScript expert that autonomously writes browser DOM manipulation code.

## Your Mission

Given a task, write IIFE JavaScript code that accomplishes it in the browser context.

## Process

1. **Understand**: Analyze the task and determine what DOM manipulation is needed
2. **Research**: If JS files are available, search them for relevant APIs or data structures
3. **Check DOM**: Use `get_dom_snapshot` to understand the current page structure
4. **Write**: Write the JavaScript code, validate it, then submit
5. **Finalize**: Call `submit_js_code` with your validated code

## Code Requirements

- IIFE format: `(function() { ... })()` or `(() => { ... })()`
- Blocked: eval, fetch, XMLHttpRequest, WebSocket, sendBeacon, addEventListener, MutationObserver, IntersectionObserver, window.close
- Use `return` to produce output; optionally use `session_storage_key` for cross-operation data

## Code Formatting

- Write readable, well-formatted JavaScript. Never write extremely long single-line IIFEs.
- Use proper indentation (2 spaces), line breaks between statements, and descriptive variable names.
- Each statement should be on its own line. Complex expressions should be broken across lines.

## When finalize tools are available

- **submit_js_code**: Submit your final validated JavaScript code
- **finalize_failure**: Report that the task cannot be accomplished with JS
- **execute_js_in_browser**: Test your JavaScript code against the live website before submitting
"""

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        js_data_store: JSDataStore | None = None,
        dom_snapshots: list[DOMSnapshotEvent] | None = None,
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str | None = None,
    ) -> None:
        self._js_data_store = js_data_store
        self._dom_snapshots = dom_snapshots or []
        self._remote_debugging_address = remote_debugging_address

        # autonomous result state
        self._js_result: JSCodeResult | None = None
        self._js_failure: JSCodeFailureResult | None = None

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
        )

        logger.debug(
            "JSSpecialist initialized: js_data_store=%s, dom_snapshots=%d, browser=%s",
            "yes" if js_data_store else "no",
            len(self._dom_snapshots),
            "yes" if remote_debugging_address else "no",
        )

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        context_parts = [self.SYSTEM_PROMPT]

        if self._js_data_store:
            stats = self._js_data_store.stats
            context_parts.append(
                f"\n\n## JS Files Context\n"
                f"- Total JS files: {stats.total_files}\n"
                f"- Unique URLs: {stats.unique_urls}\n"
            )

        if self._dom_snapshots:
            latest = self._dom_snapshots[-1]
            context_parts.append(
                f"\n\n## DOM Context\n"
                f"- {len(self._dom_snapshots)} snapshot(s) available\n"
                f"- Latest page: {latest.url}\n"
                f"- Latest title: {latest.title or 'N/A'}\n"
            )

        return "".join(context_parts)

    def _get_autonomous_system_prompt(self) -> str:
        context_parts = [self.AUTONOMOUS_SYSTEM_PROMPT]

        if self._js_data_store:
            stats = self._js_data_store.stats
            context_parts.append(
                f"\n\n## JS Files Context\n"
                f"- Total JS files: {stats.total_files}\n"
            )

        if self._dom_snapshots:
            latest = self._dom_snapshots[-1]
            context_parts.append(
                f"\n\n## DOM Context\n"
                f"- {len(self._dom_snapshots)} snapshot(s) available\n"
                f"- Latest page: {latest.url}\n"
            )

        # Urgency notices
        if self._finalize_tools_registered:
            remaining = 10 - self._autonomous_iteration
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

    def _register_tools(self) -> None:
        # validate_js_code
        self.llm_client.register_tool(
            name="validate_js_code",
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

        # search_js_files (only if data store available)
        if self._js_data_store:
            self.llm_client.register_tool(
                name="search_js_files",
                description="Search JavaScript file response bodies by terms. Returns top results ranked by relevance.",
                parameters={
                    "type": "object",
                    "properties": {
                        "terms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search terms to look for in JS file contents.",
                        }
                    },
                    "required": ["terms"],
                },
            )

            # get_js_file_detail
            self.llm_client.register_tool(
                name="get_js_file_detail",
                description="Get full JavaScript file content by request_id.",
                parameters={
                    "type": "object",
                    "properties": {
                        "request_id": {
                            "type": "string",
                            "description": "The request_id of the JS file entry.",
                        }
                    },
                    "required": ["request_id"],
                },
            )

        # get_dom_snapshot (only if snapshots available)
        if self._dom_snapshots:
            self.llm_client.register_tool(
                name="get_dom_snapshot",
                description="Get a DOM snapshot. Returns the document structure and truncated strings table. Defaults to latest snapshot.",
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

        # execute_js_in_browser (only if browser available)
        if self._remote_debugging_address:
            self.llm_client.register_tool(
                name="execute_js_in_browser",
                description=(
                    "Test JavaScript code against the live website. "
                    "Navigates to the URL and executes your IIFE, returning the result and any console output. "
                    "Use this to verify your code works before submitting."
                ),
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
                    },
                    "required": ["url", "js_code"],
                },
            )

    def _register_finalize_tools(self) -> None:
        if self._finalize_tools_registered:
            return

        self.llm_client.register_tool(
            name="submit_js_code",
            description=(
                "Submit validated JavaScript code as the final result. "
                "The code must be IIFE-wrapped and pass all validation checks."
            ),
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

        self.llm_client.register_tool(
            name="finalize_failure",
            description="Report that the JavaScript task cannot be accomplished.",
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

        logger.debug("Registered JS finalize tools")

    def _execute_tool(self, tool_name: str, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        logger.debug("Executing tool %s", tool_name)

        if tool_name == "validate_js_code":
            return self._tool_validate_js_code(tool_arguments)
        if tool_name == "search_js_files":
            return self._tool_search_js_files(tool_arguments)
        if tool_name == "get_js_file_detail":
            return self._tool_get_js_file_detail(tool_arguments)
        if tool_name == "get_dom_snapshot":
            return self._tool_get_dom_snapshot(tool_arguments)
        if tool_name == "submit_js_code":
            return self._tool_submit_js_code(tool_arguments)
        if tool_name == "finalize_failure":
            return self._tool_finalize_failure(tool_arguments)
        if tool_name == "execute_js_in_browser":
            return self._tool_execute_js_in_browser(tool_arguments)

        return {"error": f"Unknown tool: {tool_name}"}

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

    ## Tool handlers

    @token_optimized
    def _tool_validate_js_code(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        js_code = tool_arguments.get("js_code", "")
        errors = validate_js(js_code)

        # Separate hard errors from warnings
        hard_errors = [e for e in errors if not e.startswith("WARNING:")]
        warnings = [e for e in errors if e.startswith("WARNING:")]

        if hard_errors:
            return {
                "valid": False,
                "errors": hard_errors,
                "warnings": warnings,
            }
        if warnings:
            return {
                "valid": True,
                "warnings": warnings,
                "message": "Code passes validation but has formatting warnings. Consider reformatting.",
            }
        return {
            "valid": True,
            "message": "Code passes all validation checks.",
        }

    @token_optimized
    def _tool_search_js_files(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._js_data_store:
            return {"error": "No JS data store available"}

        terms = tool_arguments.get("terms", [])
        if not terms:
            return {"error": "No search terms provided"}

        results = self._js_data_store.search_by_terms(terms, top_n=20)
        if not results:
            return {"message": "No matching JS files found", "terms_searched": len(terms)}

        return {
            "terms_searched": len(terms),
            "results_found": len(results),
            "results": results,
        }

    @token_optimized
    def _tool_get_js_file_detail(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._js_data_store:
            return {"error": "No JS data store available"}

        request_id = tool_arguments.get("request_id")
        if not request_id:
            return {"error": "request_id is required"}

        entry = self._js_data_store.get_file(request_id)
        if not entry:
            return {"error": f"Entry {request_id} not found"}

        content = self._js_data_store.get_file_content(request_id)

        return {
            "request_id": request_id,
            "url": entry.url,
            "content": content,
        }

    @token_optimized
    def _tool_get_dom_snapshot(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        if not self._dom_snapshots:
            return {"error": "No DOM snapshots available"}

        index = tool_arguments.get("index", -1)
        if index < 0:
            index = len(self._dom_snapshots) + index
        if index < 0 or index >= len(self._dom_snapshots):
            return {"error": f"Snapshot index {index} out of range (0-{len(self._dom_snapshots) - 1})"}

        snapshot = self._dom_snapshots[index]

        # Truncate strings table to keep token usage reasonable
        strings = snapshot.strings
        truncated_strings = strings[:500] if len(strings) > 500 else strings

        return {
            "url": snapshot.url,
            "title": snapshot.title,
            "strings_count": len(strings),
            "strings_sample": truncated_strings,
            "documents_count": len(snapshot.documents),
            "documents": snapshot.documents,
        }

    @token_optimized
    def _tool_submit_js_code(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        js_code = tool_arguments.get("js_code", "")
        description = tool_arguments.get("description", "")
        session_storage_key = tool_arguments.get("session_storage_key")
        timeout_seconds = tool_arguments.get("timeout_seconds", 5.0)

        if not description:
            return {"error": "description is required"}

        # Validate the code
        errors = validate_js(js_code)
        # Only block on hard errors, not warnings
        hard_errors = [e for e in errors if not e.startswith("WARNING:")]
        if hard_errors:
            return {"error": "Validation failed", "errors": hard_errors}

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

    @token_optimized
    def _tool_finalize_failure(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
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

    @token_optimized
    def _tool_execute_js_in_browser(self, tool_arguments: dict[str, Any]) -> dict[str, Any]:
        """Navigate to a URL and execute JS code via CDP, returning the result."""
        if not self._remote_debugging_address:
            return {"error": "No browser connection configured"}

        url = tool_arguments.get("url", "")
        js_code = tool_arguments.get("js_code", "")
        timeout_seconds = tool_arguments.get("timeout_seconds", 5.0)

        # Validate JS first
        errors = validate_js(js_code)
        hard_errors = [e for e in errors if not e.startswith("WARNING:")]
        if hard_errors:
            return {"error": "Validation failed", "errors": hard_errors}

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
            # Cleanup: close target and dispose context
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
