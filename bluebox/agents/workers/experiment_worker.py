"""
bluebox/agents/workers/experiment_worker.py

Worker agent for Phase 2: Experimentation.

The ExperimentWorker executes experiments in a live browser while referencing
captured session data. It has two categories of tools:

- **browser_*** tools — interact with the LIVE browser (navigate, eval JS, raw CDP, DOM)
- **capture_*** tools — look up RECORDED data from a previous capture session

The worker OWNS its browser tab lifecycle — it lazily creates a persistent
incognito tab on the first browser tool call and keeps it alive across all
subsequent tool calls. Call close() to clean up.

The worker does NOT construct routines or decide strategy. It executes experiments
dispatched by an orchestrator and reports structured findings via finalize tools.
"""

from __future__ import annotations

import json
import textwrap
import time
from typing import TYPE_CHECKING, Any, Callable

from websocket import WebSocket

from bluebox.agents.abstract_agent import AgentCard, agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.cdp.connection import (
    cdp_new_tab,
    create_cdp_helpers,
    dispose_context,
)
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.code_execution_sandbox import execute_python_sandboxed
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger

if TYPE_CHECKING:
    from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader

logger = get_logger(name=__name__)


# ---------------------------------------------------------------------------
# DOM filter JS — executed via Runtime.evaluate to get a filtered DOM view
# ---------------------------------------------------------------------------

_DOM_FILTER_JS = """
(function(selector, maxDepth, includeTags) {
  function walk(node, depth) {
    if (depth > maxDepth) return null;
    var tag = node.nodeName.toLowerCase();
    if (includeTags && includeTags.length > 0 && !includeTags.includes(tag) && tag !== '#document' && tag !== 'html' && tag !== 'head' && tag !== 'body') {
      var kids = [];
      for (var i = 0; i < node.childNodes.length; i++) {
        var c = walk(node.childNodes[i], depth);
        if (c) kids.push(c);
      }
      return kids.length === 1 ? kids[0] : kids.length > 1 ? kids : null;
    }
    if (node.nodeType === 3) {
      var text = node.textContent.trim();
      return text ? text.substring(0, 200) : null;
    }
    if (node.nodeType !== 1) return null;
    var obj = { tag: tag };
    var attrs = {};
    for (var j = 0; j < node.attributes.length; j++) {
      var a = node.attributes[j];
      if (['id', 'class', 'name', 'type', 'href', 'src', 'action', 'method', 'value', 'placeholder', 'role', 'aria-label', 'data-testid'].includes(a.name)) {
        attrs[a.name] = a.value.substring(0, 200);
      }
    }
    if (Object.keys(attrs).length) obj.attrs = attrs;
    var children = [];
    for (var k = 0; k < node.childNodes.length; k++) {
      var child = walk(node.childNodes[k], depth + 1);
      if (child) {
        if (Array.isArray(child)) children = children.concat(child);
        else children.push(child);
      }
    }
    if (children.length) obj.children = children;
    return obj;
  }
  var root = selector ? document.querySelector(selector) : document.documentElement;
  if (!root) return JSON.stringify({ error: 'selector not found: ' + selector });
  var result = JSON.stringify(walk(root, 0));
  if (result.length > 15000) {
    return result.substring(0, 15000) + '... [TRUNCATED at 15K chars]';
  }
  return result;
})(%SELECTOR%, %MAX_DEPTH%, %INCLUDE_TAGS%)
""".strip()


class ExperimentWorker(AbstractSpecialist):
    """
    Worker agent that executes experiments in a live browser while referencing
    captured session data.

    Two sources of truth:
    - **capture_*** tools: recorded/stale reference data from a previous session
    - **browser_*** tools: live/current reality in the browser

    The worker OWNS its browser tab. It lazily creates a persistent incognito
    tab on the first browser tool call and reuses it for all subsequent calls.
    Call close() to tear down the tab when done.
    """

    AGENT_CARD = AgentCard(
        description=(
            "Executes experiments in a live browser while referencing captured session data. "
            "Has browser tools (navigate, eval JS, CDP, DOM) and capture lookup tools."
        ),
    )

    SYSTEM_PROMPT: str = textwrap.dedent("""\
        You are an experiment worker agent with access to TWO sources of data:

        ## Two Sources of Truth

        1. **capture_* tools** — Recorded data from a PREVIOUS browser session.
           This is stale/historical reference data. Use it to understand what the
           website looked like, what requests were made, what tokens were used.
           Think of it as "the recording."

        2. **browser_* tools** — The LIVE browser tab you control right now.
           This is current reality. Use it to navigate pages, execute JavaScript,
           read current DOM state, and test hypotheses.

        ## Your Role

        You execute experiments dispatched by an orchestrator. Each experiment has
        a specific hypothesis to test and expected output. Your job is to:
        1. Look up relevant reference data from the capture (if needed)
        2. Execute the experiment in the live browser
        3. Report your findings via finalize tools

        You do NOT decide strategy. You do NOT construct routines. You execute
        experiments and report what you find.

        ## Guidelines

        - Always check capture data first to understand context before acting in the browser
        - Use browser_eval_js for most browser interactions — it covers fetch, DOM reads,
          clicks, typing, storage access, and more
        - browser_get_dom is useful for understanding page structure before writing JS
        - Keep browser_eval_js expressions focused and concise
        - Report exact values, not approximations
    """)

    AUTONOMOUS_SYSTEM_PROMPT: str = textwrap.dedent("""\
        You are an autonomous experiment worker. Execute the given experiment,
        gather findings, and finalize with structured output.

        ## Two Sources of Truth

        1. **capture_* tools** — Recorded/stale reference data from a previous session.
        2. **browser_* tools** — Live browser tab you control right now.

        ## Process

        1. Read the experiment task carefully
        2. Look up relevant capture data for context
        3. Execute the experiment in the live browser
        4. Collect results and call finalize_with_output (or finalize_with_failure if blocked)

        ## Guidelines

        - Do NOT navigate away from the current page unless the experiment requires it
        - Use browser_eval_js as your Swiss army knife for all browser interactions
        - Report exact values and observations, not guesses
    """)

    # -----------------------------------------------------------------------
    # Constructor
    # -----------------------------------------------------------------------

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        # Browser — worker owns its own tab lifecycle
        remote_debugging_address: str | None = None,
        # Capture data loaders — each optional, gating respective capture tools
        network_data_loader: NetworkDataLoader | None = None,
        storage_data_loader: StorageDataLoader | None = None,
        dom_data_loader: DOMDataLoader | None = None,
        window_property_data_loader: WindowPropertyDataLoader | None = None,
        # Standard specialist args
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.AUTONOMOUS,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        documentation_data_loader: DocumentationDataLoader | None = None,
    ) -> None:
        # Browser connection — lazy init on first browser tool call
        self._remote_debugging_address = remote_debugging_address
        self._browser_ws: WebSocket | None = None
        self._browser_target_id: str | None = None
        self._browser_context_id: str | None = None
        self._browser_session_id: str | None = None
        self._browser_send_cmd: Callable[..., int] | None = None
        self._browser_recv_until: Callable[..., dict[str, Any]] | None = None

        # Capture data loaders
        self._network_data_loader = network_data_loader
        self._storage_data_loader = storage_data_loader
        self._dom_data_loader = dom_data_loader
        self._window_property_data_loader = window_property_data_loader

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
            "ExperimentWorker initialized: browser=%s, network=%s, storage=%s, dom=%s, window_props=%s",
            "yes" if self._remote_debugging_address else "no",
            "yes" if self._network_data_loader else "no",
            "yes" if self._storage_data_loader else "no",
            "yes" if self._dom_data_loader else "no",
            "yes" if self._window_property_data_loader else "no",
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def _has_browser(self) -> bool:
        """True when a remote debugging address is configured (tab will be created on first use)."""
        return self._remote_debugging_address is not None

    @property
    def _browser_connected(self) -> bool:
        """True when the persistent browser tab is already connected."""
        return self._browser_session_id is not None and self._browser_ws is not None

    # -----------------------------------------------------------------------
    # Browser tab lifecycle
    # -----------------------------------------------------------------------

    def _ensure_browser(self) -> None:
        """
        Lazily create a persistent incognito browser tab.

        Called automatically by browser tools on first use. Subsequent calls
        are a no-op if the tab is already connected.

        Raises:
            RuntimeError: If no remote_debugging_address is configured or
                          if tab creation fails.
        """
        if self._browser_connected:
            return

        if not self._remote_debugging_address:
            raise RuntimeError("No remote_debugging_address configured — cannot create browser tab")

        logger.info("Creating persistent browser tab at %s", self._remote_debugging_address)

        # Create incognito tab
        target_id, browser_context_id, browser_ws = cdp_new_tab(
            self._remote_debugging_address,
            incognito=True,
            url="about:blank",
        )

        send_cmd, _, recv_until = create_cdp_helpers(browser_ws)

        # Attach to target with flattened session
        deadline = time.time() + 10.0
        attach_id = send_cmd(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
        )
        attach_reply = recv_until(lambda m: m.get("id") == attach_id, deadline)
        if "error" in attach_reply:
            browser_ws.close()
            raise RuntimeError(f"Failed to attach to browser tab: {attach_reply['error']}")

        session_id = attach_reply["result"]["sessionId"]

        # Enable Page and Runtime domains
        page_id = send_cmd("Page.enable", session_id=session_id)
        recv_until(lambda m: m.get("id") == page_id, deadline)
        runtime_id = send_cmd("Runtime.enable", session_id=session_id)
        recv_until(lambda m: m.get("id") == runtime_id, deadline)

        # Store persistent connection state
        self._browser_ws = browser_ws
        self._browser_target_id = target_id
        self._browser_context_id = browser_context_id
        self._browser_session_id = session_id
        self._browser_send_cmd = send_cmd
        self._browser_recv_until = recv_until

        logger.info(
            "Browser tab ready: target_id=%s, session_id=%s",
            target_id, session_id,
        )

    def close(self) -> None:
        """
        Tear down the persistent browser tab and clean up resources.

        Safe to call multiple times. Safe to call even if no tab was created.
        """
        if self._browser_ws is not None:
            try:
                if self._browser_target_id and self._browser_send_cmd:
                    self._browser_send_cmd(
                        "Target.closeTarget",
                        {"targetId": self._browser_target_id},
                    )
            except Exception:
                pass
            try:
                self._browser_ws.close()
            except Exception:
                pass

        if self._browser_context_id and self._remote_debugging_address:
            try:
                dispose_context(
                    browser_context_id=self._browser_context_id,
                    remote_debugging_address=self._remote_debugging_address,
                )
            except Exception:
                pass

        # Reset state
        self._browser_ws = None
        self._browser_target_id = None
        self._browser_context_id = None
        self._browser_session_id = None
        self._browser_send_cmd = None
        self._browser_recv_until = None

        logger.info("Browser tab closed")

    # -----------------------------------------------------------------------
    # Abstract method implementations
    # -----------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        parts = [self.SYSTEM_PROMPT]
        parts.append(self._get_data_context_section())
        return "".join(parts)

    def _get_autonomous_system_prompt(self) -> str:
        parts = [self.AUTONOMOUS_SYSTEM_PROMPT]
        parts.append(self._get_data_context_section())
        parts.append(self._get_output_schema_prompt_section())
        parts.append(self._get_urgency_notice())
        return "".join(parts)

    def _get_autonomous_initial_message(self, task: str) -> str:
        finalize_success = "finalize_with_output" if self.has_output_schema else "finalize_result"
        return (
            f"EXPERIMENT: {task}\n\n"
            f"Execute this experiment. Use capture_* tools for reference data and "
            f"browser_* tools for live interaction. When done, call {finalize_success} "
            f"with your findings."
        )

    def _get_data_context_section(self) -> str:
        """Build a context section summarizing available data sources."""
        parts: list[str] = ["\n\n## Available Data Sources\n"]

        # Browser
        if self._has_browser:
            if self._browser_connected:
                parts.append("- **Browser**: Connected (persistent tab active)\n")
            else:
                parts.append("- **Browser**: Available (tab will be created on first use)\n")
        else:
            parts.append("- **Browser**: Not configured (browser_* tools unavailable)\n")

        # Network capture
        if self._network_data_loader:
            stats = self._network_data_loader.stats
            parts.append(
                f"- **Network capture**: {stats.total_requests} requests, "
                f"{stats.unique_urls} unique URLs\n"
            )
        else:
            parts.append("- **Network capture**: Not available\n")

        # Storage capture
        if self._storage_data_loader:
            stats = self._storage_data_loader.stats
            parts.append(
                f"- **Storage capture**: {stats.total_events} events "
                f"(cookies: {stats.cookie_events}, localStorage: {stats.local_storage_events})\n"
            )
        else:
            parts.append("- **Storage capture**: Not available\n")

        # DOM capture
        if self._dom_data_loader:
            stats = self._dom_data_loader.stats
            parts.append(
                f"- **DOM capture**: {stats.total_snapshots} snapshots, "
                f"{stats.unique_urls} unique URLs\n"
            )
        else:
            parts.append("- **DOM capture**: Not available\n")

        # Window properties
        if self._window_property_data_loader:
            stats = self._window_property_data_loader.stats
            parts.append(
                f"- **Window properties**: {stats.total_events} events, "
                f"{stats.unique_property_paths} unique paths\n"
            )
        else:
            parts.append("- **Window properties**: Not available\n")

        return "".join(parts)

    # ===================================================================
    # BROWSER TOOLS — gated by _has_browser (remote_debugging_address set)
    # ===================================================================

    @agent_tool(availability=lambda self: self._has_browser)
    @token_optimized
    def _browser_navigate(self, url: str, timeout_seconds: float = 15.0) -> dict[str, Any]:
        """
        Navigate the browser tab to a URL and wait for page load.

        Args:
            url: The URL to navigate to.
            timeout_seconds: Max time to wait for page load (default 15s).
        """
        try:
            self._ensure_browser()
        except RuntimeError as e:
            return {"error": str(e)}

        assert self._browser_send_cmd is not None
        assert self._browser_recv_until is not None

        deadline = time.time() + timeout_seconds
        try:
            nav_id = self._browser_send_cmd(
                "Page.navigate",
                {"url": url},
                session_id=self._browser_session_id,
            )
            nav_reply = self._browser_recv_until(lambda m: m.get("id") == nav_id, deadline)

            if "error" in nav_reply:
                return {"error": f"Navigation failed: {nav_reply['error']}"}

            # Wait for load event
            self._browser_recv_until(
                lambda m: m.get("method") == "Page.loadEventFired",
                deadline,
            )

            return {"status": "ok", "url": url}

        except TimeoutError:
            return {"error": f"Navigation to {url} timed out after {timeout_seconds}s"}
        except Exception as e:
            logger.error("browser_navigate failed: %s", e)
            return {"error": f"Navigation failed: {e}"}

    @agent_tool(availability=lambda self: self._has_browser)
    @token_optimized
    def _browser_eval_js(
        self,
        expression: str,
        timeout_seconds: float = 30.0,
    ) -> dict[str, Any]:
        """
        Execute JavaScript in the browser page context and return the result.

        This is your Swiss army knife — use it for:
        - Reading DOM state (document.querySelector, document.cookie, etc.)
        - Calling fetch() to test API endpoints
        - Reading/writing localStorage, sessionStorage
        - Clicking elements, filling inputs, submitting forms
        - Any browser-side computation

        The expression should return a value (use an IIFE if needed).
        Results are truncated at 10K characters.

        Args:
            expression: JavaScript expression to evaluate. Use an IIFE for multi-statement code.
            timeout_seconds: Max execution time (default 30s).
        """
        try:
            self._ensure_browser()
        except RuntimeError as e:
            return {"error": str(e)}

        assert self._browser_send_cmd is not None
        assert self._browser_recv_until is not None

        deadline = time.time() + timeout_seconds
        try:
            eval_id = self._browser_send_cmd(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": True,
                    "awaitPromise": True,
                },
                session_id=self._browser_session_id,
            )
            eval_reply = self._browser_recv_until(lambda m: m.get("id") == eval_id, deadline)

            if "error" in eval_reply:
                return {"error": f"Runtime.evaluate error: {eval_reply['error']}"}

            result_obj = eval_reply.get("result", {}).get("result", {})

            # Check for exception
            exception_details = eval_reply.get("result", {}).get("exceptionDetails")
            if exception_details:
                return {
                    "error": "JavaScript exception",
                    "exception": exception_details.get("text", str(exception_details)),
                }

            value = result_obj.get("value")

            # Truncate large results
            result_str = json.dumps(value) if not isinstance(value, str) else value
            if len(result_str) > 10_000:
                result_str = result_str[:10_000] + "... [TRUNCATED at 10K chars]"
                return {"result": result_str, "truncated": True}

            return {"result": value}

        except TimeoutError:
            return {"error": f"JS execution timed out after {timeout_seconds}s"}
        except Exception as e:
            logger.error("browser_eval_js failed: %s", e)
            return {"error": f"JS execution failed: {e}"}

    @agent_tool(availability=lambda self: self._has_browser)
    @token_optimized
    def _browser_cdp_command(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """
        Send a raw CDP command to the browser. Escape hatch for anything not
        covered by the other browser tools.

        Args:
            method: CDP method name (e.g. "Network.getCookies", "DOM.getDocument").
            params: Optional parameters dict for the CDP method.
            timeout_seconds: Max time to wait for response (default 10s).
        """
        try:
            self._ensure_browser()
        except RuntimeError as e:
            return {"error": str(e)}

        assert self._browser_send_cmd is not None
        assert self._browser_recv_until is not None

        deadline = time.time() + timeout_seconds
        try:
            cmd_id = self._browser_send_cmd(
                method,
                params or {},
                session_id=self._browser_session_id,
            )
            reply = self._browser_recv_until(lambda m: m.get("id") == cmd_id, deadline)

            if "error" in reply:
                return {"error": f"CDP error: {reply['error']}"}

            result = reply.get("result", {})

            # Truncate large results
            result_str = json.dumps(result)
            if len(result_str) > 10_000:
                return {
                    "result": json.loads(result_str[:10_000] + "}"),
                    "truncated": True,
                    "note": "Result truncated at 10K chars",
                }

            return {"result": result}

        except TimeoutError:
            return {"error": f"CDP command {method} timed out after {timeout_seconds}s"}
        except Exception as e:
            logger.error("browser_cdp_command failed: %s", e)
            return {"error": f"CDP command failed: {e}"}

    @agent_tool(availability=lambda self: self._has_browser)
    @token_optimized
    def _browser_get_dom(
        self,
        selector: str | None = None,
        max_depth: int = 5,
        include_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Get a filtered view of the current page DOM.

        Returns a JSON tree of DOM nodes with key attributes (id, class, name,
        type, href, etc.). Useful for understanding page structure before
        writing JS code.

        Args:
            selector: CSS selector to scope the view (default: entire document).
            max_depth: Maximum DOM tree depth to walk (default 6).
            include_tags: Only include these tag names (default: all tags).
                          Common values: ["form", "input", "button", "a", "select", "table"].
        """
        try:
            self._ensure_browser()
        except RuntimeError as e:
            return {"error": str(e)}

        assert self._browser_send_cmd is not None
        assert self._browser_recv_until is not None

        # Build the JS expression with parameters substituted
        selector_js = json.dumps(selector) if selector else "null"
        tags_js = json.dumps(include_tags) if include_tags else "null"

        js = _DOM_FILTER_JS.replace("%SELECTOR%", selector_js)
        js = js.replace("%MAX_DEPTH%", str(max_depth))
        js = js.replace("%INCLUDE_TAGS%", tags_js)

        deadline = time.time() + 10.0
        try:
            eval_id = self._browser_send_cmd(
                "Runtime.evaluate",
                {
                    "expression": js,
                    "returnByValue": True,
                    "awaitPromise": False,
                },
                session_id=self._browser_session_id,
            )
            eval_reply = self._browser_recv_until(lambda m: m.get("id") == eval_id, deadline)

            if "error" in eval_reply:
                return {"error": f"DOM query failed: {eval_reply['error']}"}

            result_obj = eval_reply.get("result", {}).get("result", {})
            raw_value = result_obj.get("value", "")

            # Parse JSON result
            try:
                dom_tree = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
            except json.JSONDecodeError:
                dom_tree = raw_value

            return {"dom": dom_tree, "selector": selector, "max_depth": max_depth}

        except TimeoutError:
            return {"error": "DOM query timed out"}
        except Exception as e:
            logger.error("browser_get_dom failed: %s", e)
            return {"error": f"DOM query failed: {e}"}

    # ===================================================================
    # CAPTURE LOOKUP TOOLS — gated by respective data loaders
    # ===================================================================

    @agent_tool(availability=lambda self: self._network_data_loader is not None)
    @token_optimized
    def _capture_search_transactions(
        self,
        query: str,
        top_n: int = 15,
    ) -> dict[str, Any]:
        """
        Search recorded network traffic by keyword terms.

        Searches across URLs, request/response headers, and response bodies.
        Returns ranked results with request IDs for drill-down.

        Args:
            query: Space-separated search terms (e.g. "search api flights").
            top_n: Max results to return (default 15).
        """
        assert self._network_data_loader is not None

        terms = query.split()
        if not terms:
            return {"error": "query is required"}

        results = self._network_data_loader.search_entries_by_terms(
            terms=terms, top_n=top_n,
        )
        return {
            "query": query,
            "count": len(results),
            "results": results,
        }

    @agent_tool(availability=lambda self: self._network_data_loader is not None)
    @token_optimized
    def _capture_get_transaction(self, request_id: str) -> dict[str, Any]:
        """
        Get the full recorded request/response for a specific transaction.

        Returns URL, method, status, request headers, request body,
        response headers, and response body.

        Args:
            request_id: The request ID from search results or exploration.
        """
        assert self._network_data_loader is not None

        entry = self._network_data_loader.get_entry(request_id)
        if entry is None:
            return {"error": f"No transaction found for request_id: {request_id}"}

        return {
            "request_id": entry.request_id,
            "url": entry.url,
            "method": entry.method,
            "status_code": entry.status_code,
            "request_headers": entry.request_headers,
            "request_body": entry.request_body[:5000] if entry.request_body else None,
            "response_headers": entry.response_headers,
            "response_body": entry.response_body[:5000] if entry.response_body else None,
            "response_body_truncated": bool(entry.response_body and len(entry.response_body) > 5000),
        }

    @agent_tool(availability=lambda self: self._storage_data_loader is not None)
    @token_optimized
    def _capture_search_storage(
        self,
        query: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search recorded browser storage events (cookies, localStorage, sessionStorage).

        Searches across storage keys and values.

        Args:
            query: Value to search for in storage.
            case_sensitive: Whether the search should be case-sensitive (default false).
        """
        assert self._storage_data_loader is not None

        results = self._storage_data_loader.search_values(
            value=query, case_sensitive=case_sensitive,
        )
        return {
            "query": query,
            "count": len(results),
            "results": results[:20],
        }

    @agent_tool()
    @token_optimized
    def _capture_trace_value(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search for a value across ALL recorded data sources (network, storage,
        window properties, DOM). Useful for tracing where a token or value
        originated from in the captured session.

        Args:
            value: The value to trace (token, ID, key, etc.).
            case_sensitive: Whether the search should be case-sensitive (default false).
        """
        if not value:
            return {"error": "value is required"}

        results: dict[str, Any] = {"value_searched": value}
        found_any = False

        # Network responses
        if self._network_data_loader:
            network_hits = self._network_data_loader.search_response_bodies(
                value=value, case_sensitive=case_sensitive,
            )
            results["network"] = {
                "found": len(network_hits) > 0,
                "count": len(network_hits),
                "matches": network_hits[:10],
            }
            if network_hits:
                found_any = True
        else:
            results["network"] = {"available": False}

        # Storage
        if self._storage_data_loader:
            storage_hits = self._storage_data_loader.search_values(
                value=value, case_sensitive=case_sensitive,
            )
            results["storage"] = {
                "found": len(storage_hits) > 0,
                "count": len(storage_hits),
                "matches": storage_hits[:10],
            }
            if storage_hits:
                found_any = True
        else:
            results["storage"] = {"available": False}

        # Window properties
        if self._window_property_data_loader:
            window_hits = self._window_property_data_loader.search_values(
                value=value, case_sensitive=case_sensitive,
            )
            results["window_properties"] = {
                "found": len(window_hits) > 0,
                "count": len(window_hits),
                "matches": window_hits[:10],
            }
            if window_hits:
                found_any = True
        else:
            results["window_properties"] = {"available": False}

        # DOM strings
        if self._dom_data_loader:
            dom_hits = self._dom_data_loader.search_strings(
                value=value, case_sensitive=case_sensitive,
            )
            results["dom"] = {
                "found": len(dom_hits) > 0,
                "count": len(dom_hits),
                "matches": dom_hits[:10],
            }
            if dom_hits:
                found_any = True
        else:
            results["dom"] = {"available": False}

        if not found_any and not any(
            r.get("available", True) for r in [
                results.get("network", {}),
                results.get("storage", {}),
                results.get("window_properties", {}),
                results.get("dom", {}),
            ]
        ):
            results["warning"] = "No data sources available for tracing"

        return results

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _capture_get_page_structure(
        self,
        snapshot_index: int = -1,
    ) -> dict[str, Any]:
        """
        Get forms, inputs, meta tags, and scripts from a recorded DOM snapshot.

        Provides a structural overview of what was on the page at capture time.

        Args:
            snapshot_index: Which snapshot to examine (default -1 = latest).
        """
        assert self._dom_data_loader is not None

        idx = snapshot_index if snapshot_index >= 0 else None

        forms = self._dom_data_loader.get_forms(snapshot_index=idx)
        inputs = self._dom_data_loader.get_inputs(snapshot_index=idx)
        meta_tags = self._dom_data_loader.get_meta_tags(snapshot_index=idx)
        scripts = self._dom_data_loader.get_scripts(snapshot_index=idx)

        pages = self._dom_data_loader.list_pages()
        page_info = None
        if pages:
            actual_idx = snapshot_index if snapshot_index >= 0 else len(pages) - 1
            if 0 <= actual_idx < len(pages):
                page_info = pages[actual_idx]

        return {
            "snapshot_index": snapshot_index,
            "page": page_info,
            "forms": forms[:10],
            "inputs": inputs[:30],
            "meta_tags": meta_tags[:20],
            "scripts": scripts[:10],
        }

    @agent_tool(availability=lambda self: self._dom_data_loader is not None)
    @token_optimized
    def _capture_get_element(
        self,
        element_type: str,
        snapshot_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Get specific elements from a recorded DOM snapshot.

        Args:
            element_type: Type of element to retrieve. One of:
                "forms", "inputs", "tables", "meta_tags", "scripts",
                "buttons", "links", "hidden_inputs".
            snapshot_index: Which snapshot to examine (default: all snapshots).
        """
        assert self._dom_data_loader is not None

        type_to_method: dict[str, Callable[..., list[dict[str, Any]]]] = {
            "forms": self._dom_data_loader.get_forms,
            "inputs": self._dom_data_loader.get_inputs,
            "tables": self._dom_data_loader.get_tables,
            "meta_tags": self._dom_data_loader.get_meta_tags,
            "scripts": self._dom_data_loader.get_scripts,
        }

        method = type_to_method.get(element_type)
        if method is None:
            # Try get_elements for other types
            try:
                results = self._dom_data_loader.get_elements(
                    element_type=element_type,
                    snapshot_index=snapshot_index,
                )
                return {
                    "element_type": element_type,
                    "count": len(results),
                    "elements": results[:30],
                }
            except Exception as e:
                return {
                    "error": f"Unknown element_type: {element_type}. "
                    f"Valid types: {list(type_to_method.keys())}. Error: {e}",
                }

        results = method(snapshot_index=snapshot_index)
        return {
            "element_type": element_type,
            "count": len(results),
            "elements": results[:30],
        }

    # ===================================================================
    # PYTHON SANDBOX — quick data analysis across all capture sources
    # ===================================================================

    @agent_tool()
    def _execute_python(self, code: str) -> dict[str, Any]:
        """
        Execute Python code in a sandboxed environment to analyze captured data.

        Pre-loaded variables (lists of dicts, from recorded capture data):
        - `network_entries`: Network transactions (request_id, url, method,
          status_code, request_headers, response_headers, request_body, response_body)
        - `storage_entries`: Storage events (type, origin, key, value, etc.)
        - `window_prop_entries`: Window property events (url, timestamp, changes)

        The `json` module is available. Use print() to output results.
        Imports are disabled for security.

        Examples:
            # Find all POST requests
            for e in network_entries:
                if e['method'] == 'POST':
                    print(e['url'])

            # Find all cookies with 'token' in the key
            for e in storage_entries:
                if 'token' in (e.get('key') or '').lower():
                    print(e['key'], e.get('value', '')[:100])

            # Count requests per host
            from collections import Counter  # not allowed — use dict instead
            hosts = {}
            for e in network_entries:
                host = e['url'].split('/')[2] if '/' in e['url'] else e['url']
                hosts[host] = hosts.get(host, 0) + 1
            for h, c in sorted(hosts.items(), key=lambda x: -x[1]):
                print(f"{h}: {c}")

        Args:
            code: Python code to execute. Use print() for output.
        """
        extra_globals: dict[str, Any] = {}

        if self._network_data_loader:
            extra_globals["network_entries"] = [
                e.model_dump() for e in self._network_data_loader.entries
            ]
        else:
            extra_globals["network_entries"] = []

        if self._storage_data_loader:
            extra_globals["storage_entries"] = [
                e.model_dump() for e in self._storage_data_loader.entries
            ]
        else:
            extra_globals["storage_entries"] = []

        if self._window_property_data_loader:
            extra_globals["window_prop_entries"] = [
                e.model_dump() for e in self._window_property_data_loader.entries
            ]
        else:
            extra_globals["window_prop_entries"] = []

        return execute_python_sandboxed(code, extra_globals=extra_globals)
