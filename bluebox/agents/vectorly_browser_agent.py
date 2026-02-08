"""
bluebox/agents/vectorly_browser_agent.py

Agent specialized in browsing the web using Vectorly routines.

Contains:
- VectorlyBrowserAgent: Specialist for executing browser automation via Vectorly
- Uses: AbstractSpecialist base class for all agent plumbing
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from textwrap import dedent
from typing import Any, Callable
from urllib.parse import urlparse

import requests
import websocket
from websocket import WebSocket

from bluebox.agents.abstract_agent import agent_tool
from bluebox.agents.specialists.abstract_specialist import AbstractSpecialist, RunMode
from bluebox.cdp.connection import cdp_new_tab
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    Chat,
    ChatResponseEmittedMessage,
    ChatThread,
    EmittedMessage,
)
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.routine.routine import Routine
from bluebox.utils.js_utils import (
    generate_click_js,
    generate_type_js,
    generate_scroll_element_js,
    generate_scroll_window_js,
    generate_wait_for_url_js,
    generate_get_html_js,
)
from bluebox.utils.llm_utils import token_optimized
from bluebox.utils.logger import get_logger
from bluebox.utils.web_socket_utils import send_cmd, recv_until

logger = get_logger(name=__name__)


class VectorlyBrowserAgent(AbstractSpecialist):
    """
    Vectorly browser agent that helps execute web automation routines.

    The agent uses AbstractSpecialist as its base and provides tools to list,
    inspect, and execute Vectorly routines for browser automation.
    """

    AGENT_LOOP_MAX_ITERATIONS: int = 100

    SYSTEM_PROMPT: str = dedent("""
        You are a browser automation agent with direct control over a Chrome browser tab.

        ## Fundamental Tools (Most Powerful)
        - `evaluate_js(js: str)` — Execute raw JavaScript. Use for: extracting data, DOM manipulation, complex interactions. No wrapping - runs exactly what you provide.
        - `run_cdp_command(method: str, params: dict = {})` — Run raw CDP commands. Use for: screenshots, network interception, performance metrics, DOM inspection.

        ## High-Level Browser Tools
        - `navigate(url: str)` — Go to a URL.
        - `click(selector: str)` — Click element by CSS selector. Scrolls into view, clicks center.
        - `type_text(selector: str, text: str, clear: bool = False)` — Type into input field.
        - `press_key(key: str)` — Press keyboard key (enter, tab, escape, arrows, backspace).
        - `get_page_html(selector: str | None = None)` — Get page or element HTML. Use to understand page state.
        - `wait_for_url(url_regex: str, timeout_seconds: float = 20)` — Wait for URL to match regex.
        - `scroll(delta_x: int = 0, delta_y: int = 0, selector: str | None = None)` — Scroll page or element.
        - `get_cookies(domain_filter: str = "*")` — Get cookies, optionally filtered by domain.

        ## Pre-Built Routines
        - `search_routines(keywords: list[str])` — Search routines by keywords. Use this to find matching routines.
        - `get_routine_details(routine_id: str)` — Get routine parameters before execution.
        - `execute_routine(routine_id: str, parameters: dict = {})` — Run a routine with parameters.
        - `execute_routines_parallel(routine_requests: list[dict], max_concurrency: int = 5)` — Run multiple independent routines in parallel on separate tabs. Each dict needs 'routine_id' and optionally 'parameters'. Use when routines don't depend on each other's results.

        ## IMPORTANT: Always Prioritize Routines
        Before using browser tools:
        1. Use `search_routines(keywords)` to find a matching routine
        2. If found → use `get_routine_details(routine_id)` then `execute_routine()` with proper parameters
        3. Only use browser tools when NO routine fits

        Routines are pre-built, tested, and reliable. Browser tools are for custom/exploratory tasks only.

        ## Strategy (when no routine exists)
        1. For complex/custom tasks → use `evaluate_js` or `run_cdp_command`
        2. For simple interactions → use high-level tools (click, type, navigate)
        3. Always read page state with `get_page_html` when unsure
        4. Be concise in responses
    """).strip()

    AUTONOMOUS_SYSTEM_PROMPT: str = ""  # Not used for now

    ## Magic methods

    def __init__(
        self,
        emit_message_callable: Callable[[EmittedMessage], None],
        persist_chat_callable: Callable[[Chat], Chat] | None = None,
        persist_chat_thread_callable: Callable[[ChatThread], ChatThread] | None = None,
        stream_chunk_callable: Callable[[str], None] | None = None,
        llm_model: LLMModel = OpenAIModel.GPT_5_1,
        run_mode: RunMode = RunMode.CONVERSATIONAL,
        chat_thread: ChatThread | None = None,
        existing_chats: list[Chat] | None = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
    ) -> None:
        """
        Initialize the Vectorly browser agent.

        Args:
            emit_message_callable: Callback function to emit messages to the host.
            persist_chat_callable: Optional callback to persist Chat objects.
            persist_chat_thread_callable: Optional callback to persist ChatThread.
            stream_chunk_callable: Optional callback for streaming text chunks.
            llm_model: The LLM model to use for conversation.
            run_mode: How the specialist will be run (conversational or autonomous).
            chat_thread: Existing ChatThread to continue, or None for new conversation.
            existing_chats: Existing Chat messages if loading from persistence.
            remote_debugging_address: Chrome remote debugging address for routine execution.
        """
        # Validate required config
        if not Config.VECTORLY_API_KEY:
            raise ValueError("VECTORLY_API_KEY is not set")
        if not Config.VECTORLY_API_BASE:
            raise ValueError("VECTORLY_API_BASE is not set")

        self._remote_debugging_address = remote_debugging_address
        self._routines_cache: dict[str, Routine] | None = None

        # Browser tab and CDP session state
        self._tab_id: str | None = None
        self._browser_context_id: str | None = None
        self._page_ws_url: str | None = None
        self._browser_ws: WebSocket | None = None
        self._session_id: str | None = None
        self._current_url: str = "about:blank"

        # Create browser tab and establish CDP session
        self._create_browser_tab()

        super().__init__(
            emit_message_callable=emit_message_callable,
            persist_chat_callable=persist_chat_callable,
            persist_chat_thread_callable=persist_chat_thread_callable,
            stream_chunk_callable=stream_chunk_callable,
            llm_model=llm_model,
            run_mode=run_mode,
            chat_thread=chat_thread,
            existing_chats=existing_chats,
            documentation_data_loader=None,
        )
        # Pre-load routines on boot
        self._get_all_routines()

        logger.debug(
            "VectorlyBrowserAgent initialized with model: %s, chat_thread_id: %s, tab_id: %s",
            llm_model,
            self._thread.id,
            self._tab_id,
        )

    ## Browser tab management

    def _create_browser_tab(self) -> None:
        """Create a new browser tab and establish CDP session."""
        try:
            # Create the tab
            target_id, browser_context_id, browser_ws = cdp_new_tab(
                remote_debugging_address=self._remote_debugging_address,
                incognito=False,
                url="about:blank",
            )
            # Close the browser-level websocket (we only needed it for tab creation)
            try:
                browser_ws.close()
            except Exception:
                pass

            self._tab_id = target_id
            self._browser_context_id = browser_context_id

            # Build page-level WebSocket URL
            parsed = urlparse(self._remote_debugging_address)
            host_port = f"{parsed.hostname}:{parsed.port}"
            self._page_ws_url = f"ws://{host_port}/devtools/page/{target_id}"

            # Connect to the page-level WebSocket
            self._browser_ws = websocket.create_connection(self._page_ws_url, timeout=10)

            # Attach to the target with flattened session
            attach_id = send_cmd(self._browser_ws, "Target.attachToTarget", {
                "targetId": target_id,
                "flatten": True,
            })
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == attach_id, time.time() + 10)
            self._session_id = reply["result"]["sessionId"]

            # Enable required domains
            send_cmd(self._browser_ws, "Page.enable", session_id=self._session_id)
            send_cmd(self._browser_ws, "Runtime.enable", session_id=self._session_id)
            send_cmd(self._browser_ws, "Network.enable", session_id=self._session_id)
            send_cmd(self._browser_ws, "DOM.enable", session_id=self._session_id)
            send_cmd(self._browser_ws, "Input.enable", session_id=self._session_id)

            logger.info("Created browser tab and CDP session: tab_id=%s, session_id=%s", self._tab_id, self._session_id)
        except Exception as e:
            logger.error("Failed to create browser tab: %s", e)
            raise RuntimeError(f"Failed to create browser tab: {e}")

    @property
    def tab_id(self) -> str | None:
        """Return the tab ID for this agent session."""
        return self._tab_id

    @property
    def page_ws_url(self) -> str | None:
        """Return the page-level WebSocket URL for this agent session."""
        return self._page_ws_url

    ## Abstract method implementations

    def _get_system_prompt(self) -> str:
        """Get system prompt with current time."""
        now = datetime.now()
        time_info = f"\n\n## Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S %Z').strip()}"
        return self.SYSTEM_PROMPT + time_info

    def _get_autonomous_system_prompt(self) -> str:
        """Get system prompt for autonomous mode (not implemented)."""
        return self.AUTONOMOUS_SYSTEM_PROMPT

    def _get_autonomous_initial_message(self, task: str) -> str:
        """Build initial message for autonomous mode (not implemented)."""
        return f"TASK: {task}"

    ## Routine fetching (instance-level cache)

    def _get_all_routines(self) -> dict[str, Routine]:
        """
        Get all routines from Vectorly (organization + public).

        Returns:
            Dictionary mapping routine IDs to Routine objects.
        """
        if self._routines_cache is not None:
            return self._routines_cache

        routines: list[dict[str, Any]] = []

        headers = {
            "Content-Type": "application/json",
            "X-Service-Token": Config.VECTORLY_API_KEY,
        }

        # Get organization routines
        response = requests.get(
            f"{Config.VECTORLY_API_BASE}/routines/organization_routines",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        routines.extend(response.json())

        # Get public routines
        response = requests.get(
            f"{Config.VECTORLY_API_BASE}/routines/public",
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        routines.extend(response.json())

        # Build dictionary of routine IDs to Routine objects
        routine_dict = {}
        for routine_data in routines:
            try:
                routine = Routine(**routine_data)
                routine_dict[routine_data["id"]] = routine
            except Exception:
                continue

        self._routines_cache = routine_dict
        return routine_dict

    def _clear_routine_cache(self) -> None:
        """Clear the routine cache to force a refresh."""
        self._routines_cache = None

    ## Tool handlers

    @agent_tool()
    @token_optimized
    def _search_routines(self, keywords: list[str]) -> dict[str, Any]:
        """
        Search for routines by keywords. Matches against routine name and description.

        Args:
            keywords: List of keywords to search for (case-insensitive, matches if ANY keyword is found).
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if not routines:
            return {"message": "No routines available", "matches": []}

        # Normalize keywords to lowercase
        keywords_lower = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not keywords_lower:
            return {"error": "No valid keywords provided"}

        matches = []
        for routine_id, routine in routines.items():
            # Build searchable text from name and description
            searchable = f"{routine.name} {routine.description or ''}".lower()

            # Check if any keyword matches
            matched_keywords = [kw for kw in keywords_lower if kw in searchable]
            if matched_keywords:
                matches.append({
                    "id": routine_id,
                    "name": routine.name,
                    "description": routine.description,
                    "parameter_count": len(routine.parameters) if routine.parameters else 0,
                    "matched_keywords": matched_keywords,
                })

        # Sort by number of matched keywords (most matches first)
        matches.sort(key=lambda x: len(x["matched_keywords"]), reverse=True)

        return {
            "keywords": keywords_lower,
            "match_count": len(matches),
            "matches": matches[:20],  # Limit to top 20 matches
        }

    @agent_tool()
    @token_optimized
    def _get_routine_details(self, routine_id: str) -> dict[str, Any]:
        """
        Get full details of a specific routine by ID.

        Returns the routine's name, description, parameters, and operations.
        Use this to understand what inputs are required before execution.

        Args:
            routine_id: The ID of the routine to retrieve.
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if routine_id not in routines:
            return {"error": f"Routine ID '{routine_id}' not found"}

        routine = routines[routine_id]
        return routine.model_dump()

    @agent_tool()
    def _execute_routine(
        self,
        routine_id: str,
        parameters: dict[str, Any] = {},
    ) -> dict[str, Any]:
        """
        Execute a routine with the given parameters.

        This will run the routine against the connected Chrome browser.
        Make sure Chrome is running with remote debugging enabled.

        Args:
            routine_id: The ID of the routine to execute.
            parameters: Dictionary of parameter values required by the routine.
        """
        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        if routine_id not in routines:
            return {"error": f"Routine ID '{routine_id}' not found"}

        routine = routines[routine_id]

        try:
            result = routine.execute(
                parameters_dict=parameters,
                remote_debugging_address=self._remote_debugging_address,
                tab_id=self._tab_id,
                close_tab_when_done=False,
            )
            return {
                "success": True,
                "routine_id": routine_id,
                "routine_name": routine.name,
                "result": result.model_dump() if hasattr(result, "model_dump") else result,
            }
        except Exception as e:
            logger.exception("Routine execution failed: %s", e)
            return {
                "success": False,
                "routine_id": routine_id,
                "routine_name": routine.name,
                "error": str(e),
            }

    @agent_tool()
    def _execute_routines_parallel(
        self,
        routine_requests: list[dict[str, Any]],
        max_concurrency: int = 5,
    ) -> dict[str, Any]:
        """
        Execute multiple routines in parallel, each on its own browser tab.

        Each routine runs in an isolated tab that is automatically closed after execution.
        Use this when you need to run several independent routines simultaneously for faster results.
        The agent's own browser tab is not affected.

        Args:
            routine_requests: List of routine execution requests. Each dict must have 'routine_id' (str) and optionally 'parameters' (dict).
            max_concurrency: Maximum number of routines to run simultaneously. Defaults to 5.
        """
        if not routine_requests:
            return {"error": "No routine requests provided"}

        try:
            routines = self._get_all_routines()
        except requests.RequestException as e:
            return {"error": f"Failed to fetch routines: {e}"}

        max_concurrency = max(1, min(max_concurrency, 10))

        # Validate each request
        validated: list[tuple[str, Any, dict]] = []
        validation_errors: list[dict[str, Any]] = []

        for i, req in enumerate(routine_requests):
            routine_id = req.get("routine_id")
            parameters = req.get("parameters", {})

            if not routine_id:
                validation_errors.append({"index": i, "error": "Missing 'routine_id'"})
                continue

            if routine_id not in routines:
                validation_errors.append({"index": i, "routine_id": routine_id, "error": f"Routine ID '{routine_id}' not found"})
                continue

            validated.append((routine_id, routines[routine_id], parameters))

        if not validated:
            return {
                "success": False,
                "error": "All routine requests failed validation",
                "validation_errors": validation_errors,
                "results": [],
            }

        # Execute validated routines in parallel, each on its own new tab
        def execute_one(routine_id: str, routine: Any, parameters: dict) -> dict[str, Any]:
            try:
                result = routine.execute(
                    parameters_dict=parameters,
                    remote_debugging_address=self._remote_debugging_address,
                    tab_id=None,
                    close_tab_when_done=True,
                )
                return {
                    "success": True,
                    "routine_id": routine_id,
                    "routine_name": routine.name,
                    "result": result.model_dump() if hasattr(result, "model_dump") else result,
                }
            except Exception as e:
                logger.exception("Parallel routine execution failed for %s: %s", routine_id, e)
                return {
                    "success": False,
                    "routine_id": routine_id,
                    "routine_name": routine.name,
                    "error": str(e),
                }

        results: list[dict[str, Any]] = []
        completed_count = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {
                executor.submit(execute_one, rid, routine, params): rid
                for rid, routine, params in validated
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed_count += 1

                # Stream each result as it completes
                status = "succeeded" if result.get("success") else "FAILED"
                self._emit_message(ChatResponseEmittedMessage(
                    content=f"[{completed_count}/{len(validated)}] Routine '{result.get('routine_name')}' {status}.",
                ))

        results.sort(key=lambda r: r.get("routine_id", ""))

        succeeded = sum(1 for r in results if r.get("success"))
        failed = len(results) - succeeded

        return {
            "success": failed == 0 and not validation_errors,
            "total_requested": len(routine_requests),
            "total_executed": len(validated),
            "succeeded": succeeded,
            "failed": failed,
            "validation_errors": validation_errors or None,
            "results": results,
        }

    ## Browser control tools

    @agent_tool()
    def _navigate(self, url: str) -> dict[str, Any]:
        """
        Navigate to a URL in the browser.

        Args:
            url: The URL to navigate to.
        """
        try:
            send_cmd(self._browser_ws, "Page.navigate", {"url": url}, session_id=self._session_id)
            self._current_url = url
            time.sleep(3)  # Allow page to load
            return {"success": True, "url": url}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _click(self, selector: str) -> dict[str, Any]:
        """
        Click an element by CSS selector.

        Args:
            selector: CSS selector to find the element to click.
        """
        try:
            click_js = generate_click_js(selector, ensure_visible=True)
            eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                "expression": click_js,
                "returnByValue": True,
                "timeout": 20000,
            }, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 20)

            if "error" in reply:
                return {"error": f"Failed to evaluate click: {reply['error']}"}

            click_data = reply.get("result", {}).get("result", {}).get("value", {})
            if "error" in click_data:
                return {"error": click_data["error"]}

            x, y = click_data["x"], click_data["y"]

            # Perform the click
            send_cmd(self._browser_ws, "Input.dispatchMouseEvent", {
                "type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1,
            }, session_id=self._session_id)
            time.sleep(0.05)
            send_cmd(self._browser_ws, "Input.dispatchMouseEvent", {
                "type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1,
            }, session_id=self._session_id)

            return {"success": True, "selector": selector, "coordinates": {"x": x, "y": y}}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _type_text(self, selector: str, text: str, clear: bool = False) -> dict[str, Any]:
        """
        Type text into an input element.

        Args:
            selector: CSS selector to find the input element.
            text: Text to type into the element.
            clear: Whether to clear existing text before typing.
        """
        try:
            type_js = generate_type_js(selector, clear)
            eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                "expression": type_js,
                "returnByValue": True,
                "timeout": 20000,
            }, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 20)

            if "error" in reply:
                return {"error": f"Failed to focus element: {reply['error']}"}

            type_data = reply.get("result", {}).get("result", {}).get("value", {})
            if "error" in type_data:
                return {"error": type_data["error"]}

            # Type each character
            for char in text:
                send_cmd(self._browser_ws, "Input.dispatchKeyEvent", {
                    "type": "keyDown", "text": char,
                }, session_id=self._session_id)
                send_cmd(self._browser_ws, "Input.dispatchKeyEvent", {
                    "type": "keyUp", "text": char,
                }, session_id=self._session_id)
                time.sleep(0.02)

            return {"success": True, "selector": selector, "text_length": len(text)}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _press_key(self, key: str) -> dict[str, Any]:
        """
        Press a keyboard key.

        Args:
            key: The key to press (e.g., "enter", "tab", "escape", "backspace", "arrowdown").
        """
        key_mapping = {
            "enter": "Enter", "tab": "Tab", "escape": "Escape", "esc": "Escape",
            "backspace": "Backspace", "delete": "Delete",
            "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
            "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
            "home": "Home", "end": "End", "pageup": "PageUp", "pagedown": "PageDown",
            "space": " ", "shift": "Shift", "control": "Control", "ctrl": "Control",
            "alt": "Alt", "meta": "Meta",
        }
        try:
            cdp_key = key_mapping.get(key.lower(), key)
            send_cmd(self._browser_ws, "Input.dispatchKeyEvent", {
                "type": "keyDown", "key": cdp_key,
            }, session_id=self._session_id)
            time.sleep(0.05)
            send_cmd(self._browser_ws, "Input.dispatchKeyEvent", {
                "type": "keyUp", "key": cdp_key,
            }, session_id=self._session_id)
            return {"success": True, "key": key}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _get_page_html(self, selector: str | None = None) -> dict[str, Any]:
        """
        Get the HTML content of the page or a specific element.

        Args:
            selector: Optional CSS selector. If provided, returns that element's HTML. If None, returns full page HTML.
        """
        try:
            html_js = generate_get_html_js(selector)
            eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                "expression": html_js,
                "returnByValue": True,
            }, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 30)

            if "error" in reply:
                return {"error": f"Failed to get HTML: {reply['error']}"}

            html = reply.get("result", {}).get("result", {}).get("value", "")
            return {"success": True, "html": html, "length": len(html)}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _evaluate_js(self, js: str) -> dict[str, Any]:
        """
        Execute raw JavaScript in the browser. No validation or wrapping - just runs the code.

        Args:
            js: JavaScript code to execute.
        """
        try:
            eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                "expression": js,
                "returnByValue": True,
                "awaitPromise": True,
            }, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 30)
            return reply.get("result", {})
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _wait_for_url(self, url_regex: str, timeout_seconds: float = 20.0) -> dict[str, Any]:
        """
        Wait for the browser URL to match a regex pattern.

        Args:
            url_regex: Regex pattern to match against the current URL.
            timeout_seconds: Maximum time to wait in seconds.
        """
        try:
            wait_js = generate_wait_for_url_js(url_regex)
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                    "expression": wait_js,
                    "returnByValue": True,
                }, session_id=self._session_id)
                reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 5)

                wait_data = reply.get("result", {}).get("result", {}).get("value", {})
                if wait_data.get("matches"):
                    self._current_url = wait_data.get("currentUrl", self._current_url)
                    return {"success": True, "url": self._current_url}
                time.sleep(0.2)

            return {"error": f"Timeout waiting for URL to match '{url_regex}'", "current_url": wait_data.get("currentUrl")}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _scroll(self, delta_x: int = 0, delta_y: int = 0, selector: str | None = None) -> dict[str, Any]:
        """
        Scroll the page or a specific element.

        Args:
            delta_x: Horizontal scroll amount (positive = right).
            delta_y: Vertical scroll amount (positive = down).
            selector: Optional CSS selector. If provided, scrolls that element. If None, scrolls the window.
        """
        try:
            if selector:
                scroll_js = generate_scroll_element_js(selector, delta_x, delta_y, "auto")
            else:
                scroll_js = generate_scroll_window_js(None, None, delta_x, delta_y, "auto")

            eval_id = send_cmd(self._browser_ws, "Runtime.evaluate", {
                "expression": scroll_js,
                "returnByValue": True,
            }, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == eval_id, time.time() + 10)

            scroll_data = reply.get("result", {}).get("result", {}).get("value", {})
            if "error" in scroll_data:
                return {"error": scroll_data["error"]}

            return {"success": True, "delta_x": delta_x, "delta_y": delta_y}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _get_cookies(self, domain_filter: str = "*") -> dict[str, Any]:
        """
        Get browser cookies, optionally filtered by domain.

        Args:
            domain_filter: Domain to filter cookies by. Use "*" for all cookies.
        """
        try:
            cookies_id = send_cmd(self._browser_ws, "Network.getAllCookies", {}, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == cookies_id, time.time() + 10)

            if "error" in reply:
                return {"error": f"Failed to get cookies: {reply['error']}"}

            cookies = reply.get("result", {}).get("cookies", [])

            if domain_filter != "*":
                cookies = [c for c in cookies if domain_filter in c.get("domain", "")]

            return {"success": True, "cookies": cookies, "count": len(cookies)}
        except Exception as e:
            return {"error": str(e)}

    @agent_tool()
    def _run_cdp_command(self, method: str, params: dict[str, Any] = {}) -> dict[str, Any]:
        """
        Run a raw Chrome DevTools Protocol command.

        Args:
            method: CDP method name (e.g., "Page.captureScreenshot", "DOM.getDocument").
            params: Parameters for the CDP command.
        """
        try:
            cmd_id = send_cmd(self._browser_ws, method, params, session_id=self._session_id)
            reply = recv_until(self._browser_ws, lambda m: m.get("id") == cmd_id, time.time() + 30)
            return reply.get("result", {})
        except Exception as e:
            return {"error": str(e)}
