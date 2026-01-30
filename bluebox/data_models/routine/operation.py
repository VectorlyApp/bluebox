"""
bluebox/data_models/routine/operation.py

Operation types for browser automation routines.

Contains:
- RoutineOperationTypes: Enum of all operation types
- RoutineOperation: Base class for all operations
- RoutineNavigateOperation: Navigate to URL
- RoutineSleepOperation: Pause execution
- RoutineFetchOperation: HTTP requests via browser fetch API
- RoutineReturnOperation: Return data from sessionStorage
- RoutineGetCookiesOperation: Get cookies via CDP
- RoutineClickOperation: Click elements by selector
- RoutineTypeOperation: Type text into inputs
- RoutinePressOperation: Press keyboard keys
- RoutineWaitForUrlOperation: Wait for URL regex match
- RoutineScrollOperation: Scroll page/element
- RoutineReturnHTMLOperation: Return page HTML
- RoutineDownloadOperation: Download binary files
- RoutineJsEvaluateOperation: Execute custom JavaScript
"""

import ast
import json
import re
import time
from enum import StrEnum
from typing import Annotated, ClassVar, Literal, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from bluebox.data_models.routine.endpoint import Endpoint
from bluebox.data_models.routine.execution import RoutineExecutionContext, FetchExecutionResult, OperationExecutionMetadata
from bluebox.data_models.routine.parameter import VALID_PLACEHOLDER_PREFIXES, BUILTIN_PARAMETERS
from bluebox.data_models.ui_elements import MouseButton, ScrollBehavior, HTMLScope
from bluebox.utils.data_utils import apply_params_to_str, apply_params_to_json, assert_balanced_js_delimiters
from bluebox.utils.logger import get_logger
from bluebox.utils.js_utils import (
    generate_fetch_js,
    generate_click_js,
    generate_type_js,
    generate_scroll_element_js,
    generate_scroll_window_js,
    generate_wait_for_url_js,
    generate_store_in_session_storage_js,
    generate_get_session_storage_length_js,
    generate_get_session_storage_chunk_js,
    generate_get_download_chunk_js,
    generate_get_html_js,
    generate_download_js,
    generate_js_evaluate_wrapper_js,
)
from bluebox.utils.web_socket_utils import send_cmd, recv_until

logger = get_logger(name=__name__)


# Enums ___________________________________________________________________________________________

class RoutineOperationTypes(StrEnum):
    """
    Browser operation types for running routines.
    """
    NAVIGATE = "navigate"
    SLEEP = "sleep"
    FETCH = "fetch"
    RETURN = "return"
    NETWORK_SNIFFING = "network_sniffing"
    GET_COOKIES = "get_cookies"
    DOWNLOAD = "download"

    # UI automation operations
    CLICK = "click"
    INPUT_TEXT = "input_text"
    PRESS = "press"
    WAIT_FOR_URL = "wait_for_url"
    SCROLL = "scroll"

    RETURN_HTML = "return_html"
    JS_EVALUATE = "js_evaluate"


# Base operation class ____________________________________________________________________________

class RoutineOperation(BaseModel):
    """
    Base class for routine operations.

    Args:
        type (RoutineOperationTypes): The type of operation.

    Returns:
        RoutineOperation: The interpolated operation.
    """
    type: RoutineOperationTypes

    def execute(self, routine_execution_context: RoutineExecutionContext) -> None:
        """
        Execute this operation with automatic metadata collection.

        This method wraps _execute_operation() to collect execution metadata (type, duration).
        Subclasses should override _execute_operation() to implement their specific behavior.
        Subclasses can add operation-specific data via:
            routine_execution_context.current_operation_metadata.details["key"] = value

        Args:
            routine_execution_context: Execution context containing parameters, CDP functions, and mutable state.
        """
        # Create metadata and set on context so _execute_operation can add details
        routine_execution_context.current_operation_metadata = OperationExecutionMetadata(
            type=self.type,
            duration_seconds=0.0,
        )
        start = time.perf_counter()
        try:
            self._execute_operation(routine_execution_context)
        except Exception as e:
            routine_execution_context.current_operation_metadata.error = str(e)
        finally:
            duration = time.perf_counter() - start
            routine_execution_context.current_operation_metadata.duration_seconds = duration
            routine_execution_context.result.operations_metadata.append(
                routine_execution_context.current_operation_metadata
            )
            routine_execution_context.current_operation_metadata = None

    def _store_request_response_metadata(
        self,
        routine_execution_context: RoutineExecutionContext,
        payload: dict,
    ) -> None:
        """Store request/response metadata from JS payload into operation metadata."""
        if routine_execution_context.current_operation_metadata is not None:
            if payload.get("request"):
                routine_execution_context.current_operation_metadata.details["request"] = payload["request"]
            if payload.get("response"):
                routine_execution_context.current_operation_metadata.details["response"] = payload["response"]

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """
        Implementation of operation execution.

        Subclasses must override this method to implement their specific behavior.
        Operations modify routine_execution_context directly.
        E.g., routine_execution_context.result, routine_execution_context.current_url.
        Errors should be raised as exceptions.

        Args:
            routine_execution_context: Execution context containing parameters, CDP functions, and mutable state.
        """
        raise NotImplementedError(f"_execute_operation() not implemented for {type(self).__name__}")


# Operation classes _______________________________________________________________________________

class RoutineNavigateOperation(RoutineOperation):
    """
    Navigate operation for routine.

    Args:
        type (Literal[RoutineOperationTypes.NAVIGATE]): The type of operation.
        url (str): The URL to navigate to.
        sleep_after_navigation_seconds (float): Seconds to wait after navigation for page to load.

    Returns:
        RoutineNavigateOperation: The interpolated operation.
    """
    type: Literal[RoutineOperationTypes.NAVIGATE] = RoutineOperationTypes.NAVIGATE
    url: str
    sleep_after_navigation_seconds: float = Field(
        default=3.0,
        description="Seconds to wait after navigation for page to load (allows JS to execute and populate storage)"
    )

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Navigate to the specified URL."""
        url = apply_params_to_str(self.url, routine_execution_context.parameters_dict)
        routine_execution_context.send_cmd("Page.navigate", {"url": url}, session_id=routine_execution_context.session_id)
        routine_execution_context.current_url = url

        # Wait for page to load (allows JS to execute and populate localStorage/sessionStorage)
        if self.sleep_after_navigation_seconds > 0:
            logger.info(f"Waiting {self.sleep_after_navigation_seconds}s after navigation to {url}")
            time.sleep(self.sleep_after_navigation_seconds)


class RoutineSleepOperation(RoutineOperation):
    """
    Sleep operation for routine.

    Args:
        type (Literal[RoutineOperationTypes.SLEEP]): The type of operation.
        timeout_seconds (float): The number of seconds to sleep.

    Returns:
        RoutineSleepOperation: The interpolated operation.
    """
    type: Literal[RoutineOperationTypes.SLEEP] = RoutineOperationTypes.SLEEP
    timeout_seconds: float

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Sleep for the specified duration."""
        time.sleep(self.timeout_seconds)


class RoutineFetchOperation(RoutineOperation):
    """
    Fetch operation for routine.

    Args:
        type (Literal[RoutineOperationTypes.FETCH]): The type of operation.
        endpoint (Endpoint): The endpoint to fetch.
        session_storage_key (str | None): The session storage key to save the result to (optional).

    Returns:
        RoutineFetchOperation: The interpolated operation.
    """
    type: Literal[RoutineOperationTypes.FETCH] = RoutineOperationTypes.FETCH
    endpoint: Endpoint
    session_storage_key: str | None = None

    def _execute_fetch(
        self,
        routine_execution_context: RoutineExecutionContext,
    ) -> FetchExecutionResult:
        """Execute the fetch request and return the result."""
        parameters_dict = routine_execution_context.parameters_dict or {}

        # Apply parameters to endpoint
        param_type_map = routine_execution_context.param_type_map
        fetch_url = apply_params_to_str(self.endpoint.url, parameters_dict)
        headers: dict = {}
        if self.endpoint.headers:
            headers = apply_params_to_json(self.endpoint.headers, parameters_dict, param_type_map)

        body = None
        if self.endpoint.body:
            body = apply_params_to_json(self.endpoint.body, parameters_dict, param_type_map)

        # Serialize body to JS string literal
        if body is None:
            body_js_literal = "null"
        elif isinstance(body, (dict, list)):
            body_js_literal = json.dumps(body)
        elif isinstance(body, bytes):
            body_js_literal = json.dumps(body.decode("utf-8", errors="ignore"))
        else:
            body_js_literal = json.dumps(str(body))

        # Build JS using the shared generator
        expr = generate_fetch_js(
            fetch_url=fetch_url,
            headers=headers or {},
            body_js_literal=body_js_literal,
            endpoint_method=self.endpoint.method,
            endpoint_credentials=self.endpoint.credentials,
            session_storage_key=self.session_storage_key,
        )

        # Execute the fetch
        ws = routine_execution_context.ws
        session_id = routine_execution_context.session_id
        timeout = routine_execution_context.timeout

        logger.info(f"Sending Runtime.evaluate for fetch with timeout={timeout}s")
        eval_id = send_cmd(
            ws,
            "Runtime.evaluate",
            {
                "expression": expr,
                "awaitPromise": True,
                "returnByValue": True,
                "timeout": int(timeout * 1000),
            },
            session_id=session_id,
        )

        reply = recv_until(ws, lambda m: m.get("id") == eval_id, time.time() + timeout)

        if "error" in reply:
            logger.error(f"Error in _execute_fetch (CDP error): {reply['error']}")
            return FetchExecutionResult(ok=False, error=reply["error"])

        payload = reply["result"]["result"].get("value")

        # Store request/response metadata (returned from JS)
        if isinstance(payload, dict):
            self._store_request_response_metadata(routine_execution_context, payload)

        if isinstance(payload, dict) and payload.get("__err"):
            logger.error(f"Error in _execute_fetch (JS error): {payload.get('__err')}")
            return FetchExecutionResult(
                ok=False,
                error=payload.get("__err"),
                resolved_values=payload.get("resolvedValues", {}),
            )

        logger.info(f"Fetch result payload: {str(payload)[:1000]}...")

        return FetchExecutionResult(
            ok=True,
            result=payload.get("value"),
            resolved_values=payload.get("resolvedValues", {}),
        )

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Execute the fetch operation."""

        if routine_execution_context.ws is None:
            raise RuntimeError("WebSocket not available in RoutineExecutionContext for fetch operation")

        # If current page is blank, navigate to the target origin first to avoid CORS
        if not routine_execution_context.current_url or routine_execution_context.current_url == "about:blank":
            # Extract origin URL from the fetch endpoint (scheme + netloc)
            fetch_url = apply_params_to_str(self.endpoint.url, routine_execution_context.parameters_dict)
            parsed = urlparse(fetch_url)
            origin_url = f"{parsed.scheme}://{parsed.netloc}"
            logger.info(f"Current page is blank, navigating to {origin_url} before fetch")
            routine_execution_context.send_cmd(
                "Page.navigate",
                {"url": origin_url},
                session_id=routine_execution_context.session_id
            )
            routine_execution_context.current_url = origin_url
            time.sleep(3)  # Wait for page to load

        fetch_result = self._execute_fetch(routine_execution_context)

        # Check for errors
        if not fetch_result.ok:
            raise RuntimeError(f"Fetch failed: {fetch_result.error}")

        # Collect resolved values
        if fetch_result.resolved_values:
            routine_execution_context.result.placeholder_resolution.update(fetch_result.resolved_values)
            for k, v in fetch_result.resolved_values.items():
                if v is None:
                    routine_execution_context.result.warnings.append(f"Could not resolve placeholder: {k}")


class RoutineReturnOperation(RoutineOperation):
    """
    Return operation for routine.

    Args:
        type (Literal[RoutineOperationTypes.RETURN]): The type of operation.
        session_storage_key (str): The session storage key to return.

    Returns:
        RoutineReturnOperation: The interpolated operation.
    """
    type: Literal[RoutineOperationTypes.RETURN] = RoutineOperationTypes.RETURN
    session_storage_key: str

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Get result from session storage and set it as the routine result."""

        chunk_size = 256 * 1024  # 256KB chunks

        # First get the length
        len_js = generate_get_session_storage_length_js(self.session_storage_key)
        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {"expression": len_js, "returnByValue": True},
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + routine_execution_context.timeout
        )
        if "error" in reply:
            raise RuntimeError(f"Failed to get storage length: {reply['error']}")

        total_len = reply["result"]["result"].get("value", 0)

        if total_len == 0:
            routine_execution_context.result.data = None
        else:
            stored_value = ""
            for offset in range(0, total_len, chunk_size):
                end = min(offset + chunk_size, total_len)
                chunk_js = generate_get_session_storage_chunk_js(
                    self.session_storage_key,
                    offset,
                    end,
                )
                eval_id = routine_execution_context.send_cmd(
                    "Runtime.evaluate",
                    {"expression": chunk_js, "returnByValue": True},
                    session_id=routine_execution_context.session_id,
                )
                reply = routine_execution_context.recv_until(
                    lambda m: m.get("id") == eval_id, time.time() + routine_execution_context.timeout
                )
                if "error" in reply:
                    raise RuntimeError(f"Failed to retrieve chunk at offset {offset}: {reply['error']}")

                chunk = reply["result"]["result"].get("value", "")
                stored_value += chunk
                time.sleep(0.01)

            # Try to parse as JSON
            try:
                routine_execution_context.result.data = json.loads(stored_value)
            except Exception:
                try:
                    result_literal = ast.literal_eval(stored_value)
                    if isinstance(result_literal, (dict, list)):
                        routine_execution_context.result.data = result_literal
                    else:
                        routine_execution_context.result.data = stored_value
                except Exception:
                    routine_execution_context.result.data = stored_value


class RoutineGetCookiesOperation(RoutineOperation):
    """
    Get all cookies (including HttpOnly) via CDP and store them in session storage.

    This operation uses the Chrome DevTools Protocol's Network.getAllCookies() method
    to retrieve all cookies, including HttpOnly cookies that are not accessible via
    document.cookie in JavaScript.

    Args:
        type (Literal[RoutineOperationTypes.GET_COOKIES]): The type of operation.
        session_storage_key (str): The session storage key to save the cookies to.
        domain_filter (str): Domain to filter cookies by. Use wildcard '*' to get all cookies.

    Returns:
        RoutineGetCookiesOperation: The operation instance.
    """
    type: Literal[RoutineOperationTypes.GET_COOKIES] = RoutineOperationTypes.GET_COOKIES
    session_storage_key: str
    domain_filter: str = Field(
        default="*",
        description="Domain to filter cookies by. Use wildcard '*' to get all cookies."
    )

    @field_validator("domain_filter")
    @classmethod
    def validate_domain_filter_not_empty(cls, v: str) -> str:
        """Validate that domain_filter is not empty and strip whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("domain_filter cannot be empty")
        return v

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Get all cookies via CDP and store them in session storage."""
        cookies_id = routine_execution_context.send_cmd(
            "Network.getAllCookies",
            {},
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            predicate=lambda m: m.get("id") == cookies_id,
            deadline=time.time() + routine_execution_context.timeout
        )

        if "error" in reply:
            raise RuntimeError(f"Failed to get cookies: {reply['error']}")

        cookies = reply["result"].get("cookies", [])

        # Filter by domain if specified (not wildcard)
        if self.domain_filter != "*":
            cookies = [
                cookie for cookie in cookies
                if self.domain_filter in cookie.get("domain", "")
            ]

        # Store cookies in session storage via JS
        cookies_json = json.dumps(cookies)
        store_js = generate_store_in_session_storage_js(
            self.session_storage_key,
            cookies_json,
        )

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {"expression": store_js, "returnByValue": True},
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            predicate=lambda m: m.get("id") == eval_id,
            deadline=time.time() + routine_execution_context.timeout
        )

        if "error" in reply:
            raise RuntimeError(f"Failed to store cookies in session storage: {reply['error']}")

        store_result = reply["result"]["result"].get("value", {})
        if not store_result.get("ok"):
            raise RuntimeError(store_result.get("error"))


# UI automation operations ________________________________________________________________________

class RoutineClickOperation(RoutineOperation):
    """
    Click operation for routine - clicks on an element by CSS selector.

    Important: This operation automatically validates element visibility to avoid clicking
    hidden honeypot traps. Only visible, interactable elements will be clicked.

    Args:
        type (Literal[RoutineOperationTypes.CLICK]): The type of operation.
        selector (str): CSS selector to find the element to click.
        button (str): Mouse button to use ("left", "right", "middle"). Defaults to "left".
        click_count (int): Number of clicks to perform. Defaults to 1.
        timeout_ms (int): Maximum time to wait for element in milliseconds. Defaults to 20_000.
        ensure_visible (bool): Whether to scroll element into view before clicking. Defaults to True.
    """
    type: Literal[RoutineOperationTypes.CLICK] = RoutineOperationTypes.CLICK
    selector: str
    button: MouseButton = MouseButton.LEFT
    click_count: int = 1
    timeout_ms: int = 20_000
    ensure_visible: bool = True

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Click on an element by CSS selector."""
        selector = apply_params_to_str(self.selector, routine_execution_context.parameters_dict)
        click_js = generate_click_js(selector, self.ensure_visible)

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": click_js,
                "returnByValue": True,
                "timeout": self.timeout_ms,
            },
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + (self.timeout_ms / 1000)
        )

        if "error" in reply:
            raise RuntimeError(f"Failed to evaluate click script: {reply['error']}")

        click_data = reply["result"]["result"].get("value", {})

        # Store element profile in metadata
        if routine_execution_context.current_operation_metadata is not None:
            routine_execution_context.current_operation_metadata.details["selector"] = selector
            routine_execution_context.current_operation_metadata.details["element"] = click_data.get("element")

        if "error" in click_data:
            raise RuntimeError(click_data["error"])

        x = click_data["x"]
        y = click_data["y"]

        # Store click coordinates in metadata
        if routine_execution_context.current_operation_metadata is not None:
            routine_execution_context.current_operation_metadata.details["click_coordinates"] = {"x": x, "y": y}

        # Perform the click(s) using CDP Input domain
        for _ in range(self.click_count):
            # Mouse pressed
            routine_execution_context.send_cmd(
                "Input.dispatchMouseEvent",
                {
                    "type": "mousePressed",
                    "x": x,
                    "y": y,
                    "button": self.button,
                    "clickCount": 1,
                },
                session_id=routine_execution_context.session_id,
            )
            time.sleep(0.05)

            # Mouse released
            routine_execution_context.send_cmd(
                "Input.dispatchMouseEvent",
                {
                    "type": "mouseReleased",
                    "x": x,
                    "y": y,
                    "button": self.button,
                    "clickCount": 1,
                },
                session_id=routine_execution_context.session_id,
            )

            if self.click_count > 1:
                time.sleep(0.1)


class RoutineTypeOperation(RoutineOperation):
    """
    Type operation for routine - types text into an input element.

    Important: This operation automatically validates element visibility to avoid typing
    into hidden honeypot inputs. Only visible, interactable elements will receive input.

    Args:
        type (Literal[RoutineOperationTypes.INPUT_TEXT]): The type of operation.
        selector (str): CSS selector to find the input element.
        text (str): Text to type into the element.
        clear (bool): Whether to clear existing text before typing. Defaults to False.
        timeout_ms (int): Maximum time to wait for element in milliseconds. Defaults to 20_000.
    """
    type: Literal[RoutineOperationTypes.INPUT_TEXT] = RoutineOperationTypes.INPUT_TEXT
    selector: str
    text: str
    clear: bool = False
    timeout_ms: int = 20_000

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Type text into an input element."""
        selector = apply_params_to_str(self.selector, routine_execution_context.parameters_dict)
        text = apply_params_to_str(self.text, routine_execution_context.parameters_dict)
        type_js = generate_type_js(selector, self.clear)

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": type_js,
                "returnByValue": True,
                "timeout": self.timeout_ms,
            },
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + (self.timeout_ms / 1000)
        )

        if "error" in reply:
            raise RuntimeError(f"Failed to evaluate type script: {reply['error']}")

        type_data = reply["result"]["result"].get("value", {})

        # Store element profile in metadata
        if routine_execution_context.current_operation_metadata is not None:
            routine_execution_context.current_operation_metadata.details["selector"] = selector
            routine_execution_context.current_operation_metadata.details["text_length"] = len(text)
            routine_execution_context.current_operation_metadata.details["element"] = type_data.get("element")

        if "error" in type_data:
            raise RuntimeError(type_data["error"])

        # Type the text character by character using CDP Input domain
        for char in text:
            routine_execution_context.send_cmd(
                "Input.dispatchKeyEvent",
                {"type": "keyDown", "text": char},
                session_id=routine_execution_context.session_id,
            )
            routine_execution_context.send_cmd(
                "Input.dispatchKeyEvent",
                {"type": "keyUp", "text": char},
                session_id=routine_execution_context.session_id,
            )
            time.sleep(0.02)


class RoutinePressOperation(RoutineOperation):
    """
    Press operation for routine - presses a keyboard key.
    Args:
        type (Literal[RoutineOperationTypes.PRESS]): The type of operation.
        key (str): The keyboard key to press (e.g., "enter", "tab", "escape", etc.).
    """
    type: Literal[RoutineOperationTypes.PRESS] = RoutineOperationTypes.PRESS
    key: str

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Press a keyboard key."""
        key = self.key.lower()

        # Map common key names to CDP key codes
        key_mapping = {
            "enter": "Enter",
            "tab": "Tab",
            "escape": "Escape",
            "esc": "Escape",
            "backspace": "Backspace",
            "delete": "Delete",
            "arrowup": "ArrowUp",
            "arrowdown": "ArrowDown",
            "arrowleft": "ArrowLeft",
            "arrowright": "ArrowRight",
            "home": "Home",
            "end": "End",
            "pageup": "PageUp",
            "pagedown": "PageDown",
            "space": " ",
            "shift": "Shift",
            "control": "Control",
            "ctrl": "Control",
            "alt": "Alt",
            "meta": "Meta",
        }

        cdp_key = key_mapping.get(key, key)

        routine_execution_context.send_cmd(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "key": cdp_key},
            session_id=routine_execution_context.session_id,
        )
        time.sleep(0.0525)
        routine_execution_context.send_cmd(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": cdp_key},
            session_id=routine_execution_context.session_id,
        )


class RoutineWaitForUrlOperation(RoutineOperation):
    """
    Wait for URL operation for routine - waits for the current URL to match a regex pattern.
    Args:
        type (Literal[RoutineOperationTypes.WAIT_FOR_URL]): The type of operation.
        url_regex (str): Regex pattern to match against window.location.href.
        timeout_ms (int): Maximum time to wait in milliseconds. Defaults to 20_000.
    """
    type: Literal[RoutineOperationTypes.WAIT_FOR_URL] = RoutineOperationTypes.WAIT_FOR_URL
    url_regex: str
    timeout_ms: int = 20_000

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Wait for URL to match a regex pattern."""
        timeout_sec = self.timeout_ms / 1000
        start_time = time.time()
        wait_js = generate_wait_for_url_js(self.url_regex)

        matched = False
        wait_data = {}
        while time.time() - start_time < timeout_sec:
            eval_id = routine_execution_context.send_cmd(
                "Runtime.evaluate",
                {"expression": wait_js, "returnByValue": True},
                session_id=routine_execution_context.session_id,
            )
            reply = routine_execution_context.recv_until(
                lambda m: m.get("id") == eval_id, time.time() + 5
            )

            if "error" in reply:
                raise RuntimeError(f"Failed to evaluate wait for URL script: {reply['error']}")

            wait_data = reply["result"]["result"].get("value", {})

            if wait_data.get("matches"):
                matched = True
                break

            time.sleep(0.2)

        if not matched:
            raise RuntimeError(
                f"Timeout waiting for URL to match pattern '{self.url_regex}'. "
                f"Current URL: {wait_data.get('currentUrl', 'unknown')}"
            )


class RoutineScrollOperation(RoutineOperation):
    """
    Scroll operation for routine - scrolls the page or a specific element.
    Args:
        type (Literal[RoutineOperationTypes.SCROLL]): The type of operation.
        selector (str | None): CSS selector for element to scroll. If None, scrolls the window.
        x (int | None): Absolute x position to scroll to (window only).
        y (int | None): Absolute y position to scroll to (window only).
        delta_x (int | None): Relative x scroll amount.
        delta_y (int | None): Relative y scroll amount.
        behavior (Literal["auto", "smooth"]): Scroll behavior. Defaults to "auto".
        timeout_ms (int): Maximum time to wait for element in milliseconds. Defaults to 20_000.
    """
    type: Literal[RoutineOperationTypes.SCROLL] = RoutineOperationTypes.SCROLL
    # if selector is provided, scroll that element; else scroll window
    selector: str | None = None
    x: int | None = None
    y: int | None = None
    delta_x: int | None = None
    delta_y: int | None = None
    behavior: ScrollBehavior = ScrollBehavior.AUTO
    timeout_ms: int = 20_000

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Scroll the page or a specific element."""
        if self.selector:
            selector = apply_params_to_str(self.selector, routine_execution_context.parameters_dict)
            scroll_js = generate_scroll_element_js(
                selector,
                self.delta_x or 0,
                self.delta_y or 0,
                self.behavior,
            )
        else:
            scroll_js = generate_scroll_window_js(
                self.x,
                self.y,
                self.delta_x or 0,
                self.delta_y or 0,
                self.behavior,
            )

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": scroll_js,
                "returnByValue": True,
                "timeout": self.timeout_ms,
            },
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + (self.timeout_ms / 1000)
        )

        if "error" in reply:
            raise RuntimeError(f"Failed to evaluate scroll script: {reply['error']}")

        scroll_data = reply["result"]["result"].get("value", {})

        if "error" in scroll_data:
            raise RuntimeError(scroll_data["error"])

        time.sleep(0.1)


class RoutineReturnHTMLOperation(RoutineOperation):
    """
    Return HTML operation for routine - returns HTML content from the page or element.
    Args:
        type (Literal[RoutineOperationTypes.RETURN_HTML]): The type of operation.
        scope (Literal["page", "element"]): Whether to return page or element HTML. Defaults to "page".
        selector (str | None): CSS selector for element (required if scope is "element").
        timeout_ms (int): Maximum time to wait for element in milliseconds. Defaults to 20_000.
    """
    type: Literal[RoutineOperationTypes.RETURN_HTML] = RoutineOperationTypes.RETURN_HTML
    # scope: "page" returns document.documentElement.outerHTML; "element" returns selected element.outerHTML
    scope: HTMLScope = HTMLScope.PAGE
    selector: str | None = None
    timeout_ms: int = 20_000

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Get HTML from the page or a specific element."""
        if self.scope == HTMLScope.PAGE or not self.selector:
            js = generate_get_html_js()
        else:
            selector = apply_params_to_str(self.selector, routine_execution_context.parameters_dict)
            js = generate_get_html_js(selector)

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": js,
                "returnByValue": True,
                "timeout": self.timeout_ms,
            },
            session_id=routine_execution_context.session_id,
        )
        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + routine_execution_context.timeout
        )

        if "error" in reply:
            routine_execution_context.result.data = None
        else:
            routine_execution_context.result.data = reply["result"]["result"].get("value", "")


class RoutineDownloadOperation(RoutineOperation):
    """
    Download a file and return it as base64 in the routine result.

    This operation fetches a binary file (PDF, image, audio, etc.) from an endpoint,
    converts it to base64, and directly sets it as the routine's result. This is
    typically the last operation in a routine.

    Args:
        type (Literal[RoutineOperationTypes.DOWNLOAD]): The type of operation.
        endpoint (Endpoint): The endpoint to download from.
        filename (str): Filename for the downloaded file (e.g., 'report.pdf', 'image.png').
    """
    type: Literal[RoutineOperationTypes.DOWNLOAD] = RoutineOperationTypes.DOWNLOAD
    endpoint: Endpoint
    filename: str = Field(
        ...,
        description="Filename for the downloaded file (e.g., 'report.pdf', 'image.png')"
    )

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Download a file and return it as base64."""
        # Apply parameters to endpoint
        param_type_map = routine_execution_context.param_type_map
        download_url = apply_params_to_str(self.endpoint.url, routine_execution_context.parameters_dict)
        download_headers = {}
        if self.endpoint.headers:
            download_headers = apply_params_to_json(self.endpoint.headers, routine_execution_context.parameters_dict, param_type_map)

        download_body = None
        if self.endpoint.body:
            download_body = apply_params_to_json(self.endpoint.body, routine_execution_context.parameters_dict, param_type_map)

        # Serialize body for JS
        if download_body is None:
            body_js_literal = "null"
        elif isinstance(download_body, (dict, list)):
            body_js_literal = json.dumps(download_body)
        else:
            body_js_literal = json.dumps(str(download_body))

        # Interpolate filename
        download_filename = apply_params_to_str(self.filename, routine_execution_context.parameters_dict)

        # Generate JS to fetch as binary and convert to base64
        download_js = generate_download_js(
            download_url=download_url,
            headers=download_headers,
            body_js_literal=body_js_literal,
            endpoint_method=self.endpoint.method.value,
            endpoint_credentials=self.endpoint.credentials.value,
            filename=download_filename,
        )

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": download_js,
                "awaitPromise": True,
                "returnByValue": True,
                "timeout": int(routine_execution_context.timeout * 1000),
            },
            session_id=routine_execution_context.session_id,
        )

        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id, time.time() + routine_execution_context.timeout
        )

        if "error" in reply:
            raise RuntimeError(f"Download failed (CDP error): {reply['error']}")

        payload = reply["result"]["result"].get("value", {})

        # Store request/response metadata (returned from JS)
        if isinstance(payload, dict):
            self._store_request_response_metadata(routine_execution_context, payload)

        if isinstance(payload, dict) and payload.get("__err"):
            raise RuntimeError(f"Download failed: {payload.get('__err')}")

        # Get metadata from initial response
        result_content_type = payload.get("contentType")
        result_filename = payload.get("filename")
        base64_length = payload.get("base64Length", 0)

        # Retrieve base64 data in chunks from window.__downloadData
        chunk_size = 256 * 1024  # 256KB chunks

        if base64_length == 0:
            routine_execution_context.result.data = None
        else:
            chunks = []
            for offset in range(0, base64_length, chunk_size):
                end = min(offset + chunk_size, base64_length)
                chunk_js = generate_get_download_chunk_js(offset, end)

                chunk_eval_id = routine_execution_context.send_cmd(
                    "Runtime.evaluate",
                    {"expression": chunk_js, "returnByValue": True},
                    session_id=routine_execution_context.session_id,
                )
                chunk_reply = routine_execution_context.recv_until(
                    lambda m: m.get("id") == chunk_eval_id, time.time() + routine_execution_context.timeout
                )

                if "error" in chunk_reply:
                    raise RuntimeError(f"Failed to retrieve chunk at offset {offset}: {chunk_reply['error']}")

                chunk_data = chunk_reply["result"]["result"].get("value", "")
                chunks.append(chunk_data)

            routine_execution_context.result.data = "".join(chunks)

        routine_execution_context.result.is_base64 = True
        routine_execution_context.result.content_type = result_content_type
        routine_execution_context.result.filename = result_filename


class RoutineJsEvaluateOperation(RoutineOperation):
    """
    Evaluate JavaScript code in the browser context.

    ALLOWED:
    - Promises and async/await (e.g., `new Promise()`, `.then()`, `await`)
    - setTimeout/setInterval (useful for polling)
    - Loops (while, for, do) - timeout prevents infinite loops
    - Synchronous JavaScript operations
    - DOM manipulation

    BLOCKED:
    - Dynamic code generation: eval(), Function constructor
    - Network requests: fetch(), XMLHttpRequest, WebSocket, sendBeacon (use RoutineFetchOperation instead)
    - Persistent event hooks: addEventListener(), MutationObserver, IntersectionObserver
    - Navigation/lifecycle: window.close()

    FORMAT REQUIREMENT:
    The JavaScript code MUST be wrapped in an IIFE (Immediately Invoked Function Expression):
    - Format: `(function() { ... })()` or `(() => { ... })()`
    - Example: `(function() { return document.title; })()`
    - This is validated at the data model level.

    Args:
        type (Literal[RoutineOperationTypes.JS_EVALUATE]): The type of operation.
        js (str): JavaScript code in IIFE format: (function() { ... })() or (() => { ... })()
        timeout_seconds (float): Maximum execution time in seconds. Defaults to 5.
        session_storage_key (str | None): Optional session storage key to store the result.

    Returns:
        RoutineJsEvaluateOperation: The operation instance.

    Raises:
        ValueError: If the JavaScript code contains dangerous patterns.
    """
    type: Literal[RoutineOperationTypes.JS_EVALUATE] = RoutineOperationTypes.JS_EVALUATE
    js: str
    timeout_seconds: float = Field(
        default=5.0,
        description="Maximum execution time in seconds"
    )
    session_storage_key: str | None = Field(
        default=None,
        description="Optional session storage key to store the evaluation result"
    )

    # Dangerous patterns that are blocked
    DANGEROUS_PATTERNS: ClassVar[list[str]] = [
        # Dynamic code generation
        r'eval\s*\(',
        r'(?:^|[^a-zA-Z0-9_])Function\s*\(', 

        # Network / exfiltration
        r'(?<![a-zA-Z0-9_])fetch\s*\(',
        r'XMLHttpRequest',
        r'WebSocket',
        r'sendBeacon',

        # Persistent event hooks
        r'addEventListener\s*\(',
        r'MutationObserver',
        r'IntersectionObserver',

        # Navigation / lifecycle control
        r'window\.close\s*\(',
    ]

    @field_validator("js")
    @classmethod
    def validate_js_code(cls, v: str) -> str:
        """
        Validate that JavaScript code is in IIFE format and doesn't contain dangerous patterns.

        The code must be wrapped in an IIFE (Immediately Invoked Function Expression):
        `(function() { ... })()` or `(() => { ... })()`

        Raises:
            ValueError: If code is not in IIFE format, dangerous patterns are detected, or syntax errors are found.
        """
        if not v or not v.strip():
            raise ValueError("JavaScript code cannot be empty")

        # Basic syntax sanity check (balanced brackets, terminated strings)
        # Check this BEFORE IIFE format so we catch syntax errors with better messages
        try:
            assert_balanced_js_delimiters(v)
        except ValueError as e:
            raise ValueError(f"JavaScript syntax error: {e}")

        # Validate IIFE format using regex
        # Matches: (function() { ... })() or (function(...) { ... })() or (() => { ... })()
        # Also matches async variants: (async function() { ... })() or (async () => { ... })()
        # Optional semicolon at the end: })() or })();
        iife_pattern = r'^\s*\(\s*(async\s+)?(function\s*\([^)]*\)\s*\{|\(\)\s*=>\s*\{).+\}\s*\)\s*\(\s*\)\s*;?\s*$'
        if not re.match(pattern=iife_pattern, string=v, flags=re.DOTALL):
            raise ValueError(
                "JavaScript code must be wrapped in an IIFE (Immediately Invoked Function Expression). "
                "Use format: (function() { ... })() or (() => { ... })() or (async () => { ... })()"
            )

        # Check each dangerous pattern (case-sensitive to allow "function" keyword in IIFEs)
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern=pattern, string=v, flags=re.MULTILINE):
                raise ValueError(
                    f"JavaScript code contains blocked pattern: {pattern}. "
                )

        # Check for storage/meta/window placeholders and builtin parameters
        # These placeholders are not interpolated in JS code - JS should access them directly
        placeholder_pattern = r'\{\{([^}]*)\}\}'
        builtin_names = {bp.name for bp in BUILTIN_PARAMETERS}

        for match in re.finditer(pattern=placeholder_pattern, string=v):
            content = match.group(1).strip()

            # Check if it's a storage/meta/window placeholder (has colon prefix)
            if ":" in content:
                prefix = content.split(":", 1)[0].strip()
                if prefix in VALID_PLACEHOLDER_PREFIXES:
                    raise ValueError(
                        f"JavaScript code contains placeholder '{{{{{content}}}}}' which will not be interpolated. "
                        f"Access {prefix} directly in JavaScript code instead. "
                    )

            # Check if it's a builtin parameter
            if content in builtin_names:
                raise ValueError(
                    f"JavaScript code contains builtin placeholder '{{{{{content}}}}}' which will not be interpolated. "
                    f"Access builtin values directly in JavaScript code instead. "
                    f"For example, use 'crypto.randomUUID()' for UUID or 'Date.now()' for epoch milliseconds."
                )

        return v

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        """Validate timeout is within acceptable range."""
        if v <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        if v > 10.0:
            raise ValueError("timeout_seconds cannot exceed 10 seconds")
        return v

    def _execute_operation(self, routine_execution_context: RoutineExecutionContext) -> None:
        """Execute JavaScript code and optionally store result in session storage."""
        js_code = apply_params_to_str(self.js, routine_execution_context.parameters_dict)

        # Validate again after parameter interpolation to prevent injection attacks
        RoutineJsEvaluateOperation.validate_js_code(js_code)

        # Always wrap in outer IIFE to capture console logs and handle storage
        expression = generate_js_evaluate_wrapper_js(
            iife=js_code,
            session_storage_key=self.session_storage_key,
        )

        logger.debug(
            f"Executing JS evaluation: {expression} "
            f"timeout={self.timeout_seconds}s, session_storage_key={self.session_storage_key}"
        )

        eval_id = routine_execution_context.send_cmd(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": True,
                "timeout": int(self.timeout_seconds * 1000),
            },
            session_id=routine_execution_context.session_id,
        )

        logger.info(f"JS evaluation ID: {eval_id}")

        reply = routine_execution_context.recv_until(
            lambda m: m.get("id") == eval_id,
            deadline=time.time() + self.timeout_seconds + 1.0  # Extra 1s buffer
        )

        logger.info(f"JS evaluation reply: {reply}")

        # 1. Transport-level errors (rare)
        if "error" in reply:
            raise RuntimeError(f"JS evaluation failed: {reply['error']}")

        # 2. Engine-level JS errors (syntax, reference, etc)
        if reply.get("result", {}).get("exceptionDetails"):
            details = reply["result"]["exceptionDetails"]
            description = details["exception"]["description"]
            raise RuntimeError(f"JS evaluation failed: {description}")

        # 3. Normal result from wrapper
        result_value = reply["result"]["result"].get("value")

        logger.info(f"JS evaluation result: {result_value}")

        # Store console logs and errors in metadata
        if routine_execution_context.current_operation_metadata is not None and isinstance(result_value, dict):
            routine_execution_context.current_operation_metadata.details["console_logs"] = result_value.get("console_logs")
            routine_execution_context.current_operation_metadata.details["execution_error"] = result_value.get("execution_error")
            routine_execution_context.current_operation_metadata.details["storage_error"] = result_value.get("storage_error")

        # 4. Wrapper-level runtime errors
        if isinstance(result_value, dict):
            if result_value.get("execution_error"):
                raise RuntimeError(f"JS evaluation failed: {result_value['execution_error']}")
            if result_value.get("storage_error"):
                raise RuntimeError(f"JS evaluation failed: {result_value['storage_error']}")


# Routine operation unions ________________________________________________________________________

RoutineOperationUnion = Annotated[
    Union[
        RoutineNavigateOperation,
        RoutineSleepOperation,
        RoutineFetchOperation,
        RoutineReturnOperation,
        RoutineClickOperation,
        RoutineTypeOperation,
        RoutinePressOperation,
        RoutineGetCookiesOperation,
        RoutineWaitForUrlOperation,
        RoutineScrollOperation,
        RoutineReturnHTMLOperation,
        RoutineDownloadOperation,
        RoutineJsEvaluateOperation,
        # TODO: RoutineHoverOperation
        # TODO: RoutineWaitForSelectorOperation
        # TODO: RoutineWaitForTitleOperation
        # TODO: RoutineSetFilesOperation
        # TODO: RoutineReturnScreenshotOperation
    ],
    Field(discriminator="type"),
]
