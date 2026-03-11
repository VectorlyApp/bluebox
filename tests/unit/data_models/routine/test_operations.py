"""
tests/unit/test_operations.py

Tests for routine operations, with comprehensive coverage of JS evaluation operation validation.
"""

import pytest
from pydantic import ValidationError

from bluebox.data_models.routine.operation import (
    RoutineClickOperation,
    RoutineDownloadOperation,
    RoutineGetCookiesOperation,
    RoutineJsEvaluateOperation,
    RoutineNavigateOperation,
    RoutineOperationTypes,
    RoutinePressOperation,
    RoutineReturnHTMLOperation,
    RoutineReturnOperation,
    RoutineScrollOperation,
    RoutineSleepOperation,
    RoutineTypeOperation,
    RoutineWaitForUrlOperation,
)
from bluebox.data_models.routine.endpoint import Endpoint
from bluebox.data_models.ui_elements import MouseButton, ScrollBehavior, HTMLScope
from bluebox.utils.data_utils import apply_params_to_str


# ============================================================================
# Validation Tests for Non-JsEvaluate Operation Types
# ============================================================================


class TestRoutineNavigateOperationValidation:
    """Test validation for RoutineNavigateOperation."""

    def test_valid_navigate_operation(self) -> None:
        """Test that a valid navigate operation is accepted."""
        op = RoutineNavigateOperation(url="https://example.com")
        assert op.url == "https://example.com"
        assert op.type == RoutineOperationTypes.NAVIGATE
        assert op.sleep_after_navigation_seconds == 3.0

    def test_custom_sleep_after_navigation(self) -> None:
        """Test that custom sleep_after_navigation_seconds is accepted."""
        op = RoutineNavigateOperation(url="https://example.com", sleep_after_navigation_seconds=5.0)
        assert op.sleep_after_navigation_seconds == 5.0

    def test_zero_sleep_after_navigation(self) -> None:
        """Test that zero sleep_after_navigation_seconds is accepted."""
        op = RoutineNavigateOperation(url="https://example.com", sleep_after_navigation_seconds=0.0)
        assert op.sleep_after_navigation_seconds == 0.0

    def test_url_with_placeholder(self) -> None:
        """Test that URL with placeholder template is accepted."""
        op = RoutineNavigateOperation(url="https://example.com/{{path}}")
        assert op.url == "https://example.com/{{path}}"

    def test_missing_url_rejected(self) -> None:
        """Test that missing url is rejected."""
        with pytest.raises(ValidationError):
            RoutineNavigateOperation()


class TestRoutineSleepOperationValidation:
    """Test validation for RoutineSleepOperation."""

    def test_valid_sleep_operation(self) -> None:
        """Test that a valid sleep operation is accepted."""
        op = RoutineSleepOperation(timeout_seconds=2.0)
        assert op.timeout_seconds == 2.0
        assert op.type == RoutineOperationTypes.SLEEP

    def test_fractional_sleep(self) -> None:
        """Test that fractional seconds are accepted."""
        op = RoutineSleepOperation(timeout_seconds=0.5)
        assert op.timeout_seconds == 0.5

    def test_missing_timeout_rejected(self) -> None:
        """Test that missing timeout_seconds is rejected."""
        with pytest.raises(ValidationError):
            RoutineSleepOperation()


class TestRoutineClickOperationValidation:
    """Test validation for RoutineClickOperation."""

    def test_valid_click_operation(self) -> None:
        """Test that a valid click operation is accepted."""
        op = RoutineClickOperation(selector="#submit-btn")
        assert op.selector == "#submit-btn"
        assert op.type == RoutineOperationTypes.CLICK
        assert op.button == MouseButton.LEFT
        assert op.click_count == 1
        assert op.timeout_ms == 20_000
        assert op.ensure_visible is True

    def test_right_click(self) -> None:
        """Test that right button click is accepted."""
        op = RoutineClickOperation(selector=".menu", button=MouseButton.RIGHT)
        assert op.button == MouseButton.RIGHT

    def test_double_click(self) -> None:
        """Test that double click is accepted."""
        op = RoutineClickOperation(selector=".item", click_count=2)
        assert op.click_count == 2

    def test_custom_timeout(self) -> None:
        """Test that custom timeout_ms is accepted."""
        op = RoutineClickOperation(selector=".btn", timeout_ms=5000)
        assert op.timeout_ms == 5000

    def test_ensure_visible_false(self) -> None:
        """Test that ensure_visible=False is accepted."""
        op = RoutineClickOperation(selector=".btn", ensure_visible=False)
        assert op.ensure_visible is False

    def test_missing_selector_rejected(self) -> None:
        """Test that missing selector is rejected."""
        with pytest.raises(ValidationError):
            RoutineClickOperation()

    def test_invalid_button_rejected(self) -> None:
        """Test that invalid mouse button is rejected."""
        with pytest.raises(ValidationError):
            RoutineClickOperation(selector=".btn", button="double")


class TestRoutineTypeOperationValidation:
    """Test validation for RoutineTypeOperation."""

    def test_valid_type_operation(self) -> None:
        """Test that a valid type operation is accepted."""
        op = RoutineTypeOperation(selector="#search", text="hello world")
        assert op.selector == "#search"
        assert op.text == "hello world"
        assert op.type == RoutineOperationTypes.INPUT_TEXT
        assert op.clear is False
        assert op.timeout_ms == 20_000

    def test_clear_before_typing(self) -> None:
        """Test that clear=True is accepted."""
        op = RoutineTypeOperation(selector="#input", text="new text", clear=True)
        assert op.clear is True

    def test_text_with_placeholder(self) -> None:
        """Test that text with placeholder is accepted."""
        op = RoutineTypeOperation(selector="#input", text="{{username}}")
        assert op.text == "{{username}}"

    def test_missing_selector_rejected(self) -> None:
        """Test that missing selector is rejected."""
        with pytest.raises(ValidationError):
            RoutineTypeOperation(text="hello")

    def test_missing_text_rejected(self) -> None:
        """Test that missing text is rejected."""
        with pytest.raises(ValidationError):
            RoutineTypeOperation(selector="#input")


class TestRoutinePressOperationValidation:
    """Test validation for RoutinePressOperation."""

    def test_valid_press_enter(self) -> None:
        """Test that pressing Enter is accepted."""
        op = RoutinePressOperation(key="enter")
        assert op.key == "enter"
        assert op.type == RoutineOperationTypes.PRESS

    def test_valid_press_tab(self) -> None:
        """Test that pressing Tab is accepted."""
        op = RoutinePressOperation(key="tab")
        assert op.key == "tab"

    def test_valid_press_escape(self) -> None:
        """Test that pressing Escape is accepted."""
        op = RoutinePressOperation(key="escape")
        assert op.key == "escape"

    def test_missing_key_rejected(self) -> None:
        """Test that missing key is rejected."""
        with pytest.raises(ValidationError):
            RoutinePressOperation()


class TestRoutineWaitForUrlOperationValidation:
    """Test validation for RoutineWaitForUrlOperation."""

    def test_valid_wait_for_url(self) -> None:
        """Test that a valid wait_for_url operation is accepted."""
        op = RoutineWaitForUrlOperation(url_regex=".*results.*")
        assert op.url_regex == ".*results.*"
        assert op.type == RoutineOperationTypes.WAIT_FOR_URL
        assert op.timeout_ms == 20_000

    def test_custom_timeout(self) -> None:
        """Test that custom timeout_ms is accepted."""
        op = RoutineWaitForUrlOperation(url_regex=".*", timeout_ms=10_000)
        assert op.timeout_ms == 10_000

    def test_missing_url_regex_rejected(self) -> None:
        """Test that missing url_regex is rejected."""
        with pytest.raises(ValidationError):
            RoutineWaitForUrlOperation()


class TestRoutineScrollOperationValidation:
    """Test validation for RoutineScrollOperation."""

    def test_valid_window_scroll(self) -> None:
        """Test that a valid window scroll operation is accepted."""
        op = RoutineScrollOperation(y=500)
        assert op.y == 500
        assert op.type == RoutineOperationTypes.SCROLL
        assert op.selector is None
        assert op.behavior == ScrollBehavior.AUTO

    def test_valid_element_scroll(self) -> None:
        """Test that a valid element scroll operation is accepted."""
        op = RoutineScrollOperation(selector=".content", delta_y=300)
        assert op.selector == ".content"
        assert op.delta_y == 300

    def test_smooth_scroll(self) -> None:
        """Test that smooth scroll behavior is accepted."""
        op = RoutineScrollOperation(y=100, behavior=ScrollBehavior.SMOOTH)
        assert op.behavior == ScrollBehavior.SMOOTH

    def test_invalid_behavior_rejected(self) -> None:
        """Test that invalid scroll behavior is rejected."""
        with pytest.raises(ValidationError):
            RoutineScrollOperation(behavior="instant")


class TestRoutineGetCookiesOperationValidation:
    """Test validation for RoutineGetCookiesOperation."""

    def test_valid_get_cookies(self) -> None:
        """Test that a valid get_cookies operation is accepted."""
        op = RoutineGetCookiesOperation(session_storage_key="cookies")
        assert op.session_storage_key == "cookies"
        assert op.type == RoutineOperationTypes.GET_COOKIES
        assert op.domain_filter == "*"

    def test_custom_domain_filter(self) -> None:
        """Test that a custom domain filter is accepted."""
        op = RoutineGetCookiesOperation(session_storage_key="cookies", domain_filter="example.com")
        assert op.domain_filter == "example.com"

    def test_empty_domain_filter_rejected(self) -> None:
        """Test that an empty domain filter is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineGetCookiesOperation(session_storage_key="cookies", domain_filter="")
        errors = exc_info.value.errors()
        assert any("cannot be empty" in str(e.get("msg", "")) for e in errors)

    def test_whitespace_domain_filter_rejected(self) -> None:
        """Test that a whitespace-only domain filter is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineGetCookiesOperation(session_storage_key="cookies", domain_filter="   ")
        errors = exc_info.value.errors()
        assert any("cannot be empty" in str(e.get("msg", "")) for e in errors)

    def test_domain_filter_stripped(self) -> None:
        """Test that domain filter whitespace is stripped."""
        op = RoutineGetCookiesOperation(session_storage_key="cookies", domain_filter="  example.com  ")
        assert op.domain_filter == "example.com"

    def test_missing_session_storage_key_rejected(self) -> None:
        """Test that missing session_storage_key is rejected."""
        with pytest.raises(ValidationError):
            RoutineGetCookiesOperation()


class TestRoutineReturnOperationValidation:
    """Test validation for RoutineReturnOperation."""

    def test_valid_return_operation(self) -> None:
        """Test that a valid return operation is accepted."""
        op = RoutineReturnOperation(session_storage_key="result")
        assert op.session_storage_key == "result"
        assert op.type == RoutineOperationTypes.RETURN

    def test_missing_session_storage_key_rejected(self) -> None:
        """Test that missing session_storage_key is rejected."""
        with pytest.raises(ValidationError):
            RoutineReturnOperation()


class TestRoutineReturnHTMLOperationValidation:
    """Test validation for RoutineReturnHTMLOperation."""

    def test_valid_page_scope(self) -> None:
        """Test that page scope HTML return is accepted."""
        op = RoutineReturnHTMLOperation()
        assert op.scope == HTMLScope.PAGE
        assert op.type == RoutineOperationTypes.RETURN_HTML
        assert op.selector is None

    def test_valid_element_scope(self) -> None:
        """Test that element scope with selector is accepted."""
        op = RoutineReturnHTMLOperation(scope=HTMLScope.ELEMENT, selector=".main-content")
        assert op.scope == HTMLScope.ELEMENT
        assert op.selector == ".main-content"

    def test_invalid_scope_rejected(self) -> None:
        """Test that invalid scope is rejected."""
        with pytest.raises(ValidationError):
            RoutineReturnHTMLOperation(scope="fragment")


class TestRoutineDownloadOperationValidation:
    """Test validation for RoutineDownloadOperation."""

    def test_valid_download_operation(self) -> None:
        """Test that a valid download operation is accepted."""
        endpoint = Endpoint(url="https://example.com/report.pdf", method="GET")
        op = RoutineDownloadOperation(endpoint=endpoint, filename="report.pdf")
        assert op.filename == "report.pdf"
        assert op.type == RoutineOperationTypes.DOWNLOAD

    def test_missing_filename_rejected(self) -> None:
        """Test that missing filename is rejected."""
        endpoint = Endpoint(url="https://example.com/file.pdf", method="GET")
        with pytest.raises(ValidationError):
            RoutineDownloadOperation(endpoint=endpoint)

    def test_missing_endpoint_rejected(self) -> None:
        """Test that missing endpoint is rejected."""
        with pytest.raises(ValidationError):
            RoutineDownloadOperation(filename="file.pdf")


class TestRoutineJsEvaluateOperationValidation:
    """Test validation for RoutineJsEvaluateOperation."""

    # ============================================================================
    # IIFE Format Validation Tests
    # ============================================================================

    def test_valid_iife_function_format(self) -> None:
        """Test that valid IIFE with function() syntax is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return document.title; })()"
        )
        assert operation.js == "(function() { return document.title; })()"
        assert operation.type == RoutineOperationTypes.JS_EVALUATE

    def test_valid_iife_arrow_function_format(self) -> None:
        """Test that valid IIFE with arrow function syntax is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(() => { return document.title; })()"
        )
        assert operation.js == "(() => { return document.title; })()"

    def test_valid_iife_with_whitespace(self) -> None:
        """Test that IIFE with extra whitespace is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="  (function() { return 'test'; })()  "
        )
        assert operation.js == "  (function() { return 'test'; })()  "

    def test_valid_iife_with_parameters(self) -> None:
        """Test that IIFE with function parameters is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(function(x, y) { return x + y; })()"
        )
        assert operation.js == "(function(x, y) { return x + y; })()"

    def test_valid_iife_multiline(self) -> None:
        """Test that multiline IIFE is accepted."""
        js_code = """(function() {
            const title = document.title;
            return title.toUpperCase();
        })()"""
        operation = RoutineJsEvaluateOperation(js=js_code)
        assert operation.js == js_code

    def test_valid_iife_with_semicolon(self) -> None:
        """Test that IIFE with semicolon at the end is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return document.title; })();"
        )
        assert operation.js == "(function() { return document.title; })();"

    def test_valid_iife_arrow_function_with_semicolon(self) -> None:
        """Test that arrow function IIFE with semicolon at the end is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(() => { return document.title; })();"
        )
        assert operation.js == "(() => { return document.title; })();"

    def test_valid_async_iife_function_format(self) -> None:
        """Test that valid async IIFE with function() syntax is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async function() { return await Promise.resolve('test'); })()"
        )
        assert operation.js == "(async function() { return await Promise.resolve('test'); })()"
        assert operation.type == RoutineOperationTypes.JS_EVALUATE

    def test_valid_async_iife_arrow_function_format(self) -> None:
        """Test that valid async IIFE with arrow function syntax is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async () => { return await Promise.resolve('test'); })()"
        )
        assert operation.js == "(async () => { return await Promise.resolve('test'); })()"

    def test_valid_async_iife_with_whitespace(self) -> None:
        """Test that async IIFE with extra whitespace is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="  (async function() { return 'test'; })()  "
        )
        assert operation.js == "  (async function() { return 'test'; })()  "

    def test_valid_async_iife_with_parameters(self) -> None:
        """Test that async IIFE with function parameters is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async function(x, y) { return await Promise.resolve(x + y); })()"
        )
        assert operation.js == "(async function(x, y) { return await Promise.resolve(x + y); })()"

    def test_valid_async_iife_multiline(self) -> None:
        """Test that multiline async IIFE is accepted."""
        js_code = """(async function() {
            const result = await Promise.resolve('test');
            return result.toUpperCase();
        })()"""
        operation = RoutineJsEvaluateOperation(js=js_code)
        assert operation.js == js_code

    def test_valid_async_iife_with_semicolon(self) -> None:
        """Test that async IIFE with semicolon at the end is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async function() { return await Promise.resolve('test'); })();"
        )
        assert operation.js == "(async function() { return await Promise.resolve('test'); })();"

    def test_valid_async_iife_arrow_function_with_semicolon(self) -> None:
        """Test that async arrow function IIFE with semicolon at the end is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async () => { return await Promise.resolve('test'); })();"
        )
        assert operation.js == "(async () => { return await Promise.resolve('test'); })();"

    def test_valid_async_iife_with_await(self) -> None:
        """Test that async IIFE with await is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(async () => { const data = await new Promise(r => setTimeout(() => r('done'), 100)); return data; })()"
        )
        assert operation.js is not None

    def test_valid_async_iife_with_polling(self) -> None:
        """Test that async IIFE with polling pattern is accepted."""
        js_code = """(async () => {
            const maxWait = 5000;
            const start = Date.now();
            while (Date.now() - start < maxWait) {
                const items = document.querySelectorAll('.item');
                if (items.length > 0) {
                    return items.length;
                }
                await new Promise(r => setTimeout(r, 100));
            }
            return 0;
        })()"""
        operation = RoutineJsEvaluateOperation(js=js_code)
        assert operation.js == js_code

    def test_invalid_not_iife(self) -> None:
        """Test that non-IIFE code is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(js="return document.title;")
        
        errors = exc_info.value.errors()
        assert any("IIFE" in str(e.get("msg", "")) for e in errors)

    def test_invalid_missing_invocation(self) -> None:
        """Test that function without invocation is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(js="(function() { return 'test'; })")
        
        errors = exc_info.value.errors()
        assert any("IIFE" in str(e.get("msg", "")) for e in errors)

    def test_invalid_not_wrapped(self) -> None:
        """Test that unwrapped code is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(js="function() { return 'test'; }()")
        
        errors = exc_info.value.errors()
        assert any("IIFE" in str(e.get("msg", "")) for e in errors)

    # ============================================================================
    # Empty Code Validation Tests
    # ============================================================================

    def test_empty_js_code(self) -> None:
        """Test that empty JS code is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(js="")
        
        errors = exc_info.value.errors()
        assert any("cannot be empty" in str(e.get("msg", "")) for e in errors)

    def test_whitespace_only_js_code(self) -> None:
        """Test that whitespace-only JS code is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(js="   \n\t  ")
        
        errors = exc_info.value.errors()
        assert any("cannot be empty" in str(e.get("msg", "")) for e in errors)

    # ============================================================================
    # Dangerous Patterns Validation Tests
    # ============================================================================

    def test_blocked_eval(self) -> None:
        """Test that eval() is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return eval('1+1'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("eval" in str(e.get("msg", "")) for e in errors)

    def test_blocked_function_constructor(self) -> None:
        """Test that Function constructor is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return new Function('return 1+1')(); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("Function" in str(e.get("msg", "")) for e in errors)

    def test_blocked_fetch(self) -> None:
        """Test that fetch() is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return fetch('https://example.com'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("fetch" in str(e.get("msg", "")) for e in errors)

    def test_blocked_fetch_in_async_iife(self) -> None:
        """Test that fetch() is blocked even in async IIFE."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(async function() { return await fetch('https://example.com'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("fetch" in str(e.get("msg", "")) for e in errors)

    def test_allowed_prefetch(self) -> None:
        """Test that prefetch() is not blocked by the fetch pattern."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { var link = document.createElement('link'); link.rel = 'prefetch'; return prefetch('/next-page'); })()"
        )
        assert operation.js is not None

    def test_allowed_refetch(self) -> None:
        """Test that refetch() is not blocked by the fetch pattern."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return refetch(); })()"
        )
        assert operation.js is not None

    def test_blocked_eval_in_async_iife(self) -> None:
        """Test that eval() is blocked even in async IIFE."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(async () => { return await eval('1+1'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("eval" in str(e.get("msg", "")) for e in errors)

    def test_blocked_xmlhttprequest(self) -> None:
        """Test that XMLHttpRequest is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { const xhr = new XMLHttpRequest(); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("XMLHttpRequest" in str(e.get("msg", "")) for e in errors)

    def test_blocked_websocket(self) -> None:
        """Test that WebSocket is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { const ws = new WebSocket('ws://example.com'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("WebSocket" in str(e.get("msg", "")) for e in errors)

    def test_blocked_sendbeacon(self) -> None:
        """Test that sendBeacon is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { navigator.sendBeacon('/analytics', 'data'); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("sendBeacon" in str(e.get("msg", "")) for e in errors)

    def test_blocked_addeventlistener(self) -> None:
        """Test that addEventListener is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { document.addEventListener('click', () => {}); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("addEventListener" in str(e.get("msg", "")) for e in errors)

    def test_blocked_mutation_observer(self) -> None:
        """Test that MutationObserver is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { new MutationObserver(() => {}); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("MutationObserver" in str(e.get("msg", "")) for e in errors)

    def test_blocked_intersection_observer(self) -> None:
        """Test that IntersectionObserver is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { new IntersectionObserver(() => {}); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("IntersectionObserver" in str(e.get("msg", "")) for e in errors)

    def test_blocked_window_close(self) -> None:
        """Test that window.close() is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { window.close(); })()"
            )
        
        errors = exc_info.value.errors()
        assert any("window\\.close" in str(e.get("msg", "")) or "window.close" in str(e.get("msg", "")) for e in errors)

    def test_allowed_promise(self) -> None:
        """Test that Promise is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return new Promise(resolve => resolve(42)); })()"
        )
        assert operation.js is not None

    def test_allowed_settimeout(self) -> None:
        """Test that setTimeout is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { setTimeout(() => {}, 100); })()"
        )
        assert operation.js is not None

    def test_allowed_settimeout_in_async_iife(self) -> None:
        """Test that setTimeout is allowed in async IIFE."""
        operation = RoutineJsEvaluateOperation(
            js="(async () => { await new Promise(r => setTimeout(r, 100)); return 'done'; })()"
        )
        assert operation.js is not None

    def test_allowed_dom_manipulation(self) -> None:
        """Test that DOM manipulation is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { document.getElementById('test'); })()"
        )
        assert operation.js is not None

    def test_allowed_loops(self) -> None:
        """Test that loops are allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { for (let i = 0; i < 10; i++) {} })()"
        )
        assert operation.js is not None

    # ============================================================================
    # Storage/Meta/Window Placeholder Validation Tests
    # ============================================================================

    def test_blocked_session_storage_placeholder(self) -> None:
        """Test that {{sessionStorage:...}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{sessionStorage:api_key}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("sessionStorage" in str(e.get("msg", "")) for e in errors)
        assert any("will not be interpolated" in str(e.get("msg", "")) for e in errors)

    def test_blocked_local_storage_placeholder(self) -> None:
        """Test that {{localStorage:...}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{localStorage:token}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("localStorage" in str(e.get("msg", "")) for e in errors)

    def test_blocked_cookie_placeholder(self) -> None:
        """Test that {{cookie:...}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{cookie:session_id}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("cookie" in str(e.get("msg", "")) for e in errors)

    def test_blocked_meta_placeholder(self) -> None:
        """Test that {{meta:...}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{meta:title}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("meta" in str(e.get("msg", "")) for e in errors)

    def test_blocked_window_property_placeholder(self) -> None:
        """Test that {{windowProperty:...}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{windowProperty:someProp}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("windowProperty" in str(e.get("msg", "")) for e in errors)

    def test_allowed_direct_session_storage_access(self) -> None:
        """Test that direct sessionStorage access is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return sessionStorage.getItem('key'); })()"
        )
        assert operation.js is not None

    def test_allowed_direct_local_storage_access(self) -> None:
        """Test that direct localStorage access is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return localStorage.getItem('token'); })()"
        )
        assert operation.js is not None

    # ============================================================================
    # Builtin Parameter Validation Tests
    # ============================================================================

    def test_blocked_uuid_placeholder(self) -> None:
        """Test that {{uuid}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{uuid}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("uuid" in str(e.get("msg", "")) for e in errors)
        assert any("will not be interpolated" in str(e.get("msg", "")) for e in errors)

    def test_blocked_epoch_milliseconds_placeholder(self) -> None:
        """Test that {{epoch_milliseconds}} placeholder is blocked."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{epoch_milliseconds}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("epoch_milliseconds" in str(e.get("msg", "")) for e in errors)

    def test_allowed_crypto_randomuuid(self) -> None:
        """Test that crypto.randomUUID() is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return crypto.randomUUID(); })()"
        )
        assert operation.js is not None

    def test_allowed_date_now(self) -> None:
        """Test that Date.now() is allowed."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return Date.now(); })()"
        )
        assert operation.js is not None

    # ============================================================================
    # Syntax Error Validation Tests
    # ============================================================================

    def test_unbalanced_parentheses(self) -> None:
        """Test that unbalanced parentheses are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return (1 + 2; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("syntax error" in str(e.get("msg", "")) for e in errors)

    def test_unbalanced_braces(self) -> None:
        """Test that unbalanced braces are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return 'test'; { })"
            )
        
        errors = exc_info.value.errors()
        assert any("syntax error" in str(e.get("msg", "")) for e in errors)

    def test_unterminated_string(self) -> None:
        """Test that unterminated strings are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return 'unterminated; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("syntax error" in str(e.get("msg", "")) for e in errors)

    def test_unterminated_template_literal(self) -> None:
        """Test that unterminated template literals are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return `unterminated; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("syntax error" in str(e.get("msg", "")) for e in errors)

    # ============================================================================
    # Timeout Validation Tests
    # ============================================================================

    def test_valid_timeout(self) -> None:
        """Test that valid timeout is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return 'test'; })()",
            timeout_seconds=3.5
        )
        assert operation.timeout_seconds == 3.5

    def test_default_timeout(self) -> None:
        """Test that default timeout is 5.0."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return 'test'; })()"
        )
        assert operation.timeout_seconds == 5.0

    def test_zero_timeout_rejected(self) -> None:
        """Test that zero timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return 'test'; })()",
                timeout_seconds=0.0
            )
        
        errors = exc_info.value.errors()
        assert any("greater than 0" in str(e.get("msg", "")) for e in errors)

    def test_negative_timeout_rejected(self) -> None:
        """Test that negative timeout is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return 'test'; })()",
                timeout_seconds=-1.0
            )
        
        errors = exc_info.value.errors()
        assert any("greater than 0" in str(e.get("msg", "")) for e in errors)

    def test_timeout_exceeds_max_rejected(self) -> None:
        """Test that timeout exceeding 10 seconds is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return 'test'; })()",
                timeout_seconds=10.1
            )
        
        errors = exc_info.value.errors()
        assert any("exceed" in str(e.get("msg", "")) for e in errors)
        assert any("10" in str(e.get("msg", "")) for e in errors)

    def test_max_timeout_accepted(self) -> None:
        """Test that maximum timeout (10.0) is accepted."""
        operation = RoutineJsEvaluateOperation(
            js="(function() { return 'test'; })()",
            timeout_seconds=10.0
        )
        assert operation.timeout_seconds == 10.0

    # ============================================================================
    # Post-Interpolation Validation Tests
    # ============================================================================

    def test_post_interpolation_validation_blocks_injection(self) -> None:
        """Test that validation after interpolation prevents injection attacks."""
        # Create valid operation with parameter placeholder
        operation = RoutineJsEvaluateOperation(
            js='(function() { return "{{param}}"; })()'
        )
        
        # Simulate parameter interpolation that would inject eval
        interpolated_js = apply_params_to_str(
            operation.js,
            {"param": 'test"; eval("evil"); "'}
        )
        
        # Validation should catch the injected eval
        with pytest.raises(ValueError) as exc_info:
            RoutineJsEvaluateOperation.validate_js_code(interpolated_js)
        
        # Should detect the eval injection
        assert "eval" in str(exc_info.value).lower()

    def test_post_interpolation_validation_blocks_dangerous_pattern(self) -> None:
        """Test that validation after interpolation blocks dangerous patterns."""
        # Create valid operation with parameter placeholder
        operation = RoutineJsEvaluateOperation(
            js='(function() { return "{{code}}"; })()'
        )
        
        # Simulate parameter interpolation that would inject fetch
        interpolated_js = apply_params_to_str(
            operation.js,
            {"code": 'test"; fetch("evil"); "'}
        )
        
        # Validation should catch the injected fetch
        with pytest.raises(ValueError) as exc_info:
            RoutineJsEvaluateOperation.validate_js_code(interpolated_js)
        
        # Should detect the fetch injection
        assert "fetch" in str(exc_info.value).lower()

    def test_post_interpolation_validation_allows_safe_interpolation(self) -> None:
        """Test that safe parameter interpolation still passes validation."""
        # Create valid operation with parameter placeholder
        operation = RoutineJsEvaluateOperation(
            js='(function() { return "{{name}}"; })()'
        )

        # Simulate safe parameter interpolation
        interpolated_js = apply_params_to_str(
            operation.js,
            {"name": "John"}
        )

        # Validation should pass for safe interpolated code
        validated_js = RoutineJsEvaluateOperation.validate_js_code(interpolated_js)
        assert validated_js == '(function() { return "John"; })()'

    # ============================================================================
    # Complex Real-World Examples
    # ============================================================================

    def test_allowed_string_containing_on_prefix(self) -> None:
        """Test that string literals containing 'on' followed by word chars and '=' are not blocked.

        The pattern r'on\\w+\\s*=' is meant to block event handlers like onclick=, onload=, etc.
        But it should NOT block occurrences inside string literals (e.g., 'lots_json_length=').
        """
        js_code = (
            "(function () { var r = window.chrComponents && window.chrComponents.lots; "
            "if (!(r && r.data && Array.isArray(r.data.lots) && r.data.lots.length)) { "
            "console.log('no_lots'); return null; } var a = r.data.lots; var o = []; "
            "for (var i = 0; i < a.length; i++) { var l = a[i]; o.push({ "
            "lot_number: l.lot_id_txt || null, artist: l.title_primary_txt || null, "
            "title: l.title_secondary_txt || null, estimate_text: l.estimate_txt || null, "
            "estimate_low: l.estimate_low || null, estimate_high: l.estimate_high || null, "
            "price_realised_text: l.price_realised_txt || null, price_realised: l.price_realised || null, "
            "url: l.url || null }); } var s = JSON.stringify(o); "
            "console.log('lots_json_length=' + s.length); "
            "sessionStorage.setItem('lots_json', s); return null; })()"
        )
        operation = RoutineJsEvaluateOperation(js=js_code)
        assert operation.js == js_code

    def test_complex_valid_code(self) -> None:
        """Test complex but valid JS code."""
        js_code = """(function() {
            const elements = document.querySelectorAll('.item');
            const results = [];
            for (let i = 0; i < elements.length; i++) {
                results.push({
                    text: elements[i].textContent,
                    href: elements[i].href
                });
            }
            return results;
        })()"""
        
        operation = RoutineJsEvaluateOperation(js=js_code)
        assert operation.js == js_code

    def test_multiple_placeholders_blocked(self) -> None:
        """Test that multiple invalid placeholders are all detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return \"{{sessionStorage:key}}\" and \"{{uuid}}\"; })()"
            )
        
        errors = exc_info.value.errors()
        # Should detect at least one of the invalid placeholders
        error_msg = str(errors[0].get("msg", ""))
        assert ("sessionStorage" in error_msg or "uuid" in error_msg)

    def test_placeholder_in_comment_blocked(self) -> None:
        """Test that placeholders in comments are still detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { /* {{sessionStorage:key}} */ return 'test'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("sessionStorage" in str(e.get("msg", "")) for e in errors)

    def test_placeholder_in_string_blocked(self) -> None:
        """Test that placeholders in string literals are detected."""
        with pytest.raises(ValidationError) as exc_info:
            RoutineJsEvaluateOperation(
                js="(function() { return '{{uuid}}'; })()"
            )
        
        errors = exc_info.value.errors()
        assert any("uuid" in str(e.get("msg", "")) for e in errors)

