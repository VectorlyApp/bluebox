"""
tests/unit/test_js_utils.py

Tests for JavaScript utility functions.
"""

import re
from textwrap import dedent

from bluebox.utils.js_utils import (
    DANGEROUS_JS_PATTERNS,
    IIFE_PATTERN,
    generate_js_evaluate_wrapper_js,
    validate_js,
)


class TestValidateJs:
    """Tests for validate_js function."""

    # --- Valid code ---

    def test_valid_function_iife(self) -> None:
        """Valid (function() { ... })() passes with no errors."""
        errors = validate_js("(function() { return 42; })()")
        assert errors == []

    def test_valid_arrow_iife(self) -> None:
        """Valid (() => { ... })() passes with no errors."""
        errors = validate_js("(() => { return 42; })()")
        assert errors == []

    def test_valid_async_function_iife(self) -> None:
        """Valid (async function() { ... })() passes."""
        errors = validate_js("(async function() { return 42; })()")
        assert errors == []

    def test_valid_async_arrow_iife(self) -> None:
        """Valid (async () => { ... })() passes."""
        errors = validate_js("(async () => { return 42; })()")
        assert errors == []

    def test_valid_iife_with_trailing_semicolon(self) -> None:
        """Trailing semicolon is allowed."""
        errors = validate_js("(function() { return 1; })();")
        assert errors == []

    def test_valid_multiline_iife(self) -> None:
        """Multi-line IIFE with proper formatting passes cleanly."""
        code = dedent("""
            (function() {
                const x = 1;
                const y = 2;
                return x + y;
            })()
        """)
        errors = validate_js(code)
        assert errors == []

    # --- Empty / missing code ---

    def test_empty_string(self) -> None:
        errors = validate_js("")
        assert any("cannot be empty" in e for e in errors)

    def test_whitespace_only(self) -> None:
        errors = validate_js("   \n\t  ")
        assert any("cannot be empty" in e for e in errors)

    def test_none_code(self) -> None:
        errors = validate_js(None)  # type: ignore[arg-type]
        assert any("cannot be empty" in e for e in errors)

    # --- IIFE format errors ---

    def test_bare_expression_rejected(self) -> None:
        """Code not wrapped in IIFE is rejected."""
        errors = validate_js("document.title")
        assert any("IIFE" in e for e in errors)

    def test_bare_function_declaration_rejected(self) -> None:
        errors = validate_js("function foo() { return 1; }")
        assert any("IIFE" in e for e in errors)

    def test_non_invoked_function_rejected(self) -> None:
        """Function expression without invocation is rejected."""
        errors = validate_js("(function() { return 1; })")
        assert any("IIFE" in e for e in errors)

    # --- Dangerous pattern detection ---

    def test_eval_blocked(self) -> None:
        errors = validate_js("(function() { eval('1+1'); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_function_constructor_blocked(self) -> None:
        errors = validate_js("(function() { new Function('return 1')(); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_fetch_blocked(self) -> None:
        errors = validate_js("(function() { fetch('/api'); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_xmlhttprequest_blocked(self) -> None:
        errors = validate_js("(function() { new XMLHttpRequest(); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_websocket_blocked(self) -> None:
        errors = validate_js("(function() { new WebSocket('ws://x'); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_sendbeacon_blocked(self) -> None:
        errors = validate_js("(function() { navigator.sendBeacon('/log', ''); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_addeventlistener_blocked(self) -> None:
        errors = validate_js("(function() { window.addEventListener('click', () => {}); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_mutationobserver_blocked(self) -> None:
        errors = validate_js("(function() { new MutationObserver(() => {}); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_intersectionobserver_blocked(self) -> None:
        errors = validate_js("(function() { new IntersectionObserver(() => {}); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_window_close_blocked(self) -> None:
        errors = validate_js("(function() { window.close(); })()")
        assert any("Blocked pattern" in e for e in errors)

    def test_multiple_blocked_patterns(self) -> None:
        """Multiple violations produce multiple errors."""
        code = "(function() { eval('x'); fetch('/y'); })()"
        errors = validate_js(code)
        blocked = [e for e in errors if "Blocked pattern" in e]
        assert len(blocked) >= 2

    # --- Safe patterns that look similar but should pass ---

    def test_fetch_as_variable_name_allowed(self) -> None:
        """A variable named 'prefetch' should not trigger the fetch block."""
        errors = validate_js("(function() { var prefetch = 1; return prefetch; })()")
        assert not any("Blocked pattern" in e for e in errors)

    def test_function_keyword_in_iife_allowed(self) -> None:
        """The 'function' keyword in the IIFE wrapper itself is fine."""
        errors = validate_js("(function() { return 1; })()")
        assert not any("Function" in e for e in errors)

    # --- Readability warnings ---

    def test_long_line_produces_warning(self) -> None:
        """A line > 200 chars in the IIFE body triggers a WARNING."""
        long_var = "x" * 250
        code = f"(function() {{ var {long_var} = 1; return 1; }})()"
        errors = validate_js(code)
        warnings = [e for e in errors if e.startswith("WARNING:")]
        assert len(warnings) == 1
        assert "200" in warnings[0]

    def test_long_line_warning_is_soft(self) -> None:
        """WARNING doesn't appear alongside hard errors for otherwise valid code."""
        long_var = "x" * 250
        code = f"(function() {{ var {long_var} = 1; return 1; }})()"
        errors = validate_js(code)
        hard_errors = [e for e in errors if not e.startswith("WARNING:")]
        assert hard_errors == []

    def test_short_lines_no_warning(self) -> None:
        """Well-formatted code produces no warnings."""
        code = "(function() {\n  var x = 1;\n  return x;\n})()"
        errors = validate_js(code)
        assert not any(e.startswith("WARNING:") for e in errors)

    def test_exactly_200_chars_no_warning(self) -> None:
        """A line of exactly 200 chars should not trigger the warning."""
        # pad to exactly 200 chars inside the braces
        inner = "x" * 190
        code = f"(function() {{ var {inner} = 1; return 1; }})()"
        # verify the longest body line is <= 200
        body = code[code.find("{") + 1:code.rfind("}")]
        max_len = max(len(line) for line in body.split("\n"))
        if max_len <= 200:
            errors = validate_js(code)
            assert not any(e.startswith("WARNING:") for e in errors)

    # --- Constants sanity checks ---

    def test_dangerous_patterns_not_empty(self) -> None:
        assert len(DANGEROUS_JS_PATTERNS) > 0

    def test_iife_pattern_matches_basic(self) -> None:
        assert re.match(IIFE_PATTERN, "(function() { return 1; })()", re.DOTALL)
        assert re.match(IIFE_PATTERN, "(() => { return 1; })()", re.DOTALL)
        assert not re.match(IIFE_PATTERN, "function() { return 1; }", re.DOTALL)


class TestGenerateJsEvaluateWrapperJs:
    """Tests for generate_js_evaluate_wrapper_js function."""

    def test_basic_iife_wrapping(self) -> None:
        """Test that a basic IIFE is wrapped correctly."""
        iife = "(() => { return 42; })()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should be an async IIFE
        assert result.startswith("(async () => {")
        assert result.endswith("})()")

        # Should contain the original IIFE
        assert iife in result

        # Should have console log capture setup
        assert "__consoleLogs = []" in result
        assert "__originalConsoleLog = console.log" in result

        # Should have error handling
        assert "__executionError = null" in result
        assert "__storageError = null" in result

        # Should return the expected structure
        assert "result: __result" in result
        assert "console_logs: __consoleLogs" in result
        assert "execution_error: __executionError" in result

    def test_console_log_override(self) -> None:
        """Test that console.log is properly overridden to capture logs."""
        iife = "(() => { console.log('test'); return 1; })()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should override console.log
        assert "console.log = (...args) => {" in result

        # Should capture timestamp and message
        assert "timestamp: Date.now()" in result
        assert "message: args.map" in result

        # Should call original console.log
        assert "__originalConsoleLog.apply(console, args)" in result

        # Should restore original console.log in finally block
        assert "console.log = __originalConsoleLog" in result

    def test_console_log_serialization(self) -> None:
        """Test that console.log arguments are properly serialized."""
        iife = "(() => {})()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should handle object serialization
        assert "typeof a === 'object' ? JSON.stringify(a) : String(a)" in result

        # Should join multiple arguments with space
        assert ".join(' ')" in result

    def test_with_session_storage_key(self) -> None:
        """Test that session storage code is included when key is provided."""
        iife = "(() => { return { data: 'test' }; })()"
        result = generate_js_evaluate_wrapper_js(iife, session_storage_key="my_key")

        # Should include session storage code
        assert "sessionStorage.setItem" in result
        assert '"my_key"' in result
        assert "JSON.stringify(__result)" in result

        # Should handle storage errors
        assert "SessionStorage Error" in result
        assert "storage_error: __storageError" in result

    def test_without_session_storage_key(self) -> None:
        """Test that session storage code is not included when key is None."""
        iife = "(() => { return 42; })()"
        result = generate_js_evaluate_wrapper_js(iife, session_storage_key=None)

        # Should not include session storage setItem call
        assert "sessionStorage.setItem" not in result

    def test_async_iife_handling(self) -> None:
        """Test that async IIFEs are handled correctly with await."""
        iife = "(async () => { return await Promise.resolve('async result'); })()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should wrap with Promise.resolve to handle both sync and async
        assert "await Promise.resolve(" + iife + ")" in result

    def test_execution_error_capture(self) -> None:
        """Test that execution errors are captured."""
        iife = "(() => { throw new Error('test error'); })()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should have try-catch block
        assert "try {" in result
        assert "catch(e) {" in result

        # Should capture error as string
        assert "__executionError = String(e)" in result

    def test_finally_block_restores_console(self) -> None:
        """Test that console.log is restored even if execution fails."""
        iife = "(() => {})()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should have finally block
        assert "finally {" in result

        # Should restore console.log in finally
        assert "console.log = __originalConsoleLog" in result

    def test_return_structure(self) -> None:
        """Test that the returned object has the expected structure."""
        iife = "(() => { return 'test'; })()"
        result = generate_js_evaluate_wrapper_js(iife)

        # Should return object with all expected fields
        assert "return {" in result
        assert "result: __result" in result
        assert "console_logs: __consoleLogs" in result
        assert "storage_error: __storageError" in result
        assert "execution_error: __executionError" in result

    def test_session_storage_key_with_special_characters(self) -> None:
        """Test that session storage key with special characters is properly escaped."""
        iife = "(() => { return 1; })()"
        result = generate_js_evaluate_wrapper_js(iife, session_storage_key='key"with"quotes')

        # Key should be JSON-escaped
        assert 'key\\"with\\"quotes' in result

    def test_multiline_iife(self) -> None:
        """Test that multiline IIFEs are handled correctly."""
        iife = """(() => {
            const a = 1;
            const b = 2;
            return a + b;
        })()"""
        result = generate_js_evaluate_wrapper_js(iife)

        # Should contain the multiline IIFE
        assert iife in result

        # Should still have proper structure
        assert result.startswith("(async () => {")
        assert "console_logs: __consoleLogs" in result

    def test_storage_only_when_result_defined(self) -> None:
        """Test that storage only happens when result is not undefined."""
        iife = "(() => {})()"
        result = generate_js_evaluate_wrapper_js(iife, session_storage_key="test_key")

        # Should check if result is undefined before storing
        assert "if (__result !== undefined)" in result

    def test_trailing_semicolon_stripped(self) -> None:
        """Test that trailing semicolon is stripped from IIFE to avoid syntax error."""
        # IIFE with trailing semicolon - would cause "missing ) after argument list"
        # if embedded as: Promise.resolve((function(){...})();)
        iife_with_semicolon = "(function() { return 42; })();"
        result = generate_js_evaluate_wrapper_js(iife_with_semicolon)

        # Should NOT contain the trailing semicolon in the embedded IIFE
        # The wrapper embeds as: await Promise.resolve(IIFE)
        # So we should see ")())" not "()();)"
        assert "await Promise.resolve((function() { return 42; })())" in result
        assert "();)" not in result

    def test_trailing_semicolon_with_whitespace_stripped(self) -> None:
        """Test that trailing semicolon with whitespace is stripped."""
        iife_with_semicolon_and_whitespace = "(function() { return 42; })();   \n  "
        result = generate_js_evaluate_wrapper_js(iife_with_semicolon_and_whitespace)

        # Should strip both whitespace and semicolon
        assert "await Promise.resolve((function() { return 42; })())" in result
        assert "();)" not in result

    def test_leading_whitespace_stripped(self) -> None:
        """Test that leading whitespace is stripped from IIFE."""
        iife_with_leading_whitespace = "   \n  (function() { return 42; })()"
        result = generate_js_evaluate_wrapper_js(iife_with_leading_whitespace)

        # Should embed cleanly without leading whitespace
        assert "await Promise.resolve((function() { return 42; })())" in result

    def test_no_semicolon_unchanged(self) -> None:
        """Test that IIFE without trailing semicolon is unchanged."""
        iife_no_semicolon = "(function() { return 42; })()"
        result = generate_js_evaluate_wrapper_js(iife_no_semicolon)

        # Should embed as-is
        assert "await Promise.resolve((function() { return 42; })())" in result
