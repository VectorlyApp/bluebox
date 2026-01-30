"""
tests/unit/test_js_utils.py

Tests for JavaScript utility functions.
"""

from bluebox.utils.js_utils import generate_js_evaluate_wrapper_js


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
