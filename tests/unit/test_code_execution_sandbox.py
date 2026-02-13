"""
tests/unit/test_code_execution_sandbox.py

Unit tests for the sandboxed Python code execution utility.
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bluebox.utils.code_execution_sandbox import (
    BLOCKED_MODULES,
    BLOCKED_PATTERNS,
    BLOCKED_BUILTINS,
    SENSITIVE_PATH_PREFIXES,
    check_code_safety,
    create_safe_builtins,
    execute_python_sandboxed,
    _is_docker_available,
    _execute_in_docker,
    _execute_blocklist_sandbox,
    _create_scoped_open,
)


class TestCheckCodeSafety:
    """Tests for the check_code_safety function."""

    def test_safe_code_returns_none(self) -> None:
        """Safe code should return None (no error)."""
        safe_codes = [
            "x = 1 + 2",
            "print('hello')",
            "for i in range(10): pass",
            "data = [1, 2, 3]",
            "result = sum([1, 2, 3])",
            "import collections",
            "from urllib.parse import urlparse",
        ]
        for code in safe_codes:
            assert check_code_safety(code) is None, f"Code should be safe: {code}"

    def test_blocks_open(self) -> None:
        """Should block open() calls."""
        assert check_code_safety("f = open('file.txt')") is not None
        assert "open" in check_code_safety("open('test')").lower()

    def test_blocks_exec(self) -> None:
        """Should block exec() calls."""
        assert check_code_safety("exec('print(1)')") is not None

    def test_blocks_eval(self) -> None:
        """Should block eval() calls."""
        assert check_code_safety("eval('1+1')") is not None

    def test_blocks_compile(self) -> None:
        """Should block compile() calls."""
        assert check_code_safety("compile('x=1', '', 'exec')") is not None

    def test_blocks_dunder_import(self) -> None:
        """Should block __import__ calls."""
        assert check_code_safety("__import__('os')") is not None

    def test_blocks_globals(self) -> None:
        """Should block globals() calls."""
        assert check_code_safety("g = globals()") is not None

    def test_blocks_locals(self) -> None:
        """Should block locals() calls."""
        assert check_code_safety("l = locals()") is not None

    def test_blocks_getattr(self) -> None:
        """Should block getattr() calls."""
        assert check_code_safety("getattr(obj, 'attr')") is not None

    def test_blocks_setattr(self) -> None:
        """Should block setattr() calls."""
        assert check_code_safety("setattr(obj, 'attr', 1)") is not None

    def test_blocks_dunder_builtins(self) -> None:
        """Should block __builtins__ access."""
        assert check_code_safety("x = __builtins__") is not None

    def test_blocks_dunder_class(self) -> None:
        """Should block __class__ access."""
        assert check_code_safety("x.__class__.__bases__") is not None

    def test_blocks_dunder_subclasses(self) -> None:
        """Should block __subclasses__ access."""
        assert check_code_safety("str.__subclasses__()") is not None

    def test_blocks_dunder_mro(self) -> None:
        """Should block __mro__ access."""
        assert check_code_safety("str.__mro__") is not None

    def test_blocks_dunder_globals(self) -> None:
        """Should block __globals__ access."""
        assert check_code_safety("func.__globals__") is not None

    def test_blocks_dunder_code(self) -> None:
        """Should block __code__ access."""
        assert check_code_safety("func.__code__") is not None


class TestCreateSafeBuiltins:
    """Tests for create_safe_builtins function."""

    def test_returns_dict(self) -> None:
        """Should return a dictionary."""
        builtins = create_safe_builtins()
        assert isinstance(builtins, dict)

    def test_has_common_builtins(self) -> None:
        """Should include common safe builtins."""
        builtins = create_safe_builtins()
        expected = [
            "print", "len", "str", "int", "float", "bool",
            "list", "dict", "set", "tuple", "range", "enumerate",
            "zip", "map", "filter", "sorted", "sum", "min", "max",
            "any", "all", "abs", "round", "isinstance", "type",
            "repr", "True", "False", "None",
        ]
        for name in expected:
            assert name in builtins, f"Missing builtin: {name}"

    def test_removes_dangerous_builtins(self) -> None:
        """Should not include dangerous builtins."""
        builtins = create_safe_builtins()
        for dangerous in BLOCKED_BUILTINS:
            if dangerous == "__import__":
                # __import__ is replaced, not removed
                continue
            assert dangerous not in builtins or builtins.get(dangerous) is None, \
                f"Dangerous builtin should be removed: {dangerous}"

    def test_has_safe_import(self) -> None:
        """Should have a custom __import__ function."""
        builtins = create_safe_builtins()
        assert "__import__" in builtins
        assert callable(builtins["__import__"])

    def test_safe_import_blocks_os(self) -> None:
        """Safe import should block os module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        with pytest.raises(ImportError, match="blocked"):
            safe_import("os")

    def test_safe_import_blocks_subprocess(self) -> None:
        """Safe import should block subprocess module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        with pytest.raises(ImportError, match="blocked"):
            safe_import("subprocess")

    def test_safe_import_blocks_socket(self) -> None:
        """Safe import should block socket module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        with pytest.raises(ImportError, match="blocked"):
            safe_import("socket")

    def test_safe_import_allows_collections(self) -> None:
        """Safe import should allow collections module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        module = safe_import("collections")
        assert module is not None

    def test_safe_import_allows_re(self) -> None:
        """Safe import should allow re module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        module = safe_import("re")
        assert module is not None

    def test_safe_import_allows_datetime(self) -> None:
        """Safe import should allow datetime module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        module = safe_import("datetime")
        assert module is not None

    def test_safe_import_allows_urllib_parse(self) -> None:
        """Safe import should allow urllib.parse module."""
        builtins = create_safe_builtins()
        safe_import = builtins["__import__"]
        module = safe_import("urllib.parse")
        assert module is not None


class TestExecutePythonSandboxed:
    """Tests for the execute_python_sandboxed function."""

    def test_empty_code_returns_error(self) -> None:
        """Empty code should return an error."""
        result = execute_python_sandboxed("")
        assert "error" in result

    def test_basic_print(self) -> None:
        """Basic print should work and capture output."""
        result = execute_python_sandboxed("print('hello world')")
        assert "error" not in result
        assert result["output"] == "hello world\n"

    def test_multiple_prints(self) -> None:
        """Multiple prints should be captured."""
        result = execute_python_sandboxed("print('a')\nprint('b')\nprint('c')")
        assert "error" not in result
        assert result["output"] == "a\nb\nc\n"

    def test_no_output_returns_placeholder(self) -> None:
        """Code with no output should return placeholder."""
        result = execute_python_sandboxed("x = 1 + 2")
        assert "error" not in result
        assert result["output"] == "(no output)"

    def test_computation(self) -> None:
        """Basic computation should work."""
        result = execute_python_sandboxed("print(sum([1, 2, 3, 4, 5]))")
        assert "error" not in result
        assert "15" in result["output"]

    def test_list_comprehension(self) -> None:
        """List comprehension should work."""
        result = execute_python_sandboxed("print([x**2 for x in range(5)])")
        assert "error" not in result
        assert "[0, 1, 4, 9, 16]" in result["output"]

    def test_dict_operations(self) -> None:
        """Dict operations should work."""
        code = """
d = {'a': 1, 'b': 2}
d['c'] = 3
print(sorted(d.keys()))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "['a', 'b', 'c']" in result["output"]

    def test_string_operations(self) -> None:
        """String operations should work."""
        code = """
s = 'hello world'
print(s.upper())
print(s.split())
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "HELLO WORLD" in result["output"]
        assert "['hello', 'world']" in result["output"]

    def test_json_available(self) -> None:
        """json module should be available."""
        code = """
import json
data = json.loads('{"key": "value"}')
print(data['key'])
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "value" in result["output"]

    def test_collections_import(self) -> None:
        """Should be able to import collections."""
        code = """
from collections import Counter
c = Counter(['a', 'b', 'a', 'c', 'a'])
print(c.most_common(1))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "('a', 3)" in result["output"]

    def test_re_import(self) -> None:
        """Should be able to import re."""
        code = """
import re
match = re.search(r'\\d+', 'abc123def')
print(match.group())
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "123" in result["output"]

    def test_datetime_import(self) -> None:
        """Should be able to import datetime."""
        code = """
from datetime import datetime
dt = datetime(2024, 1, 15)
print(dt.year)
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "2024" in result["output"]

    def test_urllib_parse_import(self) -> None:
        """Should be able to import urllib.parse."""
        code = """
from urllib.parse import urlparse, parse_qs
url = 'https://example.com/path?foo=bar&baz=qux'
parsed = urlparse(url)
print(parsed.netloc)
print(parse_qs(parsed.query))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "example.com" in result["output"]
        assert "foo" in result["output"]

    def test_extra_globals_available(self) -> None:
        """Extra globals should be available in execution."""
        result = execute_python_sandboxed(
            "print(len(entries))",
            extra_globals={"entries": [1, 2, 3, 4, 5]}
        )
        assert "error" not in result
        assert "5" in result["output"]

    def test_extra_globals_dict_access(self) -> None:
        """Should be able to work with dict entries in extra_globals."""
        entries = [
            {"url": "https://example.com", "status": 200},
            {"url": "https://test.com", "status": 404},
        ]
        code = """
for e in entries:
    print(f"{e['url']} -> {e['status']}")
"""
        result = execute_python_sandboxed(code, extra_globals={"entries": entries})
        assert "error" not in result
        assert "example.com" in result["output"]
        assert "200" in result["output"]
        assert "404" in result["output"]

    def test_exception_returns_error(self) -> None:
        """Exceptions should be caught and returned as errors."""
        result = execute_python_sandboxed("raise ValueError('test error')")
        assert "error" in result
        assert "test error" in result["error"]

    def test_syntax_error_returns_error(self) -> None:
        """Syntax errors should be caught and returned."""
        result = execute_python_sandboxed("def foo(")
        assert "error" in result

    def test_name_error_returns_error(self) -> None:
        """NameError should be caught and returned."""
        result = execute_python_sandboxed("print(undefined_variable)")
        assert "error" in result
        assert "undefined_variable" in result["error"]

    # Security tests - blocked patterns

    def test_blocks_open_pattern(self) -> None:
        """Should block open() in code."""
        result = execute_python_sandboxed("f = open('test.txt')")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_exec_pattern(self) -> None:
        """Should block exec() in code."""
        result = execute_python_sandboxed("exec('print(1)')")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_eval_pattern(self) -> None:
        """Should block eval() in code."""
        result = execute_python_sandboxed("eval('1+1')")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_dunder_import_pattern(self) -> None:
        """Should block __import__ in code."""
        result = execute_python_sandboxed("os = __import__('os')")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_getattr_pattern(self) -> None:
        """Should block getattr() in code."""
        result = execute_python_sandboxed("getattr(obj, 'attr')")
        assert "error" in result
        assert "Blocked" in result["error"]

    # Security tests - blocked imports (blocklist mode only)
    # Note: Docker mode allows imports but provides isolation via containerization.
    # These tests verify blocklist-mode import blocking.

    def test_blocks_os_import(self) -> None:
        """Should block os module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import os")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_subprocess_import(self) -> None:
        """Should block subprocess module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import subprocess")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_socket_import(self) -> None:
        """Should block socket module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import socket")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_pathlib_import(self) -> None:
        """Should block pathlib module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import pathlib")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_shutil_import(self) -> None:
        """Should block shutil module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import shutil")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_pickle_import(self) -> None:
        """Should block pickle module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import pickle")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_ctypes_import(self) -> None:
        """Should block ctypes module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import ctypes")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_multiprocessing_import(self) -> None:
        """Should block multiprocessing module import in blocklist mode."""
        result = _execute_blocklist_sandbox("import multiprocessing")
        assert "error" in result
        assert "blocked" in result["error"].lower()

    def test_blocks_requests_import(self) -> None:
        """Should block requests module import."""
        result = execute_python_sandboxed("import requests")
        assert "error" in result
        # requests might not be installed, so check for either blocked or not found
        assert "blocked" in result["error"].lower() or "No module" in result["error"]

    # Security tests - blocked dunder access

    def test_blocks_dunder_builtins_access(self) -> None:
        """Should block __builtins__ access."""
        result = execute_python_sandboxed("print(__builtins__)")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_dunder_class_access(self) -> None:
        """Should block __class__ access."""
        result = execute_python_sandboxed("print(''.__class__)")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_blocks_dunder_subclasses_exploit(self) -> None:
        """Should block __subclasses__ exploit attempt."""
        # Classic Python sandbox escape attempt
        code = "''.__class__.__bases__[0].__subclasses__()"
        result = execute_python_sandboxed(code)
        assert "error" in result
        assert "Blocked" in result["error"]

    # Complex data analysis tests

    def test_complex_data_analysis(self) -> None:
        """Should handle complex data analysis tasks."""
        entries = [
            {"url": "https://api.example.com/users", "status": 200, "method": "GET"},
            {"url": "https://api.example.com/posts", "status": 200, "method": "GET"},
            {"url": "https://api.example.com/users", "status": 201, "method": "POST"},
            {"url": "https://api.example.com/error", "status": 500, "method": "GET"},
        ]
        code = """
from collections import Counter

# Count status codes
status_counts = Counter(e['status'] for e in entries)
print(f"Status codes: {dict(status_counts)}")

# Count methods
method_counts = Counter(e['method'] for e in entries)
print(f"Methods: {dict(method_counts)}")

# Find errors
errors = [e for e in entries if e['status'] >= 400]
print(f"Errors: {len(errors)}")
"""
        result = execute_python_sandboxed(code, extra_globals={"entries": entries})
        assert "error" not in result
        assert "200" in result["output"]
        assert "GET" in result["output"]
        assert "Errors: 1" in result["output"]

    def test_json_parsing_in_entries(self) -> None:
        """Should be able to parse JSON strings in entry data."""
        entries = [
            {"response_body": '{"users": [{"name": "Alice"}, {"name": "Bob"}]}'},
        ]
        code = """
import json
for e in entries:
    data = json.loads(e['response_body'])
    for user in data['users']:
        print(user['name'])
"""
        result = execute_python_sandboxed(code, extra_globals={"entries": entries})
        assert "error" not in result
        assert "Alice" in result["output"]
        assert "Bob" in result["output"]

    def test_url_parsing_analysis(self) -> None:
        """Should be able to parse and analyze URLs."""
        entries = [
            {"url": "https://api.example.com/v1/users?page=1&limit=10"},
            {"url": "https://api.example.com/v1/posts?page=2&limit=20"},
        ]
        code = """
from urllib.parse import urlparse, parse_qs

for e in entries:
    parsed = urlparse(e['url'])
    params = parse_qs(parsed.query)
    print(f"Path: {parsed.path}, Params: {params}")
"""
        result = execute_python_sandboxed(code, extra_globals={"entries": entries})
        assert "error" not in result
        assert "/v1/users" in result["output"]
        assert "page" in result["output"]


class TestBlockedModulesCompleteness:
    """Tests to verify all expected dangerous modules are blocked."""

    @pytest.mark.parametrize("module", [
        "os", "pathlib", "shutil", "tempfile", "glob",
        "socket", "ssl", "http", "ftplib",
        "subprocess", "multiprocessing", "threading",
        "ctypes", "pickle", "marshal",
        "importlib", "inspect",
    ])
    def test_dangerous_module_in_blocklist(self, module: str) -> None:
        """Verify dangerous modules are in the blocklist."""
        assert module in BLOCKED_MODULES, f"{module} should be in BLOCKED_MODULES"

    @pytest.mark.parametrize("module", [
        "collections", "itertools", "functools",
        "re", "string", "textwrap",
        "datetime", "time", "calendar",
        "math", "statistics", "random",
        "json", "csv",
        "copy", "pprint",
        "urllib",  # urllib.parse is safe
    ])
    def test_safe_module_not_in_blocklist(self, module: str) -> None:
        """Verify safe modules are NOT in the blocklist."""
        assert module not in BLOCKED_MODULES, f"{module} should NOT be in BLOCKED_MODULES"


class TestBlockedPatternsCompleteness:
    """Tests to verify blocked patterns are correctly configured."""

    def test_blocked_patterns_is_tuple(self) -> None:
        """BLOCKED_PATTERNS should be a tuple for immutability."""
        assert isinstance(BLOCKED_PATTERNS, tuple)

    def test_blocked_patterns_have_messages(self) -> None:
        """Each blocked pattern should have an error message."""
        for pattern, message in BLOCKED_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(message, str)
            assert len(message) > 0

    @pytest.mark.parametrize("pattern", [
        "open(", "exec(", "eval(", "compile(",
        "__import__", "__builtins__", "__class__",
        "__subclasses__", "__globals__", "__code__",
        "getattr(", "setattr(", "delattr(",
        "globals(", "locals(", "vars(",
    ])
    def test_pattern_in_blocklist(self, pattern: str) -> None:
        """Verify dangerous patterns are in the blocklist."""
        patterns = [p for p, _ in BLOCKED_PATTERNS]
        assert pattern in patterns, f"{pattern} should be in BLOCKED_PATTERNS"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_output(self) -> None:
        """Should handle very long output."""
        code = "for i in range(1000): print(f'line {i}')"
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "line 0" in result["output"]
        assert "line 999" in result["output"]

    def test_unicode_output(self) -> None:
        """Should handle unicode output."""
        code = "print('Hello 世界 🌍')"
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "世界" in result["output"]
        assert "🌍" in result["output"]

    def test_multiline_string(self) -> None:
        """Should handle multiline strings."""
        code = '''
text = """
Line 1
Line 2
Line 3
"""
print(text.strip())
'''
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "Line 1" in result["output"]

    def test_nested_functions(self) -> None:
        """Should handle nested function definitions."""
        code = """
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
print(add_five(3))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "8" in result["output"]

    def test_class_definition(self) -> None:
        """Should handle class definitions."""
        code = """
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

c = Counter()
print(c.increment())
print(c.increment())
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "1" in result["output"]
        assert "2" in result["output"]

    def test_try_except(self) -> None:
        """Should handle try/except blocks."""
        code = """
try:
    x = 1 / 0
except ZeroDivisionError:
    print('caught division by zero')
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "caught division by zero" in result["output"]

    def test_generator_expression(self) -> None:
        """Should handle generator expressions."""
        code = """
gen = (x**2 for x in range(5))
print(list(gen))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "[0, 1, 4, 9, 16]" in result["output"]

    def test_lambda_functions(self) -> None:
        """Should handle lambda functions."""
        code = """
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(sorted(nums, key=lambda x: -x))
"""
        result = execute_python_sandboxed(code)
        assert "error" not in result
        assert "[9, 6, 5, 4, 3, 2, 1, 1]" in result["output"]

    def test_partial_output_on_error(self) -> None:
        """Should return partial output when error occurs mid-execution."""
        code = """
print('before error')
raise RuntimeError('mid error')
print('after error')
"""
        result = execute_python_sandboxed(code)
        assert "error" in result
        assert "mid error" in result["error"]
        assert "before error" in result["output"]


class TestDockerAvailability:
    """Tests for Docker availability detection."""

    def test_docker_not_available_when_binary_missing(self) -> None:
        """Should return False when docker binary is not in PATH."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        sandbox_module._docker_available = None  # Reset cache

        with patch("shutil.which", return_value=None):
            result = _is_docker_available()
            assert result is False

        sandbox_module._docker_available = None  # Reset for other tests

    def test_docker_not_available_when_daemon_not_running(self) -> None:
        """Should return False when docker daemon is not running."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        sandbox_module._docker_available = None  # Reset cache

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", return_value=mock_result):
                result = _is_docker_available()
                assert result is False

        sandbox_module._docker_available = None  # Reset for other tests

    def test_docker_available_when_daemon_running(self) -> None:
        """Should return True when docker daemon is running."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        sandbox_module._docker_available = None  # Reset cache

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("shutil.which", return_value="/usr/bin/docker"):
            with patch("subprocess.run", return_value=mock_result):
                result = _is_docker_available()
                assert result is True

        sandbox_module._docker_available = None  # Reset for other tests

    def test_docker_availability_cached(self) -> None:
        """Should cache docker availability check."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        sandbox_module._docker_available = True  # Set cache

        # Even with docker binary missing, cached value should be returned
        with patch("shutil.which", return_value=None):
            result = _is_docker_available()
            assert result is True

        sandbox_module._docker_available = None  # Reset for other tests


class TestDockerExecution:
    """Tests for Docker-based code execution."""

    def test_docker_execution_success(self) -> None:
        """Should execute code in Docker and return output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "hello from docker\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = _execute_in_docker("print('hello from docker')")
            assert "error" not in result
            assert result["output"] == "hello from docker\n"
            mock_run.assert_called_once()

    def test_docker_execution_with_extra_globals(self) -> None:
        """Should pass extra globals to Docker execution."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = _execute_in_docker(
                "print(len(data))",
                extra_globals={"data": [1, 2, 3, 4, 5]}
            )
            assert "error" not in result
            # Verify docker run was called with correct args
            call_args = mock_run.call_args
            assert "--network" in call_args[0][0]
            assert "none" in call_args[0][0]

    def test_docker_execution_error(self) -> None:
        """Should return error when Docker execution fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "NameError: name 'undefined' is not defined"

        with patch("subprocess.run", return_value=mock_result):
            result = _execute_in_docker("print(undefined)")
            assert "error" in result
            assert "NameError" in result["error"]

    def test_docker_execution_timeout(self) -> None:
        """Should handle Docker execution timeout and kill the container."""
        mock_kill_result = MagicMock()
        mock_kill_result.returncode = 0

        # First call (docker run) times out, second call (docker kill) succeeds
        with patch(
            "subprocess.run",
            side_effect=[subprocess.TimeoutExpired("docker", 30), mock_kill_result],
        ) as mock_run:
            result = _execute_in_docker("while True: pass")
            assert "error" in result
            assert "timed out" in result["error"]

            # Verify docker kill was called
            assert mock_run.call_count == 2
            kill_call = mock_run.call_args_list[1]
            assert "docker" in kill_call[0][0]
            assert "kill" in kill_call[0][0]

    def test_docker_security_flags(self) -> None:
        """Should include security flags in Docker command."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')")
            docker_cmd = mock_run.call_args[0][0]

            # Verify security flags
            assert "--network" in docker_cmd
            assert "none" in docker_cmd
            assert "--read-only" in docker_cmd
            assert "--memory" in docker_cmd
            assert "--user" in docker_cmd
            assert f"{os.getuid()}:{os.getgid()}" in docker_cmd
            assert "--security-opt" in docker_cmd
            assert "no-new-privileges" in docker_cmd


class TestSandboxModeSelection:
    """Tests for sandbox mode selection."""

    def test_blocklist_mode_uses_blocklist(self) -> None:
        """Should use blocklist sandbox when mode is 'blocklist'."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE

        try:
            sandbox_module.SANDBOX_MODE = "blocklist"
            # Should work without Docker
            result = execute_python_sandboxed("print('hello')")
            assert "error" not in result
            assert "hello" in result["output"]
        finally:
            sandbox_module.SANDBOX_MODE = original_mode

    def test_docker_mode_fails_without_docker(self) -> None:
        """Should fail when Docker mode requested but Docker unavailable."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE
        sandbox_module._docker_available = None

        try:
            sandbox_module.SANDBOX_MODE = "docker"
            with patch("shutil.which", return_value=None):
                result = execute_python_sandboxed("print('hello')")
                assert "error" in result
                assert "Docker" in result["error"]
        finally:
            sandbox_module.SANDBOX_MODE = original_mode
            sandbox_module._docker_available = None

    def test_auto_mode_falls_back_to_blocklist(self) -> None:
        """Should fall back to blocklist when Docker unavailable in auto mode."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE
        sandbox_module._docker_available = None

        try:
            sandbox_module.SANDBOX_MODE = "auto"
            with patch("shutil.which", return_value=None):
                result = execute_python_sandboxed("print('fallback')")
                assert "error" not in result
                assert "fallback" in result["output"]
        finally:
            sandbox_module.SANDBOX_MODE = original_mode
            sandbox_module._docker_available = None

    def test_auto_mode_uses_docker_when_available(self) -> None:
        """Should use Docker when available in auto mode."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE
        sandbox_module._docker_available = None

        mock_docker_info = MagicMock()
        mock_docker_info.returncode = 0

        mock_docker_run = MagicMock()
        mock_docker_run.returncode = 0
        mock_docker_run.stdout = "docker output\n"
        mock_docker_run.stderr = ""

        try:
            sandbox_module.SANDBOX_MODE = "auto"
            with patch("shutil.which", return_value="/usr/bin/docker"):
                with patch("subprocess.run", side_effect=[mock_docker_info, mock_docker_run]):
                    result = execute_python_sandboxed("print('test')")
                    assert "error" not in result
                    assert result["output"] == "docker output\n"
        finally:
            sandbox_module.SANDBOX_MODE = original_mode
            sandbox_module._docker_available = None


class TestBlocklistSandboxDirect:
    """Tests for the blocklist sandbox function directly."""

    def test_blocklist_sandbox_basic(self) -> None:
        """Should execute basic code."""
        result = _execute_blocklist_sandbox("print('direct')")
        assert "error" not in result
        assert "direct" in result["output"]

    def test_blocklist_sandbox_with_globals(self) -> None:
        """Should accept extra globals."""
        result = _execute_blocklist_sandbox(
            "print(sum(numbers))",
            extra_globals={"numbers": [1, 2, 3]}
        )
        assert "error" not in result
        assert "6" in result["output"]

    def test_blocklist_sandbox_blocks_os(self) -> None:
        """Should block os import even directly."""
        result = _execute_blocklist_sandbox("import os")
        assert "error" in result
        assert "blocked" in result["error"].lower()


class TestCheckCodeSafetyAllowFileIO:
    """Tests for check_code_safety with allow_file_io flag."""

    def test_blocks_open_by_default(self) -> None:
        """open() should be blocked when allow_file_io is False."""
        assert check_code_safety("f = open('file.txt')") is not None

    def test_allows_open_with_flag(self) -> None:
        """open() should be allowed when allow_file_io is True."""
        assert check_code_safety("f = open('file.txt')", allow_file_io=True) is None

    def test_still_blocks_exec_with_flag(self) -> None:
        """exec() should still be blocked even with allow_file_io."""
        assert check_code_safety("exec('x=1')", allow_file_io=True) is not None

    def test_still_blocks_eval_with_flag(self) -> None:
        """eval() should still be blocked even with allow_file_io."""
        assert check_code_safety("eval('1+1')", allow_file_io=True) is not None

    def test_still_blocks_dunder_import_with_flag(self) -> None:
        """__import__ should still be blocked even with allow_file_io."""
        assert check_code_safety("__import__('os')", allow_file_io=True) is not None

    def test_still_blocks_dunder_builtins_with_flag(self) -> None:
        """__builtins__ should still be blocked even with allow_file_io."""
        assert check_code_safety("x = __builtins__", allow_file_io=True) is not None


class TestScopedOpen:
    """Tests for the _create_scoped_open function."""

    def test_allows_file_within_dir(self, tmp_path: object) -> None:
        """Should allow opening files within work_dir."""
        work_dir = str(tmp_path)
        test_file = os.path.join(work_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("hello")

        scoped = _create_scoped_open(work_dir)
        with scoped("test.txt", "r") as f:
            assert f.read() == "hello"

    def test_allows_subdirectory_file(self, tmp_path: object) -> None:
        """Should allow opening files in subdirectories."""
        work_dir = str(tmp_path)
        sub_dir = os.path.join(work_dir, "sub")
        os.makedirs(sub_dir)
        test_file = os.path.join(sub_dir, "data.txt")
        with open(test_file, "w") as f:
            f.write("nested")

        scoped = _create_scoped_open(work_dir)
        with scoped("sub/data.txt", "r") as f:
            assert f.read() == "nested"

    def test_blocks_path_traversal(self, tmp_path: object) -> None:
        """Should block paths that escape work_dir via '..'."""
        work_dir = str(tmp_path / "workspace")
        os.makedirs(work_dir)

        scoped = _create_scoped_open(work_dir)
        with pytest.raises(PermissionError, match="outside the working directory"):
            scoped("../../../etc/passwd", "r")

    def test_blocks_absolute_path_outside(self, tmp_path: object) -> None:
        """Should block absolute paths outside work_dir."""
        work_dir = str(tmp_path / "workspace")
        os.makedirs(work_dir)

        scoped = _create_scoped_open(work_dir)
        with pytest.raises(PermissionError, match="outside the working directory"):
            scoped("/etc/passwd", "r")

    def test_allows_write(self, tmp_path: object) -> None:
        """Should allow writing files within work_dir."""
        work_dir = str(tmp_path)
        scoped = _create_scoped_open(work_dir)

        with scoped("output.csv", "w") as f:
            f.write("a,b,c\n1,2,3\n")

        with open(os.path.join(work_dir, "output.csv")) as f:
            assert f.read() == "a,b,c\n1,2,3\n"


class TestBlocklistSandboxWorkDir:
    """Tests for blocklist sandbox with work_dir parameter."""

    def test_can_write_file(self, tmp_path: object) -> None:
        """Should be able to write files to work_dir."""
        work_dir = str(tmp_path)
        code = """
with open("test_output.txt", "w") as f:
    f.write("hello from sandbox")
print("wrote file")
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "wrote file" in result["output"]

        output_file = os.path.join(work_dir, "test_output.txt")
        assert os.path.exists(output_file)
        with open(output_file) as f:
            assert f.read() == "hello from sandbox"

    def test_can_read_file(self, tmp_path: object) -> None:
        """Should be able to read existing files in work_dir."""
        work_dir = str(tmp_path)
        with open(os.path.join(work_dir, "input.txt"), "w") as f:
            f.write("existing data")

        code = """
with open("input.txt", "r") as f:
    content = f.read()
print(content)
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "existing data" in result["output"]

    def test_can_write_csv(self, tmp_path: object) -> None:
        """Should be able to write CSV files using the csv module."""
        work_dir = str(tmp_path)
        code = """
import csv
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "price"])
    writer.writerow(["Watch A", 5000])
    writer.writerow(["Watch B", 8000])
print("CSV written")
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "CSV written" in result["output"]

        csv_path = os.path.join(work_dir, "results.csv")
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            lines = f.readlines()
        assert len(lines) == 3
        assert "name,price" in lines[0]

    def test_can_write_json(self, tmp_path: object) -> None:
        """Should be able to write JSON files."""
        work_dir = str(tmp_path)
        code = """
data = [{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]
with open("output.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Wrote {len(data)} records")
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "Wrote 2 records" in result["output"]

        import json
        with open(os.path.join(work_dir, "output.json")) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1

    def test_can_write_to_subdirectory(self, tmp_path: object) -> None:
        """Should be able to create subdirectories and write files."""
        work_dir = str(tmp_path)
        code = """
Path("outputs").mkdir(exist_ok=True)
with open("outputs/result.txt", "w") as f:
    f.write("in subdir")
print("done")
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "done" in result["output"]
        assert os.path.exists(os.path.join(work_dir, "outputs", "result.txt"))

    def test_csv_module_available(self, tmp_path: object) -> None:
        """csv module should be pre-loaded when work_dir is set."""
        work_dir = str(tmp_path)
        code = "print(type(csv).__name__)"
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert "module" in result["output"]

    def test_path_available(self, tmp_path: object) -> None:
        """pathlib.Path should be pre-loaded when work_dir is set."""
        work_dir = str(tmp_path)
        code = "print(Path('.').resolve())"
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" not in result
        assert work_dir in result["output"]

    def test_blocks_path_traversal_write(self, tmp_path: object) -> None:
        """Should block writing outside work_dir."""
        work_dir = str(tmp_path / "workspace")
        os.makedirs(work_dir)
        code = """
with open("../../escape.txt", "w") as f:
    f.write("escaped")
"""
        result = _execute_blocklist_sandbox(code, work_dir=work_dir)
        assert "error" in result
        assert "outside" in result["error"].lower() or "denied" in result["error"].lower()

    def test_extra_globals_with_work_dir(self, tmp_path: object) -> None:
        """Extra globals should work alongside work_dir."""
        work_dir = str(tmp_path)
        data = [{"name": "A", "val": 1}, {"name": "B", "val": 2}]
        code = """
with open("from_globals.json", "w") as f:
    json.dump(routine_results, f)
print(f"Saved {len(routine_results)} results")
"""
        result = _execute_blocklist_sandbox(
            code,
            extra_globals={"routine_results": data},
            work_dir=work_dir,
        )
        assert "error" not in result
        assert "Saved 2 results" in result["output"]
        assert os.path.exists(os.path.join(work_dir, "from_globals.json"))

    def test_cwd_restored_after_execution(self, tmp_path: object) -> None:
        """Working directory should be restored after execution."""
        original_cwd = os.getcwd()
        work_dir = str(tmp_path)
        _execute_blocklist_sandbox("print('hi')", work_dir=work_dir)
        assert os.getcwd() == original_cwd

    def test_cwd_restored_after_error(self, tmp_path: object) -> None:
        """Working directory should be restored even after an error."""
        original_cwd = os.getcwd()
        work_dir = str(tmp_path)
        _execute_blocklist_sandbox("raise ValueError('boom')", work_dir=work_dir)
        assert os.getcwd() == original_cwd

    def test_without_work_dir_still_blocks_open(self) -> None:
        """Without work_dir, open() should still be blocked."""
        result = _execute_blocklist_sandbox("f = open('test.txt', 'w')")
        assert "error" in result


class TestExecutePythonSandboxedWorkDir:
    """Tests for execute_python_sandboxed with work_dir (top-level API)."""

    def test_allows_open_with_work_dir(self, tmp_path: object) -> None:
        """open() pattern should not be blocked when work_dir is set."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE

        try:
            sandbox_module.SANDBOX_MODE = "blocklist"
            work_dir = str(tmp_path)
            code = """
with open("test.txt", "w") as f:
    f.write("allowed")
print("ok")
"""
            result = execute_python_sandboxed(code, work_dir=work_dir)
            assert "error" not in result
            assert "ok" in result["output"]
            assert os.path.exists(os.path.join(work_dir, "test.txt"))
        finally:
            sandbox_module.SANDBOX_MODE = original_mode

    def test_blocks_open_without_work_dir(self) -> None:
        """open() pattern should be blocked when work_dir is not set."""
        result = execute_python_sandboxed("f = open('test.txt', 'w')")
        assert "error" in result
        assert "Blocked" in result["error"]

    def test_lambda_mode_rejects_work_dir(self) -> None:
        """Lambda mode should reject work_dir."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE

        try:
            sandbox_module.SANDBOX_MODE = "lambda"
            result = execute_python_sandboxed("print('hi')", work_dir="/tmp/test")
            assert "error" in result
            assert "not supported" in result["error"]
        finally:
            sandbox_module.SANDBOX_MODE = original_mode

    def test_docker_mode_passes_work_dir(self) -> None:
        """Docker mode should pass work_dir to _execute_in_docker."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE
        sandbox_module._docker_available = None

        mock_docker_info = MagicMock()
        mock_docker_info.returncode = 0
        mock_docker_run = MagicMock()
        mock_docker_run.returncode = 0
        mock_docker_run.stdout = "ok\n"
        mock_docker_run.stderr = ""

        try:
            sandbox_module.SANDBOX_MODE = "docker"
            with patch("shutil.which", return_value="/usr/bin/docker"):
                with patch("subprocess.run", side_effect=[mock_docker_info, mock_docker_run]) as mock_run:
                    result = execute_python_sandboxed(
                        "print('hi')",
                        work_dir="/tmp/workspace",
                    )
                    assert "error" not in result
                    # Verify -v mount flag was included
                    docker_cmd = mock_run.call_args_list[1][0][0]
                    assert "-v" in docker_cmd
                    assert any("/tmp/workspace:/data:rw" in arg for arg in docker_cmd)
        finally:
            sandbox_module.SANDBOX_MODE = original_mode
            sandbox_module._docker_available = None


class TestWorkDirValidation:
    """Tests for work_dir validation in execute_python_sandboxed."""

    @pytest.mark.parametrize("prefix", SENSITIVE_PATH_PREFIXES)
    def test_rejects_sensitive_prefix_exact(self, prefix: str) -> None:
        """Every entry in SENSITIVE_PATH_PREFIXES should be rejected as work_dir."""
        result = execute_python_sandboxed("print('hi')", work_dir=prefix)
        assert "error" in result
        assert "sensitive system path" in result["error"]

    @pytest.mark.parametrize("prefix", SENSITIVE_PATH_PREFIXES)
    def test_rejects_sensitive_prefix_subdir(self, prefix: str) -> None:
        """Subdirectories under sensitive prefixes should also be rejected."""
        result = execute_python_sandboxed("print('hi')", work_dir=f"{prefix}/subdir")
        assert "error" in result
        assert "sensitive system path" in result["error"]

    def test_normalizes_path_with_dotdot(self) -> None:
        """Paths with '..' that resolve to a sensitive prefix should be rejected."""
        result = execute_python_sandboxed("print('hi')", work_dir="/etc/../etc/nginx")
        assert "error" in result
        assert "sensitive system path" in result["error"]

    def test_allows_tmp(self, tmp_path: Path) -> None:
        """A normal temp directory should pass validation."""
        import bluebox.utils.code_execution_sandbox as sandbox_module
        original_mode = sandbox_module.SANDBOX_MODE
        try:
            sandbox_module.SANDBOX_MODE = "blocklist"
            result = execute_python_sandboxed("print('ok')", work_dir=str(tmp_path))
            assert "error" not in result
            assert "ok" in result["output"]
        finally:
            sandbox_module.SANDBOX_MODE = original_mode


class TestDockerExecutionWorkDir:
    """Tests for Docker execution with work_dir."""

    def test_docker_mounts_work_dir(self) -> None:
        """Should mount work_dir as /data:rw when provided."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')", work_dir="/tmp/workspace")
            docker_cmd = mock_run.call_args[0][0]
            assert "-v" in docker_cmd
            vol_idx = docker_cmd.index("-v")
            assert "/tmp/workspace:/data:rw" in docker_cmd[vol_idx + 1]

    def test_docker_sets_workdir_flag(self) -> None:
        """Should set -w /data when work_dir is provided."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')", work_dir="/tmp/workspace")
            docker_cmd = mock_run.call_args[0][0]
            assert "-w" in docker_cmd
            w_idx = docker_cmd.index("-w")
            assert docker_cmd[w_idx + 1] == "/data"

    def test_docker_no_mount_without_work_dir(self) -> None:
        """Should not mount any volume when work_dir is not set."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')")
            docker_cmd = mock_run.call_args[0][0]
            assert "-v" not in docker_cmd
            assert "-w" not in docker_cmd

    def test_docker_wrapper_includes_csv_and_path_with_work_dir(self) -> None:
        """Wrapper script should import csv and pathlib when work_dir is set."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')", work_dir="/tmp/workspace")
            wrapper_script = mock_run.call_args[1]["input"]
            assert "import json, csv, base64, sys, os" in wrapper_script
            assert "from pathlib import Path" in wrapper_script
            assert 'os.chdir("/data")' in wrapper_script

    def test_docker_wrapper_no_csv_without_work_dir(self) -> None:
        """Wrapper script should not import csv/pathlib without work_dir."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            _execute_in_docker("print('ok')")
            wrapper_script = mock_run.call_args[1]["input"]
            assert "import json, base64, sys" in wrapper_script
            assert "from pathlib import Path" not in wrapper_script
