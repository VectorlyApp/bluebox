"""
bluebox/utils/code_execution_sandbox.py

Sandboxed Python code execution with blocklist-based security.

Allows most Python functionality while blocking dangerous operations
like file I/O, network access, subprocess execution, etc.
"""

import builtins as real_builtins
import io
import json
import sys
from typing import Any


# Blocked modules - dangerous for file/network/system access
BLOCKED_MODULES: frozenset[str] = frozenset({
    # File system access
    "os", "pathlib", "shutil", "tempfile", "fileinput", "glob", "fnmatch",
    # Network access
    "socket", "ssl", "http", "ftplib", "poplib", "imaplib",
    "smtplib", "telnetlib", "xmlrpc", "requests", "httpx", "aiohttp",
    # Process/system execution
    "subprocess", "multiprocessing", "threading", "concurrent",
    "_thread", "pty", "tty", "termios", "resource", "syslog",
    # Code manipulation
    "importlib", "pkgutil", "modulefinder", "runpy", "compileall",
    "dis", "inspect", "ast", "code", "codeop",
    # System internals
    "ctypes", "gc", "sys", "builtins", "_io", "io",
    # Pickle (code execution via deserialization)
    "pickle", "cPickle", "shelve", "marshal",
    # Database (could access external systems)
    "sqlite3", "dbm",
})

# Patterns to block in code before execution
BLOCKED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("open(", "File operations (open) are not allowed"),
    ("__import__", "Direct __import__ is not allowed"),
    ("exec(", "exec() is not allowed"),
    ("eval(", "eval() is not allowed"),
    ("compile(", "compile() is not allowed"),
    ("globals(", "globals() is not allowed"),
    ("locals(", "locals() is not allowed"),
    ("vars(", "vars() is not allowed"),
    ("getattr(", "getattr() is not allowed - use dict access instead"),
    ("setattr(", "setattr() is not allowed"),
    ("delattr(", "delattr() is not allowed"),
    ("__builtins__", "Accessing __builtins__ is not allowed"),
    ("__class__", "Accessing __class__ is not allowed"),
    ("__bases__", "Accessing __bases__ is not allowed"),
    ("__subclasses__", "Accessing __subclasses__ is not allowed"),
    ("__mro__", "Accessing __mro__ is not allowed"),
    ("__code__", "Accessing __code__ is not allowed"),
    ("__globals__", "Accessing __globals__ is not allowed"),
)

# Builtins to remove
BLOCKED_BUILTINS: tuple[str, ...] = (
    "open", "exec", "eval", "compile", "__import__",
    "globals", "locals", "vars", "getattr", "setattr",
    "delattr", "breakpoint", "input", "memoryview",
)


def _create_safe_import() -> Any:
    """Create a safe import function that blocks dangerous modules."""
    def safe_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        # Check if the module or any parent is blocked
        root_module = name.split(".")[0]
        if root_module in BLOCKED_MODULES:
            raise ImportError(f"Import of '{name}' is blocked for security reasons")
        return __import__(name, globals_, locals_, fromlist, level)

    return safe_import


def check_code_safety(code: str) -> str | None:
    """
    Check code for blocked patterns before execution.

    Args:
        code: Python source code to check

    Returns:
        Error message if unsafe pattern found, None if safe.
    """
    for pattern, error_msg in BLOCKED_PATTERNS:
        if pattern in code:
            return error_msg
    return None


def create_safe_builtins() -> dict[str, Any]:
    """
    Create a safe builtins dict with dangerous functions removed.

    Returns:
        Dict of safe builtins with __import__ replaced by safe version.
    """
    safe_builtins = {k: v for k, v in vars(real_builtins).items()}

    # Remove dangerous builtins
    for dangerous in BLOCKED_BUILTINS:
        safe_builtins.pop(dangerous, None)

    # Add safe import function
    safe_builtins["__import__"] = _create_safe_import()

    return safe_builtins


def execute_python_sandboxed(
    code: str,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Args:
        code: Python source code to execute
        extra_globals: Additional variables to inject into the execution namespace

    Returns:
        Dict with 'output' (stdout) and optionally 'error' if execution failed.
    """
    if not code:
        return {"error": "No code provided"}

    # Check for blocked patterns before execution
    safety_error = check_code_safety(code)
    if safety_error:
        return {"error": f"Blocked: {safety_error}"}

    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    try:
        # Build execution globals with safe builtins
        exec_globals: dict[str, Any] = {
            "__builtins__": create_safe_builtins(),
            "json": json,  # Always provide json for parsing
        }

        # Add any extra globals
        if extra_globals:
            exec_globals.update(extra_globals)

        exec(code, exec_globals)  # noqa: S102 - sandboxed with blocklist

        output = captured_output.getvalue()
        return {"output": output if output else "(no output)"}

    except Exception as e:
        return {
            "error": str(e),
            "output": captured_output.getvalue(),
        }

    finally:
        sys.stdout = old_stdout
