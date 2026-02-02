"""
bluebox/utils/code_execution_sandbox.py

Sandboxed Python code execution with Docker-based isolation.

Uses Docker containers for secure execution when available, with
blocklist-based fallback for development environments.

Security layers:
1. Static pattern analysis (blocks dangerous code patterns)
2. Docker container isolation (network disabled, read-only, resource-limited)
3. Blocklist-based builtins/imports (fallback when Docker unavailable)
"""

import builtins as real_builtins
import io
import json
import logging
import os
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

# Configuration
SANDBOX_MODE: str = os.getenv("BLUEBOX_SANDBOX_MODE", "auto")  # "docker", "blocklist", "auto"
DOCKER_IMAGE: str = os.getenv("BLUEBOX_SANDBOX_IMAGE", "python:3.11-slim")
DOCKER_TIMEOUT: int = int(os.getenv("BLUEBOX_SANDBOX_TIMEOUT", "30"))
DOCKER_MEMORY_LIMIT: str = os.getenv("BLUEBOX_SANDBOX_MEMORY", "128m")

# Cache for Docker availability check
_docker_available: bool | None = None


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


def _is_docker_available() -> bool:
    """
    Check if Docker is available and running.

    Returns:
        True if Docker is available and can run containers.
    """
    global _docker_available
    if _docker_available is not None:
        return _docker_available

    # Check if docker binary exists
    if not shutil.which("docker"):
        logger.debug("Docker binary not found in PATH")
        _docker_available = False
        return False

    # Check if Docker daemon is running
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        _docker_available = result.returncode == 0
        if not _docker_available:
            logger.debug("Docker daemon not running")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        _docker_available = False

    return _docker_available


def _execute_in_docker(
    code: str,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in an isolated Docker container.

    Security measures:
    - No network access (--network none)
    - Read-only filesystem (--read-only)
    - Limited memory and CPU
    - No privilege escalation
    - Runs as non-root user
    - Automatic cleanup (--rm)

    Args:
        code: Python source code to execute
        extra_globals: Variables to inject (serialized as JSON)

    Returns:
        Dict with 'output' and optionally 'error'.
    """
    # Build the wrapper script that deserializes globals and runs user code
    globals_json = json.dumps(extra_globals) if extra_globals else "{}"

    # Escape for shell - we use base64 to avoid quote issues
    import base64
    code_b64 = base64.b64encode(code.encode()).decode()
    globals_b64 = base64.b64encode(globals_json.encode()).decode()

    wrapper_script = f"""
import json, base64, sys
extra_globals = json.loads(base64.b64decode("{globals_b64}").decode())
code = base64.b64decode("{code_b64}").decode()
exec_globals = {{"json": json, **extra_globals}}
exec(code, exec_globals)
"""

    docker_cmd = [
        "docker", "run",
        "--rm",                          # Auto-cleanup
        "--network", "none",             # No network access
        "--read-only",                   # Read-only filesystem
        "--memory", DOCKER_MEMORY_LIMIT, # Memory limit
        "--cpus", "0.5",                 # CPU limit
        "--pids-limit", "50",            # Process limit
        "--security-opt", "no-new-privileges",  # No privilege escalation
        "--user", "nobody",              # Non-root user
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # Writable /tmp for Python
        DOCKER_IMAGE,
        "python", "-c", wrapper_script,
    ]

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT,
        )

        if result.returncode == 0:
            return {"output": result.stdout if result.stdout else "(no output)"}
        else:
            return {
                "error": result.stderr.strip() if result.stderr else f"Exit code {result.returncode}",
                "output": result.stdout,
            }

    except subprocess.TimeoutExpired:
        return {"error": f"Execution timed out after {DOCKER_TIMEOUT}s"}
    except Exception as e:
        return {"error": f"Docker execution failed: {e}"}


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


def _execute_blocklist_sandbox(
    code: str,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute code using blocklist-based sandboxing (fallback method).

    WARNING: This method is not secure against adversarial input.
    Use only when Docker is unavailable.
    """
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


def execute_python_sandboxed(
    code: str,
    extra_globals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Uses Docker container isolation when available for secure execution.
    Falls back to blocklist-based sandboxing when Docker is unavailable.

    Configure via environment variables:
    - BLUEBOX_SANDBOX_MODE: "docker", "blocklist", or "auto" (default: "auto")
    - BLUEBOX_SANDBOX_IMAGE: Docker image to use (default: "python:3.11-slim")
    - BLUEBOX_SANDBOX_TIMEOUT: Execution timeout in seconds (default: 30)
    - BLUEBOX_SANDBOX_MEMORY: Container memory limit (default: "128m")

    Args:
        code: Python source code to execute
        extra_globals: Additional variables to inject into the execution namespace

    Returns:
        Dict with 'output' (stdout) and optionally 'error' if execution failed.
    """
    if not code:
        return {"error": "No code provided"}

    # Always check for blocked patterns first (fast, catches obvious issues)
    safety_error = check_code_safety(code)
    if safety_error:
        return {"error": f"Blocked: {safety_error}"}

    # Determine execution mode
    use_docker = False
    if SANDBOX_MODE == "docker":
        use_docker = True
        if not _is_docker_available():
            return {"error": "Docker sandbox requested but Docker is not available"}
    elif SANDBOX_MODE == "auto":
        use_docker = _is_docker_available()
    # else: SANDBOX_MODE == "blocklist" -> use_docker stays False

    if use_docker:
        logger.debug("Executing code in Docker sandbox")
        return _execute_in_docker(code, extra_globals)
    else:
        if SANDBOX_MODE == "auto":
            logger.warning(
                "Docker unavailable, using blocklist sandbox. "
                "This is not secure against adversarial input."
            )
        return _execute_blocklist_sandbox(code, extra_globals)
