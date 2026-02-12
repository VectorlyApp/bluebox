"""
bluebox/utils/code_execution_sandbox.py

Sandboxed Python code execution with multiple isolation backends.

Supported backends (via BLUEBOX_SANDBOX_MODE env var):
- "lambda": AWS Lambda isolation (recommended for ECS/cloud deployments)
- "docker": Docker container isolation (network disabled, read-only, resource-limited)
- "blocklist": Python-level blocklist sandboxing (fallback, not secure against adversarial input)
- "auto": Automatically selects lambda (if registered) > docker (if available) > blocklist

Security layers:
1. Static pattern analysis (blocks dangerous code patterns)
2. Backend-specific isolation (Lambda microVM, Docker container, or blocklist)
"""

import base64
import builtins as real_builtins
import csv
import io
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

from bluebox.config import Config

logger = logging.getLogger(__name__)

# Sandbox configuration from Config
SANDBOX_MODE: str = Config.SANDBOX_MODE
DOCKER_IMAGE: str = Config.SANDBOX_IMAGE
DOCKER_TIMEOUT: int = Config.SANDBOX_TIMEOUT
DOCKER_MEMORY_LIMIT: str = Config.SANDBOX_MEMORY

# cache for Docker availability check
_docker_available: bool | None = None

# AWS Lambda executor function (registered by cloud deployments at startup)
_lambda_executor_fn: Any | None = None


def register_lambda_executor(fn: Any) -> None:
    """
    Register the Lambda executor function for BLUEBOX_SANDBOX_MODE=lambda.

    This should be called at application startup in cloud deployments.
    The function must have signature: (code: str, extra_globals: dict | None) -> dict

    Args:
        fn: The Lambda executor function to register.
    """
    global _lambda_executor_fn
    _lambda_executor_fn = fn
    logger.debug("Lambda executor registered")


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

# Sensitive system paths that must never be used as work_dir
SENSITIVE_PATH_PREFIXES: tuple[str, ...] = (
    "/etc", "/var", "/usr", "/bin", "/sbin",
    "/boot", "/proc", "/sys", "/dev",
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
    work_dir: str | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in an isolated Docker container.

    Security measures:
    - No network access (--network none)
    - Read-only filesystem (--read-only), except work_dir if mounted
    - Limited memory and CPU
    - No privilege escalation
    - Runs as non-root user
    - Automatic cleanup (--rm)

    Args:
        code: Python source code to execute.
        extra_globals: Variables to inject (serialized as JSON).
        work_dir: Host directory to mount as /data with read-write access.

    Returns:
        Dict with 'output' and optionally 'error'.
    """
    # Build the wrapper script that deserializes globals and runs user code
    try:
        globals_json = json.dumps(extra_globals) if extra_globals else "{}"
    except (TypeError, ValueError) as e:
        return {"error": f"extra_globals must contain only JSON-serializable values: {e}"}

    # Escape for shell - we use base64 to avoid quote issues
    code_b64 = base64.b64encode(code.encode()).decode()
    globals_b64 = base64.b64encode(globals_json.encode()).decode()

    if work_dir:
        wrapper_script = f"""
import json, csv, base64, sys, os
from pathlib import Path
os.chdir("/data")
extra_globals = json.loads(base64.b64decode("{globals_b64}").decode())
extra_globals["csv"] = csv
extra_globals["Path"] = Path
code = base64.b64decode("{code_b64}").decode()
exec_globals = {{"json": json, "csv": csv, "Path": Path, **extra_globals}}
exec(code, exec_globals)
"""
    else:
        wrapper_script = f"""
import json, base64, sys
extra_globals = json.loads(base64.b64decode("{globals_b64}").decode())
code = base64.b64decode("{code_b64}").decode()
exec_globals = {{"json": json, **extra_globals}}
exec(code, exec_globals)
"""

    # Generate unique container name for cleanup on timeout
    container_name = f"bluebox-sandbox-{os.getpid()}-{uuid.uuid4().hex[:8]}"

    docker_cmd = [
        "docker", "run",
        "--name", container_name,        # Named for cleanup on timeout
        "--rm",                          # Auto-cleanup
        "--network", "none",             # No network access
        "--read-only",                   # Read-only filesystem
        "--memory", DOCKER_MEMORY_LIMIT, # Memory limit
        "--cpus", "0.5",                 # CPU limit
        "--pids-limit", "50",            # Process limit
        "--security-opt", "no-new-privileges",  # No privilege escalation
        "--user", "nobody",              # Non-root user
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # Writable /tmp for Python
        "-i",                            # Accept stdin
    ]

    # Mount work_dir as /data with read-write access
    if work_dir:
        abs_work_dir = os.path.abspath(work_dir)
        docker_cmd.extend(["-v", f"{abs_work_dir}:/data:rw", "-w", "/data"])

    docker_cmd.extend([DOCKER_IMAGE, "python", "-"])

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=DOCKER_TIMEOUT,
            input=wrapper_script,        # Pass script via stdin
        )

        if result.returncode == 0:
            return {"output": result.stdout if result.stdout else "(no output)"}
        else:
            return {
                "error": result.stderr.strip() if result.stderr else f"Exit code {result.returncode}",
                "output": result.stdout,
            }

    except subprocess.TimeoutExpired:
        # Kill the container to prevent resource leaks
        subprocess.run(
            ["docker", "kill", container_name],
            capture_output=True,
            timeout=5,
        )
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


def check_code_safety(code: str, allow_file_io: bool = False) -> str | None:
    """
    Check code for blocked patterns before execution.

    Args:
        code: Python source code to check.
        allow_file_io: When True, skip the open() pattern check (used with work_dir).

    Returns:
        Error message if unsafe pattern found, None if safe.
    """
    for pattern, error_msg in BLOCKED_PATTERNS:
        if allow_file_io and pattern == "open(":
            continue
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


def _create_scoped_open(work_dir: str) -> Any:
    """
    Create an open() function that restricts file access to work_dir.

    All paths are resolved relative to work_dir. Paths that escape
    work_dir (e.g. via '..') are rejected.

    Args:
        work_dir: The directory to scope file access to.
    """
    abs_work_dir = os.path.abspath(work_dir)

    def scoped_open(
        file: str,
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        resolved = os.path.abspath(os.path.join(abs_work_dir, file))
        if not resolved.startswith(abs_work_dir + os.sep) and resolved != abs_work_dir:
            raise PermissionError(f"Access denied: path '{file}' is outside the working directory")
        return open(resolved, mode, *args, **kwargs)  # noqa: SIM115

    return scoped_open


def _execute_blocklist_sandbox(
    code: str,
    extra_globals: dict[str, Any] | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """
    Execute code using blocklist-based sandboxing (fallback method).

    WARNING: This method is not secure against adversarial input.
    Use only when Docker is unavailable.

    Args:
        code: Python source code to execute.
        extra_globals: Variables to inject into the execution namespace.
        work_dir: When set, allows scoped file access to this directory.
    """
    # Capture stdout
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()

    old_cwd = os.getcwd()

    try:
        # Build execution globals with safe builtins
        safe_builtins = create_safe_builtins()

        if work_dir:
            abs_work_dir = os.path.abspath(work_dir)
            os.makedirs(abs_work_dir, exist_ok=True)
            os.chdir(abs_work_dir)
            # Restore open() scoped to work_dir
            safe_builtins["open"] = _create_scoped_open(abs_work_dir)

        exec_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "json": json,  # Always provide json for parsing
        }

        if work_dir:
            exec_globals["csv"] = csv
            exec_globals["Path"] = pathlib.Path

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
        os.chdir(old_cwd)


def execute_python_sandboxed(
    code: str,
    extra_globals: dict[str, Any] | None = None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Uses the backend specified by BLUEBOX_SANDBOX_MODE environment variable.

    Configure via environment variables:
    - BLUEBOX_SANDBOX_MODE: "lambda", "docker", "blocklist", or "auto" (default: "auto")
      - "lambda": Use AWS Lambda for isolation (requires servers' lambda_code_executor)
      - "docker": Use Docker container isolation
      - "blocklist": Use Python-level blocklist (not secure against adversarial input)
      - "auto": Use Lambda if registered, else Docker if available, else blocklist
    - BLUEBOX_SANDBOX_IMAGE: Docker image to use (default: "python:3.12-slim")
    - BLUEBOX_SANDBOX_TIMEOUT: Execution timeout in seconds (default: 30)
    - BLUEBOX_SANDBOX_MEMORY: Container memory limit (default: "128m")

    Args:
        code: Python source code to execute.
        extra_globals: Additional variables to inject into the execution namespace.
        work_dir: When set, grants read/write file access scoped to this directory.
            In Docker mode, the directory is mounted as /data. In blocklist mode,
            open() is scoped to this directory. Not supported in Lambda mode.

    Returns:
        Dict with 'output' (stdout) and optionally 'error' if execution failed.
    """
    if not code:
        return {"error": "No code provided"}

    # Validate work_dir before passing to any backend
    if work_dir:
        work_dir = os.path.abspath(work_dir)
        if any(work_dir == p or work_dir.startswith(p + os.sep) for p in SENSITIVE_PATH_PREFIXES):
            return {"error": f"work_dir points to a sensitive system path: {work_dir}"}

    # Check for blocked patterns (allow open() when work_dir is set)
    safety_error = check_code_safety(code, allow_file_io=bool(work_dir))
    if safety_error:
        return {"error": f"Blocked: {safety_error}"}

    # Lambda mode: use registered Lambda executor (set via register_lambda_executor at startup)
    if SANDBOX_MODE == "lambda":
        if work_dir:
            return {"error": "work_dir is not supported with Lambda sandbox mode"}
        if _lambda_executor_fn is None:
            return {
                "error": (
                    "BLUEBOX_SANDBOX_MODE=lambda requires a registered executor. "
                    "For local development, use BLUEBOX_SANDBOX_MODE=docker or BLUEBOX_SANDBOX_MODE=blocklist"
                )
            }
        logger.debug("Executing code via AWS Lambda sandbox")
        return _lambda_executor_fn(code, extra_globals)

    # Determine execution mode for docker/blocklist/auto
    use_docker = False
    if SANDBOX_MODE == "docker":
        use_docker = True
        if not _is_docker_available():
            return {"error": "Docker sandbox requested but Docker is not available"}
    elif SANDBOX_MODE == "auto":
        # Priority: lambda > docker > blocklist
        if _lambda_executor_fn is not None and not work_dir:
            logger.debug("Executing code via AWS Lambda sandbox (auto-selected)")
            return _lambda_executor_fn(code, extra_globals)
        use_docker = _is_docker_available()
    # else: SANDBOX_MODE == "blocklist" -> use_docker stays False

    if use_docker:
        logger.debug("Executing code in Docker sandbox (work_dir=%s)", work_dir)
        return _execute_in_docker(code, extra_globals, work_dir=work_dir)
    else:
        if SANDBOX_MODE == "auto":
            logger.warning(
                "Docker unavailable, using blocklist sandbox. "
                "This is not secure against adversarial input."
            )
        return _execute_blocklist_sandbox(code, extra_globals, work_dir=work_dir)
