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

# Workaround hints for blocked modules — returned to the LLM when a block triggers
BLOCKED_MODULE_WORKAROUNDS: dict[str, str] = {
    "os": "Use open() for file I/O and Path for paths — both are pre-loaded and scoped to the workspace.",
    "pathlib": "Path is already pre-loaded in your environment. Use it directly without importing: Path('output/file.txt').",
    "shutil": "Use open() to read source and write to destination for copying files.",
    "tempfile": "Write temporary files to the output/ directory using open().",
    "glob": "Use Path('.').glob('pattern') or Path('.').rglob('pattern') — Path is pre-loaded.",
    "fnmatch": "Use Path('.').glob('pattern') — Path is pre-loaded.",
    "io": "Use open() for file operations and str for string building.",
    "sys": "stdout is captured automatically. Use print() for output.",
    "subprocess": "External process execution is not available in the sandbox.",
    "multiprocessing": "Parallel execution is not available in the sandbox.",
    "threading": "Thread creation is not available in the sandbox.",
    "concurrent": "Parallel execution is not available in the sandbox.",
    "inspect": "Use dict access (obj['key'] or obj.get('key')) instead of inspecting objects.",
    "pickle": "Use json.dumps()/json.loads() for serialization instead.",
    "marshal": "Use json.dumps()/json.loads() for serialization instead.",
    "shelve": "Use json for data persistence. Write to output/ with open().",
    "sqlite3": "Database access is not available. Use JSON/CSV files for data storage.",
    "socket": "Network access is not available in the sandbox.",
    "requests": "Network access is not available in the sandbox.",
    "http": "Network access is not available in the sandbox.",
}

# Workaround hints for blocked code patterns.
# These fire in ALL sandbox modes (patterns are checked statically before execution).
# Note: "open(" is skipped when work_dir is set (allow_file_io=True), so its workaround
# only fires for callers that don't provide a work_dir.
BLOCKED_PATTERN_WORKAROUNDS: dict[str, str] = {
    "open(": "open() IS available when a workspace is configured. Use it directly: open('output/file.csv', 'w').",
    "getattr(": "Use dict-style access instead: obj['key'] or obj.get('key', default).",
    "setattr(": "Use dict-style assignment instead: obj['key'] = value.",
    "delattr(": "Use dict-style deletion instead: del obj['key'].",
    "eval(": "Write the logic directly in Python instead of using eval().",
    "exec(": "Write the logic directly in Python instead of using exec().",
    "compile(": "Write the logic directly in Python instead of using compile().",
    "globals(": "Access your variables directly by name — they are already in scope.",
    "locals(": "Access your variables directly by name — they are already in scope.",
    "vars(": "Use dict-style access: obj['key'] or iterate with a for loop.",
    "__import__": "Import statements are restricted. json, csv, and Path are pre-loaded. Use 'import collections', 'import re', 'import datetime', 'import math' for allowed modules.",
}


def get_workaround_for_error(error_message: str) -> str | None:
    """
    Given a sandbox error message, return a helpful workaround hint if available.

    Args:
        error_message: The error string from sandbox execution.

    Returns:
        Workaround message string, or None if no specific workaround is known.
    """
    # Blocked module: "Import of 'os' is blocked for security reasons"
    if "is blocked for security reasons" in error_message:
        for module_name, workaround in BLOCKED_MODULE_WORKAROUNDS.items():
            if f"'{module_name}'" in error_message:
                return workaround
        # Check root module for submodule imports like 'os.path'
        for module_name, workaround in BLOCKED_MODULE_WORKAROUNDS.items():
            if f"'{module_name}." in error_message:
                return workaround
        return None

    # Blocked pattern: "Blocked: File operations (open) are not allowed"
    if error_message.startswith("Blocked:"):
        blocked_msg = error_message[len("Blocked: "):]
        for pattern, err_msg in BLOCKED_PATTERNS:
            if blocked_msg == err_msg:
                return BLOCKED_PATTERN_WORKAROUNDS.get(pattern)

    return None


def get_active_sandbox_mode(work_dir_set: bool = False) -> str:
    """
    Return the effective sandbox mode that will be used at execution time.

    Args:
        work_dir_set: Whether a work_dir will be provided (affects Lambda eligibility).

    Returns:
        One of "lambda", "docker", or "blocklist".
    """
    if SANDBOX_MODE == "lambda":
        return "lambda"
    if SANDBOX_MODE == "docker":
        return "docker" if is_docker_available() else "blocklist"
    if SANDBOX_MODE == "blocklist":
        return "blocklist"
    # auto mode: lambda > docker > blocklist
    if _lambda_executor_fn is not None and not work_dir_set:
        return "lambda"
    if is_docker_available():
        return "docker"
    return "blocklist"


def is_docker_available() -> bool:
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
    read_only_paths: list[str] | None = None,
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
        read_only_paths: Optional list of paths under work_dir to mount read-only.

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
        "--user", f"{os.getuid()}:{os.getgid()}",  # Match host user for volume permissions
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",  # Writable /tmp for Python
        "-i",                            # Accept stdin
    ]

    # Mount work_dir as /data with read-write access
    if work_dir:
        abs_work_dir = os.path.abspath(work_dir)
        docker_cmd.extend(["-v", f"{abs_work_dir}:/data:rw", "-w", "/data"])
        for ro_path in read_only_paths or []:
            abs_ro_path = os.path.abspath(ro_path)
            if not os.path.exists(abs_ro_path):
                continue
            if not (abs_ro_path == abs_work_dir or abs_ro_path.startswith(abs_work_dir + os.sep)):
                logger.warning("Skipping read-only mount outside work_dir: %s", abs_ro_path)
                continue
            rel_ro = os.path.relpath(abs_ro_path, abs_work_dir)
            container_ro = "/data" if rel_ro == "." else f"/data/{rel_ro}"
            docker_cmd.extend(["-v", f"{abs_ro_path}:{container_ro}:ro"])

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


def _is_within_scoped_root(root: str, candidate: str) -> bool:
    """Return True when candidate path is equal to or under root."""
    return candidate == root or candidate.startswith(root + os.sep)


def _resolve_scoped_path(work_dir: str, file: str | os.PathLike[str]) -> str:
    """
    Resolve a user-provided path under work_dir and reject escapes.

    Resolves symlinks in existing path components to prevent symlink-based
    escapes from bypassing scope checks.
    """
    raw = os.fspath(file)
    joined = raw if os.path.isabs(raw) else os.path.join(work_dir, raw)
    resolved = os.path.realpath(joined)
    if not _is_within_scoped_root(work_dir, resolved):
        raise PermissionError(f"Access denied: path '{raw}' is outside the working directory")
    return resolved


def _assert_path_writable(
    resolved_path: str,
    *,
    raw_path: str,
    read_only_paths: tuple[str, ...],
) -> None:
    """Reject writes into configured read-only paths."""
    for ro_path in read_only_paths:
        if _is_within_scoped_root(ro_path, resolved_path):
            raise PermissionError(f"Access denied: path '{raw_path}' is read-only")


def _create_scoped_path(
    work_dir: str,
    scoped_open: Any,
    read_only_paths: list[str] | None = None,
) -> type:
    """
    Create a Path-like class scoped to work_dir.

    This prevents bypasses where pathlib operations write outside work_dir while
    still allowing familiar Path ergonomics inside the sandbox.
    """
    abs_work_dir = os.path.realpath(os.path.abspath(work_dir))
    normalized_read_only = tuple(os.path.realpath(os.path.abspath(p)) for p in (read_only_paths or []))

    class ScopedPath:
        __slots__ = ("__resolved",)

        def __init__(self, *parts: Any) -> None:
            if not parts:
                parts = (".",)
            normalized_parts: list[str] = []
            for part in parts:
                if isinstance(part, ScopedPath):
                    normalized_parts.append(str(part.__resolved))
                else:
                    normalized_parts.append(os.fspath(part))
            joined = pathlib.Path(*normalized_parts)
            resolved = _resolve_scoped_path(abs_work_dir, os.fspath(joined))
            self.__resolved = pathlib.Path(resolved)

        @property
        def name(self) -> str:
            return self.__resolved.name

        @property
        def suffix(self) -> str:
            return self.__resolved.suffix

        @property
        def stem(self) -> str:
            return self.__resolved.stem

        @property
        def parent(self) -> Any:
            return ScopedPath(str(self.__resolved.parent))

        @property
        def parts(self) -> tuple[str, ...]:
            return self.__resolved.parts

        @property
        def resolved_path(self) -> str:
            """Absolute, scoped path string."""
            return str(self.__resolved)

        def __fspath__(self) -> str:
            return str(self.__resolved)

        def __str__(self) -> str:
            return str(self.__resolved)

        def __repr__(self) -> str:
            return f"Path('{self.__resolved}')"

        def __eq__(self, other: object) -> bool:
            if isinstance(other, ScopedPath):
                return self.__resolved == other.__resolved
            if isinstance(other, (str, os.PathLike)):
                try:
                    return str(self.__resolved) == _resolve_scoped_path(abs_work_dir, os.fspath(other))
                except PermissionError:
                    return False
            return False

        def __hash__(self) -> int:
            return hash(str(self.__resolved))

        def __truediv__(self, other: Any) -> Any:
            return self.joinpath(other)

        def joinpath(self, *others: Any) -> Any:
            return ScopedPath(str(self.__resolved), *others)

        @classmethod
        def cwd(cls) -> Any:
            return cls(".")

        def resolve(self, strict: bool = False) -> Any:
            if strict and not self.__resolved.exists():
                raise FileNotFoundError(str(self.__resolved))
            return ScopedPath(str(self.__resolved))

        def exists(self) -> bool:
            return self.__resolved.exists()

        def is_file(self) -> bool:
            return self.__resolved.is_file()

        def is_dir(self) -> bool:
            return self.__resolved.is_dir()

        def open(self, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            return scoped_open(str(self.__resolved), mode, *args, **kwargs)

        def read_text(
            self,
            encoding: str | None = None,
            errors: str | None = None,
        ) -> str:
            with self.open("r", encoding=encoding, errors=errors) as f:
                return f.read()

        def write_text(
            self,
            data: str,
            encoding: str | None = None,
            errors: str | None = None,
            newline: str | None = None,
        ) -> int:
            with self.open("w", encoding=encoding, errors=errors, newline=newline) as f:
                return f.write(data)

        def read_bytes(self) -> bytes:
            with self.open("rb") as f:
                return f.read()

        def write_bytes(self, data: bytes) -> int:
            with self.open("wb") as f:
                return f.write(data)

        def mkdir(
            self,
            mode: int = 0o777,
            parents: bool = False,
            exist_ok: bool = False,
        ) -> None:
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            self.__resolved.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)

        def touch(self, mode: int = 0o666, exist_ok: bool = True) -> None:
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            self.__resolved.touch(mode=mode, exist_ok=exist_ok)

        def unlink(self, missing_ok: bool = False) -> None:
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            self.__resolved.unlink(missing_ok=missing_ok)

        def rmdir(self) -> None:
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            self.__resolved.rmdir()

        def rename(self, target: Any) -> Any:
            target_path = ScopedPath(target)
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            _assert_path_writable(
                str(target_path.__resolved),
                raw_path=os.fspath(target),
                read_only_paths=normalized_read_only,
            )
            renamed = self.__resolved.rename(str(target_path.__resolved))
            return ScopedPath(str(renamed))

        def replace(self, target: Any) -> Any:
            target_path = ScopedPath(target)
            _assert_path_writable(
                str(self.__resolved),
                raw_path=str(self),
                read_only_paths=normalized_read_only,
            )
            _assert_path_writable(
                str(target_path.__resolved),
                raw_path=os.fspath(target),
                read_only_paths=normalized_read_only,
            )
            replaced = self.__resolved.replace(str(target_path.__resolved))
            return ScopedPath(str(replaced))

        def iterdir(self) -> Any:
            for child in self.__resolved.iterdir():
                try:
                    yield ScopedPath(str(child))
                except PermissionError:
                    continue

        def glob(self, pattern: str) -> Any:
            for child in self.__resolved.glob(pattern):
                try:
                    yield ScopedPath(str(child))
                except PermissionError:
                    continue

        def rglob(self, pattern: str) -> Any:
            for child in self.__resolved.rglob(pattern):
                try:
                    yield ScopedPath(str(child))
                except PermissionError:
                    continue

    return ScopedPath


def _create_scoped_open(work_dir: str, read_only_paths: list[str] | None = None) -> Any:
    """
    Create an open() function that restricts file access to work_dir.

    All paths are resolved relative to work_dir. Paths that escape
    work_dir (e.g. via '..') are rejected.

    Args:
        work_dir: The directory to scope file access to.
        read_only_paths: Optional absolute paths under work_dir that cannot
            be opened in write/append/update modes.
    """
    abs_work_dir = os.path.realpath(os.path.abspath(work_dir))
    normalized_read_only = tuple(os.path.realpath(os.path.abspath(p)) for p in (read_only_paths or []))

    def scoped_open(
        file: str | os.PathLike[str],
        mode: str = "r",
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        raw_file = os.fspath(file)
        resolved = _resolve_scoped_path(abs_work_dir, raw_file)
        write_intent = ("w" in mode) or ("a" in mode) or ("x" in mode) or ("+" in mode)
        if write_intent:
            _assert_path_writable(
                resolved,
                raw_path=raw_file,
                read_only_paths=normalized_read_only,
            )
        return open(resolved, mode, *args, **kwargs)  # noqa: SIM115

    return scoped_open


def _execute_blocklist_sandbox(
    code: str,
    extra_globals: dict[str, Any] | None = None,
    work_dir: str | None = None,
    read_only_paths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Execute code using blocklist-based sandboxing (fallback method).

    WARNING: This method is not secure against adversarial input.
    Use only when Docker is unavailable.

    Args:
        code: Python source code to execute.
        extra_globals: Variables to inject into the execution namespace.
        work_dir: When set, allows scoped file access to this directory.
        read_only_paths: Optional list of absolute paths under work_dir that
            cannot be written.
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
            scoped_open = _create_scoped_open(
                abs_work_dir,
                read_only_paths=read_only_paths,
            )
            safe_builtins["open"] = scoped_open

        exec_globals: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "json": json,  # Always provide json for parsing
        }

        if work_dir:
            exec_globals["csv"] = csv
            exec_globals["Path"] = _create_scoped_path(
                abs_work_dir,
                scoped_open,
                read_only_paths=read_only_paths,
            )

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
    read_only_paths: list[str] | None = None,
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
        read_only_paths: Optional paths under work_dir that are read-only.

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
    if read_only_paths and not work_dir:
        return {"error": "read_only_paths requires work_dir"}

    normalized_read_only_paths: list[str] = []
    for ro_path in read_only_paths or []:
        abs_ro_path = os.path.abspath(ro_path)
        if work_dir and not (abs_ro_path == work_dir or abs_ro_path.startswith(work_dir + os.sep)):
            return {"error": f"read_only_path is outside work_dir: {ro_path}"}
        normalized_read_only_paths.append(abs_ro_path)

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
        if not is_docker_available():
            return {"error": "Docker sandbox requested but Docker is not available"}
    elif SANDBOX_MODE == "auto":
        # Priority: lambda > docker > blocklist
        if _lambda_executor_fn is not None and not work_dir:
            logger.debug("Executing code via AWS Lambda sandbox (auto-selected)")
            return _lambda_executor_fn(code, extra_globals)
        use_docker = is_docker_available()
    # else: SANDBOX_MODE == "blocklist" -> use_docker stays False

    if use_docker:
        logger.debug("Executing code in Docker sandbox (work_dir=%s)", work_dir)
        return _execute_in_docker(
            code,
            extra_globals,
            work_dir=work_dir,
            read_only_paths=normalized_read_only_paths,
        )
    else:
        if SANDBOX_MODE == "auto":
            logger.warning(
                "Docker unavailable, using blocklist sandbox. "
                "This is not secure against adversarial input."
            )
        return _execute_blocklist_sandbox(
            code,
            extra_globals,
            work_dir=work_dir,
            read_only_paths=normalized_read_only_paths,
        )
