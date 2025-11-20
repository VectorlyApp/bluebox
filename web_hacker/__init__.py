"""
Web Hacker SDK - Reverse engineer any web app!

Usage:
    from web_hacker import WebHacker
    
    # Monitor browser activity
    hacker = WebHacker()
    with hacker.monitor_browser(output_dir="./captures"):
        # User performs actions in browser
        pass
    
    # Discover routines
    routine = hacker.discover_routine(
        task="Search for flights",
        cdp_captures_dir="./captures"
    )
    
    # Execute routines
    result = hacker.execute_routine(
        routine=routine,
        parameters={"origin": "NYC", "destination": "LAX"}
    )
"""

__version__ = "1.1.0"

# Public API - High-level interface
from .sdk import WebHacker, BrowserMonitor, RoutineDiscovery, RoutineExecutor

# Data models - for advanced users
from .data_models.production_routine import (
    Routine,
    Parameter,
    RoutineOperation,
    RoutineNavigateOperation,
    RoutineFetchOperation,
    RoutineReturnOperation,
    RoutineSleepOperation,
    Endpoint,
)

# Exceptions
from .utils.exceptions import (
    WebHackerError,
    ApiKeyNotFoundError,
    RoutineExecutionError,
    BrowserConnectionError,
    TransactionIdentificationFailedError,
    LLMStructuredOutputError,
    UnsupportedFileFormat,
)

# Core modules (for advanced usage)
from . import cdp
from . import data_models
from . import routine_discovery
from . import utils

__all__ = [
    # High-level API
    "WebHacker",
    "BrowserMonitor",
    "RoutineDiscovery",
    "RoutineExecutor",
    # Data models
    "Routine",
    "Parameter",
    "RoutineOperation",
    "RoutineNavigateOperation",
    "RoutineFetchOperation",
    "RoutineReturnOperation",
    "RoutineSleepOperation",
    "Endpoint",
    # Exceptions
    "WebHackerError",
    "ApiKeyNotFoundError",
    "RoutineExecutionError",
    "BrowserConnectionError",
    "TransactionIdentificationFailedError",
    "LLMStructuredOutputError",
    "UnsupportedFileFormat",
    # Core modules
    "cdp",
    "data_models",
    "routine_discovery",
    "utils",
]

