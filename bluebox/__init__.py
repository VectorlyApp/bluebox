"""
Bluebox SDK - Reverse engineer any web app!

Usage:
    from bluebox import Bluebox

    # Monitor browser activity
    client = Bluebox()
    with client.monitor_browser(output_dir="./captures"):
        # User performs actions in browser
        pass

    # Discover routines
    routine = client.discover_routine(
        task="Search for flights",
        cdp_captures_dir="./captures"
    )

    # Execute routines
    result = client.execute_routine(
        routine=routine,
        parameters={"origin": "NYC", "destination": "LAX"}
    )
"""

__version__ = "1.1.0"

# Public API - High-level interface
from .sdk import Bluebox, BrowserMonitor, RoutineDiscovery, RoutineExecutor

# Data models - for advanced users
from .data_models.routine.routine import Routine
from .data_models.routine.parameter import Parameter
from .data_models.routine.operation import (
    RoutineOperation,
    RoutineOperationUnion,
    RoutineNavigateOperation,
    RoutineFetchOperation,
    RoutineReturnOperation,
    RoutineSleepOperation,
)
from .data_models.routine.endpoint import Endpoint

# Exceptions
from .utils.exceptions import (
    BlueboxError,
    ApiKeyNotFoundError,
    RoutineExecutionError,
    BrowserConnectionError,
    TransactionIdentificationFailedError,
    LLMStructuredOutputError,
    UnsupportedFileFormat,
)

# Core modules (for advanced usage)
from . import agents
from . import cdp
from . import data_models
from . import llms
from . import utils

__all__ = [
    # High-level API
    "Bluebox",
    "BrowserMonitor",
    "RoutineDiscovery",
    "RoutineExecutor",
    # Data models
    "Routine",
    "Parameter",
    "RoutineOperation",
    "RoutineOperationUnion",
    "RoutineNavigateOperation",
    "RoutineFetchOperation",
    "RoutineReturnOperation",
    "RoutineSleepOperation",
    "Endpoint",
    # Exceptions
    "BlueboxError",
    "ApiKeyNotFoundError",
    "RoutineExecutionError",
    "BrowserConnectionError",
    "TransactionIdentificationFailedError",
    "LLMStructuredOutputError",
    "UnsupportedFileFormat",
    # Core modules
    "agents",
    "cdp",
    "data_models",
    "llms",
    "utils",
]

