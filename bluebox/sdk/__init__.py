"""
Bluebox SDK - High-level API for web automation.
"""

from .client import Bluebox
from .monitor import BrowserMonitor
from .discovery import RoutineDiscovery
from .execution import RoutineExecutor

__all__ = [
    "Bluebox",
    "BrowserMonitor",
    "RoutineDiscovery",
    "RoutineExecutor",
]

