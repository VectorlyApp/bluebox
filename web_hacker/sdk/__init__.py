"""
Web Hacker SDK - High-level API for web automation.
"""

from .client import WebHacker
from .monitor import BrowserMonitor
from .discovery import RoutineDiscovery
from .execution import RoutineExecutor

__all__ = [
    "WebHacker",
    "BrowserMonitor",
    "RoutineDiscovery",
    "RoutineExecutor",
]

