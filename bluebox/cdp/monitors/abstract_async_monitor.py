"""
bluebox/cdp/monitors/abstract_async_monitor.py

Abstract base class for asynchronous CDP monitors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class AbstractAsyncMonitor(ABC):
    """
    Abstract base class for asynchronous CDP monitors.
    All asynchronous monitors (AsyncNetworkMonitor, AsyncStorageMonitor, AsyncWindowPropertyMonitor) should inherit from this.
    """

    # Class attributes _____________________________________________________________________________________________________

    _subclasses: ClassVar[list[type[AbstractAsyncMonitor]]] = []  # list of all subclasses of AbstractAsyncMonitor


    # Magic methods ________________________________________________________________________________________________________

    def __init_subclass__(cls: type[AbstractAsyncMonitor], **kwargs: Any) -> None:
        """
        Add the subclass to the AbstractAsyncMonitor._subclasses list when the subclass is defined.
        """
        super().__init_subclass__(**kwargs)
        cls._subclasses.append(cls)


    # Class methods ________________________________________________________________________________________________________

    @classmethod
    def get_all_subclasses(cls: type[AbstractAsyncMonitor]) -> list[type[AbstractAsyncMonitor]]:
        """
        Return a copy of the list of all subclasses of AbstractAsyncMonitor.
        """
        return cls._subclasses.copy()

    @classmethod
    def get_monitor_category(cls) -> str:
        """
        Return the category name for this monitor class.
        Returns:
            The class name (e.g., "AsyncNetworkMonitor").
        """
        return cls.__name__

    @classmethod
    @abstractmethod
    def get_ws_event_summary(cls, detail: dict[str, Any]) -> dict[str, Any]:
        """
        Extract a lightweight summary of an event for WebSocket streaming.
        Each monitor defines what fields are relevant for real-time streaming to clients.
        Args:
            detail: The full event detail dict emitted by the monitor.
        Returns:
            A simplified dict with only the fields relevant for WebSocket streaming.
        """
        # not raising UnimplementedError here because this is an abstract method
        pass

    # TODO: add additional abstract methods (e.g., `setup_monitor`, `handle_message`, etc.)
