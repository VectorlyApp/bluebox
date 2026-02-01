"""
bluebox/cdp/monitors/__init__.py

NOTE: This file is actually necessary, because it triggers the AbstractAsyncMonitor.__init_subclass__
method for all monitor classes, so that the AbstractAsyncMonitor._subclasses list is populated;
That is important because it enables using AbstractAsyncMonitor.get_all_subclasses().
"""

from bluebox.cdp.monitors.abstract_async_monitor import AbstractAsyncMonitor

# import all monitor classes to trigger AbstractAsyncMonitor.__init_subclass__
from bluebox.cdp.monitors.async_interaction_monitor import AsyncInteractionMonitor
from bluebox.cdp.monitors.async_network_monitor import AsyncNetworkMonitor
from bluebox.cdp.monitors.async_storage_monitor import AsyncStorageMonitor
from bluebox.cdp.monitors.async_dom_monitor import AsyncDOMMonitor
from bluebox.cdp.monitors.async_window_property_monitor import AsyncWindowPropertyMonitor

__all__ = [
    "AbstractAsyncMonitor",
    "AsyncDOMMonitor",
    "AsyncInteractionMonitor",
    "AsyncNetworkMonitor",
    "AsyncStorageMonitor",
    "AsyncWindowPropertyMonitor",
]
