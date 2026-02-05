"""
bluebox/llms/infra/data_store.py

Re-export of deprecated data store classes for backwards compatibility.
These classes are being migrated to the new data store architecture.
"""

# Re-export from deprecated module for backwards compatibility
# pylint: disable=unused-import
from bluebox.llms.infra.deprecated.data_store import (
    DiscoveryDataStore,
    LocalDiscoveryDataStore,
)

__all__ = ["DiscoveryDataStore", "LocalDiscoveryDataStore"]
