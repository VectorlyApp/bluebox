"""
bluebox/llms/infra/storage_data_store.py

Data store for browser storage event analysis.

Parses JSONL files with StorageEvent entries and provides
methods for token tracing - finding where values originated from.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from bluebox.data_models.cdp import StorageEvent, StorageEventType
from bluebox.utils.data_utils import read_jsonl
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


@dataclass
class StorageStats:
    """Summary statistics for storage events."""

    total_events: int = 0
    cookie_events: int = 0
    local_storage_events: int = 0
    session_storage_events: int = 0
    indexed_db_events: int = 0
    unique_origins: int = 0
    unique_keys: int = 0

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        return "\n".join([
            f"Total Events: {self.total_events}",
            f"  Cookie: {self.cookie_events}",
            f"  LocalStorage: {self.local_storage_events}",
            f"  SessionStorage: {self.session_storage_events}",
            f"  IndexedDB: {self.indexed_db_events}",
            f"Unique Origins: {self.unique_origins}",
            f"Unique Keys: {self.unique_keys}",
        ])


class StorageDataStore:
    """
    Data store for browser storage events.

    Focused on token tracing - finding where values came from
    across cookies, localStorage, sessionStorage, and IndexedDB.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the StorageDataStore from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing StorageEvent entries.
        """
        self._entries: list[StorageEvent] = []
        self._entry_index: dict[int, StorageEvent] = {}

        for line_num, data in read_jsonl(jsonl_path):
            try:
                event = StorageEvent.model_validate(data)
                self._entries.append(event)
                self._entry_index[line_num] = event
            except ValueError as e:
                logger.warning("Failed to validate line %d: %s", line_num + 1, e)
                continue

        self._stats = self._compute_stats()

        logger.info(
            "StorageDataStore initialized with %d events",
            len(self._entries),
        )

    def _compute_stats(self) -> StorageStats:
        """Compute summary statistics."""
        origins: set[str] = set()
        keys: set[str] = set()
        cookie_count = 0
        local_count = 0
        session_count = 0
        indexeddb_count = 0

        for entry in self._entries:
            if entry.origin:
                origins.add(entry.origin)
            if entry.key:
                keys.add(entry.key)

            if entry.type in StorageEventType.cookie_types():
                cookie_count += 1
            elif entry.type in StorageEventType.local_storage_types():
                local_count += 1
            elif entry.type in StorageEventType.session_storage_types():
                session_count += 1
            elif entry.type in StorageEventType.indexed_db_types():
                indexeddb_count += 1

        return StorageStats(
            total_events=len(self._entries),
            cookie_events=cookie_count,
            local_storage_events=local_count,
            session_storage_events=session_count,
            indexed_db_events=indexeddb_count,
            unique_origins=len(origins),
            unique_keys=len(keys),
        )

    @property
    def entries(self) -> list[StorageEvent]:
        """Return all storage events."""
        return self._entries

    @property
    def stats(self) -> StorageStats:
        """Return computed statistics."""
        return self._stats

    def get_entry(self, index: int) -> StorageEvent | None:
        """Get entry by index."""
        return self._entry_index.get(index)

    def get_entries_by_origin(self, origin: str) -> list[StorageEvent]:
        """
        Get all events for a specific origin.

        Args:
            origin: The origin to filter by (exact match).

        Returns:
            List of StorageEvent objects for the given origin.
        """
        return [e for e in self._entries if e.origin == origin]

    def get_entries_by_key(self, key: str) -> list[StorageEvent]:
        """
        Get all events for a specific storage key.

        Args:
            key: The key to filter by (exact match).

        Returns:
            List of StorageEvent objects for the given key.
        """
        return [e for e in self._entries if e.key == key]

    def get_entries_by_origin_and_key(self, origin: str, key: str) -> list[StorageEvent]:
        """
        Get all events for a specific origin and key combination.

        Args:
            origin: The origin to filter by (exact match).
            key: The key to filter by (exact match).

        Returns:
            List of StorageEvent objects matching both origin and key.
        """
        return [e for e in self._entries if e.origin == origin and e.key == key]

    def search_values(self, value: str, case_sensitive: bool = False) -> list[dict[str, Any]]:
        """
        Search storage values for a given string (token tracing).

        Searches across all storage event fields: value, new_value, old_value,
        added items, and modified items.

        Args:
            value: The value to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            List of dicts containing:
            - index: Entry index
            - type: Event type
            - origin: Origin
            - key: Storage key (if applicable)
            - match_locations: List of where matches were found
        """
        results: list[dict[str, Any]] = []

        if not value:
            return results

        search_value = value if case_sensitive else value.lower()

        for idx, entry in enumerate(self._entries):
            # Collect all searchable values with their location names
            searchable: list[tuple[str, str]] = []
            for field in StorageEvent.SEARCHABLE_FIELDS:
                val = getattr(entry, field)
                if val is not None:
                    searchable.append((field, val if isinstance(val, str) else json.dumps(val)))

            # Scan all values for matches
            match_locations = [
                location for location, val_str in searchable
                if search_value in (val_str if case_sensitive else val_str.lower())
            ]

            if match_locations:
                results.append({
                    "index": idx,
                    "type": entry.type,
                    "origin": entry.origin,
                    "key": entry.key,
                    "match_locations": match_locations,
                })

        return results
