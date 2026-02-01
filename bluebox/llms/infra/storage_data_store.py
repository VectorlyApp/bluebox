"""
bluebox/llms/infra/storage_data_store.py

Data store for browser storage event analysis.

Parses JSONL files with StorageEvent entries and provides
structured access to storage data (cookies, localStorage, sessionStorage, IndexedDB).
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
    """Summary statistics for a storage events file."""

    total_events: int = 0

    # Event type counts
    event_types: dict[str, int] = field(default_factory=dict)

    # Category counts
    cookie_events: int = 0
    local_storage_events: int = 0
    session_storage_events: int = 0
    indexed_db_events: int = 0

    # Origin/domain tracking
    origins: dict[str, int] = field(default_factory=dict)
    unique_origins: int = 0

    # Key tracking (for localStorage/sessionStorage)
    unique_keys: int = 0
    keys_by_origin: dict[str, set[str]] = field(default_factory=dict)

    # Cookie tracking
    cookie_domains: dict[str, int] = field(default_factory=dict)
    unique_cookie_names: int = 0

    # Change tracking
    total_items_added: int = 0
    total_items_modified: int = 0
    total_items_removed: int = 0

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Events: {self.total_events}",
            "",
            "Event Categories:",
            f"  Cookie Events: {self.cookie_events}",
            f"  LocalStorage Events: {self.local_storage_events}",
            f"  SessionStorage Events: {self.session_storage_events}",
            f"  IndexedDB Events: {self.indexed_db_events}",
            "",
            "Event Types:",
        ]
        for event_type, count in sorted(self.event_types.items(), key=lambda x: -x[1]):
            lines.append(f"  {event_type}: {count}")

        lines.append("")
        lines.append(f"Unique Origins: {self.unique_origins}")
        if self.origins:
            lines.append("Top Origins:")
            for origin, count in sorted(self.origins.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {origin}: {count}")

        if self.cookie_domains:
            lines.append("")
            lines.append("Cookie Domains:")
            for domain, count in sorted(self.cookie_domains.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {domain}: {count}")

        lines.append("")
        lines.append("Change Summary:")
        lines.append(f"  Items Added: {self.total_items_added}")
        lines.append(f"  Items Modified: {self.total_items_modified}")
        lines.append(f"  Items Removed: {self.total_items_removed}")

        return "\n".join(lines)


class StorageDataStore:
    """
    Data store for browser storage event analysis.

    Parses JSONL content and provides structured access to storage events
    including cookies, localStorage, sessionStorage, and IndexedDB.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the StorageDataStore from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing StorageEvent entries.
        """
        self._entries: list[StorageEvent] = []
        self._entry_index: dict[int, StorageEvent] = {}  # index -> event
        self._stats: StorageStats = StorageStats()

        # Load entries from JSONL
        for line_num, data in read_jsonl(jsonl_path):
            try:
                event = StorageEvent.model_validate(data)
                self._entries.append(event)
                self._entry_index[line_num] = event
            except ValueError as e:
                logger.warning("Failed to validate line %d: %s", line_num + 1, e)
                continue

        self._compute_stats()

        logger.info(
            "StorageDataStore initialized with %d events",
            len(self._entries),
        )

    @property
    def entries(self) -> list[StorageEvent]:
        """Return all storage events."""
        return self._entries

    @property
    def stats(self) -> StorageStats:
        """Return computed statistics."""
        return self._stats

    @property
    def raw_data(self) -> dict[str, Any]:
        """Return entries as a dict for compatibility."""
        return {"entries": [e.model_dump() for e in self._entries]}

    @property
    def event_type_counts(self) -> dict[str, int]:
        """Mapping of each event type to its occurrence count."""
        counts: Counter[str] = Counter(entry.type for entry in self._entries)
        return dict(counts.most_common())

    @property
    def cookie_events(self) -> list[StorageEvent]:
        """Return only cookie-related events."""
        return [e for e in self._entries if e.type in StorageEventType.cookie_types()]

    @property
    def local_storage_events(self) -> list[StorageEvent]:
        """Return only localStorage-related events."""
        return [e for e in self._entries if e.type in StorageEventType.local_storage_types()]

    @property
    def session_storage_events(self) -> list[StorageEvent]:
        """Return only sessionStorage-related events."""
        return [e for e in self._entries if e.type in StorageEventType.session_storage_types()]

    @property
    def indexed_db_events(self) -> list[StorageEvent]:
        """Return only IndexedDB-related events."""
        return [e for e in self._entries if e.type in StorageEventType.indexed_db_types()]

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        event_types: Counter[str] = Counter()
        origins: Counter[str] = Counter()
        cookie_domains: Counter[str] = Counter()
        keys_by_origin: dict[str, set[str]] = {}
        all_keys: set[str] = set()
        cookie_names: set[str] = set()

        cookie_count = 0
        local_storage_count = 0
        session_storage_count = 0
        indexed_db_count = 0

        total_added = 0
        total_modified = 0
        total_removed = 0

        for entry in self._entries:
            event_types[entry.type] += 1

            # Categorize event
            if entry.type in StorageEventType.cookie_types():
                cookie_count += 1
                # Track cookie domains and names from added/modified cookies
                for cookie_list in [entry.added, entry.modified]:
                    if cookie_list:
                        for cookie in cookie_list:
                            if isinstance(cookie, dict):
                                domain = cookie.get("domain", "")
                                name = cookie.get("name", "")
                                if domain:
                                    cookie_domains[domain] += 1
                                if name:
                                    cookie_names.add(name)
            elif entry.type in StorageEventType.local_storage_types():
                local_storage_count += 1
            elif entry.type in StorageEventType.session_storage_types():
                session_storage_count += 1
            elif entry.type in StorageEventType.indexed_db_types():
                indexed_db_count += 1

            # Track origins
            if entry.origin:
                origins[entry.origin] += 1
                if entry.key:
                    if entry.origin not in keys_by_origin:
                        keys_by_origin[entry.origin] = set()
                    keys_by_origin[entry.origin].add(entry.key)
                    all_keys.add(entry.key)

            # Track changes
            if entry.added:
                total_added += len(entry.added)
            if entry.modified:
                total_modified += len(entry.modified)
            if entry.removed:
                total_removed += len(entry.removed)

            # Track individual item events
            if entry.type in {
                StorageEventType.LOCAL_STORAGE_ITEM_ADDED,
                StorageEventType.SESSION_STORAGE_ITEM_ADDED,
            }:
                total_added += 1
            elif entry.type in {
                StorageEventType.LOCAL_STORAGE_ITEM_UPDATED,
                StorageEventType.SESSION_STORAGE_ITEM_UPDATED,
            }:
                total_modified += 1
            elif entry.type in {
                StorageEventType.LOCAL_STORAGE_ITEM_REMOVED,
                StorageEventType.SESSION_STORAGE_ITEM_REMOVED,
            }:
                total_removed += 1

        self._stats = StorageStats(
            total_events=len(self._entries),
            event_types=dict(event_types),
            cookie_events=cookie_count,
            local_storage_events=local_storage_count,
            session_storage_events=session_storage_count,
            indexed_db_events=indexed_db_count,
            origins=dict(origins),
            unique_origins=len(origins),
            unique_keys=len(all_keys),
            keys_by_origin={k: v for k, v in keys_by_origin.items()},
            cookie_domains=dict(cookie_domains),
            unique_cookie_names=len(cookie_names),
            total_items_added=total_added,
            total_items_modified=total_modified,
            total_items_removed=total_removed,
        )

    def search_entries(
        self,
        event_type: str | None = None,
        origin_contains: str | None = None,
        key_contains: str | None = None,
        category: str | None = None,
    ) -> list[StorageEvent]:
        """
        Search entries with filters.

        Args:
            event_type: Filter by exact event type (e.g., "cookieChange").
            origin_contains: Filter by origin containing substring.
            key_contains: Filter by key containing substring.
            category: Filter by category ("cookie", "localStorage", "sessionStorage", "indexedDB").

        Returns:
            List of matching StorageEvent objects.
        """
        results = []

        # Determine category filter set
        category_types: set[StorageEventType] | None = None
        if category:
            category_lower = category.lower()
            if category_lower == "cookie":
                category_types = StorageEventType.cookie_types()
            elif category_lower == "localstorage":
                category_types = StorageEventType.local_storage_types()
            elif category_lower == "sessionstorage":
                category_types = StorageEventType.session_storage_types()
            elif category_lower == "indexeddb":
                category_types = StorageEventType.indexed_db_types()

        for entry in self._entries:
            if event_type and entry.type != event_type:
                continue
            if category_types and entry.type not in category_types:
                continue
            if origin_contains and (not entry.origin or origin_contains.lower() not in entry.origin.lower()):
                continue
            if key_contains and (not entry.key or key_contains.lower() not in entry.key.lower()):
                continue

            results.append(entry)

        return results

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

    def get_origin_stats(self, origin_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Get per-origin summary statistics.

        Args:
            origin_filter: Optional substring to filter origins (case-insensitive).

        Returns:
            List of dicts sorted by event count descending, each containing:
            - origin: The origin
            - event_count: Number of events for this origin
            - event_types: Dict of event type counts
            - keys: Set of unique keys for this origin
        """
        origin_data: dict[str, dict[str, Any]] = {}

        for entry in self._entries:
            if not entry.origin:
                continue

            origin = entry.origin

            # Apply filter if provided
            if origin_filter and origin_filter.lower() not in origin.lower():
                continue

            if origin not in origin_data:
                origin_data[origin] = {
                    "event_count": 0,
                    "event_types": Counter(),
                    "keys": set(),
                }

            origin_data[origin]["event_count"] += 1
            origin_data[origin]["event_types"][entry.type] += 1
            if entry.key:
                origin_data[origin]["keys"].add(entry.key)

        results = []
        for origin, data in sorted(origin_data.items(), key=lambda x: -x[1]["event_count"]):
            results.append({
                "origin": origin,
                "event_count": data["event_count"],
                "event_types": dict(data["event_types"]),
                "keys": list(data["keys"]),
            })

        return results

    def search_values(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search storage values for a given string.

        Args:
            value: The value to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            List of dicts containing:
            - index: Entry index
            - type: Event type
            - origin: Origin
            - key: Storage key (if applicable)
            - match_location: Where the match was found ("value", "new_value", "old_value", "added", "modified")
        """
        results: list[dict[str, Any]] = []

        if not value:
            return results

        search_value = value if case_sensitive else value.lower()

        for idx, entry in enumerate(self._entries):
            match_locations: list[str] = []

            # Check value field
            if entry.value is not None:
                val_str = str(entry.value) if case_sensitive else str(entry.value).lower()
                if search_value in val_str:
                    match_locations.append("value")

            # Check new_value field
            if entry.new_value is not None:
                val_str = str(entry.new_value) if case_sensitive else str(entry.new_value).lower()
                if search_value in val_str:
                    match_locations.append("new_value")

            # Check old_value field
            if entry.old_value is not None:
                val_str = str(entry.old_value) if case_sensitive else str(entry.old_value).lower()
                if search_value in val_str:
                    match_locations.append("old_value")

            # Check added cookies/items
            if entry.added:
                for item in entry.added:
                    item_str = json.dumps(item) if case_sensitive else json.dumps(item).lower()
                    if search_value in item_str:
                        match_locations.append("added")
                        break

            # Check modified cookies/items
            if entry.modified:
                for item in entry.modified:
                    item_str = json.dumps(item) if case_sensitive else json.dumps(item).lower()
                    if search_value in item_str:
                        match_locations.append("modified")
                        break

            if match_locations:
                results.append({
                    "index": idx,
                    "type": entry.type,
                    "origin": entry.origin,
                    "key": entry.key,
                    "match_locations": match_locations,
                })

        return results

    def get_cookie_summary(self) -> dict[str, Any]:
        """
        Get a summary of cookie activity.

        Returns:
            Dict containing:
            - total_cookie_events: Total number of cookie events
            - domains: Dict of domain -> event count
            - changes: Dict with added/modified/removed counts
        """
        domains: Counter[str] = Counter()
        added_count = 0
        modified_count = 0
        removed_count = 0

        for entry in self.cookie_events:
            if entry.added:
                added_count += len(entry.added)
                for cookie in entry.added:
                    if isinstance(cookie, dict) and cookie.get("domain"):
                        domains[cookie["domain"]] += 1

            if entry.modified:
                modified_count += len(entry.modified)
                for item in entry.modified:
                    cookie = item.get("new") if isinstance(item, dict) else item
                    if isinstance(cookie, dict) and cookie.get("domain"):
                        domains[cookie["domain"]] += 1

            if entry.removed:
                removed_count += len(entry.removed)

        return {
            "total_cookie_events": len(self.cookie_events),
            "domains": dict(domains.most_common()),
            "changes": {
                "added": added_count,
                "modified": modified_count,
                "removed": removed_count,
            },
        }

    def get_storage_timeline(self) -> list[dict[str, Any]]:
        """
        Get a timeline of storage events ordered by timestamp.

        Returns:
            List of dicts with:
            - timestamp: Event timestamp
            - type: Event type
            - origin: Origin (if applicable)
            - key: Key (if applicable)
            - summary: Brief description of the event
        """
        timeline = []

        for entry in self._entries:
            summary = entry.type
            if entry.key:
                summary = f"{entry.type}: {entry.key}"
            elif entry.added:
                summary = f"{entry.type}: {len(entry.added)} item(s) added"
            elif entry.modified:
                summary = f"{entry.type}: {len(entry.modified)} item(s) modified"
            elif entry.removed:
                summary = f"{entry.type}: {len(entry.removed)} item(s) removed"

            timeline.append({
                "timestamp": entry.timestamp,
                "type": entry.type,
                "origin": entry.origin,
                "key": entry.key,
                "summary": summary,
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline
