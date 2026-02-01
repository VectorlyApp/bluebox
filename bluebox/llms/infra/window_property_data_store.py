"""
bluebox/llms/infra/window_property_data_store.py

Data store for window property event analysis.

Parses JSONL files with WindowPropertyEvent entries and provides
structured access to window property change data.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bluebox.data_models.cdp import (
    WindowPropertyChangeType,
    WindowPropertyEvent,
)
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


@dataclass
class WindowPropertyStats:
    """Summary statistics for a window property events file."""

    total_events: int = 0
    total_changes: int = 0

    # Change type counts
    changes_added: int = 0
    changes_changed: int = 0
    changes_deleted: int = 0

    # URL tracking
    urls: dict[str, int] = field(default_factory=dict)
    unique_urls: int = 0

    # Property path tracking
    property_paths: dict[str, int] = field(default_factory=dict)
    unique_property_paths: int = 0

    # Top-level property tracking (first segment of path)
    top_level_properties: dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Events: {self.total_events}",
            f"Total Changes: {self.total_changes}",
            "",
            "Change Types:",
            f"  Added: {self.changes_added}",
            f"  Changed: {self.changes_changed}",
            f"  Deleted: {self.changes_deleted}",
            "",
            f"Unique URLs: {self.unique_urls}",
            f"Unique Property Paths: {self.unique_property_paths}",
        ]

        if self.urls:
            lines.append("")
            lines.append("Top URLs:")
            for url, count in sorted(self.urls.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"  {url}: {count}")

        if self.top_level_properties:
            lines.append("")
            lines.append("Top-Level Properties:")
            for prop, count in sorted(self.top_level_properties.items(), key=lambda x: -x[1])[:15]:
                lines.append(f"  {prop}: {count}")

        if self.property_paths:
            lines.append("")
            lines.append("Most Changed Paths:")
            for path, count in sorted(self.property_paths.items(), key=lambda x: -x[1])[:15]:
                lines.append(f"  {path}: {count}")

        return "\n".join(lines)


class WindowPropertyDataStore:
    """
    Data store for window property event analysis.

    Parses JSONL content and provides structured access to window property
    change events, tracking additions, modifications, and deletions.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the WindowPropertyDataStore from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing WindowPropertyEvent entries.
        """
        self._entries: list[WindowPropertyEvent] = []
        self._entry_index: dict[int, WindowPropertyEvent] = {}  # index -> event
        self._stats: WindowPropertyStats = WindowPropertyStats()

        path = Path(jsonl_path)
        if not path.exists():
            raise ValueError(f"JSONL file does not exist: {jsonl_path}")

        # Load entries from JSONL
        with open(path, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = WindowPropertyEvent.model_validate(data)
                    self._entries.append(event)
                    self._entry_index[line_num] = event
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse line %d: %s", line_num + 1, e)
                    continue

        self._compute_stats()

        logger.info(
            "WindowPropertyDataStore initialized with %d events",
            len(self._entries),
        )

    @property
    def entries(self) -> list[WindowPropertyEvent]:
        """Return all window property events."""
        return self._entries

    @property
    def stats(self) -> WindowPropertyStats:
        """Return computed statistics."""
        return self._stats

    @property
    def raw_data(self) -> dict[str, Any]:
        """Return entries as a dict for compatibility."""
        return {"entries": [e.model_dump() for e in self._entries]}

    @property
    def all_property_paths(self) -> list[str]:
        """Return all unique property paths that have changed, sorted alphabetically."""
        paths: set[str] = set()
        for entry in self._entries:
            for change in entry.changes:
                paths.add(change.path)
        return sorted(paths)

    @property
    def top_level_properties(self) -> list[str]:
        """Return unique top-level properties (first segment of path), sorted alphabetically."""
        props: set[str] = set()
        for entry in self._entries:
            for change in entry.changes:
                top_level = change.path.split(".")[0]
                props.add(top_level)
        return sorted(props)

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        urls: Counter[str] = Counter()
        property_paths: Counter[str] = Counter()
        top_level_props: Counter[str] = Counter()

        total_changes = 0
        added_count = 0
        changed_count = 0
        deleted_count = 0

        for entry in self._entries:
            urls[entry.url] += 1

            for change in entry.changes:
                total_changes += 1
                property_paths[change.path] += 1

                # Extract top-level property
                top_level = change.path.split(".")[0]
                top_level_props[top_level] += 1

                # Count by change type
                if change.change_type == WindowPropertyChangeType.ADDED:
                    added_count += 1
                elif change.change_type == WindowPropertyChangeType.CHANGED:
                    changed_count += 1
                elif change.change_type == WindowPropertyChangeType.DELETED:
                    deleted_count += 1

        self._stats = WindowPropertyStats(
            total_events=len(self._entries),
            total_changes=total_changes,
            changes_added=added_count,
            changes_changed=changed_count,
            changes_deleted=deleted_count,
            urls=dict(urls),
            unique_urls=len(urls),
            property_paths=dict(property_paths),
            unique_property_paths=len(property_paths),
            top_level_properties=dict(top_level_props),
        )

    def search_entries(
        self,
        url_contains: str | None = None,
        path_contains: str | None = None,
        change_type: WindowPropertyChangeType | None = None,
        has_changes: bool | None = None,
    ) -> list[WindowPropertyEvent]:
        """
        Search entries with filters.

        Args:
            url_contains: Filter by URL containing substring.
            path_contains: Filter by any change path containing substring.
            change_type: Filter by change type (events must have at least one change of this type).
            has_changes: Filter by whether the event has any changes.

        Returns:
            List of matching WindowPropertyEvent objects.
        """
        results = []

        for entry in self._entries:
            if url_contains and url_contains.lower() not in entry.url.lower():
                continue

            if has_changes is not None:
                if has_changes and not entry.changes:
                    continue
                if not has_changes and entry.changes:
                    continue

            if path_contains:
                path_match = any(
                    path_contains.lower() in change.path.lower()
                    for change in entry.changes
                )
                if not path_match:
                    continue

            if change_type:
                type_match = any(
                    change.change_type == change_type
                    for change in entry.changes
                )
                if not type_match:
                    continue

            results.append(entry)

        return results

    def get_entry(self, index: int) -> WindowPropertyEvent | None:
        """Get entry by index."""
        return self._entry_index.get(index)

    def get_changes_by_path(self, path: str, exact: bool = True) -> list[dict[str, Any]]:
        """
        Get all changes for a specific property path.

        Args:
            path: The property path to filter by.
            exact: If True, match exactly. If False, match paths containing the substring.

        Returns:
            List of dicts containing:
            - index: Event index
            - timestamp: Event timestamp
            - url: Page URL
            - change_type: Type of change
            - value: New value (or None for deletions)
        """
        results = []

        for idx, entry in enumerate(self._entries):
            for change in entry.changes:
                if exact:
                    if change.path != path:
                        continue
                else:
                    if path.lower() not in change.path.lower():
                        continue

                results.append({
                    "index": idx,
                    "timestamp": entry.timestamp,
                    "url": entry.url,
                    "path": change.path,
                    "change_type": change.change_type,
                    "value": change.value,
                })

        return results

    def get_changes_by_type(
        self,
        change_type: WindowPropertyChangeType,
    ) -> list[dict[str, Any]]:
        """
        Get all changes of a specific type.

        Args:
            change_type: The type of change to filter by.

        Returns:
            List of dicts containing change details.
        """
        results = []

        for idx, entry in enumerate(self._entries):
            for change in entry.changes:
                if change.change_type == change_type:
                    results.append({
                        "index": idx,
                        "timestamp": entry.timestamp,
                        "url": entry.url,
                        "path": change.path,
                        "value": change.value,
                    })

        return results

    def get_url_stats(self, url_filter: str | None = None) -> list[dict[str, Any]]:
        """
        Get per-URL summary statistics.

        Args:
            url_filter: Optional substring to filter URLs (case-insensitive).

        Returns:
            List of dicts sorted by event count descending, each containing:
            - url: The URL
            - event_count: Number of events for this URL
            - total_changes: Total changes across all events
            - change_types: Dict of change type counts
        """
        url_data: dict[str, dict[str, Any]] = {}

        for entry in self._entries:
            url = entry.url

            # Apply filter if provided
            if url_filter and url_filter.lower() not in url.lower():
                continue

            if url not in url_data:
                url_data[url] = {
                    "event_count": 0,
                    "total_changes": 0,
                    "change_types": Counter(),
                }

            url_data[url]["event_count"] += 1
            url_data[url]["total_changes"] += len(entry.changes)

            for change in entry.changes:
                url_data[url]["change_types"][change.change_type] += 1

        results = []
        for url, data in sorted(url_data.items(), key=lambda x: -x[1]["event_count"]):
            results.append({
                "url": url,
                "event_count": data["event_count"],
                "total_changes": data["total_changes"],
                "change_types": dict(data["change_types"]),
            })

        return results

    def search_values(
        self,
        value: str,
        case_sensitive: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Search property values for a given string.

        Args:
            value: The value to search for.
            case_sensitive: Whether the search should be case-sensitive.

        Returns:
            List of dicts containing:
            - index: Event index
            - timestamp: Event timestamp
            - url: Page URL
            - path: Property path
            - change_type: Type of change
            - value: The matching value
        """
        results: list[dict[str, Any]] = []

        if not value:
            return results

        search_value = value if case_sensitive else value.lower()

        for idx, entry in enumerate(self._entries):
            for change in entry.changes:
                if change.value is None:
                    continue

                val_str = str(change.value) if case_sensitive else str(change.value).lower()
                if search_value in val_str:
                    results.append({
                        "index": idx,
                        "timestamp": entry.timestamp,
                        "url": entry.url,
                        "path": change.path,
                        "change_type": change.change_type,
                        "value": change.value,
                    })

        return results

    def get_property_timeline(self, path: str) -> list[dict[str, Any]]:
        """
        Get a timeline of changes for a specific property path.

        Args:
            path: The property path to track.

        Returns:
            List of dicts ordered by timestamp with:
            - timestamp: Event timestamp
            - url: Page URL
            - change_type: Type of change
            - value: New value (or None for deletions)
        """
        timeline = []

        for entry in self._entries:
            for change in entry.changes:
                if change.path == path:
                    timeline.append({
                        "timestamp": entry.timestamp,
                        "url": entry.url,
                        "change_type": change.change_type,
                        "value": change.value,
                    })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def get_events_timeline(self) -> list[dict[str, Any]]:
        """
        Get a timeline of all events ordered by timestamp.

        Returns:
            List of dicts with:
            - timestamp: Event timestamp
            - url: Page URL
            - total_changes: Number of changes in this event
            - added: Number of added properties
            - changed: Number of changed properties
            - deleted: Number of deleted properties
        """
        timeline = []

        for entry in self._entries:
            added = sum(1 for c in entry.changes if c.change_type == WindowPropertyChangeType.ADDED)
            changed = sum(1 for c in entry.changes if c.change_type == WindowPropertyChangeType.CHANGED)
            deleted = sum(1 for c in entry.changes if c.change_type == WindowPropertyChangeType.DELETED)

            timeline.append({
                "timestamp": entry.timestamp,
                "url": entry.url,
                "total_changes": len(entry.changes),
                "added": added,
                "changed": changed,
                "deleted": deleted,
            })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])

        return timeline
