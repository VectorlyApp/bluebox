"""
bluebox/llms/data_loaders/window_property_data_loader.py

Data loader for window property event analysis.

Parses JSONL files with WindowPropertyEvent entries and provides
methods for token tracing - finding where values originated from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bluebox.data_models.cdp import (
    WindowPropertyChangeType,
    WindowPropertyEvent,
)
from bluebox.llms.data_loaders.abstract_data_loader import AbstractDataLoader
from bluebox.utils.data_utils import read_jsonl
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


@dataclass
class WindowPropertyStats:
    """Summary statistics for window property events."""

    total_events: int = 0
    total_changes: int = 0
    changes_added: int = 0
    changes_changed: int = 0
    changes_deleted: int = 0
    unique_urls: int = 0
    unique_property_paths: int = 0

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        return "\n".join([
            f"Total Events: {self.total_events}",
            f"Total Changes: {self.total_changes}",
            f"  Added: {self.changes_added}",
            f"  Changed: {self.changes_changed}",
            f"  Deleted: {self.changes_deleted}",
            f"Unique URLs: {self.unique_urls}",
            f"Unique Property Paths: {self.unique_property_paths}",
        ])


class WindowPropertyDataLoader(AbstractDataLoader[WindowPropertyEvent, WindowPropertyStats]):
    """
    Data loader for window property events.

    Focused on token tracing - finding where values came from
    in window object properties.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the WindowPropertyDataLoader from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing WindowPropertyEvent entries.

        Raises:
            FileNotFoundError: If jsonl_path does not exist.
        """
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(f"Window property data file not found: {jsonl_path}")

        self._entries: list[WindowPropertyEvent] = []
        self._entry_index: dict[int, WindowPropertyEvent] = {}

        for line_num, data in read_jsonl(jsonl_path):
            try:
                event = WindowPropertyEvent.model_validate(data)
                self._entries.append(event)
                self._entry_index[line_num] = event
            except ValueError as e:
                logger.warning("Failed to validate line %d: %s", line_num + 1, e)
                continue

        self._compute_stats()

        logger.debug(
            "WindowPropertyDataLoader initialized with %d events",
            len(self._entries),
        )

    ## Abstract method implementations

    def get_entry_id(self, entry: WindowPropertyEvent) -> str:
        """Get unique identifier for a window property event (uses index)."""
        return str(self._entries.index(entry))

    def get_searchable_content(self, entry: WindowPropertyEvent) -> str | None:
        """Get searchable content from a window property event."""
        parts = []
        for change in entry.changes:
            if change.value is not None:
                parts.append(str(change.value))
            parts.append(change.path)
        return " ".join(parts) if parts else None

    def get_entry_url(self, entry: WindowPropertyEvent) -> str | None:
        """Get URL associated with a window property event."""
        return entry.url

    ## Private methods

    def _compute_stats(self) -> None:
        """Compute summary statistics."""
        urls: set[str] = set()
        paths: set[str] = set()
        total_changes = 0
        added = 0
        changed = 0
        deleted = 0

        for entry in self._entries:
            urls.add(entry.url)
            for change in entry.changes:
                total_changes += 1
                paths.add(change.path)
                if change.change_type == WindowPropertyChangeType.ADDED:
                    added += 1
                elif change.change_type == WindowPropertyChangeType.CHANGED:
                    changed += 1
                elif change.change_type == WindowPropertyChangeType.DELETED:
                    deleted += 1

        self._stats = WindowPropertyStats(
            total_events=len(self._entries),
            total_changes=total_changes,
            changes_added=added,
            changes_changed=changed,
            changes_deleted=deleted,
            unique_urls=len(urls),
            unique_property_paths=len(paths),
        )

    ## Public methods

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
            - path: Property path
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

    def search_values(self, value: str, case_sensitive: bool = False) -> list[dict[str, Any]]:
        """
        Search property values for a given string (token tracing).

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
