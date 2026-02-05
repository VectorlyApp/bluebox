"""
bluebox/llms/data_loaders/interactions_data_loader.py

Data loader for UI interaction events analysis.

Parses JSONL files with UIInteractionEvent entries and provides
structured access to interaction data.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bluebox.data_models.cdp import UIInteractionEvent
from bluebox.llms.data_loaders.abstract_data_loader import AbstractDataLoader
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


@dataclass
class InteractionStats:
    """
    Summary statistics for interaction events.
    """

    total_events: int = 0
    unique_urls: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    unique_elements: int = 0

    def to_summary(self) -> str:
        """
        Generate a human-readable summary of the interaction statistics.

        Returns:
            A string containing the summary.
        """
        lines = [
            f"Total Events: {self.total_events}",
            f"Unique URLs: {self.unique_urls}",
            f"Unique Elements: {self.unique_elements}",
            "",
            "Events by Type:",
        ]
        for event_type, count in sorted(self.events_by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  {event_type}: {count}")
        return "\n".join(lines)


class InteractionsDataLoader(AbstractDataLoader[UIInteractionEvent, InteractionStats]):
    """
    Data loader for UI interaction events.

    Parses JSONL content and provides structured access to interaction data
    including filtering, searching, and summary capabilities.
    """

    ## Magic methods

    def __init__(self, events: list[UIInteractionEvent]) -> None:
        """
        Initialize the interactions data loader.

        Args:
            events: List of UIInteractionEvent objects.
        """
        self._entries: list[UIInteractionEvent] = events
        self._stats: InteractionStats = InteractionStats()
        self._compute_stats()

        logger.debug(
            "InteractionsDataLoader initialized with %d events",
            len(self._entries),
        )

    ## Class methods

    @classmethod
    def from_jsonl(cls, path: str) -> InteractionsDataLoader:
        """
        Create an InteractionsDataLoader from a JSONL file.

        Args:
            path: Path to JSONL file containing UIInteractionEvent entries.

        Returns:
            InteractionsDataLoader instance.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"JSONL file does not exist: {path}")

        events: list[UIInteractionEvent] = []
        with open(file_path, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    event = UIInteractionEvent.model_validate(data)
                    events.append(event)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning("Failed to parse line %d: %s", line_num + 1, e)
                    continue

        return cls(events=events)

    ## Abstract method implementations

    def get_entry_id(self, entry: UIInteractionEvent) -> str:
        """Get unique identifier for an interaction event (uses index)."""
        return str(self._entries.index(entry))

    def get_searchable_content(self, entry: UIInteractionEvent) -> str | None:
        """Get searchable content from an interaction event."""
        parts = []
        el = entry.element
        if el.value:
            parts.append(el.value)
        if el.text:
            parts.append(el.text)
        if el.placeholder:
            parts.append(el.placeholder)
        if el.id:
            parts.append(el.id)
        if el.name:
            parts.append(el.name)
        return " ".join(parts) if parts else None

    def get_entry_url(self, entry: UIInteractionEvent) -> str | None:
        """Get URL associated with an interaction event."""
        return entry.url

    ## Private methods

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from events."""
        type_counts: Counter[str] = Counter()
        urls: set[str] = set()
        element_keys: set[str] = set()

        for event in self._entries:
            type_counts[event.type.value] += 1
            urls.add(event.url)

            #  build a unique key for the element
            el = event.element
            key = el.css_path or f"{el.tag_name}:{el.id or ''}:{el.name or ''}"
            element_keys.add(key)

        self._stats = InteractionStats(
            total_events=len(self._entries),
            unique_urls=len(urls),
            events_by_type=dict(type_counts),
            unique_elements=len(element_keys),
        )

    ## Public methods

    def filter_by_type(self, types: list[str]) -> list[UIInteractionEvent]:
        """
        Filter events by interaction type.

        Args:
            types: List of InteractionType values to include.

        Returns:
            List of matching events.
        """
        types_lower = {t.lower() for t in types}
        return [e for e in self._entries if e.type.value.lower() in types_lower]

    def filter_by_element(
        self,
        tag_name: str | None = None,
        element_id: str | None = None,
        class_name: str | None = None,
        type_attr: str | None = None,
    ) -> list[UIInteractionEvent]:
        """
        Filter events by element attributes.

        Args:
            tag_name: Filter by HTML tag name.
            element_id: Filter by element ID.
            class_name: Filter by CSS class name (substring match in class list).
            type_attr: Filter by input type attribute.

        Returns:
            List of matching events.
        """
        results: list[UIInteractionEvent] = []
        for event in self._entries:
            el = event.element
            if tag_name and el.tag_name.lower() != tag_name.lower():
                continue
            if element_id and el.id != element_id:
                continue
            if class_name:
                if not el.class_names or not any(class_name.lower() in c.lower() for c in el.class_names):
                    continue
            if type_attr and (el.type_attr or "").lower() != type_attr.lower():
                continue
            results.append(event)
        return results

    def get_form_inputs(self) -> list[dict[str, Any]]:
        """
        Get all input/change events with their values and element info.

        Returns:
            List of dicts with value, element tag, element id, element name,
            element type, css_path, and interaction type.
        """
        results: list[dict[str, Any]] = []
        for event in self._entries:
            if event.type.value not in ("input", "change"):
                continue
            el = event.element
            results.append({
                "type": event.type.value,
                "value": el.value,
                "tag_name": el.tag_name,
                "element_id": el.id,
                "element_name": el.name,
                "type_attr": el.type_attr,
                "placeholder": el.placeholder,
                "css_path": el.css_path,
                "url": event.url,
            })
        return results

    def get_unique_elements(self) -> list[dict[str, Any]]:
        """
        Get deduplicated elements with interaction counts and types.

        Returns:
            List of dicts with element info, interaction_count, and interaction_types.
        """
        element_data: dict[str, dict[str, Any]] = {}

        for event in self._entries:
            el = event.element
            key = el.css_path or f"{el.tag_name}:{el.id or ''}:{el.name or ''}"

            if key not in element_data:
                element_data[key] = {
                    "tag_name": el.tag_name,
                    "element_id": el.id,
                    "element_name": el.name,
                    "type_attr": el.type_attr,
                    "css_path": el.css_path,
                    "placeholder": el.placeholder,
                    "interaction_count": 0,
                    "interaction_types": set(),
                }

            element_data[key]["interaction_count"] += 1
            element_data[key]["interaction_types"].add(event.type.value)

        # Convert sets to sorted lists for serialization
        results: list[dict[str, Any]] = []
        for data in sorted(element_data.values(), key=lambda x: -x["interaction_count"]):
            data["interaction_types"] = sorted(data["interaction_types"])
            results.append(data)

        return results

    def get_event_detail(self, index: int) -> dict[str, Any] | None:
        """
        Get full event detail by index.

        Args:
            index: Zero-based index into the events list.

        Returns:
            Full event dict, or None if index is out of range.
        """
        if index < 0 or index >= len(self._entries):
            return None
        return self._entries[index].model_dump()
