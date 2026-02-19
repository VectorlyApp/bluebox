"""
bluebox/llms/data_loaders/dom_data_loader.py

Data loader for DOM snapshot analysis.

Parses JSONL files with DOMSnapshotEvent entries and provides
structured access to page structure data. DOM snapshots capture
full page state on load events, including the document tree and
all text content via a string interning table.
"""

import fnmatch
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from bluebox.data_models.dom import DOMSnapshotEvent
from bluebox.llms.data_loaders.abstract_data_loader import AbstractDataLoader
from bluebox.utils.data_utils import read_jsonl
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# Tag sets for element classification
FORM_INPUT_TAGS = {"INPUT", "SELECT", "TEXTAREA"}
FORM_CONTAINER_TAGS = {"FORM"}
BUTTON_TAGS = {"BUTTON"}
LINK_TAGS = {"A"}
TABLE_TAGS = {"TABLE", "THEAD", "TBODY", "TR", "TH", "TD"}
LIST_TAGS = {"UL", "OL", "LI", "DL", "DT", "DD"}
HEADING_TAGS = {"H1", "H2", "H3", "H4", "H5", "H6"}
META_TAGS = {"META"}
SCRIPT_TAGS = {"SCRIPT"}


@dataclass
class DOMStats:
    """Summary statistics for DOM snapshot data."""

    total_snapshots: int = 0
    unique_urls: int = 0
    unique_titles: int = 0
    urls: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    avg_string_count: float = 0.0
    avg_document_count: float = 0.0
    total_strings: int = 0
    hosts: dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Snapshots: {self.total_snapshots}",
            f"Unique URLs: {self.unique_urls}",
            f"Unique Titles: {self.unique_titles}",
            f"Total Strings Across All Snapshots: {self.total_strings}",
            f"Avg Strings per Snapshot: {self.avg_string_count:.0f}",
            f"Avg Documents per Snapshot: {self.avg_document_count:.1f}",
            "",
            "Pages Captured:",
        ]
        for i, url in enumerate(self.urls):
            title = self.titles[i] if i < len(self.titles) else "N/A"
            lines.append(f"  [{i}] {title} — {url}")

        if self.hosts:
            lines.append("")
            lines.append("Hosts:")
            for host, count in sorted(self.hosts.items(), key=lambda x: -x[1]):
                lines.append(f"  {host}: {count} snapshot(s)")

        return "\n".join(lines)


class DOMDataLoader(AbstractDataLoader[DOMSnapshotEvent, DOMStats]):
    """
    Data loader for DOM snapshots.

    Provides access to page structure captured via CDP DOMSnapshot.captureSnapshot.
    Each entry is a full page snapshot taken on page load, containing the document
    tree structure and a string interning table with all text content.
    """

    def __init__(self, jsonl_path: str) -> None:
        """
        Initialize the DOMDataLoader from a JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing DOMSnapshotEvent entries.

        Raises:
            FileNotFoundError: If jsonl_path does not exist.
        """
        if not Path(jsonl_path).exists():
            raise FileNotFoundError(f"DOM data file not found: {jsonl_path}")

        self._entries: list[DOMSnapshotEvent] = []
        self._stats: DOMStats = DOMStats()

        for _line_num, data in read_jsonl(jsonl_path):
            try:
                event = DOMSnapshotEvent.model_validate(data)
                self._entries.append(event)
            except ValueError as e:
                logger.warning("Failed to validate DOM snapshot: %s", e)
                continue

        self._compute_stats()

        logger.debug(
            "DOMDataLoader initialized with %d snapshots",
            len(self._entries),
        )

    # Abstract method implementations

    def get_entry_id(self, entry: DOMSnapshotEvent) -> str:
        """Get unique identifier for a DOM snapshot (uses index)."""
        return str(self._entries.index(entry))

    def get_searchable_content(self, entry: DOMSnapshotEvent) -> str | None:
        """
        Get searchable content from a DOM snapshot.

        Uses the string interning table which contains all text content,
        node names, attribute values, etc.
        """
        if not entry.strings:
            return None
        return " ".join(entry.strings)

    def get_entry_url(self, entry: DOMSnapshotEvent) -> str | None:
        """Get URL associated with a DOM snapshot."""
        return entry.url

    # Private methods

    def _compute_stats(self) -> None:
        """Compute summary statistics from snapshots."""
        urls: list[str] = []
        titles: list[str] = []
        unique_urls: set[str] = set()
        unique_titles: set[str] = set()
        hosts: Counter[str] = Counter()
        total_strings = 0
        total_documents = 0

        for entry in self._entries:
            urls.append(entry.url)
            titles.append(entry.title or "")
            unique_urls.add(entry.url)
            if entry.title:
                unique_titles.add(entry.title)
            host = urlparse(entry.url).netloc
            hosts[host] += 1
            total_strings += len(entry.strings)
            total_documents += len(entry.documents)

        n = len(self._entries) or 1
        self._stats = DOMStats(
            total_snapshots=len(self._entries),
            unique_urls=len(unique_urls),
            unique_titles=len(unique_titles),
            urls=urls,
            titles=titles,
            avg_string_count=total_strings / n,
            avg_document_count=total_documents / n,
            total_strings=total_strings,
            hosts=dict(hosts),
        )

    # Public methods — Snapshot retrieval

    def get_snapshot(self, index: int) -> DOMSnapshotEvent | None:
        """
        Get a snapshot by index.

        Args:
            index: Zero-based index into the snapshots list.

        Returns:
            The DOMSnapshotEvent, or None if index is out of range.
        """
        if index < 0 or index >= len(self._entries):
            return None
        return self._entries[index]

    def get_snapshots_by_url(self, url: str) -> list[DOMSnapshotEvent]:
        """
        Get all snapshots for a given URL (substring match).

        Args:
            url: URL substring to match.

        Returns:
            List of matching DOMSnapshotEvent objects.
        """
        return [e for e in self._entries if url in e.url]

    def get_snapshots_by_url_pattern(self, pattern: str) -> list[DOMSnapshotEvent]:
        """
        Get all snapshots whose URLs match a glob pattern.

        Args:
            pattern: Glob pattern to match URLs (e.g., "*example.com*", "*/search*").

        Returns:
            List of matching DOMSnapshotEvent objects.
        """
        return [e for e in self._entries if fnmatch.fnmatch(e.url, pattern)]

    def get_snapshots_by_host(self, host: str) -> list[DOMSnapshotEvent]:
        """
        Get all snapshots from a specific host.

        Args:
            host: Hostname to filter by (exact match on netloc).

        Returns:
            List of matching DOMSnapshotEvent objects.
        """
        return [e for e in self._entries if urlparse(e.url).netloc == host]

    # Public methods — Page-level summaries

    def get_page_titles(self) -> list[dict[str, Any]]:
        """
        Get all captured page titles with their URLs.

        Returns:
            List of dicts with index, url, and title.
        """
        return [
            {"index": idx, "url": entry.url, "title": entry.title}
            for idx, entry in enumerate(self._entries)
        ]

    def list_pages(self) -> list[dict[str, Any]]:
        """
        List all captured pages with summary info.

        Returns:
            List of dicts with index, url, title, string_count,
            document_count, and timestamp.
        """
        return [
            {
                "index": idx,
                "url": entry.url,
                "title": entry.title,
                "string_count": len(entry.strings),
                "document_count": len(entry.documents),
                "timestamp": entry.timestamp,
            }
            for idx, entry in enumerate(self._entries)
        ]

    # Public methods — String table analysis

    def get_text_content(self, index: int, max_chars: int = 5000) -> str | None:
        """
        Get the text content from a snapshot's string table.

        Filters out short strings (likely node/attribute names) and returns
        the longer strings which are more likely to be visible text content.

        Args:
            index: Zero-based snapshot index.
            max_chars: Maximum characters to return.

        Returns:
            Concatenated text content, or None if index is invalid.
        """
        snapshot = self.get_snapshot(index)
        if snapshot is None:
            return None

        # Filter to strings that are likely visible text (not tag names, short attrs)
        text_strings = [s for s in snapshot.strings if len(s) > 3 and not s.startswith("#")]
        content = "\n".join(text_strings)

        if len(content) > max_chars:
            content = content[:max_chars] + "..."

        return content

    def search_strings(
        self,
        value: str,
        case_sensitive: bool = False,
        snapshot_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search the string tables across all snapshots for a value.

        Args:
            value: The string to search for (substring match).
            case_sensitive: Whether the search is case-sensitive.
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, matching_strings (list of
            matching strings from the table), and match_count.
        """
        if not value:
            return []

        search_value = value if case_sensitive else value.lower()
        results: list[dict[str, Any]] = []

        entries_to_search: list[tuple[int, DOMSnapshotEvent]] = []
        if snapshot_index is not None:
            snapshot = self.get_snapshot(snapshot_index)
            if snapshot:
                entries_to_search = [(snapshot_index, snapshot)]
        else:
            entries_to_search = list(enumerate(self._entries))

        for idx, entry in entries_to_search:
            matching_strings: list[str] = []
            for s in entry.strings:
                compare_s = s if case_sensitive else s.lower()
                if search_value in compare_s:
                    matching_strings.append(s)

            if matching_strings:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "matching_strings": matching_strings[:50],  # cap to avoid huge results
                    "match_count": len(matching_strings),
                })

        return results

    # Private methods — Node tree walking

    def _resolve_string(self, strings: list[str], index: int) -> str | None:
        """Resolve a string index to its value, or None if invalid."""
        if 0 <= index < len(strings):
            return strings[index]
        return None

    def _get_node_attrs(self, nodes: dict[str, Any], node_index: int, strings: list[str]) -> dict[str, str]:
        """Get attributes for a node as a name→value dict."""
        attrs_list = nodes.get("attributes", [])[node_index]
        result: dict[str, str] = {}
        for j in range(0, len(attrs_list), 2):
            if j + 1 < len(attrs_list):
                name = self._resolve_string(strings, attrs_list[j])
                val = self._resolve_string(strings, attrs_list[j + 1])
                if name is not None and val is not None:
                    result[name] = val
        return result

    def _get_input_value(self, nodes: dict[str, Any], node_index: int, strings: list[str]) -> str | None:
        """Get the inputValue for a node, if any."""
        input_val_data = nodes.get("inputValue", {})
        indices = input_val_data.get("index", [])
        values = input_val_data.get("value", [])
        if node_index in indices:
            val_idx = values[indices.index(node_index)]
            return self._resolve_string(strings, val_idx)
        return None

    def _is_clickable(self, nodes: dict[str, Any], node_index: int) -> bool:
        """Check if a node is marked as clickable."""
        return node_index in nodes.get("isClickable", {}).get("index", [])

    def _get_child_text(
        self,
        nodes: dict[str, Any],
        parent_index: int,
        strings: list[str],
    ) -> str | None:
        """
        Get concatenated text content from direct child #text nodes.

        Args:
            nodes: The nodes dict from the document.
            parent_index: Index of the parent node.
            strings: The string interning table.

        Returns:
            Concatenated text, or None if no text children found.
        """
        node_names = nodes.get("nodeName", [])
        parent_indices = nodes.get("parentIndex", [])
        node_values = nodes.get("nodeValue", [])

        text_parts: list[str] = []
        for j, child_name_idx in enumerate(node_names):
            if parent_indices[j] == parent_index:
                child_tag = self._resolve_string(strings, child_name_idx)
                if child_tag == "#text":
                    val_idx = node_values[j] if j < len(node_values) else -1
                    text = self._resolve_string(strings, val_idx)
                    if text:
                        text_parts.append(text)

        return " ".join(text_parts) if text_parts else None

    def _get_entries_to_search(
        self, snapshot_index: int | None,
    ) -> list[tuple[int, DOMSnapshotEvent]]:
        """Get the list of (index, snapshot) pairs to search."""
        if snapshot_index is not None:
            snapshot = self.get_snapshot(snapshot_index)
            if snapshot:
                return [(snapshot_index, snapshot)]
            return []
        return list(enumerate(self._entries))

    def _extract_elements_by_tag(
        self,
        entry: DOMSnapshotEvent,
        tag_names: set[str],
        doc_index: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Walk the node tree for a document and extract elements matching tag names.

        Args:
            entry: The snapshot to search.
            tag_names: Set of uppercase tag names to match.
            doc_index: Which document in the snapshot (0 = main frame).

        Returns:
            List of element dicts with tag, node_index, attrs, input_value, is_clickable.
        """
        if doc_index >= len(entry.documents):
            return []

        doc = entry.documents[doc_index]
        nodes = doc.get("nodes", {})
        node_names = nodes.get("nodeName", [])
        strings = entry.strings
        results: list[dict[str, Any]] = []

        for i, name_idx in enumerate(node_names):
            tag = self._resolve_string(strings, name_idx)
            if tag is None or tag.upper() not in tag_names:
                continue

            attrs = self._get_node_attrs(nodes, i, strings)
            input_value = self._get_input_value(nodes, i, strings)
            clickable = self._is_clickable(nodes, i)

            element: dict[str, Any] = {
                "tag": tag.upper(),
                "node_index": i,
                "attrs": attrs,
                "is_clickable": clickable,
            }
            if input_value is not None:
                element["input_value"] = input_value

            results.append(element)

        return results

    # Public methods — Element extraction (tree-walking)

    def get_inputs(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all input fields (INPUT, SELECT, TEXTAREA) from snapshots.

        Walks the actual node tree — not just the string table — to extract
        real elements with their attributes, types, names, and current values.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and elements. Each element
            has tag, attrs (name, type, placeholder, value, etc.), input_value,
            and is_clickable.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            elements = self._extract_elements_by_tag(entry, FORM_INPUT_TAGS)
            if elements:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "elements": elements,
                })
        return results

    def get_buttons(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all buttons from snapshots.

        Includes both <button> elements and <input type="submit">.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and elements.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            buttons = self._extract_elements_by_tag(entry, BUTTON_TAGS)

            # Also include <input type="submit"> and <input type="button">
            inputs = self._extract_elements_by_tag(entry, {"INPUT"})
            for inp in inputs:
                input_type = inp["attrs"].get("type", "").lower()
                if input_type in ("submit", "button"):
                    buttons.append(inp)

            if buttons:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "elements": buttons,
                })
        return results

    def get_links(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all anchor links (<a>) from snapshots with their href values.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and elements. Each element
            has tag, attrs (including href), and is_clickable.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            elements = self._extract_elements_by_tag(entry, LINK_TAGS)
            if elements:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "elements": elements,
                })
        return results

    def get_forms(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all <form> elements with their action, method, and child inputs.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and forms. Each form has
            attrs (action, method, id) and a list of child input elements.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            if not entry.documents:
                continue
            doc = entry.documents[0]
            nodes = doc.get("nodes", {})
            node_names = nodes.get("nodeName", [])
            parent_indices = nodes.get("parentIndex", [])
            strings = entry.strings

            forms: list[dict[str, Any]] = []

            # Find all FORM nodes
            form_node_indices: list[int] = []
            for i, name_idx in enumerate(node_names):
                tag = self._resolve_string(strings, name_idx)
                if tag and tag.upper() == "FORM":
                    form_node_indices.append(i)

            for form_idx in form_node_indices:
                form_attrs = self._get_node_attrs(nodes, form_idx, strings)

                # Find child input elements (walk descendants)
                child_inputs: list[dict[str, Any]] = []
                for i, name_idx in enumerate(node_names):
                    tag = self._resolve_string(strings, name_idx)
                    if tag is None or tag.upper() not in FORM_INPUT_TAGS | BUTTON_TAGS:
                        continue

                    # Check if this node is a descendant of the form
                    current = i
                    is_descendant = False
                    depth = 0
                    while current != -1 and depth < 100:
                        if current == form_idx:
                            is_descendant = True
                            break
                        current = parent_indices[current] if current < len(parent_indices) else -1
                        depth += 1

                    if is_descendant:
                        attrs = self._get_node_attrs(nodes, i, strings)
                        input_value = self._get_input_value(nodes, i, strings)
                        element: dict[str, Any] = {
                            "tag": tag.upper(),
                            "node_index": i,
                            "attrs": attrs,
                        }
                        if input_value is not None:
                            element["input_value"] = input_value
                        child_inputs.append(element)

                forms.append({
                    "node_index": form_idx,
                    "attrs": form_attrs,
                    "inputs": child_inputs,
                })

            if forms:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "forms": forms,
                })
        return results

    def get_tables(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all <table> elements with their headers and row counts.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and tables. Each table has
            node_index, attrs, headers (list of header text), and row_count.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            if not entry.documents:
                continue
            doc = entry.documents[0]
            nodes = doc.get("nodes", {})
            node_names = nodes.get("nodeName", [])
            parent_indices = nodes.get("parentIndex", [])
            node_values = nodes.get("nodeValue", [])
            strings = entry.strings

            tables: list[dict[str, Any]] = []

            # Find TABLE nodes
            for i, name_idx in enumerate(node_names):
                tag = self._resolve_string(strings, name_idx)
                if tag is None or tag.upper() != "TABLE":
                    continue

                table_attrs = self._get_node_attrs(nodes, i, strings)

                # Find TH and TR descendants
                headers: list[str] = []
                row_count = 0

                for j, child_name_idx in enumerate(node_names):
                    child_tag = self._resolve_string(strings, child_name_idx)
                    if child_tag is None:
                        continue

                    # Check if descendant of this table
                    current = j
                    is_descendant = False
                    depth = 0
                    while current != -1 and depth < 100:
                        if current == i:
                            is_descendant = True
                            break
                        current = parent_indices[current] if current < len(parent_indices) else -1
                        depth += 1

                    if not is_descendant:
                        continue

                    if child_tag.upper() == "TR":
                        row_count += 1
                    elif child_tag.upper() == "TH":
                        # Get text content of the TH (look for text node children)
                        for k, grand_name_idx in enumerate(node_names):
                            if parent_indices[k] == j:
                                grand_tag = self._resolve_string(strings, grand_name_idx)
                                if grand_tag == "#text":
                                    val_idx = node_values[k] if k < len(node_values) else -1
                                    text = self._resolve_string(strings, val_idx)
                                    if text:
                                        headers.append(text)

                tables.append({
                    "node_index": i,
                    "attrs": table_attrs,
                    "headers": headers,
                    "row_count": row_count,
                })

            if tables:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "tables": tables,
                })
        return results

    def get_headings(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all heading elements (H1-H6) with their text content.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and headings. Each heading
            has tag (H1-H6), node_index, and text.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            if not entry.documents:
                continue
            doc = entry.documents[0]
            nodes = doc.get("nodes", {})
            node_names = nodes.get("nodeName", [])
            strings = entry.strings

            headings: list[dict[str, Any]] = []

            for i, name_idx in enumerate(node_names):
                tag = self._resolve_string(strings, name_idx)
                if tag is None or tag.upper() not in HEADING_TAGS:
                    continue

                headings.append({
                    "tag": tag.upper(),
                    "node_index": i,
                    "text": self._get_child_text(nodes, i, strings),
                })

            if headings:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "headings": headings,
                })
        return results

    def get_clickable_elements(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all elements marked as clickable by the browser.

        CDP marks elements as clickable based on event listeners and
        default behavior. This catches buttons, links, and any element
        with click handlers.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and elements. Each element
            has tag, node_index, and attrs.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            if not entry.documents:
                continue
            doc = entry.documents[0]
            nodes = doc.get("nodes", {})
            node_names = nodes.get("nodeName", [])
            clickable_indices = set(nodes.get("isClickable", {}).get("index", []))
            strings = entry.strings

            elements: list[dict[str, Any]] = []
            for i in clickable_indices:
                if i >= len(node_names):
                    continue
                tag = self._resolve_string(strings, node_names[i])
                if tag is None or tag.startswith("#"):
                    continue  # Skip #document, #text, etc.

                attrs = self._get_node_attrs(nodes, i, strings)
                elements.append({
                    "tag": tag.upper(),
                    "node_index": i,
                    "attrs": attrs,
                })

            if elements:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "elements": elements,
                })
        return results

    def get_meta_tags(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all <meta> elements with their attributes.

        Meta tags carry CSRF tokens, API endpoints, OG tags, verification
        keys, viewport config, and other page metadata.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and meta_tags. Each meta
            tag has node_index and attrs (name, content, property, charset, etc.).
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            elements = self._extract_elements_by_tag(entry, META_TAGS)
            if elements:
                # Simplify: meta tags don't need is_clickable or input_value
                meta_tags = [
                    {"node_index": el["node_index"], "attrs": el["attrs"]}
                    for el in elements
                ]
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "meta_tags": meta_tags,
                })
        return results

    def get_scripts(
        self,
        snapshot_index: int | None = None,
        max_inline_chars: int = 2000,
    ) -> list[dict[str, Any]]:
        """
        Get all <script> elements with their attributes and inline content.

        Extracts both external scripts (src attribute) and inline scripts
        (text content). Inline content is truncated to max_inline_chars.

        Useful for finding:
        - Framework data blobs: __NEXT_DATA__, __NUXT__, window.__INITIAL_STATE__
        - Embedded JSON: <script type="application/json">
        - Inline configuration: GTM, analytics, consent management
        - Structured data: <script type="application/ld+json">

        Args:
            snapshot_index: If provided, only search this specific snapshot.
            max_inline_chars: Max characters for inline script content (default 2000).

        Returns:
            List of dicts with snapshot_index, url, and scripts. Each script
            has node_index, attrs, and inline_content (if any).
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            if not entry.documents:
                continue
            doc = entry.documents[0]
            nodes = doc.get("nodes", {})
            node_names = nodes.get("nodeName", [])
            strings = entry.strings

            scripts: list[dict[str, Any]] = []

            for i, name_idx in enumerate(node_names):
                tag = self._resolve_string(strings, name_idx)
                if tag is None or tag.upper() != "SCRIPT":
                    continue

                attrs = self._get_node_attrs(nodes, i, strings)
                inline_content = self._get_child_text(nodes, i, strings)

                script: dict[str, Any] = {
                    "node_index": i,
                    "attrs": attrs,
                }

                if inline_content:
                    if len(inline_content) > max_inline_chars:
                        script["inline_content"] = inline_content[:max_inline_chars] + "..."
                        script["inline_content_truncated"] = True
                        script["inline_content_full_length"] = len(inline_content)
                    else:
                        script["inline_content"] = inline_content

                scripts.append(script)

            if scripts:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "scripts": scripts,
                })
        return results

    def get_hidden_inputs(self, snapshot_index: int | None = None) -> list[dict[str, Any]]:
        """
        Get all <input type="hidden"> elements.

        Hidden inputs often carry CSRF tokens, session IDs, form tokens,
        and other security-relevant values that are submitted with forms.

        Args:
            snapshot_index: If provided, only search this specific snapshot.

        Returns:
            List of dicts with snapshot_index, url, and elements. Each element
            has tag, node_index, attrs (name, value, etc.), and input_value.
        """
        results: list[dict[str, Any]] = []
        for idx, entry in self._get_entries_to_search(snapshot_index):
            all_inputs = self._extract_elements_by_tag(entry, {"INPUT"})
            hidden = [
                el for el in all_inputs
                if el["attrs"].get("type", "").lower() == "hidden"
            ]
            if hidden:
                results.append({
                    "snapshot_index": idx,
                    "url": entry.url,
                    "elements": hidden,
                })
        return results

    def get_snapshot_diff(self, index_a: int, index_b: int) -> dict[str, Any] | None:
        """
        Compare two snapshots and return strings that were added or removed.

        Useful for understanding what changed between page loads.

        Args:
            index_a: Index of the first (earlier) snapshot.
            index_b: Index of the second (later) snapshot.

        Returns:
            Dict with added (strings in B but not A), removed (strings in A
            but not B), and shared count. None if either index is invalid.
        """
        snapshot_a = self.get_snapshot(index_a)
        snapshot_b = self.get_snapshot(index_b)

        if snapshot_a is None or snapshot_b is None:
            return None

        strings_a = set(snapshot_a.strings)
        strings_b = set(snapshot_b.strings)

        added = sorted(strings_b - strings_a)
        removed = sorted(strings_a - strings_b)

        return {
            "snapshot_a": {"index": index_a, "url": snapshot_a.url},
            "snapshot_b": {"index": index_b, "url": snapshot_b.url},
            "added": added,
            "removed": removed,
            "added_count": len(added),
            "removed_count": len(removed),
            "shared_count": len(strings_a & strings_b),
        }

    def get_navigation_sequence(self) -> list[dict[str, Any]]:
        """
        Get the ordered sequence of page navigations from snapshots.

        Returns:
            List of dicts with index, url, title, host, and timestamp,
            ordered by timestamp.
        """
        sequence = []
        for idx, entry in enumerate(self._entries):
            sequence.append({
                "index": idx,
                "url": entry.url,
                "title": entry.title,
                "host": urlparse(entry.url).netloc,
                "timestamp": entry.timestamp,
            })

        sequence.sort(key=lambda x: x["timestamp"])
        return sequence
