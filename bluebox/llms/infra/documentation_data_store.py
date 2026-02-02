"""
bluebox/llms/infra/documentation_data_store.py

Data store for documentation and code file analysis.

Replaces the OpenAI vectorstore-based approach with local indexing
and search capabilities for documentation and code files.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from bluebox.utils.data_utils import (
    format_bytes,
    parse_markdown_summary,
    parse_markdown_title,
    parse_python_docstring,
)
from bluebox.utils.infra_utils import resolve_glob_patterns
from bluebox.utils.logger import get_logger


logger = get_logger(name=__name__)


class FileType(StrEnum):
    """Types of files in the documentation data store."""

    DOCUMENTATION = "documentation"
    CODE = "code"


@dataclass
class FileEntry:
    """Represents a single file in the data store."""

    path: Path
    file_type: FileType
    content: str
    summary: str | None = None  # For docs: blockquote summary, for code: docstring
    title: str | None = None  # For docs: first heading
    extension: str = ""
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": str(self.path),
            "file_type": self.file_type,
            "summary": self.summary,
            "title": self.title,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
        }


@dataclass
class DocumentationStats:
    """Summary statistics for documentation and code files."""

    total_files: int = 0
    total_docs: int = 0
    total_code: int = 0
    total_bytes: int = 0

    # Extension counts
    extensions: dict[str, int] = field(default_factory=dict)

    # Docs with summaries/titles
    docs_with_summary: int = 0
    docs_with_title: int = 0

    # Code with docstrings
    code_with_docstring: int = 0

    def to_summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Total Files: {self.total_files}",
            f"  Documentation: {self.total_docs}",
            f"  Code: {self.total_code}",
            f"Total Size: {format_bytes(self.total_bytes)}",
            "",
            "File Extensions:",
        ]
        for ext, count in sorted(self.extensions.items(), key=lambda x: -x[1]):
            lines.append(f"  {ext}: {count}")

        lines.append("")
        lines.append("Documentation Quality:")
        lines.append(f"  With Summary: {self.docs_with_summary}/{self.total_docs}")
        lines.append(f"  With Title: {self.docs_with_title}/{self.total_docs}")

        lines.append("")
        lines.append("Code Quality:")
        lines.append(f"  With Docstring: {self.code_with_docstring}/{self.total_code}")

        return "\n".join(lines)


class DocumentationDataStore:
    """
    Data store for documentation and code file analysis.

    Loads files from specified paths and provides local search capabilities
    without requiring external vectorstore services.
    """

    # Default code file extensions
    DEFAULT_CODE_EXTENSIONS: set[str] = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
        ".html", ".css", ".scss", ".sql", ".sh", ".bash",
    }

    def __init__(
        self,
        documentation_paths: list[str] | None = None,
        code_paths: list[str] | None = None,
        code_extensions: set[str] | None = None,
    ) -> None:
        """
        Initialize the DocumentationDataStore.

        Args:
            documentation_paths: Paths/patterns for documentation files (.md).
                Supports glob patterns like "docs/**/*.md".
            code_paths: Paths/patterns for code files.
                Supports glob patterns like "src/**/*.py".
            code_extensions: Set of file extensions to consider as code.
                Defaults to common programming file extensions.
        """
        self._entries: list[FileEntry] = []
        self._path_index: dict[str, FileEntry] = {}  # path string -> entry
        self._stats: DocumentationStats = DocumentationStats()

        self._documentation_paths = documentation_paths or []
        self._code_paths = code_paths or []
        self._code_extensions = code_extensions or self.DEFAULT_CODE_EXTENSIONS

        # Load files if paths are provided
        if self._documentation_paths or self._code_paths:
            self._load_files()
            self._compute_stats()

        logger.info(
            "DocumentationDataStore initialized with %d files (%d docs, %d code)",
            len(self._entries),
            self._stats.total_docs,
            self._stats.total_code,
        )

    @property
    def entries(self) -> list[FileEntry]:
        """Return all file entries."""
        return self._entries

    @property
    def stats(self) -> DocumentationStats:
        """Return computed statistics."""
        return self._stats

    @property
    def documentation_files(self) -> list[FileEntry]:
        """Return only documentation files."""
        return [e for e in self._entries if e.file_type == FileType.DOCUMENTATION]

    @property
    def code_files(self) -> list[FileEntry]:
        """Return only code files."""
        return [e for e in self._entries if e.file_type == FileType.CODE]

    def _load_files(self) -> None:
        """Load all documentation and code files."""
        # Load documentation files
        if self._documentation_paths:
            doc_files = resolve_glob_patterns(
                patterns=self._documentation_paths,
                extensions={".md"},
                recursive=True,
                raise_on_missing=False,
            )
            for file_path in doc_files:
                self._load_documentation_file(file_path)

        # Load code files
        if self._code_paths:
            code_files = resolve_glob_patterns(
                patterns=self._code_paths,
                extensions=self._code_extensions,
                recursive=True,
                raise_on_missing=False,
            )
            for file_path in code_files:
                self._load_code_file(file_path)

    def _load_documentation_file(self, file_path: Path) -> None:
        """Load a single documentation file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            title = parse_markdown_title(content)
            summary = parse_markdown_summary(content)

            entry = FileEntry(
                path=file_path,
                file_type=FileType.DOCUMENTATION,
                content=content,
                summary=summary,
                title=title,
                extension=file_path.suffix,
                size_bytes=file_path.stat().st_size,
            )
            self._entries.append(entry)
            self._path_index[str(file_path)] = entry

        except Exception as e:
            logger.warning("Failed to load documentation file %s: %s", file_path, e)

    def _load_code_file(self, file_path: Path) -> None:
        """Load a single code file."""
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            docstring = parse_python_docstring(content)

            entry = FileEntry(
                path=file_path,
                file_type=FileType.CODE,
                content=content,
                summary=docstring,
                title=file_path.name,
                extension=file_path.suffix,
                size_bytes=file_path.stat().st_size,
            )
            self._entries.append(entry)
            self._path_index[str(file_path)] = entry

        except Exception as e:
            logger.warning("Failed to load code file %s: %s", file_path, e)

    def _compute_stats(self) -> None:
        """Compute aggregate statistics from entries."""
        extensions: Counter[str] = Counter()
        total_bytes = 0
        total_docs = 0
        total_code = 0
        docs_with_summary = 0
        docs_with_title = 0
        code_with_docstring = 0

        for entry in self._entries:
            extensions[entry.extension] += 1
            total_bytes += entry.size_bytes

            if entry.file_type == FileType.DOCUMENTATION:
                total_docs += 1
                if entry.summary:
                    docs_with_summary += 1
                if entry.title:
                    docs_with_title += 1
            else:
                total_code += 1
                if entry.summary:
                    code_with_docstring += 1

        self._stats = DocumentationStats(
            total_files=len(self._entries),
            total_docs=total_docs,
            total_code=total_code,
            total_bytes=total_bytes,
            extensions=dict(extensions),
            docs_with_summary=docs_with_summary,
            docs_with_title=docs_with_title,
            code_with_docstring=code_with_docstring,
        )

    def get_file_by_path(self, path: str) -> FileEntry | None:
        """Get a file entry by its path."""
        # Try exact match first
        if path in self._path_index:
            return self._path_index[path]

        # Try matching by filename or partial path
        for entry_path, entry in self._path_index.items():
            if entry_path.endswith(path) or path in entry_path:
                return entry

        return None

    def get_file_content(self, path: str) -> str | None:
        """Get the content of a file by its path."""
        entry = self.get_file_by_path(path)
        return entry.content if entry else None

    def search_content(
        self,
        query: str,
        file_type: FileType | None = None,
        case_sensitive: bool = False,
        context_chars: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search file contents for a query string.

        Args:
            query: The string to search for.
            file_type: Optional filter by file type.
            case_sensitive: Whether search is case-sensitive.
            context_chars: Number of context characters around matches.

        Returns:
            List of dicts with path, file_type, count, and sample context.
        """
        results: list[dict[str, Any]] = []

        if not query:
            return results

        search_query = query if case_sensitive else query.lower()

        for entry in self._entries:
            if file_type and entry.file_type != file_type:
                continue

            content = entry.content if case_sensitive else entry.content.lower()
            original_content = entry.content

            count = content.count(search_query)
            if count == 0:
                continue

            # Find first occurrence and extract context
            pos = content.find(search_query)
            context_start = max(0, pos - context_chars)
            context_end = min(len(original_content), pos + len(query) + context_chars)

            sample = original_content[context_start:context_end]
            if context_start > 0:
                sample = "..." + sample
            if context_end < len(original_content):
                sample = sample + "..."

            results.append({
                "path": str(entry.path),
                "file_type": entry.file_type,
                "title": entry.title,
                "count": count,
                "sample": sample,
            })

        # Sort by count descending
        results.sort(key=lambda x: x["count"], reverse=True)

        return results

    def search_content_with_lines(
        self,
        query: str,
        file_type: FileType | None = None,
        case_sensitive: bool = False,
        max_matches_per_file: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search file contents and return matches WITH LINE NUMBERS.

        Like Cmd+F - finds exact query occurrences and returns line numbers.

        Args:
            query: The string to search for.
            file_type: Optional filter by file type (documentation/code).
            case_sensitive: Whether search is case-sensitive.
            max_matches_per_file: Maximum matches to return per file.

        Returns:
            List of dicts with:
            - path: str
            - file_type: str
            - matches: list of {line_number: int, line_content: str}
        """
        results: list[dict[str, Any]] = []

        if not query:
            return results

        search_query = query if case_sensitive else query.lower()

        for entry in self._entries:
            if file_type and entry.file_type != file_type:
                continue

            lines = entry.content.splitlines()
            matches: list[dict[str, Any]] = []

            for line_num, line in enumerate(lines, start=1):
                search_line = line if case_sensitive else line.lower()
                if search_query in search_line:
                    matches.append({
                        "line_number": line_num,
                        "line_content": line.strip(),
                    })
                    if len(matches) >= max_matches_per_file:
                        break

            if matches:
                results.append({
                    "path": str(entry.path),
                    "file_type": entry.file_type,
                    "title": entry.title,
                    "total_matches": len(matches),
                    "matches": matches,
                })

        # Sort by number of matches descending
        results.sort(key=lambda x: x["total_matches"], reverse=True)

        return results

    def get_file_lines(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> tuple[str, int] | None:
        """
        Read specific line range from a file.

        Args:
            path: File path (supports partial matching).
            start_line: Starting line (1-indexed, inclusive). None = from beginning.
            end_line: Ending line (1-indexed, inclusive). None = to end.

        Returns:
            Tuple of (content, total_lines) or None if file not found.
        """
        entry = self.get_file_by_path(path)
        if not entry:
            return None

        lines = entry.content.splitlines()
        total_lines = len(lines)

        # Handle line range
        start_idx = (start_line - 1) if start_line else 0
        end_idx = end_line if end_line else total_lines

        # Clamp to valid range
        start_idx = max(0, min(start_idx, total_lines))
        end_idx = max(start_idx, min(end_idx, total_lines))

        selected_lines = lines[start_idx:end_idx]
        content = "\n".join(selected_lines)

        return content, total_lines

    def get_documentation_index(self) -> list[dict[str, Any]]:
        """
        Get an index of documentation files with titles and summaries.

        Returns:
            List of dicts sorted by path with filename, title, summary.
        """
        results = []
        for entry in self.documentation_files:
            results.append({
                "path": str(entry.path),
                "filename": entry.path.name,
                "title": entry.title,
                "summary": entry.summary,
            })
        return sorted(results, key=lambda x: x["path"])

    def get_code_index(self) -> list[dict[str, Any]]:
        """
        Get an index of code files with docstrings.

        Returns:
            List of dicts sorted by path with path, docstring.
        """
        results = []
        for entry in self.code_files:
            results.append({
                "path": str(entry.path),
                "filename": entry.path.name,
                "docstring": entry.summary,
                "extension": entry.extension,
            })
        return sorted(results, key=lambda x: x["path"])
