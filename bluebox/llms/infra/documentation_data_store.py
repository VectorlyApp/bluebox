"""
bluebox/llms/infra/documentation_data_store.py

Data store for documentation and code file analysis.

Replaces the OpenAI vectorstore-based approach with local indexing
and search capabilities for documentation and code files.
"""

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from bluebox.utils.data_utils import format_bytes
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
            title = self._parse_doc_title(content)
            summary = self._parse_doc_summary(content)

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
            docstring = self._parse_code_docstring(content)

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

    def _parse_doc_title(self, content: str) -> str | None:
        """Parse title from markdown content (first # heading)."""
        lines = content.split("\n")
        for line in lines[:10]:
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _parse_doc_summary(self, content: str) -> str | None:
        """Parse summary from markdown content (blockquote after title)."""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("> "):
                return line[2:].strip()
            if i > 5:
                break
        return None

    def _parse_code_docstring(self, content: str) -> str | None:
        """Extract module-level docstring from code content."""
        # Read only first 2000 chars for efficiency
        content = content[:2000]

        # Look for triple-quoted docstring at the start
        for quote in ['"""', "'''"]:
            if quote in content:
                start = content.find(quote)
                if start < 50:  # Must be near the top
                    end = content.find(quote, start + 3)
                    if end != -1:
                        docstring = content[start + 3:end].strip()
                        if docstring:
                            # Replace newlines with semicolons for compact display
                            return docstring.replace("\n", "; ")
        return None

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

    def search_by_terms(
        self,
        terms: list[str],
        file_type: FileType | None = None,
        top_n: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Search files by a list of terms and rank by relevance.

        For each file, searches content for each term and computes:
        - unique_terms_found: how many different terms were found
        - total_hits: total number of term matches
        - score: (total_hits / num_terms) * unique_terms_found

        Args:
            terms: List of search terms (case-insensitive).
            file_type: Optional filter by file type.
            top_n: Number of top results to return.

        Returns:
            List of dicts with keys: path, file_type, unique_terms_found, total_hits, score
            Sorted by score descending, limited to top_n.
        """
        results: list[dict[str, Any]] = []
        terms_lower = [t.lower() for t in terms]
        num_terms = len(terms_lower)

        if num_terms == 0:
            return results

        for entry in self._entries:
            if file_type and entry.file_type != file_type:
                continue

            content_lower = entry.content.lower()

            # Count hits for each term
            unique_terms_found = 0
            total_hits = 0

            for term in terms_lower:
                count = content_lower.count(term)
                if count > 0:
                    unique_terms_found += 1
                    total_hits += count

            # Skip entries with no hits
            if unique_terms_found == 0:
                continue

            # Calculate score
            avg_hits = total_hits / num_terms
            score = avg_hits * unique_terms_found

            results.append({
                "path": str(entry.path),
                "file_type": entry.file_type,
                "title": entry.title,
                "summary": entry.summary,
                "unique_terms_found": unique_terms_found,
                "total_hits": total_hits,
                "score": score,
            })

        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_n]

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

    def search_by_pattern(
        self,
        pattern: str,
        file_type: FileType | None = None,
    ) -> list[FileEntry]:
        """
        Search files by path pattern (glob-style).

        Args:
            pattern: Glob pattern to match paths (e.g., "**/test_*.py").
            file_type: Optional filter by file type.

        Returns:
            List of matching FileEntry objects.
        """
        results = []

        for entry in self._entries:
            if file_type and entry.file_type != file_type:
                continue

            if fnmatch.fnmatch(str(entry.path), pattern):
                results.append(entry)

        return results

    def search_by_extension(self, extension: str) -> list[FileEntry]:
        """
        Get all files with a specific extension.

        Args:
            extension: File extension (e.g., ".py" or "py").

        Returns:
            List of matching FileEntry objects.
        """
        if not extension.startswith("."):
            extension = "." + extension

        return [e for e in self._entries if e.extension == extension]

    def search_functions(
        self,
        name_pattern: str | None = None,
        file_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for function/method definitions in code files.

        Args:
            name_pattern: Regex pattern to match function names.
            file_pattern: Glob pattern to filter files.

        Returns:
            List of dicts with path, function_name, line_number, signature.
        """
        results: list[dict[str, Any]] = []

        # Pattern to match Python function definitions
        func_pattern = re.compile(r"^\s*(async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE)

        for entry in self._entries:
            if entry.file_type != FileType.CODE:
                continue

            if file_pattern and not fnmatch.fnmatch(str(entry.path), file_pattern):
                continue

            # Only search Python files for now
            if entry.extension != ".py":
                continue

            for match in func_pattern.finditer(entry.content):
                async_prefix = match.group(1) or ""
                func_name = match.group(2)
                params = match.group(3)

                # Apply name filter if provided
                if name_pattern and not re.search(name_pattern, func_name):
                    continue

                # Calculate line number
                line_number = entry.content[:match.start()].count("\n") + 1

                results.append({
                    "path": str(entry.path),
                    "function_name": func_name,
                    "line_number": line_number,
                    "signature": f"{async_prefix}def {func_name}({params})",
                    "is_async": bool(async_prefix),
                })

        return results

    def search_classes(
        self,
        name_pattern: str | None = None,
        file_pattern: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for class definitions in code files.

        Args:
            name_pattern: Regex pattern to match class names.
            file_pattern: Glob pattern to filter files.

        Returns:
            List of dicts with path, class_name, line_number, bases.
        """
        results: list[dict[str, Any]] = []

        # Pattern to match Python class definitions
        class_pattern = re.compile(r"^\s*class\s+(\w+)\s*(?:\(([^)]*)\))?:", re.MULTILINE)

        for entry in self._entries:
            if entry.file_type != FileType.CODE:
                continue

            if file_pattern and not fnmatch.fnmatch(str(entry.path), file_pattern):
                continue

            # Only search Python files for now
            if entry.extension != ".py":
                continue

            for match in class_pattern.finditer(entry.content):
                class_name = match.group(1)
                bases = match.group(2) or ""

                # Apply name filter if provided
                if name_pattern and not re.search(name_pattern, class_name):
                    continue

                # Calculate line number
                line_number = entry.content[:match.start()].count("\n") + 1

                results.append({
                    "path": str(entry.path),
                    "class_name": class_name,
                    "line_number": line_number,
                    "bases": bases,
                })

        return results

    def get_file_index(self) -> list[dict[str, Any]]:
        """
        Get an index of all files with their metadata.

        Returns:
            List of dicts with path, file_type, title, summary, extension.
        """
        return [
            {
                "path": str(entry.path),
                "file_type": entry.file_type,
                "title": entry.title,
                "summary": entry.summary,
                "extension": entry.extension,
                "size_bytes": entry.size_bytes,
            }
            for entry in self._entries
        ]

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

    def generate_context_prompt(self) -> str:
        """
        Generate a prompt describing the documentation/code contents.

        Useful for providing context to LLMs about available documentation.

        Returns:
            Formatted string describing the indexed files.
        """
        lines = ["# Available Documentation and Code"]

        # Documentation section
        if self.documentation_files:
            lines.append("")
            lines.append("## Documentation Files")
            for entry in sorted(self.documentation_files, key=lambda x: str(x.path)):
                if entry.title and entry.summary:
                    lines.append(f"- `{entry.path.name}`: {entry.title} - {entry.summary}")
                elif entry.title:
                    lines.append(f"- `{entry.path.name}`: {entry.title}")
                elif entry.summary:
                    lines.append(f"- `{entry.path.name}`: {entry.summary}")
                else:
                    lines.append(f"- `{entry.path.name}`")

        # Code section
        if self.code_files:
            lines.append("")
            lines.append("## Code Files")
            for entry in sorted(self.code_files, key=lambda x: str(x.path)):
                if entry.summary:
                    # Truncate long docstrings
                    docstring = entry.summary[:100] + "..." if len(entry.summary) > 100 else entry.summary
                    lines.append(f"- `{entry.path.name}`: {docstring}")
                else:
                    lines.append(f"- `{entry.path.name}`")

        return "\n".join(lines)
