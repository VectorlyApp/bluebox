"""
bluebox/routine_discovery/data_store.py

Data store for routine discovery with vectorstore support.

Contains:
- DiscoveryDataStore: Abstract base class for data access
- LocalDiscoveryDataStore: File-based implementation with OpenAI vectorstores
- CDP data access: transactions, storage, window properties
- Documentation/code vectorstore creation for agent context
"""

import json
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bluebox.utils.data_utils import get_text_from_html
from bluebox.utils.infra_utils import resolve_glob_patterns


class DiscoveryDataStore(BaseModel, ABC):
    """Abstract base class for managing discovery data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def make_cdp_captures_vectorstore(self) -> None:
        """Make a vectorstore from the CDP captures."""
        pass
    
    @abstractmethod
    def make_documentation_vectorstore(self) -> None:
        """Make a vectorstore from the documentation."""
        pass

    @abstractmethod
    def get_all_transaction_ids(self) -> list[str]:
        """Get all transaction ids from the data store."""
        pass

    @abstractmethod
    def get_transaction_by_id(self, transaction_id: str, clean_response_body: bool = False) -> dict:
        """Get a transaction by id from the data store."""
        pass

    @abstractmethod
    def add_transaction_to_vectorstore(self, transaction_id: str, metadata: dict) -> None:
        """Add a single transaction to the vectorstore."""
        pass

    @abstractmethod
    def add_file_to_vectorstore(self, file_path: str, metadata: dict) -> None:
        """Add a file to the vectorstore."""
        pass

    @abstractmethod
    def get_transaction_ids_by_request_url(self, request_url: str) -> list[str]:
        """Get all transaction ids by request url."""
        pass

    @abstractmethod
    def get_transaction_timestamp(self, transaction_id: str) -> float:
        """Get the timestamp of a transaction."""
        pass

    @abstractmethod
    def scan_transaction_responses(self, value: str, max_timestamp: float | None = None) -> list[str]:
        """Scan the network transaction responses for a value."""
        pass

    @abstractmethod
    def scan_storage_for_value(self, value: str) -> list[str]:
        """Scan the storage for a value."""
        pass

    @abstractmethod
    def scan_window_properties_for_value(self, value: str) -> list[dict]:
        """Scan the window properties for a value."""
        pass

    @abstractmethod
    def clean_up(self) -> None:
        """Clean up the data store resources."""
        pass

    @abstractmethod
    def get_vectorstore_ids(self) -> list[str]:
        """Get all available vectorstore IDs."""
        pass


class LocalDiscoveryDataStore(DiscoveryDataStore):

    # openai client for vector store management
    client: OpenAI

    # cdp captures related fields (all optional - only required if using CDP captures)
    cdp_captures_vectorstore_id: str | None = None
    tmp_dir: str | None = None
    transactions_dir: str | None = None
    consolidated_transactions_path: str | None = None
    storage_jsonl_path: str | None = None
    window_properties_path: str | None = None
    cached_transaction_ids: list[str] | None = Field(default=None, exclude=True)
    uploaded_transaction_ids: set[str] = Field(default_factory=set, exclude=True)
    transaction_response_supported_file_extensions: list[str] = Field(default_factory=lambda: [
        ".txt",
        ".json",
        ".html",
        ".xml",
    ])

    # documentation and code related fields (both go into same vectorstore)
    documentation_vectorstore_id: str | None = None
    documentation_paths: list[str] = Field(default_factory=list)
    code_paths: list[str] = Field(default_factory=list)
    code_file_extensions: list[str] = Field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
        ".md", ".txt", ".html", ".css", ".scss", ".sql", ".sh", ".bash",
    ])

    # Cache for uploaded documentation files info (for prompt generation)
    uploaded_docs_info: list[dict] = Field(default_factory=list, exclude=True)
    uploaded_code_info: list[dict] = Field(default_factory=list, exclude=True)

    @field_validator('transactions_dir', 'consolidated_transactions_path', 'storage_jsonl_path', 'window_properties_path')
    @classmethod
    def validate_paths(cls, v: str | None) -> str | None:
        if v is not None and not Path(v).exists():
            raise ValueError(f"Path {v} does not exist")
        return v

    @model_validator(mode='after')
    def populate_cache_from_existing_vectorstore(self) -> 'LocalDiscoveryDataStore':
        """
        If documentation_vectorstore_id is provided but caches are empty,
        fetch file info from the existing vectorstore to populate the cache.
        """
        if (
            self.documentation_vectorstore_id is not None
            and not self.uploaded_docs_info
            and not self.uploaded_code_info
        ):
            self._populate_cache_from_vectorstore()
        return self

    def _populate_cache_from_vectorstore(self) -> None:
        """
        Populate cache by scanning documentation_paths and code_paths.
        Called when vectorstore_id is provided but cache is empty.

        Supports glob patterns in paths (see resolve_glob_patterns for pattern syntax).

        TODO: Consider fetching file list from vectorstore API instead of re-scanning dirs,
        which would allow cache population even when dirs aren't available locally.
        """
        # Resolve documentation patterns (recursive to include subdirectories)
        doc_files = resolve_glob_patterns(
            self.documentation_paths,
            extensions={".md"},
            recursive=True,
            raise_on_missing=False,
        )
        for file in doc_files:
            content = file.read_text(encoding="utf-8", errors="replace")[:500]
            summary = self._parse_doc_summary(content)
            self.uploaded_docs_info.append({
                'filename': file.name,
                'summary': summary
            })

        # Resolve code patterns (recursive by default for code)
        code_extensions = set(self.code_file_extensions)
        code_files = resolve_glob_patterns(
            self.code_paths,
            extensions=code_extensions,
            recursive=True,
            raise_on_missing=False,
        )
        for file in code_files:
            docstring = self._parse_code_docstring(str(file))
            self.uploaded_code_info.append({
                'path': file.name,
                'docstring': docstring
            })

    def make_cdp_captures_vectorstore(self) -> None:
        """Make a vectorstore from the CDP captures."""
        # Validate required paths are set
        if self.tmp_dir is None:
            raise ValueError("tmp_dir is required for CDP captures vectorstore")
        if self.transactions_dir is None:
            raise ValueError("transactions_dir is required for CDP captures vectorstore")
        if self.consolidated_transactions_path is None:
            raise ValueError("consolidated_transactions_path is required for CDP captures vectorstore")
        if self.storage_jsonl_path is None:
            raise ValueError("storage_jsonl_path is required for CDP captures vectorstore")
        if self.window_properties_path is None:
            raise ValueError("window_properties_path is required for CDP captures vectorstore")

        tmp_path = Path(self.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)

        if self.cdp_captures_vectorstore_id is not None:
            raise ValueError(f"Vectorstore ID already exists: {self.cdp_captures_vectorstore_id}")

        vs = self.client.vector_stores.create(
            name=f"cdp-captures-{int(time.time())}",
            expires_after={"anchor": "last_active_at", "days": 1}
        )
        self.cdp_captures_vectorstore_id = vs.id

        # Upload consolidated transactions
        self._upload_file_to_vectorstore(
            self.cdp_captures_vectorstore_id,
            self.consolidated_transactions_path,
            {"filename": "consolidated_transactions.json"}
        )

        # Convert jsonl to json (jsonl not supported by openai)
        storage_data = []
        with open(self.storage_jsonl_path, mode="r", encoding="utf-8") as f:
            for line in f:
                storage_data.append(json.loads(line))

        storage_file_path = tmp_path / "storage.json"
        storage_file_path.write_text(json.dumps(storage_data, ensure_ascii=False, indent=2), encoding="utf-8")

        self._upload_file_to_vectorstore(
            self.cdp_captures_vectorstore_id,
            str(storage_file_path),
            {"filename": "storage.json"}
        )

        # Upload window properties
        self._upload_file_to_vectorstore(
            self.cdp_captures_vectorstore_id,
            self.window_properties_path,
            {"filename": "window_properties.json"}
        )

        shutil.rmtree(tmp_path)


    def get_all_transaction_ids(self) -> list[str]:
        """
        Get all transaction ids from the data store that have a response body file with a supported extension.
        Cached per instance to avoid repeated filesystem operations.
        """
        if self.cached_transaction_ids is not None:
            return self.cached_transaction_ids

        transactions_path = Path(self.transactions_dir)
        supported_transaction_ids = [
            item.name for item in transactions_path.iterdir()
            if item.is_dir() and self.get_response_body_file_extension(item.name) in self.transaction_response_supported_file_extensions
        ]

        self.cached_transaction_ids = supported_transaction_ids
        return supported_transaction_ids


    def get_transaction_by_id(self, transaction_id: str, clean_response_body: bool = False) -> dict:
        """
        Get a transaction by id from the data store.
        Returns: {"request": ..., "response": ..., "response_body": ...}
        """
        transaction_path = Path(self.transactions_dir) / transaction_id
        result = {}

        # Load request
        request_path = transaction_path / "request.json"
        try:
            result["request"] = json.loads(request_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            result["request"] = f"No request found for transaction {transaction_id}"

        # Load response
        response_path = transaction_path / "response.json"
        try:
            result["response"] = json.loads(response_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            result["response"] = f"No response found for transaction {transaction_id}"

        # Load response body
        response_body_extension = self.get_response_body_file_extension(transaction_id)
        if response_body_extension is None:
            result["response_body"] = f"No response body found for transaction {transaction_id}"
        else:
            response_body_path = transaction_path / f"response_body{response_body_extension}"
            if response_body_extension == ".json":
                try:
                    result["response_body"] = json.loads(response_body_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    result["response_body"] = response_body_path.read_text(encoding="utf-8", errors="replace")
            else:
                result["response_body"] = response_body_path.read_text(encoding="utf-8", errors="replace")
                if response_body_extension == ".html" and clean_response_body:
                    result["response_body"] = get_text_from_html(result["response_body"])

        return result

    def clean_up(self) -> None:
        """
        Clean up all data store resources (CDP captures and documentation vectorstores).
        """
        # Clean up CDP captures vectorstore if set
        if self.cdp_captures_vectorstore_id is not None:
            try:
                self.client.vector_stores.delete(vector_store_id=self.cdp_captures_vectorstore_id)
            except Exception as e:
                raise ValueError(f"Failed to delete CDP captures vectorstore: {e}")
            self.cdp_captures_vectorstore_id = None

        # Clean up documentation vectorstore if set
        if self.documentation_vectorstore_id is not None:
            try:
                self.client.vector_stores.delete(vector_store_id=self.documentation_vectorstore_id)
            except Exception as e:
                raise ValueError(f"Failed to delete documentation vectorstore: {e}")
            self.documentation_vectorstore_id = None

    def get_vectorstore_ids(self) -> list[str]:
        """Get all available vectorstore IDs."""
        ids = []
        if self.cdp_captures_vectorstore_id is not None:
            ids.append(self.cdp_captures_vectorstore_id)
        if self.documentation_vectorstore_id is not None:
            ids.append(self.documentation_vectorstore_id)
        return ids

    def add_transaction_to_vectorstore(self, transaction_id: str, metadata: dict) -> None:
        """
        Add a single transaction to the vectorstore.
        Args:
            transaction_id: ID of the transaction to add
            metadata: Metadata to attach to the transaction file
        """
        if self.cdp_captures_vectorstore_id is None:
            raise ValueError("Vectorstore ID is not set")

        if transaction_id in self.uploaded_transaction_ids:
            return

        tmp_path = Path(self.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)

        try:
            transaction_data = self.get_transaction_by_id(transaction_id, clean_response_body=True)
            transaction_file_path = tmp_path / f"{transaction_id}.json"
            transaction_file_path.write_text(
                json.dumps(transaction_data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            self._upload_file_to_vectorstore(
                self.cdp_captures_vectorstore_id,
                str(transaction_file_path),
                metadata
            )
        finally:
            shutil.rmtree(tmp_path)
            self.uploaded_transaction_ids.add(transaction_id)


    def add_file_to_vectorstore(self, file_path: str, metadata: dict) -> None:
        """
        Add a file to the CDP captures vectorstore.
        Args:
            file_path: Path to the file to upload
            metadata: Metadata to attach to the file
        """
        if self.cdp_captures_vectorstore_id is None:
            raise ValueError("CDP captures vectorstore ID is not set")
        self._upload_file_to_vectorstore(self.cdp_captures_vectorstore_id, file_path, metadata)

    def _upload_file_to_vectorstore(self, vectorstore_id: str, file_path: str, metadata: dict) -> None:
        """
        Upload a file to a vectorstore (unified method for both CDP and documentation).
        Args:
            vectorstore_id: The vectorstore ID to upload to
            file_path: Path to the file to upload
            metadata: Metadata to attach to the file
        """
        path = Path(file_path)

        # Skip empty files (OpenAI rejects them)
        if path.stat().st_size == 0:
            return

        with open(path, mode="rb") as f:
            uploaded = self.client.files.create(file=f, purpose="assistants")

        self.client.vector_stores.files.create(
            vector_store_id=vectorstore_id,
            file_id=uploaded.id,
            attributes={"filename": path.name, **metadata}
        )


    def get_transaction_ids_by_request_url(self, request_url: str) -> list[str]:
        """
        Get all transaction ids by request url.
        Efficiently reads only the request.json file instead of the entire transaction.
        """
        transactions_path = Path(self.transactions_dir)
        transaction_ids = []
        for transaction_id in self.get_all_transaction_ids():
            try:
                request_path = transactions_path / transaction_id / "request.json"
                request_data = json.loads(request_path.read_text(encoding="utf-8"))
                if request_data.get("url") == request_url:
                    transaction_ids.append(transaction_id)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
        return transaction_ids


    def get_transaction_timestamp(self, transaction_id: str) -> float:
        """
        Get the timestamp of a transaction.
        Args:
            transaction_id: The id of the transaction.
        Returns:
            The timestamp of the transaction.
        """
        #TODO: cleaner way to get the timestamp
        parts = transaction_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid transaction_id format: {transaction_id}. Expected format: 'prefix_timestamp'")
        unix_timestamp = parts[1]
        try:
            return float(unix_timestamp)
        except ValueError as e:
            raise ValueError(
                f"Invalid timestamp in transaction_id '{transaction_id}'; {unix_timestamp} is not a valid number: {str(e)}"
            )


    def scan_transaction_responses(self, value: str, max_timestamp: float | None = None) -> list[str]:
        """
        Scan the network transaction responses for a value.

        Args:
            value: The value to scan for in the network transaction responses.
            max_timestamp: latest timestamp to scan for.
        Returns:
            A list of transaction ids that contain the value in the response body.
        """
        all_transaction_ids = self.get_all_transaction_ids()
        results = []
        for transaction_id in all_transaction_ids:
            transaction = self.get_transaction_by_id(transaction_id)
            if (
                value in str(transaction["response_body"])
                and
                (
                    max_timestamp is None
                    or self.get_transaction_timestamp(transaction_id) < max_timestamp
                )
            ):
                results.append(transaction_id)

        return list(set(results))


    def scan_storage_for_value(self, value: str) -> list[str]:
        """
        Scan the storage for a value.
        Args:
            value: The value to scan for in the storage.
        Returns:
            A list of storage items that contain the value.
        """
        storage_path = Path(self.storage_jsonl_path)
        return [line for line in storage_path.read_text(encoding="utf-8", errors="replace").splitlines() if value in line]

    def scan_window_properties_for_value(self, value: str) -> list[dict]:
        """
        Scan the window properties for a value.
        Args:
            value: The value to scan for in the window properties.
        Returns:
            A list of window properties that contain the value.
        """
        window_props_path = Path(self.window_properties_path)
        window_properties = json.loads(window_props_path.read_text(encoding="utf-8", errors="replace"))
        return [
            {key: val} for key, val in window_properties.items()
            if value in str(val)
        ]

    def get_response_body_file_extension(self, transaction_id: str) -> str | None:
        """
        Get the extension of the response body file for a transaction.
        Args:
            transaction_id: The id of the transaction.
        Returns:
            The extension of the response body file, or None if not found.
        """
        transaction_path = Path(self.transactions_dir) / transaction_id
        for file in transaction_path.iterdir():
            if file.name.startswith("response_body"):
                return file.suffix.lower()
        return None

    def make_documentation_vectorstore(self) -> None:
        """
        Create a vectorstore and upload documentation (.md files) and code files.
        Both documentation_paths and code_paths contents go into the same vectorstore.
        Uses parallel uploads for speed.

        Supports glob patterns in paths (see resolve_glob_patterns for pattern syntax):
        - "path/to/file.md" - single file
        - "path/to/dir/" - directory
        - "path/**/*.py" - recursive glob
        - "!pattern" - exclude files matching pattern
        """
        if self.documentation_vectorstore_id is not None:
            raise ValueError(f"Documentation vectorstore already exists: {self.documentation_vectorstore_id}")

        if not self.documentation_paths and not self.code_paths:
            raise ValueError("At least one of documentation_paths or code_paths must be set")

        # Clear cached info
        self.uploaded_docs_info = []
        self.uploaded_code_info = []

        # Collect all files to upload: (file_path, metadata, cache_info_type, cache_info)
        upload_tasks: list[tuple[str, dict, str, dict]] = []

        # Resolve and collect documentation files (recursive to include subdirectories)
        doc_files = resolve_glob_patterns(
            self.documentation_paths,
            extensions={".md"},
            recursive=True,
            raise_on_missing=True,
        )
        for file in doc_files:
            content = file.read_text(encoding="utf-8", errors="replace")[:500]
            summary = self._parse_doc_summary(content)
            metadata = {"type": "documentation", "filename": file.name}
            cache_info = {"filename": file.name, "summary": summary}
            upload_tasks.append((str(file), metadata, "docs", cache_info))

        # Resolve and collect code files
        code_extensions = set(self.code_file_extensions)
        code_files = resolve_glob_patterns(
            self.code_paths,
            extensions=code_extensions,
            recursive=True,
            raise_on_missing=True,
        )
        for file in code_files:
            docstring = self._parse_code_docstring(str(file))
            metadata = {
                "type": "code",
                "filename": file.name,
                "path": file.name,
            }
            cache_info = {"path": file.name, "docstring": docstring}
            upload_tasks.append((str(file), metadata, "code", cache_info))

        if not upload_tasks:
            raise ValueError("No files found to upload")

        # Create vectorstore
        vs = self.client.vector_stores.create(
            name=f"documentation-context-{int(time.time())}",
            expires_after={"anchor": "last_active_at", "days": 1}
        )
        self.documentation_vectorstore_id = vs.id

        # Upload all files in parallel
        def upload_task(task: tuple[str, dict, str, dict]) -> tuple[str, dict]:
            file_path, metadata, cache_type, cache_info = task
            self._upload_file_to_vectorstore(self.documentation_vectorstore_id, file_path, metadata)
            return cache_type, cache_info

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(upload_task, task) for task in upload_tasks]
            for future in as_completed(futures):
                cache_type, cache_info = future.result()
                if cache_type == "docs":
                    self.uploaded_docs_info.append(cache_info)
                else:
                    self.uploaded_code_info.append(cache_info)

    def _parse_doc_summary(self, content: str) -> str | None:
        """Parse summary from markdown content (blockquote after title)."""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("> "):
                return line[2:].strip()
            if i > 5:
                break
        return None

    def _parse_code_docstring(self, file_path: str) -> str | None:
        """Extract module-level docstring from a code file."""
        try:
            with open(file_path, mode="r", encoding="utf-8", errors="replace") as f:
                content = f.read(2000)  # Read first 2000 chars only

            # Look for triple-quoted docstring at the start
            for quote in ['"""', "'''"]:
                if quote in content:
                    start = content.find(quote)
                    if start < 50:  # Must be near the top
                        end = content.find(quote, start + 3)
                        if end != -1:
                            docstring = content[start + 3:end].strip()
                            if docstring:
                                # Replace newlines with semicolons for single-line display
                                return docstring.replace("\n", "; ")
        except Exception:
            pass
        return None

    def _generate_cdp_captures_vectorstore_prompt(self) -> str:
        """Generate a brief prompt describing the CDP captures vectorstore contents."""
        if self.cdp_captures_vectorstore_id is None:
            return ""

        transaction_count = len(self.get_all_transaction_ids())

        return f"""## CDP Captures Vectorstore
Contains browser session data captured via Chrome DevTools Protocol:

- `consolidated_transactions.json`: Summary of all HTTP transactions collected by CDP
  - Contains: URL, method, headers, postData (request body), status code, response headers, response body (JSON/HTML/text)
  - Search when: Looking for API endpoints, request/response payloads, auth headers, or understanding network calls

- `storage.json`: Browser storage (localStorage, sessionStorage, cookies)
  - Contains: Cookie name/value/domain/expiration, localStorage key-values, sessionStorage key-values
  - Search when: Debugging auth tokens, session IDs, cached data, or tracking when values were set

- `window_properties.json`: JavaScript window object properties
  - Contains: Custom JS properties injected into window, values captured at different page states with timestamps
  - Search when: Looking for data set by JavaScript, A/B test flags, analytics config, or dynamic page state
"""

    def _generate_documentation_vectorstore_prompt(self) -> str:
        """Generate a brief prompt describing the documentation vectorstore contents."""
        if self.documentation_vectorstore_id is None:
            return ""

        lines = ["## Documentation Vectorstore"]

        # Use cached documentation info (populated during upload)
        if self.uploaded_docs_info:
            doc_entries = []
            for info in sorted(self.uploaded_docs_info, key=lambda x: x["filename"]):
                if info["summary"]:
                    doc_entries.append(f"  - `{info['filename']}`: {info['summary']}")
                else:
                    doc_entries.append(f"  - `{info['filename']}`")
            if doc_entries:
                lines.append("**Documentation:**")
                lines.extend(doc_entries)

        # Use cached code info (populated during upload)
        if self.uploaded_code_info:
            code_entries = []
            for info in sorted(self.uploaded_code_info, key=lambda x: x["path"]):
                if info["docstring"]:
                    code_entries.append(f"  - `{info['path']}`: {info['docstring']}")
                else:
                    code_entries.append(f"  - `{info['path']}`")
            if code_entries:
                lines.append("**Code:**")
                lines.extend(code_entries)

        return "\n".join(lines)

    def generate_data_store_prompt(self) -> str:
        """Generate a combined prompt describing all vectorstore contents."""
        prompts = []

        cdp_prompt = self._generate_cdp_captures_vectorstore_prompt()
        if cdp_prompt:
            prompts.append(cdp_prompt)

        docs_prompt = self._generate_documentation_vectorstore_prompt()
        if docs_prompt:
            prompts.append(docs_prompt)

        if not prompts:
            return ""

        return "# Available Data Stores\n\n" + "\n\n".join(prompts)
