"""
bluebox/routine_discovery/data_store.py

# NOTE: THIS DATA STORE IS SOON TO BE DEPRECATED!
# NOTE: THIS DATA STORE EXITS TO SUPPORT MIGRATION FROM VECTORSTORES TO NEW DATA STORES

Data store for routine discovery with vectorstore support.

Contains:
- DiscoveryDataStore: Abstract base class for data access
- LocalDiscoveryDataStore: File-based implementation with OpenAI vectorstores
- CDP data access: transactions, storage, window properties
- Documentation/code vectorstore creation for agent context
"""

from __future__ import annotations

import json
import os
import shutil
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from bluebox.utils.data_utils import get_text_from_html
from bluebox.utils.infra_utils import resolve_glob_patterns
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# Known keys in network transaction events (for grouping)
_TRANSACTION_KNOWN_KEYS = frozenset({
    'timestamp', 'request_id',
    'url', 'method', 'type', 'request_headers', 'post_data',
    'status', 'status_text', 'response_headers', 'response_body',
    'response_body_base64', 'mime_type', 'errorText', 'failed',
})


class DiscoveryDataStore(BaseModel, ABC):
    """
    Abstract base class for managing CDP discovery data and documentation/code vectorstores.
    """

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
    """
    File-based implementation of DiscoveryDataStore with OpenAI vectorstores.
    Manages CDP captures (events.jsonl format) and documentation/code vectorstores.
    """

    # openai client for vector store management
    client: OpenAI

    # CDP captures input directory (contains network/, storage/, window_properties/ subdirs with events.jsonl)
    cdp_captures_dir: str | None = None

    # Processed output paths (generated from events.jsonl processing)
    tmp_dir: str | None = None
    network_transactions_dir: str | None = None
    consolidated_transactions_file_path: str | None = None
    consolidated_storage_items_file_path: str | None = None
    consolidated_window_properties_file_path: str | None = None

    # CDP captures vectorstore
    cdp_captures_vectorstore_id: str | None = None
    uploaded_transaction_ids: set[str] = Field(default_factory=set, exclude=True)

    # documentation and code related fields (both go into same vectorstore)
    documentation_vectorstore_id: str | None = None
    documentation_paths: list[str] = Field(default_factory=list)
    code_paths: list[str] = Field(default_factory=list)
    code_file_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
            ".md", ".txt", ".html", ".css", ".scss", ".sql", ".sh", ".bash",
        ],
    )

    # Cache for uploaded documentation files info (for prompt generation)
    uploaded_docs_info: list[dict] = Field(default_factory=list, exclude=True)
    uploaded_code_info: list[dict] = Field(default_factory=list, exclude=True)

    @model_validator(mode='after')
    def setup_cdp_captures_paths(self) -> LocalDiscoveryDataStore:
        """
        Set up CDP captures paths if cdp_captures_dir is provided.
        Also populates documentation cache if vectorstore_id is provided.
        """
        # Set up CDP captures paths from cdp_captures_dir
        if self.cdp_captures_dir is not None:
            cdp_dir = Path(self.cdp_captures_dir)
            if not cdp_dir.exists():
                raise ValueError(f"CDP captures directory does not exist: {self.cdp_captures_dir}")

            # Set up tmp_dir for processed files
            if self.tmp_dir is None:
                self.tmp_dir = str(cdp_dir / ".processed")

            # Set up processed output paths
            tmp_path = Path(self.tmp_dir)
            self.network_transactions_dir = str(tmp_path / "network_transactions")
            self.consolidated_transactions_file_path = str(tmp_path / "consolidated_transactions.json")
            self.consolidated_storage_items_file_path = str(tmp_path / "consolidated_storage_items.json")
            self.consolidated_window_properties_file_path = str(tmp_path / "consolidated_window_properties.json")

        # Populate documentation cache if vectorstore_id is provided but caches are empty
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
            patterns=self.documentation_paths,
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
            patterns=self.code_paths,
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
        """
        Make a vectorstore from the CDP captures (events.jsonl format).

        Steps:
            1. In parallel: create vectorstore + process all CDP capture JSONL files
            2. Each file type uploads after processing, but waits for vectorstore to be ready
        """
        # Validate required paths are set
        if self.cdp_captures_dir is None:
            raise ValueError("cdp_captures_dir is required for CDP captures vectorstore")

        # Validate events.jsonl files exist
        network_events_path = Path(self.cdp_captures_dir) / "network" / "events.jsonl"
        storage_events_path = Path(self.cdp_captures_dir) / "storage" / "events.jsonl"
        window_props_events_path = Path(self.cdp_captures_dir) / "window_properties" / "events.jsonl"

        if not network_events_path.exists():
            raise ValueError(f"Network events file not found: {network_events_path}")
        if not storage_events_path.exists():
            raise ValueError(f"Storage events file not found: {storage_events_path}")
        if not window_props_events_path.exists():
            raise ValueError(f"Window properties events file not found: {window_props_events_path}")

        if self.cdp_captures_vectorstore_id is not None:
            raise ValueError(f"Vectorstore ID already exists: {self.cdp_captures_vectorstore_id}")

        # Create directories for processed output
        tmp_path = Path(self.tmp_dir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        Path(self.network_transactions_dir).mkdir(parents=True, exist_ok=True)

        # Event to signal when vectorstore is ready for uploads
        vectorstore_ready = Event()

        # Helper to create vectorstore and signal ready
        def create_vectorstore() -> None:
            vs = self.client.vector_stores.create(
                name=f"cdp-captures-{int(time.time())}",
                expires_after={"anchor": "last_active_at", "days": 1}
            )
            self.cdp_captures_vectorstore_id = vs.id
            vectorstore_ready.set()

        # Helper to process files and upload (waits for vectorstore before upload)
        def process_and_upload(process_fn, consolidated_file_path: str) -> None:
            process_fn()
            vectorstore_ready.wait()
            self.add_file_to_vectorstore(file_path=consolidated_file_path, metadata={})

        # Run all tasks in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(create_vectorstore),
                executor.submit(
                    process_and_upload,
                    self._process_network_transaction_files,
                    self.consolidated_transactions_file_path,
                ),
                executor.submit(
                    process_and_upload,
                    self._process_storage_files,
                    self.consolidated_storage_items_file_path,
                ),
                executor.submit(
                    process_and_upload,
                    self._process_window_properties_files,
                    self.consolidated_window_properties_file_path,
                ),
            ]
            # Wait for all to complete and raise any exceptions
            for future in as_completed(futures):
                future.result()

        logger.info("CDP captures vectorstore created: %s", self.cdp_captures_vectorstore_id)

    def _group_transaction_details(self, transaction_details: dict) -> dict:
        """
        Group flat transaction details into request/response structure.

        Args:
            transaction_details: Flat dict from NetworkTransactionEvent.

        Returns:
            Grouped dict with timestamp, request_id, request{}, response{}, and other{}.
        """
        td = transaction_details
        request_headers = td.get('request_headers') or {}
        response_headers = td.get('response_headers') or {}
        response_body = td.get('response_body') or ""
        response_body_base64 = td.get('response_body_base64', False)
        post_data = td.get('post_data')

        request = {
            "url": td.get('url'),
            "method": td.get('method'),
            "type": td.get('type'),
            "headers": request_headers,
            "post_data": post_data,
        }

        response = {
            "status": td.get('status'),
            "status_text": td.get('status_text'),
            "headers": response_headers,
            "body": response_body,
            "body_truncated": False,
            "body_base64": response_body_base64,
            "mime_type": td.get('mime_type') or "",
            "error_text": td.get('errorText'),
            "failed": td.get('failed', False),
        }

        other = None
        for key, value in td.items():
            if key not in _TRANSACTION_KNOWN_KEYS:
                if other is None:
                    other = {}
                other[key] = value

        return {
            "timestamp": td.get('timestamp'),
            "request_id": td.get('request_id'),
            "request": request,
            "response": response,
            "other": other if other else None,
        }

    def _process_network_transaction_files(self) -> None:
        """
        Process network/events.jsonl into consolidated transactions.

        Steps:
            1. Read JSONL file line by line
            2. Generate transaction_id (timestamp_url)
            3. Group into request/response structure
            4. Save individual tx files + consolidated JSON
        """
        network_events_path = Path(self.cdp_captures_dir) / "network" / "events.jsonl"
        consolidated_transactions: dict[str, dict] = {}

        logger.info("Processing network events from: %s", network_events_path)

        with open(network_events_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    # Parse the event (NetworkTransactionEvent format)
                    transaction_details = json.loads(line)

                    # Generate transaction_id (timestamp_url format)
                    url = transaction_details.get('url', 'unknown')
                    timestamp = transaction_details.get('timestamp', 0)
                    safe_url = url.replace('/', '_').replace(':', '_')[:100]
                    transaction_id = f"{timestamp}_{safe_url}"

                    # Group transaction details
                    grouped_transaction = self._group_transaction_details(transaction_details)

                    # Save full transaction to individual file
                    transaction_file_path = Path(self.network_transactions_dir) / f"{transaction_id}.json"
                    with open(transaction_file_path, mode="w", encoding="utf-8") as out_f:
                        json.dump(grouped_transaction, out_f, indent=1, ensure_ascii=False)

                    # Create truncated version for consolidated file
                    response_body = grouped_transaction['response']['body']
                    if response_body and len(str(response_body)) > 1000:
                        truncated_response = dict(grouped_transaction['response'])
                        truncated_response['body'] = str(response_body)[:1000] + "...[truncated]"
                        truncated_response['body_truncated'] = True
                        consolidated_transactions[transaction_id] = {
                            **grouped_transaction,
                            'response': truncated_response,
                        }
                    else:
                        consolidated_transactions[transaction_id] = grouped_transaction

                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse network event line: %s", e)
                except Exception as e:
                    logger.error("Error processing network event: %s", e)

        # Write consolidated transactions file
        with open(self.consolidated_transactions_file_path, mode="w", encoding="utf-8") as f:
            json.dump(consolidated_transactions, f, indent=1, ensure_ascii=False)

        logger.info("Processed %d network transactions", len(consolidated_transactions))

    def _process_storage_files(self) -> None:
        """
        Process storage/events.jsonl into consolidated storage items.
        """
        storage_events_path = Path(self.cdp_captures_dir) / "storage" / "events.jsonl"
        consolidated_storage_items: list[dict] = []

        logger.info("Processing storage events from: %s", storage_events_path)

        with open(storage_events_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    storage_event = json.loads(line)
                    consolidated_storage_items.append(storage_event)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse storage event line: %s", e)

        # Write consolidated storage file
        with open(self.consolidated_storage_items_file_path, mode="w", encoding="utf-8") as f:
            json.dump(consolidated_storage_items, f, indent=1, ensure_ascii=False)

        logger.info("Processed %d storage events", len(consolidated_storage_items))

    def _process_window_properties_files(self) -> None:
        """
        Process window_properties/events.jsonl into consolidated window properties.
        """
        window_props_events_path = Path(self.cdp_captures_dir) / "window_properties" / "events.jsonl"
        consolidated_window_properties: list[dict] = []

        logger.info("Processing window properties events from: %s", window_props_events_path)

        with open(window_props_events_path, mode="r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    window_prop_event = json.loads(line)
                    consolidated_window_properties.append(window_prop_event)
                except json.JSONDecodeError as e:
                    logger.warning("Failed to parse window property event line: %s", e)

        # Write consolidated window properties file
        with open(self.consolidated_window_properties_file_path, mode="w", encoding="utf-8") as f:
            json.dump(consolidated_window_properties, f, indent=1, ensure_ascii=False)

        logger.info("Processed %d window property events", len(consolidated_window_properties))


    def get_all_transaction_ids(self) -> list[str]:
        """
        Get all transaction ids from the data store.
        Reads from consolidated_transactions_file_path.
        """
        if self.consolidated_transactions_file_path is None:
            raise ValueError("consolidated_transactions_file_path is not set")

        if not Path(self.consolidated_transactions_file_path).exists():
            raise ValueError(f"Consolidated transactions file not found: {self.consolidated_transactions_file_path}")

        with open(self.consolidated_transactions_file_path, mode="r", encoding="utf-8") as f:
            consolidated_transactions = json.load(f)

        return list(consolidated_transactions.keys())

    def get_transaction_by_id(self, transaction_id: str, clean_response_body: bool = False) -> dict:
        """
        Get a transaction by id from the data store.
        Returns grouped structure: {timestamp, request_id, request{}, response{}}

        If the response body was truncated in the consolidated file,
        loads the full version from the individual transaction file.
        """
        if self.consolidated_transactions_file_path is None:
            raise ValueError("consolidated_transactions_file_path is not set")

        # Load from consolidated transactions
        with open(self.consolidated_transactions_file_path, mode="r", encoding="utf-8") as f:
            consolidated_transactions = json.load(f)

        if transaction_id not in consolidated_transactions:
            raise ValueError(f"Transaction id not found: {transaction_id}")

        transaction_details = consolidated_transactions[transaction_id]

        # If body was truncated, load full version from individual file
        if transaction_details.get('response', {}).get('body_truncated', False):
            transaction_file_path = Path(self.network_transactions_dir) / f"{transaction_id}.json"
            if transaction_file_path.exists():
                with open(transaction_file_path, mode="r", encoding="utf-8") as f:
                    transaction_details = json.load(f)

        # Clean HTML response body if requested
        mime_type = transaction_details.get('response', {}).get('mime_type', '')
        if clean_response_body and "html" in mime_type.lower():
            try:
                response_body = transaction_details.get('response', {}).get('body', '')
                if response_body:
                    transaction_details['response']['body'] = get_text_from_html(html=response_body)
            except Exception as e:
                logger.warning("Error cleaning response body (leaving as-is): %s", e)

        return transaction_details

    def clean_up(self) -> None:
        """
        Clean up all data store resources (CDP captures and documentation vectorstores).
        Also cleans up processed temporary files.
        """
        # Clean up CDP captures vectorstore if set
        if self.cdp_captures_vectorstore_id is not None:
            try:
                self.client.vector_stores.delete(vector_store_id=self.cdp_captures_vectorstore_id)
                logger.info("Deleted CDP captures vectorstore: %s", self.cdp_captures_vectorstore_id)
            except Exception as e:
                logger.error("Failed to delete CDP captures vectorstore: %s", e)
            self.cdp_captures_vectorstore_id = None

        # Clean up documentation vectorstore if set
        if self.documentation_vectorstore_id is not None:
            try:
                self.client.vector_stores.delete(vector_store_id=self.documentation_vectorstore_id)
                logger.info("Deleted documentation vectorstore: %s", self.documentation_vectorstore_id)
            except Exception as e:
                logger.error("Failed to delete documentation vectorstore: %s", e)
            self.documentation_vectorstore_id = None

        # Clean up processed temporary files
        if self.tmp_dir and Path(self.tmp_dir).exists():
            try:
                shutil.rmtree(self.tmp_dir)
                logger.info("Deleted temporary directory: %s", self.tmp_dir)
            except Exception as e:
                logger.warning("Failed to delete temporary directory: %s", e)

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

        # Use a separate subdirectory for individual transaction uploads
        # to avoid deleting the consolidated files in tmp_dir
        upload_tmp_path = Path(self.tmp_dir) / "upload_tmp"
        upload_tmp_path.mkdir(parents=True, exist_ok=True)

        transaction_file_path = upload_tmp_path / f"{transaction_id}.json"
        try:
            transaction_data = self.get_transaction_by_id(transaction_id, clean_response_body=True)
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
            # Only delete the specific file, not the entire tmp_dir
            if transaction_file_path.exists():
                transaction_file_path.unlink()
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
        """
        if self.consolidated_transactions_file_path is None:
            raise ValueError("consolidated_transactions_file_path is not set")

        with open(self.consolidated_transactions_file_path, mode="r", encoding="utf-8") as f:
            consolidated_transactions = json.load(f)

        transaction_ids = [
            transaction_id for transaction_id, details in consolidated_transactions.items()
            if request_url in details.get('request', {}).get('url', '')
        ]
        return transaction_ids

    def get_transaction_timestamp(self, transaction_id: str) -> float:
        """
        Get the timestamp of a transaction.

        Args:
            transaction_id: The id of the transaction.
        Returns:
            The timestamp of the transaction.
        """
        # Try to read from individual transaction file first
        if self.network_transactions_dir:
            transaction_file_path = Path(self.network_transactions_dir) / f"{transaction_id}.json"
            if transaction_file_path.exists():
                with open(transaction_file_path, mode="r", encoding="utf-8") as f:
                    transaction_details = json.load(f)
                return float(transaction_details.get('timestamp', 0))

        # Fall back to parsing from transaction_id (format: timestamp_url)
        parts = transaction_id.split("_")
        if len(parts) >= 1:
            try:
                return float(parts[0])
            except ValueError:
                pass

        raise ValueError(f"Could not determine timestamp for transaction: {transaction_id}")

    def scan_transaction_responses(self, value: str, max_timestamp: float | None = None) -> list[str]:
        """
        Scan the network transaction responses for a value.

        Args:
            value: The value to scan for in the network transaction responses.
            max_timestamp: latest timestamp to scan for.
        Returns:
            A list of transaction ids that contain the value in the response body.
        """
        if self.consolidated_transactions_file_path is None:
            raise ValueError("consolidated_transactions_file_path is not set")

        with open(self.consolidated_transactions_file_path, mode="r", encoding="utf-8") as f:
            consolidated_transactions = json.load(f)

        # Sort by timestamp (ascending)
        all_transaction_ids = sorted(
            consolidated_transactions.keys(),
            key=lambda x: float(x.split('_')[0]) if x.split('_')[0].replace('.', '').isdigit() else 0
        )

        results: list[str] = []
        for transaction_id in all_transaction_ids:
            transaction_details = self.get_transaction_by_id(transaction_id)

            # Check timestamp constraint
            timestamp = transaction_details.get('timestamp', 0)
            if max_timestamp is not None and float(timestamp) > max_timestamp:
                break

            # Check if value is in response body
            response_body = transaction_details.get('response', {}).get('body')
            if response_body is not None and value in str(response_body):
                results.append(transaction_id)

        return results

    def scan_storage_for_value(self, value: str) -> list[str]:
        """
        Scan the storage for a value.

        Args:
            value: The value to scan for in the storage.
        Returns:
            A list of storage items (as JSON strings) that contain the value.
        """
        if self.consolidated_storage_items_file_path is None:
            raise ValueError("consolidated_storage_items_file_path is not set")

        if not Path(self.consolidated_storage_items_file_path).exists():
            return []

        with open(self.consolidated_storage_items_file_path, mode="r", encoding="utf-8") as f:
            storage_items = json.load(f)

        return [json.dumps(item) for item in storage_items if value in str(item)]

    def scan_window_properties_for_value(self, value: str) -> list[dict]:
        """
        Scan the window properties for a value.

        Args:
            value: The value to scan for in the window properties.
        Returns:
            A list of window property events that contain the value.
        """
        if self.consolidated_window_properties_file_path is None:
            raise ValueError("consolidated_window_properties_file_path is not set")

        if not Path(self.consolidated_window_properties_file_path).exists():
            return []

        with open(self.consolidated_window_properties_file_path, mode="r", encoding="utf-8") as f:
            window_properties = json.load(f)

        return [prop for prop in window_properties if value in str(prop)]

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
            patterns=self.documentation_paths,
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
            patterns=self.code_paths,
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

        return """## CDP Captures Vectorstore
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
