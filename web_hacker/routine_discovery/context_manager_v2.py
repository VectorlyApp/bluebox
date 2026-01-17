"""
LocalDiscoveryDataStore V2 - With Agent Documentation

Key difference from V1: Loads agent_docs/*.md files into the vectorstore
so the LLM has access to documentation about routines, operations, parameters, etc.
"""

from pydantic import BaseModel, field_validator, Field, ConfigDict
from openai import OpenAI
from pathlib import Path
import os
import json
import time
import shutil

from web_hacker.utils.data_utils import get_text_from_html
from web_hacker.routine_discovery.context_manager import DiscoveryDataStore


# Default path to agent_docs (relative to repo root)
DEFAULT_AGENT_DOCS_DIR = Path(__file__).parent.parent.parent / "agent_docs"


class LocalDiscoveryDataStoreV2(DiscoveryDataStore):
    """
    LocalDiscoveryDataStore with agent documentation support.

    Loads agent_docs/*.md files into the vectorstore so the LLM has
    documentation about routines, operations, parameters, placeholders, etc.
    """

    client: OpenAI
    tmp_dir: str
    transactions_dir: str
    consolidated_transactions_path: str
    storage_jsonl_path: str
    window_properties_path: str
    vectorstore_id: str | None = None

    # NEW: Path to agent_docs directory
    agent_docs_dir: Path | None = Field(default=None)

    # Generated index of doc summaries (populated during _upload_agent_docs)
    docs_index: str = Field(default="")

    supported_file_extensions: list[str] = Field(default_factory=lambda: [
        ".txt",
        ".json",
        ".html",
        ".xml",
    ])
    cached_transaction_ids: list[str] | None = Field(default=None, exclude=True)
    uploaded_transaction_ids: set[str] = Field(default_factory=set, exclude=True)

    @field_validator('transactions_dir', 'consolidated_transactions_path', 'storage_jsonl_path', 'window_properties_path')
    @classmethod
    def validate_paths(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"Path {v} does not exist")
        return v

    def model_post_init(self, __context) -> None:
        """Set default agent_docs_dir if not provided."""
        if self.agent_docs_dir is None and DEFAULT_AGENT_DOCS_DIR.exists():
            self.agent_docs_dir = DEFAULT_AGENT_DOCS_DIR

    def make_vectorstore(self) -> None:
        """Make a vectorstore from the context, including agent docs."""

        # make the tmp directory
        os.makedirs(self.tmp_dir, exist_ok=True)

        # ensure no vectorstore for this context already exists
        if self.vectorstore_id is not None:
            raise ValueError(f"Vectorstore ID already exists: {self.vectorstore_id}")

        # make the vectorstore
        vs = self.client.vector_stores.create(
            name=f"api-extraction-context-v2-{int(time.time())}"
        )

        # save the vectorstore id
        self.vectorstore_id = vs.id

        # Upload agent documentation FIRST (high priority for LLM)
        self._upload_agent_docs()

        # upload the transactions to the vectorstore
        self.add_file_to_vectorstore(
            self.consolidated_transactions_path,
            {"filename": "consolidated_transactions.json", "type": "transactions"}
        )

        # convert jsonl to json (jsonl not supported by openai)
        storage_data = []
        with open(self.storage_jsonl_path, mode="r", encoding="utf-8") as storage_jsonl_file:
            for line in storage_jsonl_file:
                obj = json.loads(line)
                storage_data.append(obj)

        # create a single storage.json file
        storage_file_path = os.path.join(self.tmp_dir, "storage.json")
        with open(storage_file_path, mode="w", encoding="utf-8") as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2)

        # upload the storage to the vectorstore
        self.add_file_to_vectorstore(storage_file_path, {"filename": "storage.json", "type": "storage"})

        # upload the window properties to the vectorstore
        self.add_file_to_vectorstore(
            self.window_properties_path,
            {"filename": "window_properties.json", "type": "window_properties"}
        )

        # delete the tmp directory
        shutil.rmtree(self.tmp_dir)

    def _parse_doc_summary(self, content: str) -> str | None:
        """Parse summary from markdown content (blockquote after title)."""
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Look for blockquote line (starts with "> ")
            if line.startswith("> "):
                return line[2:].strip()
            # Stop searching after first few lines
            if i > 5:
                break
        return None

    def _upload_agent_docs(self) -> None:
        """Upload all agent documentation files to the vectorstore and generate index."""
        if self.agent_docs_dir is None or not self.agent_docs_dir.exists():
            return

        # Find all markdown files in agent_docs (docs_*.md files)
        md_files = sorted(self.agent_docs_dir.glob("docs_*.md"))

        # Also find operation-specific docs in operations/ subdirectory
        ops_dir = self.agent_docs_dir / "operations"
        op_files = sorted(ops_dir.glob("*.md")) if ops_dir.exists() else []

        all_md_files = md_files + op_files

        if not all_md_files:
            return

        # Build docs index from summaries
        index_lines = []

        # Main docs first
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            summary = self._parse_doc_summary(content)
            if summary:
                index_lines.append(f"- `{md_file.name}`: {summary}")

        # Operation docs with header
        op_summaries = []
        for md_file in op_files:
            content = md_file.read_text(encoding="utf-8")
            summary = self._parse_doc_summary(content)
            if summary:
                op_name = md_file.stem  # e.g., "js_evaluate" from "js_evaluate.md"
                op_summaries.append(f"  - `{op_name}`: {summary}")

        if op_summaries:
            index_lines.append("\n**Operations:**")
            index_lines.extend(op_summaries)

        if index_lines:
            self.docs_index = "\n".join(index_lines)

        # Create tmp dir for combined doc
        os.makedirs(self.tmp_dir, exist_ok=True)

        # Combine all docs into a single file for better context
        combined_docs = []

        # Main docs first
        for md_file in md_files:
            content = md_file.read_text(encoding="utf-8")
            combined_docs.append(f"# FILE: {md_file.name}\n\n{content}")

        # Operation docs
        for md_file in op_files:
            content = md_file.read_text(encoding="utf-8")
            combined_docs.append(f"# FILE: operations/{md_file.name}\n\n{content}")

        combined_content = "\n\n---\n\n".join(combined_docs)

        # Write combined documentation
        combined_path = os.path.join(self.tmp_dir, "agent_documentation.md")
        with open(combined_path, mode="w", encoding="utf-8") as f:
            f.write(combined_content)

        # Upload to vectorstore with documentation type
        self.add_file_to_vectorstore(
            combined_path,
            {"filename": "agent_documentation.md", "type": "documentation"}
        )

    def get_all_transaction_ids(self) -> list[str]:
        """
        Get all transaction ids that have a response body file with a supported extension.
        Cached per instance to avoid repeated filesystem operations.
        """
        if self.cached_transaction_ids is not None:
            return self.cached_transaction_ids

        all_transaction_ids = os.listdir(self.transactions_dir)
        supported_transaction_ids = []
        for transaction_id in all_transaction_ids:
            if self.get_response_body_file_extension(transaction_id) in self.supported_file_extensions:
                supported_transaction_ids.append(transaction_id)

        self.cached_transaction_ids = supported_transaction_ids
        return supported_transaction_ids

    def get_transaction_by_id(self, transaction_id: str, clean_response_body: bool = False) -> dict:
        """
        Get a transaction by id from the context manager.
        {
            "request": ...
            "response": ...
            "response_body": ...
        }
        """
        result = {}

        try:
            with open(os.path.join(self.transactions_dir, transaction_id, "request.json"), mode="r") as f:
                result["request"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            result["request"] = f"No request found for transaction {transaction_id}"

        try:
            with open(os.path.join(self.transactions_dir, transaction_id, "response.json"), mode="r") as f:
                result["response"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            result["response"] = f"No response found for transaction {transaction_id}"

        # Get the response body file extension
        response_body_extension = self.get_response_body_file_extension(transaction_id)

        if response_body_extension is None:
            result["response_body"] = f"No response body found for transaction {transaction_id}"
        else:
            response_body_path = os.path.join(
                self.transactions_dir, transaction_id, f"response_body{response_body_extension}"
            )
            if response_body_extension == ".json":
                try:
                    with open(response_body_path, mode="r", encoding="utf-8") as f:
                        result["response_body"] = json.load(f)
                except json.JSONDecodeError:
                    with open(response_body_path, mode="r", encoding='utf-8', errors='replace') as f:
                        result["response_body"] = f.read()
            else:
                with open(response_body_path, mode="r", encoding='utf-8', errors='replace') as f:
                    result["response_body"] = f.read()

                    if response_body_extension == ".html" and clean_response_body:
                        result["response_body"] = get_text_from_html(result["response_body"])
        return result

    def clean_up(self) -> None:
        """Clean up the context manager resources."""
        if self.vectorstore_id is None:
            raise ValueError("Vectorstore ID is not set")
        try:
            self.client.vector_stores.delete(vector_store_id=self.vectorstore_id)
        except Exception as e:
            raise ValueError(f"Failed to delete vectorstore: {e}")
        self.vectorstore_id = None

    def add_transaction_to_vectorstore(self, transaction_id: str, metadata: dict) -> None:
        """Add a single transaction to the vectorstore."""
        if self.vectorstore_id is None:
            raise ValueError("Vectorstore ID is not set")

        if transaction_id in self.uploaded_transaction_ids:
            return

        os.makedirs(self.tmp_dir, exist_ok=True)

        try:
            transaction_data = self.get_transaction_by_id(transaction_id, clean_response_body=True)
            transaction_file_path = os.path.join(self.tmp_dir, f"{transaction_id}.json")

            with open(transaction_file_path, mode="w", encoding="utf-8") as f:
                json.dump(transaction_data, f, ensure_ascii=False, indent=2)

            self.add_file_to_vectorstore(transaction_file_path, {**metadata, "type": "transaction"})

        finally:
            shutil.rmtree(self.tmp_dir)
            self.uploaded_transaction_ids.add(transaction_id)

    def add_file_to_vectorstore(self, file_path: str, metadata: dict) -> None:
        """Add a file to the vectorstore."""
        assert self.vectorstore_id is not None, "Vectorstore ID is not set"

        file_name = os.path.basename(file_path)

        with open(file_path, mode="rb") as f:
            uploaded = self.client.files.create(
                file=f,
                purpose="assistants",
            )

        self.client.vector_stores.files.create(
            vector_store_id=self.vectorstore_id,
            file_id=uploaded.id,
            attributes={
                "filename": file_name,
                **metadata
            }
        )

    def get_transaction_ids_by_request_url(self, request_url: str) -> list[str]:
        """Get all transaction ids by request url."""
        all_transaction_ids = self.get_all_transaction_ids()
        transaction_ids = []
        for transaction_id in all_transaction_ids:
            try:
                request_path = os.path.join(self.transactions_dir, transaction_id, "request.json")
                with open(request_path, mode="r", encoding="utf-8") as f:
                    request_data = json.load(f)
                    if request_data.get("url") == request_url:
                        transaction_ids.append(transaction_id)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                continue
        return transaction_ids

    def get_transaction_timestamp(self, transaction_id: str) -> float:
        """Get the timestamp of a transaction."""
        parts = transaction_id.split("_")
        if len(parts) < 2:
            raise ValueError(f"Invalid transaction_id format: {transaction_id}")
        unix_timestamp = parts[1]
        try:
            return float(unix_timestamp)
        except ValueError as e:
            raise ValueError(f"Invalid timestamp in transaction_id '{transaction_id}': {str(e)}")

    def scan_transaction_responses(self, value: str, max_timestamp: float | None = None) -> list[str]:
        """Scan the network transaction responses for a value."""
        all_transaction_ids = self.get_all_transaction_ids()
        results = []
        for transaction_id in all_transaction_ids:
            transaction = self.get_transaction_by_id(transaction_id)
            if (
                value in str(transaction["response_body"])
                and (max_timestamp is None or self.get_transaction_timestamp(transaction_id) < max_timestamp)
            ):
                results.append(transaction_id)
        return list(set(results))

    def scan_storage_for_value(self, value: str) -> list[str]:
        """Scan the storage for a value."""
        results = []
        with open(self.storage_jsonl_path, mode="r", encoding='utf-8', errors='replace') as f:
            for line in f:
                if value in line:
                    results.append(line)
        return results

    def scan_window_properties_for_value(self, value: str) -> list[dict]:
        """Scan the window properties for a value."""
        result = []
        with open(self.window_properties_path, mode="r", encoding='utf-8', errors='replace') as f:
            window_properties = json.load(f)
            for key, window_property_value in window_properties.items():
                if value in str(window_property_value):
                    result.append({key: window_property_value})
        return result

    def get_response_body_file_extension(self, transaction_id: str) -> str | None:
        """Get the extension of the response body file for a transaction."""
        files = os.listdir(os.path.join(self.transactions_dir, transaction_id))
        for file in files:
            if file.startswith("response_body"):
                return os.path.splitext(file)[1].lower()
        return None
