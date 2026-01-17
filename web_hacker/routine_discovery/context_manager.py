from pydantic import BaseModel, field_validator, Field, ConfigDict
from openai import OpenAI
from abc import ABC, abstractmethod
import os
import json
import time
import shutil

from web_hacker.utils.data_utils import get_text_from_html


class DiscoveryDataStore(BaseModel, ABC):
    """Abstract base class for managing discovery data (captures, docs, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def make_vectorstore(self) -> None:
        """Make a vectorstore from the context."""
        pass

    @abstractmethod
    def get_all_transaction_ids(self) -> list[str]:
        """Get all transaction ids from the context manager."""
        pass

    @abstractmethod
    def get_transaction_by_id(self, transaction_id: str, clean_response_body: bool = False) -> dict:
        """Get a transaction by id from the context manager."""
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
        """Clean up the context manager resources."""
        pass


class LocalDiscoveryDataStore(DiscoveryDataStore):

    client: OpenAI
    tmp_dir: str
    transactions_dir: str
    consolidated_transactions_path: str
    storage_jsonl_path: str
    window_properties_path: str
    vectorstore_id: str | None = None
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
    
    
    def make_vectorstore(self) -> None:
        """Make a vectorstore from the context."""

        # make the tmp directory
        os.makedirs(self.tmp_dir, exist_ok=True)

        # ensure no vectorstore for this context already exists
        if self.vectorstore_id is not None:
            raise ValueError(f"Vectorstore ID is already exists: {self.vectorstore_id}")

        # make the vectorstore
        vs = self.client.vector_stores.create(
            name=f"api-extraction-context-{int(time.time())}"
        )

        # save the vectorstore id
        self.vectorstore_id = vs.id

        # upload the transactions to the vectorstore using add_file_to_vectorstore method
        self.add_file_to_vectorstore(self.consolidated_transactions_path, {"filename": "consolidated_transactions.json"})

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

        # upload the storage to the vectorstore using add_file_to_vectorstore method
        self.add_file_to_vectorstore(storage_file_path, {"filename": "storage.json"})
        
        # upload the window properties to the vectorstore using add_file_to_vectorstore method
        self.add_file_to_vectorstore(self.window_properties_path, {"filename": "window_properties.json"})

        # delete the tmp directory
        shutil.rmtree(self.tmp_dir)


    def get_all_transaction_ids(self) -> list[str]:
        """
        Get all transaction ids from the context manager that have a response body file with a supported extension.
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

        # Get the response body file extension to determine how to read it
        response_body_extension = self.get_response_body_file_extension(transaction_id)
        
        if response_body_extension is None:
            result["response_body"] = f"No response body found for transaction {transaction_id}"
            
        else:
            response_body_path = os.path.join(self.transactions_dir, transaction_id, f"response_body{response_body_extension}")
            # If it's a JSON file, try to parse as JSON; otherwise read as text
            if response_body_extension == ".json":
                try:
                    with open(response_body_path, mode="r", encoding="utf-8") as f:
                        result["response_body"] = json.load(f)
                except json.JSONDecodeError:
                    # Fallback to text if JSON parsing fails
                    with open(response_body_path, mode="r", encoding='utf-8', errors='replace') as f:
                        result["response_body"] = f.read()
            else:
                # Read as text for .txt, .js, .html, .xml, etc.
                with open(response_body_path, mode="r", encoding='utf-8', errors='replace') as f:
                    result["response_body"] = f.read()
                    
                    # sanitize the response body if it's an html file and clean_response_body is True
                    if response_body_extension == ".html" and clean_response_body:
                        result["response_body"] = get_text_from_html(result["response_body"])
        return result

    def clean_up(self) -> None:
        """
        Clean up the context manager resources.
        """
        if self.vectorstore_id is None:
            raise ValueError("Vectorstore ID is not set")
        try:
            self.client.vector_stores.delete(vector_store_id=self.vectorstore_id)
        except Exception as e:
            raise ValueError(f"Failed to delete vectorstore: {e}")
        self.vectorstore_id = None


    def add_transaction_to_vectorstore(self, transaction_id: str, metadata: dict) -> None:
        """
        Add a single transaction to the vectorstore.
        Args:
            transaction_id: ID of the transaction to add
            metadata: Metadata to attach to the transaction file
        """
        if self.vectorstore_id is None:
            raise ValueError("Vectorstore ID is not set")
        
        if transaction_id in self.uploaded_transaction_ids:
            return

        # make the tmp directory
        os.makedirs(self.tmp_dir, exist_ok=True)

        try:
            # get the entire transaction data
            transaction_data = self.get_transaction_by_id(transaction_id, clean_response_body=True)
            transaction_file_path = os.path.join(self.tmp_dir, f"{transaction_id}.json")

            with open(transaction_file_path, mode="w", encoding="utf-8") as f:
                json.dump(transaction_data, f, ensure_ascii=False, indent=2)
            # upload the transaction to the vectorstore using the add_file_to_vectorstore method
            self.add_file_to_vectorstore(transaction_file_path, metadata)

        finally:
            # delete the tmp directory
            shutil.rmtree(self.tmp_dir)
            
            # add the transaction id to the uploaded transaction ids
            self.uploaded_transaction_ids.add(transaction_id)


    def add_file_to_vectorstore(self, file_path: str, metadata: dict) -> None:
        """
        Add a file to the vectorstore.
        
        Args:
            file_path: Path to the file to upload
            metadata: Metadata to attach to the file
        """
        assert self.vectorstore_id is not None, "Vectorstore ID is not set"

        # get the file name
        file_name = os.path.basename(file_path)

        # Create the raw file
        with open(file_path, mode="rb") as f:
            uploaded = self.client.files.create(
                file=f,
                purpose="assistants",
            )

        # Attach file to vector store with attributes
        self.client.vector_stores.files.create(
            vector_store_id=self.vectorstore_id,
            file_id=uploaded.id,
            attributes={
                "filename": file_name,
                **metadata
            }
        )


    def get_transaction_ids_by_request_url(self, request_url: str) -> list[str]:
        """
        Get all transaction ids by request url.
        Efficiently reads only the request.json file instead of the entire transaction.
        """
        all_transaction_ids = self.get_all_transaction_ids()
        transaction_ids = []
        for transaction_id in all_transaction_ids:
            try:
                # Only read the request.json file to check the URL
                request_path = os.path.join(self.transactions_dir, transaction_id, "request.json")
                with open(request_path, mode="r", encoding="utf-8") as f:
                    request_data = json.load(f)
                    if request_data.get("url") == request_url:
                        transaction_ids.append(transaction_id)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Skip transactions with missing or malformed request files
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
        results = []
        with open(self.storage_jsonl_path, mode="r", encoding='utf-8', errors='replace') as f:
            for line in f:
                if value in line:
                    results.append(line)
        return results
    
    def scan_window_properties_for_value(self, value: str) -> list[dict]:
        """
        Scan the window properties for a value.
        Args:
            value: The value to scan for in the window properties.
        Returns:
            A list of window properties that contain the value.
        """
        result = []
        with open(self.window_properties_path, mode="r", encoding='utf-8', errors='replace') as f:
            window_properties = json.load(f)

            for key, window_property_value in window_properties.items():
                if value in str(window_property_value):
                    result.append({key: window_property_value})

        return result

    def get_response_body_file_extension(self, transaction_id: str) -> str:
        """
        Get the extension of the response body file for a transaction.
        Args:
            transaction_id: The id of the transaction.
        Returns:
            The extension of the response body file.
        """
        # get all files in the transaction directory
        files = os.listdir(os.path.join(self.transactions_dir, transaction_id))
        for file in files:
            if file.startswith("response_body"):
                return os.path.splitext(file)[1].lower()
        return None