from pydantic import BaseModel, field_validator
from openai import OpenAI
import os
import json
import time
import shutil



class ContextManager(BaseModel):
    
    client: OpenAI
    tmp_dir: str
    transactions_dir: str
    consolidated_transactions_path: str
    storage_jsonl_path: str
    vectorstore_id: str | None = None
    
    class Config:
        arbitrary_types_allowed = True
    
    
    @field_validator('transactions_dir', 'consolidated_transactions_path', 'storage_jsonl_path')
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
        with open(self.storage_jsonl_path, "r") as storage_jsonl_file:
            for line in storage_jsonl_file:
                obj = json.loads(line)
                storage_data.append(obj)
        
        # create a single storage.json file
        storage_file_path = os.path.join(self.tmp_dir, "storage.json")
        with open(storage_file_path, "w") as f:
            json.dump(storage_data, f, ensure_ascii=False, indent=2)
                    
        # upload the storage to the vectorstore using add_file_to_vectorstore method
        self.add_file_to_vectorstore(storage_file_path, {"filename": "storage.json"})
            
        # delete the tmp directory
        shutil.rmtree(self.tmp_dir)
        

    def get_all_transaction_ids(self) -> list[str]:
        """
        Get all transaction ids from the context manager.
        """
        return os.listdir(self.transactions_dir)

    
    def get_transaction_by_id(self, transaction_id: str) -> dict:
        """
        Get a transaction by id from the context manager.
        {
            "request": ...
            "response": ...
            "response_body": ...
        }
        """
        
        result = {}
        with open(os.path.join(self.transactions_dir, transaction_id, "request.json"), "r") as f:
            result["request"] = json.load(f)
        with open(os.path.join(self.transactions_dir, transaction_id, "response.json"), "r") as f:
            result["response"] = json.load(f)
        
        # Handle response body - try JSON first, fallback to text
        response_body_path = os.path.join(self.transactions_dir, transaction_id, "response_body.json")
        try:
            with open(response_body_path, "r") as f:
                result["response_body"] = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            # If it's not valid JSON or file doesn't exist, read as text
            with open(response_body_path, "r", encoding='utf-8', errors='replace') as f:
                result["response_body"] = f.read()
            
        return result
    
    
    def delete_vectorstore(self) -> None:
        """
        Delete the vectorstore from the context manager.
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
        
        # make the tmp directory
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        try:
            # get the entire transaction data
            transaction_data = self.get_transaction_by_id(transaction_id)
            transaction_file_path = os.path.join(self.tmp_dir, f"{transaction_id}.json")
            
            with open(transaction_file_path, "w") as f:
                json.dump(transaction_data, f, ensure_ascii=False, indent=2)
                
            # upload the transaction to the vectorstore using the add_file_to_vectorstore method
            self.add_file_to_vectorstore(transaction_file_path, metadata)
            
        finally:
            # delete the tmp directory
            shutil.rmtree(self.tmp_dir)
        
        
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
        with open(file_path, "rb") as f:
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
                with open(request_path, "r") as f:
                    request_data = json.load(f)
                    if request_data.get("url") == request_url:
                        transaction_ids.append(transaction_id)
            except (FileNotFoundError, json.JSONDecodeError, KeyError):
                # Skip transactions with missing or malformed request files
                continue
        return transaction_ids
        