import json
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from src.routine_discovery.context_manager import ContextManager
from src.utils.llm_utils import llm_parse_text_to_model, collect_text_from_response
from src.data_models.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    TransactionConfirmationResponse,
)
from uuid import uuid4
import os


class RoutineDiscoveryAgent(BaseModel):
    """
    Agent for discovering routines from the network transactions.
    """
    client: OpenAI
    context_manager: ContextManager
    task_description: str
    llm_model: str = "gpt-5-mini"
    message_history: list[dict] = Field(default_factory=list)
    debug_dir: str
    last_response_id: str | None = None
    tools: list[dict] = Field(default_factory=list)
    n_transaction_identification_attempts: int = 3
    current_transaction_identification_attempt: int = 0
    
    class Config:
        arbitrary_types_allowed = True
        
    SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS: str = f"""
    You are a helpful assistant that is an expert in parsing network traffic.
    You need to identify one or more network transactions that directly correspond to the user's requested task.
    You have access to vectorstore that contains network transactions and storage data
    (cookies, localStorage, sessionStorage, etc.).
    """
        
        
    def run(self) -> None:
        """
        Run the routine discovery agent.
        """
        
        # make the output dir if specified
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # validate the context manager
        assert self.context_manager.vectorstore_id is not None, "Vectorstore ID is not set"
        
        # construct the tools
        self.tools = [
            {
                "type": "file_search",
                "vector_store_ids": [self.context_manager.vectorstore_id],
            }
        ]
        
        # add the system prompt to the message history
        self.add_to_message_history("system", self.SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS)
        
        # add the user prompt to the message history
        self.add_to_message_history("user", f"Task description: {self.task_description}")
        
        self.add_to_message_history("user", f"These are the possible network transaction ids you can choose from: {self.context_manager.get_all_transaction_ids()}")

        # Step 1: Identify the network transactions that directly correspond to the user's requested task
        identified_transaction = None
        
        while identified_transaction is None:
            # identify the transaction
            identified_transaction = self.identify_transaction()
            
            # confirm the identified transaction
            confirmation_response = self.confirm_indetified_transaction(identified_transaction)
            
            # if the identified transaction is not correct, try again
            if not confirmation_response.is_correct:
                identified_transaction = None
                self.current_transaction_identification_attempt += 1
                
        if identified_transaction is None:
            raise Exception("Failed to identify the network transactions that directly correspond to the user's requested task.")
        
        # save the indentified transactions
        with open(os.path.join(self.debug_dir, "identified_transactions.json"), "w") as f:
            json.dump(identified_transaction.model_dump(), f, ensure_ascii=False, indent=2)
        
        # Step 2: Extract the variables from the identified transactions
        extracted_variables = self.extract_variables(identified_transaction)
        
        with open(os.path.join(self.debug_dir, "extracted_variables.json"), "w") as f:
            json.dump(extracted_variables.model_dump(), f, ensure_ascii=False, indent=2)
        
        
    def identify_transaction(self) -> TransactionIdentificationResponse:
        """
        Identify the network transactions that directly correspond to the user's requested task.
        """
        if self.current_transaction_identification_attempt == 0:
            self.message_history = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS
                },
                {
                    "role": "user",
                    "content": f"Task description: {self.task_description}"
                },
                {
                    "role": "user",
                    "content": f"These are the possible network transaction ids you can choose from: {self.context_manager.get_all_transaction_ids()}"
                },
                {
                    "role": "user",
                    "content": f"Please respond in the following format: {TransactionIdentificationResponse.model_json_schema()}"
                }
            ]
        else:
            message = (
                f"Please try again to identify the network transactions that directly correspond to the user's requested task."
                f"Respond in the following format: {TransactionIdentificationResponse.model_json_schema()}"
            )
            self.add_to_message_history("user", message)
        
        # call to the LLM API
        response = self.client.responses.create(
            model=self.llm_model,
            input=self.message_history if self.current_transaction_identification_attempt == 0 else [self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="required",
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self.add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-3:]]),
            pydantic_model=TransactionIdentificationResponse,
            client=self.client,
            llm_model=self.llm_model
        )
        self.add_to_message_history("assistant", parsed_response.model_dump_json())
        
        # return the parsed response
        return parsed_response


    def confirm_indetified_transaction(
        self,
        identified_transaction: TransactionIdentificationResponse,
    ) -> TransactionConfirmationResponse:
        """
        Confirm the identified network transaction that directly corresponds to the user's requested task.
        """
        
        # add the transaction to the vectorstore
        metadata = {"uuid": str(uuid4())}
        self.context_manager.add_transaction_to_vectorstore(
            transaction_id=identified_transaction.transaction_id, metadata=metadata
        )
        
        # temporarily update the tools to specifically search through these transactions
        tools = [
            {
                "type": "file_search",
                "vector_store_ids": [self.context_manager.vectorstore_id],
                "filters": {
                    "type": "eq",
                    "key": "uuid",
                    "value": [metadata["uuid"]]
                }
            }
        ]
        
        # update the message history with request to confirm the identified transaction
        message = (
            f"{identified_transaction.transaction_id} have been added to the vectorstore in full (including response bodies)."
            "Please confirm that the identified transaction is correct and that it directly corresponds to the user's requested task:"
            f"{self.task_description}"
            f"Please respond in the following format: {TransactionConfirmationResponse.model_json_schema()}"
        )
        self.add_to_message_history("user", message)
        
        # call to the LLM API for confirmation that the identified transaction is correct
        response = self.client.responses.create(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=tools,
            tool_choice="required", # forces the LLM to look at the newly added files to the vectorstore
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self.add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-3:]]),
            pydantic_model=TransactionConfirmationResponse,
            client=self.client,
            llm_model=self.llm_model
        )
        
        return parsed_response
    
    
    def extract_variables(
        self,
        identified_transactions: TransactionIdentificationResponse,
    ) -> ExtractedVariableResponse:
        """
        Extract the variables from the identified transaction.
        """
        
        # get all transaction ids by request url
        transaction_ids = self.context_manager.get_transaction_ids_by_request_url(identified_transactions.url)
        
        # get the requests of the identified transactions
        transactions = []
        for transaction_id in transaction_ids:
            transaction = self.context_manager.get_transaction_by_id(transaction_id)
            
            # Handle response_body - truncate if it's a string
            response_body = transaction["response_body"]
            if isinstance(response_body, str) and len(response_body) > 300:
                response_body = response_body[:300] + "..."
            elif isinstance(response_body, (dict, list)):
                # If it's JSON data, convert to string and truncate
                response_body_str = json.dumps(response_body, ensure_ascii=False)
                if len(response_body_str) > 300:
                    response_body = response_body_str[:300] + "..."
                else:
                    response_body = response_body_str
            
            transactions.append(
                {
                    "request": transaction["request"],
                    "response": transaction["response"],
                    "response_body": response_body
                }
            )
        
        # add message to the message history
        message = (
            f"Please extract the variables from the requests of identified network transactions:"
            f"{transactions}"
            f"Please respond in the following format: {ExtractedVariableResponse.model_json_schema()}"
        )
        self.add_to_message_history("user", message)
        
        # call to the LLM API for extraction of the variables
        response = self.client.responses.create(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="auto",
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self.message_history.append({"role": "assistant","content": response_text})
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-3:]]),
            pydantic_model=ExtractedVariableResponse,
            client=self.client,
            llm_model=self.llm_model
        )
        self.add_to_message_history("assistant", parsed_response.model_dump_json())
        
        return parsed_response
    

    def add_to_message_history(self, role: str, content: str) -> None:
        self.message_history.append({"role": role, "content": content})
        with open(os.path.join(self.debug_dir, "message_history.json"), "w") as f:
            json.dump(self.message_history, f, ensure_ascii=False, indent=2)