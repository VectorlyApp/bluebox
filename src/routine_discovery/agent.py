import json
from pydantic import BaseModel, Field, field_validator
from openai import OpenAI
from src.routine_discovery.context_manager import ContextManager
from src.utils.llm_utils import llm_parse_text_to_model, collect_text_from_response, manual_llm_parse_text_to_model
from src.data_models.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    TransactionConfirmationResponse,
    VariableType,
    ResolvedVariableResponse,
    TestParametersResponse
)
from src.data_models.production_routine import Routine as ProductionRoutine
from src.data_models.dev_routine import Routine, RoutineFetchOperation
from uuid import uuid4
import os


class RoutineDiscoveryAgent(BaseModel):
    """
    Agent for discovering routines from the network transactions.
    """
    client: OpenAI
    context_manager: ContextManager
    task: str
    llm_model: str = "gpt-5-mini"
    message_history: list[dict] = Field(default_factory=list)
    output_dir: str
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
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        self._add_to_message_history("system", self.SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS)
        
        # add the user prompt to the message history
        self._add_to_message_history("user", f"Task description: {self.task}")
        self._add_to_message_history("user", f"These are the possible network transaction ids you can choose from: {self.context_manager.get_all_transaction_ids()}")

        print("Identifying the network transaction that directly corresponds to the user's requested task...")
        
        identified_transaction = None
        
        while identified_transaction is None:
            # identify the transaction
            identified_transaction = self.identify_transaction()
            
            if identified_transaction.transaction_id is None:
                raise Exception("Failed to identify the network transactions that directly correspond to the user's requested task.")
            
            # confirm the identified transaction
            confirmation_response = self.confirm_indetified_transaction(identified_transaction)
            
            # if the identified transaction is not correct, try again
            if not confirmation_response.is_correct:
                identified_transaction = None
                self.current_transaction_identification_attempt += 1
                
        if identified_transaction is None:
            raise Exception("Failed to identify the network transactions that directly correspond to the user's requested task.")
        
        # save the indentified transactions
        save_path = os.path.join(self.output_dir, "root_transaction.json")
        with open(save_path, "w") as f:
            json.dump(identified_transaction.model_dump(), f, ensure_ascii=False, indent=2)
            
        print(f"Identified transaction: {identified_transaction.transaction_id} saved to: {save_path}")
        
        # populating the transaction queue with the identified transaction
        transaction_queue = [identified_transaction.transaction_id]
        
        # storing data for all transactions necessary for the routine construction
        routine_transactions = {}
        
        # processing the transaction queue (breadth-first search)
        while (len(transaction_queue) > 0):
            
            # make the output directory for the transaction
            os.makedirs(os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}"), exist_ok=True)
            
            # dequeue the transaction
            transaction_id = transaction_queue.pop(0)
            print(f"Processing transaction: {transaction_id}")
            
            # get the transaction
            transaction = self.context_manager.get_transaction_by_id(transaction_id)
            
            # extract variables from the transaction
            print("Extracting variables (args, cookies, tokens, browser variables) from the identified transaction...")
            extracted_variables = self.extract_variables(transaction_id)
            
            # save the extracted variables
            save_path = os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}", "extracted_variables.json")
            with open(save_path, "w") as f:
                json.dump(extracted_variables.model_dump(), f, ensure_ascii=False, indent=2)
            print(f"Extracted variables saved to: {save_path}")
                
            # resolve cookies and tokens
            print("Resolving cookies and tokens...")
            resolved_variables = self.resolve_variables(extracted_variables)
            resolved_variables_json = [resolved_variable.model_dump() for resolved_variable in resolved_variables]
            
            # save the resolved variables
            save_path = os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}", "resolved_variables.json")
            with open(save_path, "w") as f:
                json.dump(resolved_variables_json, f, ensure_ascii=False, indent=2)
            print(f"Resolved variables saved to: {save_path}")
                        
            # adding transaction data to the routine transactions
            routine_transactions[transaction_id] = {
                "request": transaction["request"],
                "extracted_variables": extracted_variables.model_dump(),
                "resolved_variables": [resolved_variable.model_dump() for resolved_variable in resolved_variables]
            }
            
            # adding transaction that need to be processed to the queue
            for resolved_variable in resolved_variables:
                if resolved_variable.transaction_source is not None:
                    new_transaction_id = resolved_variable.transaction_source.transaction_id
                    if new_transaction_id not in routine_transactions:
                        transaction_queue.append(new_transaction_id)
            
        # construct the routine
        routine = self.construct_routine(routine_transactions)

        # save the routine
        save_path = os.path.join(self.output_dir, f"routine.json")
        with open(save_path, "w") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2) 
        print(f"Routine saved to: {save_path}")
        
        # productionize the routine
        print(f"Productionizing the routine...")
        routine = self.productionize_routine(routine)
        with open(save_path, "w") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2) 
        print(f"Routine saved to: {save_path}")
    
        # get the test parameters
        print(f"Getting test parameters...")
        test_parameters = self.get_test_parameters(routine)
        test_parameters_dict = {value.name: value.value for value in test_parameters.parameters}
        
        # save the test parameters
        save_path = os.path.join(self.output_dir, f"test_parameters.json")
        with open(save_path, "w") as f:
            json.dump(test_parameters_dict, f, ensure_ascii=False, indent=2)
        print(f"Test parameters saved to: {save_path}")
        
    
        
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
                    "content": f"Task description: {self.task}"
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
            self._add_to_message_history("user", message)
            
        
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
        self._add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
            pydantic_model=TransactionIdentificationResponse,
            client=self.client,
            llm_model='gpt-5-nano'
        )
        self._add_to_message_history("assistant", parsed_response.model_dump_json())
        
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
            f"{self.task}"
            f"Please respond in the following format: {TransactionConfirmationResponse.model_json_schema()}"
        )
        self._add_to_message_history("user", message)
        
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
        self._add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
            pydantic_model=TransactionConfirmationResponse,
            client=self.client,
            llm_model='gpt-5-nano'
        )
        
        return parsed_response
    
    
    def extract_variables(self, transaction_id: str) -> ExtractedVariableResponse:
        """
        Extract the variables from the transaction.
        """
        
        # get the transaction
        transaction = self.context_manager.get_transaction_by_id(transaction_id)
        
        # get all transaction ids by request url
        transaction_ids = self.context_manager.get_transaction_ids_by_request_url(transaction["request"]["url"])
        
        # get the requests of the identified transactions
        transactions = []
        for transaction_id in transaction_ids:
            transaction = self.context_manager.get_transaction_by_id(transaction_id)
            
            # Handle response_body - truncate if it's a string
            response_body = transaction["response_body"]
            if response_body is None:
                response_body = "No response body found"
            if isinstance(response_body, str) and len(response_body) > 500:
                response_body = response_body[:500] + "..."
            elif isinstance(response_body, (dict, list)):
                # If it's JSON data, convert to string and truncate
                response_body_str = json.dumps(response_body, ensure_ascii=False)
                if len(response_body_str) > 500:
                    response_body = response_body_str[:500] + "..."
                else:
                    response_body = response_body_str
            
            transactions.append(
                {
                    "request": transaction["request"],
                    # "response": transaction["response"],
                    # "response_body": response_body
                }
            )
        
        # add message to the message history
        message = (
            f"Please extract the variables from only these network requests (requests only!): {transactions}"
            f"Please respond in the following format: {ExtractedVariableResponse.model_json_schema()}"
            "Mark each variable with requires_resolution=True if we need to dynamically resolve this variable at runtime."
            "If we can most likely hardcode this value, mark requires_resolution=False."
            "system variables are related to the device or browser environment, and are not used to identify the user."
            "token and cookie values are not used to identify the user: these may need to be resolved at runtime."
            "Only the actual values of the variables (token/cookies, etc.) should be placed into the observed_value field."
            "The values of values_to_scan_for will then be used to scan the storage and transactions for the source of the variable so only include the actual values of the variables."
            "values_to_scan_for should be possible substrings that will likely be present in the response body of a network transaction or a storage entry value."
            "This is necessary to figure out where the variable is coming from."
        )
        
        self._add_to_message_history("user", message)
        
        # call to the LLM API for extraction of the variables
        response = self.client.responses.create(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="required" if len(transactions) > 1 else "auto",
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self.message_history.append({"role": "assistant","content": response_text})
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
            pydantic_model=ExtractedVariableResponse,
            client=self.client,
            llm_model="gpt-5-nano"
        )
        self._add_to_message_history("assistant", parsed_response.model_dump_json())
        
        return parsed_response
    
    def resolve_variables(self, extracted_variables: ExtractedVariableResponse) -> list[ResolvedVariableResponse]:
        """
        Resolve the variables from the extracted variables.
        """
        
        # get the latest timestamp
        max_timestamp = self.context_manager.get_transaction_timestamp(extracted_variables.transaction_id)
        
        # get a list of cookies and tokens that require resolution
        variables_to_resolve = [
            var for var in extracted_variables.variables if var.requires_resolution and
            var.type in [
                VariableType.COOKIE,
                VariableType.TOKEN
            ]
        ]
        
        resolved_variable_responses = []
        
        # for each variable to resolve, try to find the source of the variable in the storage and transactions
        for variable in variables_to_resolve:
            
            print(f"Resolving variable: {variable.name} with values to scan for: {variable.values_to_scan_for}")
            
            # get the storage objects that contain the value and are before the latest timestamp
            storage_objects = []
            for value in variable.values_to_scan_for:
                storage_sources = self.context_manager.scan_storage_for_value(
                    value=value
                )
                storage_sources.extend(storage_sources)
                
            if len(storage_sources) > 0:
                print(f"Found {len(storage_sources)} storage sources that contain the value")
            
            # get the transaction ids that contain the value and are before the latest timestamp
            transaction_ids = {}
            for value in variable.values_to_scan_for:
                transaction_ids.extend(self.context_manager.scan_transaction_responses(
                    value=value, max_timestamp=max_timestamp
                ))
                
            # deduplicate transaction ids
            transaction_ids = list(set(transaction_ids))
                
            if len(transaction_ids) > 0:
                print(f"Found {len(transaction_ids)} transaction ids that contain the value: {transaction_ids}")

            # add the transactions to the vectorstore
            uuid = str(uuid4())
            for transaction_id in transaction_ids:
                self.context_manager.add_transaction_to_vectorstore(
                    transaction_id=transaction_id,
                    metadata={"uuid": uuid}
                )
            
            # construct the message to the LLM
            message = (
                f"Please resolve the variable: {variable.observed_value}"
                f"The variable was found in the following storage sources: {storage_sources}"
                f"The variable was found in the following transactions ids: {transaction_ids}"
                f"These transactions are added to the vectorstore in full (including response bodies)."
                f"Please respond in the following format: {ResolvedVariableResponse.model_json_schema()}"
                f"Dot paths should be like this: 'key.data.items[0].id', 'path.to.valiable.0.value', etc."
                f"For paths in transaction responses, start with the first key of the response body"
                f"For paths in storage, start with the cookie, local storage, or session storage entry name"
                f"If the variable is found in both storage and transactions, you should indicate both sources and resolve them accordinly!"
            )
            self._add_to_message_history("user", message)
            
            # custom tools to force the LLM to look at the newly added transactions to the vectorstore
            tools = [
                {
                    "type": "file_search",
                    "vector_store_ids": [self.context_manager.vectorstore_id],
                    "filters": {
                        "type": "eq",
                        "key": "uuid",
                        "value": [uuid]
                    }
                }
            ]
            
            # call to the LLM API for resolution of the variable
            response = self.client.responses.create(
                model=self.llm_model,
                input=[self.message_history[-1]],
                previous_response_id=self.last_response_id,
                tools=tools,
                tool_choice="required",
            )
            
            # save the response id
            self.last_response_id = response.id
            
            # collect the text from the response
            response_text = collect_text_from_response(response)
            self._add_to_message_history("assistant", response_text)
            
            # parse the response to the pydantic model
            parsed_response = llm_parse_text_to_model(
                text=response_text,
                context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
                pydantic_model=ResolvedVariableResponse,
                client=self.client,
                llm_model="gpt-5-nano"
            )
            self._add_to_message_history("assistant", parsed_response.model_dump_json())
            
            resolved_variable_responses.append(parsed_response)
            
            if not parsed_response.session_storage_source and not parsed_response.transaction_source:
                print(f"[WARNING] Not able to resolve variable: {parsed_response.variable.name}. Harcoding to observed value: {parsed_response.variable.observed_value}")
                
            if parsed_response.session_storage_source and parsed_response.transaction_source:
                print(f"[INFO] Variable: {parsed_response.variable.name} is resolved from both session storage and transaction. It is reasonable to use either source (fetch is slower but more stable)")
            
        return resolved_variable_responses


    def construct_routine(self, routine_transactions: dict, max_attempts: int = 3) -> Routine:
        """
        Construct the routine from the routine transactions.
        """
        message = (
            f"Please construct the routine from the routine transactions: {routine_transactions}. "
            f"Please respond in the following format: {Routine.model_json_schema()}. "
            f"Fetch operations (1 to 1 with transactions) should be constructed as follows: {RoutineFetchOperation.model_json_schema()}. "
            f"First step of the routine should be to navigate to the target web page and sleep for a bit of time (2-3 seconds). "
            f"All fetch operations should be constructed as follows: {RoutineFetchOperation.model_json_schema()}. "
            f"Parameters are only the most important arguments. "
            f"You can inject variables by using placeholders. CRITICAL: PLACEHOLDERS ARE REPLACED AT RUNTIME AND THE RESULT MUST BE VALID JSON! "
            f"For STRING values: Use \\\"{{{{parameter_name}}}}\\\" format (escaped quote + placeholder + escaped quote). "
            f"Example: \\\"name\\\": \\\"\\\"{{{{user_name}}}}\\\"\\\". At runtime, \\\"\\\"{{{{user_name}}}}\\\"\\\" is replaced and becomes \\\"name\\\": \\\"John\\\" (valid JSON string). "
            f"For NUMERIC values (int, float) or NULL: Use \\\"{{{{parameter_name}}}}\\\" format (regular quote + placeholder + quote). "
            f"Example: \\\"amount\\\": \\\"{{{{price}}}}\\\". At runtime, \\\"{{{{price}}}}\\\" is replaced with the numeric value and quotes are removed, becoming \\\"amount\\\": 99.99 (JSON number, not string). "
            f"Example: \\\"quantity\\\": \\\"{{{{count}}}}\\\" with value 5 becomes \\\"quantity\\\": 5 (JSON number). "
            f"For NULL: \\\"metadata\\\": \\\"{{{{optional_field}}}}\\\" with null value becomes \\\"metadata\\\": null (JSON null). "
            f"REMEMBER: After placeholder replacement, the JSON must be valid and parseable! "
            f"Placeholder types: {{{{parameter_name}}}} for parameters, {{{{cookie:cookie_name}}}} for cookies, {{{{sessionStorage:key.path.to.0.value}}}} for session storage, {{{{localStorage:local_storage_name}}}} for local storage. "
            f"You can hardcode unresolved variables to their observed values. "
            f"You will want to navigate to the target page, then perform the fetch operations in the proper order. "
            f"Browser variables should be hardcoded to observed values. "
            f"If tokens or cookies are resolved, they should point to values in the session storage."
            f"You can navigate to other pages in the routine by using the navigate operation. "
            f"Endpoints of the fetch operations should mimick observed network traffic requests! "
            f"Every fetch operation result is written to session storage. "
            f"At the end of the routine return the proper session storage value (likely containing the last fetch operation result). "
            f"To feed output of a fetch into a subsequent fetch, you can save result to session storage and then use {{sessionStorage:key.to.path}}. "
        )
        self._add_to_message_history("user", message)
        
        current_attempt = 0
        
        while current_attempt < max_attempts:
            
            current_attempt += 1
            
            # call to the LLM API for construction of the routine
            response = self.client.responses.create(
                model=self.llm_model,
                input=[self.message_history[-1]],
                previous_response_id=self.last_response_id,
                tools=self.tools,
                tool_choice="required",
            )
            
            # save the response id
            self.last_response_id = response.id
            
            # collect the text from the response
            response_text = collect_text_from_response(response)
            self._add_to_message_history("assistant", response_text)
            
            # parse the response to the pydantic model
            routine = llm_parse_text_to_model(
                text=response_text,
                context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
                pydantic_model=Routine,
                client=self.client,
                llm_model=self.llm_model
            )
            self._add_to_message_history("assistant", routine.model_dump_json())
            
            
            # validate the routine
            successful, errors, exception = routine.validate()
            if successful:
                return routine
            
            message = (
                f"Execution failed with error: {exception}"
                f"Routine validation failed: {errors}"
                f"Please try again to construct the routine."
            )
            self._add_to_message_history("user", message)
            
            
        raise Exception(f"Failed to construct the routine after {max_attempts} attempts")
    
    def productionize_routine(self, routine: Routine) -> Routine:
        
        message = (
            f"Please productionize the routine (from previosu step): {routine.model_dump_json()}"
            f"You need to clean up this routine to follow the following format: {ProductionRoutine.model_json_schema()}"
            f"Please respond in the following format: {ProductionRoutine.model_json_schema()}"
            f"You immediate output needs to be a valid JSON object that conforms to the production routine schema."
            f"CRITICAL: PLACEHOLDERS ARE REPLACED AT RUNTIME AND MUST RESULT IN VALID JSON! "
            f"EXPLANATION: Placeholders like {{{{key}}}} are replaced at runtime with actual values. The format you choose determines the resulting JSON type. "
            f"For STRING values: Use \\\"{{{{key}}}}\\\" format (escaped quote + placeholder + escaped quote). "
            f"This means in the JSON file you write: \\\"\\\"{{{{user_name}}}}\\\"\\\". At runtime, the \\\"{{{{user_name}}}}\\\" part gets replaced, "
            f"so \\\"\\\"{{{{user_name}}}}\\\"\\\" becomes \\\"\\\"John\\\"\\\" which becomes \\\"John\\\" (valid JSON string). "
            f"For NUMERIC/NULL values: Use \\\"{{{{key}}}}\\\" format (regular quote + placeholder + quote). "
            f"This means in the JSON file you write: \\\"{{{{item_id}}}}\\\". At runtime, the {{{{item_id}}}} part gets replaced with the number, "
            f"and the surrounding quotes are removed, so \\\"{{{{item_id}}}}\\\" with value 42 becomes just 42 (valid JSON number, not string). "
            f"Example: \\\"{{{{total_price}}}}\\\" with value 29.99 → becomes 29.99 (quotes removed, valid JSON number). "
            f"Example: \\\"{{{{optional_data}}}}\\\" with null → becomes null (quotes removed, valid JSON null). "
            """Placeholders will be resolved using this: param_pattern = r'(?:"|\\")\{\{([^}"]*)\}\}(?:"|\\")'"""
            f"The resulting JSON MUST be valid and parseable after all placeholder replacements are done."
        )
        self._add_to_message_history("user", message)
    
        
        # call to the LLM API for productionization of the routine
        response = self.client.responses.create(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self._add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        # context includes the last 2 messages (user prompt + assistant response) to help with parsing
        production_routine = manual_llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
            pydantic_model=ProductionRoutine,
            client=self.client,
            llm_model=self.llm_model
        )
        
        return production_routine
        
    
    def get_test_parameters(self, routine: Routine) -> TestParametersResponse:
        """
        Get the test parameters for the routine.
        """
        message = (
            f"Write a dictionary of parameters to test this routine (from previous step): {routine.model_dump_json()}"
            f"Please respond in the following format: {TestParametersResponse.model_json_schema()}"
            f"Ensure all parameters are present and have valid values."
        )
        self._add_to_message_history("user", message)
        
        # call to the LLM API for getting the test parameters
        response = self.client.responses.create(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
        )
        
        # save the response id
        self.last_response_id = response.id
        
        # collect the text from the response
        response_text = collect_text_from_response(response)
        self._add_to_message_history("assistant", response_text)
        
        # parse the response to the pydantic model
        parsed_response = llm_parse_text_to_model(
            text=response_text,
            context="\n".join([f"{msg['role']}: {msg['content']}" for msg in self.message_history[-2:]]),
            pydantic_model=TestParametersResponse,
            client=self.client,
            llm_model="gpt-5-nano"
        )
        self._add_to_message_history("assistant", parsed_response.model_dump_json())
        
        return parsed_response


    def _add_to_message_history(self, role: str, content: str) -> None:
        self.message_history.append({"role": role, "content": content})
        with open(os.path.join(self.output_dir, "message_history.json"), "w") as f:
            json.dump(self.message_history, f, ensure_ascii=False, indent=2)
