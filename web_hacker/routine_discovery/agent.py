"""
web_hacker/routine_discovery/agent.py

Agent for discovering routines from the network transactions.
"""

import json
from uuid import uuid4
import os

from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict

from web_hacker.config import Config
from web_hacker.routine_discovery.context_manager import ContextManager
from web_hacker.utils.llm_utils import collect_text_from_response, manual_llm_parse_text_to_model
from web_hacker.data_models.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    TransactionConfirmationResponse,
    VariableType,
    ResolvedVariableResponse,
    TestParametersResponse
)
from web_hacker.data_models.routine.routine import Routine
from web_hacker.data_models.routine.dev_routine import DevRoutine
from web_hacker.utils.exceptions import TransactionIdentificationFailedError
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)


class RoutineDiscoveryAgent(BaseModel):
    """
    Agent for discovering routines from the network transactions.
    """
    client: OpenAI
    context_manager: ContextManager
    task: str
    llm_model: str = "gpt-5.1"
    message_history: list[dict] = Field(default_factory=list)
    output_dir: str
    last_response_id: str | None = None
    tools: list[dict] = Field(default_factory=list)
    n_transaction_identification_attempts: int = 3
    current_transaction_identification_attempt: int = 0

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        logger.info("Identifying the network transaction that directly corresponds to the user's requested task...")
        logger.debug(f"\n\nMessage history:\n{self.message_history}\n\n")

        identified_transaction = None
        while identified_transaction is None:
            
            # identify the transaction
            identified_transaction = self.identify_transaction()
            logger.debug(f"\nIdentified transaction:\n{identified_transaction.model_dump_json()}")

            # ensure the identified transaction not None
            if identified_transaction.transaction_id is None:
                # get vars
                description = identified_transaction.description if identified_transaction.description is not None else 'No description provided'
                explanation = identified_transaction.explanation if identified_transaction.explanation is not None else 'No explanation provided'
                url = identified_transaction.url if identified_transaction.url is not None else 'No URL provided'
                confidence_level = identified_transaction.confidence_level if identified_transaction.confidence_level is not None else 'No confidence level provided'

                # construct the error message
                error_message = (
                    "Failed to identify the network transactions that directly correspond to the user's requested task.\n"
                    f"- Description: {description}\n"
                    f"- Explanation: {explanation}\n"
                    f"- URL: {url}\n"
                    f"- Confidence level: {confidence_level}\n"
                )
                logger.error(error_message)
                
                if self.current_transaction_identification_attempt == self.n_transaction_identification_attempts:
                    raise TransactionIdentificationFailedError(error_message)
                else:
                    self.current_transaction_identification_attempt += 1
                    continue
                
            # ensure the identified transaction is in the context manager (not hallucinated)
            if identified_transaction.transaction_id not in self.context_manager.get_all_transaction_ids():
                logger.error(f"Identified transaction: {identified_transaction.transaction_id} is not in the context manager.")
                if self.current_transaction_identification_attempt == self.n_transaction_identification_attempts:
                    raise TransactionIdentificationFailedError("Identified transaction is not in the context manager.")
                else:
                    self.current_transaction_identification_attempt += 1
                    continue

            # confirm the identified transaction is correct and directly corresponds to the user's requested task
            confirmation_response = self.confirm_identified_transaction(identified_transaction)
            logger.debug(f"\nConfirmation response:\n{confirmation_response.model_dump_json()}")

            # if the identified transaction is not correct, try again
            if not confirmation_response.is_correct:
                identified_transaction = None
                self.current_transaction_identification_attempt += 1
                logger.debug(
                    "Trying again to identify the network transaction that directly corresponds to the user's requested task... "
                    f"(attempt {self.current_transaction_identification_attempt})"
                )

        if identified_transaction is None:
            logger.error("Failed to identify the network transactions that directly correspond to the user's requested task.")
            raise TransactionIdentificationFailedError(
                "Failed to identify the network transactions that directly correspond to the user's requested task."
            )
        logger.info(f"Identified transaction: {identified_transaction.transaction_id}")

        # save the indentified transactions
        save_path = os.path.join(self.output_dir, "root_transaction.json")
        with open(save_path, mode="w", encoding="utf-8") as f:
            json.dump(obj=identified_transaction.model_dump(), fp=f, ensure_ascii=False, indent=2)

        logger.info(f"Identified transaction: {identified_transaction.transaction_id} saved to: {save_path}")

        # populating the transaction queue with the identified transaction
        transaction_queue = [identified_transaction.transaction_id]

        # storing data for all transactions necessary for the routine construction
        routine_transactions = {}
        
        # storing all resolved variables
        all_resolved_variables = []

        # processing the transaction queue (breadth-first search)
        while (len(transaction_queue) > 0):

            # make the output directory for the transaction
            os.makedirs(os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}"), exist_ok=True)

            # dequeue the transaction
            transaction_id = transaction_queue.pop(0)
            logger.info(f"Processing transaction: {transaction_id}")

            # get the transaction
            transaction = self.context_manager.get_transaction_by_id(transaction_id)
            
            # extract variables from the transaction
            logger.info("Extracting variables (args, cookies, tokens, browser variables) from the identified transaction...")
            extracted_variables = self.extract_variables(transaction_id)
            
            # save the extracted variables
            save_path = os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}", "extracted_variables.json")
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(extracted_variables.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info(f"Extracted variables saved to: {save_path}")
                
            # resolve cookies and tokens
            logger.info("Resolving cookies and tokens...")
            resolved_variables = self.resolve_variables(extracted_variables)
            all_resolved_variables.extend(resolved_variables)
            resolved_variables_json = [resolved_variable.model_dump() for resolved_variable in resolved_variables]
            
            # save the resolved variables
            save_path = os.path.join(self.output_dir, f"transaction_{len(routine_transactions)}", "resolved_variables.json")
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(resolved_variables_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Resolved variables saved to: {save_path}")

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

        # construct the routine from the routine transactions and resolved variables
        # REVERSE the order of transactions because the queue processing discovers dependencies LAST (Target -> Dep1 -> Dep2),
        # but we need to execute dependencies FIRST (Dep2 -> Dep1 -> Target).
        ordered_transactions = dict(reversed(list(routine_transactions.items())))
        
        routine = self.construct_routine(routine_transactions=ordered_transactions, resolved_variables=all_resolved_variables)

        # save the routine
        save_path = os.path.join(self.output_dir, f"routine.json")
        with open(save_path, mode="w", encoding="utf-8") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2) 
        logger.info(f"Routine saved to: {save_path}")
        
        # productionize the routine
        logger.info(f"Productionizing the routine...")
        routine = self.productionize_routine(routine)
        with open(save_path, mode="w", encoding="utf-8") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2) 
        logger.info(f"Routine saved to: {save_path}")
    
        # get the test parameters
        logger.info(f"Getting test parameters...")
        test_parameters = self.get_test_parameters(routine)
        test_parameters_dict = {value.name: value.value for value in test_parameters.parameters}
        
        # save the test parameters
        save_path = os.path.join(self.output_dir, f"test_parameters.json")
        with open(save_path, mode="w", encoding="utf-8") as f:
            json.dump(test_parameters_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Test parameters saved to: {save_path}")

    def identify_transaction(self) -> TransactionIdentificationResponse:
        """
        Identify the network transactions that directly correspond to the user's requested task.
        Returns:
            TransactionIdentificationResponse: The response from the LLM API.
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
            ]
        else:
            message = (
                f"Please try again to identify the network transactions that directly correspond to the user's requested task."
                f"The transaction id you provided does not exist or the request was not considered relevant to the user's requested task."
                f"These are the possible network transaction ids you can choose from: {self.context_manager.get_all_transaction_ids()}"
                f"Respond in the following format: {TransactionIdentificationResponse.model_json_schema()}"
            )
            self._add_to_message_history("user", message)
        
        logger.debug(f"\n\nMessage history:\n{self.message_history}\n")

        # call to the LLM API
        response = self.client.responses.parse(
            model=self.llm_model,
            input=self.message_history if self.current_transaction_identification_attempt == 0 else [self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="required",
            text_format=TransactionIdentificationResponse
        )
        transaction_identification_response = response.output_parsed
        logger.info(f"\nTransaction identification response:\n{transaction_identification_response.model_dump()}")

        # save the response id and add to the message history
        self.last_response_id = response.id
        self._add_to_message_history("assistant", transaction_identification_response.model_dump_json())

        logger.debug(f"\nParsed response:\n{transaction_identification_response.model_dump_json()}")
        logger.debug(f"New chat history:\n{self.message_history}\n")

        # return the parsed response
        return transaction_identification_response

    def confirm_identified_transaction(
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
        )
        self._add_to_message_history("user", message)
        
        # call to the LLM API for confirmation that the identified transaction is correct
        response = self.client.responses.parse(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=tools,
            tool_choice="required", # forces the LLM to look at the newly added files to the vectorstore
            text_format=TransactionConfirmationResponse
        )
        transaction_confirmation_response = response.output_parsed
        
        # save the response id and add to the message history
        self.last_response_id = response.id
        self._add_to_message_history("assistant", transaction_confirmation_response.model_dump_json())
        
        return transaction_confirmation_response

    def extract_variables(self, transaction_id: str) -> ExtractedVariableResponse:
        """
        Extract the variables from the transaction.
        """
        # save the original transaction_id before it gets shadowed in the loop
        original_transaction_id = transaction_id

        # get the transaction
        transaction = self.context_manager.get_transaction_by_id(transaction_id)

        # get all transaction ids by request url
        transaction_ids = self.context_manager.get_transaction_ids_by_request_url(request_url=transaction["request"]["url"])

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
            
            transactions.append({"request": transaction["request"]})
        
        # add message to the message history
        message = (
            f"Extract variables from these network REQUESTS only: {transactions}\n\n"
            "CRITICAL RULES:\n"
            "1. **requires_resolution=False (STATIC_VALUE)**: Default to this. HARDCODE values whenever possible.\n"
            "   - Includes: App versions, constants, User-Agents, device info.\n"
            "   - **API CONTEXT**: Fields like 'hl' (language), 'gl' (region), 'clientName', 'timeZone' are STATIC_VALUE, NOT parameters.\n"
            "   - **TELEMETRY**: Fields like 'adSignals', 'screenHeight', 'clickTrackingParams' are STATIC_VALUE.\n"
            "2. **requires_resolution=True (DYNAMIC_TOKEN)**: ONLY for dynamic security tokens that change per session.\n"
            "   - Includes: CSRF tokens, JWTs, Auth headers, 'visitorData', 'session_id'.\n"
            "   - **TRACE/REQUEST IDs**: 'x-trace-id', 'request-id', 'correlation-id' MUST be marked as DYNAMIC_TOKEN.\n"
            "   - **ALSO INCLUDE**: IDs, hashes, or blobs that are NOT user inputs but are required for the request (e.g. 'browseId', 'params' strings, 'clientVersion' if dynamic).\n"
            "   - 'values_to_scan_for' must contain the EXACT raw string value seen in the request.\n"
            "   - **RULE**: If it looks like a generated ID or state blob, IT IS A TOKEN, NOT A PARAMETER.\n"
            "3. **Parameters (PARAMETER)**: ONLY for values that represent the USER'S INTENT or INPUT.\n"
            "   - Examples: 'search_query', 'videoId', 'channelId', 'cursor', 'page_number'.\n"
            "   - If the user wouldn't explicitly provide it, it's NOT a parameter.\n"
        )

        self._add_to_message_history("user", message)

        # call to the LLM API for extraction of the variables
        response = self.client.responses.parse(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="required" if len(transactions) > 1 else "auto",
            text_format=ExtractedVariableResponse
        )
        extracted_variable_response = response.output_parsed

        # save the response id and add to the message history
        self.last_response_id = response.id
        self._add_to_message_history("assistant", extracted_variable_response.model_dump_json())

        # override the transaction_id with the one passed in, since the LLM may return an incorrect format
        extracted_variable_response.transaction_id = original_transaction_id

        return extracted_variable_response
    
    def resolve_variables(self, extracted_variables: ExtractedVariableResponse) -> list[ResolvedVariableResponse]:
        """
        Resolve the variables from the extracted variables.
        """
        # get the latest timestamp
        max_timestamp = self.context_manager.extract_timestamp_from_transaction_id(extracted_variables.transaction_id)

        # get a list of cookies and tokens that require resolution
        variables_to_resolve = [
            var for var in extracted_variables.variables
            if (
                var.requires_resolution
                and var.type == VariableType.DYNAMIC_TOKEN
            )
        ]

        resolved_variable_responses = []

        # for each variable to resolve, try to find the source of the variable in the storage and transactions
        for variable in variables_to_resolve:
            logger.info(f"Resolving variable: {variable.name} with values to scan for: {variable.values_to_scan_for}")

            # get the storage objects that contain the value and are before the latest timestamp
            storage_objects = []
            for value in variable.values_to_scan_for:
                storage_sources = self.context_manager.scan_storage_for_value(
                    value=value,
                )
                storage_objects.extend(storage_sources)
                
            if len(storage_objects) > 0:
                logger.info(f"Found {len(storage_objects)} storage sources that contain the value")

            # get the window properties that contain the value and are before the latest timestamp
            window_properties = []
            for value in variable.values_to_scan_for:
                window_properties.extend(self.context_manager.scan_window_properties_for_value(value))
                
            if len(window_properties) > 0:
                logger.info(f"Found {len(window_properties)} window properties that contain the value")

            # get the transaction ids that contain the value and are before the latest timestamp
            transaction_ids = []
            for value in variable.values_to_scan_for:
                transaction_ids_found = self.context_manager.scan_transaction_responses(
                    value=value,
                    max_timestamp=max_timestamp
                )
                transaction_ids.extend(transaction_ids_found)

            # deduplicate transaction ids
            transaction_ids = list(set(transaction_ids))

            if len(transaction_ids) > 0:
                logger.info(f"Found {len(transaction_ids)} transaction ids that contain the value: {transaction_ids}")

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
                f"The variable was found in the following storage sources: {storage_objects}"
                f"The variable was found in the following window properties: {window_properties}"
                f"The variable was found in the following transactions ids: {transaction_ids}"
                f"These transactions are added to the vectorstore in full (including response bodies)."
                f"Dot paths should be like this: 'key.data.items[0].id', 'path.to.variable.0.value', etc."
                f"For paths in transaction responses, start with the first key of the response body"
                f"For paths in storage, start with the cookie, local storage, or session storage entry name"
                f"If the variable is found in multiple places you need to resolve all of them!"
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
            response = self.client.responses.parse(
                model=self.llm_model,
                input=[self.message_history[-1]],
                previous_response_id=self.last_response_id,
                tools=tools,
                tool_choice="required",
                text_format=ResolvedVariableResponse
            )
            resolved_variable_response = response.output_parsed

            # save the response id
            self.last_response_id = response.id
            self._add_to_message_history("assistant", resolved_variable_response.model_dump_json())
            
            # parse the response to the pydantic model
            resolved_variable_responses.append(resolved_variable_response)
            
            logger.info(f"Resolved variable response: {resolved_variable_response.model_dump()}")
            
            
            resolved_transaction_source = resolved_variable_response.transaction_source
            logger.info(f"Resolved transaction source: {resolved_transaction_source}")
            resolved_session_storage_source = resolved_variable_response.session_storage_source
            logger.info(f"Resolved session storage source: {resolved_session_storage_source}")
            resolved_window_property_source = resolved_variable_response.window_property_source
            logger.info(f"Resolved window property source: {resolved_window_property_source}")
            
            # number of sources resolved
            sources = [resolved_transaction_source, resolved_session_storage_source, resolved_window_property_source]
            
            logger.info(f"Sources: {sources}")
            sources = [source for source in sources if source is not None]
            logger.info(f"Sources after filtering: {sources}")
            num_sources_resolved = sources.count(True)
            logger.info(f"Number of sources resolved: {num_sources_resolved}")
            
            if num_sources_resolved == 0:
                logger.info(f"[WARNING] Not able to resolve variable: {variable.name}. Hardcoding to observed value: {variable.observed_value}")
            elif num_sources_resolved == 1:
                logger.info(f"[INFO] Variable: {variable.name} is resolved from one source. It is reasonable to use that source.")
            elif num_sources_resolved >= 2:
                logger.info(f"[INFO] Variable: {variable.name} is resolved from multiple sources (will prioritize transaction > session storage > window property).")
                
            
        return resolved_variable_responses

    def construct_routine(self, routine_transactions: dict, resolved_variables: list[ResolvedVariableResponse] = [], max_attempts: int = 3) -> DevRoutine:
        """
        Construct the routine from the routine transactions.
        """
        message = (
            f"Please construct the routine from the routine transactions: {routine_transactions}."
            f"These are the resolved variables: {resolved_variables}"
            f"IMPORTANT: The transactions are listed in EXECUTION ORDER (Dependencies first -> Target last). "
            f"You should generally create fetch operations in the order provided."
            f"First step of the routine should be to navigate to the target web page and sleep for a bit of time (2-3 seconds). "
            f"Parameters are only the most important arguments. "
            f"You can inject variables by using placeholders. CRITICAL: PLACEHOLDERS ARE REPLACED AT RUNTIME AND THE RESULT MUST BE VALID JSON! "
            f"For STRING values: Use \\\"{{{{parameter_name}}}}\\\" format (escaped quote + placeholder + escaped quote). "
            f"Example: \\\"name\\\": \\\"\\\"{{{{user_name}}}}\\\"\\\". At runtime, \\\"\\\"{{{{user_name}}}}\\\"\\\" is replaced and becomes \\\"name\\\": \\\"John\\\" (valid JSON string). "
            f"For NUMERIC values (int, float) or NULL: Use \\\"{{{{parameter_name}}}}\\\" format (regular quote + placeholder + quote). "
            f"Example: \\\"amount\\\": \\\"{{{{price}}}}\\\". At runtime, \\\"{{{{price}}}}\\\" is replaced with the numeric value and quotes are removed, becoming \\\"amount\\\": 99.99 (JSON number, not string). "
            f"Example: \\\"quantity\\\": \\\"{{{{count}}}}\\\" with value 5 becomes \\\"quantity\\\": 5 (JSON number). "
            f"For NULL: \\\"metadata\\\": \\\"{{{{optional_field}}}}\\\" with null value becomes \\\"metadata\\\": null (JSON null). "
            f"REMEMBER: After placeholder replacement, the JSON must be valid and parseable! "
            f"Placeholder types: {{{{parameter_name}}}} for parameters, {{{{cookie:cookie_name}}}} for cookies, {{{{sessionStorage:key.path.to.0.value}}}} for session storage, {{{{localStorage:local_storage_name}}}} for local storage, {{{{windowProperty:key}}}} for window properties. "
            f"You can hardcode unresolved variables to their observed values. "
            f"You will want to navigate to the target page, then perform the fetch operations in the proper order. "
            f"Browser variables should be hardcoded to observed values. "
            f"If tokens or cookies are resolved, they should point to values in the session storage."
            f"You can navigate to other pages in the routine by using the navigate operation. "
            f"Endpoints of the fetch operations should mimick observed network traffic requests! "
            f"Every fetch operation result is written to session storage. "
            f"At the end of the routine return the proper session storage value (likely containing the last fetch operation result). "
            f"To feed output of a fetch into a subsequent fetch, you can save result to session storage and then use {{{{sessionStorage:key.to.path}}}}. "
            f"Credentials should be set to same-origin if possible. If not possible, set to include. The absolute last resort is to set to omit."
        )
        self._add_to_message_history("user", message)

        current_attempt = 0
        while current_attempt < max_attempts:
            current_attempt += 1
            
            # call to the LLM API for construction of the routine
            response = self.client.responses.parse(
                model=self.llm_model,
                input=[self.message_history[-1]],
                previous_response_id=self.last_response_id,
                tools=self.tools,
                tool_choice="required",
                text_format=DevRoutine
            )
            routine = response.output_parsed
            logger.info(f"\nRoutine:\n{routine.model_dump()}")
            
            # save the response id
            self.last_response_id = response.id
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

    def productionize_routine(self, routine: DevRoutine) -> Routine:
        """
        Productionize the routine into a production routine.
        Args:
            routine (Routine): The routine to productionize.
        Returns:
            Routine: The productionized routine.
        """
        message = (
            f"Please productionize the routine (from previous step): {routine.model_dump_json()}"
            f"You need to clean up this routine to follow the following format: {Routine.model_json_schema()}"
            f"You immediate output needs to be a valid JSON object that conforms to the production routine schema."
            f"CRITICAL: PLACEHOLDERS ARE REPLACED AT RUNTIME AND MUST RESULT IN VALID JSON! "
            f"EXPLANATION: Placeholders like {{{{key}}}} are replaced at runtime with actual values. The format you choose determines the resulting JSON type. "
            f"For STRING values: Use \\\"{{{{key}}}}\\\" format (escaped quote + placeholder + escaped quote). "
            f"This means in the JSON file you write: \"\\\"{{{{user_name}}}}\\\"\". At runtime, the \\\"{{{{user_name}}}}\\\" part gets replaced, "
            f"so \"\\\"{{{{user_name}}}}\\\"\" becomes \"John\" (valid JSON string). "
            f"For NUMERIC/NULL values: Use \"{{{{key}}}}\" format (regular quote + placeholder + quote). "
            f"This means in the JSON file you write: \"{{{{item_id}}}}\". At runtime, the {{{{item_id}}}} part gets replaced with the number, "
            f"and the surrounding quotes are removed, so \"{{{{item_id}}}}\" with value 42 becomes just 42 (valid JSON number, not string). "
            f"Example: \"{{{{total_price}}}}\" with value 29.99 → becomes 29.99 (quotes removed, valid JSON number). "
            f"Example: \"{{{{optional_data}}}}\" with null → becomes null (quotes removed, valid JSON null). "
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
            pydantic_model=Routine,
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
        response = self.client.responses.parse(
            model=self.llm_model,
            input=[self.message_history[-1]],
            previous_response_id=self.last_response_id,
            text_format=TestParametersResponse
        )
        test_parameters_response = response.output_parsed
        
        # save the response id
        self.last_response_id = response.id
        self._add_to_message_history("assistant", test_parameters_response.model_dump_json())
        
        # return the test parameters response
        return test_parameters_response

    def _add_to_message_history(self, role: str, content: str) -> None:
        """
        Add a message to the message history.
        """
        self.message_history.append(
            {"role": role, "content": content}
        )
        with open(os.path.join(self.output_dir, "message_history.json"), mode="w", encoding="utf-8") as f:
            json.dump(self.message_history, f, ensure_ascii=False, indent=2)
