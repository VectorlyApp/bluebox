"""
web_hacker/routine_discovery/agent.py

Agent for discovering routines from the network transactions.
"""

import json
from uuid import uuid4
import os
from typing import Callable

from openai import OpenAI
from pydantic import BaseModel, Field, ConfigDict
from toon import encode

from web_hacker.routine_discovery.data_store import DiscoveryDataStore
from web_hacker.utils.llm_utils import collect_text_from_response, manual_llm_parse_text_to_model
from web_hacker.data_models.routine_discovery.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    TransactionConfirmationResponse,
    VariableType,
    ResolvedVariableResponse,
    TestParametersResponse
)
from web_hacker.data_models.routine_discovery.message import (
    RoutineDiscoveryMessage,
    RoutineDiscoveryMessageType,
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
    data_store: DiscoveryDataStore
    task: str
    emit_message_callable: Callable[[RoutineDiscoveryMessage], None]
    llm_model: str = Field(default="gpt-5.1")
    message_history: list[dict] = Field(default_factory=list)
    output_dir: str | None = Field(default=None)
    last_response_id: str | None = Field(default=None)
    tools: list[dict] = Field(default_factory=list)
    n_transaction_identification_attempts: int = Field(default=3)
    current_transaction_identification_attempt: int = Field(default=1)
    timeout: int = Field(default=600)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS: str = """
    You are a helpful assistant that is an expert in parsing network traffic.
    You need to identify one or more network transactions that directly correspond to the user's requested task.
    You have access to vectorstore that contains network transactions and storage data
    (cookies, localStorage, sessionStorage, etc.).
    """

    PLACEHOLDER_INSTRUCTIONS: str = (
        "PLACEHOLDER SYNTAX:\n"
        "- PARAMS: {{param_name}} (NO prefix, name matches parameter definition)\n"
        "- SOURCES (use dot paths): {{cookie:name}}, {{sessionStorage:path.to.value}}, {{localStorage:key}}, {{windowProperty:obj.key}}\n\n"
        "JSON VALUE RULES (TWO sets of quotes needed for strings!):\n"
        '- String: "key": \\"{{x}}\\"  (OUTER quotes = JSON string, INNER \\" = escaped quotes around placeholder)\n'
        '- Number/bool/null: "key": "{{x}}"  (only outer quotes, they get stripped)\n'
        '- Inside larger string: "prefix\\"{{x}}\\"suffix"  (escaped quotes wrap placeholder)\n\n'
        "EXAMPLES:\n"
        '1. String param:     "name": \\"{{username}}\\"           -> "name": "john"\n'
        '2. Number param:     "count": "{{limit}}"                -> "count": 50\n'
        '3. Bool param:       "active": "{{is_active}}"           -> "active": true\n'
        '4. String in string: "msg_\\"{{id}}\\""                  -> "msg_abc"\n'
        '5. Number in string: "page\\"{{num}}\\""                 -> "page5"\n'
        '6. URL with param:   "/api/\\"{{user_id}}\\"/data"       -> "/api/123/data"\n'
        '7. Session storage:  "token": \\"{{sessionStorage:auth.access_token}}\\"\n'
        '8. Cookie:           "sid": \\"{{cookie:session_id}}\\"'
        'IMPORTANT: YOU MUST ENSURE THAT EACH PLACEHOLDER IS SURROUNDED BY QUOTES OR ESCAPED QUOTES!'
    )

    def _save_to_output_dir(self, relative_path: str, data: dict | list | str) -> None:
        """Save data to output_dir if it is specified."""
        if self.output_dir is None:
            return
        save_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(data, dict) or isinstance(data, list):
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif isinstance(data, str):
            with open(save_path, mode="w", encoding="utf-8") as f:
                f.write(data)

    def _handle_transaction_identification_failure(self) -> None:
        """Handle failure to identify a valid transaction."""
        logger.error(
            "_handle_transaction_identification_failure: current_transaction_identification_attempt: %d, n_transaction_identification_attempts: %d",
            self.current_transaction_identification_attempt,
            self.n_transaction_identification_attempts
        )
        error_message = (
            "Failed to identify the network transactions that directly correspond to your task.\n\n"
            "Possible fixes:\n"
            "1. Make the task description more specific and detailed.\n"
            "2. Streamline the browser session to reduce noise (close unrelated tabs, avoid extraneous clicks).\n"
        )
        logger.error(error_message)

        if self.current_transaction_identification_attempt >= self.n_transaction_identification_attempts:
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.ERROR,
                content=error_message
            ))
            raise TransactionIdentificationFailedError(error_message)

        self.current_transaction_identification_attempt += 1
        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content=f"Retrying transaction identification (attempt {self.current_transaction_identification_attempt})..."
        ))

    def run(self) -> Routine:
        """
        Run the routine discovery agent.

        Returns:
            Routine: The discovered and productionized routine.
        """
        # validate the data store
        assert self.data_store.cdp_captures_vectorstore_id is not None, "Vectorstore ID is not set"

        # Push initial message
        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.INITIATED,
            content=f"Discovery initiated"
        ))

        # construct the tools with all available vectorstores
        vector_store_ids = [self.data_store.cdp_captures_vectorstore_id]
        if self.data_store.documentation_vectorstore_id is not None:
            vector_store_ids.append(self.data_store.documentation_vectorstore_id)

        self.tools = [
            {
                "type": "file_search",
                "vector_store_ids": vector_store_ids,
            }
        ]

        # add the system prompt to the message history (including data store context)
        system_prompt = self.SYSTEM_PROMPT_IDENTIFY_TRANSACTIONS
        data_store_prompt = self.data_store.generate_data_store_prompt()
        if data_store_prompt:
            system_prompt = f"{system_prompt}\n\n{data_store_prompt}"
        self._add_to_message_history("system", system_prompt)

        # add the user prompt to the message history
        self._add_to_message_history("user", f"Task description: {self.task}")
        self._add_to_message_history("user", f"These are the possible network transaction ids you can choose from:\n{encode(self.data_store.get_all_transaction_ids())}")

        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Identifying relevant network transactions"
        ))
        logger.debug(f"\n\nMessage history:\n{self.message_history}\n\n")

        identified_transaction = None
        while identified_transaction is None:

            # identify the transaction
            identified_transaction = self.identify_transaction()
            logger.debug(f"\nIdentified transaction:\n{identified_transaction.model_dump_json()}")

            # ensure the identified transaction not None
            if identified_transaction.transaction_id in [None, "None", ""]:
                self._handle_transaction_identification_failure()
                identified_transaction = None
                continue

            # ensure the identified transaction is in the data store (not hallucinated)
            if identified_transaction.transaction_id not in self.data_store.get_all_transaction_ids():
                logger.error(f"Identified transaction: {identified_transaction.transaction_id} is not in the data store.")
                self._handle_transaction_identification_failure()
                identified_transaction = None
                continue

            # confirm the identified transaction is correct and directly corresponds to the user's requested task
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content=f"Confirming identified network transaction"
            ))
            confirmation_response = self.confirm_identified_transaction(identified_transaction)
            logger.debug(f"\nConfirmation response:\n{confirmation_response.model_dump_json()}")

            # if the identified transaction is not correct, try again
            if not confirmation_response.is_correct:
                identified_transaction = None
                self._handle_transaction_identification_failure()
                logger.debug(
                    "Trying again to identify the network transaction that directly corresponds to the user's requested task... "
                    f"(attempt {self.current_transaction_identification_attempt})"
                )

        if identified_transaction is None:
            error_msg = "Failed to identify the network transactions that directly correspond to the user's requested task."
            logger.error(error_msg)
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.ERROR,
                content=error_msg
            ))
            raise TransactionIdentificationFailedError(error_msg)

        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content=f"Successfully identified network transaction relevant to the task"
        ))
        logger.info(f"Identified transaction: {identified_transaction.transaction_id}")

        # save the indentified transaction (optional)
        self._save_to_output_dir("root_transaction.json", identified_transaction.model_dump())

        # populating the transaction queue with the identified transaction
        transaction_queue = [identified_transaction.transaction_id]

        # storing data for all transactions necessary for the routine construction
        routine_transactions = {}

        # storing all resolved variables
        all_resolved_variables = []

        # processing the transaction queue (breadth-first search)
        while (len(transaction_queue) > 0):

            # dequeue the transaction
            transaction_id = transaction_queue.pop(0)
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content=f"Processing network transaction {len(routine_transactions) + 1}"
            ))
            logger.info("Processing transaction: %s", transaction_id)

            # extract variables from the transaction
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content="Extracting variables (args, cookies, tokens, browser variables)"
            ))
            logger.info("Extracting variables (args, cookies, tokens, browser variables) from the identified transaction...")
            extracted_variables = self.extract_variables(transaction_id)

            # resolve cookies and tokens
            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content="Resolving cookies, tokens, and api keys"
            ))
            logger.info("Resolving cookies and tokens...")
            resolved_variables = self.resolve_variables(extracted_variables)
            all_resolved_variables.extend(resolved_variables)
            resolved_variables_json = [resolved_variable.model_dump() for resolved_variable in resolved_variables]

            self.emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
                content=f"Successfully processed network transaction {len(routine_transactions) + 1}"
            ))

            # save the extracted and resolved variables (optional)
            self._save_to_output_dir(
                f"transaction_{len(routine_transactions)}/extracted_variables.json",
                extracted_variables.model_dump()
            )
            self._save_to_output_dir(
                f"transaction_{len(routine_transactions)}/resolved_variables.json",
                resolved_variables_json
            )

            # adding transaction data to the routine transactions
            routine_transactions[transaction_id] = {
                "request": self.data_store.get_transaction_by_id(transaction_id)['request'],
                "extracted_variables": extracted_variables.model_dump(),
                "resolved_variables": [resolved_variable.model_dump() for resolved_variable in resolved_variables]
            }

            # adding transactions that need to be processed to the queue
            for resolved_variable in resolved_variables:
                if resolved_variable.transaction_source is not None:
                    new_transaction_id = resolved_variable.transaction_source.transaction_id
                    if new_transaction_id not in routine_transactions:
                        transaction_queue.append(new_transaction_id)

        # construct the routine from the routine transactions and resolved variables
        # REVERSE the order of transactions because the queue processing discovers dependencies LAST (Target -> Dep1 -> Dep2),
        # but we need to execute dependencies FIRST (Dep2 -> Dep1 -> Target).
        # Reverse the items and create a new dict to preserve order (Python 3.7+ preserves dict insertion order)
        ordered_transactions = {k: v for k, v in reversed(list(routine_transactions.items()))}

        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Constructing the template of the routine"
        ))
        dev_routine = self.construct_routine(
            routine_transactions=ordered_transactions,
            resolved_variables=all_resolved_variables
        )

        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content="Successfully constructed the template of the routine"
        ))

        # productionize the routine
        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Productionizing the routine"
        ))
        logger.info(f"Productionizing the routine")
        production_routine = self.productionize_routine(dev_routine)

        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content=f"Productionized the routine"
        ))
        logger.info("Productionized the routine")

        # Save routines
        self._save_to_output_dir("routine.json", production_routine.model_dump())

        # Mark as finished
        self.emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.FINISHED,
            content="Routine generated successfully"
        ))

        return production_routine

    def identify_transaction(self) -> TransactionIdentificationResponse:
        """
        Identify the network transactions that directly correspond to the user's requested task.
        Returns:
            TransactionIdentificationResponse: The response from the LLM API.
        """
        is_first_attempt = self.current_transaction_identification_attempt == 1

        # On retry attempts, add a "try again" message
        if not is_first_attempt:
            message = (
                "Try again. The transaction id you provided does not exist or was not relevant. "
                f"Choose from: {encode(self.data_store.get_all_transaction_ids())}"
            )
            self._add_to_message_history("user", message)

        logger.debug(f"\n\nMessage history:\n{self.message_history}\n")

        # First attempt: send full history. Retries: send only the last message (uses previous_response_id for context)
        response = self.client.responses.parse(
            model=self.llm_model,
            input=self.message_history if is_first_attempt else [self.message_history[-1]],
            previous_response_id=self.last_response_id,
            tools=self.tools,
            tool_choice="required",
            text_format=TransactionIdentificationResponse
        )
        transaction_identification_response = response.output_parsed
        logger.info(f"\nTransaction identification response:\n{transaction_identification_response.model_dump()}")

        # save the response id and add to the message history
        self.last_response_id = response.id
        self._add_to_message_history("assistant", encode(transaction_identification_response.model_dump()))

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
        self.data_store.add_transaction_to_vectorstore(
            transaction_id=identified_transaction.transaction_id, metadata=metadata
        )

        # temporarily update the tools to specifically search through these transactions
        tools = [
            {
                "type": "file_search",
                "vector_store_ids": [self.data_store.cdp_captures_vectorstore_id],
                "filters": {
                    "type": "eq",
                    "key": "uuid",
                    "value": [metadata["uuid"]]
                }
            }
        ]
        
        # update the message history with request to confirm the identified transaction
        message = (
            f"{identified_transaction.transaction_id} have been added to the vectorstore in full (including response bodies).\n"
            "Confirm the identified transaction is correct and directly corresponds to the user's requested task:\n"
            f"{self.task}\n\n"
            "IMPORTANT: Focus on whether this transaction accomplishes the user's INTENT, not the literal wording. "
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
        self._add_to_message_history("assistant", encode(transaction_confirmation_response.model_dump()))
        
        return transaction_confirmation_response

    def extract_variables(self, transaction_id: str) -> ExtractedVariableResponse:
        """
        Extract the variables from the transaction.
        """
        # save the original transaction_id before it gets shadowed in the loop
        original_transaction_id = transaction_id

        # get the transaction
        transaction = self.data_store.get_transaction_by_id(transaction_id)
        
        # add message to the message history
        message = (
            f"Extract variables from these network REQUESTS only: {encode(transaction['request'])}\n\n"
            "CRITICAL RULES:\n"
            "1. **requires_dynamic_resolution=False (STATIC_VALUE)**: Default to this. HARDCODE values whenever possible.\n"
            "   - Includes: App versions, constants, User-Agents, device info.\n"
            "   - **API CONTEXT**: Fields like 'hl' (language), 'gl' (region), 'clientName', 'timeZone' are STATIC_VALUE, NOT parameters.\n"
            "   - **TELEMETRY**: Fields like 'adSignals', 'screenHeight', 'clickTrackingParams' are STATIC_VALUE.\n"
            "2. **requires_dynamic_resolution=True (DYNAMIC_TOKEN)**: ONLY for dynamic security tokens that change per session.\n"
            "   - Includes: CSRF tokens, JWTs, Auth headers, 'visitorData', 'session_id'.\n"
            "   - **TRACE/REQUEST IDs**: 'x-trace-id', 'request-id', 'correlation-id' MUST be marked as DYNAMIC_TOKEN.\n"
            "   - **ALSO INCLUDE**: IDs, hashes, or blobs that are NOT user inputs but are required for the request (e.g. 'browseId', 'params' strings, 'clientVersion' if dynamic).\n"
            "   - 'values_to_scan_for' must contain the EXACT raw string value seen in the request.\n"
            "   - **RULE**: If it looks like a generated ID or state blob, IT IS A TOKEN, NOT A PARAMETER.\n"
            "   - **If the value can be hardcoded, set requires_dynamic_resolution=False (we dont need to waste time figureing out the source)\n"
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
            tool_choice="auto",
            text_format=ExtractedVariableResponse
        )
        extracted_variable_response = response.output_parsed

        # save the response id and add to the message history
        self.last_response_id = response.id
        self._add_to_message_history("assistant", encode(extracted_variable_response.model_dump()))

        # override the transaction_id with the one passed in, since the LLM may return an incorrect format
        extracted_variable_response.transaction_id = original_transaction_id

        return extracted_variable_response
    
    def resolve_variables(self, extracted_variables: ExtractedVariableResponse) -> list[ResolvedVariableResponse]:
        """
        Resolve the variables from the extracted variables.
        """
        # get the latest timestamp
        max_timestamp = self.data_store.get_transaction_timestamp(extracted_variables.transaction_id)

        # get a list of cookies and tokens that require resolution
        variables_to_resolve = [
            var for var in extracted_variables.variables
            if (
                var.requires_dynamic_resolution
                and var.type == VariableType.DYNAMIC_TOKEN
            )
        ]

        resolved_variable_responses = []

        # for each variable to resolve, try to find the source of the variable in the storage and transactions
        for variable in variables_to_resolve:
            logger.info(f"Resolving variable: {variable.name} with values to scan for: {variable.values_to_scan_for}")

            # get the storage objects that contain the value and are before the latest timestamp
            storage_objects: list[dict] = []
            for value in variable.values_to_scan_for:
                storage_sources_found = self.data_store.scan_storage_for_value(
                    value=value,
                )
                # scan_storage_for_value returns list[str], so we can use it directly
                storage_objects.extend(storage_sources_found)
                
            if len(storage_objects) > 0:
                logger.info(f"Found {len(storage_objects)} storage sources that contain the value")

            # get the window properties that contain the value and are before the latest timestamp
            window_properties: list[dict] = []
            for value in variable.values_to_scan_for:
                window_properties_found = self.data_store.scan_window_properties_for_value(value)
                # scan_window_properties_for_value returns list[dict], so we can use it directly
                window_properties.extend(window_properties_found)
                
            if len(window_properties) > 0:
                logger.info(f"Found {len(window_properties)} window properties that contain the value")

            # get the transaction ids that contain the value and are before the latest timestamp
            transaction_ids: list[str] = []
            for value in variable.values_to_scan_for:
                transaction_ids_found = self.data_store.scan_transaction_responses(
                    value=value,
                    max_timestamp=max_timestamp
                )
                transaction_ids.extend(transaction_ids_found)

            # deduplicate transaction ids (limit to 2 for now)
            transaction_ids = list(set(transaction_ids))[:2]

            if len(transaction_ids) > 0:
                logger.info(f"Found {len(transaction_ids)} transaction ids that contain the value: {transaction_ids}")

            # add the transactions to the vectorstore
            uuid = str(uuid4())
            for transaction_id in transaction_ids:
                self.data_store.add_transaction_to_vectorstore(
                    transaction_id=transaction_id,
                    metadata={"uuid": uuid}
                )

            # construct the message to the LLM
            message = (
                f"Resolve variable: {encode(variable.model_dump())}\n\n"
                f"Found in:\n"
                f"- Storage: {encode(storage_objects[:3])}\n"
                f"- Window properties: {encode(window_properties[:3])}\n"
                f"- Transactions (in vectorstore): {encode(transaction_ids)}\n\n"
                "Use dot paths like 'key.data.items[0].id'. For transaction responses, start with first key. "
                "For storage, start with entry name. Resolve ALL occurrences if found in multiple places."
            )
            self._add_to_message_history("user", message)

            # custom tools to force the LLM to look at the newly added transactions to the vectorstore
            tools = [
                {
                    "type": "file_search",
                    "vector_store_ids": [self.data_store.cdp_captures_vectorstore_id],
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
            self._add_to_message_history("assistant", encode(resolved_variable_response.model_dump()))
            
            # parse the response to the pydantic model
            resolved_variable_responses.append(resolved_variable_response)
            
            # Count resolved sources
            resolved_sources = [
                s for s in [
                    resolved_variable_response.transaction_source,
                    resolved_variable_response.session_storage_source,
                    resolved_variable_response.window_property_source,
                ] if s is not None
            ]
            
            if len(resolved_sources) == 0:
                logger.warning(f"Unable to resolve variable '{variable.name}'. Hardcoding to: {variable.observed_value}")
            elif len(resolved_sources) == 1:
                logger.info(f"Variable '{variable.name}' resolved from: {type(resolved_sources[0]).__name__}")
            else:
                logger.info(f"Variable '{variable.name}' resolved from {len(resolved_sources)} sources (prioritizing: transaction > session storage > window property)")
            
        return resolved_variable_responses

    def construct_routine(self, routine_transactions: dict, resolved_variables: list[ResolvedVariableResponse] = [], max_attempts: int = 3) -> DevRoutine:
        """
        Construct the routine from the routine transactions.
        """
        # Convert resolved_variables to dicts for encoding
        resolved_variables_dicts = [rv.model_dump() for rv in resolved_variables]
        
        message = (
            f"Construct routine from transactions:\n{encode(routine_transactions)}\n\n"
            f"Resolved variables:\n{encode(resolved_variables_dicts)}\n\n"
            f"Rules:\n"
            f"1. Transactions are in EXECUTION ORDER (dependencies first -> target last)\n"
            f"2. First step: navigate to target page + sleep 2-3s\n"
            f"3. KEEP PARAMS MINIMAL: only what user MUST provide. If only 1 value observed, hardcode it. Focus on user's original request.\n"
            f"4. {self.PLACEHOLDER_INSTRUCTIONS}\n"
            f"5. Hardcode unresolved variables to observed values\n"
            f"6. Fetch results go to sessionStorage; chain fetches via {{{{sessionStorage:path}}}}\n"
            f"7. Return final sessionStorage value at end\n"
            f"8. Credentials: same-origin > include > omit"
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
            self._add_to_message_history("assistant", encode(routine.model_dump()))
            
            # validate the routine
            successful, errors, exception = routine.validate()
            if successful:
                return routine
            
            message = (
                f"Execution failed with error: {exception}\n\n"
                f"Routine validation failed:\n{encode(errors)}\n\n"
                f"Try again to construct the routine."
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
            f"Productionize routine:\n{encode(routine.model_dump())}\n\n"
            f"Output schema:\n{encode(Routine.model_json_schema())}\n\n"
            f"Output valid JSON only. {self.PLACEHOLDER_INSTRUCTIONS}"
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
            pydantic_model=Routine,
            client=self.client,
            context=encode(self.message_history[-2:]) + f"\n\n{self.PLACEHOLDER_INSTRUCTIONS}",
            llm_model=self.llm_model,
            n_tries=5
        )

        return production_routine

    def get_test_parameters(self, routine: Routine) -> TestParametersResponse:
        """
        Get the test parameters for the routine.
        """
        message = (
            f"Write a dictionary of parameters to test this routine (from previous step):\n{encode(routine.model_dump())}\n\n"
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
        self._add_to_message_history("assistant", encode(test_parameters_response.model_dump()))

        # save test parameters as a simple dict {name: value}
        test_params_dict = {param.name: param.value for param in test_parameters_response.parameters}
        self._save_to_output_dir("test_parameters.json", test_params_dict)

        # return the test parameters response
        return test_parameters_response

    def _add_to_message_history(self, role: str, content: str) -> None:
        """
        Add a message to the message history.
        """
        self.message_history.append({"role": role, "content": content})
        self._save_to_output_dir("message_history.json", self.message_history)
