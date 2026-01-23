"""
web_hacker/routine_discovery/agent.py

LLM-powered agent for generating routines from CDP captures.

Contains:
- RoutineDiscoveryAgent: Uses LLMClient with file_search tools to analyze captures and build routines
- Multi-step pipeline: identify → extract → resolve → construct → productionize
- Outputs: validated Routine JSON with parameters and operations
"""

import json
import os
from typing import Any, Callable, TypeVar
from uuid import uuid4

from pydantic import BaseModel
from toon import encode

from web_hacker.data_models.llms.vendors import OpenAIModel
from web_hacker.data_models.routine.dev_routine import DevRoutine
from web_hacker.data_models.routine.routine import Routine
from web_hacker.data_models.routine_discovery.llm_responses import (
    ExtractedVariableResponse,
    ResolvedVariableResponse,
    TestParametersResponse,
    TransactionConfirmationResponse,
    TransactionIdentificationResponse,
    VariableType,
)
from web_hacker.data_models.routine_discovery.message import (
    RoutineDiscoveryMessage,
    RoutineDiscoveryMessageType,
)
from web_hacker.llms.llm_client import LLMClient
from web_hacker.routine_discovery.data_store import DiscoveryDataStore
from web_hacker.utils.exceptions import TransactionIdentificationFailedError
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)


T = TypeVar("T", bound=BaseModel)


class RoutineDiscoveryAgent:
    """
    Agent for discovering routines from network transactions.

    Uses LLMClient with file_search tools to analyze CDP captures
    and construct parameterized routines via a multi-step pipeline:
    1. Identify the target transaction
    2. Extract variables from each transaction
    3. Resolve dynamic tokens to their sources
    4. Construct a DevRoutine from resolved transactions
    5. Productionize into a final Routine
    """

    # Class constants ______________________________________________________________________________________________________

    DATA_STORE_PROMPT: str = """
You have access to the following data and you must refer to it when searching for transactions and resolving variables!
It is essential that you use this data, documentation, and code:
{data_store_prompt}
"""

    SYSTEM_PROMPT: str = """You are a helpful assistant that is an expert in parsing network traffic.
You need to identify one or more network transactions that directly correspond to the user's requested task.
You have access to vectorstore that contains network transactions and storage data
(cookies, localStorage, sessionStorage, etc.)."""

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

    N_TRANSACTION_IDENTIFICATION_ATTEMPTS: int = 3
    DEFAULT_TIMEOUT: int = 600

    # Magic methods ________________________________________________________________________________________________________

    def __init__(
        self,
        data_store: DiscoveryDataStore,
        task: str,
        emit_message_callable: Callable[[RoutineDiscoveryMessage], None],
        llm_model: OpenAIModel = OpenAIModel.GPT_5_1,
        output_dir: str | None = None,
        n_transaction_identification_attempts: int = N_TRANSACTION_IDENTIFICATION_ATTEMPTS,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initialize the RoutineDiscoveryAgent.

        Args:
            data_store: Data store providing access to transactions and vectorstores.
            task: Description of the task to discover routines for.
            emit_message_callable: Callback for emitting progress messages.
            llm_model: The OpenAI model to use for LLM calls.
            output_dir: Optional directory for saving intermediate outputs.
            n_transaction_identification_attempts: Max attempts for transaction identification.
            timeout: Timeout in seconds for the discovery process.
        """
        self._data_store = data_store
        self._task = task
        self._emit_message_callable = emit_message_callable
        self._output_dir = output_dir
        self._n_transaction_identification_attempts = n_transaction_identification_attempts
        self._timeout = timeout

        # LLM client setup
        self._llm_client = LLMClient(llm_model)
        self._previous_response_id: str | None = None
        self._message_history: list[dict[str, str]] = []

        # State
        self._current_transaction_identification_attempt: int = 1

        # Configure base vectorstores for file_search
        vector_store_ids = self._data_store.get_vectorstore_ids()
        self._llm_client.set_file_search_vectorstores(vector_store_ids)

        logger.info(
            "Initialized RoutineDiscoveryAgent with model: %s, task: %s",
            llm_model,
            task[:100],
        )

    # Properties ___________________________________________________________________________________________________________

    @property
    def data_store(self) -> DiscoveryDataStore:
        """Return the data store."""
        return self._data_store

    # Private methods ______________________________________________________________________________________________________

    def _get_system_prompt(self) -> str:
        """Get system prompt with data store context if available."""
        system_prompt = self.SYSTEM_PROMPT
        if self._data_store:
            data_store_prompt = self.DATA_STORE_PROMPT.format(
                data_store_prompt=self._data_store.generate_data_store_prompt()
            )
            if data_store_prompt:
                system_prompt = f"{system_prompt}\n\n{data_store_prompt}"
        return system_prompt

    def _build_filtered_tools(self, uuid_filter: str) -> list[dict[str, Any]]:
        """
        Build file_search tools with UUID filter for CDP captures
        and unfiltered for other vectorstores.

        Args:
            uuid_filter: The UUID to filter CDP captures by.

        Returns:
            List of file_search tool definitions using all available vectorstores.
        """
        tools: list[dict[str, Any]] = []

        if self._data_store.cdp_captures_vectorstore_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [self._data_store.cdp_captures_vectorstore_id],
                "filters": {
                    "type": "eq",
                    "key": "uuid",
                    "value": [uuid_filter],
                },
            })

        if self._data_store.documentation_vectorstore_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [self._data_store.documentation_vectorstore_id],
            })

        return tools

    def _save_to_output_dir(self, relative_path: str, data: dict | list | str) -> None:
        """Save data to output_dir if it is specified."""
        if self._output_dir is None:
            return
        save_path = os.path.join(self._output_dir, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(data, (dict, list)):
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif isinstance(data, str):
            with open(save_path, mode="w", encoding="utf-8") as f:
                f.write(data)

    def _add_to_message_history(self, role: str, content: str) -> None:
        """Add a message to the message history."""
        self._message_history.append({"role": role, "content": content})
        self._save_to_output_dir("message_history.json", self._message_history)

    def _call_llm(
        self,
        response_model: type[T],
        tool_choice: str = "auto",
        tools_override: list[dict[str, Any]] | None = None,
        send_full_history: bool = False,
    ) -> T:
        """
        Make an LLM call with structured output.

        Args:
            response_model: Pydantic model for structured response.
            tool_choice: Tool choice mode ("required", "auto", "none").
            tools_override: Optional per-call tool override (e.g., filtered file_search).
            send_full_history: If True, send full history. If False, send only last message
                             and rely on previous_response_id for context.

        Returns:
            The parsed Pydantic model instance.
        """
        messages = self._message_history if send_full_history else [self._message_history[-1]]

        result = self._llm_client.call_sync(
            messages=messages,
            system_prompt=self._get_system_prompt(),
            previous_response_id=self._previous_response_id,
            response_model=response_model,
            tool_choice=tool_choice,
            tools_override=tools_override,
        )

        # Update previous_response_id for conversation chaining
        self._previous_response_id = self._llm_client.last_response_id

        return result

    def _handle_transaction_identification_failure(self) -> None:
        """Handle failure to identify a valid transaction."""
        logger.error(
            "Transaction identification failed: attempt %d of %d",
            self._current_transaction_identification_attempt,
            self._n_transaction_identification_attempts,
        )
        error_message = (
            "Failed to identify the network transactions that directly correspond to your task.\n\n"
            "Possible fixes:\n"
            "1. Make the task description more specific and detailed.\n"
            "2. Streamline the browser session to reduce noise (close unrelated tabs, avoid extraneous clicks).\n"
        )

        if self._current_transaction_identification_attempt >= self._n_transaction_identification_attempts:
            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.ERROR,
                content=error_message,
            ))
            raise TransactionIdentificationFailedError(error_message)

        self._current_transaction_identification_attempt += 1
        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content=f"Retrying transaction identification (attempt {self._current_transaction_identification_attempt})...",
        ))

    # Public methods _______________________________________________________________________________________________________

    def run(self) -> Routine:
        """
        Run the routine discovery agent.

        Returns:
            Routine: The discovered and productionized routine.
        """
        assert self._data_store.cdp_captures_vectorstore_id is not None, "Vectorstore ID is not set"

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.INITIATED,
            content="Discovery initiated",
        ))

        # Add the system prompt and initial user messages to the message history
        self._add_to_message_history("system", self._get_system_prompt())
        self._add_to_message_history("user", f"Task description: {self._task}")
        self._add_to_message_history(
            "user",
            f"These are the possible network transaction ids you can choose from:\n"
            f"{encode(self._data_store.get_all_transaction_ids())}",
        )

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Identifying relevant network transactions",
        ))

        # Step 1: Identify the target transaction
        identified_transaction = None
        while identified_transaction is None:
            identified_transaction = self.identify_transaction()

            if identified_transaction.transaction_id in [None, "None", ""]:
                self._handle_transaction_identification_failure()
                identified_transaction = None
                continue

            if identified_transaction.transaction_id not in self._data_store.get_all_transaction_ids():
                logger.error("Identified transaction %s is not in the data store", identified_transaction.transaction_id)
                self._handle_transaction_identification_failure()
                identified_transaction = None
                continue

            # Confirm the identified transaction
            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content="Confirming identified network transaction",
            ))
            confirmation = self.confirm_identified_transaction(identified_transaction)

            if not confirmation.is_correct:
                identified_transaction = None
                self._handle_transaction_identification_failure()

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content="Successfully identified network transaction relevant to the task",
        ))
        logger.info("Identified transaction: %s", identified_transaction.transaction_id)
        self._save_to_output_dir("root_transaction.json", identified_transaction.model_dump())

        # Step 2: Process transaction queue (BFS for dependencies)
        transaction_queue = [identified_transaction.transaction_id]
        routine_transactions: dict[str, dict[str, Any]] = {}
        all_resolved_variables: list[ResolvedVariableResponse] = []

        while len(transaction_queue) > 0:
            transaction_id = transaction_queue.pop(0)
            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content=f"Processing network transaction {len(routine_transactions) + 1}",
            ))
            logger.info("Processing transaction: %s", transaction_id)

            # Extract variables
            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content="Extracting variables (args, cookies, tokens, browser variables)",
            ))
            extracted_variables = self.extract_variables(transaction_id)

            # Resolve dynamic tokens
            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
                content="Resolving cookies, tokens, and api keys",
            ))
            resolved_variables = self.resolve_variables(extracted_variables)
            all_resolved_variables.extend(resolved_variables)

            self._emit_message_callable(RoutineDiscoveryMessage(
                type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
                content=f"Successfully processed network transaction {len(routine_transactions) + 1}",
            ))

            # Save intermediate results
            tx_index = len(routine_transactions)
            self._save_to_output_dir(
                f"transaction_{tx_index}/extracted_variables.json",
                extracted_variables.model_dump(),
            )
            self._save_to_output_dir(
                f"transaction_{tx_index}/resolved_variables.json",
                [rv.model_dump() for rv in resolved_variables],
            )

            # Store transaction data
            routine_transactions[transaction_id] = {
                "request": self._data_store.get_transaction_by_id(transaction_id)["request"],
                "extracted_variables": extracted_variables.model_dump(),
                "resolved_variables": [rv.model_dump() for rv in resolved_variables],
            }

            # Enqueue dependency transactions
            for resolved_variable in resolved_variables:
                if resolved_variable.transaction_source is not None:
                    new_id = resolved_variable.transaction_source.transaction_id
                    if new_id not in routine_transactions:
                        transaction_queue.append(new_id)

        # Step 3: Construct the routine (reverse order: dependencies first)
        ordered_transactions = {k: v for k, v in reversed(list(routine_transactions.items()))}

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Constructing the template of the routine",
        ))
        dev_routine = self.construct_routine(ordered_transactions, all_resolved_variables)

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content="Successfully constructed the template of the routine",
        ))

        # Step 4: Productionize
        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_THINKING,
            content="Productionizing the routine",
        ))
        production_routine = self.productionize_routine(dev_routine)

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.PROGRESS_RESULT,
            content="Productionized the routine",
        ))
        logger.info("Productionized the routine")

        self._save_to_output_dir("routine.json", production_routine.model_dump())

        self._emit_message_callable(RoutineDiscoveryMessage(
            type=RoutineDiscoveryMessageType.FINISHED,
            content="Routine generated successfully",
        ))

        return production_routine

    def identify_transaction(self) -> TransactionIdentificationResponse:
        """
        Identify the network transaction corresponding to the user's task.

        Returns:
            TransactionIdentificationResponse with the identified transaction.
        """
        is_first_attempt = self._current_transaction_identification_attempt == 1

        if not is_first_attempt:
            message = (
                "Try again. The transaction id you provided does not exist or was not relevant. "
                f"Choose from: {encode(self._data_store.get_all_transaction_ids())}"
            )
            self._add_to_message_history("user", message)

        response = self._call_llm(
            response_model=TransactionIdentificationResponse,
            tool_choice="required",
            send_full_history=is_first_attempt,
        )

        self._add_to_message_history("assistant", encode(response.model_dump()))
        logger.info("Transaction identification response: %s", response.model_dump())

        return response

    def confirm_identified_transaction(
        self,
        identified_transaction: TransactionIdentificationResponse,
    ) -> TransactionConfirmationResponse:
        """
        Confirm the identified transaction is correct and relevant to the task.

        Args:
            identified_transaction: The transaction to confirm.

        Returns:
            TransactionConfirmationResponse with confirmation result.
        """
        # Add the transaction to the vectorstore for detailed inspection
        metadata = {"uuid": str(uuid4())}
        self._data_store.add_transaction_to_vectorstore(
            transaction_id=identified_transaction.transaction_id,
            metadata=metadata,
        )

        # Build filtered tools to search through the specific transaction
        tools = self._build_filtered_tools(uuid_filter=metadata["uuid"])

        message = (
            f"{identified_transaction.transaction_id} have been added to the vectorstore in full (including response bodies).\n"
            "Confirm the identified transaction is correct and directly corresponds to the user's requested task:\n"
            f"{self._task}\n\n"
            "IMPORTANT: Focus on whether this transaction accomplishes the user's INTENT, not the literal wording. "
        )
        self._add_to_message_history("user", message)

        response = self._call_llm(
            response_model=TransactionConfirmationResponse,
            tool_choice="required",
            tools_override=tools,
        )

        self._add_to_message_history("assistant", encode(response.model_dump()))
        return response

    def extract_variables(self, transaction_id: str) -> ExtractedVariableResponse:
        """
        Extract variables from a transaction's request.

        Args:
            transaction_id: The transaction to extract variables from.

        Returns:
            ExtractedVariableResponse with extracted variables.
        """
        original_transaction_id = transaction_id
        transaction = self._data_store.get_transaction_by_id(transaction_id)

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

        response = self._call_llm(
            response_model=ExtractedVariableResponse,
            tool_choice="auto",
        )

        self._add_to_message_history("assistant", encode(response.model_dump()))

        # Override the transaction_id since the LLM may return an incorrect format
        response.transaction_id = original_transaction_id

        return response

    def resolve_variables(self, extracted_variables: ExtractedVariableResponse) -> list[ResolvedVariableResponse]:
        """
        Resolve dynamic tokens to their sources (storage, window properties, or prior transactions).

        Args:
            extracted_variables: The extracted variables to resolve.

        Returns:
            List of resolved variable responses.
        """
        max_timestamp = self._data_store.get_transaction_timestamp(extracted_variables.transaction_id)

        variables_to_resolve = [
            var for var in extracted_variables.variables
            if var.requires_dynamic_resolution and var.type == VariableType.DYNAMIC_TOKEN
        ]

        resolved_variable_responses: list[ResolvedVariableResponse] = []

        for variable in variables_to_resolve:
            logger.info("Resolving variable: %s with values to scan for: %s", variable.name, variable.values_to_scan_for)

            # Scan storage for the variable's value
            storage_objects: list[dict] = []
            for value in variable.values_to_scan_for:
                storage_objects.extend(self._data_store.scan_storage_for_value(value=value))

            if storage_objects:
                logger.info("Found %d storage sources that contain the value", len(storage_objects))

            # Scan window properties
            window_properties: list[dict] = []
            for value in variable.values_to_scan_for:
                window_properties.extend(self._data_store.scan_window_properties_for_value(value))

            if window_properties:
                logger.info("Found %d window properties that contain the value", len(window_properties))

            # Scan transaction responses
            transaction_ids: list[str] = []
            for value in variable.values_to_scan_for:
                transaction_ids.extend(
                    self._data_store.scan_transaction_responses(value=value, max_timestamp=max_timestamp)
                )

            # Deduplicate and limit
            transaction_ids = list(set(transaction_ids))[:2]

            if transaction_ids:
                logger.info("Found %d transaction ids that contain the value: %s", len(transaction_ids), transaction_ids)

            # Add transactions to vectorstore for LLM inspection
            uuid = str(uuid4())
            for tx_id in transaction_ids:
                self._data_store.add_transaction_to_vectorstore(
                    transaction_id=tx_id,
                    metadata={"uuid": uuid},
                )

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

            # Use filtered tools to force the LLM to look at the specific transactions
            tools = self._build_filtered_tools(uuid_filter=uuid)

            response = self._call_llm(
                response_model=ResolvedVariableResponse,
                tool_choice="required",
                tools_override=tools,
            )

            self._add_to_message_history("assistant", encode(response.model_dump()))
            resolved_variable_responses.append(response)

            # Log resolution result
            resolved_sources = [
                s for s in [
                    response.transaction_source,
                    response.session_storage_source,
                    response.window_property_source,
                ] if s is not None
            ]

            if len(resolved_sources) == 0:
                logger.warning("Unable to resolve variable '%s'. Hardcoding to: %s", variable.name, variable.observed_value)
            elif len(resolved_sources) == 1:
                logger.info("Variable '%s' resolved from: %s", variable.name, type(resolved_sources[0]).__name__)
            else:
                logger.info("Variable '%s' resolved from %d sources", variable.name, len(resolved_sources))

        return resolved_variable_responses

    def construct_routine(
        self,
        routine_transactions: dict[str, Any],
        resolved_variables: list[ResolvedVariableResponse],
        max_attempts: int = 3,
    ) -> DevRoutine:
        """
        Construct a DevRoutine from the resolved transactions.

        Args:
            routine_transactions: Ordered dict of transaction data (dependencies first).
            resolved_variables: All resolved variables across transactions.
            max_attempts: Maximum retry attempts for validation failures.

        Returns:
            A validated DevRoutine.

        Raises:
            Exception: If construction fails after max_attempts.
        """
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

        for attempt in range(1, max_attempts + 1):
            routine = self._call_llm(
                response_model=DevRoutine,
                tool_choice="required",
            )

            self._add_to_message_history("assistant", encode(routine.model_dump()))
            logger.info("Constructed routine (attempt %d): %s", attempt, routine.model_dump())

            # Validate the routine
            successful, errors, exception = routine.validate()
            if successful:
                return routine

            # Add validation error and retry
            message = (
                f"Execution failed with error: {exception}\n\n"
                f"Routine validation failed:\n{encode(errors)}\n\n"
                f"Try again to construct the routine."
            )
            self._add_to_message_history("user", message)

        raise Exception(f"Failed to construct the routine after {max_attempts} attempts")

    def productionize_routine(self, routine: DevRoutine) -> Routine:
        """
        Convert a DevRoutine into a production Routine.

        Args:
            routine: The DevRoutine to productionize.

        Returns:
            A production-ready Routine.
        """
        message = (
            f"Productionize routine:\n{encode(routine.model_dump())}\n\n"
            f"Output schema:\n{encode(Routine.model_json_schema())}\n\n"
            f"Output valid JSON only. {self.PLACEHOLDER_INSTRUCTIONS}"
        )
        self._add_to_message_history("user", message)

        response = self._call_llm(
            response_model=Routine,
            tool_choice="auto",
        )

        self._add_to_message_history("assistant", encode(response.model_dump()))
        return response

    def get_test_parameters(self, routine: Routine) -> TestParametersResponse:
        """
        Generate test parameters for the routine.

        Args:
            routine: The routine to generate test parameters for.

        Returns:
            TestParametersResponse with parameter values for testing.
        """
        message = (
            f"Write a dictionary of parameters to test this routine (from previous step):\n{encode(routine.model_dump())}\n\n"
            f"Ensure all parameters are present and have valid values."
        )
        self._add_to_message_history("user", message)

        response = self._call_llm(
            response_model=TestParametersResponse,
            tool_choice="auto",
        )

        self._add_to_message_history("assistant", encode(response.model_dump()))

        # Save test parameters
        test_params_dict = {param.name: param.value for param in response.parameters}
        self._save_to_output_dir("test_parameters.json", test_params_dict)

        return response
