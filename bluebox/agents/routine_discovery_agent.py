"""
bluebox/agents/routine_discovery_agent.py

LLM-powered agent for generating routines from CDP captures.

This agent uses an LLM-driven agentic loop with tools to:
1. Identify network transactions matching the user's task
2. Extract variables (parameters, tokens, static values)
3. Resolve dynamic token sources (storage, window properties, prior transactions)
4. Construct and validate routines

The workflow is guided by a system prompt, with the LLM deciding which
tools to call at each step.
"""

import json
import os
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from toon import encode

from bluebox.llms.infra.data_store import DiscoveryDataStore
from bluebox.llms.llm_client import LLMClient
from bluebox.llms.tools.routine_discovery_tools import TOOL_DEFINITIONS
from bluebox.llms.tools.execute_routine_tool import execute_routine
from bluebox.data_models.routine_discovery.llm_responses import (
    TransactionIdentificationResponse,
    ExtractedVariableResponse,
    Variable,
    VariableType,
    ResolvedVariableResponse,
    SessionStorageSource,
    SessionStorageType,
    TransactionSource,
    WindowPropertySource,
    TestParameter,
    TestParametersResponse,
)
from bluebox.data_models.routine_discovery.state import (
    DiscoveryPhase,
    RoutineDiscoveryState,
)
from bluebox.data_models.routine_discovery.message import (
    RoutineDiscoveryMessage,
    RoutineDiscoveryMessageType,
)
from bluebox.data_models.routine.routine import Routine
from bluebox.data_models.routine.dev_routine import DevRoutine
from bluebox.data_models.routine.endpoint import HTTPMethod
from bluebox.utils.exceptions import TransactionIdentificationFailedError
from bluebox.utils.llm_utils import manual_llm_parse_text_to_model
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


class RoutineDiscoveryAgent(BaseModel):
    """
    Agent for discovering routines from network transactions using an LLM-driven loop.

    The agent uses tools to explore CDP captures and build routines step by step,
    with the LLM deciding the workflow based on what it discovers.
    """

    llm_client: LLMClient
    data_store: DiscoveryDataStore
    task: str
    emit_message_callable: Callable[[RoutineDiscoveryMessage], None]
    message_history: list[dict] = Field(default_factory=list)
    output_dir: str | None = Field(default=None)
    last_response_id: str | None = Field(default=None)
    n_transaction_identification_attempts: int = Field(default=3)
    max_iterations: int = Field(default=50)
    timeout: int = Field(default=600)
    remote_debugging_address: str | None = Field(default=None, description="Chrome remote debugging address for routine validation (e.g., http://127.0.0.1:9222)")
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Internal state (not part of constructor interface)
    _state: RoutineDiscoveryState | None = PrivateAttr(default=None)

    # === Prompts ===

    PLACEHOLDER_INSTRUCTIONS: str = (
        "PLACEHOLDER SYNTAX:\n"
        "- PARAMS: {{param_name}} (NO prefix, name matches parameter definition)\n"
        "- SOURCES (use dot paths): {{cookie:name}}, {{sessionStorage:path.to.value}}, "
        "{{localStorage:key}}, {{windowProperty:obj.key}}\n\n"
        "JSON VALUE RULES (TWO sets of quotes needed for strings!):\n"
        '- String: "key": \\"{{x}}\\"  (OUTER quotes = JSON string, INNER \\" = escaped quotes)\n'
        '- Number/bool/null: "key": "{{x}}"  (only outer quotes, they get stripped)\n'
        '- Inside larger string: "prefix\\"{{x}}\\"suffix"  (escaped quotes wrap placeholder)\n\n'
        "EXAMPLES:\n"
        '1. String param:     "name": \\"{{username}}\\"           -> "name": "john"\n'
        '2. Number param:     "count": "{{limit}}"                -> "count": 50\n'
        '3. Bool param:       "active": "{{is_active}}"           -> "active": true\n'
        '4. Session storage:  "token": \\"{{sessionStorage:auth.access_token}}\\"\n'
        '5. Cookie:           "sid": \\"{{cookie:session_id}}\\"'
    )

    SYSTEM_PROMPT: str = """You are an expert at analyzing network traffic and building web automation routines.

## Your Task
Analyze captured browser network data to create a reusable routine that accomplishes the user's task.

## Workflow
Follow these phases in order:

### Phase 1: Identify Transaction
1. Use `list_transactions` to see available transactions
2. Use `get_transaction` to examine promising candidates
3. Use `record_identified_transaction` when you find the transaction that accomplishes the user's task

### Phase 2: Process Transactions (BFS Queue)
For each transaction in the queue:
1. Use `get_transaction` to see full details
2. Use `record_extracted_variables` to log variables found in the request:
   - PARAMETER: User input (search_query, item_id) - things the user explicitly provides
   - DYNAMIC_TOKEN: Auth/session values (CSRF, JWT, session_id) - require resolution
   - STATIC_VALUE: Constants (app version, User-Agent) - can be hardcoded
3. For each DYNAMIC_TOKEN, use `scan_for_value` to find its source
4. Use `record_resolved_variable` to record where each token comes from
   - If source is another transaction, it will be auto-added to the queue
   - IMPORTANT: If value is found in BOTH storage AND a prior transaction,
     use source_type='transaction' as the primary source. Session storage may
     be empty in a fresh session - prefer network sources for reliability.
5. Use `mark_transaction_complete` when done with current transaction
6. Continue until queue is empty

### Phase 3: Construct and Finalize Routine
1. Use `construct_routine` to build the routine from all processed data
2. If validation fails, fix the errors and try again
3. On success, the routine is automatically finalized to production format

### Phase 4: Validate Routine (Optional, if browser available)
1. Use `execute_routine` to test the routine with real parameters
2. If execution fails, check the diagnostics:
   - `failed_placeholders`: Variables that couldn't be resolved at runtime
   - `operation_errors`: Operations that threw errors
3. For failed placeholders, use `scan_for_value` to re-verify the source
4. Update the routine with `construct_routine` and test again

## Variable Classification Rules

**PARAMETER** (requires_dynamic_resolution=false):
- Values the user explicitly provides as input
- Examples: search_query, item_id, page_number, username
- Rule: If the user wouldn't directly provide this value, it's NOT a parameter

**DYNAMIC_TOKEN** (requires_dynamic_resolution=true):
- Auth/session values that change per session
- Examples: CSRF tokens, JWTs, session_id, visitorData, auth headers
- Also: trace IDs, request IDs, correlation IDs
- Rule: If it looks like a generated ID or security token, it's a DYNAMIC_TOKEN

**STATIC_VALUE** (requires_dynamic_resolution=false):
- Constants that don't change between sessions
- Examples: App version, User-Agent, clientName, timeZone, language codes
- Rule: If you can hardcode it and it will work across sessions, it's STATIC

## Important Notes
- Focus on the user's INTENT, not literal wording
- Keep parameters MINIMAL - only what the user MUST provide
- If only one value was observed and it could be hardcoded, hardcode it
- Credentials for fetch operations: same-origin > include > omit
- PREFER NETWORK SOURCES: When a value appears in both session storage AND a prior
  transaction response, use source_type='transaction' as the PRIMARY source, not storage.
  Session storage may be empty in a fresh session, so the routine must fetch via network.

{placeholder_instructions}
"""

    DATA_STORE_PROMPT: str = """
## Available Data
You have access to captured browser data including:
{data_store_prompt}
"""

    def _get_system_prompt(self) -> str:
        """Build the complete system prompt with current state."""
        prompt = self.SYSTEM_PROMPT.format(
            placeholder_instructions=self.PLACEHOLDER_INSTRUCTIONS
        )

        # Add data store context
        if self.data_store:
            data_store_prompt = self.data_store.generate_data_store_prompt()
            if data_store_prompt:
                prompt += self.DATA_STORE_PROMPT.format(data_store_prompt=data_store_prompt)

        # Add current state context
        if self._state:
            status = self._state.get_queue_status()
            prompt += f"""
## Current State
- Phase: {self._state.phase.value}
- Queue: {status['pending_count']} pending, {status['processed_count']} processed
- Current transaction: {status['current'] or 'None'}
"""

        return prompt

    def _register_tools(self) -> None:
        """Register all discovery tools with the LLM client."""
        self.llm_client.clear_tools()
        for tool_def in TOOL_DEFINITIONS:
            self.llm_client.register_tool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
            )

    def _set_vectorstores(self, uuid_filter: str | None = None) -> None:
        """Configure the LLMClient's file_search vectorstores."""
        vector_store_ids = self.data_store.get_vectorstore_ids()
        if uuid_filter and self.data_store.cdp_captures_vectorstore_id:
            filters = {"type": "eq", "key": "uuid", "value": [uuid_filter]}
            self.llm_client.set_file_search_vectorstores(vector_store_ids, filters=filters)
        else:
            self.llm_client.set_file_search_vectorstores(vector_store_ids)

    def _add_to_message_history(self, role: str, content: str, tool_calls: list | None = None) -> None:
        """Add a message to the history."""
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.tool_arguments) if isinstance(tc.tool_arguments, dict) else tc.tool_arguments,
                    },
                }
                for tc in tool_calls
            ]
        self.message_history.append(msg)
        self._save_to_output_dir("message_history.json", self.message_history)

    def _add_tool_result(self, call_id: str, result: dict) -> None:
        """Add a tool result to the message history."""
        self.message_history.append({
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": call_id,
        })
        self._save_to_output_dir("message_history.json", self.message_history)

    def _save_to_output_dir(self, relative_path: str, data: dict | list | str) -> None:
        """Save data to output_dir if specified."""
        if self.output_dir is None:
            return
        save_path = os.path.join(self.output_dir, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if isinstance(data, (dict, list)):
            with open(save_path, mode="w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif isinstance(data, str):
            with open(save_path, mode="w", encoding="utf-8") as f:
                f.write(data)

    def _emit_progress(self, message: str, msg_type: RoutineDiscoveryMessageType = RoutineDiscoveryMessageType.PROGRESS_THINKING) -> None:
        """Emit a progress message."""
        self.emit_message_callable(RoutineDiscoveryMessage(type=msg_type, content=message))

    # === Tool Implementations ===

    def _tool_list_transactions(self) -> dict:
        """List all available transaction IDs."""
        tx_ids = self.data_store.get_all_transaction_ids()
        return {
            "transaction_ids": tx_ids,
            "count": len(tx_ids),
        }

    def _tool_get_transaction(self, transaction_id: str) -> dict:
        """Get full details of a transaction."""
        all_ids = self.data_store.get_all_transaction_ids()
        if transaction_id not in all_ids:
            return {"error": f"Transaction {transaction_id} not found. Available: {all_ids[:10]}..."}

        tx = self.data_store.get_transaction_by_id(transaction_id)
        return {
            "transaction_id": transaction_id,
            "request": tx.get("request", {}),
            "response": tx.get("response", {}),
        }

    def _tool_scan_for_value(self, value: str, before_transaction_id: str | None = None) -> dict:
        """Scan storage, window properties, and transactions for a value."""
        max_timestamp = None
        if before_transaction_id:
            max_timestamp = self.data_store.get_transaction_timestamp(before_transaction_id)

        # Scan storage
        storage_sources = self.data_store.scan_storage_for_value(value)

        # Scan window properties
        window_sources = self.data_store.scan_window_properties_for_value(value)

        # Scan transaction responses
        tx_sources = self.data_store.scan_transaction_responses(value, max_timestamp=max_timestamp)

        return {
            "storage_sources": storage_sources[:5],  # Limit results
            "window_property_sources": window_sources[:5],
            "transaction_sources": tx_sources[:5],
            "found_count": len(storage_sources) + len(window_sources) + len(tx_sources),
        }

    def _tool_add_to_queue(self, transaction_id: str, reason: str) -> dict:
        """Add a transaction to the processing queue."""
        all_ids = self.data_store.get_all_transaction_ids()
        if transaction_id not in all_ids:
            return {"success": False, "error": f"Transaction {transaction_id} not found"}

        added, position = self._state.add_to_queue(transaction_id)
        return {
            "success": True,
            "added": added,
            "queue_position": position,
            "already_processed": transaction_id in self._state.processed_transactions,
            "reason": reason,
        }

    def _tool_get_queue_status(self) -> dict:
        """Get current queue status."""
        return self._state.get_queue_status()

    def _tool_mark_complete(self, transaction_id: str) -> dict:
        """Mark a transaction as complete and get the next one."""
        next_tx = self._state.mark_transaction_complete(transaction_id)

        # Check if we should advance phase
        if not next_tx and not self._state.transaction_queue:
            self._state.phase = DiscoveryPhase.CONSTRUCT_ROUTINE
            self._emit_progress("All transactions processed, ready to construct routine", RoutineDiscoveryMessageType.PROGRESS_RESULT)

        return {
            "success": True,
            "next_transaction": next_tx,
            "remaining_count": len(self._state.transaction_queue),
            "phase": self._state.phase.value,
        }

    def _tool_record_identified_transaction(self, args: dict) -> dict:
        """Record the identified root transaction."""
        tx_id = args["transaction_id"]

        # Validate transaction exists
        all_ids = self.data_store.get_all_transaction_ids()
        if tx_id not in all_ids:
            self._state.identification_attempts += 1
            if self._state.identification_attempts >= self.n_transaction_identification_attempts:
                raise TransactionIdentificationFailedError(
                    f"Failed to identify transaction after {self.n_transaction_identification_attempts} attempts"
                )
            return {
                "success": False,
                "error": f"Transaction {tx_id} not found. Choose from: {all_ids}",
                "attempts_remaining": self.n_transaction_identification_attempts - self._state.identification_attempts,
            }

        # Record the root transaction
        self._state.root_transaction = TransactionIdentificationResponse(
            transaction_id=tx_id,
            description=args["description"],
            url=args["url"],
            method=HTTPMethod(args["method"]),
            short_explanation=args.get("short_explanation", ""),
        )

        # Add to queue and set as current
        self._state.add_to_queue(tx_id)
        self._state.get_next_transaction()  # Set as current
        self._state.phase = DiscoveryPhase.PROCESS_QUEUE

        self._emit_progress(f"Identified transaction: {tx_id}", RoutineDiscoveryMessageType.PROGRESS_RESULT)
        self._save_to_output_dir("root_transaction.json", self._state.root_transaction.model_dump())

        return {
            "success": True,
            "transaction_id": tx_id,
            "added_to_queue": True,
            "message": "Transaction identified and added to processing queue",
        }

    def _tool_record_extracted_variables(self, args: dict) -> dict:
        """Record extracted variables for a transaction."""
        tx_id = args["transaction_id"]
        variables_data = args["variables"]

        # Convert to Variable objects
        variables = []
        variables_needing_resolution = []
        for v in variables_data:
            var = Variable(
                type=VariableType(v["type"]),
                requires_dynamic_resolution=v["requires_dynamic_resolution"],
                name=v["name"],
                observed_value=v["observed_value"],
                values_to_scan_for=v.get("values_to_scan_for", [v["observed_value"]]),
            )
            variables.append(var)
            if var.requires_dynamic_resolution and var.type == VariableType.DYNAMIC_TOKEN:
                variables_needing_resolution.append(var.name)

        # Store in state
        extracted = ExtractedVariableResponse(transaction_id=tx_id, variables=variables)
        self._state.store_transaction_data(tx_id, extracted_variables=extracted)

        # Also store the request
        tx = self.data_store.get_transaction_by_id(tx_id)
        self._state.store_transaction_data(tx_id, request=tx.get("request", {}))

        self._save_to_output_dir(f"transaction_{len(self._state.processed_transactions)}/extracted_variables.json", extracted.model_dump())

        return {
            "success": True,
            "transaction_id": tx_id,
            "total_variables": len(variables),
            "variables_needing_resolution": variables_needing_resolution,
        }

    def _tool_record_resolved_variable(self, args: dict) -> dict:
        """Record a resolved variable source."""
        var_name = args["variable_name"]
        tx_id = args["transaction_id"]
        source_type = args["source_type"]

        # Find the variable in extracted variables
        tx_data = self._state.transaction_data.get(tx_id, {})
        extracted = tx_data.get("extracted_variables")
        if not extracted:
            return {"success": False, "error": f"No extracted variables found for transaction {tx_id}"}

        variable = None
        for v in extracted.variables:
            if v.name == var_name:
                variable = v
                break
        if not variable:
            return {"success": False, "error": f"Variable {var_name} not found in transaction {tx_id}"}

        # Build resolved variable response
        session_storage_source = None
        transaction_source = None
        window_property_source = None
        needs_dependency = False
        dependency_tx_id = None

        if source_type == "storage" and args.get("storage_source"):
            ss = args["storage_source"]
            session_storage_source = SessionStorageSource(
                type=SessionStorageType(ss["type"]),
                dot_path=ss["dot_path"],
            )
        elif source_type == "window_property" and args.get("window_property_source"):
            wp = args["window_property_source"]
            window_property_source = WindowPropertySource(dot_path=wp["dot_path"])
        elif source_type == "transaction" and args.get("transaction_source"):
            ts = args["transaction_source"]
            transaction_source = TransactionSource(
                transaction_id=ts["transaction_id"],
                dot_path=ts["dot_path"],
            )
            # Auto-add dependency to queue
            dep_tx_id = ts["transaction_id"]
            if dep_tx_id not in self._state.processed_transactions:
                added, _ = self._state.add_to_queue(dep_tx_id)
                if added:
                    needs_dependency = True
                    dependency_tx_id = dep_tx_id

        resolved = ResolvedVariableResponse(
            variable=variable,
            session_storage_source=session_storage_source,
            transaction_source=transaction_source,
            window_property_source=window_property_source,
            short_explanation=f"Resolved from {source_type}",
        )

        # Store in state
        self._state.store_transaction_data(tx_id, resolved_variable=resolved)

        result = {
            "success": True,
            "variable_name": var_name,
            "source_type": source_type,
            "needs_dependency_processing": needs_dependency,
        }
        if dependency_tx_id:
            result["dependency_transaction_id"] = dependency_tx_id
            result["message"] = f"Added {dependency_tx_id} to queue for processing"

        return result

    def _tool_construct_routine(self, args: dict) -> dict:
        """Construct the DevRoutine from processed data and finalize it."""
        self._state.construction_attempts += 1

        try:
            dev_routine = DevRoutine(
                name=args["name"],
                description=args["description"],
                parameters=args.get("parameters", []),
                operations=args["operations"],
            )

            # Validate
            valid, errors, exc = dev_routine.validate()
            if not valid:
                return {
                    "success": False,
                    "validation_errors": errors,
                    "message": "Fix the errors and try again",
                    "attempt": self._state.construction_attempts,
                }

            # Store dev routine
            self._state.dev_routine = dev_routine
            self._emit_progress("Routine constructed successfully", RoutineDiscoveryMessageType.PROGRESS_RESULT)
            self._save_to_output_dir("dev_routine.json", dev_routine.model_dump())

            # Finalize: convert DevRoutine to production Routine
            self._emit_progress("Productionizing routine")

            productionize_message = {
                "role": "user",
                "content": (
                    f"Productionize routine:\n{encode(dev_routine.model_dump())}\n\n"
                    f"Output schema:\n{encode(Routine.model_json_schema())}\n\n"
                    f"Output valid JSON only. {self.PLACEHOLDER_INSTRUCTIONS}"
                ),
            }

            # Use a separate message list for this LLM call - don't inject into main
            # message_history during tool execution, as that would corrupt the
            # conversation sequence (must be: assistant(tool_call) -> tool_result)
            response = self.llm_client.call_sync(
                messages=[productionize_message],
            )

            response_text = response.content or ""

            # Parse the response
            production_routine = manual_llm_parse_text_to_model(
                text=response_text,
                pydantic_model=Routine,
                client=self.llm_client._client._client,
                context=encode([productionize_message, {"role": "assistant", "content": response_text}]) + f"\n\n{self.PLACEHOLDER_INSTRUCTIONS}",
                llm_model=self.llm_client.llm_model.value,
                n_tries=5,
            )

            self._state.production_routine = production_routine
            self._save_to_output_dir("routine.json", production_routine.model_dump())

            # Transition to validation phase if browser available, otherwise complete
            if self.remote_debugging_address:
                self._state.phase = DiscoveryPhase.VALIDATE_ROUTINE
                return {
                    "success": True,
                    "routine_name": production_routine.name,
                    "operations_count": len(dev_routine.operations),
                    "parameters_count": len(dev_routine.parameters),
                    "next_step": "Use execute_routine to validate the routine works correctly",
                }
            else:
                self._state.phase = DiscoveryPhase.COMPLETE
                return {
                    "success": True,
                    "routine_name": production_routine.name,
                    "operations_count": len(dev_routine.operations),
                    "parameters_count": len(dev_routine.parameters),
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempt": self._state.construction_attempts,
            }

    def _tool_execute_routine(self, args: dict) -> dict:
        """Execute the constructed routine to validate it works."""
        if not self._state.production_routine:
            return {
                "success": False,
                "error": "No routine has been constructed yet. Call construct_routine first.",
            }

        if not self.remote_debugging_address:
            return {
                "success": False,
                "error": "No remote_debugging_address configured. Cannot execute routine without a browser connection.",
            }

        self._state.validation_attempts += 1
        parameters = args.get("parameters", {})
        self._emit_progress(f"Executing routine with parameters: {list(parameters.keys())}")

        result = execute_routine(
            routine=self._state.production_routine.model_dump(),
            parameters=parameters,
            remote_debugging_address=self.remote_debugging_address,
            timeout=self.timeout,
            close_tab_when_done=True,
        )

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "Unknown execution error"),
            }

        exec_result = result.get("result")
        if exec_result is None:
            return {
                "success": False,
                "error": "Execution returned no result",
            }

        # Extract diagnostic information for the agent
        if not exec_result.ok:
            # Identify failed placeholders
            failed_placeholders = [
                key for key, value in exec_result.placeholder_resolution.items()
                if value is None
            ]

            # Identify operation errors
            operation_errors = [
                (op.type, op.error)
                for op in exec_result.operations_metadata
                if op.error is not None
            ]

            self._emit_progress(
                f"Routine execution failed: {len(failed_placeholders)} placeholder failures, {len(operation_errors)} operation errors",
                RoutineDiscoveryMessageType.PROGRESS_RESULT
            )

            # Build guidance for fixing the issues
            next_steps = []
            if failed_placeholders:
                next_steps.append(
                    f"Use scan_for_value to verify the sources for: {failed_placeholders}. "
                    "The placeholders couldn't be resolved at runtime."
                )
            if operation_errors:
                next_steps.append(
                    "Check the operation_errors for details on which operations failed and why."
                )
            next_steps.append("After fixing, call construct_routine again with the corrected values.")

            return {
                "success": False,
                "error": exec_result.error or "Routine execution failed",
                "failed_placeholders": failed_placeholders,
                "operation_errors": operation_errors,
                "warnings": exec_result.warnings,
                "placeholder_resolution": exec_result.placeholder_resolution,
                "attempt": self._state.validation_attempts,
                "next_steps": next_steps,
            }

        # Success - mark as complete
        self._state.phase = DiscoveryPhase.COMPLETE
        self._emit_progress("Routine validated successfully", RoutineDiscoveryMessageType.PROGRESS_RESULT)
        return {
            "success": True,
            "message": "Routine executed and validated successfully",
            "placeholder_resolution": exec_result.placeholder_resolution,
            "data_preview": str(exec_result.data)[:500] if exec_result.data else None,
        }

    def _execute_tool(self, tool_name: str, tool_arguments: dict) -> dict:
        """Execute a tool and return the result."""
        logger.info(f"Executing tool: {tool_name} with args: {tool_arguments}")

        try:
            if tool_name == "list_transactions":
                return self._tool_list_transactions()
            elif tool_name == "get_transaction":
                return self._tool_get_transaction(tool_arguments["transaction_id"])
            elif tool_name == "scan_for_value":
                return self._tool_scan_for_value(
                    tool_arguments["value"],
                    tool_arguments.get("before_transaction_id"),
                )
            elif tool_name == "add_transaction_to_queue":
                return self._tool_add_to_queue(
                    tool_arguments["transaction_id"],
                    tool_arguments["reason"],
                )
            elif tool_name == "get_queue_status":
                return self._tool_get_queue_status()
            elif tool_name == "mark_transaction_complete":
                return self._tool_mark_complete(tool_arguments["transaction_id"])
            elif tool_name == "record_identified_transaction":
                return self._tool_record_identified_transaction(tool_arguments)
            elif tool_name == "record_extracted_variables":
                return self._tool_record_extracted_variables(tool_arguments)
            elif tool_name == "record_resolved_variable":
                return self._tool_record_resolved_variable(tool_arguments)
            elif tool_name == "construct_routine":
                return self._tool_construct_routine(tool_arguments)
            elif tool_name == "execute_routine":
                return self._tool_execute_routine(tool_arguments)
            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool execution error for {tool_name}: {e}")
            return {"error": str(e)}

    def _run_agent_loop(self) -> Routine:
        """Run the agentic loop until routine is complete or max iterations reached."""
        for iteration in range(self.max_iterations):
            logger.debug(f"Agent loop iteration {iteration + 1}, phase: {self._state.phase.value}")

            # Check if complete
            if self._state.phase == DiscoveryPhase.COMPLETE:
                if self._state.production_routine:
                    return self._state.production_routine
                raise RuntimeError("Discovery marked complete but no routine produced")

            # Build messages for LLM (only send recent messages with previous_response_id)
            if self.last_response_id:
                # Find last assistant message index and send only messages after it
                last_assistant_idx = -1
                for i, msg in enumerate(self.message_history):
                    if msg["role"] == "assistant":
                        last_assistant_idx = i
                messages = self.message_history[last_assistant_idx + 1:] if last_assistant_idx >= 0 else self.message_history
            else:
                messages = self.message_history

            # Call LLM
            response = self.llm_client.call_sync(
                messages=messages,
                system_prompt=self._get_system_prompt(),
                previous_response_id=self.last_response_id,
                tool_choice="auto",
            )
            self.last_response_id = response.response_id

            # Process response content
            if response.content:
                self._add_to_message_history("assistant", response.content, tool_calls=response.tool_calls)
                logger.debug(f"Assistant response: {response.content[:200]}...")

            # Execute tool calls
            if response.tool_calls:
                if not response.content:
                    self._add_to_message_history("assistant", "", tool_calls=response.tool_calls)

                for tool_call in response.tool_calls:
                    result = self._execute_tool(tool_call.tool_name, tool_call.tool_arguments)
                    self._add_tool_result(tool_call.call_id, result)

                    # Check for errors that should stop the loop
                    if "error" in result and "not found" not in result.get("error", "").lower():
                        logger.warning(f"Tool error: {result['error']}")

            # If no tool calls and not complete, prompt the agent to continue
            elif self._state.phase != DiscoveryPhase.COMPLETE:
                status = self._state.get_queue_status()
                prompt = (
                    f"[ACTION REQUIRED] Current phase: {self._state.phase.value}. "
                    f"Queue: {status['pending_count']} pending, {status['processed_count']} processed. "
                )
                if self._state.phase == DiscoveryPhase.IDENTIFY_TRANSACTION:
                    prompt += "Use list_transactions and get_transaction to find the relevant transaction, then record_identified_transaction."
                elif self._state.phase == DiscoveryPhase.PROCESS_QUEUE:
                    if status['current']:
                        prompt += f"Currently processing: {status['current']}. Extract and resolve variables, then mark_transaction_complete."
                    elif status['pending_count'] > 0:
                        prompt += "Get the next transaction from the queue."
                    else:
                        prompt += "Queue is empty. Call construct_routine to build the routine."
                elif self._state.phase == DiscoveryPhase.CONSTRUCT_ROUTINE:
                    prompt += "Build the routine using construct_routine."
                elif self._state.phase == DiscoveryPhase.VALIDATE_ROUTINE:
                    if self._state.validation_attempts > 0:
                        prompt += (
                            f"Previous validation attempt {self._state.validation_attempts} failed. "
                            "Review the execution errors above. If there are failed placeholders or operation errors, "
                            "use scan_for_value to re-verify sources, then call construct_routine with fixes. "
                            "If the issue is with test parameters, try execute_routine again with corrected values."
                        )
                    else:
                        prompt += "Validate the routine using execute_routine with test parameters."

                self._add_to_message_history("system", prompt)

        raise TimeoutError(f"Discovery did not complete in {self.max_iterations} iterations")

    def run(self) -> Routine:
        """
        Run the routine discovery agent.

        Returns:
            Routine: The discovered and productionized routine.
        """
        # Validate data store
        assert self.data_store.cdp_captures_vectorstore_id is not None, "Vectorstore ID is not set"

        # Initialize state
        self._state = RoutineDiscoveryState()

        # Emit start message
        self._emit_progress("Discovery initiated", RoutineDiscoveryMessageType.INITIATED)

        # Register tools and configure vectorstores
        self._register_tools()
        self._set_vectorstores()

        # Initialize message history
        self._add_to_message_history("system", self._get_system_prompt())
        self._add_to_message_history("user", f"Task: {self.task}")

        all_tx_ids = self.data_store.get_all_transaction_ids()
        self._add_to_message_history(
            "user",
            f"Available transaction IDs ({len(all_tx_ids)} total):\n{encode(all_tx_ids)}"
        )

        # Run the agentic loop
        try:
            routine = self._run_agent_loop()

            self._emit_progress("Routine generated successfully", RoutineDiscoveryMessageType.FINISHED)
            return routine

        except TransactionIdentificationFailedError as e:
            self._emit_progress(str(e), RoutineDiscoveryMessageType.ERROR)
            raise
        except Exception as e:
            logger.exception(f"Discovery failed: {e}")
            self._emit_progress(f"Discovery failed: {e}", RoutineDiscoveryMessageType.ERROR)
            raise

    def get_test_parameters(self, routine: Routine) -> TestParametersResponse:
        """
        Generate test parameters for the routine based on observed values from discovery.

        Args:
            routine: The production routine to generate test parameters for.

        Returns:
            TestParametersResponse with parameter name/value pairs.
        """
        test_params = []

        # Get observed values from extracted variables in state
        observed_values = {}
        if self._state:
            for tx_data in self._state.transaction_data.values():
                extracted = tx_data.get("extracted_variables")
                if extracted:
                    for var in extracted.variables:
                        observed_values[var.name] = var.observed_value

        # Match routine parameters with observed values (all values as strings)
        for param in routine.parameters:
            value = observed_values.get(param.name, "")
            # Use observed value or provide sensible defaults as strings
            if not value:
                if param.type == "integer":
                    value = "1"
                elif param.type == "number":
                    value = "1.0"
                elif param.type == "boolean":
                    value = "false"
            test_params.append(TestParameter(name=param.name, value=str(value)))

        return TestParametersResponse(parameters=test_params)
