"""
bluebox/data_models/routine/routine.py

Main Routine model for browser automation workflows.

Contains:
- Routine: JSON-serializable workflow with operations, parameters, validation
- execute(): Run routine with CDP connection
- execute_with_session(): Run with existing CDP session
- Validation: parameter usage, placeholder resolution, builtin handling
"""

import ast
import json
import time
from collections import defaultdict
from typing import Any, get_args

from pydantic import BaseModel, Field, model_validator

from bluebox.data_models.routine.execution import RoutineExecutionContext, RoutineExecutionResult
from bluebox.data_models.routine.operation import (
    RoutineDownloadOperation,
    RoutineFetchOperation,
    RoutineJsEvaluateOperation,
    RoutineNavigateOperation,
    RoutineOperationUnion,
    RoutineReturnHTMLOperation,
    RoutineReturnOperation,
    RoutineSleepOperation,
)
from bluebox.data_models.routine.parameter import (
    Parameter,
    ParameterType,
    BUILTIN_PARAMETERS,
    VALID_PLACEHOLDER_PREFIXES,
)
from bluebox.cdp.connection import cdp_new_tab, cdp_attach_to_existing_tab, dispose_context
from bluebox.data_models.routine.endpoint import Endpoint
from bluebox.utils.pydantic_utils import format_model_fields
from bluebox.data_models.routine.placeholder import (
    PlaceholderQuoteType,
    extract_placeholders_from_json_str,
)
from bluebox.utils.data_utils import extract_base_url_from_url
from bluebox.utils.logger import get_logger

from bluebox.utils.web_socket_utils import send_cmd, recv_until

logger = get_logger(name=__name__)


# Routine model ___________________________________________________________________________________

class Routine(BaseModel):
    """
    Routine model with comprehensive parameter validation.
    """
    # routine details
    name: str
    description: str
    operations: list[RoutineOperationUnion]
    parameters: list[Parameter] = Field(
        default_factory=list,
        description="List of parameters"
    )

    @model_validator(mode='after')
    def validate_routine_structure(self) -> 'Routine':
        """
        Pydantic model validator for routine structure and parameter usage.

        Validates:
        - Routine must have at least 2 operations
        - Last operation must be 'return' or 'return_html'
        - Return operation's session_storage_key must be set by a prior fetch or js_evaluate
        - All defined parameters must be used
        - No undefined parameters should be used

        Raises ValueError with all errors collected together.
        """
        errors: list[str] = []

        # === STRUCTURAL VALIDATION ===

        # Check 1: Must have at least 2 operations
        if len(self.operations) < 2:
            errors.append(
                f"Routine must have at least 2 operations, found {len(self.operations)}"
            )

        # Check 2: Last operation must be return or return_html
        if self.operations:
            last_op = self.operations[-1]
            if not isinstance(last_op, (RoutineReturnOperation, RoutineReturnHTMLOperation)):
                last_op_type = last_op.type if hasattr(last_op, 'type') else type(last_op).__name__
                errors.append(
                    f"Last operation must be 'return' or 'return_html', found '{last_op_type}'"
                )
            else:
                # Check 3: Return's session_storage_key must be set by prior fetch or js_evaluate
                return_key = last_op.session_storage_key
                if return_key:
                    # Collect all session_storage_keys set by fetch or js_evaluate operations
                    available_keys: set[str] = set()
                    for op in self.operations[:-1]:  # Exclude the return operation
                        if isinstance(op, RoutineFetchOperation) and op.session_storage_key:
                            available_keys.add(op.session_storage_key)
                        elif isinstance(op, RoutineJsEvaluateOperation) and op.session_storage_key:
                            available_keys.add(op.session_storage_key)

                    if return_key not in available_keys:
                        errors.append(
                            f"Return operation uses session_storage_key '{return_key}' but it was not set by any prior fetch or js_evaluate operation. "
                            f"Available keys: {sorted(available_keys) if available_keys else 'none'}"
                        )

        # === PARAMETER VALIDATION ===

        # Convert the entire routine to JSON string for searching
        routine_json = self.model_dump_json()

        # Build lookup maps for parameters
        defined_parameters = {param.name for param in self.parameters}
        param_type_map = {param.name: param.type for param in self.parameters}
        builtin_parameter_names = {bp.name for bp in BUILTIN_PARAMETERS}

        # Types that allow both quoted "{{...}}" and escape-quoted \"{{...}}\"
        non_string_types = {
            ParameterType.INTEGER,
            ParameterType.NUMBER, 
            ParameterType.BOOLEAN
        }

        # Extract all placeholders with their quote types
        placeholders = extract_placeholders_from_json_str(routine_json)

        # Track used parameters
        used_parameters: set[str] = set()

        # Validate each placeholder
        for placeholder in placeholders:
            content = placeholder.content
            quote_type = placeholder.quote_type

            # Check if it's a storage/meta/window placeholder (has colon prefix)
            if ":" in content:
                prefix, path = [p.strip() for p in content.split(":", 1)]
                if prefix not in VALID_PLACEHOLDER_PREFIXES:
                    errors.append(f"Invalid prefix in placeholder: {prefix}")
                if not path:
                    errors.append(f"Path is required for {prefix}: placeholder")
                # Storage/meta/window placeholders can use either QUOTED or ESCAPE_QUOTED - valid
                continue

            # Check if it's a builtin parameter
            if content in builtin_parameter_names:
                # Builtins can use either QUOTED or ESCAPE_QUOTED - valid
                continue

            # It's a regular user-defined parameter
            used_parameters.add(content)

            # Get the parameter type (if defined)
            param_type = param_type_map.get(content)

            if param_type is not None:
                # Validate quote type based on parameter type
                if param_type in non_string_types:
                    # int, number, bool: can use either "{{...}}" or \"{{...}}\"
                    pass  # Both QUOTED and ESCAPE_QUOTED are valid
                else:
                    # string types: MUST use escape-quoted \"{{...}}\"
                    if quote_type != PlaceholderQuoteType.ESCAPE_QUOTED:
                        errors.append(
                            f"String parameter '{{{{{content}}}}}' must use escape-quoted format. "
                            f"Use '\\\"{{{{content}}}}\\\"' instead of '\"{{{{content}}}}\"'."
                        )

        # Check: All defined parameters must be used
        unused_parameters = defined_parameters - used_parameters
        if unused_parameters:
            error_message = f"Unused parameters: {list(unused_parameters)}. "
            for unused_parameter in unused_parameters:
                # scan for unquoted placeholders in the routine...
                if f"{{{{{unused_parameter}}}}}" in routine_json:
                    error_message += f"Unquoted placeholder '{{{{{unused_parameter}}}}}' found. "
                    error_message += "Ensure all placeholders are surrounded by quotes or escaped quotes."
            errors.append(error_message)

        # Check: No undefined parameters should be used
        undefined_parameters = used_parameters - defined_parameters
        if undefined_parameters:
            errors.append(
                f"Undefined parameters: {list(undefined_parameters)}. "
                f"All parameters used must be defined in parameters list."
            )

        # Raise all errors together
        if errors:
            raise ValueError(f"Routine '{self.name}' validation failed:\n- " + "\n- ".join(errors))

        return self

    @staticmethod
    def model_schema_markdown() -> str:
        """
        Generate a compact markdown schema reference from the Pydantic models.

        Like model_json_schema() but formatted for LLM system prompts — compact,
        readable, no $ref indirection. Auto-derived from Routine, Parameter,
        Endpoint, and all operation models to stay in sync with the code.
        """
        lines: list[str] = ["## Routine Schema Reference", ""]

        # Routine (top level)
        lines.append("### Routine (top level)")
        lines.extend(format_model_fields(Routine, skip_fields={"operations", "parameters"}))
        lines.append("- operations: list[operation] (required) — ≥2, last must be return or return_html")
        lines.append("- parameters: list[parameter] = []")
        lines.append("")

        # Parameter
        lines.append("### Parameter")
        lines.extend(format_model_fields(Parameter, skip_fields={"observed_value"}))
        lines.append("")

        # Endpoint (referenced by fetch and download)
        lines.append("### Endpoint (used by fetch and download)")
        lines.extend(format_model_fields(Endpoint, skip_fields={"description"}))
        lines.append("")

        # All operation types (auto-derived from RoutineOperationUnion)
        union_inner = get_args(RoutineOperationUnion)[0]  # Unwrap Annotated
        op_models: tuple[type, ...] = get_args(union_inner)  # Unwrap Union

        for model_cls in op_models:
            type_default = model_cls.model_fields["type"].default
            op_name = type_default.value if hasattr(type_default, "value") else str(type_default)
            lines.append(f"### Operation: {op_name}")
            lines.extend(format_model_fields(model_cls, skip_fields={"type"}))
            lines.append("")

        return "\n".join(lines)

    def compute_base_urls_from_operations(self) -> str | None:
        """
        Computes comma-separated base URLs from routine operations.
        Extracts unique base URLs from navigate, fetch, and download operations.

        Returns:
            Comma-separated string of unique base URLs (sorted), or None if none found.
        """
        urls: list[str] = []

        # Collect all URLs from operations
        for operation in self.operations:
            if isinstance(operation, RoutineNavigateOperation):
                if operation.url:
                    urls.append(operation.url)
            elif isinstance(operation, RoutineFetchOperation):
                if operation.endpoint and operation.endpoint.url:
                    urls.append(operation.endpoint.url)
            elif isinstance(operation, RoutineDownloadOperation):
                if operation.endpoint and operation.endpoint.url:
                    urls.append(operation.endpoint.url)

        # Extract base URLs from collected URLs
        base_urls: set[str] = set()
        for url in urls:
            base_url = extract_base_url_from_url(url)
            if base_url:
                base_urls.add(base_url)

        if len(base_urls) == 0:
            return None

        # Return comma-separated unique base URLs (sorted for consistency)
        return ','.join(sorted(base_urls))

    def get_structure_warnings(self) -> list[str]:
        """
        Check routine structure and return warnings for potential issues.

        Note: Critical errors (missing return, invalid structure, placeholder issues)
        are caught by the model validator. This method returns non-blocking warnings only.

        Checks:
        - First operation should be navigate
        - Second-to-last should be fetch or js_evaluate
        - Routine should have at least one navigate operation
        - Fetch operations should have session_storage_key set
        - Multiple fetches should not overwrite the same session_storage_key
        - Sleep operations should not be excessively long (> 30s)
        - POST/PUT/PATCH requests should have a body

        Returns:
            List of warning messages for potential issues.
        """
        warnings: list[str] = []

        if not self.operations:
            return warnings  # Can't check further without operations

        # 1. First operation should be navigate
        first_op = self.operations[0]
        if not isinstance(first_op, RoutineNavigateOperation):
            first_op_type = first_op.type if hasattr(first_op, 'type') else type(first_op).__name__
            warnings.append(f"First operation is '{first_op_type}', expected 'navigate'")

        # 2. Second-to-last should be fetch or js_evaluate
        if len(self.operations) >= 2:
            second_last_op = self.operations[-2]
            if not isinstance(second_last_op, (RoutineFetchOperation, RoutineJsEvaluateOperation)):
                second_last_type = second_last_op.type if hasattr(second_last_op, 'type') else type(second_last_op).__name__
                warnings.append(f"Second-to-last operation is '{second_last_type}', expected 'fetch' or 'js_evaluate'")

        # 3. Check if routine has any navigate operation
        has_navigate = any(isinstance(op, RoutineNavigateOperation) for op in self.operations)
        if not has_navigate:
            warnings.append("Routine has no 'navigate' operation - browser may not load the target page")

        # 4. Check fetch operations for missing session_storage_key
        session_storage_keys_seen: defaultdict[str, int] = defaultdict(int)
        for i, op in enumerate(self.operations):
            if isinstance(op, RoutineFetchOperation):
                if not op.session_storage_key:
                    warnings.append(f"Fetch operation at index {i} has no session_storage_key - response data will be lost")
                else:
                    session_storage_keys_seen[op.session_storage_key] += 1

        # 5. Check for duplicate session_storage_keys (overwrites)
        for key, count in session_storage_keys_seen.items():
            if count > 1:
                warnings.append(f"session_storage_key '{key}' is set by {count} fetch operations - later fetches will overwrite earlier data")

        # 6. Check for excessively long sleep durations
        for i, op in enumerate(self.operations):
            if isinstance(op, RoutineSleepOperation):
                if op.timeout_seconds > 30:
                    warnings.append(f"Sleep operation at index {i} has long duration ({op.timeout_seconds}s) - consider if this is intentional")

        # 7. Check POST/PUT/PATCH requests have body
        for i, op in enumerate(self.operations):
            if isinstance(op, (RoutineFetchOperation, RoutineDownloadOperation)):
                if op.endpoint and op.endpoint.method in ("POST", "PUT", "PATCH"):
                    if not op.endpoint.body:
                        warnings.append(f"{op.endpoint.method} request at index {i} has empty body - this may be unintentional")

        return warnings

    def execute(
        self,
        parameters_dict: dict | None = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        timeout: float = 180.0,
        close_tab_when_done: bool = True,
        tab_id: str | None = None,
        proxy_address: str | None = None,
        incognito: bool = True,
    ) -> RoutineExecutionResult:
        """
        Execute this routine using Chrome DevTools Protocol.

        Executes a sequence of operations (navigate, sleep, fetch, return) in a browser
        session, maintaining state between operations.

        Args:
            parameters_dict: Parameters for URL/header/body interpolation.
            remote_debugging_address: Chrome debugging server address.
            timeout: Operation timeout in seconds.
            close_tab_when_done: Whether to close the tab when finished.
            tab_id: If provided, attach to this existing tab. If None, create a new tab.
            proxy_address: If provided, use this proxy address.
            incognito: Whether to create an incognito browser context.
        Returns:
            RoutineExecutionResult: Result of the routine execution.
        """
        if parameters_dict is None:
            parameters_dict = {}

        # Get a tab for the routine (returns browser-level WebSocket)
        try:
            if tab_id is not None:
                target_id, browser_context_id, browser_ws = cdp_attach_to_existing_tab(
                    remote_debugging_address=remote_debugging_address,
                    target_id=tab_id,
                )
            else:
                target_id, browser_context_id, browser_ws = cdp_new_tab(
                    remote_debugging_address=remote_debugging_address,
                    incognito=incognito,
                    url="about:blank",
                    proxy_address=proxy_address,
                )
        except Exception as e:
            return RoutineExecutionResult(
                ok=False,
                error=f"Failed to {'attach to' if tab_id else 'create'} tab: {e}"
            )

        try:
            # Attach to target using flattened session (allows multiplexing via session_id)
            attach_id = send_cmd(browser_ws, "Target.attachToTarget", {"targetId": target_id, "flatten": True})
            reply = recv_until(browser_ws, lambda m: m.get("id") == attach_id, time.time() + timeout)
            session_id = reply["result"]["sessionId"]

            # Enable domains
            send_cmd(browser_ws, "Page.enable", session_id=session_id)
            send_cmd(browser_ws, "Runtime.enable", session_id=session_id)
            send_cmd(browser_ws, "Network.enable", session_id=session_id)
            send_cmd(browser_ws, "DOM.enable", session_id=session_id)

            # Create execution context
            routine_execution_context = RoutineExecutionContext(
                session_id=session_id,
                ws=browser_ws,
                send_cmd=lambda method, params=None, **kwargs: send_cmd(browser_ws, method, params, **kwargs),
                recv_until=lambda predicate, deadline: recv_until(browser_ws, predicate, deadline),
                parameters_dict=parameters_dict,
                timeout=timeout,
            )

            # Execute operations
            logger.info(f"Executing routine '{self.name}' with {len(self.operations)} operations")
            for i, operation in enumerate(self.operations):
                logger.info(
                    f"Executing operation {i+1}/{len(self.operations)}: {type(operation).__name__}"
                )
                operation.execute(routine_execution_context)

            # Try to parse string results as JSON or Python literals (skip for base64)
            result = routine_execution_context.result
            if isinstance(result.data, str) and not result.is_base64:
                try:
                    result.data = json.loads(result.data)
                except Exception:
                    try:
                        result_literal = ast.literal_eval(result.data)
                        if isinstance(result_literal, (dict, list)):
                            result.data = result_literal
                    except Exception:
                        pass  # Keep as string if both fail

            return routine_execution_context.result

        except Exception as e:
            return RoutineExecutionResult(
                ok=False,
                error=f"Routine execution failed: {e}",
            )

        finally:
            try:
                if close_tab_when_done:
                    send_cmd(browser_ws, "Target.closeTarget", {"targetId": target_id})
                    if browser_context_id:
                        dispose_context(browser_context_id, ws=browser_ws)
            except Exception:
                pass
            try:
                browser_ws.close()
            except Exception:
                pass

# NOTE: Update server to use this model.
class RoutineInfo(BaseModel):
    """Information about an available routine."""
    routine_id: str = Field(..., description="Unique identifier of the routine")
    name: str = Field(..., description="Routine name")
    description: str = Field(..., description="Routine description")
    parameters: list[Parameter] = Field(..., description="List of parameters for the routine")
    
class RoutineExecutionRequest(BaseModel):
    """A single routine to execute."""
    routine_id: str = Field(..., description="The routine_id returned by search_routines")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the routine, keyed by parameter name. Example: {\"origin \": \"California\"}",
    )
