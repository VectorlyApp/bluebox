"""
web_hacker/data_models/routine/routine.py

Routine data model.
"""

import ast
import json
import time

from pydantic import BaseModel, Field, model_validator

from web_hacker.data_models.routine.execution import RoutineExecutionContext, RoutineExecutionResult
from web_hacker.data_models.routine.operation import (
    RoutineDownloadOperation,
    RoutineFetchOperation,
    RoutineNavigateOperation,
    RoutineOperationUnion,
)
from web_hacker.data_models.routine.parameter import (
    Parameter,
    ParameterType,
    BUILTIN_PARAMETERS,
    VALID_PLACEHOLDER_PREFIXES,
)
from web_hacker.cdp.connection import cdp_new_tab, cdp_attach_to_existing_tab, dispose_context
from web_hacker.data_models.routine.placeholder import (
    PlaceholderQuoteType,
    extract_placeholders_from_json_str,
)
from web_hacker.utils.data_utils import extract_base_url_from_url
from web_hacker.utils.logger import get_logger
from web_hacker.utils.web_socket_utils import send_cmd, recv_until

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
    incognito: bool = Field(
        default=True,
        description="Whether to use incognito mode when executing the routine"
    )
    parameters: list[Parameter] = Field(
        default_factory=list,
        description="List of parameters"
    )

    @model_validator(mode='after')
    def validate_parameter_usage(self) -> 'Routine':
        """
        Pydantic model validator to ensure all defined parameters are used in the routine
        and no undefined parameters are used.
        Raises ValueError if unused parameters are found or undefined parameters are used.
        Also automatically computes base_urls from operations.
        """
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
                    raise ValueError(f"Invalid prefix in placeholder: {prefix}")
                if not path:
                    raise ValueError(f"Path is required for {prefix}: placeholder")
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
                        raise ValueError(
                            f"String parameter '{{{{{content}}}}}' in routine '{self.name}' must use escape-quoted format. "
                            f"Use '\\\"{{{{content}}}}\\\"' instead of '\"{{{{content}}}}\"'."
                        )

        # Check 1: All defined parameters must be used
        unused_parameters = defined_parameters - used_parameters
        if unused_parameters:
            raise ValueError(
                f"Unused parameters found in routine '{self.name}': {list(unused_parameters)}. "
                f"All defined parameters must be used somewhere in the routine operations."
            )

        # Check 2: No undefined parameters should be used
        undefined_parameters = used_parameters - defined_parameters
        if undefined_parameters:
            raise ValueError(
                f"Undefined parameters found in routine '{self.name}': {list(undefined_parameters)}. "
                f"All parameters used in the routine must be defined in parameters."
            )


        return self

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

    def execute(
        self,
        parameters_dict: dict | None = None,
        remote_debugging_address: str = "http://127.0.0.1:9222",
        timeout: float = 180.0,
        close_tab_when_done: bool = True,
        tab_id: str | None = None,
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
                    incognito=self.incognito,
                    url="about:blank",
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
                    if browser_context_id and self.incognito:
                        dispose_context(remote_debugging_address, browser_context_id)
            except Exception:
                pass
            try:
                browser_ws.close()
            except Exception:
                pass


