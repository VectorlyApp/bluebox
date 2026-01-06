"""
web_hacker/data_models/routine/dev_routine.py

This primarily serves as an intermediate data model for routine discovery.
This data model is much simpler than the production routine data model,
which makes it easier for the LLM agent to generate.

Key differences from production Routine:
- DevEndpoint has headers/body as strings (not dicts) - simpler for LLM
- Only 4 operation types: navigate, sleep, fetch, return
- No execution logic - just data validation
"""

import re
from typing import Union, Literal

from pydantic import BaseModel

from web_hacker.data_models.routine.endpoint import HTTPMethod, CREDENTIALS
from web_hacker.data_models.routine.operation import RoutineOperationTypes
from web_hacker.data_models.routine.parameter import Parameter


class DevEndpoint(BaseModel):
    """
    Simplified endpoint model for LLM generation.
    Headers and body are strings (not dicts) for easier generation.
    """
    url: str
    description: str | None = None
    method: HTTPMethod
    headers: str  # JSON string, not dict
    body: str     # JSON string, not dict
    credentials: CREDENTIALS = CREDENTIALS.SAME_ORIGIN


# Dev operation classes - simple data models without execute() methods
# Uses production RoutineOperationTypes enum for consistency

class DevNavigateOperation(BaseModel):
    """Navigate operation for dev routine (data-only, no execute)."""
    type: Literal[RoutineOperationTypes.NAVIGATE] = RoutineOperationTypes.NAVIGATE
    url: str


class DevSleepOperation(BaseModel):
    """Sleep operation for dev routine (data-only, no execute)."""
    type: Literal[RoutineOperationTypes.SLEEP] = RoutineOperationTypes.SLEEP
    timeout_seconds: float


class DevFetchOperation(BaseModel):
    """
    Fetch operation for dev routine (data-only, no execute).
    Key difference: uses DevEndpoint (string headers/body) instead of Endpoint (dict).
    """
    type: Literal[RoutineOperationTypes.FETCH] = RoutineOperationTypes.FETCH
    endpoint: DevEndpoint
    session_storage_key: str


class DevReturnOperation(BaseModel):
    """Return operation for dev routine (data-only, no execute)."""
    type: Literal[RoutineOperationTypes.RETURN] = RoutineOperationTypes.RETURN
    session_storage_key: str


# Dev routine operation union
DevOperationUnion = Union[
    DevNavigateOperation,
    DevSleepOperation,
    DevFetchOperation,
    DevReturnOperation,
]


class DevRoutine(BaseModel):
    """
    Simplified routine model for LLM generation during discovery.
    """
    name: str
    description: str
    operations: list[DevOperationUnion]
    parameters: list[Parameter]
    
    def validate(self) -> tuple[bool, list[str], Exception | None]:
        """
        Validate the dev routine structure.
        
        Returns:
            tuple containing:
                - result: True if valid, False otherwise
                - errors: List of error messages
                - exception: Exception if one occurred
        """
        result = True
        errors = []
        exception = None
        
        try: 
            # Must have at least 3 operations (navigate, fetch, return)
            if len(self.operations) < 3:
                result = False
                errors.append("Must have at least 3 operations (navigate, fetch, return)")
            
            # First operation should be navigate
            if not isinstance(self.operations[0], DevNavigateOperation):
                result = False
                errors.append("First operation should be a navigate operation")
                
            # Last operation should be return
            if not isinstance(self.operations[-1], DevReturnOperation):
                result = False
                errors.append("Last operation should be a return operation")
                
            # Second to last should be fetch
            if not isinstance(self.operations[-2], DevFetchOperation):
                result = False
                errors.append("Second to last operation should be a fetch operation")
                
            # Get all placeholders
            all_placeholders = set(self._get_all_placeholders(self.model_dump_json()))
                
            # Check that every parameter is used
            defined_parameters = {p.name for p in self.parameters}
            unused_parameters = defined_parameters - all_placeholders
            if unused_parameters:
                result = False
                for param_name in unused_parameters:
                    errors.append(f"Parameter '{param_name}' is not used in the routine operations")
            
            # Remaining placeholders (not parameters)
            remaining_placeholders = all_placeholders - defined_parameters
                    
            # All remaining placeholders should be valid prefixes
            valid_prefixes = ["sessionStorage", "cookie", "localStorage", "uuid", "epoch_milliseconds", "meta", "windowProperty"]
            for placeholder in remaining_placeholders:
                prefix = placeholder.split(":")[0]
                if prefix not in valid_prefixes:
                    result = False
                    errors.append(f"Placeholder '{placeholder}' has invalid prefix. Valid: {valid_prefixes}")
                        
            # Get used session storage keys from placeholders
            used_session_storage_keys = set()
            for placeholder in all_placeholders:
                if placeholder.split(":")[0] == "sessionStorage":
                    used_session_storage_keys.add(placeholder.split(":")[1].split(".")[0])
            
            # Include return operation's key
            if isinstance(self.operations[-1], DevReturnOperation):
                used_session_storage_keys.add(self.operations[-1].session_storage_key)
                    
            # Get all fetch session storage keys
            all_fetch_keys = set()
            for operation in self.operations:
                if isinstance(operation, DevFetchOperation):
                    all_fetch_keys.add(operation.session_storage_key)
            
            # Check unused fetch keys
            unused_keys = all_fetch_keys - used_session_storage_keys
            if unused_keys:
                result = False
                for key in unused_keys:
                    errors.append(f"Fetch session storage key '{key}' is not used")
                    
            # Last fetch key should match return key
            if len(self.operations) >= 3:
                if isinstance(self.operations[-1], DevReturnOperation) and isinstance(self.operations[-2], DevFetchOperation):
                    if self.operations[-1].session_storage_key != self.operations[-2].session_storage_key:
                        result = False
                        errors.append("Last fetch session_storage_key should match return session_storage_key")
            
        except Exception as e:
            result = False
            errors.append(f"Exception: {e}")
            exception = e
            
        return result, errors, exception
    
    def _get_all_placeholders(self, routine_string: str) -> list[str]:
        """
        Extract all placeholders {{...}} from routine string.
        """
        placeholders = re.findall(r'{{.*?}}', routine_string)
        return [placeholder[2:-2] for placeholder in set(placeholders)]


# Backward-compatible aliases for existing imports 
# (SPECIFICALLY FOR production DISCOVERY AGENT)
#TODO: Remove this once the production discovery agent is updated
Routine = DevRoutine
