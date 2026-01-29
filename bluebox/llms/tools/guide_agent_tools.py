"""
bluebox/llms/tools/guide_agent_tools.py

Tool functions for the guide agent.
"""

from typing import Any

from pydantic import ValidationError

from bluebox.data_models.routine.routine import Routine


def validate_routine(routine_dict: dict) -> dict:
    """
    Validates a routine dictionary against the Routine schema.

    IMPORTANT: You MUST pass the COMPLETE routine JSON object as routine_dict.
    If you have a routine from get_current_routine, pass that exact JSON here.

    The routine_dict must be a JSON object containing:
    - "name" (string): The name of the routine
    - "description" (string): Description of what the routine does
    - "parameters" (array): Parameter definitions with name, description, type, required fields
    - "operations" (array): Operation definitions

    Args:
        routine_dict: The complete routine JSON object with name, description, parameters, and operations

    Returns:
        Dict with 'valid' bool and either 'message' (success) or 'error' (failure)
    """
    if not routine_dict:
        return {
            "valid": False,
            "error": "routine_dict is empty. You must pass the complete routine JSON object. If you have a routine from get_current_routine, pass that exact routine_json here.",
        }

    try:
        routine = Routine(**routine_dict)
        return {
            "valid": True,
            "message": f"Routine '{routine.name}' is valid with {len(routine.operations)} operations and {len(routine.parameters)} parameters.",
        }
    except ValidationError as e:
        return {
            "valid": False,
            "error": str(e),
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Unexpected error: {str(e)}",
        }
