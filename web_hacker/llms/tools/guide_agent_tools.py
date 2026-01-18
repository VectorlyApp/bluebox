"""
web_hacker/llms/tools/guide_agent_tools.py

Tool functions for the guide agent.
"""

from typing import Any

from pydantic import ValidationError

from web_hacker.data_models.routine.routine import Routine


def validate_routine(routine_dict: dict) -> dict:
    """
    Validates a routine dictionary against the Routine schema.

    IMPORTANT: You MUST construct and pass the COMPLETE routine JSON object as the
    routine_dict argument. Do NOT call this with empty arguments {}.

    The routine_dict must be a JSON object containing:
    - "name" (string): The name of the routine
    - "description" (string): Description of what the routine does
    - "parameters" (array): Parameter definitions with name, description, type, required fields
    - "operations" (array): Operation definitions

    WORKFLOW:
    1. First construct the complete routine JSON in your response
    2. Then call this tool with that object
    3. If validation fails, read the error, fix the issues, and retry up to 3 times

    Args:
        routine_dict: The complete routine JSON object with name, description, parameters, and operations

    Returns:
        Dict with 'valid' bool and either 'message' (success) or 'error' (failure)
    """
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
