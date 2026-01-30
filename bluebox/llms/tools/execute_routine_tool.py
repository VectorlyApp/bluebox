"""
bluebox/llms/tools/execute_routine_tool.py

Shared routine execution utility for agents.
"""

import json
from typing import Any

from bluebox.data_models.routine.routine import Routine
from bluebox.sdk.execution import RoutineExecutor
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def execute_routine(
    routine: str | dict[str, Any],
    parameters: dict[str, Any],
    remote_debugging_address: str = "http://127.0.0.1:9222",
    timeout: float = 180.0,
    close_tab_when_done: bool = True,
    tab_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute a routine from JSON string or dictionary.

    Args:
        routine: JSON string or dictionary representation of the routine
        parameters: Parameters for the routine
        remote_debugging_address: Chrome debugging address
        timeout: Execution timeout in seconds
        close_tab_when_done: Whether to close tab after execution
        tab_id: Optional existing tab ID to use

    Returns:
        dict with 'success', 'result' or 'error'
    """
    try:
        if isinstance(routine, str):
            routine_dict = json.loads(routine)
        else:
            routine_dict = routine
        routine_obj = Routine(**routine_dict)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid routine JSON: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to parse routine: {e}"}

    try:
        executor = RoutineExecutor(remote_debugging_address=remote_debugging_address)
        result = executor.execute(
            routine=routine_obj,
            parameters=parameters,
            timeout=timeout,
            close_tab_when_done=close_tab_when_done,
            tab_id=tab_id,
        )

        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        logger.exception("Routine execution failed")
        return {
            "success": False,
            "error": str(e),
        }
