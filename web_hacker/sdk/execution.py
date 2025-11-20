"""
Routine execution SDK wrapper.
"""

from typing import Any
from ..cdp.routine_execution import execute_routine
from ..data_models.production_routine import Routine


class RoutineExecutor:
    """
    High-level interface for executing routines.
    
    Example:
        >>> executor = RoutineExecutor()
        >>> result = executor.execute(
        ...     routine=routine,
        ...     parameters={"origin": "NYC", "destination": "LAX"}
        ... )
    """
    
    def __init__(
        self,
        remote_debugging_address: str = "http://127.0.0.1:9222",
    ):
        self.remote_debugging_address = remote_debugging_address
    
    def execute(
        self,
        routine: Routine,
        parameters: dict[str, Any],
        timeout: float = 180.0,
        wait_after_navigate_sec: float = 3.0,
        close_tab_when_done: bool = True,
        incognito: bool = False,
    ) -> dict[str, Any]:
        """
        Execute a routine.
        
        Returns:
            Result dictionary with "ok" status and "result" data.
        """
        return execute_routine(
            routine=routine,
            parameters_dict=parameters,
            remote_debugging_address=self.remote_debugging_address,
            timeout=timeout,
            wait_after_navigate_sec=wait_after_navigate_sec,
            close_tab_when_done=close_tab_when_done,
            incognito=incognito,
        )

