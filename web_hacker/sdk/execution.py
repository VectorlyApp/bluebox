"""
Routine execution SDK wrapper.
"""

from typing import Any

from ..data_models.routine.execution import RoutineExecutionResult
from ..data_models.routine.routine import Routine


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
        close_tab_when_done: bool = True,
    ) -> RoutineExecutionResult:
        """
        Execute a routine.

        Args:
            routine: The routine to execute.
            parameters: Parameters for URL/header/body interpolation.
            timeout: Operation timeout in seconds.
            close_tab_when_done: Whether to close the tab when finished.

        Returns:
            RoutineExecutionResult with execution status and data.
        """
        return routine.execute(
            parameters_dict=parameters,
            remote_debugging_address=self.remote_debugging_address,
            timeout=timeout,
            close_tab_when_done=close_tab_when_done,
        )
