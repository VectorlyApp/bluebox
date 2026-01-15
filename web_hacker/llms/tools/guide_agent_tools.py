"""
web_hacker/llms/tools/guide_agent_tools.py

Tool functions for the guide agent.
"""

from typing import Any


def start_routine_discovery_job_creation(
    task_description: str,
    expected_output_description: str,
    input_parameters: list[dict[str, str]] | None = None,
    filters_or_constraints: list[str] | None = None,
    target_website: str | None = None,
) -> dict[str, Any]:
    """
    Initiates the routine discovery process.

    Call this when you have gathered enough information about:
    1) What task the user wants to automate
    2) What data/output they expect
    3) What input parameters the routine should accept
    4) Any filters or constraints

    This tool requests user confirmation before executing.

    Args:
        task_description: Description of the task/routine the user wants to create
        expected_output_description: Description of what data the routine should return
        input_parameters: List of input parameters with 'name' and 'description' keys
        filters_or_constraints: Any filters or constraints the user mentioned
        target_website: Target website/URL if mentioned by user

    Returns:
        Result dict to be passed to routine discovery agent
    """
    # TODO: implement the actual handoff logic
    raise NotImplementedError("start_routine_discovery_job_creation not yet implemented")
