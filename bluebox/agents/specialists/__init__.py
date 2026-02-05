"""
Bluebox specialist agents module.

Specialists are domain-expert agents that an orchestrator deploys for specific tasks.
"""

from bluebox.agents.specialists.abstract_specialist import (
    AbstractSpecialist,
    specialist_tool,
    RunMode,
    AutonomousConfig,
)

__all__ = [
    "AbstractSpecialist",
    "specialist_tool",
    "RunMode",
    "AutonomousConfig",
]
