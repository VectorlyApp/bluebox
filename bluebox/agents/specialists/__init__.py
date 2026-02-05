"""
bluebox/agents/specialists/__init__.py

NOTE: This file is necessary because it triggers AbstractSpecialist.__init_subclass__
for all specialist classes, populating AbstractSpecialist._subclasses list.
This enables using AbstractSpecialist.get_all_subclasses() to discover specialists.
"""

from bluebox.agents.specialists.abstract_specialist import (
    AbstractSpecialist,
    AutonomousConfig,
    RunMode,
    specialist_tool,
)

# Import all specialist classes to trigger AbstractSpecialist.__init_subclass__
from bluebox.agents.specialists.docs_digger_agent import DocsDiggerAgent
from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist
from bluebox.agents.specialists.js_specialist import JSSpecialist
from bluebox.agents.specialists.network_spy_agent import NetworkSpyAgent
from bluebox.agents.specialists.trace_hound_agent import TraceHoundAgent

__all__ = [
    # Base class and utilities
    "AbstractSpecialist",
    "AutonomousConfig",
    "RunMode",
    "specialist_tool",
    # Concrete specialists
    "DocsDiggerAgent",
    "InteractionSpecialist",
    "JSSpecialist",
    "NetworkSpyAgent",
    "TraceHoundAgent",
]
