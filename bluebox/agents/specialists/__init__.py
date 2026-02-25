"""
bluebox/agents/specialists/__init__.py

NOTE: This file imports specialist classes so AbstractAgent subclass registration
runs for each concrete specialist at import time.
"""

from bluebox.agents.abstract_agent import AgentExecutionMode, AutonomousRunConfig, agent_tool

# Import all specialist classes so AbstractAgent.__init_subclass__ registers them
from bluebox.agents.specialists.dom_specialist import DOMSpecialist
from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist
from bluebox.agents.specialists.js_specialist import JSSpecialist
from bluebox.agents.specialists.network_specialist import NetworkSpecialist
from bluebox.agents.specialists.value_trace_resolver_specialist import ValueTraceResolverSpecialist

__all__ = [
    # Utilities
    "AutonomousRunConfig",
    "AgentExecutionMode",
    "agent_tool",
    # Concrete specialists
    "DOMSpecialist",
    "InteractionSpecialist",
    "JSSpecialist",
    "NetworkSpecialist",
    "ValueTraceResolverSpecialist",
]
