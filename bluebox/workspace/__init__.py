"""
Workspace interfaces and implementations.
"""

from bluebox.workspace.abstract_workspace import AgentWorkspace
from bluebox.workspace.local_workspace import LocalAgentWorkspace

__all__ = [
    "AgentWorkspace",
    "LocalAgentWorkspace",
]
