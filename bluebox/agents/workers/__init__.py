"""
bluebox/agents/workers/__init__.py

Worker agents for Phase 2: Experimentation.

Workers execute experiments dispatched by an orchestrator. They have access
to both live browser tools and recorded capture data for reference.
"""

from bluebox.agents.workers.experiment_worker import ExperimentWorker

__all__ = ["ExperimentWorker"]
