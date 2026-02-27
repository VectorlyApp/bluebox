"""
tests/unit/data_models/orchestration/test_state.py

Tests for AgentOrchestrationState.
"""

from bluebox.data_models.orchestration.state import AgentOrchestrationState
from bluebox.data_models.orchestration.task import (
    SpecialistAgentType,
    SubAgent,
    Task,
    TaskStatus,
)


class TestAgentOrchestrationState:
    """Tests for AgentOrchestrationState."""

    def test_defaults(self) -> None:
        state = AgentOrchestrationState()
        assert state.tasks == {}
        assert state.subagents == {}

    def test_add_task(self) -> None:
        state = AgentOrchestrationState()
        task = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="test")
        result = state.add_task(task)
        assert result is task
        assert task.id in state.tasks

    def test_get_pending_tasks(self) -> None:
        state = AgentOrchestrationState()
        t1 = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t1")
        t2 = Task(agent_type=SpecialistAgentType.DOM_SPECIALIST, prompt="t2", status=TaskStatus.IN_PROGRESS)
        t3 = Task(agent_type=SpecialistAgentType.JS_SPECIALIST, prompt="t3")
        state.add_task(t1)
        state.add_task(t2)
        state.add_task(t3)
        pending = state.get_pending_tasks()
        assert len(pending) == 2

    def test_get_in_progress_tasks(self) -> None:
        state = AgentOrchestrationState()
        t1 = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t", status=TaskStatus.IN_PROGRESS)
        state.add_task(t1)
        assert len(state.get_in_progress_tasks()) == 1

    def test_get_paused_tasks(self) -> None:
        state = AgentOrchestrationState()
        t1 = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t", status=TaskStatus.PAUSED)
        state.add_task(t1)
        assert len(state.get_paused_tasks()) == 1

    def test_get_completed_tasks(self) -> None:
        state = AgentOrchestrationState()
        t1 = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t", status=TaskStatus.COMPLETED)
        state.add_task(t1)
        assert len(state.get_completed_tasks()) == 1

    def test_get_failed_tasks(self) -> None:
        state = AgentOrchestrationState()
        t1 = Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t", status=TaskStatus.FAILED)
        state.add_task(t1)
        assert len(state.get_failed_tasks()) == 1

    def test_get_queue_status(self) -> None:
        state = AgentOrchestrationState()
        state.add_task(Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="p"))
        state.add_task(Task(agent_type=SpecialistAgentType.DOM_SPECIALIST, prompt="p", status=TaskStatus.IN_PROGRESS))
        state.add_task(Task(agent_type=SpecialistAgentType.JS_SPECIALIST, prompt="p", status=TaskStatus.COMPLETED))
        state.add_task(Task(agent_type=SpecialistAgentType.EXPERIMENT_WORKER, prompt="p", status=TaskStatus.FAILED))

        status = state.get_queue_status()
        assert status["pending_tasks"] == 1
        assert status["in_progress_tasks"] == 1
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1
        assert status["paused_tasks"] == 0

    def test_reset(self) -> None:
        state = AgentOrchestrationState()
        state.add_task(Task(agent_type=SpecialistAgentType.NETWORK_SPECIALIST, prompt="t"))
        state.subagents["agent1"] = SubAgent(
            type=SpecialistAgentType.NETWORK_SPECIALIST,
            llm_model="gpt-5.2",
        )
        state.reset()
        assert state.tasks == {}
        assert state.subagents == {}
