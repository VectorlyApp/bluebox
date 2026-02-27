"""
tests/unit/data_models/orchestration/test_task.py

Tests for Task, SubAgent, TaskStatus, SpecialistAgentType, generate_short_id.
"""

import pytest

from bluebox.data_models.orchestration.task import (
    SubAgent,
    SpecialistAgentType,
    Task,
    TaskStatus,
    generate_short_id,
)


class TestGenerateShortId:
    """Tests for generate_short_id."""

    def test_default_length(self) -> None:
        assert len(generate_short_id()) == 6

    def test_custom_length(self) -> None:
        assert len(generate_short_id(length=10)) == 10

    def test_no_ambiguous_chars(self) -> None:
        for _ in range(100):
            id_ = generate_short_id()
            for ch in "0o1l":
                assert ch not in id_

    def test_uniqueness(self) -> None:
        ids = {generate_short_id() for _ in range(100)}
        assert len(ids) == 100

    def test_lowercase_alphanumeric(self) -> None:
        for _ in range(50):
            id_ = generate_short_id()
            assert id_.isalnum()


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_values(self) -> None:
        expected = {"pending", "in_progress", "paused", "completed", "failed"}
        assert {s.value for s in TaskStatus} == expected


class TestSpecialistAgentType:
    """Tests for SpecialistAgentType enum."""

    def test_pipeline_types_exist(self) -> None:
        assert SpecialistAgentType.NETWORK_SPECIALIST == "network_specialist"
        assert SpecialistAgentType.DOM_SPECIALIST == "dom_specialist"
        assert SpecialistAgentType.EXPERIMENT_WORKER == "experiment_worker"
        assert SpecialistAgentType.ROUTINE_INSPECTOR == "routine_inspector"
        assert SpecialistAgentType.VALUE_TRACE_RESOLVER == "value_trace_resolver"


class TestTask:
    """Tests for Task model."""

    def test_defaults(self) -> None:
        task = Task(
            agent_type=SpecialistAgentType.NETWORK_SPECIALIST,
            prompt="Analyze the /api/search endpoint",
        )
        assert len(task.id) == 6
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.error is None
        assert task.max_loops == 5
        assert task.loops_used == 0
        assert task.context == {}
        assert task.output_schema is None

    def test_unique_ids(self) -> None:
        tasks = [
            Task(agent_type=SpecialistAgentType.DOM_SPECIALIST, prompt="t")
            for _ in range(20)
        ]
        assert len({t.id for t in tasks}) == 20

    def test_with_context_and_schema(self) -> None:
        task = Task(
            agent_type=SpecialistAgentType.EXPERIMENT_WORKER,
            prompt="Run experiment",
            context={"endpoint": "/api/search", "method": "POST"},
            output_schema={"type": "object", "properties": {"success": {"type": "boolean"}}},
            output_description="Whether the experiment succeeded",
        )
        assert task.context["endpoint"] == "/api/search"
        assert task.output_schema is not None

    def test_model_dump_roundtrip(self) -> None:
        task = Task(
            agent_type=SpecialistAgentType.ROUTINE_INSPECTOR,
            prompt="Inspect routine",
            max_loops=10,
        )
        data = task.model_dump()
        task2 = Task.model_validate(data)
        assert task2.agent_type == task.agent_type
        assert task2.max_loops == 10


class TestSubAgent:
    """Tests for SubAgent model."""

    def test_defaults(self) -> None:
        agent = SubAgent(
            type=SpecialistAgentType.NETWORK_SPECIALIST,
            llm_model="gpt-5.2",
        )
        assert len(agent.id) == 6
        assert agent.task_ids == []

    def test_task_tracking(self) -> None:
        agent = SubAgent(
            type=SpecialistAgentType.DOM_SPECIALIST,
            llm_model="gpt-5.2",
            task_ids=["abc", "def"],
        )
        assert len(agent.task_ids) == 2

    def test_unique_ids(self) -> None:
        agents = [
            SubAgent(type=SpecialistAgentType.JS_SPECIALIST, llm_model="gpt-5.2")
            for _ in range(20)
        ]
        assert len({a.id for a in agents}) == 20
