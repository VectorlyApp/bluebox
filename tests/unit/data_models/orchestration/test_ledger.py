"""
tests/unit/data_models/orchestration/test_ledger.py

Tests for DiscoveryLedger, RoutineSpec, RoutineAttempt, RoutineCatalog, ShippedRoutine.
"""

import pytest

from bluebox.data_models.orchestration.experiment import (
    ExperimentEntry,
    ExperimentStatus,
    ExperimentVerdict,
)
from bluebox.data_models.orchestration.ledger import (
    DiscoveryLedger,
    RoutineAttempt,
    RoutineAttemptStatus,
    RoutineCatalog,
    RoutineSpec,
    RoutineSpecStatus,
    ShippedRoutine,
)


class TestRoutineSpecStatus:
    """Tests for RoutineSpecStatus enum."""

    def test_all_statuses(self) -> None:
        expected = {"planned", "experimenting", "assembling", "validating", "shipped", "failed"}
        assert {s.value for s in RoutineSpecStatus} == expected


class TestRoutineAttemptStatus:
    """Tests for RoutineAttemptStatus enum."""

    def test_all_statuses(self) -> None:
        expected = {"draft", "validating", "executing", "inspecting", "passed", "failed"}
        assert {s.value for s in RoutineAttemptStatus} == expected


class TestRoutineSpec:
    """Tests for RoutineSpec."""

    def test_defaults(self) -> None:
        spec = RoutineSpec(name="get_prices", description="Get train prices")
        assert len(spec.id) == 6
        assert spec.status == RoutineSpecStatus.PLANNED
        assert spec.priority == 1
        assert spec.experiment_ids == []
        assert spec.attempt_ids == []
        assert spec.shipped_attempt_id is None

    def test_unique_ids(self) -> None:
        specs = [RoutineSpec(name="s", description="d") for _ in range(20)]
        assert len({s.id for s in specs}) == 20


class TestRoutineAttempt:
    """Tests for RoutineAttempt."""

    def test_defaults(self) -> None:
        attempt = RoutineAttempt(
            routine_spec_id="abc",
            routine_json={"name": "test", "operations": []},
        )
        assert len(attempt.id) == 6
        assert attempt.status == RoutineAttemptStatus.DRAFT
        assert attempt.test_parameters is None
        assert attempt.execution_result is None
        assert attempt.overall_pass is None
        assert attempt.blocking_issues == []

    def test_with_inspection_results(self) -> None:
        attempt = RoutineAttempt(
            routine_spec_id="abc",
            routine_json={"name": "test", "operations": []},
            status=RoutineAttemptStatus.FAILED,
            overall_pass=False,
            blocking_issues=["Missing auth header", "Wrong URL"],
            recommendations=["Add retry logic"],
        )
        assert attempt.overall_pass is False
        assert len(attempt.blocking_issues) == 2

    def test_lineage(self) -> None:
        parent = RoutineAttempt(
            routine_spec_id="abc",
            routine_json={"name": "v1", "operations": []},
        )
        child = RoutineAttempt(
            routine_spec_id="abc",
            routine_json={"name": "v2", "operations": []},
            parent_attempt_id=parent.id,
            changes_from_parent="Added auth header to fetch",
        )
        assert child.parent_attempt_id == parent.id
        assert "auth header" in child.changes_from_parent


class TestShippedRoutine:
    """Tests for ShippedRoutine."""

    def test_creation(self) -> None:
        sr = ShippedRoutine(
            routine_spec_id="abc",
            routine_json={"name": "search_trains", "operations": []},
            name="search_trains",
            description="Search for train schedules",
            when_to_use="When user wants to find train times and prices",
            parameters_summary=["origin: departure station", "destination: arrival station"],
            inspection_score=85,
        )
        assert sr.name == "search_trains"
        assert sr.inspection_score == 85
        assert len(sr.parameters_summary) == 2


class TestRoutineCatalog:
    """Tests for RoutineCatalog."""

    def test_defaults(self) -> None:
        catalog = RoutineCatalog(
            site="amtrak.com",
            user_task="Search for train tickets",
        )
        assert catalog.routines == []
        assert catalog.failed_routines == []
        assert catalog.total_experiments == 0
        assert catalog.total_attempts == 0

    def test_with_routines(self) -> None:
        sr = ShippedRoutine(
            routine_spec_id="abc",
            routine_json={},
            name="search",
            description="Search trains",
            when_to_use="Find trains",
            inspection_score=90,
        )
        catalog = RoutineCatalog(
            site="amtrak.com",
            user_task="Search for train tickets",
            routines=[sr],
            total_experiments=5,
            total_attempts=3,
        )
        assert len(catalog.routines) == 1
        assert catalog.total_experiments == 5


class TestDiscoveryLedger:
    """Tests for DiscoveryLedger."""

    def _make_ledger(self) -> DiscoveryLedger:
        return DiscoveryLedger(user_task="Search for train tickets")

    def test_creation(self) -> None:
        ledger = self._make_ledger()
        assert ledger.user_task == "Search for train tickets"
        assert ledger.routine_specs == []
        assert ledger.experiments == []
        assert ledger.attempts == []
        assert ledger.catalog is None

    # --- Spec management ---

    def test_add_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search trains")
        result = ledger.add_spec(spec)
        assert result is spec
        assert len(ledger.routine_specs) == 1

    def test_get_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search trains")
        ledger.add_spec(spec)
        assert ledger.get_spec(spec.id) is spec

    def test_get_spec_not_found(self) -> None:
        ledger = self._make_ledger()
        assert ledger.get_spec("nonexistent") is None

    def test_get_active_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search")
        ledger.add_spec(spec)
        ledger.active_spec_id = spec.id
        assert ledger.get_active_spec() is spec

    def test_get_active_spec_none(self) -> None:
        ledger = self._make_ledger()
        assert ledger.get_active_spec() is None

    # --- Experiment management ---

    def test_add_experiment(self) -> None:
        ledger = self._make_ledger()
        exp = ExperimentEntry(hypothesis="h", rationale="r", methodology="m")
        result = ledger.add_experiment(exp)
        assert result is exp
        assert len(ledger.experiments) == 1

    def test_add_experiment_links_to_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="s", description="d")
        ledger.add_spec(spec)
        exp = ExperimentEntry(
            hypothesis="h",
            rationale="r",
            methodology="m",
            routine_spec_id=spec.id,
        )
        ledger.add_experiment(exp)
        assert exp.id in spec.experiment_ids

    def test_get_experiment(self) -> None:
        ledger = self._make_ledger()
        exp = ExperimentEntry(hypothesis="h", rationale="r", methodology="m")
        ledger.add_experiment(exp)
        assert ledger.get_experiment(exp.id) is exp

    def test_get_experiment_not_found(self) -> None:
        ledger = self._make_ledger()
        assert ledger.get_experiment("nope") is None

    def test_get_running_experiments(self) -> None:
        ledger = self._make_ledger()
        e1 = ExperimentEntry(hypothesis="h1", rationale="r", methodology="m", status=ExperimentStatus.RUNNING)
        e2 = ExperimentEntry(hypothesis="h2", rationale="r", methodology="m", status=ExperimentStatus.DONE)
        ledger.add_experiment(e1)
        ledger.add_experiment(e2)
        assert len(ledger.get_running_experiments()) == 1

    def test_get_confirmed_experiments(self) -> None:
        ledger = self._make_ledger()
        e1 = ExperimentEntry(hypothesis="h1", rationale="r", methodology="m", verdict=ExperimentVerdict.CONFIRMED)
        e2 = ExperimentEntry(hypothesis="h2", rationale="r", methodology="m", verdict=ExperimentVerdict.REFUTED)
        ledger.add_experiment(e1)
        ledger.add_experiment(e2)
        assert len(ledger.get_confirmed_experiments()) == 1

    def test_get_experiments_for_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="s", description="d")
        ledger.add_spec(spec)
        exp = ExperimentEntry(
            hypothesis="h",
            rationale="r",
            methodology="m",
            routine_spec_id=spec.id,
        )
        ledger.add_experiment(exp)
        result = ledger.get_experiments_for_spec(spec.id)
        assert len(result) == 1
        assert result[0] is exp

    def test_get_experiments_for_nonexistent_spec(self) -> None:
        ledger = self._make_ledger()
        assert ledger.get_experiments_for_spec("nope") == []

    # --- Attempt management ---

    def test_add_attempt(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="s", description="d")
        ledger.add_spec(spec)
        attempt = RoutineAttempt(
            routine_spec_id=spec.id,
            routine_json={"name": "test", "operations": []},
        )
        ledger.add_attempt(attempt)
        assert len(ledger.attempts) == 1
        assert attempt.id in spec.attempt_ids

    def test_get_attempt(self) -> None:
        ledger = self._make_ledger()
        attempt = RoutineAttempt(routine_spec_id="x", routine_json={})
        ledger.attempts.append(attempt)
        assert ledger.get_attempt(attempt.id) is attempt

    def test_get_attempt_not_found(self) -> None:
        ledger = self._make_ledger()
        assert ledger.get_attempt("nope") is None

    def test_get_attempts_for_spec(self) -> None:
        ledger = self._make_ledger()
        a1 = RoutineAttempt(routine_spec_id="spec1", routine_json={})
        a2 = RoutineAttempt(routine_spec_id="spec2", routine_json={})
        a3 = RoutineAttempt(routine_spec_id="spec1", routine_json={})
        ledger.attempts.extend([a1, a2, a3])
        result = ledger.get_attempts_for_spec("spec1")
        assert len(result) == 2

    # --- Summary rendering ---

    def test_to_summary_empty(self) -> None:
        ledger = self._make_ledger()
        assert ledger.to_summary() == "(no activity yet)"

    def test_to_summary_with_specs(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search_trains", description="Search for trains")
        ledger.add_spec(spec)
        summary = ledger.to_summary()
        assert "Routine Catalog Plan" in summary
        assert "search_trains" in summary

    def test_to_summary_with_active_spec(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search")
        ledger.add_spec(spec)
        ledger.active_spec_id = spec.id
        summary = ledger.to_summary()
        assert "ACTIVE" in summary

    def test_to_summary_with_shipped(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search", status=RoutineSpecStatus.SHIPPED)
        ledger.add_spec(spec)
        summary = ledger.to_summary()
        assert "Shipped Routines" in summary

    def test_to_summary_with_experiments(self) -> None:
        ledger = self._make_ledger()
        exp = ExperimentEntry(
            hypothesis="API uses Bearer auth",
            rationale="r",
            methodology="m",
            verdict=ExperimentVerdict.CONFIRMED,
            summary="Confirmed",
        )
        ledger.add_experiment(exp)
        summary = ledger.to_summary()
        assert "Experiment History" in summary
        assert "API uses Bearer auth" in summary

    def test_to_summary_with_proven_artifacts(self) -> None:
        ledger = self._make_ledger()
        ledger.proven.fetches.append({"method": "POST", "url": "/api/search"})
        summary = ledger.to_summary()
        assert "Proven Artifacts" in summary
        assert "FETCH" in summary

    def test_to_summary_with_failed_attempt(self) -> None:
        ledger = self._make_ledger()
        spec = RoutineSpec(name="search", description="Search")
        ledger.add_spec(spec)
        attempt = RoutineAttempt(
            routine_spec_id=spec.id,
            routine_json={},
            status=RoutineAttemptStatus.FAILED,
            blocking_issues=["Missing auth"],
        )
        ledger.add_attempt(attempt)
        summary = ledger.to_summary()
        assert "LAST FAILURE" in summary
