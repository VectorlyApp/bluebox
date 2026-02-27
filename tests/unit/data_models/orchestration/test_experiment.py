"""
tests/unit/data_models/orchestration/test_experiment.py

Tests for ExperimentEntry, ExperimentLog, ProvenArtifacts, ExperimentTakeaway.
"""

import pytest

from bluebox.data_models.orchestration.experiment import (
    ArtifactType,
    ExperimentEntry,
    ExperimentLog,
    ExperimentStatus,
    ExperimentTakeaway,
    ExperimentVerdict,
    ProvenArtifacts,
)


class TestExperimentEnums:
    """Tests for experiment-related enums."""

    def test_verdict_values(self) -> None:
        assert set(ExperimentVerdict) == {
            ExperimentVerdict.CONFIRMED,
            ExperimentVerdict.REFUTED,
            ExperimentVerdict.PARTIAL,
            ExperimentVerdict.NEEDS_FOLLOWUP,
        }

    def test_status_values(self) -> None:
        assert set(ExperimentStatus) == {
            ExperimentStatus.PENDING,
            ExperimentStatus.RUNNING,
            ExperimentStatus.DONE,
            ExperimentStatus.FAILED,
        }

    def test_artifact_type_values(self) -> None:
        assert set(ArtifactType) == {
            ArtifactType.FETCH,
            ArtifactType.NAVIGATION,
            ArtifactType.TOKEN,
            ArtifactType.PARAMETER,
        }


class TestExperimentTakeaway:
    """Tests for ExperimentTakeaway."""

    def test_minimal(self) -> None:
        t = ExperimentTakeaway(claim="Auth uses Bearer JWT from sessionStorage")
        assert t.claim == "Auth uses Bearer JWT from sessionStorage"
        assert t.evidence is None
        assert t.tags == []

    def test_full(self) -> None:
        t = ExperimentTakeaway(
            claim="CSRF token rotates per page load",
            evidence="Observed 3 different values across 3 page loads",
            how_to_apply_next="Extract CSRF from meta tag before each POST",
            confidence=0.9,
            tags=["auth", "anti_bot"],
        )
        assert t.confidence == 0.9
        assert "auth" in t.tags


class TestExperimentEntry:
    """Tests for ExperimentEntry."""

    def test_defaults(self) -> None:
        entry = ExperimentEntry(
            hypothesis="POST /api/search returns train schedules",
            rationale="Observed this endpoint during recording",
            methodology="Send a POST with sample params and check response",
        )
        assert len(entry.id) == 6
        assert entry.status == ExperimentStatus.PENDING
        assert entry.verdict is None
        assert entry.takeaways == []
        assert entry.output is None

    def test_unique_ids(self) -> None:
        entries = [
            ExperimentEntry(hypothesis="h", rationale="r", methodology="m")
            for _ in range(20)
        ]
        assert len({e.id for e in entries}) == 20

    def test_with_result(self) -> None:
        entry = ExperimentEntry(
            hypothesis="test",
            rationale="test",
            methodology="test",
            status=ExperimentStatus.DONE,
            verdict=ExperimentVerdict.CONFIRMED,
            summary="Endpoint works as expected",
            takeaways=[ExperimentTakeaway(claim="It works")],
            output={"status": 200, "data": {"trains": []}},
        )
        assert entry.verdict == ExperimentVerdict.CONFIRMED
        assert len(entry.takeaways) == 1


class TestProvenArtifacts:
    """Tests for ProvenArtifacts."""

    def test_defaults_empty(self) -> None:
        pa = ProvenArtifacts()
        assert pa.fetches == []
        assert pa.navigations == []
        assert pa.tokens == []
        assert pa.parameters == []

    def test_accumulation(self) -> None:
        pa = ProvenArtifacts()
        pa.fetches.append({"method": "POST", "url": "/api/search"})
        pa.tokens.append({"name": "csrf", "source": "meta tag"})
        pa.parameters.append({"name": "origin", "type": "string"})
        assert len(pa.fetches) == 1
        assert len(pa.tokens) == 1
        assert len(pa.parameters) == 1


class TestExperimentLog:
    """Tests for ExperimentLog."""

    def _make_log(self) -> ExperimentLog:
        return ExperimentLog(user_task="Search for train tickets")

    def test_creation(self) -> None:
        log = self._make_log()
        assert log.user_task == "Search for train tickets"
        assert log.experiments == []
        assert log.unresolved == []

    def test_add_experiment(self) -> None:
        log = self._make_log()
        entry = ExperimentEntry(hypothesis="h", rationale="r", methodology="m")
        result = log.add_experiment(entry)
        assert result is entry
        assert len(log.experiments) == 1

    def test_get_experiment_found(self) -> None:
        log = self._make_log()
        entry = ExperimentEntry(hypothesis="h", rationale="r", methodology="m")
        log.add_experiment(entry)
        found = log.get_experiment(entry.id)
        assert found is entry

    def test_get_experiment_not_found(self) -> None:
        log = self._make_log()
        assert log.get_experiment("nonexistent") is None

    def test_get_running_experiments(self) -> None:
        log = self._make_log()
        e1 = ExperimentEntry(hypothesis="h1", rationale="r", methodology="m", status=ExperimentStatus.RUNNING)
        e2 = ExperimentEntry(hypothesis="h2", rationale="r", methodology="m", status=ExperimentStatus.DONE)
        e3 = ExperimentEntry(hypothesis="h3", rationale="r", methodology="m", status=ExperimentStatus.RUNNING)
        log.add_experiment(e1)
        log.add_experiment(e2)
        log.add_experiment(e3)
        running = log.get_running_experiments()
        assert len(running) == 2

    def test_get_confirmed_experiments(self) -> None:
        log = self._make_log()
        e1 = ExperimentEntry(hypothesis="h1", rationale="r", methodology="m", verdict=ExperimentVerdict.CONFIRMED)
        e2 = ExperimentEntry(hypothesis="h2", rationale="r", methodology="m", verdict=ExperimentVerdict.REFUTED)
        log.add_experiment(e1)
        log.add_experiment(e2)
        confirmed = log.get_confirmed_experiments()
        assert len(confirmed) == 1
        assert confirmed[0].hypothesis == "h1"

    def test_to_summary_empty(self) -> None:
        log = self._make_log()
        assert log.to_summary() == "(no experiments yet)"

    def test_to_summary_with_experiments(self) -> None:
        log = self._make_log()
        entry = ExperimentEntry(
            hypothesis="API uses Bearer auth",
            rationale="saw headers",
            methodology="test",
            status=ExperimentStatus.DONE,
            verdict=ExperimentVerdict.CONFIRMED,
            summary="Confirmed Bearer auth pattern",
        )
        log.add_experiment(entry)
        summary = log.to_summary()
        assert "Experiment History" in summary
        assert "API uses Bearer auth" in summary
        assert "Confirmed Bearer auth pattern" in summary

    def test_to_summary_with_proven_artifacts(self) -> None:
        log = self._make_log()
        log.proven.fetches.append({"method": "POST", "url": "/api/search"})
        log.proven.tokens.append({"name": "csrf", "source": "meta"})
        summary = log.to_summary()
        assert "Proven Artifacts" in summary
        assert "FETCH" in summary
        assert "TOKEN" in summary

    def test_to_summary_with_unresolved(self) -> None:
        log = self._make_log()
        log.unresolved.append("How does pagination work?")
        summary = log.to_summary()
        assert "Unresolved Questions" in summary
        assert "pagination" in summary

    def test_to_summary_with_takeaways(self) -> None:
        log = self._make_log()
        entry = ExperimentEntry(
            hypothesis="h",
            rationale="r",
            methodology="m",
            takeaways=[ExperimentTakeaway(claim="CSRF rotates per page load")],
        )
        log.add_experiment(entry)
        summary = log.to_summary()
        assert "takeaways" in summary
