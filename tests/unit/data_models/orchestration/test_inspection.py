"""
tests/unit/data_models/orchestration/test_inspection.py

Tests for DimensionScore, RoutineInspectionDimensions, RoutineInspectionResult.
"""

import pytest

from bluebox.data_models.orchestration.inspection import (
    DimensionScore,
    RoutineInspectionDimensions,
    RoutineInspectionResult,
)


class TestDimensionScore:
    """Tests for DimensionScore with clamping validator."""

    def test_normal_score(self) -> None:
        ds = DimensionScore(score=7, reasoning="Good")
        assert ds.score == 7

    def test_score_clamped_above_10(self) -> None:
        ds = DimensionScore(score=15, reasoning="over")
        assert ds.score == 10

    def test_score_clamped_below_0(self) -> None:
        ds = DimensionScore(score=-5, reasoning="under")
        assert ds.score == 0

    def test_score_from_string(self) -> None:
        ds = DimensionScore(score="8", reasoning="from string")
        assert ds.score == 8

    def test_score_string_clamped(self) -> None:
        ds = DimensionScore(score="20", reasoning="over string")
        assert ds.score == 10

    def test_score_from_float(self) -> None:
        ds = DimensionScore(score=7.9, reasoning="float")
        assert ds.score == 7

    def test_bool_true(self) -> None:
        ds = DimensionScore(score=True, reasoning="bool")
        assert ds.score == 1

    def test_bool_false(self) -> None:
        ds = DimensionScore(score=False, reasoning="bool")
        assert ds.score == 0

    def test_boundary_0(self) -> None:
        ds = DimensionScore(score=0, reasoning="zero")
        assert ds.score == 0

    def test_boundary_10(self) -> None:
        ds = DimensionScore(score=10, reasoning="ten")
        assert ds.score == 10


def _make_dims(score: int = 8) -> RoutineInspectionDimensions:
    ds = DimensionScore(score=score, reasoning="test")
    return RoutineInspectionDimensions(
        task_completion=ds,
        data_quality=ds,
        parameter_coverage=ds,
        routine_robustness=ds,
        structural_correctness=ds,
        documentation_quality=ds,
    )


class TestRoutineInspectionDimensions:
    """Tests for RoutineInspectionDimensions."""

    def test_all_dimensions(self) -> None:
        dims = _make_dims(9)
        assert dims.task_completion.score == 9
        assert dims.data_quality.score == 9
        assert dims.parameter_coverage.score == 9
        assert dims.routine_robustness.score == 9
        assert dims.structural_correctness.score == 9
        assert dims.documentation_quality.score == 9

    def test_missing_dimension_raises(self) -> None:
        ds = DimensionScore(score=5, reasoning="test")
        with pytest.raises(Exception):
            RoutineInspectionDimensions(task_completion=ds)


class TestRoutineInspectionResult:
    """Tests for RoutineInspectionResult."""

    def test_passing(self) -> None:
        result = RoutineInspectionResult(
            overall_pass=True,
            overall_score=85,
            dimensions=_make_dims(),
            summary="Looks good.",
        )
        assert result.overall_pass is True
        assert result.overall_score == 85
        assert result.blocking_issues == []
        assert result.recommendations == []

    def test_failing_with_issues(self) -> None:
        result = RoutineInspectionResult(
            overall_pass=False,
            overall_score=35,
            dimensions=_make_dims(3),
            summary="Needs work.",
            blocking_issues=["Missing auth", "Wrong endpoint"],
            recommendations=["Add retry"],
        )
        assert result.overall_pass is False
        assert len(result.blocking_issues) == 2
        assert len(result.recommendations) == 1

    def test_score_bounds_valid(self) -> None:
        RoutineInspectionResult(
            overall_pass=True, overall_score=0, dimensions=_make_dims(), summary="s"
        )
        RoutineInspectionResult(
            overall_pass=True, overall_score=100, dimensions=_make_dims(), summary="s"
        )

    def test_score_out_of_bounds(self) -> None:
        with pytest.raises(Exception):
            RoutineInspectionResult(
                overall_pass=True, overall_score=101, dimensions=_make_dims(), summary="s"
            )
        with pytest.raises(Exception):
            RoutineInspectionResult(
                overall_pass=True, overall_score=-1, dimensions=_make_dims(), summary="s"
            )

    def test_roundtrip(self) -> None:
        result = RoutineInspectionResult(
            overall_pass=True,
            overall_score=80,
            dimensions=_make_dims(),
            summary="Good routine.",
            blocking_issues=[],
            recommendations=["Minor: add descriptions"],
        )
        data = result.model_dump()
        result2 = RoutineInspectionResult.model_validate(data)
        assert result2.overall_score == 80
        assert result2.recommendations == ["Minor: add descriptions"]
