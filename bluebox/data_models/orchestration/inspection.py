"""
bluebox/data_models/orchestration/inspection.py

Data models for the RoutineInspector — the independent quality gate
that judges routines after construction.
"""

from typing import Any

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    score: int = Field(ge=0, le=10, description="Score from 0 (terrible) to 10 (perfect)")
    reasoning: str = Field(description="Why this score was given")


class RoutineInspectionResult(BaseModel):
    """
    Output of the RoutineInspector.

    The inspector scores the routine on 5 dimensions (0-10 each),
    identifies blocking issues vs. non-blocking recommendations,
    and renders a pass/fail verdict.
    """

    overall_pass: bool = Field(
        description="Whether the routine should ship (True) or needs fixes (False)"
    )
    overall_score: int = Field(
        ge=0,
        le=100,
        description="Sum of dimension scores × 2 (0-100 scale)",
    )
    dimensions: dict[str, DimensionScore] = Field(
        description=(
            "Scores per dimension: task_completion, data_quality, "
            "parameter_coverage, routine_robustness, structural_correctness"
        ),
    )
    blocking_issues: list[str] = Field(
        default_factory=list,
        description="Issues that MUST be fixed before shipping",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Issues that SHOULD be fixed but are non-blocking",
    )
    summary: str = Field(
        description="2-3 sentence overall assessment",
    )
