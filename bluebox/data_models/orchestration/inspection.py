"""
bluebox/data_models/orchestration/inspection.py

Data models for the RoutineInspector — the independent quality gate
that judges routines after construction.
"""

from pydantic import BaseModel, Field, field_validator


class DimensionScore(BaseModel):
    """Score for a single quality dimension."""

    score: int = Field(ge=0, description="Score from 0 (terrible) to 10 (perfect)")
    reasoning: str = Field(description="Why this score was given")

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, value: object) -> object:
        """
        Clamp model-provided scores into the valid [0, 10] range.

        This prevents minor over/under-shoots (e.g. 11) from propagating.
        """
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str)):
            try:
                as_int = int(value)
            except (TypeError, ValueError):
                return value
            return max(0, min(10, as_int))
        return value


class RoutineInspectionDimensions(BaseModel):
    """Required quality dimensions for routine inspection."""

    task_completion: DimensionScore = Field(description="How well the routine fulfills its stated purpose")
    data_quality: DimensionScore = Field(description="Quality and usefulness of returned data")
    parameter_coverage: DimensionScore = Field(description="Whether parameterization is complete and appropriate")
    routine_robustness: DimensionScore = Field(description="Resilience across sessions and site variance")
    structural_correctness: DimensionScore = Field(description="Correct operation ordering and dependency structure")
    documentation_quality: DimensionScore = Field(description="Naming/description/parameter-doc quality for discoverability")


class RoutineInspectionResult(BaseModel):
    """
    Output of the RoutineInspector.

    The inspector scores the routine on 6 dimensions (0-10 each),
    identifies blocking issues vs. non-blocking recommendations,
    and renders a pass/fail verdict.
    """

    overall_pass: bool = Field(
        description="Whether the routine should ship (True) or needs fixes (False)"
    )
    overall_score: int = Field(
        ge=0,
        le=100,
        description="Round(sum of 6 dimension scores / 60 * 100)",
    )
    dimensions: RoutineInspectionDimensions = Field(
        description="Scores per required quality dimension",
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
