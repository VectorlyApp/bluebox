"""
bluebox/data_models/orchestration/ledger.py

Discovery Ledger — the PI's pipeline-level tracker for multi-routine
catalog construction.

Tracks:
- What routines the PI plans to build (RoutineSpec catalog plan)
- All experiments across all routines (shared pool)
- All routine attempts, executions, and inspections
- The final deliverable: a RoutineCatalog of shipped routines
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from bluebox.data_models.orchestration.experiment import (
    ExperimentEntry,
    ExperimentStatus,
    ExperimentVerdict,
    ProvenArtifacts,
)
from bluebox.data_models.orchestration.inspection import RoutineInspectionResult
from bluebox.data_models.orchestration.task import generate_short_id


# ---------------------------------------------------------------------------
# Routine lifecycle
# ---------------------------------------------------------------------------

class RoutineSpecStatus(StrEnum):
    """Status of a planned routine."""
    PLANNED = "planned"
    EXPERIMENTING = "experimenting"
    ASSEMBLING = "assembling"
    VALIDATING = "validating"
    SHIPPED = "shipped"
    FAILED = "failed"


class RoutineAttemptStatus(StrEnum):
    """Status of a routine attempt."""
    DRAFT = "draft"
    VALIDATING = "validating"
    EXECUTING = "executing"
    INSPECTING = "inspecting"
    PASSED = "passed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Routine planning — what to build
# ---------------------------------------------------------------------------

class RoutineSpec(BaseModel):
    """A planned routine in the catalog."""

    id: str = Field(default_factory=generate_short_id)
    name: str = Field(description="Short name e.g. 'get_league_standings'")
    description: str = Field(description="What this routine does")
    status: RoutineSpecStatus = Field(default=RoutineSpecStatus.PLANNED)
    priority: int = Field(default=1, description="1=must-have, 2=should-have, 3=nice-to-have")
    depends_on_specs: list[str] = Field(
        default_factory=list,
        description="Other RoutineSpec IDs that share dependencies (e.g. same auth)",
    )
    experiment_ids: list[str] = Field(
        default_factory=list,
        description="IDs of experiments run for this routine",
    )
    attempt_ids: list[str] = Field(
        default_factory=list,
        description="IDs of routine attempts for this routine",
    )
    shipped_attempt_id: str | None = Field(
        default=None,
        description="The attempt that passed inspection and was shipped",
    )
    failure_reason: str | None = Field(
        default=None,
        description="Why the routine was abandoned (if status=failed)",
    )


# ---------------------------------------------------------------------------
# Routine attempts — per routine spec
# ---------------------------------------------------------------------------

class RoutineAttempt(BaseModel):
    """A single attempt at constructing a routine for a spec."""

    id: str = Field(default_factory=generate_short_id)
    routine_spec_id: str = Field(description="Which RoutineSpec this attempt is for")
    routine_json: dict[str, Any] = Field(description="The full routine as a dict")
    status: RoutineAttemptStatus = Field(default=RoutineAttemptStatus.DRAFT)

    # Execution
    test_parameters: dict[str, Any] | None = Field(default=None)
    execution_result: dict[str, Any] | None = Field(default=None)
    execution_error: str | None = Field(default=None)

    # Inspection
    inspection_result: dict[str, Any] | None = Field(default=None)
    overall_pass: bool | None = Field(default=None)
    blocking_issues: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)

    # Lineage
    parent_attempt_id: str | None = Field(
        default=None,
        description="Previous attempt this was derived from",
    )
    changes_from_parent: str | None = Field(
        default=None,
        description="What changed from the parent attempt",
    )


# ---------------------------------------------------------------------------
# Final deliverable
# ---------------------------------------------------------------------------

class ShippedRoutine(BaseModel):
    """A routine that passed inspection and is ready for use."""

    routine_spec_id: str = Field(description="Which spec this fulfills")
    routine_json: dict[str, Any] = Field(description="The final routine")
    name: str = Field(description="Routine name")
    description: str = Field(description="What it does")
    when_to_use: str = Field(description="Guidance for the user on when to use this routine")
    parameters_summary: list[str] = Field(
        default_factory=list,
        description="Human-readable parameter descriptions",
    )
    inspection_score: int = Field(description="Inspector's final score (0-100)")


class RoutineCatalog(BaseModel):
    """The final output of the entire pipeline — a catalog of shipped routines."""

    site: str = Field(description="Primary domain e.g. 'premierleague.com'")
    user_task: str = Field(description="The original user task")
    routines: list[ShippedRoutine] = Field(
        default_factory=list,
        description="All shipped routines",
    )
    usage_guide: str = Field(
        default="",
        description="How to use these routines together",
    )
    failed_routines: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Routines we couldn't build + why",
    )
    total_experiments: int = Field(default=0)
    total_attempts: int = Field(default=0)


# ---------------------------------------------------------------------------
# The Ledger — everything the PI needs to see
# ---------------------------------------------------------------------------

class DiscoveryLedger(BaseModel):
    """
    Pipeline-level tracker for the PI's entire API indexing lifecycle.

    Tracks routine specs, experiments, proven artifacts, routine attempts,
    and the final catalog. A compact view is rendered into the PI's
    system prompt every iteration so the PI can track progress.
    """

    # Context
    user_task: str = Field(description="The user's original task description")

    # Planning — what routines to build
    routine_specs: list[RoutineSpec] = Field(default_factory=list)
    active_spec_id: str | None = Field(
        default=None,
        description="Which routine the PI is currently working on",
    )

    # Experiments (shared pool across all routines)
    experiments: list[ExperimentEntry] = Field(default_factory=list)
    proven: ProvenArtifacts = Field(default_factory=ProvenArtifacts)
    unresolved: list[str] = Field(
        default_factory=list,
        description="Questions we still don't have answers to",
    )

    # Routine attempts (across all specs)
    attempts: list[RoutineAttempt] = Field(default_factory=list)

    # Final output
    catalog: RoutineCatalog | None = Field(
        default=None,
        description="Built when PI calls mark_complete",
    )

    # -----------------------------------------------------------------------
    # Convenience methods — specs
    # -----------------------------------------------------------------------

    def add_spec(self, spec: RoutineSpec) -> RoutineSpec:
        """Add a routine spec to the catalog plan."""
        self.routine_specs.append(spec)
        return spec

    def get_spec(self, spec_id: str) -> RoutineSpec | None:
        """Find a routine spec by ID."""
        for spec in self.routine_specs:
            if spec.id == spec_id:
                return spec
        return None

    def get_active_spec(self) -> RoutineSpec | None:
        """Get the currently active routine spec."""
        if self.active_spec_id is None:
            return None
        return self.get_spec(self.active_spec_id)

    # -----------------------------------------------------------------------
    # Convenience methods — experiments
    # -----------------------------------------------------------------------

    def add_experiment(self, experiment: ExperimentEntry) -> ExperimentEntry:
        """Add an experiment to the shared pool."""
        self.experiments.append(experiment)
        return experiment

    def get_experiment(self, experiment_id: str) -> ExperimentEntry | None:
        """Find an experiment by ID."""
        for exp in self.experiments:
            if exp.id == experiment_id:
                return exp
        return None

    def get_running_experiments(self) -> list[ExperimentEntry]:
        """Get all currently running experiments."""
        return [e for e in self.experiments if e.status == ExperimentStatus.RUNNING]

    def get_confirmed_experiments(self) -> list[ExperimentEntry]:
        """Get all confirmed experiments."""
        return [e for e in self.experiments if e.verdict == ExperimentVerdict.CONFIRMED]

    def get_experiments_for_spec(self, spec_id: str) -> list[ExperimentEntry]:
        """Get experiments for a specific routine spec."""
        spec = self.get_spec(spec_id)
        if spec is None:
            return []
        return [e for e in self.experiments if e.id in spec.experiment_ids]

    # -----------------------------------------------------------------------
    # Convenience methods — attempts
    # -----------------------------------------------------------------------

    def add_attempt(self, attempt: RoutineAttempt) -> RoutineAttempt:
        """Add a routine attempt."""
        self.attempts.append(attempt)
        spec = self.get_spec(attempt.routine_spec_id)
        if spec:
            spec.attempt_ids.append(attempt.id)
        return attempt

    def get_attempt(self, attempt_id: str) -> RoutineAttempt | None:
        """Find a routine attempt by ID."""
        for attempt in self.attempts:
            if attempt.id == attempt_id:
                return attempt
        return None

    def get_attempts_for_spec(self, spec_id: str) -> list[RoutineAttempt]:
        """Get all attempts for a specific routine spec."""
        return [a for a in self.attempts if a.routine_spec_id == spec_id]

    # -----------------------------------------------------------------------
    # Summary rendering for system prompt
    # -----------------------------------------------------------------------

    def to_summary(self) -> str:
        """Render a compact view of the ledger for the PI's system prompt."""
        lines: list[str] = []

        # Routine catalog plan
        if self.routine_specs:
            lines.append("## Routine Catalog Plan")
            for i, spec in enumerate(self.routine_specs, 1):
                status_badge = f"[{spec.status.value}]"
                lines.append(f"  {i}. {status_badge:16s} {spec.name} — \"{spec.description}\"")
            if self.active_spec_id:
                active = self.get_active_spec()
                if active:
                    lines.append(f"\n  ACTIVE: {active.name}")
                    spec_exps = self.get_experiments_for_spec(active.id)
                    confirmed = sum(1 for e in spec_exps if e.verdict == ExperimentVerdict.CONFIRMED)
                    lines.append(
                        f"    Experiments: {len(spec_exps)} run ({confirmed} confirmed)"
                    )
                    spec_attempts = self.get_attempts_for_spec(active.id)
                    if spec_attempts:
                        latest = spec_attempts[-1]
                        lines.append(
                            f"    Latest attempt: {latest.id} [{latest.status.value}]"
                        )
            lines.append("")

        # Shipped routines
        shipped = [s for s in self.routine_specs if s.status == RoutineSpecStatus.SHIPPED]
        if shipped:
            lines.append("## Shipped Routines")
            for spec in shipped:
                attempt = self.get_attempt(spec.shipped_attempt_id) if spec.shipped_attempt_id else None
                score = ""
                if attempt and attempt.inspection_result:
                    score = f" (score: {attempt.inspection_result.get('overall_score', '?')}/100)"
                lines.append(f"  {spec.name}{score}")
            lines.append("")

        # Experiment history
        if self.experiments:
            lines.append("## Experiment History")
            for exp in self.experiments:
                verdict_icon = {
                    ExperimentVerdict.CONFIRMED: "+",
                    ExperimentVerdict.REFUTED: "x",
                    ExperimentVerdict.PARTIAL: "~",
                    ExperimentVerdict.NEEDS_FOLLOWUP: "?",
                }.get(exp.verdict, " ") if exp.verdict else " "
                lines.append(f"  [{verdict_icon}] {exp.id}: {exp.hypothesis}")
                if exp.summary:
                    lines.append(f"      → {exp.summary}")
            lines.append("")

        # Proven artifacts
        proven = self.proven
        if proven.fetches or proven.navigations or proven.tokens or proven.parameters:
            lines.append("## Proven Artifacts")
            for f in proven.fetches:
                lines.append(f"  FETCH: {f.get('method', '?')} {f.get('url', '?')}")
            for n in proven.navigations:
                lines.append(f"  NAV: {n.get('url', '?')}")
            for t in proven.tokens:
                lines.append(f"  TOKEN: {t.get('name', '?')} from {t.get('source', '?')}")
            for p in proven.parameters:
                lines.append(f"  PARAM: {p.get('name', '?')} ({p.get('type', '?')})")
            lines.append("")

        # Unresolved
        if self.unresolved:
            lines.append("## Unresolved Questions")
            for q in self.unresolved:
                lines.append(f"  - {q}")
            lines.append("")

        return "\n".join(lines) if lines else "(no activity yet)"
