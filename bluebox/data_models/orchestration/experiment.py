"""
bluebox/data_models/orchestration/experiment.py

Experiment log models for the PrincipalInvestigator orchestrator.

The experiment log is the PI's working memory — it tracks hypotheses,
results, proven artifacts, and unresolved questions across the entire
routine construction process.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from bluebox.data_models.orchestration.task import generate_short_id


class ExperimentVerdict(StrEnum):
    """Verdict after reviewing an experiment result."""
    CONFIRMED = "confirmed"
    REFUTED = "refuted"
    PARTIAL = "partial"
    NEEDS_FOLLOWUP = "needs_followup"


class ExperimentStatus(StrEnum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ExperimentTakeaway(BaseModel):
    """Reusable lesson extracted from an experiment result."""

    claim: str = Field(description="Concrete claim learned from the experiment")
    evidence: str | None = Field(
        default=None,
        description="Evidence supporting the claim (endpoint/status/value/log snippet)",
    )
    how_to_apply_next: str | None = Field(
        default=None,
        description="How future workers should apply this claim",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence in [0, 1]",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags like auth, pagination, anti_bot, endpoint, parameter",
    )


class ExperimentEntry(BaseModel):
    """A single experiment in the PI's log."""

    id: str = Field(default_factory=generate_short_id)
    hypothesis: str = Field(description="Specific, falsifiable hypothesis being tested")
    rationale: str = Field(description="Why we're testing this — evidence, reasoning, expectations")
    prompt: str = Field(description="Instructions sent to the worker")
    routine_spec_id: str | None = Field(
        default=None,
        description="RoutineSpec this experiment supports; None for shared/global experiments",
    )
    priority: int = Field(default=1, description="1=critical, 2=important, 3=nice-to-have")
    task_id: str | None = Field(default=None, description="Reference to the dispatched Task")
    status: ExperimentStatus = Field(default=ExperimentStatus.PENDING)
    verdict: ExperimentVerdict | None = Field(default=None)
    summary: str | None = Field(default=None, description="What we learned (recorded by PI)")
    takeaways: list[ExperimentTakeaway] = Field(
        default_factory=list,
        description="Reusable lessons that future workers can apply",
    )
    output: dict[str, Any] | None = Field(default=None, description="Raw worker output")


class ArtifactType(StrEnum):
    """Types of proven artifacts."""
    FETCH = "fetch"
    NAVIGATION = "navigation"
    TOKEN = "token"
    PARAMETER = "parameter"


class ProvenArtifacts(BaseModel):
    """Accumulates proven artifacts from confirmed experiments."""

    fetches: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Proven API calls: url, method, headers, body, response_preview",
    )
    navigations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Proven navigation steps: url, sets_up (cookies, storage_keys)",
    )
    tokens: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Proven tokens: name, source, storage_type, key_name",
    )
    parameters: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Proven user parameters: name, type, description, example_value",
    )


class ExperimentLog(BaseModel):
    """
    The PI's working memory.

    Tracks all experiments, their results, proven artifacts, and
    unresolved questions. Injected into the PI's system prompt so it
    can track progress across iterations.
    """

    user_task: str = Field(description="The user's original task description")
    experiments: list[ExperimentEntry] = Field(default_factory=list)
    proven: ProvenArtifacts = Field(default_factory=ProvenArtifacts)
    unresolved: list[str] = Field(
        default_factory=list,
        description="Questions we still don't have answers to",
    )

    # Convenience methods

    def add_experiment(self, experiment: ExperimentEntry) -> ExperimentEntry:
        """Add an experiment to the log."""
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

    def to_summary(self) -> str:
        """Render the experiment log as a text summary for the PI's system prompt."""
        lines: list[str] = []

        # Experiments
        if self.experiments:
            lines.append("## Experiment History")
            for exp in self.experiments:
                status_str = f"[{exp.status.value}]"
                if exp.verdict:
                    status_str += f" verdict={exp.verdict.value}"
                lines.append(f"- {exp.id}: {status_str} {exp.hypothesis}")
                if exp.summary:
                    lines.append(f"  → {exp.summary}")
                if exp.takeaways:
                    first_claim = exp.takeaways[0].claim.strip()
                    preview = first_claim[:120] + ("..." if len(first_claim) > 120 else "")
                    lines.append(f"  → takeaways={len(exp.takeaways)} (e.g. {preview})")
            lines.append("")

        # Proven artifacts
        proven = self.proven
        if proven.fetches or proven.navigations or proven.tokens or proven.parameters:
            lines.append("## Proven Artifacts")
            for f in proven.fetches:
                lines.append(f"- FETCH: {f.get('method', '?')} {f.get('url', '?')}")
            for n in proven.navigations:
                lines.append(f"- NAV: {n.get('url', '?')}")
            for t in proven.tokens:
                lines.append(f"- TOKEN: {t.get('name', '?')} from {t.get('source', '?')}")
            for p in proven.parameters:
                lines.append(f"- PARAM: {p.get('name', '?')} ({p.get('type', '?')})")
            lines.append("")

        # Unresolved
        if self.unresolved:
            lines.append("## Unresolved Questions")
            for q in self.unresolved:
                lines.append(f"- {q}")
            lines.append("")

        return "\n".join(lines) if lines else "(no experiments yet)"
