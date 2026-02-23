"""
bluebox/data_models/agents/context.py

Data model for BlueBoxAgent context files.

A context file captures the successful path through a BlueBoxAgent
conversation so a new agent instance can replay it without trial and error.

Supports dual format: canonical JSON (Pydantic) and human-readable Markdown,
with round-trip parsing between both.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class UsedRoutineParameter(BaseModel):
    """A single parameter key-value pair used in a routine execution."""

    key: str = Field(..., description="Parameter name")
    value: str | bool | int | float = Field(..., description="Parameter value")


class UsedRoutine(BaseModel):
    """One routine that was successfully executed during the session."""

    routine_id: str = Field(..., description="Routine ID from search_routines results")
    routine_name: str = Field(..., description="Human-readable routine name")
    parameters: list[UsedRoutineParameter] = Field(
        default_factory=list,
        description="Parameter key-value pairs that produced correct results",
    )

    def parameters_as_dict(self) -> dict[str, str | bool | int | float]:
        """Convert parameters list to a dict for convenience."""
        return {p.key: p.value for p in self.parameters}

    @classmethod
    def from_dict_params(
        cls, routine_id: str, routine_name: str, parameters: dict[str, Any],
    ) -> UsedRoutine:
        """Convenience constructor that accepts a dict of parameters."""
        return cls(
            routine_id=routine_id,
            routine_name=routine_name,
            parameters=[UsedRoutineParameter(key=k, value=v) for k, v in parameters.items()],
        )


class BlueBoxAgentContext(BaseModel):
    """
    Structured snapshot of a successful BlueBoxAgent session.

    Serialized to JSON and saved to context/. Consumed by a new
    BlueBoxAgent instance via system prompt injection.
    """

    version: int = Field(default=1, description="Schema version for forward compatibility")
    goal: str = Field(..., description="The user's original request, in their own words")
    routines_used: list[UsedRoutine] = Field(
        default_factory=list,
        description="Routines that produced useful results, in execution order",
    )
    python_code: str | None = Field(
        default=None,
        description="The final working Python post-processing snippet",
    )
    output_files: list[str] = Field(
        default_factory=list,
        description="Relative paths of output files written to outputs/",
    )
    output_description: str = Field(
        ...,
        description="Prose description of the output: format, key fields, row count if known",
    )
    summary: str = Field(
        ...,
        description="1-2 sentence human-readable summary of what was accomplished",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc),
        description="When this context was generated",
    )

    # ── Markdown serialization ───────────────────────────────────────────

    def to_markdown(self) -> str:
        """Render as structured Markdown with fenced sections for round-tripping."""
        lines: list[str] = []
        lines.append("# BlueBox Agent Context")
        lines.append("")
        lines.append(f"**Version:** {self.version}")
        lines.append(f"**Generated:** {self.generated_at.isoformat()}")
        lines.append("")

        lines.append("## Goal")
        lines.append("")
        lines.append(self.goal)
        lines.append("")

        lines.append("## Summary")
        lines.append("")
        lines.append(self.summary)
        lines.append("")

        if self.routines_used:
            lines.append("## Routines Used")
            lines.append("")
            for r in self.routines_used:
                lines.append(f"### {r.routine_name} (`{r.routine_id}`)")
                lines.append("")
                if r.parameters:
                    lines.append("**Parameters:**")
                    lines.append("```json")
                    lines.append(json.dumps(r.parameters_as_dict(), indent=2, default=str))
                    lines.append("```")
                else:
                    lines.append("No parameters.")
                lines.append("")

        if self.python_code:
            lines.append("## Python Code")
            lines.append("")
            lines.append("```python")
            lines.append(self.python_code)
            lines.append("```")
            lines.append("")

        if self.output_files:
            lines.append("## Output Files")
            lines.append("")
            for f in self.output_files:
                lines.append(f"- `{f}`")
            lines.append("")

        lines.append("## Output Description")
        lines.append("")
        lines.append(self.output_description)
        lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str) -> BlueBoxAgentContext:
        """Parse structured Markdown back into BlueBoxAgentContext."""
        sections = _split_markdown_sections(text)

        # Version and generated_at from header
        version = 1
        generated_at = datetime.now(tz=timezone.utc)
        header = sections.get("BlueBox Agent Context", "")
        version_match = re.search(r"\*\*Version:\*\*\s*(\d+)", header)
        if version_match:
            version = int(version_match.group(1))
        generated_match = re.search(r"\*\*Generated:\*\*\s*(.+)", header)
        if generated_match:
            try:
                generated_at = datetime.fromisoformat(generated_match.group(1).strip())
            except ValueError:
                pass

        goal = sections.get("Goal", "").strip()
        summary = sections.get("Summary", "").strip()
        output_description = sections.get("Output Description", "").strip()

        # Parse routines from subsections
        routines_used = _parse_routines_section(sections.get("Routines Used", ""))

        # Parse python code from fenced block
        python_code = _extract_fenced_block(sections.get("Python Code", ""), "python")

        # Parse output files
        output_files: list[str] = []
        for line in sections.get("Output Files", "").splitlines():
            match = re.match(r"^-\s*`(.+)`", line.strip())
            if match:
                output_files.append(match.group(1))

        return cls(
            version=version,
            goal=goal,
            summary=summary,
            output_description=output_description,
            routines_used=routines_used,
            python_code=python_code,
            output_files=output_files,
            generated_at=generated_at,
        )


# ── Markdown parsing helpers ─────────────────────────────────────────────


def _split_markdown_sections(text: str) -> dict[str, str]:
    """Split Markdown into {heading: body} pairs. Handles H1 and H2 levels."""
    sections: dict[str, str] = {}
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        heading_match = re.match(r"^#{1,2}\s+(.+)$", line)
        if heading_match:
            if current_heading is not None:
                sections[current_heading] = "\n".join(current_lines)
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections[current_heading] = "\n".join(current_lines)

    return sections


def _extract_fenced_block(text: str, language: str | None = None) -> str | None:
    """Extract the first fenced code block from text, optionally matching language."""
    if language:
        pattern = rf"```{re.escape(language)}\n(.*?)```"
    else:
        pattern = r"```\w*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).rstrip("\n")
    return None


def _parse_routines_section(text: str) -> list[UsedRoutine]:
    """Parse the Routines Used section into UsedRoutine objects."""
    routines: list[UsedRoutine] = []
    if not text.strip():
        return routines

    # Split on H3 headers: ### RoutineName (`routine_id`)
    parts = re.split(r"^###\s+", text, flags=re.MULTILINE)
    for part in parts:
        if not part.strip():
            continue
        # Parse header: "RoutineName (`routine_id`)"
        header_match = re.match(r"^(.+?)\s*\(`([^`]+)`\)", part)
        if not header_match:
            continue
        routine_name = header_match.group(1).strip()
        routine_id = header_match.group(2).strip()

        # Parse parameters from JSON code block
        param_list: list[UsedRoutineParameter] = []
        params_json = _extract_fenced_block(part, "json")
        if params_json:
            try:
                params_dict = json.loads(params_json)
                param_list = [UsedRoutineParameter(key=k, value=v) for k, v in params_dict.items()]
            except (json.JSONDecodeError, TypeError):
                pass

        routines.append(UsedRoutine(
            routine_id=routine_id,
            routine_name=routine_name,
            parameters=param_list,
        ))

    return routines
