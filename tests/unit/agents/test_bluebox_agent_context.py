"""
tests/unit/agents/test_bluebox_agent_context.py

Unit tests for BlueBoxAgentContext data model and context generation/loading
in BlueBoxAgent.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bluebox.data_models.agents.context import BlueBoxAgentContext, RoutineUsed


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_context() -> BlueBoxAgentContext:
    """A fully populated context for testing."""
    return BlueBoxAgentContext(
        version=1,
        goal="Find one-way train tickets from NYC to Boston on March 15, 2026",
        routines_used=[
            RoutineUsed(
                routine_id="Routine_abc123",
                routine_name="AmtrakOneWaySearch",
                parameters={"origin": "New York", "destination": "Boston", "date": "2026-03-15"},
            ),
            RoutineUsed(
                routine_id="Routine_def456",
                routine_name="AmtrakPriceFilter",
                parameters={"max_price": 100},
            ),
        ],
        python_code=(
            'import csv\n'
            'with open("outputs/trains.csv", "w") as f:\n'
            '    writer = csv.DictWriter(f, fieldnames=["departure", "price"])\n'
            '    writer.writeheader()\n'
            '    for rr in routine_results:\n'
            '        for train in rr["result"]["data"]["trains"]:\n'
            '            writer.writerow(train)\n'
            'print("Done")'
        ),
        output_files=["outputs/trains.csv"],
        output_description="CSV with columns: departure, price. 12 rows of Amtrak trains under $100.",
        summary="Searched Amtrak for NYC-Boston trains on March 15, filtered by price, and exported to CSV.",
        generated_at=datetime(2026, 2, 22, 10, 30, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def minimal_context() -> BlueBoxAgentContext:
    """A context with only required fields."""
    return BlueBoxAgentContext(
        goal="Search for flights",
        output_description="JSON with flight data",
        summary="Found flights.",
    )


# =============================================================================
# BlueBoxAgentContext model tests
# =============================================================================


class TestBlueBoxAgentContextModel:
    """Tests for the Pydantic model itself."""

    def test_json_roundtrip(self, sample_context: BlueBoxAgentContext) -> None:
        """Serialize to JSON and back, verify equality."""
        json_str = sample_context.model_dump_json(indent=2)
        restored = BlueBoxAgentContext.model_validate_json(json_str)
        assert restored.version == sample_context.version
        assert restored.goal == sample_context.goal
        assert restored.summary == sample_context.summary
        assert restored.output_description == sample_context.output_description
        assert restored.python_code == sample_context.python_code
        assert restored.output_files == sample_context.output_files
        assert len(restored.routines_used) == 2
        assert restored.routines_used[0].routine_id == "Routine_abc123"
        assert restored.routines_used[1].parameters == {"max_price": 100}
        assert isinstance(restored.generated_at, datetime)

    def test_version_defaults_to_1(self, minimal_context: BlueBoxAgentContext) -> None:
        assert minimal_context.version == 1

    def test_generated_at_defaults_to_now(self, minimal_context: BlueBoxAgentContext) -> None:
        assert isinstance(minimal_context.generated_at, datetime)
        # Should be recent (within last 10 seconds)
        delta = datetime.now(tz=timezone.utc) - minimal_context.generated_at
        assert delta.total_seconds() < 10

    def test_optional_fields_default(self, minimal_context: BlueBoxAgentContext) -> None:
        assert minimal_context.routines_used == []
        assert minimal_context.python_code is None
        assert minimal_context.output_files == []


# =============================================================================
# Markdown round-trip tests
# =============================================================================


class TestMarkdownRoundTrip:
    """Tests for to_markdown() and from_markdown()."""

    def test_to_markdown_has_expected_sections(self, sample_context: BlueBoxAgentContext) -> None:
        md = sample_context.to_markdown()
        assert "# BlueBox Agent Context" in md
        assert "## Goal" in md
        assert "## Summary" in md
        assert "## Routines Used" in md
        assert "## Python Code" in md
        assert "## Output Files" in md
        assert "## Output Description" in md
        assert "**Version:** 1" in md
        assert "**Generated:**" in md

    def test_to_markdown_contains_routine_details(self, sample_context: BlueBoxAgentContext) -> None:
        md = sample_context.to_markdown()
        assert "AmtrakOneWaySearch" in md
        assert "Routine_abc123" in md
        assert '"origin": "New York"' in md

    def test_to_markdown_contains_python_code(self, sample_context: BlueBoxAgentContext) -> None:
        md = sample_context.to_markdown()
        assert "```python" in md
        assert "csv.DictWriter" in md

    def test_from_markdown_roundtrip(self, sample_context: BlueBoxAgentContext) -> None:
        """from_markdown(to_markdown(ctx)) should produce an equivalent model."""
        md = sample_context.to_markdown()
        restored = BlueBoxAgentContext.from_markdown(md)
        assert restored.version == sample_context.version
        assert restored.goal == sample_context.goal
        assert restored.summary == sample_context.summary
        assert restored.output_description == sample_context.output_description
        assert restored.python_code == sample_context.python_code
        assert restored.output_files == sample_context.output_files
        assert len(restored.routines_used) == len(sample_context.routines_used)
        for orig, rest in zip(sample_context.routines_used, restored.routines_used):
            assert rest.routine_id == orig.routine_id
            assert rest.routine_name == orig.routine_name
            assert rest.parameters == orig.parameters

    def test_from_markdown_no_python_code(self, minimal_context: BlueBoxAgentContext) -> None:
        """Markdown with no Python Code section should parse python_code as None."""
        md = minimal_context.to_markdown()
        assert "## Python Code" not in md
        restored = BlueBoxAgentContext.from_markdown(md)
        assert restored.python_code is None

    def test_from_markdown_no_routines(self, minimal_context: BlueBoxAgentContext) -> None:
        md = minimal_context.to_markdown()
        assert "## Routines Used" not in md
        restored = BlueBoxAgentContext.from_markdown(md)
        assert restored.routines_used == []

    def test_from_markdown_no_output_files(self, minimal_context: BlueBoxAgentContext) -> None:
        md = minimal_context.to_markdown()
        restored = BlueBoxAgentContext.from_markdown(md)
        assert restored.output_files == []


# =============================================================================
# Context loading tests (BlueBoxAgent integration)
# =============================================================================


class TestContextLoading:
    """Tests for context file loading in BlueBoxAgent."""

    def _make_agent(
        self,
        workspace_dir: Path,
        context_file: str | None = None,
    ) -> Any:
        """Create a BlueBoxAgent with mocked dependencies."""
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        from bluebox.agents.workspace import LocalWorkspace

        return BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace(str(workspace_dir)),
            auth_headers_provider=lambda: {"X-Service-Token": "test"},
            context_file=context_file,
        )

    def test_loads_json_context_file(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        ctx_file = tmp_path / "my_context.json"
        ctx_file.write_text(sample_context.model_dump_json(indent=2))

        agent = self._make_agent(tmp_path, context_file=str(ctx_file))
        assert agent._agent_context is not None
        assert agent._agent_context.goal == sample_context.goal

    def test_loads_markdown_context_file(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        ctx_file = tmp_path / "my_context.md"
        ctx_file.write_text(sample_context.to_markdown())

        agent = self._make_agent(tmp_path, context_file=str(ctx_file))
        assert agent._agent_context is not None
        assert agent._agent_context.goal == sample_context.goal

    def test_workspace_relative_path(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        ctx_file = context_dir / "my_context.json"
        ctx_file.write_text(sample_context.model_dump_json(indent=2))

        agent = self._make_agent(tmp_path, context_file="context/my_context.json")
        assert agent._agent_context is not None
        assert agent._agent_context.goal == sample_context.goal

    def test_auto_discovers_from_workspace(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        ctx_file = context_dir / "agent_context.json"
        ctx_file.write_text(sample_context.model_dump_json(indent=2))

        agent = self._make_agent(tmp_path)
        assert agent._agent_context is not None
        assert agent._agent_context.goal == sample_context.goal

    def test_auto_discovers_most_recent(self, tmp_path: Path) -> None:
        """When multiple context files exist, loads the most recently modified."""
        import time

        context_dir = tmp_path / "context"
        context_dir.mkdir()

        old = BlueBoxAgentContext(goal="old goal", output_description="old", summary="old")
        (context_dir / "old.json").write_text(old.model_dump_json())
        time.sleep(0.05)  # ensure mtime differs

        new = BlueBoxAgentContext(goal="new goal", output_description="new", summary="new")
        (context_dir / "new.json").write_text(new.model_dump_json())

        agent = self._make_agent(tmp_path)
        assert agent._agent_context is not None
        assert agent._agent_context.goal == "new goal"

    def test_explicit_context_file_overrides_auto_discovery(
        self, tmp_path: Path, sample_context: BlueBoxAgentContext,
    ) -> None:
        # Put one context in workspace
        context_dir = tmp_path / "context"
        context_dir.mkdir()
        auto_ctx = BlueBoxAgentContext(goal="auto goal", output_description="auto", summary="auto")
        (context_dir / "auto.json").write_text(auto_ctx.model_dump_json())

        # Put explicit context elsewhere
        explicit_file = tmp_path / "explicit.json"
        explicit_file.write_text(sample_context.model_dump_json(indent=2))

        agent = self._make_agent(tmp_path, context_file=str(explicit_file))
        assert agent._agent_context is not None
        assert agent._agent_context.goal == sample_context.goal

    def test_invalid_context_file_ignored(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path, context_file="/nonexistent/path.json")
        assert agent._agent_context is None

    def test_malformed_json_ignored(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json!!!")
        agent = self._make_agent(tmp_path, context_file=str(bad_file))
        assert agent._agent_context is None

    def test_no_context_dir_no_error(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        assert agent._agent_context is None


# =============================================================================
# System prompt injection tests
# =============================================================================


class TestContextPromptInjection:
    """Tests for _get_context_prompt_section and system prompt integration."""

    def _make_agent(self, tmp_path: Path, context: BlueBoxAgentContext) -> Any:
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        from bluebox.agents.workspace import LocalWorkspace

        ctx_file = tmp_path / "context.json"
        ctx_file.write_text(context.model_dump_json(indent=2))

        return BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace(str(tmp_path)),
            auth_headers_provider=lambda: {"X-Service-Token": "test"},
            context_file=str(ctx_file),
        )

    def test_context_section_in_system_prompt(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path, sample_context)
        prompt = agent._get_system_prompt()
        assert "## Prior Context" in prompt
        assert sample_context.goal in prompt
        assert sample_context.summary in prompt
        assert "Routine_abc123" in prompt
        assert "AmtrakOneWaySearch" in prompt

    def test_context_section_includes_python_code(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path, sample_context)
        prompt = agent._get_system_prompt()
        assert "```python" in prompt
        assert "csv.DictWriter" in prompt

    def test_context_section_truncation(self, tmp_path: Path) -> None:
        """Context over 20K chars gets truncated with a hint."""
        big_context = BlueBoxAgentContext(
            goal="x" * 25_000,
            output_description="desc",
            summary="summary",
        )
        agent = self._make_agent(tmp_path, big_context)
        section = agent._get_context_prompt_section()
        assert len(section) < 25_000
        assert "context truncated" in section
        assert "read_workspace_file" in section

    def test_no_context_no_section(self, tmp_path: Path) -> None:
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        from bluebox.agents.workspace import LocalWorkspace

        agent = BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace(str(tmp_path)),
            auth_headers_provider=lambda: {"X-Service-Token": "test"},
        )
        prompt = agent._get_system_prompt()
        assert "## Prior Context" not in prompt


# =============================================================================
# generate_context tool tests
# =============================================================================


class TestGenerateContextTool:
    """Tests for the _generate_context agent tool."""

    def _make_agent(self, tmp_path: Path) -> Any:
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        from bluebox.agents.workspace import LocalWorkspace

        return BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace(str(tmp_path)),
            auth_headers_provider=lambda: {"X-Service-Token": "test"},
        )

    def test_tool_is_registered(self) -> None:
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        tools = BlueBoxAgent._collect_tools()
        tool_names = [meta.name for meta, _ in tools]
        assert "generate_context" in tool_names

    def test_saves_both_files(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path)
        result = agent._generate_context(
            goal=sample_context.goal,
            summary=sample_context.summary,
            output_description=sample_context.output_description,
            routines_used=[r.model_dump() for r in sample_context.routines_used],
            python_code=sample_context.python_code,
            output_files=sample_context.output_files,
        )

        assert result["success"] is True
        assert result["context_json"] is not None
        assert result["context_md"] is not None

        # Verify JSON file exists and is valid
        json_path = tmp_path / result["context_json"]
        assert json_path.is_file()
        loaded = BlueBoxAgentContext.model_validate_json(json_path.read_text())
        assert loaded.goal == sample_context.goal

        # Verify MD file exists
        md_path = tmp_path / result["context_md"]
        assert md_path.is_file()
        assert "## Goal" in md_path.read_text()

    def test_saves_to_context_subdirectory(self, tmp_path: Path, minimal_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path)
        result = agent._generate_context(
            goal=minimal_context.goal,
            summary=minimal_context.summary,
            output_description=minimal_context.output_description,
        )
        assert "context/" in result["context_json"]
        assert "context/" in result["context_md"]

    def test_validates_bad_routines_used(self, tmp_path: Path) -> None:
        agent = self._make_agent(tmp_path)
        result = agent._generate_context(
            goal="test",
            summary="test",
            output_description="test",
            routines_used=[{"bad_key": "missing routine_id"}],
        )
        assert "error" in result
