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

from bluebox.data_models.agents.context import BlueBoxAgentContext, UsedRoutine, UsedRoutineParameter


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
            UsedRoutine.from_dict_params(
                routine_id="Routine_abc123",
                routine_name="AmtrakOneWaySearch",
                parameters={"origin": "New York", "destination": "Boston", "date": "2026-03-15"},
            ),
            UsedRoutine.from_dict_params(
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
        assert restored.routines_used[1].parameters_as_dict() == {"max_price": 100}
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
            assert rest.parameters_as_dict() == orig.parameters_as_dict()

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
# generate_context (structured output) tests
# =============================================================================


class TestGenerateContext:
    """Tests for the generate_context public method (structured output)."""

    def _make_agent(self, tmp_path: Path) -> Any:
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        from bluebox.agents.workspace import LocalWorkspace

        return BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace(str(tmp_path)),
            auth_headers_provider=lambda: {"X-Service-Token": "test"},
        )

    def _mock_llm_response(self, context: BlueBoxAgentContext) -> MagicMock:
        """Create a mock LLMChatResponse with parsed context."""
        response = MagicMock()
        response.parsed = context
        return response

    def test_tool_is_not_registered(self) -> None:
        """generate_context should NOT be an agent tool anymore."""
        from bluebox.agents.bluebox_agent import BlueBoxAgent
        tools = BlueBoxAgent._collect_tools()
        tool_names = [meta.name for meta, _ in tools]
        assert "generate_context" not in tool_names

    def test_saves_both_files(self, tmp_path: Path, sample_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path)
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(sample_context))

        result = agent.generate_context()

        assert result.goal == sample_context.goal
        assert result.summary == sample_context.summary

        # Verify both JSON and MD files were saved
        context_dir = tmp_path / "context"
        json_files = list(context_dir.glob("*.json"))
        md_files = list(context_dir.glob("*.md"))
        assert len(json_files) == 1
        assert len(md_files) == 1

        # Verify JSON is valid
        loaded = BlueBoxAgentContext.model_validate_json(json_files[0].read_text())
        assert loaded.goal == sample_context.goal

        # Verify MD has expected sections
        assert "## Goal" in md_files[0].read_text()

    def test_saves_to_context_subdirectory(self, tmp_path: Path, minimal_context: BlueBoxAgentContext) -> None:
        agent = self._make_agent(tmp_path)
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(minimal_context))

        agent.generate_context()

        context_dir = tmp_path / "context"
        assert context_dir.is_dir()
        assert len(list(context_dir.glob("*.json"))) == 1
        assert len(list(context_dir.glob("*.md"))) == 1

    def test_raises_on_none_parsed(self, tmp_path: Path) -> None:
        """Should raise ValueError when LLM returns None parsed result."""
        agent = self._make_agent(tmp_path)
        response = MagicMock()
        response.parsed = None
        agent.llm_client.call_sync = MagicMock(return_value=response)

        with pytest.raises(ValueError, match="failed to produce"):
            agent.generate_context()

    def test_auto_populates_routines_from_raw(self, tmp_path: Path) -> None:
        """When LLM returns empty routines_used, auto-populate from raw/."""
        agent = self._make_agent(tmp_path)

        # Write a fake routine result to raw/
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "result_1.json").write_text(json.dumps({
            "routine_id": "Routine_abc",
            "routine_name": "TestRoutine",
            "status": "completed",
            "parameters": {"city": "NYC"},
            "result": {"ok": True, "data": {}},
        }))

        # LLM returns context with empty routines_used
        context_from_llm = BlueBoxAgentContext(
            goal="test goal",
            summary="test summary",
            output_description="test output",
            routines_used=[],
        )
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(context_from_llm))

        result = agent.generate_context()

        assert len(result.routines_used) == 1
        assert result.routines_used[0].routine_id == "Routine_abc"
        assert result.routines_used[0].routine_name == "TestRoutine"
        assert result.routines_used[0].parameters_as_dict() == {"city": "NYC"}

    def test_auto_populate_deduplicates_routines(self, tmp_path: Path) -> None:
        """Same routine_id executed multiple times should appear once."""
        agent = self._make_agent(tmp_path)

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        for i in range(3):
            (raw_dir / f"result_{i}.json").write_text(json.dumps({
                "routine_id": "Routine_same",
                "routine_name": "SameRoutine",
                "status": "completed",
                "parameters": {"q": f"query_{i}"},
                "result": {"ok": True, "data": {}},
            }))

        context_from_llm = BlueBoxAgentContext(
            goal="test", summary="test", output_description="test",
            routines_used=[],
        )
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(context_from_llm))

        result = agent.generate_context()
        assert len(result.routines_used) == 1

    def test_llm_provided_routines_not_overridden(self, tmp_path: Path) -> None:
        """When LLM provides routines_used, don't auto-populate from raw/."""
        agent = self._make_agent(tmp_path)

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(exist_ok=True)
        (raw_dir / "result_1.json").write_text(json.dumps({
            "routine_id": "Routine_from_raw",
            "routine_name": "RawRoutine",
            "status": "completed",
            "parameters": {},
            "result": {"ok": True, "data": {}},
        }))

        context_from_llm = BlueBoxAgentContext(
            goal="test", summary="test", output_description="test",
            routines_used=[UsedRoutine.from_dict_params(
                routine_id="Routine_llm_provided",
                routine_name="LLMRoutine",
                parameters={"x": 1},
            )],
        )
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(context_from_llm))

        result = agent.generate_context()
        assert len(result.routines_used) == 1
        assert result.routines_used[0].routine_id == "Routine_llm_provided"

    def test_passes_focus_to_system_prompt(self, tmp_path: Path, minimal_context: BlueBoxAgentContext) -> None:
        """Focus text should be included in the system prompt sent to LLM."""
        agent = self._make_agent(tmp_path)
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(minimal_context))

        agent.generate_context(focus="focus on the flight search part")

        call_kwargs = agent.llm_client.call_sync.call_args
        system_prompt = call_kwargs.kwargs.get("system_prompt") or call_kwargs[1].get("system_prompt", "")
        assert "focus on the flight search part" in system_prompt

    def test_passes_response_model(self, tmp_path: Path, minimal_context: BlueBoxAgentContext) -> None:
        """Should call llm_client.call_sync with response_model=BlueBoxAgentContext."""
        agent = self._make_agent(tmp_path)
        agent.llm_client.call_sync = MagicMock(return_value=self._mock_llm_response(minimal_context))

        agent.generate_context()

        call_kwargs = agent.llm_client.call_sync.call_args
        assert call_kwargs.kwargs.get("response_model") is BlueBoxAgentContext
