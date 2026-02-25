"""
tests/unit/agents/test_dumb_agent.py

Unit tests for DumbAgent minimal behavior.
"""

from unittest.mock import MagicMock

import pytest

from bluebox.agents.dumb_agent import DumbAgent


@pytest.fixture
def mock_emit() -> MagicMock:
    return MagicMock()


@pytest.fixture
def agent(mock_emit: MagicMock) -> DumbAgent:
    return DumbAgent(emit_message_callable=mock_emit)


class TestDumbAgent:
    def test_defaults_to_no_workspace_with_python_enabled(self, agent: DumbAgent) -> None:
        assert agent.has_workspace is False
        assert agent._allow_code_execution is True

    def test_registers_only_execute_python_by_default(self, agent: DumbAgent) -> None:
        agent._sync_tools()
        assert "execute_python" in agent._registered_tool_names
        assert "list_files" not in agent._registered_tool_names
        assert "read_file" not in agent._registered_tool_names
        assert "search_files" not in agent._registered_tool_names
        assert "add_note" not in agent._registered_tool_names
        assert "finalize_result" not in agent._registered_tool_names
        assert "finalize_failure" not in agent._registered_tool_names
        assert "finalize_with_output" not in agent._registered_tool_names
        assert "finalize_with_failure" not in agent._registered_tool_names

    def test_execute_python_runs_compute_without_workspace(self, agent: DumbAgent) -> None:
        result = agent._execute_tool("execute_python", {"code": "print(2 + 3)"})
        assert "error" not in result
        assert "5" in result.get("output", "")

    def test_execute_python_blocks_open_without_workspace(self, agent: DumbAgent) -> None:
        result = agent._execute_tool(
            "execute_python",
            {"code": "open('output/x.txt', 'w').write('x')"},
        )
        assert "error" in result
        assert "Blocked" in result["error"]
        assert "open" in result["error"].lower()
