"""
tests/unit/agents/test_bluebox_agent_execution_counter.py

Unit tests verifying that routine results are saved with a globally unique
sequential index via itertools.count on the agent instance.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from bluebox.agents.bluebox_agent import BlueBoxAgent
from bluebox.agents.workspace import LocalWorkspace


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agent(tmp_path: Path) -> BlueBoxAgent:
    """Create a BlueBoxAgent with a local workspace for testing."""
    return BlueBoxAgent(
        emit_message_callable=MagicMock(),
        workspace=LocalWorkspace(str(tmp_path)),
        auth_headers_provider=lambda: {"X-Service-Token": "test"},
    )


def _mock_response(json_data: dict) -> MagicMock:
    """Create a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.json.return_value = json_data
    resp.raise_for_status.return_value = None
    return resp


# =============================================================================
# Tests
# =============================================================================


class TestRoutineResultFilenameIndex:
    """Tests that saved routine result files contain a globally unique index."""

    @patch("bluebox.agents.bluebox_agent.requests.post")
    def test_single_execution_has_index_1(
        self, mock_post: MagicMock, agent: BlueBoxAgent, tmp_path: Path,
    ) -> None:
        """A single routine execution should save with index 1."""
        mock_post.return_value = _mock_response({"result": {"data": "ok"}})

        agent._execute_routines_in_parallel(routine_executions=[
            MagicMock(routine_id="r1", parameters={}),
        ])

        raw_files = list((tmp_path / "raw").glob("*routine_result_*.json"))
        assert len(raw_files) == 1
        assert "_1.json" in raw_files[0].name

    @patch("bluebox.agents.bluebox_agent.requests.post")
    def test_multiple_executions_have_sequential_indices(
        self, mock_post: MagicMock, agent: BlueBoxAgent, tmp_path: Path,
    ) -> None:
        """Three routines should produce files with indices 1, 2, 3."""
        mock_post.return_value = _mock_response({"result": {"data": "ok"}})

        agent._execute_routines_in_parallel(routine_executions=[
            MagicMock(routine_id="r1", parameters={}),
            MagicMock(routine_id="r2", parameters={}),
            MagicMock(routine_id="r3", parameters={}),
        ])

        raw_files = sorted((tmp_path / "raw").glob("*routine_result_*.json"))
        assert len(raw_files) == 3
        indices = sorted(int(f.stem.split("_")[-1]) for f in raw_files)
        assert indices == [1, 2, 3]

    @patch("bluebox.agents.bluebox_agent.requests.post")
    def test_indices_continue_across_batches(
        self, mock_post: MagicMock, agent: BlueBoxAgent, tmp_path: Path,
    ) -> None:
        """3 batches of 2 routines each should produce 6 files with indices 1-6."""
        mock_post.return_value = _mock_response({"result": {"data": "ok"}})

        for _ in range(3):
            agent._execute_routines_in_parallel(routine_executions=[
                MagicMock(routine_id="ra", parameters={}),
                MagicMock(routine_id="rb", parameters={}),
            ])

        raw_files = sorted((tmp_path / "raw").glob("*routine_result_*.json"))
        assert len(raw_files) == 6
        indices = sorted(int(f.stem.split("_")[-1]) for f in raw_files)
        assert indices == [1, 2, 3, 4, 5, 6]

    @patch("bluebox.agents.bluebox_agent.requests.post")
    def test_error_results_also_get_index(
        self, mock_post: MagicMock, agent: BlueBoxAgent, tmp_path: Path,
    ) -> None:
        """Even failed executions should save with a unique index."""
        mock_post.side_effect = requests.RequestException("timeout")

        agent._execute_routines_in_parallel(routine_executions=[
            MagicMock(routine_id="r1", parameters={}),
            MagicMock(routine_id="r2", parameters={}),
        ])

        raw_files = sorted((tmp_path / "raw").glob("*routine_result_*.json"))
        assert len(raw_files) == 2
        indices = sorted(int(f.stem.split("_")[-1]) for f in raw_files)
        assert indices == [1, 2]

        for f in raw_files:
            data = json.loads(f.read_text())
            assert "error" in data

    @patch("bluebox.agents.bluebox_agent.requests.post")
    def test_no_duplicate_files_on_parallel_execution(
        self, mock_post: MagicMock, agent: BlueBoxAgent, tmp_path: Path,
    ) -> None:
        """Parallel execution of many routines should never produce duplicate indices."""
        mock_post.return_value = _mock_response({"result": {"data": "ok"}})

        agent._execute_routines_in_parallel(routine_executions=[
            MagicMock(routine_id=f"r{i}", parameters={}) for i in range(10)
        ])

        raw_files = list((tmp_path / "raw").glob("*routine_result_*.json"))
        assert len(raw_files) == 10
        indices = sorted(int(f.stem.split("_")[-1]) for f in raw_files)
        assert indices == list(range(1, 11))
