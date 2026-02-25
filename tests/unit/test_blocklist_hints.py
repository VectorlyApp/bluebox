"""
tests/unit/test_blocklist_hints.py

Unit tests for blocklist workaround hints and sandbox-aware system prompts.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bluebox.agents.workspace import LocalWorkspace
from bluebox.utils.code_execution_sandbox import (
    BLOCKED_MODULE_WORKAROUNDS,
    BLOCKED_PATTERN_WORKAROUNDS,
    get_active_sandbox_mode,
    get_workaround_for_error,
)


class TestGetWorkaroundForError:
    """Tests for the get_workaround_for_error function."""

    def test_blocked_module_os(self) -> None:
        """Should return workaround for blocked 'os' module."""
        error = "Import of 'os' is blocked for security reasons"
        result = get_workaround_for_error(error)
        assert result is not None
        assert "open()" in result
        assert "Path" in result

    def test_blocked_module_pathlib(self) -> None:
        """Should return workaround for blocked 'pathlib' module."""
        error = "Import of 'pathlib' is blocked for security reasons"
        result = get_workaround_for_error(error)
        assert result is not None
        assert "pre-loaded" in result.lower()

    def test_blocked_submodule(self) -> None:
        """Should return workaround for blocked submodule like 'os.path'."""
        error = "Import of 'os.path' is blocked for security reasons"
        result = get_workaround_for_error(error)
        assert result is not None
        assert result == BLOCKED_MODULE_WORKAROUNDS["os"]

    def test_blocked_pattern_getattr(self) -> None:
        """Should return workaround for blocked getattr() pattern."""
        error = "Blocked: getattr() is not allowed - use dict access instead"
        result = get_workaround_for_error(error)
        assert result is not None
        assert "dict" in result.lower()

    def test_blocked_pattern_open(self) -> None:
        """Should return workaround for blocked open() pattern."""
        error = "Blocked: File operations (open) are not allowed"
        result = get_workaround_for_error(error)
        assert result is not None
        assert "open()" in result

    def test_blocked_pattern_eval(self) -> None:
        """Should return workaround for blocked eval() pattern."""
        error = "Blocked: eval() is not allowed"
        result = get_workaround_for_error(error)
        assert result is not None
        assert "directly" in result.lower()

    def test_unknown_error_returns_none(self) -> None:
        """Should return None for unrecognized error messages."""
        assert get_workaround_for_error("SomeRandomError: something broke") is None
        assert get_workaround_for_error("TypeError: unsupported operand") is None
        assert get_workaround_for_error("") is None

    def test_all_workaround_modules_are_blocked(self) -> None:
        """Every module in BLOCKED_MODULE_WORKAROUNDS should be in BLOCKED_MODULES."""
        from bluebox.utils.code_execution_sandbox import BLOCKED_MODULES
        for module_name in BLOCKED_MODULE_WORKAROUNDS:
            assert module_name in BLOCKED_MODULES, (
                f"Module '{module_name}' has a workaround but is not in BLOCKED_MODULES"
            )

    def test_all_workaround_patterns_are_blocked(self) -> None:
        """Every pattern in BLOCKED_PATTERN_WORKAROUNDS should be in BLOCKED_PATTERNS."""
        from bluebox.utils.code_execution_sandbox import BLOCKED_PATTERNS
        blocked_pattern_keys = {p for p, _ in BLOCKED_PATTERNS}
        for pattern in BLOCKED_PATTERN_WORKAROUNDS:
            assert pattern in blocked_pattern_keys, (
                f"Pattern '{pattern}' has a workaround but is not in BLOCKED_PATTERNS"
            )


class TestGetActiveSandboxMode:
    """Tests for the get_active_sandbox_mode function."""

    @patch("bluebox.utils.code_execution_sandbox.is_docker_available", return_value=False)
    @patch("bluebox.utils.code_execution_sandbox.SANDBOX_MODE", "auto")
    @patch("bluebox.utils.code_execution_sandbox._lambda_executor_fn", None)
    def test_auto_no_docker_returns_blocklist(self, mock_docker: MagicMock) -> None:
        """Auto mode with no Docker and no Lambda should return blocklist."""
        assert get_active_sandbox_mode() == "blocklist"

    @patch("bluebox.utils.code_execution_sandbox.is_docker_available", return_value=True)
    @patch("bluebox.utils.code_execution_sandbox.SANDBOX_MODE", "auto")
    @patch("bluebox.utils.code_execution_sandbox._lambda_executor_fn", None)
    def test_auto_with_docker_returns_docker(self, mock_docker: MagicMock) -> None:
        """Auto mode with Docker available should return docker."""
        assert get_active_sandbox_mode() == "docker"

    @patch("bluebox.utils.code_execution_sandbox.SANDBOX_MODE", "blocklist")
    def test_explicit_blocklist_mode(self) -> None:
        """Explicit blocklist mode should return blocklist regardless."""
        assert get_active_sandbox_mode() == "blocklist"

    @patch("bluebox.utils.code_execution_sandbox.SANDBOX_MODE", "docker")
    @patch("bluebox.utils.code_execution_sandbox.is_docker_available", return_value=False)
    def test_docker_mode_no_docker_returns_blocklist(self, mock_docker: MagicMock) -> None:
        """Docker mode with no Docker should fall back to blocklist."""
        assert get_active_sandbox_mode() == "blocklist"

    @patch("bluebox.utils.code_execution_sandbox.SANDBOX_MODE", "lambda")
    def test_lambda_mode(self) -> None:
        """Lambda mode should return lambda."""
        assert get_active_sandbox_mode() == "lambda"


class TestBlueBoxAgentBlocklistPrompt:
    """Tests for blocklist-aware system prompt in BlueBoxAgent."""

    @patch("bluebox.agents.abstract_agent.get_active_sandbox_mode", return_value="blocklist")
    @patch("bluebox.agents.bluebox_agent.Config")
    def test_blocklist_prompt_included(
        self,
        mock_config: MagicMock,
        mock_mode: MagicMock,
        tmp_path: Path,
    ) -> None:
        """System prompt should include sandbox restrictions in blocklist mode."""
        mock_config.VECTORLY_SERVICE_TOKEN = "test-token"
        mock_config.VECTORLY_API_BASE = "http://test"

        from bluebox.agents.bluebox_agent import BlueBoxAgent
        agent = BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace.from_directory_path(tmp_path),
        )
        prompt = agent._get_system_prompt()
        assert "Sandbox Restrictions" in prompt
        assert "Blocked imports" in prompt
        assert "do NOT import" in prompt

    @patch("bluebox.agents.abstract_agent.get_active_sandbox_mode", return_value="docker")
    @patch("bluebox.agents.bluebox_agent.Config")
    def test_blocklist_prompt_excluded(
        self,
        mock_config: MagicMock,
        mock_mode: MagicMock,
        tmp_path: Path,
    ) -> None:
        """System prompt should NOT include sandbox restrictions when Docker is available."""
        mock_config.VECTORLY_SERVICE_TOKEN = "test-token"
        mock_config.VECTORLY_API_BASE = "http://test"

        from bluebox.agents.bluebox_agent import BlueBoxAgent
        agent = BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace.from_directory_path(tmp_path),
        )
        prompt = agent._get_system_prompt()
        assert "Sandbox Restrictions" not in prompt


class TestRunPythonCodeBlockedHint:
    """Tests for workaround hints in _run_python_code error responses."""

    @patch("bluebox.agents.abstract_agent.get_active_sandbox_mode", return_value="blocklist")
    @patch("bluebox.agents.bluebox_agent.execute_python_sandboxed")
    @patch("bluebox.agents.bluebox_agent.Config")
    def test_blocked_module_returns_workaround_hint(
        self,
        mock_config: MagicMock,
        mock_sandbox: MagicMock,
        mock_mode: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When sandbox returns a blocked module error, _hint should contain workaround."""
        mock_config.VECTORLY_SERVICE_TOKEN = "test-token"
        mock_config.VECTORLY_API_BASE = "http://test"
        mock_sandbox.return_value = {
            "error": "Import of 'os' is blocked for security reasons",
            "output": "",
        }

        from bluebox.agents.bluebox_agent import BlueBoxAgent
        agent = BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace.from_directory_path(tmp_path),
        )
        result = agent._run_python_code("import os")
        kwargs = mock_sandbox.call_args.kwargs
        read_only_paths = kwargs["read_only_paths"]
        assert any(p.endswith("/raw") for p in read_only_paths)
        assert any(p.endswith("/meta") for p in read_only_paths)
        assert "Sandbox restriction:" in result["_hint"]
        assert "open()" in result["_hint"]

    @patch("bluebox.agents.abstract_agent.get_active_sandbox_mode", return_value="blocklist")
    @patch("bluebox.agents.bluebox_agent.execute_python_sandboxed")
    @patch("bluebox.agents.bluebox_agent.Config")
    def test_generic_error_returns_generic_hint(
        self,
        mock_config: MagicMock,
        mock_sandbox: MagicMock,
        mock_mode: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When sandbox returns a non-blocklist error, _hint should be generic."""
        mock_config.VECTORLY_SERVICE_TOKEN = "test-token"
        mock_config.VECTORLY_API_BASE = "http://test"
        mock_sandbox.return_value = {
            "error": "NameError: name 'foo' is not defined",
            "output": "",
        }

        from bluebox.agents.bluebox_agent import BlueBoxAgent
        agent = BlueBoxAgent(
            emit_message_callable=MagicMock(),
            workspace=LocalWorkspace.from_directory_path(tmp_path),
        )
        result = agent._run_python_code("print(foo)")
        kwargs = mock_sandbox.call_args.kwargs
        read_only_paths = kwargs["read_only_paths"]
        assert any(p.endswith("/raw") for p in read_only_paths)
        assert any(p.endswith("/meta") for p in read_only_paths)
        assert "Sandbox restriction:" not in result["_hint"]
        assert "Code failed" in result["_hint"]
