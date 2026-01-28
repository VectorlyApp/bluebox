"""
tests/unit/cdp/test_file_event_writer.py

Tests for FileEventWriter.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from bluebox.cdp.file_event_writer import FileEventWriter


class TestFileEventWriterInit:
    """
    Tests for FileEventWriter initialization.
    """

    def test_init_creates_directories(self, tmp_path: Path) -> None:
        """Parent directories should be created on init."""
        network_path = tmp_path / "subdir1" / "network" / "events.jsonl"
        storage_path = tmp_path / "subdir2" / "storage" / "events.jsonl"
        window_path = tmp_path / "subdir3" / "window" / "events.jsonl"
        interaction_path = tmp_path / "subdir4" / "interaction" / "events.jsonl"

        paths = {
            "network_events_path": str(network_path),
            "storage_events_path": str(storage_path),
            "window_properties_path": str(window_path),
            "interaction_events_path": str(interaction_path),
        }

        FileEventWriter(paths=paths)

        # directories should exist
        assert network_path.parent.exists()
        assert storage_path.parent.exists()
        assert window_path.parent.exists()
        assert interaction_path.parent.exists()


class TestFileEventWriterWriteEvent:
    """
    Tests for FileEventWriter.write_event method.
    """

    @pytest.mark.asyncio
    async def test_write_event_network(self, tmp_path: Path) -> None:
        """Writes to network events file."""
        network_path = tmp_path / "network" / "events.jsonl"
        paths = {
            "network_events_path": str(network_path),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("AsyncNetworkMonitor", {"url": "https://example.com"})

        assert network_path.exists()
        content = network_path.read_text()
        data = json.loads(content.strip())
        assert data["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_write_event_storage(self, tmp_path: Path) -> None:
        """Writes to storage events file."""
        storage_path = tmp_path / "storage" / "events.jsonl"
        paths = {
            "network_events_path": str(tmp_path / "network" / "events.jsonl"),
            "storage_events_path": str(storage_path),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("AsyncStorageMonitor", {"type": "cookieChange"})

        assert storage_path.exists()
        content = storage_path.read_text()
        data = json.loads(content.strip())
        assert data["type"] == "cookieChange"

    @pytest.mark.asyncio
    async def test_write_event_window_properties(self, tmp_path: Path) -> None:
        """Writes to window properties file."""
        window_path = tmp_path / "window" / "events.jsonl"
        paths = {
            "network_events_path": str(tmp_path / "network" / "events.jsonl"),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(window_path),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("AsyncWindowPropertyMonitor", {"changes": []})

        assert window_path.exists()
        content = window_path.read_text()
        data = json.loads(content.strip())
        assert data["changes"] == []

    @pytest.mark.asyncio
    async def test_write_event_interaction(self, tmp_path: Path) -> None:
        """Writes to interaction events file."""
        interaction_path = tmp_path / "interaction" / "events.jsonl"
        paths = {
            "network_events_path": str(tmp_path / "network" / "events.jsonl"),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(interaction_path),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("AsyncInteractionMonitor", {"type": "click"})

        assert interaction_path.exists()
        content = interaction_path.read_text()
        data = json.loads(content.strip())
        assert data["type"] == "click"

    @pytest.mark.asyncio
    async def test_write_event_unknown_category(self, tmp_path: Path, caplog) -> None:
        """Unknown category logs warning and doesn't write."""
        paths = {
            "network_events_path": str(tmp_path / "network" / "events.jsonl"),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("UnknownMonitor", {"data": "test"})

        # none of the files should be created (except directories)
        assert not (tmp_path / "network" / "events.jsonl").exists()
        assert not (tmp_path / "storage" / "events.jsonl").exists()
        assert not (tmp_path / "window" / "events.jsonl").exists()
        assert not (tmp_path / "interaction" / "events.jsonl").exists()

    @pytest.mark.asyncio
    async def test_write_event_pydantic_model(self, tmp_path: Path) -> None:
        """Pydantic model calls model_dump() if available."""
        network_path = tmp_path / "network" / "events.jsonl"
        paths = {
            "network_events_path": str(network_path),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        # mock pydantic model
        mock_model = MagicMock()
        mock_model.model_dump.return_value = {"field": "value", "nested": {"a": 1}}

        await writer.write_event("AsyncNetworkMonitor", mock_model)

        mock_model.model_dump.assert_called_once()
        assert network_path.exists()
        content = network_path.read_text()
        data = json.loads(content.strip())
        assert data["field"] == "value"
        assert data["nested"]["a"] == 1

    @pytest.mark.asyncio
    async def test_write_event_appends(self, tmp_path: Path) -> None:
        """Multiple writes append to the same file."""
        network_path = tmp_path / "network" / "events.jsonl"
        paths = {
            "network_events_path": str(network_path),
            "storage_events_path": str(tmp_path / "storage" / "events.jsonl"),
            "window_properties_path": str(tmp_path / "window" / "events.jsonl"),
            "interaction_events_path": str(tmp_path / "interaction" / "events.jsonl"),
        }
        writer = FileEventWriter(paths=paths)

        await writer.write_event("AsyncNetworkMonitor", {"id": 1})
        await writer.write_event("AsyncNetworkMonitor", {"id": 2})
        await writer.write_event("AsyncNetworkMonitor", {"id": 3})

        lines = network_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[1])["id"] == 2
        assert json.loads(lines[2])["id"] == 3


class TestFileEventWriterFactory:
    """
    Tests for FileEventWriter.create_from_output_dir factory method.
    """

    def test_create_from_output_dir(self, tmp_path: Path) -> None:
        """Factory creates correct paths structure."""
        output_dir = tmp_path / "captures"
        writer = FileEventWriter.create_from_output_dir(output_dir)

        # check paths are set correctly
        assert writer.paths["output_dir"] == str(output_dir)
        assert writer.paths["network_dir"] == str(output_dir / "network")
        assert writer.paths["storage_dir"] == str(output_dir / "storage")
        assert writer.paths["window_properties_dir"] == str(output_dir / "window_properties")
        assert writer.paths["interaction_dir"] == str(output_dir / "interaction")

        # check event paths
        assert writer.paths["network_events_path"] == str(output_dir / "network" / "events.jsonl")
        assert writer.paths["storage_events_path"] == str(output_dir / "storage" / "events.jsonl")
        assert writer.paths["window_properties_path"] == str(
            output_dir / "window_properties" / "events.jsonl"
        )
        assert writer.paths["interaction_events_path"] == str(
            output_dir / "interaction" / "events.jsonl"
        )

        # check consolidated output paths
        assert writer.paths["consolidated_transactions_json_path"] == str(
            output_dir / "network" / "consolidated_transactions.json"
        )
        assert writer.paths["network_har_path"] == str(output_dir / "network" / "network.har")
        assert writer.paths["summary_path"] == str(output_dir / "session_summary.json")

        # directories should be created
        assert (output_dir / "network").exists()
        assert (output_dir / "storage").exists()
        assert (output_dir / "window_properties").exists()
        assert (output_dir / "interaction").exists()


class TestFileEventWriterCategoryMapping:
    """
    Tests for FileEventWriter category to path mapping.
    """

    def test_category_to_path_key_mapping(self) -> None:
        """Verify CATEGORY_TO_PATH_KEY mappings are correct."""
        mapping = FileEventWriter.CATEGORY_TO_PATH_KEY

        assert mapping["AsyncNetworkMonitor"] == "network_events_path"
        assert mapping["AsyncStorageMonitor"] == "storage_events_path"
        assert mapping["AsyncWindowPropertyMonitor"] == "window_properties_path"
        assert mapping["AsyncInteractionMonitor"] == "interaction_events_path"
        assert len(mapping) == 4
