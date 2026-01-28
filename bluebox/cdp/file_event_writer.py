"""
bluebox/cdp/file_event_writer.py

File-based event writer callback for AsyncCDPSession.
Used by web-hacker CLI and SDK to write CDP events to disk.
"""

import json
from pathlib import Path
from typing import Any

from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class FileEventWriter:
    """
    Callback adapter that writes CDP events to files.

    Usage:
        writer = FileEventWriter(paths={
            "network_events_path": "./captures/network/events.jsonl",
            "storage_events_path": "./captures/storage/events.jsonl",
            "window_properties_path": "./captures/window_properties/events.jsonl",
        })
        session = AsyncCDPSession(
            ws_url=ws_url,
            session_start_dtm=datetime.now().isoformat(),
            event_callback_fn=writer.write_event,
            paths=writer.paths,
        )
    """

    # Map monitor category names to path keys
    CATEGORY_TO_PATH_KEY = {
        "AsyncNetworkMonitor": "network_events_path",
        "AsyncStorageMonitor": "storage_events_path",
        "AsyncWindowPropertyMonitor": "window_properties_path",
        "AsyncInteractionMonitor": "interaction_events_path",
    }

    def __init__(self, paths: dict[str, str]) -> None:
        """
        Initialize FileEventWriter.

        Args:
            paths: Dict with file paths. Expected keys:
                - 'network_events_path': Path for network events JSONL
                - 'storage_events_path': Path for storage events JSONL
                - 'window_properties_path': Path for window property events JSONL
                - 'interaction_events_path': Path for interaction events JSONL
                Additional keys are preserved and passed through to AsyncCDPSession.
        """
        self.paths = paths

        # Get specific paths (with defaults)
        self.network_events_path = Path(
            paths.get("network_events_path", "./network/events.jsonl")
        )
        self.storage_events_path = Path(
            paths.get("storage_events_path", "./storage/events.jsonl")
        )
        self.window_properties_path = Path(
            paths.get("window_properties_path", "./window_properties/events.jsonl")
        )
        self.interaction_events_path = Path(
            paths.get("interaction_events_path", "./interaction/events.jsonl")
        )

        # Ensure parent directories exist
        self.network_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.window_properties_path.parent.mkdir(parents=True, exist_ok=True)
        self.interaction_events_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("📁 FileEventWriter initialized")
        logger.info("   Network events: %s", self.network_events_path)
        logger.info("   Storage events: %s", self.storage_events_path)
        logger.info("   Window properties: %s", self.window_properties_path)
        logger.info("   Interaction events: %s", self.interaction_events_path)

    async def write_event(self, category: str, event: Any) -> None:
        """
        Async callback that writes events to appropriate files.

        Args:
            category: Event category (monitor class name, e.g., "AsyncNetworkMonitor").
            event: Event data (Pydantic model with .model_dump() or dict).
        """
        # Convert Pydantic model to dict if needed
        if hasattr(event, "model_dump"):
            event_dict = event.model_dump()
        elif isinstance(event, dict):
            event_dict = event
        else:
            event_dict = {"data": str(event)}

        # Determine output file based on category
        if category == "AsyncNetworkMonitor":
            output_path = self.network_events_path
        elif category == "AsyncStorageMonitor":
            output_path = self.storage_events_path
        elif category == "AsyncWindowPropertyMonitor":
            output_path = self.window_properties_path
        elif category == "AsyncInteractionMonitor":
            output_path = self.interaction_events_path
        else:
            # Unknown category - log warning but don't fail
            logger.warning("⚠️ Unknown event category: %s", category)
            return

        # Write to JSONL file (append mode)
        try:
            with open(output_path, mode="a", encoding="utf-8") as f:
                json_line = json.dumps(event_dict, ensure_ascii=False)
                f.write(json_line + "\n")
        except Exception as e:
            logger.error("❌ Failed to write event to %s: %s", output_path, e)

    @classmethod
    def create_from_output_dir(cls, output_dir: str | Path) -> "FileEventWriter":
        """
        Factory method to create FileEventWriter from an output directory.

        Creates standard subdirectory structure:
            output_dir/
            ├── network/
            │   ├── events.jsonl
            │   ├── consolidated_transactions.json
            │   └── network.har
            ├── storage/
            │   └── events.jsonl
            ├── window_properties/
            │   └── events.jsonl
            └── interaction/
                └── events.jsonl

        Args:
            output_dir: Base output directory path.

        Returns:
            Configured FileEventWriter instance.
        """
        output_dir = Path(output_dir)
        paths = {
            # Directories
            "output_dir": str(output_dir),
            "network_dir": str(output_dir / "network"),
            "storage_dir": str(output_dir / "storage"),
            "window_properties_dir": str(output_dir / "window_properties"),
            "interaction_dir": str(output_dir / "interaction"),
            # Event JSONL paths (written by this writer)
            "network_events_path": str(output_dir / "network" / "events.jsonl"),
            "storage_events_path": str(output_dir / "storage" / "events.jsonl"),
            "window_properties_path": str(output_dir / "window_properties" / "events.jsonl"),
            "interaction_events_path": str(output_dir / "interaction" / "events.jsonl"),
            # Consolidated output paths (written by finalize())
            "consolidated_transactions_json_path": str(
                output_dir / "network" / "consolidated_transactions.json"
            ),
            "network_har_path": str(output_dir / "network" / "network.har"),
            "summary_path": str(output_dir / "session_summary.json"),
        }
        return cls(paths=paths)
