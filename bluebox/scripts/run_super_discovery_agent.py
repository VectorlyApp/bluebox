"""
bluebox/scripts/run_super_discovery_agent.py

Run the SuperDiscoveryAgent orchestrator for routine discovery.

Usage:
    bluebox-super-discovery --task "Search for trains" --cdp-captures-dir ./cdp_captures

    bluebox-super-discovery --task "get the live standings of a premier league football season" \
        --network-jsonl ./cdp_captures/network/events.jsonl \
        --storage-jsonl ./cdp_captures/storage/events.jsonl \
        --window-props-jsonl ./cdp_captures/window_properties/events.jsonl

    bluebox-super-discovery --task "Search for flights" \
        --network-jsonl ./cdp_captures/network/events.jsonl \
        --storage-jsonl ./cdp_captures/storage/events.jsonl
"""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

from bluebox.agents.super_discovery_agent import SuperDiscoveryAgent
from bluebox.config import Config
from bluebox.data_models.llms.interaction import (
    EmittedMessage,
    ChatResponseEmittedMessage,
    ErrorEmittedMessage,
    ToolInvocationResultEmittedMessage,
)
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.js_data_loader import JSDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.utils.exceptions import ApiKeyNotFoundError
from bluebox.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    parser = ArgumentParser(description="Discover routines using the SuperDiscoveryAgent orchestrator.")
    parser.add_argument("--task", type=str, required=True, help="The task description.")

    # CDP captures directory (convenience option - auto-discovers JSONL files)
    parser.add_argument(
        "--cdp-captures-dir",
        type=str,
        default=None,
        help="Directory with CDP captures. Auto-discovers JSONL files within.",
    )

    # Individual JSONL file paths (explicit option)
    parser.add_argument("--network-jsonl", type=str, default=None, help="Path to network events JSONL file.")
    parser.add_argument("--storage-jsonl", type=str, default=None, help="Path to storage events JSONL file.")
    parser.add_argument("--window-props-jsonl", type=str, default=None, help="Path to window properties JSONL file.")
    parser.add_argument("--js-jsonl", type=str, default=None, help="Path to JavaScript events JSONL file.")

    # Output and model options
    parser.add_argument("--output-dir", type=str, default="./super_discovery_output", help="Output directory.")
    parser.add_argument("--llm-model", type=str, default="gpt-5.1", help="LLM model for orchestrator.")
    parser.add_argument("--subagent-llm-model", type=str, default=None, help="LLM model for subagents (defaults to --llm-model).")
    parser.add_argument("--remote-debugging-address", type=str, default=None, help="Chrome remote debugging address (e.g., http://127.0.0.1:9222).")
    parser.add_argument("--max-iterations", type=int, default=50, help="Max iterations for discovery loop.")
    args = parser.parse_args()

    if Config.OPENAI_API_KEY is None:
        logger.error("OPENAI_API_KEY is not set")
        raise ApiKeyNotFoundError("OPENAI_API_KEY is not set")

    logger.info("-" * 100)
    logger.info("Starting SuperDiscovery for task:\n%s", args.task)
    logger.info("-" * 100)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve JSONL paths - explicit paths take precedence over cdp-captures-dir
    network_jsonl = args.network_jsonl
    storage_jsonl = args.storage_jsonl
    window_props_jsonl = args.window_props_jsonl
    js_jsonl = args.js_jsonl

    if args.cdp_captures_dir:
        cdp_dir = Path(args.cdp_captures_dir)
        if not network_jsonl:
            candidate = cdp_dir / "network" / "events.jsonl"
            if candidate.exists():
                network_jsonl = str(candidate)
        if not storage_jsonl:
            candidate = cdp_dir / "storage" / "events.jsonl"
            if candidate.exists():
                storage_jsonl = str(candidate)
        if not window_props_jsonl:
            candidate = cdp_dir / "window_properties" / "events.jsonl"
            if candidate.exists():
                window_props_jsonl = str(candidate)
        if not js_jsonl:
            candidate = cdp_dir / "network" / "javascript_events.jsonl"
            if candidate.exists():
                js_jsonl = str(candidate)

    # Validate that we have at least network data
    if not network_jsonl:
        logger.error("No network data source provided. Use --network-jsonl or --cdp-captures-dir")
        raise ValueError("Network data is required for routine discovery")

    # Load data stores
    logger.info("Loading data stores...")

    network_data_store = NetworkDataStore(network_jsonl)
    logger.info("Network data loaded: %d transactions", network_data_store.stats.total_requests)

    storage_data_store: StorageDataStore | None = None
    if storage_jsonl and Path(storage_jsonl).exists():
        storage_data_store = StorageDataStore(storage_jsonl)
        logger.info("Storage data loaded: %d events", storage_data_store.stats.total_events)

    window_property_data_store: WindowPropertyDataStore | None = None
    if window_props_jsonl and Path(window_props_jsonl).exists():
        window_property_data_store = WindowPropertyDataStore(window_props_jsonl)
        logger.info("Window property data loaded: %d events", window_property_data_store.stats.total_events)

    js_data_store: JSDataStore | None = None
    if js_jsonl and Path(js_jsonl).exists():
        js_data_store = JSDataStore(js_jsonl)
        logger.info("JS data loaded: %d files", js_data_store.stats.total_files)

    # Initialize documentation data store with defaults from run_docs_digger.py
    BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_DOCS_DIR = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
    DEFAULT_CODE_PATHS = [
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
        str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
        str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
        "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
    ]

    documentation_data_store = DocumentationDataStore(
        documentation_paths=[DEFAULT_DOCS_DIR],
        code_paths=DEFAULT_CODE_PATHS,
    )
    logger.info("Documentation data loaded: %d docs, %d code files",
                documentation_data_store.stats.total_docs,
                documentation_data_store.stats.total_code)

    # Message history storage
    message_history: list[dict] = []
    message_history_path = os.path.join(args.output_dir, "message_history.json")

    # Message handler
    def handle_message(message: EmittedMessage) -> None:
        # Store message in history
        message_dict = {
            "type": message.__class__.__name__,
            "timestamp": message.timestamp if hasattr(message, "timestamp") else None,
        }

        if isinstance(message, ChatResponseEmittedMessage):
            logger.info("💬 %s", message.content[:200] + "..." if len(message.content) > 200 else message.content)
            message_dict["content"] = message.content
        elif isinstance(message, ErrorEmittedMessage):
            logger.error("❌ %s", message.error)
            message_dict["error"] = message.error
        elif isinstance(message, ToolInvocationResultEmittedMessage):
            tool_name = message.tool_invocation.tool_name
            status = message.tool_invocation.status.value
            logger.info("🔧 %s [%s]", tool_name, status)
            message_dict["tool_name"] = tool_name
            message_dict["status"] = status
            message_dict["result"] = message.tool_invocation.result if hasattr(message.tool_invocation, "result") else None

        message_history.append(message_dict)

        # Save message history after every message
        with open(message_history_path, mode="w", encoding="utf-8") as f:
            json.dump(message_history, f, ensure_ascii=False, indent=2)

    # Initialize agent
    llm_model = OpenAIModel(args.llm_model)
    subagent_model = OpenAIModel(args.subagent_llm_model) if args.subagent_llm_model else None

    agent = SuperDiscoveryAgent(
        emit_message_callable=handle_message,
        network_data_store=network_data_store,
        task=args.task,
        storage_data_store=storage_data_store,
        window_property_data_store=window_property_data_store,
        js_data_store=js_data_store,
        documentation_data_store=documentation_data_store,
        llm_model=llm_model,
        subagent_llm_model=subagent_model,
        max_iterations=args.max_iterations,
        remote_debugging_address=args.remote_debugging_address,
    )
    logger.info("SuperDiscoveryAgent initialized.")
    logger.info("📝 Message history will be saved to: %s", message_history_path)

    if args.remote_debugging_address:
        logger.info("Validation enabled via: %s", args.remote_debugging_address)

    logger.info("-" * 100)
    logger.info("Running SuperDiscoveryAgent...")
    logger.info("-" * 100)

    routine = agent.run()

    if routine:
        routine_path = os.path.join(args.output_dir, "routine.json")
        with open(routine_path, mode="w", encoding="utf-8") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info("🎉 Routine saved to: %s", routine_path)

        # Save test parameters
        test_params = {p.name: p.observed_value or "" for p in routine.parameters}
        test_params_path = os.path.join(args.output_dir, "test_parameters.json")
        with open(test_params_path, mode="w", encoding="utf-8") as f:
            json.dump(test_params, f, ensure_ascii=False, indent=2)
        logger.info("Test parameters saved to: %s", test_params_path)
    else:
        logger.error("❌ Discovery failed - no routine produced")

    logger.info("📝 Message history saved (%d messages): %s", len(message_history), message_history_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
