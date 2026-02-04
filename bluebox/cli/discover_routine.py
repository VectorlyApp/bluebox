"""
bluebox/cli/discover_routine.py

Script for discovering routines from the network transactions.
"""

from argparse import ArgumentParser
import os
import json
from pathlib import Path

from openai import OpenAI

# Package root for code_paths (bluebox/cli/ -> bluebox/)
PACKAGE_ROOT = Path(__file__).resolve().parent.parent

from bluebox.config import Config
from bluebox.utils.exceptions import ApiKeyNotFoundError
from bluebox.agents.routine_discovery_agent import RoutineDiscoveryAgent
from bluebox.llms.infra.data_store import LocalDiscoveryDataStore
from bluebox.llms.llm_client import LLMClient
from bluebox.data_models.llms.vendors import OpenAIModel
from bluebox.data_models.routine_discovery.message import (
    RoutineDiscoveryMessage,
    RoutineDiscoveryMessageType,
)
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    # parse arguments
    parser = ArgumentParser(description="Discover routines from the network transactions.")
    parser.add_argument("--task", type=str, required=True, help="The description of the task to discover routines for.")
    parser.add_argument("--cdp-captures-dir", type=str, default="./cdp_captures", help="The directory containing the CDP captures.")
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="The directory to save the output to.")
    parser.add_argument("--llm-model", type=str, default="gpt-5.1", help="The LLM model to use.")
    parser.add_argument("--remote-debugging-address", type=str, default=None, help="Chrome remote debugging address for routine validation (e.g., http://127.0.0.1:9222). If provided, the agent will execute the routine to validate it works.")
    args = parser.parse_args()

    # ensure OpenAI API key is set
    if Config.OPENAI_API_KEY is None:
        logger.error("OPENAI_API_KEY is not set")
        raise ApiKeyNotFoundError("OPENAI_API_KEY is not set")

    logger.info(f"\n{'-' * 100}")
    logger.info("Starting routine discovery for task:\n%s", args.task)
    logger.info(f"\n{'-' * 100}\n")

    # initialize OpenAI client
    openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # initialize context manager
    data_store = LocalDiscoveryDataStore(
        client=openai_client,
        cdp_captures_dir=args.cdp_captures_dir,
        tmp_dir=os.path.join(args.output_dir, "tmp"),
        documentation_paths=[str(PACKAGE_ROOT / "agent_docs")],
        code_paths=[
            str(PACKAGE_ROOT / "data_models" / "routine"),
            str(PACKAGE_ROOT / "data_models" / "ui_elements.py"),
            str(PACKAGE_ROOT / "utils" / "js_utils.py"),
            str(PACKAGE_ROOT / "utils" / "data_utils.py"),
            "!" + str(PACKAGE_ROOT / "**" / "__init__.py"),
        ],
    )
    logger.info("Data store initialized.")

    # make the vectorstores
    data_store.make_cdp_captures_vectorstore()
    logger.info("CDP captures vectorstore created: %s", data_store.cdp_captures_vectorstore_id)

    data_store.make_documentation_vectorstore()
    logger.info("Documentation vectorstore created: %s", data_store.documentation_vectorstore_id)

    # define message handler for progress updates
    def handle_discovery_message(message: RoutineDiscoveryMessage) -> None:
        """Handle routine discovery progress messages."""
        if message.type == RoutineDiscoveryMessageType.INITIATED:
            logger.info(f"🚀 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.PROGRESS_THINKING:
            logger.info(f"🤔 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.PROGRESS_RESULT:
            logger.info(f"✅ {message.content}")
        elif message.type == RoutineDiscoveryMessageType.FINISHED:
            logger.info(f"🎉 {message.content}")
        elif message.type == RoutineDiscoveryMessageType.ERROR:
            logger.error(f"❌ {message.content}")

    # initialize routine discovery agent
    llm_client = LLMClient(llm_model=OpenAIModel(args.llm_model))
    routine_discovery_agent = RoutineDiscoveryAgent(
        llm_client=llm_client,
        data_store=data_store,
        task=args.task,
        emit_message_callable=handle_discovery_message,
        output_dir=args.output_dir,
        remote_debugging_address=args.remote_debugging_address,
    )
    logger.info("Routine discovery agent initialized.")
    if args.remote_debugging_address:
        logger.info(f"Routine validation enabled via: {args.remote_debugging_address}")

    logger.info(f"\n{'-' * 100}")
    logger.info("Running routine discovery agent.")
    logger.info(f"\n{'-' * 100}\n")

    # run the routine discovery agent and get the routine
    try:
        routine = routine_discovery_agent.run()
        logger.info("Routine discovery agent run complete.")

        # save the final routine to output
        routine_path = os.path.join(args.output_dir, "routine.json")
        with open(routine_path, mode="w", encoding="utf-8") as f:
            json.dump(routine.model_dump(), f, ensure_ascii=False, indent=2)
        logger.info(f"Production routine saved to: {routine_path}")

        # Extract test parameters directly from routine parameters
        logger.info("Extracting test parameters...")
        test_parameters_dict = {param.name: param.observed_value or "" for param in routine.parameters}

        # save test parameters
        test_params_path = os.path.join(args.output_dir, "test_parameters.json")
        with open(test_params_path, mode="w", encoding="utf-8") as f:
            json.dump(test_parameters_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Test parameters saved to: {test_params_path}")

    finally:
        # clean up the vectorstore
        logger.info("Cleaning up vectorstore...")
        data_store.clean_up()
        logger.info("Vectorstore cleaned up.")


if __name__ == "__main__":
    main()
