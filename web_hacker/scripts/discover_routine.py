"""
web_hacker/scripts/discover_routine.py

Script for discovering routines from the network transactions.
"""

from argparse import ArgumentParser
import os
import json

from openai import OpenAI

from web_hacker.config import Config
from web_hacker.utils.exceptions import ApiKeyNotFoundError
from web_hacker.routine_discovery.agent import RoutineDiscoveryAgent
from web_hacker.routine_discovery.context_manager import LocalDiscoveryDataStore
from web_hacker.data_models.routine_discovery.message import (
    RoutineDiscoveryMessage,
    RoutineDiscoveryMessageType,
)
from web_hacker.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    # parse arguments
    parser = ArgumentParser(description="Discover routines from the network transactions.")
    parser.add_argument("--task", type=str, required=True, help="The description of the task to discover routines for.")
    parser.add_argument("--cdp-captures-dir", type=str, default="./cdp_captures", help="The directory containing the CDP captures.")
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="The directory to save the output to.")
    parser.add_argument("--llm-model", type=str, default="gpt-5.1", help="The LLM model to use.")
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
    context_manager = LocalDiscoveryDataStore(
        client=openai_client,
        tmp_dir=os.path.join(args.output_dir, "tmp"),
        transactions_dir=os.path.join(args.cdp_captures_dir, "network/transactions"),
        consolidated_transactions_path=os.path.join(args.cdp_captures_dir, "network/consolidated_transactions.json"),
        storage_jsonl_path=os.path.join(args.cdp_captures_dir, "storage/events.jsonl"),
        window_properties_path=os.path.join(args.cdp_captures_dir, "window_properties/window_properties.json"),
    )
    logger.info("Context manager initialized.")

    # make the vectorstore
    context_manager.make_vectorstore()
    logger.info("Vectorstore created: %s", context_manager.vectorstore_id)

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
    routine_discovery_agent = RoutineDiscoveryAgent(
        client=openai_client,
        context_manager=context_manager,
        task=args.task,
        emit_message_callable=handle_discovery_message,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
    )
    logger.info("Routine discovery agent initialized.")

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

        # get test parameters
        logger.info("Generating test parameters...")
        test_parameters = routine_discovery_agent.get_test_parameters(routine)
        test_parameters_dict = {param.name: param.value for param in test_parameters.parameters}

        # save test parameters
        test_params_path = os.path.join(args.output_dir, "test_parameters.json")
        with open(test_params_path, mode="w", encoding="utf-8") as f:
            json.dump(test_parameters_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Test parameters saved to: {test_params_path}")

    finally:
        # clean up the vectorstore
        logger.info("Cleaning up vectorstore...")
        context_manager.clean_up()
        logger.info("Vectorstore cleaned up.")


if __name__ == "__main__":
    main()
