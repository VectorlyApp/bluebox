"""
Script for discovering routines from the network transactions.
"""

from argparse import ArgumentParser
import logging
import os

from openai import OpenAI

from src.config import Config
from src.utils.exceptions import ApiKeyNotFoundError
from src.routine_discovery.agent import RoutineDiscoveryAgent
from src.routine_discovery.context_manager import ContextManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    
    # parse arguments
    parser = ArgumentParser(description="Discover routines from the network transactions.")
    parser.add_argument("--task", type=str, required=True, help="The description of the task to discover routines for.")
    parser.add_argument("--cdp-captures-dir", type=str, default="./cdp_captures", help="The directory containing the CDP captures.")
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="The directory to save the output to.")
    parser.add_argument("--llm-model", type=str, default="gpt-5", help="The LLM model to use.")
    args = parser.parse_args()

    # ensure OpenAI API key is set
    if Config.OPENAI_API_KEY is None:
        logger.error("OPENAI_API_KEY is not set")
        raise ApiKeyNotFoundError("OPENAI_API_KEY is not set")

    logger.info(f"\n{'-' * 100}")
    logger.info(f"Starting routine discovery for task:\n{args.task}")
    logger.info(f"{'-' * 100}\n")
    
    # initialize OpenAI client
    openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)
    
    # create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # initialize context manager
    context_manager = ContextManager(
        client=openai_client,
        tmp_dir=os.path.join(args.output_dir, "tmp"),
        transactions_dir=os.path.join(args.cdp_captures_dir, "network/transactions"),
        consolidated_transactions_path=os.path.join(args.cdp_captures_dir, "network/consolidated_transactions.json"),
        storage_jsonl_path=os.path.join(args.cdp_captures_dir, "storage/events.jsonl")
    )
    
    logger.info(f"Context manager initialized.")
    
    # make the vectorstore
    context_manager.make_vectorstore()
    logger.info(f"Vectorstore created: {context_manager.vectorstore_id}")
    
    # initialize routine discovery agent
    routine_discovery_agent = RoutineDiscoveryAgent(
        client=openai_client,
        context_manager=context_manager,
        task=args.task,
        llm_model=args.llm_model,
        output_dir=args.output_dir,
    )
    logger.info(f"Routine discovery agent initialized.")
    
    logger.info(f"\n{'-' * 100}")
    logger.info(f"Running routine discovery agent.")
    logger.info(f"{'-' * 100}\n")
    
    # run the routine discovery agent
    routine_discovery_agent.run()
    logger.info(f"Routine discovery agent run complete")


if __name__ == "__main__":
    main()
