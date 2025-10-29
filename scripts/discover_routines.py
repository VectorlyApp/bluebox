"""
Script for discovering routines from the network transactions.
"""
from argparse import ArgumentParser
from openai import OpenAI
from src.routine_discovery.agent import RoutineDiscoveryAgent
from src.routine_discovery.context_manager import ContextManager
from dotenv import load_dotenv
import os

def main() -> None:
    
    # parse arguments
    parser = ArgumentParser(description="Discover routines from the network transactions.")
    parser.add_argument("--task-description", type=str, required=True, help="The description of the task to discover routines for.")
    parser.add_argument("--cdp-captures-dir", type=str, default="./cdp_captures", help="The directory containing the CDP captures.")
    parser.add_argument("--output-dir", type=str, default="./routine_discovery_output", help="The directory to save the output to.")
    parser.add_argument("--llm-model", type=str, default="gpt-5", help="The LLM model to use.")
    args = parser.parse_args()
    
    # load environment variables
    load_dotenv()
    
    # ensure OpenAI API key is set
    if os.getenv("OPENAI_API_KEY") is None:
        raise ValueError("OPENAI_API_KEY is not set")
    
    
    print(f"\n{'-' * 100}")
    print(f"Starting routine discovery for task:\n{args.task_description}")
    print(f"{'-' * 100}\n")
    
    # initialize OpenAI client
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
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
    print(f"Context manager initialized.")
    
    # make the vectorstore
    context_manager.make_vectorstore()
    print(f"Vectorstore made: {context_manager.vectorstore_id}")
    
    # initialize routine discovery agent
    routine_discovery_agent = RoutineDiscoveryAgent(
        client=openai_client,
        context_manager=context_manager,
        task_description=args.task_description,
        llm_model=args.llm_model,
        debug_dir=os.path.join(args.output_dir, "llm_debug"),
    )
    print(f"Routine discovery agent initialized.")
    
    print(f"\n{'-' * 100}")
    print(f"Running routine discovery agent.")
    print(f"{'-' * 100}\n")
    
    # run the routine discovery agent
    routine_discovery_agent.run()
    print(f"Routine discovery agent run complete")



if __name__ == "__main__":
    main()
