"""
Python script to execute a routine.

Example commands:

    # Execute routine with parameters from a file
    python scripts/execute_routine.py \
        --routine-path routine_discovery_output/routine.json \
        --parameters-path routine_discovery_output/test_parameters.json

    # Execute routine with parameters from a dictionary
    python scripts/execute_routine.py \
        --routine-path example_routines/amtrak_one_way_train_search_routine.json \
        --parameters-dict "{'origin': 'boston', 'destination': 'new york', 'departureDate': '2026-03-22'}"
"""

import argparse
import json
import logging

from web_hacker.config import Config
from web_hacker.cdp.routine_execution import execute_routine
from web_hacker.data_models.production_routine import Routine

logging.basicConfig(level=Config.LOG_LEVEL, format=Config.LOG_FORMAT, datefmt=Config.LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


def main(routine_path: str | None = None, parameters_path: str | None = None, parameters_dict: str | None = None):
    """
    Main function for executing a routine.
    Can be called with arguments (for direct execution) or without (for CLI entry point).
    """
    # If called as CLI entry point, parse arguments
    if routine_path is None:
        parser = argparse.ArgumentParser(description="Execute a routine")
        parser.add_argument("--routine-path", type=str, required=True, help="Path to the routine JSON file")
        parser.add_argument("--parameters-path", type=str, required=False, help="Path to the parameters JSON file")
        parser.add_argument("--parameters-dict", type=str, required=False, help="Dictionary of parameters")
        args = parser.parse_args()
        routine_path = args.routine_path
        parameters_path = args.parameters_path
        parameters_dict = args.parameters_dict
    
    # ensure only one of parameters_path or parameters_dict is provided
    if parameters_path and parameters_dict:
        raise ValueError("Only one of --parameters-path or --parameters-dict must be provided")
    
    # Load routine data
    if parameters_path:
        parameters_dict_parsed = json.load(open(parameters_path))
    elif parameters_dict:
        parameters_dict_parsed = json.loads(parameters_dict)
    else:
        raise ValueError("Either --parameters-path or --parameters-dict must be provided")
        
    # load routine data
    routine_data = json.load(open(routine_path))
    routine = Routine(**routine_data)
    
    # Execute routine
    try:
        result = execute_routine(
            routine=routine,
            parameters_dict=parameters_dict_parsed,
            timeout=60.0,
            wait_after_navigate_sec=3.0,
            close_tab_when_done=False,
            incognito=True,
        )
        logger.info(f"Result: {result}")
        
    except Exception as e:
        logger.error("Error executing routine: %s", e)


if __name__ == "__main__":
    main()
