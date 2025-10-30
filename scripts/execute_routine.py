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

import json
import argparse
from src.cdp.routine_execution import execute_routine
from src.data_models.production_routine import Routine



def main(routine_path: str, parameters_path: str | None = None, parameters_dict: dict | None = None):
    
    # ensure only one of parameters_path or parameters_dict is provided
    if parameters_path and parameters_dict:
        raise ValueError("Only one of --parameters-path or --parameters-dict must be provided")
    
    # Load routine data
    if parameters_path:
        parameters_dict = json.load(open(parameters_path))
    elif parameters_dict:
        parameters_dict = json.loads(parameters_dict)
    else:
        raise ValueError("Either --parameters-path or --parameters-dict must be provided")
        
    # load routine data
    routine_data = json.load(open(routine_path))
    routine = Routine(**routine_data)
    
    # Execute routine
    try:
        result = execute_routine(
            routine=routine,
            parameters_dict=parameters_dict,
            timeout=60.0,
            wait_after_navigate_sec=3.0,
            close_tab_when_done=False,
            incognito=True,
        )
        print(f"Result: {result}")
        
    except Exception as e:
        print("Error executing routine: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute a routine")
    parser.add_argument("--routine-path", type=str, required=True, help="Path to the routine JSON file")
    parser.add_argument("--parameters-path", type=str, required=False, help="Path to the parameters JSON file")
    parser.add_argument("--parameters-dict", type=str, required=False, help="Dictionary of parameters")
    args = parser.parse_args()
    main(args.routine_path, args.parameters_path, args.parameters_dict)
