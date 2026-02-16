"""
Execute a routine from the command line.

Usage:
    bluebox-execute --routine-path <path> --parameters-path <path> [--output <path>] [--download-dir <dir>] [--keep-open]
    bluebox-execute --routine-path <path> --parameters-dict '<json>' [--output <path>] [--download-dir <dir>] [--keep-open]

Examples:
    bluebox-execute --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json --parameters-path example_data/example_routines/amtrak_one_way_train_search_input.json
    bluebox-execute --routine-path example_data/example_routines/amtrak_one_way_train_search_routine.json --parameters-dict '{"origin": "boston", "destination": "new york", "departureDate": "2026-03-22"}' --output output.json --keep-open
"""

import argparse
import json
import os

from bluebox.data_models.routine.execution import RoutineExecutionResult
from bluebox.data_models.routine.routine import Routine
from bluebox.utils.data_utils import save_data_to_file, write_json_file
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def save_result(
    result: RoutineExecutionResult,
    output_path: str | None = None,
    download_dir: str | None = None,
) -> None:
    """
    Save execution result to files.
    
    Args:
        result: The routine execution result.
        output_path: Path to save the full RoutineExecutionResult as JSON.
        download_dir: Directory to save data file when result.filename is present.
    """
    # Save data to filename if present
    if result.filename:
        if download_dir and not os.path.isabs(result.filename):
            file_path = os.path.join(download_dir, result.filename)
        else:
            file_path = result.filename
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        save_data_to_file(result.data, file_path, is_base64=result.is_base64)
    
    # Save full result as JSON if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        write_json_file(output_path, result.model_dump())
        logger.info(f"Saved RoutineExecutionResult to: {output_path}")


def main(
    routine_path: str | None = None,
    parameters_path: str | None = None,
    parameters_dict: str | None = None,
    output: str | None = None,
    download_dir: str | None = None,
    keep_open: bool = False,
    remote_debugging_address: str = "http://127.0.0.1:9222",
    incognito: bool = True,
) -> None:
    """Execute a routine with given parameters."""
    # Parse CLI arguments if not called programmatically
    if routine_path is None:
        parser = argparse.ArgumentParser(description="Execute a routine")
        parser.add_argument("--routine-path", type=str, required=True)
        parser.add_argument("--parameters-path", type=str)
        parser.add_argument("--parameters-dict", type=str)
        parser.add_argument("--output", type=str, help="Save full RoutineExecutionResult as JSON")
        parser.add_argument("--download-dir", type=str, help="Directory for downloaded files")
        parser.add_argument("--keep-open", action="store_true", help="Keep the browser tab open after execution (default: False)")
        parser.add_argument("--remote-debugging-address", type=str, default="http://127.0.0.1:9222", help="Chrome debugging address (default: http://127.0.0.1:9222)")
        parser.add_argument("--no-incognito", action="store_true", help="Disable incognito mode (default: incognito is on)")
        args = parser.parse_args()
        routine_path = args.routine_path
        parameters_path = args.parameters_path
        parameters_dict = args.parameters_dict
        output = args.output
        download_dir = args.download_dir
        keep_open = args.keep_open
        remote_debugging_address = args.remote_debugging_address
        incognito = not args.no_incognito
    
    # Validate parameters
    if parameters_path and parameters_dict:
        raise ValueError("Only one of --parameters-path or --parameters-dict must be provided")
    if not parameters_path and not parameters_dict:
        raise ValueError("Either --parameters-path or --parameters-dict must be provided")
    
    # Load parameters
    if parameters_path:
        with open(parameters_path, encoding="utf-8") as f:
            params = json.load(f)
    else:
        params = json.loads(parameters_dict)  # type: ignore[arg-type]
    
    # Load and execute routine
    with open(routine_path, encoding="utf-8") as f:
        routine = Routine(**json.load(f))
    
    try:
        result = routine.execute(
            parameters_dict=params,
            remote_debugging_address=remote_debugging_address,
            timeout=60.0,
            close_tab_when_done=not keep_open,
            incognito=incognito,
        )
        logger.info(f"Result: {result}")
        save_result(result, output, download_dir)
    except Exception as e:
        logger.error("Error executing routine: %s", e)


if __name__ == "__main__":
    main()
