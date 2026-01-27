#!/usr/bin/env python3
"""
Quickstart: Full workflow for bluebox-sdk using the Python SDK.
This script guides you through: Monitor → Discover → Execute

Usage:
    python quickstart.py
"""

import json
import sys
import time
from pathlib import Path

import websocket

from bluebox.sdk import Bluebox, BrowserMonitor
from bluebox.data_models.routine.routine import Routine
from bluebox.cdp.connection import get_existing_tabs
from bluebox.utils.chrome_utils import check_chrome_running, launch_chrome
from bluebox.utils.terminal_utils import (
    GREEN, YELLOW, BLUE, CYAN,
    print_colored, print_header, ask_yes_no,
)
from bluebox.utils.infra_utils import clear_directory

# Configuration
PORT = 9222
REMOTE_DEBUGGING_ADDRESS = f"http://127.0.0.1:{PORT}"
CDP_CAPTURES_DIR = Path("./cdp_captures")
DISCOVERY_OUTPUT_DIR = Path("./routine_discovery_output")


def step_1_monitor_browser(cdp_captures_dir: Path) -> bool:
    """Step 1: Monitor browser activity (launches Chrome if needed)."""
    print_header("Step 1: Monitor Browser Activity")

    if ask_yes_no("Skip monitoring step?"):
        new_dir = input(f"Enter CDP captures directory [default: {cdp_captures_dir}]: ").strip()
        if new_dir:
            cdp_captures_dir = Path(new_dir)
        print_colored(f"⏭️  Using existing captures from: {cdp_captures_dir}", GREEN)
        return True

    # Check for existing data
    if cdp_captures_dir.exists() and any(cdp_captures_dir.iterdir()):
        print_colored(f"⚠️  Directory {cdp_captures_dir} contains existing data.", YELLOW)
        if ask_yes_no("Clear existing data?"):
            clear_directory(cdp_captures_dir)
            print_colored(f"✅ Cleared {cdp_captures_dir}", GREEN)

    print()
    print_colored("📋 Instructions:", YELLOW)
    print("   1. Chrome will launch (if not already running)")
    print("   2. Navigate to your target website")
    print("   3. Perform the actions you want to automate")
    print("   4. Press Ctrl+C when done")
    print()
    input("Press Enter to start monitoring...")
    print()

    # Launch Chrome if not already running
    if check_chrome_running(PORT):
        print_colored(f"✅ Chrome is already running on port {PORT}", GREEN)
    else:
        launch_chrome(PORT)
        if not check_chrome_running(PORT):
            print_colored("❌ Chrome is not running. Cannot continue.", YELLOW)
            return False

    print("🔍 Starting browser monitor...")
    print_colored(f"   Output directory: {cdp_captures_dir}", BLUE)
    print()

    monitor = BrowserMonitor(
        remote_debugging_address=REMOTE_DEBUGGING_ADDRESS,
        output_dir=str(cdp_captures_dir),
        create_tab=False,
    )

    try:
        monitor.start()
        print_colored("✅ Monitoring started! Perform your actions in the browser.", GREEN)
        print_colored("   Press Ctrl+C when done...", YELLOW)
        print()

        # Wait for user to press Ctrl+C or tab to close
        while monitor.is_alive:
            time.sleep(1)

    except KeyboardInterrupt:
        print()
        print("⏹️  Stopping monitor...")
    finally:
        summary = monitor.stop()
        # Keep Chrome running for execution step

    print()
    print_colored("✅ Monitoring complete!", GREEN)
    if summary:
        print(f"   Duration: {summary.get('duration', 0):.1f}s")
        print(f"   Transactions captured: {summary.get('network_transactions', 0)}")

    return True


def step_2_discover_routine(
    client: Bluebox,
    cdp_captures_dir: Path,
    discovery_output_dir: Path,
) -> Routine | None:
    """Step 2: Discover routine from captured data."""
    print_header("Step 2: Discover Routine")

    # Check if capture data exists
    transactions_dir = cdp_captures_dir / "network" / "transactions"
    if not transactions_dir.exists() or not any(transactions_dir.iterdir()):
        print_colored("⚠️  No capture data found. Cannot run discovery.", YELLOW)
        print("   Make sure you performed actions during monitoring.")
        return None

    if ask_yes_no("Skip discovery step?"):
        routine_file = discovery_output_dir / "routine.json"
        if routine_file.exists():
            print_colored(f"⏭️  Loading existing routine from: {routine_file}", GREEN)
            return Routine.model_validate_json(routine_file.read_text())
        else:
            print_colored(f"⚠️  No existing routine found at {routine_file}", YELLOW)
            return None

    # Check for existing routine
    routine_file = discovery_output_dir / "routine.json"
    if routine_file.exists():
        print_colored(f"📁 Found existing routine at {routine_file}", YELLOW)
        if not ask_yes_no("Overwrite?"):
            print_colored("⏭️  Using existing routine.", GREEN)
            return Routine.model_validate_json(routine_file.read_text())

    # Clear existing discovery output
    if discovery_output_dir.exists() and any(discovery_output_dir.iterdir()):
        print_colored(f"⚠️  Directory {discovery_output_dir} contains existing data.", YELLOW)
        if ask_yes_no("Clear existing data?"):
            clear_directory(discovery_output_dir)
            print_colored(f"✅ Cleared {discovery_output_dir}", GREEN)

    print()
    print_colored("📋 Let's define your routine:", YELLOW)
    print("   We'll analyze the recorded session and turn it into a reusable routine.")
    print()

    # Step 1: What data to return
    print_colored("   What data do you want this routine to return? *", CYAN)
    print_colored("   (e.g., flight prices, product details, search results)", BLUE)
    data_output = ""
    while not data_output:
        try:
            data_output = input("   → ").strip()
            if not data_output:
                print_colored("   ⚠️  This field is required.", YELLOW)
        except KeyboardInterrupt:
            print()
            return None
    print()

    # Step 2: What inputs/filters
    print_colored("   What inputs or filters does it need?", CYAN)
    print_colored("   (e.g., search query, date range, location)", BLUE)
    try:
        inputs_needed = input("   → ").strip()
    except KeyboardInterrupt:
        print()
        return None
    print()

    # Step 3: Additional context
    print_colored("   Anything else? (optional)", CYAN)
    print_colored("   (e.g., notes, special handling, edge cases)", BLUE)
    try:
        extra_context = input("   → ").strip()
    except KeyboardInterrupt:
        print()
        return None
    print()

    # Build the task description
    task_parts = [f"Create a web routine that returns {data_output}"]
    if inputs_needed:
        task_parts.append(f"given {inputs_needed}")
    if extra_context:
        task_parts.append(f"({extra_context})")
    task = " ".join(task_parts) + "."

    # Show summary
    print_colored("   ─────────────────────────────────────────────", BLUE)
    print_colored("   ✓ Task:", YELLOW)
    print(f"   \"{task}\"")
    print_colored("   ─────────────────────────────────────────────", BLUE)

    print()
    print("🤖 Running routine discovery agent...")
    print_colored(f"   Task: {task}", BLUE)
    print_colored(f"   Captures: {cdp_captures_dir}", BLUE)
    print_colored(f"   Output: {discovery_output_dir}", BLUE)
    print()

    try:
        result = client.discover_routine(
            task=task,
            cdp_captures_dir=str(cdp_captures_dir),
            output_dir=str(discovery_output_dir),
        )
        routine = result.routine

        print()
        print_colored("✅ Routine discovered successfully!", GREEN)
        print(f"   Name: {routine.name}")
        print(f"   Operations: {len(routine.operations)}")
        print(f"   Parameters: {len(routine.parameters)}")

        return routine

    except Exception as e:
        print_colored(f"❌ Discovery failed: {e}", YELLOW)
        return None


def step_3_execute_routine(
    client: Bluebox,
    routine: Routine,
    discovery_output_dir: Path,
) -> None:
    """Step 3: Execute the discovered routine."""
    print_header("Step 3: Execute Routine")

    print_colored("📋 Routine Details:", BLUE)
    print(f"   Name: {routine.name}")
    print(f"   Description: {routine.description or 'N/A'}")
    print()

    print_colored("📋 Parameters:", BLUE)
    for param in routine.parameters:
        required = "required" if param.required else "optional"
        default = f", default: {param.default}" if param.default else ""
        print(f"   • {param.name} ({param.type}, {required}{default})")
        if param.description:
            print(f"     {param.description}")
    print()

    # Try to load test parameters
    test_params_file = discovery_output_dir / "test_parameters.json"
    parameters: dict[str, str] = {}

    if test_params_file.exists():
        try:
            parameters = json.loads(test_params_file.read_text())
            print_colored(f"📁 Loaded test parameters from: {test_params_file}", GREEN)
            print(f"   {json.dumps(parameters, indent=2)}")
            print()

            if not ask_yes_no("Use these parameters?"):
                parameters = {}
        except Exception:
            pass

    # Collect parameters if not using test params
    if not parameters:
        print_colored("Enter parameter values:", YELLOW)
        for param in routine.parameters:
            default_hint = f" [default: {param.default}]" if param.default else ""
            value = input(f"   {param.name}{default_hint}: ").strip()
            if value:
                parameters[param.name] = value
            elif param.default:
                parameters[param.name] = param.default
            elif param.required:
                print_colored(f"   ⚠️  {param.name} is required!", YELLOW)
                return

    print()
    if not ask_yes_no("Execute routine?"):
        print_colored("⏭️  Skipping execution.", GREEN)
        return

    # Launch Chrome only after user confirms execution
    if not check_chrome_running(PORT):
        print_colored("⚠️  Chrome not running. Launching for execution...", YELLOW)
        launch_chrome(PORT)
        if not check_chrome_running(PORT):
            print_colored("❌ Chrome is not running. Cannot execute routine.", YELLOW)
            return

    # Get existing tab to reuse
    tabs = get_existing_tabs(REMOTE_DEBUGGING_ADDRESS)
    page_tabs = [t for t in tabs if t.get("type") == "page"]
    tab_id = page_tabs[0]["id"] if page_tabs else None
    if tab_id:
        print_colored(f"📎 Reusing existing tab: {page_tabs[0].get('url', 'unknown')[:50]}...", BLUE)

    print()
    print("🚀 Executing routine...")
    print_colored(f"   Parameters: {json.dumps(parameters)}", BLUE)
    print()

    try:
        result = client.execute_routine(
            routine=routine,
            parameters=parameters,
            timeout=60.0,
            close_tab_when_done=True,
            tab_id=tab_id,  # Reuse existing tab if available
        )

        print()
        if result.ok:
            print_colored("✅ Execution successful!", GREEN)

            # Save result
            output_file = discovery_output_dir / "execution_result.json"
            output_data = {
                "ok": result.ok,
                "data": result.data,
                "placeholder_resolution": result.placeholder_resolution,
                "warnings": result.warnings,
            }
            output_file.write_text(json.dumps(output_data, indent=2))
            print_colored(f"   Result saved to: {output_file}", BLUE)

            # Preview
            if result.data:
                data_str = json.dumps(result.data, indent=2)
                preview = data_str[:500] + "..." if len(data_str) > 500 else data_str
                print()
                print_colored("📄 Result preview:", BLUE)
                print(preview)
        else:
            print_colored(f"❌ Execution failed: {result.error}", YELLOW)

    except Exception as e:
        print_colored(f"❌ Execution error: {e}", YELLOW)


def main() -> None:
    """Main workflow."""
    print_colored("╔════════════════════════════════════════════════════════════╗", BLUE)
    print_colored("║           Bluebox - Quickstart Workflow                    ║", BLUE)
    print_colored("╚════════════════════════════════════════════════════════════╝", BLUE)
    print()

    print_colored("Pipeline Overview:", CYAN)
    print("  1. Monitor browser interactions (or skip with existing captures)")
    print("  2. Discover routine from captures")
    print("  3. Execute routine")
    print()

    input("Press Enter to start: ")

    # Configuration
    cdp_captures_dir = CDP_CAPTURES_DIR
    discovery_output_dir = DISCOVERY_OUTPUT_DIR

    # Step 1: Monitor (handles Chrome launch internally if needed)
    if not step_1_monitor_browser(cdp_captures_dir):
        return

    # Initialize client
    print()
    print("🔧 Initializing Bluebox...")
    try:
        client = Bluebox(
            remote_debugging_address=REMOTE_DEBUGGING_ADDRESS,
            llm_model="gpt-5.1",
        )
        print_colored("✅ Ready!", GREEN)
    except Exception as e:
        print_colored(f"❌ Failed to initialize: {e}", YELLOW)
        print("   Make sure OPENAI_API_KEY is set.")
        return

    # Step 2: Discover
    routine = step_2_discover_routine(client, cdp_captures_dir, discovery_output_dir)
    if not routine:
        print_colored("⚠️  No routine available. Exiting.", YELLOW)
        return

    # Step 3: Execute
    step_3_execute_routine(client, routine, discovery_output_dir)

    print()
    print_colored("═" * 60, GREEN)
    print_colored("  🎉 Quickstart complete!", GREEN)
    print_colored("═" * 60, GREEN)
    print()
    print_colored("Next steps:", CYAN)
    print(f"  • Review routine: {discovery_output_dir / 'routine.json'}")
    print(f"  • Check results: {discovery_output_dir / 'execution_result.json'}")
    print("  • Deploy to production: https://console.vectorly.app")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("⚠️  Interrupted by user.", YELLOW)
        sys.exit(0)
