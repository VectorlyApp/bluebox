#!/usr/bin/env python3
"""
Quickstart script: Full workflow for web-hacker
This script guides you through: Launch Chrome → Monitor → Discover → Execute
"""

import os
import sys
import time
import platform
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional
import requests
import websocket

# Colors for output (ANSI codes work on modern Windows 10+ terminals)
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color

# Configuration
PORT = 9222
CDP_CAPTURES_DIR = Path("./cdp_captures")
DISCOVERY_OUTPUT_DIR = Path("./routine_discovery_output")


def print_colored(text: str, color: str = NC) -> None:
    """Print colored text."""
    print(f"{color}{text}{NC}")


def check_chrome_running(port: int) -> bool:
    """Check if Chrome is already running in debug mode."""
    try:
        response = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=1)
        return response.status_code == 200
    except (requests.RequestException, requests.Timeout):
        return False


def open_url_in_chrome(port: int, url: str) -> bool:
    """Navigate the existing Chrome tab to a URL using CDP."""
    try:
        # Get list of existing tabs
        tabs_response = requests.get(f"http://127.0.0.1:{port}/json", timeout=2)
        if tabs_response.status_code != 200:
            return False
        
        tabs = tabs_response.json()
        if not tabs:
            return False
        
        # Use the first available tab
        first_tab = tabs[0]
        target_id = first_tab.get("id")
        if not target_id:
            return False
        
        # Navigate the existing tab using WebSocket
        try:
            # Get browser WebSocket URL (not the tab's)
            version_response = requests.get(f"http://127.0.0.1:{port}/json/version", timeout=2)
            if version_response.status_code != 200:
                return False
            
            browser_ws_url = version_response.json().get("webSocketDebuggerUrl")
            if not browser_ws_url:
                return False
            
            ws = websocket.create_connection(browser_ws_url, timeout=5)
            try:
                next_id = 1
                
                # Attach to the target
                attach_id = next_id
                attach_msg = {
                    "id": attach_id,
                    "method": "Target.attachToTarget",
                    "params": {"targetId": target_id, "flatten": True}
                }
                ws.send(json.dumps(attach_msg))
                next_id += 1
                
                # Read attach response (may need to skip event messages)
                ws.settimeout(5)
                session_id = None
                while True:
                    try:
                        msg = json.loads(ws.recv())
                        # Look for the response with matching ID
                        if msg.get("id") == attach_id:
                            if "error" in msg:
                                print_colored(f"⚠️  Attach error: {msg.get('error')}", YELLOW)
                                return False
                            if "result" in msg:
                                session_id = msg["result"].get("sessionId")
                                if session_id:
                                    break
                                else:
                                    print_colored(f"⚠️  No sessionId in attach response: {msg}", YELLOW)
                                    return False
                    except websocket.WebSocketTimeoutException:
                        print_colored("⚠️  Timeout waiting for attach response", YELLOW)
                        return False
                
                if not session_id:
                    print_colored("⚠️  Failed to get session ID", YELLOW)
                    return False
                
                # Enable Page domain
                enable_msg = {
                    "id": next_id,
                    "method": "Page.enable",
                    "sessionId": session_id
                }
                ws.send(json.dumps(enable_msg))
                next_id += 1
                
                # Read enable response (skip if timeout)
                ws.settimeout(1)
                try:
                    while True:
                        msg = json.loads(ws.recv())
                        if msg.get("id") == next_id - 1:
                            break
                except websocket.WebSocketTimeoutException:
                    pass  # Continue anyway
                
                # Navigate to URL
                navigate_msg = {
                    "id": next_id,
                    "method": "Page.navigate",
                    "params": {"url": url},
                    "sessionId": session_id
                }
                ws.send(json.dumps(navigate_msg))
                
                # Wait briefly for navigate response
                ws.settimeout(1)
                try:
                    while True:
                        msg = json.loads(ws.recv())
                        if msg.get("id") == next_id:
                            return True
                        if msg.get("error"):
                            return False
                except websocket.WebSocketTimeoutException:
                    # Timeout is okay, navigation was sent
                    return True
            finally:
                ws.close()
        except ImportError:
            # websocket library not available - this shouldn't happen if web-hacker is installed
            print_colored("⚠️  websocket library not available. Cannot navigate tab.", YELLOW)
            return False
        except Exception as e:
            # Print error for debugging
            print_colored(f"⚠️  Error navigating tab: {e}", YELLOW)
            return False
    except (requests.RequestException, requests.Timeout) as e:
        print_colored(f"⚠️  Error connecting to Chrome: {e}", YELLOW)
        return False
    except Exception as e:
        print_colored(f"⚠️  Unexpected error: {e}", YELLOW)
        return False


def find_chrome_path() -> Optional[str]:
    """Find Chrome executable path based on OS."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.isfile(chrome_path):
            return chrome_path
    elif system == "Linux":
        # Try common Linux Chrome/Chromium names
        for name in ["google-chrome", "chromium-browser", "chromium", "chrome"]:
            chrome_path = shutil.which(name)
            if chrome_path:
                return chrome_path
    elif system == "Windows":
        # Common Windows Chrome locations
        possible_paths = [
            os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
            os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        ]
        for path in possible_paths:
            if os.path.isfile(path):
                return path
        # Try to find in PATH
        chrome_path = shutil.which("chrome") or shutil.which("google-chrome")
        if chrome_path:
            return chrome_path
    
    return None


def launch_chrome(port: int) -> Optional[subprocess.Popen]:
    """Launch Chrome in debug mode."""
    chrome_path = find_chrome_path()
    
    if not chrome_path:
        print_colored("⚠️  Chrome not found automatically.", YELLOW)
        print("   Please launch Chrome manually with:")
        print(f"   --remote-debugging-port={port}")
        print()
        input("Press Enter when Chrome is running in debug mode...")
        return None
    
    # Create user data directory
    if platform.system() == "Windows":
        chrome_user_dir = os.path.expandvars(r"%USERPROFILE%\tmp\chrome")
    else:
        chrome_user_dir = os.path.expanduser("~/tmp/chrome")
    
    os.makedirs(chrome_user_dir, exist_ok=True)
    
    # Build Chrome arguments
    chrome_args = [
        chrome_path,
        f"--remote-debugging-address=127.0.0.1",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={chrome_user_dir}",
        "--remote-allow-origins=*",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    
    # Launch Chrome
    print("🚀 Launching Chrome...")
    try:
        # On Windows, use CREATE_NEW_PROCESS_GROUP to detach
        creation_flags = 0
        if platform.system() == "Windows":
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        
        process = subprocess.Popen(
            chrome_args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )
        
        # Wait for Chrome to be ready
        print("⏳ Waiting for Chrome to start...")
        for _ in range(10):
            if check_chrome_running(port):
                print_colored("✅ Chrome is ready!", GREEN)
                # Give Chrome a moment to fully initialize tabs
                time.sleep(0.5)
                # Open documentation page explaining what's happening
                doc_url = "https://github.com/VectorlyApp/web-hacker/blob/main/scripts/chrome-debug-mode-explanation.md"
                if open_url_in_chrome(port, doc_url):
                    print_colored("✅ Documentation page opened", GREEN)
                else:
                    print_colored("⚠️  Could not navigate Chrome tab. There may be an issue with the Chrome connection.", YELLOW)
                return process
            time.sleep(1)
        
        # Chrome didn't start in time
        print_colored("⚠️  Chrome failed to start automatically.", YELLOW)
        try:
            process.terminate()
            time.sleep(0.5)
            process.kill()
        except Exception:
            pass
        
        print("   Please launch Chrome manually with:")
        print(f"   --remote-debugging-port={port}")
        print()
        input("Press Enter when Chrome is running in debug mode...")
        return None
        
    except Exception as e:
        print_colored(f"⚠️  Error launching Chrome: {e}", YELLOW)
        print("   Please launch Chrome manually with:")
        print(f"   --remote-debugging-port={port}")
        print()
        input("Press Enter when Chrome is running in debug mode...")
        return None


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False
    except KeyboardInterrupt:
        print()
        print_colored("⚠️  Command interrupted.", YELLOW)
        return False
    except FileNotFoundError:
        print_colored(f"⚠️  Command not found: {cmd[0]}", YELLOW)
        print("   Make sure web-hacker is installed: pip install -e .")
        return False


def main():
    """Main workflow."""
    # Use local variables that can be updated
    cdp_captures_dir = CDP_CAPTURES_DIR
    discovery_output_dir = DISCOVERY_OUTPUT_DIR
    
    print_colored("╔════════════════════════════════════════════════════════════╗", BLUE)
    print_colored("║         Web Hacker - Quickstart Workflow                   ║", BLUE)
    print_colored("╚════════════════════════════════════════════════════════════╝", BLUE)
    print()

    # Pipeline overview
    print_colored("Web-hacker Pipeline:", BLUE)
    print()
    print("  Step 1: Launch Chrome in debug mode")
    print("  Step 2: Monitor browser interactions")
    print("  Step 3: Discover web action routine")
    print("  Step 4 (optional): Test routine execution")
    print()
    input("Press Enter to start: ")
    print()
    
    # Step 1: Launch Chrome
    print_colored("Step 1: Launching Chrome in debug mode...", GREEN)
    
    chrome_process = None
    if check_chrome_running(PORT):
        print_colored(f"✅ Chrome is already running in debug mode on port {PORT}", GREEN)
    else:
        chrome_process = launch_chrome(PORT)
    
    print()
    
    # Step 2: Monitor
    print_colored("Step 2: Starting browser monitoring...", GREEN)
    
    skip = input("   Skip monitoring step? (y/n): ").strip().lower()
    if skip == 'y':
        new_dir = input(f"   Enter CDP captures directory path [Press Enter to use: {CDP_CAPTURES_DIR.resolve()}]: ").strip()
        if new_dir:
            cdp_captures_dir = Path(new_dir)
            print_colored(f"✅ Using CDP captures directory: {cdp_captures_dir}", GREEN)
        print_colored("⏭️  Skipping monitoring step.", GREEN)
        print()
    else:
        # Check if directory exists and has content before running monitoring
        if cdp_captures_dir.exists() and any(cdp_captures_dir.iterdir()):
            print_colored(f"⚠️  Directory {cdp_captures_dir} already exists and contains files.", YELLOW)
            confirm = input("   Remove existing data before monitoring? (Data may be overwritten if not removed) (y/n): ").strip().lower()
            if confirm == 'y':
                # Remove all data but keep the directory
                for item in cdp_captures_dir.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                print_colored(f"✅ Cleared data in {cdp_captures_dir}", GREEN)
            else:
                print_colored(f"⚠️  Keeping existing data in {cdp_captures_dir}", YELLOW)
        
        print_colored("📋 Instructions:", YELLOW)
        print("   1. A new Chrome tab will open")
        print("   2. Navigate to your target website")
        print("   3. Perform the actions you want to automate (search, login, etc.)")
        print("   4. Press Ctrl+C when you're done")
        print()
        input("Press Enter to open a new tab and start monitoring...")
        
        print()
        print("🚀 Starting monitor (press Ctrl+C when done)...")
        
        monitor_cmd = [
            "web-hacker-monitor",
            "--host", "127.0.0.1",
            "--port", str(PORT),
            "--output-dir", str(cdp_captures_dir),
            "--url", "about:blank",
            "--incognito",
        ]
        
        run_command(monitor_cmd, "monitoring")
        print()
    
    # Step 3: Discover
    print_colored("Step 3: Discovering routine from captured data...", GREEN)
    
    # Check if capture data exists first
    transactions_dir = cdp_captures_dir / "network" / "transactions"
    if not cdp_captures_dir.exists() or not transactions_dir.exists() or not any(transactions_dir.iterdir()):
        print_colored("⚠️  No capture data found. Skipping discovery step.", YELLOW)
        print("   Make sure you performed actions during monitoring.")
        return
    
    skip = input("   Skip discovery step? (y/n): ").strip().lower()
    
    if skip == 'y':
        # Use default directory when skipping - user can specify routine path in step 4 if needed
        discovery_output_dir = DISCOVERY_OUTPUT_DIR
        
        print_colored("⏭️  Skipping discovery step.", GREEN)
        print_colored(f"   Using default discovery output directory: {discovery_output_dir.resolve()}", GREEN)
        print()
        
        # Set routine_file for step 4 even if skipped
        routine_file = discovery_output_dir / "routine.json"
    else:
        new_output_dir = input(f"   Enter discovery output directory path [Press Enter to use: {DISCOVERY_OUTPUT_DIR.resolve()}]: ").strip()
        if new_output_dir:
            discovery_output_dir = Path(new_output_dir)
            print_colored(f"✅ Using discovery output directory: {discovery_output_dir}", GREEN)
        
        # Check if routine already exists
        routine_file = discovery_output_dir / "routine.json"
        has_existing_routine = routine_file.exists()
        
        if has_existing_routine:
            print_colored(f"📁 Found existing routine at {routine_file}", YELLOW)
            overwrite = input("   Overwrite existing routine? (y/n): ").strip().lower()
            if overwrite != 'y':
                print_colored("⏭️  Keeping existing routine. Skipping discovery step.", GREEN)
                print()
            else:
                # Check if directory exists and has content before running discovery
                if discovery_output_dir.exists() and any(discovery_output_dir.iterdir()):
                    print_colored(f"⚠️  Directory {discovery_output_dir} already exists and contains files.", YELLOW)
                    confirm = input("   Remove existing data before discovery? (Data may be overwritten if not removed) (y/n): ").strip().lower()
                    if confirm == 'y':
                        # Remove all data but keep the directory
                        for item in discovery_output_dir.iterdir():
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        print_colored(f"✅ Cleared data in {discovery_output_dir}", GREEN)
                    else:
                        print_colored(f"⚠️  Keeping existing data in {discovery_output_dir}", YELLOW)
                
                print_colored("📋 Enter a description of what you want to automate:", YELLOW)
                print("   Example: 'Search for flights and get prices'")
                print("   (Press Ctrl+C to exit)")
                
                task = ""
                while not task:
                    try:
                        task = input("   Task: ").strip()
                        if not task:
                            print_colored("⚠️  Task cannot be empty. Please enter a description (or Ctrl+C to exit).", YELLOW)
                    except KeyboardInterrupt:
                        print()
                        print_colored("⚠️  Discovery cancelled by user.", YELLOW)
                        return
                
                print()
                print("🤖 Running routine discovery agent...")
                
                discover_cmd = [
                    "web-hacker-discover",
                    "--task", task,
                    "--cdp-captures-dir", str(cdp_captures_dir),
                    "--output-dir", str(discovery_output_dir),
                    "--llm-model", "gpt-5",
                ]
                
                run_command(discover_cmd, "discovery")
                print()
        else:
            # Check if directory exists and has content before running discovery
            if discovery_output_dir.exists() and any(discovery_output_dir.iterdir()):
                print_colored(f"⚠️  Directory {discovery_output_dir} already exists and contains files.", YELLOW)
                confirm = input("   Remove existing data before discovery? (Data may be overwritten if not removed) (y/n): ").strip().lower()
                if confirm == 'y':
                    # Remove all data but keep the directory
                    for item in discovery_output_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                    print_colored(f"✅ Cleared data in {discovery_output_dir}", GREEN)
                else:
                    print_colored(f"⚠️  Keeping existing data in {discovery_output_dir}", YELLOW)
            
            print_colored("📋 Enter a description of what you want to automate:", YELLOW)
            print("   Example: 'Search for flights and get prices'")
            print("   (Press Ctrl+C to exit)")
            
            task = ""
            while not task:
                try:
                    task = input("   Task: ").strip()
                    if not task:
                        print_colored("⚠️  Task cannot be empty. Please enter a description (or Ctrl+C to exit).", YELLOW)
                except KeyboardInterrupt:
                    print()
                    print_colored("⚠️  Discovery cancelled by user.", YELLOW)
                    return
            
            print()
            print("🤖 Running routine discovery agent...")
            
            discover_cmd = [
                "web-hacker-discover",
                "--task", task,
                "--cdp-captures-dir", str(cdp_captures_dir),
                "--output-dir", str(discovery_output_dir),
                "--llm-model", "gpt-5",
            ]
            
            run_command(discover_cmd, "discovery")
            print()
    
    # Step 4: Execute (optional)
    if not routine_file.exists():
        print_colored(f"⚠️  Routine not found at {routine_file}", YELLOW)
        return
    
    print_colored("Step 4: Ready to execute routine!", GREEN)
    print()
    print("✅ Routine discovered successfully!")
    print(f"   Location: {routine_file}")
    print()
    print_colored("To execute the routine, run:", YELLOW)
    print("   web-hacker-execute \\")
    print(f"     --routine-path {routine_file} \\")
    
    test_params_file = discovery_output_dir / "test_parameters.json"
    if test_params_file.exists():
        print(f"     --parameters-path {test_params_file}")
    else:
        print("     --parameters-dict '{\"param1\": \"value1\", \"param2\": \"value2\"}'")
    
    print()
    print_colored(f"💡 Tip: Review {routine_file} before executing", BLUE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print_colored("⚠️  Interrupted by user.", YELLOW)
        sys.exit(0)