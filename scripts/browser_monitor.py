"""
src/scripts/browser_monitor.py

CDP-based web scraper that blocks trackers and captures network requests.
"""

import argparse
import logging
import os
import json
import time
import shutil
import sys

from src.cdp.cdp_session import CDPSession
from src.data_models.network import ResourceType
from src.cdp.tab_managements import cdp_new_tab, dispose_context

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Configuration ----

BLOCK_PATTERNS = [
    "*://*.doubleclick.net/*",
    "*://*.googletagmanager.com/*",
    "*://*.google-analytics.com/*",
    "*://*.g.doubleclick.net/*",
    "*://*.facebook.com/tr/*",
    "*://connect.facebook.net/*",
    "*://tr.snapchat.com/*",
    "*://sc-static.net/*",
    "*://*.scorecardresearch.com/*",
    "*://*.quantserve.com/*",
    "*://*.krxd.net/*",
    "*://*.adobedtm.com/*",
    "*://*.omtrdc.net/*",
    "*://*.demdex.net/*",
    "*://*.optimizely.com/*",
    "*://cdn.cookielaw.org/*",
    "*://*.segment.io/*",
    "*://*.mixpanel.com/*",
    "*://*.hotjar.com/*",
    "*://*.clarity.ms/*",
    "*://*.taboola.com/*",
    "*://*.outbrain.com/*",
]

BLOCK_REGEXES = [
    r"(^|://)([^/]*\.)?doubleclick\.net/",
    r"(^|://)([^/]*\.)?googletagmanager\.com/",
    r"(^|://)([^/]*\.)?google-analytics\.com/",
    r"(^|://)([^/]*\.)?facebook\.com/tr/?(\?|$)",
    r"(^|://)connect\.facebook\.net/",
    r"(^|://)tr\.snapchat\.com/",
    r"(^|://)sc-static\.net/",
    r"(^|://)([^/]*\.)?scorecardresearch\.com/",
    r"(^|://)([^/]*\.)?quantserve\.com/",
    r"(^|://)([^/]*\.)?krxd\.net/",
    r"(^|://)([^/]*\.)?adobedtm\.com/",
    r"(^|://)([^/]*\.)?omtrdc\.net/",
    r"(^|://)([^/]*\.)?demdex\.net/",
    r"(^|://)([^/]*\.)?optimizely\.com/",
    r"(^|://)([^/]*\.)?segment\.io/",
    r"(^|://)([^/]*\.)?mixpanel\.com/",
    r"(^|://)([^/]*\.)?hotjar\.com/",
    r"(^|://)([^/]*\.)?clarity\.ms/",
    r"(^|://)([^/]*\.)?taboola\.com/",
    r"(^|://)([^/]*\.)?outbrain\.com/",
]

# Default values - can be overridden by command line args
DEFAULT_CAPTURE_RESOURCE_TYPES = {ResourceType.XHR, ResourceType.FETCH}


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CDP-based web scraper that blocks trackers and captures network requests. By default, clears all cookies and storage before monitoring.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_cdp.py <TAB_ID>                    # Use existing tab
  python run_cdp.py                             # Create new tab automatically
  python run_cdp.py --incognito                 # Create new incognito tab
  python run_cdp.py --tab-id <TAB_ID> --url https://example.com
  python run_cdp.py -t <TAB_ID> --output-dir ./captures --no-navigate
  python run_cdp.py -t <TAB_ID> --capture-resources XHR Fetch --block-resources Image Font
  python run_cdp.py -t <TAB_ID> --no-clear-all --url https://example.com
  python run_cdp.py -t <TAB_ID> --no-clear-cookies --no-clear-storage

Get TAB_ID from chrome://inspect/#devices or http://127.0.0.1:9222/json
If no TAB_ID is provided, a new tab will be created automatically.
        """
    )
    
    parser.add_argument(
        "tab_id", 
        nargs="?",
        help="Chrome DevTools tab ID (optional - will create new tab if not provided)"
    )
    
    parser.add_argument(
        "-t", "--tab-id",
        dest="tab_id_alt",
        help="Chrome DevTools tab ID (alternative to positional argument)"
    )
    
    parser.add_argument(
        "--incognito",
        action="store_true",
        help="Create new tab in incognito mode (only used when no tab_id provided)"
    )
    
    parser.add_argument(
        "-u", "--url",
        default="about:blank",
        help="URL to navigate to (default: about:blank)"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="./cdp_captures",
        help="Output directory for captures (default: ./cdp_captures)"
    )
    
    parser.add_argument(
        "--no-navigate",
        action="store_true",
        help="Don't navigate to URL, just attach to existing tab"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=9222,
        help="Chrome DevTools port (default: 9222)"
    )
    
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Chrome DevTools host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear output directory before starting (default behavior)"
    )
    
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="Keep existing files in output directory"
    )

    
    parser.add_argument(
        "--capture-resources",
        nargs="*", 
        default=list(DEFAULT_CAPTURE_RESOURCE_TYPES),
        help="Resource types to capture and save (default: XHR Fetch)"
    )
    
    parser.add_argument(
        "--no-clear-cookies",
        action="store_true",
        help="Don't clear browser cookies before starting monitoring (cookies are cleared by default)"
    )
    
    parser.add_argument(
        "--no-clear-storage",
        action="store_true",
        help="Don't clear localStorage and sessionStorage before starting monitoring (storage is cleared by default)"
    )
    
    parser.add_argument(
        "--no-clear-all",
        action="store_true",
        help="Don't clear cookies or storage before starting monitoring (disables default clearing)"
    )
    
    args = parser.parse_args()
    
    # Determine tab_id from positional or named argument
    tab_id = args.tab_id or args.tab_id_alt
    # tab_id is now optional - will create new tab if not provided
    
    # Validate conflicting options
    if args.clear_output and args.keep_output:
        parser.error("--clear-output and --keep-output are mutually exclusive")
    
    # Convert resource lists to sets
    args.capture_resources = set(args.capture_resources)
    
    # Set clearing defaults (enabled by default)
    args.clear_cookies = True
    args.clear_storage = True
    
    # Handle no-clear options (disable clearing)
    if args.no_clear_all:
        args.clear_cookies = False
        args.clear_storage = False
    else:
        if args.no_clear_cookies:
            args.clear_cookies = False
        if args.no_clear_storage:
            args.clear_storage = False
    
    return args, tab_id


def setup_output_directory(output_dir, keep_output):
    """Setup the output directory structure and return paths for log files."""
    # Handle main output directory
    if os.path.exists(output_dir) and not keep_output:
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create organized subdirectories
    network_dir = os.path.join(output_dir, "network")
    storage_dir = os.path.join(output_dir, "storage")
    
    os.makedirs(network_dir, exist_ok=True)
    os.makedirs(storage_dir, exist_ok=True)
    
    # Create transactions directory for unified request/response storage
    transactions_dir = os.path.join(network_dir, "transactions")
    os.makedirs(transactions_dir, exist_ok=True)
    
    return {
        # Main directories
        'network_dir': network_dir,
        'storage_dir': storage_dir,
        'transactions_dir': transactions_dir,
        
        
        # Storage files  
        'storage_jsonl_path': os.path.join(storage_dir, "events.jsonl"),
        
        # Summary file
        'summary_path': os.path.join(output_dir, "session_summary.json")
    }


def save_session_summary(paths, summary, args, start_time, end_time, created_tab=False, context_id=None):
    """Save detailed session summary to JSON file."""
    session_summary = {
        "session_info": {
            "start_time": start_time,
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "tab_id": args.tab_id or args.tab_id_alt,
            "url": args.url,
            "output_dir": args.output_dir,
            "tab_created": created_tab,
            "context_id": context_id
        },
        "configuration": {
            "capture_resources": list(args.capture_resources),
            "block_patterns_count": len(BLOCK_PATTERNS),
            "navigated": not args.no_navigate,
            "cleared_cookies": args.clear_cookies,
            "cleared_storage": args.clear_storage
        },
        "monitoring_summary": summary,
        "output_files": {
            "network": {
                "transactions": paths['transactions_dir']
            },
            "storage": {
                "events": paths['storage_jsonl_path']
            }
        }
    }
    
    with open(paths['summary_path'], mode='w', encoding='utf-8') as f:
        json.dump(session_summary, f, indent=2, ensure_ascii=False)
    
    return session_summary


def main():
    """Main function."""
    start_time = time.time()
    
    # Parse arguments
    args, tab_id = parse_arguments()
    
    # Setup output directory and paths
    paths = setup_output_directory(args.output_dir, args.keep_output)
    
    # Handle tab creation if no tab_id provided
    created_tab = False
    context_id = None
    remote_debugging_address = f"http://{args.host}:{args.port}"
    
    if not tab_id:
        logger.info("No tab ID provided, creating new tab...")
        try:
            tab_id, context_id = cdp_new_tab(
                remote_debugging_address=remote_debugging_address,
                incognito=args.incognito,
                url=args.url if not args.no_navigate else "about:blank"
            )
            created_tab = True
            logger.info(f"Created new tab: {tab_id}")
            if context_id:
                logger.info(f"Browser context: {context_id}")
        except Exception as e:
            logger.info(f"Error creating new tab: {e}")
            sys.exit(1)
    
    # Build WebSocket URL
    ws_url = f"ws://{args.host}:{args.port}/devtools/page/{tab_id}"
    navigate_to = None if args.no_navigate else args.url
    
    logger.info(f"Starting CDP monitoring session...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target URL: {navigate_to or 'No navigation (attach only)'}")
    logger.info(f"Tab ID: {tab_id}")

    # Create and run CDP session
    try:
        session = CDPSession(
            ws_url, 
            paths['network_dir'],  # Use network directory for response bodies
            paths, 
            capture_resources=args.capture_resources,
            block_patterns=BLOCK_PATTERNS,
            clear_cookies=args.clear_cookies,
            clear_storage=args.clear_storage
        )
        session.setup_cdp(navigate_to)
        session.run()

    except KeyboardInterrupt:
        logger.info("\nSession stopped by user")
    except Exception as e:
        logger.info(f"Error: {e}")
        sys.exit(1)
    finally:
        # Cleanup: dispose context if we created a tab
        if created_tab and context_id:
            try:
                logger.info("Cleaning up created browser context...")
                dispose_context(remote_debugging_address, context_id)
            except Exception as e:
                logger.info(f"Warning: Could not dispose browser context: {e}")

        end_time = time.time()

        # Get final summary and save it
        try:
            summary = session.get_monitoring_summary()
            save_session_summary(paths, summary, args, start_time, end_time, created_tab, context_id)

            # Print organized summary
            logger.info("\n" + "="*60)
            logger.info("SESSION SUMMARY")
            logger.info("="*60)
            logger.info(f"Duration: {end_time - start_time:.1f} seconds")
            logger.info(f"Tab created: {'Yes' if created_tab else 'No'}")
            if created_tab and context_id:
                logger.info(f"Browser context: {context_id}")
            logger.info(f"Network requests tracked: {summary['network']['requests_tracked']}")
            logger.info(f"Cookies tracked: {summary['storage']['cookies_count']}")
            logger.info(f"LocalStorage origins: {len(summary['storage']['local_storage_origins'])}")
            logger.info(f"SessionStorage origins: {len(summary['storage']['session_storage_origins'])}")
            logger.info("OUTPUT STRUCTURE:")
            logger.info(f"├── session_summary.json")
            logger.info(f"├── network/")
            logger.info(f"│   ├── consolidated_transactions.json")
            logger.info(f"│   ├── network.har")
            logger.info(f"│   └── transactions/")
            logger.info(f"│       └── [timestamp_url_id]/")
            logger.info(f"│           ├── request.json")
            logger.info(f"│           ├── response.json")
            logger.info(f"│           └── response_body.[ext]")
            logger.info(f"└── storage/")
            logger.info(f"    └── events.jsonl")
            logger.info()
            logger.info(f"Session complete! Check {args.output_dir} for all outputs.")

        except Exception as e:
            logger.info("Warning: Could not generate summary: %s", e)


if __name__ == "__main__":
    main()
