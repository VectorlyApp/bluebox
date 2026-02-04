"""
bluebox/scripts/browser_monitor.py

CDP-based web scraper that blocks trackers and captures network requests.
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone

from bluebox.cdp.async_cdp_session import AsyncCDPSession
from bluebox.cdp.file_event_writer import FileEventWriter
from bluebox.cdp.connection import cdp_new_tab, dispose_context
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def parse_arguments() -> tuple[argparse.Namespace, str | None]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CDP-based web scraper that blocks trackers and captures network requests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bluebox-monitor <TAB_ID>                    # Use existing tab
  bluebox-monitor                             # Create new tab automatically
  bluebox-monitor --incognito                 # Create new incognito tab
  bluebox-monitor --tab-id <TAB_ID> --url https://example.com
  bluebox-monitor -t <TAB_ID> --output-dir ./captures --no-navigate

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

    args = parser.parse_args()

    # Determine tab_id from positional or named argument
    tab_id = args.tab_id or args.tab_id_alt

    # Validate conflicting options
    if args.clear_output and args.keep_output:
        parser.error("--clear-output and --keep-output are mutually exclusive")

    return args, tab_id


def save_session_summary(
    paths: dict,
    summary: dict,
    args: argparse.Namespace,
    start_time: float,
    end_time: float,
    created_tab: bool = False,
    context_id: str | None = None,
) -> dict:
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
            "navigated": not args.no_navigate,
        },
        "monitoring_summary": summary,
    }

    summary_path = paths.get("summary_path", os.path.join(args.output_dir, "session_summary.json"))
    with open(summary_path, mode='w', encoding='utf-8') as f:
        json.dump(session_summary, f, indent=2, ensure_ascii=False)

    return session_summary


async def async_main(args: argparse.Namespace, tab_id: str | None) -> None:
    """Async entry point for CDP monitoring."""
    start_time = time.time()

    # Clear output directory if needed
    if os.path.exists(args.output_dir) and not args.keep_output:
        shutil.rmtree(args.output_dir)

    # Set up file writer (creates output directory structure)
    writer = FileEventWriter.create_from_output_dir(args.output_dir)

    # Handle tab creation if no tab_id provided
    created_tab = False
    context_id = None
    remote_debugging_address = f"http://{args.host}:{args.port}"

    if not tab_id:
        logger.info("No tab ID provided, creating new tab...")
        try:
            tab_id, context_id, browser_ws = cdp_new_tab(
                remote_debugging_address=remote_debugging_address,
                incognito=args.incognito,
                url=args.url if not args.no_navigate else "about:blank"
            )
            try:
                browser_ws.close()
            except Exception:
                pass
            created_tab = True
            logger.info(f"Created new tab: {tab_id}")
            if context_id:
                logger.info(f"Browser context: {context_id}")
        except Exception as e:
            logger.error(f"Error creating new tab: {e}")
            sys.exit(1)

    # Build page-level WebSocket URL
    ws_url = f"ws://{args.host}:{args.port}/devtools/page/{tab_id}"

    logger.info(f"Starting CDP monitoring session...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Target URL: {args.url if not args.no_navigate else 'No navigation (attach only)'}")
    logger.info(f"Tab ID: {tab_id}")

    session = AsyncCDPSession(
        ws_url=ws_url,
        session_start_dtm=datetime.now(timezone.utc).isoformat(),
        event_callback_fn=writer.write_event,
        paths=writer.paths,
    )

    try:
        await session.run()
    except KeyboardInterrupt:
        logger.info("\nSession stopped by user")
    except asyncio.CancelledError:
        logger.info("\nSession cancelled")
    except Exception:
        logger.error("Session crashed!", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Running cleanup...")

        # Finalize session (consolidate transactions, HAR, etc.)
        try:
            await session.finalize()
        except Exception as e:
            logger.warning(f"Could not finalize session: {e}")

        # Dispose browser context if we created one
        if created_tab and context_id:
            try:
                logger.info(f"Disposing browser context {context_id}...")
                dispose_context(remote_debugging_address, context_id)
                logger.info("Browser context disposed")
            except Exception as e:
                logger.error(f"Failed to dispose browser context: {e}", exc_info=True)

        end_time = time.time()

        # Save and print summary
        try:
            summary = session.get_monitoring_summary()
            save_session_summary(writer.paths, summary, args, start_time, end_time, created_tab, context_id)

            logger.info("\n" + "=" * 60)
            logger.info("SESSION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Duration: {end_time - start_time:.1f} seconds")
            logger.info(f"Tab created: {'Yes' if created_tab else 'No'}")
            if created_tab and context_id:
                logger.info(f"Browser context: {context_id}")
            logger.info(f"Network: {summary.get('network', {})}")
            logger.info(f"Storage: {summary.get('storage', {})}")
            logger.info(f"Interaction: {summary.get('interaction', {})}")
            logger.info(f"Window Properties: {summary.get('window_properties', {})}")
            logger.info(f"\nSession complete! Check {args.output_dir} for all outputs.")
        except Exception as e:
            logger.warning("Could not generate summary: %s", e)


def main() -> None:
    """Main function."""
    args, tab_id = parse_arguments()
    asyncio.run(async_main(args, tab_id))


if __name__ == "__main__":
    main()
