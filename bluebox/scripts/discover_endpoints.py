"""
Discover API endpoints from JavaScript bundle files.

Performs deterministic static analysis on captured JS files to extract:
- fetch() calls with URLs, methods, headers, body schemas
- XMLHttpRequest .open() calls
- axios/jQuery/generic HTTP client calls
- URL string patterns from config objects and route definitions

Usage:
    bluebox-endpoints --input cdp_captures/network/javascript_events.jsonl
    bluebox-endpoints --input cdp_captures/network/javascript_events.jsonl --format summary
    bluebox-endpoints --input cdp_captures/network/javascript_events.jsonl --output endpoints.json --format json
    bluebox-endpoints --input cdp_captures/network/javascript_events.jsonl --first-party-host www.premierleague.com
    bluebox-endpoints --input cdp_captures/network/javascript_events.jsonl --skip-third-party --format table
"""

import argparse
import sys

from bluebox.endpoint_discovery.analyzer import EndpointAnalyzer
from bluebox.endpoint_discovery.reporter import write_json, write_summary, write_table
from bluebox.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """CLI entry point for endpoint discovery."""
    parser = argparse.ArgumentParser(
        prog="bluebox-endpoints",
        description="Discover API endpoints from JavaScript bundle files via deterministic static analysis.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to javascript_events.jsonl file",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "summary", "table"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Analyze all JS files, even duplicates with same URL",
    )
    parser.add_argument(
        "--skip-third-party",
        action="store_true",
        help="Skip known third-party analytics/tracking JS files",
    )
    parser.add_argument(
        "--first-party-host",
        action="append",
        dest="first_party_hosts",
        default=None,
        help="Only analyze JS from this host (can be specified multiple times)",
    )
    parser.add_argument(
        "--include-snippets",
        action="store_true",
        help="Include raw code snippets in JSON output (increases size)",
    )

    args = parser.parse_args()

    # Run analysis
    try:
        analyzer = EndpointAnalyzer(
            jsonl_path=args.input,
            deduplicate_js=not args.no_dedup,
            skip_third_party=args.skip_third_party,
            first_party_hosts=args.first_party_hosts,
        )
        result = analyzer.analyze()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Write output
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            _write_output(result, f, args.format, args.include_snippets)
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        _write_output(result, sys.stdout, args.format, args.include_snippets)


def _write_output(result, output, fmt: str, include_snippets: bool) -> None:
    """Dispatch to the appropriate writer."""
    if fmt == "json":
        write_json(result, output, include_snippets=include_snippets)
    elif fmt == "table":
        write_table(result, output)
    else:
        write_summary(result, output)


if __name__ == "__main__":
    main()
