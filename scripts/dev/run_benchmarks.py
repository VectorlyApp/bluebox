#!/usr/bin/env python3
"""
Script to run routine discovery benchmarks from S3 zip files.

Downloads CDP capture zips from S3, extracts them, runs routine discovery
and evaluation tests, then saves results.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from openai import OpenAI

from bluebox.config import Config
from bluebox.data_models.benchmarks import DeterministicTest, LLMTest, RoutineDiscoveryEvaluation
from bluebox.data_models.routine.routine import Routine
from bluebox.utils.infra_utils import clear_directory, download_zip, extract_zip, remove_directory
from bluebox.utils.terminal_utils import BLUE, CYAN, GREEN, YELLOW, ask_yes_no, print_colored

# OpenAI client for LLM tests
openai_client = OpenAI(api_key=Config.OPENAI_API_KEY)

S3_ZIP_URLS = [
    "https://web-hacker-test-cdp-captures.s3.us-east-1.amazonaws.com/amtrak_search_one_way_routine_discovery.zip",
]


def run_benchmark_evaluation(
    config_path: Path,
    extract_dir: Path,
    llm_model: str,
    run_llm_tests: bool,
    verbose: bool,
) -> RoutineDiscoveryEvaluation:
    """
    Run a complete benchmark evaluation using RoutineDiscoveryEvaluation.run().

    Returns:
        RoutineDiscoveryEvaluation with results
    """
    # Load the test config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create the evaluation object upfront
    evaluation = RoutineDiscoveryEvaluation(
        name=config.get("name", "Unknown"),
        description=config.get("description", ""),
        task=config.get("task", ""),
        ground_truth_routine=Routine.model_validate(config.get("ground_truth_routine", {})),
        deterministic_tests=[DeterministicTest.model_validate(t) for t in config.get("deterministic_tests", [])],
        llm_tests=[LLMTest.model_validate(t) for t in config.get("llm_tests", [])],
    )

    # Use hardcoded paths (assumed to exist)
    cdp_dir = extract_dir / "cdp_captures"

    # Run the full evaluation pipeline
    evaluation.run(
        model=llm_model,
        cdp_captures_dir=str(cdp_dir),
        client=openai_client,
        run_llm_tests=run_llm_tests,
        verbose=verbose,
    )

    return evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Run routine discovery benchmarks from S3 zip files"
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="LLM model to use for routine discovery (default: gpt-5.1)"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Output directory for results (default: benchmarks)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()

    start_timestamp = datetime.now(timezone.utc).isoformat()
    output_dir = Path(args.output_dir)

    print_colored("=" * 60, BLUE)
    print_colored("  S3 Benchmark Runner", BLUE)
    print_colored("=" * 60, BLUE)
    print()
    print(f"  Model: {args.model}")
    print(f"  Output: {output_dir}")
    print(f"  Benchmarks: {len(S3_ZIP_URLS)}")
    print()

    # Check if output directory exists and has content
    if output_dir.exists() and any(output_dir.iterdir()):
        print_colored(f"Output directory '{output_dir}' already contains files.", YELLOW)
        if ask_yes_no("Clear existing data?"):
            clear_directory(output_dir)
            print_colored(f"Cleared {output_dir}", GREEN)
        else:
            print_colored("Keeping existing files. New results will overwrite existing ones.", YELLOW)
        print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each S3 URL
    results_summary = []

    for i, url in enumerate(S3_ZIP_URLS, 1):
        # Extract zip name from URL
        parsed = urlparse(url)
        filename = os.path.basename(parsed.path)
        if filename.endswith(".zip"):
            filename = filename[:-4]
        zip_name = filename

        print_colored(f"\n{'=' * 60}", CYAN)
        print_colored(f"  [{i}/{len(S3_ZIP_URLS)}] {zip_name}", CYAN)
        print_colored(f"{'=' * 60}", CYAN)

        # Create extraction directory in output_dir
        extract_path = output_dir / zip_name
        extract_path.mkdir(parents=True, exist_ok=True)

        # Download zip to a temporary location (just for download, will be deleted)
        with tempfile.TemporaryDirectory(prefix="s3_benchmark_download_") as temp_download:
            zip_path = Path(temp_download) / f"{zip_name}.zip"

            # Download
            if not download_zip(url, zip_path):
                results_summary.append({
                    "name": zip_name,
                    "status": "DOWNLOAD_FAILED",
                    "error": "Failed to download zip file"
                })
                continue

            # Extract directly to output_dir
            if not extract_zip(zip_path, extract_path):
                results_summary.append({
                    "name": zip_name,
                    "status": "EXTRACT_FAILED",
                    "error": "Failed to extract zip file"
                })
                continue

        # Use hardcoded path (assumed to exist)
        config_path = extract_path / "test_configs.json"

        print_colored(f"\n  Running benchmark evaluation...", CYAN)
        print(f"    Config: {config_path}")

        # Run the benchmark evaluation
        try:
            evaluation = run_benchmark_evaluation(
                config_path=config_path,
                extract_dir=extract_path,
                llm_model=args.model,
                run_llm_tests=True,
                verbose=args.verbose,
            )

            # Save results to output_dir (not in extract_path since we'll clean it up)
            results_file = output_dir / f"{zip_name}.json"
            with open(results_file, "w") as f:
                json.dump(evaluation.model_dump(), f, indent=2)

            print_colored(f"\n  Evaluation complete", GREEN)
            print(f"    Results saved to: {results_file}")

            # Build result entry
            summary = evaluation.summary
            status = summary.get("status", "UNKNOWN")
            det_passed = summary.get("deterministic_tests", {}).get("passed", 0)
            det_total = summary.get("deterministic_tests", {}).get("total", 0)
            llm_passed = summary.get("llm_tests", {}).get("passed", 0)
            llm_total = summary.get("llm_tests", {}).get("total", 0)

            result_entry = {
                "name": zip_name,
                "status": status,
                "discovery_duration": evaluation.discovery_duration,
                "error": evaluation.error,
                "deterministic_tests": f"{det_passed}/{det_total}",
                "llm_tests": f"{llm_passed}/{llm_total}",
            }
            results_summary.append(result_entry)

            # Print result immediately
            status_icon = "✓" if status == "SUCCESS" else "✗"
            status_color = GREEN if status == "SUCCESS" else YELLOW
            duration_str = f" ({evaluation.discovery_duration:.1f}s)" if evaluation.discovery_duration else ""
            print_colored(f"\n  {status_icon} {zip_name}{duration_str}", status_color)
            if result_entry["error"]:
                print(f"      Error: {result_entry['error']}")
            print(f"      Deterministic: {result_entry['deterministic_tests']}")
            print(f"      LLM: {result_entry['llm_tests']}")

            # Clean up extracted directory after evaluation is complete
            if extract_path.exists():
                remove_directory(extract_path)

        except Exception as e:
            result_entry = {
                "name": zip_name,
                "status": "EVALUATION_FAILED",
                "discovery_duration": None,
                "error": str(e),
                "deterministic_tests": "0/0",
                "llm_tests": None,
            }
            results_summary.append(result_entry)

            # Print failure immediately
            print_colored(f"\n  ✗ {zip_name}", YELLOW)
            print(f"      Error: {e}")

            # Clean up extracted directory even on failure
            if extract_path.exists():
                remove_directory(extract_path)
            continue

    # Save summary to _summary.json
    end_timestamp = datetime.now(timezone.utc).isoformat()
    summary_file = output_dir / "_summary.json"
    success_count = sum(1 for r in results_summary if r["status"] == "SUCCESS")
    total_count = len(results_summary)

    final_summary = {
        "model": args.model,
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "total": total_count,
        "succeeded": success_count,
        "failed": total_count - success_count,
        "results": results_summary,
    }

    with open(summary_file, "w") as f:
        json.dump(final_summary, f, indent=2)

    print()
    print_colored(f"  Total: {success_count}/{total_count} succeeded", GREEN if success_count == total_count else YELLOW)
    print_colored(f"  Summary saved to: {summary_file}", BLUE)
    print()


if __name__ == "__main__":
    main()
