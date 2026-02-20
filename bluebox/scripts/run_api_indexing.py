"""
bluebox/scripts/run_api_indexing.py

End-to-end API indexing pipeline.

Phase 1: Run 4 exploration specialists in parallel to analyze CDP captures
Phase 2: Run the PrincipalInvestigator to build a catalog of routines

Output is written incrementally to disk so every experiment, attempt, and
routine is available for debugging even if the pipeline crashes mid-run.

Usage:
    python -m bluebox.scripts.run_api_indexing \
        --cdp-captures-dir ./cdp_captures \
        --task "Browse Premier League standings and view team details"

    python -m bluebox.scripts.run_api_indexing \
        --cdp-captures-dir ./cdp_captures \
        --task "Search for flights" \
        --skip-exploration \
        --output-dir ./api_indexing_output
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bluebox.agents.principal_investigator import PrincipalInvestigator
from bluebox.data_models.api_indexing.exploration import (
    DOMExplorationSummary,
    NetworkExplorationSummary,
    StorageExplorationSummary,
    UIExplorationSummary,
)
from bluebox.data_models.llms.interaction import EmittedMessage
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.data_models.orchestration.ledger import DiscoveryLedger, RoutineCatalog
from bluebox.llms.data_loaders.documentation_data_loader import DocumentationDataLoader
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.network_data_loader import NetworkDataLoader
from bluebox.llms.data_loaders.storage_data_loader import StorageDataLoader
from bluebox.llms.data_loaders.window_property_data_loader import WindowPropertyDataLoader
from bluebox.scripts.run_dom_exploration import run_dom_exploration
from bluebox.scripts.run_network_exploration import run_network_exploration
from bluebox.scripts.run_storage_exploration import run_storage_exploration
from bluebox.scripts.run_ui_exploration import run_ui_exploration
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

BLUEBOX_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_model(model_str: str) -> LLMModel:
    """Resolve a model string to an LLMModel enum value."""
    for member in OpenAIModel:
        if member.value == model_str:
            return member
    raise ValueError(
        f"Unknown model: {model_str}. "
        f"Available: {[m.value for m in OpenAIModel]}"
    )


def _emit_message(msg: EmittedMessage) -> None:
    """Print emitted messages to stderr."""
    if hasattr(msg, "content") and msg.content:
        print(f"[agent] {msg.content}", file=sys.stderr)
    elif hasattr(msg, "error") and msg.error:
        print(f"[error] {msg.error}", file=sys.stderr)


MAX_PI_ATTEMPTS = 3


def _load_if_exists(loader_cls: type, jsonl_path: Path) -> Any:
    """Load a data loader if its JSONL file exists, else return None."""
    if jsonl_path.exists():
        return loader_cls(jsonl_path=str(jsonl_path))
    logger.info("Skipping %s — %s not found", loader_cls.__name__, jsonl_path)
    return None


def _write_json(path: Path, data: Any) -> None:
    """Write JSON to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Phase 1: Exploration
# ---------------------------------------------------------------------------


def run_explorations(
    cdp_captures_dir: Path,
    output_dir: Path,
    llm_model: LLMModel,
) -> dict[str, str]:
    """
    Run all 4 exploration specialists in parallel.

    Returns:
        Dict mapping domain name → JSON summary string (for PI system prompt).
    """
    exploration_dir = output_dir / "exploration"
    exploration_dir.mkdir(parents=True, exist_ok=True)

    runners = {
        "network": run_network_exploration,
        "storage": run_storage_exploration,
        "dom": run_dom_exploration,
        "ui": run_ui_exploration,
    }

    summaries: dict[str, str] = {}

    print("\n=== Phase 1: Exploration (4 domains in parallel) ===\n", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fn, cdp_captures_dir, llm_model): domain
            for domain, fn in runners.items()
        }

        for future in as_completed(futures):
            domain = futures[future]
            try:
                result = future.result()
                if result is not None:
                    summary_json = result.model_dump_json(indent=2)
                    # Save to disk
                    _write_json(exploration_dir / f"{domain}.json", json.loads(summary_json))
                    summaries[domain] = summary_json
                    print(f"  [+] {domain} exploration complete", file=sys.stderr)
                else:
                    print(f"  [-] {domain} exploration returned no result", file=sys.stderr)
            except Exception as e:
                logger.error("Exploration failed for %s: %s", domain, e)
                print(f"  [!] {domain} exploration failed: {e}", file=sys.stderr)

    print(f"\n  Explorations complete: {len(summaries)}/4 domains\n", file=sys.stderr)
    return summaries


def load_explorations(output_dir: Path) -> dict[str, str]:
    """Load previously saved exploration summaries from disk."""
    exploration_dir = output_dir / "exploration"
    summaries: dict[str, str] = {}

    model_map = {
        "network": NetworkExplorationSummary,
        "storage": StorageExplorationSummary,
        "dom": DOMExplorationSummary,
        "ui": UIExplorationSummary,
    }

    for domain, model_cls in model_map.items():
        path = exploration_dir / f"{domain}.json"
        if path.exists():
            raw = json.loads(path.read_text())
            # Validate it parses
            model_cls(**raw)
            summaries[domain] = json.dumps(raw, indent=2)
            print(f"  [+] Loaded {domain} exploration from {path}", file=sys.stderr)
        else:
            print(f"  [-] No saved {domain} exploration at {path}", file=sys.stderr)

    return summaries


# ---------------------------------------------------------------------------
# Phase 2: PI loop with incremental persistence
# ---------------------------------------------------------------------------


class PipelinePersistence:
    """
    Writes ledger state and agent threads to disk incrementally as the PI works.

    Output structure:
        output_dir/
        ├── experiments/
        │   ├── exp_abc123.json      # Each experiment as its own file
        │   └── exp_def456.json
        ├── attempts/
        │   ├── attempt_xyz789.json  # Each routine attempt
        │   └── ...
        ├── routines/
        │   ├── get_standings.json   # Shipped routine files
        │   └── ...
        ├── agent_threads/
        │   ├── principal_investigator.json  # PI's full conversation
        │   ├── worker_abc123.json           # Worker message histories
        │   └── inspector_get_standings.json # Inspector conversations
        ├── ledger.json              # Full ledger snapshot (overwritten each time)
        └── catalog.json             # Final catalog (written on mark_complete)
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._experiments_dir = output_dir / "experiments"
        self._attempts_dir = output_dir / "attempts"
        self._routines_dir = output_dir / "routines"
        self._threads_dir = output_dir / "agent_threads"

        # Create directories
        for d in [self._experiments_dir, self._attempts_dir, self._routines_dir, self._threads_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def on_ledger_change(self, ledger: DiscoveryLedger, reason: str) -> None:
        """
        Called by the PI after every ledger mutation.

        Writes:
        1. Each experiment as experiments/exp_{id}.json
        2. Each attempt as attempts/attempt_{id}.json
        3. Shipped routines as routines/{name}.json
        4. Full ledger snapshot as ledger.json
        5. Catalog as catalog.json (if built)
        """
        # Individual experiment files
        for exp in ledger.experiments:
            exp_path = self._experiments_dir / f"exp_{exp.id}.json"
            _write_json(exp_path, exp.model_dump())

        # Individual attempt files
        for attempt in ledger.attempts:
            attempt_path = self._attempts_dir / f"attempt_{attempt.id}.json"
            _write_json(attempt_path, attempt.model_dump())

        # Shipped routine files
        for spec in ledger.routine_specs:
            if spec.shipped_attempt_id:
                attempt = ledger.get_attempt(spec.shipped_attempt_id)
                if attempt:
                    routine_path = self._routines_dir / f"{spec.name}.json"
                    _write_json(routine_path, attempt.routine_json)

        # Full ledger snapshot
        _write_json(self._output_dir / "ledger.json", ledger.model_dump())

        # Catalog (if built)
        if ledger.catalog is not None:
            _write_json(self._output_dir / "catalog.json", ledger.catalog.model_dump())

        logger.debug("Persisted ledger to disk (reason: %s)", reason)

    def on_agent_thread(
        self,
        agent_label: str,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """
        Called by the PI after a worker/inspector finishes or PI itself completes.

        Writes the full message history to agent_threads/{label}.json.
        """
        thread_path = self._threads_dir / f"{agent_label}.json"
        _write_json(thread_path, {
            "agent_label": agent_label,
            "thread_id": thread_id,
            "message_count": len(messages),
            "messages": messages,
        })
        logger.debug("Persisted agent thread: %s (%d messages)", agent_label, len(messages))


def run_pi_with_recovery(
    task: str,
    summaries: dict[str, str],
    cdp_captures_dir: Path,
    output_dir: Path,
    llm_model: LLMModel,
    remote_debugging_address: str,
    max_pi_iterations: int,
    min_experiments_before_fail: int = 10,
    num_workers: int = 3,
    num_inspectors: int = 1,
) -> RoutineCatalog | None:
    """
    Run the PrincipalInvestigator with automatic recovery.

    If the PI fails for ANY reason (context exhaustion, API error, etc.),
    it preserves the DiscoveryLedger and spins up a fresh PI to continue.
    Retries up to MAX_PI_ATTEMPTS total (default 3).
    """
    # Build data loaders
    network_loader = _load_if_exists(
        NetworkDataLoader, cdp_captures_dir / "network" / "events.jsonl",
    )
    storage_loader = _load_if_exists(
        StorageDataLoader, cdp_captures_dir / "storage" / "events.jsonl",
    )
    dom_loader = _load_if_exists(
        DOMDataLoader, cdp_captures_dir / "dom" / "events.jsonl",
    )
    window_prop_loader = _load_if_exists(
        WindowPropertyDataLoader, cdp_captures_dir / "window_properties" / "events.jsonl",
    )

    # Documentation loader — gives PI access to Routine schema docs and source code
    docs_dir = str(BLUEBOX_PACKAGE_ROOT / "agent_docs")
    code_paths = [
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "routine"),
        str(BLUEBOX_PACKAGE_ROOT / "data_models" / "ui_elements.py"),
        str(BLUEBOX_PACKAGE_ROOT / "agents" / "routine_discovery_agent.py"),
        str(BLUEBOX_PACKAGE_ROOT / "llms" / "infra" / "data_store.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "js_utils.py"),
        str(BLUEBOX_PACKAGE_ROOT / "utils" / "data_utils.py"),
        "!" + str(BLUEBOX_PACKAGE_ROOT / "**" / "__init__.py"),
    ]
    documentation_data_loader = DocumentationDataLoader(
        documentation_paths=[docs_dir],
        code_paths=code_paths,
    )

    # Persistence layer — writes to disk incrementally
    persistence = PipelinePersistence(output_dir)

    ledger: DiscoveryLedger | None = None
    catalog: RoutineCatalog | None = None

    print("\n=== Phase 2: Routine Construction (PI loop) ===\n", file=sys.stderr)

    for attempt in range(MAX_PI_ATTEMPTS):
        if attempt > 0:
            print(
                f"\n  [!] PI attempt {attempt + 1}/{MAX_PI_ATTEMPTS} "
                f"(fresh PI, ledger preserved)\n",
                file=sys.stderr,
            )

        pi = PrincipalInvestigator(
            emit_message_callable=_emit_message,
            task=task,
            exploration_summaries=summaries,
            network_data_loader=network_loader,
            storage_data_loader=storage_loader,
            dom_data_loader=dom_loader,
            window_property_data_loader=window_prop_loader,
            documentation_data_loader=documentation_data_loader,
            remote_debugging_address=remote_debugging_address,
            llm_model=llm_model,
            ledger=ledger,
            max_iterations=max_pi_iterations,
            min_experiments_before_fail=min_experiments_before_fail,
            num_workers=num_workers,
            num_inspectors=num_inspectors,
            on_ledger_change=persistence.on_ledger_change,
            on_agent_thread=persistence.on_agent_thread,
        )

        try:
            catalog = pi.run()
            break
        except Exception as e:
            logger.error("PI attempt %d/%d failed: %s", attempt + 1, MAX_PI_ATTEMPTS, e)
            # Preserve ledger for next attempt
            ledger = pi._ledger
            if ledger.experiments:
                persistence.on_ledger_change(ledger, f"recovery_attempt_{attempt + 1}")

            if attempt + 1 >= MAX_PI_ATTEMPTS:
                print(
                    f"\n  [!] All {MAX_PI_ATTEMPTS} PI attempts exhausted. "
                    "Returning partial results.\n",
                    file=sys.stderr,
                )
                # Return whatever was shipped before the crash
                return pi._build_partial_catalog()
        finally:
            pi.close()

    return catalog


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_api_indexing(
    cdp_captures_dir: Path,
    task: str,
    output_dir: Path = Path("./api_indexing_output"),
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    remote_debugging_address: str = "http://127.0.0.1:9222",
    skip_exploration: bool = False,
    max_pi_iterations: int = 200,
    min_experiments_before_fail: int = 10,
    num_workers: int = 3,
    num_inspectors: int = 1,
) -> RoutineCatalog | None:
    """
    Run the full API indexing pipeline end-to-end.

    Phase 1: 4 parallel explorations (network, storage, DOM, UI)
    Phase 2: PI loop with experiment workers

    Args:
        cdp_captures_dir: Path to CDP captures directory.
        task: What the user was trying to do.
        output_dir: Where to write output files.
        llm_model: LLM model to use.
        remote_debugging_address: Chrome debugging URL for live browser experiments.
        skip_exploration: Skip Phase 1, load existing summaries from output_dir.
        max_pi_iterations: Max PI loop iterations per session.
        min_experiments_before_fail: Min experiments before PI can call mark_failed.
        num_workers: Max concurrent ExperimentWorker agents (default 3).
        num_inspectors: Max concurrent RoutineInspector agents (default 1).

    Returns:
        RoutineCatalog if successful, None if no routines could be built.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    print(f"\nAPI Indexing Pipeline", file=sys.stderr)
    print(f"  Task: {task}", file=sys.stderr)
    print(f"  Captures: {cdp_captures_dir}", file=sys.stderr)
    print(f"  Output: {output_dir}", file=sys.stderr)
    print(f"  Model: {llm_model.value}", file=sys.stderr)
    print(f"  Browser: {remote_debugging_address}", file=sys.stderr)

    # Phase 1: Exploration
    if skip_exploration:
        print("\n  Skipping exploration (--skip-exploration), loading from disk...", file=sys.stderr)
        summaries = load_explorations(output_dir)
    else:
        summaries = run_explorations(cdp_captures_dir, output_dir, llm_model)

    if not summaries:
        print("\n  [!] No exploration summaries available. Cannot proceed.", file=sys.stderr)
        return None

    # Clean up Phase 2 artifacts from previous runs (preserve exploration/)
    for subdir in ["experiments", "attempts", "routines", "agent_threads"]:
        p = output_dir / subdir
        if p.exists():
            shutil.rmtree(p)
            logger.info("Cleaned up %s", p)
    for f in ["ledger.json", "catalog.json"]:
        p = output_dir / f
        if p.exists():
            p.unlink()
            logger.info("Cleaned up %s", p)

    # Phase 2: PI loop
    catalog = run_pi_with_recovery(
        task=task,
        summaries=summaries,
        cdp_captures_dir=cdp_captures_dir,
        output_dir=output_dir,
        llm_model=llm_model,
        remote_debugging_address=remote_debugging_address,
        max_pi_iterations=max_pi_iterations,
        min_experiments_before_fail=min_experiments_before_fail,
        num_workers=num_workers,
        num_inspectors=num_inspectors,
    )

    elapsed = time.time() - start_time

    # Final summary
    print(f"\n=== Pipeline Complete ({elapsed:.1f}s) ===\n", file=sys.stderr)
    if catalog:
        print(f"  Routines shipped: {len(catalog.routines)}", file=sys.stderr)
        print(f"  Routines failed:  {len(catalog.failed_routines)}", file=sys.stderr)
        print(f"  Experiments run:  {catalog.total_experiments}", file=sys.stderr)
        print(f"  Total attempts:   {catalog.total_attempts}", file=sys.stderr)
        for routine in catalog.routines:
            print(f"    [+] {routine.name} — {routine.description}", file=sys.stderr)
        for failed in catalog.failed_routines:
            print(f"    [-] {failed['name']} — {failed.get('reason', '?')}", file=sys.stderr)
        print(f"\n  Output: {output_dir}", file=sys.stderr)
    else:
        print("  No routines produced.", file=sys.stderr)

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for the API indexing pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full API indexing pipeline: exploration → experimentation → routine catalog",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        required=True,
        help="Path to CDP captures directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task description — what the user was doing in the captured session",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./api_indexing_output"),
        help="Where to write output files (default: ./api_indexing_output)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--remote-debugging-address",
        type=str,
        default="http://127.0.0.1:9222",
        help="Chrome remote debugging address (default: http://127.0.0.1:9222)",
    )
    parser.add_argument(
        "--skip-exploration",
        action="store_true",
        help="Skip Phase 1 exploration, load existing summaries from output-dir",
    )
    parser.add_argument(
        "--max-pi-iterations",
        type=int,
        default=200,
        help="Max PI loop iterations per session (default: 200)",
    )
    parser.add_argument(
        "--min-experiments-before-fail",
        type=int,
        default=10,
        help="Min experiments before PI can abandon the pipeline (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=3,
        help="Max concurrent ExperimentWorker agents (default: 3)",
    )
    parser.add_argument(
        "--num-inspectors",
        type=int,
        default=1,
        help="Max concurrent RoutineInspector agents (default: 1)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    if not args.cdp_captures_dir.exists():
        print(f"Error: {args.cdp_captures_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    llm_model = _resolve_model(args.model)

    catalog = run_api_indexing(
        cdp_captures_dir=args.cdp_captures_dir,
        task=args.task,
        output_dir=args.output_dir,
        llm_model=llm_model,
        remote_debugging_address=args.remote_debugging_address,
        skip_exploration=args.skip_exploration,
        max_pi_iterations=args.max_pi_iterations,
        min_experiments_before_fail=args.min_experiments_before_fail,
        num_workers=args.num_workers,
        num_inspectors=args.num_inspectors,
    )

    if catalog is None:
        print("\nPipeline produced no routines.", file=sys.stderr)
        sys.exit(1)

    # Print catalog JSON to stdout
    print(catalog.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
