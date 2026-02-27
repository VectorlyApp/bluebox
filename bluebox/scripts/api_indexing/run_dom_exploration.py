"""
bluebox/scripts/api_indexing/run_dom_exploration.py

Prototype script for Phase 1 (Exploration) — DOM domain.

Runs the DOMSpecialist in autonomous mode with an output schema matching
DOMExplorationSummary. The agent uses DOM tools (list_pages, get_forms,
get_meta_tags, get_scripts, get_hidden_inputs, etc.) to produce a
structured summary of what exists on the page.

Usage:
    python -m bluebox.scripts.api_indexing.run_dom_exploration --cdp-captures-dir ./cdp_captures
    python -m bluebox.scripts.api_indexing.run_dom_exploration --cdp-captures-dir ./cdp_captures --model gpt-5.1
    python -m bluebox.scripts.api_indexing.run_dom_exploration --cdp-captures-dir ./cdp_captures --output /tmp/exploration_dom.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from bluebox.agents.abstract_agent import AutonomousRunConfig
from bluebox.agents.specialists.dom_specialist import DOMSpecialist
from bluebox.workspace import LocalAgentWorkspace
from bluebox.data_models.api_indexing.exploration import DOMExplorationSummary
from bluebox.data_models.llms.interaction import EmittedMessage
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Output schema — derived from DOMExplorationSummary
# ---------------------------------------------------------------------------

DOM_EXPLORATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_snapshots": {
            "type": "integer",
            "description": "Total number of DOM snapshots in the capture.",
        },
        "pages": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one visited page: URL, title, key elements, purpose.",
        },
        "forms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one form: page, action URL, method, fields, purpose.",
        },
        "embedded_tokens": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one token/key in the DOM: location, name, type, size.",
        },
        "data_blobs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one server-side data blob: container, data shape, size.",
        },
        "tables": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one data table: page, columns, row count, data shown.",
        },
        "inferred_framework": {
            "type": "string",
            "description": "Frontend framework inferred from DOM signals (e.g. 'Next.js', 'Angular', 'vanilla/unknown').",
        },
        "narrative": {
            "type": "string",
            "description": "Freeform observations: navigation flow, patterns, anything else worth noting.",
        },
    },
    "required": [
        "total_snapshots",
        "pages",
        "forms",
        "embedded_tokens",
        "data_blobs",
        "tables",
        "inferred_framework",
        "narrative",
    ],
}

# ---------------------------------------------------------------------------
# Autonomous system prompt — DOM exploration
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """
You are a DOM exploration agent. Survey ALL captured browser snapshots
to produce a structured summary of what exists on each page.

## Your Mission

1. **Pages** — what was visited, what forms/inputs/buttons exist on each page
2. **Tokens** — CSRF tokens in meta tags, hidden inputs, API keys in scripts
3. **Data blobs** — __NEXT_DATA__, ld+json, inline JSON config, __NUXT__
4. **Tables** — data tables with columns and row counts
5. **Framework** — infer from DOM signals (Next.js -> __NEXT_DATA__, Angular -> ng-*, ASP.NET -> __VIEWSTATE, React -> data-reactroot, Vue -> data-v-*)

## Process

1. `list_pages` + `get_navigation_sequence` -> survey all pages
2. `get_forms` -> forms with fields and actions
3. `get_meta_tags` + `get_hidden_inputs` -> embedded tokens
4. `get_scripts` -> server-side data blobs and framework signals
5. `get_tables` -> data displays
6. `finalize_with_output(output={...})` with the COMPLETE JSON

## How to Finalize

Pass the ENTIRE output as a single JSON object. Example:

```
finalize_with_output(output={
  "total_snapshots": 5,
  "pages": ["/ (Home) -- landing page with search widget"],
  "forms": ["/book/flights search form -- POST, fields: origin, destination, date"],
  "embedded_tokens": ["meta[name=csrf-token] -- 64-char hex, rotates per load"],
  "data_blobs": ["script#__NEXT_DATA__ -- JSON (~15kb) with feature flags"],
  "tables": [],
  "inferred_framework": "Angular",
  "narrative": "SPA with Angular, search form submits to /api/search..."
})
```

## Guidelines

- Examine ALL snapshots, not just the first one.
- For tokens, note whether they rotate across snapshots or stay static.
- For data blobs, mention interesting keys, not raw content.
- Keep descriptions to one sentence each.
- Sort pages in navigation order.
""".strip()


def _emit_message(msg: EmittedMessage) -> None:
    """Print emitted messages to stderr for visibility."""
    if hasattr(msg, "content") and msg.content:
        print(f"[agent] {msg.content}", file=sys.stderr)
    elif hasattr(msg, "error") and msg.error:
        print(f"[error] {msg.error}", file=sys.stderr)


def _resolve_model(model_str: str) -> LLMModel:
    """Resolve a model string to an LLMModel enum value."""
    for member in OpenAIModel:
        if member.value == model_str:
            return member
    raise ValueError(
        f"Unknown model: {model_str}. "
        f"Available: {[m.value for m in OpenAIModel]}"
    )


def run_dom_exploration(
    cdp_captures_dir: Path,
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    min_iterations: int = 3,
    max_iterations: int = 15,
    workspace_dir: Path | None = None,
) -> DOMExplorationSummary | None:
    """
    Run DOM exploration on a CDP captures directory.

    Args:
        cdp_captures_dir: Path to directory containing dom/events.jsonl.
        llm_model: LLM model to use.
        min_iterations: Minimum iterations before finalize is available.
        max_iterations: Maximum iterations before the loop exits.
        workspace_dir: Workspace directory for artifacts and mounted inputs.

    Returns:
        DOMExplorationSummary if successful, None if the agent failed or timed out.
    """
    dom_jsonl = cdp_captures_dir / "dom" / "events.jsonl"
    if not dom_jsonl.exists():
        logger.error("dom/events.jsonl not found in %s", cdp_captures_dir)
        return None

    # Load DOM data
    dom_loader = DOMDataLoader(jsonl_path=str(dom_jsonl))
    logger.info(
        "Loaded %d DOM snapshots (%d unique URLs, %d unique titles)",
        dom_loader.stats.total_snapshots,
        dom_loader.stats.unique_urls,
        dom_loader.stats.unique_titles,
    )

    workspace = LocalAgentWorkspace.from_directory_path(
        workspace_dir or Path("./agent_workspace/dom_exploration"),
    )
    workspace.attach_input_file("dom_events", dom_jsonl)

    specialist = DOMSpecialist(
        emit_message_callable=_emit_message,
        dom_data_loader=dom_loader,
        llm_model=llm_model,
        workspace=workspace,
    )

    # Bump max_output_tokens for the finalize call
    specialist.llm_client._client.DEFAULT_MAX_TOKENS = 16_384

    # Monkey-patch the autonomous system prompt for exploration
    def _exploration_system_prompt() -> str:
        stats = dom_loader.stats
        context_parts: list[str] = [
            EXPLORATION_SYSTEM_PROMPT,
            f"\n\n## DOM Data Context\n"
            f"- Snapshots: {stats.total_snapshots}\n"
            f"- Unique URLs: {stats.unique_urls}\n"
            f"- Hosts: {', '.join(stats.hosts.keys())}",
        ]

        pages = dom_loader.list_pages()
        if pages:
            page_lines = [f"- [{p['index']}] {p['url']} — {p['title']}" for p in pages[:20]]
            context_parts.append(f"\n\n## Captured Pages\n" + "\n".join(page_lines))

        return (
            "".join(context_parts)
            + specialist._get_output_schema_prompt_section()
            + specialist._get_urgency_notice()
        )

    specialist._get_autonomous_system_prompt = _exploration_system_prompt  # type: ignore[assignment]

    # Override initial message so exploration framing is explicit.
    def _exploration_initial_message(task_text: str) -> str:
        return (
            f"DOM EXPLORATION TASK: {task_text}\n\n"
            "This is broad DOM exploration. Cover pages, forms, embedded tokens, "
            "data blobs, and table structures across snapshots, then finalize with "
            "the complete structured output."
        )

    specialist._get_autonomous_initial_message = _exploration_initial_message  # type: ignore[assignment]

    # Build task message
    task = (
        "Explore ALL DOM snapshots in this capture. Survey pages, scan forms, "
        "find embedded tokens (meta tags, hidden inputs), discover data blobs "
        "(scripts with __NEXT_DATA__, ld+json), examine tables, and infer the "
        f"frontend framework. There are {dom_loader.stats.total_snapshots} snapshots. "
        "Then call finalize_with_output(output={...}) with the COMPLETE JSON."
    )

    config = AutonomousRunConfig(
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )

    result = specialist.run_autonomous(
        task=task,
        config=config,
        output_schema=DOM_EXPLORATION_OUTPUT_SCHEMA,
        output_description=(
            "A DOMExplorationSummary with pages, forms, tokens, data blobs, "
            "tables, framework, and narrative."
        ),
    )

    if result is None:
        logger.warning("DOM exploration did not produce a result (timed out or failed)")
        return None

    if not result.success:
        logger.warning("DOM exploration failed: %s", result.failure_reason)
        return None

    # Parse into our Pydantic model
    try:
        summary = DOMExplorationSummary(**result.output)
        logger.info(
            "DOM exploration complete: %d pages, %d forms, %d tokens, %d blobs, framework: %s",
            len(summary.pages),
            len(summary.forms),
            len(summary.embedded_tokens),
            len(summary.data_blobs),
            summary.inferred_framework,
        )
        return summary
    except Exception as e:
        logger.error("Failed to parse exploration output: %s", e)
        logger.debug("Raw output: %s", result.output)
        return None


def main() -> None:
    """CLI entrypoint for DOM exploration."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 DOM Exploration on CDP captures",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        required=True,
        help="Path to CDP captures directory (expects dom/events.jsonl inside)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.1",
        help="LLM model to use (default: gpt-5.1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("exploration_dom.json"),
        help="Path to write output JSON (default: exploration_dom.json)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("./agent_workspace/dom_exploration"),
        help="Workspace directory for artifacts and mounted inputs.",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=3,
        help="Minimum iterations before finalize is available (default: 3)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum iterations before the loop exits (default: 15)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    if not args.cdp_captures_dir.exists():
        print(f"Error: {args.cdp_captures_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    llm_model = _resolve_model(args.model)

    summary = run_dom_exploration(
        cdp_captures_dir=args.cdp_captures_dir,
        llm_model=llm_model,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        workspace_dir=args.workspace_dir,
    )

    if summary is None:
        print("DOM exploration failed — no output produced.", file=sys.stderr)
        sys.exit(1)

    output_json = summary.model_dump_json(indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"DOM exploration summary written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
