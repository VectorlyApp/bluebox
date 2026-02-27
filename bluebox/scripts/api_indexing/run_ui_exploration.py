"""
bluebox/scripts/api_indexing/run_ui_exploration.py

Prototype script for Phase 1 (Exploration) — UI / Interaction domain.

Runs the InteractionSpecialist in autonomous mode with an output schema
matching UIExplorationSummary. The agent uses interaction tools (form inputs,
clicks, elements) and optionally DOM tools (pages, forms, inputs) to
understand what the user did and infer their intent.

Usage:
    python -m bluebox.scripts.api_indexing.run_ui_exploration --cdp-captures-dir ./cdp_captures
    python -m bluebox.scripts.api_indexing.run_ui_exploration --cdp-captures-dir ./cdp_captures --model gpt-5.1
    python -m bluebox.scripts.api_indexing.run_ui_exploration --cdp-captures-dir ./cdp_captures --output /tmp/exploration_ui.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from bluebox.agents.abstract_agent import AutonomousRunConfig
from bluebox.agents.specialists.interaction_specialist import InteractionSpecialist
from bluebox.workspace import LocalAgentWorkspace
from bluebox.data_models.api_indexing.exploration import UIExplorationSummary
from bluebox.data_models.llms.interaction import EmittedMessage
from bluebox.data_models.llms.vendors import LLMModel, OpenAIModel
from bluebox.llms.data_loaders.dom_data_loader import DOMDataLoader
from bluebox.llms.data_loaders.interactions_data_loader import InteractionsDataLoader
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)

# ---------------------------------------------------------------------------
# Output schema — derived from UIExplorationSummary
# ---------------------------------------------------------------------------

UI_EXPLORATION_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "total_events": {
            "type": "integer",
            "description": "Total number of interaction events in the capture.",
        },
        "user_inputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one user input: element, value, type.",
        },
        "clicks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Each entry describes one meaningful click: element, action, page.",
        },
        "navigation_flow": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered page transitions the user made.",
        },
        "inferred_intent": {
            "type": "string",
            "description": "What the user was trying to accomplish.",
        },
        "narrative": {
            "type": "string",
            "description": "Freeform observations about user behavior and interaction patterns.",
        },
    },
    "required": [
        "total_events",
        "user_inputs",
        "clicks",
        "navigation_flow",
        "inferred_intent",
        "narrative",
    ],
}

# ---------------------------------------------------------------------------
# Autonomous system prompt — UI exploration
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """
You are a UI interaction exploration agent. Your job is to analyze recorded
browser interaction events to understand what the user did on the page.

## Your Mission

1. **User inputs** — what did the user type, select, or fill in?
2. **Clicks** — what buttons, links, or elements did the user click?
3. **Navigation flow** — what pages did the user visit and in what order?
4. **Intent** — what was the user trying to accomplish?

## Process

1. `get_interaction_summary` -> overview of all events
2. `get_form_inputs` -> what the user typed/selected in forms
3. `get_unique_elements` -> which elements were interacted with
4. `search_interactions_by_type(types=["click"])` -> button/link clicks
5. If DOM tools available: `list_pages` + `get_forms` for page context
6. `finalize_with_output(output={...})` with the COMPLETE JSON

## How to Finalize

Pass the ENTIRE output as a single JSON object. Example:

```
finalize_with_output(output={
  "total_events": 42,
  "user_inputs": [
    "input#origin (text) -- user typed 'LAX'",
    "input#destination (text) -- user typed 'JFK'",
    "input#date (date) -- user selected '2025-03-15'"
  ],
  "clicks": [
    "button.search-btn 'Search Flights' -- submits search form",
    "a.result-link 'Select Flight' -- picks first result"
  ],
  "navigation_flow": [
    "/ (Home) -> /book/flights (Search)",
    "/book/flights (Search) -> /results (Results)"
  ],
  "inferred_intent": "Search for one-way flights from LAX to JFK on 2025-03-15",
  "narrative": "User navigated to flight search, filled origin/destination/date, clicked search..."
})
```

## Guidelines

- Focus on meaningful interactions — ignore scroll, hover, focus/blur noise.
- For inputs, include the actual value the user typed/selected.
- For clicks, include the button/link text or identifier.
- Keep descriptions to one sentence each.
- The inferred intent should be a single clear sentence.
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


def run_ui_exploration(
    cdp_captures_dir: Path,
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    min_iterations: int = 3,
    max_iterations: int = 15,
    workspace_dir: Path | None = None,
) -> UIExplorationSummary | None:
    """
    Run UI/interaction exploration on a CDP captures directory.

    Requires interaction data. Optionally loads DOM data for structural context.

    Args:
        cdp_captures_dir: Path to directory containing interactions/events.jsonl.
        llm_model: LLM model to use.
        min_iterations: Minimum iterations before finalize is available.
        max_iterations: Maximum iterations before the loop exits.
        workspace_dir: Workspace directory for artifacts and mounted inputs.

    Returns:
        UIExplorationSummary if successful, None if the agent failed or timed out.
    """
    # Support both "interactions/" and "interaction/" directory names
    interaction_jsonl = cdp_captures_dir / "interactions" / "events.jsonl"
    if not interaction_jsonl.exists():
        interaction_jsonl = cdp_captures_dir / "interaction" / "events.jsonl"
    if not interaction_jsonl.exists():
        logger.error("interactions/events.jsonl not found in %s", cdp_captures_dir)
        return None

    # Load interaction data (required)
    interaction_loader = InteractionsDataLoader.from_jsonl(str(interaction_jsonl))
    logger.info(
        "Loaded %d interaction events (%d unique elements)",
        interaction_loader.stats.total_events,
        interaction_loader.stats.unique_elements,
    )

    # Try to load DOM data (optional, for structural context)
    dom_loader: DOMDataLoader | None = None
    dom_jsonl = cdp_captures_dir / "dom" / "events.jsonl"
    if dom_jsonl.exists():
        dom_loader = DOMDataLoader(jsonl_path=str(dom_jsonl))
        logger.info(
            "Loaded %d DOM snapshots for structural context",
            dom_loader.stats.total_snapshots,
        )
    else:
        logger.info("No DOM data found — will use interaction events only")

    workspace = LocalAgentWorkspace.from_directory_path(
        workspace_dir or Path("./agent_workspace/ui_exploration"),
    )
    workspace.attach_input_file("interaction_events", interaction_jsonl)
    if dom_jsonl.exists():
        workspace.attach_input_file("dom_events", dom_jsonl)

    specialist = InteractionSpecialist(
        emit_message_callable=_emit_message,
        interaction_data_loader=interaction_loader,
        dom_data_loader=dom_loader,
        llm_model=llm_model,
        workspace=workspace,
    )

    # Bump max_output_tokens for the finalize call
    specialist.llm_client._client.DEFAULT_MAX_TOKENS = 16_384

    # Monkey-patch the autonomous system prompt for exploration
    def _exploration_system_prompt() -> str:
        i_stats = interaction_loader.stats
        context_parts: list[str] = [
            EXPLORATION_SYSTEM_PROMPT,
            f"\n\n## Interaction Data Context\n"
            f"- Events: {i_stats.total_events}\n"
            f"- Unique URLs: {i_stats.unique_urls}\n"
            f"- Unique Elements: {i_stats.unique_elements}\n"
            f"- Events by Type: {i_stats.events_by_type}",
        ]

        if dom_loader:
            d_stats = dom_loader.stats
            context_parts.append(
                f"\n\n## DOM Data (for context)\n"
                f"- Snapshots: {d_stats.total_snapshots}\n"
                f"- Unique URLs: {d_stats.unique_urls}\n"
                f"- Use DOM tools (list_pages, get_forms) for structural context."
            )

        return (
            "".join(context_parts)
            + specialist._get_output_schema_prompt_section()
            + specialist._get_urgency_notice()
        )

    specialist._get_autonomous_system_prompt = _exploration_system_prompt  # type: ignore[assignment]

    # Override initial message so exploration framing is explicit.
    def _exploration_initial_message(task_text: str) -> str:
        return (
            f"UI EXPLORATION TASK: {task_text}\n\n"
            "This is broad interaction exploration. Reconstruct inputs, clicks, "
            "navigation flow, and user intent across all events, then finalize with "
            "the complete structured output."
        )

    specialist._get_autonomous_initial_message = _exploration_initial_message  # type: ignore[assignment]

    # Build task message
    task = (
        "Analyze ALL interaction events in this capture. Discover what the user "
        "typed, clicked, and selected. Map the navigation flow. Infer what the "
        f"user was trying to accomplish. There are {interaction_loader.stats.total_events} "
        "interaction events. "
        "Then call finalize_with_output(output={...}) with the COMPLETE JSON."
    )

    config = AutonomousRunConfig(
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )

    result = specialist.run_autonomous(
        task=task,
        config=config,
        output_schema=UI_EXPLORATION_OUTPUT_SCHEMA,
        output_description=(
            "A UIExplorationSummary with user inputs, clicks, navigation flow, "
            "inferred intent, and narrative."
        ),
    )

    if result is None:
        logger.warning("UI exploration did not produce a result (timed out or failed)")
        return None

    if not result.success:
        logger.warning("UI exploration failed: %s", result.failure_reason)
        return None

    # Parse into our Pydantic model
    try:
        summary = UIExplorationSummary(**result.output)
        logger.info(
            "UI exploration complete: %d inputs, %d clicks, intent: %s",
            len(summary.user_inputs),
            len(summary.clicks),
            summary.inferred_intent[:80] if summary.inferred_intent else "(none)",
        )
        return summary
    except Exception as e:
        logger.error("Failed to parse exploration output: %s", e)
        logger.debug("Raw output: %s", result.output)
        return None


def main() -> None:
    """CLI entrypoint for UI exploration."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 UI/Interaction Exploration on CDP captures",
    )
    parser.add_argument(
        "--cdp-captures-dir",
        type=Path,
        required=True,
        help="Path to CDP captures directory (expects interactions/events.jsonl inside)",
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
        default=Path("exploration_ui.json"),
        help="Path to write output JSON (default: exploration_ui.json)",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path("./agent_workspace/ui_exploration"),
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

    summary = run_ui_exploration(
        cdp_captures_dir=args.cdp_captures_dir,
        llm_model=llm_model,
        min_iterations=args.min_iterations,
        max_iterations=args.max_iterations,
        workspace_dir=args.workspace_dir,
    )

    if summary is None:
        print("UI exploration failed — no output produced.", file=sys.stderr)
        sys.exit(1)

    output_json = summary.model_dump_json(indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"UI exploration summary written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
