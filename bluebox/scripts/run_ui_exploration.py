"""
bluebox/scripts/run_ui_exploration.py

Prototype script for Phase 1 (Exploration) — UI domain (DOM + Interactions).

Runs the DOMSpecialist in autonomous mode with an output schema matching
UIExplorationSummary. The agent uses its existing tools (list_pages, get_forms,
get_inputs, get_buttons, get_meta_tags, get_scripts, get_hidden_inputs, etc.)
to explore captured DOM snapshots and produce a structured exploration summary.
Interaction events are summarized and injected into the system prompt context
so the agent can cross-reference what the user did with what the page shows.

Usage:
    python -m bluebox.scripts.run_ui_exploration --cdp-captures-dir ./cdp_captures
    python -m bluebox.scripts.run_ui_exploration --cdp-captures-dir ./cdp_captures --model gpt-5.1
    python -m bluebox.scripts.run_ui_exploration --cdp-captures-dir ./cdp_captures --output /tmp/exploration_ui.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from bluebox.agents.specialists.abstract_specialist import AutonomousConfig, RunMode
from bluebox.agents.specialists.dom_specialist import DOMSpecialist
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
        "total_snapshots": {
            "type": "integer",
            "description": "Total number of DOM snapshots in the capture.",
        },
        "total_interaction_events": {
            "type": "integer",
            "description": "Total number of UI interaction events in the capture.",
        },
        "pages": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each entry is a freeform description of one visited page. "
                "Include: URL, page title, what forms/inputs/buttons exist, "
                "any embedded tokens (CSRF in meta tags, hidden inputs), "
                "any server-side data blobs (__NEXT_DATA__, inline JSON)."
            ),
        },
        "user_inputs": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Each entry is a freeform description of one user input action. "
                "Include: what element was interacted with (tag, id/name, label), "
                "what value was entered or selected, what type of input it is."
            ),
        },
        "narrative": {
            "type": "string",
            "description": (
                "Freeform observations: user flow/journey, page transitions, "
                "form submission patterns, anything else worth noting."
            ),
        },
    },
    "required": [
        "total_snapshots",
        "total_interaction_events",
        "pages",
        "user_inputs",
        "narrative",
    ],
}

# ---------------------------------------------------------------------------
# Autonomous system prompt override — exploration-specific
# ---------------------------------------------------------------------------

EXPLORATION_SYSTEM_PROMPT = """
You are a UI exploration agent. Your job is to survey ALL captured DOM snapshots
and UI interaction events to build a complete picture of the user's journey
through the web application.

## Your Mission

Explore every page snapshot and cross-reference with interaction data. Find two things:

1. **Pages** — what pages were visited, what forms/inputs/buttons exist on each,
   what tokens or data blobs are embedded in the HTML (CSRF tokens in meta tags,
   hidden inputs, __NEXT_DATA__ blobs, inline JSON config).
2. **User inputs** — what the user actually typed, selected, or clicked. These
   are the actions that become routine parameters. Focus on form inputs, typed
   values, dropdown selections, date pickers, and meaningful button clicks.

Everything else (scroll events, hover, focus/blur without input, framework noise,
analytics scripts) is noise — don't catalog it.

## Process

1. **Survey pages**: Use `list_pages` to see all captured pages and their URLs/titles.
2. **Scan each page**: For each page, use `get_forms`, `get_inputs`, `get_buttons`
   to understand what interactive elements exist.
3. **Check for embedded tokens**: Use `get_meta_tags` to find CSRF tokens, API
   configs, verification keys. Use `get_hidden_inputs` to find hidden form tokens
   and session IDs.
4. **Check for data blobs**: Use `get_scripts` to find __NEXT_DATA__, inline JSON,
   structured data (ld+json), and other server-side state embedded in the page.
5. **Cross-reference interactions**: The interaction event summary is provided in
   context below. Match interaction events to the DOM elements you found — which
   inputs did the user fill in? What buttons did they click?
6. **Track navigation**: Use `get_navigation_sequence` to understand the page flow.
7. **Finalize**: Call `finalize_with_output(output={...})` with the COMPLETE JSON.

## CRITICAL: How to Finalize

When you call `finalize_with_output`, you MUST pass the ENTIRE output as a single
JSON object in the `output` parameter. Do NOT call it empty or with partial data.

```
finalize_with_output(output={
  "total_snapshots": 5,
  "total_interaction_events": 23,
  "pages": [
    "/search (Flight Search) — form with origin (text input), destination (text input), date (date picker), passengers (dropdown) + 'Search Flights' submit button. Hidden input 'csrf_token' with value rotated per page load.",
    "/results (Search Results) — table with flight options (airline, time, price). 'Select' button per row. No forms."
  ],
  "user_inputs": [
    "input#origin (text) — user typed 'LAX', likely an airport code parameter",
    "input#destination (text) — user typed 'JFK', likely an airport code parameter",
    "input#date (date) — user selected '2025-03-15', departure date parameter",
    "button.search-btn — user clicked 'Search Flights' to submit the search form"
  ],
  "narrative": "User flow: landed on /search, filled in origin/destination/date, clicked search, arrived at /results. The search form POSTs to /api/search. CSRF token found in hidden input. No __NEXT_DATA__ or framework blobs."
})
```

## Guidelines

- Be thorough — scan ALL pages, not just the first one.
- For pages: describe what's ON the page (forms, inputs, buttons, embedded tokens),
  not just that it exists.
- For user inputs: describe what the user DID, not just that the element exists.
  "User typed 'LAX' into origin field" is useful. "Input field exists" is not.
- The narrative should tell the story of the user's journey through the app.
- Cross-reference DOM elements with interaction events to paint the full picture.
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


def _build_interaction_context(interaction_loader: InteractionsDataLoader) -> str:
    """Build a summary of interaction events for the system prompt."""
    stats = interaction_loader.stats
    lines: list[str] = [
        "\n\n## Interaction Events Context",
        f"- Total Events: {stats.total_events}",
        f"- Unique URLs: {stats.unique_urls}",
        f"- Unique Elements: {stats.unique_elements}",
    ]

    if stats.events_by_type:
        type_parts = ", ".join(
            f"{t}: {c}" for t, c in sorted(stats.events_by_type.items(), key=lambda x: -x[1])
        )
        lines.append(f"- Events by Type: {type_parts}")

    # Include form input details so the agent can cross-reference
    form_inputs = interaction_loader.get_form_inputs()
    if form_inputs:
        lines.append(f"\n### User Input Events ({len(form_inputs)} total)")
        for inp in form_inputs[:50]:
            tag = inp.get("tag_name", "?")
            elem_id = inp.get("element_id", "")
            name = inp.get("element_name", "")
            value = inp.get("value", "")
            event_type = inp.get("type", "")
            identifier = elem_id or name or "unknown"
            lines.append(f"- {tag}#{identifier} ({event_type}): value='{value}'")
        if len(form_inputs) > 50:
            lines.append(f"  ... and {len(form_inputs) - 50} more input events")

    return "\n".join(lines)


def run_ui_exploration(
    cdp_captures_dir: Path,
    llm_model: LLMModel = OpenAIModel.GPT_5_1,
    min_iterations: int = 3,
    max_iterations: int = 15,
) -> UIExplorationSummary | None:
    """
    Run UI exploration on a CDP captures directory.

    Args:
        cdp_captures_dir: Path to directory containing dom/events.jsonl
            and optionally interactions/events.jsonl.
        llm_model: LLM model to use.
        min_iterations: Minimum iterations before finalize is available.
        max_iterations: Maximum iterations before the loop exits.

    Returns:
        UIExplorationSummary if successful, None if the agent failed or timed out.
    """
    # Load DOM snapshots — required
    dom_jsonl = cdp_captures_dir / "dom" / "events.jsonl"
    if not dom_jsonl.exists():
        logger.error("dom/events.jsonl not found in %s", cdp_captures_dir)
        return None

    dom_loader = DOMDataLoader(jsonl_path=str(dom_jsonl))
    logger.info(
        "Loaded %d DOM snapshots (%d unique URLs)",
        dom_loader.stats.total_snapshots,
        dom_loader.stats.unique_urls,
    )

    # Load interaction events — optional but very useful
    interaction_loader: InteractionsDataLoader | None = None
    interaction_jsonl = cdp_captures_dir / "interactions" / "events.jsonl"
    if interaction_jsonl.exists():
        interaction_loader = InteractionsDataLoader(jsonl_path=str(interaction_jsonl))
        logger.info(
            "Loaded %d interaction events (%d unique elements)",
            interaction_loader.stats.total_events,
            interaction_loader.stats.unique_elements,
        )
    else:
        logger.warning(
            "interactions/events.jsonl not found — UI exploration will proceed "
            "with DOM snapshots only (no user input data)"
        )

    # Build the specialist — uses DOMSpecialist for rich page structure tools
    specialist = DOMSpecialist(
        emit_message_callable=_emit_message,
        dom_data_loader=dom_loader,
        llm_model=llm_model,
        run_mode=RunMode.AUTONOMOUS,
    )

    # Bump max_output_tokens for the finalize call
    specialist.llm_client._client.DEFAULT_MAX_TOKENS = 16_384

    # Monkey-patch the autonomous system prompt for exploration
    def _exploration_system_prompt() -> str:
        dom_stats = dom_loader.stats
        context_parts: list[str] = [
            EXPLORATION_SYSTEM_PROMPT,
            "\n\n## DOM Data Context",
            f"\n- Snapshots: {dom_stats.total_snapshots}",
            f"\n- Unique URLs: {dom_stats.unique_urls}",
            f"\n- Unique Titles: {dom_stats.unique_titles}",
            f"\n- Hosts: {', '.join(dom_stats.hosts.keys())}",
        ]

        if dom_stats.urls:
            urls_list = "\n".join(f"  - {url}" for url in dom_stats.urls[:20])
            context_parts.append(f"\n- Pages:\n{urls_list}")

        if interaction_loader:
            context_parts.append(_build_interaction_context(interaction_loader))
        else:
            context_parts.append(
                "\n\n## Interaction Events Context\n"
                "- No interaction events available. Focus on DOM structure only."
            )

        return (
            "".join(context_parts)
            + specialist._get_output_schema_prompt_section()
            + specialist._get_urgency_notice()
        )

    specialist._get_autonomous_system_prompt = _exploration_system_prompt  # type: ignore[assignment]

    # Build task message
    interaction_count = interaction_loader.stats.total_events if interaction_loader else 0
    task = (
        "Explore ALL DOM snapshots and UI interaction data in this capture. "
        "Map out every page: what forms, inputs, buttons, and embedded tokens exist. "
        "Cross-reference with interaction events to identify what the user actually did. "
        f"There are {dom_loader.stats.total_snapshots} DOM snapshots and "
        f"{interaction_count} interaction events. "
        "Then call finalize_with_output(output={...}) with a COMPLETE JSON object containing: "
        "total_snapshots (int), total_interaction_events (int), pages (array of strings), "
        "user_inputs (array of strings), narrative (string)."
    )

    config = AutonomousConfig(
        min_iterations=min_iterations,
        max_iterations=max_iterations,
    )

    result = specialist.run_autonomous(
        task=task,
        config=config,
        output_schema=UI_EXPLORATION_OUTPUT_SCHEMA,
        output_description=(
            "A UIExplorationSummary with all discovered pages and user inputs, "
            "plus a narrative of the user's journey."
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
            "UI exploration complete: %d pages, %d user inputs discovered",
            len(summary.pages),
            len(summary.user_inputs),
        )
        return summary
    except Exception as e:
        logger.error("Failed to parse exploration output: %s", e)
        logger.debug("Raw output: %s", result.output)
        return None


def main() -> None:
    """CLI entrypoint for UI exploration."""
    parser = argparse.ArgumentParser(
        description="Run Phase 1 UI Exploration on CDP captures",
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
        default=Path("exploration_ui.json"),
        help="Path to write output JSON (default: exploration_ui.json)",
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
        import logging
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
    )

    if summary is None:
        print("Exploration failed — no output produced.", file=sys.stderr)
        sys.exit(1)

    output_json = summary.model_dump_json(indent=2)

    if args.output:
        args.output.write_text(output_json)
        print(f"Exploration summary written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
