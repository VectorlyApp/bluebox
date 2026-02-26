"""
bluebox/scripts/analyze_pipeline_output.py

Analyze API indexing pipeline output: per-agent tool call traces and usage stats.

Reads the agent_threads/ directory from a pipeline run and produces a summary
showing each agent's tool call trace (ordered) and a counter of how many times
each tool was called.

Usage:
    python -m bluebox.scripts.analyze_pipeline_output \
        --output-dir ./api_indexing_output

    python -m bluebox.scripts.analyze_pipeline_output \
        --output-dir ./api_indexing_output --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _load_agent_thread(path: Path) -> dict[str, Any]:
    """Load a single agent thread JSON file."""
    return json.loads(path.read_text())


def _extract_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ordered tool calls from an agent's message history."""
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                calls.append({
                    "tool_name": tc["tool_name"],
                    "call_id": tc.get("call_id", ""),
                    "arguments": tc.get("arguments", {}),
                })
    return calls


def _extract_tool_results(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map call_id -> truncated result content for tool responses."""
    results: dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "tool" and msg.get("tool_call_id"):
            content = msg.get("content", "")
            # Truncate long results
            if len(content) > 200:
                content = content[:200] + "..."
            results[msg["tool_call_id"]] = content
    return results


# ---------------------------------------------------------------------------
# Overflow / persistence detection
# ---------------------------------------------------------------------------

_OVERFLOW_KEYWORDS = [
    ("overflow", "persist_mode: overflow"),
    ("always_persist", "persist_mode: always"),
    ("output_too_large", "output too large"),
    ("body_truncated", "response_body_truncated: true"),
    ("result_truncated", "truncated: true"),
]


def _classify_overflow(content: str) -> dict[str, Any] | None:
    """Detect if a tool result was overflowed / persisted / truncated.

    Returns a dict with classification details, or None if nothing notable.
    """
    # Skip false positives: response_body_truncated: false
    for kind, marker in _OVERFLOW_KEYWORDS:
        if marker not in content:
            continue
        # Guard: "truncated: true" can appear as "response_body_truncated: true"
        # which we already handle with the body_truncated kind
        if kind == "result_truncated" and "response_body_truncated" in content:
            # Only flag if there's a standalone "truncated: true" outside of
            # "response_body_truncated"
            stripped = content.replace("response_body_truncated: true", "").replace(
                "response_body_truncated: false", ""
            )
            if "truncated: true" not in stripped:
                continue

        info: dict[str, Any] = {"kind": kind}

        # Extract artifact path if present
        if "artifact_path:" in content:
            idx = content.index("artifact_path:")
            line = content[idx:].split("\\n")[0].split("\n")[0]
            info["artifact_path"] = line.split(":", 1)[1].strip().rstrip('"')

        # Extract artifact_id if present
        if "artifact_id:" in content:
            idx = content.index("artifact_id:")
            line = content[idx:].split("\\n")[0].split("\n")[0]
            info["artifact_id"] = line.split(":", 1)[1].strip().rstrip('"')

        return info

    return None


def _extract_overflows(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find all tool results that overflowed / were persisted / truncated.

    Returns a list of dicts, each with: call_id, tool_name, kind, and optional
    artifact_path / artifact_id.
    """
    # Build call_id -> tool_name map from assistant messages
    call_id_to_tool: dict[str, str] = {}
    call_id_to_args: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                cid = tc.get("call_id", "")
                call_id_to_tool[cid] = tc["tool_name"]
                call_id_to_args[cid] = tc.get("arguments", {})

    overflows: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        info = _classify_overflow(content)
        if info is None:
            continue

        call_id = msg.get("tool_call_id", "")
        info["call_id"] = call_id
        info["tool_name"] = call_id_to_tool.get(call_id, "?")
        info["arguments"] = call_id_to_args.get(call_id, {})
        overflows.append(info)

    return overflows


# ---------------------------------------------------------------------------
# PI delegation analysis
# ---------------------------------------------------------------------------


def _analyze_pi_delegations(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Analyze PI's experiment dispatches: output contract usage, etc.

    Only applicable to principal_investigator agents. Returns None for others.
    """
    dispatched_experiments: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            if tc["tool_name"] not in ("dispatch_experiments_batch", "dispatch_experiment"):
                continue
            args = tc.get("arguments", {})
            # dispatch_experiments_batch has a list; dispatch_experiment is a single
            if tc["tool_name"] == "dispatch_experiments_batch":
                experiments = args.get("experiments", [])
            else:
                experiments = [args]

            for exp in experiments:
                dispatched_experiments.append({
                    "hypothesis": exp.get("hypothesis", ""),
                    "has_output_description": bool(exp.get("output_description")),
                    "output_description": exp.get("output_description", ""),
                })

    if not dispatched_experiments:
        return None

    total = len(dispatched_experiments)
    with_description = sum(1 for e in dispatched_experiments if e["has_output_description"])

    return {
        "total_experiments_dispatched": total,
        "with_output_description": with_description,
        "description_usage_rate": f"{with_description / total * 100:.0f}%" if total else "N/A",
        "experiments": dispatched_experiments,
    }


def analyze_agent(thread_data: dict[str, Any]) -> dict[str, Any]:
    """Analyze a single agent thread and return structured stats."""
    label = thread_data["agent_label"]
    messages = thread_data["messages"]
    message_count = thread_data["message_count"]

    tool_calls = _extract_tool_calls(messages)
    tool_results = _extract_tool_results(messages)
    overflows = _extract_overflows(messages)

    # Count tool usage
    tool_counter: Counter[str] = Counter()
    for tc in tool_calls:
        tool_counter[tc["tool_name"]] += 1

    # Build trace with results
    trace: list[dict[str, Any]] = []
    for i, tc in enumerate(tool_calls):
        entry: dict[str, Any] = {
            "step": i + 1,
            "tool_name": tc["tool_name"],
            "call_id": tc["call_id"],
            "arguments": tc["arguments"],
        }
        result = tool_results.get(tc["call_id"])
        if result:
            entry["result_preview"] = result
        trace.append(entry)

    result: dict[str, Any] = {
        "agent_label": label,
        "message_count": message_count,
        "total_tool_calls": len(tool_calls),
        "tool_usage_counts": dict(tool_counter.most_common()),
        "unique_tools_used": len(tool_counter),
        "tool_call_trace": trace,
        "overflows": overflows,
    }

    # PI-specific delegation analysis
    if label.startswith("principal"):
        delegations = _analyze_pi_delegations(messages)
        if delegations:
            result["pi_delegations"] = delegations

    return result


# ---------------------------------------------------------------------------
# Ledger analysis
# ---------------------------------------------------------------------------


def _analyze_ledger(output_dir: Path) -> dict[str, Any] | None:
    """Analyze the discovery ledger for spec/experiment/attempt stats."""
    ledger_path = output_dir / "ledger.json"
    if not ledger_path.exists():
        return None

    data = json.loads(ledger_path.read_text())

    # --- Specs ---
    specs = data.get("routine_specs", [])
    specs_by_status: Counter[str] = Counter()
    spec_details: list[dict[str, Any]] = []
    for spec in specs:
        specs_by_status[spec["status"]] += 1
        spec_attempts = [a for a in data.get("attempts", []) if a["routine_spec_id"] == spec["id"]]
        passed_attempts = [a for a in spec_attempts if a.get("overall_pass")]
        failed_attempts = [a for a in spec_attempts if a.get("status") == "failed"]

        detail: dict[str, Any] = {
            "id": spec["id"],
            "name": spec["name"],
            "description": spec["description"],
            "status": spec["status"],
            "priority": spec["priority"],
            "total_attempts": len(spec_attempts),
            "passed_attempts": len(passed_attempts),
            "failed_attempts": len(failed_attempts),
            "shipped_attempt_id": spec.get("shipped_attempt_id"),
            "failure_reason": spec.get("failure_reason"),
        }

        # Collect blocking issues from failed attempts
        all_blocking: list[str] = []
        for att in failed_attempts:
            all_blocking.extend(att.get("blocking_issues", []))
        if all_blocking:
            detail["blocking_issues"] = all_blocking

        spec_details.append(detail)

    # --- Experiments ---
    experiments = data.get("experiments", [])
    exp_by_status: Counter[str] = Counter()
    exp_by_verdict: Counter[str] = Counter()
    for exp in experiments:
        exp_by_status[exp["status"]] += 1
        verdict = exp.get("verdict") or "none"
        exp_by_verdict[verdict] += 1

    # --- Attempts ---
    attempts = data.get("attempts", [])
    att_by_status: Counter[str] = Counter()
    for att in attempts:
        att_by_status[att["status"]] += 1

    total_passed = sum(1 for a in attempts if a.get("overall_pass"))
    total_failed = sum(1 for a in attempts if a.get("status") == "failed")

    # --- Catalog ---
    catalog = data.get("catalog")
    catalog_summary: dict[str, Any] | None = None
    if catalog:
        catalog_summary = {
            "site": catalog.get("site", ""),
            "shipped_count": len(catalog.get("routines", [])),
            "failed_count": len(catalog.get("failed_routines", [])),
            "total_experiments": catalog.get("total_experiments", 0),
            "total_attempts": catalog.get("total_attempts", 0),
            "routine_names": [r["name"] for r in catalog.get("routines", [])],
            "failed_names": [r.get("name", "?") for r in catalog.get("failed_routines", [])],
            "usage_guide": catalog.get("usage_guide", ""),
        }

    return {
        "user_task": data.get("user_task", ""),
        "specs": {
            "total": len(specs),
            "by_status": dict(specs_by_status.most_common()),
            "details": spec_details,
        },
        "experiments": {
            "total": len(experiments),
            "by_status": dict(exp_by_status.most_common()),
            "by_verdict": dict(exp_by_verdict.most_common()),
        },
        "attempts": {
            "total": len(attempts),
            "by_status": dict(att_by_status.most_common()),
            "passed": total_passed,
            "failed": total_failed,
            "pass_rate": f"{total_passed / len(attempts) * 100:.0f}%" if attempts else "N/A",
        },
        "catalog": catalog_summary,
    }


def analyze_pipeline_output(output_dir: Path) -> dict[str, Any]:
    """Analyze all agent threads in a pipeline output directory."""
    threads_dir = output_dir / "agent_threads"
    if not threads_dir.exists():
        print(f"Error: {threads_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    thread_files = sorted(threads_dir.glob("*.json"))
    if not thread_files:
        print(f"Error: No agent thread files found in {threads_dir}", file=sys.stderr)
        sys.exit(1)

    agents: list[dict[str, Any]] = []
    global_tool_counter: Counter[str] = Counter()
    total_tool_calls = 0

    for tf in thread_files:
        thread_data = _load_agent_thread(tf)
        analysis = analyze_agent(thread_data)
        agents.append(analysis)
        total_tool_calls += analysis["total_tool_calls"]
        for tool_name, count in analysis["tool_usage_counts"].items():
            global_tool_counter[tool_name] += count

    # Classify agents
    pi_agents = [a for a in agents if a["agent_label"].startswith("principal")]
    worker_agents = [a for a in agents if a["agent_label"].startswith("worker")]
    inspector_agents = [a for a in agents if a["agent_label"].startswith("inspector")]

    # Global overflow stats
    global_overflow_counter: Counter[str] = Counter()  # tool_name -> count
    global_overflow_kind_counter: Counter[str] = Counter()  # kind -> count
    total_overflows = 0
    for agent in agents:
        for ov in agent.get("overflows", []):
            global_overflow_counter[ov["tool_name"]] += 1
            global_overflow_kind_counter[ov["kind"]] += 1
            total_overflows += 1

    # Ledger analysis
    ledger_analysis = _analyze_ledger(output_dir)

    return {
        "output_dir": str(output_dir),
        "total_agents": len(agents),
        "total_tool_calls": total_tool_calls,
        "global_tool_usage": dict(global_tool_counter.most_common()),
        "total_overflows": total_overflows,
        "global_overflow_by_tool": dict(global_overflow_counter.most_common()),
        "global_overflow_by_kind": dict(global_overflow_kind_counter.most_common()),
        "agent_breakdown": {
            "principal_investigators": len(pi_agents),
            "workers": len(worker_agents),
            "inspectors": len(inspector_agents),
        },
        "ledger": ledger_analysis,
        "agents": agents,
    }


def print_text_report(result: dict[str, Any]) -> None:
    """Print a human-readable report to stdout."""
    print(f"\n{'=' * 70}")
    print(f"  Pipeline Output Analysis: {result['output_dir']}")
    print(f"{'=' * 70}\n")

    print(f"  Total agents:     {result['total_agents']}")
    breakdown = result["agent_breakdown"]
    print(f"    PI:             {breakdown['principal_investigators']}")
    print(f"    Workers:        {breakdown['workers']}")
    print(f"    Inspectors:     {breakdown['inspectors']}")
    print(f"  Total tool calls: {result['total_tool_calls']}")

    print(f"  Total overflows: {result['total_overflows']}")

    print(f"\n{'─' * 70}")
    print("  Global Tool Usage (all agents combined)")
    print(f"{'─' * 70}")
    for tool_name, count in result["global_tool_usage"].items():
        print(f"    {tool_name:<40} {count:>5}x")

    if result["total_overflows"] > 0:
        print(f"\n{'─' * 70}")
        print("  Global Overflows/Persisted Results")
        print(f"{'─' * 70}")
        print(f"\n  By tool:")
        for tool_name, count in result["global_overflow_by_tool"].items():
            print(f"    {tool_name:<40} {count:>5}x")
        print(f"\n  By kind:")
        for kind, count in result["global_overflow_by_kind"].items():
            label = _overflow_kind_label(kind)
            print(f"    {label:<40} {count:>5}x")

    # Ledger analysis
    ledger = result.get("ledger")
    if ledger:
        print(f"\n{'═' * 70}")
        print("  Discovery Ledger")
        print(f"{'═' * 70}")
        print(f"\n  Task: {ledger['user_task']}")

        # Specs
        specs = ledger["specs"]
        print(f"\n  {'─' * 60}")
        print(f"  Routine Specs ({specs['total']})")
        print(f"  {'─' * 60}")
        print(f"    By status: {specs['by_status']}")
        print()
        for spec in specs["details"]:
            status_icon = {
                "shipped": "+", "failed": "x", "planned": " ",
                "experimenting": "~", "assembling": "~", "validating": "~",
            }.get(spec["status"], "?")
            print(f"    [{status_icon}] {spec['name']}")
            print(f"        status={spec['status']}  priority=P{spec['priority']}  "
                  f"attempts={spec['total_attempts']} (passed={spec['passed_attempts']}, failed={spec['failed_attempts']})")
            print(f"        {spec['description']}")
            if spec.get("shipped_attempt_id"):
                print(f"        shipped_attempt: {spec['shipped_attempt_id']}")
            if spec.get("failure_reason"):
                print(f"        failure: {spec['failure_reason']}")
            if spec.get("blocking_issues"):
                print(f"        blocking issues ({len(spec['blocking_issues'])}):")
                for bi in spec["blocking_issues"]:
                    issue_text = bi if len(bi) <= 100 else bi[:97] + "..."
                    print(f"          - {issue_text}")
            print()

        # Experiments
        exps = ledger["experiments"]
        print(f"  {'─' * 60}")
        print(f"  Experiments ({exps['total']})")
        print(f"  {'─' * 60}")
        print(f"    By status:  {exps['by_status']}")
        print(f"    By verdict: {exps['by_verdict']}")

        # Attempts
        atts = ledger["attempts"]
        print(f"\n  {'─' * 60}")
        print(f"  Routine Attempts ({atts['total']})")
        print(f"  {'─' * 60}")
        print(f"    By status:  {atts['by_status']}")
        print(f"    Passed: {atts['passed']}  Failed: {atts['failed']}  Pass rate: {atts['pass_rate']}")

        # Catalog
        cat = ledger.get("catalog")
        if cat:
            print(f"\n  {'─' * 60}")
            print(f"  Final Catalog")
            print(f"  {'─' * 60}")
            print(f"    Site: {cat['site']}")
            print(f"    Shipped: {cat['shipped_count']}  Failed: {cat['failed_count']}")
            print(f"    Total experiments: {cat['total_experiments']}  Total attempts: {cat['total_attempts']}")
            if cat["routine_names"]:
                print(f"    Shipped routines:")
                for name in cat["routine_names"]:
                    print(f"      [+] {name}")
            if cat["failed_names"]:
                print(f"    Failed routines:")
                for name in cat["failed_names"]:
                    print(f"      [-] {name}")
            if cat.get("usage_guide"):
                guide = cat["usage_guide"]
                if len(guide) > 300:
                    guide = guide[:297] + "..."
                print(f"    Usage guide: {guide}")

    for agent in result["agents"]:
        print(f"\n{'═' * 70}")
        print(f"  Agent: {agent['agent_label']}")
        print(f"{'═' * 70}")
        print(f"  Messages: {agent['message_count']}  |  Tool calls: {agent['total_tool_calls']}  |  Unique tools: {agent['unique_tools_used']}")

        print(f"\n  Tool Usage Counts:")
        for tool_name, count in agent["tool_usage_counts"].items():
            print(f"    {tool_name:<40} {count:>5}x")

        overflows = agent.get("overflows", [])
        if overflows:
            print(f"\n  Overflows / Persisted Results ({len(overflows)}):")
            for ov in overflows:
                kind_label = _overflow_kind_label(ov["kind"])
                args_summary = _summarize_args(ov.get("arguments", {}))
                print(f"    {ov['tool_name']:<35} [{kind_label}]")
                if args_summary:
                    print(f"      args: {args_summary}")
                if ov.get("artifact_path"):
                    print(f"      artifact: {ov['artifact_path']}")

        # PI delegation analysis
        delegations = agent.get("pi_delegations")
        if delegations:
            print(f"\n  PI Experiment Delegations:")
            print(f"    Total dispatched:       {delegations['total_experiments_dispatched']}")
            print(f"    With output_description:{delegations['with_output_description']}  ({delegations['description_usage_rate']})")

            print(f"\n    Experiments:")
            for i, exp in enumerate(delegations["experiments"]):
                hypothesis = exp["hypothesis"]
                if len(hypothesis) > 90:
                    hypothesis = hypothesis[:87] + "..."
                desc_tag = "desc" if exp["has_output_description"] else "no desc"
                print(f"      [{i + 1:>2}] [{desc_tag}] {hypothesis}")

        if "tool_call_trace" in agent:
            print(f"\n  Tool Call Trace:")
            for entry in agent["tool_call_trace"]:
                args_summary = _summarize_args(entry["arguments"])
                # Mark overflowed calls with a tag
                overflow_tag = ""
                for ov in overflows:
                    if ov.get("call_id") == entry.get("call_id", object()):
                        overflow_tag = f" !! {ov['kind']}"
                        break
                print(f"    [{entry['step']:>3}] {entry['tool_name']}{overflow_tag}")
                if args_summary:
                    print(f"          args: {args_summary}")

    print(f"\n{'=' * 70}")
    print("  End of report")
    print(f"{'=' * 70}\n")


_OVERFLOW_KIND_LABELS = {
    "overflow": "overflow (too large, saved to artifact)",
    "always_persist": "always_persist (tool persists by default)",
    "output_too_large": "output_too_large (read_file exceeded line limit)",
    "body_truncated": "body_truncated (HTTP response body cut off)",
    "result_truncated": "result_truncated (tool output cut at char limit)",
}


def _overflow_kind_label(kind: str) -> str:
    """Human-readable label for an overflow kind."""
    return _OVERFLOW_KIND_LABELS.get(kind, kind)


def _summarize_args(args: dict[str, Any]) -> str:
    """Create a short summary of tool call arguments."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 80:
            v_str = v_str[:77] + "..."
        parts.append(f"{k}={v_str}")
    summary = ", ".join(parts)
    if len(summary) > 120:
        summary = summary[:117] + "..."
    return summary


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Analyze API indexing pipeline output: per-agent tool call traces and usage stats.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./api_indexing_output"),
        help="Path to pipeline output directory (default: ./api_indexing_output)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter to a specific agent label (e.g., 'principal_investigator', 'worker_sxv8h9')",
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Omit the full tool call trace (show only counts)",
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: {args.output_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    result = analyze_pipeline_output(args.output_dir)

    # Filter to specific agent if requested
    if args.agent:
        matching = [a for a in result["agents"] if args.agent in a["agent_label"]]
        if not matching:
            print(f"Error: No agent matching '{args.agent}' found", file=sys.stderr)
            available = [a["agent_label"] for a in result["agents"]]
            print(f"Available: {', '.join(available)}", file=sys.stderr)
            sys.exit(1)
        result["agents"] = matching

    # Optionally strip traces
    if args.no_trace:
        for agent in result["agents"]:
            del agent["tool_call_trace"]

    if args.json_output:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_text_report(result)


if __name__ == "__main__":
    main()
