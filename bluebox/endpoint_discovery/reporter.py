"""
bluebox/endpoint_discovery/reporter.py

Output formatters for endpoint discovery results.

Supports:
- JSON: Full structured output
- Summary: Human-readable console output
- Table: Compact tabular format
"""

import json
from typing import TextIO

from bluebox.endpoint_discovery.models import EndpointDiscoveryResult


def write_json(result: EndpointDiscoveryResult, output: TextIO, include_snippets: bool = False) -> None:
    """
    Write full results as JSON.

    Args:
        result: The discovery result to serialize.
        output: File-like object to write to.
        include_snippets: If False, strip raw_snippet from call sites to reduce size.
    """
    data = result.model_dump(mode="json")

    if not include_snippets:
        for endpoint in data.get("endpoints", []):
            for site in endpoint.get("call_sites", []):
                site.pop("raw_snippet", None)
        for site in data.get("call_sites", []):
            site.pop("raw_snippet", None)

    json.dump(data, output, indent=2, ensure_ascii=False)
    output.write("\n")


def write_summary(result: EndpointDiscoveryResult, output: TextIO) -> None:
    """
    Write a human-readable summary of discovered endpoints.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("ENDPOINT DISCOVERY RESULTS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"JS files analyzed: {result.total_js_files_analyzed} ({result.total_unique_js_files} unique)")
    lines.append(f"Call sites found:  {result.total_call_sites_found}")
    lines.append(f"Endpoints found:   {result.total_endpoints_discovered}")
    lines.append("")

    if result.call_type_breakdown:
        lines.append("Call type breakdown:")
        for call_type, count in sorted(result.call_type_breakdown.items(), key=lambda x: -x[1]):
            lines.append(f"  {call_type:20s} {count}")
        lines.append("")

    if result.js_hosts_analyzed:
        lines.append("JS hosts analyzed:")
        for host in result.js_hosts_analyzed:
            lines.append(f"  {host}")
        lines.append("")

    lines.append("-" * 70)
    lines.append("DISCOVERED ENDPOINTS")
    lines.append("-" * 70)
    lines.append("")

    # Group endpoints by tag
    tagged: dict[str, list] = {}
    for ep in result.endpoints:
        primary_tag = ep.tags[0] if ep.tags else "other"
        tagged.setdefault(primary_tag, []).append(ep)

    for tag, endpoints in sorted(tagged.items()):
        lines.append(f"[{tag.upper()}] ({len(endpoints)} endpoints)")
        lines.append("")

        for ep in endpoints:
            methods_str = ",".join(ep.methods)
            conf_str = f"{ep.confidence:.0%}"
            lines.append(f"  {methods_str:8s} {ep.url_pattern}")
            if ep.base_url:
                lines.append(f"           Base: {ep.base_url}")
            if ep.query_params:
                params_str = ", ".join(f"{k}={v}" for k, v in ep.query_params.items())
                lines.append(f"           Params: {params_str}")
            if ep.headers:
                for hk, hv in ep.headers.items():
                    lines.append(f"           Header: {hk}: {hv[:80]}")
            if ep.body_schema:
                keys_str = ", ".join(ep.body_schema.keys())
                lines.append(f"           Body keys: {keys_str}")
            if ep.response_type:
                lines.append(f"           Response: {ep.response_type}")
            lines.append(f"           Confidence: {conf_str} | Sources: {len(ep.call_sites)}")
            lines.append("")

    output.write("\n".join(lines))


def write_table(result: EndpointDiscoveryResult, output: TextIO) -> None:
    """
    Write a compact table of discovered endpoints.
    """
    lines: list[str] = []

    # Header
    lines.append(f"{'METHOD':<8} {'CONFIDENCE':<11} {'TAG':<10} {'ENDPOINT'}")
    lines.append("-" * 100)

    for ep in result.endpoints:
        methods = ",".join(ep.methods)
        conf = f"{ep.confidence:.0%}"
        tag = ep.tags[0] if ep.tags else ""
        lines.append(f"{methods:<8} {conf:<11} {tag:<10} {ep.url_pattern[:120]}")

    lines.append("-" * 100)
    lines.append(f"Total: {result.total_endpoints_discovered} endpoints from {result.total_call_sites_found} call sites")

    output.write("\n".join(lines))
    output.write("\n")
