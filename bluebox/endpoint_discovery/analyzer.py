"""
bluebox/endpoint_discovery/analyzer.py

Main orchestrator for the endpoint discovery pipeline.

Pipeline:
1. Load JS files from JSONL (deduplicate by URL)
2. For each file: run all extractors to find HTTP call sites
3. For each call site: run variable resolver to improve URL resolution
4. Aggregate and deduplicate into canonical endpoints
5. Classify and score each endpoint

The entire pipeline is deterministic — no LLM calls, no randomness.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs, urlencode

from bluebox.endpoint_discovery.extractors import extract_all_call_sites, _classify_url, _compute_confidence
from bluebox.endpoint_discovery.models import (
    CallType,
    DiscoveredEndpoint,
    EndpointCallSite,
    EndpointDiscoveryResult,
)
from bluebox.endpoint_discovery.resolver import VariableResolver
from bluebox.utils.logger import get_logger

logger = get_logger(name=__name__)


class EndpointAnalyzer:
    """
    Analyzes JavaScript bundle files to discover API endpoints.

    Usage:
        analyzer = EndpointAnalyzer("cdp_captures/network/javascript_events.jsonl")
        result = analyzer.analyze()
        print(result.model_dump_json(indent=2))
    """

    def __init__(
        self,
        jsonl_path: str,
        deduplicate_js: bool = True,
        skip_third_party: bool = False,
        first_party_hosts: list[str] | None = None,
    ) -> None:
        """
        Initialize the analyzer.

        Args:
            jsonl_path: Path to javascript_events.jsonl file.
            deduplicate_js: If True, only analyze each unique JS URL once.
            skip_third_party: If True, skip JS files from known third-party hosts.
            first_party_hosts: If provided, only analyze JS from these hosts.
        """
        self._jsonl_path = Path(jsonl_path)
        self._deduplicate_js = deduplicate_js
        self._skip_third_party = skip_third_party
        self._first_party_hosts = set(first_party_hosts) if first_party_hosts else None

        if not self._jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    def analyze(self) -> EndpointDiscoveryResult:
        """
        Run the full endpoint discovery pipeline.

        Returns:
            EndpointDiscoveryResult with all discovered endpoints and metadata.
        """
        logger.info("Starting endpoint discovery from %s", self._jsonl_path)

        # Step 1: Load and deduplicate JS files
        js_files = self._load_js_files()
        logger.info("Loaded %d JS files (%d unique)", len(js_files), len(set(f['url'] for f in js_files)))

        # Step 2: Extract call sites from all files
        all_call_sites: list[EndpointCallSite] = []
        for js_file in js_files:
            sites = self._extract_from_file(js_file)
            all_call_sites.extend(sites)

        logger.info("Extracted %d raw call sites", len(all_call_sites))

        # Step 3: Aggregate into unique endpoints
        endpoints = self._aggregate_endpoints(all_call_sites)
        logger.info("Aggregated into %d unique endpoints", len(endpoints))

        # Step 4: Build result
        hosts = sorted(set(urlparse(f['url']).netloc for f in js_files))
        call_type_breakdown: dict[str, int] = {}
        for site in all_call_sites:
            call_type_breakdown[site.call_type] = call_type_breakdown.get(site.call_type, 0) + 1

        result = EndpointDiscoveryResult(
            endpoints=endpoints,
            call_sites=all_call_sites,
            total_js_files_analyzed=len(js_files),
            total_unique_js_files=len(set(f['url'] for f in js_files)),
            total_call_sites_found=len(all_call_sites),
            total_endpoints_discovered=len(endpoints),
            js_hosts_analyzed=hosts,
            call_type_breakdown=call_type_breakdown,
        )

        return result

    def _load_js_files(self) -> list[dict[str, Any]]:
        """Load JS entries from the JSONL file."""
        entries: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        with open(self._jsonl_path, mode="r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed JSON at line %d", line_num + 1)
                    continue

                url = data.get('url', '')
                body = data.get('response_body', '')
                request_id = data.get('request_id', f'line-{line_num}')

                # Skip empty bodies
                if not body:
                    continue

                # Deduplicate by URL
                if self._deduplicate_js and url in seen_urls:
                    continue
                seen_urls.add(url)

                # Host filtering
                host = urlparse(url).netloc
                if self._first_party_hosts and host not in self._first_party_hosts:
                    continue
                if self._skip_third_party and self._is_third_party(host):
                    continue

                entries.append({
                    'url': url,
                    'request_id': request_id,
                    'body': body,
                    'host': host,
                })

        return entries

    def _is_third_party(self, host: str) -> bool:
        """Check if a host is a known third-party analytics/tracking service."""
        third_party_patterns = [
            'google-analytics', 'googletagmanager', 'googleapis.com',
            'facebook.net', 'fbcdn.net',
            'twitter.com', 'twimg.com',
            'analytics', 'tracking', 'pixel',
            'chartbeat', 'hotjar', 'mixpanel', 'segment',
            'doubleclick', 'adsense',
            'tiktok.com', 'bytedance',
            'instagram.com',
            'spotify.com',
        ]
        host_lower = host.lower()
        return any(p in host_lower for p in third_party_patterns)

    def _extract_from_file(self, js_file: dict[str, Any]) -> list[EndpointCallSite]:
        """Extract call sites from a single JS file with variable resolution."""
        source = js_file['body']
        source_url = js_file['url']
        request_id = js_file['request_id']

        # Extract raw call sites
        call_sites = extract_all_call_sites(source, source_url, request_id)

        if not call_sites:
            return []

        # Build variable resolver for this file
        resolver = VariableResolver(source)
        resolver.scan()

        # Enhance each call site with resolved variables
        enhanced_sites: list[EndpointCallSite] = []
        for site in call_sites:
            enhanced = self._enhance_with_resolver(site, resolver)
            enhanced_sites.append(enhanced)

        return enhanced_sites

    def _enhance_with_resolver(
        self,
        site: EndpointCallSite,
        resolver: VariableResolver,
    ) -> EndpointCallSite:
        """
        Enhance a call site by attempting to resolve its URL and variables.
        """
        # Try to resolve the URL if it wasn't already fully resolved
        if not site.url_resolved or '${' in (site.url_resolved or ''):
            resolved_url, used_vars = resolver.resolve_url_at_callsite(
                site.url_raw, site.position
            )
            if resolved_url:
                site.url_resolved = resolved_url
                site.variables_used = used_vars

                # Re-templatize with better resolution
                from bluebox.endpoint_discovery.extractors import _templatize_url
                site.url_template = _templatize_url(resolved_url)

        return site

    def _aggregate_endpoints(self, call_sites: list[EndpointCallSite]) -> list[DiscoveredEndpoint]:
        """
        Aggregate call sites into unique endpoint definitions.

        Groups call sites by their URL template (canonical form) and merges
        information from all call sites that reference the same endpoint.
        """
        # Group by URL template or url_resolved
        groups: dict[str, list[EndpointCallSite]] = defaultdict(list)

        for site in call_sites:
            key = self._get_grouping_key(site)
            if key:
                groups[key].append(site)

        endpoints: list[DiscoveredEndpoint] = []
        for key, sites in groups.items():
            endpoint = self._merge_call_sites(key, sites)
            if endpoint:
                endpoints.append(endpoint)

        # Sort by confidence (highest first), then by path
        endpoints.sort(key=lambda e: (-e.confidence, e.path))

        return endpoints

    def _get_grouping_key(self, site: EndpointCallSite) -> str | None:
        """Get a canonical key for grouping call sites into endpoints."""
        # Prefer url_template, fall back to url_resolved, then url_raw
        url = site.url_template or site.url_resolved or site.url_raw

        if not url:
            return None

        # Filter out obviously bad URLs (minified variable names, empty strings, etc.)
        stripped = url.strip().strip('"').strip("'").strip('`')
        if len(stripped) < 3:
            return None
        # Skip if it's just a variable name (no slash, no dot, no colon)
        if re.fullmatch(r'[a-zA-Z_$][\w$]*', stripped):
            return None
        # Skip spread expressions
        if stripped.startswith('...'):
            return None

        # Normalize: strip query params for grouping
        url = url.split('?')[0]

        # Strip fragment
        url = url.split('#')[0]

        # Normalize trailing slash
        url = url.rstrip('/')

        return url

    def _merge_call_sites(self, key: str, sites: list[EndpointCallSite]) -> DiscoveredEndpoint | None:
        """Merge multiple call sites into a single DiscoveredEndpoint."""
        if not sites:
            return None

        # Determine the best URL representation
        best_resolved = None
        best_template = None
        for site in sites:
            if site.url_resolved and (not best_resolved or len(site.url_resolved) > len(best_resolved)):
                best_resolved = site.url_resolved
            if site.url_template and (not best_template or len(site.url_template) > len(best_template)):
                best_template = site.url_template

        url_pattern = best_template or best_resolved or key

        # Parse URL components
        base_url, path, query_params = self._parse_url_components(url_pattern)

        # Merge methods from all call sites
        methods = sorted(set(site.method for site in sites))

        # Merge headers
        merged_headers: dict[str, str] = {}
        for site in sites:
            if site.headers:
                merged_headers.update(site.headers)

        # Merge body schemas
        merged_body: dict[str, Any] = {}
        for site in sites:
            if site.body_schema:
                merged_body.update(site.body_schema)

        # Take first non-None credentials
        credentials = next((s.credentials for s in sites if s.credentials), None)

        # Take first non-None response type
        response_type = next((s.response_handling for s in sites if s.response_handling), None)

        # Compute confidence as average
        confidences = [_compute_confidence(s) for s in sites]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Classify
        tags = _classify_url(url_pattern)

        return DiscoveredEndpoint(
            url_pattern=url_pattern,
            base_url=base_url,
            path=path,
            query_params=query_params or None,
            methods=methods,
            headers=merged_headers or None,
            body_schema=merged_body or None,
            credentials=credentials,
            response_type=response_type,
            call_sites=sites,
            confidence=round(avg_confidence, 2),
            tags=tags,
        )

    def _parse_url_components(self, url: str) -> tuple[str | None, str, dict[str, str] | None]:
        """
        Parse a URL into (base_url, path, query_params).

        Returns:
            Tuple of (base_url or None, path, query_params dict or None)
        """
        # Handle template-style URLs that start with ${...}
        if url.startswith('$'):
            # Try to extract path after the variable
            path_match = re.search(r'(/[^\s?#]+)', url)
            if path_match:
                path = path_match.group(1)
            else:
                path = url
            return None, path, None

        try:
            parsed = urlparse(url)
        except ValueError:
            return None, url, None

        base_url = None
        if parsed.scheme and parsed.netloc:
            base_url = f"{parsed.scheme}://{parsed.netloc}"

        path = parsed.path or '/'

        query_params = None
        if parsed.query:
            params = parse_qs(parsed.query, keep_blank_values=True)
            query_params = {k: v[0] if len(v) == 1 else ','.join(v) for k, v in params.items()}

        return base_url, path, query_params


def analyze_js_bundles(
    jsonl_path: str,
    deduplicate: bool = True,
    skip_third_party: bool = False,
    first_party_hosts: list[str] | None = None,
) -> EndpointDiscoveryResult:
    """
    Convenience function to run the full endpoint discovery pipeline.

    Args:
        jsonl_path: Path to javascript_events.jsonl file.
        deduplicate: Only analyze each unique JS URL once.
        skip_third_party: Skip known third-party tracking/analytics JS.
        first_party_hosts: Only analyze JS from these hosts.

    Returns:
        EndpointDiscoveryResult with all discovered endpoints.
    """
    analyzer = EndpointAnalyzer(
        jsonl_path=jsonl_path,
        deduplicate_js=deduplicate,
        skip_third_party=skip_third_party,
        first_party_hosts=first_party_hosts,
    )
    return analyzer.analyze()
