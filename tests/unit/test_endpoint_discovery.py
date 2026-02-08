"""
tests/unit/test_endpoint_discovery.py

Unit tests for the endpoint discovery module.
Tests extractors, variable resolver, and the full analyzer pipeline.
"""

import json
import os
import tempfile

import pytest

from bluebox.endpoint_discovery.extractors import (
    _extract_object_keys,
    _extract_string_value,
    _find_balanced_braces,
    _find_balanced_parens,
    _is_asset_url,
    _is_noise_url,
    _templatize_url,
    extract_all_call_sites,
    extract_axios_calls,
    extract_fetch_calls,
    extract_jquery_calls,
    extract_url_strings,
    extract_xhr_calls,
)
from bluebox.endpoint_discovery.models import CallType, EndpointCallSite
from bluebox.endpoint_discovery.resolver import VariableResolver
from bluebox.endpoint_discovery.analyzer import EndpointAnalyzer


# ---------------------------------------------------------------------------
# Helper: create a temporary JSONL file with JS entries
# ---------------------------------------------------------------------------

def _make_jsonl(entries: list[dict]) -> str:
    """Write entries to a temporary JSONL file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return path


def _js_entry(
    body: str,
    url: str = "https://example.com/app.js",
    request_id: str = "test-001",
) -> dict:
    """Create a minimal JS JSONL entry."""
    return {
        "timestamp": 1700000000.0,
        "request_id": request_id,
        "url": url,
        "method": "GET",
        "status": 200,
        "response_body": body,
        "mime_type": "application/javascript",
        "failed": False,
    }


# ===========================================================================
# Extractor helper tests
# ===========================================================================

class TestExtractorHelpers:
    def test_extract_string_value_double_quotes(self) -> None:
        assert _extract_string_value('"hello"') == "hello"

    def test_extract_string_value_single_quotes(self) -> None:
        assert _extract_string_value("'hello'") == "hello"

    def test_extract_string_value_backticks(self) -> None:
        assert _extract_string_value('`hello ${name}`') == "hello ${name}"

    def test_extract_string_value_none_for_non_string(self) -> None:
        assert _extract_string_value("variable") is None

    def test_find_balanced_parens_simple(self) -> None:
        source = 'fetch("url", {method: "POST"})'
        result = _find_balanced_parens(source, 5)
        assert result == '"url", {method: "POST"}'

    def test_find_balanced_parens_nested(self) -> None:
        source = 'fn(a(b), c)'
        result = _find_balanced_parens(source, 2)
        assert result == 'a(b), c'

    def test_find_balanced_braces_simple(self) -> None:
        source = '{key: "value", nested: {a: 1}}'
        result = _find_balanced_braces(source, 0)
        assert result == 'key: "value", nested: {a: 1}'

    def test_extract_object_keys(self) -> None:
        obj_src = '"Content-Type": "application/json", Accept: "text/html", count: 5'
        keys = _extract_object_keys(obj_src)
        assert "Content-Type" in keys
        assert "Accept" in keys
        assert "count" in keys

    def test_is_asset_url(self) -> None:
        assert _is_asset_url("https://example.com/bundle.js") is True
        assert _is_asset_url("https://example.com/style.css") is True
        assert _is_asset_url("https://example.com/logo.png") is True
        assert _is_asset_url("https://example.com/api/v1/users") is False
        assert _is_asset_url("/v1/data") is False

    def test_is_noise_url(self) -> None:
        assert _is_noise_url("http://www.w3.org/2000/svg") is True
        assert _is_noise_url("https://facebook.com") is True
        assert _is_noise_url("https://example.com") is True
        assert _is_noise_url("https://api.example.com/v1/users") is False
        assert _is_noise_url("/v1/competitions/123/matches") is False

    def test_templatize_url_with_template_expressions(self) -> None:
        url = "/v1/competitions/${e}/seasons/${t}"
        result = _templatize_url(url)
        assert "${" not in result
        assert "{competitionId}" in result
        assert "{seasonId}" in result

    def test_templatize_url_strips_function_calls(self) -> None:
        url = "/v1/competitions/${e}/seasons/${t}/players${Kd(n)}"
        result = _templatize_url(url)
        assert "Kd" not in result
        assert "{competitionId}" in result

    def test_templatize_url_preserves_static_segments(self) -> None:
        url = "/v1/matches/commentary"
        result = _templatize_url(url)
        assert result == "/v1/matches/commentary"


# ===========================================================================
# Fetch extractor tests
# ===========================================================================

class TestFetchExtractor:
    def test_simple_fetch_with_string_url(self) -> None:
        source = 'const data = await fetch("https://api.example.com/users");'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].call_type == CallType.FETCH
        assert sites[0].url_resolved == "https://api.example.com/users"
        assert sites[0].method == "GET"

    def test_fetch_with_template_literal(self) -> None:
        source = 'const r = await fetch(`${BASE_URL}/api/v1/users/${userId}`);'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert "${BASE_URL}" in (sites[0].url_resolved or "")
        assert "${userId}" in (sites[0].url_resolved or "")

    def test_fetch_with_post_method(self) -> None:
        source = 'fetch("https://api.example.com/login", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({username: u, password: p})})'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "POST"
        assert sites[0].headers is not None
        assert "Content-Type" in sites[0].headers
        assert sites[0].body_schema is not None
        assert "username" in sites[0].body_schema

    def test_fetch_with_credentials(self) -> None:
        source = 'fetch("/api/data", {credentials: "include"})'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].credentials == "include"

    def test_fetch_response_json(self) -> None:
        source = 'const r = await fetch("/api/data"); const d = await r.json();'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].response_handling == ".json()"

    def test_fetch_response_text(self) -> None:
        source = 'const r = await fetch("/api/data"); const t = await r.text();'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].response_handling == ".text()"

    def test_multiple_fetch_calls(self) -> None:
        source = 'fetch("/api/a"); fetch("/api/b"); fetch("/api/c");'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 3

    def test_fetch_with_variable_url(self) -> None:
        source = 'const url = "/api/users"; const r = await fetch(url);'
        sites = extract_fetch_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].url_raw == "url"


# ===========================================================================
# XHR extractor tests
# ===========================================================================

class TestXHRExtractor:
    def test_xhr_open_get(self) -> None:
        source = 'var xhr = new XMLHttpRequest(); xhr.open("GET", "https://api.example.com/data"); xhr.send();'
        sites = extract_xhr_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].call_type == CallType.XHR
        assert sites[0].method == "GET"
        assert sites[0].url_resolved == "https://api.example.com/data"

    def test_xhr_open_post_with_headers(self) -> None:
        source = '''
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/api/submit");
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.setRequestHeader("Authorization", "Bearer token123");
        xhr.send(JSON.stringify({data: value}));
        '''
        sites = extract_xhr_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "POST"
        assert sites[0].headers is not None
        assert "Content-Type" in sites[0].headers
        assert "Authorization" in sites[0].headers

    def test_no_xhr_when_not_present(self) -> None:
        source = 'fetch("/api/data");'
        sites = extract_xhr_calls(source, "test.js", "req-1")
        assert len(sites) == 0


# ===========================================================================
# Axios extractor tests
# ===========================================================================

class TestAxiosExtractor:
    def test_axios_get(self) -> None:
        source = 'const res = await axios.get("https://api.example.com/users");'
        sites = extract_axios_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].call_type == CallType.AXIOS
        assert sites[0].method == "GET"
        assert sites[0].url_resolved == "https://api.example.com/users"

    def test_axios_post_with_data(self) -> None:
        source = 'axios.post("/api/users", {name: "John", email: "john@example.com"});'
        sites = extract_axios_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "POST"
        assert sites[0].body_raw is not None

    def test_axios_delete(self) -> None:
        source = 'await axios.delete(`/api/users/${id}`);'
        sites = extract_axios_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "DELETE"

    def test_http_client_get(self) -> None:
        source = 'const res = await http.get("/api/config");'
        sites = extract_axios_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "GET"


# ===========================================================================
# jQuery extractor tests
# ===========================================================================

class TestJQueryExtractor:
    def test_jquery_ajax(self) -> None:
        source = '$.ajax({url: "/api/data", type: "POST", data: {key: "value"}});'
        sites = extract_jquery_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].call_type == CallType.JQUERY_AJAX
        assert sites[0].method == "POST"

    def test_jquery_get(self) -> None:
        source = '$.get("/api/info");'
        sites = extract_jquery_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "GET"

    def test_jquery_post(self) -> None:
        source = '$.post("/api/submit", {name: "test"});'
        sites = extract_jquery_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "POST"

    def test_jquery_getjson(self) -> None:
        source = '$.getJSON("/api/data.json");'
        sites = extract_jquery_calls(source, "test.js", "req-1")
        assert len(sites) == 1
        assert sites[0].method == "GET"


# ===========================================================================
# URL string extractor tests
# ===========================================================================

class TestURLStringExtractor:
    def test_api_path_detection(self) -> None:
        source = 'const endpoint = "/api/v2/users/list"; const other = "/api/v1/config";'
        sites = extract_url_strings(source, "test.js", "req-1")
        urls = [s.url_resolved for s in sites]
        assert "/api/v2/users/list" in urls
        assert "/api/v1/config" in urls

    def test_full_url_detection(self) -> None:
        source = 'const url = "https://api.example.com/v1/data/fetch";'
        sites = extract_url_strings(source, "test.js", "req-1")
        assert len(sites) >= 1
        assert any("api.example.com" in (s.url_resolved or "") for s in sites)

    def test_skips_asset_urls(self) -> None:
        source = 'const script = "https://cdn.example.com/bundle.min.js";'
        sites = extract_url_strings(source, "test.js", "req-1")
        assert all(".js" not in (s.url_resolved or "") for s in sites)

    def test_skips_noise_urls(self) -> None:
        source = 'const ns = "http://www.w3.org/2000/svg"; const fb = "https://facebook.com";'
        sites = extract_url_strings(source, "test.js", "req-1")
        assert len(sites) == 0


# ===========================================================================
# Variable resolver tests
# ===========================================================================

class TestVariableResolver:
    def test_resolve_string_assignment(self) -> None:
        source = 'const BASE_URL = "https://api.example.com"; fetch(BASE_URL + "/users");'
        resolver = VariableResolver(source)
        binding = resolver.resolve_variable("BASE_URL")
        assert binding is not None
        assert binding.value == "https://api.example.com"

    def test_resolve_expression_literal(self) -> None:
        source = 'const x = "hello";'
        resolver = VariableResolver(source)
        result = resolver.resolve_expression('"world"')
        assert result == "world"

    def test_resolve_expression_variable(self) -> None:
        source = 'const apiUrl = "https://api.example.com";'
        resolver = VariableResolver(source)
        result = resolver.resolve_expression("apiUrl")
        assert result == "https://api.example.com"

    def test_resolve_concatenation(self) -> None:
        source = 'const base = "https://api.example.com"; const path = "/v1/users";'
        resolver = VariableResolver(source)
        result = resolver.resolve_expression('base + path')
        assert result == "https://api.example.com/v1/users"

    def test_resolve_template_literal(self) -> None:
        source = 'const host = "api.example.com";'
        resolver = VariableResolver(source)
        result = resolver.resolve_expression('`https://${host}/data`')
        assert result == "https://api.example.com/data"

    def test_resolve_property_access(self) -> None:
        source = 'const config = {baseUrl: "https://api.example.com", timeout: 5000};'
        resolver = VariableResolver(source)
        binding = resolver.resolve_variable("config.baseUrl")
        assert binding is not None
        assert binding.value == "https://api.example.com"

    def test_get_url_config_values(self) -> None:
        source = '''
        const API_URL = "https://api.example.com";
        const BASE_URL = "https://cdn.example.com";
        const name = "John";
        const imgUrl = "https://images.example.com/logo.png";
        '''
        resolver = VariableResolver(source)
        configs = resolver.get_all_url_config_values()
        assert "API_URL" in configs
        assert "BASE_URL" in configs
        assert "imgUrl" in configs  # starts with https://

    def test_resolve_url_at_callsite(self) -> None:
        source = 'const base = "https://api.example.com"; function load() { const endpoint = "/users"; fetch(base + endpoint); }'
        resolver = VariableResolver(source)
        url, vars_used = resolver.resolve_url_at_callsite('base + endpoint', len(source) - 10)
        assert url is not None
        assert "api.example.com" in url


# ===========================================================================
# extract_all_call_sites integration test
# ===========================================================================

class TestExtractAllCallSites:
    def test_deduplication_of_overlapping_sites(self) -> None:
        """URL strings found near fetch calls should not be double-counted."""
        source = 'await fetch("https://api.example.com/v1/users"); console.log("done");'
        sites = extract_all_call_sites(source, "test.js", "req-1")
        # Should have the fetch site but not a duplicate URL string site
        fetch_sites = [s for s in sites if s.call_type == CallType.FETCH]
        assert len(fetch_sites) == 1

    def test_mixed_call_types(self) -> None:
        source = '''
        fetch("/api/a");
        var xhr = new XMLHttpRequest(); xhr.open("POST", "/api/b");
        $.get("/api/c");
        const url = "/api/v1/data/endpoint";
        '''
        sites = extract_all_call_sites(source, "test.js", "req-1")
        call_types = {s.call_type for s in sites}
        assert CallType.FETCH in call_types
        assert CallType.XHR in call_types
        assert CallType.JQUERY_AJAX in call_types


# ===========================================================================
# Full analyzer pipeline test
# ===========================================================================

class TestEndpointAnalyzer:
    def test_full_pipeline(self) -> None:
        """End-to-end test: JSONL -> analysis -> structured result."""
        js_code = '''
        const BASE = "https://api.example.com";
        async function loadUsers() {
            const resp = await fetch(BASE + "/v1/users", {
                method: "GET",
                headers: {"Authorization": "Bearer token123"}
            });
            return await resp.json();
        }
        async function createUser(data) {
            return fetch("https://api.example.com/v1/users", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({name: data.name, email: data.email})
            });
        }
        async function deleteUser(id) {
            return fetch(`https://api.example.com/v1/users/${id}`, {method: "DELETE"});
        }
        '''
        path = _make_jsonl([_js_entry(js_code)])
        try:
            analyzer = EndpointAnalyzer(path)
            result = analyzer.analyze()

            assert result.total_js_files_analyzed == 1
            assert result.total_call_sites_found >= 3

            # Check we found the /v1/users endpoint
            api_endpoints = [ep for ep in result.endpoints if '/v1/users' in ep.path]
            assert len(api_endpoints) >= 1

            # Check methods were detected
            all_methods = set()
            for ep in api_endpoints:
                all_methods.update(ep.methods)
            assert "POST" in all_methods or "GET" in all_methods

            # Verify result model can serialize
            json_output = result.model_dump_json()
            assert len(json_output) > 100
        finally:
            os.unlink(path)

    def test_deduplication_by_url(self) -> None:
        """Duplicate JS URLs should only be analyzed once."""
        js_code = 'fetch("https://api.example.com/v1/data");'
        entries = [
            _js_entry(js_code, url="https://example.com/app.js", request_id="req-1"),
            _js_entry(js_code, url="https://example.com/app.js", request_id="req-2"),
        ]
        path = _make_jsonl(entries)
        try:
            analyzer = EndpointAnalyzer(path, deduplicate_js=True)
            result = analyzer.analyze()
            assert result.total_js_files_analyzed == 1
        finally:
            os.unlink(path)

    def test_no_deduplication(self) -> None:
        """When dedup is off, all entries are analyzed."""
        js_code = 'fetch("https://api.example.com/v1/data");'
        entries = [
            _js_entry(js_code, url="https://example.com/app.js", request_id="req-1"),
            _js_entry(js_code, url="https://example.com/app.js", request_id="req-2"),
        ]
        path = _make_jsonl(entries)
        try:
            analyzer = EndpointAnalyzer(path, deduplicate_js=False)
            result = analyzer.analyze()
            assert result.total_js_files_analyzed == 2
        finally:
            os.unlink(path)

    def test_first_party_host_filter(self) -> None:
        """Only JS from specified hosts should be analyzed."""
        entries = [
            _js_entry('fetch("/api/a");', url="https://mysite.com/app.js", request_id="req-1"),
            _js_entry('fetch("/api/b");', url="https://cdn.tracker.io/analytics.js", request_id="req-2"),
        ]
        path = _make_jsonl(entries)
        try:
            analyzer = EndpointAnalyzer(path, first_party_hosts=["mysite.com"])
            result = analyzer.analyze()
            assert result.total_js_files_analyzed == 1
            assert any("/api/a" in s.url_raw for s in result.call_sites)
        finally:
            os.unlink(path)

    def test_file_not_found(self) -> None:
        """Should raise FileNotFoundError for missing JSONL."""
        with pytest.raises(FileNotFoundError):
            EndpointAnalyzer("/nonexistent/path.jsonl")

    def test_empty_jsonl(self) -> None:
        """Should handle empty JSONL gracefully."""
        path = _make_jsonl([])
        try:
            analyzer = EndpointAnalyzer(path)
            result = analyzer.analyze()
            assert result.total_endpoints_discovered == 0
        finally:
            os.unlink(path)


# ===========================================================================
# Real data integration test (only runs if capture data exists)
# ===========================================================================

_REAL_DATA_PATH = "cdp_captures/network/javascript_events.jsonl"


@pytest.mark.skipif(
    not os.path.exists(_REAL_DATA_PATH),
    reason="Real capture data not available",
)
class TestRealDataIntegration:
    def test_analyzes_real_captures(self) -> None:
        """Smoke test against real captured JS data."""
        analyzer = EndpointAnalyzer(
            _REAL_DATA_PATH,
            skip_third_party=True,
        )
        result = analyzer.analyze()

        # Should find a reasonable number of endpoints
        assert result.total_endpoints_discovered > 10
        assert result.total_call_sites_found > 20

        # Should find Premier League API endpoints
        api_paths = [ep.path for ep in result.endpoints if 'api' in ep.tags]
        assert any('/v1/competitions' in p for p in api_paths)
        assert any('/v1/matches' in p for p in api_paths)

        # Should have breakdown by call type
        assert len(result.call_type_breakdown) >= 1
        assert result.call_type_breakdown.get("fetch", 0) > 0

    def test_first_party_filtering(self) -> None:
        """Test that first-party host filtering reduces noise."""
        all_result = EndpointAnalyzer(_REAL_DATA_PATH).analyze()
        filtered_result = EndpointAnalyzer(
            _REAL_DATA_PATH,
            first_party_hosts=["www.premierleague.com"],
        ).analyze()

        # Filtered should have fewer but still meaningful results
        assert filtered_result.total_js_files_analyzed <= all_result.total_js_files_analyzed
        assert filtered_result.total_endpoints_discovered > 10
