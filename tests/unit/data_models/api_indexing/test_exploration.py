"""
tests/unit/data_models/api_indexing/test_exploration.py

Tests for exploration summary models: NetworkExplorationSummary,
StorageExplorationSummary, DOMExplorationSummary, UIExplorationSummary.
"""

import pytest

from bluebox.data_models.api_indexing.exploration import (
    DOMExplorationSummary,
    EndpointCategory,
    EndpointCluster,
    InterestLevel,
    NetworkExplorationSummary,
    StorageExplorationSummary,
    UIExplorationSummary,
)


class TestEndpointCategory:
    """Tests for EndpointCategory enum."""

    def test_all_values(self) -> None:
        expected = {"action", "data", "auth", "navigation"}
        assert {c.value for c in EndpointCategory} == expected


class TestInterestLevel:
    """Tests for InterestLevel enum."""

    def test_all_values(self) -> None:
        expected = {"high", "medium", "low"}
        assert {l.value for l in InterestLevel} == expected


class TestEndpointCluster:
    """Tests for EndpointCluster."""

    def test_creation(self) -> None:
        cluster = EndpointCluster(
            url_pattern="/api/v2/search",
            method="POST",
            category=EndpointCategory.ACTION,
            hit_count=3,
            description="Train route search",
            interest=InterestLevel.HIGH,
            request_ids=["req-1", "req-2", "req-3"],
        )
        assert cluster.url_pattern == "/api/v2/search"
        assert cluster.method == "POST"
        assert cluster.category == EndpointCategory.ACTION
        assert cluster.hit_count == 3
        assert cluster.interest == InterestLevel.HIGH
        assert len(cluster.request_ids) == 3

    def test_defaults(self) -> None:
        cluster = EndpointCluster(
            url_pattern="/api/data",
            method="GET",
            category=EndpointCategory.DATA,
            hit_count=1,
            description="Get data",
            interest=InterestLevel.LOW,
        )
        assert cluster.request_ids == []


class TestNetworkExplorationSummary:
    """Tests for NetworkExplorationSummary."""

    def test_minimal(self) -> None:
        summary = NetworkExplorationSummary(total_requests=150)
        assert summary.total_requests == 150
        assert summary.endpoints == []
        assert summary.auth_observations == []
        assert summary.narrative == ""

    def test_full(self) -> None:
        cluster = EndpointCluster(
            url_pattern="/api/search",
            method="POST",
            category=EndpointCategory.ACTION,
            hit_count=2,
            description="Search endpoint",
            interest=InterestLevel.HIGH,
        )
        summary = NetworkExplorationSummary(
            total_requests=200,
            endpoints=[cluster],
            auth_observations=["Bearer JWT on all /api/ requests"],
            narrative="Standard REST API with JWT auth",
        )
        assert len(summary.endpoints) == 1
        assert summary.endpoints[0].interest == InterestLevel.HIGH
        assert "Bearer" in summary.auth_observations[0]

    def test_roundtrip(self) -> None:
        cluster = EndpointCluster(
            url_pattern="/api/data",
            method="GET",
            category=EndpointCategory.DATA,
            hit_count=5,
            description="Data endpoint",
            interest=InterestLevel.MEDIUM,
        )
        summary = NetworkExplorationSummary(
            total_requests=100,
            endpoints=[cluster],
        )
        data = summary.model_dump()
        summary2 = NetworkExplorationSummary.model_validate(data)
        assert summary2.total_requests == 100
        assert summary2.endpoints[0].url_pattern == "/api/data"


class TestStorageExplorationSummary:
    """Tests for StorageExplorationSummary."""

    def test_minimal(self) -> None:
        summary = StorageExplorationSummary(total_events=50, noise_filtered=30)
        assert summary.total_events == 50
        assert summary.noise_filtered == 30
        assert summary.tokens == []
        assert summary.data_blocks == []

    def test_with_tokens_and_data(self) -> None:
        summary = StorageExplorationSummary(
            total_events=100,
            noise_filtered=70,
            tokens=[
                "sessionStorage[auth_token] -- JWT (~1.2kb), written once on page load",
                "cookie[csrf_token] -- 64-char hex, rotates per request",
            ],
            data_blocks=[
                "localStorage[user_profile] -- JSON (~2kb) with name, email, subscription tier",
            ],
            narrative="Standard JWT auth with CSRF protection",
        )
        assert len(summary.tokens) == 2
        assert len(summary.data_blocks) == 1
        assert "JWT" in summary.tokens[0]


class TestDOMExplorationSummary:
    """Tests for DOMExplorationSummary."""

    def test_minimal(self) -> None:
        summary = DOMExplorationSummary(total_snapshots=5)
        assert summary.total_snapshots == 5
        assert summary.pages == []
        assert summary.forms == []
        assert summary.embedded_tokens == []
        assert summary.data_blobs == []
        assert summary.tables == []
        assert summary.inferred_framework == ""

    def test_full(self) -> None:
        summary = DOMExplorationSummary(
            total_snapshots=8,
            pages=["/book/flights -- flight search page with form"],
            forms=["/book/flights search form -- POST to /api/search, fields: origin, destination, date"],
            embedded_tokens=["meta[name=csrf-token] -- 64-char hex, rotates per page load"],
            data_blobs=["script#__NEXT_DATA__ -- JSON (~15kb) with session data"],
            tables=["results table -- columns: Departure, Arrival, Price. 12 rows"],
            inferred_framework="Next.js",
            narrative="Server-side rendered with client hydration",
        )
        assert len(summary.pages) == 1
        assert len(summary.forms) == 1
        assert summary.inferred_framework == "Next.js"


class TestUIExplorationSummary:
    """Tests for UIExplorationSummary."""

    def test_minimal(self) -> None:
        summary = UIExplorationSummary(total_events=25)
        assert summary.total_events == 25
        assert summary.user_inputs == []
        assert summary.clicks == []
        assert summary.navigation_flow == []
        assert summary.inferred_intent == ""

    def test_full(self) -> None:
        summary = UIExplorationSummary(
            total_events=40,
            user_inputs=[
                "input#origin (text) -- user typed 'LAX'",
                "input#destination (text) -- user typed 'JFK'",
            ],
            clicks=[
                "button.search-btn 'Search Flights' on /book/flights -- submits search form",
            ],
            navigation_flow=[
                "/ (Home) -> /book/flights (Search) -> /results (Results)",
            ],
            inferred_intent="Search for one-way flights from LAX to JFK",
            narrative="Straightforward search flow, no pagination",
        )
        assert len(summary.user_inputs) == 2
        assert len(summary.clicks) == 1
        assert "LAX" in summary.inferred_intent

    def test_roundtrip(self) -> None:
        summary = UIExplorationSummary(
            total_events=10,
            inferred_intent="Book a train ticket",
        )
        data = summary.model_dump()
        summary2 = UIExplorationSummary.model_validate(data)
        assert summary2.inferred_intent == "Book a train ticket"
