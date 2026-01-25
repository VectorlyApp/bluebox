"""
tests/unit/test_endpoint.py

Tests for endpoint data model including validation, type checking, and parameter interpolation.
"""

import pytest
from pydantic import ValidationError

from web_hacker.data_models.routine.endpoint import (
    Endpoint,
    HTTPMethod,
    CREDENTIALS,
)


class TestEndpointCreation:
    """Test basic endpoint creation with valid data."""

    def test_create_endpoint_minimal(self) -> None:
        """Test creating an endpoint with minimal required fields."""
        endpoint = Endpoint(
            url="https://api.example.com/users",
            method=HTTPMethod.GET,
            headers={"Content-Type": "application/json"},
            body={}
        )

        assert endpoint.url == "https://api.example.com/users"
        assert endpoint.method == HTTPMethod.GET
        assert endpoint.headers == {"Content-Type": "application/json"}
        assert endpoint.body == {}
        assert endpoint.credentials == CREDENTIALS.SAME_ORIGIN
        assert endpoint.description is None

    def test_create_endpoint_with_all_fields(self) -> None:
        """Test creating an endpoint with all fields."""
        endpoint = Endpoint(
            url="https://api.example.com/users/{{user_id}}",
            description="Get user details by ID",
            method=HTTPMethod.PATCH,
            headers={"Authorization": "Bearer {{token}}"},
            body={"name": "{{new_name}}"},
            credentials=CREDENTIALS.INCLUDE,
        )

        assert endpoint.url == "https://api.example.com/users/{{user_id}}"
        assert endpoint.description == "Get user details by ID"
        assert endpoint.method == HTTPMethod.PATCH
        assert endpoint.headers == {"Authorization": "Bearer {{token}}"}
        assert endpoint.body == {"name": "{{new_name}}"}
        assert endpoint.credentials == CREDENTIALS.INCLUDE


class TestEndpointHTTPMethods:
    """Test all supported HTTP methods."""

    def test_get_method(self) -> None:
        """Test GET method."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={}
        )
        assert endpoint.method == HTTPMethod.GET

    def test_post_method(self) -> None:
        """Test POST method."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.POST,
            headers={},
            body={}
        )
        assert endpoint.method == HTTPMethod.POST

    def test_put_method(self) -> None:
        """Test PUT method."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.PUT,
            headers={},
            body={}
        )
        assert endpoint.method == HTTPMethod.PUT

    def test_delete_method(self) -> None:
        """Test DELETE method."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.DELETE,
            headers={},
            body={}
        )
        assert endpoint.method == HTTPMethod.DELETE

    def test_patch_method(self) -> None:
        """Test PATCH method."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.PATCH,
            headers={},
            body={}
        )
        assert endpoint.method == HTTPMethod.PATCH


class TestEndpointCredentials:
    """Test all supported credentials modes."""

    def test_same_origin_credentials(self) -> None:
        """Test same-origin credentials mode."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={},
            credentials=CREDENTIALS.SAME_ORIGIN
        )
        assert endpoint.credentials == CREDENTIALS.SAME_ORIGIN

    def test_include_credentials(self) -> None:
        """Test include credentials mode."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={},
            credentials=CREDENTIALS.INCLUDE
        )
        assert endpoint.credentials == CREDENTIALS.INCLUDE

    def test_omit_credentials(self) -> None:
        """Test omit credentials mode."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={},
            credentials=CREDENTIALS.OMIT
        )
        assert endpoint.credentials == CREDENTIALS.OMIT

    def test_default_credentials(self) -> None:
        """Test default credentials mode (same-origin)."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={}
        )
        assert endpoint.credentials == CREDENTIALS.SAME_ORIGIN


class TestEndpointValidationErrors:
    """Test validation errors for invalid data."""

    def test_headers_as_string_raises_error(self) -> None:
        """Test that using string for headers raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Endpoint(
                url="https://api.example.com/data",
                method=HTTPMethod.GET,
                headers='{"Authorization": "Bearer token"}',  # str instead of dict
                body={}
            )
        assert "Input should be a valid dictionary" in str(exc_info.value)

    def test_body_as_string_raises_error(self) -> None:
        """Test that using string for body raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            Endpoint(
                url="https://api.example.com/data",
                method=HTTPMethod.POST,
                headers={"Content-Type": "application/json"},
                body='{"name": "value"}'  # String instead of dict
            )
        assert "Input should be a valid dictionary" in str(exc_info.value)

    def test_headers_and_body_as_strings_raises_error(self) -> None:
        """Test that using strings for both headers and body raises validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            Endpoint(
                url="https://api.example.com/data",
                method=HTTPMethod.POST,
                headers='{"Authorization": "Bearer token"}',  # String
                body='{"key": "value"}'  # String
            )
        error_str = str(exc_info.value)
        assert "Input should be a valid dictionary" in error_str

    def test_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError):
            Endpoint(
                method=HTTPMethod.GET,
                headers={},
                body={}
                # Missing url
            )

    def test_invalid_http_method(self) -> None:
        """Test that invalid HTTP method raises validation error."""
        with pytest.raises(ValidationError):
            Endpoint(
                url="https://api.example.com/data",
                method="INVALID",  # Invalid method
                headers={},
                body={}
            )


class TestEndpointParameterInterpolation:
    """Test parameter interpolation in endpoints."""

    def test_parameter_in_url(self) -> None:
        """Test parameter placeholders in URL."""
        endpoint = Endpoint(
            url="https://api.example.com/{{user_id}}/posts/{{post_id}}",
            method=HTTPMethod.GET,
            headers={},
            body={}
        )
        assert "{{user_id}}" in endpoint.url
        assert "{{post_id}}" in endpoint.url

    def test_parameter_in_headers(self) -> None:
        """Test parameter placeholders in headers."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={
                "Authorization": "Bearer {{sessionStorage:auth.token}}",
                "X-User-Id": "{{user_id}}"
            },
            body={}
        )
        assert "{{sessionStorage:auth.token}}" in endpoint.headers["Authorization"]
        assert endpoint.headers["X-User-Id"] == "{{user_id}}"

    def test_parameter_in_body(self) -> None:
        """Test parameter placeholders in body."""
        endpoint = Endpoint(
            url="https://api.example.com/users",
            method=HTTPMethod.POST,
            headers={"Content-Type": "application/json"},
            body={
                "name": "{{user_name}}",
                "email": "{{user_email}}",
                "preferences": "{{localStorage:user.preferences}}"
            }
        )
        assert endpoint.body["name"] == "{{user_name}}"
        assert endpoint.body["email"] == "{{user_email}}"
        assert "{{localStorage:user.preferences}}" in endpoint.body["preferences"]

    def test_nested_parameters(self) -> None:
        """Test nested parameter structures."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.POST,
            headers={
                "Authorization": "Bearer {{token}}",
                "Content-Type": "application/json"
            },
            body={
                "user": {
                    "id": "{{user_id}}",
                    "metadata": {
                        "session": "{{sessionStorage:session.id}}"
                    }
                }
            }
        )
        assert endpoint.body["user"]["id"] == "{{user_id}}"
        assert "{{sessionStorage:session.id}}" in endpoint.body["user"]["metadata"]["session"]

    def test_complex_interpolation(self) -> None:
        """Test complex parameter interpolation with all types."""
        endpoint = Endpoint(
            url="https://api.example.com/{{user_id}}/data",
            method=HTTPMethod.PUT,
            headers={
                "Authorization": "Bearer {{sessionStorage:auth.token}}",
                "X-Timestamp": "{{meta:timestamp}}"
            },
            body={
                "filter": "{{localStorage:user.preferences.filter}}",
                "page": "{{page_number}}"
            }
        )
        # verify all parameter types are present
        assert "{{user_id}}" in endpoint.url
        assert "{{sessionStorage:auth.token}}" in endpoint.headers["Authorization"]
        assert "{{meta:timestamp}}" in endpoint.headers["X-Timestamp"]
        assert "{{localStorage:user.preferences.filter}}" in endpoint.body["filter"]
        assert endpoint.body["page"] == "{{page_number}}"


class TestEndpointEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_headers(self) -> None:
        """Test endpoint with empty headers."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={},
            body={}
        )
        assert endpoint.headers == {}

    def test_empty_body(self) -> None:
        """Test endpoint with empty body."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={"Content-Type": "application/json"},
            body={}
        )
        assert endpoint.body == {}

    def test_multiple_headers(self) -> None:
        """Test endpoint with multiple headers."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.GET,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer token",
                "X-Custom-Header": "value",
                "Cache-Control": "no-cache"
            },
            body={}
        )
        assert len(endpoint.headers) == 4
        assert endpoint.headers["Authorization"] == "Bearer token"

    def test_complex_body_structure(self) -> None:
        """Test endpoint with complex nested body structure."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.POST,
            headers={"Content-Type": "application/json"},
            body={
                "user": {
                    "profile": {
                        "name": "John",
                        "settings": {"theme": "dark"}
                    }
                },
                "metadata": ["item1", "item2"]
            }
        )
        assert endpoint.body["user"]["profile"]["name"] == "John"
        assert endpoint.body["metadata"] == ["item1", "item2"]

