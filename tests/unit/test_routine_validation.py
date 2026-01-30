"""
tests/unit/test_routine_validation.py

Tests for routine validation functionality including parameter validation,
interpolation pattern matching, and error handling.
"""

import pytest
from pydantic import ValidationError

from bluebox.data_models.routine.operation import (
    RoutineDownloadOperation,
    RoutineFetchOperation,
    RoutineNavigateOperation,
    RoutineSleepOperation,
    RoutineReturnOperation,
    RoutineJsEvaluateOperation,
)
from bluebox.data_models.routine.routine import Routine
from bluebox.data_models.routine.parameter import Parameter, ParameterType
from bluebox.data_models.routine.endpoint import Endpoint, HTTPMethod
from bluebox.data_models.routine.placeholder import extract_placeholders_from_json_str
from bluebox.utils.data_utils import extract_base_url_from_url


class TestExtractPlaceholdersFromJson:
    """Test the standalone extract_placeholders_from_json_str function."""

    def test_extract_placeholder(self) -> None:
        """Test extracting a simple placeholder."""
        json_string = '{"body": {"n_trials": "{{n_trials}}"}}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert placeholders == ["n_trials"]

    def test_extract_multiple_placeholders(self) -> None:
        """Test extracting multiple placeholders."""
        json_string = '{"url": "https://example.com/{{user_id}}", "body": {"count": "{{count}}"}}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert set(placeholders) == {"user_id", "count"}

    def test_extract_storage_placeholder(self) -> None:
        """Test extracting storage placeholder."""
        json_string = '{"header": "{{sessionStorage:api_key}}"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert placeholders == ["sessionStorage:api_key"]

    def test_extract_builtin_placeholder(self) -> None:
        """Test extracting builtin placeholders."""
        json_string = '{"id": "{{uuid}}", "ts": "{{epoch_milliseconds}}"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert set(placeholders) == {"uuid", "epoch_milliseconds"}

    def test_extract_no_placeholders(self) -> None:
        """Test with no placeholders."""
        json_string = '{"url": "https://example.com/static"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert len(placeholders) == 0

    def test_extract_whitespace_in_placeholder(self) -> None:
        """Test that whitespace inside placeholder is stripped."""
        json_string = '{"url": "{{ user_id }}"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert placeholders == ["user_id"]

    def test_deduplication(self) -> None:
        """Test that duplicate placeholders are deduplicated."""
        json_string = '{"a": "{{x}}", "b": "{{x}}"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert placeholders == ["x"]

    def test_substring_placeholder(self) -> None:
        """Test extracting placeholder from substring context."""
        json_string = '{"date": "{{dep_date}}T00:00:00"}'
        placeholders = extract_placeholders_from_json_str(json_string)
        assert placeholders == ["dep_date"]


class TestRoutineParameterValidation:
    """Test routine parameter validation and interpolation pattern matching."""

    def test_validate_parameter_usage_basic_usage(self, make_routine) -> None:
        """Test basic parameter usage validation."""
        parameters = [
            Parameter(name="user_id", type=ParameterType.STRING, description="User ID"),
            Parameter(name="page", type=ParameterType.STRING, description="Page name")
        ]
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{user_id}}/{{page}}")],
            parameters=parameters
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_undefined_parameter_raises_error(self, make_routine) -> None:
        """Test that using undefined parameters raises validation error."""
        with pytest.raises(ValueError, match="Undefined parameters found in routine"):
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{undefined_param}}")]
            )

    def test_validate_parameter_usage_storage_prefixes(self, make_routine) -> None:
        """Test validation of storage parameter prefixes."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{sessionStorage:user.name}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="selectors.button")
            ]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_invalid_storage_prefix_raises_error(self, make_routine) -> None:
        """Test that invalid storage prefixes raise validation error."""
        with pytest.raises(ValueError, match="Invalid prefix in placeholder: invalidStorage"):
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{invalidStorage:user.name}}")]
            )

    def test_validate_parameter_usage_meta_prefix(self, make_routine) -> None:
        """Test validation of meta parameter prefix."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{meta:timestamp}}")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_mixed_usage(self, make_routine) -> None:
        """Test validation with mixed parameter types."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{user_id}}"),
                RoutineSleepOperation(timeout_seconds=2.0),
                RoutineReturnOperation(session_storage_key="selectors.button"),
                RoutineReturnOperation(session_storage_key="default_text")
            ],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_fetch_operation(self, make_routine) -> None:
        """Test parameter validation in fetch operations."""
        endpoint = Endpoint(
            url="https://api.example.com/{{user_id}}/data",
            method=HTTPMethod.GET,
            headers={"Authorization": "Bearer {{sessionStorage:auth.token}}"},
            body={"filter": "{{localStorage:user.preferences.filter}}"}
        )
        routine = make_routine(
            operations=[RoutineFetchOperation(endpoint=endpoint)],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_nested_json_parameters(self, make_routine) -> None:
        """Test parameter validation in nested JSON structures."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{user_id}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="ui.selectors.submit_button")
            ],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_whitespace_handling(self, make_routine) -> None:
        """Test that whitespace in parameter patterns is handled correctly."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{ user_id }}"),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="user.name")
            ],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_empty_parameters_list(self, make_routine) -> None:
        """Test validation with empty parameters list."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{sessionStorage:user.id}}")]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_no_interpolation(self, make_routine) -> None:
        """Test validation when no parameter interpolation is used."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/static-page"),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="static_data")
            ]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_jseval_op_with_valid_param(self, make_routine) -> None:
        """Test validation with valid parameter in JS evaluation operation."""
        routine = make_routine(
            operations=[RoutineJsEvaluateOperation(js='(function() { return "{{user_name}}"; })()')],
            parameters=[Parameter(name="user_name", type=ParameterType.STRING, description="User name", required=True)]
        )
        routine.validate_parameter_usage()

    def test_validate_parameter_usage_jseval_op_with_undefined_param_raises_error(self, make_routine) -> None:
        """Test that undefined parameter in JS evaluation operation raises validation error."""
        with pytest.raises(ValueError, match="Undefined parameters found in routine"):
            make_routine(
                operations=[RoutineJsEvaluateOperation(js='(function() { return "{{undefined_param}}"; })()')]
            )

    def test_validate_parameter_usage_complex_nested_operations(self, make_routine) -> None:
        """Test validation with complex nested operation structures."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{user_id}}/{{page}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="ui.selectors.menu"),
                RoutineReturnOperation(session_storage_key="{{user_email}}")
            ],
            parameters=[
                Parameter(name="user_id", type=ParameterType.STRING, description="User ID"),
                Parameter(name="page", type=ParameterType.STRING, description="Page name"),
                Parameter(name="user_email", type=ParameterType.STRING, description="User email")
            ]
        )
        routine.validate_parameter_usage()


class TestRoutineValidationErrorMessages:
    """Test that validation error messages are clear and helpful."""

    def test_undefined_parameter_error_message(self, make_routine) -> None:
        """Test that undefined parameter error message is clear."""
        with pytest.raises(ValueError) as exc_info:
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{undefined_param}}")]
            )
        assert "Undefined parameters found in routine" in str(exc_info.value)

    def test_invalid_storage_prefix_error_message(self, make_routine) -> None:
        """Test that invalid storage prefix error message is clear."""
        with pytest.raises(ValueError) as exc_info:
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{invalidStorage:user.name}}")]
            )
        assert "Invalid prefix in placeholder: invalidStorage" in str(exc_info.value)

    def test_invalid_jseval_op_code_error_message(self, make_routine) -> None:
        """Test that invalid JS code in JS evaluation operation raises clear error."""
        with pytest.raises(ValidationError) as exc_info:
            make_routine(
                operations=[RoutineJsEvaluateOperation(js="return document.title;")]  # Not wrapped in IIFE
            )
        assert "IIFE" in str(exc_info.value)


class TestParameterReservedPrefixes:
    """Test that Parameter model rejects names starting with reserved prefixes."""

    def test_parameter_name_cannot_start_with_sessionStorage(self) -> None:
        """Test that parameter name cannot start with 'sessionStorage'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="sessionStorageToken",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'sessionStorage'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_localStorage(self) -> None:
        """Test that parameter name cannot start with 'localStorage'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="localStorageValue",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'localStorage'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_cookie(self) -> None:
        """Test that parameter name cannot start with 'cookie'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="cookieValue",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'cookie'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_meta(self) -> None:
        """Test that parameter name cannot start with 'meta'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="metaTag",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'meta'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_uuid(self) -> None:
        """Test that parameter name cannot start with 'uuid'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="uuidValue",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'uuid'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_epoch_milliseconds(self) -> None:
        """Test that parameter name cannot start with 'epoch_milliseconds'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="epoch_millisecondsValue",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'epoch_milliseconds'" in str(exc_info.value)

    def test_parameter_name_cannot_start_with_windowProperty(self) -> None:
        """Test that parameter name cannot start with 'windowProperty'."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="windowPropertyValue",
                type=ParameterType.STRING,
                description="Should fail"
            )
        assert "cannot start with 'windowProperty'" in str(exc_info.value)

    def test_parameter_name_can_contain_reserved_prefix_in_middle(self) -> None:
        """Test that parameter name can contain reserved prefix in the middle."""
        # This should succeed - reserved prefixes only checked at start
        param = Parameter(
            name="my_sessionStorage_token",
            type=ParameterType.STRING,
            description="Should succeed"
        )
        assert param.name == "my_sessionStorage_token"

    def test_parameter_name_can_contain_reserved_prefix_at_end(self) -> None:
        """Test that parameter name can contain reserved prefix at the end."""
        # This should succeed - reserved prefixes only checked at start
        param = Parameter(
            name="token_sessionStorage",
            type=ParameterType.STRING,
            description="Should succeed"
        )
        assert param.name == "token_sessionStorage"

    def test_valid_parameter_names_succeed(self) -> None:
        """Test that valid parameter names without reserved prefixes succeed."""
        param1 = Parameter(
            name="user_id",
            type=ParameterType.STRING,
            description="Valid name"
        )
        assert param1.name == "user_id"

        param2 = Parameter(
            name="session_token",
            type=ParameterType.STRING,
            description="Valid name"
        )
        assert param2.name == "session_token"

        param3 = Parameter(
            name="local_value",
            type=ParameterType.STRING,
            description="Valid name"
        )
        assert param3.name == "local_value"


class TestParameterUsageValidation:
    """Test the specific parameter matching logic in validate_parameter_usage (lines 496-507)."""

    def test_builtin_parameters_are_not_tracked_as_used(self, make_routine) -> None:
        """Test that builtin parameters (uuid, epoch_milliseconds) are skipped and not tracked."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{uuid}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
            ]
        )
        routine.validate_parameter_usage()

    def test_builtin_parameters_uuid_not_tracked(self, make_routine) -> None:
        """Test that uuid builtin parameter is not tracked as used_parameters."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{uuid}}")]
        )
        routine.validate_parameter_usage()

    def test_builtin_parameters_epoch_milliseconds_not_tracked(self, make_routine) -> None:
        """Test that epoch_milliseconds builtin parameter is not tracked as used_parameters."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{epoch_milliseconds}}")]
        )
        routine.validate_parameter_usage()

    def test_placeholder_params_with_colon_are_not_tracked(self, make_routine) -> None:
        """Test that placeholder params with ':' (sessionStorage, localStorage, etc.) are not tracked."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{sessionStorage:user.id}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
            ]
        )
        routine.validate_parameter_usage()

    def test_all_placeholder_prefixes_are_not_tracked(self, make_routine) -> None:
        """Test that all valid placeholder prefixes are not tracked as used_parameters."""
        make_routine(
            operations=[
                RoutineNavigateOperation(url="https://example.com/{{sessionStorage:token}}"),
                RoutineSleepOperation(timeout_seconds=1.0),
            ]
        ).validate_parameter_usage()
        
        make_routine(
            name="test_routine2",
            operations=[RoutineNavigateOperation(url="https://example.com/{{localStorage:pref}}")]
        ).validate_parameter_usage()
        
        make_routine(
            name="test_routine3",
            operations=[RoutineNavigateOperation(url="https://example.com/{{cookie:session}}")]
        ).validate_parameter_usage()
        
        make_routine(
            name="test_routine4",
            operations=[RoutineNavigateOperation(url="https://example.com/{{meta:csrf}}")]
        ).validate_parameter_usage()
        
        make_routine(
            name="test_routine5",
            operations=[RoutineNavigateOperation(url="https://example.com/{{windowProperty:app.config}}")]
        ).validate_parameter_usage()

    def test_regular_parameters_are_tracked(self, make_routine) -> None:
        """Test that regular parameters (without ':') are tracked as used_parameters."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{user_id}}")],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_mixed_builtin_and_regular_parameters(self, make_routine) -> None:
        """Test that builtin parameters don't interfere with regular parameter tracking."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{uuid}}/{{user_id}}")],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_mixed_placeholder_and_regular_parameters(self, make_routine) -> None:
        """Test that placeholder parameters don't interfere with regular parameter tracking."""
        endpoint = Endpoint(
            url="https://api.example.com/data",
            method=HTTPMethod.POST,
            headers={"Authorization": "Bearer {{sessionStorage:token}}"},
            body={"user_id": "{{user_id}}"}
        )
        routine = make_routine(
            name="test_routine2",
            operations=[RoutineFetchOperation(endpoint=endpoint)],
            parameters=[Parameter(name="user_id", type=ParameterType.STRING, description="User ID")]
        )
        routine.validate_parameter_usage()

    def test_placeholder_param_with_empty_path_raises_error(self, make_routine) -> None:
        """Test that placeholder params with empty path after ':' raise an error."""
        with pytest.raises(ValueError) as exc_info:
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{sessionStorage:}}")]
            )
        assert "Path is required" in str(exc_info.value)

    def test_placeholder_param_with_whitespace_only_path_raises_error(self, make_routine) -> None:
        """Test that placeholder params with whitespace-only path raise an error."""
        with pytest.raises(ValueError) as exc_info:
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{sessionStorage:   }}")]
            )
        assert "Path is required" in str(exc_info.value)

    def test_invalid_placeholder_prefix_raises_error(self, make_routine) -> None:
        """Test that invalid placeholder prefixes raise an error."""
        with pytest.raises(ValueError) as exc_info:
            make_routine(
                operations=[RoutineNavigateOperation(url="https://example.com/{{invalidPrefix:value}}")]
            )
        assert "Invalid prefix in placeholder: invalidPrefix" in str(exc_info.value)

    def test_placeholder_param_path_with_whitespace_is_valid(self, make_routine) -> None:
        """Test that placeholder params with whitespace in path (but not empty) are valid."""
        routine = make_routine(
            operations=[RoutineNavigateOperation(url="https://example.com/{{sessionStorage: user . token }}")]
        )
        routine.validate_parameter_usage()


class TestBaseUrlExtraction:
    """Test base URL extraction functionality."""

    def test_extract_base_url_from_url_basic(self) -> None:
        """Test extracting base URL from basic URLs."""
        assert extract_base_url_from_url("https://www.example.com") == "example.com"
        assert extract_base_url_from_url("https://api.example.com") == "example.com"
        assert extract_base_url_from_url("http://example.com") == "example.com"
        assert extract_base_url_from_url("https://example.com/path") == "example.com"

    def test_extract_base_url_from_url_with_port(self) -> None:
        """Test extracting base URL from URLs with ports."""
        assert extract_base_url_from_url("https://www.example.com:8080") == "example.com"
        assert extract_base_url_from_url("http://api.example.com:3000/path") == "example.com"

    def test_extract_base_url_from_url_special_tlds(self) -> None:
        """Test extracting base URL from URLs with special TLDs."""
        assert extract_base_url_from_url("https://www.example.co.uk") == "example.co.uk"
        assert extract_base_url_from_url("https://api.example.co.uk") == "example.co.uk"
        assert extract_base_url_from_url("https://subdomain.example.co.uk/path") == "example.co.uk"

    def test_extract_base_url_from_url_with_subdomains(self) -> None:
        """Test extracting base URL from URLs with various subdomains."""
        assert extract_base_url_from_url("https://www.amtrak.com") == "amtrak.com"
        assert extract_base_url_from_url("https://api.amtrak.com") == "amtrak.com"
        assert extract_base_url_from_url("https://mobile.api.amtrak.com") == "amtrak.com"

    def test_extract_base_url_from_url_with_placeholders(self) -> None:
        """Test extracting base URL from URLs with parameter placeholders."""
        # URLs with placeholders should still extract the base domain
        result = extract_base_url_from_url("https://api.example.com/{{user_id}}")
        assert result == "example.com"
        
        result = extract_base_url_from_url("https://{{domain}}.example.com/path")
        # Should handle gracefully - might return None or extract what it can
        assert result is not None

    def test_extract_base_url_from_url_invalid_urls(self) -> None:
        """Test extracting base URL from invalid URLs."""
        assert extract_base_url_from_url("not-a-url") is None
        assert extract_base_url_from_url("") is None
        assert extract_base_url_from_url("://invalid") is None

    def test_extract_base_url_from_url_with_query_params(self) -> None:
        """Test extracting base URL from URLs with query parameters."""
        assert extract_base_url_from_url("https://example.com?param=value") == "example.com"
        assert extract_base_url_from_url("https://api.example.com/path?foo=bar&baz=qux") == "example.com"

    def test_extract_base_url_from_url_with_fragments(self) -> None:
        """Test extracting base URL from URLs with fragments."""
        assert extract_base_url_from_url("https://example.com#section") == "example.com"
        assert extract_base_url_from_url("https://api.example.com/path#anchor") == "example.com"

    def test_routine_base_urls_auto_populated_navigate(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from navigate operations."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineSleepOperation(timeout_seconds=1.0),
            ]
        )
        assert routine.compute_base_urls_from_operations() == "example.com"

    def test_routine_base_urls_auto_populated_fetch(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from fetch operations."""
        endpoint = Endpoint(url="https://api.example.com/data", method=HTTPMethod.GET, headers={}, body={})
        routine = make_routine(operations=[RoutineFetchOperation(endpoint=endpoint)])
        assert routine.compute_base_urls_from_operations() == "example.com"

    def test_routine_base_urls_auto_populated_mixed(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from mixed operations."""
        endpoint = Endpoint(url="https://api.example.com/data", method=HTTPMethod.GET, headers={}, body={})
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineFetchOperation(endpoint=endpoint),
                RoutineNavigateOperation(url="https://www.otherdomain.com"),
            ]
        )
        assert routine.compute_base_urls_from_operations() == "example.com,otherdomain.com"

    def test_routine_base_urls_auto_populated_no_urls(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations returns None when no URL operations exist."""
        routine = make_routine(
            operations=[
                RoutineSleepOperation(timeout_seconds=1.0),
                RoutineReturnOperation(session_storage_key="test"),
            ]
        )
        assert routine.compute_base_urls_from_operations() is None

    def test_routine_base_urls_with_placeholders(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations works with URLs containing placeholders."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com/{{user_id}}"),
                RoutineFetchOperation(
                    endpoint=Endpoint(url="https://api.example.com/{{param}}/data", method=HTTPMethod.GET, headers={}, body={})
                ),
            ],
            parameters=[
                Parameter(name="user_id", type=ParameterType.STRING, description="User ID"),
                Parameter(name="param", type=ParameterType.STRING, description="Param"),
            ]
        )
        assert routine.compute_base_urls_from_operations() == "example.com"

    def test_routine_base_urls_special_tlds(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations handles special TLDs correctly."""
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.co.uk"),
                RoutineFetchOperation(
                    endpoint=Endpoint(url="https://api.example.co.uk/data", method=HTTPMethod.GET, headers={}, body={})
                ),
            ]
        )
        assert routine.compute_base_urls_from_operations() == "example.co.uk"

    def test_routine_base_urls_auto_populated_download(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from download operations."""
        endpoint = Endpoint(url="https://cdn.example.com/files/report.pdf", method=HTTPMethod.GET, headers={}, body={})
        routine = make_routine(operations=[RoutineDownloadOperation(endpoint=endpoint, filename="report.pdf")])
        assert routine.compute_base_urls_from_operations() == "example.com"

    def test_routine_base_urls_mixed_with_download(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from mixed operations including download."""
        fetch_endpoint = Endpoint(url="https://api.example.com/data", method=HTTPMethod.GET, headers={}, body={})
        download_endpoint = Endpoint(url="https://cdn.example.com/files/report.pdf", method=HTTPMethod.GET, headers={}, body={})
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineFetchOperation(endpoint=fetch_endpoint),
                RoutineDownloadOperation(endpoint=download_endpoint, filename="report.pdf"),
            ]
        )
        # Should deduplicate example.com
        assert routine.compute_base_urls_from_operations() == "example.com"

    def test_routine_base_urls_download_different_domain(self, make_routine) -> None:
        """Test that compute_base_urls_from_operations extracts base URLs from download operations on different domains."""
        download_endpoint = Endpoint(url="https://cdn.downloads.com/files/report.pdf", method=HTTPMethod.GET, headers={}, body={})
        routine = make_routine(
            operations=[
                RoutineNavigateOperation(url="https://www.example.com"),
                RoutineDownloadOperation(endpoint=download_endpoint, filename="report.pdf"),
            ]
        )
        assert routine.compute_base_urls_from_operations() == "downloads.com,example.com"


class TestPremierLeagueRoutineValidation:
    """Test validation for Premier League Get Matchweek Games routine."""

    def test_premier_league_routine_valid_parameters(self) -> None:
        """Test Premier League routine with valid integer parameters."""
        routine = Routine(
            name="Premier League Get Matchweek Games",
            description="Get all matchweek games (played or scheduled) for the english premier league.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches?competition=8&season={{season_year}}&matchweek={{matchweek}}&_limit=100",
                        method=HTTPMethod.GET,
                        headers={},
                        body={},
                        credentials="omit"
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="season_year",
                    type=ParameterType.INTEGER,
                    description="Start year of the EPL season",
                    required=True,
                    default=2025,
                    min_value=2024,
                    max_value=2025,
                    examples=[2025]
                ),
                Parameter(
                    name="matchweek",
                    type=ParameterType.INTEGER,
                    description="EPL matchweek",
                    required=True,
                    min_value=1,
                    max_value=38,
                    examples=[10, 1, 23]
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        
        # Should not raise any validation errors
        routine.validate_parameter_usage()

    def test_premier_league_routine_base_url_extraction(self) -> None:
        """Test that compute_base_urls_from_operations correctly extracts base URL for Premier League routine."""
        routine = Routine(
            name="Premier League Get Matchweek Games",
            description="Get all matchweek games for the EPL.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches?competition=8&season={{season_year}}&matchweek={{matchweek}}&_limit=100",
                        method=HTTPMethod.GET,
                        headers={},
                        body={},
                        credentials="omit"
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="season_year",
                    type=ParameterType.INTEGER,
                    description="Start year of the EPL season",
                    required=True,
                    default=2025,
                    min_value=2024,
                    max_value=2025
                ),
                Parameter(
                    name="matchweek",
                    type=ParameterType.INTEGER,
                    description="EPL matchweek",
                    required=True,
                    min_value=1,
                    max_value=38
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        
        # Base URL should be extracted from pulselive.com
        assert routine.compute_base_urls_from_operations() == "pulselive.com"

    def test_premier_league_routine_missing_parameter_raises_error(self) -> None:
        """Test that using undefined parameters raises validation error."""
        with pytest.raises(ValueError, match="Undefined parameters found in routine"):
            Routine(
                name="Premier League Get Matchweek Games",
                description="Get all matchweek games for the EPL.",
                operations=[
                    RoutineFetchOperation(
                        endpoint=Endpoint(
                            url="https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches?competition=8&season={{season_year}}&matchweek={{matchweek}}&_limit=100",
                            method=HTTPMethod.GET,
                            headers={},
                            body={},
                            credentials="omit"
                        ),
                        session_storage_key="result"
                    ),
                    RoutineReturnOperation(session_storage_key="result")
                ],
                # Missing matchweek parameter
                parameters=[
                    Parameter(
                        name="season_year",
                        type=ParameterType.INTEGER,
                        description="Start year of the EPL season",
                        required=True
                    )
                ],
                created_by="test_user",
                project_id="test_project"
            )

    def test_premier_league_routine_unquoted_placeholder_is_valid(self) -> None:
        """Test that placeholders in URLs are detected and valid."""
        # All {{...}} patterns are now detected regardless of quoting
        routine = Routine(
            name="Premier League Get Matchweek Games",
            description="Get all matchweek games for the EPL.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://sdp-prem-prod.premier-league-prod.pulselive.com/api/v2/matches?competition=8&season={{season_year}}&matchweek={{matchweek}}&_limit=100",
                        method=HTTPMethod.GET,
                        headers={},
                        body={},
                        credentials="omit"
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="season_year",
                    type=ParameterType.INTEGER,
                    description="Start year of the EPL season",
                    required=True
                ),
                Parameter(
                    name="matchweek",
                    type=ParameterType.INTEGER,
                    description="EPL matchweek",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        routine.validate_parameter_usage()

    def test_premier_league_routine_integer_parameter_types(self) -> None:
        """Test that integer parameters with min/max values are valid."""
        season_param = Parameter(
            name="season_year",
            type=ParameterType.INTEGER,
            description="Start year of the EPL season",
            required=True,
            default=2025,
            min_value=2024,
            max_value=2025,
            examples=[2025]
        )
        assert season_param.type == ParameterType.INTEGER
        assert season_param.min_value == 2024
        assert season_param.max_value == 2025
        assert season_param.default == 2025
        
        matchweek_param = Parameter(
            name="matchweek",
            type=ParameterType.INTEGER,
            description="EPL matchweek",
            required=True,
            min_value=1,
            max_value=38,
            examples=[10, 1, 23]
        )
        assert matchweek_param.type == ParameterType.INTEGER
        assert matchweek_param.min_value == 1
        assert matchweek_param.max_value == 38
        assert matchweek_param.examples == [10, 1, 23]


class TestParameterTypeQuoteValidation:
    """Test validation rules based on parameter types.

    All parameter types use uniform {{param}} syntax.
    Type resolution is determined by the Parameter's type field at runtime.
    """

    def test_integer_param_in_body_quoted_format_valid(self) -> None:
        """Test INTEGER param using quoted format "{{param}}" in body is valid."""
        routine = Routine(
            name="API with Integer Body Param",
            description="Routine with integer param in body field.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={"n_trials": "{{n_trials}}", "limit": "{{limit}}"}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="n_trials",
                    type=ParameterType.INTEGER,
                    description="Number of trials",
                    required=True
                ),
                Parameter(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Result limit",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - INTEGER can use "{{...}}" format
        routine.validate_parameter_usage()

    def test_number_param_in_body_quoted_format_valid(self) -> None:
        """Test NUMBER param using quoted format "{{param}}" in body is valid."""
        routine = Routine(
            name="API with Number Body Param",
            description="Routine with number param in body field.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={"threshold": "{{threshold}}"}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="threshold",
                    type=ParameterType.NUMBER,
                    description="Threshold value (float)",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - NUMBER can use "{{...}}" format
        routine.validate_parameter_usage()

    def test_boolean_param_in_body_quoted_format_valid(self) -> None:
        """Test BOOLEAN param using quoted format "{{param}}" in body is valid."""
        routine = Routine(
            name="API with Boolean Body Param",
            description="Routine with boolean param in body field.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={"enabled": "{{enabled}}", "verbose": "{{verbose}}"}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="enabled",
                    type=ParameterType.BOOLEAN,
                    description="Enable feature",
                    required=True
                ),
                Parameter(
                    name="verbose",
                    type=ParameterType.BOOLEAN,
                    description="Verbose output",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - BOOLEAN can use "{{...}}" format
        routine.validate_parameter_usage()

    def test_string_param_in_body_quoted_format_is_valid(self) -> None:
        """Test STRING param using {{param}} in body is valid (type resolution is schema-driven)."""
        routine = Routine(
            name="API with String Body Param",
            description="Routine with string param in body field.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={"name": "{{user_name}}"}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="user_name",
                    type=ParameterType.STRING,
                    description="User name",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        routine.validate_parameter_usage()

    def test_string_param_in_body_valid(self) -> None:
        """Test STRING param using {{param}} format in body is valid."""
        routine = Routine(
            name="API with String Body Param",
            description="Routine with string param in body field.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        # STRING param using {{...}} IS allowed
                        body={"name": "{{user_name}}"}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="user_name",
                    type=ParameterType.STRING,
                    description="User name",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - STRING using {{...}} is valid
        routine.validate_parameter_usage()

    def test_integer_param_in_url_valid(self) -> None:
        """Test INTEGER param using {{param}} format in URL is valid."""
        routine = Routine(
            name="API with Integer URL Param",
            description="Routine with integer param in URL.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data/{{item_id}}",
                        method=HTTPMethod.GET,
                        headers={},
                        body={}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(
                    name="item_id",
                    type=ParameterType.INTEGER,
                    description="Item ID",
                    required=True
                )
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - INTEGER can use {{...}} too
        routine.validate_parameter_usage()

    def test_mixed_param_types_in_body(self) -> None:
        """Test mixing different param types in body with appropriate quote formats."""
        routine = Routine(
            name="API with Mixed Param Types",
            description="Routine with mixed param types in body.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.POST,
                        headers={},
                        body={
                            "name": "{{user_name}}",
                            "count": "{{count}}",
                            "active": "{{is_active}}",
                            "rate": "{{rate}}"
                        }
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[
                Parameter(name="user_name", type=ParameterType.STRING, description="User name", required=True),
                Parameter(name="count", type=ParameterType.INTEGER, description="Count", required=True),
                Parameter(name="is_active", type=ParameterType.BOOLEAN, description="Is active", required=True),
                Parameter(name="rate", type=ParameterType.NUMBER, description="Rate", required=True)
            ],
            created_by="test_user",
            project_id="test_project"
        )
        # Should not raise - all params use correct quote format for their type
        routine.validate_parameter_usage()

    def test_storage_placeholder_in_header_quoted_valid(self) -> None:
        """Test storage placeholder using quoted format in header is valid."""
        routine = Routine(
            name="API with Storage Header",
            description="Routine with storage placeholder in header.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET,
                        headers={},
                        # Storage can use "{{...}}" format
                        body={}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[],
            created_by="test_user",
            project_id="test_project"
        )
        # Update headers after creation since Endpoint validation might be strict
        routine.operations[0].endpoint.headers = {"X-API-Key": "{{sessionStorage:api_key}}"}
        # Should not raise - storage can use "{{...}}" format
        routine.validate_parameter_usage()

    def test_routine_with_session_storage_in_endpoint_header_quoted(self) -> None:
        """Test routine using quoted sessionStorage placeholder in endpoint header."""
        routine = Routine(
            name="API with Session Storage Header (Quoted)",
            description="Routine that uses sessionStorage value in endpoint header with quotes.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET,
                        headers={},
                        body={}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[],
            created_by="test_user",
            project_id="test_project"
        )
        # Update headers after creation
        routine.operations[0].endpoint.headers = {
            "X-API-Key": "{{sessionStorage:api_key}}",
            "Authorization": "Bearer {{sessionStorage:auth.token}}"
        }
        
        # Should not raise any validation errors
        routine.validate_parameter_usage()

    def test_routine_with_session_storage_in_endpoint_header_unquoted(self) -> None:
        """Test routine using unquoted sessionStorage placeholder in endpoint header.
        
        Storage placeholders (sessionStorage, localStorage, cookie, meta, windowProperty)
        do NOT require escaped quotes because they can resolve to int, float, or bool values.
        """
        routine = Routine(
            name="API with Session Storage Header (Unquoted)",
            description="Routine that uses sessionStorage value in endpoint header without quotes.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET,
                        headers={},
                        body={}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[],
            created_by="test_user",
            project_id="test_project"
        )
        # Update headers after creation
        routine.operations[0].endpoint.headers = {
            # Unquoted is valid for storage placeholders - can be int/float/bool
            "X-API-Key": "{{sessionStorage:api_key}}",
            "X-User-ID": "{{sessionStorage:user.id}}",
            "X-Rate-Limit": "{{sessionStorage:rate_limit}}"
        }
        
        # Should not raise any validation errors - storage placeholders don't need quotes
        routine.validate_parameter_usage()

    def test_routine_with_mixed_quoted_unquoted_storage_placeholders(self) -> None:
        """Test routine mixing quoted and unquoted storage placeholders."""
        routine = Routine(
            name="API with Mixed Storage Placeholders",
            description="Routine with both quoted and unquoted storage placeholders.",
            operations=[
                RoutineFetchOperation(
                    endpoint=Endpoint(
                        url="https://api.example.com/data",
                        method=HTTPMethod.GET,
                        headers={},
                        body={}
                    ),
                    session_storage_key="result"
                ),
                RoutineReturnOperation(session_storage_key="result")
            ],
            parameters=[],
            created_by="test_user",
            project_id="test_project"
        )
        # Update headers and body after creation
        routine.operations[0].endpoint.headers = {
            # Quoted string value
            "Authorization": "Bearer {{sessionStorage:auth.token}}",
            # Unquoted int value
            "X-User-ID": "{{sessionStorage:user.id}}"
        }
        routine.operations[0].endpoint.body = {
            # Unquoted values in body (can be int/bool)
            "limit": "{{localStorage:page_limit}}",
            "enabled": "{{sessionStorage:feature_enabled}}"
        }
        
        # Should not raise any validation errors
        routine.validate_parameter_usage()
