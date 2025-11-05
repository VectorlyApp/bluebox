"""
tests/unit/test_production_routine.py

Unit tests for production routine data models.
"""

import re
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.utils.data_utils import load_data
from src.data_models.production_routine import (
    ResourceBase,
    Routine,
    Parameter,
    ParameterType,
    Endpoint,
    HTTPMethod,
    CREDENTIALS,
    RoutineNavigateOperation,
    RoutineSleepOperation,
    RoutineFetchOperation,
    RoutineReturnOperation,
)


def _make_basic_routine(
    parameters: list[Parameter],
    operations: list,
    name: str = "test_routine",
    description: str = "Test routine",
) -> Routine:
    """
    Helper to create a basic routine for testing.
    Args:
        parameters: List of parameters to include.
        operations: List of operations to include.
        name: Routine name.
        description: Routine description.
    Returns:
        Routine instance.
    """
    return Routine(
        name=name,
        description=description,
        parameters=parameters,
        operations=operations,
    )


def _make_fetch_operation(
    url: str = "https://api.example.com/endpoint",
    headers: dict | None = None,
    body: dict | None = None,
    session_storage_key: str = "result",
) -> RoutineFetchOperation:
    """
    Helper to create a fetch operation.
    Args:
        url: Endpoint URL.
        headers: Request headers.
        body: Request body.
        session_storage_key: Session storage key for result.
    Returns:
        RoutineFetchOperation instance.
    """
    return RoutineFetchOperation(
        endpoint=Endpoint(
            url=url,
            method=HTTPMethod.POST,
            headers=headers or {},
            body=body or {},
            credentials=CREDENTIALS.INCLUDE,
        ),
        session_storage_key=session_storage_key,
    )


class TestResourceBase:
    """Tests for ResourceBase class."""
    
    def test_id_generation_format(self) -> None:
        """ID should be generated in format ClassName_uuid."""

        class TestResource(ResourceBase):
            pass

        resource = TestResource()
        # check format: ClassName_uuid
        assert resource.id.startswith("TestResource_")
        # check uuid portion is valid
        uuid_part = resource.id.split("_", 1)[1]
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        assert re.match(uuid_pattern, uuid_part)
    
    def test_different_subclasses_different_ids(self) -> None:
        """Different subclasses should generate IDs with their own class names."""

        class ResourceA(ResourceBase):
            pass
        
        class ResourceB(ResourceBase):
            pass
        
        a = ResourceA()
        b = ResourceB()
        
        assert a.id.startswith("ResourceA_")
        assert b.id.startswith("ResourceB_")
    
    def test_created_at_timestamp(self) -> None:
        """created_at should be a valid unix timestamp."""

        class TestResource(ResourceBase):
            pass
        
        before = int(time.time())
        resource = TestResource()
        after = int(time.time())
        
        assert before <= resource.created_at <= after
        assert isinstance(resource.created_at, int)
    
    def test_updated_at_timestamp(self) -> None:
        """updated_at should be a valid unix timestamp."""

        class TestResource(ResourceBase):
            pass
        
        before = int(time.time())
        resource = TestResource()
        after = int(time.time())
        
        assert before <= resource.updated_at <= after
        assert isinstance(resource.updated_at, int)
    
    def test_resource_type_property(self) -> None:
        """resource_type property should return class name."""

        class MyCustomResource(ResourceBase):
            pass
        
        resource = MyCustomResource()
        assert resource.resource_type == "MyCustomResource"
    
    def test_custom_id_provided(self) -> None:
        """Should accept custom ID if provided."""

        class TestResource(ResourceBase):
                pass
        
        custom_id = "TestResource_custom-123"
        resource = TestResource(id=custom_id)
        assert resource.id == custom_id
    
    def test_custom_timestamps_provided(self) -> None:
        """Should accept custom timestamps if provided."""

        class TestResource(ResourceBase):
            pass
        
        created = 1609459200  # 2021-01-01
        updated = 1640995200  # 2022-01-01
        resource = TestResource(created_at=created, updated_at=updated)
        assert resource.created_at == created
        assert resource.updated_at == updated
    
    def test_serialization(self) -> None:
        """ResourceBase instances should be serializable."""

        class _TestResource(ResourceBase):
            name: str

        resource = _TestResource(name="test")
        data = resource.model_dump()

        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert data["name"] == "test"


class TestParameter:
    """Tests for Parameter class."""

    def test_valid_string_parameter(self, input_data_dir: Path) -> None:
        """Valid string parameter should be created successfully."""
        data = load_data(input_data_dir / "production_routine" / "parameter_valid_string.json")
        param = Parameter(**data)

        assert param.name == "user_name"
        assert param.type == ParameterType.STRING
        assert param.required is True
        assert param.default == "guest"
        assert len(param.examples) == 2
    
    def test_valid_integer_parameter(self, input_data_dir: Path) -> None:
        """Valid integer parameter should be created successfully."""
        data = load_data(input_data_dir / "production_routine" / "parameter_valid_integer.json")
        param = Parameter(**data)

        assert param.name == "page_number"
        assert param.type == ParameterType.INTEGER
        assert param.required is False
        assert param.default == 1
        assert param.min_value == 1
        assert param.max_value == 1000
    
    def test_valid_enum_parameter(self, input_data_dir: Path) -> None:
        """Valid enum parameter should be created successfully."""
        data = load_data(input_data_dir / "production_routine" / "parameter_valid_enum.json")
        param = Parameter(**data)

        assert param.name == "status"
        assert param.type == ParameterType.ENUM
        assert param.enum_values == ["active", "pending", "completed", "cancelled"]
        assert param.default == "active"

    @pytest.mark.parametrize("invalid_name", [
        "123invalid",  # starts with number
        "invalid-name",  # contains hyphen
        "invalid name",  # contains space
        "invalid.name",  # contains dot
    ])
    def test_invalid_parameter_names(self, invalid_name: str) -> None:
        """Invalid parameter names should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(name=invalid_name, description="test")
        
        error_msg = str(exc_info.value)
        assert "not a valid Python identifier" in error_msg
    
    @pytest.mark.parametrize("reserved_prefix", [
        "sessionStorage",
        "localStorage",
        "cookie",
        "meta",
        "uuid",
        "epoch_milliseconds",
    ])
    def test_reserved_prefix_names(self, reserved_prefix: str) -> None:
        """Parameter names with reserved prefixes should raise ValidationError."""
        invalid_name = f"{reserved_prefix}_data"
        with pytest.raises(ValidationError) as exc_info:
            Parameter(name=invalid_name, description="test")
        
        error_msg = str(exc_info.value)
        assert f"cannot start with '{reserved_prefix}'" in error_msg
    
    def test_invalid_enum_without_values(self, input_data_dir: Path) -> None:
        """Enum parameter without enum_values should raise ValidationError."""
        data = load_data(input_data_dir / "production_routine" / "parameter_invalid_enum_no_values.json")
        with pytest.raises(ValidationError) as exc_info:
            Parameter(**data)
        
        error_msg = str(exc_info.value)
        assert "enum_values must be provided" in error_msg
    
    def test_default_value_type_conversion_integer(self) -> None:
        """Default value should be converted to correct type for INTEGER."""
        param = Parameter(
            name="count",
            description="Count",
            type=ParameterType.INTEGER,
            default="42"
        )
        assert param.default == 42
        assert isinstance(param.default, int)
    
    def test_default_value_type_conversion_number(self) -> None:
        """Default value should be converted to correct type for NUMBER."""
        param = Parameter(
            name="price",
            description="Price",
            type=ParameterType.NUMBER,
            default="19.99"
        )
        assert param.default == 19.99
        assert isinstance(param.default, float)
    
    @pytest.mark.parametrize("bool_value,expected", [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ])
    def test_default_value_boolean_conversion(self, bool_value: str, expected: bool) -> None:
        """Boolean default values should be converted correctly."""
        param = Parameter(
            name="enabled",
            description="Enabled flag",
            type=ParameterType.BOOLEAN,
            default=bool_value
        )
        assert param.default == expected
    
    def test_invalid_default_value_for_integer(self) -> None:
        """Invalid default value for INTEGER should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="count",
                description="Count",
                type=ParameterType.INTEGER,
                default="not_a_number"
            )
        
        error_msg = str(exc_info.value)
        assert "cannot be converted to integer" in error_msg
    
    def test_examples_type_conversion_integer(self) -> None:
        """Examples should be converted to correct type for INTEGER."""
        param = Parameter(
            name="count",
            description="Count",
            type=ParameterType.INTEGER,
            examples=["1", "2", "3"]
        )
        assert param.examples == [1, 2, 3]
        assert all(isinstance(ex, int) for ex in param.examples)
    
    def test_examples_type_conversion_number(self) -> None:
        """Examples should be converted to correct type for NUMBER."""
        param = Parameter(
            name="price",
            description="Price",
            type=ParameterType.NUMBER,
            examples=["10.5", "20.99", "5"]
        )
        assert param.examples == [10.5, 20.99, 5.0]
        assert all(isinstance(ex, float) for ex in param.examples)
    
    def test_invalid_examples_for_integer(self) -> None:
        """Invalid examples for INTEGER should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Parameter(
                name="count",
                description="Count",
                type=ParameterType.INTEGER,
                examples=["1", "invalid", "3"]
            )
        error_msg = str(exc_info.value)
        assert "cannot be converted to integer" in error_msg
    
    def test_parameter_with_all_fields(self) -> None:
        """Parameter with all optional fields should work."""
        param = Parameter(
            name="advanced_param",
            description="Advanced parameter",
            type=ParameterType.STRING,
            required=False,
            default="default_value",
            examples=["example1", "example2"],
            min_length=5,
            max_length=100,
            pattern=r"^[a-z]+$",
            format="lowercase"
        )
        assert param.name == "advanced_param"
        assert param.min_length == 5
        assert param.max_length == 100
        assert param.pattern == r"^[a-z]+$"
        assert param.format == "lowercase"


class TestRoutineParameterValidation:
    """Tests for Routine.validate_parameter_usage method."""
    
    def test_parameter_in_url_query_string(self) -> None:
        """Parameter used in URL query string should be valid."""
        params = [
            Parameter(name="query", description="Search query", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com/search?q={{query}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_in_url_path(self) -> None:
        """Parameter used in URL path should be valid."""
        params = [
            Parameter(name="user_id", description="User ID", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com/users/{{user_id}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_multiple_parameters_in_url(self) -> None:
        """Multiple parameters in same URL should be valid."""
        params = [
            Parameter(name="query", description="Search query", type=ParameterType.STRING),
            Parameter(name="limit", description="Result limit", type=ParameterType.INTEGER),
            Parameter(name="offset", description="Result offset", type=ParameterType.INTEGER),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/search?q={{query}}&limit={{limit}}&offset={{offset}}"
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_in_headers(self) -> None:
        """Parameter used in headers should be valid."""
        params = [
            Parameter(name="api_token", description="API token", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                headers={"Authorization": "Bearer {{api_token}}"}
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_in_body(self) -> None:
        """Parameter used in request body should be valid."""
        params = [
            Parameter(name="username", description="Username", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                body={"username": "{{username}}"}
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_in_nested_body(self) -> None:
        """Parameter used in nested request body should be valid."""
        params = [
            Parameter(name="email", description="Email", type=ParameterType.EMAIL),
        ]
        ops = [
            _make_fetch_operation(
                body={"user": {"contact": {"email": "{{email}}"}}}
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_in_navigate_url(self) -> None:
        """Parameter used in navigate operation URL should be valid."""
        params = [
            Parameter(name="page_id", description="Page ID", type=ParameterType.STRING),
        ]
        ops = [
            RoutineNavigateOperation(url="https://example.com/page/{{page_id}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_used_multiple_times(self) -> None:
        """Same parameter used multiple times should be valid."""
        params = [
            Parameter(name="user_id", description="User ID", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/users/{{user_id}}/profile",
                headers={"X-User-ID": "{{user_id}}"},
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_all_parameters_used_across_operations(self) -> None:
        """All parameters used across different operations should be valid."""
        params = [
            Parameter(name="search_term", description="Search term", type=ParameterType.STRING),
            Parameter(name="result_id", description="Result ID", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/search?q={{search_term}}",
                session_storage_key="search_results",
            ),
            RoutineSleepOperation(timeout_seconds=1.0),
            _make_fetch_operation(
                url="https://api.example.com/items/{{result_id}}",
                session_storage_key="item_details",
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_uuid_builtin(self) -> None:
        """Using {{uuid}} builtin should not require parameter definition."""
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/create",
                body={"id": "{{uuid}}"},
            ),
        ]
        routine = _make_basic_routine(parameters=[], operations=ops)
        assert routine.name == "test_routine"
    
    def test_epoch_milliseconds_builtin(self) -> None:
        """Using {{epoch_milliseconds}} builtin should not require parameter definition."""
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/events",
                body={"timestamp": "{{epoch_milliseconds}}"},
            ),
        ]
        routine = _make_basic_routine(parameters=[], operations=ops)
        assert routine.name == "test_routine"
    
    def test_builtin_with_regular_params(self) -> None:
        """Builtins mixed with regular parameters should work."""
        params = [
            Parameter(name="event_name", description="Event name", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/events",
                body={
                    "id": "{{uuid}}",
                    "name": "{{event_name}}",
                    "timestamp": "{{epoch_milliseconds}}",
                },
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    @pytest.mark.parametrize("storage_type,path", [
        ("sessionStorage", "user_data"),
        ("localStorage", "settings.theme"),
        ("cookie", "auth_token"),
        ("meta", "page.title"),
    ])
    def test_storage_parameter_types(self, storage_type: str, path: str) -> None:
        """Storage parameters should not require parameter definition."""
        ops = [
            _make_fetch_operation(
                body={"value": f"{{{{{storage_type}:{path}}}}}"}
            ),
        ]
        routine = _make_basic_routine(parameters=[], operations=ops)
        assert routine.name == "test_routine"
    
    def test_storage_with_regular_params(self) -> None:
        """Storage parameters mixed with regular parameters should work."""
        params = [
            Parameter(name="new_value", description="New value", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                body={
                    "old_value": "{{sessionStorage:cached_data}}",
                    "new_value": "{{new_value}}",
                    "timestamp": "{{epoch_milliseconds}}",
                }
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_storage_in_url(self) -> None:
        """Storage parameter in URL should work."""
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/data?token={{cookie:session_id}}"
            ),
        ]
        routine = _make_basic_routine(parameters=[], operations=ops)
        assert routine.name == "test_routine"
    
    def test_unused_parameter(self) -> None:
        """Defined but unused parameter should raise ValidationError."""
        params = [
            Parameter(name="unused_param", description="Unused", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com/endpoint"),
        ]
        with pytest.raises(ValidationError) as exc_info:
            _make_basic_routine(parameters=params, operations=ops)
        
        error_msg = str(exc_info.value)
        assert "Unused parameters" in error_msg
        assert "unused_param" in error_msg
    
    def test_multiple_unused_parameters(self) -> None:
        """Multiple unused parameters should be reported."""
        params = [
            Parameter(name="unused_one", description="Unused 1", type=ParameterType.STRING),
            Parameter(name="unused_two", description="Unused 2", type=ParameterType.STRING),
            Parameter(name="used_param", description="Used", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{used_param}}"),
        ]
        with pytest.raises(ValidationError) as exc_info:
            _make_basic_routine(parameters=params, operations=ops)
        
        error_msg = str(exc_info.value)
        assert "Unused parameters" in error_msg
        assert "unused_one" in error_msg or "unused_two" in error_msg
    
    def test_undefined_parameter(self) -> None:
        """Used but undefined parameter should raise ValidationError."""
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{undefined_param}}"),
        ]
        with pytest.raises(ValidationError) as exc_info:
            _make_basic_routine(parameters=[], operations=ops)
        
        error_msg = str(exc_info.value)
        assert "Undefined parameters" in error_msg
        assert "undefined_param" in error_msg
    
    def test_multiple_undefined_parameters(self) -> None:
        """Multiple undefined parameters should be reported."""
        ops = [
            _make_fetch_operation(
                url="https://api.example.com?q={{param_one}}&limit={{param_two}}"
            ),
        ]
        with pytest.raises(ValidationError) as exc_info:
            _make_basic_routine(parameters=[], operations=ops)
        
        error_msg = str(exc_info.value)
        assert "Undefined parameters" in error_msg
        assert "param_one" in error_msg or "param_two" in error_msg
    
    def test_both_unused_and_undefined(self) -> None:
        """Having both unused and undefined parameters should fail."""
        params = [
            Parameter(name="defined_unused", description="Defined but unused", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{undefined_used}}"),
        ]
        with pytest.raises(ValidationError) as exc_info:
            _make_basic_routine(parameters=params, operations=ops)
        
        error_msg = str(exc_info.value)
        # should report unused first (based on validation order)
        assert "Unused parameters" in error_msg or "Undefined parameters" in error_msg
    
    def test_empty_parameters_empty_operations(self) -> None:
        """Routine with no parameters and no operations should be valid."""
        routine = _make_basic_routine(parameters=[], operations=[])
        assert routine.name == "test_routine"
    
    def test_parameter_in_return_operation(self) -> None:
        """Parameters cannot be in return operations (only storage keys)."""
        params = [
            Parameter(name="query", description="Query", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com?q={{query}}",
                session_storage_key="result",
            ),
            RoutineReturnOperation(session_storage_key="result"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_with_whitespace_in_placeholder(self) -> None:
        """Parameter with whitespace in placeholder should be handled."""
        params = [
            Parameter(name="query", description="Query", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{ query }}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_similar_parameter_names(self) -> None:
        """Similar parameter names should be distinguished."""
        params = [
            Parameter(name="user", description="User", type=ParameterType.STRING),
            Parameter(name="user_id", description="User ID", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com?user={{user}}&id={{user_id}}"
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_parameter_value_containing_braces(self) -> None:
        """Regular parameters should not conflict with storage syntax."""
        params = [
            Parameter(name="data", description="Data", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(body={"value": "{{data}}"}),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"
    
    def test_complex_nested_structure(self) -> None:
        """Parameters in complex nested structures should be found."""
        params = [
            Parameter(name="token", description="Token", type=ParameterType.STRING),
            Parameter(name="user_id", description="User ID", type=ParameterType.STRING),
            Parameter(name="action", description="Action", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(
                url="https://api.example.com/users/{{user_id}}/{{action}}",
                headers={
                    "Authorization": "Bearer {{token}}",
                    "X-Request-ID": "{{uuid}}",
                },
                body={
                    "metadata": {
                        "user": {"id": "{{user_id}}"},
                        "action": {"type": "{{action}}"},
                        "timestamp": "{{epoch_milliseconds}}",
                        "session": "{{sessionStorage:current_session}}",
                    }
                },
            ),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.name == "test_routine"


class TestRoutine:
    """Tests for Routine class (general functionality)."""
    
    def test_valid_simple_routine_from_json(self, input_data_dir: Path) -> None:
        """Valid simple routine should be loaded from JSON."""
        data = load_data(input_data_dir / "production_routine" / "routine_valid_simple.json")
        routine = Routine(**data)
        
        assert routine.name == "simple_search_routine"
        assert routine.incognito is True
        assert len(routine.parameters) == 1
        assert len(routine.operations) == 3
        assert routine.parameters[0].name == "query"
    
    def test_valid_complex_routine_from_json(self, input_data_dir: Path) -> None:
        """Valid complex routine should be loaded from JSON."""
        data = load_data(input_data_dir / "production_routine" / "routine_valid_complex.json")
        routine = Routine(**data)
        
        assert routine.name == "complex_routine"
        assert routine.incognito is False
        assert len(routine.parameters) == 3
        assert len(routine.operations) == 4
        
        # verify parameter names
        param_names = {p.name for p in routine.parameters}
        assert param_names == {"user_id", "api_token", "limit"}
    
    def test_invalid_routine_unused_param_from_json(self, input_data_dir: Path) -> None:
        """Routine with unused parameter should fail validation."""
        data = load_data(input_data_dir / "production_routine" / "routine_invalid_unused_param.json")
        with pytest.raises(ValidationError) as exc_info:
            Routine(**data)
        
        error_msg = str(exc_info.value)
        assert "Unused parameters" in error_msg
        assert "unused_param" in error_msg

    def test_invalid_routine_param_in_description_from_json(self, input_data_dir: Path) -> None:
        """Routine with parameter placeholder in description should fail validation."""
        data = load_data(input_data_dir / "production_routine" / "routine_invalid_param_in_description.json")
        with pytest.raises(ValidationError) as exc_info:
            Routine(**data)

        error_msg = str(exc_info.value)
        assert "Parameter placeholders found in routine description" in error_msg
        assert "param" in error_msg
        assert "metadata field" in error_msg

    def test_invalid_routine_param_in_name_from_json(self, input_data_dir: Path) -> None:
        """Routine with parameter placeholder in name should fail validation."""
        data = load_data(input_data_dir / "production_routine" / "routine_invalid_param_in_name.json")
        with pytest.raises(ValidationError) as exc_info:
            Routine(**data)

        error_msg = str(exc_info.value)
        assert "Parameter placeholders found in routine name" in error_msg
        assert "user_id" in error_msg
        assert "metadata field" in error_msg

    def test_routine_with_url_params_only_from_json(self, input_data_dir: Path) -> None:
        """Routine with parameters used only in URLs should be valid."""
        data = load_data(input_data_dir / "production_routine" / "routine_url_params_only.json")
        routine = Routine(**data)

        assert routine.name == "url_params_only_routine"
        assert len(routine.parameters) == 3

        # verify all three parameters
        param_names = {p.name for p in routine.parameters}
        assert param_names == {"user_id", "page", "filter"}

        # verify parameters are used in URLs (check navigate operation)
        assert "{{user_id}}" in routine.operations[0].url

        # verify parameters are used in fetch URL
        fetch_op = routine.operations[2]
        assert isinstance(fetch_op, RoutineFetchOperation)
        assert "{{user_id}}" in fetch_op.endpoint.url
        assert "{{page}}" in fetch_op.endpoint.url
        assert "{{filter}}" in fetch_op.endpoint.url

        # verify NO parameters in headers (only static header)
        assert fetch_op.endpoint.headers == {"Content-Type": "application/json"}

        # verify empty body (no parameters)
        assert fetch_op.endpoint.body == {}
    
    def test_routine_with_escaped_string_params_from_json(self, input_data_dir: Path) -> None:
        """Routine with properly escaped string parameters should be valid."""
        data = load_data(input_data_dir / "production_routine" / "routine_escaped_string_params.json")
        routine = Routine(**data)

        assert routine.name == "routine_with_escaped_string_params"
        assert len(routine.parameters) == 6

        # verify all parameters
        param_names = {p.name for p in routine.parameters}
        assert param_names == {"api_key", "search_query", "user_agent", "page_size", "timeout_ms", "price_threshold"}

        # verify parameter types
        param_types = {p.name: p.type for p in routine.parameters}
        assert param_types["api_key"] == ParameterType.STRING
        assert param_types["search_query"] == ParameterType.STRING
        assert param_types["user_agent"] == ParameterType.STRING
        assert param_types["page_size"] == ParameterType.INTEGER
        assert param_types["timeout_ms"] == ParameterType.INTEGER
        assert param_types["price_threshold"] == ParameterType.NUMBER

        # get the fetch operation
        fetch_op = routine.operations[2]
        assert isinstance(fetch_op, RoutineFetchOperation)

        # verify STRING parameters are ESCAPED in headers with \"{{param}}\"
        assert fetch_op.endpoint.headers["Authorization"] == '"{{api_key}}"'
        assert fetch_op.endpoint.headers["User-Agent"] == '"{{user_agent}}"'
        assert fetch_op.endpoint.headers["X-Search-Query"] == '"{{search_query}}"'

        # verify NON-STRING parameters are NOT ESCAPED in headers (just {{param}})
        assert fetch_op.endpoint.headers["X-Page-Size"] == "{{page_size}}"
        assert fetch_op.endpoint.headers["X-Timeout-Ms"] == "{{timeout_ms}}"

        # verify STRING parameters are ESCAPED in body with \"{{param}}\"
        assert fetch_op.endpoint.body["query"] == '"{{search_query}}"'
        assert fetch_op.endpoint.body["api_key"] == '"{{api_key}}"'
        assert fetch_op.endpoint.body["metadata"]["user_agent"] == '"{{user_agent}}"'

        # verify NON-STRING parameters are NOT ESCAPED in body (just {{param}})
        assert fetch_op.endpoint.body["page_size"] == "{{page_size}}"
        assert fetch_op.endpoint.body["timeout_ms"] == "{{timeout_ms}}"
        assert fetch_op.endpoint.body["threshold"] == "{{price_threshold}}"

        # verify builtin parameters (not escaped)
        assert fetch_op.endpoint.body["metadata"]["timestamp"] == "{{epoch_milliseconds}}"

        # verify parameters in URL (no escaping needed in URLs)
        assert "{{search_query}}" in fetch_op.endpoint.url
        assert "{{page_size}}" in fetch_op.endpoint.url
    
    def test_yahoo_finance_routine_from_discovery_output(self) -> None:
        """Real-world example: Yahoo Finance routine with escaped string headers."""
        # load the actual routine from routine_discovery_output
        routine_path = Path("/home/ec2-user/web-hacker/routine_discovery_output/routine.json")
        if not routine_path.exists():
            pytest.skip("Yahoo Finance routine not found in routine_discovery_output")
        
        data = load_data(routine_path)
        routine = Routine(**data)

        assert routine.name == "Yahoo Finance - Search Ticker"
        assert len(routine.parameters) == 6

        # verify all parameters are string or integer types
        param_types = {p.name: p.type for p in routine.parameters}
        assert param_types["query"] == ParameterType.STRING
        assert param_types["lang"] == ParameterType.STRING
        assert param_types["region"] == ParameterType.STRING
        assert param_types["quotesCount"] == ParameterType.INTEGER
        assert param_types["newsCount"] == ParameterType.INTEGER
        assert param_types["listsCount"] == ParameterType.INTEGER

        # get the fetch operation
        fetch_op = routine.operations[2]
        assert isinstance(fetch_op, RoutineFetchOperation)

        # verify STRING parameters are ESCAPED in x-param headers
        assert fetch_op.endpoint.headers["x-param-query"] == '"{{query}}"'
        assert fetch_op.endpoint.headers["x-param-lang"] == '"{{lang}}"'
        assert fetch_op.endpoint.headers["x-param-region"] == '"{{region}}"'

        # verify INTEGER parameters are ESCAPED in x-param headers (note: even integers get escaped in this pattern)
        assert fetch_op.endpoint.headers["x-param-quotesCount"] == '"{{quotesCount}}"'
        assert fetch_op.endpoint.headers["x-param-newsCount"] == '"{{newsCount}}"'
        assert fetch_op.endpoint.headers["x-param-listsCount"] == '"{{listsCount}}"'

        # verify parameters are in URL (no escaping in URLs)
        assert "{{query}}" in fetch_op.endpoint.url
        assert "{{lang}}" in fetch_op.endpoint.url
        assert "{{region}}" in fetch_op.endpoint.url
        assert "{{quotesCount}}" in fetch_op.endpoint.url
        assert "{{newsCount}}" in fetch_op.endpoint.url
        assert "{{listsCount}}" in fetch_op.endpoint.url

    def test_routine_inherits_from_resource_base(self) -> None:
        """Routine should inherit ResourceBase functionality."""
        params = [
            Parameter(name="test_param", description="Test", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{test_param}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        
        # check ResourceBase fields
        assert routine.id.startswith("Routine_")
        assert isinstance(routine.created_at, int)
        assert isinstance(routine.updated_at, int)
        assert routine.resource_type == "Routine"
    
    def test_routine_default_incognito_true(self) -> None:
        """Routine should default to incognito=True."""
        params = [
            Parameter(name="test_param", description="Test", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{test_param}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        assert routine.incognito is True
    
    def test_routine_with_all_operation_types(self) -> None:
        """Routine with all operation types should work."""
        params = [
            Parameter(name="query", description="Query", type=ParameterType.STRING),
        ]
        ops = [
            RoutineNavigateOperation(url="https://example.com/search"),
            RoutineSleepOperation(timeout_seconds=2.0),
            _make_fetch_operation(
                url="https://api.example.com/search?q={{query}}",
                session_storage_key="results"
            ),
            RoutineReturnOperation(session_storage_key="results"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        
        assert len(routine.operations) == 4
        assert routine.operations[0].type == "navigate"
        assert routine.operations[1].type == "sleep"
        assert routine.operations[2].type == "fetch"
        assert routine.operations[3].type == "return"
    
    def test_routine_serialization(self) -> None:
        """Routine should be serializable to dict/JSON."""
        params = [
            Parameter(name="query", description="Query", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{query}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        
        # to dict
        data = routine.model_dump()
        assert data["name"] == "test_routine"
        assert len(data["parameters"]) == 1
        assert len(data["operations"]) == 1
        
        # to JSON
        json_str = routine.model_dump_json()
        assert isinstance(json_str, str)
        assert "test_routine" in json_str
    
    def test_routine_deserialization(self) -> None:
        """Routine should be deserializable from dict."""
        params = [
            Parameter(name="query", description="Query", type=ParameterType.STRING),
        ]
        ops = [
            _make_fetch_operation(url="https://api.example.com?q={{query}}"),
        ]
        routine1 = _make_basic_routine(parameters=params, operations=ops)
        
        # serialize then deserialize
        data = routine1.model_dump()
        routine2 = Routine(**data)
        
        assert routine2.name == routine1.name
        assert len(routine2.parameters) == len(routine1.parameters)
        assert len(routine2.operations) == len(routine1.operations)
    
    def test_routine_with_no_parameters(self) -> None:
        """Routine with no parameters should work if none are used."""
        ops = [
            RoutineNavigateOperation(url="https://example.com"),
            RoutineSleepOperation(timeout_seconds=1.0),
            _make_fetch_operation(
                url="https://api.example.com/data",
                body={"id": "{{uuid}}"},
                session_storage_key="data"
            ),
        ]
        routine = _make_basic_routine(parameters=[], operations=ops)
        assert len(routine.parameters) == 0
        assert len(routine.operations) == 3
    
    def test_routine_operation_discriminator(self) -> None:
        """Operations should be correctly discriminated by type."""
        params = [
            Parameter(name="url_param", description="URL param", type=ParameterType.STRING),
        ]
        ops = [
            RoutineNavigateOperation(url="https://example.com/{{url_param}}"),
        ]
        routine = _make_basic_routine(parameters=params, operations=ops)
        
        # should be correctly typed
        assert isinstance(routine.operations[0], RoutineNavigateOperation)
        assert routine.operations[0].type == "navigate"
