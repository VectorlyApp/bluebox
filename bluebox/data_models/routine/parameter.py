"""
bluebox/data_models/routine/parameter.py

Parameter definitions for routine inputs.

Contains:
- ParameterType: Enum (string, integer, number, boolean, date, datetime, email, url, enum)
- Parameter: User-defined input with name, type, validation, description
- BuiltinParameter: Auto-generated values (uuid, epoch_milliseconds, iso_date, etc.)
- VALID_PLACEHOLDER_PREFIXES: sessionStorage, localStorage, cookie, windowProperty
"""

import re
import time
from typing import Any, ClassVar, Callable
import uuid

from pydantic import BaseModel, Field, field_validator, model_validator

from bluebox.data_models.routine.parameter_type import ParameterType, VALID_PLACEHOLDER_PREFIXES


class Parameter(BaseModel):
    """
    Parameter model with comprehensive validation and type information.

    Fields:
        name (str): Parameter name (must be valid Python identifier)
        type (ParameterType): Parameter data type
        required (bool): Whether parameter is required
        description (str): Human-readable parameter description
        default (str | int | float | bool | None): Default value if not provided
        examples (list[str | int | float | bool]): Example values
        min_length (int | None): Minimum length for strings
        max_length (int | None): Maximum length for strings
        min_value (int | float | None): Minimum value for numbers
        max_value (int | float | None): Maximum value for numbers
        pattern (str | None): Regex pattern for string validation
        enum_values (list[str] | None): Allowed values for enum type
        format (str | None): Format specification (e.g., 'YYYY-MM-DD')
    """

    # reserved prefixes: names that cannot be used at the beginning of a parameter name
    RESERVED_PREFIXES: ClassVar[list[str]] = [
        *VALID_PLACEHOLDER_PREFIXES,
        "uuid", "epoch_milliseconds",
    ]


    name: str = Field(..., description="Parameter name (must be valid Python identifier)")
    type: ParameterType = Field(
        default=ParameterType.STRING,
        description="Parameter data type"
    )
    required: bool = Field(
        default=True,
        description="Whether parameter is required"
    )
    description: str = Field(..., description="Human-readable parameter description")
    default: str | int | float | bool | None = Field(
        default=None,
        description="Default value if not provided"
    )
    examples: list[str | int | float | bool] = Field(
        default_factory=list,
        description="Example values"
    )

    # Type-specific validation
    min_length: int | None = Field(
        default=None,
        description="Minimum length for strings"
    )
    max_length: int | None = Field(
        default=None,
        description="Maximum length for strings"
    )
    min_value: int | float | None = Field(
        default=None,
        description="Minimum value for numbers")
    max_value: int | float | None = Field(
        default=None,
        description="Maximum value for numbers")
    pattern: str | None = Field(
        default=None,
        description="Regex pattern for string validation"
    )
    enum_values: list[str] | None = Field(
        default=None,
        description="Allowed values for enum type"
    )

    # Format specifications
    format: str | None = Field(
        default=None,
        description="Format specification (e.g., 'YYYY-MM-DD')"
    )

    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Ensure parameter name is a valid Python identifier and not reserved."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"Parameter name '{v}' is not a valid Python identifier")

        # Check for reserved prefixes
        for prefix in cls.RESERVED_PREFIXES:
            if v.startswith(prefix):
                raise ValueError(
                    f"Parameter name '{v}' cannot start with '{prefix}'. "
                    f"Reserved prefixes: {cls.RESERVED_PREFIXES}"
                )

        return v

    @model_validator(mode='after')
    def validate_type_consistency(self):
        """Validate type-specific constraints are consistent."""
        if self.type == ParameterType.ENUM and not self.enum_values:
            raise ValueError("enum_values must be provided for enum type")
        return self

    @field_validator('default')
    @classmethod
    def validate_default_type(cls, v, info):
        """Ensure default value matches parameter type."""
        if v is None:
            return v

        param_type = info.data.get('type', ParameterType.STRING)
        if param_type == ParameterType.INTEGER and not isinstance(v, int):
            try:
                return int(v)
            except (ValueError, TypeError):
                raise ValueError(f"Default value {v} cannot be converted to integer")
        elif param_type == ParameterType.NUMBER and not isinstance(v, (int, float)):
            try:
                return float(v)
            except (ValueError, TypeError):
                raise ValueError(f"Default value {v} cannot be converted to number")
        elif param_type == ParameterType.BOOLEAN and not isinstance(v, bool):
            if isinstance(v, str):
                lower_v = v.lower()
                if lower_v in ('true', '1', 'yes', 'on'):
                    return True
                elif lower_v in ('false', '0', 'no', 'off'):
                    return False
                else:
                    raise ValueError(f"Default value {v} is not a valid boolean value")
            raise ValueError(f"Default value {v} cannot be converted to boolean")

        return v

    @field_validator('examples')
    @classmethod
    def validate_examples_type(cls, v, info):
        """Ensure examples match parameter type."""
        if not v:
            return v

        param_type = info.data.get('type', ParameterType.STRING)
        validated_examples = []

        for example in v:
            if param_type == ParameterType.INTEGER:
                try:
                    validated_examples.append(int(example))
                except (ValueError, TypeError):
                    raise ValueError(f"Example {example} cannot be converted to integer")
            elif param_type == ParameterType.NUMBER:
                try:
                    validated_examples.append(float(example))
                except (ValueError, TypeError):
                    raise ValueError(f"Example {example} cannot be converted to number")
            elif param_type == ParameterType.BOOLEAN:
                if isinstance(example, str):
                    lower_example = example.lower()
                    if lower_example in ('true', '1', 'yes', 'on'):
                        validated_examples.append(True)
                    elif lower_example in ('false', '0', 'no', 'off'):
                        validated_examples.append(False)
                    else:
                        raise ValueError(f"Example {example} is not a valid boolean value")
                else:
                    validated_examples.append(bool(example))
            else:
                validated_examples.append(str(example))

        return validated_examples


class BuiltinParameter(BaseModel):
    """
    Builtin parameter model.
    """
    name: str = Field(
        ...,
        description="Builtin parameter name"
    )
    description: str = Field(
        ...,
        description="Human-readable builtin parameter description"
    )
    value_generator: Callable[[], Any] = Field(
        ...,
        description="Function to generate the builtin parameter value"
    )


BUILTIN_PARAMETERS = [
    BuiltinParameter(
        name="uuid",
        description="UUID parameter",
        value_generator=lambda: str(uuid.uuid4())
    ),
    BuiltinParameter(
        name="epoch_milliseconds",
        description="Epoch milliseconds parameter",
        value_generator=lambda: str(int(time.time() * 1000))
    ),
]
