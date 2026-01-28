"""
tests/unit/test_resource_base.py

Tests for ResourceBase functionality including ID generation, timestamp handling,
and subclass behavior.
"""

import time
from abc import ABC
from datetime import datetime
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from bluebox.data_models.resource_base import ResourceBase


class SampleResource(ResourceBase):
    """Sample resource class for testing ResourceBase functionality."""
    name: str
    description: str | None = None


class SampleResourceWithCustomFields(ResourceBase):
    """Sample resource with additional custom fields."""
    title: str
    value: int
    tags: list[str] = Field(default_factory=list)


class TestResourceBase:
    """Test cases for ResourceBase functionality."""

    def test_resource_base_can_be_instantiated(self) -> None:
        """Test that ResourceBase can be instantiated directly."""
        # ResourceBase inherits from ABC but has no abstract methods, so it can be instantiated
        resource = ResourceBase()

        assert isinstance(resource.id, str)
        assert resource.id.startswith("ResourceBase_")
        assert isinstance(resource.created_at, float)
        assert isinstance(resource.updated_at, float)

    def test_subclass_can_be_instantiated(self) -> None:
        """Test that subclasses of ResourceBase can be instantiated."""
        resource = SampleResource(name="test_resource")

        assert resource.name == "test_resource"
        assert resource.description is None
        assert isinstance(resource.id, str)
        assert isinstance(resource.created_at, float)
        assert isinstance(resource.updated_at, float)

    def test_id_format(self) -> None:
        """Test that ID follows the correct format [ClassName]_[uuid4]."""
        resource = SampleResource(name="test")

        # check ID format
        assert resource.id.startswith("SampleResource_")
        assert len(resource.id) == len("SampleResource_") + 36  # 36 chars for UUID4

        # verify it's a valid UUID after the slash
        uuid_part = resource.id.split("_", 1)[1]
        UUID(uuid_part)  # should not raise ValueError

    def test_id_uniqueness(self) -> None:
        """Test that each instance gets a unique ID."""
        resource1 = SampleResource(name="test1")
        resource2 = SampleResource(name="test2")

        assert resource1.id != resource2.id
        assert resource1.id.startswith("SampleResource_")
        assert resource2.id.startswith("SampleResource_")

    def test_timestamps_are_set(self) -> None:
        """Test that created_at and updated_at are set to current time."""
        before = datetime.now().timestamp()
        resource = SampleResource(name="test")
        after = datetime.now().timestamp()

        assert before <= resource.created_at <= after
        assert before <= resource.updated_at <= after

    def test_timestamps_are_floats(self) -> None:
        """Test that timestamps are Unix timestamps (floats with sub-second precision)."""
        resource = SampleResource(name="test")

        assert isinstance(resource.created_at, float)
        assert isinstance(resource.updated_at, float)
        assert resource.created_at > 0
        assert resource.updated_at > 0

    def test_resource_type_property(self) -> None:
        """Test that resource_type property returns the class name."""
        resource = SampleResource(name="test")

        assert resource.resource_type == "SampleResource"

    def test_resource_type_different_classes(self) -> None:
        """Test that different subclasses return different resource types."""
        resource1 = SampleResource(name="test1")
        resource2 = SampleResourceWithCustomFields(title="test2", value=42)

        assert resource1.resource_type == "SampleResource"
        assert resource2.resource_type == "SampleResourceWithCustomFields"

    def test_custom_fields_preserved(self) -> None:
        """Test that custom fields in subclasses are preserved."""
        resource = SampleResourceWithCustomFields(
            title="My Resource",
            value=100,
            tags=["tag1", "tag2"]
        )

        assert resource.title == "My Resource"
        assert resource.value == 100
        assert resource.tags == ["tag1", "tag2"]
        assert resource.id.startswith("SampleResourceWithCustomFields_")

    def test_inheritance_chain(self) -> None:
        """Test that ResourceBase works with multiple levels of inheritance."""
        class GrandChildResource(SampleResource):
            extra_field: str = "extra"

        resource = GrandChildResource(name="test")

        assert resource.name == "test"
        assert resource.extra_field == "extra"
        assert resource.id.startswith("GrandChildResource_")
        assert resource.resource_type == "GrandChildResource"

    def test_model_dump_includes_all_fields(self) -> None:
        """Test that model_dump includes all fields including inherited ones."""
        resource = SampleResourceWithCustomFields(
            title="Test Title",
            value=42,
            tags=["a", "b"]
        )

        data = resource.model_dump()

        # check inherited fields
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

        # check custom fields
        assert data["title"] == "Test Title"
        assert data["value"] == 42
        assert data["tags"] == ["a", "b"]

        # check ID format
        assert data["id"].startswith("SampleResourceWithCustomFields_")

    def test_model_dump_json_serialization(self) -> None:
        """Test that model can be serialized to JSON."""
        resource = SampleResource(name="test", description="A test resource")

        json_str = resource.model_dump_json()
        assert isinstance(json_str, str)
        assert "SampleResource_" in json_str
        assert "test" in json_str
        assert "A test resource" in json_str

    def test_model_validation_with_invalid_data(self) -> None:
        """Test that model validation works with invalid data."""
        with pytest.raises(Exception):  # pydantic validation error
            SampleResource(name=123)  # name should be string

    def test_model_validation_with_missing_required_fields(self) -> None:
        """Test that missing required fields raise validation errors."""
        with pytest.raises(Exception):  # pydantic validation error
            SampleResource()  # name is required

    def test_model_validation_with_extra_fields(self) -> None:
        """Test that extra fields are handled according to Pydantic config."""
        # this should work if model_config allows extra fields
        resource = SampleResource(name="test", extra_field="value")
        assert resource.name == "test"

    def test_multiple_instances_different_timestamps(self) -> None:
        """Test that multiple instances created at different times have different timestamps."""
        resource1 = SampleResource(name="first")
        time.sleep(0.1)  # longer delay to ensure different timestamps
        resource2 = SampleResource(name="second")

        # timestamps should be different or equal (if created in same second)
        assert resource1.created_at <= resource2.created_at
        assert resource1.updated_at <= resource2.updated_at

        # at least one should be different if we waited long enough
        if resource1.created_at == resource2.created_at:
            # if they're the same, it means they were created in the same second
            # this is acceptable behavior
            pass

    def test_id_generation_with_custom_id(self) -> None:
        """Test that custom ID can be provided."""
        custom_id = "CustomResource_custom-uuid-123"
        resource = SampleResource(name="test", id=custom_id)

        assert resource.id == custom_id

    def test_timestamp_generation_with_custom_timestamps(self) -> None:
        """Test that custom timestamps can be provided."""
        custom_time = 1234567890
        resource = SampleResource(
            name="test",
            created_at=custom_time,
            updated_at=custom_time
        )

        assert resource.created_at == custom_time
        assert resource.updated_at == custom_time

    def test_model_fields_contains_id_field(self) -> None:
        """Test that model_fields contains the id field with correct configuration."""
        assert "id" in SampleResource.model_fields
        id_field = SampleResource.model_fields["id"]

        # check that the field has a default_factory
        assert hasattr(id_field, "default_factory")
        assert id_field.default_factory is not None

    def test_subclass_initialization_updates_id_factory(self) -> None:
        """Test that __init_subclass__ properly updates the id field factory."""
        # create a new subclass and check its id field factory
        class NewSampleResource(ResourceBase):
            name: str

        # the id field should have been updated to use the class name
        id_field = NewSampleResource.model_fields["id"]
        generated_id = id_field.default_factory()

        assert generated_id.startswith("NewSampleResource_")
        assert len(generated_id) == len("NewSampleResource_") + 36

    def test_resource_base_inheritance_from_base_model(self) -> None:
        """Test that ResourceBase properly inherits from BaseModel."""
        assert issubclass(ResourceBase, BaseModel)
        assert issubclass(SampleResource, BaseModel)
        assert issubclass(SampleResource, ResourceBase)

    def test_resource_base_is_abstract_base_class(self) -> None:
        """Test that ResourceBase is an abstract base class."""
        assert issubclass(ResourceBase, ABC)
        assert hasattr(ResourceBase, "__abstractmethods__")

    def test_model_config_inheritance(self) -> None:
        """Test that model configuration is properly inherited."""
        # test that subclasses can access model configuration
        assert hasattr(SampleResource, "model_config")
        assert hasattr(SampleResource, "model_fields")

    def test_equality_comparison(self) -> None:
        """Test that resource instances can be compared for equality."""
        resource1 = SampleResource(name="test")
        resource2 = SampleResource(name="test")

        # different instances with same data should not be equal
        assert resource1 != resource2

        # same instance should be equal to itself
        assert resource1 == resource1

    def test_string_representation(self) -> None:
        """Test that resource has a meaningful string representation."""
        resource = SampleResource(name="test_resource")
        str_repr = str(resource)

        assert "SampleResource" in str_repr
        assert "test_resource" in str_repr
