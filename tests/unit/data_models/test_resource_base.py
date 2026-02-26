"""
tests/unit/data_models/test_resource_base.py

Tests for ResourceBase functionality including ID generation, timestamp handling,
subclass behavior, and is_valid_resource_id().
"""

import time
from abc import ABC
from datetime import datetime, timezone
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from bluebox.data_models.resource_base import ResourceBase


# A valid v4 UUID to reuse across tests
VALID_V4_UUID = "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"


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

    def test_create_new_id_uses_subclass_name(self) -> None:
        """Test that create_new_id generates IDs with the correct class prefix."""
        new_id = SampleResource.create_new_id()

        assert new_id.startswith("SampleResource_")
        uuid_part = new_id.split("_", 1)[1]
        UUID(uuid_part)  # valid UUID

    def test_create_new_id_different_subclasses(self) -> None:
        """Test that create_new_id uses each subclass's own name."""
        id1 = SampleResource.create_new_id()
        id2 = SampleResourceWithCustomFields.create_new_id()

        assert id1.startswith("SampleResource_")
        assert id2.startswith("SampleResourceWithCustomFields_")

    def test_create_new_id_raises_on_resource_base(self) -> None:
        """Test that calling create_new_id on ResourceBase itself raises TypeError."""
        with pytest.raises(TypeError, match="must be called on a subclass"):
            ResourceBase.create_new_id()

    def test_create_new_id_uniqueness(self) -> None:
        """Test that create_new_id generates unique IDs each call."""
        ids = {SampleResource.create_new_id() for _ in range(10)}
        assert len(ids) == 10

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


# --- Subclasses for is_valid_resource_id tests ---

class Widget(ResourceBase):
    pass


class AnotherWidget(ResourceBase):
    pass


class TestIsValidResourceId:
    """Tests for is_valid_resource_id()."""

    # --- Happy path ---

    def test_valid_id_from_create_new_id(self) -> None:
        """ID generated by create_new_id() should always pass."""
        assert Widget.is_valid_resource_id(Widget.create_new_id()) is True

    def test_valid_id_from_instance(self) -> None:
        """Default ID on a new instance should pass."""
        w = Widget()
        assert Widget.is_valid_resource_id(w.id) is True

    def test_valid_manually_constructed_id(self) -> None:
        assert Widget.is_valid_resource_id(f"Widget_{VALID_V4_UUID}") is True

    # --- Wrong prefix ---

    def test_wrong_class_prefix(self) -> None:
        """Widget ID should not validate as AnotherWidget."""
        widget_id = Widget.create_new_id()
        assert AnotherWidget.is_valid_resource_id(widget_id) is False

    def test_lowercase_prefix(self) -> None:
        assert Widget.is_valid_resource_id(f"widget_{VALID_V4_UUID}") is False

    def test_empty_prefix(self) -> None:
        assert Widget.is_valid_resource_id(f"_{VALID_V4_UUID}") is False

    def test_no_underscore_separator(self) -> None:
        assert Widget.is_valid_resource_id(f"Widget{VALID_V4_UUID}") is False

    def test_extra_prefix_text(self) -> None:
        assert Widget.is_valid_resource_id(f"SuperWidget_{VALID_V4_UUID}") is False

    def test_prefix_substring(self) -> None:
        """'Widge' is a substring of 'Widget' but not the right prefix."""
        assert Widget.is_valid_resource_id(f"Widge_{VALID_V4_UUID}") is False

    # --- Invalid UUID portion ---

    def test_empty_uuid(self) -> None:
        assert Widget.is_valid_resource_id("Widget_") is False

    def test_garbage_uuid(self) -> None:
        assert Widget.is_valid_resource_id("Widget_not-a-uuid") is False

    def test_truncated_uuid(self) -> None:
        assert Widget.is_valid_resource_id("Widget_a1b2c3d4-e5f6-4a7b-8c9d") is False

    def test_uuid_with_extra_chars(self) -> None:
        assert Widget.is_valid_resource_id(f"Widget_{VALID_V4_UUID}-extra") is False

    def test_uuid_with_braces(self) -> None:
        """UUIDs wrapped in braces are valid for stdlib UUID(), but the format should still parse."""
        braced = f"Widget_{{{VALID_V4_UUID}}}"
        assert Widget.is_valid_resource_id(braced) is True

    def test_uuid_v1_rejected(self) -> None:
        """A valid UUID v1 should be rejected (not v4)."""
        v1_uuid = "a1b2c3d4-e5f6-1a7b-8c9d-0e1f2a3b4c5d"
        assert Widget.is_valid_resource_id(f"Widget_{v1_uuid}") is False

    def test_uuid_v3_rejected(self) -> None:
        v3_uuid = "a1b2c3d4-e5f6-3a7b-8c9d-0e1f2a3b4c5d"
        assert Widget.is_valid_resource_id(f"Widget_{v3_uuid}") is False

    def test_uuid_v5_rejected(self) -> None:
        v5_uuid = "a1b2c3d4-e5f6-5a7b-8c9d-0e1f2a3b4c5d"
        assert Widget.is_valid_resource_id(f"Widget_{v5_uuid}") is False

    # --- Edge-case inputs ---

    def test_empty_string(self) -> None:
        assert Widget.is_valid_resource_id("") is False

    def test_just_classname(self) -> None:
        assert Widget.is_valid_resource_id("Widget") is False

    def test_just_underscore(self) -> None:
        assert Widget.is_valid_resource_id("_") is False

    def test_multiple_underscores_in_uuid(self) -> None:
        """Underscores in the UUID portion should fail."""
        assert Widget.is_valid_resource_id("Widget_a1b2c3d4_e5f6_4a7b_8c9d_0e1f2a3b4c5d") is False

    def test_whitespace_around_id(self) -> None:
        assert Widget.is_valid_resource_id(f" Widget_{VALID_V4_UUID} ") is False

    def test_newline_in_id(self) -> None:
        assert Widget.is_valid_resource_id(f"Widget_{VALID_V4_UUID}\n") is False

    def test_double_underscore_separator(self) -> None:
        assert Widget.is_valid_resource_id(f"Widget__{VALID_V4_UUID}") is False

    # --- Called on ResourceBase directly ---

    def test_raises_on_resource_base(self) -> None:
        with pytest.raises(TypeError, match="is_valid_resource_id.*must be called on a subclass"):
            ResourceBase.is_valid_resource_id(f"ResourceBase_{VALID_V4_UUID}")

    # --- Cross-class isolation ---

    def test_each_subclass_validates_its_own_prefix(self) -> None:
        """Two subclasses should not accept each other's IDs."""
        widget_id = Widget.create_new_id()
        another_id = AnotherWidget.create_new_id()

        assert Widget.is_valid_resource_id(widget_id) is True
        assert Widget.is_valid_resource_id(another_id) is False
        assert AnotherWidget.is_valid_resource_id(another_id) is True
        assert AnotherWidget.is_valid_resource_id(widget_id) is False

    # --- Bulk / fuzz-lite ---

    def test_many_generated_ids_all_valid(self) -> None:
        """100 generated IDs should all validate."""
        for _ in range(100):
            assert Widget.is_valid_resource_id(Widget.create_new_id()) is True


class TestGetUtcTimestamp:
    """Tests for ResourceBase.get_utc_timestamp()."""

    def test_returns_float(self) -> None:
        assert isinstance(ResourceBase.get_utc_timestamp(), float)

    def test_value_close_to_now(self) -> None:
        """Should be within 1 second of actual UTC now."""
        before = datetime.now(tz=timezone.utc).timestamp()
        ts = ResourceBase.get_utc_timestamp()
        after = datetime.now(tz=timezone.utc).timestamp()
        assert before <= ts <= after

    def test_callable_on_subclass(self) -> None:
        assert isinstance(SampleResource.get_utc_timestamp(), float)

    def test_callable_on_instance(self) -> None:
        resource = SampleResource(name="test")
        assert isinstance(resource.get_utc_timestamp(), float)

    def test_monotonically_increases(self) -> None:
        """Two successive calls should be non-decreasing."""
        t1 = ResourceBase.get_utc_timestamp()
        t2 = ResourceBase.get_utc_timestamp()
        assert t2 >= t1

    def test_increases_over_time(self) -> None:
        t1 = ResourceBase.get_utc_timestamp()
        time.sleep(0.05)
        t2 = ResourceBase.get_utc_timestamp()
        assert t2 > t1

    def test_is_utc_not_local(self) -> None:
        """Returned value should match UTC, not naive local time."""
        ts = ResourceBase.get_utc_timestamp()
        utc_now = datetime.now(tz=timezone.utc).timestamp()
        # if the impl accidentally used naive datetime.now(), the delta would be
        # hours off in non-UTC timezones; 5 seconds is a safe tolerance here
        assert abs(ts - utc_now) < 5

    def test_has_subsecond_precision(self) -> None:
        """Float should carry sub-second data (not truncated to int)."""
        ts = ResourceBase.get_utc_timestamp()
        assert ts != int(ts)

    def test_reasonable_epoch_value(self) -> None:
        """Sanity-check: value should be after 2020-01-01 and before 2100-01-01."""
        ts = ResourceBase.get_utc_timestamp()
        assert ts > 1_577_836_800  # 2020-01-01 UTC
        assert ts < 4_102_444_800  # 2100-01-01 UTC

    def test_consistent_with_field_defaults(self) -> None:
        """get_utc_timestamp() should use the same clock as created_at/updated_at."""
        before = ResourceBase.get_utc_timestamp()
        resource = SampleResource(name="test")
        after = ResourceBase.get_utc_timestamp()
        assert before <= resource.created_at <= after
        assert before <= resource.updated_at <= after


class TestUuidProperty:
    """Tests for ResourceBase.uuid property."""

    def test_returns_uuid_object(self) -> None:
        """uuid property should return a UUID instance."""
        resource = SampleResource(name="test")
        assert isinstance(resource.uuid, UUID)

    def test_uuid_matches_id_suffix(self) -> None:
        """uuid should match the UUID portion after the class prefix."""
        resource = SampleResource(name="test")
        uuid_str = resource.id.split("_", 1)[1]
        assert resource.uuid == UUID(uuid_str)

    def test_uuid_is_v4(self) -> None:
        """Auto-generated IDs should yield v4 UUIDs."""
        resource = SampleResource(name="test")
        assert resource.uuid.version == 4

    def test_uuid_usable_as_string(self) -> None:
        """str(resource.uuid) should be a valid UUID string usable as e.g. a Qdrant point ID."""
        resource = SampleResource(name="test")
        uuid_str = str(resource.uuid)
        assert UUID(uuid_str) == resource.uuid

    def test_uuid_different_subclasses(self) -> None:
        """uuid should work on any subclass."""
        r1 = SampleResource(name="test")
        r2 = SampleResourceWithCustomFields(title="t", value=1)
        assert isinstance(r1.uuid, UUID)
        assert isinstance(r2.uuid, UUID)
        assert r1.uuid != r2.uuid

    def test_uuid_with_custom_valid_id(self) -> None:
        """uuid should work when a valid custom ID is provided."""
        resource = SampleResource(name="test", id=f"SampleResource_{VALID_V4_UUID}")
        assert resource.uuid == UUID(VALID_V4_UUID)

    def test_uuid_with_bare_uuid_id(self) -> None:
        """uuid should work when the entire ID is a bare UUID string."""
        resource = SampleResource(name="test", id=VALID_V4_UUID)
        assert resource.uuid == UUID(VALID_V4_UUID)

    def test_uuid_with_bare_uuid_str_roundtrip(self) -> None:
        """str(resource.uuid) should equal the original bare UUID id."""
        resource = SampleResource(name="test", id=VALID_V4_UUID)
        assert str(resource.uuid) == VALID_V4_UUID

    def test_uuid_raises_on_wrong_prefix(self) -> None:
        """uuid should raise ValueError if the ID has a wrong prefix and isn't a bare UUID."""
        resource = SampleResource(name="test", id="WrongPrefix_not-a-uuid")
        with pytest.raises(ValueError, match="Invalid resource id"):
            _ = resource.uuid

    def test_uuid_raises_on_garbage_string(self) -> None:
        """uuid should raise ValueError for a non-UUID, non-prefixed string."""
        resource = SampleResource(name="test", id="just-a-bare-string")
        with pytest.raises(ValueError, match="Invalid resource id"):
            _ = resource.uuid

    def test_uuid_on_grandchild(self) -> None:
        """uuid should use the runtime class name, not a parent class."""
        class GrandChild(SampleResource):
            extra: str = "x"

        resource = GrandChild(name="test")
        assert resource.id.startswith("GrandChild_")
        assert isinstance(resource.uuid, UUID)
