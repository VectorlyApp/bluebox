"""
bluebox/data_models/resource_base.py

Base class for all resources that provides a standardized ID format.

ID format: [resourceType]_[uuidv4]
Examples: "Routine_123e4567-e89b-12d3-a456-426614174000"
"""

from abc import ABC
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResourceBase(BaseModel, ABC):
    """
    Base class for all resources that provides a standardized ID format.
    
    ID format: [resourceType]_[uuidv4]
    Examples: "Routine_123e4567-e89b-12d3-a456-426614174000"
    """

    # standardized resource ID in format "[resourceType]_[uuidv4]"
    id: str = Field(
        default_factory=lambda: f"ResourceBase_{uuid4()}",
        description="Resource ID in format [resourceType]_[uuidv4]",
    )

    created_at: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Unix timestamp in seconds (float) when resource was created",
    )
    updated_at: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Unix timestamp in seconds (float) when resource was last updated",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata for the resource. Anything that is not suitable for a regular field.",
    )

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Set up the correct ``default_factory`` for the ``id`` field.

        Called automatically when a class inherits from ``ResourceBase``.
        Ensures each subclass generates IDs in the format
        ``[ClassName]_[uuid4]``.
        """
        super().__init_subclass__(**kwargs)
        # override the default_factory for the id field to use the actual class name
        if hasattr(cls, 'model_fields') and 'id' in cls.model_fields:
            cls.model_fields['id'].default_factory = lambda: f"{cls.__name__}_{uuid4()}"

    @classmethod
    def is_valid_resource_id(cls, id_str: str) -> bool:
        """
        Check whether an ID string matches this resource's expected format.

        The expected format is ``[ClassName]_[uuidv4]``, where *ClassName*
        is the name of the subclass this method is called on.

        Args:
            id_str: The ID string to validate.

        Returns:
            ``True`` if *id_str* has the correct prefix and a valid v4 UUID
            suffix, ``False`` otherwise.

        Raises:
            TypeError: If called directly on ``ResourceBase`` rather than a
                subclass.
        """
        if cls is ResourceBase:
            raise TypeError("is_valid_resource_id() must be called on a subclass, not ResourceBase")
        prefix = f"{cls.__name__}_"
        if not id_str.startswith(prefix):
            return False
        uuid_part = id_str[len(prefix):]
        try:
            # UUID() accepts any valid UUID; check version == 4 explicitly
            return UUID(uuid_part).version == 4
        except ValueError:
            return False

    @classmethod
    def create_new_id(cls) -> str:
        """
        Generate a new ID in the format ``[ClassName]_[uuidv4]``.

        Returns:
            A string ID with the subclass name as prefix and a fresh UUIDv4.

        Raises:
            TypeError: If called directly on ``ResourceBase`` rather than a
                subclass.
        """
        if cls is ResourceBase:
            raise TypeError("create_new_id() must be called on a subclass, not ResourceBase")
        return f"{cls.__name__}_{uuid4()}"

    @staticmethod
    def get_utc_timestamp() -> float:
        """
        Return the current UTC time as a Unix timestamp.
            Useful for updating ``updated_at`` in a consistent manner.

        Returns:
            Seconds since the Unix epoch as a float with millisecond precision,
            equivalent to ``datetime.now(tz=timezone.utc).timestamp()``.
        """
        return datetime.now(tz=timezone.utc).timestamp()

    @property
    def resource_type(self) -> str:
        """Return the resource type name (class name) for this instance."""
        return self.__class__.__name__
