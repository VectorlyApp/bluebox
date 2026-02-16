"""
bluebox/data_models/resource_base.py

Base class for all resources that provides a standardized ID format.

ID format: [resourceType]_[uuidv4]
Examples: "Routine_123e4567-e89b-12d3-a456-426614174000"
"""

from abc import ABC
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

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
        description="Resource ID in format [resourceType]_[uuidv4]"
    )

    created_at: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Unix timestamp (seconds) when resource was created"
    )
    updated_at: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Unix timestamp (seconds) when resource was last updated"
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata for the resource. Anythning that is not suitable for a regular field."
    )

    @classmethod
    def create_new_id(cls) -> str:
        """Generate a new ID in format [ClassName]_[uuidv4]."""
        return f"{cls.__name__}_{uuid4()}"

    @property
    def resource_type(self) -> str:
        """
        Return the resource type name (class name) for this class.
        """
        return self.__class__.__name__

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Initialize subclass by setting up the correct default_factory for the id field.
        This method is called when a class inherits from ResourceBase. It ensures
        that each subclass gets an id field with a default_factory that generates
        IDs in the format "[ClassName]_[uuid4]".
        Args:
            cls: The subclass being initialized
            **kwargs: Additional keyword arguments passed to the subclass
        """
        super().__init_subclass__(**kwargs)
        # override the default_factory for the id field to use the actual class name
        if hasattr(cls, 'model_fields') and 'id' in cls.model_fields:
            cls.model_fields['id'].default_factory = lambda: f"{cls.__name__}_{uuid4()}"
