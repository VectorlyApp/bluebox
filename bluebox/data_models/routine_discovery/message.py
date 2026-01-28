"""
bluebox/data_models/routine_discovery/message.py

Routine discovery message data models.
"""

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class RoutineDiscoveryMessageType(StrEnum):
    """
    Enum for routine discovery message types.
    """
    INITIATED = "initiated"
    PROGRESS_THINKING = "progress-thinking"
    PROGRESS_RESULT = "progress-result"
    FINISHED = "finished"
    ERROR = "error"


class RoutineDiscoveryMessage(BaseModel):
    """
    Base model for all routine discovery messages.
    """
    type: RoutineDiscoveryMessageType = Field(
        ...,
        description="The type of the discovery message"
    )
    timestamp: float = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc).timestamp(),
        description="Timestamp when the message was created"
    )
    content: str = Field(
        ...,
        description="The content of the message"
    )
