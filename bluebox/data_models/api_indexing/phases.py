"""
Phases for the API indexing pipeline.
"""

from enum import Enum


class ApiIndexingPhase(str, Enum):
    """The current phase of the API indexing pipeline."""

    EXPLORATION = "exploration"
