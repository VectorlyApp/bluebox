"""
web_hacker/data_models/monitor/window_property.py

Window property data models.
"""

from typing import Any

from pydantic import BaseModel, Field


class WindowPropertyValue(BaseModel):
    """
    A single window property value entry with timestamp and URL context.
    """
    timestamp: float = Field(..., description="Timestamp when the property value was observed")
    value: Any = Field(..., description="The value of the window property (can be int, str, bool, etc.)")
    url: str = Field(..., description="The URL where this property value was observed")


class WindowProperty(BaseModel):
    """
    A window property identified by its dot-separated path, containing a list of observed values.
    """
    path: str = Field(..., description="Dot-separated path to the window property (e.g., 'ncbi.awesome.articlePage.logger.levels.DEBUG')")
    values: list[WindowPropertyValue] = Field(..., description="List of observed values for this property over time")

