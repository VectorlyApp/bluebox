"""
web_hacker/data_models/llm_responses.py

LLM response data models.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from web_hacker.data_models.routine.endpoint import HTTPMethod


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TransactionIdentificationResponse(BaseModel):
    """
    Response from the LLM for identifying the network transaction that directly corresponds to
    the user's requested task. 
    """
    transaction_id: str | None
    description: str
    url: str
    method: HTTPMethod
    explanation: str
    confidence_level: ConfidenceLevel = Field(
        description="The confidence level of the identification. May be LOW, MEDIUM, or HIGH."
    )


class TransactionConfirmationResponse(BaseModel):
    """
    Response obejct for confirming the identified network transactions that directly correspond to
    the user's requested task.
    """
    is_correct: bool = Field(
        description="Whether the identified network transaction directly corresponds to the user's requested task."
    )
    confirmed_transaction_id: str = Field(
        description="The ID of the network transaction that directly corresponds to the user's requested task."
    )
    explanation: str = Field(
        description="The explanation of the confirmation. Should be a short explanation of why the transaction is correct."
    )
    confidence_level: ConfidenceLevel = Field(
        description="The confidence level of the confirmation. May be LOW, MEDIUM, or HIGH."
    )


class VariableType(StrEnum):
    PARAMETER = "parameter"          # User input (e.g. query, item_id)
    DYNAMIC_TOKEN = "dynamic_token"  # Auth tokens, CSRF, rotating cookies
    STATIC_VALUE = "static_value"    # Hardcoded constants, version strings, user-agent parts
    

class Variable(BaseModel):
    """
    A variable that was extracted from the network transaction.
    """
    type: VariableType
    requires_resolution: bool = Field(description="Whether the variable requires resolution.")
    name: str
    observed_value: str
    values_to_scan_for: list[str]
    description: str


class ExtractedVariableResponse(BaseModel):
    """
    Response from the LLM for extracting variables from the network transaction.
    """
    transaction_id: str
    variables: list[Variable]
    explanation: str


class SessionStorageType(StrEnum):
    COOKIE = "cookie"
    LOCAL_STORAGE = "localStorage"
    SESSION_STORAGE = "sessionStorage"
    

class SessionStorageSource(BaseModel):
    """
    Source of the session storage.
    """
    type: SessionStorageType = Field(description="The type of the session storage.")
    dot_path: str = Field(description="The dot path to the variable in the session storage.")
    
class WindowPropertySource(BaseModel):
    """
    Source of the window property.
    """
    dot_path: str = Field(description="The dot path to the variable in the window property (key of the window property dictionary)")


class TransactionSource(BaseModel):
    """
    Source of the transaction.
    """
    transaction_id: str = Field(description="The ID of the transaction that contains the variable.")
    dot_path: str = Field(description="The dot path to the variable in the transaction response body.")


class ResolvedVariableResponse(BaseModel):
    """
    Response from the LLM for resolving cookies and tokens.
    """
    variable: Variable
    session_storage_source: SessionStorageSource | None = None
    transaction_source: TransactionSource | None = None
    window_property_source: WindowPropertySource | None = None
    explanation: str


class TestParameter(BaseModel):
    """
    A test parameter for a routine.
    """
    name: str
    value: str


class TestParametersResponse(BaseModel):
    """
    Response from the LLM for getting the test parameters for a routine.
    """
    parameters: list[TestParameter]
