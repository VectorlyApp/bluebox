"""
src/data_models/llm_responses.py

LLM response data models.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

from src.data_models.network import Method


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
    method: Method
    explanation: str
    confidence_level: ConfidenceLevel
    
    
class TransactionConfirmationResponse(BaseModel):
    """
    Response obejct for confirming the identified network transactions that directly correspond to
    the user's requested task.
    """
    is_correct: bool
    confirmed_transaction_id: str
    explanation: str
    confidence_level: ConfidenceLevel


class VariableType(StrEnum):
    ARGUMENT = "argument"
    COOKIE = "cookie"
    TOKEN = "token"
    BROWSER_VARIABLE = "browser_variable"
    # CONSTANT = "constant"
    

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
