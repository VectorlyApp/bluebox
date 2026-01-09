"""
Test file to demonstrate TOON encoding of all LLM response model schemas.
"""

import json

import tiktoken
from toon import encode

from web_hacker.data_models.routine_discovery.llm_responses import (
    ConfidenceLevel,
    ExtractedVariableResponse,
    ResolvedVariableResponse,
    SessionStorageSource,
    TestParameter,
    TestParametersResponse,
    TransactionConfirmationResponse,
    TransactionIdentificationResponse,
    Variable,
    VariableType,
    WindowPropertySource,
    TransactionSource,
    SessionStorageType,
)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in a text string using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def test_encode_all_model_schemas() -> None:
    """Encode all LLM response model schemas to TOON format."""
    
    # Collect all BaseModel classes
    models = [
        TransactionIdentificationResponse,
        TransactionConfirmationResponse,
        Variable,
        ExtractedVariableResponse,
        SessionStorageSource,
        WindowPropertySource,
        TransactionSource,
        ResolvedVariableResponse,
        TestParameter,
        TestParametersResponse,
    ]
    
    print("\n" + "=" * 80)
    print("TOON Encoding of LLM Response Model Schemas")
    print("=" * 80 + "\n")
    
    # Track totals
    total_json_tokens = 0
    total_toon_tokens = 0
    
    for model in models:
        model_name = model.__name__
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name}")
        print(f"{'─' * 80}\n")
        
        # Get the JSON schema
        schema = model.model_json_schema()
        json_string = json.dumps(schema, indent=2)
        
        # Encode to TOON
        toon_string = encode(schema)
        
        # Count tokens
        json_tokens = count_tokens(json_string)
        toon_tokens = count_tokens(toon_string)
        tokens_saved = json_tokens - toon_tokens
        percent_saved = (tokens_saved / json_tokens * 100) if json_tokens > 0 else 0
        
        # Update totals
        total_json_tokens += json_tokens
        total_toon_tokens += toon_tokens
        
        print("JSON Schema:")
        print(json_string)
        
        print("\n" + "-" * 80)
        print("TOON Encoding:")
        print("-" * 80 + "\n")
        print(toon_string)
        
        print("\n" + "=" * 80)
        print("Token Statistics:")
        print("=" * 80)
        print(f"JSON tokens:  {json_tokens:,}")
        print(f"TOON tokens:  {toon_tokens:,}")
        print(f"Tokens saved: {tokens_saved:,} ({percent_saved:.1f}%)")
        print("=" * 80)
    
    # Print summary
    total_tokens_saved = total_json_tokens - total_toon_tokens
    total_percent_saved = (total_tokens_saved / total_json_tokens * 100) if total_json_tokens > 0 else 0
    
    print("\n\n" + "=" * 80)
    print("SUMMARY - All Models Combined")
    print("=" * 80)
    print(f"Total JSON tokens:  {total_json_tokens:,}")
    print(f"Total TOON tokens:  {total_toon_tokens:,}")
    print(f"Total tokens saved: {total_tokens_saved:,} ({total_percent_saved:.1f}%)")
    print("=" * 80 + "\n")


def create_sample_instances() -> dict:
    """Create sample instances of all models."""
    from web_hacker.data_models.routine.endpoint import HTTPMethod
    
    return {
        TransactionIdentificationResponse: TransactionIdentificationResponse(
            transaction_id="txn_12345",
            description="API call to search for flights",
            url="https://api.example.com/flights/search",
            method=HTTPMethod.POST,
            explanation="This transaction matches the user's request to search for flights",
            confidence_level=ConfidenceLevel.HIGH,
        ),
        TransactionConfirmationResponse: TransactionConfirmationResponse(
            is_correct=True,
            confirmed_transaction_id="txn_12345",
            explanation="This is the correct transaction that handles flight searches",
            confidence_level=ConfidenceLevel.HIGH,
        ),
        Variable: Variable(
            type=VariableType.PARAMETER,
            requires_dynamic_resolution=False,
            name="search_query",
            observed_value="New York to London",
            values_to_scan_for=["New York", "London", "search_query"],
            description="The search query parameter for flight search",
        ),
        ExtractedVariableResponse: ExtractedVariableResponse(
            transaction_id="txn_12345",
            variables=[
                Variable(
                    type=VariableType.PARAMETER,
                    requires_dynamic_resolution=False,
                    name="origin",
                    observed_value="NYC",
                    values_to_scan_for=["NYC", "origin"],
                    description="Origin airport code",
                ),
                Variable(
                    type=VariableType.PARAMETER,
                    requires_dynamic_resolution=False,
                    name="destination",
                    observed_value="LHR",
                    values_to_scan_for=["LHR", "destination"],
                    description="Destination airport code",
                ),
            ],
            explanation="Extracted origin and destination parameters from the transaction",
        ),
        SessionStorageSource: SessionStorageSource(
            type=SessionStorageType.COOKIE,
            dot_path="session.auth_token",
        ),
        WindowPropertySource: WindowPropertySource(
            dot_path="window.userPreferences.theme",
        ),
        TransactionSource: TransactionSource(
            transaction_id="txn_67890",
            dot_path="response.data.token",
        ),
        ResolvedVariableResponse: ResolvedVariableResponse(
            variable=Variable(
                type=VariableType.DYNAMIC_TOKEN,
                requires_dynamic_resolution=True,
                name="auth_token",
                observed_value="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
                values_to_scan_for=["auth_token", "token", "jwt"],
                description="JWT authentication token",
            ),
            session_storage_source=SessionStorageSource(
                type=SessionStorageType.COOKIE,
                dot_path="session.auth_token",
            ),
            explanation="Token found in session cookie",
        ),
        TestParameter: TestParameter(
            name="origin",
            value="NYC",
        ),
        TestParametersResponse: TestParametersResponse(
            parameters=[
                TestParameter(name="origin", value="NYC"),
                TestParameter(name="destination", value="LHR"),
                TestParameter(name="date", value="2024-12-25"),
            ],
        ),
    }


def test_encode_all_model_instances() -> None:
    """Encode all LLM response model instances to TOON format."""
    
    instances = create_sample_instances()
    
    print("\n" + "=" * 80)
    print("TOON Encoding of LLM Response Model Instances")
    print("=" * 80 + "\n")
    
    # Track totals
    total_json_tokens = 0
    total_toon_tokens = 0
    
    for model_class, instance in instances.items():
        model_name = model_class.__name__
        print(f"\n{'─' * 80}")
        print(f"Model: {model_name}")
        print(f"{'─' * 80}\n")
        
        # Convert to JSON
        json_string = instance.model_dump_json(indent=2)
        
        # Encode to TOON using model_dump() to get dict
        instance_dict = instance.model_dump()
        toon_string = encode(instance_dict)
        
        # Count tokens
        json_tokens = count_tokens(json_string)
        toon_tokens = count_tokens(toon_string)
        tokens_saved = json_tokens - toon_tokens
        percent_saved = (tokens_saved / json_tokens * 100) if json_tokens > 0 else 0
        
        # Update totals
        total_json_tokens += json_tokens
        total_toon_tokens += toon_tokens
        
        print("JSON Instance:")
        print(json_string)
        
        print("\n" + "-" * 80)
        print("TOON Encoding:")
        print("-" * 80 + "\n")
        print(toon_string)
        
        print("\n" + "=" * 80)
        print("Token Statistics:")
        print("=" * 80)
        print(f"JSON tokens:  {json_tokens:,}")
        print(f"TOON tokens:  {toon_tokens:,}")
        print(f"Tokens saved: {tokens_saved:,} ({percent_saved:.1f}%)")
        print("=" * 80)
    
    # Print summary
    total_tokens_saved = total_json_tokens - total_toon_tokens
    total_percent_saved = (total_tokens_saved / total_json_tokens * 100) if total_json_tokens > 0 else 0
    
    print("\n\n" + "=" * 80)
    print("SUMMARY - All Model Instances Combined")
    print("=" * 80)
    print(f"Total JSON tokens:  {total_json_tokens:,}")
    print(f"Total TOON tokens:  {total_toon_tokens:,}")
    print(f"Total tokens saved: {total_tokens_saved:,} ({total_percent_saved:.1f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_encode_all_model_schemas()
    print("\n\n")
    test_encode_all_model_instances()

