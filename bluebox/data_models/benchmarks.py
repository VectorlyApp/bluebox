"""
bluebox/data_models/benchmarks.py

Data models for routine evaluation and benchmarking.
"""

import json
import re
import statistics
import tempfile
import time
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

import jmespath
from jmespath.exceptions import JMESPathError
from openai import OpenAI
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from bluebox.data_models.routine.routine import Routine
from bluebox.sdk import Bluebox


class ExpressionOperator(StrEnum):
    """Operators for evaluating expressions against data."""
    
    # Equality
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    
    # Containment
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    
    # Type checks
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    IS_TYPE = "is_type"
    
    # Comparison (for numbers)
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    
    # String operations
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    MATCHES_REGEX = "matches_regex"
    
    # Collection operations
    LENGTH_EQUALS = "length_equals"
    LENGTH_GREATER_THAN = "length_greater_than"
    LENGTH_LESS_THAN = "length_less_than"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    
    # Existence (for checking if path exists)
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


# Unary operators that only need value_1 (no value_2)
UNARY_OPERATORS: set[ExpressionOperator] = {
    ExpressionOperator.IS_NULL,
    ExpressionOperator.IS_NOT_NULL,
    ExpressionOperator.IS_EMPTY,
    ExpressionOperator.IS_NOT_EMPTY,
    ExpressionOperator.EXISTS,
    ExpressionOperator.NOT_EXISTS,
}


# Operator display symbols for pretty printing
OPERATOR_SYMBOLS: dict[str, str] = {
    "equals": "==",
    "not_equals": "!=",
    "contains": "contains",
    "not_contains": "not contains",
    "is_null": "is null",
    "is_not_null": "is not null",
    "is_type": "is type",
    "greater_than": ">",
    "greater_than_or_equal": ">=",
    "less_than": "<",
    "less_than_or_equal": "<=",
    "starts_with": "starts with",
    "ends_with": "ends with",
    "matches_regex": "matches",
    "length_equals": "length ==",
    "length_greater_than": "length >",
    "length_less_than": "length <",
    "is_empty": "is empty",
    "is_not_empty": "is not empty",
    "exists": "exists",
    "not_exists": "not exists",
}


def _format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "null"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    else:
        return str(value)


def _get_value_at_path(data: Any, path: str) -> tuple[bool, Any]:
    """
    Get value at a JMESPath expression from data.

    Uses JMESPath syntax:
        - Object keys: "user.name"
        - Array indices: "items[0].name", "items[-1]" (last element)
        - Wildcards: "users[*].name" (all names)
        - Filters: "users[?type == 'admin']" (filter by condition)
        - Pipes: "users[?active] | [0]" (first active user)

    See https://jmespath.org/ for full syntax.

    Returns:
        tuple[bool, Any]: (exists, value) - exists is False if path doesn't exist or returns None
    """
    if not path:
        return True, data

    try:
        result = jmespath.search(path, data)
        # JMESPath returns None for non-existent paths
        # We treat None as "exists but is None" only if the path is valid
        # and the actual value is None. Otherwise, it means path doesn't exist.
        if result is None:
            # Check if the path genuinely has a None value vs doesn't exist
            # by checking if parent exists and has the key
            return _check_path_exists(data, path), None
        return True, result
    except JMESPathError:
        return False, None


def _check_path_exists(data: Any, path: str) -> bool:
    """
    Check if a JMESPath actually exists (vs returning None because path is invalid).

    This is needed because JMESPath returns None for both:
    - Path exists but value is None
    - Path doesn't exist
    """
    # For simple paths without filters/wildcards, we can check existence
    # For complex paths, assume None means doesn't exist
    if not path or any(c in path for c in '[]*|?@&!'):
        # Complex JMESPath - if result is None, treat as non-existent
        # unless the entire data is None
        return data is None and path == ''

    # Simple dot-notation path - traverse manually to check existence
    parts = path.split('.')
    current = data

    for part in parts:
        if current is None:
            return False
        if isinstance(current, dict):
            if part not in current:
                return False
            current = current[part]
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return False

    return True


class PathReference(BaseModel):
    """
    A reference to a path in the data, used in SimpleExpression to extract values dynamically.

    Both value_1 and value_2 in SimpleExpression can be PathReferences.

    Example - compare two paths:
        {
            "type": "simple",
            "value_1": {"type": "path", "path": "generated_routine.operations[0].type"},
            "operator": "equals",
            "value_2": {"type": "path", "path": "ground_truth_routine.operations[0].type"}
        }
    """

    type: Literal["path"] = Field(
        default="path",
        description="Discriminator to identify this as a path reference"
    )

    path: str = Field(
        description="JMESPath expression to get the comparison value from the data"
    )


def _resolve_value(value: Any, data: Any) -> tuple[bool, Any]:
    """
    Resolve a value, which may be a literal or a PathReference.

    Args:
        value: The value to resolve (literal or PathReference)
        data: The data to resolve path references against

    Returns:
        tuple[bool, Any]: (success, resolved_value)
            - For literals: (True, value)
            - For PathReference: result of _get_value_at_path
    """
    if isinstance(value, PathReference):
        return _get_value_at_path(data, value.path)
    if isinstance(value, dict) and value.get("type") == "path" and "path" in value:
        # Handle dict representation of PathReference (from JSON)
        return _get_value_at_path(data, value["path"])
    return True, value


class SimpleExpression(BaseModel):
    """
    A simple expression that evaluates a condition using two values and an operator.

    Both value_1 and value_2 can be:
    - A literal value (str, int, float, bool, list, dict, None)
    - A PathReference to extract a value from the data: {"type": "path", "path": "..."}

    Examples:
        Path vs literal:
        {"type": "simple", "value_1": {"type": "path", "path": "steps[0].type"}, "operator": "equals", "value_2": "navigate"}

        Path vs path:
        {"type": "simple",
         "value_1": {"type": "path", "path": "steps[?type == 'fetch'] | [-1].session"},
         "operator": "equals",
         "value_2": {"type": "path", "path": "steps[?type == 'return'] | [-1].session"}}

        Unary operators (value_2 not needed):
        {"type": "simple", "value_1": {"type": "path", "path": "user.name"}, "operator": "exists"}
    """

    type: Literal["simple"] = Field(
        default="simple",
        description="Expression type discriminator"
    )

    value_1: Any = Field(
        description="First operand. Can be a literal or PathReference {'type': 'path', 'path': '...'}"
    )

    operator: ExpressionOperator = Field(
        description="The operator to use for comparison"
    )

    value_2: Any = Field(
        default=None,
        description="Second operand. Can be a literal or PathReference. Not needed for unary operators (exists, is_null, etc.)"
    )

    @model_validator(mode="after")
    def convert_path_dicts_to_path_reference(self) -> "SimpleExpression":
        """Convert dict-format PathReferences to PathReference objects."""
        if isinstance(self.value_1, dict) and self.value_1.get("type") == "path" and "path" in self.value_1:
            object.__setattr__(self, 'value_1', PathReference(path=self.value_1["path"]))
        if isinstance(self.value_2, dict) and self.value_2.get("type") == "path" and "path" in self.value_2:
            object.__setattr__(self, 'value_2', PathReference(path=self.value_2["path"]))
        return self

    def stringify(self) -> str:
        """Convert expression to human-readable string."""
        op_symbol = OPERATOR_SYMBOLS.get(self.operator.value, self.operator.value)

        def format_operand(val: Any) -> str:
            """Format an operand which may be a literal or PathReference."""
            if isinstance(val, PathReference):
                return f"${{{val.path}}}"
            if isinstance(val, dict) and val.get("type") == "path" and "path" in val:
                return f"${{{val['path']}}}"
            return _format_value(val)

        v1_str = format_operand(self.value_1)

        # Unary operators don't need a second value
        if self.operator in UNARY_OPERATORS:
            return f"{v1_str} {op_symbol}"

        v2_str = format_operand(self.value_2)
        return f"{v1_str} {op_symbol} {v2_str}"
    
    def evaluate(self, data: Any) -> bool:
        """
        Evaluate this expression against the given data.

        Args:
            data: The data object to evaluate against (dict, object, etc.)

        Returns:
            bool: True if the expression passes, False otherwise
        """
        # Resolve value_1 (may be literal or PathReference)
        v1_exists, v1 = _resolve_value(self.value_1, data)

        # Handle existence operators first (only need value_1)
        if self.operator == ExpressionOperator.EXISTS:
            return v1_exists
        if self.operator == ExpressionOperator.NOT_EXISTS:
            return not v1_exists

        # For all other operators, value_1 must exist/resolve
        if not v1_exists:
            return False

        # Null checks (unary - only need value_1)
        if self.operator == ExpressionOperator.IS_NULL:
            return v1 is None
        if self.operator == ExpressionOperator.IS_NOT_NULL:
            return v1 is not None

        # Empty checks (unary - only need value_1)
        if self.operator == ExpressionOperator.IS_EMPTY:
            if v1 is None:
                return True
            if isinstance(v1, (str, list, dict, tuple)):
                return len(v1) == 0
            return False
        if self.operator == ExpressionOperator.IS_NOT_EMPTY:
            if v1 is None:
                return False
            if isinstance(v1, (str, list, dict, tuple)):
                return len(v1) > 0
            return True

        # Resolve value_2 for binary operators
        v2_exists, v2 = _resolve_value(self.value_2, data)
        if not v2_exists:
            return False

        # Type check
        if self.operator == ExpressionOperator.IS_TYPE:
            type_map = {
                "str": str, "string": str,
                "int": int, "integer": int,
                "float": float, "number": (int, float),
                "bool": bool, "boolean": bool,
                "list": list, "array": list,
                "dict": dict, "object": dict,
                "none": type(None), "null": type(None),
            }
            expected_type = type_map.get(str(v2).lower())
            if expected_type:
                return isinstance(v1, expected_type)
            return False

        # Equality
        if self.operator == ExpressionOperator.EQUALS:
            return v1 == v2
        if self.operator == ExpressionOperator.NOT_EQUALS:
            return v1 != v2

        # Containment
        if self.operator == ExpressionOperator.CONTAINS:
            if isinstance(v1, str) and isinstance(v2, str):
                return v2 in v1
            if isinstance(v1, (list, tuple)):
                return v2 in v1
            if isinstance(v1, dict):
                return v2 in v1
            return False
        if self.operator == ExpressionOperator.NOT_CONTAINS:
            if isinstance(v1, str) and isinstance(v2, str):
                return v2 not in v1
            if isinstance(v1, (list, tuple)):
                return v2 not in v1
            if isinstance(v1, dict):
                return v2 not in v1
            return True

        # Comparison (numbers)
        if self.operator == ExpressionOperator.GREATER_THAN:
            try:
                return float(v1) > float(v2)
            except (TypeError, ValueError):
                return False
        if self.operator == ExpressionOperator.GREATER_THAN_OR_EQUAL:
            try:
                return float(v1) >= float(v2)
            except (TypeError, ValueError):
                return False
        if self.operator == ExpressionOperator.LESS_THAN:
            try:
                return float(v1) < float(v2)
            except (TypeError, ValueError):
                return False
        if self.operator == ExpressionOperator.LESS_THAN_OR_EQUAL:
            try:
                return float(v1) <= float(v2)
            except (TypeError, ValueError):
                return False

        # String operations
        if self.operator == ExpressionOperator.STARTS_WITH:
            if isinstance(v1, str) and isinstance(v2, str):
                return v1.startswith(v2)
            return False
        if self.operator == ExpressionOperator.ENDS_WITH:
            if isinstance(v1, str) and isinstance(v2, str):
                return v1.endswith(v2)
            return False
        if self.operator == ExpressionOperator.MATCHES_REGEX:
            if isinstance(v1, str) and isinstance(v2, str):
                return bool(re.search(v2, v1))
            return False

        # Length operations
        if self.operator == ExpressionOperator.LENGTH_EQUALS:
            if hasattr(v1, '__len__'):
                return len(v1) == v2
            return False
        if self.operator == ExpressionOperator.LENGTH_GREATER_THAN:
            if hasattr(v1, '__len__'):
                return len(v1) > v2
            return False
        if self.operator == ExpressionOperator.LENGTH_LESS_THAN:
            if hasattr(v1, '__len__'):
                return len(v1) < v2
            return False

        return False


class CompositeExpression(BaseModel):
    """
    A composite expression that combines multiple expressions with AND/OR logic.
    
    Examples:
        {"type": "composite", "logic": "and", "expressions": [...]}
        {"type": "composite", "logic": "or", "expressions": [...]}
    """
    
    type: Literal["composite"] = Field(
        default="composite",
        description="Expression type discriminator"
    )
    
    logic: Literal["and", "or"] = Field(
        description="Logic operator: 'and' (all must pass) or 'or' (at least one must pass)"
    )
    
    expressions: list["Expression"] = Field(
        description="List of expressions to combine. Can be simple or composite (nested)."
    )
    
    def stringify(self) -> str:
        """Convert expression to human-readable string."""
        logic_str = " AND " if self.logic == "and" else " OR "
        parts = [stringify_expression(expr) for expr in self.expressions]
        inner = logic_str.join(parts)
        return f"({inner})"
    
    def evaluate(self, data: Any) -> bool:
        """
        Evaluate this composite expression against the given data.
        
        Args:
            data: The data object to evaluate against (dict, object, etc.)
            
        Returns:
            bool: True if the expression passes, False otherwise
                  - AND: all expressions must pass
                  - OR: at least one expression must pass
        """
        if self.logic == "and":
            return all(evaluate_expression(expr, data) for expr in self.expressions)
        elif self.logic == "or":
            return any(evaluate_expression(expr, data) for expr in self.expressions)
        else:
            raise ValueError(f"Invalid logic: {self.logic}")


def evaluate_expression(expr: "SimpleExpression | CompositeExpression", data: Any) -> bool:
    """
    Evaluate any expression against the given data.
    
    Args:
        expr: The expression to evaluate (simple or composite)
        data: The data object to evaluate against
        
    Returns:
        bool: True if the expression passes, False otherwise
    """
    return expr.evaluate(data)


def stringify_expression(expr: "SimpleExpression | CompositeExpression") -> str:
    """
    Convert any expression to a human-readable string.

    Examples:
        >>> expr = SimpleExpression(
        ...     value_1={"type": "path", "path": "user.name"},
        ...     operator="equals",
        ...     value_2="John"
        ... )
        >>> stringify_expression(expr)
        '${user.name} == "John"'
    """
    return expr.stringify()


# Union type with discriminator for JSON parsing
Expression = Annotated[
    Union[
        SimpleExpression,
        CompositeExpression,
    ],
    Field(discriminator="type")
]

# Update forward reference for CompositeExpression
CompositeExpression.model_rebuild()


class DeterministicTest(BaseModel):
    """
    A deterministic test with a root expression that must pass.

    The expression can be:
    - A simple expression (type: "simple", value_1 + operator + value_2)
    - A composite expression (type: "composite", logic + expressions)

    Example JSON:
        {
            "name": "first_operation_is_navigate",
            "description": "First operation should be navigate",
            "expression": {
                "type": "simple",
                "value_1": {"type": "path", "path": "generated_routine.operations[0].type"},
                "operator": "equals",
                "value_2": "navigate"
            }
        }
    """

    name: str = Field(
        description="Name of the test"
    )

    description: str = Field(
        default="",
        description="Description of what this test validates"
    )

    expression: SimpleExpression | CompositeExpression = Field(
        description="The expression to evaluate. Can be simple or composite (with AND/OR logic)."
    )

    result: bool | None = Field(
        default=None,
        description="Result of the test after running. True if passed, False if failed, None if not run."
    )

    def run(self, data: Any) -> bool:
        """
        Run the deterministic test against the provided data.

        Args:
            data: The data object to evaluate against

        Returns:
            bool: True if the test passed, False otherwise
        """
        self.result = evaluate_expression(self.expression, data)
        return self.result


class LLMTestResult(BaseModel):
    """
    Result of running an LLMTest.
    """

    score: float = Field(
        description="Normalized score produced by the LLM"
    )

    rationale: str | None = Field(
        default=None,
        description="LLM explanation for the score"
    )

    confidence: float | None = Field(
        default=None,
        description="Optional confidence estimate (0.0–1.0)"
    )

    def passed(self, threshold: float | None) -> bool | None:
        if threshold is None:
            return None
        return self.score >= threshold


class LLMTest(BaseModel):
    """
    A non-deterministic test evaluated by an LLM.

    The LLM inspects some data and answers a question or produces a score.
    """

    name: str = Field(
        description="Name of the test"
    )

    description: str = Field(
        default="",
        description="What this test evaluates"
    )

    prompt: str = Field(
        description=(
            "The evaluation prompt given to the LLM. "
            "May reference the data under test."
        )
    )

    model: str = Field(
        description="LLM model identifier used for evaluation (e.g. gpt-4.1, claude-3.5-sonnet)"
    )

    n_trials: int = Field(
        default=3,
        ge=1,
        description="Number of independent LLM evaluations to run"
    )

    score_range: tuple[float, float] = Field(
        default=(0.0, 1.0),
        description="Minimum and maximum possible score"
    )

    passing_threshold: float | None = Field(
        default=None,
        description=(
            "Optional threshold above which the test is considered passing. "
            "If None, the test does not produce pass/fail directly."
        )
    )

    aggregation: Literal["mean", "median", "min", "max"] = Field(
        default="mean",
        description="How to aggregate scores across trials"
    )

    results: list[LLMTestResult] = Field(
        default_factory=list,
        description="Results from running this test. Populated after run() is called."
    )

    def run(self, data: dict[str, Any], client: OpenAI) -> LLMTestResult:
        """
        Run the LLM test against the provided data.

        Uses self.prompt as the evaluation prompt. Runs n_trials evaluations
        and aggregates results according to self.aggregation.

        Args:
            data: Data dictionary to include in the prompt
            client: OpenAI client instance

        Returns:
            LLMTestResult: Aggregated result from n_trials evaluations
        """
        data_dump = json.dumps(data, indent=2)
        full_prompt = (
            f"{self.prompt}\n\n"
            f"Data:\n{data_dump}\n\n"
            f"Provide a score between {self.score_range[0]} and {self.score_range[1]}.\n"
            f"Respond with JSON: {{\"score\": <number>, \"rationale\": \"<explanation>\"}}"
        )

        # Run n_trials evaluations
        trial_results: list[LLMTestResult] = []
        for _ in range(self.n_trials):
            response = client.responses.parse(
                model=self.model,
                input=[{"role": "user", "content": full_prompt}],
                text_format=LLMTestResult
            )
            trial_results.append(response.output_parsed)

        # Aggregate scores
        scores = [r.score for r in trial_results]
        if self.aggregation == "mean":
            final_score = statistics.mean(scores)
        elif self.aggregation == "median":
            final_score = statistics.median(scores)
        elif self.aggregation == "min":
            final_score = min(scores)
        else:  # max
            final_score = max(scores)

        # Aggregate confidence if present
        confidences = [r.confidence for r in trial_results if r.confidence is not None]
        final_confidence = statistics.mean(confidences) if confidences else None

        # Use rationale from result closest to aggregated score
        closest_result = min(trial_results, key=lambda r: abs(r.score - final_score))

        aggregated_result = LLMTestResult(
            score=final_score,
            rationale=closest_result.rationale,
            confidence=final_confidence
        )
        self.results.append(aggregated_result)
        return aggregated_result


class RoutineDiscoveryEvaluation(BaseModel):
    """
    A test case for evaluating routine discovery.

    Contains the task description, expected ground truth routine,
    model to use for discovery, and tests to validate the discovered routine.
    """

    name: str = Field(description="The name of the evaluation")
    description: str = Field(default="", description="The description of the evaluation")
    task: str = Field(description="The task description given to the routine discovery agent")
    ground_truth_routine: Routine = Field(description="The expected routine that should be discovered")
    deterministic_tests: list[DeterministicTest] = Field(
        default_factory=list,
        description="List of deterministic tests to run against the discovered routine"
    )
    llm_tests: list[LLMTest] = Field(
        default_factory=list,
        description="List of LLM-based tests to run against the discovered routine"
    )

    # Results populated after running the evaluation
    model: str | None = Field(default=None, description="The LLM model used for routine discovery")
    generated_routine: Routine | None = Field(default=None, description="The routine generated by the discovery agent")
    discovery_duration: float | None = Field(default=None, description="Time taken to discover the routine in seconds")
    error: str | None = Field(default=None, description="Error message if discovery or evaluation failed")
    summary: dict | None = Field(default=None, description="Structured summary of the evaluation results")

    # Runtime state (excluded from serialization)
    _verbose: bool = PrivateAttr(default=False)

    def _print_if_verbose(self, *args: Any, **kwargs: Any) -> None:
        """Print if verbose mode is enabled."""
        if self._verbose:
            print(*args, **kwargs)

    def run(
        self,
        model: str,
        cdp_captures_dir: str,
        client: OpenAI,
        output_dir: str | None = None,
        run_llm_tests: bool = False,
        verbose: bool = False,
    ) -> "RoutineDiscoveryEvaluation":
        """
        Run the full evaluation pipeline: discover routine, run tests, generate summary.

        Args:
            model: LLM model to use for routine discovery
            cdp_captures_dir: Path to the directory containing CDP captures
            client: OpenAI client instance
            output_dir: Path to output directory (uses temp dir if not provided)
            run_llm_tests: Whether to run LLM-based tests
            verbose: Whether to print detailed progress

        Returns:
            Self with results populated (generated_routine, discovery_duration, error, summary)
        """
        self.model = model
        self._verbose = verbose

        self._print_if_verbose(f"\nRunning routine discovery...")
        self._print_if_verbose(f"  Task: {self.task}")
        self._print_if_verbose(f"  CDP captures: {cdp_captures_dir}")
        self._print_if_verbose(f"  Model: {model}")

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="benchmark_discovery_")
        self._print_if_verbose(f"  Output: {output_dir}")

        start_time = time.time()

        try:
            client = Bluebox(llm_model=model)
            result = client.discover_routine(
                task=self.task,
                cdp_captures_dir=cdp_captures_dir,
                output_dir=output_dir,
            )
            self.discovery_duration = time.time() - start_time
            self._print_if_verbose(f"  Completed in {self.discovery_duration:.1f}s")

            if result and result.routine:
                self.generated_routine = result.routine
            else:
                self.error = "Discovery returned no routine"
                self._print_if_verbose(f"  Warning: {self.error}")

        except Exception as e:
            self.discovery_duration = time.time() - start_time
            self.error = str(e)
            self._print_if_verbose(f"  Failed after {self.discovery_duration:.1f}s: {self.error}")

        # Run tests if we have a generated routine
        if self.generated_routine is not None:
            test_data = {
                "task": self.task,
                "ground_truth_routine": self.ground_truth_routine.model_dump(),
                "generated_routine": self.generated_routine.model_dump(),
            }

            # Deterministic tests
            if self.deterministic_tests:
                self._print_if_verbose(f"\n{'='*60}")
                self._print_if_verbose("Deterministic Tests:")
                self._print_if_verbose(f"{'='*60}")

                for i, test in enumerate(self.deterministic_tests, 1):
                    passed = test.run(test_data)
                    icon = "✓" if passed else "✗"
                    self._print_if_verbose(f"  [{i}/{len(self.deterministic_tests)}] {icon} {test.name}")
                    if not passed:
                        self._print_if_verbose(f"        Expression: {test.expression.stringify()}")

            # LLM tests
            if run_llm_tests and self.llm_tests:
                self._print_if_verbose(f"\n{'='*60}")
                self._print_if_verbose("LLM Tests:")
                self._print_if_verbose(f"{'='*60}")

                for i, test in enumerate(self.llm_tests, 1):
                    result = test.run(data=test_data, client=client)
                    passed = result.passed(test.passing_threshold)
                    icon = "✓" if passed else "✗"
                    self._print_if_verbose(f"  [{i}/{len(self.llm_tests)}] {icon} {test.name}: {result.score:.2f} (threshold: {test.passing_threshold})")

        self.summary = self.summarize_results()
        self._print_if_verbose(f"\n{'='*60}")
        self._print_if_verbose("Summary:")
        self._print_if_verbose(f"{'='*60}")
        det = self.summary.get("deterministic_tests", {})
        llm = self.summary.get("llm_tests", {})
        self._print_if_verbose(f"  Deterministic: {det.get('passed', 0)}/{det.get('total', 0)} ({det.get('pass_rate', 0)*100:.0f}%)")
        self._print_if_verbose(f"  LLM: {llm.get('passed', 0)}/{llm.get('total', 0)} ({llm.get('pass_rate', 0)*100:.0f}%)")

        return self

    def summarize_results(self) -> dict:
        """Generate a structured summary of the evaluation results."""
        # Determine status
        if self.error:
            status = "FAILED"
        elif self.generated_routine is None:
            status = "NOT_RUN"
        else:
            status = "SUCCESS"

        # Deterministic test results
        det_passed = sum(1 for t in self.deterministic_tests if t.result is True)
        det_total = len(self.deterministic_tests)
        det_failed = [t.name for t in self.deterministic_tests if t.result is False]

        deterministic_summary = {
            "passed": det_passed,
            "total": det_total,
            "pass_rate": round(det_passed / det_total, 3) if det_total > 0 else 0.0,
            "failed_tests": det_failed,
        }

        # LLM test results
        llm_results = {}
        llm_passed = 0
        llm_total = 0
        for t in self.llm_tests:
            if t.results:
                latest = t.results[-1]
                passed = latest.passed(t.passing_threshold)
                llm_results[t.name] = {
                    "score": round(latest.score, 3),
                    "threshold": t.passing_threshold,
                    "passed": passed,
                    "rationale": latest.rationale,
                }
                if passed:
                    llm_passed += 1
                llm_total += 1

        llm_summary = {
            "passed": llm_passed,
            "total": llm_total,
            "pass_rate": round(llm_passed / llm_total, 3) if llm_total > 0 else 0.0,
            "results": llm_results,
        }

        # Overall
        total_passed = det_passed + llm_passed
        total_tests = det_total + llm_total

        return {
            "status": status,
            "error": self.error,
            "discovery_duration_seconds": round(self.discovery_duration, 2) if self.discovery_duration else None,
            "deterministic_tests": deterministic_summary,
            "llm_tests": llm_summary,
            "overall": {
                "passed": total_passed,
                "total": total_tests,
                "pass_rate": round(total_passed / total_tests, 3) if total_tests > 0 else 0.0,
            },
        }

