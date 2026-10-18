"""Prompt-derived property verification for Python code candidates.

This module adds a lightweight deterministic verifier for HumanEval-style code
tasks. It derives extra checks from the prompt intent, function signature,
docstring examples, and official test harness, then converts failures into
pipeline-compatible constraint results for repair feedback.

Spec: REQ-CODE-007, REQ-CODE-008,
      SCENARIO-CODE-006, SCENARIO-CODE-007
"""

from __future__ import annotations

import ast
import copy
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from carnot.pipeline.extract import ConstraintResult
from carnot.verify.python_types import safe_exec_function

ExampleCase = tuple[tuple[Any, ...], Any]

_SORT_ASCENDING_PATTERN = re.compile(
    r"\b(sort|sorted|ascending|in ascending order|increasing order)\b",
    flags=re.IGNORECASE,
)
_SORT_DESCENDING_PATTERN = re.compile(
    r"\b(descending|decreasing order|reverse sorted)\b",
    flags=re.IGNORECASE,
)
_REVERSE_PATTERN = re.compile(
    r"\b(reverse|reversed|backwards)\b",
    flags=re.IGNORECASE,
)
_IN_PLACE_PATTERN = re.compile(
    r"\b(in place|mutate|modify the input|update the input)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class DerivedProperty:
    """One derived property with its source and description."""

    name: str
    source: str
    description: str


@dataclass(frozen=True)
class PropertyFailure:
    """One prompt-derived property failure."""

    property_name: str
    source: str
    input_args: tuple[Any, ...]
    description: str
    actual: str = ""
    expected: str = ""
    error: str | None = None

    def to_constraint_result(self) -> ConstraintResult:
        """Convert the failure into pipeline-compatible repair feedback."""
        outcome = f"raised {self.error}" if self.error is not None else f"returned {self.actual}"

        expected_suffix = f" expected {self.expected};" if self.expected else ""
        return ConstraintResult(
            constraint_type="property_code",
            description=(
                f"{self.property_name} ({self.source}) failed for input={self.input_args}:"
                f"{expected_suffix} {outcome}"
            ),
            metadata={
                "property_name": self.property_name,
                "source": self.source,
                "input_args": self.input_args,
                "expected": self.expected,
                "actual": self.actual,
                "error": self.error,
            },
        )


@dataclass
class PropertyCodeVerificationResult:
    """Result of running prompt-derived property verification."""

    derived_properties: list[DerivedProperty] = field(default_factory=list)
    failures: list[PropertyFailure] = field(default_factory=list)
    wall_clock_seconds: float = 0.0

    @property
    def verified(self) -> bool:
        return len(self.failures) == 0

    def to_constraint_results(self) -> list[ConstraintResult]:
        """Convert failures into VerifyRepairPipeline-compatible constraints."""
        return [failure.to_constraint_result() for failure in self.failures]

    def repair_feedback(self) -> str:
        """Render failures with the existing pipeline formatter."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        return VerifyRepairPipeline._format_violations(self.to_constraint_results())


@dataclass(frozen=True)
class _FunctionSignature:
    param_kinds: list[str]
    return_kind: str | None


def extract_prompt_examples(prompt: str, entry_point: str) -> list[ExampleCase]:
    """Extract doctest-style examples from the prompt docstring."""
    function_node = _find_function_node(prompt, entry_point)
    if function_node is None:
        return []

    docstring = ast.get_docstring(function_node) or ""
    lines = docstring.splitlines()
    examples: list[ExampleCase] = []

    for index, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line.startswith(">>> "):
            continue

        expected = _next_nonempty_line(lines, index + 1)
        if expected is None:
            continue

        call_source = line[4:].strip()
        try:
            parsed = ast.parse(call_source, mode="eval")
        except SyntaxError:
            continue

        call = parsed.body
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == entry_point
        ):
            continue

        try:
            args = tuple(ast.literal_eval(arg) for arg in call.args)
            expected_value = ast.literal_eval(expected)
        except (SyntaxError, ValueError):
            continue

        examples.append((args, expected_value))

    return examples


def extract_official_test_examples(official_tests: str) -> list[ExampleCase]:
    """Extract literal examples from a HumanEval-style `check(candidate)` harness."""
    try:
        tree = ast.parse(official_tests)
    except SyntaxError:
        return []

    examples: list[ExampleCase] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            for statement in node.body:
                if not isinstance(statement, ast.Assert):
                    continue
                literal = _literal_example_from_assert(statement.test)
                if literal is not None:
                    examples.append(literal)

    return examples


class PropertyCodeVerifier:
    """Derive lightweight code properties from prompt, signature, and tests."""

    def __init__(self, max_failures: int = 20) -> None:
        self._max_failures = max_failures

    def verify(
        self,
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str,
    ) -> PropertyCodeVerificationResult:
        """Verify code with prompt-derived properties."""
        start = time.perf_counter()

        prompt_examples = extract_prompt_examples(prompt, entry_point)
        official_examples = extract_official_test_examples(official_tests)
        signature = _extract_signature(prompt, entry_point) or _extract_signature(code, entry_point)

        result = PropertyCodeVerificationResult()
        sample_inputs = _build_sample_inputs(signature, prompt_examples, official_examples, prompt)

        if prompt_examples:
            result.derived_properties.append(
                DerivedProperty(
                    name="example_regression",
                    source="docstring_example",
                    description="Prompt doctest examples remain correct.",
                )
            )
            for args, expected in prompt_examples:
                self._check_expected_output(
                    result,
                    code,
                    entry_point,
                    args,
                    expected,
                    property_name="example_regression",
                    source="docstring_example",
                    description="Prompt example should remain correct.",
                )

        if signature is not None and signature.return_kind is not None and sample_inputs:
            result.derived_properties.append(
                DerivedProperty(
                    name="annotated_return_type",
                    source="signature",
                    description="The return value matches the prompt annotation.",
                )
            )
            for args in sample_inputs:
                if len(result.failures) >= self._max_failures:
                    break
                actual, error = safe_exec_function(code, entry_point, args)
                if error is not None:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="annotated_return_type",
                            source="signature",
                            input_args=args,
                            description="The return value matches the prompt annotation.",
                            expected=signature.return_kind,
                            error=str(error),
                        ),
                    )
                    continue
                if not _matches_kind(actual, signature.return_kind):
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="annotated_return_type",
                            source="signature",
                            input_args=args,
                            description="The return value matches the prompt annotation.",
                            expected=signature.return_kind,
                            actual=repr(actual),
                        ),
                    )

        if sample_inputs:
            source = "official_tests" if official_examples else "signature"
            result.derived_properties.append(
                DerivedProperty(
                    name="deterministic",
                    source=source,
                    description="Repeated calls on the same input are stable.",
                )
            )
            for args in sample_inputs:
                if len(result.failures) >= self._max_failures:
                    break
                first, first_error = safe_exec_function(code, entry_point, args)
                second, second_error = safe_exec_function(code, entry_point, args)
                if first_error is not None or second_error is not None:
                    error_text = str(first_error or second_error)
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="deterministic",
                            source=source,
                            input_args=args,
                            description="Repeated calls on the same input are stable.",
                            error=error_text,
                        ),
                    )
                    continue
                if first != second:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="deterministic",
                            source=source,
                            input_args=args,
                            description="Repeated calls on the same input are stable.",
                            expected=repr(first),
                            actual=repr(second),
                        ),
                    )

        if (
            signature is not None
            and sample_inputs
            and _has_mutable_parameter(signature)
            and not _allows_input_mutation(prompt)
        ):
            result.derived_properties.append(
                DerivedProperty(
                    name="input_immutability",
                    source="signature",
                    description="The function does not mutate caller-owned inputs.",
                )
            )
            for args in sample_inputs:
                if len(result.failures) >= self._max_failures:
                    break
                original_args = copy.deepcopy(args)
                _, error = safe_exec_function(code, entry_point, args)
                if error is not None:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="input_immutability",
                            source="signature",
                            input_args=original_args,
                            description="The function does not mutate caller-owned inputs.",
                            error=str(error),
                        ),
                    )
                    continue
                if args != original_args:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="input_immutability",
                            source="signature",
                            input_args=original_args,
                            description="The function does not mutate caller-owned inputs.",
                            expected=repr(original_args),
                            actual=repr(args),
                        ),
                    )

        if sample_inputs and _SORT_ASCENDING_PATTERN.search(prompt):
            reverse = bool(_SORT_DESCENDING_PATTERN.search(prompt))
            result.derived_properties.append(
                DerivedProperty(
                    name="sorted_output",
                    source="prompt_intent",
                    description="Sorted tasks return an ordered permutation of the input.",
                )
            )
            for args in _sequence_inputs(sample_inputs):
                if len(result.failures) >= self._max_failures:
                    break
                actual, error = safe_exec_function(code, entry_point, args)
                input_sequence = args[0]
                if error is not None:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="sorted_output",
                            source="prompt_intent",
                            input_args=args,
                            description="Sorted tasks return an ordered permutation of the input.",
                            error=str(error),
                        ),
                    )
                    continue
                if not _is_ordered_permutation(input_sequence, actual, reverse=reverse):
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="sorted_output",
                            source="prompt_intent",
                            input_args=args,
                            description="Sorted tasks return an ordered permutation of the input.",
                            expected=repr(sorted(input_sequence, reverse=reverse)),
                            actual=repr(actual),
                        ),
                    )

        if sample_inputs and _REVERSE_PATTERN.search(prompt):
            result.derived_properties.append(
                DerivedProperty(
                    name="reverse_output",
                    source="prompt_intent",
                    description="Reverse tasks mirror the input order.",
                )
            )
            for args in _sequence_inputs(sample_inputs):
                if len(result.failures) >= self._max_failures:
                    break
                actual, error = safe_exec_function(code, entry_point, args)
                expected = _reverse_like(args[0])
                if error is not None:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="reverse_output",
                            source="prompt_intent",
                            input_args=args,
                            description="Reverse tasks mirror the input order.",
                            error=str(error),
                        ),
                    )
                    continue
                if actual != expected:
                    self._append_failure(
                        result,
                        PropertyFailure(
                            property_name="reverse_output",
                            source="prompt_intent",
                            input_args=args,
                            description="Reverse tasks mirror the input order.",
                            expected=repr(expected),
                            actual=repr(actual),
                        ),
                    )

        result.wall_clock_seconds = time.perf_counter() - start
        return result

    def _check_expected_output(
        self,
        result: PropertyCodeVerificationResult,
        code: str,
        entry_point: str,
        args: tuple[Any, ...],
        expected: Any,
        *,
        property_name: str,
        source: str,
        description: str,
    ) -> None:
        if len(result.failures) >= self._max_failures:
            return

        actual, error = safe_exec_function(code, entry_point, args)
        if error is not None:
            self._append_failure(
                result,
                PropertyFailure(
                    property_name=property_name,
                    source=source,
                    input_args=args,
                    description=description,
                    expected=repr(expected),
                    error=str(error),
                ),
            )
            return
        if actual != expected:
            self._append_failure(
                result,
                PropertyFailure(
                    property_name=property_name,
                    source=source,
                    input_args=args,
                    description=description,
                    expected=repr(expected),
                    actual=repr(actual),
                ),
            )

    def _append_failure(
        self,
        result: PropertyCodeVerificationResult,
        failure: PropertyFailure,
    ) -> None:
        if len(result.failures) < self._max_failures:
            result.failures.append(failure)


def _find_function_node(source: str, entry_point: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    return None


def _extract_signature(source: str, entry_point: str) -> _FunctionSignature | None:
    function_node = _find_function_node(source, entry_point)
    if function_node is None:
        return None

    param_kinds: list[str] = []
    for arg in function_node.args.args:
        if arg.arg == "self":
            continue
        param_kinds.append(_annotation_kind(arg.annotation) or "unknown")

    return _FunctionSignature(
        param_kinds=param_kinds,
        return_kind=_annotation_kind(function_node.returns),
    )


def _annotation_kind(annotation: ast.expr | None) -> str | None:
    if annotation is None:
        return None

    if isinstance(annotation, ast.Name):
        return annotation.id.lower()
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        return annotation.value.lower()
    if isinstance(annotation, ast.Subscript):
        text = ast.unparse(annotation).lower()
        if text.startswith("list"):
            return "list"
        if text.startswith("tuple"):
            return "tuple"
        if text.startswith("dict"):
            return "dict"
        if text.startswith("set"):
            return "set"
    return ast.unparse(annotation).lower()


def _next_nonempty_line(lines: list[str], start: int) -> str | None:
    for line in lines[start:]:
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _literal_example_from_assert(node: ast.expr) -> ExampleCase | None:
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
        if not isinstance(node.left, ast.Call):
            return None
        call = node.left
        if not (isinstance(call.func, ast.Name) and call.func.id == "candidate"):
            return None
        try:
            args = tuple(ast.literal_eval(arg) for arg in call.args)
            expected = ast.literal_eval(node.comparators[0])
        except (SyntaxError, ValueError):
            return None

        if isinstance(node.ops[0], ast.Eq):
            return (args, expected)
        return None

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "candidate"
    ):
        try:
            args = tuple(ast.literal_eval(arg) for arg in node.args)
        except (SyntaxError, ValueError):
            return None
        return (args, True)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        inner = node.operand
        if (
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "candidate"
        ):
            try:
                args = tuple(ast.literal_eval(arg) for arg in inner.args)
            except (SyntaxError, ValueError):
                return None
            return (args, False)

    return None


def _build_sample_inputs(
    signature: _FunctionSignature | None,
    prompt_examples: list[ExampleCase],
    official_examples: list[ExampleCase],
    prompt: str,
) -> list[tuple[Any, ...]]:
    samples = [args for args, _expected in prompt_examples]
    samples.extend(args for args, _expected in official_examples)

    if signature is None:
        return _unique_args(samples)

    if not samples:
        generated = _generate_signature_samples(signature)
        if generated:
            samples.extend(generated)

    if _SORT_ASCENDING_PATTERN.search(prompt):
        generated_sorted = _generate_intent_samples(signature, intent="sorted")
        samples.extend(generated_sorted)

    if _REVERSE_PATTERN.search(prompt):
        generated_reverse = _generate_intent_samples(signature, intent="reverse")
        samples.extend(generated_reverse)

    return _unique_args(samples)


def _generate_signature_samples(signature: _FunctionSignature) -> list[tuple[Any, ...]]:
    if not signature.param_kinds:
        return [tuple()]

    samples_by_kind = [_sample_values_for_kind(kind) for kind in signature.param_kinds]
    if any(not values for values in samples_by_kind):
        return []
    n_samples = min(len(values) for values in samples_by_kind if values)

    return [tuple(values[index] for values in samples_by_kind) for index in range(n_samples)]


def _generate_intent_samples(
    signature: _FunctionSignature,
    *,
    intent: str,
) -> list[tuple[Any, ...]]:
    if not signature.param_kinds:
        return []

    values_by_kind = [_sample_values_for_kind(kind) for kind in signature.param_kinds]
    if any(not values for values in values_by_kind):
        return []
    n_samples = min(len(values) for values in values_by_kind if values)

    if intent == "sorted":
        for index, kind in enumerate(signature.param_kinds):
            if kind in {"list", "tuple"}:
                sorted_cases = []
                for values_index in range(n_samples):
                    args = [values[values_index] for values in values_by_kind]
                    if kind == "list":
                        args[index] = [3, 1, 2] if values_index == 0 else [2, 2, 1]
                    else:
                        args[index] = (3, 1, 2) if values_index == 0 else (2, 2, 1)
                    sorted_cases.append(tuple(args))
                return sorted_cases

    if intent == "reverse":
        for index, kind in enumerate(signature.param_kinds):
            if kind in {"str", "list", "tuple"}:
                reverse_cases = []
                for values_index in range(n_samples):
                    args = [values[values_index] for values in values_by_kind]
                    if kind == "str":
                        args[index] = "stressed" if values_index == 0 else "drawer"
                    elif kind == "list":
                        args[index] = [1, 2, 3] if values_index == 0 else [4, 5]
                    else:
                        args[index] = (1, 2, 3) if values_index == 0 else (4, 5)
                    reverse_cases.append(tuple(args))
                return reverse_cases

    return []


def _sample_values_for_kind(kind: str | None) -> list[Any]:
    if kind is None:
        return []
    if kind == "int":
        return [0, 1]
    if kind == "float":
        return [0.0, 1.5]
    if kind == "bool":
        return [True, False]
    if kind == "str":
        return ["abc", ""]
    if kind == "list":
        return [[1, 2, 3], []]
    if kind == "tuple":
        return [(1, 2, 3), tuple()]
    if kind == "dict":
        return [{"a": 1}, {}]
    if kind == "set":
        return [{1, 2}, set()]
    return []


def _unique_args(samples: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    seen: set[str] = set()
    unique: list[tuple[Any, ...]] = []
    for args in samples:
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        unique.append(args)
    return unique


def _matches_kind(value: Any, kind: str) -> bool:
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "str":
        return isinstance(value, str)
    if kind == "list":
        return isinstance(value, list)
    if kind == "tuple":
        return isinstance(value, tuple)
    if kind == "dict":
        return isinstance(value, dict)
    if kind == "set":
        return isinstance(value, set)
    return True


def _has_mutable_parameter(signature: _FunctionSignature) -> bool:
    return any(kind in {"list", "dict", "set"} for kind in signature.param_kinds)


def _allows_input_mutation(prompt: str) -> bool:
    return _IN_PLACE_PATTERN.search(prompt) is not None


def _sequence_inputs(samples: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [args for args in samples if args and isinstance(args[0], (list, tuple, str))]


def _is_ordered_permutation(input_sequence: Any, output_sequence: Any, *, reverse: bool) -> bool:
    if not isinstance(input_sequence, (list, tuple)):
        return False
    if not isinstance(output_sequence, (list, tuple)):
        return False

    if Counter(output_sequence) != Counter(input_sequence):
        return False
    return list(output_sequence) == sorted(input_sequence, reverse=reverse)


def _reverse_like(value: Any) -> Any:
    if isinstance(value, str):
        return value[::-1]
    if isinstance(value, list):
        return list(reversed(value))
    if isinstance(value, tuple):
        return tuple(reversed(value))
    return value
