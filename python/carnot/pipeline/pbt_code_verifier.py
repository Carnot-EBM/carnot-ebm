"""Hypothesis-backed property-based verification for generated Python code.

This module derives bounded property checks for HumanEval-style Python
functions and uses Hypothesis to search for concrete counterexamples. It is an
additive verifier: the official tests still matter, but this path can surface
prompt-implied failures that example-based execution misses.

Spec: REQ-CODE-009, REQ-CODE-010, REQ-CODE-011,
      SCENARIO-CODE-008, SCENARIO-CODE-009, SCENARIO-CODE-010
"""

from __future__ import annotations

import copy
import string
import time
from dataclasses import dataclass, field
from typing import Any

from hypothesis import Phase, find, settings
from hypothesis import strategies as st
from hypothesis.errors import NoSuchExample

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.property_code_verifier import (
    _REVERSE_PATTERN,
    _SORT_ASCENDING_PATTERN,
    _SORT_DESCENDING_PATTERN,
    _allows_input_mutation,
    _extract_signature,
    _is_ordered_permutation,
    _matches_kind,
    _reverse_like,
    extract_official_test_examples,
    extract_prompt_examples,
)
from carnot.verify.python_types import safe_exec_function


@dataclass(frozen=True)
class PBTDerivedProperty:
    """One Hypothesis-backed property checked against generated code."""

    name: str
    source: str
    description: str


@dataclass(frozen=True)
class PBTPropertyFailure:
    """One counterexample found by the Hypothesis-backed verifier."""

    property_name: str
    source: str
    input_args: tuple[Any, ...]
    description: str
    actual: str = ""
    expected: str = ""
    error: str | None = None

    def to_constraint_result(self) -> ConstraintResult:
        """Convert a counterexample into pipeline-compatible feedback."""
        outcome = f"raised {self.error}" if self.error is not None else f"returned {self.actual}"
        expected_suffix = f" expected {self.expected};" if self.expected else ""
        return ConstraintResult(
            constraint_type="pbt_code",
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
                "satisfied": False,
            },
        )


@dataclass
class PBTCodeVerificationResult:
    """Result of Hypothesis-backed property-based verification."""

    derived_properties: list[PBTDerivedProperty] = field(default_factory=list)
    failures: list[PBTPropertyFailure] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    max_examples: int = 0

    @property
    def verified(self) -> bool:
        return not self.failures

    def to_constraint_results(self) -> list[ConstraintResult]:
        """Convert failures into VerifyRepairPipeline-compatible constraints."""
        return [failure.to_constraint_result() for failure in self.failures]

    def repair_feedback(self) -> str:
        """Render failures with the existing pipeline violation formatter."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        return VerifyRepairPipeline._format_violations(self.to_constraint_results())


@dataclass(frozen=True)
class _PropertyCheck:
    name: str
    source: str
    description: str
    strategy: st.SearchStrategy[tuple[Any, ...]]
    checker: str
    sequence_index: int | None = None
    return_kind: str | None = None
    reverse: bool = False


class PBTCodeVerifier:
    """Use Hypothesis to search for generated-code counterexamples."""

    def __init__(
        self,
        max_examples: int = 64,
        timeout_per_call: float = 1.0,
        max_failures: int = 10,
    ) -> None:
        self._max_examples = max_examples
        self._timeout_per_call = timeout_per_call
        self._max_failures = max_failures

    def verify(
        self,
        code: str,
        prompt: str,
        entry_point: str,
        official_tests: str,
    ) -> PBTCodeVerificationResult:
        """Verify generated code by searching for property counterexamples."""
        start = time.perf_counter()

        prompt_examples = extract_prompt_examples(prompt, entry_point)
        official_examples = extract_official_test_examples(official_tests)
        signature = _extract_signature(prompt, entry_point) or _extract_signature(code, entry_point)
        inferred_kinds = _infer_param_kinds(prompt_examples, official_examples)
        if signature is None:
            kinds = inferred_kinds
        else:
            kinds = _merge_param_kinds(signature.param_kinds, inferred_kinds)
        return_kind = signature.return_kind if signature is not None else None
        example_args = [args for args, _expected in prompt_examples]
        example_args.extend(args for args, _expected in official_examples)

        checks = self._build_checks(
            prompt=prompt,
            param_kinds=kinds,
            return_kind=return_kind,
            example_args=example_args,
        )
        result = PBTCodeVerificationResult(
            derived_properties=[
                PBTDerivedProperty(
                    name=check.name,
                    source=check.source,
                    description=check.description,
                )
                for check in checks
            ],
            max_examples=self._max_examples,
        )

        for check in checks:
            if len(result.failures) >= self._max_failures:
                break
            failure = self._find_failure(code, entry_point, check)
            if failure is not None:
                result.failures.append(failure)

        result.wall_clock_seconds = time.perf_counter() - start
        return result

    def _build_checks(
        self,
        *,
        prompt: str,
        param_kinds: list[str],
        return_kind: str | None,
        example_args: list[tuple[Any, ...]],
    ) -> list[_PropertyCheck]:
        base_strategy = _build_tuple_strategy(param_kinds, example_args)
        checks: list[_PropertyCheck] = []

        if base_strategy is not None:
            checks.append(
                _PropertyCheck(
                    name="no_exception",
                    source="signature" if param_kinds else "examples",
                    description="Generated edge-case inputs should not raise exceptions.",
                    strategy=base_strategy,
                    checker="no_exception",
                )
            )
            checks.append(
                _PropertyCheck(
                    name="deterministic",
                    source="signature" if param_kinds else "examples",
                    description="Repeated calls on the same generated input stay stable.",
                    strategy=base_strategy,
                    checker="deterministic",
                )
            )

        if base_strategy is not None and return_kind is not None:
            checks.append(
                _PropertyCheck(
                    name="annotated_return_type",
                    source="signature",
                    description="Generated outputs match the annotated return type.",
                    strategy=base_strategy,
                    checker="annotated_return_type",
                    return_kind=return_kind,
                )
            )

        if (
            base_strategy is not None
            and param_kinds
            and _has_mutable_kind(param_kinds)
            and not _allows_input_mutation(prompt)
        ):
            immutability_strategy = _build_immutability_strategy(param_kinds, example_args)
            assert immutability_strategy is not None
            checks.append(
                _PropertyCheck(
                    name="input_immutability",
                    source="signature" if return_kind is not None else "examples",
                    description="The function does not mutate caller-owned inputs.",
                    strategy=immutability_strategy,
                    checker="input_immutability",
                )
            )

        if _SORT_ASCENDING_PATTERN.search(prompt):
            sequence_index = _sequence_param_index(param_kinds)
            sort_strategy = _build_sorted_strategy(
                param_kinds,
                example_args,
                reverse=bool(_SORT_DESCENDING_PATTERN.search(prompt)),
            )
            if sort_strategy is not None and sequence_index is not None:
                checks.append(
                    _PropertyCheck(
                        name="sorted_output",
                        source="prompt_intent",
                        description="Sorted tasks return an ordered permutation of the input.",
                        strategy=sort_strategy,
                        checker="sorted_output",
                        sequence_index=sequence_index,
                        reverse=bool(_SORT_DESCENDING_PATTERN.search(prompt)),
                    )
                )

        if _REVERSE_PATTERN.search(prompt):
            sequence_index = _reverse_param_index(param_kinds, example_args)
            reverse_strategy = _build_reverse_strategy(param_kinds, example_args)
            if reverse_strategy is not None and sequence_index is not None:
                checks.append(
                    _PropertyCheck(
                        name="reverse_output",
                        source="prompt_intent",
                        description="Reverse tasks mirror the input order.",
                        strategy=reverse_strategy,
                        checker="reverse_output",
                        sequence_index=sequence_index,
                    )
                )

        return checks

    def _find_failure(
        self,
        code: str,
        entry_point: str,
        check: _PropertyCheck,
    ) -> PBTPropertyFailure | None:
        config = settings(
            database=None,
            deadline=None,
            derandomize=True,
            max_examples=self._max_examples,
            phases=(Phase.generate, Phase.target, Phase.shrink),
        )

        try:
            args = find(
                check.strategy,
                lambda candidate_args: (
                    self._evaluate_counterexample(
                        code,
                        entry_point,
                        check,
                        candidate_args,
                    )
                    is not None
                ),
                settings=config,
            )
        except NoSuchExample:
            return None

        return self._evaluate_counterexample(code, entry_point, check, args)

    def _evaluate_counterexample(
        self,
        code: str,
        entry_point: str,
        check: _PropertyCheck,
        args: tuple[Any, ...],
    ) -> PBTPropertyFailure | None:
        if check.checker == "no_exception":
            _actual, error = safe_exec_function(
                code,
                entry_point,
                args,
                timeout=self._timeout_per_call,
            )
            if error is None:
                return None
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=args,
                description=check.description,
                error=str(error),
            )

        if check.checker == "deterministic":
            first, first_error = safe_exec_function(
                code,
                entry_point,
                copy.deepcopy(args),
                timeout=self._timeout_per_call,
            )
            second, second_error = safe_exec_function(
                code,
                entry_point,
                copy.deepcopy(args),
                timeout=self._timeout_per_call,
            )
            if first_error is not None or second_error is not None:
                return PBTPropertyFailure(
                    property_name=check.name,
                    source=check.source,
                    input_args=args,
                    description=check.description,
                    error=str(first_error or second_error),
                )
            if first == second:
                return None
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=args,
                description=check.description,
                expected=repr(first),
                actual=repr(second),
            )

        if check.checker == "annotated_return_type":
            actual, error = safe_exec_function(
                code,
                entry_point,
                args,
                timeout=self._timeout_per_call,
            )
            if error is not None:
                return PBTPropertyFailure(
                    property_name=check.name,
                    source=check.source,
                    input_args=args,
                    description=check.description,
                    expected=check.return_kind or "",
                    error=str(error),
                )
            assert check.return_kind is not None
            if _matches_kind(actual, check.return_kind):
                return None
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=args,
                description=check.description,
                expected=check.return_kind,
                actual=repr(actual),
            )

        if check.checker == "input_immutability":
            mutable_args = copy.deepcopy(args)
            original_args = copy.deepcopy(args)
            _actual, error = safe_exec_function(
                code,
                entry_point,
                mutable_args,
                timeout=self._timeout_per_call,
            )
            if error is not None:
                return PBTPropertyFailure(
                    property_name=check.name,
                    source=check.source,
                    input_args=original_args,
                    description=check.description,
                    error=str(error),
                )
            if mutable_args == original_args:
                return None
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=original_args,
                description=check.description,
                expected=repr(original_args),
                actual=repr(mutable_args),
            )

        if check.checker == "sorted_output":
            actual, error = safe_exec_function(
                code,
                entry_point,
                args,
                timeout=self._timeout_per_call,
            )
            if error is not None:
                return PBTPropertyFailure(
                    property_name=check.name,
                    source=check.source,
                    input_args=args,
                    description=check.description,
                    error=str(error),
                )
            assert check.sequence_index is not None
            input_sequence = args[check.sequence_index]
            if _is_ordered_permutation(input_sequence, actual, reverse=check.reverse):
                return None
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=args,
                description=check.description,
                expected=repr(sorted(input_sequence, reverse=check.reverse)),
                actual=repr(actual),
            )

        actual, error = safe_exec_function(
            code,
            entry_point,
            args,
            timeout=self._timeout_per_call,
        )
        if error is not None:
            return PBTPropertyFailure(
                property_name=check.name,
                source=check.source,
                input_args=args,
                description=check.description,
                error=str(error),
            )
        assert check.sequence_index is not None
        expected = _reverse_like(args[check.sequence_index])
        if actual == expected:
            return None
        return PBTPropertyFailure(
            property_name=check.name,
            source=check.source,
            input_args=args,
            description=check.description,
            expected=repr(expected),
            actual=repr(actual),
        )


def _infer_param_kinds(
    prompt_examples: list[tuple[tuple[Any, ...], Any]],
    official_examples: list[tuple[tuple[Any, ...], Any]],
) -> list[str]:
    all_examples = [args for args, _expected in prompt_examples]
    all_examples.extend(args for args, _expected in official_examples)
    if not all_examples:
        return []

    arity = len(all_examples[0])
    if any(len(args) != arity for args in all_examples):
        return []

    kinds: list[str] = []
    for index in range(arity):
        values = [args[index] for args in all_examples]
        kinds.append(_kind_from_examples(values))
    return kinds


def _merge_param_kinds(signature_kinds: list[str], inferred_kinds: list[str]) -> list[str]:
    if not signature_kinds:
        return inferred_kinds

    merged: list[str] = []
    for index, kind in enumerate(signature_kinds):
        if kind != "unknown":
            merged.append(kind)
            continue
        if index < len(inferred_kinds) and inferred_kinds[index] != "unknown":
            merged.append(inferred_kinds[index])
        else:
            merged.append(kind)
    return merged


def _kind_from_examples(values: list[Any]) -> str:
    if not values:
        return "unknown"

    first_kind = _kind_from_value(values[0])
    if all(_kind_from_value(value) == first_kind for value in values):
        return first_kind
    return "unknown"


def _kind_from_value(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, set):
        return "set"
    return "unknown"


def _build_tuple_strategy(
    param_kinds: list[str],
    example_args: list[tuple[Any, ...]],
) -> st.SearchStrategy[tuple[Any, ...]] | None:
    if not param_kinds:
        if example_args:
            return st.sampled_from(example_args)
        return st.just(tuple())

    field_strategies: list[st.SearchStrategy[Any]] = []
    for index, kind in enumerate(param_kinds):
        strategy = _strategy_for_kind(kind)
        if strategy is None:
            values = [args[index] for args in example_args if len(args) > index]
            if not values:
                return None
            strategy = st.sampled_from(values)
        field_strategies.append(strategy)
    return st.tuples(*field_strategies)


def _build_immutability_strategy(
    param_kinds: list[str],
    example_args: list[tuple[Any, ...]],
) -> st.SearchStrategy[tuple[Any, ...]] | None:
    base = _build_tuple_strategy(param_kinds, example_args)
    if base is None:
        return None

    mutable_index = next(
        (index for index, kind in enumerate(param_kinds) if kind in {"list", "dict", "set"}),
        None,
    )
    if mutable_index is None:
        return base

    return base.filter(lambda args: _is_nonempty_mutable(args[mutable_index]))


def _build_sorted_strategy(
    param_kinds: list[str],
    example_args: list[tuple[Any, ...]],
    *,
    reverse: bool,
) -> st.SearchStrategy[tuple[Any, ...]] | None:
    sequence_index = _sequence_param_index(param_kinds)
    if sequence_index is None:
        return None
    base = _build_tuple_strategy(param_kinds, example_args)
    if base is None:
        return None

    return base.filter(
        lambda args: list(args[sequence_index]) != sorted(args[sequence_index], reverse=reverse)
    )


def _build_reverse_strategy(
    param_kinds: list[str],
    example_args: list[tuple[Any, ...]],
) -> st.SearchStrategy[tuple[Any, ...]] | None:
    index = _reverse_param_index(param_kinds, example_args)
    if index is None:
        return None
    base = _build_tuple_strategy(param_kinds, example_args)
    if base is None:
        return None

    return base.filter(lambda args: args[index] != _reverse_like(args[index]))


def _sequence_param_index(param_kinds: list[str]) -> int | None:
    for index, kind in enumerate(param_kinds):
        if kind in {"list", "tuple"}:
            return index
    return None


def _reverse_param_index(
    param_kinds: list[str],
    example_args: list[tuple[Any, ...]],
) -> int | None:
    for index, kind in enumerate(param_kinds):
        if kind in {"str", "list", "tuple"}:
            return index
    if example_args:
        for index, value in enumerate(example_args[0]):
            if isinstance(value, (str, list, tuple)):
                return index
    return None


def _is_nonempty_mutable(value: Any) -> bool:
    if isinstance(value, (list, dict, set)):
        return bool(value)
    return False


def _has_mutable_kind(param_kinds: list[str]) -> bool:
    return any(kind in {"list", "dict", "set"} for kind in param_kinds)


def _strategy_for_kind(kind: str) -> st.SearchStrategy[Any] | None:
    if kind == "int":
        return st.integers(min_value=-20, max_value=20)
    if kind == "float":
        return st.floats(
            min_value=-20.0,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
        )
    if kind == "bool":
        return st.booleans()
    if kind == "str":
        return st.text(alphabet=string.ascii_letters, min_size=0, max_size=8)
    if kind == "list":
        return st.lists(st.integers(min_value=-9, max_value=9), min_size=0, max_size=6)
    if kind == "tuple":
        return st.builds(
            tuple,
            st.lists(st.integers(min_value=-9, max_value=9), min_size=0, max_size=6),
        )
    if kind == "dict":
        return st.dictionaries(
            st.text(alphabet="abc", min_size=1, max_size=3),
            st.integers(min_value=-9, max_value=9),
            min_size=0,
            max_size=4,
        )
    if kind == "set":
        return st.sets(st.integers(min_value=-9, max_value=9), min_size=0, max_size=6)
    return None
