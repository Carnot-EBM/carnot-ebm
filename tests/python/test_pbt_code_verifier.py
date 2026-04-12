"""Tests for the Hypothesis-backed PBT code verifier.

Spec coverage: REQ-CODE-009, REQ-CODE-010, REQ-CODE-011,
SCENARIO-CODE-008, SCENARIO-CODE-009, SCENARIO-CODE-010
"""

from __future__ import annotations

from dataclasses import dataclass

from carnot.pipeline.humaneval_live_benchmark import execute_humaneval
from carnot.pipeline.pbt_code_verifier import (
    PBTCodeVerificationResult,
    PBTCodeVerifier,
    _build_immutability_strategy,
    _build_reverse_strategy,
    _build_sorted_strategy,
    _build_tuple_strategy,
    _has_mutable_kind,
    _infer_param_kinds,
    _is_nonempty_mutable,
    _kind_from_examples,
    _kind_from_value,
    _merge_param_kinds,
    _PropertyCheck,
    _reverse_param_index,
    _sequence_param_index,
    _strategy_for_kind,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline
from hypothesis import strategies as st


def _problem(*, prompt: str, test: str, entry_point: str) -> dict[str, str]:
    return {
        "task_id": "HumanEval/0",
        "prompt": prompt,
        "test": test,
        "entry_point": entry_point,
    }


@dataclass(frozen=True)
class _BenchmarkCase:
    prompt: str
    official_tests: str
    entry_point: str
    buggy_code: str
    correct_code: str


def test_sorted_prompt_pbt_finds_counterexample_missed_by_execution() -> None:
    """SCENARIO-CODE-008: Hypothesis PBT finds a sorting bug the official tests miss."""
    prompt = (
        "def sort_numbers(nums: list[int]) -> list[int]:\n"
        '    """Return numbers sorted in ascending order."""\n'
    )
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate([]) == []\n"
        "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
    )
    buggy_code = "def sort_numbers(nums: list[int]) -> list[int]:\n    return nums\n"

    harness = execute_humaneval(
        buggy_code,
        _problem(prompt=prompt, test=official_tests, entry_point="sort_numbers"),
        timeout=1.0,
    )
    verification = PBTCodeVerifier(max_examples=64).verify(
        buggy_code,
        prompt,
        "sort_numbers",
        official_tests,
    )

    assert harness.passed is True
    assert verification.verified is False
    assert any(prop.name == "sorted_output" for prop in verification.derived_properties)
    assert any(failure.property_name == "sorted_output" for failure in verification.failures)
    assert any(failure.source == "prompt_intent" for failure in verification.failures)


def test_reverse_prompt_type_failures_become_constraint_results_and_feedback() -> None:
    """REQ-CODE-009: PBT failures preserve type, property, and counterexample details."""
    prompt = 'def reverse_string(text: str) -> str:\n    """Return the reversed string."""\n'
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate('') == ''\n"
        "    assert candidate('aa') == 'aa'\n"
    )
    buggy_code = (
        "def reverse_string(text: str) -> str:\n"
        "    if text == text[::-1]:\n"
        "        return text\n"
        "    return list(reversed(text))\n"
    )

    verification = PBTCodeVerifier(max_examples=64).verify(
        buggy_code,
        prompt,
        "reverse_string",
        official_tests,
    )
    constraints = verification.to_constraint_results()
    feedback = verification.repair_feedback()

    assert verification.verified is False
    assert any(f.property_name == "annotated_return_type" for f in verification.failures)
    assert any(f.property_name == "reverse_output" for f in verification.failures)
    assert constraints
    assert constraints[0].constraint_type == "pbt_code"
    assert "input=" in constraints[0].description
    assert "prompt_intent" in feedback or "signature" in feedback


def test_example_types_drive_strategy_when_signature_is_missing() -> None:
    """REQ-CODE-009: example inputs infer a strategy when annotations are absent."""
    prompt = 'def reverse_text(text):\n    """Return the reversed string."""\n'
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate('') == ''\n"
        "    assert candidate('aa') == 'aa'\n"
    )
    buggy_code = "def reverse_text(text):\n    return text\n"

    verification = PBTCodeVerifier(max_examples=64).verify(
        buggy_code,
        prompt,
        "reverse_text",
        official_tests,
    )

    assert verification.verified is False
    assert any(failure.property_name == "reverse_output" for failure in verification.failures)


def test_zero_arg_functions_report_execution_errors() -> None:
    """REQ-CODE-009: zero-argument functions still get bounded PBT feedback."""
    prompt = 'def broken() -> int:\n    """Return the constant seven."""\n'
    official_tests = "def check(candidate):\n    assert candidate() == 7\n"
    buggy_code = "def broken() -> int:\n    raise RuntimeError('boom')\n"

    verification = PBTCodeVerifier(max_examples=32).verify(
        buggy_code,
        prompt,
        "broken",
        official_tests,
    )

    assert verification.verified is False
    assert any(failure.error == "boom" for failure in verification.failures)


def test_helper_branches_cover_inference_and_strategy_fallbacks() -> None:
    """REQ-CODE-009: helper branches stay bounded for missing or mixed type evidence."""
    assert _infer_param_kinds([], []) == []
    assert _infer_param_kinds([((1,), 1)], [((1, 2), 3)]) == []
    assert _merge_param_kinds(["unknown", "int"], ["str", "bool"]) == ["str", "int"]
    assert _merge_param_kinds(["unknown"], []) == ["unknown"]
    assert _kind_from_examples([]) == "unknown"
    assert _kind_from_examples([1, 2.0]) == "unknown"
    assert _kind_from_value(True) == "bool"
    assert _kind_from_value(3) == "int"
    assert _kind_from_value(1.5) == "float"
    assert _kind_from_value("x") == "str"
    assert _kind_from_value([1]) == "list"
    assert _kind_from_value((1,)) == "tuple"
    assert _kind_from_value({"a": 1}) == "dict"
    assert _kind_from_value({1}) == "set"
    assert _kind_from_value(object()) == "unknown"
    assert _build_tuple_strategy(["unknown"], []) is None
    assert _build_tuple_strategy(["unknown"], [("x",)]) is not None
    assert _build_tuple_strategy([], []) is not None
    assert _build_immutability_strategy(["int"], [tuple()]) is not None
    assert _build_immutability_strategy(["unknown"], []) is None
    assert _build_sorted_strategy(["int"], [tuple()], reverse=False) is None
    assert _build_sorted_strategy(["list", "unknown"], [], reverse=False) is None
    assert _build_reverse_strategy([], []) is None
    assert _build_reverse_strategy(["str", "unknown"], []) is None
    assert _sequence_param_index(["int"]) is None
    assert _reverse_param_index([], [("abc",)]) == 0
    assert _reverse_param_index([], []) is None
    assert _is_nonempty_mutable([]) is False
    assert _is_nonempty_mutable([1]) is True
    assert _is_nonempty_mutable(1) is False
    assert _has_mutable_kind(["int", "list"]) is True
    assert _has_mutable_kind(["int", "str"]) is False
    assert _strategy_for_kind("float") is not None
    assert _strategy_for_kind("bool") is not None
    assert _strategy_for_kind("dict") is not None
    assert _strategy_for_kind("set") is not None
    assert _strategy_for_kind("tuple") is not None
    assert _strategy_for_kind("mystery") is None


def test_internal_counterexample_evaluation_covers_error_and_nondeterministic_paths() -> None:
    """REQ-CODE-009: internal property evaluators preserve structured branch coverage."""
    verifier = PBTCodeVerifier(max_examples=16)

    deterministic_check = _PropertyCheck(
        name="deterministic",
        source="signature",
        description="Repeated calls stay stable.",
        strategy=st.just((1,)),
        checker="deterministic",
    )
    deterministic_failure = verifier._evaluate_counterexample(
        "import time\ndef unstable(x):\n    return time.time_ns()\n",
        "unstable",
        deterministic_check,
        (1,),
    )
    assert deterministic_failure is not None
    assert deterministic_failure.property_name == "deterministic"

    type_error_check = _PropertyCheck(
        name="annotated_return_type",
        source="signature",
        description="Generated outputs match the annotation.",
        strategy=st.just((1,)),
        checker="annotated_return_type",
        return_kind="int",
    )
    type_error_failure = verifier._evaluate_counterexample(
        "def explode(x: int) -> int:\n    raise ValueError('boom')\n",
        "explode",
        type_error_check,
        (1,),
    )
    assert type_error_failure is not None
    assert type_error_failure.error == "boom"

    immutability_check = _PropertyCheck(
        name="input_immutability",
        source="signature",
        description="The function does not mutate inputs.",
        strategy=st.just(([1],)),
        checker="input_immutability",
    )
    immutability_failure = verifier._evaluate_counterexample(
        "def mutate(nums):\n    raise RuntimeError('mutate')\n",
        "mutate",
        immutability_check,
        ([1],),
    )
    assert immutability_failure is not None
    assert immutability_failure.error == "mutate"

    sorted_check = _PropertyCheck(
        name="sorted_output",
        source="prompt_intent",
        description="Sorted tasks return ordered permutations.",
        strategy=st.just(([2, 1],)),
        checker="sorted_output",
        sequence_index=0,
    )
    sorted_failure = verifier._evaluate_counterexample(
        "def sort_numbers(nums):\n    raise RuntimeError('sort')\n",
        "sort_numbers",
        sorted_check,
        ([2, 1],),
    )
    assert sorted_failure is not None
    assert sorted_failure.error == "sort"

    reverse_check = _PropertyCheck(
        name="reverse_output",
        source="prompt_intent",
        description="Reverse tasks mirror the input order.",
        strategy=st.just(("ab",)),
        checker="reverse_output",
        sequence_index=0,
    )
    reverse_failure = verifier._evaluate_counterexample(
        "def reverse_text(text):\n    raise RuntimeError('reverse')\n",
        "reverse_text",
        reverse_check,
        ("ab",),
    )
    assert reverse_failure is not None
    assert reverse_failure.error == "reverse"


def test_verifier_respects_max_failures_and_signatureless_invalid_code() -> None:
    """REQ-CODE-009: the verifier caps failures and handles signatureless invalid code."""
    capped = PBTCodeVerifier(max_examples=32, max_failures=1).verify(
        "def reverse_text(text: str) -> str:\n    return list(reversed(text))\n",
        'def reverse_text(text: str) -> str:\n    """Return the reversed string."""\n',
        "reverse_text",
        (
            "def check(candidate):\n"
            "    assert candidate('') == ''\n"
            "    assert candidate('aa') == 'aa'\n"
        ),
    )
    invalid = PBTCodeVerifier(max_examples=8).verify(
        "not python",
        "also not python",
        "broken",
        "def check(candidate):\n    assert True\n",
    )

    assert len(capped.failures) == 1
    assert invalid.verified is False
    assert any(failure.error is not None for failure in invalid.failures)


def test_result_helpers_cover_verified_feedback_paths() -> None:
    """REQ-CODE-009: result helpers stay lightweight when there are no failures."""
    result = PBTCodeVerificationResult()

    assert result.verified is True
    assert result.to_constraint_results() == []
    assert result.repair_feedback() == "No violations found."


def test_verify_generated_code_merges_static_and_pbt_violations() -> None:
    """SCENARIO-CODE-009: VerifyRepairPipeline returns structured PBT violations for code."""
    prompt = (
        "def increment_all(nums: list[int]) -> list[int]:\n"
        '    """Return a new list with every integer incremented by one."""\n'
    )
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate([]) == []\n"
        "    assert candidate([1, 2]) == [2, 3]\n"
    )
    mutating_code = (
        "def increment_all(nums: list[int]) -> list[int]:\n"
        "    for index, value in enumerate(nums):\n"
        "        nums[index] = value + 1\n"
        "    return nums\n"
    )

    pipeline = VerifyRepairPipeline()
    result = pipeline.verify_generated_code(
        mutating_code,
        prompt,
        "increment_all",
        official_tests,
    )

    assert result.verified is False
    assert result.certificate["n_constraints"] >= len(result.violations)
    assert "pbt_summary" in result.certificate
    assert any(v.constraint_type == "pbt_code" for v in result.violations)
    assert any(v.metadata.get("property_name") == "input_immutability" for v in result.violations)


def test_verify_generated_code_allows_static_only_mode() -> None:
    """REQ-CODE-010: callers can disable the PBT path without breaking verification."""
    prompt = 'def add(a: int, b: int) -> int:\n    """Return a + b."""\n'
    official_tests = "def check(candidate):\n    assert candidate(1, 2) == 3\n"
    code = "def add(a: int, b: int) -> int:\n    return a + b\n"

    result = VerifyRepairPipeline().verify_generated_code(
        code,
        prompt,
        "add",
        official_tests,
        include_pbt=False,
    )

    assert result.verified is True
    assert result.certificate["pbt_summary"]["enabled"] is False


def test_five_problem_humaneval_slice_shows_extra_pbt_detection_without_false_positives() -> None:
    """SCENARIO-CODE-010: a deterministic five-problem slice shows extra PBT detection."""
    cases = [
        _BenchmarkCase(
            prompt=(
                "def sort_numbers(nums: list[int]) -> list[int]:\n"
                '    """Return numbers sorted in ascending order."""\n'
            ),
            official_tests=(
                "def check(candidate):\n"
                "    assert candidate([]) == []\n"
                "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
            ),
            entry_point="sort_numbers",
            buggy_code="def sort_numbers(nums: list[int]) -> list[int]:\n    return nums\n",
            correct_code=(
                "def sort_numbers(nums: list[int]) -> list[int]:\n    return sorted(nums)\n"
            ),
        ),
        _BenchmarkCase(
            prompt=(
                "def sort_desc(nums: list[int]) -> list[int]:\n"
                '    """Return numbers sorted in descending order."""\n'
            ),
            official_tests=(
                "def check(candidate):\n"
                "    assert candidate([]) == []\n"
                "    assert candidate([3, 2, 1]) == [3, 2, 1]\n"
            ),
            entry_point="sort_desc",
            buggy_code="def sort_desc(nums: list[int]) -> list[int]:\n    return nums\n",
            correct_code=(
                "def sort_desc(nums: list[int]) -> list[int]:\n"
                "    return sorted(nums, reverse=True)\n"
            ),
        ),
        _BenchmarkCase(
            prompt=(
                'def reverse_string(text: str) -> str:\n    """Return the reversed string."""\n'
            ),
            official_tests=(
                "def check(candidate):\n"
                "    assert candidate('') == ''\n"
                "    assert candidate('aa') == 'aa'\n"
            ),
            entry_point="reverse_string",
            buggy_code="def reverse_string(text: str) -> str:\n    return text\n",
            correct_code=("def reverse_string(text: str) -> str:\n    return text[::-1]\n"),
        ),
        _BenchmarkCase(
            prompt=(
                "def reverse_items(items: list[int]) -> list[int]:\n"
                '    """Return the list with its items reversed."""\n'
            ),
            official_tests=(
                "def check(candidate):\n"
                "    assert candidate([]) == []\n"
                "    assert candidate([1]) == [1]\n"
            ),
            entry_point="reverse_items",
            buggy_code="def reverse_items(items: list[int]) -> list[int]:\n    return items\n",
            correct_code=(
                "def reverse_items(items: list[int]) -> list[int]:\n"
                "    return list(reversed(items))\n"
            ),
        ),
        _BenchmarkCase(
            prompt=(
                "def increment_all(nums: list[int]) -> list[int]:\n"
                '    """Return a new list with every integer incremented by one."""\n'
            ),
            official_tests=(
                "def check(candidate):\n"
                "    assert candidate([]) == []\n"
                "    assert candidate([1, 2]) == [2, 3]\n"
            ),
            entry_point="increment_all",
            buggy_code=(
                "def increment_all(nums: list[int]) -> list[int]:\n"
                "    for index, value in enumerate(nums):\n"
                "        nums[index] = value + 1\n"
                "    return nums\n"
            ),
            correct_code=(
                "def increment_all(nums: list[int]) -> list[int]:\n"
                "    return [value + 1 for value in nums]\n"
            ),
        ),
    ]

    pipeline = VerifyRepairPipeline()
    execution_detected = 0
    pbt_detected = 0

    for case in cases:
        buggy_harness = execute_humaneval(
            case.buggy_code,
            _problem(
                prompt=case.prompt,
                test=case.official_tests,
                entry_point=case.entry_point,
            ),
            timeout=1.0,
        )
        buggy_result = pipeline.verify_generated_code(
            case.buggy_code,
            case.prompt,
            case.entry_point,
            case.official_tests,
            include_static=False,
        )
        correct_result = pipeline.verify_generated_code(
            case.correct_code,
            case.prompt,
            case.entry_point,
            case.official_tests,
            include_static=False,
        )

        execution_detected += int(not buggy_harness.passed)
        pbt_detected += int(not buggy_result.verified)
        assert correct_result.verified is True

    assert execution_detected < pbt_detected
    assert pbt_detected == len(cases)
