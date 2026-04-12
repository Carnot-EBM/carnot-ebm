"""Tests for prompt-derived property verification on Python code.

Spec coverage: REQ-CODE-007, REQ-CODE-008,
               SCENARIO-CODE-006, SCENARIO-CODE-007
"""

from __future__ import annotations

import ast

from carnot.pipeline.humaneval_live_benchmark import execute_humaneval
from carnot.pipeline.property_code_verifier import (
    DerivedProperty,
    PropertyCodeVerifier,
    _annotation_kind,
    _build_sample_inputs,
    _extract_signature,
    _find_function_node,
    _FunctionSignature,
    _generate_intent_samples,
    _generate_signature_samples,
    _is_ordered_permutation,
    _literal_example_from_assert,
    _matches_kind,
    _reverse_like,
    _sample_values_for_kind,
    _sequence_inputs,
    _unique_args,
    extract_official_test_examples,
    extract_prompt_examples,
)


def _problem(*, prompt: str, test: str, entry_point: str) -> dict[str, str]:
    return {
        "task_id": "HumanEval/0",
        "prompt": prompt,
        "test": test,
        "entry_point": entry_point,
    }


def test_extract_prompt_and_official_test_examples() -> None:
    """REQ-CODE-007: prompt doctests and official asserts become deterministic examples."""
    prompt = (
        "def sort_numbers(nums: list[int]) -> list[int]:\n"
        '    """Return numbers sorted in ascending order.\n'
        "\n"
        "    >>> sort_numbers([3, 1, 2])\n"
        "    [1, 2, 3]\n"
        '    """\n'
    )
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate([]) == []\n"
        "    assert candidate([1, 2, 3]) == [1, 2, 3]\n"
        "    assert candidate([4]) == [4]\n"
    )

    prompt_examples = extract_prompt_examples(prompt, "sort_numbers")
    official_examples = extract_official_test_examples(official_tests)

    assert prompt_examples == [(([3, 1, 2],), [1, 2, 3])]
    assert official_examples == [
        (([],), []),
        (([1, 2, 3],), [1, 2, 3]),
        (([4],), [4]),
    ]


def test_sorted_intent_catches_bug_missed_by_official_tests() -> None:
    """SCENARIO-CODE-006: prompt-derived invariants catch an identity implementation."""
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
    verification = PropertyCodeVerifier().verify(
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


def test_docstring_examples_become_exact_regressions() -> None:
    """REQ-CODE-007: prompt examples add extra regression checks beyond the official harness."""
    prompt = (
        "def reverse_string(text: str) -> str:\n"
        '    """Return the reversed string.\n'
        "\n"
        "    >>> reverse_string('stressed')\n"
        "    'desserts'\n"
        '    """\n'
    )
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate('') == ''\n"
        "    assert candidate('aa') == 'aa'\n"
    )
    buggy_code = "def reverse_string(text: str) -> str:\n    return text\n"

    verification = PropertyCodeVerifier().verify(
        buggy_code,
        prompt,
        "reverse_string",
        official_tests,
    )

    assert any(prop.name == "example_regression" for prop in verification.derived_properties)
    assert any(failure.source == "docstring_example" for failure in verification.failures)


def test_in_place_prompts_skip_non_mutation_invariant() -> None:
    """REQ-CODE-007: prompts that explicitly say 'in place' do not emit immutability checks."""
    prompt = (
        "def sort_in_place(nums: list[int]) -> list[int]:\n"
        '    """Sort nums in place and return the same list."""\n'
    )
    official_tests = (
        "def check(candidate):\n    values = [3, 1, 2]\n    assert candidate(values) == [1, 2, 3]\n"
    )
    mutating_code = (
        "def sort_in_place(nums: list[int]) -> list[int]:\n    nums.sort()\n    return nums\n"
    )

    verification = PropertyCodeVerifier().verify(
        mutating_code,
        prompt,
        "sort_in_place",
        official_tests,
    )

    assert all(prop.name != "input_immutability" for prop in verification.derived_properties)


def test_failures_convert_to_pipeline_compatible_feedback() -> None:
    """SCENARIO-CODE-007: property failures render as pipeline-compatible repair feedback."""
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

    verification = PropertyCodeVerifier().verify(
        buggy_code,
        prompt,
        "sort_numbers",
        official_tests,
    )
    constraints = verification.to_constraint_results()
    feedback = verification.repair_feedback()

    assert constraints
    assert constraints[0].constraint_type == "property_code"
    assert "sorted_output" in constraints[0].description
    assert "prompt_intent" in feedback
    assert "input=" in feedback


def test_result_dataclass_defaults() -> None:
    """REQ-CODE-008: derived-property metadata is lightweight and serializable."""
    prop = DerivedProperty(
        name="deterministic",
        source="signature",
        description="Calling the function twice on the same input returns the same value.",
    )

    assert prop.name == "deterministic"
    assert prop.source == "signature"

    error_constraint = (
        PropertyCodeVerifier(max_failures=1)
        .verify(
            "def boom(x: int) -> int:\n    raise ValueError('boom')\n",
            'def boom(x: int) -> int:\n    """Return an int."""\n',
            "boom",
            "def check(candidate):\n    assert candidate(1) == 1\n",
        )
        .to_constraint_results()[0]
    )
    assert "raised boom" in error_constraint.description


def test_extract_prompt_examples_handles_invalid_variants() -> None:
    """REQ-CODE-007: prompt example extraction skips malformed, missing, and mismatched doctests."""
    assert extract_prompt_examples("not python", "demo") == []

    missing_expected = 'def demo(x: int) -> int:\n    """\n    >>> demo(1)\n    """\n'
    assert extract_prompt_examples(missing_expected, "demo") == []

    invalid_call = 'def demo(x: int) -> int:\n    """\n    >>> demo(\n    1\n    """\n'
    assert extract_prompt_examples(invalid_call, "demo") == []

    wrong_function = 'def demo(x: int) -> int:\n    """\n    >>> other(1)\n    1\n    """\n'
    assert extract_prompt_examples(wrong_function, "demo") == []

    non_literal = 'def demo(x: int) -> int:\n    """\n    >>> demo(name)\n    1\n    """\n'
    assert extract_prompt_examples(non_literal, "demo") == []


def test_extract_official_test_examples_handles_truthy_and_falsey_asserts() -> None:
    """REQ-CODE-007: official test extraction supports equality, truthy, and falsey asserts."""
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate(1) == 2\n"
        "    assert candidate(3)\n"
        "    assert not candidate(4)\n"
    )

    assert extract_official_test_examples(official_tests) == [
        ((1,), 2),
        ((3,), True),
        ((4,), False),
    ]
    assert extract_official_test_examples("def check(") == []


def test_literal_example_from_assert_rejects_unsupported_shapes() -> None:
    """REQ-CODE-007: unsupported assert shapes are ignored rather than raising."""
    compare_non_call = "assert 1 == 1"
    compare_wrong_func = "assert other(1) == 2"
    compare_not_equal = "assert candidate(1) != 2"
    nonliteral_true = "assert candidate(name)"
    nonliteral_false = "assert not candidate(name)"
    plain_name = "assert flag"

    compare_non_call_node = ast.parse(compare_non_call).body[0].test
    compare_wrong_func_node = ast.parse(compare_wrong_func).body[0].test
    compare_not_equal_node = ast.parse(compare_not_equal).body[0].test
    nonliteral_true_node = ast.parse(nonliteral_true).body[0].test
    nonliteral_false_node = ast.parse(nonliteral_false).body[0].test
    plain_name_node = ast.parse(plain_name).body[0].test

    assert _literal_example_from_assert(compare_non_call_node) is None
    assert _literal_example_from_assert(compare_wrong_func_node) is None
    assert _literal_example_from_assert(compare_not_equal_node) is None
    assert _literal_example_from_assert(nonliteral_true_node) is None
    assert _literal_example_from_assert(nonliteral_false_node) is None
    assert _literal_example_from_assert(plain_name_node) is None


def test_find_function_and_signature_helpers_cover_annotation_variants() -> None:
    """REQ-CODE-007: signature parsing handles missing functions and typed collections."""
    source = (
        "def demo(self, values: list[int], pair: tuple[int, int], "
        "meta: dict[str, int], tags: set[str],"
        " raw: UnknownType) -> 'Answer':\n"
        "    return 'ok'\n"
    )

    assert _find_function_node("def demo(", "demo") is None
    assert _find_function_node(source, "other") is None
    assert _extract_signature(source, "other") is None

    signature = _extract_signature(source, "demo")
    assert signature is not None
    assert signature.param_kinds == ["list", "tuple", "dict", "set", "unknowntype"]
    assert signature.return_kind == "answer"

    annotation = ast.parse("x: list[int]", mode="single").body[0].annotation
    assert _annotation_kind(annotation) == "list"
    fallback_annotation = ast.parse("x: int | None", mode="single").body[0].annotation
    assert _annotation_kind(fallback_annotation) == "int | none"
    assert _annotation_kind(None) is None


def test_build_sample_inputs_and_generators_cover_fallback_paths() -> None:
    """REQ-CODE-007: sample generation falls back from examples to signature-derived probes."""
    signature = _FunctionSignature(param_kinds=["int"], return_kind="int")
    assert _build_sample_inputs(signature, [], [], "def demo(x: int) -> int:\n") == [(0,), (1,)]
    assert _build_sample_inputs(None, [((1,), 2)], [], "prompt") == [(1,)]
    generated_inputs = _build_sample_inputs(
        _FunctionSignature(param_kinds=["list"], return_kind="list"),
        [],
        [],
        "Return numbers sorted in ascending order and reversed later.",
    )
    assert generated_inputs == [
        ([1, 2, 3],),
        ([],),
        ([3, 1, 2],),
        ([2, 2, 1],),
        ([4, 5],),
    ]
    assert _generate_signature_samples(_FunctionSignature(param_kinds=[], return_kind=None)) == [
        tuple()
    ]
    assert (
        _generate_signature_samples(_FunctionSignature(param_kinds=["unknown"], return_kind=None))
        == []
    )
    assert (
        _generate_intent_samples(
            _FunctionSignature(param_kinds=[], return_kind=None), intent="sorted"
        )
        == []
    )
    assert (
        _generate_intent_samples(
            _FunctionSignature(param_kinds=["unknown"], return_kind=None),
            intent="sorted",
        )
        == []
    )
    assert (
        _generate_intent_samples(
            _FunctionSignature(param_kinds=["list"], return_kind="list"),
            intent="noop",
        )
        == []
    )


def test_generate_intent_samples_cover_sorted_and_reverse_shapes() -> None:
    """REQ-CODE-007: intent sampling supports tuple sorting and string/list/tuple reversal cases."""
    tuple_sorted = _generate_intent_samples(
        _FunctionSignature(param_kinds=["tuple"], return_kind="tuple"),
        intent="sorted",
    )
    str_reverse = _generate_intent_samples(
        _FunctionSignature(param_kinds=["str"], return_kind="str"),
        intent="reverse",
    )
    list_reverse = _generate_intent_samples(
        _FunctionSignature(param_kinds=["list"], return_kind="list"),
        intent="reverse",
    )
    tuple_reverse = _generate_intent_samples(
        _FunctionSignature(param_kinds=["tuple"], return_kind="tuple"),
        intent="reverse",
    )

    assert tuple_sorted == [((3, 1, 2),), ((2, 2, 1),)]
    assert str_reverse == [("stressed",), ("drawer",)]
    assert list_reverse == [([1, 2, 3],), ([4, 5],)]
    assert tuple_reverse == [((1, 2, 3),), ((4, 5),)]


def test_helper_predicates_cover_value_shapes() -> None:
    """REQ-CODE-007: helper predicates and generators cover typed value branches."""
    assert _sample_values_for_kind("float") == [0.0, 1.5]
    assert _sample_values_for_kind("bool") == [True, False]
    assert _sample_values_for_kind("dict") == [{"a": 1}, {}]
    assert _sample_values_for_kind("set") == [{1, 2}, set()]
    assert _sample_values_for_kind(None) == []

    assert _matches_kind(1, "int") is True
    assert _matches_kind(True, "int") is False
    assert _matches_kind(1.5, "float") is True
    assert _matches_kind(False, "float") is False
    assert _matches_kind(True, "bool") is True
    assert _matches_kind("x", "str") is True
    assert _matches_kind([], "list") is True
    assert _matches_kind((), "tuple") is True
    assert _matches_kind({}, "dict") is True
    assert _matches_kind(set(), "set") is True
    assert _matches_kind(object(), "unknown") is True


def test_collection_helpers_cover_duplicates_and_non_sequence_outputs() -> None:
    """REQ-CODE-007: duplicate inputs are deduped and ordering checks reject invalid shapes."""
    assert _unique_args([(1,), (1,), (2,)]) == [(1,), (2,)]
    assert _sequence_inputs([(1,), ([1, 2],), ("ab",), ((1, 2),)]) == [
        ([1, 2],),
        ("ab",),
        ((1, 2),),
    ]
    assert _is_ordered_permutation("ab", ["a", "b"], reverse=False) is False
    assert _is_ordered_permutation([1, 2], "ab", reverse=False) is False
    assert _is_ordered_permutation([1, 2], [2, 2], reverse=False) is False


def test_reverse_like_covers_all_supported_shapes() -> None:
    """REQ-CODE-007: reverse helper mirrors strings, lists, tuples, and leaves others alone."""
    assert _reverse_like("abc") == "cba"
    assert _reverse_like([1, 2, 3]) == [3, 2, 1]
    assert _reverse_like((1, 2, 3)) == (3, 2, 1)
    marker = object()
    assert _reverse_like(marker) is marker


def test_verify_records_wrong_return_type_and_runtime_error() -> None:
    """REQ-CODE-007: signature-derived checks capture wrong return types and execution errors."""
    verifier = PropertyCodeVerifier()
    prompt = 'def demo(x: int) -> int:\n    """Return an int."""\n'
    official_tests = "def check(candidate):\n    assert candidate(1) == 1\n"

    wrong_type = "def demo(x: int) -> int:\n    return 'oops'\n"
    wrong_result = verifier.verify(wrong_type, prompt, "demo", official_tests)
    assert any(
        failure.property_name == "annotated_return_type" for failure in wrong_result.failures
    )

    raising = "def demo(x: int) -> int:\n    raise ValueError('boom')\n"
    raising_result = verifier.verify(raising, prompt, "demo", official_tests)
    assert any(
        failure.property_name == "annotated_return_type" and failure.error == "boom"
        for failure in raising_result.failures
    )


def test_prompt_example_runtime_error_becomes_structured_failure() -> None:
    """REQ-CODE-008: docstring-example execution errors are preserved as structured failures."""
    prompt = (
        "def crash(nums: list[int]) -> list[int]:\n"
        '    """\n'
        "    >>> crash([1, 2])\n"
        "    [1, 2]\n"
        '    """\n'
    )
    code = "def crash(nums: list[int]) -> list[int]:\n    raise RuntimeError('boom')\n"

    result = PropertyCodeVerifier().verify(
        code,
        prompt,
        "crash",
        "def check(candidate):\n    assert candidate([1]) == [1]\n",
    )

    assert any(
        failure.source == "docstring_example" and failure.error == "boom"
        for failure in result.failures
    )


def test_verify_detects_nondeterminism_and_reverse_intent() -> None:
    """REQ-CODE-007: reverse-intent tasks catch both unstable and non-reversing implementations."""
    prompt = 'def reverse_string(text: str) -> str:\n    """Return the reversed string."""\n'
    official_tests = (
        "def check(candidate):\n"
        "    assert candidate('') == ''\n"
        "    assert candidate('aa') == 'aa'\n"
    )
    nondeterministic = (
        "def reverse_string(text: str) -> str:\n    return __import__('uuid').uuid4().hex\n"
    )

    result = PropertyCodeVerifier().verify(
        nondeterministic,
        prompt,
        "reverse_string",
        official_tests,
    )

    assert any(failure.property_name == "deterministic" for failure in result.failures)
    assert any(failure.property_name == "reverse_output" for failure in result.failures)


def test_verify_detects_input_mutation_when_prompt_is_not_in_place() -> None:
    """REQ-CODE-007: signature-derived immutability checks catch caller-visible mutations."""
    prompt = (
        "def touch(nums: list[int]) -> list[int]:\n"
        '    """Return nums without changing the caller input."""\n'
    )
    official_tests = "def check(candidate):\n    assert candidate([1, 2]) == [1, 2, 0]\n"
    mutating = "def touch(nums: list[int]) -> list[int]:\n    nums.append(0)\n    return nums\n"

    result = PropertyCodeVerifier().verify(mutating, prompt, "touch", official_tests)

    assert any(failure.property_name == "input_immutability" for failure in result.failures)


def test_verify_sorted_error_path_executes() -> None:
    """REQ-CODE-007: sorted-intent checks keep structured failures when execution raises."""
    prompt = 'def weird(nums: list[int]) -> list[int]:\n    """Return the reverse sorted list."""\n'
    official_tests = "def check(candidate):\n    assert candidate([1, 2]) == [2, 1]\n"
    raising = "def weird(nums: list[int]) -> list[int]:\n    raise RuntimeError('boom')\n"

    result = PropertyCodeVerifier().verify(raising, prompt, "weird", official_tests)

    assert any(failure.property_name == "sorted_output" for failure in result.failures)


def test_verify_reverse_error_path_executes() -> None:
    """REQ-CODE-007: reverse-intent checks keep structured failures when execution raises."""
    prompt = (
        'def reverse_list(nums: list[int]) -> list[int]:\n    """Return the reversed list."""\n'
    )
    official_tests = "def check(candidate):\n    assert candidate([1, 2]) == [2, 1]\n"
    raising = "def reverse_list(nums: list[int]) -> list[int]:\n    raise RuntimeError('boom')\n"

    result = PropertyCodeVerifier().verify(
        raising,
        prompt,
        "reverse_list",
        official_tests,
    )

    assert any(failure.property_name == "reverse_output" for failure in result.failures)


def test_max_failures_short_circuits_additional_checks() -> None:
    """REQ-CODE-008: failure caps short-circuit the verifier without crashing later checks."""
    prompt = (
        "def weird(nums: list[int]) -> list[int]:\n"
        '    """Return the reverse sorted list.\n'
        "\n"
        "    >>> weird([3, 1, 2])\n"
        "    [3, 2, 1]\n"
        '    """\n'
    )
    official_tests = "def check(candidate):\n    assert candidate([1, 2]) == [2, 1]\n"
    buggy = "def weird(nums: list[int]) -> list[int]:\n    return nums\n"

    result = PropertyCodeVerifier(max_failures=0).verify(
        buggy,
        prompt,
        "weird",
        official_tests,
    )

    assert result.failures == []
