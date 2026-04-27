"""Tests for Experiment 967: IterativeSelfRepair 100q + DebugRepair hypothesis step.

Covers only the new code introduced in experiment_967_*.py.  The underlying
IterativeSelfRepair pipeline is tested separately in test_iterative_self_repair.py.

Spec: REQ-CODE-033, SCENARIO-CODE-031, REQ-REPAIR-022
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path before importing experiment module
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import importlib.util
import types

_SPEC = importlib.util.spec_from_file_location(
    "exp967",
    _REPO_ROOT / "scripts" / "experiment_967_iterativeselfrepair_100q_debugrepair.py",
)
assert _SPEC is not None, "Could not find experiment_967 script"
_MOD: types.ModuleType = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Inline problem sets
# ---------------------------------------------------------------------------


def test_inline_problem_counts() -> None:
    """SCENARIO-CODE-031: inline CODE problem set must have exactly 50 extra problems."""
    assert len(_MOD._CODE_PROBLEMS_50_99) == 50


def test_inline_math_problem_count() -> None:
    """SCENARIO-CODE-031: inline MATH problem set must have exactly 50 GSM8K problems."""
    assert len(_MOD._MATH_PROBLEMS) == 50


def test_math_problems_have_required_fields() -> None:
    """Each math problem must have question, answer, and expected_str fields."""
    for prob in _MOD._MATH_PROBLEMS:
        assert "question" in prob, f"Missing 'question' in {prob}"
        assert "answer" in prob, f"Missing 'answer' in {prob}"
        assert "expected_str" in prob, f"Missing 'expected_str' in {prob}"
        assert str(prob["answer"]) == prob["expected_str"], (
            f"answer and expected_str mismatch: {prob['answer']} vs {prob['expected_str']}"
        )


def test_code_problems_have_required_fields() -> None:
    """Each code problem must have task_id, prompt, entry_point, and test fields."""
    for prob in _MOD._CODE_PROBLEMS_50_99:
        for field in ("task_id", "prompt", "entry_point", "test"):
            assert field in prob, f"Missing '{field}' in {prob['task_id']}"


# ---------------------------------------------------------------------------
# Hypothesis prompt builders
# ---------------------------------------------------------------------------


def test_build_hypothesis_prompt_contains_error() -> None:
    """REQ-CODE-033: hypothesis prompt must include the error text."""
    problem = {"prompt": "def foo():\n    pass\n", "entry_point": "foo", "test": ""}
    prompt = _MOD._build_hypothesis_prompt(problem, "def foo(): return 1", "AssertionError: 1 != 2")
    assert "AssertionError: 1 != 2" in prompt
    assert "WHY" in prompt or "diagnosis" in prompt.lower()


def test_build_hypothesis_prompt_contains_code() -> None:
    """REQ-CODE-033: hypothesis prompt must include the initial code for context."""
    problem = {"prompt": "def bar():\n    pass\n", "entry_point": "bar", "test": ""}
    code = "def bar(): return None"
    prompt = _MOD._build_hypothesis_prompt(problem, code, "TypeError")
    assert code in prompt


def test_build_debug_repair_prompt_contains_hypothesis() -> None:
    """REQ-CODE-033: debug repair prompt must embed the hypothesis text."""
    problem = {"prompt": "def baz():\n    pass\n", "entry_point": "baz", "test": ""}
    hypothesis = "The function returns None instead of an integer."
    prompt = _MOD._build_debug_repair_prompt(
        problem, "def baz(): return None", "TypeError: expected int", hypothesis
    )
    assert hypothesis in prompt
    assert "diagnosis" in prompt.lower()


def test_build_standard_repair_prompt_no_hypothesis() -> None:
    """Ablation baseline repair prompt must NOT contain hypothesis framing."""
    problem = {"prompt": "def qux():\n    pass\n", "entry_point": "qux", "test": ""}
    prompt = _MOD._build_standard_repair_prompt(problem, "def qux(): pass", "NameError")
    assert "diagnosis" not in prompt.lower()
    assert "hypothesis" not in prompt.lower()
    assert "NameError" in prompt


# ---------------------------------------------------------------------------
# check_math_answer
# ---------------------------------------------------------------------------


def test_check_math_answer_exact_match() -> None:
    """_check_math_answer should find the exact expected number in the response."""
    assert _MOD._check_math_answer("The answer is 42.", "42") is True


def test_check_math_answer_no_match() -> None:
    """_check_math_answer should return False when the expected number is absent."""
    assert _MOD._check_math_answer("The answer is 7.", "42") is False


def test_check_math_answer_no_partial_match() -> None:
    """_check_math_answer must not match sub-string of a larger number (word boundary)."""
    # '4' should not match inside '42'
    assert _MOD._check_math_answer("The answer is 42.", "4") is False


def test_check_math_answer_with_commas() -> None:
    """Commas in the response (e.g., '1,000') should be stripped before matching."""
    assert _MOD._check_math_answer("The answer is 1,000.", "1000") is True


# ---------------------------------------------------------------------------
# _compute_hypothesis_contribution
# ---------------------------------------------------------------------------


def test_hypothesis_contribution_positive() -> None:
    """hypothesis_contribution should be positive when with_hyp has higher repair pass rate."""
    without = [{"repair_passed": False}, {"repair_passed": False}, {"repair_passed": True}]
    with_h = [{"repair_passed": True}, {"repair_passed": True}, {"repair_passed": True}]
    contrib = _MOD._compute_hypothesis_contribution(without, with_h)
    assert contrib > 0


def test_hypothesis_contribution_zero_when_equal() -> None:
    """hypothesis_contribution should be 0.0 when pass rates are identical."""
    same = [{"repair_passed": True}, {"repair_passed": False}]
    contrib = _MOD._compute_hypothesis_contribution(same, same)
    assert contrib == 0.0


def test_hypothesis_contribution_handles_empty() -> None:
    """hypothesis_contribution should return 0.0 for empty result lists."""
    assert _MOD._compute_hypothesis_contribution([], []) == 0.0


def test_hypothesis_contribution_uses_min_length() -> None:
    """hypothesis_contribution should use only overlapping results when lists differ in length."""
    without = [{"repair_passed": False}]
    with_h = [{"repair_passed": True}, {"repair_passed": True}, {"repair_passed": True}]
    contrib = _MOD._compute_hypothesis_contribution(without, with_h)
    # Only 1 problem overlaps: 0% vs 100% = +1.0
    assert contrib == 1.0


# ---------------------------------------------------------------------------
# _pass_rates
# ---------------------------------------------------------------------------


def test_pass_rates_all_pass() -> None:
    """_pass_rates should return (1.0, 1.0, 0.0) when all problems pass baseline and repair."""
    results = [{"baseline_passed": True, "repair_passed": True} for _ in range(5)]
    base, repair, delta = _MOD._pass_rates(results)
    assert base == 1.0
    assert repair == 1.0
    assert delta == 0.0


def test_pass_rates_none_pass() -> None:
    """_pass_rates should return (0.0, 0.0, 0.0) when nothing passes."""
    results = [{"baseline_passed": False, "repair_passed": False} for _ in range(3)]
    base, repair, delta = _MOD._pass_rates(results)
    assert base == 0.0
    assert repair == 0.0
    assert delta == 0.0


def test_pass_rates_improvement() -> None:
    """_pass_rates should detect positive delta when repair adds passes."""
    results = [
        {"baseline_passed": False, "repair_passed": True},
        {"baseline_passed": False, "repair_passed": False},
    ]
    base, repair, delta = _MOD._pass_rates(results)
    assert base == 0.0
    assert repair == 0.5
    assert delta == 0.5


def test_pass_rates_empty() -> None:
    """_pass_rates should return (0.0, 0.0, 0.0) for empty input."""
    assert _MOD._pass_rates([]) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# _exec_code
# ---------------------------------------------------------------------------


def test_exec_code_passing() -> None:
    """_exec_code should return (True, None) for valid code + passing assertions."""
    code = "def add(a, b): return a + b"
    tests = ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    passed, error = _MOD._exec_code(code, tests)
    assert passed is True
    assert error is None


def test_exec_code_failing() -> None:
    """_exec_code should return (False, str) when assertion fails."""
    code = "def add(a, b): return a - b"
    tests = ["assert add(1, 2) == 3"]
    passed, error = _MOD._exec_code(code, tests)
    assert passed is False
    assert error is not None
    assert len(error) > 0


def test_exec_code_syntax_error() -> None:
    """_exec_code should handle syntax errors gracefully without raising."""
    code = "def add(a, b) return a + b"  # missing colon
    tests = ["assert add(1, 2) == 3"]
    passed, error = _MOD._exec_code(code, tests)
    assert passed is False
    assert error is not None


def test_exec_code_timeout() -> None:
    """_exec_code should return (False, timeout_msg) for code that loops forever."""
    code = "import time\nwhile True: time.sleep(0.1)"
    tests = []
    passed, error = _MOD._exec_code(code, tests, timeout_s=0.3)
    assert passed is False
    assert error is not None
    assert "Timeout" in error or "timeout" in error.lower()
