"""Tests for Exp 744: TwoRoundCodeRepairPipeline.

Coverage targets (REQ-CODE-031, REQ-CODE-032):
- build_repair_prompt includes traceback and expected output.
- execute captures SyntaxError correctly.
- pass rates computed correctly from round results.
- error classification maps correctly to error types.
- TwoRoundResult: round pass/fail tracked across rounds.
- classify_verdict covers all three branches.
- compute_pass_rates computes cumulative fractions.
- compute_error_type_breakdown counts per error type per round.

Spec: REQ-CODE-031, REQ-CODE-032, SCENARIO-CODE-029, SCENARIO-CODE-030
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from carnot.pipeline.two_round_repair import (  # noqa: E402
    TwoRoundCodeRepairPipeline,
    TwoRoundResult,
    ExecutionResult,
)
import experiment_744_iterative_2round_repair as exp744  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_llm(responses: list[str]):
    """Return a deterministic LLM mock that cycles through a list of responses."""
    call_count = {"n": 0}

    def _caller(prompt: str) -> str:  # noqa: ARG001
        idx = call_count["n"] % len(responses)
        call_count["n"] += 1
        return responses[idx]

    return _caller


# ---------------------------------------------------------------------------
# REQ-CODE-031: build_repair_prompt
# ---------------------------------------------------------------------------


class TestBuildRepairPrompt:
    """Tests for TwoRoundCodeRepairPipeline.build_repair_prompt.

    Spec: REQ-CODE-031
    """

    def test_prompt_includes_traceback(self):
        """build_repair_prompt MUST include the full traceback in the output.

        Spec: REQ-CODE-031 — the repair prompt must inject the traceback so the
        LLM sees the exact error type and failing line.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        prompt = pipeline.build_repair_prompt(
            original_problem="def f(x): ...",
            failed_code="def f(x): return x",
            traceback_str="Traceback (most recent call last):\n  AssertionError: 1 != 2",
            expected_output="2",
            actual_output="1",
        )
        assert "AssertionError: 1 != 2" in prompt, "Traceback must appear in repair prompt"

    def test_prompt_includes_expected_output(self):
        """build_repair_prompt MUST include the expected output value.

        Spec: REQ-CODE-031 — the model needs to see what the correct answer is.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        prompt = pipeline.build_repair_prompt(
            original_problem="def add(a, b): ...",
            failed_code="def add(a, b): return a - b",
            traceback_str="",
            expected_output="5",
            actual_output="3",
        )
        assert "5" in prompt, "Expected output must appear in repair prompt"
        assert "3" in prompt, "Actual output must appear in repair prompt"

    def test_prompt_includes_original_problem(self):
        """build_repair_prompt MUST include the original problem statement.

        Spec: REQ-CODE-031 — without the problem, the model cannot verify its fix.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        original = "def multiply(a, b) -> int: ..."
        prompt = pipeline.build_repair_prompt(
            original_problem=original,
            failed_code="def multiply(a, b): return a + b",
            traceback_str="AssertionError",
            expected_output="6",
            actual_output="5",
        )
        assert original.strip() in prompt, "Original problem must appear in repair prompt"

    def test_prompt_includes_failing_code(self):
        """build_repair_prompt MUST include the failing code.

        Spec: REQ-CODE-031 — the model must see its own incorrect code to fix it.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        bad_code = "def f(x): return x * x * x"
        prompt = pipeline.build_repair_prompt(
            original_problem="def f(x): ...",
            failed_code=bad_code,
            traceback_str="AssertionError",
            expected_output="4",
            actual_output="8",
        )
        assert bad_code.strip() in prompt, "Failing code must appear in repair prompt"

    def test_prompt_has_repair_instruction(self):
        """build_repair_prompt MUST contain a clear fix instruction.

        Spec: REQ-CODE-031 — a plain repair directive outperforms elaborate instructions.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        prompt = pipeline.build_repair_prompt(
            original_problem="def g(): ...",
            failed_code="def g(): pass",
            traceback_str="",
            expected_output="1",
            actual_output="None",
        )
        assert "Fix" in prompt or "fix" in prompt, "Repair instruction must appear"

    def test_prompt_with_no_traceback_shows_fallback(self):
        """build_repair_prompt MUST handle empty traceback gracefully.

        Spec: REQ-CODE-031 — some failures are wrong-output, not exceptions.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        prompt = pipeline.build_repair_prompt(
            original_problem="def h(): ...",
            failed_code="def h(): return 0",
            traceback_str="",
            expected_output="1",
            actual_output="0",
        )
        # Should not crash; should still mention the wrong output
        assert "0" in prompt


# ---------------------------------------------------------------------------
# REQ-CODE-031: execute — error capture
# ---------------------------------------------------------------------------


class TestExecute:
    """Tests for TwoRoundCodeRepairPipeline.execute.

    Spec: REQ-CODE-031
    """

    def test_execute_captures_syntax_error(self):
        """execute MUST return error_type='syntax_error' when code has SyntaxError.

        Spec: REQ-CODE-031 — SyntaxError is easiest to repair according to arXiv 2604.10508;
        we must classify it separately from runtime errors.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(x:\n    return x",  # Missing closing paren — SyntaxError
            test_cases=[{"call": "f(1)", "expected": 1}],
        )
        assert result.passed is False
        assert result.error_type == "syntax_error"
        assert "SyntaxError" in result.traceback_str

    def test_execute_passes_on_correct_code(self):
        """execute MUST return passed=True when all test cases pass.

        Spec: REQ-CODE-031 — the execution gate must correctly identify passing code.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def add(a, b):\n    return a + b",
            test_cases=[{"call": "add(1, 2)", "expected": 3}],
        )
        assert result.passed is True
        assert result.error_type == ""

    def test_execute_captures_assertion_error(self):
        """execute MUST return error_type='assertion_error' on wrong output.

        Spec: REQ-CODE-031 — assertion errors (~45% repair rate) must be tracked.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(x):\n    return x + 1",
            test_cases=[{"call": "f(1)", "expected": 10}],  # wrong expected
        )
        assert result.passed is False
        assert result.error_type == "assertion_error"
        assert result.actual_output == "2"
        assert result.expected_output == "10"

    def test_execute_captures_name_error(self):
        """execute MUST return error_type='name_error' when code uses undefined name.

        Spec: REQ-CODE-031 — NameError is one of the easiest errors to repair.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(x):\n    return undefined_var + x",
            test_cases=[{"call": "f(1)", "expected": 2}],
        )
        assert result.passed is False
        assert result.error_type == "name_error"

    def test_execute_captures_runtime_error_as_other(self):
        """execute MUST return error_type='other' for unclassified runtime errors.

        Spec: REQ-CODE-031 — not every error is one of the named categories.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(x):\n    return x / 0",
            test_cases=[{"call": "f(1)", "expected": 1}],
        )
        assert result.passed is False
        assert result.error_type == "other"

    def test_execute_no_test_cases_passes(self):
        """execute with empty test_cases MUST return passed=True.

        Spec: REQ-CODE-031 — no test cases means no failure.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(): pass",
            test_cases=[],
        )
        assert result.passed is True

    def test_execute_test_without_expected_does_not_fail_on_value(self):
        """execute MUST not assert equality when no 'expected' key is present.

        Spec: REQ-CODE-031 — test cases without an expected value only check
        that no exception is raised.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        result = pipeline.execute(
            "def f(x):\n    return x * 2",
            test_cases=[{"call": "f(5)"}],  # no "expected"
        )
        assert result.passed is True


# ---------------------------------------------------------------------------
# REQ-CODE-031: error classification
# ---------------------------------------------------------------------------


class TestClassifyError:
    """Tests for TwoRoundCodeRepairPipeline._classify_error.

    Spec: REQ-CODE-031
    """

    def test_syntax_error_classified(self):
        """_classify_error MUST return 'syntax_error' when SyntaxError in traceback.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("... SyntaxError: invalid syntax") == "syntax_error"

    def test_assertion_error_classified(self):
        """_classify_error MUST return 'assertion_error' when AssertionError in traceback.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("AssertionError: blah") == "assertion_error"

    def test_name_error_classified(self):
        """_classify_error MUST return 'name_error' when NameError in traceback.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("NameError: name 'x' is not defined") == "name_error"

    def test_timeout_classified(self):
        """_classify_error MUST return 'timeout' when TimeoutError in traceback.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("TimeoutError: execution exceeded 10s") == "timeout"

    def test_other_classified(self):
        """_classify_error MUST return 'other' for unknown error types.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("ZeroDivisionError: division by zero") == "other"

    def test_empty_traceback_returns_empty_string(self):
        """_classify_error MUST return '' for empty traceback.

        Spec: REQ-CODE-031
        """
        pipeline = TwoRoundCodeRepairPipeline()
        assert pipeline._classify_error("") == ""


# ---------------------------------------------------------------------------
# REQ-CODE-031: full run() method
# ---------------------------------------------------------------------------


class TestTwoRoundRun:
    """Tests for TwoRoundCodeRepairPipeline.run.

    Spec: REQ-CODE-031, REQ-CODE-032
    """

    def test_run_round0_pass(self):
        """run MUST return round0_pass=True when initial code is correct.

        Spec: REQ-CODE-031 — no repair needed if round 0 passes.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        llm = _make_llm(["def add(a, b):\n    return a + b"])
        result = pipeline.run(
            problem="def add(a, b): ...",
            test_cases=[{"call": "add(1, 2)", "expected": 3}],
            llm_caller=llm,
        )
        assert result.round0_pass is True
        assert result.round1_pass is False
        assert result.round2_pass is False
        assert result.error_types == []

    def test_run_round1_pass_after_repair(self):
        """run MUST return round1_pass=True when first repair fixes the bug.

        Spec: REQ-CODE-031, REQ-CODE-032 — repair in round 1 must be recorded.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        # Round 0: wrong code; round 1: correct code
        llm = _make_llm([
            "def add(a, b):\n    return a - b",   # round 0: wrong
            "def add(a, b):\n    return a + b",   # round 1: correct
        ])
        result = pipeline.run(
            problem="def add(a, b): ...",
            test_cases=[{"call": "add(1, 2)", "expected": 3}],
            llm_caller=llm,
        )
        assert result.round0_pass is False
        assert result.round1_pass is True
        assert result.round2_pass is False
        assert len(result.error_types) == 1

    def test_run_round2_pass_after_two_repairs(self):
        """run MUST attempt a second repair when round 1 also fails.

        Spec: REQ-CODE-031, REQ-CODE-032 — two repair rounds are the target.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        llm = _make_llm([
            "def add(a, b):\n    return a - b",  # round 0: wrong
            "def add(a, b):\n    return a * b",  # round 1: still wrong
            "def add(a, b):\n    return a + b",  # round 2: correct
        ])
        result = pipeline.run(
            problem="def add(a, b): ...",
            test_cases=[{"call": "add(1, 2)", "expected": 3}],
            llm_caller=llm,
        )
        assert result.round0_pass is False
        assert result.round1_pass is False
        assert result.round2_pass is True
        assert len(result.error_types) == 2

    def test_run_all_fail_still_returns_result(self):
        """run MUST return a complete TwoRoundResult even if all rounds fail.

        Spec: REQ-CODE-031 — a "not repaired" outcome is still a valid result.
        """
        pipeline = TwoRoundCodeRepairPipeline()
        llm = _make_llm(["def add(a, b):\n    return a - b"])
        result = pipeline.run(
            problem="def add(a, b): ...",
            test_cases=[{"call": "add(1, 2)", "expected": 3}],
            llm_caller=llm,
        )
        assert result.round0_pass is False
        assert result.round1_pass is False
        assert result.round2_pass is False
        assert isinstance(result, TwoRoundResult)


# ---------------------------------------------------------------------------
# REQ-CODE-032: pass rate computation
# ---------------------------------------------------------------------------


class TestComputePassRates:
    """Tests for exp744.compute_pass_rates.

    Spec: REQ-CODE-032
    """

    def test_all_pass_round0(self):
        """compute_pass_rates MUST return 1.0 for all rounds when all pass round 0.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
        ]
        rates = exp744.compute_pass_rates(results)
        assert rates["pass_round0"] == 1.0
        assert rates["pass_round1"] == 1.0
        assert rates["pass_round2"] == 1.0

    def test_cumulative_counting(self):
        """compute_pass_rates MUST count cumulatively (not per-round increments).

        Spec: REQ-CODE-032 — a problem that passes round 1 also counts in round 2.
        """
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),   # passed r0
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),   # passed r1
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=True),   # passed r2
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),  # never passed
        ]
        rates = exp744.compute_pass_rates(results)
        assert rates["pass_round0"] == 0.25       # 1/4
        assert rates["pass_round1"] == 0.5        # 2/4
        assert rates["pass_round2"] == 0.75       # 3/4

    def test_empty_list_returns_zeros(self):
        """compute_pass_rates MUST return zeros for empty input.

        Spec: REQ-CODE-032
        """
        rates = exp744.compute_pass_rates([])
        assert rates["pass_round0"] == 0.0
        assert rates["pass_round1"] == 0.0
        assert rates["pass_round2"] == 0.0

    def test_no_pass(self):
        """compute_pass_rates MUST return 0.0 for all when nothing ever passes.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        rates = exp744.compute_pass_rates(results)
        assert rates["pass_round0"] == 0.0
        assert rates["pass_round1"] == 0.0
        assert rates["pass_round2"] == 0.0


# ---------------------------------------------------------------------------
# REQ-CODE-032: classify_verdict
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """Tests for exp744.classify_verdict.

    Spec: REQ-CODE-032
    """

    def test_confirmed_at_threshold(self):
        """classify_verdict MUST return 'confirmed' when improvement >= 0.02.

        Spec: REQ-CODE-032 — 2pp is the target from arXiv 2604.10508.
        """
        assert exp744.classify_verdict(0.02) == "two_round_repair_confirmed"
        assert exp744.classify_verdict(0.10) == "two_round_repair_confirmed"

    def test_marginal_below_threshold(self):
        """classify_verdict MUST return 'marginal' for 0 < improvement < 0.02.

        Spec: REQ-CODE-032
        """
        assert exp744.classify_verdict(0.01) == "two_round_repair_marginal"
        assert exp744.classify_verdict(0.001) == "two_round_repair_marginal"

    def test_no_improvement_at_zero(self):
        """classify_verdict MUST return 'no_improvement' when improvement <= 0.

        Spec: REQ-CODE-032
        """
        assert exp744.classify_verdict(0.0) == "two_round_repair_no_improvement"
        assert exp744.classify_verdict(-0.05) == "two_round_repair_no_improvement"


# ---------------------------------------------------------------------------
# REQ-CODE-032: error_type_breakdown
# ---------------------------------------------------------------------------


class TestComputeErrorTypeBreakdown:
    """Tests for exp744.compute_error_type_breakdown.

    Spec: REQ-CODE-032
    """

    def test_counts_repaired_in_round1(self):
        """Error type fixed in round 1 must appear in repaired_round1 count.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(
                round0_pass=False, round1_pass=True, round2_pass=False,
                error_types=["syntax_error"],
            ),
        ]
        breakdown = exp744.compute_error_type_breakdown(results)
        assert breakdown["syntax_error"]["repaired_round1"] == 1
        assert breakdown["syntax_error"]["repaired_round2"] == 0
        assert breakdown["syntax_error"]["not_repaired"] == 0

    def test_counts_repaired_in_round2(self):
        """Error type fixed in round 2 must appear in repaired_round2 count.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(
                round0_pass=False, round1_pass=False, round2_pass=True,
                error_types=["assertion_error", "assertion_error"],
            ),
        ]
        breakdown = exp744.compute_error_type_breakdown(results)
        assert breakdown["assertion_error"]["repaired_round2"] == 1

    def test_counts_not_repaired(self):
        """Error type never fixed must appear in not_repaired count.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(
                round0_pass=False, round1_pass=False, round2_pass=False,
                error_types=["name_error"],
            ),
        ]
        breakdown = exp744.compute_error_type_breakdown(results)
        assert breakdown["name_error"]["not_repaired"] == 1

    def test_round0_pass_not_counted(self):
        """Problems that pass round 0 MUST NOT appear in error breakdown.

        Spec: REQ-CODE-032 — round 0 passes have no error to track.
        """
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False, error_types=[]),
        ]
        breakdown = exp744.compute_error_type_breakdown(results)
        assert breakdown == {}

    def test_multiple_error_types(self):
        """Breakdown MUST aggregate counts across multiple problems of same error type.

        Spec: REQ-CODE-032
        """
        results = [
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False, error_types=["other"]),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False, error_types=["other"]),
        ]
        breakdown = exp744.compute_error_type_breakdown(results)
        assert breakdown["other"]["repaired_round1"] == 1
        assert breakdown["other"]["not_repaired"] == 1
