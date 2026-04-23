"""Tests for Exp 759 iterative 2-round code repair helpers.

Spec: REQ-REPAIR-020, REQ-REPAIR-021, SCENARIO-REPAIR-040, SCENARIO-REPAIR-041
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.pipeline.two_round_repair import TwoRoundResult  # noqa: E402
from scripts.experiment_759_iterative_code_repair_live import (  # noqa: E402
    build_repair_prompt_759,
    classify_honest_verdict,
    compute_pass_at_1,
    compute_signed_improvement,
)


class TestBuildRepairPrompt759:
    """REQ-REPAIR-020, SCENARIO-REPAIR-040: repair prompt includes traceback + test case."""

    def test_prompt_includes_traceback(self):
        # Traceback must appear verbatim in the repair prompt so the model sees the error.
        prompt = build_repair_prompt_759(
            original_problem="def foo(x): ...",
            failed_code="def foo(x): return x",
            traceback_str="AssertionError: 1 != 2",
            test_case_call="foo(1)",
            expected_output="2",
            actual_output="1",
        )
        assert "AssertionError: 1 != 2" in prompt

    def test_prompt_includes_test_case_call(self):
        # REQ-REPAIR-020: test case input must appear in prompt.
        prompt = build_repair_prompt_759(
            original_problem="def foo(x): ...",
            failed_code="def foo(x): return x",
            traceback_str="NameError: foo not defined",
            test_case_call="foo(42)",
            expected_output="84",
            actual_output="42",
        )
        assert "foo(42)" in prompt

    def test_prompt_differs_from_generation_prompt(self):
        # Repair prompt must add error context — it cannot be identical to the generation prompt.
        gen_prompt = (
            "You are an expert Python programmer.  Write a correct Python function "
            "that solves the following problem.  Return ONLY the function definition "
            "with no extra explanation.\n\ndef foo(x): ..."
        )
        repair_prompt = build_repair_prompt_759(
            original_problem="def foo(x): ...",
            failed_code="def foo(x): return x",
            traceback_str="AssertionError",
            test_case_call="foo(1)",
            expected_output="2",
            actual_output="1",
        )
        # Repair prompt must contain error context not present in generation prompt.
        assert "Execution Error" in repair_prompt
        assert repair_prompt != gen_prompt

    def test_prompt_includes_expected_and_actual(self):
        # Expected vs actual must appear so the model understands the correctness gap.
        prompt = build_repair_prompt_759(
            original_problem="def bar(): ...",
            failed_code="def bar(): return 0",
            traceback_str="",
            test_case_call="bar()",
            expected_output="42",
            actual_output="0",
        )
        assert "42" in prompt
        assert "0" in prompt


class TestComputePassAt1:
    """REQ-CODE-032: pass@1 computation from TwoRoundResult lists."""

    def test_all_pass_round1(self):
        # All pass on initial generation — round2 cumulative equals round1.
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
        ]
        r1, r2 = compute_pass_at_1(results)
        assert r1 == 1.0
        assert r2 == 1.0

    def test_none_pass_round1(self):
        # No initial passes; some repaired in round1.
        results = [
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        r1, r2 = compute_pass_at_1(results)
        assert r1 == 0.0
        assert r2 == 0.5

    def test_mixed_pass_rates(self):
        # 2 of 4 pass initially; 1 more repaired → round2 cumulative is 3/4.
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        r1, r2 = compute_pass_at_1(results)
        assert r1 == 0.5
        assert r2 == 0.75

    def test_empty_results(self):
        r1, r2 = compute_pass_at_1([])
        assert r1 == 0.0
        assert r2 == 0.0


class TestComputeSignedImprovement:
    """REQ-CODE-032: signed_improvement = pass_at_1_round2 - pass_at_1_round1."""

    def test_positive_improvement(self):
        assert compute_signed_improvement(0.5, 0.7) == 0.2

    def test_zero_improvement(self):
        assert compute_signed_improvement(0.6, 0.6) == 0.0

    def test_negative_improvement(self):
        # Unexpected but must be handled (e.g. repair corrupts passing code via timeout).
        assert compute_signed_improvement(0.7, 0.5) == -0.2


class TestClassifyHonestVerdict:
    """REQ-REPAIR-020, REQ-REPAIR-021, SCENARIO-REPAIR-041: verdict maps correctly."""

    def test_blocked_when_inference_mode_blocked(self):
        # REQ-REPAIR-021, SCENARIO-REPAIR-041: no live GPU → blocked verdict.
        verdict = classify_honest_verdict(0.1, "blocked")
        assert verdict == "blocked_no_live_gpu"

    def test_blocked_regardless_of_improvement(self):
        # Inference mode "blocked" always wins over the improvement value.
        assert classify_honest_verdict(0.5, "blocked") == "blocked_no_live_gpu"
        assert classify_honest_verdict(0.0, "blocked") == "blocked_no_live_gpu"

    def test_positive_improvement_live_gpu(self):
        # REQ-REPAIR-020: positive improvement on live GPU → confirmed repair.
        verdict = classify_honest_verdict(0.05, "live_gpu")
        assert verdict == "code_repair_positive"

    def test_zero_improvement_live_gpu(self):
        verdict = classify_honest_verdict(0.0, "live_gpu")
        assert verdict == "code_repair_zero"

    def test_negative_improvement_live_gpu(self):
        # Unexpected case: repair made things worse (e.g., timeout on repaired code).
        verdict = classify_honest_verdict(-0.05, "live_gpu")
        assert verdict == "code_repair_negative"
