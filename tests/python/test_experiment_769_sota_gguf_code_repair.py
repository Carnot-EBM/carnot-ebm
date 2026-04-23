"""Tests for Exp 769 SOTA GGUF 2-round code repair helpers.

Spec: REQ-REPAIR-022, REQ-REPAIR-023, SCENARIO-REPAIR-042, SCENARIO-REPAIR-043
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from carnot.pipeline.two_round_repair import TwoRoundResult  # noqa: E402
from scripts.experiment_769_sota_gguf_code_repair import (  # noqa: E402
    build_repair_prompt_769,
    classify_verdict_769,
    compute_repair_metrics,
)


class TestBuildRepairPrompt769:
    """REQ-REPAIR-022, SCENARIO-REPAIR-042: repair prompt includes error message from round 1."""

    def test_prompt_includes_error_message(self):
        # REQ-REPAIR-022: error message from round 1 must appear in repair prompt.
        prompt = build_repair_prompt_769(
            original_problem="def add(a, b): ...",
            failed_code="def add(a, b): return a - b",
            error_message="AssertionError: expected 5 got 1",
        )
        assert "AssertionError: expected 5 got 1" in prompt

    def test_prompt_includes_failing_code(self):
        # The model must see what it wrote so it can identify the bug.
        prompt = build_repair_prompt_769(
            original_problem="def foo(): ...",
            failed_code="def foo(): return None",
            error_message="TypeError: expected int",
        )
        assert "def foo(): return None" in prompt

    def test_prompt_differs_from_generation_prompt(self):
        # Repair prompt must contain error context absent in the generation prompt.
        repair_prompt = build_repair_prompt_769(
            original_problem="def bar(): ...",
            failed_code="def bar(): return 0",
            error_message="AssertionError",
        )
        assert "Error" in repair_prompt
        # Must not be the plain generation prompt (which lacks error context).
        assert "Fix the bug" in repair_prompt

    def test_prompt_empty_error_handled(self):
        # Empty error_message must not crash — replaced with fallback string.
        prompt = build_repair_prompt_769(
            original_problem="def baz(): ...",
            failed_code="def baz(): pass",
            error_message="",
        )
        assert "(no traceback)" in prompt


class TestComputeRepairMetrics:
    """REQ-REPAIR-023, SCENARIO-REPAIR-043: signed_improvement and n_repaired are correct."""

    def test_pass_at_1_round2_gte_round1(self):
        # REQ-REPAIR-023: round2 cumulative pass rate can only be >= round1.
        # (Repair can improve or stay same — it cannot unrepair a passing answer.)
        results = [
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics(results)
        assert m["pass_at_1_round2"] >= m["pass_at_1_round1"]

    def test_signed_improvement_equals_delta(self):
        # REQ-REPAIR-023: signed_improvement = pass_at_1_round2 - pass_at_1_round1.
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics(results)
        expected_si = round(m["pass_at_1_round2"] - m["pass_at_1_round1"], 4)
        assert m["signed_improvement"] == expected_si

    def test_n_repaired_matches_definition(self):
        # n_repaired = count(NOT round0_pass AND round1_pass).
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics(results)
        assert m["n_repaired"] == 2

    def test_n_round2_attempted_counts_failures(self):
        # n_round2_attempted = count(NOT round0_pass).
        results = [
            TwoRoundResult(round0_pass=True, round1_pass=False, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=True, round2_pass=False),
            TwoRoundResult(round0_pass=False, round1_pass=False, round2_pass=False),
        ]
        m = compute_repair_metrics(results)
        assert m["n_round2_attempted"] == 2

    def test_empty_results(self):
        m = compute_repair_metrics([])
        assert m["pass_at_1_round1"] == 0.0
        assert m["pass_at_1_round2"] == 0.0
        assert m["signed_improvement"] == 0.0
        assert m["n_repaired"] == 0
        assert m["n_round2_attempted"] == 0


class TestClassifyVerdict769:
    """REQ-REPAIR-022: honest_verdict maps correctly to signed_improvement and inference_mode."""

    def test_blocked_no_live_gpu(self):
        # CARNOT_FORCE_LIVE not set → blocked_no_live_gpu regardless of improvement.
        assert classify_verdict_769(0.1, "blocked") == "blocked_no_live_gpu"
        assert classify_verdict_769(0.0, "blocked") == "blocked_no_live_gpu"
        assert classify_verdict_769(-0.1, "blocked") == "blocked_no_live_gpu"

    def test_blocked_model_load_failed(self):
        # llama-cpp load failure → blocked_model_load_failed.
        assert classify_verdict_769(0.0, "blocked_model_load_failed") == "blocked_model_load_failed"

    def test_positive_improvement_live_gpu(self):
        # signed_improvement > 0 on live GPU → sota_code_repair_positive.
        assert classify_verdict_769(0.1, "live_gpu") == "sota_code_repair_positive"

    def test_zero_improvement_live_gpu(self):
        # signed_improvement == 0 on live GPU → sota_code_repair_zero.
        assert classify_verdict_769(0.0, "live_gpu") == "sota_code_repair_zero"

    def test_negative_improvement_live_gpu(self):
        # signed_improvement < 0 on live GPU → sota_code_repair_negative.
        assert classify_verdict_769(-0.05, "live_gpu") == "sota_code_repair_negative"
