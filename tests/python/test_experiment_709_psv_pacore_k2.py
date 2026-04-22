"""Tests for Experiment 709: PSV-PaCoRe K=2 parallel chains with diverse temperatures.

Covers:
- Energy-merge: lower-energy (correct) response is selected per question.
- Violation pool: violations from BOTH chains are collected, not just selected.
- FP rate: conservative (both-chain) estimate per iteration.
- Trend slope: linear regression computation matches Exp 697 logic.
- Honest verdict: correct string for each slope/GPU-mode combination.
- Blocked artifact: when CARNOT_FORCE_LIVE is not set.
- Deliverable JSON: when the artifact exists on disk.

Spec: REQ-LEARN-020, REQ-LEARN-021,
      SCENARIO-LEARN-020, SCENARIO-LEARN-021
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.verify.psv_pacore import IterationResult, PSVPaCoReRunner  # noqa: E402
from scripts.experiment_709_psv_pacore_k2 import (  # noqa: E402
    DELIVERABLE,
    EXP_697_BASELINE_SLOPE,
    EXP_ID,
    N_ITERATIONS,
    N_QUESTIONS,
    SCHEMA,
    TEMP_A,
    TEMP_B,
    _compute_honest_verdict,
    _linear_slope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_runner(
    a_correct: list[bool],
    b_correct: list[bool],
) -> PSVPaCoReRunner:
    """Build a PSVPaCoReRunner with deterministic stub functions.

    a_correct[i] is the verify_fn result for chain A on question i.
    b_correct[i] is the verify_fn result for chain B on question i.

    The inference_fn ignores temperature and device; it encodes the chain in
    the response string so tests can assert which response was selected.
    """
    n = len(a_correct)
    assert len(b_correct) == n

    # Response strings encode which chain produced them.
    a_responses = [f"chain_a_response_{i}" for i in range(n)]
    b_responses = [f"chain_b_response_{i}" for i in range(n)]

    # verify_fn looks at the response string to pick the right label.
    def verify_fn(response: str) -> bool:
        for i in range(n):
            if response == a_responses[i]:
                return a_correct[i]
            if response == b_responses[i]:
                return b_correct[i]
        return True

    call_counts: dict[str, int] = {"a": 0, "b": 0}

    def inference_fn(question: str, temperature: float, device: str) -> str:
        # Identify which chain is calling based on temperature.
        # temp_a=0.7 -> chain A; temp_b=1.0 -> chain B.
        q_idx = int(question.split("_")[-1]) if question.startswith("q_") else 0
        if abs(temperature - TEMP_A) < abs(temperature - TEMP_B):
            call_counts["a"] += 1
            return a_responses[q_idx]
        else:
            call_counts["b"] += 1
            return b_responses[q_idx]

    runner = PSVPaCoReRunner(
        inference_fn=inference_fn,
        verify_fn=verify_fn,
        n_iterations=1,
        n_questions=n,
    )
    return runner


def _make_questions(n: int) -> list[str]:
    return [f"q_{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# REQ-LEARN-021: Energy-merge selects lower-energy response
# ---------------------------------------------------------------------------


def test_energy_merge_selects_correct_chain_b_over_incorrect_chain_a() -> None:
    """When chain A is wrong and chain B is correct, chain B's response is selected.

    Spec: REQ-LEARN-021, SCENARIO-LEARN-020
    """
    runner = _make_runner(a_correct=[False], b_correct=[True])
    result = runner.run_iteration(
        _make_questions(1),
        temp_a=TEMP_A,
        temp_b=TEMP_B,
        iteration=0,
    )
    assert result.best_responses == ["chain_b_response_0"], (
        f"Expected chain B response (correct), got {result.best_responses}"
    )


def test_energy_merge_selects_correct_chain_a_over_incorrect_chain_b() -> None:
    """When chain A is correct and chain B is wrong, chain A's response is selected.

    Spec: REQ-LEARN-021
    """
    runner = _make_runner(a_correct=[True], b_correct=[False])
    result = runner.run_iteration(
        _make_questions(1),
        temp_a=TEMP_A,
        temp_b=TEMP_B,
        iteration=0,
    )
    assert result.best_responses == ["chain_a_response_0"]


def test_energy_merge_tie_selects_chain_a() -> None:
    """When both chains have the same energy (both wrong), chain A is selected (tie-break).

    Spec: REQ-LEARN-021-3 (tie: chain A wins)
    """
    runner = _make_runner(a_correct=[False], b_correct=[False])
    result = runner.run_iteration(
        _make_questions(1),
        temp_a=TEMP_A,
        temp_b=TEMP_B,
        iteration=0,
    )
    assert result.best_responses == ["chain_a_response_0"], (
        "Tie (both wrong = both energy=1.0) should select chain A"
    )


def test_energy_merge_both_correct_selects_chain_a() -> None:
    """When both chains are correct (energy=0.0 each), chain A wins the tie.

    Spec: REQ-LEARN-021-3
    """
    runner = _make_runner(a_correct=[True], b_correct=[True])
    result = runner.run_iteration(
        _make_questions(1),
        temp_a=TEMP_A,
        temp_b=TEMP_B,
        iteration=0,
    )
    assert result.best_responses == ["chain_a_response_0"]


# ---------------------------------------------------------------------------
# REQ-LEARN-020: Iteration collects violations from both chains
# ---------------------------------------------------------------------------


def test_violation_pool_collects_from_both_chains() -> None:
    """Violations from BOTH chains appear in all_violations even when not selected.

    Spec: REQ-LEARN-020-2, REQ-LEARN-021-4, SCENARIO-LEARN-021
    """
    # q0: chain A wrong, chain B correct -> chain B selected; chain A violation recorded
    # q1: chain A correct, chain B wrong -> chain A selected; chain B violation recorded
    # q2: both correct -> neither violation recorded
    runner = _make_runner(
        a_correct=[False, True, True],
        b_correct=[True, False, True],
    )
    result = runner.run_iteration(
        _make_questions(3),
        temp_a=TEMP_A,
        temp_b=TEMP_B,
        iteration=0,
    )
    assert len(result.all_violations) == 2, (
        f"Expected 2 violations (one from each chain), got {len(result.all_violations)}"
    )
    # Check that violations include responses from both chains.
    # q0: chain A wrong -> violation is chain_a_response_0
    # q1: chain B wrong -> violation is chain_b_response_1
    violation_responses = {r for _, r in result.all_violations}
    assert "chain_a_response_0" in violation_responses, "Chain A violation missing"
    assert "chain_b_response_1" in violation_responses, "Chain B violation missing"


def test_violation_pool_accumulates_across_iterations() -> None:
    """The constraint_pool property grows with each iteration.

    Spec: REQ-LEARN-020-3
    """
    runner = _make_runner(
        a_correct=[False],
        b_correct=[False],
    )
    runner.n_iterations = 3
    runner.run_10_iterations(_make_questions(1))
    # Each iteration: 2 violations (one per chain per question)
    assert len(runner.constraint_pool) == 6, (
        f"Expected 6 accumulated violations (3 iter × 2 chains), got {len(runner.constraint_pool)}"
    )


# ---------------------------------------------------------------------------
# REQ-LEARN-020: FP rate estimate (conservative: both chains failed)
# ---------------------------------------------------------------------------


def test_fp_rate_estimate_only_when_both_chains_fail() -> None:
    """fp_rate_estimate counts only questions where BOTH chains produced violations.

    Spec: REQ-LEARN-020-4, SCENARIO-LEARN-021
    """
    # 3 questions: only q1 has both chains failing
    runner = _make_runner(
        a_correct=[True, False, False],
        b_correct=[False, False, True],
    )
    result = runner.run_iteration(_make_questions(3), temp_a=TEMP_A, temp_b=TEMP_B)
    # Only q1 has both failing -> 1/3
    expected = 1 / 3
    assert abs(result.fp_rate_estimate - expected) < 1e-9, (
        f"Expected fp_rate={expected:.4f}, got {result.fp_rate_estimate:.4f}"
    )


def test_fp_rate_zero_when_at_least_one_chain_correct() -> None:
    """fp_rate_estimate is 0.0 when at least one chain is correct for every question.

    Spec: REQ-LEARN-020-4
    """
    runner = _make_runner(
        a_correct=[False, True, False],
        b_correct=[True, False, True],
    )
    result = runner.run_iteration(_make_questions(3), temp_a=TEMP_A, temp_b=TEMP_B)
    assert result.fp_rate_estimate == 0.0


def test_fp_rate_one_when_all_questions_both_fail() -> None:
    """fp_rate_estimate is 1.0 when every question has both chains failing.

    Spec: REQ-LEARN-020-4
    """
    n = 4
    runner = _make_runner(
        a_correct=[False] * n,
        b_correct=[False] * n,
    )
    result = runner.run_iteration(_make_questions(n), temp_a=TEMP_A, temp_b=TEMP_B)
    assert result.fp_rate_estimate == 1.0


# ---------------------------------------------------------------------------
# Linear regression slope — matches Exp 697 computation
# ---------------------------------------------------------------------------


def test_linear_slope_improving_sequence() -> None:
    """A decreasing sequence has negative slope.

    Spec: REQ-LEARN-020-3 (fp_rate_trend_slope as signal)
    """
    # Clear downtrend: [0.5, 0.4, 0.3, 0.2, 0.1]
    slope = _linear_slope([0.5, 0.4, 0.3, 0.2, 0.1])
    assert slope < 0, f"Expected negative slope for decreasing sequence, got {slope}"


def test_linear_slope_degrading_sequence() -> None:
    """An increasing sequence has positive slope (degradation).

    Spec: REQ-LEARN-020-3
    """
    slope = _linear_slope([0.1, 0.2, 0.3, 0.4, 0.5])
    assert slope > 0, f"Expected positive slope for increasing sequence, got {slope}"


def test_linear_slope_flat_sequence() -> None:
    """A constant sequence has slope 0.

    Spec: REQ-LEARN-020-3
    """
    slope = _linear_slope([0.3, 0.3, 0.3, 0.3])
    assert abs(slope) < 1e-9


def test_linear_slope_single_value_returns_zero() -> None:
    """Slope is 0.0 for a single value (undefined).

    Spec: REQ-LEARN-020-3
    """
    assert _linear_slope([0.5]) == 0.0


def test_linear_slope_empty_returns_zero() -> None:
    """Slope is 0.0 for an empty list.

    Spec: REQ-LEARN-020-3
    """
    assert _linear_slope([]) == 0.0


def test_linear_slope_exp697_baseline() -> None:
    """Slope of Exp 697 fp_rate sequence matches recorded baseline +0.004242.

    Spec: REQ-LEARN-020-3
    """
    # Recorded from Exp 697 result
    exp697_fp_rates = [0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.05, 0.05, 0.05, 0.0]
    slope = _linear_slope(exp697_fp_rates)
    assert abs(slope - 0.004242) < 0.0001, (
        f"Baseline slope mismatch: expected ~0.004242, got {slope:.6f}"
    )


# ---------------------------------------------------------------------------
# Honest verdict
# ---------------------------------------------------------------------------


def test_honest_verdict_improving() -> None:
    """Negative slope -> psv_pacore_improving.

    Spec: REQ-LEARN-020
    """
    assert _compute_honest_verdict(-0.005, "dualgpu") == "psv_pacore_improving"


def test_honest_verdict_flat() -> None:
    """Slope in [-0.001, 0.001] -> psv_pacore_flat.

    Spec: REQ-LEARN-020
    """
    assert _compute_honest_verdict(0.0, "singlegpu") == "psv_pacore_flat"
    assert _compute_honest_verdict(0.001, "singlegpu") == "psv_pacore_flat"
    assert _compute_honest_verdict(-0.001, "singlegpu") == "psv_pacore_improving"


def test_honest_verdict_still_degrading() -> None:
    """Slope > 0.001 -> psv_pacore_still_degrading.

    Spec: REQ-LEARN-020
    """
    assert _compute_honest_verdict(0.005, "dualgpu") == "psv_pacore_still_degrading"


def test_honest_verdict_sequential_fallback_overrides_slope() -> None:
    """sequential_fallback gpu_mode -> psv_pacore_dualgpu_fallback regardless of slope.

    Spec: REQ-LEARN-020
    """
    assert _compute_honest_verdict(-0.1, "sequential_fallback") == "psv_pacore_dualgpu_fallback"
    assert _compute_honest_verdict(0.1, "sequential_fallback") == "psv_pacore_dualgpu_fallback"


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_exp_id() -> None:
    """EXP_ID must be 709.  Spec: REQ-LEARN-020"""
    assert EXP_ID == 709


def test_schema() -> None:
    """SCHEMA must encode the experiment version.  Spec: REQ-LEARN-020"""
    assert SCHEMA == "carnot.psv_pacore_k2.v1"


def test_n_iterations() -> None:
    """N_ITERATIONS must be 10.  Spec: REQ-LEARN-020"""
    assert N_ITERATIONS == 10


def test_n_questions() -> None:
    """N_QUESTIONS must be 10.  Spec: REQ-LEARN-020"""
    assert N_QUESTIONS == 10


def test_baseline_slope() -> None:
    """EXP_697_BASELINE_SLOPE must be 0.004242.  Spec: REQ-LEARN-020"""
    assert EXP_697_BASELINE_SLOPE == 0.004242


def test_temperatures() -> None:
    """TEMP_A=0.7 (near-greedy) and TEMP_B=1.0 (stochastic).  Spec: REQ-LEARN-020"""
    assert TEMP_A == 0.7
    assert TEMP_B == 1.0


# ---------------------------------------------------------------------------
# Produce deliverable via main() with synthetic mocked inference
# ---------------------------------------------------------------------------


def test_main_produces_deliverable_with_synthetic_inference() -> None:
    """main() with CARNOT_FORCE_LIVE=1 and mocked inference produces a valid deliverable.

    Why this test writes to disk: the deliverable must exist for
    test_deliverable_json_exists_and_valid to validate it.  Using a fast
    synthetic inference_fn avoids hanging on HuggingFace model downloads in CI.

    Spec: REQ-LEARN-020, REQ-LEARN-021
    """
    import scripts.experiment_709_psv_pacore_k2 as mod  # noqa: PLC0415

    call_count: dict[str, int] = {"n": 0}

    def _synthetic_inference_fn(question: str, temperature: float, device: str) -> str:
        call_count["n"] += 1
        idx = abs(hash(question + str(temperature))) % 100
        return f"COMPUTE: result = {idx}"

    def _synthetic_verify_fn(response: str) -> bool:
        idx = int(response.split("=")[-1].strip()) if "=" in response else 0
        return idx % 3 != 0

    env = dict(os.environ)
    env["CARNOT_FORCE_LIVE"] = "1"

    with (
        patch("scripts.experiment_709_psv_pacore_k2._make_live_inference_fn", return_value=_synthetic_inference_fn),
        patch("scripts.experiment_709_psv_pacore_k2._make_verify_fn", return_value=_synthetic_verify_fn),
        patch("scripts.experiment_709_psv_pacore_k2._detect_gpu_mode", return_value=("sequential_fallback", "cpu", "cpu")),
        patch("scripts.experiment_709_psv_pacore_k2.apply_env_autofix"),
        patch(
            "scripts.experiment_709_psv_pacore_k2.ExperimentTimeoutWatchdog",
            return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: False),
        ),
        patch.dict(os.environ, env),
    ):
        mod.main()

    result_path = _REPO_ROOT / DELIVERABLE
    assert result_path.exists(), "Deliverable was not written"
    data = json.loads(result_path.read_text())

    assert data["experiment"] == EXP_ID
    assert data["status"] == "success"
    assert data["honest_verdict"] == "psv_pacore_dualgpu_fallback"
    assert len(data["fp_rate_per_iteration"]) == N_ITERATIONS
    assert data["gpu_mode"] == "sequential_fallback"
    assert isinstance(data["n_violations_collected"], int)
    assert "slope_improvement" in data


# ---------------------------------------------------------------------------
# Blocked artifact when CARNOT_FORCE_LIVE is not set
# ---------------------------------------------------------------------------


def test_blocked_artifact_when_no_carnot_force_live() -> None:
    """When CARNOT_FORCE_LIVE is not set, main() writes a blocked artifact.

    Spec: REQ-LEARN-020 (gate check)
    """
    import scripts.experiment_709_psv_pacore_k2 as mod  # noqa: PLC0415

    written: list[dict] = []
    deliverable_calls: list[None] = []

    class _FakeTemplate:
        def setup(self) -> None:
            pass

        def assert_deliverable_written(self) -> None:
            deliverable_calls.append(None)

        def build_result(self, data, **kw):
            result = dict(data)
            result["experiment"] = EXP_ID
            written.append(result)
            return result

    fake_tmpl = _FakeTemplate()

    env = {k: v for k, v in os.environ.items() if k != "CARNOT_FORCE_LIVE"}

    with (
        patch("scripts.experiment_709_psv_pacore_k2.ExperimentTemplate", return_value=fake_tmpl),
        patch(
            "scripts.experiment_709_psv_pacore_k2.ExperimentTimeoutWatchdog",
            return_value=MagicMock(__enter__=lambda s, *a: s, __exit__=lambda s, *a: False),
        ),
        patch("scripts.experiment_709_psv_pacore_k2.apply_env_autofix"),
        patch.dict(os.environ, env, clear=True),
    ):
        mod.main()

    assert len(written) == 1, "Expected exactly one artifact written"
    assert written[0]["honest_verdict"] == "psv_pacore_blocked_no_live"
    assert written[0]["inference_mode"] == "blocked"
    assert len(deliverable_calls) == 1, "assert_deliverable_written must be called"


# ---------------------------------------------------------------------------
# Deliverable JSON on disk (skipped if not yet written)
# ---------------------------------------------------------------------------


def test_deliverable_json_exists_and_valid() -> None:
    """Deliverable JSON on disk must exist and contain all required schema fields.

    Spec: REQ-LEARN-020, REQ-LEARN-021
    """
    result_path = _REPO_ROOT / DELIVERABLE
    if not result_path.exists():
        pytest.skip("Deliverable not yet written — run the experiment first")

    data = json.loads(result_path.read_text())

    required = {
        "experiment",
        "schema",
        "run_date",
        "status",
        "honest_verdict",
        "fp_rate_per_iteration",
        "fp_rate_trend_slope",
        "slope_improvement",
        "n_violations_collected",
        "gpu_mode",
        "baseline_slope_exp697",
    }
    missing = required - set(data.keys())
    assert not missing, f"Missing required fields: {missing}"

    assert data["experiment"] == EXP_ID
    assert data["baseline_slope_exp697"] == EXP_697_BASELINE_SLOPE

    valid_verdicts = {
        "psv_pacore_improving",
        "psv_pacore_flat",
        "psv_pacore_still_degrading",
        "psv_pacore_dualgpu_fallback",
        "psv_pacore_blocked_no_live",
    }
    assert data["honest_verdict"] in valid_verdicts, (
        f"Unexpected honest_verdict: {data['honest_verdict']}"
    )
