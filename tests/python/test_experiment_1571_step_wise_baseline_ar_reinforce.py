"""Tests for Exp 1571 step-wise AR-REINFORCE baselines.

Spec refs: REQ-VERIFY-1571, SCENARIO-VERIFY-1571.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.training import ar_reinforce_stepwise_baseline as exp1571


def test_spec_mentions_exp1571_contract() -> None:
    """REQ-VERIFY-1571, SCENARIO-VERIFY-1571: Exp 1571 is spec-anchored."""

    spec = (exp1571.PROJECT_ROOT / "openspec/capabilities/verification/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-VERIFY-1571" in spec
    assert "SCENARIO-VERIFY-1571" in spec
    assert "experiment_1571_step_wise_baseline_AR_REINFORCE.json" in spec
    assert "gradient_variance_reduction_factor >= 10.0" in spec


def test_config_records_requested_linear_ar_benchmark() -> None:
    """REQ-VERIFY-1571: the benchmark uses n=32, k=15, and Linear-AR couplings."""

    config = exp1571.StepWiseBaselineConfig()

    assert config.n == 32
    assert config.k == 15
    assert config.constraints_per_k == 10
    assert config.noise_fraction == pytest.approx(0.03)
    assert config.linear_ar_parameter_count == 528
    assert config.linear_ar_coupling_parameter_count == 496


def test_step_wise_baseline_uses_prefix_only_information() -> None:
    """REQ-VERIFY-1571: token baselines must not inspect x_t or future samples."""

    config = exp1571.StepWiseBaselineConfig(batch_size=128)
    problem = exp1571.build_problem(config)
    states = np.vstack([problem.planted_target, problem.planted_target]).astype(np.float64)
    states[1, 10:] = 1.0 - states[1, 10:]

    baseline = exp1571.compute_step_wise_baseline(states, problem)
    rewards = exp1571.evaluate_and_reward(states, problem.constraints)

    assert baseline[0, 10] == pytest.approx(baseline[1, 10])
    assert baseline[:, 15] == pytest.approx(rewards)
    assert baseline.shape == (2, 32)


def test_gradient_variance_gate_passes_for_ar_couplings() -> None:
    """SCENARIO-VERIFY-1571: step-wise baseline cuts AR-coupling variance by >=10x."""

    config = exp1571.StepWiseBaselineConfig()

    result = exp1571.run_ab_test(config)

    assert result.gradient_variance.reduction_factor >= 10.0
    assert result.gradient_variance.step_wise_coupling_trace < (
        result.gradient_variance.scalar_coupling_trace / 10.0
    )
    assert result.gradient_variance.metric == "linear_ar_coupling_trace_variance"


def test_noisy_convergence_rate_matches_theorem_2_proxy() -> None:
    """REQ-VERIFY-1571: 3% noisy step-wise estimator preserves convergence proxy."""

    config = exp1571.StepWiseBaselineConfig()

    result = exp1571.run_ab_test(config)

    assert result.convergence.matches_theorem_2 is True
    assert result.convergence.noisy_to_clean_rate_ratio >= config.convergence_rate_floor
    assert result.convergence.noisy_step_wise_snr > 0.0
    assert result.convergence.clean_step_wise_snr > 0.0


def test_run_experiment_writes_complete_required_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1571: runner writes the terminal JSON schema."""

    output = tmp_path / "experiment_1571.json"

    artifact = exp1571.run_experiment(
        output_path=output,
        config=exp1571.StepWiseBaselineConfig(batch_size=2048),
    )

    assert exp1571.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["status"] == "complete"
    assert artifact["step_wise_baseline_implemented"] is True
    assert artifact["gradient_variance_reduction_factor"] >= 10.0
    assert artifact["convergence_rate_matches_theorem_2"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_validate_artifact_rejects_missing_fields_and_failed_gates() -> None:
    """REQ-VERIFY-1571: terminal artifacts are schema- and gate-checked."""

    valid = exp1571.build_artifact(
        config=exp1571.StepWiseBaselineConfig(batch_size=16),
        result=exp1571.ABTestResult(
            gradient_variance=exp1571.GradientVarianceResult(
                metric="linear_ar_coupling_trace_variance",
                scalar_coupling_trace=100.0,
                step_wise_coupling_trace=5.0,
                scalar_full_trace=120.0,
                step_wise_full_trace=20.0,
                reduction_factor=20.0,
            ),
            convergence=exp1571.ConvergenceRateResult(
                clean_step_wise_snr=0.02,
                noisy_step_wise_snr=0.0198,
                noisy_to_clean_rate_ratio=0.99,
                matches_theorem_2=True,
            ),
        ),
    )

    assert exp1571.validate_artifact(valid) is None

    missing = dict(valid)
    missing.pop("gradient_variance_reduction_factor")
    with pytest.raises(ValueError, match="missing required fields"):
        exp1571.validate_artifact(missing)

    bad_status = dict(valid, status="in_progress")
    with pytest.raises(ValueError, match="status must be complete"):
        exp1571.validate_artifact(bad_status)

    bad_verdict = dict(valid, honest_verdict="passed")
    with pytest.raises(ValueError, match="honest_verdict"):
        exp1571.validate_artifact(bad_verdict)

    bad_variance = dict(valid, gradient_variance_reduction_factor=9.9)
    with pytest.raises(ValueError, match="variance reduction"):
        exp1571.validate_artifact(bad_variance)

    bad_convergence = dict(valid, convergence_rate_matches_theorem_2=False)
    with pytest.raises(ValueError, match="convergence"):
        exp1571.validate_artifact(bad_convergence)


def test_config_rejects_invalid_shape_values() -> None:
    """REQ-VERIFY-1571: invalid AR/AND benchmark shapes fail before sampling."""

    with pytest.raises(ValueError, match="n must be positive"):
        exp1571.StepWiseBaselineConfig(n=0).validate()
    with pytest.raises(ValueError, match="k must satisfy"):
        exp1571.StepWiseBaselineConfig(n=8, k=9).validate()
    with pytest.raises(ValueError, match="constraint_prefix_span"):
        exp1571.StepWiseBaselineConfig(k=15, constraint_prefix_span=14).validate()
    with pytest.raises(ValueError, match="batch_size"):
        exp1571.StepWiseBaselineConfig(batch_size=1).validate()
