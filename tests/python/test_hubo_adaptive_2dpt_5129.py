"""Tests for Exp 5129 adaptive HUBO/p-spin 2D parallel tempering.

Spec refs: REQ-SAMPLE-5129, SCENARIO-SAMPLE-5129.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_5129_hubo_adaptive_2dpt_v470 as exp
from carnot.samplers.hubo_2dpt import (
    AdaptiveHubo2DPTConfig,
    AdaptiveHubo2DParallelTemperingSampler,
    adapt_inverse_temperature_ladder,
    build_synthetic_hubo_families,
    exact_enumerate,
)
from scripts import experiment_5129_hubo_adaptive_2dpt_v470 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"
ARTIFACT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_sample_5129_spec_declares_adaptive_contract() -> None:
    """REQ-SAMPLE-5129: OpenSpec declares the adaptive HUBO 2D PT contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAMPLE-5129") :]

    for marker in (
        "REQ-SAMPLE-5129",
        "SCENARIO-SAMPLE-5129",
        "adaptive inverse-temperature",
        exp.RESULT_RELATIVE_PATH,
    ):
        assert marker in section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_sample_5129_ladder_update_preserves_monotonic_order() -> None:
    """REQ-SAMPLE-5129: adaptive ladder updates preserve ordered betas."""

    initial = (0.35, 0.8, 1.6, 3.0)
    updated = adapt_inverse_temperature_ladder(
        initial,
        pair_acceptance_rates=(0.9, 0.45, 0.05),
        target_acceptance=0.45,
        learning_rate=0.5,
        min_beta_gap=0.05,
    )

    assert updated[0] == initial[0]
    assert updated[-1] == initial[-1]
    assert list(updated) == sorted(updated)
    assert all(right > left for left, right in zip(updated[:-1], updated[1:], strict=True))
    assert updated != initial
    assert updated[1] - updated[0] > initial[1] - initial[0]
    assert updated[-1] - updated[-2] < initial[-1] - initial[-2]


def test_scenario_sample_5129_sampler_records_residuals_and_balance() -> None:
    """SCENARIO-SAMPLE-5129: adaptive sampler records residual and balance telemetry."""

    problem = build_synthetic_hubo_families()[0]
    config = AdaptiveHubo2DPTConfig(
        initial_beta_grid=(0.35, 0.8, 1.6, 3.0),
        penalty_grid=(0.5, 2.0, 4.0),
        sweeps=12,
        swap_interval=1,
        adaptation_interval=3,
        target_acceptance=0.45,
        adaptation_learning_rate=0.5,
    )
    exact = exact_enumerate(problem, penalty=config.target_penalty)
    result = AdaptiveHubo2DParallelTemperingSampler(config).run(
        problem,
        seed=5129,
        exact_optimum_energy=exact.optimum_energy,
    )

    assert len(result.residual_energy_trace) == config.sweeps
    assert all(residual >= 0.0 for residual in result.residual_energy_trace)
    assert len(result.beta_grid_history) > 1
    assert result.beta_grid_history[0] == config.initial_beta_grid
    assert result.beta_grid_history[-1] != config.initial_beta_grid
    assert all(
        list(grid) == sorted(grid)
        and all(right > left for left, right in zip(grid[:-1], grid[1:], strict=True))
        for grid in result.beta_grid_history
    )
    assert result.swap_stats["beta_axis"].attempts > 0
    assert result.swap_stats["penalty_axis"].attempts > 0
    assert result.detailed_balance_sanity["passed"] is True
    assert result.detailed_balance_sanity["local_log_ratio_antisymmetry_max_abs"] <= 1e-9
    assert result.round_trip_proxy["mean_beta_span_fraction"] >= 0.0
    assert result.as_dict()["algorithm"] == "adaptive_two_d_beta_penalty_pt"


def test_req_sample_5129_artifact_schema_ready_gate_and_no_hardware_claim(tmp_path: Path) -> None:
    """REQ-SAMPLE-5129: artifact emits required fields and an honest ready gate."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_adaptive_2dpt_5129.py"],
    )
    payload = json.loads((tmp_path / exp.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    exp.validate_artifact(artifact)
    assert set(exp.REQUIRED_ARTIFACT_FIELDS).issubset(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["milestone"] == exp.MILESTONE
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["exp5116_baseline_loaded"] is True
    assert artifact["exact_enumeration_checked"] is True
    assert artifact["adaptive_2dpt_ready"] is True
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["conductor_modified"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["detailed_balance_sanity"]["passed"] is True
    assert artifact["best_energy_delta_vs_baselines"]["adaptive_vs_fixed_grid_2dpt"] <= 0.0
    assert artifact["optimum_hit_rate"]["adaptive_two_d_beta_penalty_pt"] >= artifact["optimum_hit_rate"]["fixed_grid_2dpt"]


def test_req_sample_5129_validation_rejects_missing_required_field(tmp_path: Path) -> None:
    """REQ-SAMPLE-5129: artifact validation rejects malformed terminal payloads."""

    artifact = exp.write_artifact(
        root=tmp_path,
        run_date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_adaptive_2dpt_5129.py"],
    )
    malformed = dict(artifact)
    malformed.pop("adaptive_2dpt_ready")

    with pytest.raises(ValueError, match="missing required fields"):
        exp.validate_artifact(malformed)


def test_scenario_sample_5129_script_entrypoint_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-5129: CLI wrapper writes the configured JSON artifact."""

    path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=1.0,
        tests_run=["tests/python/test_hubo_adaptive_2dpt_5129.py"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / exp.RESULT_RELATIVE_PATH
    exp.validate_artifact(payload)
    assert payload["hardware_speedup_claimed"] is False
    assert payload["conductor_modified"] is False


def test_deliverable_file_validates_for_scenario_sample_5129() -> None:
    """SCENARIO-SAMPLE-5129: checked-in deliverable satisfies the terminal schema."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    exp.validate_artifact(artifact)
    assert artifact["experiment_id"] == exp.EXPERIMENT_ID
    assert artifact["adaptive_2dpt_ready"] is True
