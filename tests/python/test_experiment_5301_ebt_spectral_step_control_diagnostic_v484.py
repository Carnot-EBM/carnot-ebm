"""Tests for Exp 5301 EBT spectral step-control diagnostic.

Spec refs: REQ-INFER-5301, SCENARIO-INFER-5301.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5301_ebt_spectral_step_control_diagnostic_v484 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, object], field: str) -> object:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_infer_5301_spec_declares_spectral_control_contract() -> None:
    """REQ-INFER-5301: OpenSpec anchors the stability diagnostic."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-INFER-5301") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-INFER-5301",
        "SCENARIO-INFER-5301",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "fixed conservative",
        "fixed aggressive",
        "adaptive spectral step control",
        "power-iteration Hessian-vector proxy",
        "SHALL NOT make an LLM quality",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_req_infer_5301_power_iteration_estimates_local_lambda_max() -> None:
    """REQ-INFER-5301: lambda-max is estimated from Hessian-vector products."""

    fixture = mod.build_sharpened_fixture()
    estimate = mod.estimate_lambda_max(
        fixture,
        fixture.initial_state,
        seed=mod.RANDOM_SEED,
        iterations=mod.POWER_ITERATIONS,
    )
    hvp = fixture.hessian_vector_product(fixture.initial_state, (0.0, 0.0, 1.0))

    assert estimate == pytest.approx(100.0, rel=1e-6)
    assert hvp == pytest.approx((0.0, 0.0, 100.0))
    assert fixture.condition_number == pytest.approx(100.0)


def test_scenario_infer_5301_conservative_policy_decreases_energy_monotonically() -> None:
    """SCENARIO-INFER-5301: conservative fixed alpha is a safe control."""

    result = mod.run_policy(mod.build_sharpened_fixture(), "fixed_conservative")
    energies_before = [step.energy_before for step in result.steps]
    energies_after = [step.energy_after for step in result.steps]

    assert result.diverged is False
    assert result.converged is True
    assert result.recovered is False
    assert result.total_recovery_shrinks == 0
    assert result.final_energy < result.initial_energy
    assert all(after < before for before, after in zip(energies_before, energies_after))
    assert all(step.alpha == pytest.approx(mod.FIXED_CONSERVATIVE_ALPHA) for step in result.steps)


def test_scenario_infer_5301_aggressive_policy_detects_divergence() -> None:
    """SCENARIO-INFER-5301: aggressive fixed alpha is flagged on the sharp axis."""

    result = mod.run_policy(mod.build_sharpened_fixture(), "fixed_aggressive")

    assert result.diverged is True
    assert result.converged is False
    assert result.recovered is False
    assert len(result.steps) == 1
    assert result.steps[0].energy_after > result.steps[0].energy_before
    assert result.steps[0].divergence_detected is True
    assert result.steps[0].alpha == pytest.approx(mod.FIXED_AGGRESSIVE_ALPHA)


def test_scenario_infer_5301_adaptive_spectral_policy_recovers() -> None:
    """SCENARIO-INFER-5301: adaptive spectral alpha shrinks and remains stable."""

    result = mod.run_policy(mod.build_sharpened_fixture(), "adaptive_spectral")

    assert result.diverged is False
    assert result.converged is True
    assert result.recovered is True
    assert result.total_recovery_shrinks >= 1
    assert result.final_energy < result.initial_energy
    assert result.steps[0].recovery_shrinks == 1
    assert result.steps[0].alpha == pytest.approx(0.012)
    assert all(step.energy_after <= step.energy_before for step in result.steps)
    assert all(step.alpha <= mod.STABILITY_LIMIT_FACTOR / step.lambda_max_estimate for step in result.steps)


def test_req_infer_5301_diagnostic_summary_compares_three_policies() -> None:
    """REQ-INFER-5301: all three alpha policies are reported together."""

    diagnostic = mod.run_diagnostic()
    policies = diagnostic.policy_results
    summary = diagnostic.summary()

    assert set(policies) == {"fixed_conservative", "fixed_aggressive", "adaptive_spectral"}
    assert summary["spectral_control_ready"] is True
    assert summary["divergence_recovery"]["aggressive_diverged"] is True
    assert summary["divergence_recovery"]["adaptive_recovered"] is True
    assert summary["divergence_recovery"]["adaptive_diverged"] is False
    assert summary["lambda_max_estimates"]["fixture"] == "ill_conditioned_sharpened_quadratic"
    assert summary["alpha_policy_results"]["adaptive_spectral"]["recovered"] is True
    assert len(summary["alpha_policy_results"]["fixed_conservative"]["steps"]) == mod.MAX_STEPS
    assert len(summary["alpha_policy_results"]["fixed_aggressive"]["steps"]) == 1
    assert len(summary["alpha_policy_results"]["adaptive_spectral"]["steps"]) == mod.MAX_STEPS
    assert len(mod.reproducibility_checksum(diagnostic)) == 64


def test_req_infer_5301_artifact_fields_are_principle_wrapped(tmp_path: Path) -> None:
    """REQ-INFER-5301: terminal artifact exposes required wrapped fields."""

    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.write_outputs(
        artifact_path=artifact_path,
        duration_s=0.5,
        tests_run=[{"command": "unit exp5301", "outcome": "passed"}],
    )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "spectral step-control is usable" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert _value(artifact, "spectral_control_ready") is True
    assert _value(artifact, "lambda_max_estimates")["fixture"] == "ill_conditioned_sharpened_quadratic"
    assert _value(artifact, "alpha_policy_results")["fixed_aggressive"]["diverged"] is True
    assert _value(artifact, "divergence_recovery")["adaptive_recovered"] is True
    assert _value(artifact, "random_seed") == mod.RANDOM_SEED
    assert len(_value(artifact, "reproducibility_checksum")) == 64
    assert artifact["tests_run"] == [{"command": "unit exp5301", "outcome": "passed"}]
    assert "REQ-INFER-5301" in artifact["spec_refs"]
    assert artifact["llm_quality_claimed"] is False


def test_req_infer_5301_validation_fails_closed_on_schema_drift() -> None:
    """REQ-INFER-5301: invalid stability, substrate, and checksum claims fail."""

    artifact = mod.build_artifact(
        duration_s=0.1,
        tests_run=[{"command": "unit exp5301", "outcome": "passed"}],
    )

    broken = copy.deepcopy(artifact)
    broken["spectral_control_ready"] = mod.wrap_field("spectral_control_ready", False)
    with pytest.raises(AssertionError, match="spectral control"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"] = mod.wrap_field("inference_substrate", "live_llm")
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["reproducibility_checksum"] = mod.wrap_field("reproducibility_checksum", "0" * 64)
    with pytest.raises(AssertionError, match="checksum"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_infer_5301() -> None:
    """SCENARIO-INFER-5301: committed deliverable satisfies the V484 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert _value(artifact, "spectral_control_ready") is True
    assert _value(artifact, "divergence_recovery")["aggressive_diverged"] is True
    assert _value(artifact, "divergence_recovery")["adaptive_recovered"] is True
    assert _value(artifact, "alpha_policy_results")["adaptive_spectral"]["final_energy"] < _value(
        artifact,
        "alpha_policy_results",
    )["adaptive_spectral"]["initial_energy"]
