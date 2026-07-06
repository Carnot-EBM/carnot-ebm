"""Tests for Exp 5317 EBT telemetry audit re-emission.

Spec refs: REQ-INFER-5317, SCENARIO-INFER-5317.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5301_ebt_spectral_step_control_diagnostic_v484 as exp5301
from carnot import experiment_5317_ebt_telemetry_audit_reemit_v485 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/llm-ebm-inference/spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def _value(artifact: dict[str, Any], field: str) -> Any:
    wrapped = artifact[field]
    assert isinstance(wrapped, dict)
    return wrapped["value"]


def test_req_infer_5317_spec_declares_audit_contract() -> None:
    """REQ-INFER-5317: OpenSpec anchors the V485 telemetry audit."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-INFER-5317") : spec.index("## Implementation Status")]

    for marker in (
        "REQ-INFER-5317",
        "SCENARIO-INFER-5317",
        str(mod.RESULT_RELATIVE_PATH),
        mod.INFERENCE_SUBSTRATE,
        "`ebt_telemetry_audited`",
        "`methodology_duration_s`",
        "`methodology_flag_cleared`",
        "workload counters",
        "future energy-descent headline claims",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_scenario_infer_5317_prior_flag_audit_keeps_claim_bounded() -> None:
    """SCENARIO-INFER-5317: prior flags are explained without reviving Exp3872."""

    prior_5301 = mod.read_json(REPO / mod.PRIOR_EXP5301_RELATIVE_PATH)
    prior_3872 = mod.read_json(REPO / mod.PRIOR_EXP3872_RELATIVE_PATH)
    audit = mod.audit_prior_artifacts(prior_5301, prior_3872)

    assert audit["exp5301_flagged_adversarial"] is True
    assert audit["exp5301_flag_kinds"] == ["DURATION_TOO_SHORT", "METHODOLOGY_MISSING"]
    assert audit["exp5301_underlying_diagnostic_valid"] is True
    assert audit["exp3872_pre_gate_blocked"] is True
    assert audit["exp3872_system2_claim_usable"] is False
    assert audit["methodology_issue"] == "duration_and_methodology_record_incomplete"
    assert audit["v485_action"] == "rerun_deterministic_diagnostic_with_explicit_workload_counters"


def test_req_infer_5317_workload_counters_are_explicit() -> None:
    """REQ-INFER-5317: workload counters explain the tiny deterministic duration."""

    counters = mod.compute_workload_counters(exp5301.run_diagnostic())

    assert counters["policy_count"] == 3
    assert counters["logged_steps_by_policy"] == {
        "adaptive_spectral": 8,
        "fixed_aggressive": 1,
        "fixed_conservative": 8,
    }
    assert counters["total_logged_steps"] == 17
    assert counters["alpha_attempts_total"] == 25
    assert counters["adaptive_recovery_shrink_count"] == 8
    assert counters["lambda_power_iterations_per_logged_step"] == exp5301.POWER_ITERATIONS
    assert counters["hessian_vector_products"] == 425
    assert counters["forward_energy_evaluations"] == 45
    assert counters["analytic_gradient_evaluations"] == 875
    assert counters["autograd_backward_calls"] == 0
    assert counters["random_probe_vectors"] == 17
    assert counters["llm_forward_passes"] == 0
    assert counters["hardware_invocations"] == 0
    assert mod.value_of("bare") == "bare"


def test_scenario_infer_5317_builds_required_artifact_fields(tmp_path: Path) -> None:
    """SCENARIO-INFER-5317: artifact carries V485 required fields and no claims."""

    artifact = mod.write_outputs(
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        duration_s=0.125,
        tests_run=[{"command": "unit exp5317", "outcome": "passed"}],
        run_date="20260706",
    )
    payload = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert payload == artifact
    mod.validate_artifact(artifact)
    for field in mod.PRINCIPLE_WRAPPED_FIELDS:
        assert isinstance(artifact[field], dict)
        assert artifact[field]["principle"] == mod.FIELD_PRINCIPLES[field]
        assert "value" in artifact[field]
    for field in mod.BARE_FIELDS:
        assert not isinstance(artifact[field], dict)

    assert _value(artifact, "experiment_id") == mod.EXPERIMENT_ID
    assert _value(artifact, "milestone") == mod.MILESTONE
    assert _value(artifact, "status") == "complete"
    assert _value(artifact, "honest_verdict").startswith("complete:")
    assert "methodology flag cleared" in _value(artifact, "honest_verdict")
    assert _value(artifact, "inference_substrate") == mod.INFERENCE_SUBSTRATE
    assert artifact["ebt_telemetry_audited"] is True
    assert artifact["methodology_duration_s"] == pytest.approx(0.125)
    assert artifact["methodology_flag_cleared"] is True
    assert artifact["lambda_max_logged"] is True
    assert artifact["step_control_recovery_logged"] is True
    assert artifact["no_sota_quality_claim"] is True
    assert artifact["no_hardware_speedup_claim"] is True
    assert _value(artifact, "workload_counters")["alpha_attempts_total"] == 25
    assert _value(artifact, "tests_run") == [{"command": "unit exp5317", "outcome": "passed"}]
    assert artifact["claim_quarantine"]["future_energy_descent_claims_eligible"] is False
    assert artifact["runtime_breakdown_s"]["wall_clock_total"] == pytest.approx(0.125)
    assert artifact["memory_utilization_proxies"]["process_max_rss_kb"] is None or isinstance(
        artifact["memory_utilization_proxies"]["process_max_rss_kb"],
        int,
    )


def test_req_infer_5317_validation_fails_closed_on_schema_drift() -> None:
    """REQ-INFER-5317: invalid substrate, bare fields, and claims fail."""

    artifact = mod.build_artifact(
        duration_s=0.2,
        tests_run=[{"command": "unit exp5317", "outcome": "passed"}],
        run_date="20260706",
    )

    broken = copy.deepcopy(artifact)
    broken["inference_substrate"]["value"] = "live_llm_inference"
    with pytest.raises(AssertionError, match="inference_substrate"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["methodology_duration_s"] = {"value": 0.2, "principle": "bad"}
    with pytest.raises(AssertionError, match="bare"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["no_sota_quality_claim"] = False
    with pytest.raises(AssertionError, match="SOTA"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["workload_counters"]["value"]["alpha_attempts_total"] = 0
    with pytest.raises(AssertionError, match="workload"):
        mod.validate_artifact(broken)

    broken = copy.deepcopy(artifact)
    broken["honest_verdict"]["value"] = "telemetry methodology flag cleared"
    with pytest.raises(AssertionError, match="honest_verdict"):
        mod.validate_artifact(broken)


def test_deliverable_file_validates_for_scenario_infer_5317() -> None:
    """SCENARIO-INFER-5317: committed deliverable satisfies the V485 contract."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["ebt_telemetry_audited"] is True
    assert artifact["methodology_flag_cleared"] is True
    assert artifact["no_sota_quality_claim"] is True
    assert artifact["no_hardware_speedup_claim"] is True
    assert _value(artifact, "workload_counters")["total_logged_steps"] == 17
