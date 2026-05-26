"""Tests for Exp 3131 KAN PWA/MILP verifier abstraction audit.

Spec refs: REQ-KAN-3131, SCENARIO-KAN-3131.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.eval import kan_pwa_milp_verifier_abstraction_audit_v1 as exp


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "kan" / "spec.md"


def _config(tmp_path: Path) -> exp.ExperimentConfig:
    return exp.ExperimentConfig(
        repo_root=REPO_ROOT,
        output_path=tmp_path / "results" / exp.ARTIFACT_FILENAME,
        started_at=100.0,
        clock=lambda: 100.75,
        backend_name="z3",
        tests_run=("focused-req-kan-3131",),
    )


def test_req_kan_3131_spec_anchor_exists() -> None:
    """REQ-KAN-3131: the audit schema is declared before implementation."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-KAN-3131" in spec
    assert "SCENARIO-KAN-3131" in spec
    assert exp.ARTIFACT_FILENAME in spec
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in spec


def test_scenario_kan_3131_writes_complete_bounded_audit(tmp_path: Path) -> None:
    """SCENARIO-KAN-3131: existing tiny fixture produces the required audit."""

    artifact = exp.run_experiment(_config(tmp_path))
    saved = json.loads((tmp_path / "results" / exp.ARTIFACT_FILENAME).read_text("utf-8"))

    assert saved == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["kan_pwa_milp_audit_v1_ready"] is True
    assert artifact["kan_code_present"] is True
    assert artifact["abstraction_count"] == 2
    assert artifact["milp_property_check_count"] == 1
    assert artifact["milp_property_pass_count"] == 1
    assert artifact["implementation_blockers"] == []
    assert artifact["honest_verdict"].startswith("complete_")
    assert artifact["duration_s"] == pytest.approx(0.75)
    assert artifact["tests_run"] == ["focused-req-kan-3131"]

    local_summary = artifact["local_error_bound_summary"]
    assert local_summary["procedure"] == "max_per_segment_midpoint_residual"
    assert local_summary["max_local_error_bound"] == pytest.approx(0.0625)
    assert local_summary["unit_count"] == 2
    assert local_summary["segment_count"] == 8
    assert [unit["name"] for unit in local_summary["per_unit"]] == [
        "u0_x_squared",
        "u1_shifted_scaled_quadratic",
    ]

    global_summary = artifact["global_error_bound_summary"]
    assert global_summary["procedure"] == "weighted_output_error_propagation"
    assert global_summary["global_error_bound"] == pytest.approx(0.09375)
    assert global_summary["bounds_distinct_by_construction"] is True
    assert global_summary["weighted_contributions"][0]["contribution"] == pytest.approx(0.0625)
    assert global_summary["weighted_contributions"][1]["contribution"] == pytest.approx(0.03125)

    property_check = artifact["milp_property_checks"][0]
    assert property_check["property_verified"] is True
    assert property_check["solver_status"] == "optimal"
    assert property_check["milp_backend_name"] == "z3"
    assert property_check["certified_upper_bound"] == pytest.approx(0.53125)
    assert property_check["property_threshold"] == pytest.approx(0.625)

    substrate = artifact["inference_substrate"]
    assert substrate["mode"] == "cpu_pwa_milp_abstraction_audit"
    assert substrate["live_llm_inference"] is False
    assert substrate["live_model_inference"] is False
    assert substrate["model_weight_training"] is False
    assert substrate["model_weight_mutation"] is False
    assert substrate["deployed_verifier_improvement_claim"] is False
    assert substrate["hardware_execution"] is False

    source_paths = {item["path"] for item in artifact["source_artifacts"]}
    assert "python/carnot/verify/kan_pwa_milp_corrigendum.py" in source_paths
    assert "openspec/capabilities/kan/spec.md" in source_paths
    exp.validate_artifact(artifact)


def test_req_kan_3131_accounting_is_derived_from_existing_fixture() -> None:
    """REQ-KAN-3131: local/global summaries come from the PWA fixture."""

    fixture = exp.build_source_fixture()
    local_summary = exp.local_error_bound_summary(fixture)
    global_summary = exp.global_error_bound_summary(fixture)
    checks = exp.milp_property_checks(fixture, backend_name="z3")

    assert local_summary["max_local_error_bound"] == pytest.approx(fixture.local_error_bound)
    assert global_summary["global_error_bound"] == pytest.approx(fixture.global_error_bound)
    assert global_summary["global_error_bound"] != pytest.approx(
        local_summary["max_local_error_bound"]
    )
    assert checks[0]["property_verified"] is True
    assert checks[0]["counterexample_or_certificate"]["kind"] == "certificate"
    assert exp.compute_readiness(True, 2, checks, []) is True


def test_req_kan_3131_validation_blocks_overclaims(tmp_path: Path) -> None:
    """REQ-KAN-3131: validation rejects missing fields and unbounded claims."""

    artifact = exp.run_experiment(_config(tmp_path))
    missing_required = dict(artifact)
    missing_required.pop("honest_verdict")

    invalid_cases = [
        (missing_required, "missing required fields"),
        (artifact | {"honest_verdict": "waiting"}, "honest_verdict"),
        (artifact | {"kan_code_present": False, "implementation_blockers": []}, "blockers"),
        (artifact | {"abstraction_count": 0}, "abstraction_count"),
        (artifact | {"milp_property_check_count": 2}, "check count"),
        (artifact | {"milp_property_pass_count": 2}, "pass count"),
        (artifact | {"source_artifacts": []}, "source_artifacts"),
        (artifact | {"inference_substrate": "bad"}, "inference_substrate"),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_llm_inference": True}
            },
            "live LLM inference",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"live_model_inference": True}
            },
            "live model inference",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"model_weight_training": True}
            },
            "model weights",
        ),
        (
            artifact
            | {"inference_substrate": artifact["inference_substrate"] | {"hardware_execution": True}},
            "hardware execution",
        ),
        (
            artifact
            | {
                "inference_substrate": artifact["inference_substrate"]
                | {"deployed_verifier_improvement_claim": True}
            },
            "deployed verifier",
        ),
        (
            artifact
            | {
                "kan_code_present": False,
                "implementation_blockers": ["missing"],
            },
            "ready audit requires KAN code",
        ),
        (
            artifact
            | {
                "implementation_blockers": ["missing"],
            },
            "ready audit cannot have implementation blockers",
        ),
        (
            artifact
            | {
                "milp_property_check_count": 0,
                "milp_property_pass_count": 0,
                "milp_property_checks": [],
            },
            "ready audit requires every MILP property check to pass",
        ),
    ]
    for bad_artifact, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(bad_artifact)


def test_req_kan_3131_design_boundary_when_code_is_missing() -> None:
    """REQ-KAN-3131: absent KAN code produces exact implementation blockers."""

    blockers = exp.implementation_blockers_for_missing_code()

    assert exp.compute_readiness(False, 0, [], blockers) is False
    assert exp.honest_verdict(False).startswith("complete_")
    assert "python/carnot/verify/kan_pwa_milp_corrigendum.py" in blockers
    assert "tests/python/test_experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.py" in blockers
    assert "openspec/capabilities/kan/spec.md: REQ-KAN-3131" in blockers
    assert exp._sha256_file(Path("/missing/kan/fixture.py")) == ""
