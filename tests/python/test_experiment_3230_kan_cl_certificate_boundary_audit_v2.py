"""Tests for Exp 3230 FR-11 KAN-CL certificate boundary audit.

Spec refs: REQ-LEARN-3230, SCENARIO-LEARN-3230,
SCENARIO-LEARN-3230-BLOCKED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from carnot.eval import fr11_kan_cl_certificate_boundary_audit_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec/capabilities/self-learning/spec.md"


def _write_json(root: Path, rel_path: Path | str, payload: Mapping[str, Any]) -> None:
    path = root / Path(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_substrate() -> dict[str, Any]:
    return {
        "controller_memory_replay_only": True,
        "uses_checked_in_artifacts_only": True,
        "executes_live_model_inference": False,
        "fresh_live_inference_calls": 0,
        "model_weight_learning": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "base_model_weights_updated": False,
        "kan_model_weight_training": False,
        "hidden_state_mutation_claimed": False,
    }


def _exp3201_payload(
    *,
    terminal: bool = True,
    certificate_boundary: Mapping[str, Any] | None = None,
    fresh_live_calls: int = 0,
) -> dict[str, Any]:
    substrate = _safe_substrate()
    substrate["fresh_live_inference_calls"] = fresh_live_calls
    payload: dict[str, Any] = {
        "artifact": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1",
        "heldout_replay_count": 12,
        "drift_replay_count": 9,
        "negative_control_regression_count": 0,
        "locality_violation_count": 0,
        "rollback_triggered": False,
        "model_weight_update_performed": False,
        "sidecar_promotion_allowed": False,
        "audit_metric_schema": {
            "feature_space": ["historical_exact_evidence_key", "row_id", "routing_outcome"],
            "metrics": {
                "locality_boundary": "same exact evidence key retains routing bin",
                "negative_control_regression": "negative controls must not change",
            },
            "not_authority_for": ["KAN training", "sidecar verifier promotion"],
        },
        "inference_substrate": substrate,
        "honest_verdict": "complete: unit exp3201" if terminal else "draft: unit exp3201",
    }
    if certificate_boundary is not None:
        payload["certificate_boundary"] = dict(certificate_boundary)
    return payload


def _exp3216_payload(
    *,
    terminal: bool = True,
    budget_exceeded: bool = False,
) -> dict[str, Any]:
    return {
        "artifact": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
        "schema_version": "1.0",
        "experiment_id": "experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1",
        "milestone": "2026.05.297",
        "continuous_self_learning_task": True,
        "nonforgetting_queue_defined": True,
        "nonforgetting_queue_value": 2.0 if not budget_exceeded else 3.0,
        "nonforgetting_budget_exceeded": budget_exceeded,
        "nonforgetting_queue": {
            "nonforgetting_budget": 2.0,
            "nonforgetting_budget_exceeded": budget_exceeded,
            "nonforgetting_queue_defined": True,
            "nonforgetting_queue_value": 2.0 if not budget_exceeded else 3.0,
            "pressure_terms": {
                "affected_heldout_or_drift_routes": 2,
                "negative_control_regressions": 0,
                "unrouted_retraction_pressure": 0,
            },
        },
        "model_weight_update_claimed": False,
        "controller_memory_promotion_allowed": False,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "inference_substrate": _safe_substrate(),
        "honest_verdict": "complete: unit exp3216" if terminal else "draft: unit exp3216",
    }


def _complete_certificate_boundary() -> dict[str, Any]:
    return {
        "bounded_input_domains": {
            "defined": True,
            "domains": [{"feature": "routing_outcome_bin", "min": 0.0, "max": 1.0}],
        },
        "per_knot_budget": {
            "defined": True,
            "budget": {"knot:0": 0.05, "knot:1": 0.05},
        },
        "monotonicity_lipschitz_evidence": {
            "defined": True,
            "checks": [{"knot": "knot:0", "local_lipschitz_bound": 1.0}],
        },
        "pwa_milp_abstraction": {
            "ready": True,
            "segments": [{"feature": "routing_outcome_bin", "pieces": 2}],
            "error_bounds": {"global": 0.01},
            "property_checks": [{"property": "monotone_nonforgetting", "verified": True}],
        },
    }


def _write_sources(
    root: Path,
    *,
    exp3201: Mapping[str, Any] | None = None,
    exp3216: Mapping[str, Any] | None = None,
) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("no hidden model weight updates\n", encoding="utf-8")
    (root / "research-program.md").write_text("Continuous Self-Learning\n", encoding="utf-8")
    (root / "research-references.md").write_text(
        "KAN-CL\ncertificate boundaries\n", encoding="utf-8"
    )
    spec = root / mod.SPEC_REL_PATH
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-LEARN-3230\nSCENARIO-LEARN-3230\nSCENARIO-LEARN-3230-BLOCKED\n",
        encoding="utf-8",
    )
    _write_json(root, mod.EXP3201_REL_PATH, exp3201 or _exp3201_payload())
    _write_json(root, mod.EXP3216_REL_PATH, exp3216 or _exp3216_payload())


def test_req_learn_3230_spec_anchor_exists() -> None:
    """REQ-LEARN-3230: OpenSpec declares the certificate audit contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-LEARN-3230" in spec
    assert "SCENARIO-LEARN-3230" in spec
    assert "SCENARIO-LEARN-3230-BLOCKED" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "certificate_requirements" in spec
    assert "requirement_evidence_matrix" in spec
    assert "kan_sidecar_promotion_allowed=false" in spec


def test_req_learn_3230_evidence_matrix_marks_missing_certificates() -> None:
    """REQ-LEARN-3230-2/3/4: each requirement receives an evidence row."""

    matrix = mod.requirement_evidence_matrix(_exp3201_payload(), _exp3216_payload())
    by_id = {row["requirement_id"]: row for row in matrix}

    assert [req["requirement_id"] for req in mod.certificate_requirements()] == [
        "bounded_input_domains",
        "per_knot_budget",
        "monotonicity_or_lipschitz_evidence",
        "pwa_milp_abstraction",
        "nonforgetting_budget_check",
        "model_weight_immutability",
    ]
    assert by_id["bounded_input_domains"]["evidence_status"] == "missing"
    assert by_id["per_knot_budget"]["evidence_status"] == "missing"
    assert by_id["monotonicity_or_lipschitz_evidence"]["evidence_status"] == "missing"
    assert by_id["pwa_milp_abstraction"]["evidence_status"] == "missing"
    assert by_id["nonforgetting_budget_check"]["evidence_status"] == "present"
    assert by_id["model_weight_immutability"]["evidence_status"] == "present"
    assert mod.missing_certificate_count(matrix) == 4
    assert mod.per_knot_budget_defined(matrix) is False
    assert mod.pwa_milp_abstraction_ready(matrix) is False


def test_req_learn_3230_complete_certificate_sources_allow_boundary_readiness() -> None:
    """REQ-LEARN-3230-5/6/7: complete synthetic certificates become ready."""

    exp3201 = _exp3201_payload(certificate_boundary=_complete_certificate_boundary())
    exp3216 = _exp3216_payload()
    matrix = mod.requirement_evidence_matrix(exp3201, exp3216)

    assert mod.missing_certificate_count(matrix) == 0
    assert mod.per_knot_budget_defined(matrix) is True
    assert mod.pwa_milp_abstraction_ready(matrix) is True
    assert mod.certificate_boundary_ready(matrix) is True
    assert mod.kan_sidecar_promotion_allowed(matrix, exp3201, exp3216) is True
    assert mod.source_claims_live_or_mutation({"inference_substrate": "unsafe"}) is True
    assert mod.detected_model_weight_update({"model_weight_update_claimed": True}) is True
    assert mod.pwa_milp_evidence({"pwa_milp_abstraction": []})["evidence_status"] == "missing"


def test_scenario_learn_3230_writes_fail_closed_audit(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3230: missing certificates block KAN sidecar promotion."""

    _write_sources(tmp_path)
    output = mod.write_artifact(
        tmp_path,
        output_path=tmp_path / mod.OUTPUT_REL_PATH,
        started_s=10.0,
        now_s=12.0,
        tests_run=["SCENARIO-LEARN-3230 focused"],
    )
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert mod.REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["continuous_self_learning_task"] is True
    assert artifact["missing_certificate_count"] == 4
    assert artifact["per_knot_budget_defined"] is False
    assert artifact["pwa_milp_abstraction_ready"] is False
    assert artifact["certificate_boundary_ready"] is False
    assert artifact["kan_sidecar_promotion_allowed"] is False
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert artifact["tests_run"] == ["SCENARIO-LEARN-3230 focused"]
    assert artifact["duration_s"] == pytest.approx(2.0)
    assert artifact["honest_verdict"].startswith("complete:")
    assert "kan_sidecar_promotion_allowed=false" in artifact["honest_verdict"]
    assert all(row["exists"] for row in artifact["source_artifacts"] if row["required"])
    mod.validate_artifact(artifact)


def test_scenario_learn_3230_blocked_sources_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-LEARN-3230-BLOCKED: unsafe evidence blocks every certificate."""

    _write_sources(
        tmp_path,
        exp3201=_exp3201_payload(fresh_live_calls=1),
        exp3216=_exp3216_payload(),
    )
    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.5)

    assert artifact["blocked_reason"] == "exp3201_live_inference_or_weight_update_claimed"
    assert artifact["missing_certificate_count"] == len(mod.certificate_requirements())
    assert artifact["per_knot_budget_defined"] is False
    assert artifact["pwa_milp_abstraction_ready"] is False
    assert artifact["certificate_boundary_ready"] is False
    assert artifact["kan_sidecar_promotion_allowed"] is False
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["honest_verdict"].startswith("complete:")
    mod.validate_artifact(artifact)
