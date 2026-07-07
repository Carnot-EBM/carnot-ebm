"""Tests for Exp5358 solver-authoritative projection/cut bridge.

Spec refs: REQ-VERIFY-5358, SCENARIO-VERIFY-5358.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5358_solver_projection_cut_bridge_v488 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_verify_5358_spec_declares_projection_contract() -> None:
    """REQ-VERIFY-5358: OpenSpec anchors solver projection and cuts."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5358") : spec.index("### REQ-VERIFY-5345")]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5358",
        "SCENARIO-VERIFY-5358",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "valid",
        "near-valid",
        "invalid-repairable",
        "invalid-unrepairable",
        "misleading-neural",
        "no proposal",
        "solver_projection_ready",
        "deterministic_solver_projection",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_req_verify_5358_loads_qstr_and_kan_cut_sources() -> None:
    """REQ-VERIFY-5358: diagnostic uses QSTR plus one KAN/Ising cut fixture."""

    fixtures = exp.load_source_fixtures()
    proposals = exp.build_projection_proposals(fixtures)

    assert fixtures["qstr_ready"] is True
    assert fixtures["kan_cut_available"] is True
    assert fixtures["selected_cut"]["cut_id"].startswith("cut_forbid_")
    assert str(exp.qstr.RESULT_RELATIVE_PATH) in fixtures["source_artifacts"]
    assert str(exp.kan_bridge.RESULT_RELATIVE_PATH) in fixtures["source_artifacts"]
    assert [proposal.proposal_class for proposal in proposals] == list(
        exp.PROPOSAL_CLASS_NAMES
    )
    assert len(exp.PROPOSAL_CLASS_NAMES) == 6


def test_scenario_verify_5358_solver_projects_or_rejects_before_acceptance() -> None:
    """SCENARIO-VERIFY-5358: bad proposals cannot certify themselves."""

    diagnostic = exp.run_projection_diagnostic()
    rows = diagnostic["projection_results"]
    by_class = {row["proposal_class"]: row for row in rows}

    assert diagnostic["solver_authoritative"] is True
    assert diagnostic["proposal_class_count"] == 6
    assert diagnostic["projection_success_rate"] == pytest.approx(0.6)
    assert diagnostic["post_projection_validity_rate"] == pytest.approx(1.0)
    assert diagnostic["fallback_completeness_rate"] == pytest.approx(1.0)
    assert diagnostic["counterexample_cut_count"] == 1
    assert diagnostic["conflict_delta_vs_no_hint"] == pytest.approx(1.0)
    assert diagnostic["search_delta_vs_no_hint"] == pytest.approx(2.0)
    assert diagnostic["neural_corrector_agreement_rate"] == pytest.approx(5 / 6)
    assert diagnostic["unsafe_false_accepts"] == 0
    assert diagnostic["repairable_class_benefited"] is True
    assert diagnostic["solver_projection_ready"] is True

    assert by_class["valid"]["solver_action"] == "accept_exact"
    assert by_class["near-valid"]["solver_action"] == "project_to_intersection"
    assert by_class["invalid-repairable"]["solver_action"] == "repair_with_counterexample_cut"
    assert by_class["invalid-unrepairable"]["solver_action"] == "reject_with_counterexample_cut"
    assert by_class["misleading-neural"]["solver_action"] == "reject_and_fallback"
    assert by_class["no proposal"]["solver_action"] == "unguided_search"

    assert by_class["invalid-unrepairable"]["final_status"] == "rejected"
    assert by_class["misleading-neural"]["final_status"] == "unsatisfiable"
    assert all(row["post_projection_valid"] is True for row in rows)
    assert all(row["false_accept"] is False for row in rows)
    assert by_class["invalid-repairable"]["repairable_benefited"] is True


def test_req_verify_5358_artifact_schema_and_required_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5358: artifact exposes principle fields and bare metrics."""

    tests_run = [{"command": "unit exp5358", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_NAME
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "solver_projection_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["solver_authoritative"] is True
    assert artifact["proposal_class_count"] == 6
    assert artifact["projection_success_rate"] == pytest.approx(0.6)
    assert artifact["post_projection_validity_rate"] == pytest.approx(1.0)
    assert artifact["fallback_completeness_rate"] == pytest.approx(1.0)
    assert artifact["counterexample_cut_count"] == 1
    assert artifact["conflict_delta_vs_no_hint"] == pytest.approx(1.0)
    assert artifact["neural_corrector_agreement_rate"] == pytest.approx(5 / 6)
    assert artifact["unsafe_false_accepts"] == 0
    assert artifact["solver_projection_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run


def test_req_verify_5358_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5358: committed artifact is stable under replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=result["tests_run"]["value"])

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["solver_authoritative"] is True
    assert result["unsafe_false_accepts"] == 0
    assert result["solver_projection_ready"] is True
    exp.validate_artifact(result)


def test_req_verify_5358_validation_rejects_schema_drift() -> None:
    """REQ-VERIFY-5358: validation fails closed on unsafe drift."""

    artifact = exp.build_artifact(
        tests_run=[{"command": "unit exp5358", "outcome": "passed"}]
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_authority = deepcopy(artifact)
    bad_authority["solver_authoritative"] = False
    with pytest.raises(ValueError, match="solver_authoritative"):
        exp.validate_artifact(bad_authority)

    bad_accept = deepcopy(artifact)
    bad_accept["unsafe_false_accepts"] = 1
    with pytest.raises(ValueError, match="unsafe_false_accepts"):
        exp.validate_artifact(bad_accept)

    bad_ready = deepcopy(artifact)
    bad_ready["solver_projection_ready"] = {"value": True}
    with pytest.raises(ValueError, match="solver_projection_ready"):
        exp.validate_artifact(bad_ready)

    bad_metric = deepcopy(artifact)
    bad_metric["projection_success_rate"] = "0.6"
    with pytest.raises(ValueError, match="projection_success_rate"):
        exp.validate_artifact(bad_metric)
