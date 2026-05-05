"""Tests for Exp 1378 publication-hold v16 claim-boundary review.

Spec: REQ-PUBLISH-017, SCENARIO-PUBLISH-018.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import publication_hold_v16_claim_boundary as exp1378


def _sources() -> dict[str, dict[str, Any]]:
    return {
        "exp1362": {
            "status": "complete",
            "publication_hold_state": "active",
            "publication_hold_rationale": {
                "active_blockers": [
                    "certificate_parse_truthfulness_and_unknown_preservation_gates_not_satisfied",
                    "semantic_validator_and_dvi_paths_gate_blocked_or_replay_only",
                    "fresh_verifier_selected_self_learning_samples_absent",
                    "external_ebt_arm_kona_extropic_parity_not_locally_demonstrated",
                ]
            },
            "ebt_arm_claim_boundary": {
                "external_dependency_claim_allowed": False,
                "local_ebt_arm_equivalence_proven": False,
            },
            "honest_verdict": (
                "publication_hold_active_local_evidence_does_not_support_ebt_arm_kona_or_hardware_claims"
            ),
        },
        "exp1366": {
            "status": "complete",
            "certificate_case_count": 4,
            "certificate_parse_rate": 1.0,
            "certificate_truthfulness_rate": 1.0,
            "prefix_injection_supported": True,
            "headline_result_allowed": True,
            "unknown_preservation_rate": 1.0,
            "honest_verdict": "tag_first_prefix_injection_crane_positive_parse_rate_1_0",
        },
        "exp1369": {
            "status": "complete",
            "parsed_certificate_cases": 4,
            "validator_execution_pass_rate": 1.0,
            "semantic_validator_claim_allowed": True,
            "unknown_preservation_rate": 1.0,
            "z3_constraint_pass_rate": 1.0,
            "honest_verdict": "semantic_validator_v2_complete_unknown_preserved",
        },
        "exp1370": {
            "status": "complete",
            "repair_claim_allowed": True,
            "mcs_localization_rate": 1.0,
            "repair_hint_precision": 1.0,
            "semantic_equivalence_pass_rate": 1.0,
            "accepted_repair_count": 2,
            "rejected_repair_count": 1,
            "honest_verdict": "verge_mcs_repair_localization_complete_claim_allowed",
        },
        "exp1371": {
            "status": "complete",
            "triage_claim_allowed": True,
            "false_acceptance_rate": 0.0,
            "unknown_silently_accepted_count": 0,
            "observed_full_verifier_call_reduction": 0.25,
            "honest_verdict": "margin_aware_scheduler_claim_allowed_zero_false_acceptance",
        },
        "exp1372": {
            "status": "complete",
            "formal_property_verified": True,
            "kan_formal_claim_allowed": True,
            "milp_verification_result": "verified",
            "milp_solver_status": "optimal",
            "property_energy_threshold": 5.562736705311322,
            "lp_certified_upper_bound": 5.5571795257855365,
            "hardware_execution_claimed": False,
            "honest_verdict": "cpu_only_gskan_energy_bound_verified_no_hardware_claim",
        },
        "exp1374": {
            "status": "complete",
            "headline_result_allowed": True,
            "path_used": "primary_semantic_verified",
            "dvi_ready": True,
            "fresh_verified_sample_count": 4,
            "self_learning_delta_overall": 1.596429,
            "nonforgetting_certificate_rate": 1.0,
            "memory_regression_count": 0,
            "accepted_violation_delta": -0.846154,
            "honest_verdict": (
                "continuous_self_learning_v3_primary_semantic_verified_dvi_ready_headline_allowed"
            ),
        },
    }


def _write_sources(project: Path, sources: dict[str, dict[str, Any]]) -> None:
    for key, rel_path in exp1378.SOURCE_PATHS.items():
        path = project / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sources[key]), encoding="utf-8")


def test_req_publish_017_builds_lift_recommended_boundary() -> None:
    """REQ-PUBLISH-017: resolved primary blockers recommend lift without parity claims."""

    artifact = exp1378.build_artifact(_sources(), project_root="/repo", run_date="20260505")

    exp1378.validate_artifact(artifact)
    assert exp1378.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["publication_hold_state"] == "lift_recommended"
    assert artifact["all_primary_blockers_resolved"] is True
    assert artifact["hold_blocker_resolved_certificate"] is True
    assert artifact["hold_blocker_resolved_semantic_repair"] is True
    assert artifact["hold_blocker_resolved_self_learning"] is True
    assert artifact["dvi_ready"] is True
    assert artifact["external_dependency_claim_allowed"] is False
    assert artifact["honest_verdict"] == exp1378.HONEST_VERDICT_LIFT_RECOMMENDED


def test_scenario_publish_018_summarizes_full_106_evidence_chain() -> None:
    """SCENARIO-PUBLISH-018: the .106 source fields flow into evidence summaries."""

    artifact = exp1378.build_artifact(_sources(), project_root="/repo")

    assert artifact["certificate_evidence_summary"]["certificate_parse_rate"] == 1.0
    assert artifact["certificate_evidence_summary"]["prefix_injection_supported"] is True
    assert artifact["semantic_repair_evidence_summary"]["validator_execution_pass_rate"] == 1.0
    assert artifact["semantic_repair_evidence_summary"]["repair_claim_allowed"] is True
    assert artifact["semantic_repair_evidence_summary"]["triage_claim_allowed"] is True
    assert artifact["semantic_repair_evidence_summary"]["false_acceptance_rate"] == 0.0
    assert artifact["kan_formal_evidence_summary"]["formal_property_verified"] is True
    assert artifact["kan_formal_evidence_summary"]["kan_formal_claim_allowed"] is True
    assert artifact["self_learning_evidence_summary"]["path_used"] == "primary_semantic_verified"
    assert artifact["self_learning_evidence_summary"]["headline_result_allowed"] is True


def test_req_publish_017_keeps_hold_active_when_primary_blocker_remains() -> None:
    """REQ-PUBLISH-017: any unresolved primary blocker keeps the state active."""

    sources = _sources()
    sources["exp1371"] = dict(
        sources["exp1371"],
        triage_claim_allowed=False,
        false_acceptance_rate=0.25,
    )

    artifact = exp1378.build_artifact(sources, project_root="/repo")

    exp1378.validate_artifact(artifact)
    assert artifact["publication_hold_state"] == "active"
    assert artifact["all_primary_blockers_resolved"] is False
    assert artifact["hold_blocker_resolved_semantic_repair"] is False
    assert "semantic_repair_chain_incomplete" in artifact["remaining_primary_blockers"]
    assert artifact["honest_verdict"] == exp1378.HONEST_VERDICT_ACTIVE


def test_req_publish_017_validation_rejects_overclaims() -> None:
    """REQ-PUBLISH-017: validation rejects external parity and premature lift."""

    artifact = exp1378.build_artifact(_sources(), project_root="/repo")

    external = dict(artifact, external_dependency_claim_allowed=True)
    with pytest.raises(ValueError, match="external_dependency_claim_allowed"):
        exp1378.validate_artifact(external)

    premature = dict(
        artifact,
        all_primary_blockers_resolved=False,
        publication_hold_state="lift_recommended",
    )
    with pytest.raises(ValueError, match="publication hold active"):
        exp1378.validate_artifact(premature)


def test_req_publish_017_run_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """REQ-PUBLISH-017: the runner writes bootstrap before loading local evidence."""

    _write_sources(tmp_path, _sources())
    output = tmp_path / exp1378.DEFAULT_OUT_PATH
    writes: list[str] = []

    artifact = exp1378.run(
        project_root=tmp_path,
        out_path=output,
        run_date="20260505",
        write_observer=lambda _path, payload: writes.append(str(payload["status"])),
    )

    assert writes == ["in_progress", "complete"]
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["publication_hold_state"] == "lift_recommended"
