"""Tests for Exp5425 bounded KAN/KANDy measurement-access certificate.

Spec refs: REQ-KAN-5425, SCENARIO-KAN-5425.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5425_kan_measurement_access_certificate_v493 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def test_req_kan_5425_spec_declares_measurement_access_contract() -> None:
    """REQ-KAN-5425: OpenSpec anchors observable-vs-missing evidence."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5425") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-KAN-5425",
        "SCENARIO-KAN-5425",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp 5412 bounded KAN/KANDy",
        "unsupported board, token, and internal-state claims",
        "`broad_kan_verification_claim` MUST be false",
        exp.INFERENCE_SUBSTRATE,
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f"{field}`: {principle}" in normalized


def test_req_kan_5425_replays_rows_with_stable_provenance_checksums() -> None:
    """REQ-KAN-5425: reused Exp5412/Exp5406 rows carry row-level checksums."""

    rows = exp.load_measurement_rows()
    provenance = exp.build_row_provenance(rows)
    diagnostic = exp.evaluate_measurement_access_certificate()

    assert len(rows) == 16
    assert {row["hint_mode"] for row in rows} == {
        "no_hint",
        "stale_hint",
        "adversarial_hint",
        "candidate_hint",
    }
    assert len(provenance) == len(rows)
    assert len({row["row_checksum"] for row in provenance}) == len(rows)
    assert all(row["row_checksum"].startswith("sha256:") for row in provenance)
    assert diagnostic["row_checksums"] == [row["row_checksum"] for row in provenance]
    assert diagnostic["source_experiment"] == exp.SOURCE_EXPERIMENT


def test_scenario_kan_5425_rejects_false_properties_and_missing_evidence() -> None:
    """SCENARIO-KAN-5425: false controls are rejected or marked unsupported."""

    diagnostic = exp.evaluate_measurement_access_certificate()
    controls = diagnostic["measurement_access_controls"]
    false_controls = [row for row in controls if row["control_kind"] == "false_property"]
    active_false = [
        row for row in false_controls if row["classification"] == "observable_false"
    ]
    unsupported = [
        row
        for row in false_controls
        if row["classification"] == "missing_evidence_unsupported"
    ]

    assert diagnostic["property_family"] == exp.PROPERTY_FAMILY
    assert diagnostic["certificate_count"] == len(controls) == 14
    assert diagnostic["false_property_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["missing_evidence_detected"] is True
    assert diagnostic["broad_kan_verification_claim"] is False
    assert len(active_false) == 8
    assert len(unsupported) == 3
    assert {row["unsupported_claim_type"] for row in unsupported} == {
        "board_timing",
        "token_access",
        "internal_state",
    }

    for row in active_false:
        assert row["rejected"] is True
        assert row["missing_evidence"] == []
        assert row["counterexample"]["actual_route"] in {
            "reject_to_solver_fallback",
            "overwrite_with_solver_active_set",
        }
        assert row["row_provenance"][0]["row_checksum"] in diagnostic["row_checksums"]

    for row in unsupported:
        assert row["rejected"] is True
        assert row["counterexample"] is None
        assert row["row_provenance"] == []
        assert row["missing_evidence"]
        assert row["evidence_status"] == "missing_required_evidence"


def test_scenario_kan_5425_preserves_true_properties_under_same_certificate() -> None:
    """SCENARIO-KAN-5425: true measured controls remain provable."""

    diagnostic = exp.evaluate_measurement_access_certificate()
    true_controls = [
        row
        for row in diagnostic["measurement_access_controls"]
        if row["control_kind"] == "true_property"
    ]

    assert diagnostic["true_property_preservation_rate"] == pytest.approx(1.0)
    assert {row["property_id"] for row in true_controls} == {
        "true_exact_candidate_hints_supported_by_measured_rows",
        "true_no_hint_solver_baseline_supported_by_measured_rows",
        "true_solver_authority_and_final_validity_supported_by_measured_rows",
    }
    assert all(row["classification"] == "observable_supported" for row in true_controls)
    assert all(row["preserved"] is True for row in true_controls)
    assert all(row["row_provenance"] for row in true_controls)
    assert all(
        provenance["row_checksum"] in diagnostic["row_checksums"]
        for row in true_controls
        for provenance in row["row_provenance"]
    )


def test_req_kan_5425_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-KAN-5425: run() writes the required terminal artifact."""

    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=exp.default_tests_run())

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["certificate_count"] == len(artifact["measurement_access_controls"])
    assert artifact["property_family"] == exp.PROPERTY_FAMILY
    assert artifact["false_property_rejection_rate"] == pytest.approx(1.0)
    assert artifact["true_property_preservation_rate"] == pytest.approx(1.0)
    assert artifact["row_checksums"]
    assert artifact["missing_evidence_detected"] is True
    assert artifact["broad_kan_verification_claim"] is False
    assert artifact["kan_measurement_access_certificate_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == exp.default_tests_run()
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_kan_5425_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-KAN-5425: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["kan_measurement_access_certificate_ready"] is True
    assert checked_in["broad_kan_verification_claim"] is False
    exp.validate_artifact(checked_in)


def test_req_kan_5425_validation_rejects_broad_or_unproven_claims() -> None:
    """REQ-KAN-5425: validation fails closed on claim drift."""

    artifact = exp.build_artifact(tests_run=exp.default_tests_run())

    blocked = exp.build_artifact(tests_run=[])
    assert blocked["status"] == "blocked"
    assert blocked["kan_measurement_access_certificate_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(blocked)

    missing = deepcopy(artifact)
    missing.pop("measurement_access_controls")
    with pytest.raises(ValueError, match="measurement_access_controls"):
        exp.validate_artifact(missing)

    bad_family = deepcopy(artifact)
    bad_family["property_family"] = "broad_kan_verification"
    with pytest.raises(ValueError, match="property_family"):
        exp.validate_artifact(bad_family)

    broad_flag = deepcopy(artifact)
    broad_flag["broad_kan_verification_claim"] = True
    with pytest.raises(ValueError, match="broad_kan_verification_claim"):
        exp.validate_artifact(broad_flag)

    broad_verdict = deepcopy(artifact)
    broad_verdict["honest_verdict"] = "complete: broad KAN verification proved"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(broad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    no_missing = deepcopy(artifact)
    no_missing["missing_evidence_detected"] = False
    with pytest.raises(ValueError, match="missing_evidence_detected"):
        exp.validate_artifact(no_missing)

    bad_false_rate = deepcopy(artifact)
    bad_false_rate["false_property_rejection_rate"] = 0.5
    with pytest.raises(ValueError, match="false_property_rejection_rate"):
        exp.validate_artifact(bad_false_rate)

    bad_true_rate = deepcopy(artifact)
    bad_true_rate["true_property_preservation_rate"] = 0.5
    with pytest.raises(ValueError, match="true_property_preservation_rate"):
        exp.validate_artifact(bad_true_rate)

    no_checksums = deepcopy(artifact)
    no_checksums["row_checksums"] = []
    with pytest.raises(ValueError, match="row_checksums"):
        exp.validate_artifact(no_checksums)

    unchecked = deepcopy(artifact)
    unchecked["measurement_access_controls"][0]["rejected"] = False
    with pytest.raises(ValueError, match="measurement_access_controls"):
        exp.validate_artifact(unchecked)

    bad_provenance = deepcopy(artifact)
    bad_provenance["measurement_access_controls"][0]["row_provenance"][0][
        "row_checksum"
    ] = "sha256:not-present"
    with pytest.raises(ValueError, match="measurement_access_controls"):
        exp.validate_artifact(bad_provenance)


def test_req_kan_5425_readiness_blockers_are_explicit() -> None:
    """REQ-KAN-5425: blocked certificates name every failed readiness gate."""

    diagnostic = exp.evaluate_measurement_access_certificate()
    diagnostic["broad_kan_verification_claim"] = True
    diagnostic["false_property_rejection_rate"] = 0.5
    diagnostic["true_property_preservation_rate"] = 0.5
    diagnostic["missing_evidence_detected"] = False
    diagnostic["certificate_count"] = 0
    diagnostic["row_checksums"] = []
    diagnostic["claim_limits"] = []

    assert exp._readiness_blockers(diagnostic, ()) == [
        "broad_kan_claim",
        "false_properties_not_rejected",
        "true_properties_not_preserved",
        "missing_evidence_not_detected",
        "no_certificate_controls",
        "missing_row_checksums",
        "claim_limits_not_explicit",
        "tests_not_recorded",
    ]
