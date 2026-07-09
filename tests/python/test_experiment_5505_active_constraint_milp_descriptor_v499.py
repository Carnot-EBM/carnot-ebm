"""Tests for Exp5505 active-constraint MILP/MaxSAT/CSP descriptors.

Spec refs: REQ-VERIFY-5505, SCENARIO-VERIFY-5505.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as fixture_mod
from carnot import experiment_5505_active_constraint_milp_descriptor_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5505_active_constraint_milp_descriptor_v499.py")


def test_req_verify_5505_spec_declares_milp_maxsat_csp_descriptor_contract() -> None:
    """REQ-VERIFY-5505: OpenSpec anchors paths, fields, styles, and no-speedup scope."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5505") : spec.index("### REQ-VERIFY-5462")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5505" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.DESCRIPTOR_RELATIVE_PATH) in section
    assert str(mod.SCHEMA_RELATIVE_PATH) in section
    assert "MILP-style" in section
    assert "MaxSAT-style" in section
    assert "CSP-style" in section
    assert fixture_mod.REFERENCE_SOLVER_PATH in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert "hardware_speedup_claim` SHALL be false" in section
    for target in mod.HARDWARE_TARGETS:
        assert target in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5505_descriptors_cover_styles_and_exact_fallbacks() -> None:
    """SCENARIO-VERIFY-5505: every descriptor is executable and exact-authoritative."""

    descriptors = mod.build_descriptors()
    summary = mod.summarize_descriptors(descriptors)

    assert summary["num_descriptor_rows"] == 7
    assert summary["milp_style_rows"] == 2
    assert summary["maxsat_style_rows"] == 3
    assert summary["csp_style_rows"] == 2
    assert summary["exact_fallback_agreement_rate"] == pytest.approx(1.0)
    assert summary["partition_update_fields_present"] is True
    assert summary["descriptor_ready_for_hardware"] is True
    assert {row["descriptor_style"] for row in descriptors} == {"milp", "maxsat", "csp"}

    for descriptor in descriptors:
        assert set(mod.REQUIRED_DESCRIPTOR_FIELDS) <= set(descriptor)
        assert descriptor["typed_variables"]
        assert descriptor["hard_constraints"]
        assert descriptor["soft_preferences"]
        assert descriptor["objective_weights"]
        assert descriptor["exact_fallback"]["required"] is True
        assert descriptor["exact_fallback"]["solver"] == mod.EXACT_SOLVER_NAME
        assert descriptor["exact_fallback"]["agreement_with_expected"] is True
        assert descriptor["admissible_hardware_mapping"]["advisory_only"] is True
        assert descriptor["admissible_hardware_mapping"]["speedup_claim_allowed"] is False
        assert descriptor["admissible_hardware_mapping"]["board_timing_collected"] is False
        assert descriptor["partition_update"]["receipt_targets"] == list(mod.HARDWARE_TARGETS)
        assert mod.descriptor_exact_fallback_agrees(descriptor) is True
        assert mod.descriptor_partition_update_fields_present(descriptor) is True
        mod.validate_descriptor(descriptor)


def test_req_verify_5505_maxsat_rows_match_exp5499_reference_fixture() -> None:
    """REQ-VERIFY-5505: MaxSAT descriptors agree with Exp5499 exact references."""

    fixture = fixture_mod.build_fixture()
    references = {
        row["instance_id"]: fixture_mod.solve_reference(row) for row in fixture["instances"]
    }
    maxsat_rows = [row for row in mod.build_descriptors() if row["descriptor_style"] == "maxsat"]

    assert {row["source_instance_id"] for row in maxsat_rows} == set(references)
    assert any(row["exact_fallback"]["status"] == "infeasible" for row in maxsat_rows)
    for descriptor in maxsat_rows:
        reference = references[descriptor["source_instance_id"]]
        assert descriptor["exp5499_reference"] == reference
        assert descriptor["exact_fallback"]["status"] == reference["status"]
        assert descriptor["exact_fallback"]["solution"] == reference["assignment"]
        assert descriptor["exact_fallback"]["objective_score"] == reference["objective_score"]
        assert descriptor["expected_outputs"]["status"] == reference["status"]
        assert descriptor["expected_outputs"]["solution"] == reference["assignment"]
        assert descriptor["expected_outputs"]["objective_score"] == reference["objective_score"]


def test_req_verify_5505_artifact_writes_required_paths_and_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5505: result JSON emits the prompt-required bare fields."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    descriptors = json.loads((tmp_path / mod.DESCRIPTOR_RELATIVE_PATH).read_text(encoding="utf-8"))
    schema = json.loads((tmp_path / mod.SCHEMA_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert descriptors["descriptor_rows"] == artifact["descriptor_rows"]
    assert schema == mod.build_schema()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["descriptor_paths"] == [mod.DESCRIPTOR_RELATIVE_PATH.as_posix()]
    assert artifact["schema_paths"] == [mod.SCHEMA_RELATIVE_PATH.as_posix()]
    assert artifact["test_paths"] == [TEST_PATH.as_posix()]
    assert artifact["num_descriptor_rows"] == 7
    assert artifact["milp_style_rows"] == 2
    assert artifact["maxsat_style_rows"] == 3
    assert artifact["csp_style_rows"] == 2
    assert artifact["exact_fallback_agreement_rate"] == pytest.approx(1.0)
    assert artifact["partition_update_fields_present"] is True
    assert artifact["descriptor_ready_for_hardware"] is True
    assert artifact["hardware_speedup_claim"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == [{"command": str(TEST_PATH), "outcome": "passed"}]
    assert artifact["research_conductor_modified"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5505_validation_fails_closed_on_artifact_drift() -> None:
    """REQ-VERIFY-5505: artifact validation rejects missing fields and overclaims."""

    artifact = mod.build_artifact(tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}])
    mod.validate_artifact(artifact)
    assert mod.honest_verdict(False, ["exact_fallback_disagreement"]).startswith("blocked:")

    missing = deepcopy(artifact)
    missing.pop("descriptor_paths")
    with pytest.raises(ValueError, match="descriptor_paths"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    speedup = deepcopy(artifact)
    speedup["hardware_speedup_claim"] = True
    with pytest.raises(ValueError, match="hardware_speedup_claim"):
        mod.validate_artifact(speedup)

    bad_count = deepcopy(artifact)
    bad_count["milp_style_rows"] = 0
    with pytest.raises(ValueError, match="milp_style_rows"):
        mod.validate_artifact(bad_count)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5505_descriptor_validation_rejects_unsafe_rows() -> None:
    """REQ-VERIFY-5505: descriptor validation rejects exact and partition drift."""

    descriptor = mod.build_descriptors()[0]
    mod.validate_descriptor(descriptor)

    bad_expected = deepcopy(descriptor)
    bad_expected["expected_outputs"]["solution_hash"] = "bad"
    with pytest.raises(ValueError, match="exact_fallback"):
        mod.validate_descriptor(bad_expected)

    bad_fallback = deepcopy(descriptor)
    bad_fallback["exact_fallback"]["solution_hash"] = "bad"
    assert mod.descriptor_exact_fallback_agrees(bad_fallback) is False
    with pytest.raises(ValueError, match="exact_fallback"):
        mod.validate_descriptor(bad_fallback)

    bad_mapping = deepcopy(descriptor)
    bad_mapping["admissible_hardware_mapping"]["speedup_claim_allowed"] = True
    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        mod.validate_descriptor(bad_mapping)

    malformed_partition = deepcopy(descriptor)
    malformed_partition["partition_update"] = "not-a-mapping"
    assert mod.descriptor_partition_update_fields_present(malformed_partition) is False
    with pytest.raises(ValueError, match="partition_update"):
        mod.validate_descriptor(malformed_partition)

    incomplete_partition = deepcopy(descriptor)
    incomplete_partition["partition_update"].pop("expected_output_hash")
    assert mod.descriptor_partition_update_fields_present(incomplete_partition) is False
    with pytest.raises(ValueError, match="partition_update"):
        mod.validate_descriptor(incomplete_partition)

    bad_partition = deepcopy(descriptor)
    bad_partition["partition_update"]["receipt_targets"] = ["cpu"]
    with pytest.raises(ValueError, match="partition_update"):
        mod.validate_descriptor(bad_partition)

    bad_constraint = deepcopy(descriptor)
    bad_constraint["hard_constraints"] = [{"id": "HC_UNKNOWN", "type": "unknown"}]
    with pytest.raises(ValueError, match="constraint_type"):
        mod.validate_descriptor(bad_constraint)

    bad_preference = deepcopy(descriptor)
    bad_preference["soft_preferences"] = [{"id": "SP_UNKNOWN", "type": "unknown", "weight": 1}]
    with pytest.raises(ValueError, match="preference_type"):
        mod.validate_descriptor(bad_preference)
