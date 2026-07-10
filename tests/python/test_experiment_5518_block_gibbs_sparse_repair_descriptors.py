"""Tests for Exp5518 exact-checked sparse block repair descriptors.

Spec refs: REQ-VERIFY-5518, SCENARIO-VERIFY-5518.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5518_block_gibbs_sparse_repair_descriptors as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5518_block_gibbs_sparse_repair_descriptors.py")


def test_req_verify_5518_spec_declares_sparse_repair_contract() -> None:
    """REQ-VERIFY-5518: OpenSpec anchors paths, fields, exact fallback, and no speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5518") : spec.index("### REQ-VERIFY-5506")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5518" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.DESCRIPTOR_RELATIVE_PATH) in section
    assert str(mod.FIXTURE_RELATIVE_PATH) in section
    assert "without training a diffusion model" in section
    assert "without invoking a live LLM" in normalized
    assert "`speedup_claim_allowed` SHALL be false" in section
    assert mod.INFERENCE_SUBSTRATE in section
    assert mod.SPARSE_BLOCK_POLICY in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5518_sparse_blocks_are_strict_and_exact_checked() -> None:
    """SCENARIO-VERIFY-5518: sparse repair candidates are all exact-accepted or rejected."""

    fixtures = mod.build_selected_fixtures()
    descriptors = mod.build_sparse_descriptors(fixtures)
    comparison = mod.run_policy_comparison(fixtures=fixtures, descriptors=descriptors)

    assert len(fixtures["instances"]) == 2
    assert {row["instance_id"] for row in fixtures["instances"]} == {
        "claim_support_preference",
        "claim_safety_conflict",
    }
    for descriptor in descriptors["sparse_repair_descriptors"]:
        mod.validate_sparse_descriptor(descriptor)
        assert 0 < len(descriptor["repair_block_variables"]) < descriptor["variable_count"]
        assert descriptor["sparse_subset"] is True
        assert descriptor["exact_fallback"]["required"] is True
        assert descriptor["exact_fallback"]["status"] == "optimal"

    assert comparison["exact_only_success_rate"] == pytest.approx(1.0)
    assert comparison["sparse_repair_success_rate"] == pytest.approx(1.0)
    assert comparison["random_block_success_rate"] < comparison["sparse_repair_success_rate"]
    assert comparison["mean_iterations_exact_only"] > comparison["mean_iterations_sparse_repair"]
    assert comparison["all_candidates_exact_checked"] is True
    assert comparison["unchecked_candidate_count"] == 0
    assert comparison["wall_time_exact_only_s"] >= 0.0
    assert comparison["wall_time_sparse_repair_s"] >= 0.0
    assert comparison["wall_time_random_block_s"] >= 0.0

    for policy_name in ("exact_only", "sparse_repair", "random_block"):
        for attempt in comparison["policy_results"][policy_name]:
            assert attempt["seed"] in mod.SEEDS
            assert attempt["candidate_checks"]
            assert all("exact_validator_decision" in row for row in attempt["candidate_checks"])
            assert {row["exact_validator_decision"] for row in attempt["candidate_checks"]} <= {
                "accepted",
                "rejected",
            }


def test_req_verify_5518_exact_validator_rejects_hard_and_soft_drift() -> None:
    """REQ-VERIFY-5518: exact fallback rejects hard failures and soft-suboptimal repairs."""

    fixtures = mod.build_selected_fixtures()
    support = fixtures["instances"][0]
    optimum = mod.validate_candidate_assignment(support, support["exact_reference"]["assignment"])
    hard_violation = mod.validate_candidate_assignment(
        support,
        {"support": "unsupported", "source_quality": "primary", "scope": "bounded"},
    )
    soft_suboptimal = mod.validate_candidate_assignment(
        support,
        {"support": "entailed", "source_quality": "secondary", "scope": "bounded"},
    )

    assert optimum["accepted"] is True
    assert optimum["exact_validator_decision"] == "accepted"
    assert hard_violation["accepted"] is False
    assert hard_violation["exact_validator_decision"] == "rejected"
    assert hard_violation["reject_reason"] == "hard_constraints_failed"
    assert soft_suboptimal["accepted"] is False
    assert soft_suboptimal["exact_validator_decision"] == "rejected"
    assert soft_suboptimal["reject_reason"] == "not_exact_optimum"


def test_req_verify_5518_artifact_writes_required_payloads_and_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5518: result JSON emits the prompt-required fields and payload paths."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    descriptors = json.loads((tmp_path / mod.DESCRIPTOR_RELATIVE_PATH).read_text(encoding="utf-8"))
    fixtures = json.loads((tmp_path / mod.FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert descriptors == artifact["descriptor_payload"]
    assert fixtures == artifact["fixture_payload"]
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["descriptor_path"] == mod.DESCRIPTOR_RELATIVE_PATH.as_posix()
    assert artifact["fixture_paths"] == [
        mod.FIXTURE_RELATIVE_PATH.as_posix(),
        mod.EXP5499_FIXTURE_PATH.as_posix(),
    ]
    assert artifact["exact_fallback_used"] is True
    assert artifact["sparse_block_policy"] == mod.SPARSE_BLOCK_POLICY
    assert artifact["seeds"] == list(mod.SEEDS)
    assert artifact["exact_only_success_rate"] == pytest.approx(1.0)
    assert artifact["sparse_repair_success_rate"] == pytest.approx(1.0)
    assert artifact["random_block_success_rate"] < artifact["sparse_repair_success_rate"]
    assert artifact["mean_iterations_exact_only"] > artifact["mean_iterations_sparse_repair"]
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["active_constraint_sparse_repair_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == [{"command": str(TEST_PATH), "outcome": "passed"}]
    assert artifact["research_conductor_modified"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5518_validation_fails_closed_on_overclaims_and_drift() -> None:
    """REQ-VERIFY-5518: artifact validation rejects missing fields and unchecked candidates."""

    artifact = mod.build_artifact(tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}])
    mod.validate_artifact(artifact)
    assert mod.honest_verdict(False, ["candidate_missing_exact_decision"]).startswith("blocked:")

    missing = deepcopy(artifact)
    missing.pop("descriptor_path")
    with pytest.raises(ValueError, match="descriptor_path"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "verifier_ensemble_against_cached_candidates"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    speedup = deepcopy(artifact)
    speedup["speedup_claim_allowed"] = True
    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        mod.validate_artifact(speedup)

    unchecked = deepcopy(artifact)
    unchecked["unchecked_candidate_count"] = 1
    with pytest.raises(ValueError, match="unchecked_candidate_count"):
        mod.validate_artifact(unchecked)

    sparse_failed = deepcopy(artifact)
    sparse_failed["sparse_repair_success_rate"] = 0.5
    with pytest.raises(ValueError, match="sparse_repair_success_rate"):
        mod.validate_artifact(sparse_failed)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5518_descriptor_validation_rejects_non_sparse_or_unchecked_rows() -> None:
    """REQ-VERIFY-5518: descriptor validation rejects full blocks and missing exact fallback."""

    descriptor = mod.build_sparse_descriptors(mod.build_selected_fixtures())[
        "sparse_repair_descriptors"
    ][0]
    mod.validate_sparse_descriptor(descriptor)

    full_block = deepcopy(descriptor)
    full_block["repair_block_variables"] = list(full_block["variables"])
    with pytest.raises(ValueError, match="sparse_subset"):
        mod.validate_sparse_descriptor(full_block)

    no_fallback = deepcopy(descriptor)
    no_fallback["exact_fallback"]["required"] = False
    with pytest.raises(ValueError, match="exact_fallback"):
        mod.validate_sparse_descriptor(no_fallback)

    bad_policy = deepcopy(descriptor)
    bad_policy["sparse_block_policy"] = "trained_diffusion_model"
    with pytest.raises(ValueError, match="sparse_block_policy"):
        mod.validate_sparse_descriptor(bad_policy)


def test_req_verify_5518_readiness_blockers_name_non_ready_conditions() -> None:
    """REQ-VERIFY-5518: blocked readiness reports the precise failed gate."""

    fixtures = mod.build_selected_fixtures()
    descriptors = mod.build_sparse_descriptors(fixtures)
    comparison = mod.run_policy_comparison(fixtures=fixtures, descriptors=descriptors)

    bad_descriptors = deepcopy(descriptors)
    bad_descriptors["descriptor_count"] = 1
    assert "descriptor_count" in mod.readiness_blockers(bad_descriptors, comparison)

    unchecked = deepcopy(comparison)
    unchecked["unchecked_candidate_count"] = 1
    assert "unchecked_candidate_count" in mod.readiness_blockers(descriptors, unchecked)

    sparse_failed = deepcopy(comparison)
    sparse_failed["sparse_repair_success_rate"] = 0.0
    assert "sparse_repair_success_rate" in mod.readiness_blockers(descriptors, sparse_failed)

    not_checked = deepcopy(comparison)
    not_checked["all_candidates_exact_checked"] = False
    assert "all_candidates_exact_checked" in mod.readiness_blockers(descriptors, not_checked)
