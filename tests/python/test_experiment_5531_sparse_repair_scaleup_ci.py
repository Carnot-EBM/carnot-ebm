"""Tests for Exp5531 sparse repair scale-up confidence intervals.

Spec refs: REQ-VERIFY-5531, SCENARIO-VERIFY-5531.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5531_sparse_repair_scaleup_ci as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5531_sparse_repair_scaleup_ci.py")


def test_req_verify_5531_spec_declares_scaleup_contract() -> None:
    """REQ-VERIFY-5531: OpenSpec anchors required fields, CIs, and no speedup."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5531") : spec.index("### REQ-VERIFY-5519")]
    normalized = " ".join(section.split())

    assert "SCENARIO-VERIFY-5531" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert "without modifying `scripts/research_conductor.py`" in section
    assert "without invoking live model inference" in section
    assert "`matched_timing_available` SHALL be false" in section
    assert "`speedup_claim_allowed` SHALL be false" in section
    assert mod.INFERENCE_SUBSTRATE in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_verify_5531_fixtures_are_varied_sparse_and_exact_solvable() -> None:
    """SCENARIO-VERIFY-5531: scale-up fixtures stay exact-solvable with sparse blocks."""

    fixtures = mod.build_scaleup_fixtures()
    descriptors = mod.build_sparse_descriptors(fixtures)

    assert fixtures["schema"] == mod.FIXTURE_SCHEMA
    assert len(fixtures["instances"]) >= 5
    assert set(fixtures["fixture_families"]) >= {
        "exp5518_typed_claims",
        "four_variable_active_claims",
        "ternary_active_claims",
    }
    variable_counts = {len(instance["typed_claims"]) for instance in fixtures["instances"]}
    assert max(variable_counts) > 3

    for instance, descriptor in zip(
        fixtures["instances"], descriptors["sparse_repair_descriptors"], strict=True
    ):
        mod.validate_scaleup_instance(instance)
        mod.validate_sparse_descriptor(descriptor)
        assert instance["exact_reference"]["status"] == "optimal"
        assert instance["violated_hard_constraints"]
        assert 0 < len(descriptor["repair_block_variables"]) < descriptor["variable_count"]
        assert set(descriptor["repair_block_variables"]) <= {
            row["name"] for row in instance["typed_claims"]
        }


def test_scenario_verify_5531_multi_seed_comparison_has_ci_and_exact_checks() -> None:
    """SCENARIO-VERIFY-5531: policy rows aggregate over seeds with exact decisions."""

    fixtures = mod.build_scaleup_fixtures()
    descriptors = mod.build_sparse_descriptors(fixtures)
    comparison = mod.run_policy_comparison(fixtures=fixtures, descriptors=descriptors)

    assert comparison["n_instances"] == len(fixtures["instances"])
    assert comparison["n_seeds"] == len(mod.SEEDS)
    assert comparison["exact_only_success_rate"] == pytest.approx(1.0)
    assert comparison["sparse_repair_success_rate"] == pytest.approx(1.0)
    assert comparison["random_block_success_rate"] < comparison["sparse_repair_success_rate"]
    assert comparison["mean_iterations_exact_only"] > comparison["mean_iterations_sparse_repair"]
    assert comparison["exact_fallback_rate"] == pytest.approx(1.0)
    assert comparison["all_candidates_exact_checked"] is True
    assert comparison["unchecked_candidate_count"] == 0

    for field in (
        "exact_only_success_rate",
        "sparse_repair_success_rate",
        "random_block_success_rate",
        "mean_iterations_exact_only",
        "mean_iterations_sparse_repair",
        "exact_fallback_rate",
    ):
        ci = comparison["confidence_intervals"][field]
        assert ci["low"] <= comparison[field] <= ci["high"]
        assert ci["method"]

    expected_attempts = len(fixtures["instances"]) * len(mod.SEEDS)
    for policy_name in ("exact_only", "sparse_repair", "random_block"):
        assert len(comparison["policy_results"][policy_name]) == expected_attempts
        for attempt in comparison["policy_results"][policy_name]:
            assert attempt["candidate_checks"]
            assert attempt["exact_fallback_used"] is True
            assert {row["exact_validator_decision"] for row in attempt["candidate_checks"]} <= {
                "accepted",
                "rejected",
            }


def test_req_verify_5531_artifact_writes_required_result_json(tmp_path: Path) -> None:
    """REQ-VERIFY-5531: result JSON emits the required scale-up artifact fields."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixture_families"] == mod.fixture_families(artifact["fixture_payload"])
    assert artifact["n_instances"] >= 5
    assert artifact["n_seeds"] == len(mod.SEEDS)
    assert artifact["exact_only_success_rate"] == pytest.approx(1.0)
    assert artifact["sparse_repair_success_rate"] == pytest.approx(1.0)
    assert artifact["random_block_success_rate"] < artifact["sparse_repair_success_rate"]
    assert artifact["exact_fallback_rate"] == pytest.approx(1.0)
    assert artifact["matched_timing_available"] is False
    assert artifact["speedup_claim_allowed"] is False
    assert artifact["active_constraint_sparse_repair_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert str(TEST_PATH) in artifact["tests_added_or_reused"]
    assert artifact["research_conductor_modified"] is False
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5531_validation_fails_closed_on_missing_ci_or_overclaim() -> None:
    """REQ-VERIFY-5531: validation rejects omitted CIs, unchecked rows, and speedup claims."""

    artifact = mod.build_artifact(tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}])
    mod.validate_artifact(artifact)
    assert mod.honest_verdict(False, ["confidence_intervals"]).startswith("blocked:")

    missing_field = deepcopy(artifact)
    missing_field.pop("fixture_families")
    with pytest.raises(ValueError, match="fixture_families"):
        mod.validate_artifact(missing_field)

    missing_ci = deepcopy(artifact)
    missing_ci["confidence_intervals"].pop("random_block_success_rate")
    with pytest.raises(ValueError, match="confidence_intervals"):
        mod.validate_artifact(missing_ci)

    unchecked = deepcopy(artifact)
    unchecked["unchecked_candidate_count"] = 1
    with pytest.raises(ValueError, match="unchecked_candidate_count"):
        mod.validate_artifact(unchecked)

    speedup = deepcopy(artifact)
    speedup["speedup_claim_allowed"] = True
    with pytest.raises(ValueError, match="speedup_claim_allowed"):
        mod.validate_artifact(speedup)

    timing_overclaim = deepcopy(artifact)
    timing_overclaim["matched_timing_available"] = True
    with pytest.raises(ValueError, match="matched_timing_available"):
        mod.validate_artifact(timing_overclaim)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)


def test_req_verify_5531_readiness_blockers_name_failed_gates() -> None:
    """REQ-VERIFY-5531: blocked readiness reports precise gate failures."""

    fixtures = mod.build_scaleup_fixtures()
    descriptors = mod.build_sparse_descriptors(fixtures)
    comparison = mod.run_policy_comparison(fixtures=fixtures, descriptors=descriptors)

    too_small = deepcopy(comparison)
    too_small["n_instances"] = 1
    assert "n_instances" in mod.readiness_blockers(fixtures, descriptors, too_small)

    too_few_seeds = deepcopy(comparison)
    too_few_seeds["n_seeds"] = 1
    assert "n_seeds" in mod.readiness_blockers(fixtures, descriptors, too_few_seeds)

    bad_descriptor_count = deepcopy(descriptors)
    bad_descriptor_count["descriptor_count"] = 1
    assert "descriptor_count" in mod.readiness_blockers(
        fixtures, bad_descriptor_count, comparison
    )

    sparse_failed = deepcopy(comparison)
    sparse_failed["sparse_repair_success_rate"] = 0.0
    assert "sparse_repair_success_rate" in mod.readiness_blockers(
        fixtures, descriptors, sparse_failed
    )

    unchecked = deepcopy(comparison)
    unchecked["all_candidates_exact_checked"] = False
    assert "all_candidates_exact_checked" in mod.readiness_blockers(
        fixtures, descriptors, unchecked
    )

    unchecked_count = deepcopy(comparison)
    unchecked_count["unchecked_candidate_count"] = 1
    assert "unchecked_candidate_count" in mod.readiness_blockers(
        fixtures, descriptors, unchecked_count
    )

    no_ci = deepcopy(comparison)
    no_ci["confidence_intervals"] = {}
    assert "confidence_intervals" in mod.readiness_blockers(fixtures, descriptors, no_ci)

    no_fallback = deepcopy(comparison)
    no_fallback["exact_fallback_rate"] = 0.5
    assert "exact_fallback_rate" in mod.readiness_blockers(fixtures, descriptors, no_fallback)
