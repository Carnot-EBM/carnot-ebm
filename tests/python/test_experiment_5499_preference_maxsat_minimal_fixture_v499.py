"""Tests for Exp5499 Preference-MaxSAT minimal typed claim-state fixture.

Spec refs: REQ-VERIFY-5499, SCENARIO-VERIFY-5499.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5499_preference_maxsat_minimal_fixture_v499 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/verification/spec.md"
TEST_PATH = Path("tests/python/test_experiment_5499_preference_maxsat_minimal_fixture_v499.py")


def test_req_verify_5499_spec_declares_minimal_fixture_contract() -> None:
    """REQ-VERIFY-5499: OpenSpec anchors the exact Preference-MaxSAT fixture."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[
        spec.index("### REQ-VERIFY-5499") : spec.index("### REQ-VERIFY-5462")
    ]

    assert "SCENARIO-VERIFY-5499" in section
    assert str(mod.RESULT_RELATIVE_PATH) in section
    assert str(mod.FIXTURE_RELATIVE_PATH) in section
    assert "hard-infeasible negative control" in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section


def test_scenario_verify_5499_fixture_has_typed_claims_and_negative_control() -> None:
    """SCENARIO-VERIFY-5499: fixture rows expose typed domains and hard/soft rows."""

    fixture = mod.build_fixture()

    assert len(fixture["instances"]) == 3
    assert any(row["expected_status"] == "infeasible" for row in fixture["instances"])
    assert all(row["typed_claims"] for row in fixture["instances"])
    assert all(row["hard_constraints"] for row in fixture["instances"])
    assert all(row["soft_preferences"] for row in fixture["instances"])
    assert all(row["candidates"] for row in fixture["instances"])
    assert fixture["guided_decoding_used"] is False
    assert fixture["token_steering_used"] is False
    mod.validate_fixture(fixture)


def test_req_verify_5499_reference_solver_enumerates_exact_optima() -> None:
    """REQ-VERIFY-5499: exact enumerator is independent final authority."""

    fixture = mod.build_fixture()
    by_id = {row["instance_id"]: row for row in fixture["instances"]}

    support = mod.solve_reference(by_id["claim_support_preference"])
    assert support["status"] == "optimal"
    assert support["assignment"] == {
        "support": "entailed",
        "source_quality": "primary",
        "scope": "bounded",
    }
    assert support["objective_score"] == pytest.approx(11.0)

    safety = mod.solve_reference(by_id["claim_safety_conflict"])
    assert safety["status"] == "optimal"
    assert safety["assignment"] == {
        "safety": "safe",
        "citation": "present",
        "action": "accept",
    }
    assert safety["objective_score"] == pytest.approx(12.0)

    infeasible = mod.solve_reference(by_id["claim_infeasible_negative_control"])
    assert infeasible["status"] == "infeasible"
    assert infeasible["assignment"] is None
    assert infeasible["objective_score"] is None


def test_scenario_verify_5499_validators_measure_required_rates() -> None:
    """SCENARIO-VERIFY-5499: exact validators compute pass, optimality, and false accepts."""

    fixture = mod.build_fixture()
    report = mod.evaluate_fixture(fixture)

    assert report["num_instances"] == 3
    assert report["hard_constraint_pass_rate"] == pytest.approx(1.0)
    assert report["preference_optimality_rate"] == pytest.approx(1.0)
    assert report["independent_reference_agreement_rate"] == pytest.approx(1.0)
    assert report["false_accept_rate"] == pytest.approx(0.0)
    assert report["preference_maxsat_fixture_ready"] is True
    assert report["accepted_candidate_ids"] == [
        "support_exact_optimum",
        "safety_exact_optimum",
    ]
    assert report["rejected_candidate_ids"] == [
        "support_hard_violation",
        "support_soft_suboptimal",
        "safety_hard_violation_high_soft",
        "safety_soft_suboptimal",
        "infeasible_false_accept_probe",
    ]


def test_req_verify_5499_validation_rejects_false_accepts_and_suboptimal_accepts() -> None:
    """REQ-VERIFY-5499: final validators fail closed on bad accepted candidates."""

    fixture = mod.build_fixture()

    hard_violation = deepcopy(fixture)
    hard_violation["instances"][0]["candidates"][1]["accept"] = True
    bad_hard = mod.evaluate_fixture(hard_violation)
    assert bad_hard["preference_maxsat_fixture_ready"] is False
    assert bad_hard["false_accept_rate"] > 0.0
    assert any("false_accept" in blocker for blocker in bad_hard["readiness_blockers"])
    assert mod.honest_verdict(False, bad_hard["readiness_blockers"]).startswith("blocked:")

    suboptimal = deepcopy(fixture)
    suboptimal["instances"][0]["candidates"][2]["accept"] = True
    bad_soft = mod.evaluate_fixture(suboptimal)
    assert bad_soft["preference_maxsat_fixture_ready"] is False
    assert bad_soft["preference_optimality_rate"] < 1.0
    assert any("preference_optimality" in blocker for blocker in bad_soft["readiness_blockers"])

    missing_negative = deepcopy(fixture)
    missing_negative["instances"] = [
        row for row in missing_negative["instances"] if row["expected_status"] != "infeasible"
    ]
    with pytest.raises(ValueError, match="negative_control"):
        mod.validate_fixture(missing_negative)


def test_req_verify_5499_artifact_writes_required_deliverable_fields(tmp_path: Path) -> None:
    """REQ-VERIFY-5499: result JSON emits the required prompt schema fields."""

    artifact = mod.run(
        repo_root=tmp_path,
        tests_run=[
            {"command": str(TEST_PATH), "outcome": "passed"},
        ],
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    fixture = json.loads((tmp_path / mod.FIXTURE_RELATIVE_PATH).read_text(encoding="utf-8"))

    assert written == artifact
    assert fixture == mod.build_fixture()
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixture_path"] == mod.FIXTURE_RELATIVE_PATH.as_posix()
    assert artifact["reference_solver_path"] == mod.REFERENCE_SOLVER_PATH
    assert artifact["test_paths"] == [TEST_PATH.as_posix()]
    assert artifact["num_instances"] == 3
    assert artifact["hard_constraint_pass_rate"] == pytest.approx(1.0)
    assert artifact["preference_optimality_rate"] == pytest.approx(1.0)
    assert artifact["independent_reference_agreement_rate"] == pytest.approx(1.0)
    assert artifact["false_accept_rate"] == pytest.approx(0.0)
    assert artifact["preference_maxsat_fixture_ready"] is True
    assert artifact["guided_decoding_used"] is False
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)
    mod.validate_artifact(artifact)


def test_req_verify_5499_artifact_validation_fails_closed() -> None:
    """REQ-VERIFY-5499: artifact validator rejects schema drift and forbidden scope."""

    artifact = mod.build_artifact(tests_run=[{"command": str(TEST_PATH), "outcome": "passed"}])
    mod.validate_artifact(artifact)

    missing = deepcopy(artifact)
    missing.pop("fixture_path")
    with pytest.raises(ValueError, match="fixture_path"):
        mod.validate_artifact(missing)

    bad_guidance = deepcopy(artifact)
    bad_guidance["guided_decoding_used"] = True
    with pytest.raises(ValueError, match="guided_decoding_used"):
        mod.validate_artifact(bad_guidance)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_rate = deepcopy(artifact)
    bad_rate["false_accept_rate"] = 0.5
    with pytest.raises(ValueError, match="false_accept_rate"):
        mod.validate_artifact(bad_rate)

    bad_checksum = deepcopy(artifact)
    bad_checksum["reproducibility_checksum"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        mod.validate_artifact(bad_checksum)
