"""Tests for Exp5412 bounded KAN/KANDy active-constraint certificate.

Spec refs: REQ-KAN-5412, SCENARIO-KAN-5412.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5412_kan_active_constraint_certificate_v492 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/kan/spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5412_kan_active_constraint_certificate_v492.py "
    "-q --no-cov"
)


def test_req_kan_5412_spec_declares_active_constraint_certificate_contract() -> None:
    """REQ-KAN-5412: OpenSpec anchors the bounded active-constraint certificate."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## REQ-KAN-5412") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-KAN-5412",
        "SCENARIO-KAN-5412",
        str(exp.RESULT_RELATIVE_PATH),
        "Exp 5406 active-constraint",
        "stale partial active-set hints",
        "adversarial contradictory active-set hints",
        "verifier_ensemble_against_cached_candidates",
        "`broad_kan_verification_claim` MUST be false",
        "`scripts/research_conductor.py`",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert f"{field}`: {principle}" in normalized


def test_req_kan_5412_replays_exp5406_active_constraint_rows() -> None:
    """REQ-KAN-5412: Exp5406 row records are the bounded evidence source."""

    rows = exp.load_active_constraint_rows()
    modes = {row["hint_mode"] for row in rows}
    certificate = exp.build_certificate_records(rows)

    assert len(rows) == 16
    assert modes == {"no_hint", "stale_hint", "adversarial_hint", "candidate_hint"}
    assert len(certificate["counterexample_regions"]) == 8
    assert len(certificate["false_property_checks"]) == 8
    assert len(certificate["true_property_checks"]) == 3
    assert {row["source_experiment"] for row in certificate["counterexample_regions"]} == {
        "experiment_5406_active_constraint_warmstart_guidance_v492"
    }
    assert {row["hint_mode"] for row in certificate["counterexample_regions"]} == {
        "stale_hint",
        "adversarial_hint",
    }


def test_scenario_kan_5412_rejects_false_properties_and_preserves_controls() -> None:
    """SCENARIO-KAN-5412: false hint-routing properties produce regions."""

    diagnostic = exp.evaluate_certificate()

    assert diagnostic["certificate_family"] == exp.CERTIFICATE_FAMILY
    assert diagnostic["false_property_rejection_rate"] == pytest.approx(1.0)
    assert diagnostic["true_property_preservation_rate"] == pytest.approx(1.0)
    assert diagnostic["counterexample_region_count"] == 8
    assert diagnostic["deterministic_verifier_passed"] is True
    assert diagnostic["broad_kan_verification_claim"] is False

    for check in diagnostic["false_property_checks"]:
        assert check["rejected"] is True
        assert check["false_claimed_route"] == "accept_candidate_hint"
        assert check["actual_route"] in {
            "reject_to_solver_fallback",
            "overwrite_with_solver_active_set",
        }
        assert check["deterministic_check_passed"] is True

    for check in diagnostic["true_property_checks"]:
        assert check["preserved"] is True
        assert check["deterministic_check_passed"] is True

    for region in diagnostic["counterexample_regions"]:
        assert region["bounded_fixture_only"] is True
        assert region["rejects_false_property"] is True
        assert region["deterministic_check_passed"] is True
        assert region["feature_bounds"]

    assert any("bounded active-constraint" in item for item in diagnostic["claim_limits"])
    assert any("no broad KAN verification claim" in item for item in diagnostic["claim_limits"])


def test_req_kan_5412_artifact_schema_and_run_write(tmp_path: Path) -> None:
    """REQ-KAN-5412: run() writes the required bounded certificate artifact."""

    tests_run = [
        {"command": TEST_COMMAND, "outcome": "passed"},
        {
            "command": (
                ".venv/bin/coverage run "
                "--include=python/carnot/experiment_5412_kan_active_constraint_certificate_v492.py "
                "-m pytest tests/python/test_experiment_5412_kan_active_constraint_certificate_v492.py "
                "-q --no-cov -n 0"
            ),
            "outcome": "passed",
        },
        {
            "command": (
                ".venv/bin/coverage report "
                "--include=python/carnot/experiment_5412_kan_active_constraint_certificate_v492.py "
                "--fail-under=100"
            ),
            "outcome": "passed",
        },
    ]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["field_principles"] == exp.FIELD_PRINCIPLES
    assert artifact["certificate_family"] == exp.CERTIFICATE_FAMILY
    assert artifact["counterexample_region_count"] == 8
    assert artifact["false_property_rejection_rate"] == pytest.approx(1.0)
    assert artifact["true_property_preservation_rate"] == pytest.approx(1.0)
    assert artifact["certificate_size_bytes"] == exp.certificate_size_bytes(
        artifact["certificate_records"]
    )
    assert artifact["broad_kan_verification_claim"] is False
    assert artifact["deterministic_verifier_passed"] is True
    assert artifact["kan_active_constraint_certificate_ready"] is True
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["tests_run"] == tests_run
    assert artifact["spec_refs"] == list(exp.SPEC_REFS)
    assert artifact["reproducibility_checksum"].startswith("sha256:")
    exp.validate_artifact(artifact)


def test_req_kan_5412_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-KAN-5412: checked-in JSON is stable under deterministic replay."""

    checked_in = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(tests_run=checked_in["tests_run"])

    assert checked_in == replay
    assert checked_in["broad_kan_verification_claim"] is False
    assert checked_in["kan_active_constraint_certificate_ready"] is True
    exp.validate_artifact(checked_in)


def test_req_kan_5412_validation_rejects_broad_or_unchecked_claims() -> None:
    """REQ-KAN-5412: validation fails closed on broad KAN claim drift."""

    artifact = exp.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])

    blocked = exp.build_artifact(tests_run=())
    assert blocked["kan_active_constraint_certificate_ready"] is False
    assert blocked["honest_verdict"].startswith("blocked:")
    exp.validate_artifact(blocked)

    missing = deepcopy(artifact)
    missing.pop("certificate_family")
    with pytest.raises(ValueError, match="certificate_family"):
        exp.validate_artifact(missing)

    broad_flag = deepcopy(artifact)
    broad_flag["broad_kan_verification_claim"] = True
    with pytest.raises(ValueError, match="broad_kan_verification_claim"):
        exp.validate_artifact(broad_flag)

    broad_family = deepcopy(artifact)
    broad_family["certificate_family"] = "broad_kan_verification"
    with pytest.raises(ValueError, match="certificate_family"):
        exp.validate_artifact(broad_family)

    broad_verdict = deepcopy(artifact)
    broad_verdict["honest_verdict"] = "complete: broad KAN verification proved"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(broad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    unchecked = deepcopy(artifact)
    unchecked["deterministic_verifier_passed"] = False
    with pytest.raises(ValueError, match="deterministic_verifier_passed"):
        exp.validate_artifact(unchecked)

    bad_size = deepcopy(artifact)
    bad_size["certificate_size_bytes"] = 0
    with pytest.raises(ValueError, match="certificate_size_bytes"):
        exp.validate_artifact(bad_size)


def test_req_kan_5412_readiness_blockers_are_explicit() -> None:
    """REQ-KAN-5412: blocked certificates name every failed gate."""

    diagnostic = exp.evaluate_certificate()
    diagnostic["broad_kan_verification_claim"] = True
    diagnostic["false_property_rejection_rate"] = 0.5
    diagnostic["true_property_preservation_rate"] = 0.5
    diagnostic["deterministic_verifier_passed"] = False
    diagnostic["counterexample_region_count"] = 0

    assert exp._readiness_blockers(diagnostic, 0, ()) == [
        "broad_kan_claim",
        "false_properties_not_rejected",
        "true_properties_not_preserved",
        "deterministic_verifier_failed",
        "no_counterexample_regions",
        "empty_certificate",
        "tests_not_recorded",
    ]
