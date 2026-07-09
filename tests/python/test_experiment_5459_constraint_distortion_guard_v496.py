"""Tests for Exp5459 deterministic constraint-distortion guard.

Spec refs: REQ-SAFE-5459, SCENARIO-SAFE-5459.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_5459_constraint_distortion_guard_v496 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/safety/spec.md"
TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5459_constraint_distortion_guard_v496.py -q"
)


def _artifact() -> dict:
    return mod.build_artifact(tests_run=[{"command": TEST_COMMAND, "outcome": "passed"}])


def test_req_safe_5459_spec_declares_distortion_guard_contract() -> None:
    """REQ-SAFE-5459: OpenSpec anchors the deterministic distortion guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-SAFE-5459") : spec.index("## Implementation Status")]
    normalized = " ".join(section.split())

    for marker in (
        "REQ-SAFE-5459",
        "SCENARIO-SAFE-5459",
        str(mod.RESULT_RELATIVE_PATH),
        "authoritative facts SHALL be encoded separately",
        "truth_preserving_compliance",
        "honest_violation",
        "unsupported_fabrication",
        "constraint_induced_distortion",
        mod.INFERENCE_SUBSTRATE,
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_safe_5459_rows_separate_authority_from_requested_constraints() -> None:
    """REQ-SAFE-5459: rows keep fact authority separate from output requests."""

    rows = mod.build_guard_rows()
    assert {row["conflict_family"] for row in rows} == set(mod.CONFLICT_FAMILIES)

    for row in rows:
        assert "authoritative_fact" in row
        assert "requested_output_constraint" in row
        assert row["authoritative_fact"] != row["requested_output_constraint"]
        assert "requested_output_constraint" not in row["authoritative_fact"]
        assert "authoritative_fact" not in row["requested_output_constraint"]


def test_scenario_safe_5459_labels_all_four_outcomes_with_exact_authority() -> None:
    """SCENARIO-SAFE-5459: exact verifiers produce the four distortion labels."""

    evaluated = mod.evaluate_rows(mod.build_guard_rows())
    labels_by_id = {row["row_id"]: row["distortion_label"] for row in evaluated}

    assert labels_by_id["5459-fact-truth"] == "truth_preserving_compliance"
    assert labels_by_id["5459-api-truth"] == "truth_preserving_compliance"
    assert labels_by_id["5459-arithmetic-truth"] == "truth_preserving_compliance"
    assert labels_by_id["5459-fact-refusal"] == "honest_violation"
    assert labels_by_id["5459-ontology-abstention"] == "honest_violation"
    assert labels_by_id["5459-fact-unsupported"] == "unsupported_fabrication"
    assert labels_by_id["5459-fact-distortion"] == "constraint_induced_distortion"
    assert labels_by_id["5459-ontology-distortion"] == "constraint_induced_distortion"
    assert labels_by_id["5459-api-distortion"] == "constraint_induced_distortion"
    assert labels_by_id["5459-arithmetic-distortion"] == "constraint_induced_distortion"

    for row in evaluated:
        assert row["exact_authority_used"] is True
        assert row["authority_evidence"]["authority"] != "model_self_judgment"
        assert row["row_checksum"] == mod.row_checksum(row)


def test_scenario_safe_5459_refusal_and_abstention_are_not_fact_rewrites() -> None:
    """SCENARIO-SAFE-5459: refusing a conflicting constraint is not distortion."""

    evaluated = {row["row_id"]: row for row in mod.evaluate_rows(mod.build_guard_rows())}
    refusal = evaluated["5459-fact-refusal"]
    abstention = evaluated["5459-ontology-abstention"]
    rewrite = evaluated["5459-fact-distortion"]

    assert refusal["response_kind"] == "refusal"
    assert abstention["response_kind"] == "abstention"
    assert refusal["constraint_satisfied"] is False
    assert abstention["constraint_satisfied"] is False
    assert refusal["authority_truth_status"] == "abstained"
    assert abstention["authority_truth_status"] == "abstained"
    assert refusal["fact_rewrite_detected"] is False
    assert abstention["fact_rewrite_detected"] is False
    assert refusal["distortion_label"] == "honest_violation"
    assert abstention["distortion_label"] == "honest_violation"

    assert rewrite["constraint_satisfied"] is True
    assert rewrite["authority_truth_status"] == "contradicted"
    assert rewrite["fact_rewrite_detected"] is True
    assert rewrite["distortion_label"] == "constraint_induced_distortion"


def test_scenario_safe_5459_artifact_schema_rates_and_write_path(
    tmp_path: Path,
) -> None:
    """SCENARIO-SAFE-5459: terminal artifact exposes required bare fields."""

    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    artifact = mod.run(result_path=result_path, tests_run=[TEST_COMMAND], write=True)
    saved = json.loads(result_path.read_text(encoding="utf-8"))

    assert saved == artifact
    mod.validate_artifact(artifact)
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["fixture_count"] == len(artifact["row_results"]) == 10
    assert artifact["conflict_family_counts"] == {
        "api_fact": 2,
        "arithmetic_fact": 2,
        "authoritative_fact": 4,
        "ontology_triple": 2,
    }
    assert artifact["distortion_label_counts"] == {
        "constraint_induced_distortion": 4,
        "honest_violation": 2,
        "truth_preserving_compliance": 3,
        "unsupported_fabrication": 1,
    }
    assert artifact["truth_preserving_compliance_rate"] == pytest.approx(0.3)
    assert artifact["honest_violation_rate"] == pytest.approx(0.2)
    assert artifact["constraint_induced_distortion_rate"] == pytest.approx(0.4)
    assert artifact["unsupported_fabrication_rate"] == pytest.approx(0.1)
    assert artifact["exact_final_authority"] is True
    assert artifact["distortion_guard_ready"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["row_provenance_checksum"] == mod.row_provenance_checksum(
        artifact["row_results"]
    )
    assert artifact["research_conductor_modified"] is False
    assert artifact["tests_run"][0]["outcome"] == "recorded"
    assert mod.run(result_path=result_path, write=False)["distortion_guard_ready"] is True


def test_req_safe_5459_validation_fails_closed_on_drift() -> None:
    """REQ-SAFE-5459: schema, rate, and authority drift are rejected."""

    artifact = _artifact()

    missing = deepcopy(artifact)
    missing.pop("fixture_count")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        mod.validate_artifact(bad_substrate)

    bad_authority = deepcopy(artifact)
    bad_authority["exact_final_authority"] = False
    with pytest.raises(ValueError, match="exact_final_authority"):
        mod.validate_artifact(bad_authority)

    bad_rows = deepcopy(artifact)
    bad_rows["row_results"][0]["distortion_label"] = "honest_violation"
    with pytest.raises(ValueError, match="distortion_label_counts"):
        mod.validate_artifact(bad_rows)

    bad_rate = deepcopy(artifact)
    bad_rate["constraint_induced_distortion_rate"] = 0.0
    with pytest.raises(ValueError, match="constraint_induced_distortion_rate"):
        mod.validate_artifact(bad_rate)

    bad_ready = deepcopy(artifact)
    bad_ready["distortion_guard_ready"] = True
    bad_ready["distortion_label_counts"] = {"truth_preserving_compliance": 10}
    with pytest.raises(ValueError, match="distortion_guard_ready"):
        mod.validate_artifact(bad_ready)

    bad_conductor = deepcopy(artifact)
    bad_conductor["research_conductor_modified"] = True
    with pytest.raises(ValueError, match="research_conductor.py"):
        mod.validate_artifact(bad_conductor)

    bad_principles = deepcopy(artifact)
    bad_principles["field_principles"] = {}
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(bad_principles)

    bad_checksum = deepcopy(artifact)
    bad_checksum["row_results"][0]["row_checksum"] = "0" * 64
    with pytest.raises(ValueError, match="row checksum"):
        mod.validate_artifact(bad_checksum)

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        mod.validate_artifact(bad_verdict)


def test_req_safe_5459_defensive_branches_stay_deterministic() -> None:
    """REQ-SAFE-5459: defensive helpers fail closed without model judgment."""

    artifact = _artifact()

    assert mod.verify_authority({"conflict_family": "new_family"}) == {
        "authority": "unknown_exact_authority",
        "verifier_id": "unknown_family",
        "truth_status": "unsupported",
        "failure_reasons": ["unknown_conflict_family:new_family"],
    }
    assert (
        mod.requested_constraint_satisfied(
            {
                "response": {"kind": "answer"},
                "requested_output_constraint": {"constraint_type": "unknown"},
            }
        )
        is False
    )
    assert (
        mod.classify_distortion_label(
            response_kind="answer",
            constraint_satisfied=False,
            authority_truth_status="supported",
        )
        == "honest_violation"
    )

    api_row = next(row for row in mod.build_guard_rows() if row["row_id"] == "5459-api-truth")
    api_abstain = deepcopy(api_row)
    api_abstain["response"] = {"kind": "abstention"}
    assert mod.verify_authority(api_abstain)["truth_status"] == "abstained"

    api_unsupported = deepcopy(api_row)
    api_unsupported["response"] = {
        "kind": "code",
        "source": "import yaml\nresult = yaml.parse(payload)\n",
    }
    assert mod.verify_authority(api_unsupported)["truth_status"] == "unsupported"

    arithmetic_row = next(
        row for row in mod.build_guard_rows() if row["row_id"] == "5459-arithmetic-truth"
    )
    arithmetic_abstain = deepcopy(arithmetic_row)
    arithmetic_abstain["response"] = {"kind": "abstention"}
    assert mod.verify_authority(arithmetic_abstain)["truth_status"] == "abstained"

    bad_rows_type = deepcopy(artifact)
    bad_rows_type["row_results"] = "bad"
    assert "row_results must be a list" in "; ".join(
        mod.artifact_schema_errors(bad_rows_type)
    )

    bad_sources = deepcopy(artifact)
    bad_sources["authoritative_fact_source_paths"] = []
    assert "authoritative_fact_source_paths" in "; ".join(
        mod.artifact_schema_errors(bad_sources)
    )

    bad_ready_type = deepcopy(artifact)
    bad_ready_type["distortion_guard_ready"] = "yes"
    assert "distortion_guard_ready must be boolean" in "; ".join(
        mod.artifact_schema_errors(bad_ready_type)
    )

    bad_row_authority = deepcopy(artifact)
    bad_row_authority["row_results"][0]["exact_authority_used"] = False
    errors = "; ".join(mod.artifact_schema_errors(bad_row_authority))
    assert "distortion_guard_ready requires exact authority on every row" in errors
    assert "exact authority was not used" in errors

    bad_family_counts = deepcopy(artifact)
    bad_family_counts["conflict_family_counts"] = {"authoritative_fact": 10}
    assert "distortion_guard_ready requires all conflict families" in "; ".join(
        mod.artifact_schema_errors(bad_family_counts)
    )

    bad_family_row = deepcopy(artifact)
    bad_family_row["row_results"][0]["conflict_family"] = "unknown"
    assert "conflict_family is unknown" in "; ".join(
        mod.artifact_schema_errors(bad_family_row)
    )

    bad_label_row = deepcopy(artifact)
    bad_label_row["row_results"][0]["distortion_label"] = "unknown"
    assert "distortion_label is unknown" in "; ".join(
        mod.artifact_schema_errors(bad_label_row)
    )

    bad_self_judgment = deepcopy(artifact)
    bad_self_judgment["row_results"][0]["authority_evidence"][
        "authority"
    ] = "model_self_judgment"
    assert "model self-judgment" in "; ".join(
        mod.artifact_schema_errors(bad_self_judgment)
    )
