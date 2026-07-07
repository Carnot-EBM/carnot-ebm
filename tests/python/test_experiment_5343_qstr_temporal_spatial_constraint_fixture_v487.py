"""Tests for Exp5343 deterministic QSTR temporal/spatial fixture.

Spec refs: REQ-VERIFY-5343, SCENARIO-VERIFY-5343.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from carnot import experiment_5343_qstr_temporal_spatial_constraint_fixture_v487 as exp


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"
RESULT_PATH = REPO / exp.RESULT_RELATIVE_PATH


def _relation_rows(evaluation: exp.JsonDict) -> dict[str, exp.JsonDict]:
    return {row["case_id"]: row for row in evaluation["relation_results"]}


def _composition_rows(evaluation: exp.JsonDict) -> dict[str, exp.JsonDict]:
    return {row["case_id"]: row for row in evaluation["composition_results"]}


def test_req_verify_5343_spec_declares_qstr_contract() -> None:
    """REQ-VERIFY-5343: OpenSpec anchors the QSTR fixture and artifact fields."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-5343") :]
    normalized_section = " ".join(section.split())

    for marker in (
        "REQ-VERIFY-5343",
        "SCENARIO-VERIFY-5343",
        str(exp.RESULT_RELATIVE_PATH),
        exp.INFERENCE_SUBSTRATE,
        "before, overlaps, during, meets, contradiction, and ambiguous",
        "disconnected, overlap, containment, cardinal east-of, and contradiction",
        "finite exact interval enumeration",
        "solver_authoritative",
        "false_accept_count",
        "failure_localization_rate",
        "qstr_fixture_ready",
        "scripts/research_conductor.py",
    ):
        assert marker in section

    for field, principle in exp.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized_section


def test_req_verify_5343_fixture_contains_requested_case_families() -> None:
    """REQ-VERIFY-5343: fixture has the requested temporal and spatial families."""

    evaluation = exp.evaluate_fixture(exp.build_fixture())

    assert evaluation["calculus_count"] == 2
    assert evaluation["temporal_case_type_counts"] == {
        "before": 1,
        "overlaps": 1,
        "during": 1,
        "meets": 1,
        "contradiction": 1,
        "ambiguous": 1,
    }
    assert evaluation["spatial_case_type_counts"] == {
        "disconnected": 1,
        "overlap": 1,
        "containment": 1,
        "cardinal_direction": 1,
        "contradiction": 1,
    }
    assert evaluation["composition_case_count"] == 6
    assert evaluation["contradiction_case_count"] == 2
    assert evaluation["solver_authoritative"] is True


def test_scenario_verify_5343_accepts_true_relations_and_rejects_false_properties() -> None:
    """SCENARIO-VERIFY-5343: true properties accept and false properties reject."""

    evaluation = exp.evaluate_fixture(exp.build_fixture())
    rows = _relation_rows(evaluation)

    for case_id, relation in (
        ("t-before", "before"),
        ("t-overlaps", "overlaps"),
        ("t-during", "during"),
        ("t-meets", "meets"),
        ("s-disconnected", "disconnected"),
        ("s-overlap", "overlap"),
        ("s-containment", "contains"),
        ("s-east-of", "east_of"),
    ):
        row = rows[case_id]
        assert row["accepted"] is True
        assert row["actual_label"] == "satisfiable"
        assert row["actual_relation"] == relation
        assert row["violation_ids"] == []

    temporal_false = rows["t-contradiction-before-vs-meets"]
    assert temporal_false["accepted"] is False
    assert temporal_false["actual_label"] == "unsatisfiable"
    assert temporal_false["claimed_relation"] == "before"
    assert temporal_false["actual_relation"] == "meets"
    assert temporal_false["violation_ids"] == [
        "t-contradiction-before-vs-meets:claim-before"
    ]

    spatial_false = rows["s-contradiction-contains-vs-disconnected"]
    assert spatial_false["accepted"] is False
    assert spatial_false["claimed_relation"] == "contains"
    assert "disconnected" in spatial_false["actual_relations"]
    assert spatial_false["violation_ids"] == [
        "s-contradiction-contains-vs-disconnected:claim-contains"
    ]

    assert evaluation["false_accept_count"] == 0
    assert evaluation["deterministic_checks_passed"] is True


def test_scenario_verify_5343_composition_and_converse_are_authoritative() -> None:
    """SCENARIO-VERIFY-5343: composition and converse use exact typed rules."""

    evaluation = exp.evaluate_fixture(exp.build_fixture())
    relation_rows = _relation_rows(evaluation)
    composition_rows = _composition_rows(evaluation)

    assert relation_rows["t-before"]["converse_relation"] == "after"
    assert relation_rows["t-meets"]["converse_relation"] == "met_by"
    assert relation_rows["t-overlaps"]["converse_relation"] == "overlapped_by"
    assert relation_rows["t-during"]["converse_relation"] == "contains"
    assert relation_rows["s-containment"]["converse_relation"] == "inside"
    assert relation_rows["s-east-of"]["converse_relation"] == "west_of"
    assert all(row["converse_valid"] is True for row in relation_rows.values())

    assert exp.compose_relations("temporal", "before", "meets") == ("before",)
    assert composition_rows["tc-before-meets"]["actual_relation_ac"] == "before"
    assert composition_rows["tc-before-meets"]["accepted"] is True
    assert (
        composition_rows["tc-before-meets"]["composition_source"]
        == exp.TEMPORAL_COMPOSITION_SOURCE
    )

    assert composition_rows["sc-contains-contains"]["actual_relation_ac"] == "contains"
    assert composition_rows["sc-east-east"]["actual_relation_ac"] == "east_of"
    assert (
        composition_rows["sc-east-east"]["composition_source"]
        == exp.SPATIAL_COMPOSITION_SOURCE
    )
    assert all(row["accepted"] is True for row in composition_rows.values())


def test_scenario_verify_5343_accepts_declared_ambiguity_without_overclaiming() -> None:
    """SCENARIO-VERIFY-5343: ambiguous alternatives are accepted only if exact."""

    evaluation = exp.evaluate_fixture(exp.build_fixture())
    rows = _relation_rows(evaluation)
    composition_rows = _composition_rows(evaluation)

    ambiguous = rows["t-ambiguous-before-or-meets"]
    assert ambiguous["ambiguous"] is True
    assert ambiguous["allowed_relations"] == ["before", "meets"]
    assert ambiguous["actual_relation"] == "meets"
    assert ambiguous["accepted"] is True
    assert ambiguous["actual_label"] == "satisfiable"
    assert ambiguous["violation_ids"] == []

    composition = composition_rows["tc-overlaps-during"]
    assert composition["accepted"] is True
    assert composition["actual_relation_ac"] == "overlaps"
    assert "overlaps" in composition["possible_composed_relations"]
    assert len(composition["possible_composed_relations"]) > 1


def test_req_verify_5343_failure_localization_is_exact() -> None:
    """REQ-VERIFY-5343: unsatisfiable rows localize the failing relation claim."""

    fixture = exp.build_fixture()
    evaluation = exp.evaluate_fixture(fixture)
    invalid_rows = [
        row for row in evaluation["relation_results"] if row["expected_satisfiable"] is False
    ]

    assert evaluation["failure_localization_rate"] == pytest.approx(1.0)
    assert len(invalid_rows) == 2
    assert all(row["localized_failure"] is True for row in invalid_rows)
    assert all(row["violation_ids"] == row["expected_failure_ids"] for row in invalid_rows)

    target = next(
        case
        for case in fixture.relation_cases
        if case.case_id == "t-contradiction-before-vs-meets"
    )
    wrong_expected = replace(target, expected_failure_ids=("wrong-location",))
    mutated = replace(
        fixture,
        relation_cases=tuple(
            wrong_expected if case.case_id == target.case_id else case
            for case in fixture.relation_cases
        ),
    )
    mutated_evaluation = exp.evaluate_fixture(mutated)

    assert mutated_evaluation["failure_localization_rate"] == pytest.approx(0.5)
    assert mutated_evaluation["deterministic_checks_passed"] is False


def test_req_verify_5343_run_writes_required_artifact_schema(tmp_path: Path) -> None:
    """REQ-VERIFY-5343: run() writes principle fields and bare downstream gates."""

    tests_run = [{"command": "unit qstr fixture", "outcome": "passed"}]
    result_path = tmp_path / exp.RESULT_RELATIVE_PATH
    artifact = exp.run(result_path=result_path, tests_run=tests_run)

    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    exp.validate_artifact(artifact)
    assert artifact["experiment_id"]["value"] == exp.EXPERIMENT_NAME
    assert artifact["milestone"]["value"] == exp.MILESTONE
    assert artifact["status"]["value"] == "qstr_fixture_ready"
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert artifact["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert artifact["calculus_count"] == 2
    assert artifact["composition_case_count"] == 6
    assert artifact["contradiction_case_count"] == 2
    assert artifact["solver_authoritative"] is True
    assert artifact["false_accept_count"] == 0
    assert artifact["failure_localization_rate"] == pytest.approx(1.0)
    assert artifact["qstr_fixture_ready"] is True
    assert artifact["tests_run"]["value"] == tests_run


def test_req_verify_5343_repository_artifact_matches_deterministic_replay() -> None:
    """REQ-VERIFY-5343: checked-in artifact is stable under deterministic replay."""

    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    replay = exp.build_artifact(
        exp.build_fixture(),
        tests_run=result["tests_run"]["value"],
    )

    assert result == replay
    assert result["honest_verdict"]["value"].startswith("complete:")
    assert result["inference_substrate"]["value"] == exp.INFERENCE_SUBSTRATE
    assert result["solver_authoritative"] is True
    assert result["false_accept_count"] == 0
    assert result["qstr_fixture_ready"] is True
    exp.validate_artifact(result)


def test_req_verify_5343_validation_rejects_schema_drift() -> None:
    """REQ-VERIFY-5343: artifact validation rejects wrapped and bare field drift."""

    artifact = exp.build_artifact(
        exp.build_fixture(),
        tests_run=[{"command": "unit qstr schema", "outcome": "passed"}],
    )

    bad_verdict = deepcopy(artifact)
    bad_verdict["honest_verdict"]["value"] = "done"
    with pytest.raises(ValueError, match="honest_verdict"):
        exp.validate_artifact(bad_verdict)

    bad_substrate = deepcopy(artifact)
    bad_substrate["inference_substrate"]["value"] = "live_llm_inference"
    with pytest.raises(ValueError, match="inference_substrate"):
        exp.validate_artifact(bad_substrate)

    bad_count = deepcopy(artifact)
    bad_count["calculus_count"] = True
    with pytest.raises(ValueError, match="calculus_count"):
        exp.validate_artifact(bad_count)

    bad_authority = deepcopy(artifact)
    bad_authority["solver_authoritative"] = False
    with pytest.raises(ValueError, match="solver_authoritative"):
        exp.validate_artifact(bad_authority)

    bad_localization = deepcopy(artifact)
    bad_localization["failure_localization_rate"] = "1.0"
    with pytest.raises(ValueError, match="failure_localization_rate"):
        exp.validate_artifact(bad_localization)

    bad_ready = deepcopy(artifact)
    bad_ready["qstr_fixture_ready"] = {"value": True}
    with pytest.raises(ValueError, match="qstr_fixture_ready"):
        exp.validate_artifact(bad_ready)

    bad_tests = deepcopy(artifact)
    bad_tests["tests_run"] = [{"command": "lost principle", "outcome": "passed"}]
    with pytest.raises(ValueError, match="tests_run"):
        exp.validate_artifact(bad_tests)
