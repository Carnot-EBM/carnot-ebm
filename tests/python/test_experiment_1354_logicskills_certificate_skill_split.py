"""Tests for Exp 1354 LogicSkills certificate skill split.

Spec: REQ-VERIFY-1354,
      SCENARIO-VERIFY-1354
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import logicskills_certificate_skill_split as mod


def _exp1353_missing_tags() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "sota_triggered_certificate_v7_measured",
        "certificate_rows": [
            {
                "case_id": "sat_unit_clause",
                "expected_state": "SAT",
                "parseable": False,
                "truthful": False,
                "unknown_preserved": False,
                "trigger_token_hit": False,
                "errors": ["missing_structural_tag"],
            },
            {
                "case_id": "unsat_unit_conflict",
                "expected_state": "UNSAT",
                "parseable": False,
                "truthful": False,
                "unknown_preserved": False,
                "trigger_token_hit": False,
                "errors": ["missing_structural_tag"],
            },
            {
                "case_id": "unknown_missing_bound",
                "expected_state": "UNKNOWN",
                "parseable": False,
                "truthful": False,
                "unknown_preserved": False,
                "trigger_token_hit": False,
                "errors": ["missing_structural_tag"],
            },
            {
                "case_id": "repair_missing_upper_bound",
                "expected_state": "REPAIR_HINT",
                "parseable": False,
                "truthful": False,
                "unknown_preserved": False,
                "trigger_token_hit": False,
                "errors": ["missing_structural_tag"],
            },
        ],
    }


def test_req1354_missing_exp1353_tags_become_symbolization_gap() -> None:
    """REQ-VERIFY-1354: rejected missing-tag cases are symbolization failures."""
    artifact = mod.build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact=_exp1353_missing_tags(),
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["certificate_cases_used"] == 4
    assert artifact["symbolization_pass_rate"] == pytest.approx(0.0)
    assert artifact["countermodel_pass_rate"] == pytest.approx(0.0)
    assert artifact["validity_pass_rate"] == pytest.approx(0.0)
    assert artifact["z3_verified_case_count"] == 2
    assert artifact["dominant_skill_gap"] == "symbolization"
    assert artifact["skill_split_claim_allowed"] is True
    assert artifact["honest_verdict"] == (
        "logic_skill_split_supported_symbolization_dominates_exp1353"
    )
    assert {row["classification"] for row in artifact["classification_rows"]} == {"rejected"}
    assert artifact["skill_gap_counts"] == {
        "countermodel": 0,
        "symbolization": 4,
        "unknown_preservation": 0,
        "validity_assessment": 0,
    }


def test_req1354_classifies_truth_unknown_validity_and_countermodel_evidence() -> None:
    """REQ-VERIFY-1354: row evidence maps to distinct LogicSkills labels."""
    rows = [
        {
            "case_id": "sat_unit_clause",
            "expected_state": "SAT",
            "parseable": True,
            "truthful": True,
            "unknown_preserved": False,
            "errors": [],
        },
        {
            "case_id": "unknown_missing_bound",
            "expected_state": "UNKNOWN",
            "parseable": True,
            "truthful": True,
            "unknown_preserved": True,
            "errors": [],
        },
        {
            "case_id": "invalid_truth_value",
            "expected_state": "UNSAT",
            "parseable": True,
            "truthful": False,
            "unknown_preserved": False,
            "errors": ["final_answer_mismatch"],
        },
        {
            "case_id": "bad_witness",
            "expected_state": "SAT",
            "parseable": True,
            "truthful": False,
            "unknown_preserved": False,
            "errors": ["countermodel_witness_invalid"],
        },
        {
            "case_id": "unknown_collapsed",
            "expected_state": "UNKNOWN",
            "parseable": True,
            "truthful": False,
            "unknown_preserved": False,
            "errors": ["forced_sat_unsat"],
        },
    ]
    artifact = mod.build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact={"status": "complete", "certificate_rows": rows},
        run_date="20260505",
        project_root="/repo",
    )

    by_case = {row["case_id"]: row for row in artifact["classification_rows"]}
    assert by_case["sat_unit_clause"]["classification"] == "semantically truth-preserving"
    assert by_case["unknown_missing_bound"]["classification"] == "UNKNOWN-preserving"
    assert by_case["invalid_truth_value"]["skill_gap"] == "validity_assessment"
    assert by_case["bad_witness"]["skill_gap"] == "countermodel"
    assert by_case["unknown_collapsed"]["skill_gap"] == "unknown_preservation"
    assert artifact["symbolization_pass_rate"] == pytest.approx(1.0)
    assert artifact["countermodel_pass_rate"] == pytest.approx(0.5)
    assert artifact["validity_pass_rate"] == pytest.approx(0.4)
    assert artifact["dominant_skill_gap"] == "countermodel"
    assert artifact["skill_split_claim_allowed"] is True


def test_scenario1354_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1354: runner persists bootstrap and terminal artifact."""
    results = tmp_path / "results"
    results.mkdir()
    exp1353_path = results / "exp1353.json"
    output_path = results / "exp1354.json"
    exp1353_path.write_text(json.dumps(_exp1353_missing_tags()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1353_path=exp1353_path,
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["run_date"] == "20260505"


def test_req1354_empty_or_malformed_rows_do_not_allow_claim() -> None:
    """REQ-VERIFY-1354: no evidenced cases means no skill-split claim."""
    artifact = mod.build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact={"status": "complete", "certificate_rows": ["bad-row"]},
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["certificate_cases_used"] == 0
    assert artifact["skill_split_claim_allowed"] is False
    assert artifact["dominant_skill_gap"] == "none"
    assert artifact["honest_verdict"] == "no_exp1353_certificate_cases_available"
    assert artifact["z3_verified_case_count"] == 0

    non_list = mod.build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact={"status": "complete", "certificate_rows": "bad-schema"},
        run_date="20260505",
        project_root="/repo",
    )
    assert non_list["certificate_cases_used"] == 0


def test_req1354_all_evidenced_rows_can_have_no_measured_gap() -> None:
    """REQ-VERIFY-1354: successful classified rows keep the claim bounded."""
    artifact = mod.build_logicskills_certificate_skill_split_artifact(
        exp1353_artifact={
            "status": "complete",
            "certificate_rows": [
                {
                    "case_id": "unknown_missing_bound",
                    "expected_state": "UNKNOWN",
                    "parseable": True,
                    "truthful": True,
                    "unknown_preserved": True,
                    "errors": [],
                }
            ],
        },
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["dominant_skill_gap"] == "none"
    assert artifact["skill_split_claim_allowed"] is True
    assert artifact["honest_verdict"] == "logic_skill_split_supported_no_measured_gap"
