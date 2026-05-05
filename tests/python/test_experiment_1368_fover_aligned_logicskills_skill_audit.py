"""Tests for Exp 1368 FoVer-aligned LogicSkills certificate audit.

Spec: REQ-VERIFY-1368,
      SCENARIO-VERIFY-1368
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import fover_aligned_logicskills_skill_audit as mod


def _exp1366_all_pass() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "tag_first_prefix_injection_crane_positive_parse_rate_1_0",
        "certificate_parse_rate": 1.0,
        "certificate_rows": [
            {
                "case_id": "sat_unit_clause",
                "expected_state": "SAT",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": False,
                "errors": [],
            },
            {
                "case_id": "unsat_unit_conflict",
                "expected_state": "UNSAT",
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
                "case_id": "repair_missing_upper_bound",
                "expected_state": "REPAIR_HINT",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": False,
                "errors": [],
            },
        ],
    }


def test_req1368_exp1366_parse_cleared_rows_have_no_dominant_gap() -> None:
    """REQ-VERIFY-1368-2/5/6/7/8: Exp 1366 success rows audit cleanly."""

    artifact = mod.build_fover_aligned_logicskills_skill_audit_artifact(
        exp1366_artifact=_exp1366_all_pass(),
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["certificate_cases_used"] == 4
    assert artifact["symbolization_pass_rate"] == pytest.approx(1.0)
    assert artifact["countermodel_pass_rate"] == pytest.approx(1.0)
    assert artifact["validity_pass_rate"] == pytest.approx(1.0)
    assert artifact["z3_verified_case_count"] == 2
    assert artifact["fover_symbolization_alignment"] == pytest.approx(0.0)
    assert artifact["fover_validity_alignment"] == pytest.approx(0.0)
    assert artifact["dominant_skill_gap"] == "none"
    assert artifact["fover_training_data_applicable"] is False
    assert artifact["honest_verdict"] == "fover_aligned_logicskills_audit_no_skill_gap"
    assert {row["logicskills_category"] for row in artifact["classification_rows"]} == {"pass"}


def test_req1368_failure_categories_map_to_fover_alignment_fractions() -> None:
    """REQ-VERIFY-1368-3/6/7: Carnot failures map onto FoVer analog buckets."""

    exp1366 = _exp1366_all_pass()
    exp1366["certificate_rows"] = [
        {
            "case_id": "sat_unit_clause",
            "expected_state": "SAT",
            "parseable": False,
            "truthful": False,
            "unknown_preserved": False,
            "errors": ["missing_structural_tag"],
        },
        {
            "case_id": "unsat_unit_conflict",
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
            "case_id": "unknown_missing_bound",
            "expected_state": "UNKNOWN",
            "parseable": True,
            "truthful": False,
            "unknown_preserved": False,
            "errors": ["forced_sat_unsat"],
        },
    ]
    exp1366["certificate_parse_rate"] = 0.75

    artifact = mod.build_fover_aligned_logicskills_skill_audit_artifact(
        exp1366_artifact=exp1366,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["skill_failure_counts"] == {
        "countermodel": 1,
        "symbolization": 1,
        "unknown": 1,
        "validity": 1,
    }
    assert artifact["dominant_skill_gap"] == "symbolization"
    assert artifact["fover_symbolization_alignment"] == pytest.approx(0.25)
    assert artifact["fover_validity_alignment"] == pytest.approx(0.25)
    assert artifact["fover_training_data_applicable"] is True
    assert artifact["honest_verdict"] == (
        "fover_aligned_logicskills_audit_symbolization_gap_has_training_analog"
    )


def test_req1368_parse_gate_blocks_unqualified_exp1366_rows() -> None:
    """REQ-VERIFY-1368-2: below-gate Exp 1366 rows are not audited as evidence."""

    exp1366 = _exp1366_all_pass()
    exp1366["certificate_parse_rate"] = 0.5

    artifact = mod.build_fover_aligned_logicskills_skill_audit_artifact(
        exp1366_artifact=exp1366,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "complete"
    assert artifact["certificate_cases_used"] == 0
    assert artifact["terminal_blocker"] == "exp1366_parse_gate_failed:0.5_lt_0.75"
    assert artifact["classification_rows"] == []
    assert artifact["dominant_skill_gap"] == "none"
    assert artifact["fover_training_data_applicable"] is False
    assert artifact["honest_verdict"] == "exp1366_parse_gate_failed_no_skill_audit"


def test_scenario1368_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1368: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1366_path = results / "exp1366.json"
    output_path = results / "exp1368.json"
    exp1366_path.write_text(json.dumps(_exp1366_all_pass()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1366_path=exp1366_path,
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
