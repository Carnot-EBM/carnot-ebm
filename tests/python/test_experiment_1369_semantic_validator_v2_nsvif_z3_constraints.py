"""Tests for Exp 1369 semantic validator v2.

Spec: REQ-VERIFY-1369,
      SCENARIO-VERIFY-1369
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import semantic_validator_v2_nsvif_z3_constraints as mod


def _exp1366_all_pass() -> dict[str, Any]:
    return {
        "status": "complete",
        "honest_verdict": "tag_first_prefix_injection_crane_positive_parse_rate_1_0",
        "certificate_parse_rate": 1.0,
        "certificate_rows": [
            {
                "case_id": "sat_unit_clause",
                "expected_state": "SAT",
                "dispatched_state": "SAT",
                "tag_state": "SAT",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": False,
                "errors": [],
            },
            {
                "case_id": "unsat_unit_conflict",
                "expected_state": "UNSAT",
                "dispatched_state": "UNSAT",
                "tag_state": "UNSAT",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": False,
                "errors": [],
            },
            {
                "case_id": "unknown_missing_bound",
                "expected_state": "UNKNOWN",
                "dispatched_state": "UNKNOWN",
                "tag_state": "UNKNOWN",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": True,
                "errors": [],
            },
            {
                "case_id": "repair_missing_upper_bound",
                "expected_state": "REPAIR_HINT",
                "dispatched_state": "REPAIR_HINT",
                "tag_state": "REPAIR_HINT",
                "parseable": True,
                "truthful": True,
                "unknown_preserved": False,
                "errors": [],
            },
        ],
        "generation_rows": [
            {"case_id": "unknown_missing_bound", "certificate_body": "UNKNOWN"},
            {
                "case_id": "repair_missing_upper_bound",
                "certificate_body": "REPAIR_HINT: add bound.",
            },
        ],
    }


def test_req1369_parse_cleared_rows_run_z3_and_logitext_constraints() -> None:
    """REQ-VERIFY-1369-2/3/4/5/6/7/8/9: Parse-cleared rows validate."""

    artifact = mod.build_semantic_validator_v2_artifact(
        exp1366_artifact=_exp1366_all_pass(),
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["parsed_certificate_cases"] == 4
    assert artifact["fully_formal_claim_count"] == 2
    assert artifact["nltc_claim_count"] == 2
    assert artifact["z3_constraint_pass_rate"] == pytest.approx(1.0)
    assert artifact["unknown_preservation_rate"] == pytest.approx(1.0)
    assert artifact["smt_text_constraint_pass_rate"] == pytest.approx(1.0)
    assert artifact["validator_execution_pass_rate"] == pytest.approx(1.0)
    assert artifact["coverage_delta_over_fol_only"] == pytest.approx(0.5)
    assert artifact["semantic_validator_claim_allowed"] is True
    assert artifact["honest_verdict"] == "semantic_validator_v2_complete_unknown_preserved"
    assert {row["claim_route"] for row in artifact["semantic_validator_rows"]} == {
        "nltc_partial_smt",
        "z3_fully_formal",
    }


def test_req1369_unknown_collapse_blocks_semantic_validator_claim() -> None:
    """REQ-VERIFY-1369-6/8: UNKNOWN collapse is preserved as a failed claim gate."""

    exp1366 = _exp1366_all_pass()
    exp1366["certificate_rows"][2]["dispatched_state"] = "SAT"
    exp1366["certificate_rows"][2]["tag_state"] = "SAT"
    exp1366["certificate_rows"][2]["truthful"] = False
    exp1366["certificate_rows"][2]["unknown_preserved"] = False

    artifact = mod.build_semantic_validator_v2_artifact(
        exp1366_artifact=exp1366,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "complete"
    assert artifact["parsed_certificate_cases"] == 4
    assert artifact["unknown_preservation_rate"] == pytest.approx(0.0)
    assert artifact["validator_execution_pass_rate"] == pytest.approx(0.75)
    assert artifact["semantic_validator_claim_allowed"] is False
    assert artifact["honest_verdict"] == "semantic_validator_v2_ran_unknown_collapsed"


def test_req1369_parse_gate_blocks_unqualified_exp1366_rows() -> None:
    """REQ-VERIFY-1369-2: Below-gate Exp 1366 rows are not semantic evidence."""

    exp1366 = _exp1366_all_pass()
    exp1366["certificate_parse_rate"] = 0.5

    artifact = mod.build_semantic_validator_v2_artifact(
        exp1366_artifact=exp1366,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "blocked"
    assert artifact["parsed_certificate_cases"] == 0
    assert artifact["fully_formal_claim_count"] == 0
    assert artifact["nltc_claim_count"] == 0
    assert artifact["validator_execution_pass_rate"] == pytest.approx(0.0)
    assert artifact["coverage_delta_over_fol_only"] == pytest.approx(0.0)
    assert artifact["semantic_validator_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_exp1366_parse_gate_below_0_75"


def test_scenario1369_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1369: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1366_path = results / "exp1366.json"
    output_path = results / "exp1369.json"
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
