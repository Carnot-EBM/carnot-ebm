"""Tests for Exp 1370 VERGE MCS repair localization.

Spec: REQ-VERIFY-1370,
      SCENARIO-VERIFY-1370
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import verge_mcs_repair_localization_v2 as mod


def _exp1369_semantic_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "validator_execution_pass_rate": 1.0,
        "honest_verdict": "semantic_validator_v2_complete_unknown_preserved",
        "semantic_validator_rows": [
            {
                "case_id": "sat_unit_clause",
                "claim_route": "z3_fully_formal",
                "constraint_evaluated": True,
                "constraint_passed": True,
                "expected_state": "SAT",
                "certificate_state": "SAT",
                "semantic_result": "SAT",
                "nsvif_encoding": "x1",
            },
            {
                "case_id": "unsat_unit_conflict",
                "claim_route": "z3_fully_formal",
                "constraint_evaluated": True,
                "constraint_passed": True,
                "expected_state": "UNSAT",
                "certificate_state": "UNSAT",
                "semantic_result": "UNSAT",
                "nsvif_encoding": "And(x1, Not(x1))",
            },
            {
                "case_id": "unknown_missing_bound",
                "claim_route": "nltc_partial_smt",
                "constraint_evaluated": True,
                "constraint_passed": True,
                "expected_state": "UNKNOWN",
                "certificate_state": "UNKNOWN",
                "semantic_result": "UNKNOWN",
                "text_constraints": ["missing_capacity_bound_requires_unknown"],
            },
            {
                "case_id": "repair_missing_upper_bound",
                "claim_route": "nltc_partial_smt",
                "constraint_evaluated": True,
                "constraint_passed": True,
                "expected_state": "REPAIR_HINT",
                "certificate_state": "REPAIR_HINT",
                "semantic_result": "REPAIR_HINT",
                "text_constraints": ["missing_upper_bound_requires_repair_hint"],
            },
        ],
    }


def test_req1370_exp1369_non_sat_semantic_rows_produce_localized_hints() -> None:
    """REQ-VERIFY-1370-3/4/5/6/7/8/9: MCS replay localizes semantic failures."""

    artifact = mod.build_verge_mcs_repair_localization_artifact(
        exp1369_artifact=_exp1369_semantic_artifact(),
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["semantic_cases_used"] == 3
    assert artifact["repair_hint_count"] == 3
    assert artifact["mcs_localization_rate"] == pytest.approx(1.0)
    assert artifact["repair_hint_precision"] == pytest.approx(1.0)
    assert artifact["semantic_equivalence_pass_rate"] == pytest.approx(1.0)
    assert artifact["iteration_count_to_accept"] == pytest.approx(1.0)
    assert artifact["accepted_violation_delta"] == -2
    assert artifact["repair_claim_allowed"] is True
    assert artifact["honest_verdict"] == "verge_mcs_repair_localization_complete_claim_allowed"

    rows = {row["case_id"]: row for row in artifact["repair_localization_rows"]}
    assert rows["unsat_unit_conflict"]["mcs_candidates"] == [
        ["cnf_clause_x1"],
        ["cnf_clause_not_x1"],
    ]
    assert rows["unsat_unit_conflict"]["accepted"] is False
    assert rows["unknown_missing_bound"]["localized_constraint"] == "capacity_bound_B"
    assert rows["unknown_missing_bound"]["accepted"] is True
    assert rows["repair_missing_upper_bound"]["localized_constraint"] == "upper_bound_premise"
    assert rows["repair_missing_upper_bound"]["accepted"] is True


def test_req1370_execution_gate_blocks_unqualified_exp1369_artifact() -> None:
    """REQ-VERIFY-1370-2: Below-gate semantic validator rows are not repair evidence."""

    exp1369 = _exp1369_semantic_artifact()
    exp1369["validator_execution_pass_rate"] = 0.25

    artifact = mod.build_verge_mcs_repair_localization_artifact(
        exp1369_artifact=exp1369,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "blocked"
    assert artifact["semantic_cases_used"] == 0
    assert artifact["mcs_localization_rate"] == pytest.approx(0.0)
    assert artifact["repair_hint_count"] == 0
    assert artifact["repair_hint_precision"] == pytest.approx(0.0)
    assert artifact["semantic_equivalence_pass_rate"] == pytest.approx(0.0)
    assert artifact["iteration_count_to_accept"] == 0
    assert artifact["accepted_violation_delta"] == 0
    assert artifact["repair_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_exp1369_validator_execution_pass_rate_below_0_5"


def test_scenario1370_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1370: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1369_path = results / "exp1369.json"
    output_path = results / "exp1370.json"
    exp1369_path.write_text(json.dumps(_exp1369_semantic_artifact()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1369_path=exp1369_path,
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
