"""Tests for Exp 1371 margin-aware Cactus/BEAVER scheduler replay.

Spec: REQ-VERIFY-1371,
      SCENARIO-VERIFY-1371
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import margin_aware_cactus_beaver_scheduler_v3 as mod


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


def _exp1370_repair_artifact() -> dict[str, Any]:
    return {
        "status": "complete",
        "repair_hint_precision": 1.0,
        "honest_verdict": "verge_mcs_repair_localization_complete_claim_allowed",
        "repair_localization_rows": [
            {
                "case_id": "unsat_unit_conflict",
                "repair_hint": "Relax exactly one contradictory unit clause.",
                "precision_match": True,
            },
            {
                "case_id": "unknown_missing_bound",
                "repair_hint": "Add the missing capacity bound B.",
                "precision_match": True,
            },
            {
                "case_id": "repair_missing_upper_bound",
                "repair_hint": "Add the missing upper-bound premise.",
                "precision_match": True,
            },
        ],
    }


def test_req1371_conservative_policy_reduces_full_calls_only_at_zero_false_acceptance() -> None:
    """REQ-VERIFY-1371: high-margin SAT accepts, all non-SAT states escalate."""

    artifact = mod.build_margin_aware_scheduler_artifact(
        exp1369_artifact=_exp1369_semantic_artifact(),
        exp1370_artifact=_exp1370_repair_artifact(),
        run_date="20260505",
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["proxy_accept_rate"] == pytest.approx(0.25)
    assert artifact["low_margin_escalation_rate"] == pytest.approx(0.75)
    assert artifact["full_verifier_call_reduction"] == pytest.approx(0.25)
    assert artifact["false_acceptance_rate"] == pytest.approx(0.0)
    assert artifact["repair_hint_reuse_rate"] == pytest.approx(1.0)
    assert artifact["verifier_cost_reduction_proxy"] == pytest.approx(0.2)
    assert artifact["triage_claim_allowed"] is True
    assert (
        artifact["honest_verdict"] == "margin_aware_scheduler_claim_allowed_zero_false_acceptance"
    )

    rows = {row["case_id"]: row for row in artifact["scheduler_rows"]}
    assert rows["sat_unit_clause"]["scheduler_action"] == "proxy_accept"
    assert rows["unknown_missing_bound"]["scheduler_action"] == "escalate_full_verifier"
    assert rows["unknown_missing_bound"]["unknown_silently_accepted"] is False
    assert rows["repair_missing_upper_bound"]["repair_hint_reused"] is True


def test_req1371_repair_hint_precision_gate_blocks_scheduler_claim() -> None:
    """REQ-VERIFY-1371: repair hints must clear the precision gate before replay."""

    exp1370 = _exp1370_repair_artifact()
    exp1370["repair_hint_precision"] = 0.25

    artifact = mod.build_margin_aware_scheduler_artifact(
        exp1369_artifact=_exp1369_semantic_artifact(),
        exp1370_artifact=exp1370,
        run_date="20260505",
        project_root="/repo",
    )

    assert artifact["status"] == "blocked"
    assert artifact["full_verifier_call_reduction"] == pytest.approx(0.0)
    assert artifact["false_acceptance_rate"] == pytest.approx(0.0)
    assert artifact["repair_hint_reuse_rate"] == pytest.approx(0.0)
    assert artifact["triage_claim_allowed"] is False
    assert artifact["honest_verdict"] == "blocked_exp1370_repair_hint_precision_below_0_5"


def test_scenario1371_run_experiment_writes_in_progress_then_complete(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1371: runner persists bootstrap and terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1369_path = results / "exp1369.json"
    exp1370_path = results / "exp1370.json"
    output_path = results / "exp1371.json"
    exp1369_path.write_text(json.dumps(_exp1369_semantic_artifact()), encoding="utf-8")
    exp1370_path.write_text(json.dumps(_exp1370_repair_artifact()), encoding="utf-8")
    writes: list[dict[str, Any]] = []

    artifact = mod.run_experiment(
        project_root=tmp_path,
        run_date="20260505",
        exp1369_path=exp1369_path,
        exp1370_path=exp1370_path,
        output_path=output_path,
        write_observer=lambda _path, payload: writes.append(dict(payload)),
    )

    assert [payload["status"] for payload in writes] == ["in_progress", "complete"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
    assert artifact["artifact_metadata"]["project_root"] == str(tmp_path)
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
