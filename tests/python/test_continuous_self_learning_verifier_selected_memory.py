"""Tests for Exp 1358 verifier-selected continuous self-learning memory.

Spec: REQ-LEARN-1358, SCENARIO-LEARN-1358.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import continuous_self_learning_verifier_selected_memory as mod


def _exp1344() -> dict[str, Any]:
    return {
        "status": "complete",
        "replay_cases_used": 12,
        "self_learning_delta_overall": 0.75,
        "nonforgetting_certificate_rate": 1.0,
        "memory_regression_count": 0,
        "accepted_violation_delta": -0.25,
        "failure_type_policy": {
            "semantic_invalidity": {
                "action": "promote",
                "failure_count": 4,
                "source_next_actions": ["run semantic validator before promotion"],
            },
            "possible_hardcoded_solution_leakage": {
                "action": "demote",
                "failure_count": 2,
                "source_next_actions": ["exclude verifier-label repair leakage"],
            },
            "unknown_state_mishandling": {
                "action": "quarantine",
                "failure_count": 1,
                "source_next_actions": ["preserve UNKNOWN"],
            },
            "parser_schema_mismatch": {
                "action": "request_fresh_verifier",
                "failure_count": 3,
                "source_next_actions": ["recover parser schema"],
            },
        },
        "honest_verdict": "failure_type_memory_policy_dvi_ready_replay_non_headline",
    }


def _exp1353_without_verified_rows() -> dict[str, Any]:
    return {
        "status": "complete",
        "certificate_rows": [
            {
                "case_id": "sat_unit_clause",
                "parseable": False,
                "truthful": False,
                "errors": ["missing_structural_tag"],
            },
            {
                "case_id": "unsat_unit_conflict",
                "parseable": True,
                "truthful": False,
                "errors": ["wrong_state"],
            },
        ],
    }


def _blocked_exp1355() -> dict[str, Any]:
    return {"status": "blocked", "honest_verdict": "blocked_gate_check_failed"}


def test_req_learn_1358_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1358-1: bootstrap output exists before source artifacts load."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["artifact_metadata"] == {
        "project_root": "/home/ianblenke/github.com/ianblenke/carnot",
        "run_date": "20260505",
    }
    assert written["fresh_verified_sample_count"] == 0
    assert written["headline_result_allowed"] is False


def test_scenario_learn_1358_replay_fallback_is_dvi_ready_but_non_headline() -> None:
    """SCENARIO-LEARN-1358: absent fresh accepted rows preserve replay-only honesty."""

    artifact = mod.build_artifact(
        exp1344_artifact=_exp1344(),
        exp1353_artifact=_exp1353_without_verified_rows(),
        exp1355_artifact=_blocked_exp1355(),
        input_resolution={"exp1344": {"requested": "alias", "used": "fallback"}},
        source_artifacts=[
            "results/experiment_1344_continuous_self_learning_failure_type_memory_policy.json"
        ],
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["replay_cases_used"] == 12
    assert artifact["fresh_verified_sample_count"] == 0
    assert artifact["variant_question_count"] == 4
    assert artifact["self_learning_delta_overall"] == 0.75
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["memory_regression_count"] == 0
    assert artifact["accepted_violation_delta"] == -0.25
    assert artifact["promoted_memory_count"] == 4
    assert artifact["demoted_memory_count"] == 3
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == (
        "verifier_selected_memory_replay_only_dvi_ready_non_headline"
    )
    assert artifact["memory_updates"]["promoted"] == ["replay:semantic_invalidity"]
    assert set(artifact["memory_updates"]["demoted"]) == {
        "replay:possible_hardcoded_solution_leakage",
        "replay:unknown_state_mishandling",
    }


def test_scenario_learn_1358_fresh_verified_samples_are_promoted_for_headline() -> None:
    """SCENARIO-LEARN-1358: verifier-accepted fresh rows make the update non-replay-only."""

    exp1353 = {
        "status": "complete",
        "certificate_rows": [
            {
                "case_id": "fresh-cert-1",
                "expected_state": "SAT",
                "parseable": True,
                "truthful": True,
                "errors": [],
            }
        ],
    }
    exp1355 = {
        "status": "complete",
        "semantic_validator_rows": [
            {
                "case_id": "fresh-semantic-1",
                "verifier_accepted": True,
                "semantic_rejected": False,
            },
            {
                "case_id": "fresh-semantic-reject",
                "verifier_accepted": False,
                "semantic_rejected": True,
            },
        ],
    }

    artifact = mod.build_artifact(
        exp1344_artifact=_exp1344(),
        exp1353_artifact=exp1353,
        exp1355_artifact=exp1355,
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["fresh_verified_sample_count"] == 2
    assert artifact["variant_question_count"] == 3
    assert artifact["promoted_memory_count"] == 2
    assert artifact["demoted_memory_count"] == 1
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == (
        "verifier_selected_memory_fresh_verified_dvi_ready_headline_eligible"
    )
    assert "fresh:exp1353:fresh-cert-1" in artifact["memory_updates"]["promoted"]
    assert "fresh:exp1355:fresh-semantic-1" in artifact["memory_updates"]["promoted"]
    assert artifact["memory_updates"]["demoted"] == ["fresh:exp1355:fresh-semantic-reject"]


def test_req_learn_1358_run_records_exp1344_alias_fallback(tmp_path: Path) -> None:
    """REQ-LEARN-1358-2: requested Exp 1344 alias absence is auditable."""

    results = tmp_path / "results"
    results.mkdir()
    (results / mod.EXP1344_FALLBACK_FILE).write_text(json.dumps(_exp1344()), encoding="utf-8")
    (results / mod.EXP1353_FILE).write_text(
        json.dumps(_exp1353_without_verified_rows()),
        encoding="utf-8",
    )
    (results / mod.EXP1355_FILE).write_text(json.dumps(_blocked_exp1355()), encoding="utf-8")
    out_path = results / mod.OUTPUT_FILE

    artifact = mod.run(
        results_dir=results,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["status"] == "complete"
    assert f"results/{mod.EXP1344_REQUESTED_FILE}" in artifact["inputs_unavailable"]
    assert artifact["input_resolution"]["exp1344"]["used"] == (
        f"results/{mod.EXP1344_FALLBACK_FILE}"
    )
    assert artifact["headline_result_allowed"] is False


def test_req_learn_1358_dvi_controls_block_headline_when_controls_regress() -> None:
    """REQ-LEARN-1358-7: fresh samples cannot override control regressions."""

    artifact = mod.build_artifact(
        exp1344_artifact=dict(_exp1344(), memory_regression_count=1),
        exp1353_artifact={
            "status": "complete",
            "certificate_rows": [
                {
                    "case_id": "fresh-cert-1",
                    "parseable": True,
                    "truthful": True,
                    "errors": [],
                }
            ],
        },
        exp1355_artifact={},
        project_root="/repo",
    )

    assert artifact["fresh_verified_sample_count"] == 1
    assert artifact["dvi_ready"] is False
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "verifier_selected_memory_controls_blocked_non_headline"


def test_req_learn_1358_validation_rejects_malformed_artifacts() -> None:
    """REQ-LEARN-1358-6: required reconciliation fields are schema checked."""

    artifact = mod.build_artifact(
        exp1344_artifact=_exp1344(),
        exp1353_artifact={},
        exp1355_artifact={},
        project_root="/repo",
    )

    missing = dict(artifact)
    del missing["fresh_verified_sample_count"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    bad_rate = dict(artifact, nonforgetting_certificate_rate=1.5)
    with pytest.raises(AssertionError, match="between 0 and 1"):
        mod.validate_artifact(bad_rate)

    bad_status = dict(artifact, status="running")
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)
