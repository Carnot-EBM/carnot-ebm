"""Tests for Exp 1374 continuous self-learning v3.

Spec: REQ-LEARN-1374, SCENARIO-LEARN-1374.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import continuous_self_learning_v3_verifier_selected_or_csp_fallback as mod


def _exp1358() -> dict[str, Any]:
    return {
        "status": "complete",
        "replay_cases_used": 282,
        "fresh_verified_sample_count": 0,
        "variant_question_count": 6,
        "self_learning_delta_overall": 1.596429,
        "nonforgetting_certificate_rate": 1.0,
        "memory_regression_count": 0,
        "accepted_violation_delta": -0.846154,
        "promoted_memory_count": 30,
        "demoted_memory_count": 39,
        "headline_result_allowed": False,
        "honest_verdict": "verifier_selected_memory_replay_only_dvi_ready_non_headline",
        "variant_questions": [
            {
                "variant_id": "replay:semantic_invalidity",
                "case_id": "semantic_invalidity",
                "source": "exp1358_replay",
                "support": 30,
                "verifier_accepted": True,
                "semantic_rejected": False,
            },
            {
                "variant_id": "replay:unknown_state_mishandling",
                "case_id": "unknown_state_mishandling",
                "source": "exp1358_replay",
                "support": 4,
                "verifier_accepted": False,
                "semantic_rejected": True,
            },
        ],
    }


def _exp1369_open() -> dict[str, Any]:
    return {
        "status": "complete",
        "semantic_validator_claim_allowed": True,
        "honest_verdict": "semantic_validator_v2_complete_unknown_preserved",
        "semantic_validator_rows": [
            {
                "case_id": "sat_unit_clause",
                "certificate_state": "SAT",
                "expected_state": "SAT",
                "semantic_result": "SAT",
                "constraint_evaluated": True,
                "constraint_passed": True,
            },
            {
                "case_id": "unknown_missing_bound",
                "certificate_state": "UNKNOWN",
                "expected_state": "UNKNOWN",
                "semantic_result": "UNKNOWN",
                "constraint_evaluated": True,
                "constraint_passed": True,
            },
            {
                "case_id": "bad_claim",
                "certificate_state": "SAT",
                "expected_state": "SAT",
                "semantic_result": "UNSAT",
                "constraint_evaluated": True,
                "constraint_passed": False,
            },
        ],
    }


def _exp1369_closed() -> dict[str, Any]:
    return {
        "status": "complete",
        "semantic_validator_claim_allowed": False,
        "semantic_validator_rows": [],
        "honest_verdict": "semantic_gate_closed",
    }


def _exp1365_viable() -> dict[str, Any]:
    return {
        "status": "complete",
        "corpus_cases_used": 100,
        "csp_feasibility_rate": 0.74,
        "eidoku_csp_viable": True,
        "honest_verdict": "eidoku_csp_viable_local_fover_probe",
    }


def test_req_learn_1374_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1374-1: bootstrap output exists before source artifacts load."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["path_used"] is None
    assert written["fresh_verified_sample_count"] == 0
    assert written["csp_selected_sample_count"] == 0
    assert written["headline_result_allowed"] is False


def test_scenario_learn_1374_primary_semantic_verified_promotes_fresh_cases() -> None:
    """SCENARIO-LEARN-1374: semantic gate uses fresh verifier-accepted rows."""

    artifact = mod.build_artifact(
        exp1358_artifact=_exp1358(),
        exp1365_artifact={},
        exp1369_artifact=_exp1369_open(),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["path_used"] == "primary_semantic_verified"
    assert artifact["replay_cases_used"] == 282
    assert artifact["fresh_verified_sample_count"] == 2
    assert artifact["csp_selected_sample_count"] == 0
    assert artifact["variant_question_count"] == 3
    assert artifact["promoted_memory_count"] == 2
    assert artifact["demoted_memory_count"] == 1
    assert artifact["self_learning_delta_overall"] == 1.596429
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["memory_regression_count"] == 0
    assert artifact["accepted_violation_delta"] == -0.846154
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is True
    assert artifact["honest_verdict"] == (
        "continuous_self_learning_v3_primary_semantic_verified_dvi_ready_headline_allowed"
    )


def test_scenario_learn_1374_csp_fallback_stays_non_headline() -> None:
    """SCENARIO-LEARN-1374: CSP fallback can promote memory but not headline."""

    artifact = mod.build_artifact(
        exp1358_artifact=_exp1358(),
        exp1365_artifact=_exp1365_viable(),
        exp1369_artifact=_exp1369_closed(),
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["path_used"] == "fallback_csp_selected"
    assert artifact["fresh_verified_sample_count"] == 0
    assert artifact["csp_selected_sample_count"] == 74
    assert artifact["variant_question_count"] == 1
    assert artifact["promoted_memory_count"] == 74
    assert artifact["demoted_memory_count"] == 0
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == (
        "continuous_self_learning_v3_csp_selected_dvi_ready_non_headline"
    )


def test_scenario_learn_1374_replay_fallback_preserves_1358_controls() -> None:
    """SCENARIO-LEARN-1374: replay fallback remains non-headline."""

    artifact = mod.build_artifact(
        exp1358_artifact=_exp1358(),
        exp1365_artifact={},
        exp1369_artifact={},
        project_root="/repo",
    )

    mod.validate_artifact(artifact)
    assert artifact["path_used"] == "fallback_replay"
    assert artifact["fresh_verified_sample_count"] == 0
    assert artifact["csp_selected_sample_count"] == 0
    assert artifact["variant_question_count"] == 2
    assert artifact["promoted_memory_count"] == 30
    assert artifact["demoted_memory_count"] == 4
    assert artifact["dvi_ready"] is True
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == (
        "continuous_self_learning_v3_replay_only_dvi_ready_non_headline"
    )


def test_req_learn_1374_run_loads_sources_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """REQ-LEARN-1374-1/5: runner writes bootstrap then terminal artifact."""

    results = tmp_path / "results"
    results.mkdir()
    (results / mod.EXP1358_FILE).write_text(json.dumps(_exp1358()), encoding="utf-8")
    (results / mod.EXP1365_FILE).write_text(json.dumps(_exp1365_viable()), encoding="utf-8")
    (results / mod.EXP1369_FILE).write_text(json.dumps(_exp1369_open()), encoding="utf-8")
    out_path = results / mod.OUTPUT_FILE

    artifact = mod.run(results_dir=results, out_path=out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "complete"
    assert written["path_used"] == "primary_semantic_verified"
    assert written["source_artifacts"] == [
        f"results/{mod.EXP1358_FILE}",
        f"results/{mod.EXP1365_FILE}",
        f"results/{mod.EXP1369_FILE}",
    ]


def test_req_learn_1374_validation_rejects_malformed_artifacts() -> None:
    """REQ-LEARN-1374-5/7: artifact schema and headline gates are enforced."""

    artifact = mod.build_artifact(
        exp1358_artifact=_exp1358(),
        exp1365_artifact={},
        exp1369_artifact=_exp1369_open(),
        project_root="/repo",
    )

    missing = dict(artifact)
    del missing["path_used"]
    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact(missing)

    malformed_headline = dict(
        artifact,
        path_used="fallback_csp_selected",
        headline_result_allowed=True,
    )
    with pytest.raises(AssertionError, match="primary semantic"):
        mod.validate_artifact(malformed_headline)
