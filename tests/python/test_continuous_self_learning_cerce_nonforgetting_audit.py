"""Tests for Exp 1315 CerCE non-forgetting audit.

Spec: REQ-LEARN-1315, SCENARIO-LEARN-1315.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import continuous_self_learning_cerce_nonforgetting_audit as exp


def _candidate(
    *,
    pattern: str,
    selected_decision: str,
    verifier_result: str,
    routing: str,
    support: int,
) -> dict[str, object]:
    return {
        "skill_id": f"fixture/{pattern}/{selected_decision}/{routing}",
        "constraint_pattern": pattern,
        "selected_decision": selected_decision,
        "verifier_result": verifier_result,
        "memory_routing_decision": routing,
        "repair_hint": (
            "recompute_arithmetic_result"
            if selected_decision == "repair"
            else "accept_verified_answer"
        ),
        "replay_evidence": {
            "support": support,
            "verifier_backed": True,
            "source_experiment": 1288,
        },
    }


def _exp1302_fixture() -> dict[str, object]:
    return {
        "experiment": "1302_skill_graph_promotion_demotion_v2",
        "status": "complete",
        "promoted_memory_count": 2,
        "demoted_memory_count": 1,
        "expired_memory_count": 1,
        "skill_graph_candidates": [
            _candidate(
                pattern="math:addition",
                selected_decision="repair",
                verifier_result="failed",
                routing="promote",
                support=8,
            ),
            _candidate(
                pattern="math:addition",
                selected_decision="accept",
                verifier_result="passed",
                routing="promote",
                support=7,
            ),
            _candidate(
                pattern="math:general",
                selected_decision="accept",
                verifier_result="passed",
                routing="demote",
                support=2,
            ),
            _candidate(
                pattern="math:ratio",
                selected_decision="accept",
                verifier_result="passed",
                routing="expire",
                support=1,
            ),
        ],
    }


def _exp1303_fixture() -> dict[str, object]:
    return {
        "experiment": "1303_querybandits_ngc_online_memory_policy",
        "status": "complete",
        "self_learning_delta_overall": 1.25,
        "accepted_violation_delta": -0.5,
        "selected_policy_counts": {
            "replay_memory": 3,
            "rewrite_repair_prompt": 1,
            "abstain_escalate": 0,
            "demote_expire_memory": 2,
        },
        "headline_result_allowed": False,
        "honest_verdict": "online_memory_policy_improved_non_headline",
    }


def _exp1288_fixture() -> dict[str, object]:
    return {
        "experiment": "1288_interwhen_dvi_verifier_feedback_replay",
        "status": "complete",
        "memory_update_written": True,
        "clause_prediction_records": [
            {
                "constraint_pattern": "math:addition",
                "selected_decision": "repair",
                "verifier_result": "failed",
                "support": 8,
            }
        ],
        "replay_slices": [
            {
                "case_id": "old-promoted",
                "chronological_index": 0,
                "constraint_pattern": "math:addition",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
            {
                "case_id": "old-demoted",
                "chronological_index": 1,
                "constraint_pattern": "math:general",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
            {
                "case_id": "old-expired",
                "chronological_index": 2,
                "constraint_pattern": "math:ratio",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
            {
                "case_id": "old-missing",
                "chronological_index": 3,
                "constraint_pattern": "math:missing",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
            {
                "case_id": "improved-promote",
                "chronological_index": 4,
                "target_decision": "repair",
                "verifier_result": "failed",
            },
            "skip-non-record",
        ],
    }


def test_req_learn_1315_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1315-1: the workflow writes a durable in-progress artifact."""

    out_path = tmp_path / "results" / exp.OUTPUT_FILE

    artifact = exp.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["artifact_metadata"]["run_date"] == "20260505"
    assert written["nonforgetting_certificate_rate"] == 0.0
    assert written["honest_verdict"] == "in_progress"


def test_req_learn_1315_missing_inputs_write_terminal_blocker(tmp_path: Path) -> None:
    """REQ-LEARN-1315-2: absent source artifacts produce a terminal blocker."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / exp.EXP1302_FILE).write_text(
        json.dumps(_exp1302_fixture()),
        encoding="utf-8",
    )
    (results_dir / exp.EXP1288_FILE).write_text(
        json.dumps(_exp1288_fixture()),
        encoding="utf-8",
    )
    out_path = results_dir / exp.OUTPUT_FILE

    artifact = exp.run(
        results_dir=results_dir,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    assert artifact["status"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_missing_inputs"
    assert artifact["missing_inputs"] == [f"results/{exp.EXP1303_FILE}"]
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact


def test_scenario_learn_1315_audits_nonforgetting_and_policy_decisions() -> None:
    """SCENARIO-LEARN-1315: old verified behavior survives the policy audit."""

    artifact = exp.build_artifact(
        _exp1302_fixture(),
        _exp1303_fixture(),
        _exp1288_fixture(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["nonforgetting_certificate_rate"] == 1.0
    assert artifact["memory_regression_count"] == 0
    assert artifact["lagrangian_violation_penalty"] == 0.0
    assert artifact["accepted_violation_delta"] < 0.0
    assert artifact["self_learning_delta_overall"] == 1.25
    assert artifact["promoted_memory_count"] == 2
    assert artifact["demoted_memory_count"] == 2
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "cerce_nonforgetting_preserved_improved_non_headline"
    assert artifact["replay_cohort_counts"] == {
        "old_verified": 4,
        "101_improved": 2,
        "adversarial_unknown": 2,
    }
    for action in ("promote", "demote", "rewrite", "abstain", "expire"):
        assert artifact["audit_decision_counts"][action] > 0
    assert artifact["adversarial_promotion_count"] == 0
    assert all(
        row["audit_decision"] == "abstain"
        for row in artifact["policy_audit_records"]
        if row["cohort"] == "adversarial_unknown"
    )


def test_req_learn_1315_run_loads_sources_and_writes_final_schema(tmp_path: Path) -> None:
    """REQ-LEARN-1315-2/6: run loads all sources and writes required fields."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / exp.EXP1302_FILE).write_text(
        json.dumps(_exp1302_fixture()),
        encoding="utf-8",
    )
    (results_dir / exp.EXP1303_FILE).write_text(
        json.dumps(_exp1303_fixture()),
        encoding="utf-8",
    )
    (results_dir / exp.EXP1288_FILE).write_text(
        json.dumps(_exp1288_fixture()),
        encoding="utf-8",
    )
    out_path = results_dir / exp.OUTPUT_FILE

    artifact = exp.run(
        results_dir=results_dir,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["source_artifacts"] == [
        f"results/{exp.EXP1302_FILE}",
        f"results/{exp.EXP1303_FILE}",
        f"results/{exp.EXP1288_FILE}",
    ]
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["status"] == "complete"


def test_req_learn_1315_validation_and_verdict_edges() -> None:
    """REQ-LEARN-1315-4/5/7: validation and verdict derivation stay strict."""

    assert exp._support(None) == 0
    assert exp._fallback_pattern([], "repair", "failed", 0) == "unknown"
    assert exp._cohort("abstain", "unknown") == "adversarial_unknown"
    assert (
        exp.derive_honest_verdict(
            nonforgetting_certificate_rate=1.0,
            memory_regression_count=0,
            self_learning_delta_overall=0.1,
            lagrangian_violation_penalty=0.0,
        )
        == "cerce_nonforgetting_preserved_improved_non_headline"
    )
    assert (
        exp.derive_honest_verdict(
            nonforgetting_certificate_rate=0.5,
            memory_regression_count=1,
            self_learning_delta_overall=0.1,
            lagrangian_violation_penalty=0.5,
        )
        == "cerce_nonforgetting_regression_non_headline"
    )
    assert (
        exp.derive_honest_verdict(
            nonforgetting_certificate_rate=1.0,
            memory_regression_count=0,
            self_learning_delta_overall=-0.1,
            lagrangian_violation_penalty=0.0,
        )
        == "cerce_nonforgetting_preserved_regressed_non_headline"
    )
    assert (
        exp.derive_honest_verdict(
            nonforgetting_certificate_rate=1.0,
            memory_regression_count=0,
            self_learning_delta_overall=0.0,
            lagrangian_violation_penalty=0.0,
        )
        == "cerce_nonforgetting_preserved_neutral_non_headline"
    )

    artifact = exp.build_artifact(
        _exp1302_fixture(),
        _exp1303_fixture(),
        _exp1288_fixture(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )
    for key, message in [
        ("status", "status must be complete or blocked"),
        ("accepted_violation_delta", "missing required fields"),
        ("memory_regression_count", "memory_regression_count must be non-negative"),
        ("nonforgetting_certificate_rate", "nonforgetting_certificate_rate must be between 0 and 1"),
        ("lagrangian_violation_penalty", "lagrangian_violation_penalty must be non-negative"),
        ("promoted_memory_count", "promoted_memory_count must be non-negative"),
        ("demoted_memory_count", "demoted_memory_count must be non-negative"),
        ("headline_result_allowed", "headline_result_allowed must be boolean"),
        ("honest_verdict", "missing required fields"),
    ]:
        bad = dict(artifact)
        if key == "status":
            bad[key] = "in_progress"
        elif key == "memory_regression_count":
            bad[key] = -1
        elif key == "nonforgetting_certificate_rate":
            bad[key] = 1.1
        elif key == "lagrangian_violation_penalty":
            bad[key] = -0.1
        elif key == "promoted_memory_count":
            bad[key] = -1
        elif key == "demoted_memory_count":
            bad[key] = -1
        elif key == "headline_result_allowed":
            bad[key] = "false"
        else:
            del bad[key]
        try:
            exp.validate_artifact(bad)
        except AssertionError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"validation accepted invalid {key}")

    unsupported_verdict = dict(artifact)
    unsupported_verdict["honest_verdict"] = "unsupported"
    try:
        exp.validate_artifact(unsupported_verdict)
    except AssertionError as exc:
        assert "honest_verdict is unsupported" in str(exc)
    else:
        raise AssertionError("validation accepted unsupported honest_verdict")
