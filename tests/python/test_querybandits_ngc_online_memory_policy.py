"""Tests for Exp 1303 QueryBandits + NGC online memory policy.

Spec: REQ-LEARN-1303, SCENARIO-LEARN-1303.
"""

from __future__ import annotations

import json
from pathlib import Path

from carnot.reporting import querybandits_ngc_online_memory_policy as exp


def _candidate(
    *,
    pattern: str,
    selected_decision: str,
    verifier_result: str,
    routing: str,
    support: int,
) -> dict[str, object]:
    return {
        "skill_id": f"fixture/{pattern}/{selected_decision}",
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
        "skill_graph_candidates": [
            _candidate(
                pattern="arithmetic:addition",
                selected_decision="repair",
                verifier_result="failed",
                routing="promote",
                support=8,
            ),
            _candidate(
                pattern="arithmetic:addition",
                selected_decision="accept",
                verifier_result="passed",
                routing="promote",
                support=7,
            ),
            _candidate(
                pattern="arithmetic:general",
                selected_decision="accept",
                verifier_result="passed",
                routing="demote",
                support=2,
            ),
            _candidate(
                pattern="arithmetic:ratio",
                selected_decision="repair",
                verifier_result="failed",
                routing="promote",
                support=6,
            ),
        ],
    }


def _exp1288_fixture() -> dict[str, object]:
    return {
        "experiment": "1288_interwhen_dvi_verifier_feedback_replay",
        "status": "complete",
        "clause_prediction_records": [
            {
                "constraint_pattern": "arithmetic:addition",
                "selected_decision": "repair",
                "verifier_result": "failed",
                "support": 8,
            },
            {
                "constraint_pattern": "arithmetic:general",
                "selected_decision": "accept",
                "verifier_result": "passed",
                "support": 2,
            },
        ],
        "replay_slices": [
            {
                "case_id": "case-0",
                "chronological_index": 0,
                "constraint_pattern": "arithmetic:addition",
                "target_decision": "repair",
                "verifier_result": "failed",
            },
            {
                "case_id": "case-1",
                "chronological_index": 1,
                "constraint_pattern": "arithmetic:ratio",
                "target_decision": "repair",
                "verifier_result": "failed",
            },
            {
                "case_id": "case-2",
                "chronological_index": 2,
                "constraint_pattern": "arithmetic:addition",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
            {
                "case_id": "case-3",
                "chronological_index": 3,
                "constraint_pattern": "arithmetic:general",
                "target_decision": "accept",
                "verifier_result": "passed",
            },
        ],
    }


def test_req_learn_1303_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-LEARN-1303-1: the workflow writes a durable in-progress artifact first."""

    out_path = tmp_path / "results" / "experiment_1303_querybandits_ngc_online_memory_policy.json"

    artifact = exp.write_in_progress_artifact(
        out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["artifact_metadata"]["run_date"] == "20260505"
    assert written["selected_policy_distribution"] == {
        action: 0.0 for action in exp.ACTIONS
    }
    assert written["honest_verdict"] == "in_progress"


def test_scenario_learn_1303_simulates_four_policy_actions() -> None:
    """SCENARIO-LEARN-1303: verifier memory produces honest bandit metrics."""

    artifact = exp.build_artifact(
        _exp1302_fixture(),
        _exp1288_fixture(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    exp.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["action_space"] == list(exp.ACTIONS)
    assert set(artifact["selected_policy_distribution"]) == set(exp.ACTIONS)
    assert sum(artifact["selected_policy_counts"].values()) == artifact["n_examples"]
    assert sum(artifact["selected_policy_distribution"].values()) == 1.0
    assert artifact["selected_policy_counts"][exp.ACTION_REPLAY_MEMORY] >= 3
    assert artifact["selected_policy_counts"][exp.ACTION_DEMOTE_EXPIRE_MEMORY] == 1
    assert artifact["memory_demotion_count"] == 1
    assert artifact["accepted_violation_delta"] < 0.0
    assert artifact["self_learning_delta_overall"] > 0.0
    assert artifact["bandit_regret"] >= 0.0
    assert artifact["headline_result_allowed"] is False
    assert artifact["honest_verdict"] == "online_memory_policy_improved_non_headline"


def test_req_learn_1303_run_loads_exp1302_and_exp1288_results(tmp_path: Path) -> None:
    """REQ-LEARN-1303-2/5: run loads source artifacts and writes final schema."""

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "experiment_1302_skill_graph_promotion_demotion_v2.json").write_text(
        json.dumps(_exp1302_fixture()),
        encoding="utf-8",
    )
    (results_dir / "experiment_1288_interwhen_dvi_verifier_feedback_replay.json").write_text(
        json.dumps(_exp1288_fixture()),
        encoding="utf-8",
    )
    out_path = results_dir / "experiment_1303_querybandits_ngc_online_memory_policy.json"

    artifact = exp.run(
        results_dir=results_dir,
        out_path=out_path,
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert artifact["source_artifacts"] == [
        "results/experiment_1302_skill_graph_promotion_demotion_v2.json",
        "results/experiment_1288_interwhen_dvi_verifier_feedback_replay.json",
    ]
    assert artifact["artifact_metadata"]["run_date"] == "20260505"
    assert artifact["status"] == "complete"


def test_req_learn_1303_infers_pattern_and_penalizes_missing_memory() -> None:
    """REQ-LEARN-1303-2/4: fallback pattern inference and violation scoring work."""

    examples = exp.build_feedback_examples(
        {"skill_graph_candidates": []},
        {
            "clause_prediction_records": [
                {
                    "constraint_pattern": "arithmetic:fallback",
                    "selected_decision": "repair",
                    "verifier_result": "failed",
                }
            ],
            "replay_slices": [
                {
                    "case_id": "missing-memory",
                    "chronological_index": 0,
                    "verifier_result": "failed",
                }
            ],
        },
    )

    assert examples[0]["constraint_pattern"] == "arithmetic:fallback"
    assert examples[0]["target_decision"] == "repair"
    assert examples[0]["memory_available"] is False
    assert exp._score_action(exp.ACTION_REPLAY_MEMORY, examples[0]) == (-1.0, 1)


def test_req_learn_1303_validation_and_verdict_edges() -> None:
    """REQ-LEARN-1303-5/6: validation and verdict derivation stay strict."""

    assert exp.derive_honest_verdict(0.1) == "online_memory_policy_improved_non_headline"
    assert exp.derive_honest_verdict(0.0) == "online_memory_policy_neutral_non_headline"
    assert exp.derive_honest_verdict(-0.1) == "online_memory_policy_regressed_non_headline"

    artifact = exp.build_artifact(
        _exp1302_fixture(),
        _exp1288_fixture(),
        project_root="/home/ianblenke/github.com/ianblenke/carnot",
    )
    for key, message in [
        ("status", "status must be complete"),
        ("self_learning_delta_overall", "missing required fields"),
        ("accepted_violation_delta", "missing required fields"),
        ("bandit_regret", "bandit_regret must be non-negative"),
        ("selected_policy_distribution", "selected_policy_distribution must cover actions"),
        ("memory_demotion_count", "memory_demotion_count must be non-negative"),
        ("headline_result_allowed", "headline_result_allowed must be boolean"),
        ("honest_verdict", "missing required fields"),
    ]:
        bad = dict(artifact)
        if key == "status":
            bad[key] = "in_progress"
        elif key == "bandit_regret":
            bad[key] = -0.1
        elif key == "selected_policy_distribution":
            bad[key] = {exp.ACTION_REPLAY_MEMORY: 1.0}
        elif key == "memory_demotion_count":
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

    invalid_verdict = dict(artifact)
    invalid_verdict["honest_verdict"] = "unsupported"
    try:
        exp.validate_artifact(invalid_verdict)
    except AssertionError as exc:
        assert "honest_verdict is unsupported" in str(exc)
    else:
        raise AssertionError("validation accepted unsupported honest_verdict")
