"""Tests for Exp 1484 FR-11 v9 query-time memory policy replay.

Spec: REQ-LEARN-1484, SCENARIO-LEARN-1484, SCENARIO-LEARN-1485.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline import query_time_memory_policy as policy
from carnot.reporting import fr11_v9_query_time_memory_policy as mod


def _promoted(count: int = 3) -> list[str]:
    return [f"dvi_v8:verified:exp1449:ltlzinc-memory-positive-{index}" for index in range(count)]


def _demoted(count: int = 3) -> list[str]:
    return [f"dvi_v8:verified:fover:memory-control-{index}" for index in range(count)]


def _exp1471(
    *,
    promoted_count: int = 3,
    demoted_count: int = 3,
    nonforgetting_rate: float = 1.0,
    soundness_mistakes: int = 0,
    completeness_mistakes: int = 140,
) -> dict[str, Any]:
    promoted = _promoted(promoted_count)
    demoted = _demoted(demoted_count)
    return {
        "experiment": "1471_fr11_v8_verified_memory_growth_pivot",
        "status": "complete",
        "model_specs": mod.MODEL_SPECS,
        "headline_result_allowed": True,
        "pivot_preserved": True,
        "self_learning_delta_overall": len(promoted),
        "new_promoted_count": len(promoted),
        "memory_entries_added": len(promoted),
        "session_memory_updated": bool(promoted),
        "nonforgetting_rate": nonforgetting_rate,
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "memory_updates": {
            "promoted": promoted,
            "demoted": demoted,
            "promoted_memory_count": len(promoted),
            "demoted_memory_count": len(demoted),
            "rejection_reason_counts": {"verifier_rejection": len(demoted)},
        },
    }


def _exp1472(*, soundness_mistakes: int = 0, completeness_mistakes: int = 140) -> dict[str, Any]:
    return {
        "experiment": "1472_online_verifier_asymmetric_mistake_budget",
        "status": "complete",
        "source_experiment": "experiment_1471_fr11_v8_verified_memory_growth_pivot",
        "soundness_mistakes": soundness_mistakes,
        "completeness_mistakes": completeness_mistakes,
        "source_nonforgetting_rate": 1.0,
        "self_learning_claim_preserved": soundness_mistakes == 0,
        "honest_verdict": "self_learning_claim_preserved_zero_soundness_mistakes",
    }


def test_req_learn_1484_query_memory_signal_is_opt_in() -> None:
    """REQ-LEARN-1484-3: memory is disabled unless the caller opts in."""

    empty_eval = policy.evaluate_query_time_memory_policy(
        (),
        policy.VerifiedMemoryIndex.from_ids(()),
    )
    assert empty_eval.task_success_rate == 0.0

    cases = (
        policy.QueryReplayCase(case_id="memory-positive", expects_memory_signal=True),
        policy.QueryReplayCase(case_id="negative-control", expects_memory_signal=False),
    )
    memory_index = policy.VerifiedMemoryIndex.from_ids(["memory-positive"])

    default_eval = policy.evaluate_query_time_memory_policy(cases, memory_index)
    enabled_eval = policy.evaluate_query_time_memory_policy(
        cases,
        memory_index,
        memory_enabled=True,
    )

    assert default_eval.memory_enabled is False
    assert default_eval.task_success_rate == pytest.approx(0.5)
    assert default_eval.completeness_mistakes == 1
    assert enabled_eval.memory_enabled is True
    assert enabled_eval.task_success_rate == pytest.approx(1.0)
    assert enabled_eval.soundness_mistakes == 0
    assert enabled_eval.completeness_mistakes == 0


def test_scenario_learn_1484_bounded_replay_improves_without_false_accepts() -> None:
    """SCENARIO-LEARN-1484: opt-in memory improves the same bounded replay cases."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(promoted_count=3, demoted_count=3),
        exp1472_artifact=_exp1472(soundness_mistakes=0),
        project_root="/repo",
        commands_run=["pytest targeted"],
        max_replay_pairs=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["policy_integration_ready"] is True
    assert artifact["replay_cases_evaluated"] == 6
    assert artifact["baseline_task_success_rate"] == pytest.approx(0.5)
    assert artifact["memory_task_success_rate"] == pytest.approx(1.0)
    assert artifact["task_success_delta"] == pytest.approx(0.5)
    assert artifact["soundness_mistakes"] == 0
    assert artifact["completeness_mistakes"] == 140
    assert artifact["memory_policy_replay"]["memory_enabled"]["completeness_mistakes"] == 0
    assert artifact["promotion_allowed"] is True
    assert artifact["honest_verdict"] == (
        "query_time_memory_policy_improves_bounded_replay_without_false_accepts"
    )
    assert artifact["tests_run"] == ["pytest targeted"]


def test_scenario_learn_1485_source_soundness_blocks_promotion() -> None:
    """SCENARIO-LEARN-1485: source false accepts block promotion."""

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(soundness_mistakes=0),
        exp1472_artifact=_exp1472(soundness_mistakes=1),
        project_root="/repo",
        max_replay_pairs=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["policy_integration_ready"] is False
    assert artifact["task_success_delta"] >= 0.0
    assert artifact["soundness_mistakes"] == 1
    assert artifact["promotion_allowed"] is False
    assert artifact["honest_verdict"] == "query_time_memory_policy_blocked_by_soundness_risk"


def test_req_learn_1484_source_gate_and_edge_helpers_remain_conservative() -> None:
    """REQ-LEARN-1484-2/6: malformed controls and source gates stay conservative."""

    assert mod._as_str_list("not-a-list") == []
    assert (
        mod._honest_verdict(
            soundness_mistakes=0,
            policy_integration_ready=True,
            task_success_delta=0.0,
        )
        == "query_time_memory_policy_no_positive_task_benefit"
    )

    exp1472 = _exp1472(soundness_mistakes=0)
    exp1472["self_learning_claim_preserved"] = False
    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(),
        exp1472_artifact=exp1472,
        project_root="/repo",
        max_replay_pairs=3,
    )

    mod.validate_artifact(artifact)
    assert artifact["policy_integration_ready"] is False
    assert artifact["promotion_allowed"] is False
    assert artifact["honest_verdict"] == "query_time_memory_policy_blocked_by_source_or_replay_gate"


def test_req_learn_1484_run_writes_bootstrap_then_complete_artifact(tmp_path: Path) -> None:
    """REQ-LEARN-1484-1/7: run writes the required terminal artifact."""

    exp1471_path = tmp_path / "results" / "experiment_1471.json"
    exp1472_path = tmp_path / "results" / "experiment_1472.json"
    out_path = tmp_path / "results" / mod.OUTPUT_FILE
    exp1471_path.parent.mkdir(parents=True, exist_ok=True)
    exp1471_path.write_text(json.dumps(_exp1471(), sort_keys=True), encoding="utf-8")
    exp1472_path.write_text(json.dumps(_exp1472(), sort_keys=True), encoding="utf-8")

    artifact = mod.run(
        exp1471_path=exp1471_path,
        exp1472_path=exp1472_path,
        out_path=out_path,
        project_root=tmp_path,
        commands_run=["pytest targeted"],
    )

    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["tests_run"] == ["pytest targeted"]


def test_req_learn_1484_validation_rejects_bad_contract() -> None:
    """REQ-LEARN-1484-4/6/7: validation enforces rates and promotion gate."""

    in_progress = mod.write_in_progress_artifact(Path("/tmp/nonpersistent_exp1484.json"))
    mod.validate_artifact(in_progress)

    artifact = mod.build_artifact(
        exp1471_artifact=_exp1471(),
        exp1472_artifact=_exp1472(),
        project_root="/repo",
        max_replay_pairs=3,
    )

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({key: value for key, value in artifact.items() if key != "status"})

    bad_delta = dict(artifact, task_success_delta=-1.0)
    with pytest.raises(AssertionError, match="task_success_delta"):
        mod.validate_artifact(bad_delta)

    bad_promotion = dict(artifact, promotion_allowed=True, soundness_mistakes=1)
    with pytest.raises(AssertionError, match="promotion_allowed"):
        mod.validate_artifact(bad_promotion)

    bad_cases = dict(artifact, replay_cases_evaluated=0)
    with pytest.raises(AssertionError, match="replay_cases_evaluated"):
        mod.validate_artifact(bad_cases)

    bad_nonforgetting = dict(
        artifact,
        policy_integration_ready=True,
        nonforgetting_rate=0.5,
        promotion_allowed=False,
    )
    with pytest.raises(AssertionError, match="nonforgetting"):
        mod.validate_artifact(bad_nonforgetting)
