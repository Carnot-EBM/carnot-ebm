"""Tests for the Exp 3919 ARC-AGI-3 synthetic harness scaffold.

Spec coverage: REQ-PHASE4-006, SCENARIO-PHASE4-006.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.agentic.arc_agi3_harness import (
    ACTIONS,
    HARNESS_MODULE_PATH,
    UNIT_TEST_PATH,
    ArcAgi3Harness,
    PreconditionResult,
    SyntheticGridEnv,
    VerifierRouter,
    build_result_artifact,
    check_preconditions,
    stable_reproducibility_checksum,
    write_result_artifact,
)


def test_synthetic_env_reset_and_step_work() -> None:
    """REQ-PHASE4-006: the scaffold exposes a tiny reset/step grid env."""
    env = SyntheticGridEnv()
    obs = env.reset()

    assert obs.position == (0, 0)
    assert obs.goal == (2, 0)
    assert obs.grid[0][0] == 1
    assert obs.grid[0][2] == 2

    next_obs, reward, done = env.step(ACTIONS["east"])
    assert next_obs.position == (1, 0)
    assert reward == 0.0
    assert done is False

    final_obs, reward, done = env.step(ACTIONS["east"])
    assert final_obs.position == (2, 0)
    assert reward == 1.0
    assert done is True


def test_router_selects_action_by_verifier_score_and_prunes() -> None:
    """SCENARIO-PHASE4-006: verifier routing selects and prunes actions."""
    env = SyntheticGridEnv()
    obs = env.reset()
    router = VerifierRouter(keep_threshold=0.93)

    decision = router.select_action(obs, env.candidate_actions())

    assert decision.action.name == "east"
    assert decision.fallback_used is False
    assert decision.pruned_count > 0
    assert len(decision.retained_action_names) < len(env.candidate_actions())
    assert decision.score_for("east") == max(score.verification_score for score in decision.scores)
    assert decision.score_for("east") > decision.score_for("stay")


def test_router_fallback_only_when_all_candidates_verify_poorly() -> None:
    """REQ-PHASE4-006: the router escalates only when every action is poor."""

    def all_poor(_items: tuple[dict[str, object], ...]) -> dict[str, object]:
        return {"scores": [0.99 for _ in _items], "est_tokens": 0, "est_flops": 0}

    env = SyntheticGridEnv()
    router = VerifierRouter(keep_threshold=0.5, verifier_fn=all_poor)

    decision = router.select_action(env.reset(), env.candidate_actions())

    assert decision.fallback_used is True
    assert decision.action.name == "stay"
    assert decision.pruned_count == len(env.candidate_actions())


def test_harness_solves_tiny_synthetic_task() -> None:
    """SCENARIO-PHASE4-006: the verifier-router loop solves the synthetic task."""
    result = ArcAgi3Harness(
        env=SyntheticGridEnv(),
        router=VerifierRouter(keep_threshold=0.93),
        random_seed=3919,
    ).run(max_steps=4)

    assert result.solved is True
    assert result.actions_taken == ("east", "east")
    assert result.synthetic_task_solved is True
    assert result.total_pruned_count > 0
    assert result.is_synthetic_not_real_benchmark is True


def test_harness_can_report_unsolved_without_claiming_benchmark_performance() -> None:
    """REQ-PHASE4-006: unsolved synthetic runs stay scoped and honest."""
    result = ArcAgi3Harness(env=SyntheticGridEnv(), router=VerifierRouter()).run(max_steps=0)

    assert result.solved is False
    assert result.synthetic_task_solved is False
    assert result.actions_taken == ()
    assert result.is_synthetic_not_real_benchmark is True


def test_preconditions_support_success_and_blocked_import() -> None:
    """REQ-PHASE4-006: readiness is blocked when `carnot.verify` cannot import."""
    ok = check_preconditions()

    assert ok == PreconditionResult(
        preconditions_checked=True,
        carnot_verify_imported=True,
        blocked_resource="",
        detail="import carnot.verify OK",
    )

    def fail_import(_name: str) -> object:
        raise ModuleNotFoundError("synthetic import break")

    blocked = check_preconditions(import_fn=fail_import)
    assert blocked.preconditions_checked is True
    assert blocked.carnot_verify_imported is False
    assert blocked.blocked_resource == "blocked_carnot_verify_import"
    assert "synthetic import break" in blocked.detail


def test_artifact_fields_are_bare_scalars_and_checksum_is_stable(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-006: the Exp 3919 artifact exposes the required gate fields."""
    result = ArcAgi3Harness(
        env=SyntheticGridEnv(),
        router=VerifierRouter(keep_threshold=0.93),
        random_seed=3919,
    ).run(max_steps=4)
    checksum = stable_reproducibility_checksum(
        {
            "result": result.as_checksum_payload(),
            "unit_test_passed": True,
            "random_seed": 3919,
        }
    )

    artifact = build_result_artifact(
        result,
        preconditions=check_preconditions(),
        unit_test_passed=True,
        duration_s=0.125,
        reproducibility_checksum=checksum,
    )
    output_path = write_result_artifact(artifact, tmp_path / "experiment_3919.json")

    assert artifact["harness_module_path"] == HARNESS_MODULE_PATH
    assert artifact["unit_test_path"] == UNIT_TEST_PATH
    assert artifact["unit_test_passed"] is True
    assert artifact["synthetic_task_solved"] is True
    assert isinstance(artifact["action_pruned_count"], int)
    assert artifact["action_pruned_count"] > 0
    assert artifact["is_synthetic_not_real_benchmark"] is True
    assert artifact["preconditions_checked"] is True
    assert artifact["random_seed"] == 3919
    assert artifact["reproducibility_checksum"] == checksum
    assert artifact["inference_substrate"] == "synthetic_grid_plus_cpu_energy_verifier"
    assert artifact["harness_ready"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "_synthetic_only_agentic_proof_can_follow_offline_proof" in artifact["honest_verdict"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

    not_ready = build_result_artifact(
        result,
        preconditions=PreconditionResult(True, True, "", "import carnot.verify OK"),
        unit_test_passed=False,
        duration_s=0.125,
        reproducibility_checksum=checksum,
    )
    assert not_ready["harness_ready"] is False
    assert not_ready["honest_verdict"] == "complete: arc_agi3_scaffold_NOT_READY_unit_testFalse"

    blocked = build_result_artifact(
        result,
        preconditions=PreconditionResult(
            True,
            False,
            "blocked_carnot_verify_import",
            "ModuleNotFoundError",
        ),
        unit_test_passed=False,
        duration_s=0.125,
        reproducibility_checksum=checksum,
    )
    assert blocked["harness_ready"] is False
    assert blocked["honest_verdict"] == "blocked_carnot_verify_import"


def test_router_rejects_empty_candidate_sets() -> None:
    """REQ-PHASE4-006: verifier routing requires at least one candidate action."""
    with pytest.raises(ValueError, match="candidate action"):
        VerifierRouter().select_action(SyntheticGridEnv().reset(), ())
