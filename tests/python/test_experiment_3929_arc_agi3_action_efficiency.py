"""Tests for Exp 3929 ARC-AGI-3 synthetic action-efficiency measurement.

Spec coverage: REQ-PHASE4-007, SCENARIO-PHASE4-007.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from carnot.agentic import arc_agi3_action_efficiency as exp3929


def fake_energy_verifier(items: tuple[dict[str, object], ...]) -> dict[str, object]:
    """Return low error only for candidates that the synthetic potential marks useful."""

    return {
        "scores": [
            0.01 if int(item["potential_delta"]) > 0 else 0.99
            for item in items
        ],
        "est_tokens": len(items),
        "est_flops": len(items) * 10,
    }


def test_rich_env_is_multistep_deterministic_and_has_larger_action_space() -> None:
    """REQ-PHASE4-007: the richer synthetic task is deterministic and non-tiny."""
    config = exp3929.RichArcTaskConfig(
        start=(0, 0),
        key=(1, 5),
        switch=(5, 5),
        gate=(5, 2),
        goal=(6, 0),
    )
    env = exp3929.RichSyntheticArcEnv(config)
    observation = env.reset()

    assert observation.position == config.start
    assert observation.goal == config.goal
    assert observation.stage == "find_key"
    assert len(env.candidate_actions()) > 8

    first, reward, done = env.step(exp3929.RICH_ACTIONS["jump_south"])
    second, reward_again, done_again = env.step(exp3929.RICH_ACTIONS["jump_south"])

    replay = exp3929.RichSyntheticArcEnv(config)
    replay.reset()
    replay_first, _, _ = replay.step(exp3929.RICH_ACTIONS["jump_south"])
    replay_second, _, _ = replay.step(exp3929.RICH_ACTIONS["jump_south"])

    assert first.position == replay_first.position == (0, 2)
    assert second.position == replay_second.position == (0, 4)
    assert reward == 0.0
    assert reward_again == 0.0
    assert done is False
    assert done_again is False


def test_verifier_pruner_selects_lowest_energy_progress_action() -> None:
    """SCENARIO-PHASE4-007: verifier-pruned selection minimizes candidate energy."""
    env = exp3929.RichSyntheticArcEnv(exp3929.build_episode_configs(1, random_seed=3929)[0])
    observation = env.reset()

    decision = exp3929.select_verifier_pruned_action(
        observation,
        env.candidate_actions(),
        verifier_fn=fake_energy_verifier,
    )

    assert decision.action.name == "jump_south"
    assert decision.pruned_count > 0
    assert decision.energy_for("jump_south") == min(score.energy_score for score in decision.scores)
    assert decision.energy_for("stay") > decision.energy_for("jump_south")


def test_measurement_reports_ratio_ci_solve_rates_and_honest_synthetic_scope() -> None:
    """SCENARIO-PHASE4-007: Exp 3929 reports the required bare artifact fields."""
    measurement = exp3929.run_action_efficiency_measurement(
        n_episodes=30,
        random_seed=3929,
        verifier_fn=fake_energy_verifier,
        max_steps=240,
        bootstrap_resamples=300,
    )
    artifact = exp3929.build_result_artifact(
        measurement,
        preconditions=exp3929.PreconditionResult(
            preconditions_checked=True,
            carnot_verify_imported=True,
            arc_harness_imported=True,
            blocked_resource="",
            detail="test preconditions OK",
        ),
        real_benchmark_preflight=exp3929.RealBenchmarkPreflight(
            reachable=False,
            note="test stub: no real benchmark access attempted",
            url=exp3929.REAL_ARC_AGI3_BASE_URL,
        ),
        duration_s=0.25,
    )

    assert artifact["action_efficiency_ratio"] > 1.0
    assert artifact["action_efficiency_ci95"]["low"] > 1.0
    assert artifact["action_efficiency_ci95"]["high"] >= artifact["action_efficiency_ci95"]["low"]
    assert artifact["verifier_mean_actions"] < artifact["baseline_mean_actions"]
    assert artifact["n_episodes"] == 30
    assert artifact["solve_rate_with_verifier"] >= artifact["solve_rate_baseline"]
    assert artifact["is_synthetic_not_real_benchmark"] is True
    assert artifact["real_benchmark_reachable"] is False
    assert artifact["preconditions_checked"] is True
    assert artifact["random_seed"] == 3929
    assert isinstance(artifact["reproducibility_checksum"], str)
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["duration_s"] == 0.25
    assert artifact["inference_substrate"] == "synthetic_arc_grid_cpu_energy_verifier"
    assert artifact["falsification_gate"] == "VERIFIER_ROUTER_HELPS"
    assert artifact["honest_verdict"].startswith("complete:")
    assert "_synthetic_first_agentic_step_real_benchmark_reachablefalse" in artifact["honest_verdict"]


def test_no_advantage_verdict_fires_when_ci_lower_bound_does_not_clear_gate() -> None:
    """SCENARIO-PHASE4-007: no-advantage synthetic findings stay honest."""
    measurement = exp3929.ActionEfficiencyMeasurement(
        verifier_episodes=(
            exp3929.EpisodeResult(True, ("a", "b"), 0),
            exp3929.EpisodeResult(True, ("a", "b"), 0),
        ),
        baseline_episodes=(
            exp3929.EpisodeResult(True, ("a", "b"), 0),
            exp3929.EpisodeResult(True, ("a", "b"), 0),
        ),
        action_efficiency_ratio=1.0,
        action_efficiency_ci95=exp3929.RatioConfidenceInterval(low=1.0, high=1.0),
        random_seed=3929,
        bootstrap_resamples=10,
    )

    artifact = exp3929.build_result_artifact(
        measurement,
        preconditions=exp3929.PreconditionResult(True, True, True, "", "ok"),
        real_benchmark_preflight=exp3929.RealBenchmarkPreflight(True, "ok", "https://example.test"),
        duration_s=0.1,
    )

    assert artifact["falsification_gate"] == "NO_ACTION_ADVANTAGE"
    assert artifact["honest_verdict"] == (
        "complete: arc_agi3_verifier_router_NO_ADVANTAGE_ratio1.000_synthetic_finding"
    )


def test_blocked_preconditions_exit_before_measurement_claims() -> None:
    """REQ-PHASE4-007: failed imports produce a blocked terminal artifact."""
    env = exp3929.RichSyntheticArcEnv(exp3929.build_episode_configs(1, random_seed=3929)[0])
    blocked = exp3929.build_blocked_artifact(
        preconditions=exp3929.PreconditionResult(
            True,
            True,
            False,
            "blocked_arc_harness_import",
            "ModuleNotFoundError('arc harness')",
        ),
        real_benchmark_preflight=exp3929.RealBenchmarkPreflight(False, "not attempted", ""),
        duration_s=0.0,
        final_observation=env.reset(),
    )

    assert blocked["honest_verdict"] == "blocked_arc_harness_import"
    assert blocked["status"] == "blocked_arc_harness_import"
    assert blocked["action_efficiency_ratio"] == 0.0
    assert blocked["preconditions_checked"] is True
    assert blocked["is_synthetic_not_real_benchmark"] is True


def test_preconditions_check_both_verify_and_arc_harness_imports() -> None:
    """REQ-PHASE4-007: preconditions fail closed on missing verify or harness imports."""
    seen: list[str] = []

    def ok_import(name: str) -> object:
        seen.append(name)
        return object()

    ok = exp3929.check_preconditions(import_fn=ok_import)
    assert ok.carnot_verify_imported is True
    assert ok.arc_harness_imported is True
    assert ok.blocked_resource == ""
    assert seen == ["carnot.verify", "carnot.agentic.arc_agi3_harness"]

    def fail_verify(name: str) -> object:
        if name == "carnot.verify":
            raise ModuleNotFoundError("verify missing")
        return object()

    verify_blocked = exp3929.check_preconditions(import_fn=fail_verify)
    assert verify_blocked.blocked_resource == "blocked_carnot_verify_import"
    assert verify_blocked.arc_harness_imported is False

    def fail_harness(name: str) -> object:
        if name == "carnot.agentic.arc_agi3_harness":
            raise ModuleNotFoundError("harness missing")
        return object()

    harness_blocked = exp3929.check_preconditions(import_fn=fail_harness)
    assert harness_blocked.carnot_verify_imported is True
    assert harness_blocked.arc_harness_imported is False
    assert harness_blocked.blocked_resource == "blocked_arc_harness_import"


def test_real_benchmark_preflight_records_reachable_and_error_notes() -> None:
    """REQ-PHASE4-007: preflight records official access reachability only."""

    class Response:
        status = 204
        url = "https://three.arcprize.org/environments"

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

    def ok_open(_request: object, *, timeout: float) -> Response:
        assert timeout == 1.5
        return Response()

    reachable = exp3929.probe_real_benchmark_access(opener=ok_open, timeout_s=1.5)
    assert reachable.reachable is True
    assert "HTTP 204" in reachable.note

    def http_error(_request: object, *, timeout: float) -> object:
        raise HTTPError("https://three.arcprize.org", 403, "Forbidden", {}, None)

    forbidden = exp3929.probe_real_benchmark_access(opener=http_error)
    assert forbidden.reachable is True
    assert "HTTP 403" in forbidden.note

    def url_error(_request: object, *, timeout: float) -> object:
        raise URLError("network unavailable")

    unavailable = exp3929.probe_real_benchmark_access(opener=url_error)
    assert unavailable.reachable is False
    assert "network unavailable" in unavailable.note


def test_artifact_writer_round_trips_json(tmp_path: Path) -> None:
    """SCENARIO-PHASE4-007: result artifacts are written as plain JSON."""
    artifact = {"experiment": 3929, "honest_verdict": "complete: test"}
    output = exp3929.write_result_artifact(artifact, tmp_path / "exp3929.json")

    assert json.loads(output.read_text(encoding="utf-8")) == artifact


def test_measurement_rejects_too_few_episodes_and_empty_candidates() -> None:
    """REQ-PHASE4-007: measurement gates reject undersized or malformed runs."""
    config = exp3929.build_episode_configs(1, random_seed=3929)[0]
    observation = exp3929.RichSyntheticArcEnv(config).reset()

    with pytest.raises(ValueError, match="at least 30"):
        exp3929.run_action_efficiency_measurement(
            n_episodes=29,
            verifier_fn=fake_energy_verifier,
        )

    with pytest.raises(ValueError, match="candidate action"):
        exp3929.select_verifier_pruned_action(
            observation,
            (),
            verifier_fn=fake_energy_verifier,
        )

    with pytest.raises(ValueError, match="candidate action"):
        exp3929.RandomGreedyNoVerifierPolicy(random_seed=3929).select_action(
            observation,
            (),
        )

    with pytest.raises(ValueError, match="paired verifier and baseline"):
        exp3929.bootstrap_ratio_ci((), (), random_seed=3929, resamples=10)

    verifier_unsolved = exp3929.run_verifier_episode(
        config,
        verifier_fn=fake_energy_verifier,
        max_steps=0,
    )
    baseline_unsolved = exp3929.run_baseline_episode(config, random_seed=3929, max_steps=0)

    assert verifier_unsolved.solved is False
    assert verifier_unsolved.actions_taken == ()
    assert baseline_unsolved.solved is False
    assert baseline_unsolved.actions_taken == ()
