"""Tests for the Phase 4 active-inference pilot.

Spec coverage: REQ-KONA-012, SCENARIO-KONA-012
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.active_inference_pilot import (
    ARC3Action,
    ARC3PuzzleEnv,
    ActiveInferencePilot,
    LATENT_DIM,
    build_default_k5_ensemble_energies,
    build_experiment_artifact,
    energy_trace_monotone,
    run_phase4_vs_baseline,
    run_random_baseline_episode,
    write_experiment_artifact,
)
from carnot.phase3.snap_validity import snap_to_action


class RecordingSampler:
    def __init__(self, chain: np.ndarray | None = None) -> None:
        self.chain = (
            np.asarray(chain, dtype=np.float64)
            if chain is not None
            else np.zeros((2, LATENT_DIM), dtype=np.float64)
        )
        self.calls: list[tuple[int, tuple[float, ...]]] = []

    def sample(self, energy_fn, init_state, n_steps):  # type: ignore[no-untyped-def]
        self.calls.append((int(n_steps), tuple(np.asarray(init_state, dtype=np.float64))))
        assert np.isfinite(float(energy_fn(np.asarray(init_state, dtype=np.float64))))
        return self.chain


def test_arc3_env_exposes_ten_5x5_puzzles_with_finite_solution_traces() -> None:
    """REQ-KONA-012: the stub has ten finite 5x5 ARC-like puzzle traces."""
    env = ARC3PuzzleEnv()

    assert len(env.puzzle_ids) == 10
    for puzzle_id in env.puzzle_ids:
        state = env.reset(puzzle_id)
        assert np.asarray(state.grid).shape == (5, 5)
        steps = 0
        done = False
        while not done:
            legal = env.legal_actions(state)
            assert 3 <= len(legal) <= 5
            expected = next(action for action in legal if action.name == state.expected_action_name)
            state, done, info = env.step(state, expected)
            assert info["correct"] is True
            steps += 1
        assert 3 <= steps <= 10


def test_env_rejects_unknown_puzzles_and_nonlegal_actions() -> None:
    """REQ-KONA-012: environment transitions stay inside declared legal actions."""
    env = ARC3PuzzleEnv()
    state = env.reset("color_fill")
    wrong = next(
        action for action in env.legal_actions(state) if action.name != state.expected_action_name
    )
    next_state, done, info = env.step(state, wrong)

    assert next_state.step_index == state.step_index
    assert done is False
    assert info["correct"] is False

    with pytest.raises(KeyError, match="unknown puzzle_id"):
        env.reset("missing")
    with pytest.raises(ValueError, match="action is not legal"):
        env.step(state, ARC3Action("illegal", tuple(np.ones(LATENT_DIM)), "bad", -1, -1))


def test_snap_to_action_selects_nearest_object_and_array_action() -> None:
    """SCENARIO-KONA-012: latent states snap to the nearest legal action."""
    left = ARC3Action("left", tuple(-np.ones(LATENT_DIM)), "move", 0, 0)
    right = ARC3Action("right", tuple(np.ones(LATENT_DIM)), "move", 1, 0)

    assert snap_to_action(np.full(LATENT_DIM, 0.9), [left, right]) == right
    snapped_array = snap_to_action(
        np.array([0.8, -0.2]),
        [np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
    )
    assert np.allclose(snapped_array, np.array([1.0, 0.0]))


def test_minimize_free_energy_uses_sampler_snap_weights_and_best_candidate() -> None:
    """REQ-KONA-012: free-energy minimisation returns the best snapped action."""
    env = ARC3PuzzleEnv()
    state = env.reset("pattern_copy")
    legal = env.legal_actions(state)
    expected = next(action for action in legal if action.name == state.expected_action_name)
    decoy = next(action for action in legal if action.name != state.expected_action_name)
    sampler = RecordingSampler(np.asarray([decoy.latent], dtype=np.float64))

    def exact_energy(action, board_state):
        return 0.0 if action.name == board_state.expected_action_name else 4.0

    def one_arg_energy(action):
        return 0.0 if action.name == expected.name else 2.0

    pilot = ActiveInferencePilot(
        [exact_energy, one_arg_energy],
        snap_to_action,
        sampler,
        rng_seed=1,
    )
    pilot.bind_board_state(state)
    z_minimized, trace = pilot.minimize_free_energy(
        np.asarray(decoy.latent, dtype=np.float64),
        n_gibbs_sweeps=3,
        weights=[0.5, 0.5],
    )

    assert sampler.calls[0][0] == 3
    assert snap_to_action(z_minimized, legal).name == expected.name
    assert trace[0] > trace[-1]
    assert energy_trace_monotone(trace)


def test_pilot_validates_configuration_before_sampling() -> None:
    """REQ-KONA-012: invalid pilot inputs fail clearly."""
    env = ARC3PuzzleEnv()
    state = env.reset("row_rotate")
    sampler = RecordingSampler()

    with pytest.raises(ValueError, match="at least one verifier energy"):
        ActiveInferencePilot([], snap_to_action, sampler)

    pilot = ActiveInferencePilot(build_default_k5_ensemble_energies(), snap_to_action, sampler)
    with pytest.raises(RuntimeError, match="bind_board_state"):
        pilot.minimize_free_energy(np.zeros(LATENT_DIM), n_gibbs_sweeps=1, weights=None)

    pilot.bind_board_state(state)
    with pytest.raises(ValueError, match="weights length"):
        pilot.minimize_free_energy(np.zeros(LATENT_DIM), n_gibbs_sweeps=1, weights=[1.0])
    with pytest.raises(ValueError, match="n_gibbs_sweeps"):
        pilot.minimize_free_energy(np.zeros(LATENT_DIM), n_gibbs_sweeps=0, weights=None)


def test_select_action_and_run_episode_solve_with_default_verifiers() -> None:
    """SCENARIO-KONA-012: Phase 4 selects the expected action and solves episodes."""
    env = ARC3PuzzleEnv()
    state = env.reset("color_fill")
    pilot = ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        RecordingSampler(),
        rng_seed=2,
    )

    action, z_minimized, free_energy = pilot.select_action(state, n_gibbs_sweeps=2)
    result = pilot.run_episode("color_fill", max_actions=10, n_gibbs_sweeps=2)
    unsolved = pilot.run_episode("color_fill", max_actions=1, n_gibbs_sweeps=2)

    assert action.name == state.expected_action_name
    assert np.max(np.abs(z_minimized)) <= 1.0
    assert free_energy == pytest.approx(0.0)
    assert result.solved is True
    assert result.action_count == 3
    assert len(result.actions_taken) == 3
    assert unsolved.solved is False
    assert unsolved.action_count == 1


def test_baseline_and_experiment_artifact_report_required_fields(tmp_path: Path) -> None:
    """REQ-KONA-012: experiment summaries emit the required result schema."""
    env = ARC3PuzzleEnv()
    pilot = ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        RecordingSampler(),
        rng_seed=3,
    )

    baseline = run_random_baseline_episode(
        env,
        "color_fill",
        np.random.default_rng(4),
        max_actions=20,
    )
    summary = run_phase4_vs_baseline(
        pilot,
        env,
        n_episodes=1,
        max_actions=20,
        n_gibbs_sweeps=2,
        baseline_seed=5,
    )
    artifact = build_experiment_artifact(
        summary,
        blocked_gibbs_params={"n_sweeps": 2, "n_blocks": LATENT_DIM, "step_size": 0.01},
    )
    output_path = write_experiment_artifact(artifact, tmp_path / "artifact.json")

    assert baseline.action_count >= 3
    assert artifact["prototype_operational"] is True
    assert artifact["n_puzzles_evaluated"] == 10
    assert artifact["phase4_mean_action_count"] < artifact["baseline_mean_action_count"]
    assert artifact["action_count_ratio"] < 1.0
    assert artifact["honest_verdict"] == "phase4_better_than_baseline"
    assert "Seed IQ" in artifact["comparison_to_seed_iq"]
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

    blocked = build_experiment_artifact(
        {
            "phase4": [type(baseline)(1, False, [], [])],
            "baseline": [type(baseline)(1, True, [], [])],
            "n_puzzles": 10,
        },
        blocked_gibbs_params={"n_sweeps": 2, "n_blocks": LATENT_DIM, "step_size": 0.01},
    )
    assert blocked["honest_verdict"] == "prototype_only_no_convergence"

    worse = build_experiment_artifact(
        {
            "phase4": [type(baseline)(3, True, [3.0], [])],
            "baseline": [type(baseline)(1, True, [], [])],
            "n_puzzles": 10,
        },
        blocked_gibbs_params={"n_sweeps": 2, "n_blocks": LATENT_DIM, "step_size": 0.01},
    )
    assert worse["honest_verdict"] == "phase4_worse_than_baseline"
