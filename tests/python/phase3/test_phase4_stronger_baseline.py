"""Tests for the Phase 4 stronger BFS baseline at 5x5 and 10x10.

Spec coverage: REQ-KONA-015, SCENARIO-KONA-015
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.active_inference_pilot import (
    ARC3PuzzleEnv,
    ActiveInferencePilot,
    BFSBaseline,
    BFSResult,
    BFS_INTRACTABLE_STATE_LIMIT,
    EpisodeResult,
    LATENT_DIM,
    SUPPORTED_GRID_SIZES,
    _ratio_and_wins,
    _stronger_baseline_verdict,
    build_default_k5_ensemble_energies,
    build_stronger_baseline_artifact,
    run_phase4_vs_bfs,
    write_experiment_artifact,
)
from carnot.phase3.snap_validity import snap_to_action


class _NullSampler:
    """Minimal blocked-Gibbs stand-in: returns the start state, never moves."""

    def sample(self, energy_fn, init_state, n_steps):  # type: ignore[no-untyped-def]
        # Touch the energy at init to satisfy the contract that ``energy_fn`` is
        # finite at the start state, mirroring the production sampler's first
        # diagnostic call. The result is unused.
        assert np.isfinite(float(energy_fn(np.asarray(init_state, dtype=np.float64))))
        return np.asarray([init_state], dtype=np.float64)


def test_arc3_env_supports_5_and_10_grid_sizes() -> None:
    """REQ-KONA-015: env exposes both grid sizes with the documented bounds."""
    assert SUPPORTED_GRID_SIZES == (5, 10)
    env5 = ARC3PuzzleEnv()  # default 5x5
    env10 = ARC3PuzzleEnv(grid_size=10)

    assert env5.grid_size == 5
    assert env10.grid_size == 10
    assert len(env5.puzzle_ids) == 10
    assert len(env10.puzzle_ids) == 10
    assert all(p.endswith("_10x10") for p in env10.puzzle_ids)
    assert set(env5.puzzle_ids).isdisjoint(set(env10.puzzle_ids))

    state5 = env5.reset(env5.puzzle_ids[0])
    state10 = env10.reset(env10.puzzle_ids[0])
    assert np.asarray(state5.grid).shape == (5, 5)
    assert np.asarray(state10.grid).shape == (10, 10)
    assert 5 <= len(env10.legal_actions(state10)) <= 8


def test_arc3_env_rejects_unsupported_grid_size() -> None:
    """REQ-KONA-015: only 5 and 10 are accepted; other values fail clearly."""
    with pytest.raises(ValueError, match="grid_size must be one of"):
        ARC3PuzzleEnv(grid_size=7)


def test_10x10_wrong_action_mutates_grid_so_bfs_branches() -> None:
    """REQ-KONA-015: 10x10 wrong moves must produce a new grid for BFS to branch."""
    env = ARC3PuzzleEnv(grid_size=10)
    state = env.reset(env.puzzle_ids[0])
    legal = env.legal_actions(state)
    wrong = next(action for action in legal if action.name != state.expected_action_name)
    next_state, done, info = env.step(state, wrong)

    assert info["correct"] is False
    assert done is False
    assert next_state.step_index == state.step_index
    assert next_state.grid != state.grid


def test_5x5_wrong_action_preserves_grid_for_backward_compat() -> None:
    """REQ-KONA-015: 5x5 wrong moves leave the grid unchanged (Exp 1165 behavior)."""
    env = ARC3PuzzleEnv()
    state = env.reset(env.puzzle_ids[0])
    legal = env.legal_actions(state)
    wrong = next(action for action in legal if action.name != state.expected_action_name)
    next_state, _, info = env.step(state, wrong)
    assert info["correct"] is False
    assert next_state.grid == state.grid


def test_bfs_baseline_solves_5x5_in_solution_length_actions() -> None:
    """REQ-KONA-015: BFS finds the optimal path on tractable 5x5 puzzles."""
    env = ARC3PuzzleEnv()
    bfs = BFSBaseline()
    for puzzle_id in env.puzzle_ids:
        result = bfs.bfs_solve(env, puzzle_id)
        assert result.solved
        assert result.intractable is False
        # The optimum equals the length of the canonical solution trace.
        assert len(result.actions) == len(env.puzzles[puzzle_id].solution_names)


def test_bfs_baseline_marks_intractable_when_state_cap_exceeded() -> None:
    """REQ-KONA-015: BFS reports intractable when the popped-state cap is hit."""
    env = ARC3PuzzleEnv(grid_size=10)
    # Length-10 puzzle with branching 5..8 cannot fit under a 50-state budget.
    bfs = BFSBaseline(state_limit=50)
    result = bfs.bfs_solve(env, env.puzzle_ids[6])
    assert result.intractable is True
    assert result.actions is None
    assert result.solved is False
    assert result.n_states_explored > 50


def test_bfs_baseline_reports_unsolved_when_queue_exhausts() -> None:
    """REQ-KONA-015: BFS can report no path without marking intractability."""
    env = ARC3PuzzleEnv()
    initial = env.reset(env.puzzle_ids[0])

    class _NoSolutionEnv:
        def reset(self, puzzle_id):  # type: ignore[no-untyped-def]
            return initial

        def legal_actions(self, board_state):  # type: ignore[no-untyped-def]
            return []

        def step(self, board_state, action):  # type: ignore[no-untyped-def]
            raise AssertionError("no legal actions should be stepped")

    result = BFSBaseline().bfs_solve(_NoSolutionEnv(), initial.puzzle_id)  # type: ignore[arg-type]
    assert result.solved is False
    assert result.actions is None
    assert result.intractable is False
    assert result.n_states_explored == 1


def test_bfs_baseline_validates_state_limit() -> None:
    """REQ-KONA-015: state_limit must be positive."""
    with pytest.raises(ValueError, match="state_limit must be positive"):
        BFSBaseline(state_limit=0)


def test_bfs_returns_empty_actions_when_initial_state_already_solved() -> None:
    """Edge case: a puzzle that starts solved returns an empty action sequence."""
    env = ARC3PuzzleEnv()
    state = env.reset("color_fill")
    # color_fill has a 3-step solution; advance it manually to the goal.
    while not state.solved:
        expected = next(a for a in env.legal_actions(state) if a.name == state.expected_action_name)
        state, _, _ = env.step(state, expected)

    class _GoalEnv:
        grid_size = env.grid_size
        puzzles = env.puzzles

        def reset(self, puzzle_id):  # type: ignore[no-untyped-def]
            return state

        def legal_actions(self, board_state):  # type: ignore[no-untyped-def]
            return env.legal_actions(board_state)

        def step(self, board_state, action):  # type: ignore[no-untyped-def]
            return env.step(board_state, action)

    bfs = BFSBaseline()
    result = bfs.bfs_solve(_GoalEnv(), "color_fill")  # type: ignore[arg-type]
    assert result.solved is True
    assert result.actions == ()
    assert result.intractable is False


def _make_pilot(env: ARC3PuzzleEnv) -> ActiveInferencePilot:
    return ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        _NullSampler(),
        latent_dim=LATENT_DIM,
        rng_seed=1189,
    )


def test_run_phase4_vs_bfs_emits_per_puzzle_rows_with_required_fields() -> None:
    """REQ-KONA-015: per-puzzle row reports both methods + full energy trace."""
    env = ARC3PuzzleEnv()
    pilot = _make_pilot(env)
    bfs = BFSBaseline()
    rows = run_phase4_vs_bfs(pilot, env, bfs, max_actions=20, n_gibbs_sweeps=2)

    assert len(rows) == 10
    for row in rows:
        for field in (
            "puzzle_id",
            "grid_size",
            "phase4_action_count",
            "phase4_solved",
            "phase4_energy_trace",
            "phase4_actions",
            "bfs_action_count",
            "bfs_solved",
            "bfs_states_explored",
            "bfs_intractable",
        ):
            assert field in row, f"missing {field} in row {row['puzzle_id']}"
        assert row["bfs_solved"] is True
        assert len(row["phase4_energy_trace"]) > 0
        # Phase 4 with the perfect-info verifier matches BFS optimum on 5x5.
        assert row["phase4_action_count"] == row["bfs_action_count"]


def test_ratio_and_wins_handles_mixed_intractable_and_comparable_rows() -> None:
    """REQ-KONA-015: action-ratio reduction skips intractable + unsolved rows."""
    rows = [
        {
            "phase4_action_count": 5,
            "bfs_action_count": 5,
            "bfs_solved": True,
            "bfs_intractable": False,
        },
        {
            "phase4_action_count": 4,
            "bfs_action_count": 5,
            "bfs_solved": True,
            "bfs_intractable": False,
        },
        {
            "phase4_action_count": 7,
            "bfs_action_count": None,
            "bfs_solved": False,
            "bfs_intractable": True,
        },
        {
            "phase4_action_count": 9,
            "bfs_action_count": None,
            "bfs_solved": False,
            "bfs_intractable": False,
        },
    ]
    ratio, wins, comparable, intractable = _ratio_and_wins(rows)
    assert ratio == pytest.approx(9 / 10)  # 5+4 phase4 / 5+5 bfs
    assert wins == 1  # second row: 4 < 5
    assert comparable == 2
    assert intractable == 1


def test_ratio_and_wins_returns_inf_when_no_comparable_rows() -> None:
    """REQ-KONA-015: zero-comparable case returns inf, not silent zero ratio."""
    rows = [
        {
            "phase4_action_count": 9,
            "bfs_action_count": None,
            "bfs_solved": False,
            "bfs_intractable": True,
        }
    ]
    ratio, wins, comparable, intractable = _ratio_and_wins(rows)
    assert ratio == float("inf")
    assert wins == 0
    assert comparable == 0
    assert intractable == 1


def test_stronger_baseline_verdict_maps_each_outcome_class() -> None:
    """REQ-KONA-015: verdict mapping covers the four allowed outcomes."""
    # Tied (ratio ~ 1.0 on both, no intractability).
    assert (
        _stronger_baseline_verdict(ratio_5x5=1.0, ratio_10x10=1.0, intractable_10x10=0, n_10x10=10)
        == "phase4_tied_with_bfs"
    )
    # Phase 4 strictly better on 10x10 hard puzzles.
    assert (
        _stronger_baseline_verdict(ratio_5x5=1.0, ratio_10x10=0.5, intractable_10x10=0, n_10x10=10)
        == "phase4_beats_bfs_on_hard_puzzles"
    )
    # Phase 4 worse on every grid size.
    assert (
        _stronger_baseline_verdict(ratio_5x5=2.0, ratio_10x10=2.0, intractable_10x10=0, n_10x10=10)
        == "phase4_loses_to_bfs_all_sizes"
    )
    # Most 10x10 puzzles intractable for BFS.
    assert (
        _stronger_baseline_verdict(
            ratio_5x5=1.0,
            ratio_10x10=float("inf"),
            intractable_10x10=8,
            n_10x10=10,
        )
        == "bfs_mostly_intractable"
    )


def test_build_stronger_baseline_artifact_writes_required_fields(tmp_path: Path) -> None:
    """REQ-KONA-015 + SCENARIO-KONA-015: artifact carries every required field."""
    env5 = ARC3PuzzleEnv()
    env10 = ARC3PuzzleEnv(grid_size=10)
    pilot5 = _make_pilot(env5)
    pilot10 = ActiveInferencePilot(
        build_default_k5_ensemble_energies(),
        snap_to_action,
        _NullSampler(),
        latent_dim=LATENT_DIM,
        rng_seed=1189,
    )
    bfs_tractable = BFSBaseline()
    bfs_tight = BFSBaseline(state_limit=50)

    rows_5x5 = run_phase4_vs_bfs(pilot5, env5, bfs_tractable, max_actions=20, n_gibbs_sweeps=2)
    rows_10x10 = run_phase4_vs_bfs(pilot10, env10, bfs_tight, max_actions=20, n_gibbs_sweeps=2)

    artifact = build_stronger_baseline_artifact(
        rows_5x5,
        rows_10x10,
        blocked_gibbs_params={"n_sweeps": 2, "n_blocks": LATENT_DIM, "step_size": 0.01},
    )

    for field in (
        "bfs_baseline_implemented",
        "stronger_baseline_implemented",
        "grid_sizes_tested",
        "n_5x5_puzzles",
        "n_10x10_puzzles",
        "phase4_5x5_action_ratio",
        "phase4_10x10_action_ratio",
        "phase4_better_than_bfs_5x5",
        "phase4_better_than_bfs_10x10",
        "bfs_intractable_10x10",
        "free_energy_values_all_puzzles",
        "paper_narrative",
        "honest_verdict",
    ):
        assert field in artifact, f"missing {field}"
    assert artifact["bfs_baseline_implemented"] is True
    assert artifact["stronger_baseline_implemented"] is True
    assert artifact["grid_sizes_tested"] == [5, 10]
    assert artifact["n_5x5_puzzles"] == 10
    assert artifact["n_10x10_puzzles"] == 10
    assert artifact["free_energy_values_all_puzzles"] is True
    assert artifact["honest_verdict"] in {
        "phase4_beats_bfs_on_hard_puzzles",
        "phase4_tied_with_bfs",
        "phase4_loses_to_bfs_all_sizes",
        "bfs_mostly_intractable",
    }
    assert "ISSUE-9" in artifact["paper_narrative"]
    assert artifact["bfs_intractable_10x10"] >= 1  # tight cap forces some intractable

    out_path = write_experiment_artifact(artifact, tmp_path / "exp1189.json")
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact


def test_bfs_result_solved_property_reflects_actions_field() -> None:
    """REQ-KONA-015: BFSResult.solved is the truthiness of actions."""
    solved = BFSResult("p", (), 0, False)
    unsolved = BFSResult("p", None, 5, True)
    assert solved.solved is True
    assert unsolved.solved is False
    assert BFS_INTRACTABLE_STATE_LIMIT == 100_000


def test_episode_result_export_helper_used_by_artifact() -> None:
    """Sanity: EpisodeResult is the type produced by run_episode in this module."""
    result = EpisodeResult(2, True, [1.0, 0.5], ["a", "b"])
    assert result.action_count == 2
    assert result.solved is True
