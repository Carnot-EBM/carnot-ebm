"""Tests for Exp 1210: Phase 4 vs BFS on BFS-intractable scrambled-grid puzzles.

Spec: REQ-KONA-016, SCENARIO-KONA-016
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.phase3.scrambled_grid_puzzle import (
    BFS_INTRACTABLE_STATE_LIMIT,
    DEFAULT_GRID_SIZE,
    DEFAULT_LATENT_DIM,
    DEFAULT_N_PUZZLES,
    DEFAULT_N_SCRAMBLE_STEPS,
    FlipAction,
    Phase4Result,
    apply_action,
    bfs_solve,
    build_action_space,
    build_artifact,
    generate_puzzle_batch,
    generate_scrambled_puzzle,
    hamming_to_goal,
    phase4_solve,
    run_experiment,
    write_artifact,
    _GreedyOverActionsSampler,
    _grid_to_bitmask,
    _latent_for_cell,
    _snap_latent_to_action,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_action_space_size_matches_grid_size_squared() -> None:
    # REQ-KONA-016: branching = grid_size**2 (every cell flippable).
    actions = build_action_space(grid_size=4)
    assert len(actions) == 16
    assert all(isinstance(a, FlipAction) for a in actions)


def test_action_space_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        build_action_space(grid_size=0)
    with pytest.raises(ValueError):
        build_action_space(grid_size=3, latent_dim=0)


def test_apply_action_toggles_single_cell_mod2() -> None:
    grid = ((0, 0), (0, 0))
    actions = build_action_space(grid_size=2)
    next_grid = apply_action(grid, actions[0])
    assert next_grid[0][0] == 1
    again = apply_action(next_grid, actions[0])
    assert again == grid  # self-inverse


def test_apply_action_rejects_out_of_bounds() -> None:
    grid = ((0, 0), (0, 0))
    bogus = FlipAction(name="bad", row=5, col=5, latent=(0.0,) * DEFAULT_LATENT_DIM)
    with pytest.raises(ValueError):
        apply_action(grid, bogus)


def test_hamming_to_goal_counts_differences() -> None:
    goal = ((0, 0), (0, 0))
    assert hamming_to_goal(goal, goal) == 0
    assert hamming_to_goal(((1, 0), (0, 0)), goal) == 1
    assert hamming_to_goal(((1, 1), (1, 1)), goal) == 4


def test_grid_to_bitmask_matches_xor_semantics() -> None:
    # SCENARIO-KONA-016: BFS uses bitmask dedup; mask of empty grid is 0.
    assert _grid_to_bitmask(()) == 0
    assert _grid_to_bitmask(((0, 0), (0, 0))) == 0
    grid = ((1, 0), (0, 1))
    mask = _grid_to_bitmask(grid)
    # bits 0 (row 0 col 0) and 3 (row 1 col 1) are set
    assert mask == (1 << 0) | (1 << 3)


def test_latent_for_cell_is_unit_bounded_and_deterministic() -> None:
    a = _latent_for_cell(2, 3, grid_size=5, latent_dim=DEFAULT_LATENT_DIM)
    b = _latent_for_cell(2, 3, grid_size=5, latent_dim=DEFAULT_LATENT_DIM)
    assert a == b
    assert all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in a)


def test_snap_latent_to_action_picks_nearest() -> None:
    actions = build_action_space(grid_size=3, latent_dim=DEFAULT_LATENT_DIM)
    target = actions[4]
    snapped = _snap_latent_to_action(np.asarray(target.latent), actions)
    assert snapped is target


def test_generate_scrambled_puzzle_has_positive_initial_energy() -> None:
    # REQ-KONA-016: every generated puzzle must have initial_energy > 0.
    puzzle = generate_scrambled_puzzle(
        puzzle_id="t",
        grid_size=15,
        n_scramble_steps=50,
        seed=1210000,
    )
    assert puzzle.initial_energy > 0
    assert puzzle.grid_size == 15
    assert puzzle.n_scramble_steps == 50


def test_generate_scrambled_puzzle_fallback_flips_when_scramble_undoes_itself() -> None:
    # When scrambling happens to land on goal, the deterministic fallback
    # flip ensures initial_energy > 0 anyway. We construct a tiny grid with
    # scramble steps designed to (sometimes) cancel out and still expect
    # initial_energy > 0.
    for seed in range(20):
        puzzle = generate_scrambled_puzzle(
            puzzle_id=f"tiny_{seed}",
            grid_size=2,
            n_scramble_steps=2,
            seed=seed,
        )
        assert puzzle.initial_energy > 0


def test_generate_scrambled_puzzle_rejects_invalid_args() -> None:
    with pytest.raises(ValueError):
        generate_scrambled_puzzle(puzzle_id="x", grid_size=0)
    with pytest.raises(ValueError):
        generate_scrambled_puzzle(puzzle_id="x", grid_size=3, n_scramble_steps=0)


def test_generate_puzzle_batch_size_and_uniqueness() -> None:
    puzzles = generate_puzzle_batch(n_puzzles=5, grid_size=10, n_scramble_steps=20, base_seed=42)
    assert len(puzzles) == 5
    assert all(p.initial_energy > 0 for p in puzzles)
    assert len({p.puzzle_id for p in puzzles}) == 5
    with pytest.raises(ValueError):
        generate_puzzle_batch(n_puzzles=0)


def test_bfs_solve_short_circuits_when_initial_equals_goal() -> None:
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=3, n_scramble_steps=2, seed=0)
    # Manually build a puzzle whose initial == goal to hit the short-circuit.
    same_goal = puzzle.__class__(
        puzzle_id="solved",
        grid_size=3,
        n_scramble_steps=2,
        initial_grid=puzzle.goal_grid,
        goal_grid=puzzle.goal_grid,
        initial_energy=0,
    )
    result = bfs_solve(same_goal, state_limit=10)
    assert result.solved
    assert result.actions == ()
    assert result.n_states_explored == 0


def test_bfs_solve_finds_goal_for_small_puzzle() -> None:
    # For a 2x2 puzzle scrambled by 1 flip, BFS should find a 1-action solution.
    actions = build_action_space(grid_size=2)
    goal = ((0, 0), (0, 0))
    initial = apply_action(goal, actions[0])
    from carnot.phase3.scrambled_grid_puzzle import ScrambledPuzzle

    puzzle = ScrambledPuzzle(
        puzzle_id="t",
        grid_size=2,
        n_scramble_steps=1,
        initial_grid=initial,
        goal_grid=goal,
        initial_energy=1,
    )
    result = bfs_solve(puzzle, state_limit=100)
    assert result.solved
    assert result.actions == (actions[0].name,)
    assert not result.intractable


def test_bfs_solve_reports_intractable_with_tiny_state_limit() -> None:
    # SCENARIO-KONA-016: intractable=True when more than state_limit pops.
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=10, n_scramble_steps=30, seed=2)
    result = bfs_solve(puzzle, state_limit=5)
    assert result.intractable
    assert result.actions is None
    assert result.n_states_explored > 5


def test_bfs_solve_rejects_invalid_state_limit() -> None:
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=3, n_scramble_steps=1, seed=1)
    with pytest.raises(ValueError):
        bfs_solve(puzzle, state_limit=0)


def test_phase4_solve_solves_small_puzzle_and_records_trace() -> None:
    # SCENARIO-KONA-016: Phase 4 trace starts > 0 and decreases to 0.
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=4, n_scramble_steps=8, seed=11)
    result = phase4_solve(puzzle, max_actions=50, n_gibbs_sweeps=20, seed=11)
    assert result.initial_energy_positive
    assert result.energy_trace[0] == puzzle.initial_energy
    assert result.solved
    assert result.energy_trace[-1] == 0
    # Greedy descent on Hamming distance: trace must be monotone non-increasing.
    diffs = np.diff(result.energy_trace)
    assert all(d <= 0 for d in diffs)


def test_phase4_solve_short_circuits_when_grid_starts_at_goal() -> None:
    from carnot.phase3.scrambled_grid_puzzle import ScrambledPuzzle

    goal = ((0, 0), (0, 0))
    puzzle = ScrambledPuzzle(
        puzzle_id="solved",
        grid_size=2,
        n_scramble_steps=0,
        initial_grid=goal,
        goal_grid=goal,
        initial_energy=0,
    )
    result = phase4_solve(puzzle, max_actions=5, n_gibbs_sweeps=3, seed=0)
    assert result.solved
    assert result.n_actions == 0
    assert tuple(result.energy_trace) == (0,)


def test_phase4_solve_rejects_invalid_max_actions() -> None:
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=3, n_scramble_steps=2, seed=0)
    with pytest.raises(ValueError):
        phase4_solve(puzzle, max_actions=0)


def test_phase4_solve_caps_at_max_actions() -> None:
    # When max_actions is below the initial energy, Phase 4 cannot finish.
    puzzle = generate_scrambled_puzzle(puzzle_id="t", grid_size=4, n_scramble_steps=8, seed=11)
    result = phase4_solve(puzzle, max_actions=1, n_gibbs_sweeps=5, seed=11)
    assert result.n_actions == 1
    # If initial energy was > 1, the puzzle won't be solved after only 1 action.
    if puzzle.initial_energy > 1:
        assert not result.solved


def test_greedy_sampler_validates_inputs() -> None:
    actions = build_action_space(grid_size=2, latent_dim=DEFAULT_LATENT_DIM)
    with pytest.raises(ValueError):
        _GreedyOverActionsSampler([], latent_dim=DEFAULT_LATENT_DIM)
    with pytest.raises(ValueError):
        _GreedyOverActionsSampler(actions, latent_dim=0)
    sampler = _GreedyOverActionsSampler(actions, latent_dim=DEFAULT_LATENT_DIM)
    with pytest.raises(ValueError):
        sampler.sample(lambda z: 0.0, np.zeros(DEFAULT_LATENT_DIM), 0)


def test_greedy_sampler_returns_chain_of_top_k_action_latents() -> None:
    actions = build_action_space(grid_size=3, latent_dim=DEFAULT_LATENT_DIM)
    sampler = _GreedyOverActionsSampler(actions, latent_dim=DEFAULT_LATENT_DIM)
    target = actions[2]

    def energy_fn(z: np.ndarray) -> float:
        delta = np.asarray(target.latent) - np.asarray(z)
        return float(np.dot(delta, delta))

    chain = sampler.sample(energy_fn, np.zeros(DEFAULT_LATENT_DIM), n_steps=3)
    assert chain.shape == (3, DEFAULT_LATENT_DIM)
    # The minimum-energy latent (the target's own latent) must come first.
    assert np.allclose(chain[0], np.asarray(target.latent))


def test_greedy_sampler_rejects_mismatched_latent_dim() -> None:
    bad_actions = [
        FlipAction(name="x", row=0, col=0, latent=(0.0, 0.0)),
    ]
    with pytest.raises(ValueError):
        _GreedyOverActionsSampler(bad_actions, latent_dim=DEFAULT_LATENT_DIM)


def test_build_artifact_validates_lengths() -> None:
    with pytest.raises(ValueError):
        build_artifact(
            puzzles=[],
            bfs_results=[None],  # type: ignore[list-item]
            phase4_results=[],
            blocked_gibbs_params={},
            grid_size=15,
            n_scramble_steps=50,
        )


def _toy_run(n_puzzles: int = 3, grid_size: int = 4, scramble: int = 8) -> dict:
    return run_experiment(
        n_puzzles=n_puzzles,
        grid_size=grid_size,
        n_scramble_steps=scramble,
        bfs_state_limit=50,
        max_phase4_actions=40,
        n_gibbs_sweeps=10,
        base_seed=42,
    )


def test_run_experiment_emits_required_artifact_fields() -> None:
    artifact = _toy_run()
    required_fields = {
        "n_puzzles_total",
        "grid_size",
        "n_scramble_steps",
        "initial_energy_nonzero_fraction",
        "bfs_intractable_count",
        "bfs_intractable_fraction",
        "phase4_solved_on_intractable",
        "phase4_energy_traces_all_nonzero_initial",
        "phase4_bfs_intractable_fraction_above_50pct",
        "honest_verdict",
    }
    assert required_fields <= set(artifact.keys())
    assert artifact["n_puzzles_total"] == 3
    assert artifact["initial_energy_nonzero_fraction"] == 1.0
    assert artifact["honest_verdict"] in {
        "phase4_advantage_on_intractable",
        "phase4_tied_with_bfs_again",
        "puzzle_generator_fixed_but_bfs_still_tractable",
        "blocked",
    }


def test_run_experiment_per_puzzle_rows_have_positive_initial_energy() -> None:
    artifact = _toy_run()
    for row in artifact["per_puzzle"]:
        assert row["initial_energy"] > 0


def test_write_artifact_round_trip(tmp_path: Path) -> None:
    artifact = _toy_run()
    target = tmp_path / "exp1210.json"
    written = write_artifact(artifact, target)
    assert written.exists()
    parsed = json.loads(written.read_text(encoding="utf-8"))
    assert parsed["n_puzzles_total"] == artifact["n_puzzles_total"]
    assert parsed["honest_verdict"] == artifact["honest_verdict"]


def test_artifact_schema_includes_paper_narrative_and_per_puzzle() -> None:
    # SCENARIO-KONA-016: the artifact must surface a paper-ready narrative
    # and a per-puzzle row breakdown so downstream readers can audit any
    # individual puzzle's BFS / Phase 4 outcome.
    artifact = _toy_run(n_puzzles=2, grid_size=3, scramble=4)
    assert "paper_narrative" in artifact
    assert isinstance(artifact["paper_narrative"], str)
    assert artifact["per_puzzle"] and len(artifact["per_puzzle"]) == 2
    for row in artifact["per_puzzle"]:
        assert "phase4_energy_trace" in row
        assert row["phase4_energy_trace"][0] == row["initial_energy"]
