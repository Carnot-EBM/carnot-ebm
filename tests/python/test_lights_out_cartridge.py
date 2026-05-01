"""Tests for the Lights Out WOPR cartridge (Ising ground-state search).

Spec: REQ-WOPR-002 — Lights Out cartridge Ising ground-state search.

Each test traces to REQ-LIGHTS-OUT-001: the cartridge must use
ParallelIsingSampler to reach E=0 within 10000 iterations.
"""

import sys
import os

import pytest

# Add the spaces/wopr-games directory so games/ is importable
_WOPR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "spaces", "wopr-games")
if _WOPR_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(_WOPR_DIR))

from games.lights_out import LightsOutGame, lights_out_energy


def _run_solve(seed: int = 17, max_iterations: int = 10000):
    """Helper: instantiate game, solve, return list of (energy,) per step."""
    game = LightsOutGame(seed=seed)
    steps = game.carnot_solve(max_iterations=max_iterations)
    return game, steps


def test_energy_decreases_monotonically_during_solve():
    """Energy must be non-increasing at every step of the solve.

    The Ising-derived solution is replayed in greedy order so each press
    reduces (or at worst maintains) the lit-cell count.
    REQ-LIGHTS-OUT-001 / SCENARIO-MONOTONE-DESCENT
    """
    _, steps = _run_solve()
    assert steps, "carnot_solve returned no steps"
    for i in range(1, len(steps)):
        assert steps[i].energy <= steps[i - 1].energy, (
            f"Energy increased at step {i}: {steps[i - 1].energy} -> {steps[i].energy}"
        )


def test_final_energy_zero_when_solved():
    """The last step must have energy == 0.0.

    REQ-LIGHTS-OUT-001 / SCENARIO-GROUND-STATE
    """
    _, steps = _run_solve()
    assert steps, "carnot_solve returned no steps"
    assert steps[-1].energy == 0.0, f"Final energy is {steps[-1].energy}, expected 0.0"


def test_all_cells_dark_at_solution():
    """Every cell must be False (dark) in the solved state.

    REQ-LIGHTS-OUT-001 / SCENARIO-ALL-DARK
    """
    _, steps = _run_solve()
    assert steps, "carnot_solve returned no steps"
    final_state = steps[-1].state
    lit = [(r, c) for r in range(5) for c in range(5) if final_state.grid[r][c]]
    assert lit == [], f"Cells still lit at solution: {lit}"


def test_ising_sampler_reaches_ground_state_within_10000_iters():
    """The Ising-powered solver must reach E=0 within 10000 carnot_step calls.

    For a 5x5 = 25-spin system, the solution has at most 25 button presses,
    so this bound is very conservative.
    REQ-LIGHTS-OUT-001 / SCENARIO-EFFICIENCY
    """
    game, steps = _run_solve(max_iterations=10000)
    solved = any(s.is_solved for s in steps)
    assert solved, (
        f"Did not reach E=0 within 10000 iterations. "
        f"Best energy: {min(s.energy for s in steps) if steps else 'N/A'}"
    )
    assert game.ising_solver_used, "ParallelIsingSampler was not invoked"
