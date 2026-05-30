"""Tests for the Kona global-opt correctness-first solve-rate gate.

These tests trace to REQ-KONA-3440 / SCENARIO-KONA-3440 and
SCENARIO-KONA-3440-ENCODING-INVALID. They cover the module
``carnot.phase3.sudoku_global_opt``: deterministic puzzle generation, the
board-level validity oracle, Step-0a encoding validity (the gating
precondition), the optimizer ladder, the constraint-propagation hybrid, and the
top-level artifact contract. Optimization budgets are kept tiny so the suite
runs fast -- the science budget lives in the experiment driver, not the tests.
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import pytest

from carnot.phase3.sudoku_global_opt import (
    ENCODING_EPS,
    EncodingValidity,
    OptimizeResult,
    TIER_CLUES,
    board_is_valid_solution,
    check_encoding_validity,
    constraint_propagation_solve,
    count_violated_constraints,
    dig_holes,
    generate_full_grid,
    hybrid_solve,
    make_puzzle_set,
    optimize_board,
    reproducibility_checksum,
    run_correctness_gate,
)


@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches to keep the pytest watchdog memory flat."""
    yield
    jax.clear_caches()


# --------------------------------------------------------------------------- #
# Puzzle generation + validity oracle (REQ-KONA-3440).
# --------------------------------------------------------------------------- #
def test_generate_full_grid_is_valid_and_deterministic():
    """A generated full grid is a legal Latin square and reproducible by seed."""
    g1 = generate_full_grid(7)
    g2 = generate_full_grid(7)
    assert g1 == g2  # deterministic
    assert generate_full_grid(8) != g1  # seed-sensitive
    # A full grid is its own clue set and must validate.
    assert board_is_valid_solution(g1, g1)
    assert count_violated_constraints(g1) == 0


def test_dig_holes_preserves_solution_and_clue_count():
    """Digging holes leaves exactly n_clues givens, all matching the solution."""
    full = generate_full_grid(11)
    puzzle = dig_holes(full, n_clues=40, seed=11)
    given = sum(1 for r in range(9) for c in range(9) if puzzle[r][c] != 0)
    assert given == 40
    # Every surviving clue equals the solution, so the full grid still solves it.
    assert board_is_valid_solution(full, puzzle)


def test_dig_holes_rejects_out_of_range_clue_count():
    """n_clues outside 1..81 is a programming error and must raise."""
    full = generate_full_grid(1)
    with pytest.raises(ValueError):
        dig_holes(full, n_clues=0, seed=1)


def test_board_is_valid_solution_detects_violations():
    """The oracle rejects clue mismatch, row/col/box duplicates."""
    full = generate_full_grid(3)
    # Clue mismatch: change a cell the clue pins.
    bad_clue = [row[:] for row in full]
    clues = [row[:] for row in full]
    bad_clue[0][0] = 1 + (full[0][0] % 9)
    assert not board_is_valid_solution(bad_clue, clues)
    # Row duplicate.
    dup = [row[:] for row in full]
    dup[0][0], dup[0][1] = 5, 5
    assert not board_is_valid_solution(dup, [[0] * 9 for _ in range(9)])


def test_make_puzzle_set_has_at_least_20_puzzles_across_tiers():
    """The set clears the >=20 sample-size floor and spans all three tiers."""
    puzzles = make_puzzle_set(seed=3440)
    assert len(puzzles) >= 20
    tiers = {p.difficulty for p in puzzles}
    assert tiers == set(TIER_CLUES)
    # Every puzzle's stored solution is valid for its clues.
    assert all(board_is_valid_solution(p.solution, p.clues) for p in puzzles)


def test_count_violated_constraints_counts_groups():
    """A board with one duplicated row digit reports at least one violated group."""
    full = generate_full_grid(5)
    broken = [row[:] for row in full]
    broken[0][0] = broken[0][1]
    assert count_violated_constraints(broken) >= 1


# --------------------------------------------------------------------------- #
# Step 0a: encoding validity (gating precondition).
# --------------------------------------------------------------------------- #
def test_encoding_validity_valid_board_scores_zero():
    """A known-valid solved board scores ~0 -- the Step-0a invariant."""
    full = generate_full_grid(42)
    ev = check_encoding_validity(full)
    assert isinstance(ev, EncodingValidity)
    assert ev.is_valid
    assert ev.total_energy < ENCODING_EPS
    assert set(ev.residual_by_type) == {"row", "col", "box", "clue"}
    # Round-trips through JSON.
    assert ev.as_dict()["is_valid"] is True


def test_encoding_validity_broken_board_scores_positive():
    """A board with colliding digits scores > eps and is flagged invalid."""
    full = generate_full_grid(42)
    broken = [row[:] for row in full]
    broken[0][0] = broken[0][1]  # collide two cells in row 0
    ev = check_encoding_validity(broken)
    assert not ev.is_valid
    assert ev.total_energy > ENCODING_EPS


# --------------------------------------------------------------------------- #
# Optimizer ladder.
# --------------------------------------------------------------------------- #
def test_optimize_board_returns_scored_result():
    """The optimizer returns a rounded board scored on the discrete board."""
    puzzles = make_puzzle_set(3440)
    res = optimize_board(
        puzzles[0].clues, seed=1, variant="annealed", n_steps=50, n_restarts=1
    )
    assert isinstance(res, OptimizeResult)
    assert len(res.board) == 9 and all(len(r) == 9 for r in res.board)
    assert all(1 <= v <= 9 for row in res.board for v in row)
    assert res.final_energy >= 0.0
    # solved is scored on the board, must agree with the oracle.
    assert res.solved == board_is_valid_solution(res.board, puzzles[0].clues)


def test_optimize_board_restarts_variant_runs():
    """The annealed_restarts variant runs multiple inits and returns the best."""
    puzzles = make_puzzle_set(3440)
    res = optimize_board(
        puzzles[0].clues,
        seed=2,
        variant="annealed_restarts",
        n_steps=30,
        n_restarts=2,
    )
    assert isinstance(res, OptimizeResult)
    assert 0 <= res.n_violated <= 27


def test_optimize_board_rejects_unknown_variant():
    """An unknown optimizer variant is a programming error."""
    puzzles = make_puzzle_set(3440)
    with pytest.raises(ValueError):
        optimize_board(puzzles[0].clues, seed=1, variant="nope")


# --------------------------------------------------------------------------- #
# Hybrid: energy proposes, constraint propagation closes.
# --------------------------------------------------------------------------- #
def test_constraint_propagation_solves_valid_puzzle():
    """The CP solver finds a valid completion for a real puzzle."""
    puzzles = make_puzzle_set(3440)
    hard = next(p for p in puzzles if p.difficulty == "hard")
    solved = constraint_propagation_solve(hard.clues)
    assert solved is not None
    assert board_is_valid_solution(solved, hard.clues)


def test_constraint_propagation_node_budget_returns_none():
    """An impossibly small node budget yields None (treated as unsolved)."""
    puzzles = make_puzzle_set(3440)
    hard = next(p for p in puzzles if p.difficulty == "hard")
    assert constraint_propagation_solve(hard.clues, max_nodes=1) is None


def test_hybrid_solve_returns_valid_board():
    """The hybrid returns a valid board and a True flag for a solvable puzzle."""
    puzzles = make_puzzle_set(3440)
    p = puzzles[0]
    board, ok = hybrid_solve(p.clues, energy_board=[[0] * 9 for _ in range(9)])
    assert ok is True
    assert board is not None
    assert board_is_valid_solution(board, p.clues)


def test_hybrid_solve_handles_unsolvable_within_budget(monkeypatch):
    """When CP returns None, the hybrid reports an unsolved instance."""
    import carnot.phase3.sudoku_global_opt as mod

    monkeypatch.setattr(mod, "constraint_propagation_solve", lambda clues: None)
    board, ok = mod.hybrid_solve([[0] * 9 for _ in range(9)], energy_board=[[0] * 9 for _ in range(9)])
    assert board is None and ok is False


# --------------------------------------------------------------------------- #
# Reproducibility + top-level artifact contract.
# --------------------------------------------------------------------------- #
def test_reproducibility_checksum_is_stable():
    """The checksum is deterministic for the same puzzles/seed/config."""
    puzzles = make_puzzle_set(3440)
    cfg = {"n_steps": 10}
    a = reproducibility_checksum(puzzles, 3440, cfg)
    b = reproducibility_checksum(puzzles, 3440, cfg)
    assert a == b
    assert a != reproducibility_checksum(puzzles, 9999, cfg)


def test_run_correctness_gate_emits_required_fields(monkeypatch):
    """The full gate emits every required artifact field with a terminal verdict.

    The real optimizer is covered by the ``test_optimize_board_*`` cases above;
    here we stub it with a fast "never solves" result so this orchestration test
    does not trigger 21 JIT compiles (and the pytest memory watchdog). That stub
    is also the honest finding: pure energy descent plateaus, so the gate must
    fall through to the "hybrid solves, pure descent does not" verdict while the
    Step-0a gating and schema contract hold.
    """
    import carnot.phase3.sudoku_global_opt as mod

    def _stub_opt(clues, *, seed, variant, n_steps, n_restarts):
        return OptimizeResult(
            board=[[1] * 9 for _ in range(9)],
            final_energy=10.0,
            solved=False,
            n_violated=20,
        )

    monkeypatch.setattr(mod, "optimize_board", _stub_opt)
    artifact = run_correctness_gate(seed=3440, n_steps=20, n_restarts=1)
    required = {
        "encoding_validity_E0",
        "easy_tier_solve_rate",
        "n_violated_constraints_at_plateau",
        "hybrid_solve_rate",
        "solve_rate",
        "n_puzzles",
        "solve_rate_by_difficulty",
        "time_to_solution_solved_only",
        "optimizer_variant",
        "random_seed",
        "reproducibility_checksum",
        "honest_verdict",
    }
    assert required <= set(artifact)
    assert artifact["n_puzzles"] >= 20
    assert artifact["encoding_validity_E0"]["is_valid"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    # The hybrid (CP) must solve every valid puzzle.
    assert artifact["hybrid_solve_rate"] == 1.0
    # No compute-bound (GGUF/CUDA) markers -- this is a CPU Ising optimization.
    assert "model_specs" not in artifact


def test_run_correctness_gate_forks_on_invalid_encoding(monkeypatch):
    """If Step 0a fails, no optimization runs and the blocked verdict is emitted."""
    import carnot.phase3.sudoku_global_opt as mod

    bad = EncodingValidity(
        total_energy=123.0,
        is_valid=False,
        residual_by_type={"row": 123.0, "col": 0.0, "box": 0.0, "clue": 0.0},
    )
    monkeypatch.setattr(mod, "check_encoding_validity", lambda solution: bad)
    artifact = mod.run_correctness_gate(seed=3440, n_steps=20, n_restarts=1)
    assert artifact["honest_verdict"] == (
        "complete: blocked_energy_encoding_invalid_per_constraint_residual_reported"
    )
    assert artifact["solve_rate"] is None
    assert artifact["encoding_validity_E0"]["is_valid"] is False
