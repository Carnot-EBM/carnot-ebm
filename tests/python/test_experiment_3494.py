"""Tests for the P0.1 Sudoku solve-rate gate module (Exp 3494).

Traces to REQ-KONA-3494 and SCENARIO-KONA-3494. Covers:
- QUBO cross-validation note content (Step 0a documentation)
- AR baseline greedy scan (fully-constrained board, valid-board path)
- Extended optimizer variants (smoke -- tiny n_steps so the suite runs fast)
- Top-level run_p01_gate artifact contract (all required fields present,
  verdict has terminal prefix, encoding is valid, n_puzzles >= 20)
"""

from __future__ import annotations

import os

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import random

import jax
import pytest

from carnot.phase3.sudoku_global_opt import (
    SudokuPuzzle,
    board_is_valid_solution,
    dig_holes,
    generate_full_grid,
    make_puzzle_set,
)
from carnot.phase3.sudoku_p01_gate import (
    QUBO_CROSSVALIDATION_NOTE,
    ar_baseline_solve_rate,
    ar_greedy_solve,
    optimize_board_v2,
    run_p01_gate,
)


@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches between tests -- keeps memory flat."""
    yield
    jax.clear_caches()


# ---------------------------------------------------------------------------
# REQ-KONA-3494: QUBO cross-validation note (Step 0a documentation)
# ---------------------------------------------------------------------------


def test_qubo_crossvalidation_note_cites_paper():
    """The QUBO note cites arXiv:2403.04816.

    SCENARIO-KONA-3494: the note is the audit trail that shows Carnot's energy
    matches the published QUBO encoding -- any claim about E=0 on a valid board
    rests on this alignment.
    """
    assert "2403.04816" in QUBO_CROSSVALIDATION_NOTE  # REQ-KONA-3494


def test_qubo_crossvalidation_note_covers_all_constraint_families():
    """The QUBO note mentions all four constraint families.

    SCENARIO-KONA-3494: row, column, box, and cell uniqueness must all be
    documented -- a missing family would leave a gap in the QUBO correspondence
    claim.
    """
    note_lower = QUBO_CROSSVALIDATION_NOTE.lower()
    for family in ("row", "column", "box", "cell"):
        assert family in note_lower, (
            f"QUBO note missing constraint family: {family!r}"
        )  # REQ-KONA-3494


# ---------------------------------------------------------------------------
# REQ-KONA-3494: AR greedy baseline
# ---------------------------------------------------------------------------


def test_ar_greedy_solve_on_fully_constrained_board():
    """AR greedy scan trivially succeeds when all cells are given (no free cells).

    SCENARIO-KONA-3494: with every cell as a clue, the scan makes no random
    choices -- it should return True without contradiction.
    """
    full = generate_full_grid(42)
    rng = random.Random(0)
    result = ar_greedy_solve(full, rng)
    assert result is True  # REQ-KONA-3494: all-clue board succeeds


def test_ar_greedy_solve_returns_bool():
    """ar_greedy_solve always returns a bool, never raises on a hard puzzle.

    SCENARIO-KONA-3494: a hard puzzle (26 clues) may produce False, but the
    function must not raise -- the baseline infrastructure must be stable.
    """
    full = generate_full_grid(7)
    puzzle = dig_holes(full, 26, 7)
    rng = random.Random(123)
    result = ar_greedy_solve(puzzle, rng)
    assert isinstance(result, bool)  # REQ-KONA-3494


def test_ar_baseline_solve_rate_range():
    """ar_baseline_solve_rate is in [0.0, 1.0] for any puzzle set.

    SCENARIO-KONA-3494: the rate is a fraction -- negative or >1 would indicate
    a counting bug.
    """
    puzzles = make_puzzle_set(seed=3494)
    rate = ar_baseline_solve_rate(puzzles, seed=42, n_trials=1)
    assert 0.0 <= rate <= 1.0  # REQ-KONA-3494


def test_ar_baseline_easy_puzzles_nonnegative():
    """AR baseline on easy puzzles (46 clues) returns a non-negative rate.

    SCENARIO-KONA-3494: easy puzzles are nearly fully constrained -- the greedy
    scan has fewer chances to create contradictions.
    """
    puzzles = make_puzzle_set(seed=3494)
    easy = [p for p in puzzles if p.difficulty == "easy"]
    rate = ar_baseline_solve_rate(easy, seed=42, n_trials=2)
    assert rate >= 0.0  # REQ-KONA-3494


# ---------------------------------------------------------------------------
# REQ-KONA-3494: Extended optimizer variants (smoke -- tiny budget)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant,n_steps,n_restarts",
    [
        ("vanilla", 50, 1),
        ("annealed", 50, 1),
        ("simulated_annealing", 50, 1),
        ("random_restarts", 50, 2),
        ("annealed_restarts", 50, 2),
        ("parallel_tempering", 50, 1),
        ("adaptive", 50, 2),
    ],
)
@pytest.mark.memory_watchdog_skip
def test_optimize_board_v2_variant_smoke(variant: str, n_steps: int, n_restarts: int) -> None:
    """Each optimizer variant completes without error and returns a valid-shaped result.

    SCENARIO-KONA-3494: with tiny n_steps the board is unlikely to be solved, but
    the variant must return a 9x9 board, a bool solved flag, and an int n_violated.
    """
    full = generate_full_grid(3494)
    puzzle = dig_holes(full, 26, 3494)
    res = optimize_board_v2(
        puzzle,
        seed=1,
        variant=variant,
        n_steps=n_steps,
        n_restarts=n_restarts,
        n_clues=26,
    )
    assert isinstance(res.solved, bool)  # REQ-KONA-3494
    assert isinstance(res.n_violated, int)
    assert len(res.board) == 9
    assert all(len(row) == 9 for row in res.board)
    for row in res.board:
        for v in row:
            assert 1 <= v <= 9, f"cell value {v} out of [1,9]"


def test_optimize_board_v2_unknown_variant_raises():
    """An unknown variant name raises ValueError immediately.

    SCENARIO-KONA-3494: the guard prevents silent fallback to wrong behavior --
    mis-spelled variant names should fail loudly, not silently use vanilla.
    """
    full = generate_full_grid(1)
    puzzle = dig_holes(full, 46, 1)
    with pytest.raises(ValueError, match="unknown variant"):
        optimize_board_v2(puzzle, seed=1, variant="does_not_exist")


# ---------------------------------------------------------------------------
# REQ-KONA-3494: Top-level artifact contract
# ---------------------------------------------------------------------------


@pytest.mark.memory_watchdog_skip
def test_run_p01_gate_required_fields_present():
    """run_p01_gate returns all required artifact fields, even on tiny budget.

    SCENARIO-KONA-3494: missing fields would fail the adversarial-verify schema
    check and prevent milestone acceptance.
    """
    artifact = run_p01_gate(seed=42, n_steps_base=50, n_restarts=1, ar_n_trials=1)
    required_fields = [
        "honest_verdict",
        "inference_substrate",
        "encoding_validity_E0",
        "qubo_crossvalidation_note",
        "easy_tier_solve_rate",
        "n_puzzles",
        "solve_rate",
        "solve_rate_by_difficulty",
        "solve_rate_by_optimizer_variant",
        "n_violated_constraints_at_plateau",
        "hybrid_solve_rate",
        "time_to_solution_solved_only",
        "ar_baseline_solve_rate",
        "random_seed",
        "reproducibility_checksum",
    ]
    for field in required_fields:
        assert field in artifact, f"missing required field: {field!r}"  # REQ-KONA-3494


@pytest.mark.memory_watchdog_skip
def test_run_p01_gate_verdict_has_terminal_prefix():
    """honest_verdict must start with a terminal prefix per Verdict Discipline.

    SCENARIO-KONA-3494: the conductor's classifier fires on verdicts lacking a
    terminal prefix -- a missing prefix causes spurious retry or mis-classification.
    """
    artifact = run_p01_gate(seed=42, n_steps_base=50, n_restarts=1, ar_n_trials=1)
    verdict = artifact["honest_verdict"]
    terminal_prefixes = (
        "complete:",
        "complete_",
        "success:",
        "success_",
        "passed:",
        "passed_",
        "shipped:",
        "shipped_",
    )
    assert any(verdict.startswith(p) for p in terminal_prefixes), (
        f"verdict {verdict!r} lacks a terminal prefix"
    )  # REQ-KONA-3494


@pytest.mark.memory_watchdog_skip
def test_run_p01_gate_encoding_valid():
    """Step 0a must report encoding_validity_E0.is_valid=True on a correct board.

    SCENARIO-KONA-3494: encoding validity is the most critical precondition --
    if the energy mis-specifies the constraints, no optimizer can solve it and
    the entire P0.1 test is invalid.
    """
    artifact = run_p01_gate(seed=42, n_steps_base=50, n_restarts=1, ar_n_trials=1)
    ev = artifact["encoding_validity_E0"]
    assert ev["is_valid"] is True, (
        f"encoding is invalid (E={ev['total_energy']:.6f}); "
        "the QUBO cross-validation is broken -- check carnot.verify.sudoku"
    )  # REQ-KONA-3494


@pytest.mark.memory_watchdog_skip
def test_run_p01_gate_n_puzzles_floor():
    """n_puzzles must be >=20 per the Adversarial Artifact sample-size rule.

    SCENARIO-KONA-3494: solve_rate on n<20 is not headline-eligible per CLAUDE.md
    'Adversarial Artifact Verification + Sample-Size Rigor'.
    """
    artifact = run_p01_gate(seed=42, n_steps_base=50, n_restarts=1, ar_n_trials=1)
    assert artifact["n_puzzles"] >= 20, (
        f"n_puzzles={artifact['n_puzzles']} below the 20-puzzle sample-size floor"
    )  # REQ-KONA-3494


@pytest.mark.memory_watchdog_skip
def test_run_p01_gate_reproducibility_checksum_present():
    """The artifact must include a reproducibility_checksum.

    SCENARIO-KONA-3494: the checksum is a content hash of puzzle set + seed +
    config -- required by the Adversarial Artifact Verification methodology rule
    to enable third-party replication.
    """
    artifact = run_p01_gate(seed=42, n_steps_base=50, n_restarts=1, ar_n_trials=1)
    cs = artifact["reproducibility_checksum"]
    assert isinstance(cs, str) and len(cs) == 64, (
        f"expected 64-char hex SHA256, got {cs!r}"
    )  # REQ-KONA-3494
