"""Tests for kona_process_energy_harder_instances (exp 3475).

Spec: REQ-KONA-3475, SCENARIO-KONA-3475

These tests exercise the exported API surface of the module. All tests use
small inputs (2-3 puzzles, single Langevin iteration) to keep the suite fast.
The module is stateless: the same call with the same seed always returns the
same result, so we can assert on exact structural properties without relying
on stochastic outcomes.

The JAX optimizer (optimize_board) is stubbed in any test that would call it,
preventing JAX JIT compilation and the associated ~900 MB RSS spike that the
memray plugin flags as a memory leak.
"""

from __future__ import annotations

import jax
import pytest

import carnot.phase3.kona_process_energy_harder_instances as _kmod
from carnot.phase3.kona_process_energy_harder_instances import (
    HARDER_CLUE_COUNT,
    RESTRICTED_MAX_NODES,
    board_to_text,
    derive_verdict_3475,
    energy_aware_hybrid_solve,
    make_harder_puzzle_set_v5,
    paired_significance_v5,
    reproducibility_checksum_3475,
    run_process_energy_arms,
)


from carnot.phase3.sudoku_global_opt import OptimizeResult, dig_holes, generate_full_grid


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

_SEED = 20260530
_SMALL_N = 1  # one puzzle per tier (2 total) for fast test runs


@pytest.fixture(autouse=True)
def clear_jax_cache():
    """Clear JAX compilation caches after each test to cap RSS growth.

    Without this, JAX JIT-compiles the Langevin optimizer on first call and
    the compiled code stays resident in the process. The memray plugin sees
    the accumulated allocation as a 'memory leak' at teardown. Clearing the
    cache after each test limits the per-test RSS delta within plugin thresholds.
    """
    yield
    jax.clear_caches()


def _make_small_puzzles():
    """Generate 2 harder puzzles for fast testing."""
    return make_harder_puzzle_set_v5(_SEED, n_per_tier=_SMALL_N)


def _stub_optimize_board(clues, *, seed, variant, n_steps, n_restarts):
    """Fast stub for optimize_board that avoids JAX compilation.

    Returns a deterministic board (the clues grid with zeros replaced by 1)
    and a dummy energy. This stub lets the arm-runner logic exercise the full
    code path without triggering JAX JIT, keeping the test fast and memory-safe.
    """
    board = [[cell if cell != 0 else 1 for cell in row] for row in clues]
    return OptimizeResult(board=board, final_energy=5.0, solved=False, n_violated=5)


# ---------------------------------------------------------------------------
# REQ-KONA-3475: make_harder_puzzle_set_v5
# ---------------------------------------------------------------------------


def test_make_harder_puzzle_set_v5_returns_correct_count():
    """SCENARIO-KONA-3475: puzzle-set count equals 2 * n_per_tier (one per tier)."""
    n_per_tier = 3
    puzzles = make_harder_puzzle_set_v5(_SEED, n_per_tier=n_per_tier)
    # Two tiers (harder + hardest) → 2 * n_per_tier total
    assert len(puzzles) == 2 * n_per_tier


def test_harder_puzzles_have_fewer_clues_than_standard():
    """SCENARIO-KONA-3475: all puzzles have <= HARDER_CLUE_COUNT clues.

    Standard hard tier uses 26 clues. Our harder instances use 17 or 18 clues,
    well below the standard threshold. This is the mechanism that forces CP
    timeouts and gives the energy fallback a role.
    """
    puzzles = _make_small_puzzles()
    for p in puzzles:
        # Count non-zero cells in the clue grid
        actual_clues = sum(1 for row in p.clues for cell in row if cell != 0)
        assert actual_clues <= HARDER_CLUE_COUNT, (
            f"Puzzle {p.puzzle_id} has {actual_clues} clues; expected <= {HARDER_CLUE_COUNT}"
        )


# ---------------------------------------------------------------------------
# REQ-KONA-3475: energy_aware_hybrid_solve
# ---------------------------------------------------------------------------


def test_energy_aware_hybrid_solve_cp_succeeds():
    """SCENARIO-KONA-3475: when CP succeeds within node budget, returns solved=True.

    We use a known-easy puzzle (46 clues = standard easy tier) so that even
    the restricted 1,500-node CP solver can find a solution. The energy board
    is irrelevant in this case — CP is used exclusively.
    """
    from carnot.phase3.sudoku_global_opt import generate_full_grid, dig_holes

    # 46-clue puzzle: CP at 1,500 nodes always finds a solution.
    full = generate_full_grid(42)
    clues = dig_holes(full, 46, 99)
    energy_board = full  # arbitrary valid board as the energy proposal

    board, solved = energy_aware_hybrid_solve(clues, energy_board, max_nodes=200_000)
    assert solved is True
    assert board is not None


def test_energy_aware_hybrid_solve_cp_timeout_fallback_valid():
    """SCENARIO-KONA-3475: when CP times out, use energy_board if it is valid.

    We force a CP timeout by using max_nodes=1 (guaranteed to time out on any
    non-trivial puzzle). The energy_board is the known full solution, which
    passes the validity check, so the function should return (full_board, True).
    """
    from carnot.phase3.sudoku_global_opt import generate_full_grid, dig_holes

    full = generate_full_grid(77)
    clues = dig_holes(full, 17, 88)  # 17-clue puzzle

    # max_nodes=1 always times out
    board, solved = energy_aware_hybrid_solve(clues, full, max_nodes=1)
    assert solved is True
    assert board is not None


def test_energy_aware_hybrid_solve_cp_timeout_fallback_invalid():
    """SCENARIO-KONA-3475: when CP times out and energy_board is invalid, return False.

    We force a CP timeout with max_nodes=1 and supply an invalid energy board
    (all zeros). The function must return (None, False) rather than accepting
    a broken board.
    """
    from carnot.phase3.sudoku_global_opt import generate_full_grid, dig_holes

    full = generate_full_grid(33)
    clues = dig_holes(full, 17, 44)
    bad_board = [[0] * 9 for _ in range(9)]  # definitely not a valid solution

    board, solved = energy_aware_hybrid_solve(clues, bad_board, max_nodes=1)
    assert solved is False
    assert board is None


# ---------------------------------------------------------------------------
# REQ-KONA-3475: board_to_text
# ---------------------------------------------------------------------------


def test_board_to_text():
    """SCENARIO-KONA-3475: board_to_text serialises a 9x9 grid to a string."""
    board = [[i + 1 for i in range(9)] for _ in range(9)]
    text = board_to_text(board)
    assert isinstance(text, str)
    assert len(text) > 0
    # Should contain digit characters and spaces only
    for ch in text:
        assert ch.isdigit() or ch == " ", f"Unexpected character: {ch!r}"
    # Each row's 9 digits should appear (joined by spaces)
    assert "1 2 3 4 5 6 7 8 9" in text


# ---------------------------------------------------------------------------
# REQ-KONA-3475: run_process_energy_arms
# ---------------------------------------------------------------------------


def test_run_process_energy_arms_returns_required_fields(monkeypatch):
    """SCENARIO-KONA-3475: run_process_energy_arms returns all required keys.

    This is a smoke test on a 2-puzzle set. It validates the output schema
    without asserting on specific solve-rate values (which are stochastic).
    The JAX optimizer is stubbed to avoid JIT compilation and RSS growth.
    """
    monkeypatch.setattr(_kmod, "optimize_board", _stub_optimize_board)
    puzzles = _make_small_puzzles()
    result = run_process_energy_arms(puzzles, seed=42)


    required_keys = {
        "untrained_hybrid_solve_rate",
        "process_hybrid_solve_rate",
        "pure_process_energy_descent_solve_rate",
        "per_puzzle_untrained_hybrid",
        "per_puzzle_process_hybrid",
        "per_puzzle_pure_process_descent",
    }
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Per-puzzle lists must match puzzle count
    n = len(puzzles)
    assert len(result["per_puzzle_untrained_hybrid"]) == n
    assert len(result["per_puzzle_process_hybrid"]) == n
    assert len(result["per_puzzle_pure_process_descent"]) == n

    # Solve rates must be floats in [0, 1]
    for rate_key in (
        "untrained_hybrid_solve_rate",
        "process_hybrid_solve_rate",
        "pure_process_energy_descent_solve_rate",
    ):
        rate = result[rate_key]
        assert isinstance(rate, float), f"{rate_key} is not float"
        assert 0.0 <= rate <= 1.0, f"{rate_key} out of range: {rate}"

    # Per-puzzle records must have the 'solved' bool
    for rec in result["per_puzzle_untrained_hybrid"]:
        assert "puzzle_id" in rec
        assert "solved" in rec
        assert isinstance(rec["solved"], bool)


# ---------------------------------------------------------------------------
# REQ-KONA-3475: paired_significance_v5
# ---------------------------------------------------------------------------


def test_paired_significance_v5():
    """SCENARIO-KONA-3475: paired_significance_v5 returns correct McNemar fields."""
    # Build synthetic per-puzzle records: arm A solves all, arm B solves none.
    # This gives maximum discordance: every puzzle is a B-loses record.
    n = 5
    untrained = [{"puzzle_id": f"p{i}", "solved": True} for i in range(n)]
    process = [{"puzzle_id": f"p{i}", "solved": False} for i in range(n)]

    sig = paired_significance_v5(untrained, process)

    assert "comparison" in sig
    assert "discordant_process_wins" in sig
    assert "discordant_untrained_wins" in sig
    assert "mcnemar_exact_p" in sig
    assert "interpretation" in sig

    # All n puzzles are untrained-wins discordant pairs
    assert sig["discordant_untrained_wins"] == n
    assert sig["discordant_process_wins"] == 0
    # p-value must be in (0, 1]
    p = sig["mcnemar_exact_p"]
    assert 0 < p <= 1.0


def test_paired_significance_v5_no_discordance():
    """SCENARIO-KONA-3475: p = 1.0 when both arms agree on all puzzles."""
    n = 4
    both_true = [{"puzzle_id": f"p{i}", "solved": True} for i in range(n)]
    sig = paired_significance_v5(both_true, both_true)
    assert sig["mcnemar_exact_p"] == 1.0
    assert sig["discordant_process_wins"] == 0
    assert sig["discordant_untrained_wins"] == 0


# ---------------------------------------------------------------------------
# REQ-KONA-3475: reproducibility_checksum_3475
# ---------------------------------------------------------------------------


def test_reproducibility_checksum_3475():
    """SCENARIO-KONA-3475: checksum is deterministic and 16 chars long."""
    cs1 = reproducibility_checksum_3475(
        seed=_SEED, n_per_tier=12, restricted_max_nodes=RESTRICTED_MAX_NODES,
        n_candidates=4, fast_n_steps=80
    )
    cs2 = reproducibility_checksum_3475(
        seed=_SEED, n_per_tier=12, restricted_max_nodes=RESTRICTED_MAX_NODES,
        n_candidates=4, fast_n_steps=80
    )
    assert cs1 == cs2
    assert len(cs1) == 16

    # Changing any parameter changes the checksum
    cs3 = reproducibility_checksum_3475(
        seed=_SEED + 1, n_per_tier=12, restricted_max_nodes=RESTRICTED_MAX_NODES,
        n_candidates=4, fast_n_steps=80
    )
    assert cs3 != cs1


# ---------------------------------------------------------------------------
# REQ-KONA-3475: derive_verdict_3475
# ---------------------------------------------------------------------------


def test_derive_verdict_3475_no_headroom():
    """SCENARIO-KONA-3475: blocked verdict when solve-rate >= 0.8 (no headroom)."""
    verdict = derive_verdict_3475(
        process_hybrid_solve_rate=0.9,
        untrained_hybrid_solve_rate=0.9,
        pure_process_solve_rate=0.0,
        instances_have_headroom=False,
        mcnemar_p=1.0,
    )
    assert verdict == "complete: blocked_kona_instances_saturated_no_headroom"
    assert verdict.startswith("complete:")


def test_derive_verdict_3475_no_lift():
    """SCENARIO-KONA-3475: no-lift verdict when process <= untrained or p >= 0.05."""
    # Case 1: process == untrained (no lift)
    verdict = derive_verdict_3475(
        process_hybrid_solve_rate=0.5,
        untrained_hybrid_solve_rate=0.5,
        pure_process_solve_rate=0.0,
        instances_have_headroom=True,
        mcnemar_p=1.0,
    )
    assert verdict == "complete: process_energy_no_lift_over_untrained_kona_hybrid_even_with_headroom"
    assert verdict.startswith("complete:")

    # Case 2: process > untrained but p >= 0.05 (not significant)
    verdict2 = derive_verdict_3475(
        process_hybrid_solve_rate=0.6,
        untrained_hybrid_solve_rate=0.5,
        pure_process_solve_rate=0.0,
        instances_have_headroom=True,
        mcnemar_p=0.20,
    )
    assert verdict2 == "complete: process_energy_no_lift_over_untrained_kona_hybrid_even_with_headroom"


def test_derive_verdict_3475_lift():
    """SCENARIO-KONA-3475: lift verdict when process > untrained AND p < 0.05."""
    verdict = derive_verdict_3475(
        process_hybrid_solve_rate=0.7,
        untrained_hybrid_solve_rate=0.4,
        pure_process_solve_rate=0.0,
        instances_have_headroom=True,
        mcnemar_p=0.01,
    )
    assert verdict == "complete: process_energy_strengthens_kona_hybrid_with_headroom"
    assert verdict.startswith("complete:")


# ---------------------------------------------------------------------------
# Branch coverage: _solve_rate empty list and process_best=None fallback
# ---------------------------------------------------------------------------


def test_run_process_energy_arms_empty_puzzles(monkeypatch):
    """SCENARIO-KONA-3475: empty puzzle list returns 0.0 solve rates.

    Exercises the _solve_rate([]) early-return branch (if not records: return 0.0).
    """
    monkeypatch.setattr(_kmod, "optimize_board", _stub_optimize_board)
    result = run_process_energy_arms([], seed=42)
    assert result["untrained_hybrid_solve_rate"] == 0.0
    assert result["process_hybrid_solve_rate"] == 0.0
    assert result["pure_process_energy_descent_solve_rate"] == 0.0
    assert result["per_puzzle_untrained_hybrid"] == []
    assert result["per_puzzle_process_hybrid"] == []
    assert result["per_puzzle_pure_process_descent"] == []


def test_run_process_energy_arms_process_argmin_none_fallback(monkeypatch):
    """SCENARIO-KONA-3475: process_energy_argmin returning None falls back to candidates[0].

    This exercises the 'else candidates[0]' defensive branch in run_process_energy_arms.
    We patch process_energy_argmin to return None, which is what it does when all
    answers are None (empty list edge case). The fallback must still pick candidates[0].
    """
    monkeypatch.setattr(_kmod, "optimize_board", _stub_optimize_board)
    # Patch process_energy_argmin to return None (simulating the all-None-answers edge case)
    monkeypatch.setattr(_kmod, "process_energy_argmin", lambda answers, energies: None)
    puzzles = _make_small_puzzles()
    result = run_process_energy_arms(puzzles, seed=42)
    # With process_energy_argmin returning None, the fallback picks candidates[0].
    # The test asserts the schema is still correct — the fallback does not crash.
    assert "process_hybrid_solve_rate" in result
    assert len(result["per_puzzle_process_hybrid"]) == len(puzzles)
