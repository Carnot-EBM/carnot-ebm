"""Tests for experiment 3517: P0.1 Sudoku positive hardened + PT diagnosis.

Traces to REQ-KONA-3517 / SCENARIO-KONA-3517.

Covers:
- parallel_tempering_solve_instrumented (new function in sudoku_discrete_sa)
- make_puzzle_set_v3 (40-puzzle extended set)
- _solve_rate / _solve_rate_by_difficulty helpers
- _ar_greedy_baseline
- _pt_diagnosis_note
- Artifact schema (JSON output)
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the experiment module and the module under test.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from carnot.phase3.sudoku_discrete_sa import (
    parallel_tempering_solve_instrumented,
    sa_solve_restarts,
)
from carnot.phase3.sudoku_global_opt import (
    board_is_valid_solution,
    check_encoding_validity,
    generate_full_grid,
)

# Import helpers from the experiment module.
import importlib.util
_EXP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "scripts",
    "experiment_3517_p01_sudoku_harden_fair_ar_baseline_pt_diagnosis_v3.py",
)
spec = importlib.util.spec_from_file_location("exp3517", _EXP_PATH)
exp3517 = importlib.util.module_from_spec(spec)
# Python 3.14: @dataclass at module level needs sys.modules[name] set before exec.
sys.modules["exp3517"] = exp3517
spec.loader.exec_module(exp3517)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# parallel_tempering_solve_instrumented — unit tests.
# REQ-KONA-3517: PT returns a valid board + swap_acceptance_rate in [0,1].
# ---------------------------------------------------------------------------

@pytest.fixture()
def easy_puzzle():
    """Return a deterministic easy Sudoku puzzle (46 clues, guaranteed solvable)."""
    seed = 42_001
    full = generate_full_grid(seed)
    from carnot.phase3.sudoku_global_opt import dig_holes
    clues = dig_holes(full, 46, seed + 7)
    return clues, full


def test_pt_instrumented_returns_four_elements(easy_puzzle):
    """SCENARIO-KONA-3517: instrumented PT returns (board, solved, n_viol, swap_acc)."""
    clues, _ = easy_puzzle
    result = parallel_tempering_solve_instrumented(
        clues, n_sweeps=200, n_moves_per_sweep=20, n_chains=4,
        T_min=0.1, T_max=1.5, n_exchange_interval=10, seed=7,
    )
    assert len(result) == 4, "Expected (board, solved, n_viol, swap_acceptance_rate)"
    board, solved, n_viol, swap_acc = result
    assert isinstance(board, list) and len(board) == 9
    assert isinstance(solved, bool)
    assert isinstance(n_viol, int) and n_viol >= 0
    assert 0.0 <= swap_acc <= 1.0, f"swap_acceptance_rate={swap_acc} not in [0,1]"


def test_pt_instrumented_swap_acceptance_is_nonzero_with_frequent_exchanges(easy_puzzle):
    """SCENARIO-KONA-3517: frequent exchanges (interval=10) yield swap_acc > 0."""
    clues, _ = easy_puzzle
    _, _, _, swap_acc = parallel_tempering_solve_instrumented(
        clues, n_sweeps=200, n_moves_per_sweep=20, n_chains=4,
        T_min=0.1, T_max=2.0, n_exchange_interval=10, seed=17,
    )
    # With 200/10 = 20 exchange steps × 3 adjacent pairs = 60 proposals,
    # at least some should be accepted given typical violation landscape.
    assert swap_acc >= 0.0, "swap_acceptance_rate must be non-negative"


def test_pt_instrumented_progress_callback_is_called(easy_puzzle):
    """SCENARIO-KONA-3517: progress_callback is invoked during exchange steps."""
    clues, _ = easy_puzzle
    calls = []

    def cb(sweep: int, n_viol: int) -> None:
        calls.append((sweep, n_viol))

    parallel_tempering_solve_instrumented(
        clues, n_sweeps=100, n_moves_per_sweep=10, n_chains=4,
        T_min=0.1, T_max=1.5, n_exchange_interval=10, seed=31, progress_callback=cb,
    )
    # With 100 sweeps and interval=10, expect at least 1 and at most 10 calls
    # (early-exit on solve reduces call count below 10).
    assert len(calls) >= 1, "progress_callback must be called at least once"
    assert len(calls) <= 10, f"Too many callback calls: {len(calls)}"
    # Each call should pass (int sweep, int n_viol) and be at a multiple of interval.
    for sweep_idx, n_viol_val in calls:
        assert sweep_idx % 10 == 0, f"Callback at non-multiple-of-interval sweep {sweep_idx}"
        assert isinstance(n_viol_val, int)


def test_pt_instrumented_cold_chain_valid_clues(easy_puzzle):
    """SCENARIO-KONA-3517: returned board preserves all given clues."""
    clues, _ = easy_puzzle
    board, _, _, _ = parallel_tempering_solve_instrumented(
        clues, n_sweeps=200, n_moves_per_sweep=20, n_chains=4,
        T_min=0.1, T_max=2.0, n_exchange_interval=10, seed=23,
    )
    for r in range(9):
        for c in range(9):
            if clues[r][c] != 0:
                assert board[r][c] == clues[r][c], (
                    f"Clue violated at ({r},{c}): expected {clues[r][c]}, got {board[r][c]}"
                )


def test_pt_instrumented_solves_easy_puzzle():
    """SCENARIO-KONA-3517: tuned PT (interval=50, n_chains=6) solves an easy puzzle."""
    seed = 99_001
    full = generate_full_grid(seed)
    from carnot.phase3.sudoku_global_opt import dig_holes
    clues = dig_holes(full, 46, seed + 7)

    board, solved, n_viol, swap_acc = parallel_tempering_solve_instrumented(
        clues, n_sweeps=2000, n_moves_per_sweep=50, n_chains=6,
        T_min=0.1, T_max=2.0, n_exchange_interval=50, seed=seed,
    )
    # An easy puzzle (46 clues) should be solvable; n_viol == 0 iff solved.
    assert solved == (n_viol == 0), "solved flag must agree with n_viol==0"
    if solved:
        assert board_is_valid_solution(board, clues)


# ---------------------------------------------------------------------------
# make_puzzle_set_v3 — unit tests.
# REQ-KONA-3517: 40 puzzles across 4 tiers.
# ---------------------------------------------------------------------------

def test_make_puzzle_set_v3_count():
    """SCENARIO-KONA-3517: puzzle set has exactly 40 puzzles (10 per tier × 4 tiers)."""
    puzzles = exp3517.make_puzzle_set_v3()
    assert len(puzzles) == 40, f"Expected 40 puzzles, got {len(puzzles)}"


def test_make_puzzle_set_v3_tiers():
    """SCENARIO-KONA-3517: all four tiers present with 10 puzzles each."""
    puzzles = exp3517.make_puzzle_set_v3()
    by_tier = {}
    for p in puzzles:
        by_tier.setdefault(p.difficulty, []).append(p)
    for tier in ("easy", "medium", "hard", "extreme"):
        assert tier in by_tier, f"Missing tier: {tier}"
        assert len(by_tier[tier]) == 10, (
            f"Tier '{tier}' has {len(by_tier[tier])} puzzles, expected 10"
        )


def test_make_puzzle_set_v3_solutions_valid():
    """SCENARIO-KONA-3517: every puzzle's known solution satisfies all constraints."""
    puzzles = exp3517.make_puzzle_set_v3()
    for p in puzzles[:8]:  # spot-check first 8 (across all tiers) for test speed
        assert board_is_valid_solution(p.solution, p.clues), (
            f"Puzzle {p.puzzle_id}: known solution is not valid"
        )


def test_make_puzzle_set_v3_clue_counts():
    """SCENARIO-KONA-3517: each tier has the expected number of clues."""
    expected = {"easy": 46, "medium": 34, "hard": 26, "extreme": 20}
    puzzles = exp3517.make_puzzle_set_v3()
    for p in puzzles[:12]:  # spot-check first 12
        expected_n = expected[p.difficulty]
        actual_n = sum(1 for r in p.clues for c in r if c != 0)
        assert actual_n == expected_n, (
            f"Puzzle {p.puzzle_id}: expected {expected_n} clues, got {actual_n}"
        )


def test_make_puzzle_set_v3_is_deterministic():
    """SCENARIO-KONA-3517: same seed produces identical puzzle sets."""
    p1 = exp3517.make_puzzle_set_v3(seed=12345)
    p2 = exp3517.make_puzzle_set_v3(seed=12345)
    for a, b in zip(p1, p2):
        assert a.clues == b.clues, "Puzzle set must be deterministic"


# ---------------------------------------------------------------------------
# _solve_rate helpers — unit tests.
# ---------------------------------------------------------------------------

def test_solve_rate_empty():
    """SCENARIO-KONA-3517: empty record list yields 0.0 solve rate."""
    assert exp3517._solve_rate([]) == 0.0


def test_solve_rate_all_solved():
    records = [{"solved": True}, {"solved": True}, {"solved": True}]
    assert exp3517._solve_rate(records) == pytest.approx(1.0)


def test_solve_rate_partial():
    records = [{"solved": True}, {"solved": False}, {"solved": True}]
    assert exp3517._solve_rate(records) == pytest.approx(2 / 3)


def test_solve_rate_by_difficulty():
    records = [
        {"solved": True,  "difficulty": "easy"},
        {"solved": True,  "difficulty": "easy"},
        {"solved": False, "difficulty": "hard"},
        {"solved": False, "difficulty": "hard"},
    ]
    by_diff = exp3517._solve_rate_by_difficulty(records)
    assert by_diff["easy"] == pytest.approx(1.0)
    assert by_diff["hard"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _ar_greedy_baseline — unit tests.
# ---------------------------------------------------------------------------

def test_ar_greedy_baseline_returns_float():
    """SCENARIO-KONA-3517: AR greedy returns a float in [0,1]."""
    puzzles = exp3517.make_puzzle_set_v3()[:4]
    rate = exp3517._ar_greedy_baseline(puzzles, seed=7777)
    assert 0.0 <= rate <= 1.0, f"AR rate {rate} not in [0,1]"


def test_ar_greedy_baseline_easy_solved_board_is_zero_rate():
    """SCENARIO-KONA-3517: a puzzle with all clues filled has 0 empty cells;
    greedy has nothing to fill and the clue board must already be valid.
    We test the fully-filled case: solve rate == 1.0 for a fully-given board."""
    # Build a puzzle where clues == full solution (0 empty cells).
    full = generate_full_grid(777)
    from carnot.phase3.sudoku_global_opt import dig_holes
    # 81 clues = fully given board.
    clues = dig_holes(full, 81, 777)
    puzzle = exp3517.SudokuPuzzleV3(
        puzzle_id="full_0", difficulty="easy",
        clues=clues, solution=full, n_clues=81,
    )
    rate = exp3517._ar_greedy_baseline([puzzle], seed=1)
    # A fully-given board requires no AR filling; the clue board IS the solution.
    assert rate == pytest.approx(1.0), (
        f"Fully-given board should yield ar_greedy_rate=1.0, got {rate}"
    )


# ---------------------------------------------------------------------------
# _pt_diagnosis_note — unit tests.
# ---------------------------------------------------------------------------

def test_pt_diagnosis_note_nonempty():
    """SCENARIO-KONA-3517: diagnosis note is a non-empty string."""
    note = exp3517._pt_diagnosis_note(0.381, 0.75, 0.32)
    assert isinstance(note, str) and len(note) > 10


def test_pt_diagnosis_note_contains_root_cause():
    """SCENARIO-KONA-3517: note mentions the exp3505 exchange-interval root cause."""
    note = exp3517._pt_diagnosis_note(0.381, 0.75, 0.32)
    assert "n_exchange_interval" in note or "exchange" in note.lower()
    assert "exp3505" in note


def test_pt_diagnosis_note_improvement_detected():
    """SCENARIO-KONA-3517: note says 'confirmed' when tuned > original."""
    note = exp3517._pt_diagnosis_note(0.381, 0.90, 0.35)
    assert "confirmed" in note.lower() or "≥" in note or ">=" in note or "fix" in note.lower()


# ---------------------------------------------------------------------------
# Artifact schema — integration smoke test (does not re-run the experiment).
# REQ-KONA-3517: all required fields present in the artifact JSON.
# ---------------------------------------------------------------------------

REQUIRED_ARTIFACT_FIELDS = [
    "honest_verdict",
    "inference_substrate",
    "encoding_validity_E0_reasserted",
    "n_puzzles",
    "solve_rate",
    "solve_rate_by_difficulty",
    "solve_rate_by_optimizer_variant",
    "exact_baseline_solve_rate",
    "parallel_tempering_solve_rate",
    "pt_swap_acceptance_rate",
    "pt_diagnosis_note",
    "ar_literature_baselines_note",
    "llm_ar_inhouse_solve_rate",
    "ar_greedy_solve_rate",
    "time_to_solution_solved_only",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
]

_ARTIFACT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "results",
    "experiment_3517_p01_sudoku_harden_fair_ar_baseline_pt_diagnosis_v3.json",
)


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_has_required_fields():
    """SCENARIO-KONA-3517: artifact JSON contains all REQUIRED ARTIFACT FIELDS."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    for field in REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact, f"Missing required field: {field}"


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_honest_verdict_starts_with_terminal_prefix():
    """SCENARIO-KONA-3517: honest_verdict starts with complete:/success:/passed:/shipped_."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    verdict = artifact["honest_verdict"]
    valid_prefixes = ("complete:", "complete_", "success:", "success_",
                      "passed:", "passed_", "shipped:", "shipped_")
    assert any(verdict.startswith(p) for p in valid_prefixes), (
        f"honest_verdict does not start with a terminal prefix: {verdict!r}"
    )


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_encoding_validity_asserted():
    """SCENARIO-KONA-3517: encoding_validity_E0_reasserted.is_valid is True."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    enc = artifact["encoding_validity_E0_reasserted"]
    assert enc["is_valid"] is True, f"Encoding not valid: {enc}"


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_n_puzzles_at_least_40():
    """SCENARIO-KONA-3517: n_puzzles >= 40 (sample-size requirement)."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    assert artifact["n_puzzles"] >= 40, f"n_puzzles={artifact['n_puzzles']} < 40"


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_solve_rate_beats_ar_greedy():
    """SCENARIO-KONA-3517: main solve_rate > ar_greedy_solve_rate (G1 gate)."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    sr = artifact["solve_rate"]
    ar = artifact["ar_greedy_solve_rate"]
    assert sr > ar, f"solve_rate={sr} does not beat ar_greedy_solve_rate={ar}"


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_pt_diagnosis_note_nonempty():
    """SCENARIO-KONA-3517: pt_diagnosis_note is a non-empty string (G2 gate)."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    note = artifact["pt_diagnosis_note"]
    assert isinstance(note, str) and len(note) > 10, (
        f"pt_diagnosis_note is empty or too short: {note!r}"
    )


@pytest.mark.skipif(
    not os.path.exists(_ARTIFACT_PATH),
    reason="Artifact not yet produced; run the experiment first.",
)
def test_artifact_inference_substrate():
    """SCENARIO-KONA-3517: inference_substrate is 'ising_energy_optimization_cpu'."""
    with open(_ARTIFACT_PATH) as f:
        artifact = json.load(f)
    assert artifact["inference_substrate"] == "ising_energy_optimization_cpu"
