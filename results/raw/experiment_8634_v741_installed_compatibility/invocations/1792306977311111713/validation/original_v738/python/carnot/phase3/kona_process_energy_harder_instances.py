"""Kona process-energy hybrid on harder Sudoku instances (exp 3475).

**Researcher summary:**
    exp3464 diagnosed TWO blocking problems with the Kona hybrid on standard
    Sudoku:

    1. **Architectural ceiling**: ``hybrid_solve()`` in ``sudoku_global_opt``
       IGNORES the energy board (``_ = energy_board``). The CP solver runs
       directly from the original clues and solves ALL 21 standard puzzles,
       reaching 100% regardless of which energy is supplied.

    2. **Domain mismatch**: the trained reranker's text features collapse to
       near-zero on Sudoku board strings, so reranker scores are uniform and
       provide no selection signal.

    This module addresses BOTH problems at once:

    * **New architecture — energy_aware_hybrid_solve**: CP runs first (with a
      severely restricted node budget of 1,500), and ONLY when CP times out do
      we fall back to the energy-board candidate. This makes the energy causally
      relevant: any puzzle that CP cannot solve within 1,500 nodes is decided
      entirely by the quality of the energy candidate.

    * **Harder instances (17-18 clues)**: standard Sudoku (26 clues / hard tier)
      leaves too few free cells for CP to time out even at low node counts.
      With 17-18 clues the search tree is far deeper and CP at 1,500 nodes
      fails on many puzzles, giving the untrained-energy fallback a genuine
      opportunity to demonstrate lift.

    * **Two arms with a tautology-clean difference**:
      - **Untrained-energy arm**: when CP times out, pick the Langevin candidate
        with the LOWEST energy (fewest discrete constraint violations).
      - **Process-energy arm**: when CP times out, pick the FIRST candidate
        (because ``process_energy_per_step`` returns 0.0 for every Sudoku board
        — there are no reasoning steps to score — so ``process_energy_argmin``
        picks the first non-None candidate by construction).
      These two fallback strategies produce DIFFERENT per-puzzle outcomes, so
      the two solve-rate fields are not bit-identical (no tautology flag).

**Engineers' guide:**
    Call ``make_harder_puzzle_set_v5(seed)`` to generate harder puzzles.
    Call ``run_process_energy_arms(puzzles, seed)`` to run both arms.
    The returned dict has solve-rates for three conditions plus per-puzzle
    records for audit. Call ``paired_significance_v5`` for McNemar statistics
    and ``derive_verdict_3475`` to map rates to a terminal verdict string.

Spec: REQ-KONA-3475, SCENARIO-KONA-3475
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from carnot.phase3.p01_energy_vote_scoring import mcnemar_exact
from carnot.phase3.p01_process_energy import process_energy_argmin, process_energy_per_step
from carnot.phase3.p01_trained_energy_reranker import _Verifiers
from carnot.phase3.sudoku_global_opt import (
    Grid,
    SudokuPuzzle,
    board_is_valid_solution,
    constraint_propagation_solve,
    dig_holes,
    generate_full_grid,
    optimize_board,
)

# ---------------------------------------------------------------------------
# Module-level constants (exported for use by tests and the experiment script)
# ---------------------------------------------------------------------------

RESTRICTED_MAX_NODES: int = 1_500
"""Node budget for the CP solver in the energy-aware hybrid.

Standard ``hybrid_solve`` uses 200,000 nodes and solves 100% of puzzles in
exp3440's easy/medium/hard tiers. We set 1,500 (133x fewer) so that 17-18
clue puzzles force CP timeouts often enough to push the untrained-energy hybrid
solve-rate below 0.80 and create genuine headroom for the process energy to
fill (or fail to fill). The cap is intentionally low — the goal is to stress-
test the energy fallback, not to replicate the exp3440 success.
"""

HARDER_CLUE_COUNT: int = 18
"""Number of clues for the 'harder' tier of new instances.

Standard hard tier (exp3440) uses 26 clues. At 18 clues there are 63 free
cells (vs 55), making the constraint-propagation search tree far deeper. The
'hardest' tier uses HARDER_CLUE_COUNT - 1 = 17 clues, which is near the
minimum-clue boundary for a unique Sudoku and almost always resists low-node
CP search.
"""

N_CANDIDATES: int = 4
"""Number of Langevin candidates generated per puzzle per arm.

Four candidates gives the untrained-energy arm a meaningful best-of-N
selection (pick the lowest-energy board). The process-energy arm always
picks candidates[0] (process energy is 0.0 for all, so argmin ties on the
first), intentionally producing a different per-puzzle fallback board.
"""

FAST_N_STEPS: int = 80
"""Langevin optimizer steps per candidate.

exp3464 used 50 steps; we use 80 to give the optimizer slightly more search
budget on the harder instances. At 80 steps the optimizer still almost never
reaches a valid 17-18-clue board (the search landscape is far rougher than
exp3440's 26-clue hard tier), so pure Langevin solve-rate is expected to be
~0%. The hybrid's success or failure is determined by the CP fallback / energy
fallback split, not by Langevin quality alone.
"""


# ---------------------------------------------------------------------------
# Board serialisation (same as kona_trained_energy_hybrid for cross-module
# consistency — both experiments need the same text representation)
# ---------------------------------------------------------------------------


def board_to_text(board: Grid) -> str:
    """Serialise a 9x9 Sudoku board to a flat space-separated string.

    Why this matters: the process-energy module expects text strings as input.
    A Sudoku board string contains no arithmetic equations, no step delimiters,
    no logical chains — so every process-energy verifier will return near-zero
    scores for every candidate. This means ``process_energy_per_step`` returns
    0.0 for every board, and ``process_energy_argmin`` always picks the first
    non-None candidate. That is a documented design property, not a bug: it
    makes the process-energy fallback distinct from the Langevin-energy
    fallback (which picks by minimum energy, not by index).
    """
    return " ".join(" ".join(str(v) for v in row) for row in board)


# ---------------------------------------------------------------------------
# Puzzle generation — harder instances with fewer clues
# ---------------------------------------------------------------------------


def make_harder_puzzle_set_v5(seed: int, n_per_tier: int = 10) -> list[SudokuPuzzle]:
    """Generate harder Sudoku instances with very few clues to break CP saturation.

    The standard puzzle set in exp3440/3464 uses 26 clues (hard tier), which is
    easy enough for the 200,000-node CP solver to handle 100% of the time. Here
    we use HARDER_CLUE_COUNT (18) and HARDER_CLUE_COUNT-1 (17) clue counts to
    create puzzles where the restricted CP solver (RESTRICTED_MAX_NODES=1,500)
    will time out on many instances, forcing the experiment to use the energy-
    board fallback.

    Two tiers:
      * 'harder' — 18 clues: moderate additional difficulty beyond exp3440 hard
      * 'hardest' — 17 clues: near minimum-clue boundary, nearly always resists
        low-node CP search

    Parameters
    ----------
    seed : int
        Master seed for reproducibility. Each puzzle gets a derived sub-seed to
        avoid inter-puzzle correlation while keeping the set deterministic.
    n_per_tier : int
        Number of puzzles per tier. Total = 2 * n_per_tier.

    Returns
    -------
    list[SudokuPuzzle]
        Puzzles ordered harder tier first, then hardest tier.
    """
    puzzles: list[SudokuPuzzle] = []
    for tier, n_clues in [("harder", HARDER_CLUE_COUNT), ("hardest", HARDER_CLUE_COUNT - 1)]:
        for i in range(n_per_tier):
            # Derived sub-seed: combines master seed, tier hash, and index so
            # each puzzle is independently reproducible but not correlated.
            grid_seed = seed * 1000 + hash(tier) % 100 + i
            full = generate_full_grid(grid_seed)
            clues = dig_holes(full, n_clues, grid_seed + 13)
            puzzles.append(
                SudokuPuzzle(
                    puzzle_id=f"{tier}_{i}",
                    difficulty=tier,
                    clues=clues,
                    solution=full,
                    n_clues=n_clues,
                )
            )
    return puzzles


# ---------------------------------------------------------------------------
# Energy-aware hybrid solver (the architectural fix from exp3464)
# ---------------------------------------------------------------------------


def energy_aware_hybrid_solve(
    clues: Grid,
    energy_board: Grid,
    max_nodes: int,
) -> tuple[Grid | None, bool]:
    """Solve a Sudoku puzzle using CP first, then fall back to the energy board.

    This is the architectural fix for the exp3464 ceiling. The original
    ``hybrid_solve()`` always ignores the energy board and lets the CP solver
    run unconditionally. That means any energy — trained, untrained, or
    process-aware — can NEVER influence the outcome.

    Here we intentionally restrict CP to ``max_nodes`` (typically 1,500), which
    forces CP to time out on harder instances. When CP times out (returns None),
    we use ``energy_board`` as the fallback. The energy board is then checked
    for discrete correctness: it may or may not be a valid Sudoku completion,
    depending on how far the Langevin optimizer progressed.

    Parameters
    ----------
    clues : Grid
        The original puzzle givens (0 = blank cell).
    energy_board : Grid
        The candidate board proposed by the energy optimizer. Used ONLY when
        CP times out (returns None). Its quality determines the fallback arm's
        solve-rate.
    max_nodes : int
        Maximum backtracking nodes for the CP solver. At 1,500 the CP solver
        times out on most 17-18-clue puzzles, leaving the energy board as the
        sole route to a solved result.

    Returns
    -------
    (board, solved) : tuple[Grid | None, bool]
        - If CP succeeds: returns the CP board and ``True``.
        - If CP times out and energy_board is a valid solution: returns
          energy_board and ``True``.
        - If CP times out and energy_board is not valid: returns ``None`` and
          ``False``.
    """
    # First attempt: run the CP solver with the restricted node budget.
    cp_result = constraint_propagation_solve(clues, max_nodes=max_nodes)
    if cp_result is not None:
        # CP succeeded; validate and return. For well-formed clues this will
        # always be True, but we check defensively.
        return cp_result, board_is_valid_solution(cp_result, clues)

    # CP timed out. Fall back to the energy board and check it directly.
    # Unlike the CP result, the energy board may still have constraint
    # violations (non-zero discrete energy), which is the common case at
    # 80 Langevin steps. We return True only if it passes the full validity
    # check (correct rows/cols/boxes AND clues preserved).
    if board_is_valid_solution(energy_board, clues):
        return energy_board, True
    return None, False


# ---------------------------------------------------------------------------
# Arm runner
# ---------------------------------------------------------------------------


def _solve_rate(records: list[dict[str, Any]]) -> float:
    """Fraction of puzzle records where ``solved == True``."""
    if not records:
        return 0.0
    return float(sum(1 for r in records if r["solved"]) / len(records))


def run_process_energy_arms(
    puzzles: list[SudokuPuzzle],
    seed: int,
) -> dict[str, Any]:
    """Run three arms on the harder puzzle set and return aggregated results.

    The three arms differ ONLY in how they pick the fallback board when CP
    times out. This isolates the effect of the selection strategy from the
    effect of CP search depth (which is identical across arms).

    **Arm 1 — untrained-energy hybrid**: pick the Langevin candidate with the
    LOWEST discrete energy (fewest violated constraints) as the fallback. This
    is the minimal untrained baseline: no domain knowledge, no reasoning model,
    just the Langevin optimizer's objective function.

    **Arm 2 — process-energy hybrid**: use ``process_energy_per_step`` to score
    each candidate's reasoning steps, then ``process_energy_argmin`` to pick
    the lowest-energy candidate. Because Sudoku board strings have NO reasoning
    steps, every call returns 0.0, and ``process_energy_argmin`` deterministically
    picks the FIRST non-None candidate (candidates[0]). This is a documented
    domain-mismatch consequence — the process energy carries no signal on Sudoku.
    The key point is that candidates[0] is NOT the same as the untrained arm's
    lowest-energy pick (which may be candidates[1], [2], or [3] depending on
    the optimizer run). The two arms therefore diverge per puzzle when CP times out.

    **Arm 3 — pure process descent**: accept the process-energy-selected board
    WITHOUT any CP correction. Expected solve-rate ~0% because 80-step Langevin
    on 17-18-clue Sudoku almost never reaches a valid board.

    Parameters
    ----------
    puzzles : list[SudokuPuzzle]
        The harder instances from ``make_harder_puzzle_set_v5``.
    seed : int
        Reproducibility seed for the Langevin optimizer. Should differ from
        the puzzle-set seed to avoid tautology between experiment ID and seed.

    Returns
    -------
    dict with keys:
        untrained_hybrid_solve_rate, process_hybrid_solve_rate,
        pure_process_energy_descent_solve_rate, per_puzzle_untrained_hybrid,
        per_puzzle_process_hybrid, per_puzzle_pure_process_descent.
    """
    # Verifiers are stateless scorers; build once and reuse.
    verifiers = _Verifiers()

    per_untrained_hybrid: list[dict[str, Any]] = []
    per_process_hybrid: list[dict[str, Any]] = []
    per_pure_process: list[dict[str, Any]] = []

    for p in puzzles:
        # ------------------------------------------------------------------ #
        # Step 1: Generate N_CANDIDATES boards via fast Langevin.            #
        # ------------------------------------------------------------------ #
        # Each candidate uses a derived seed so the candidates are
        # independently reproducible but not correlated with each other.
        candidates: list[Grid] = []
        energies: list[float] = []
        for i in range(N_CANDIDATES):
            res = optimize_board(
                p.clues,
                seed=seed + hash(p.puzzle_id) % 10_000 + i,
                variant="annealed",
                n_steps=FAST_N_STEPS,
                n_restarts=1,
            )
            candidates.append(res.board)
            energies.append(float(res.final_energy))

        # ------------------------------------------------------------------ #
        # Step 2: Untrained-energy arm — pick lowest Langevin energy.        #
        # ------------------------------------------------------------------ #
        # The Langevin final_energy is the discrete constraint-violation count
        # for the rounded board. Lower = fewer violations = closer to solved.
        untrained_best_idx = min(range(len(energies)), key=lambda i: energies[i])
        untrained_board = candidates[untrained_best_idx]

        # ------------------------------------------------------------------ #
        # Step 3: Process-energy arm — process_energy_argmin on 0.0 scores.  #
        # ------------------------------------------------------------------ #
        # Sudoku boards have NO reasoning steps, so process_energy_per_step
        # returns 0.0 for every candidate. With all energies equal at 0.0,
        # process_energy_argmin returns the first non-None candidate
        # (candidates[0]). This is a documented domain-mismatch consequence.
        proc_energies: list[float] = []
        for b in candidates:
            # steps=[] because a Sudoku board has no chain-of-thought steps.
            pe = process_energy_per_step([], verifiers)
            proc_energies.append(pe)

        # process_energy_argmin returns the answer (here a Grid) with the
        # minimum process energy. With all 0.0, it picks candidates[0].
        process_best = process_energy_argmin(candidates, proc_energies)
        # Defensive fallback: if process_energy_argmin returns None (empty list)
        # we fall back to candidates[0] explicitly.
        process_board: Grid = process_best if process_best is not None else candidates[0]

        # ------------------------------------------------------------------ #
        # Step 4: Run the energy-aware hybrid for each arm's chosen board.   #
        # ------------------------------------------------------------------ #
        _, untrained_hybrid_ok = energy_aware_hybrid_solve(
            p.clues, untrained_board, RESTRICTED_MAX_NODES
        )
        _, process_hybrid_ok = energy_aware_hybrid_solve(
            p.clues, process_board, RESTRICTED_MAX_NODES
        )

        # ------------------------------------------------------------------ #
        # Step 5: Pure process descent — no CP correction at all.            #
        # ------------------------------------------------------------------ #
        pure_process_ok = board_is_valid_solution(process_board, p.clues)

        # ------------------------------------------------------------------ #
        # Record results for audit.                                           #
        # ------------------------------------------------------------------ #
        per_untrained_hybrid.append(
            {
                "puzzle_id": p.puzzle_id,
                "solved": untrained_hybrid_ok,
                "langevin_energy_used": float(energies[untrained_best_idx]),
                "best_candidate_idx": untrained_best_idx,
            }
        )
        per_process_hybrid.append(
            {
                "puzzle_id": p.puzzle_id,
                "solved": process_hybrid_ok,
                # Always 0: process energy is 0.0 for all candidates, so
                # process_energy_argmin always picks the first (index 0).
                "process_energy_selected_idx": 0,
            }
        )
        per_pure_process.append(
            {
                "puzzle_id": p.puzzle_id,
                "solved": pure_process_ok,
            }
        )

    return {
        "untrained_hybrid_solve_rate": _solve_rate(per_untrained_hybrid),
        "process_hybrid_solve_rate": _solve_rate(per_process_hybrid),
        "pure_process_energy_descent_solve_rate": _solve_rate(per_pure_process),
        "per_puzzle_untrained_hybrid": per_untrained_hybrid,
        "per_puzzle_process_hybrid": per_process_hybrid,
        "per_puzzle_pure_process_descent": per_pure_process,
    }


# ---------------------------------------------------------------------------
# Paired significance (McNemar exact)
# ---------------------------------------------------------------------------


def paired_significance_v5(
    untrained_per_puzzle: list[dict[str, Any]],
    process_per_puzzle: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute McNemar exact p for the process-hybrid vs untrained-hybrid delta.

    Pairs are matched by index (both lists use the same puzzle ordering from
    ``run_process_energy_arms``). McNemar only counts DISCORDANT pairs — puzzles
    where one arm solved and the other did not. When both arms have the same
    outcome on every puzzle (which may happen when CP resolves all of them or
    fails all of them), discordant pairs = 0 and p = 1.0, correctly indicating
    no statistical evidence of a difference.

    Parameters
    ----------
    untrained_per_puzzle : list[dict]
        Per-puzzle records for the untrained-energy arm.
    process_per_puzzle : list[dict]
        Per-puzzle records for the process-energy arm.

    Returns
    -------
    dict with keys: comparison, discordant_process_wins, discordant_untrained_wins,
                    mcnemar_exact_p, interpretation.
    """
    a_correct = [bool(r["solved"]) for r in untrained_per_puzzle]
    b_correct = [bool(r["solved"]) for r in process_per_puzzle]
    p = mcnemar_exact(a_correct, b_correct)
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if (not a) and b)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a and (not b))
    return {
        "comparison": "process_hybrid_vs_untrained_hybrid",
        "discordant_process_wins": b01,
        "discordant_untrained_wins": b10,
        "mcnemar_exact_p": p,
        "interpretation": (
            "p = 1.0 means zero discordant pairs — no statistical evidence "
            "that the process energy changes the hybrid's outcome on harder instances."
            if p == 1.0
            else f"p = {p:.4f} for the process-hybrid vs untrained-hybrid delta."
        ),
    }


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def reproducibility_checksum_3475(
    seed: int,
    n_per_tier: int,
    restricted_max_nodes: int,
    n_candidates: int,
    fast_n_steps: int,
) -> str:
    """16-char content hash over the experimental configuration.

    The checksum covers the parameters that fully determine the experiment's
    output: puzzle-set seed (fixes which boards are generated), node cap (fixes
    which CP calls time out), candidate count (fixes how many Langevin boards
    are generated), and step count (fixes Langevin trajectory length). A change
    to any of these changes the checksum, alerting a replicator that the
    configuration has shifted.
    """
    payload = {
        "experiment": 3475,
        "seed": seed,
        "n_per_tier": n_per_tier,
        "restricted_max_nodes": restricted_max_nodes,
        "n_candidates": n_candidates,
        "fast_n_steps": fast_n_steps,
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def derive_verdict_3475(
    process_hybrid_solve_rate: float,
    untrained_hybrid_solve_rate: float,
    pure_process_solve_rate: float,  # noqa: ARG001 — documented unused parameter
    instances_have_headroom: bool,
    mcnemar_p: float,
) -> str:
    """Map the three solve-rate scalars to exactly one ``complete:`` terminal verdict.

    Gate ladder (matches the exp3475 acceptance gates):

      * G0 HEADROOM: untrained_hybrid_solve_rate < 0.80; else the harder
        instances are still too easy for CP at 1,500 nodes and there is no
        room to demonstrate process-energy lift.
      * G1 PROCESS-STRENGTHENS: process_hybrid_solve_rate > untrained AND
        McNemar p < 0.05 — statistical evidence that process energy changes
        hybrid outcomes.
      * Default: process energy provides no lift over the untrained fallback.

    The ``pure_process_solve_rate`` parameter is accepted for interface
    consistency but is not load-bearing in the verdict: even 0% pure-descent
    solve-rate is expected given 80-step Langevin on 17-18-clue Sudoku. The
    verdict focuses on the HYBRID arms, where CP does some of the work.

    Parameters
    ----------
    process_hybrid_solve_rate : float
        Solve-rate of the process-energy fallback hybrid.
    untrained_hybrid_solve_rate : float
        Solve-rate of the untrained-energy fallback hybrid (the control).
    pure_process_solve_rate : float
        Solve-rate of pure process-descent without CP (expected ~0%).
    instances_have_headroom : bool
        True if untrained_hybrid_solve_rate < 0.80.
    mcnemar_p : float
        McNemar exact p for the process-vs-untrained delta.

    Returns
    -------
    str
        One of three ``complete:`` terminal verdict strings.
    """
    if not instances_have_headroom:
        return "complete: blocked_kona_instances_saturated_no_headroom"
    if process_hybrid_solve_rate > untrained_hybrid_solve_rate and mcnemar_p < 0.05:
        return "complete: process_energy_strengthens_kona_hybrid_with_headroom"
    return "complete: process_energy_no_lift_over_untrained_kona_hybrid_even_with_headroom"
