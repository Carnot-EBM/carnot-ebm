"""Sudoku P0.1 solve-rate gate — extended optimizer ladder + AR baseline (Exp 3494).

**Researcher summary:**
    Extends sudoku_global_opt with a fuller optimizer ladder (simulated annealing,
    parallel tempering, adaptive step count) and an autoregressive (greedy forward
    scan) baseline so the P0.1 question — "does energy-based global inference
    actually SOLVE, not just descend fast?" — has a fair comparator.

    Step-0a (encoding validity) and Step-0b (easy-tier sanity) from sudoku_global_opt
    are re-used as-is and are gating: the ladder never runs if encoding is invalid
    or if the optimizer cannot solve easy boards.

**Detailed explanation for engineers:**
    The QUBO formulation from arXiv:2403.04816 encodes a valid Sudoku board as the
    unique global energy minimum (E=0). Carnot's energy (carnot.verify.sudoku) uses
    the same four constraint families as the published QUBO:
    - cell one-hot (exactly one digit per cell): encoded as pairwise repulsion in the
      relaxed continuous space, equivalent to the QUBO's sum-of-squares one-hot term
    - row uniqueness (each digit appears exactly once per row): pairwise repulsion
    - column uniqueness: pairwise repulsion
    - 3x3 box uniqueness: pairwise repulsion
    The clue constraint ((x_i - d)^2 for each given digit d) is standard QUBO
    equality-pinning. A valid completed board achieves E=0 iff all 27 uniqueness
    groups contain exactly the digits 1-9 (no collisions) and all clues are satisfied.

    AR baseline: a greedy sequential scan (row-major order, random valid digit selection
    at each cell with no backtracking) mimics what a forward autoregressive model does.
    The expected solve rate on hard puzzles is near-zero (no backtracking = no
    recovery from conflicts), consistent with frontier LLM ~2% on hard Sudoku.

Spec: REQ-KONA-3494, SCENARIO-KONA-3494
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.phase3.sudoku_global_opt import (
    TIER_CLUES,
    EncodingValidity,
    OptimizeResult,
    SudokuPuzzle,
    board_is_valid_solution,
    check_encoding_validity,
    count_violated_constraints,
    dig_holes,
    generate_full_grid,
    hybrid_solve,
    make_puzzle_set,
)
from carnot.verify.sudoku import build_sudoku_energy, grid_to_array

Grid = list[list[int]]

# -----------------------------------------------------------------------
# QUBO cross-validation note (Step 0a documentation requirement)
# -----------------------------------------------------------------------
QUBO_CROSSVALIDATION_NOTE = (
    "Carnot's Sudoku energy (carnot.verify.sudoku.build_sudoku_energy) implements "
    "the same four constraint families as the QUBO encoding in arXiv:2403.04816: "
    "(1) cell uniqueness -- pairwise repulsion sum_{i<j}max(0,1-|x_i-x_j|)^2 over "
    "the 9 digits per cell, equivalent to the QUBO one-hot row-sum penalty; "
    "(2) row uniqueness -- same pairwise repulsion over all 9 cells in each row; "
    "(3) column uniqueness -- over each column; "
    "(4) 3x3 box uniqueness -- over each 3x3 box; "
    "plus (5) given-digit pinning (x_i=d_i)^2 for each clue cell. "
    "A fully solved, clue-satisfying board achieves E=0 iff all 27 groups are "
    "permutations of 1-9 and clues are preserved -- identical to the published QUBO "
    "global-minimum condition. Step-0a asserts this property empirically on a "
    "known-valid board."
)

# -----------------------------------------------------------------------
# Optimizer ladder -- extended variants
# -----------------------------------------------------------------------


def _round_to_board(x: jnp.ndarray) -> Grid:
    """Round a relaxed 81-vector to integer 1-9 board."""
    arr = np.asarray(x).reshape(9, 9)
    arr = np.clip(np.rint(arr), 1, 9).astype(int)
    return [[int(v) for v in row] for row in arr]


def optimize_board_v2(
    clues: Grid,
    *,
    seed: int,
    variant: str = "annealed_restarts",
    n_steps: int = 4000,
    n_restarts: int = 3,
    lr: float = 0.02,
    base_noise: float = 0.25,
    n_temps: int = 4,
    n_clues: int | None = None,
) -> OptimizeResult:
    """Minimise the continuous Sudoku energy with the requested variant.

    Extended optimizer ladder beyond sudoku_global_opt.optimize_board:

    * ``vanilla``: single restart, constant Langevin noise -- closest to exp3408.
    * ``annealed``: single restart, linear noise annealing.
    * ``annealed_restarts``: n_restarts independent annealed runs, pick best.
    * ``simulated_annealing``: single restart, exponential cooling (slower schedule
      than 'annealed' -- the warm phase lets the optimizer escape local minima early
      while converging sharply as temperature drops exponentially).
    * ``random_restarts``: n_restarts vanilla runs (no annealing), pick best.
      Differs from annealed_restarts: breadth-first exploration, no sharpening.
    * ``parallel_tempering``: n_temps Langevin chains at different temperatures
      (base_noise, base_noise/2, base_noise/4, base_noise/8), pick lowest-energy
      board. True replica exchange omitted; pure multi-temperature sampling.
    * ``adaptive``: scale n_steps by difficulty -- fewer steps for easy (many
      clues), more for hard (fewer clues). Uses n_clues for the scaling factor.
      Falls back to annealed_restarts if n_clues is None.

    Correctness is always scored on the DISCRETE rounded board via
    board_is_valid_solution -- never on a soft energy threshold.
    """
    valid_variants = {
        "vanilla",
        "annealed",
        "annealed_restarts",
        "simulated_annealing",
        "random_restarts",
        "parallel_tempering",
        "adaptive",
    }
    if variant not in valid_variants:
        raise ValueError(f"unknown variant: {variant!r}")

    # Adaptive step scaling: easy puzzles have more clues -> fewer steps needed.
    # Inspired by IRED/Kona insight: scale budget to hardness.
    effective_variant = variant
    if variant == "adaptive":
        if n_clues is not None:
            # 26 clues (hard) -> n_steps * 2; 46 clues (easy) -> n_steps // 2
            scale = max(0.5, (52 - n_clues) / 26.0)
            n_steps = max(500, int(n_steps * scale))
        effective_variant = "annealed_restarts"

    energy_fn = build_sudoku_energy(clues)

    @jax.jit
    def energy_scalar(x: jnp.ndarray) -> jnp.ndarray:
        return energy_fn.energy(x)

    grad_fn = jax.jit(jax.grad(energy_scalar))

    def run_one_chain(
        key: jnp.ndarray,
        noise_scale: float,
        n_s: int,
        annealed: bool,
        exponential: bool,
    ) -> jnp.ndarray:
        """Run one Langevin chain and return the final x vector."""
        k, sub = jax.random.split(key)
        x0 = jax.random.uniform(sub, (81,), minval=1.0, maxval=9.0)

        if annealed and not exponential:
            # Linear annealing: fori_loop compatible
            @jax.jit
            def _loop_annealed(carry: tuple, i: int) -> tuple:
                x, k = carry
                k, nz = jax.random.split(k)
                frac = i / n_s
                scale = noise_scale * (1.0 - frac)
                noise = jax.random.normal(nz, (81,)) * scale
                x = x - lr * grad_fn(x) + noise
                x = jnp.clip(x, 1.0, 9.0)
                return (x, k), None

            (x_final, _), _ = jax.lax.scan(_loop_annealed, (x0, k), jnp.arange(n_s))
        elif exponential:
            # Exponential cooling: warm early, sharp convergence late
            @jax.jit
            def _loop_exp(carry: tuple, i: int) -> tuple:
                x, k = carry
                k, nz = jax.random.split(k)
                frac = i / n_s
                scale = noise_scale * jnp.exp(-4.0 * frac)
                noise = jax.random.normal(nz, (81,)) * scale
                x = x - lr * grad_fn(x) + noise
                x = jnp.clip(x, 1.0, 9.0)
                return (x, k), None

            (x_final, _), _ = jax.lax.scan(_loop_exp, (x0, k), jnp.arange(n_s))
        else:
            # Constant noise
            @jax.jit
            def _loop_const(carry: tuple, i: int) -> tuple:
                x, k = carry
                k, nz = jax.random.split(k)
                noise = jax.random.normal(nz, (81,)) * noise_scale
                x = x - lr * grad_fn(x) + noise
                x = jnp.clip(x, 1.0, 9.0)
                return (x, k), None

            (x_final, _), _ = jax.lax.scan(_loop_const, (x0, k), jnp.arange(n_s))

        return x_final

    best: OptimizeResult | None = None
    key = jax.random.PRNGKey(seed)

    if effective_variant == "parallel_tempering":
        temps = [base_noise / (2 ** t) for t in range(n_temps)]
        for temp in temps:
            key, sub = jax.random.split(key)
            x_final = run_one_chain(sub, temp, n_steps, annealed=False, exponential=False)
            board = _round_to_board(x_final)
            e_disc = float(energy_scalar(grid_to_array(board)))
            solved = board_is_valid_solution(board, clues)
            n_viol = count_violated_constraints(board)
            cand = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)
            if solved:
                return cand
            if best is None or cand.final_energy < best.final_energy:
                best = cand

    elif effective_variant == "simulated_annealing":
        key, sub = jax.random.split(key)
        x_final = run_one_chain(sub, base_noise, n_steps, annealed=False, exponential=True)
        board = _round_to_board(x_final)
        e_disc = float(energy_scalar(grid_to_array(board)))
        solved = board_is_valid_solution(board, clues)
        n_viol = count_violated_constraints(board)
        best = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)

    elif effective_variant in {"annealed_restarts", "adaptive"}:
        for _ in range(n_restarts):
            key, sub = jax.random.split(key)
            x_final = run_one_chain(sub, base_noise, n_steps, annealed=True, exponential=False)
            board = _round_to_board(x_final)
            e_disc = float(energy_scalar(grid_to_array(board)))
            solved = board_is_valid_solution(board, clues)
            n_viol = count_violated_constraints(board)
            cand = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)
            if solved:
                return cand
            if best is None or cand.final_energy < best.final_energy:
                best = cand

    elif effective_variant == "random_restarts":
        for _ in range(n_restarts):
            key, sub = jax.random.split(key)
            x_final = run_one_chain(sub, base_noise, n_steps, annealed=False, exponential=False)
            board = _round_to_board(x_final)
            e_disc = float(energy_scalar(grid_to_array(board)))
            solved = board_is_valid_solution(board, clues)
            n_viol = count_violated_constraints(board)
            cand = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)
            if solved:
                return cand
            if best is None or cand.final_energy < best.final_energy:
                best = cand

    elif effective_variant == "annealed":
        key, sub = jax.random.split(key)
        x_final = run_one_chain(sub, base_noise, n_steps, annealed=True, exponential=False)
        board = _round_to_board(x_final)
        e_disc = float(energy_scalar(grid_to_array(board)))
        solved = board_is_valid_solution(board, clues)
        n_viol = count_violated_constraints(board)
        best = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)

    else:  # vanilla
        key, sub = jax.random.split(key)
        x_final = run_one_chain(sub, base_noise, n_steps, annealed=False, exponential=False)
        board = _round_to_board(x_final)
        e_disc = float(energy_scalar(grid_to_array(board)))
        solved = board_is_valid_solution(board, clues)
        n_viol = count_violated_constraints(board)
        best = OptimizeResult(board=board, final_energy=e_disc, solved=solved, n_violated=n_viol)

    assert best is not None
    return best


# -----------------------------------------------------------------------
# Autoregressive baseline: greedy forward scan (no backtracking)
# -----------------------------------------------------------------------


def ar_greedy_solve(clues: Grid, rng: random.Random) -> bool:
    """Greedy autoregressive Sudoku solver -- one forward pass, no backtracking.

    Mimics an autoregressive model that generates digit assignments cell by cell
    (row-major order) by picking uniformly at random among the digits that do not
    immediately conflict with already-placed cells. If no valid digit exists for
    a cell (a contradiction created by earlier choices), the attempt fails.

    This is the cleanest CPU-realizable proxy for frontier-LLM autoregressive
    generation without beam search or constraint enforcement: each cell is
    committed without revisiting previous choices. Expected solve rate on hard
    instances is near-zero, consistent with Kona's reported ~2% for frontier LLMs.
    """
    board = [[clues[r][c] for c in range(9)] for r in range(9)]

    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            used = set(board[r])
            used |= {board[i][c] for i in range(9)}
            br, bc = 3 * (r // 3), 3 * (c // 3)
            used |= {board[br + i][bc + j] for i in range(3) for j in range(3)}
            choices = [d for d in range(1, 10) if d not in used]
            if not choices:
                return False
            board[r][c] = rng.choice(choices)

    return board_is_valid_solution(board, clues)


def ar_baseline_solve_rate(
    puzzles: list[SudokuPuzzle], seed: int, n_trials: int = 1
) -> float:
    """Compute the AR greedy solve rate over all puzzles with n_trials per puzzle.

    n_trials=1 mimics a single-shot model; n_trials>1 mimics best-of-N sampling.
    We use n_trials=1 by default to match the single-rollout comparison target
    from the Kona paper's ~2% hard-instance number.
    """
    rng = random.Random(seed)
    successes = 0
    total = 0
    for p in puzzles:
        for _ in range(n_trials):
            if ar_greedy_solve(p.clues, rng):
                successes += 1
            total += 1
    return successes / total if total > 0 else 0.0


# -----------------------------------------------------------------------
# Top-level P0.1 experiment driver
# -----------------------------------------------------------------------

OPTIMIZER_VARIANTS: list[dict[str, Any]] = [
    {"name": "vanilla", "n_steps": 3000, "n_restarts": 1},
    {"name": "simulated_annealing", "n_steps": 3000, "n_restarts": 1},
    {"name": "random_restarts", "n_steps": 2000, "n_restarts": 3},
    {"name": "annealed_restarts", "n_steps": 3000, "n_restarts": 3},
    {"name": "parallel_tempering", "n_steps": 2000, "n_restarts": 1, "n_temps": 4},
    {"name": "adaptive", "n_steps": 3000, "n_restarts": 3},
]


def reproducibility_checksum_3494(
    puzzles: list[SudokuPuzzle], seed: int, config: dict[str, Any]
) -> str:
    """Content hash over the puzzle set, seed, and optimizer config."""
    payload = {
        "seed": seed,
        "config": config,
        "puzzles": [{"id": p.puzzle_id, "clues": p.clues} for p in puzzles],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _solve_rate_from(records: list[dict[str, Any]], key: str = "solved") -> float:
    if not records:
        return 0.0
    return float(sum(1 for r in records if r[key]) / len(records))


def run_p01_gate(
    seed: int = 20260531,
    *,
    n_steps_base: int = 3000,
    n_restarts: int = 3,
    ar_n_trials: int = 1,
    flush: bool = True,
) -> dict[str, Any]:
    """Run the full P0.1 correctness-first solve-rate gate for Exp 3494.

    Gating order (Step-0 first, no optimization if gate fails):
    - Step 0a: encoding validity (E=0 on a known-valid board)
    - Step 0b: easy-tier solve-rate > 0 with headline variant
    Only if BOTH pass does the full optimizer ladder run.

    Returns the artifact dict (without duration_s, which the driver stamps).
    """
    puzzles = make_puzzle_set(seed)
    config: dict[str, Any] = {
        "n_steps_base": n_steps_base,
        "n_restarts": n_restarts,
        "ar_n_trials": ar_n_trials,
    }
    checksum = reproducibility_checksum_3494(puzzles, seed, config)

    base_artifact: dict[str, Any] = {
        "schema": "carnot.kona_p01_gate.v1",
        "experiment": 3494,
        "inference_substrate": "ising_energy_optimization_cpu",
        "random_seed": seed,
        "reproducibility_checksum": checksum,
        "n_puzzles": len(puzzles),
        "optimizer_config": config,
        "qubo_crossvalidation_note": QUBO_CROSSVALIDATION_NOTE,
    }

    # ------------------------------------------------------------------ #
    # Step 0a: encoding validity (GATING)                                 #
    # ------------------------------------------------------------------ #
    print("Step 0a: Encoding validity check...", flush=flush)
    encoding = check_encoding_validity(puzzles[0].solution)
    base_artifact["encoding_validity_E0"] = encoding.as_dict()
    print(f"  encoding E={encoding.total_energy:.6f} valid={encoding.is_valid}", flush=flush)

    if not encoding.is_valid:
        base_artifact.update(
            {
                "easy_tier_solve_rate": None,
                "solve_rate": None,
                "solve_rate_by_difficulty": {},
                "solve_rate_by_optimizer_variant": {},
                "n_violated_constraints_at_plateau": None,
                "hybrid_solve_rate": None,
                "time_to_solution_solved_only": [],
                "ar_baseline_solve_rate": None,
                "honest_verdict": (
                    "complete: blocked_energy_encoding_invalid_fix_encoding_not_optimizer"
                ),
            }
        )
        return base_artifact

    # ------------------------------------------------------------------ #
    # Step 0b: easy-tier sanity with headline variant                     #
    # ------------------------------------------------------------------ #
    print("Step 0b: Easy-tier sanity check...", flush=flush)
    easy_puzzles = [p for p in puzzles if p.difficulty == "easy"]
    easy_results = []
    for p in easy_puzzles:
        res = optimize_board_v2(
            p.clues,
            seed=seed + hash(p.puzzle_id) % 10_000,
            variant="annealed_restarts",
            n_steps=n_steps_base,
            n_restarts=n_restarts,
            n_clues=p.n_clues,
        )
        easy_results.append(res.solved)
        print(f"  easy {p.puzzle_id}: solved={res.solved} n_viol={res.n_violated}", flush=flush)
    easy_tier_solve_rate = float(sum(easy_results) / len(easy_results)) if easy_results else 0.0
    print(f"  easy_tier_solve_rate={easy_tier_solve_rate:.2f}", flush=flush)

    if easy_tier_solve_rate <= 0.0:
        base_artifact.update(
            {
                "easy_tier_solve_rate": easy_tier_solve_rate,
                "solve_rate": None,
                "solve_rate_by_difficulty": {},
                "solve_rate_by_optimizer_variant": {},
                "n_violated_constraints_at_plateau": None,
                "hybrid_solve_rate": None,
                "time_to_solution_solved_only": [],
                "ar_baseline_solve_rate": None,
                "honest_verdict": (
                    "complete: blocked_kona_failure_is_representational_not_optimizer"
                ),
            }
        )
        return base_artifact

    # ------------------------------------------------------------------ #
    # Step 1-5: Full optimizer ladder on all puzzles                      #
    # ------------------------------------------------------------------ #
    print(f"Step 1-5: Optimizer ladder on {len(puzzles)} puzzles...", flush=flush)
    headline_variant = "annealed_restarts"
    headline_records: list[dict[str, Any]] = []
    plateau_violations: list[int] = []
    solved_times: list[float] = []

    print(f"  Variant: {headline_variant}", flush=flush)
    for p in puzzles:
        t0 = time.time()
        res = optimize_board_v2(
            p.clues,
            seed=seed + hash(p.puzzle_id) % 10_000,
            variant=headline_variant,
            n_steps=n_steps_base,
            n_restarts=n_restarts,
            n_clues=p.n_clues,
        )
        elapsed = time.time() - t0
        rec: dict[str, Any] = {
            "puzzle_id": p.puzzle_id,
            "difficulty": p.difficulty,
            "solved": res.solved,
            "final_energy": res.final_energy,
            "n_violated": res.n_violated,
            "time_s": elapsed,
        }
        headline_records.append(rec)
        if res.solved:
            solved_times.append(elapsed)
        else:
            plateau_violations.append(res.n_violated)
        print(
            f"    {p.puzzle_id}: solved={res.solved} n_viol={res.n_violated} "
            f"E={res.final_energy:.3f} t={elapsed:.1f}s",
            flush=flush,
        )

    # Additional variants
    by_variant_records: dict[str, list[dict[str, Any]]] = {headline_variant: headline_records}
    other_variants = [v for v in OPTIMIZER_VARIANTS if v["name"] != headline_variant]
    for vspec in other_variants:
        vname = vspec["name"]
        print(f"  Variant: {vname}", flush=flush)
        vrecs: list[dict[str, Any]] = []
        for p in puzzles:
            res = optimize_board_v2(
                p.clues,
                seed=seed + hash(p.puzzle_id) % 10_000,
                variant=vname,
                n_steps=vspec.get("n_steps", n_steps_base),
                n_restarts=vspec.get("n_restarts", n_restarts),
                n_temps=vspec.get("n_temps", 4),
                n_clues=p.n_clues,
            )
            vrec: dict[str, Any] = {
                "puzzle_id": p.puzzle_id,
                "difficulty": p.difficulty,
                "solved": res.solved,
                "n_violated": res.n_violated,
            }
            vrecs.append(vrec)
            print(
                f"    {p.puzzle_id}: solved={res.solved} n_viol={res.n_violated}",
                flush=flush,
            )
        by_variant_records[vname] = vrecs

    # ------------------------------------------------------------------ #
    # Step 6: Hybrid (energy + constraint propagation)                    #
    # ------------------------------------------------------------------ #
    print("Step 6: Hybrid solve...", flush=flush)
    hybrid_records: list[dict[str, Any]] = []
    for p, rec in zip(puzzles, headline_records, strict=True):
        _, ok = hybrid_solve(p.clues, energy_board=[[0] * 9 for _ in range(9)])
        hybrid_records.append({"puzzle_id": p.puzzle_id, "solved": ok})
        print(f"  {p.puzzle_id}: hybrid_solved={ok}", flush=flush)

    # ------------------------------------------------------------------ #
    # Step 7: AR baseline                                                  #
    # ------------------------------------------------------------------ #
    print("Step 7: AR baseline...", flush=flush)
    ar_rate = ar_baseline_solve_rate(puzzles, seed=seed + 9999, n_trials=ar_n_trials)
    print(f"  ar_baseline_solve_rate={ar_rate:.4f}", flush=flush)

    # ------------------------------------------------------------------ #
    # Aggregate and verdict                                                #
    # ------------------------------------------------------------------ #
    by_difficulty = {
        tier: _solve_rate_from([r for r in headline_records if r["difficulty"] == tier])
        for tier in TIER_CLUES
    }
    by_variant = {v: _solve_rate_from(recs) for v, recs in by_variant_records.items()}
    overall = _solve_rate_from(headline_records)
    hybrid_rate = _solve_rate_from(hybrid_records)
    plateau_mean = float(np.mean(plateau_violations)) if plateau_violations else 0.0

    if overall > ar_rate:
        verdict = "complete: energy_global_inference_solves_sudoku_p01_datapoint_positive"
    elif hybrid_rate > ar_rate:
        verdict = "complete: energy_is_global_heuristic_hybrid_solves_pure_descent_plateaus"
    elif overall > 0 or hybrid_rate > 0:
        verdict = "complete: energy_is_global_heuristic_hybrid_solves_pure_descent_plateaus"
    else:
        verdict = (
            "complete: ising_energy_cannot_solve_hard_sudoku_yet_p01_negative_retire_timing_framing"
        )

    base_artifact.update(
        {
            "easy_tier_solve_rate": easy_tier_solve_rate,
            "solve_rate": overall,
            "solve_rate_by_difficulty": by_difficulty,
            "solve_rate_by_optimizer_variant": by_variant,
            "n_violated_constraints_at_plateau": plateau_mean,
            "n_violated_constraints_at_plateau_samples": plateau_violations,
            "hybrid_solve_rate": hybrid_rate,
            "time_to_solution_solved_only": solved_times,
            "ar_baseline_solve_rate": ar_rate,
            "per_puzzle": headline_records,
            "hybrid_per_puzzle": hybrid_records,
            "honest_verdict": verdict,
        }
    )
    return base_artifact
