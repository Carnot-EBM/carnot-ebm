"""Experiment 3529: P0.1 Sudoku discriminating hard tier (v1).

CONTEXT:
    exp3517 (.324) hardened the Sudoku positive but is CEILING_SATURATED:
    discrete_sa_single achieves 1.0 on EVERY difficulty tier including the
    20-clue "extreme" tier with 50k sweeps. This means "any non-AR optimizer
    beats AR" not that energy-specific machinery (PT/restarts) is uniquely capable.
    This experiment builds a DISCRIMINATING hard tier where trivial single-SA falls
    below ceiling, making the optimizer-power gradient visible.

    Design:
    - "ultra_hard" tier: 17-clue puzzles (minimum near the Sudoku uniqueness boundary)
    - Budget-matched comparison per arXiv:2506.04596: each optimizer gets equal
      per-unit budget; SA_single gets 1x, restarts get 20x, PT gets 8 chains x 4x
    - discrete_sa_single: 5000 sweeps x 100 moves (reduced from exp3517's 50000)
    - SA_restarts20: 20 x 5000 = 100000 sweeps per puzzle
    - PT_tuned: 8 chains x 20000 sweeps, T_min=0.05, T_max=4.0, interval=50
    - exact_cp confirms solvability (all puzzles satisfiable by construction)

    Literature references:
    - Sudoku-Bench arXiv:2505.16135: SOTA LLMs solve <15% unaided
    - Kona EBM (Logical Intelligence): 96.2% vs frontier LLMs ~2% on hard Sudoku
    - BDH (Pathway): 97.4% on Sudoku Extreme vs leading LLMs ~0%
    - QUBO solver benchmark arXiv:2506.04596: SA/PT/SB/Gurobi under uniform budget
    - Sudoku-QUBO arXiv:2510.19835: QUBO formulation for Sudoku

Spec: REQ-KONA-3529, SCENARIO-KONA-3529

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot &&
    JAX_PLATFORMS=cpu .venv/bin/python
        scripts/experiment_3529_p01_sudoku_headroom_discriminating_tier_v1.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from carnot.phase3.sudoku_global_opt import (
    ENCODING_EPS,
    board_is_valid_solution,
    check_encoding_validity,
    constraint_propagation_solve,
    dig_holes,
    generate_full_grid,
)
from carnot.phase3.sudoku_discrete_sa import (
    parallel_tempering_solve_instrumented,
    sa_solve_restarts,
)

# ---------------------------------------------------------------------------
# Experiment identity — seed is content-derived, NOT the experiment number.
# WHY content-derived: using the experiment number as the seed creates a
# TAUTOLOGY adversarial flag (experiment_id == random_seed). The content hash
# is unique to this experiment's design without being numerically equal.
# ---------------------------------------------------------------------------
_SEED_BYTES = b"exp3529_sudoku_discriminating_tier_ultra_hard_v1"
SEED = int(hashlib.sha256(_SEED_BYTES).hexdigest(), 16) % (2**31)

OUT_PATH = "results/experiment_3529_p01_sudoku_headroom_discriminating_tier_v1.json"

# ---------------------------------------------------------------------------
# Tier configuration.
# WHY 3 tiers x 15 = 45 puzzles: satisfies >= 40 requirement; 3 tiers let us
# compare easy/bridge/discriminating difficulty in the same run.
# WHY 17 clues for ultra_hard: 17 is empirically the minimum number of clues
# for a uniquely-solvable Sudoku (McGuire et al. 2012 proved no 16-clue unique
# puzzles exist). This creates maximum difficulty near the feasibility boundary.
# ---------------------------------------------------------------------------
TIER_CLUES_V4: dict[str, int] = {
    "hard": 26,       # reference tier (same as exp3517)
    "extreme": 20,    # bridge tier (was ceiling-saturated in exp3517)
    "ultra_hard": 17, # new discriminating tier: single-SA must fail here
}
PUZZLES_PER_TIER = 15  # 15 x 3 = 45 >= 40 requirement

# ---------------------------------------------------------------------------
# SA budget — REDUCED for budget-matched comparison.
# WHY reduced: arXiv:2506.04596 mandates that comparisons be made under equal
# compute budgets. exp3517 used 50k sweeps x 100 moves = 5M moves per puzzle
# for the extreme tier, which produced CEILING_SATURATION (solve_rate=1.0
# for single-SA on ALL tiers). With only 500k moves on 17-clue puzzles,
# single-SA should fail ~30-60%, creating discriminating power.
# ---------------------------------------------------------------------------
SA_SINGLE_CFG: dict[str, dict] = {
    "hard":       {"n_sweeps": 5_000, "n_moves": 100, "T_init": 0.5, "T_final": 0.01},
    "extreme":    {"n_sweeps": 5_000, "n_moves": 100, "T_init": 0.5, "T_final": 0.005},
    "ultra_hard": {"n_sweeps": 5_000, "n_moves": 100, "T_init": 0.5, "T_final": 0.005},
}

SA_N_RESTARTS = 20
# SA restarts: same per-restart budget as single, but 20x the total compute.
# WHY same per-restart budget: the budget-matched design says "single gets 1x
# unit, restarts get 20x the same unit". Same unit = same sweep budget per run.
SA_RESTARTS_CFG = SA_SINGLE_CFG  # 20 restarts x 5000 sweeps = 100k sweeps per puzzle

# ---------------------------------------------------------------------------
# PT config — 8 chains, wider temperature range, more sweeps per chain.
# WHY 8 chains (vs 6 in exp3517): wider ensemble helps harder puzzles.
# WHY T_max=4.0 (vs 2.0 in exp3517): ultra_hard landscape requires more
# thermal exploration to escape deep local minima near the uniqueness boundary.
# WHY 20k sweeps for ultra_hard: compensates for the increased n_chains so the
# total PT compute is proportional to the SA_restarts budget.
# ---------------------------------------------------------------------------
PT_CFG_TUNED: dict[str, dict] = {
    "hard":       {"n_sweeps": 5_000,  "n_moves": 100, "n_chains": 8,
                   "T_min": 0.05, "T_max": 3.0, "n_exchange_interval": 50},
    "extreme":    {"n_sweeps": 10_000, "n_moves": 100, "n_chains": 8,
                   "T_min": 0.05, "T_max": 3.0, "n_exchange_interval": 50},
    "ultra_hard": {"n_sweeps": 20_000, "n_moves": 100, "n_chains": 8,
                   "T_min": 0.05, "T_max": 4.0, "n_exchange_interval": 50},
}

# Hard wall-clock budget: 36 minutes.
# WHY a hard budget: the experiment task spec requires graceful termination if
# the SA/PT runs on hard puzzles take longer than expected. Rather than timeout-
# killing the process, we check the budget at each puzzle and write a partial
# artifact if time expires.
BUDGET_SECS = 36 * 60  # 36 min wall-clock cap

# Hardness gate probe size: use a small subset to check quickly.
# WHY 10: probing all 15 ultra_hard puzzles with single-SA takes ~1-2 min;
# 10 is statistically sufficient (if 9+/10 pass, the gate succeeds with ~0.9
# rate which is our threshold).
HARDNESS_GATE_PROBE_N = 10

_RUN_START: float = 0.0


def _elapsed() -> float:
    """Seconds elapsed since _RUN_START."""
    return time.time() - _RUN_START


def _over_budget() -> bool:
    """True if the experiment has exceeded its hard wall-clock budget."""
    return _elapsed() > BUDGET_SECS


# ---------------------------------------------------------------------------
# Puzzle generation — 3 tiers x 15 puzzles = 45 total.
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class SudokuPuzzleV4:
    """One puzzle with its known valid solution and difficulty tag.

    WHY frozen=True: experiment artifacts are immutable once generated; a mutable
    puzzle record would allow silent corruption of the known-good solution field,
    which is used as the encoding-validity reference in step 0a.
    """
    puzzle_id: str
    difficulty: str
    clues: list
    solution: list
    n_clues: int


def make_puzzle_set_v4(
    seed: int = SEED,
    tier_clues: dict[str, int] | None = None,
    puzzles_per_tier: int = PUZZLES_PER_TIER,
) -> list[SudokuPuzzleV4]:
    """Build a deterministic puzzle set spanning 3 difficulty tiers.

    WHY 15 per tier: 15 x 3 = 45 clears the >= 40 sample-size requirement.
    WHY the seed formula (seed + hash(tier) % 10000 + i * 997):
    - hash(tier) separates tiers so that "hard_0" and "extreme_0" start from
      different sequences (tier names hash to different values modulo 10000).
    - i * 997 gives a large spacing within a tier so consecutive puzzles are
      not correlated (997 is prime; spacing >> puzzle-generation period).

    Args:
        seed: base random seed (content-derived, not experiment ID)
        tier_clues: override tier -> n_clues mapping (for hardness gate escalation)
        puzzles_per_tier: number of puzzles to generate per tier

    Returns:
        List of SudokuPuzzleV4 objects in tier order.
    """
    if tier_clues is None:
        tier_clues = TIER_CLUES_V4
    puzzles: list[SudokuPuzzleV4] = []
    for tier, n_clues in tier_clues.items():
        for i in range(puzzles_per_tier):
            grid_seed = seed + hash(tier) % 10_000 + i * 997
            full = generate_full_grid(grid_seed)
            clues = dig_holes(full, n_clues, grid_seed + 37)
            puzzles.append(
                SudokuPuzzleV4(
                    puzzle_id=f"{tier}_{i}",
                    difficulty=tier,
                    clues=clues,
                    solution=full,
                    n_clues=n_clues,
                )
            )
    return puzzles


def _solve_rate(records: list[dict]) -> float:
    """Compute the fraction of records where solved==True.

    WHY float division: the solve rate is reported as a float in [0, 1] for
    consistency with the adversarial_verify schema (float fields) and for
    direct comparison with literature-reported LLM solve rates.
    """
    if not records:
        return 0.0
    return float(sum(1 for r in records if r["solved"]) / len(records))


def _solve_rate_by_difficulty(records: list[dict]) -> dict[str, float]:
    """Group records by difficulty tier and compute per-tier solve rates.

    WHY separate by tier: the core claim of exp3529 is that the optimizer-power
    gradient is tier-dependent — SA_single saturates on easy/hard but fails on
    ultra_hard, while PT/restarts maintain high rates. Without per-tier breakdown
    the gradient is invisible in the aggregate number.
    """
    by_tier: dict[str, list[dict]] = {}
    for r in records:
        by_tier.setdefault(r["difficulty"], []).append(r)
    return {tier: _solve_rate(recs) for tier, recs in by_tier.items()}


def _reproducibility_checksum(
    puzzles: list[SudokuPuzzleV4],
    seed: int,
    config: dict,
) -> str:
    """SHA-256 checksum over the experiment's defining inputs.

    WHY content-addressed: a third party can regenerate the puzzle set from seed
    + config and verify the checksum matches. Diffs in checksum reveal silent
    corpus or config drift between this artifact and any future replication.
    The 'experiment' field uses 3529 (NOT seed) to distinguish this checksum
    from exp3517's checksum even if seed values coincide.
    """
    payload = {
        "experiment": 3529,
        "seed": seed,
        "config": config,
        "puzzles": [{"id": p.puzzle_id, "n_clues": p.n_clues} for p in puzzles],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Optimizer runners.
# ---------------------------------------------------------------------------

def _run_discrete_sa_single(
    puzzles: list[SudokuPuzzleV4],
    seed: int,
    cfg: dict[str, dict] | None = None,
) -> dict:
    """Run discrete SA with a single restart (budget-matched comparison baseline).

    WHY single restart: this is the 'unit' in the budget-matched design.
    SA_restarts gets 20x this unit; PT gets 8 chains x 4x this unit. Single-SA
    is the discriminator: if it saturates on ultra_hard, the experiment is
    ceiling-saturated; if it fails, the power gradient is measurable.

    WHY progress_callback=None: single-SA doesn't need callback; the loop prints
    per-puzzle lines which is sufficient for progress monitoring.
    """
    print("  Variant: discrete_sa_single (1 restart, reduced budget)", flush=True)
    if cfg is None:
        cfg = SA_SINGLE_CFG
    records = []
    for p in puzzles:
        if _over_budget():
            break
        tier_cfg = cfg[p.difficulty]
        _, solved, n_viol = sa_solve_restarts(
            p.clues,
            n_sweeps=tier_cfg["n_sweeps"],
            n_moves_per_sweep=tier_cfg["n_moves"],
            T_init=tier_cfg["T_init"],
            T_final=tier_cfg["T_final"],
            n_restarts=1,
            seed=seed + hash(p.puzzle_id) % 10_000,
        )
        records.append({
            "puzzle_id": p.puzzle_id,
            "difficulty": p.difficulty,
            "solved": solved,
            "n_violated": n_viol,
        })
        print(f"    {p.puzzle_id}: solved={solved} n_viol={n_viol}", flush=True)
    return {"discrete_sa_single": records}


def _run_discrete_sa_restarts(
    puzzles: list[SudokuPuzzleV4],
    seed: int,
    cfg: dict[str, dict] | None = None,
) -> dict:
    """Run discrete SA with 20 restarts (20x compute budget of single-SA).

    WHY 20 restarts: this is 20x the single-SA compute. Each restart is independent
    from a fresh random row fill; the first solved restart is returned immediately
    (early stopping). If no restart solves, the restart with fewest violations is
    returned to measure progress.
    """
    print(f"  Variant: discrete_sa_restarts{SA_N_RESTARTS} (20x budget)", flush=True)
    if cfg is None:
        cfg = SA_RESTARTS_CFG
    records = []

    for p in puzzles:
        if _over_budget():
            break

        def _cb(k: int, n_viol: int, solved: bool) -> None:
            if k % 5 == 0 or solved:
                print(
                    f"    {p.puzzle_id} restart {k}: n_viol={n_viol} solved={solved}",
                    flush=True,
                )

        tier_cfg = cfg[p.difficulty]
        t0 = time.time()
        _, solved, n_viol = sa_solve_restarts(
            p.clues,
            n_sweeps=tier_cfg["n_sweeps"],
            n_moves_per_sweep=tier_cfg["n_moves"],
            T_init=tier_cfg["T_init"],
            T_final=tier_cfg["T_final"],
            n_restarts=SA_N_RESTARTS,
            seed=seed + hash(p.puzzle_id) % 10_000,
            progress_callback=_cb,
        )
        elapsed = time.time() - t0
        records.append({
            "puzzle_id": p.puzzle_id,
            "difficulty": p.difficulty,
            "solved": solved,
            "n_violated": n_viol,
            "time_s": elapsed,
        })
        print(
            f"    {p.puzzle_id}: FINAL solved={solved} n_viol={n_viol} t={elapsed:.1f}s",
            flush=True,
        )

    return {f"discrete_sa_restarts{SA_N_RESTARTS}": records}


def _run_parallel_tempering_tuned(
    puzzles: list[SudokuPuzzleV4],
    seed: int,
) -> dict:
    """Run instrumented PT with 8 chains and a wider temperature range.

    WHY 8 chains (vs 6 in exp3517): harder puzzles benefit from a wider ensemble
    of temperatures. Additional chains fill the gap between T_min and T_max more
    densely, improving replica-exchange probability.
    WHY T_max=4.0 for ultra_hard: the 17-clue landscape has very deep local
    minima near the satisfiability boundary. A hotter maximum temperature is
    needed for the top chain to explore broadly enough for effective mixing.
    WHY interval=50: exp3517 confirmed that n_exchange_interval=50 produces
    sufficient exchange proposals (n_sweeps/50 attempts vs the original n_sweeps//5+1
    that gave only ~4 attempts and killed chain mixing).
    """
    print("  Variant: parallel_tempering_tuned (8 chains, T_max=4.0 for ultra_hard)", flush=True)
    records = []
    swap_rates: list[float] = []

    for p in puzzles:
        if _over_budget():
            break

        def _cb(sweep: int, n_viol: int) -> None:
            print(
                f"    {p.puzzle_id} PT sweep {sweep}: coldest n_viol={n_viol}",
                flush=True,
            )

        cfg = PT_CFG_TUNED[p.difficulty]
        _, solved, n_viol, swap_acc = parallel_tempering_solve_instrumented(
            p.clues,
            n_sweeps=cfg["n_sweeps"],
            n_moves_per_sweep=cfg["n_moves"],
            n_chains=cfg["n_chains"],
            T_min=cfg["T_min"],
            T_max=cfg["T_max"],
            n_exchange_interval=cfg["n_exchange_interval"],
            seed=seed + hash(p.puzzle_id) % 10_000,
            progress_callback=_cb,
        )
        swap_rates.append(swap_acc)
        records.append({
            "puzzle_id": p.puzzle_id,
            "difficulty": p.difficulty,
            "solved": solved,
            "n_violated": n_viol,
            "swap_acceptance_rate": swap_acc,
        })
        print(
            f"    {p.puzzle_id}: solved={solved} n_viol={n_viol} swap_acc={swap_acc:.3f}",
            flush=True,
        )

    mean_swap = float(np.mean(swap_rates)) if swap_rates else 0.0
    return {"parallel_tempering_tuned": records, "_pt_mean_swap_acceptance": mean_swap}


def _run_exact_cp(puzzles: list[SudokuPuzzleV4]) -> dict:
    """Exact constraint-propagation solver (MRV backtracking).

    WHY exact_cp: provides the ground-truth upper bound (all puzzles are
    satisfiable by construction — constraint propagation always finds the solution
    given enough search budget). exact_cp_solve_rate == 1.0 verifies puzzle
    feasibility. If exact_cp < 1.0, the puzzle is not uniquely solvable within
    our search budget and should be escalated.
    """
    print("  Variant: exact_cp (constraint propagation + MRV backtracking)", flush=True)
    records = []
    for p in puzzles:
        if _over_budget():
            break
        t0 = time.time()
        result = constraint_propagation_solve(p.clues, max_nodes=5_000_000)
        elapsed = time.time() - t0
        solved = result is not None and board_is_valid_solution(result, p.clues)
        records.append({
            "puzzle_id": p.puzzle_id,
            "difficulty": p.difficulty,
            "solved": solved,
            "time_s": elapsed,
        })
        print(f"    {p.puzzle_id}: solved={solved} t={elapsed:.2f}s", flush=True)
    return {"exact_cp": records}


def _ar_greedy_baseline(puzzles: list[SudokuPuzzleV4], seed: int) -> float:
    """Naive greedy AR: fill empty cells sequentially with any valid digit.

    WHY greedy (no backtrack): this matches the 'greedy AR' from exp3505/exp3517.
    It is NOT a fair comparator to state-of-the-art LLMs (Sudoku-Bench shows
    frontier LLMs solve <15% on hard boards, not 0%). The greedy baseline is
    reported as ar_greedy_solve_rate and the literature LLM numbers are in
    ar_literature_baselines_note. Together they establish the AR performance
    envelope: the real LLM number lies somewhere between 0% and 15%.

    WHY sequential (row-major): greedy AR fills cells left-to-right, top-to-bottom.
    This is a reasonable approximation to token-by-token LLM generation.
    """
    rng = np.random.default_rng(seed)
    n_solved = 0
    for p in puzzles:
        board = [row[:] for row in p.clues]
        ok = True
        for r in range(9):
            for c in range(9):
                if board[r][c] == 0:
                    used = (
                        {board[r][cc] for cc in range(9) if board[r][cc] != 0}
                        | {board[rr][c] for rr in range(9) if board[rr][c] != 0}
                        | {
                            board[r // 3 * 3 + i][c // 3 * 3 + j]
                            for i in range(3)
                            for j in range(3)
                            if board[r // 3 * 3 + i][c // 3 * 3 + j] != 0
                        }
                    )
                    choices = [d for d in range(1, 10) if d not in used]
                    if not choices:
                        ok = False
                        break
                    board[r][c] = int(rng.choice(choices))
            if not ok:
                break
        if ok and board_is_valid_solution(board, p.clues):
            n_solved += 1
    return n_solved / len(puzzles) if puzzles else 0.0


def _probe_hardness_gate(
    probe_puzzles: list[SudokuPuzzleV4],
    seed: int,
    n_sweeps: int = 5_000,
    n_moves: int = 100,
    T_init: float = 0.5,
    T_final: float = 0.005,
) -> float:
    """Run discrete_sa_single on probe_puzzles and return the solve rate.

    WHY a dedicated probe function: the hardness gate (step 0b) needs to run
    single-SA on ultra_hard puzzles with reduced budget BEFORE committing to the
    full 45-puzzle optimizer ladder. If single-SA solves >= 90% of the probe set,
    the tier is not discriminating and we must escalate or block.

    Args:
        probe_puzzles: subset of ultra_hard puzzles to probe
        seed: base random seed for reproducibility
        n_sweeps, n_moves, T_init, T_final: SA hyperparameters for the probe

    Returns:
        Solve rate (float in [0, 1]) of single-SA on the probe set.
    """
    n_solved = 0
    for p in probe_puzzles:
        _, solved, _ = sa_solve_restarts(
            p.clues,
            n_sweeps=n_sweeps,
            n_moves_per_sweep=n_moves,
            T_init=T_init,
            T_final=T_final,
            n_restarts=1,
            seed=seed + hash(p.puzzle_id) % 10_000,
        )
        if solved:
            n_solved += 1
        print(
            f"    hardness_gate probe {p.puzzle_id}: solved={solved}",
            flush=True,
        )
    return n_solved / len(probe_puzzles) if probe_puzzles else 0.0


def main() -> None:
    global _RUN_START
    _RUN_START = time.time()

    print(
        f"Exp 3529: P0.1 Sudoku discriminating hard tier (v1) — "
        f"start {time.strftime('%H:%M:%S')}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 0a: Build 45-puzzle set and assert encoding validity.
    # WHY gating: if E != 0 on a valid board, NO optimizer can solve it.
    # We check this first to avoid wasting budget on a broken encoding.
    # ------------------------------------------------------------------
    puzzles = make_puzzle_set_v4(SEED)
    print(
        f"Puzzle set: {len(puzzles)} puzzles across "
        f"{list(TIER_CLUES_V4.keys())}",
        flush=True,
    )
    assert len(puzzles) >= 40, f"Expected >= 40 puzzles, got {len(puzzles)}"

    config = {
        "sa_n_restarts": SA_N_RESTARTS,
        "sa_single_cfg": SA_SINGLE_CFG,
        "sa_restarts_cfg": SA_RESTARTS_CFG,
        "pt_cfg_tuned": PT_CFG_TUNED,
        "tier_clues": TIER_CLUES_V4,
        "puzzles_per_tier": PUZZLES_PER_TIER,
        "hardness_gate_probe_n": HARDNESS_GATE_PROBE_N,
    }
    checksum = _reproducibility_checksum(puzzles, SEED, config)

    print("\nStep 0a: Encoding validity regression check...", flush=True)
    encoding = check_encoding_validity(puzzles[0].solution)
    print(f"  E={encoding.total_energy:.8f} is_valid={encoding.is_valid}", flush=True)

    if not encoding.is_valid:
        artifact: dict = {
            "schema": "carnot.kona_p01_gate.v4",
            "experiment": 3529,
            "inference_substrate": "ising_energy_optimization_cpu",
            "random_seed": SEED,
            "reproducibility_checksum": checksum,
            "n_puzzles": len(puzzles),
            "encoding_validity_E0_reasserted": encoding.as_dict(),
            "honest_verdict": "complete: blocked_energy_encoding_invalid_regression",
            "duration_s": _elapsed(),
        }
        os.makedirs("results", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"BLOCKED: encoding invalid. Artifact: {OUT_PATH}", flush=True)
        return

    # ------------------------------------------------------------------
    # Step 0b: Hardness gate.
    # WHY this gate: exp3517 showed CEILING_SATURATION (single-SA solved 100%
    # of all tiers). Before running the full 45-puzzle ladder, we probe a
    # subset of ultra_hard puzzles to confirm single-SA fails. If it doesn't
    # fail, we escalate to 15-clue (the absolute minimum unique-solvable
    # boundary) and reduce the budget further. If it STILL doesn't fail,
    # we block and report — this experiment cannot provide discriminating data.
    # ------------------------------------------------------------------
    print("\nStep 0b: Hardness gate (probing ultra_hard discriminating tier)...", flush=True)

    ultra_hard_puzzles = [p for p in puzzles if p.difficulty == "ultra_hard"]
    probe_puzzles = ultra_hard_puzzles[:HARDNESS_GATE_PROBE_N]

    hardness_sweep: list[dict] = []

    # First probe: 17-clue puzzles with 5k sweeps x 100 moves.
    gate_probe_rate = _probe_hardness_gate(
        probe_puzzles,
        seed=SEED,
        n_sweeps=5_000,
        n_moves=100,
        T_init=0.5,
        T_final=0.005,
    )
    print(
        f"  Hardness gate probe 1 (17-clue, 5k sweeps): SA_single solve_rate={gate_probe_rate:.3f}",
        flush=True,
    )
    hardness_sweep.append({
        "n_clues": 17,
        "n_sweeps": 5_000,
        "probe_solve_rate": gate_probe_rate,
    })

    final_tier_clues = dict(TIER_CLUES_V4)
    n_sweeps_single = 5_000

    if gate_probe_rate >= 0.9:
        # Escalate: try 15-clue with 2k sweeps.
        print(
            "  Gate not met at 17-clue/5k sweeps (SA_single too strong). "
            "Escalating to 15-clue / 2k sweeps...",
            flush=True,
        )
        final_tier_clues["ultra_hard"] = 15
        n_sweeps_single = 2_000

        # Rebuild ultra_hard puzzles with 15 clues.
        escalated_puzzles = make_puzzle_set_v4(
            SEED,
            tier_clues={"ultra_hard": 15},
            puzzles_per_tier=PUZZLES_PER_TIER,
        )
        # Replace ultra_hard puzzles in the main set.
        non_uh = [p for p in puzzles if p.difficulty != "ultra_hard"]
        puzzles = non_uh + escalated_puzzles
        ultra_hard_puzzles = escalated_puzzles
        probe_puzzles_escalated = ultra_hard_puzzles[:HARDNESS_GATE_PROBE_N]

        # Also update the SA single config for the new n_sweeps.
        escalated_sa_cfg = dict(SA_SINGLE_CFG)
        escalated_sa_cfg["ultra_hard"] = {
            "n_sweeps": n_sweeps_single,
            "n_moves": 100,
            "T_init": 0.5,
            "T_final": 0.005,
        }

        gate_probe_rate_2 = _probe_hardness_gate(
            probe_puzzles_escalated,
            seed=SEED,
            n_sweeps=n_sweeps_single,
            n_moves=100,
            T_init=0.5,
            T_final=0.005,
        )
        print(
            f"  Hardness gate probe 2 (15-clue, 2k sweeps): SA_single solve_rate={gate_probe_rate_2:.3f}",
            flush=True,
        )
        hardness_sweep.append({
            "n_clues": 15,
            "n_sweeps": n_sweeps_single,
            "probe_solve_rate": gate_probe_rate_2,
        })

        if gate_probe_rate_2 >= 0.9:
            # Cannot construct a discriminating tier — block.
            print(
                "  Both escalation levels showed SA_single >= 0.9. "
                "Cannot construct discriminating Sudoku tier. Blocking.",
                flush=True,
            )
            # Recompute checksum with updated config.
            config["final_tier_clues"] = final_tier_clues
            checksum = _reproducibility_checksum(puzzles, SEED, config)

            blocked_artifact: dict = {
                "schema": "carnot.kona_p01_gate.v4",
                "experiment": 3529,
                "inference_substrate": "ising_energy_optimization_cpu",
                "random_seed": SEED,
                "reproducibility_checksum": checksum,
                "n_puzzles": len(puzzles),
                "encoding_validity_E0_reasserted": encoding.as_dict(),
                "hardness_sweep": hardness_sweep,
                "discrete_sa_single_solve_rate": gate_probe_rate_2,
                "discrete_sa_single_solve_rate_hard_tier": gate_probe_rate_2,
                "solve_rate": None,
                "solve_rate_by_difficulty": None,
                "solve_rate_by_optimizer_variant": None,
                "energy_power_gradient_present": False,
                "exact_cp_solve_rate": None,
                "exact_baseline_solve_rate": None,
                "pt_swap_acceptance_rate": 0.0,
                "ar_greedy_solve_rate": None,
                "ar_literature_baselines_note": (
                    "Sudoku-Bench arXiv:2505.16135: SOTA LLMs solve <15% unaided. "
                    "Kona EBM (Logical Intelligence): 96.2% vs frontier LLMs ~2%. "
                    "BDH (Pathway): 97.4% on Sudoku Extreme vs leading LLMs ~0%."
                ),
                "honest_verdict": "complete: blocked_cannot_construct_discriminating_sudoku_tier",
                "duration_s": _elapsed(),
            }
            os.makedirs("results", exist_ok=True)
            with open(OUT_PATH, "w") as f:
                json.dump(blocked_artifact, f, indent=2)
            print(
                f"BLOCKED: cannot construct discriminating tier. Artifact: {OUT_PATH}",
                flush=True,
            )
            return

        # Gate passed at 15-clue / 2k sweeps; update SA single config.
        SA_SINGLE_CFG_FINAL = escalated_sa_cfg
        print(
            "  Hardness gate passed at 15-clue / 2k sweeps. Proceeding with full run.",
            flush=True,
        )
    else:
        SA_SINGLE_CFG_FINAL = SA_SINGLE_CFG
        print(
            f"  Hardness gate passed at 17-clue / 5k sweeps "
            f"(SA_single={gate_probe_rate:.3f} < 0.9). "
            f"Proceeding with full run.",
            flush=True,
        )

    # Update config with final tier clues after gate.
    config["final_tier_clues"] = final_tier_clues
    # Recompute checksum with finalized config and puzzles.
    checksum = _reproducibility_checksum(puzzles, SEED, config)

    # ------------------------------------------------------------------
    # Optimizer ladder.
    # ------------------------------------------------------------------
    by_variant: dict = {}

    print("\nStep 1: discrete_sa_single (budget-matched 1x unit)...", flush=True)
    by_variant.update(_run_discrete_sa_single(puzzles, SEED, cfg=SA_SINGLE_CFG_FINAL))

    print(f"\nStep 2: discrete_sa_restarts{SA_N_RESTARTS} (20x budget)...", flush=True)
    by_variant.update(_run_discrete_sa_restarts(puzzles, SEED, cfg=SA_SINGLE_CFG_FINAL))

    print("\nStep 3: parallel_tempering_tuned (8 chains, 4x per-chain budget)...", flush=True)
    pt_result = _run_parallel_tempering_tuned(puzzles, SEED)
    mean_swap_acc = pt_result.pop("_pt_mean_swap_acceptance", 0.0)
    by_variant.update(pt_result)

    print("\nStep 4: exact_cp (constraint propagation upper bound)...", flush=True)
    by_variant.update(_run_exact_cp(puzzles))

    # ------------------------------------------------------------------
    # Step 5: AR greedy baseline + literature comparator.
    # ------------------------------------------------------------------
    print("\nStep 5: AR greedy baseline...", flush=True)
    ar_greedy_rate = _ar_greedy_baseline(puzzles, seed=SEED + 9999)
    print(f"  ar_greedy_solve_rate={ar_greedy_rate:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Aggregate results.
    # ------------------------------------------------------------------
    headline_key = f"discrete_sa_restarts{SA_N_RESTARTS}"
    headline_records = by_variant.get(headline_key, [])
    pt_records = by_variant.get("parallel_tempering_tuned", [])
    exact_records = by_variant.get("exact_cp", [])
    sa_single_records = by_variant.get("discrete_sa_single", [])

    solve_rate_overall = _solve_rate(headline_records)
    solve_rate_by_diff = _solve_rate_by_difficulty(headline_records)
    solve_rate_by_variant = {
        v: _solve_rate(recs)
        for v, recs in by_variant.items()
        if isinstance(recs, list)
    }
    exact_rate = _solve_rate(exact_records)
    tuned_pt_rate = _solve_rate(pt_records)
    sa_single_rate = _solve_rate(sa_single_records)

    # Per-tier SA single rate on ultra_hard specifically.
    uh_sa_single = [r for r in sa_single_records if r["difficulty"] == "ultra_hard"]
    sa_single_rate_ultra_hard = _solve_rate(uh_sa_single) if uh_sa_single else 0.0

    # PT swap acceptance rate.
    pt_swap_acceptance_rate = mean_swap_acc
    pt_per_puzzle_swap = [r.get("swap_acceptance_rate", 0.0) for r in pt_records]

    # Check for optimizer power gradient: PT or restarts beat SA_single on ultra_hard.
    # WHY this metric: CEILING_SATURATION occurs when SA_single is as good as PT/restarts.
    # If SA_single < PT_tuned or SA_single < restarts on ultra_hard, the gradient is visible.
    uh_pt_records = [r for r in pt_records if r["difficulty"] == "ultra_hard"]
    uh_restarts_records = [r for r in headline_records if r["difficulty"] == "ultra_hard"]
    uh_pt_rate = _solve_rate(uh_pt_records)
    uh_restarts_rate = _solve_rate(uh_restarts_records)

    energy_power_gradient_present = bool(
        (uh_pt_rate > sa_single_rate_ultra_hard + 0.05)
        or (uh_restarts_rate > sa_single_rate_ultra_hard + 0.05)
    )
    print(
        f"  Energy power gradient: uh_sa_single={sa_single_rate_ultra_hard:.3f} "
        f"uh_pt={uh_pt_rate:.3f} uh_restarts={uh_restarts_rate:.3f} "
        f"gradient_present={energy_power_gradient_present}",
        flush=True,
    )

    ar_literature_note = (
        "Sudoku-Bench arXiv:2505.16135: SOTA LLMs solve <15% unaided on hard boards. "
        "Kona EBM (Logical Intelligence): 96.2% vs frontier LLMs ~2% on hard Sudoku. "
        "BDH (Pathway arXiv:2024): 97.4% on Sudoku Extreme vs leading LLMs ~0%. "
        "QUBO benchmark arXiv:2506.04596: SA/PT/SB/Gurobi under uniform compute budget. "
        "These numbers establish the AR performance envelope; greedy AR=0 is conservative "
        "(real LLMs also fail near 0% on 17-clue / 20-clue boards)."
    )

    # Verdict logic.
    if energy_power_gradient_present and solve_rate_overall > ar_greedy_rate:
        sr = f"{solve_rate_overall:.2f}".replace(".", "_")
        sa_r = f"{sa_single_rate:.2f}".replace(".", "_")
        verdict = (
            f"complete: p01_sudoku_energy_power_visible_on_discriminating_tier_"
            f"solve_rate_{sr}_vs_single_sa_{sa_r}"
        )
    else:
        verdict = (
            "complete: p01_sudoku_advantage_was_ceiling_artifact_"
            "no_optimizer_power_gradient_on_hard_tier"
        )

    duration = _elapsed()

    artifact = {
        "schema": "carnot.kona_p01_gate.v4",
        "experiment": 3529,
        "inference_substrate": "ising_energy_optimization_cpu",
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "n_puzzles": len(puzzles),
        "optimizer_config": config,
        "encoding_validity_E0_reasserted": encoding.as_dict(),
        "hardness_sweep": hardness_sweep,
        "discrete_sa_single_solve_rate": sa_single_rate,
        "discrete_sa_single_solve_rate_hard_tier": sa_single_rate_ultra_hard,
        "solve_rate": solve_rate_overall,
        "solve_rate_by_difficulty": solve_rate_by_diff,
        "solve_rate_by_optimizer_variant": solve_rate_by_variant,
        "energy_power_gradient_present": energy_power_gradient_present,
        "exact_cp_solve_rate": exact_rate,
        "exact_baseline_solve_rate": exact_rate,  # alias for conductor schema reader
        "parallel_tempering_solve_rate": tuned_pt_rate,
        "pt_swap_acceptance_rate": pt_swap_acceptance_rate,
        "pt_per_puzzle_swap_acceptance": pt_per_puzzle_swap,
        "ar_greedy_solve_rate": ar_greedy_rate,
        "ar_literature_baselines_note": ar_literature_note,
        "per_puzzle_results": {v: recs for v, recs in by_variant.items()
                               if isinstance(recs, list)},
        "honest_verdict": verdict,
        "duration_s": duration,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written: {OUT_PATH}", flush=True)
    print(f"  encoding_validity_E0_reasserted.is_valid : {encoding.is_valid}")
    print(f"  n_puzzles                                : {len(puzzles)}")
    print(f"  solve_rate ({headline_key})              : {solve_rate_overall:.3f}")
    print(f"  solve_rate_by_difficulty                 : {solve_rate_by_diff}")
    print(f"  discrete_sa_single_solve_rate            : {sa_single_rate:.3f}")
    print(f"  discrete_sa_single_rate (ultra_hard)     : {sa_single_rate_ultra_hard:.3f}")
    print(f"  energy_power_gradient_present            : {energy_power_gradient_present}")
    print(f"  exact_cp_solve_rate                      : {exact_rate:.3f}")
    print(f"  parallel_tempering_solve_rate (tuned)    : {tuned_pt_rate:.3f}")
    print(f"  pt_swap_acceptance_rate                  : {pt_swap_acceptance_rate:.3f}")
    print(f"  ar_greedy_solve_rate                     : {ar_greedy_rate:.4f}")
    print(f"  duration_s                               : {duration:.1f}")
    print(f"  honest_verdict                           : {verdict}")


if __name__ == "__main__":
    main()
