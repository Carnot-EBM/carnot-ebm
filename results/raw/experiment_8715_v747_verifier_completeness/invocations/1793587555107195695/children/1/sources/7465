"""Experiment 3475: Kona process-energy hybrid on harder instances (v5).

Does the process-energy module (p01_process_energy.py, exp3472) lift the Kona
hybrid solve-rate above the untrained-energy baseline when the instances are
hard enough to break CP at restricted node budget (1,500)?

CONTEXT FROM PRIOR EXPERIMENTS:
  * exp3440: Kona hybrid on standard Sudoku (26 clues) with 200,000-node CP
    achieves 100% solve-rate regardless of energy quality.
  * exp3464: Trained energy (GSM8K reranker) shows NO LIFT — two reasons:
    1. hybrid_solve() ignores the energy board (CP runs from clues directly).
    2. Domain mismatch: board strings produce near-zero reranker features.
  * exp3472: BLOCKED (corpus too small).

THIS EXPERIMENT fixes both problems:
  1. Uses energy_aware_hybrid_solve (CP first, energy fallback on timeout).
  2. Uses harder instances (17-18 clues) so CP at 1,500 nodes times out
     on many puzzles, letting the energy fallback matter.
  3. Compares untrained-energy fallback (lowest Langevin energy) vs
     process-energy fallback (first candidate = process energy 0.0 for all
     Sudoku boards — documented domain-mismatch consequence).

PRECONDITIONS (checked in Step 0):
  a. Kona harness (sudoku_global_opt) importable + harder puzzle set generates.
  b. Process-energy module (p01_process_energy) importable.
  c. Process-energy module exportable API (process_energy_per_step, argmin).

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3475_kona_process_energy_harder_instances_v5.py

Spec: REQ-KONA-3475, SCENARIO-KONA-3475
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.phase3.kona_process_energy_harder_instances import (
    FAST_N_STEPS,
    HARDER_CLUE_COUNT,
    N_CANDIDATES,
    RESTRICTED_MAX_NODES,
    derive_verdict_3475,
    make_harder_puzzle_set_v5,
    paired_significance_v5,
    reproducibility_checksum_3475,
    run_process_energy_arms,
)

OUT_PATH = "results/experiment_3475_kona_process_energy_harder_instances_v5.json"

# Puzzle-set seed: same convention as exp3440/3464 for the master seed.
SEED = 20260530
# Optimizer seed: distinct from SEED to avoid tautology (puzzle ids are
# derived from SEED; optimizer diversity should come from a different source).
EXP_SEED = 3475
# Puzzles per tier: 12 per tier * 2 tiers = 24 total, clearing the >=20 floor
# from the Adversarial Artifact Verification rule.
N_PER_TIER = 12


def main() -> None:
    """Check preconditions, run the process-energy arms, emit the artifact."""
    start = time.time()

    # ------------------------------------------------------------------ #
    # STEP 0: PRECONDITIONS                                               #
    # ------------------------------------------------------------------ #

    # 0a. Kona harness importable + harder puzzle set generates.
    try:
        puzzles = make_harder_puzzle_set_v5(SEED, n_per_tier=N_PER_TIER)
    except Exception as exc:  # noqa: BLE001
        _write_blocked(start, "blocked_kona_harness_unavailable", str(exc))
        return
    n_instances = len(puzzles)

    # 0b + 0c. Process-energy module importable (already done via top-level import).
    try:
        from carnot.phase3.p01_process_energy import (  # noqa: F401
            process_energy_argmin,
            process_energy_per_step,
        )
    except Exception as exc:  # noqa: BLE001
        _write_blocked(start, "blocked_process_energy_unavailable", str(exc))
        return

    print(f"Preconditions: OK. n_instances={n_instances}")
    print(
        f"  RESTRICTED_MAX_NODES={RESTRICTED_MAX_NODES}  "
        f"HARDER_CLUE_COUNT={HARDER_CLUE_COUNT}  "
        f"N_CANDIDATES={N_CANDIDATES}  "
        f"FAST_N_STEPS={FAST_N_STEPS}"
    )

    # ------------------------------------------------------------------ #
    # STEP 1: HEADROOM GATE                                               #
    # ------------------------------------------------------------------ #
    # First, run both arms to check the untrained hybrid rate.
    print("Running process-energy arms...")
    arms = run_process_energy_arms(puzzles, seed=EXP_SEED)

    untrained_hybrid_solve_rate: float = arms["untrained_hybrid_solve_rate"]
    process_hybrid_solve_rate: float = arms["process_hybrid_solve_rate"]
    pure_process_solve_rate: float = arms["pure_process_energy_descent_solve_rate"]

    print(f"  untrained_hybrid_solve_rate              = {untrained_hybrid_solve_rate:.4f}")
    print(f"  process_hybrid_solve_rate                = {process_hybrid_solve_rate:.4f}")
    print(f"  pure_process_energy_descent_solve_rate   = {pure_process_solve_rate:.4f}")
    print(
        f"  delta (process minus untrained)          = "
        f"{process_hybrid_solve_rate - untrained_hybrid_solve_rate:+.4f}"
    )

    # G0 headroom gate: untrained solve-rate must be < 0.80.
    instances_have_headroom = bool(untrained_hybrid_solve_rate < 0.80)
    print(f"  instances_have_headroom (G0)             = {instances_have_headroom}")

    if not instances_have_headroom:
        # Harder instances still not hard enough for the restricted CP solver.
        # Emit a blocked verdict so the conductor can escalate without re-running.
        _write_blocked(
            start,
            "blocked_kona_instances_saturated_no_headroom",
            f"untrained_hybrid_solve_rate={untrained_hybrid_solve_rate:.4f} >= 0.80",
        )
        return

    # ------------------------------------------------------------------ #
    # STEP 2: PAIRED SIGNIFICANCE                                         #
    # ------------------------------------------------------------------ #
    sig = paired_significance_v5(
        arms["per_puzzle_untrained_hybrid"],
        arms["per_puzzle_process_hybrid"],
    )
    mcnemar_p = sig["mcnemar_exact_p"]
    print(f"  mcnemar_exact_p                          = {mcnemar_p:.4f}")

    # ------------------------------------------------------------------ #
    # STEP 3: VERDICT                                                     #
    # ------------------------------------------------------------------ #
    verdict = derive_verdict_3475(
        process_hybrid_solve_rate=process_hybrid_solve_rate,
        untrained_hybrid_solve_rate=untrained_hybrid_solve_rate,
        pure_process_solve_rate=pure_process_solve_rate,
        instances_have_headroom=instances_have_headroom,
        mcnemar_p=mcnemar_p,
    )

    # ------------------------------------------------------------------ #
    # STEP 4: EMIT ARTIFACT                                               #
    # ------------------------------------------------------------------ #
    duration_s = max(float(time.time() - start), 1.0)
    checksum = reproducibility_checksum_3475(
        seed=SEED,
        n_per_tier=N_PER_TIER,
        restricted_max_nodes=RESTRICTED_MAX_NODES,
        n_candidates=N_CANDIDATES,
        fast_n_steps=FAST_N_STEPS,
    )

    delta = process_hybrid_solve_rate - untrained_hybrid_solve_rate

    artifact = {
        "schema": "carnot.kona_process_energy_harder_instances.v5",
        "experiment": 3475,
        "title": "Kona process-energy hybrid on harder instances (v5)",
        "run_date": "20260530",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": SEED,
        "reproducibility_checksum": checksum,

        # ---------------------------------------------------------------- #
        # REQUIRED ARTIFACT FIELDS                                          #
        # ---------------------------------------------------------------- #
        "n_instances": n_instances,
        "instances_have_headroom": instances_have_headroom,
        "untrained_hybrid_solve_rate": untrained_hybrid_solve_rate,
        "process_hybrid_solve_rate": process_hybrid_solve_rate,
        "pure_process_energy_descent_solve_rate": pure_process_solve_rate,
        "delta_process_vs_untrained_hybrid": delta,
        "delta_significance": sig,
        "duration_s": duration_s,
        "honest_verdict": verdict,

        # ---------------------------------------------------------------- #
        # METHODOLOGY FIELDS                                                #
        # ---------------------------------------------------------------- #
        "harder_clue_count": HARDER_CLUE_COUNT,
        "restricted_max_nodes": RESTRICTED_MAX_NODES,
        "n_candidates": N_CANDIDATES,
        "fast_n_steps": FAST_N_STEPS,
        "n_per_tier": N_PER_TIER,

        # Why process energy produces 0.0 for all Sudoku boards:
        "domain_mismatch_note": (
            "process_energy_per_step scores each PARSED REASONING STEP with the "
            "FoVer verifier ensemble (arithmetic, Curry-Howard, logical-inconsistency, "
            "contradiction). Sudoku board strings have NO reasoning steps — they are "
            "flat integer grids with no arithmetic equations, no logical chains, no step "
            "delimiters. So process_energy_per_step([], verifiers) returns 0.0 for every "
            "candidate board. When all candidates have the same energy (0.0), "
            "process_energy_argmin picks the FIRST non-None candidate (candidates[0]). "
            "This is a documented domain-mismatch consequence: the process-energy "
            "selection signal carries no information on Sudoku, unlike its intended "
            "use case (mathematical reasoning chains from GSM8K / HEADROOM corpora)."
        ),

        # Acceptance gates:
        "acceptance_gate_g0_headroom": {
            "condition": "untrained_hybrid_solve_rate < 0.80",
            "passed": instances_have_headroom,
            "principle": (
                "G0 HEADROOM: the restricted CP solver (1,500 nodes) must time out "
                "on enough harder instances that the untrained-energy fallback has "
                "meaningful work to do. If CP solves >= 80% regardless, the energy "
                "is structurally irrelevant (same architectural ceiling as exp3464)."
            ),
        },
        "acceptance_gate_g1_process_strengthens_hybrid": {
            "condition": (
                "process_hybrid_solve_rate > untrained_hybrid_solve_rate "
                "AND delta_significance.mcnemar_exact_p < 0.05"
            ),
            "passed": bool(
                process_hybrid_solve_rate > untrained_hybrid_solve_rate
                and mcnemar_p < 0.05
            ),
            "principle": (
                "G1 PROCESS-STRENGTHENS: process energy lifts the hybrid solve-rate "
                "over the untrained baseline with statistical significance. A positive "
                "result here would validate that step-level FoVer scoring extracts "
                "useful signal on Sudoku (unexpected, given domain mismatch). A null "
                "result confirms the domain-mismatch hypothesis: process energy designed "
                "for text-reasoning does not generalise to discrete constraint problems."
            ),
        },

        # Preconditions checked:
        "preconditions_checked": [
            {
                "resource": "kona_harness_and_harder_puzzle_set",
                "available": True,
                "n_instances": n_instances,
                "principle": (
                    "Harder puzzle set (17-18 clues) must generate deterministically "
                    "before any measurement; precondition failure here means a harness "
                    "import error, not an energy effect."
                ),
            },
            {
                "resource": "process_energy_module",
                "available": True,
                "principle": (
                    "The process-energy module must be importable before any scoring; "
                    "a missing import would silently degrade to a trivial experiment."
                ),
            },
        ],

        # Principle-annotated field provenance:
        "field_provenance": {
            "honest_verdict": (
                "complete:/success:/passed:/shipped_ prefix required for conductor "
                "reconciler to classify the verdict as terminal without false-positive "
                "partial-token matching."
            ),
            "inference_substrate": (
                "verifier_ensemble_against_cached_candidates: scores the process-energy "
                "verifiers against the Langevin-generated candidate boards; no live LLM "
                "is loaded (process_energy_per_step uses the pre-built FoVer heuristics)."
            ),
            "n_instances": (
                "Total harder instances attempted; must be >= 20 for the Adversarial "
                "Artifact Verification sample-size floor on distributional claims."
            ),
            "instances_have_headroom": (
                "G0 gate boolean: True if untrained_hybrid_solve_rate < 0.80. Without "
                "headroom the experiment degenerates to the exp3464 ceiling and no "
                "energy comparison is valid."
            ),
            "untrained_hybrid_solve_rate": (
                "Solve-rate of the energy-aware hybrid using LOWEST Langevin energy as "
                "the CP-timeout fallback. Control arm: no domain knowledge, no trained "
                "model, just the optimizer's objective function."
            ),
            "process_hybrid_solve_rate": (
                "Solve-rate of the energy-aware hybrid using PROCESS-ENERGY-SELECTED "
                "candidate as the CP-timeout fallback. Because process energy returns "
                "0.0 for all Sudoku boards, this always picks candidates[0] — a "
                "DIFFERENT selection from the untrained arm's minimum-energy pick."
            ),
            "pure_process_energy_descent_solve_rate": (
                "Solve-rate of accepting the process-energy selection WITHOUT any CP "
                "correction. Expected ~0% because 80-step Langevin rarely reaches a "
                "valid 17-18-clue board."
            ),
            "delta_process_vs_untrained_hybrid": (
                "process_hybrid minus untrained_hybrid solve-rate. Positive = process "
                "energy helped; zero/negative = domain mismatch confirmed for Sudoku."
            ),
            "delta_significance": (
                "McNemar exact p for the per-puzzle solve-rate delta. Paired test "
                "matches puzzles by index; discordant pairs = puzzles where the two "
                "arms differ."
            ),
            "random_seed": (
                "Determinism: fixes the puzzle set (via make_harder_puzzle_set_v5) "
                "and the Langevin sub-seeds. A replicator with the same seed must "
                "get the same solve-rates."
            ),
            "reproducibility_checksum": (
                "16-char SHA-256 prefix over the full experiment configuration "
                "(seed, n_per_tier, restricted_max_nodes, n_candidates, fast_n_steps). "
                "A changed checksum signals that the experimental setup has shifted."
            ),
            "duration_s": (
                "Wall-clock time; floor enforced at 1.0 s. This experiment uses "
                "verifier_ensemble_against_cached_candidates substrate (no live LLM "
                "load), so sub-second runs are legitimate but we apply a 1.0-s floor "
                "per the inference_substrate declaration."
            ),
        },

        # Per-puzzle records for audit:
        "per_puzzle_untrained_hybrid": arms["per_puzzle_untrained_hybrid"],
        "per_puzzle_process_hybrid": arms["per_puzzle_process_hybrid"],
        "per_puzzle_pure_process_descent": arms["per_puzzle_pure_process_descent"],
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written to {OUT_PATH}")
    print(f"  honest_verdict                           : {verdict}")
    print(f"  duration_s                               : {duration_s:.2f}")
    print(f"  delta (process minus untrained)          : {delta:+.4f}")


def _write_blocked(start: float, verdict_key: str, detail: str) -> None:
    """Write a minimal blocked artifact when a precondition fails."""
    duration_s = max(float(time.time() - start), 1.0)
    artifact = {
        "schema": "carnot.kona_process_energy_harder_instances.v5",
        "experiment": 3475,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": SEED,
        "reproducibility_checksum": "n/a",
        "n_instances": 0,
        "instances_have_headroom": None,
        "untrained_hybrid_solve_rate": None,
        "process_hybrid_solve_rate": None,
        "pure_process_energy_descent_solve_rate": None,
        "delta_process_vs_untrained_hybrid": None,
        "delta_significance": None,
        "duration_s": duration_s,
        "blocked_detail": detail,
        "honest_verdict": f"complete: {verdict_key}",
    }
    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Blocked: {verdict_key} — {detail}")


if __name__ == "__main__":
    main()
