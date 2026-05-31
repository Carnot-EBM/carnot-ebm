"""Experiment 3518: P0.1 generalization — energy global inference vs AR on a SECOND CSP.

CONTEXT:
    Exp 3505 (.322) produced P0.1's FIRST CLEAN POSITIVE: energy-guided global
    inference (discrete SA, solve_rate=1.0) beats AR (greedy, solve_rate=0.0)
    on Sudoku — a CSP encoding that maps to Ising via a QUBO penalty (row +
    column + box uniqueness). The Sudoku positive is confirmed by Exp 3517
    (>=40 puzzles, 4 tiers). But the claim is Sudoku-ONLY until we test a
    second CSP family.

    Literature basis for expecting generalization:
    - "Beyond Autoregression" (arXiv:2410.14157): non-AR global-inference beats
      greedy AR on Sudoku (100% vs 20.7%), Boolean SAT, and Countdown (91.5%
      vs 45.8%) — THREE CSP families, all with the same contrast pattern.
    - DIFUSCO (arXiv:2302.08224): diffusion (global-inference) beats AR on
      graph coloring and TSP.

    CHOSEN CSP: Graph k-coloring. Reasons:
    1. Standard QUBO/Ising encoding (Lucas 2014, Frontiers in Physics 2(5) §5):
       E = sum_{(u,v)∈E} x_{u,c}*x_{v,c} (conflict penalty; E=0 iff valid).
    2. Well-studied AR baseline: greedy sequential coloring (assign lowest
       available color, no backtracking) — the natural AR analog.
    3. DIFUSCO (arXiv:2302.08224) already showed non-AR beats AR on this CSP,
       providing independent empirical precedent.

PRECONDITION (Step 0a):
    E == 0 on the known-valid coloring of the first generated graph.
    If not → complete: blocked_second_csp_encoding_invalid.

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3518_p01_second_csp_energy_vs_ar_generalization_v1.py

Spec: REQ-KONA-3518, SCENARIO-KONA-3518
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from carnot.phase3.graph_coloring_ising import (
    check_encoding_validity,
    generate_colorable_graph,
    gc_ar_greedy_solve,
    gc_exact_solve,
    gc_parallel_tempering_solve_instrumented,
    gc_sa_solve_once,
    gc_sa_solve_restarts,
    is_valid_coloring,
)

# ---------------------------------------------------------------------------
# Experiment identity — seed is content-derived, NOT the experiment number.
# WHY content-derived: the seed "exp3518" from the TAUTOLOGY fix (exp3505
# used random_seed=3505 == experiment ID, flagged by adversarial_verify).
# ---------------------------------------------------------------------------
_SEED_BYTES = b"exp3518_graph_coloring_second_csp_p01_generalization_v1"
SEED = int(hashlib.sha256(_SEED_BYTES).hexdigest(), 16) % (2**31)

OUT_PATH = (
    "results/experiment_3518_p01_second_csp_energy_vs_ar_generalization_v1.json"
)

# Hard wall: exit clean at ~28 min.
BUDGET_SECS = 28 * 60
_RUN_START: float = 0.0


def _elapsed() -> float:
    return time.time() - _RUN_START


def _over_budget() -> bool:
    return _elapsed() > BUDGET_SECS


# ---------------------------------------------------------------------------
# Difficulty tiers for graph k-coloring.
#
# WHY these parameters:
# - easy: small sparse graphs (n=10, p=0.3, k=3) — SA solves trivially,
#   greedy often succeeds too (low conflict pressure).
# - medium: larger, denser (n=15, p=0.5, k=3) — greedy starts to fail on
#   unlucky vertex orderings; SA nearly always succeeds.
# - hard: n=20, p=0.65, k=3 — greedy frequently fails; SA solves well.
# - extreme: n=25, p=0.75, k=4 — even with 4 colors, high density creates
#   strong conflict pressure; greedy fails often; SA demonstrates the gap.
#
# 5 instances per tier × 4 tiers = 20 total (meets >=20 requirement).
# ---------------------------------------------------------------------------
TIER_CONFIG: dict[str, dict] = {
    "easy":    {"n_vertices": 10, "n_colors": 3, "edge_prob": 0.30, "n_instances": 5},
    "medium":  {"n_vertices": 15, "n_colors": 3, "edge_prob": 0.50, "n_instances": 5},
    "hard":    {"n_vertices": 20, "n_colors": 3, "edge_prob": 0.65, "n_instances": 5},
    "extreme": {"n_vertices": 25, "n_colors": 4, "edge_prob": 0.75, "n_instances": 5},
}

# Optimizer settings per difficulty tier.
SA_CONFIG = {
    "easy":    {"n_sweeps": 2_000,  "n_moves": 30, "T_init": 1.0, "T_final": 0.01},
    "medium":  {"n_sweeps": 4_000,  "n_moves": 40, "T_init": 1.0, "T_final": 0.01},
    "hard":    {"n_sweeps": 6_000,  "n_moves": 50, "T_init": 1.0, "T_final": 0.005},
    "extreme": {"n_sweeps": 8_000,  "n_moves": 50, "T_init": 1.5, "T_final": 0.005},
}
SA_N_RESTARTS = 15

# PT config (tuned per Exp 3517 lessons: n_exchange_interval=50 for adequate mixing).
PT_CONFIG = {
    "easy":    {"n_sweeps": 2_000,  "n_moves": 30, "n_chains": 6,
                "T_min": 0.1, "T_max": 2.0, "n_exchange_interval": 50},
    "medium":  {"n_sweeps": 4_000,  "n_moves": 40, "n_chains": 6,
                "T_min": 0.1, "T_max": 2.0, "n_exchange_interval": 50},
    "hard":    {"n_sweeps": 6_000,  "n_moves": 50, "n_chains": 6,
                "T_min": 0.1, "T_max": 2.0, "n_exchange_interval": 50},
    "extreme": {"n_sweeps": 8_000,  "n_moves": 50, "n_chains": 6,
                "T_min": 0.1, "T_max": 3.0, "n_exchange_interval": 50},
}


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass(frozen=True)
class GraphColoringInstance:
    """One graph coloring instance with its known valid solution."""
    instance_id: str
    difficulty: str
    n_vertices: int
    n_colors: int
    edges: list
    known_coloring: list
    n_edges: int


# ---------------------------------------------------------------------------
# Instance set generation
# ---------------------------------------------------------------------------

def make_instance_set(seed: int = SEED) -> list[GraphColoringInstance]:
    """Build a deterministic 20-instance set spanning 4 difficulty tiers.

    WHY 5 per tier: 5 × 4 = 20 meets the >=20 sample-size requirement while
    keeping total runtime well within the 28-minute wall-clock budget.
    """
    instances: list[GraphColoringInstance] = []
    for tier, cfg in TIER_CONFIG.items():
        for i in range(cfg["n_instances"]):
            inst_seed = seed + hash(tier) % 10_000 + i * 997
            edges, known_coloring = generate_colorable_graph(
                n_vertices=cfg["n_vertices"],
                n_colors=cfg["n_colors"],
                edge_probability=cfg["edge_prob"],
                seed=inst_seed,
            )
            instances.append(GraphColoringInstance(
                instance_id=f"{tier}_{i}",
                difficulty=tier,
                n_vertices=cfg["n_vertices"],
                n_colors=cfg["n_colors"],
                edges=edges,
                known_coloring=known_coloring,
                n_edges=len(edges),
            ))
    return instances


# ---------------------------------------------------------------------------
# Aggregate statistics helpers
# ---------------------------------------------------------------------------

def _solve_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return float(sum(1 for r in records if r["solved"]) / len(records))


def _solve_rate_by_difficulty(records: list[dict]) -> dict[str, float]:
    by_tier: dict[str, list[dict]] = {}
    for r in records:
        by_tier.setdefault(r["difficulty"], []).append(r)
    return {tier: _solve_rate(recs) for tier, recs in by_tier.items()}


def _reproducibility_checksum(
    instances: list[GraphColoringInstance],
    seed: int,
    config: dict,
) -> str:
    payload = {
        "experiment": 3518,
        "seed": seed,
        "config": config,
        "instances": [
            {"id": inst.instance_id, "n_vertices": inst.n_vertices,
             "n_edges": inst.n_edges, "n_colors": inst.n_colors}
            for inst in instances
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Optimizer runners
# ---------------------------------------------------------------------------

def _run_vanilla_descent(instances: list[GraphColoringInstance], seed: int) -> dict:
    """Greedy descent (T=0 SA) — contrast baseline expected to get stuck."""
    print("  Variant: vanilla_descent (T=0, single restart — contrast baseline)", flush=True)
    records = []
    for inst in instances:
        if _over_budget():
            break
        cfg = SA_CONFIG[inst.difficulty]
        _, solved, n_conflicts = gc_sa_solve_once(
            inst.edges, inst.n_vertices, inst.n_colors,
            n_sweeps=cfg["n_sweeps"], n_moves_per_sweep=cfg["n_moves"],
            T_init=0.0, T_final=0.0,  # T=0: pure greedy descent
            seed=seed + hash(inst.instance_id) % 10_000,
        )
        records.append({
            "instance_id": inst.instance_id, "difficulty": inst.difficulty,
            "solved": solved, "n_conflicts": n_conflicts,
        })
        print(f"    {inst.instance_id}: solved={solved} n_conflicts={n_conflicts}", flush=True)
    return {"vanilla_descent": records}


def _run_sa_single(instances: list[GraphColoringInstance], seed: int) -> dict:
    """Single SA trajectory per instance."""
    print("  Variant: sa_single (1 restart)", flush=True)
    records = []
    for inst in instances:
        if _over_budget():
            break
        cfg = SA_CONFIG[inst.difficulty]
        _, solved, n_conflicts = gc_sa_solve_once(
            inst.edges, inst.n_vertices, inst.n_colors,
            n_sweeps=cfg["n_sweeps"], n_moves_per_sweep=cfg["n_moves"],
            T_init=cfg["T_init"], T_final=cfg["T_final"],
            seed=seed + hash(inst.instance_id) % 10_000,
        )
        records.append({
            "instance_id": inst.instance_id, "difficulty": inst.difficulty,
            "solved": solved, "n_conflicts": n_conflicts,
        })
        print(f"    {inst.instance_id}: solved={solved} n_conflicts={n_conflicts}", flush=True)
    return {"sa_single": records}


def _run_sa_restarts(instances: list[GraphColoringInstance], seed: int) -> dict:
    """SA with multiple restarts — headline energy optimizer."""
    print(f"  Variant: sa_restarts{SA_N_RESTARTS}", flush=True)
    records = []
    for inst in instances:
        if _over_budget():
            break

        def _cb(k: int, n_conflicts: int, solved: bool) -> None:
            if k % 5 == 0 or solved:
                print(
                    f"    {inst.instance_id} restart {k}: "
                    f"n_conflicts={n_conflicts} solved={solved}",
                    flush=True,
                )

        cfg = SA_CONFIG[inst.difficulty]
        t0 = time.time()
        _, solved, n_conflicts = gc_sa_solve_restarts(
            inst.edges, inst.n_vertices, inst.n_colors,
            n_sweeps=cfg["n_sweeps"], n_moves_per_sweep=cfg["n_moves"],
            T_init=cfg["T_init"], T_final=cfg["T_final"],
            n_restarts=SA_N_RESTARTS,
            seed=seed + hash(inst.instance_id) % 10_000,
            progress_callback=_cb,
        )
        elapsed = time.time() - t0
        records.append({
            "instance_id": inst.instance_id, "difficulty": inst.difficulty,
            "solved": solved, "n_conflicts": n_conflicts, "time_s": elapsed,
        })
        print(
            f"    {inst.instance_id}: FINAL solved={solved} "
            f"n_conflicts={n_conflicts} t={elapsed:.1f}s",
            flush=True,
        )
    return {f"sa_restarts{SA_N_RESTARTS}": records}


def _run_parallel_tempering(instances: list[GraphColoringInstance], seed: int) -> dict:
    """PT with instrumented swap-acceptance tracking."""
    print("  Variant: parallel_tempering (6 chains, interval=50)", flush=True)
    records = []
    swap_rates: list[float] = []

    for inst in instances:
        if _over_budget():
            break

        def _cb(sweep: int, n_conflicts: int) -> None:
            print(
                f"    {inst.instance_id} PT sweep {sweep}: "
                f"coldest n_conflicts={n_conflicts}",
                flush=True,
            )

        cfg = PT_CONFIG[inst.difficulty]
        _, solved, n_conflicts, swap_acc = gc_parallel_tempering_solve_instrumented(
            inst.edges, inst.n_vertices, inst.n_colors,
            n_sweeps=cfg["n_sweeps"], n_moves_per_sweep=cfg["n_moves"],
            n_chains=cfg["n_chains"], T_min=cfg["T_min"], T_max=cfg["T_max"],
            n_exchange_interval=cfg["n_exchange_interval"],
            seed=seed + hash(inst.instance_id) % 10_000,
            progress_callback=_cb,
        )
        swap_rates.append(swap_acc)
        records.append({
            "instance_id": inst.instance_id, "difficulty": inst.difficulty,
            "solved": solved, "n_conflicts": n_conflicts,
            "swap_acceptance_rate": swap_acc,
        })
        print(
            f"    {inst.instance_id}: solved={solved} n_conflicts={n_conflicts} "
            f"swap_acc={swap_acc:.3f}",
            flush=True,
        )

    mean_swap = float(np.mean(swap_rates)) if swap_rates else 0.0
    return {"parallel_tempering": records, "_pt_mean_swap_acceptance": mean_swap}


def _run_exact(instances: list[GraphColoringInstance]) -> dict:
    """Exact backtracking solver — optimality reference."""
    print("  Variant: exact_backtracking (MRV degree ordering)", flush=True)
    records = []
    for inst in instances:
        if _over_budget():
            break
        t0 = time.time()
        result = gc_exact_solve(
            inst.edges, inst.n_vertices, inst.n_colors, max_nodes=5_000_000
        )
        elapsed = time.time() - t0
        if result is not None:
            solved = is_valid_coloring(result, inst.edges, inst.n_colors)
        else:
            solved = False
        records.append({
            "instance_id": inst.instance_id, "difficulty": inst.difficulty,
            "solved": solved, "time_s": elapsed,
        })
        print(
            f"    {inst.instance_id}: solved={solved} t={elapsed:.2f}s", flush=True
        )
    return {"exact_backtracking": records}


def _run_ar_greedy(instances: list[GraphColoringInstance], seed: int) -> float:
    """Greedy AR baseline: assign each vertex its lowest available color.

    WHY greedy as AR: sequential lowest-available-color is the canonical
    AR analog for graph coloring — it commits to each vertex in order without
    backtracking, mirroring AR token generation. It fails when it runs out of
    available colors before finishing (needs k+1 colors despite the graph
    being k-colorable).
    """
    rng = np.random.default_rng(seed)
    n_solved = 0
    for inst in instances:
        result = gc_ar_greedy_solve(
            inst.edges, inst.n_vertices, inst.n_colors,
            seed=int(rng.integers(0, 2**31)),
        )
        if result is not None and is_valid_coloring(result, inst.edges, inst.n_colors):
            n_solved += 1
    return n_solved / len(instances) if instances else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _RUN_START
    _RUN_START = time.time()

    print(
        f"Exp 3518: P0.1 second-CSP generalization (graph coloring) — "
        f"start {time.strftime('%H:%M:%S')}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build 20-instance set (4 tiers × 5 each).
    # ------------------------------------------------------------------
    instances = make_instance_set(SEED)
    print(
        f"Instance set: {len(instances)} instances across "
        f"{list(TIER_CONFIG.keys())}",
        flush=True,
    )
    assert len(instances) >= 20, f"Expected >=20 instances, got {len(instances)}"

    config = {
        "csp_family": "graph_coloring",
        "sa_n_restarts": SA_N_RESTARTS,
        "sa_config": SA_CONFIG,
        "pt_config": PT_CONFIG,
        "tier_config": {k: dict(v) for k, v in TIER_CONFIG.items()},
    }
    checksum = _reproducibility_checksum(instances, SEED, config)

    # ------------------------------------------------------------------
    # Step 0a: Encoding validity — GATING.
    # Use the known-valid coloring of the first instance (E must be 0).
    # ------------------------------------------------------------------
    print("\nStep 0a: Encoding validity check (known-valid solution → E must be 0)...", flush=True)
    first = instances[0]
    enc = check_encoding_validity(
        first.known_coloring, first.edges, first.n_vertices, first.n_colors
    )
    print(
        f"  E={enc.total_energy:.6f} is_valid={enc.is_valid} "
        f"n_conflicts={enc.n_conflicts} n_uncolored={enc.n_uncolored}",
        flush=True,
    )

    if not enc.is_valid:
        artifact: dict = {
            "schema": "carnot.kona_p01_gate.graph_coloring.v1",
            "experiment": 3518,
            "csp_family": "graph_coloring",
            "inference_substrate": "ising_energy_optimization_cpu",
            "random_seed": SEED,
            "reproducibility_checksum": checksum,
            "n_instances": len(instances),
            "encoding_validity_E0": enc.as_dict(),
            "honest_verdict": "complete: blocked_second_csp_encoding_invalid",
            "duration_s": _elapsed(),
        }
        os.makedirs("results", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(artifact, f, indent=2)
        print(f"BLOCKED: encoding invalid. Artifact: {OUT_PATH}", flush=True)
        return

    print("  Encoding validity: PASS (E=0 on known-valid coloring).", flush=True)

    # Also verify the exact solver confirms instances are solvable
    # (quick check on first instance, non-blocking).
    exact_first = gc_exact_solve(first.edges, first.n_vertices, first.n_colors)
    print(
        f"  Exact solver on first instance: "
        f"{'SOLVED' if exact_first is not None else 'FAILED/TIMEOUT'}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 1: Optimizer ladder.
    # ------------------------------------------------------------------
    by_variant: dict = {}

    print("\nStep 1: vanilla_descent (T=0 contrast baseline)...", flush=True)
    by_variant.update(_run_vanilla_descent(instances, SEED))

    print("\nStep 2: sa_single...", flush=True)
    by_variant.update(_run_sa_single(instances, SEED))

    print(f"\nStep 3: sa_restarts{SA_N_RESTARTS} (headline energy optimizer)...", flush=True)
    by_variant.update(_run_sa_restarts(instances, SEED))

    print("\nStep 4: parallel_tempering (instrumented, 6 chains)...", flush=True)
    pt_result = _run_parallel_tempering(instances, SEED)
    mean_swap_acc = pt_result.pop("_pt_mean_swap_acceptance", 0.0)
    by_variant.update(pt_result)

    print("\nStep 5: exact_backtracking (optimality reference)...", flush=True)
    by_variant.update(_run_exact(instances))

    # ------------------------------------------------------------------
    # Step 6: AR greedy baseline.
    # ------------------------------------------------------------------
    print("\nStep 6: AR greedy baseline...", flush=True)
    ar_solve_rate = _run_ar_greedy(instances, seed=SEED + 9999)
    print(f"  ar_greedy_solve_rate={ar_solve_rate:.4f}", flush=True)

    # ------------------------------------------------------------------
    # Step 7: Aggregate.
    # ------------------------------------------------------------------
    headline_key = f"sa_restarts{SA_N_RESTARTS}"
    headline_records = by_variant.get(headline_key, [])
    pt_records = by_variant.get("parallel_tempering", [])
    exact_records = by_variant.get("exact_backtracking", [])

    solve_rate = _solve_rate(headline_records)
    solve_rate_by_diff = _solve_rate_by_difficulty(headline_records)
    solve_rate_by_variant = {v: _solve_rate(recs) for v, recs in by_variant.items()}
    exact_rate = _solve_rate(exact_records)
    pt_rate = _solve_rate(pt_records)

    pt_swap_acc = mean_swap_acc
    pt_per_puzzle_swap = [r.get("swap_acceptance_rate", 0.0) for r in pt_records]

    tts = [
        r["time_s"] for r in headline_records
        if r.get("solved") and "time_s" in r
    ]

    generalizes = solve_rate > ar_solve_rate

    ar_literature_note = (
        "Graph k-coloring AR baseline: greedy sequential coloring (lowest available "
        "color, random vertex order) — canonical AR analog (no backtracking). "
        "Literature: DIFUSCO (arXiv:2302.08224) showed non-AR diffusion beats greedy "
        "AR on graph coloring. 'Beyond Autoregression' (arXiv:2410.14157): "
        "global-inference beats AR on Sudoku (100% vs 20.7%), SAT, Countdown (91.5% "
        "vs 45.8%) — same regime tested here. Greedy AR uses at most Δ+1 colors "
        "(Brooks' theorem), so it fails on k-colorable graphs with Δ ≥ k-1 in "
        "adversarial orderings."
    )

    # Verdict
    if not enc.is_valid:
        verdict = "complete: blocked_second_csp_encoding_invalid"
    elif generalizes:
        verdict = (
            f"complete: p01_energy_vs_ar_generalizes_to_graph_coloring_"
            f"solve_rate_{solve_rate:.2f}_vs_ar_{ar_solve_rate:.2f}"
        ).replace(".", "_")
    else:
        verdict = (
            f"complete: p01_energy_vs_ar_positive_is_sudoku_specific_"
            f"does_not_generalize_to_graph_coloring"
        )

    duration = _elapsed()
    artifact = {
        "schema": "carnot.kona_p01_gate.graph_coloring.v1",
        "experiment": 3518,
        "csp_family": "graph_coloring",
        "inference_substrate": "ising_energy_optimization_cpu",
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "n_instances": len(instances),
        "optimizer_config": config,
        "encoding_validity_E0": enc.as_dict(),
        "solve_rate": solve_rate,
        "solve_rate_by_difficulty": solve_rate_by_diff,
        "solve_rate_by_optimizer_variant": solve_rate_by_variant,
        "exact_baseline_solve_rate": exact_rate,
        "parallel_tempering_solve_rate": pt_rate,
        "pt_swap_acceptance_rate": pt_swap_acc,
        "pt_per_instance_swap_acceptance": pt_per_puzzle_swap,
        "ar_baseline_solve_rate": ar_solve_rate,
        "ar_literature_baselines_note": ar_literature_note,
        "llm_ar_inhouse_solve_rate": None,  # CUDA-gated; CUDA unavailable in this run
        "time_to_solution_solved_only": tts,
        "generalizes_beyond_sudoku": generalizes,
        "per_instance_results": {v: recs for v, recs in by_variant.items()},
        "honest_verdict": verdict,
        "duration_s": duration,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written: {OUT_PATH}", flush=True)
    print(f"  csp_family                              : graph_coloring")
    print(f"  encoding_validity_E0.is_valid           : {enc.is_valid}")
    print(f"  n_instances                             : {len(instances)}")
    print(f"  solve_rate ({headline_key})             : {solve_rate:.3f}")
    print(f"  solve_rate_by_difficulty                : {solve_rate_by_diff}")
    print(f"  exact_baseline_solve_rate               : {exact_rate:.3f}")
    print(f"  parallel_tempering_solve_rate           : {pt_rate:.3f}")
    print(f"  pt_swap_acceptance_rate                 : {pt_swap_acc:.3f}")
    print(f"  ar_baseline_solve_rate                  : {ar_solve_rate:.4f}")
    print(f"  generalizes_beyond_sudoku               : {generalizes}")
    print(f"  duration_s                              : {duration:.1f}")
    print(f"  honest_verdict                          : {verdict}")


if __name__ == "__main__":
    main()
