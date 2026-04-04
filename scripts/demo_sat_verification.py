#!/usr/bin/env python3
"""Demo: LLM-EBM anti-hallucination pipeline on SAT and graph coloring.

This script demonstrates the core Carnot thesis: an LLM proposes a solution
to a constraint satisfaction problem, the EBM scores it, repairs violations
via gradient descent, and issues a verification certificate.

No LLM API required — uses random assignments to simulate LLM output.
For real LLM integration, see the parse_llm_sat_assignment() and
parse_llm_coloring() functions in carnot.inference.verify_and_repair.

Usage:
    python scripts/demo_sat_verification.py
"""

from __future__ import annotations

import time

import jax.numpy as jnp

from carnot.inference.benchmark import (
    generate_random_graph,
    generate_random_sat,
    run_coloring_benchmark,
    run_sat_benchmark,
)
from carnot.inference.verify_and_repair import verify_and_repair
from carnot.verify.graph_coloring import build_coloring_energy, coloring_to_array
from carnot.verify.sat import assignment_to_array, build_sat_energy


def demo_sat() -> None:
    """Demonstrate SAT verify-and-repair."""
    print("=" * 70)
    print("DEMO 1: SAT Constraint Satisfaction")
    print("=" * 70)

    # Create a 3-SAT instance: 10 variables, 25 clauses
    n_vars, n_clauses = 10, 25
    clauses = generate_random_sat(n_vars, n_clauses, clause_size=3, seed=42)
    energy = build_sat_energy(clauses, n_vars)

    print(f"\nSAT instance: {n_vars} variables, {n_clauses} clauses")
    print(f"Total constraints: {energy.num_constraints} ({n_clauses} clauses + 1 binary penalty)")

    # Simulate an LLM attempt (random assignment)
    import random

    rng = random.Random(123)
    llm_assignment = [rng.random() < 0.5 for _ in range(n_vars)]
    x = assignment_to_array(llm_assignment)

    print(f"\nSimulated LLM assignment: {['T' if v else 'F' for v in llm_assignment]}")

    # Verify initial assignment
    initial = energy.verify(x)
    failing = initial.verdict.failing
    print(f"\n--- Initial Verification ---")
    print(f"Total energy: {initial.total_energy:.4f}")
    print(f"Verified: {initial.verdict.verified}")
    print(f"Failing constraints: {len(failing)}/{energy.num_constraints}")

    # Repair via gradient descent
    print(f"\n--- Gradient Repair (200 steps, step_size=0.1) ---")

    def round_fn(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(arr >= 0.5, 1.0, 0.0)

    start = time.time()
    result = verify_and_repair(
        x, energy, step_size=0.1, max_repair_steps=200, round_fn=round_fn
    )
    elapsed = time.time() - start

    print(f"Repair steps: {result.n_repair_steps}")
    print(f"Time: {elapsed:.2f}s")

    if result.repair_trajectory:
        print(f"Energy trajectory: {result.repair_trajectory[0]:.4f} → {result.repair_trajectory[-1]:.4f}")

    # Final verification
    assert result.rounded_verification is not None
    r_failing = result.rounded_verification.verdict.failing
    print(f"\n--- After Repair + Rounding ---")
    print(f"Total energy: {result.rounded_verification.total_energy:.4f}")
    print(f"Verified: {result.rounded_verification.verdict.verified}")
    print(f"Failing constraints: {len(r_failing)}/{energy.num_constraints}")

    if initial.total_energy > 0:
        reduction = (1 - result.rounded_verification.total_energy / initial.total_energy) * 100
        print(f"Energy reduction: {reduction:.1f}%")

    # Show repaired assignment
    repaired_bool = [float(v) >= 0.5 for v in result.rounded_assignment]
    print(f"Repaired assignment: {['T' if v else 'F' for v in repaired_bool]}")


def demo_graph_coloring() -> None:
    """Demonstrate graph coloring verify-and-repair."""
    print("\n" + "=" * 70)
    print("DEMO 2: Graph Coloring")
    print("=" * 70)

    # Create a small graph: 6 nodes, 3 colors
    n_nodes, n_colors = 6, 3
    edges = generate_random_graph(n_nodes, edge_probability=0.4, seed=42)
    energy = build_coloring_energy(edges, n_nodes, n_colors)

    print(f"\nGraph: {n_nodes} nodes, {len(edges)} edges, {n_colors} colors")
    print(f"Edges: {edges}")

    # Simulate a bad LLM coloring (all same color)
    bad_coloring = [0] * n_nodes
    x = coloring_to_array(bad_coloring)

    print(f"Simulated LLM coloring: {bad_coloring} (all same color!)")

    initial = energy.verify(x)
    print(f"\n--- Initial Verification ---")
    print(f"Total energy: {initial.total_energy:.4f}")
    print(f"Verified: {initial.verdict.verified}")
    print(f"Failing: {len(initial.verdict.failing)}/{energy.num_constraints}")

    def round_fn(arr: jnp.ndarray) -> jnp.ndarray:
        return jnp.round(jnp.clip(arr, 0.0, float(n_colors - 1)))

    result = verify_and_repair(
        x, energy, step_size=0.1, max_repair_steps=200, round_fn=round_fn
    )

    assert result.rounded_verification is not None
    print(f"\n--- After Repair + Rounding ---")
    print(f"Total energy: {result.rounded_verification.total_energy:.4f}")
    print(f"Verified: {result.rounded_verification.verdict.verified}")

    repaired_colors = [int(round(float(v))) for v in result.rounded_assignment]
    print(f"Repaired coloring: {repaired_colors}")


def demo_benchmark() -> None:
    """Run a small benchmark."""
    print("\n" + "=" * 70)
    print("DEMO 3: Benchmark (10 instances each)")
    print("=" * 70)

    print("\n--- SAT Benchmark (10 vars, 30 clauses) ---")
    sat_summary = run_sat_benchmark(
        n_instances=10, n_vars=10, n_clauses=30, max_repair_steps=50
    )
    print(f"Avg initial violations: {sat_summary.avg_initial_violations:.1f}")
    print(f"Avg repaired violations: {sat_summary.avg_repaired_violations:.1f}")
    print(f"Repair success rate: {sat_summary.repair_success_rate:.0%}")
    print(f"Avg energy reduction: {sat_summary.avg_energy_reduction_pct:.1f}%")

    print("\n--- Graph Coloring Benchmark (8 nodes, 3 colors) ---")
    color_summary = run_coloring_benchmark(
        n_instances=10, n_nodes=8, n_colors=3, max_repair_steps=50
    )
    print(f"Avg initial violations: {color_summary.avg_initial_violations:.1f}")
    print(f"Avg repaired violations: {color_summary.avg_repaired_violations:.1f}")
    print(f"Repair success rate: {color_summary.repair_success_rate:.0%}")
    print(f"Avg energy reduction: {color_summary.avg_energy_reduction_pct:.1f}%")


if __name__ == "__main__":
    print("Carnot LLM-EBM Anti-Hallucination Demo")
    print("LLM proposes → EBM scores → Gradient repairs → Certificate verifies\n")

    demo_sat()
    demo_graph_coloring()
    demo_benchmark()

    print("\n" + "=" * 70)
    print("CONCLUSION: The EBM catches and fixes what the LLM gets wrong.")
    print("This is the foundation for hallucination-proof AI reasoning.")
    print("=" * 70)
