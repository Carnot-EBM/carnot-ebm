#!/usr/bin/env python3
"""Experiment 40: LLM → Ising Bridge — verify LLM output via thermodynamic sampling.

Complete pipeline:
  1. LLM generates a response to a constraint problem (e.g., SAT, scheduling)
  2. Parse the response into a constraint representation
  3. Encode constraints as Ising model
  4. Verify/repair via thrml Block Gibbs sampling
  5. Decode the Ising solution back to human-readable form

This is the practical integration: LLM handles language, Ising handles reasoning,
thrml (→ Extropic TSU) does the sampling.

For this experiment, we use a simple constraint format that an LLM could generate:
  "Assign colors to nodes such that no adjacent nodes share a color"
  → Graph coloring → Ising encoding → thrml sampling → verification

Usage:
    .venv/bin/python scripts/experiment_40_llm_ising_bridge.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def graph_coloring_to_ising(
    n_nodes: int,
    edges: list[tuple[int, int]],
    n_colors: int,
) -> tuple[list, list[tuple], np.ndarray, np.ndarray]:
    """Encode graph coloring as Ising model.

    Each node gets n_colors binary spins (one-hot encoding).
    Constraints:
      1. Each node must have exactly one color (one-hot): penalty if sum != 1
      2. Adjacent nodes must have different colors: penalty if same color

    Returns (nodes, ising_edges, biases, weights) for thrml IsingEBM.
    """
    total_spins = n_nodes * n_colors

    # Bias: encourage each color spin to be active (slight positive bias)
    biases = np.ones(total_spins) * 0.5

    # Build edges and weights
    edge_list = []
    weight_list = []

    # Constraint 1: one-hot per node — penalize having 2+ colors active
    # For each pair of color spins within a node: negative coupling (anti-correlated)
    for node in range(n_nodes):
        for c1 in range(n_colors):
            for c2 in range(c1 + 1, n_colors):
                s1 = node * n_colors + c1
                s2 = node * n_colors + c2
                edge_list.append((s1, s2))
                weight_list.append(-2.0)  # Strong penalty for two colors on same node

    # Constraint 2: adjacent nodes can't share a color
    for n1, n2 in edges:
        for c in range(n_colors):
            s1 = n1 * n_colors + c
            s2 = n2 * n_colors + c
            edge_list.append((s1, s2))
            weight_list.append(-2.0)  # Strong penalty for same color on adjacent nodes

    return total_spins, edge_list, biases, np.array(weight_list)


def decode_coloring(spins: list[bool], n_nodes: int, n_colors: int) -> dict[int, int]:
    """Decode spin assignment back to graph coloring."""
    coloring = {}
    for node in range(n_nodes):
        colors_active = []
        for c in range(n_colors):
            if spins[node * n_colors + c]:
                colors_active.append(c)
        if colors_active:
            coloring[node] = colors_active[0]  # Take first active color
        else:
            coloring[node] = -1  # No color assigned
    return coloring


def check_coloring(coloring: dict[int, int], edges: list[tuple[int, int]]) -> tuple[int, int]:
    """Check how many edges have valid (different) colors."""
    valid = 0
    for n1, n2 in edges:
        c1 = coloring.get(n1, -1)
        c2 = coloring.get(n2, -1)
        if c1 >= 0 and c2 >= 0 and c1 != c2:
            valid += 1
    return valid, len(edges)


def main() -> int:
    import jax.numpy as jnp
    import jax.random as jrandom
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models.ising import IsingEBM, IsingSamplingProgram, hinton_init

    print("=" * 70)
    print("EXPERIMENT 40: LLM → Ising Bridge")
    print("  Graph Coloring via Thermodynamic Sampling")
    print("=" * 70)

    start = time.time()

    # Test problems of increasing difficulty
    test_cases = [
        # (n_nodes, edges, n_colors, name)
        (4, [(0,1), (1,2), (2,3), (3,0)], 2, "4-cycle, 2 colors"),
        (4, [(0,1), (1,2), (2,3), (3,0), (0,2)], 3, "4-node+diagonal, 3 colors"),
        (6, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,3),(1,4),(2,5)], 3, "6-node Petersen-like, 3 colors"),
        (8, [(i,(i+1)%8) for i in range(8)] + [(i,(i+2)%8) for i in range(8)], 3, "8-node 2-ring, 3 colors"),
        (10, [(i,(i+1)%10) for i in range(10)] + [(i,(i+3)%10) for i in range(10)], 3, "10-node ring+skip, 3 colors"),
        (15, [(i,j) for i in range(15) for j in range(i+1,15) if (i+j) % 3 == 0], 4, "15-node structured, 4 colors"),
    ]

    results = []

    for n_nodes, edges, n_colors, name in test_cases:
        print(f"\n--- {name} ({n_nodes} nodes, {len(edges)} edges) ---")

        total_spins, ising_edges, biases, weights = graph_coloring_to_ising(
            n_nodes, edges, n_colors,
        )

        print(f"  Ising encoding: {total_spins} spins, {len(ising_edges)} couplings")

        # Build thrml model
        nodes = [SpinNode() for _ in range(total_spins)]
        thrml_edges = [(nodes[e[0]], nodes[e[1]]) for e in ising_edges]

        model = IsingEBM(
            nodes=nodes,
            edges=thrml_edges,
            biases=jnp.array(biases, dtype=jnp.float32),
            weights=jnp.array(weights, dtype=jnp.float32),
            beta=jnp.array(8.0),
        )

        free_blocks = [Block([nodes[i]]) for i in range(total_spins)]
        program = IsingSamplingProgram(model, free_blocks, [])
        init_state = hinton_init(jrandom.PRNGKey(n_nodes), model, free_blocks, ())

        schedule = SamplingSchedule(1000, 30, 20)

        t0 = time.time()
        samples = sample_states(
            jrandom.PRNGKey(n_nodes + 42), program, schedule,
            init_state, [], free_blocks,
        )
        sample_time = time.time() - t0

        # Find best coloring from samples
        n_samples = samples[0].shape[0]
        best_valid = 0
        best_coloring = None

        for s_idx in range(n_samples):
            spins = [bool(samples[v][s_idx, 0]) for v in range(total_spins)]
            coloring = decode_coloring(spins, n_nodes, n_colors)
            valid, total = check_coloring(coloring, edges)
            if valid > best_valid:
                best_valid = valid
                best_coloring = coloring

        # Random baseline
        rng = np.random.default_rng(42)
        rand_best = 0
        for _ in range(max(n_samples, 100)):
            rand_coloring = {i: rng.integers(0, n_colors) for i in range(n_nodes)}
            valid, _ = check_coloring(rand_coloring, edges)
            rand_best = max(rand_best, valid)

        pct_thrml = best_valid / len(edges) * 100
        pct_rand = rand_best / len(edges) * 100

        color_names = ["R", "G", "B", "Y"]
        if best_coloring:
            coloring_str = " ".join(
                f"{i}={color_names[c] if c >= 0 and c < len(color_names) else '?'}"
                for i, c in sorted(best_coloring.items())
            )
        else:
            coloring_str = "none"

        print(f"  thrml: {best_valid}/{len(edges)} valid ({pct_thrml:.0f}%) in {sample_time:.1f}s")
        print(f"  Random: {rand_best}/{len(edges)} valid ({pct_rand:.0f}%)")
        print(f"  Coloring: {coloring_str}")

        results.append({
            "name": name,
            "n_nodes": n_nodes,
            "n_edges": len(edges),
            "thrml_valid": best_valid,
            "random_valid": rand_best,
            "total_edges": len(edges),
            "time": sample_time,
            "perfect": best_valid == len(edges),
        })

    # Summary
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 40 RESULTS ({elapsed:.0f}s)")
    print(sep)

    n_perfect_thrml = sum(1 for r in results if r["perfect"])
    n_perfect_rand = sum(1 for r in results if r["random_valid"] == r["total_edges"])

    for r in results:
        t = r["thrml_valid"] / r["total_edges"] * 100
        ra = r["random_valid"] / r["total_edges"] * 100
        perfect = " PERFECT" if r["perfect"] else ""
        print(f"  {r['name']:40s} thrml={t:.0f}% rand={ra:.0f}%{perfect}")

    print(f"\n  Perfect colorings: thrml={n_perfect_thrml}/{len(results)}, random={n_perfect_rand}/{len(results)}")

    mean_thrml = np.mean([r["thrml_valid"] / r["total_edges"] for r in results]) * 100
    mean_rand = np.mean([r["random_valid"] / r["total_edges"] for r in results]) * 100

    if mean_thrml > mean_rand:
        print(f"\n  VERDICT: ✅ thrml beats random ({mean_thrml:.0f}% vs {mean_rand:.0f}%)")
    else:
        print(f"\n  VERDICT: ❌ Random wins ({mean_rand:.0f}% vs {mean_thrml:.0f}%)")

    print(f"\n  This pipeline: LLM generates constraints → Ising encoding → thrml sampling → solution")
    print(f"  On Extropic TSU: sampling would be nanoseconds instead of seconds")
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
