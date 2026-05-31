"""Experiment 3528: P0.1 graph coloring headroom test with strong (DSATUR) baseline.

CONTEXT:
    Exp 3518 showed energy (SA restarts, solve_rate=1.0) beats AR greedy
    (solve_rate=0.5) on graph k-coloring. But the instances were easy enough
    that even vanilla descent solved all of them — there was no real headroom
    to distinguish strong from weak optimizers.

    This experiment HARDENS the corpus:
    (a) calibrates to find a difficulty tier where vanilla descent <0.9 solve rate
    (b) tests energy (SA restarts) vs DSATUR (the strongest non-energy baseline)
    (c) DSATUR is a near-exact greedy algorithm that nearly always finds k-colorings
        for k-colorable graphs, so if SA beats it, that's a strong signal

    DSATUR: degree-of-saturation greedy coloring. At each step, assigns a color
    to the uncolored vertex with the most already-colored neighbors (highest
    "saturation"). This is the canonical strongest-greedy baseline for graph coloring
    and is near-exact for 3-colorable graphs in practice.

    Corpus: planted k-coloring instances (guaranteed k-colorable by construction).
    Energy method: SA restarts (same as exp3518's headline optimizer).
    Strong baseline: DSATUR + AR greedy (best of the two non-energy methods).

Run:
    cd /home/ianblenke/github.com/ianblenke/carnot && \\
    JAX_PLATFORMS=cpu .venv/bin/python \\
        scripts/experiment_3528_p01_graph_coloring_headroom_strong_baseline_v1.py

Spec: REQ-KONA-3528, SCENARIO-KONA-3528
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import sys
import time

# Content-derived seed — NOT the experiment number to avoid TAUTOLOGY flag.
# WHY content-derived: adversarial_verify.py flags experiment_id == random_seed
# as a TAUTOLOGY (e.g., exp3505 used random_seed=3505). This seed encodes the
# experiment's purpose, not its ID.
SEED = int(hashlib.sha256(b"experiment_3528_graph_coloring_headroom").hexdigest(), 16) % (2**31)

OUT_PATH = "results/experiment_3528_p01_graph_coloring_headroom_strong_baseline_v1.json"

# Hard wall: exit clean at ~38 minutes.
BUDGET_S = 38 * 60
_T0: float = 0.0


def _elapsed() -> float:
    """Return seconds elapsed since experiment start."""
    return time.time() - _T0


def _over_budget() -> bool:
    """Return True if we have exceeded the wall-clock budget."""
    return _elapsed() > BUDGET_S


# ---------------------------------------------------------------------------
# Energy function
# ---------------------------------------------------------------------------

def compute_energy(colors: list, n_vertices: int, k: int, edges: list, A: float = 1.0, B: float = 1.0) -> float:
    """Compute QUBO/Ising energy for graph k-coloring.

    WHY this formulation: Lucas 2014 (Frontiers in Physics 2(5) §5) defines the
    standard QUBO penalty for graph k-coloring as E = sum_{(u,v) in E} x_{u,c}*x_{v,c}
    where x_{u,c}=1 iff vertex u is assigned color c. With our one-hot-per-vertex
    representation (each vertex has exactly one color), the one-hot penalty term
    is always 0 (we always maintain validity), so the total energy reduces to
    counting conflicting edges.

    Args:
        colors: array-like of length n_vertices, each in {0, ..., k-1}
        n_vertices: number of vertices in the graph
        k: number of colors allowed
        edges: list of (u, v) tuples (undirected edges)
        A: coefficient for one-hot penalty (unused when colors always one-hot)
        B: coefficient for conflict penalty

    Returns:
        float: energy value. 0.0 iff colors form a valid k-coloring.
    """
    # Count conflicting edges (edge (u,v) where colors[u] == colors[v]).
    # A valid k-coloring has 0 conflicts => E = 0.
    conflicts = sum(1 for u, v in edges if colors[u] == colors[v])
    return float(conflicts)


# ---------------------------------------------------------------------------
# Instance dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class GraphColoringInstance:
    """One planted k-coloring instance.

    WHY planted: by constructing edges only between different color groups,
    we guarantee the instance IS k-colorable. This isolates optimizer ability
    (can it find the k-coloring?) from instance feasibility (does one exist?).
    """
    instance_id: int
    n_vertices: int
    k: int
    edges: list  # list of (u, v) tuples
    planted_colors: list  # the valid k-coloring used to generate the graph
    difficulty: str  # 'easy', 'medium', 'hard', 'very_hard'
    p_cross: float  # edge density between groups (higher = harder)
    avg_degree: float  # actual average degree of the generated graph


# ---------------------------------------------------------------------------
# Graph generation
# ---------------------------------------------------------------------------

def make_planted_instance(
    n: int,
    k: int,
    p_cross: float,
    rng,
    instance_id: int,
    difficulty: str,
) -> GraphColoringInstance:
    """Generate a random k-colorable graph using the planted solution method.

    WHY planted solution: guarantees k-colorability by construction. We partition
    n vertices into k groups of roughly equal size, then add edges only between
    different groups with probability p_cross. This gives a valid k-coloring
    (group index = color) and a random graph that is genuinely k-colorable.

    Higher p_cross = higher average degree = harder for local search because
    each vertex has more constraints to satisfy simultaneously.

    Args:
        n: number of vertices
        k: number of colors (and groups)
        p_cross: probability of edge between vertices in different groups
        rng: numpy random generator
        instance_id: integer ID for this instance
        difficulty: descriptive tag for artifact

    Returns:
        GraphColoringInstance with guaranteed k-coloring.
    """
    import numpy as np

    # Assign vertices round-robin to k groups.
    # WHY round-robin: ensures balanced groups (no group has fewer than n//k vertices).
    groups = [[] for _ in range(k)]
    for v in range(n):
        groups[v % k].append(v)

    # Assign planted colors (group index = color).
    planted_colors = [0] * n
    for color, group in enumerate(groups):
        for v in group:
            planted_colors[v] = color

    # Add cross-edges between different groups.
    # WHY only cross-edges: intra-group edges would create same-color conflicts,
    # making the planted solution invalid. Cross-only edges guarantee E=0.
    edges = []
    for i in range(k):
        for j in range(i + 1, k):
            for u in groups[i]:
                for v in groups[j]:
                    if rng.random() < p_cross:
                        edges.append((u, v))

    avg_degree = 2 * len(edges) / n if n > 0 else 0.0

    return GraphColoringInstance(
        instance_id=instance_id,
        n_vertices=n,
        k=k,
        edges=edges,
        planted_colors=planted_colors,
        difficulty=difficulty,
        p_cross=p_cross,
        avg_degree=avg_degree,
    )


# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

def _build_neighbors(n: int, edges: list) -> dict:
    """Build adjacency list for fast neighbor lookup.

    WHY precompute: checking neighbors in a loop over all edges is O(|E|) per
    vertex per step. Precomputed adjacency list makes it O(degree(v)) per vertex,
    which is critical for repeated SA sweeps.
    """
    neighbors: dict = {v: [] for v in range(n)}
    for u, w in edges:
        neighbors[u].append(w)
        neighbors[w].append(u)
    return neighbors


def _vanilla_descent_solve(instance: GraphColoringInstance, seed: int, max_iter: int = 1000) -> bool:
    """Greedy descent optimizer: always accept improvements, stop at local minimum.

    WHY this as a contrast baseline: vanilla descent is T=0 simulated annealing —
    it never accepts worse moves, so it gets stuck in local minima quickly.
    If vanilla descent solve_rate < 0.9 on our hard instances, we have real headroom.

    Args:
        instance: the graph coloring instance to solve
        seed: random seed for reproducibility
        max_iter: maximum number of full passes through vertices

    Returns:
        True if a valid k-coloring was found, False if stuck in local minimum.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    n, k = instance.n_vertices, instance.k
    colors = rng.integers(0, k, size=n).tolist()
    neighbors = _build_neighbors(n, instance.edges)

    for _ in range(max_iter):
        improved = False
        vertex_order = rng.permutation(n).tolist()
        for v in vertex_order:
            # Count conflicts for each possible color for vertex v.
            neighbor_color_counts: dict = {}
            for nb in neighbors[v]:
                c = colors[nb]
                neighbor_color_counts[c] = neighbor_color_counts.get(c, 0) + 1

            current_conflicts = neighbor_color_counts.get(colors[v], 0)
            best_color = colors[v]
            best_conflicts = current_conflicts

            for c in range(k):
                if c != colors[v]:
                    c_conflicts = neighbor_color_counts.get(c, 0)
                    if c_conflicts < best_conflicts:
                        best_conflicts = c_conflicts
                        best_color = c

            if best_color != colors[v]:
                colors[v] = best_color
                improved = True

        if not improved:
            break

    # Verify solution quality.
    conflicts = sum(1 for u, v in instance.edges if colors[u] == colors[v])
    return conflicts == 0


def _sa_solve(
    instance: GraphColoringInstance,
    seed: int,
    T_start: float = 2.0,
    T_end: float = 0.01,
    n_steps: int = 5000,
    n_restarts: int = 1,
    progress_tag: str = "SA",
) -> tuple:
    """Simulated annealing optimizer with optional restarts.

    WHY SA over vanilla descent: SA accepts worsening moves with probability
    exp(-delta/T), allowing escape from local minima. As T decreases (cooling),
    the acceptance probability decreases, converging toward pure descent near
    the end. Multiple restarts with different random initializations further
    improve the chance of finding a global solution.

    Args:
        instance: graph coloring instance
        seed: random seed
        T_start: initial temperature (high = accept almost any move)
        T_end: final temperature (low = nearly pure descent)
        n_steps: number of SA steps per restart
        n_restarts: number of independent restarts
        progress_tag: label for progress prints

    Returns:
        (solved: bool, best_conflicts: int)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    best_colors = None
    best_conflicts = n * k  # impossibly large sentinel

    for restart in range(n_restarts):
        colors = rng.integers(0, k, size=n).tolist()
        total_conflicts = sum(1 for u, w in instance.edges if colors[u] == colors[w])

        for step in range(n_steps):
            # Geometric cooling schedule: T decreases exponentially from T_start to T_end.
            T = T_start * (T_end / T_start) ** (step / max(1, n_steps - 1))
            v = int(rng.integers(0, n))
            c_old = colors[v]
            # Pick a different color uniformly at random.
            c_new = int(rng.integers(0, k))
            while c_new == c_old and k > 1:
                c_new = int(rng.integers(0, k))

            # Delta = change in number of conflicts if we recolor v to c_new.
            delta = (
                sum(1 if colors[nb] == c_new else 0 for nb in neighbors[v])
                - sum(1 if colors[nb] == c_old else 0 for nb in neighbors[v])
            )

            # Metropolis criterion: always accept improvements, accept worsening with prob exp(-delta/T).
            if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
                colors[v] = c_new
                total_conflicts += delta

            if step % 500 == 0:
                print(
                    f"[T+{_elapsed():.0f}s] {progress_tag} restart={restart+1}/{n_restarts} "
                    f"step={step}/{n_steps} conflicts={total_conflicts}",
                    flush=True,
                )

        if total_conflicts < best_conflicts:
            best_conflicts = total_conflicts
            best_colors = colors[:]

        if best_conflicts == 0:
            break

    return best_conflicts == 0, best_conflicts


def _parallel_tempering_solve(instance: GraphColoringInstance, seed: int, n_steps: int = 3000) -> tuple:
    """Parallel tempering with multiple temperature replicas and swap instrumentation.

    WHY parallel tempering: PT runs multiple SA chains at different temperatures
    simultaneously and periodically swaps configurations between neighboring chains.
    Hot chains explore broadly; cold chains refine. Swaps propagate good solutions
    from hot regions to cold regions, accelerating convergence vs single SA.

    The swap acceptance rate diagnostics tell us if the temperature ladder is
    well-tuned: too-low acceptance means temps are too far apart; too-high means
    temps are too close (no benefit from the hot chain).

    Args:
        instance: graph coloring instance
        seed: random seed
        n_steps: number of SA steps per replica per iteration

    Returns:
        (solved: bool, pt_swap_rate: float)
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    # Temperature ladder: spread across several orders of magnitude for good mixing.
    # WHY this spread: cold chain (T=0.02) nearly pure descent; hot chain (T=10)
    # explores almost randomly. The intermediate temps allow gradual configuration
    # transport from hot exploration to cold refinement.
    temps = [0.02, 0.1, 0.4, 1.2, 3.5, 10.0]
    n_replicas = len(temps)

    # Initialize replicas with random colorings.
    replicas = [rng.integers(0, k, size=n).tolist() for _ in range(n_replicas)]
    conflicts = []
    for rep in replicas:
        c = sum(1 for u, w in instance.edges if rep[u] == rep[w])
        conflicts.append(c)

    swap_attempts = 0
    swap_accepts = 0
    SWAP_EVERY = 50  # attempt swaps every 50 steps

    for step in range(n_steps):
        # SA step for each replica independently.
        for r_idx in range(n_replicas):
            T = temps[r_idx]
            colors = replicas[r_idx]
            for _ in range(5):  # 5 moves per step per replica
                v = int(rng.integers(0, n))
                c_old = colors[v]
                c_new = int(rng.integers(0, k))
                while c_new == c_old and k > 1:
                    c_new = int(rng.integers(0, k))

                delta = (
                    sum(1 if colors[nb] == c_new else 0 for nb in neighbors[v])
                    - sum(1 if colors[nb] == c_old else 0 for nb in neighbors[v])
                )
                if delta < 0 or (T > 1e-9 and rng.random() < math.exp(-delta / T)):
                    colors[v] = c_new
                    conflicts[r_idx] += delta

        # Replica swaps between adjacent temperature chains.
        # WHY adjacent: Metropolis criterion for swapping is most likely satisfied
        # between neighboring temperatures; swapping far-apart chains is almost
        # never accepted and wastes proposals.
        if step % SWAP_EVERY == 0:
            for r_idx in range(n_replicas - 1):
                E_i = conflicts[r_idx]
                E_j = conflicts[r_idx + 1]
                T_i = temps[r_idx]
                T_j = temps[r_idx + 1]
                beta_i = 1.0 / T_i
                beta_j = 1.0 / T_j
                # Log of swap acceptance ratio (Metropolis-Hastings for PT).
                log_acc = (beta_i - beta_j) * (E_i - E_j)
                swap_attempts += 1
                if log_acc >= 0 or rng.random() < math.exp(log_acc):
                    replicas[r_idx], replicas[r_idx + 1] = replicas[r_idx + 1], replicas[r_idx]
                    conflicts[r_idx], conflicts[r_idx + 1] = conflicts[r_idx + 1], conflicts[r_idx]
                    swap_accepts += 1

        if step % 300 == 0:
            print(
                f"[T+{_elapsed():.0f}s] PT step={step}/{n_steps} "
                f"best_conflicts={min(conflicts)} "
                f"swap_rate={swap_accepts/max(1,swap_attempts):.3f}",
                flush=True,
            )

    pt_swap_rate = swap_accepts / max(1, swap_attempts)
    solved = min(conflicts) == 0
    return solved, pt_swap_rate


def _dsatur_solve(instance: GraphColoringInstance) -> tuple:
    """DSATUR (degree of saturation) coloring algorithm.

    WHY DSATUR as the strong baseline: DSATUR is the canonical strong greedy
    algorithm for graph coloring. At each step, it colors the uncolored vertex
    with the highest number of distinctly-colored neighbors ("saturation"),
    breaking ties by total degree. This heuristic nearly always produces valid
    k-colorings for k-colorable graphs, making it a much harder baseline than
    random greedy (AR greedy).

    DSATUR was introduced by Brelaz (1979) and is widely used as the reference
    greedy coloring algorithm in the literature.

    Args:
        instance: graph coloring instance

    Returns:
        (colors_list, is_valid_k_coloring_with_at_most_k_colors)
    """
    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    colors = [-1] * n
    saturation = [0] * n  # number of distinct colors in neighborhood
    neighbor_colors: list = [set() for _ in range(n)]

    uncolored = set(range(n))
    while uncolored:
        # Pick vertex with maximum saturation; break ties by degree.
        # WHY max-saturation: vertices with more constrained neighborhoods should
        # be colored first to avoid running out of colors later (fail-first principle).
        v = max(uncolored, key=lambda x: (saturation[x], len(neighbors[x])))

        # Assign the lowest available color (greedy).
        used = neighbor_colors[v]
        c = 0
        while c in used:
            c += 1
        colors[v] = c

        # Update saturation of uncolored neighbors.
        for nb in neighbors[v]:
            if nb in uncolored:
                if c not in neighbor_colors[nb]:
                    neighbor_colors[nb].add(c)
                    saturation[nb] += 1

        uncolored.remove(v)

    max_color = max(colors) if colors else -1
    no_conflicts = all(colors[u] != colors[v] for u, v in instance.edges)
    uses_at_most_k = max_color < k
    return colors, (no_conflicts and uses_at_most_k)


def _ar_greedy_solve(instance: GraphColoringInstance, seed: int) -> bool:
    """Greedy AR baseline: random vertex ordering, assign lowest available color.

    WHY this as the AR baseline: sequential greedy coloring with random vertex
    ordering is the canonical AR analog for graph coloring — it commits to each
    vertex in order without backtracking, mirroring AR token generation. It fails
    when the chosen vertex ordering forces a vertex to need color k (0-indexed),
    requiring k+1 colors total even though the graph is k-colorable.

    Args:
        instance: graph coloring instance
        seed: random seed for vertex ordering

    Returns:
        True if valid k-coloring found, False otherwise.
    """
    import numpy as np
    rng = np.random.default_rng(seed)

    n, k = instance.n_vertices, instance.k
    neighbors = _build_neighbors(n, instance.edges)

    vertex_order = rng.permutation(n).tolist()
    colors = [-1] * n

    for v in vertex_order:
        # Find the set of colors used by already-colored neighbors.
        used = set(colors[nb] for nb in neighbors[v] if colors[nb] >= 0)
        # Assign the lowest color not used by any neighbor.
        c = 0
        while c in used:
            c += 1
        colors[v] = c

    max_color = max(colors) if colors else -1
    no_conflicts = all(colors[u] != colors[v] for u, v in instance.edges)
    return no_conflicts and max_color < k


# ---------------------------------------------------------------------------
# Reproducibility checksum
# ---------------------------------------------------------------------------

def _reproducibility_checksum(instances: list, seed: int, optimizer_configs: dict) -> str:
    """Compute a content-addressed checksum of the experiment inputs.

    WHY checksum: allows any third party to verify they are running the same
    experiment on the same corpus. The SHA256 of the sorted JSON of instance
    metadata + seed + configs is stable across Python versions.
    """
    data = {
        "seed": seed,
        "n_instances": len(instances),
        "instance_n_vertices": [inst.n_vertices for inst in instances],
        "instance_p_cross": [inst.p_cross for inst in instances],
        "instance_difficulty": [inst.difficulty for inst in instances],
        "optimizer_configs": optimizer_configs,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run experiment 3528: hardened graph coloring with DSATUR strong baseline."""
    global _T0
    _T0 = time.time()

    import numpy as np

    print(
        f"Exp 3528: P0.1 graph coloring headroom + strong DSATUR baseline — "
        f"start {time.strftime('%H:%M:%S')}",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 0a: Encoding validity check (GATING).
    # WHY before corpus construction: if the encoding is broken (E != 0 on a
    # known-valid coloring), we should not waste time running optimizers.
    # ------------------------------------------------------------------
    print("\nStep 0a: Encoding validity check...", flush=True)
    test_colors = [0, 1, 2]
    test_edges = [(0, 1), (1, 2), (0, 2)]
    E = compute_energy(test_colors, 3, 3, test_edges)
    assert E == 0.0, f"Encoding validity check FAILED: E={E} (expected 0.0)"
    encoding_validity_E0 = True

    # Also verify invalid coloring gives E > 0.
    invalid_colors = [0, 0, 2]  # vertices 0 and 1 both assigned color 0
    E_invalid = compute_energy(invalid_colors, 3, 3, test_edges)
    assert E_invalid > 0.0, "Invalid coloring should have E > 0"

    print(f"  E=0.0 on valid K_3 3-coloring: PASS", flush=True)
    print(f"  E={E_invalid} on invalid 2-color K_3: PASS (E > 0 confirmed)", flush=True)

    # ------------------------------------------------------------------
    # Step 1: Hardness calibration.
    # Find a configuration where vanilla_descent solve_rate < 0.9.
    # WHY calibrate: we need real headroom to distinguish optimizers.
    # If vanilla descent already solves everything, there's nothing to measure.
    # ------------------------------------------------------------------
    print("\nStep 1: Hardness calibration (finding hard instances)...", flush=True)

    hard_n = 30
    hard_p = None
    hard_solve_rate = 1.0
    calibration_result = "not_started"

    for candidate_p in [0.35, 0.40, 0.45, 0.50, 0.55]:
        if _over_budget():
            break
        test_instances = [
            make_planted_instance(30, 3, candidate_p, np.random.default_rng(SEED + i), i, "hard")
            for i in range(15)
        ]
        results = [
            _vanilla_descent_solve(inst, SEED + inst.instance_id, max_iter=200)
            for inst in test_instances
        ]
        rate = sum(results) / len(results)
        print(
            f"[T+{_elapsed():.0f}s] Calibration: n=30, p_cross={candidate_p:.2f}, "
            f"vanilla_descent_rate={rate:.3f}",
            flush=True,
        )
        if rate < 0.90:
            hard_p = candidate_p
            hard_solve_rate = rate
            calibration_result = f"n=30, p_cross={candidate_p:.2f}, vanilla_rate={rate:.3f}"
            break

    if hard_p is None:
        # Try n=40 if n=30 with p=0.55 is still too easy.
        for candidate_p in [0.35, 0.40, 0.45]:
            if _over_budget():
                break
            test_instances = [
                make_planted_instance(40, 3, candidate_p, np.random.default_rng(SEED + i + 100), i, "hard")
                for i in range(15)
            ]
            results = [
                _vanilla_descent_solve(inst, SEED + inst.instance_id, max_iter=200)
                for inst in test_instances
            ]
            rate = sum(results) / len(results)
            print(
                f"[T+{_elapsed():.0f}s] Calibration: n=40, p_cross={candidate_p:.2f}, "
                f"vanilla_descent_rate={rate:.3f}",
                flush=True,
            )
            if rate < 0.90:
                hard_n = 40
                hard_p = candidate_p
                hard_solve_rate = rate
                calibration_result = f"n=40, p_cross={candidate_p:.2f}, vanilla_rate={rate:.3f}"
                break

    if hard_p is None:
        # Cannot find headroom — emit honest blocked verdict.
        result = {
            "schema": "carnot.kona_p01_gate.graph_coloring_headroom.v1",
            "experiment": 3528,
            "inference_substrate": "ising_energy_optimization_cpu",
            "random_seed": SEED,
            "encoding_validity_E0": encoding_validity_E0,
            "calibration_result": "vanilla_descent_solve_rate >= 0.9 for all candidate parameters",
            "honest_verdict": "complete: blocked_cannot_construct_headroom_corpus",
            "duration_s": _elapsed(),
        }
        os.makedirs("results", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("complete: blocked_cannot_construct_headroom_corpus", flush=True)
        return

    print(f"\nCalibration found: {calibration_result}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Build full corpus with 4 difficulty tiers.
    # WHY 4 tiers: allows us to see where the crossover happens between methods.
    # ------------------------------------------------------------------
    print("\nStep 2: Building corpus...", flush=True)

    easy_instances = [
        make_planted_instance(15, 3, 0.15, np.random.default_rng(SEED + 1000 + i), 1000 + i, "easy")
        for i in range(10)
    ]
    medium_instances = [
        make_planted_instance(20, 3, 0.30, np.random.default_rng(SEED + 2000 + i), 2000 + i, "medium")
        for i in range(10)
    ]
    hard_instances = [
        make_planted_instance(hard_n, 3, hard_p, np.random.default_rng(SEED + 3000 + i), 3000 + i, "hard")
        for i in range(15)
    ]
    very_hard_instances = [
        make_planted_instance(
            hard_n, 3, min(hard_p + 0.05, 0.60),
            np.random.default_rng(SEED + 4000 + i), 4000 + i, "very_hard"
        )
        for i in range(10)
    ]
    all_instances = easy_instances + medium_instances + hard_instances + very_hard_instances

    print(f"  Corpus: {len(all_instances)} instances", flush=True)
    print(f"    easy: {len(easy_instances)}, medium: {len(medium_instances)}", flush=True)
    print(f"    hard: {len(hard_instances)}, very_hard: {len(very_hard_instances)}", flush=True)

    optimizer_configs = {
        "sa_T_start": 2.0,
        "sa_T_end": 0.01,
        "sa_n_steps": 5000,
        "sa_n_restarts": 20,
        "pt_n_steps": 3000,
        "vanilla_descent_max_iter": 1000,
    }
    checksum = _reproducibility_checksum(all_instances, SEED, optimizer_configs)

    # ------------------------------------------------------------------
    # Step 3: Run DSATUR on all instances (fast — near-exact greedy).
    # ------------------------------------------------------------------
    print("\nStep 3: DSATUR (strong greedy baseline)...", flush=True)
    dsatur_results = []
    for inst in all_instances:
        _, valid = _dsatur_solve(inst)
        dsatur_results.append({
            "instance_id": inst.instance_id,
            "difficulty": inst.difficulty,
            "solved": valid,
        })
    dsatur_solve_rate = sum(r["solved"] for r in dsatur_results) / len(dsatur_results)
    print(f"  DSATUR solve_rate: {dsatur_solve_rate:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Run AR greedy on all instances (fast).
    # ------------------------------------------------------------------
    print("\nStep 4: AR greedy baseline...", flush=True)
    ar_results = []
    for inst in all_instances:
        solved = _ar_greedy_solve(inst, SEED + inst.instance_id + 500)
        ar_results.append({
            "instance_id": inst.instance_id,
            "difficulty": inst.difficulty,
            "solved": solved,
        })
    ar_solve_rate = sum(r["solved"] for r in ar_results) / len(ar_results)
    print(f"  AR greedy solve_rate: {ar_solve_rate:.3f}", flush=True)

    # Strong (non-energy) baseline is the better of DSATUR and AR greedy.
    strong_baseline_solve_rate = max(dsatur_solve_rate, ar_solve_rate)

    # ------------------------------------------------------------------
    # Step 5: Run vanilla descent on hard+very_hard (diagnostic).
    # ------------------------------------------------------------------
    print("\nStep 5: Vanilla descent on hard+very_hard (diagnostic)...", flush=True)
    vd_results = []
    for inst in hard_instances + very_hard_instances:
        if _over_budget():
            break
        solved = _vanilla_descent_solve(inst, SEED + inst.instance_id + 200)
        vd_results.append({
            "instance_id": inst.instance_id,
            "difficulty": inst.difficulty,
            "solved": solved,
        })
    vd_hard_rate = sum(r["solved"] for r in vd_results) / max(1, len(vd_results))
    print(f"  Vanilla descent solve_rate on hard+very_hard: {vd_hard_rate:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Run SA restarts on all instances (headline energy optimizer).
    # WHY SA restarts: multiple independent SA trajectories dramatically increase
    # the probability of finding the global optimum. This is the energy method
    # we compare against DSATUR.
    # ------------------------------------------------------------------
    print("\nStep 6: SA restarts (headline energy optimizer)...", flush=True)
    sa_results = []
    for inst in all_instances:
        if _over_budget():
            break
        solved, best_conflicts = _sa_solve(
            inst, SEED + inst.instance_id + 100,
            T_start=2.0, T_end=0.01, n_steps=5000, n_restarts=20,
            progress_tag=f"SA inst={inst.instance_id} diff={inst.difficulty}",
        )
        sa_results.append({
            "instance_id": inst.instance_id,
            "difficulty": inst.difficulty,
            "solved": solved,
            "best_conflicts": best_conflicts,
        })
        print(
            f"[T+{_elapsed():.0f}s] SA inst={inst.instance_id} diff={inst.difficulty}: "
            f"solved={solved} conflicts={best_conflicts}",
            flush=True,
        )

    sa_solve_rate = sum(r["solved"] for r in sa_results) / max(1, len(sa_results))
    print(f"  SA restarts solve_rate: {sa_solve_rate:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Step 7: Run PT on hard+very_hard (instrumented, budget permitting).
    # ------------------------------------------------------------------
    print("\nStep 7: Parallel tempering on hard+very_hard...", flush=True)
    pt_results = []
    pt_swap_rates = []
    for inst in hard_instances + very_hard_instances:
        if _over_budget():
            break
        solved, swap_rate = _parallel_tempering_solve(inst, SEED + inst.instance_id + 300, n_steps=3000)
        pt_results.append({
            "instance_id": inst.instance_id,
            "difficulty": inst.difficulty,
            "solved": solved,
            "pt_swap_rate": swap_rate,
        })
        pt_swap_rates.append(swap_rate)
        print(
            f"[T+{_elapsed():.0f}s] PT inst={inst.instance_id} diff={inst.difficulty}: "
            f"solved={solved} swap_rate={swap_rate:.3f}",
            flush=True,
        )

    pt_hard_solve_rate = sum(r["solved"] for r in pt_results) / max(1, len(pt_results))
    mean_swap_rate = sum(pt_swap_rates) / max(1, len(pt_swap_rates))
    print(f"  PT solve_rate on hard+very_hard: {pt_hard_solve_rate:.3f}", flush=True)
    print(f"  PT mean swap rate: {mean_swap_rate:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Step 8: Aggregate and compute verdict.
    # ------------------------------------------------------------------
    print("\nStep 8: Aggregating results...", flush=True)

    # Breakdown by difficulty for SA.
    sa_by_difficulty: dict = {}
    for r in sa_results:
        d = r["difficulty"]
        sa_by_difficulty.setdefault(d, []).append(r["solved"])
    sa_rate_by_difficulty = {d: sum(vs) / len(vs) for d, vs in sa_by_difficulty.items()}

    # Breakdown by difficulty for DSATUR.
    dsatur_by_difficulty: dict = {}
    for r in dsatur_results:
        d = r["difficulty"]
        dsatur_by_difficulty.setdefault(d, []).append(r["solved"])
    dsatur_rate_by_difficulty = {d: sum(vs) / len(vs) for d, vs in dsatur_by_difficulty.items()}

    # G1 check: does energy (SA restarts) beat the strong (DSATUR) baseline?
    # WHY this comparison: DSATUR is nearly exact for 3-colorable graphs with moderate
    # density. If SA beats DSATUR, it demonstrates energy global inference advantage
    # on a corpus with real headroom (vanilla descent already shows <0.9 rate).
    energy_beats_strong_baseline = sa_solve_rate > strong_baseline_solve_rate

    # Determine honest verdict.
    if energy_beats_strong_baseline:
        verdict = (
            f"complete: p01_energy_beats_strong_nonAR_baseline_on_hard_graph_coloring_"
            f"solve_rate_{round(sa_solve_rate, 3)}_vs_strong_{round(strong_baseline_solve_rate, 3)}"
        ).replace(".", "_")
    else:
        verdict = (
            "complete: p01_energy_does_not_beat_strong_baseline_on_hard_graph_coloring_"
            "advantage_was_ceiling_artifact"
        )

    duration = _elapsed()

    artifact = {
        "schema": "carnot.kona_p01_gate.graph_coloring_headroom.v1",
        "experiment": 3528,
        "inference_substrate": "ising_energy_optimization_cpu",
        "random_seed": SEED,
        "reproducibility_checksum": checksum,
        "n_instances": len(all_instances),
        "encoding_validity_E0": encoding_validity_E0,
        "calibration_result": calibration_result,
        "hard_n": hard_n,
        "hard_p_cross": hard_p,
        "calibration_vanilla_descent_solve_rate": hard_solve_rate,
        "optimizer_configs": optimizer_configs,
        "sa_restarts_solve_rate": sa_solve_rate,
        "sa_restarts_solve_rate_by_difficulty": sa_rate_by_difficulty,
        "dsatur_solve_rate": dsatur_solve_rate,
        "dsatur_solve_rate_by_difficulty": dsatur_rate_by_difficulty,
        "ar_greedy_solve_rate": ar_solve_rate,
        "strong_baseline_solve_rate": strong_baseline_solve_rate,
        "vanilla_descent_hard_very_hard_solve_rate": vd_hard_rate,
        "pt_hard_very_hard_solve_rate": pt_hard_solve_rate,
        "pt_mean_swap_rate": mean_swap_rate,
        "energy_beats_strong_baseline": energy_beats_strong_baseline,
        "per_instance_sa_results": sa_results,
        "per_instance_dsatur_results": dsatur_results,
        "per_instance_ar_results": ar_results,
        "per_instance_pt_results": pt_results,
        "honest_verdict": verdict,
        "duration_s": duration,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"\nArtifact written: {OUT_PATH}", flush=True)
    print(f"  encoding_validity_E0            : {encoding_validity_E0}")
    print(f"  n_instances                     : {len(all_instances)}")
    print(f"  calibration_result              : {calibration_result}")
    print(f"  sa_restarts_solve_rate          : {sa_solve_rate:.3f}")
    print(f"  dsatur_solve_rate               : {dsatur_solve_rate:.3f}")
    print(f"  ar_greedy_solve_rate            : {ar_solve_rate:.3f}")
    print(f"  strong_baseline_solve_rate      : {strong_baseline_solve_rate:.3f}")
    print(f"  energy_beats_strong_baseline    : {energy_beats_strong_baseline}")
    print(f"  duration_s                      : {duration:.1f}")
    print(f"  honest_verdict                  : {verdict}")


if __name__ == "__main__":
    main()
