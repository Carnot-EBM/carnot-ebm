#!/usr/bin/env python3
"""Experiment 52: Cross-domain transfer of learned Ising models.

**The big question:**
    Do Ising models learned on one SAT domain transfer to another?
    Unlike activation-based EBMs (which showed ~50% transfer in prior work),
    Ising constraints encode STRUCTURAL rules via pairwise couplings. If
    structure is shared across domains (e.g., all 3-SAT instances share the
    3-literal-per-clause structure), the learned couplings might generalize.

**Experimental design:**
    Train Ising models on three different SAT structures:
      (a) Random 3-SAT at clause/variable ratio 4.26 (phase transition boundary)
      (b) Random 3-SAT at ratio 3.0 (underconstrained, many solutions)
      (c) Graph coloring encoded as SAT (structured, non-random clauses)

    Test each model on all three domains. Report a 3x3 transfer matrix
    where entry (i,j) = mean SAT% when model trained on domain i is
    evaluated on domain j's test instances.

**Why this matters:**
    If transfer works, a single Ising model trained on easy instances could
    help solve harder instances in a different domain. This would make
    thermodynamic reasoning engines more practical: train once, deploy across
    constraint types. If transfer fails, it tells us that Ising couplings
    are instance-specific, not structure-specific.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_52_transfer_ising.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


# --- SAT instance generators ---

def generate_random_3sat_data(
    n_vars: int,
    ratio: float,
    n_samples: int,
    seed: int = 42,
) -> tuple[list, np.ndarray]:
    """Generate satisfying assignments for a random 3-SAT instance.

    Uses brute-force random search to find assignments that satisfy all clauses.
    Falls back to near-satisfying (all but 1 clause) if exact solutions are scarce,
    which happens near the phase transition ratio ~4.26.

    Args:
        n_vars: Number of Boolean variables.
        ratio: Clause-to-variable ratio. 4.26 is the phase transition for 3-SAT;
            below that, most instances are satisfiable with many solutions.
        n_samples: Number of satisfying assignments to collect.
        seed: Random seed for reproducibility.

    Returns:
        (clauses, data) where clauses is a list of 3-literal lists and
        data is shape (n_found, n_vars) float32 with values in {0, 1}.
    """
    from experiment_39_thrml_sat import random_3sat, check_assignment

    n_clauses = int(n_vars * ratio)
    clauses = random_3sat(n_vars, n_clauses, seed=seed)

    rng = np.random.default_rng(seed)
    data = []
    attempts = 0
    max_attempts = n_samples * 2000

    # First pass: find exact satisfying assignments.
    while len(data) < n_samples and attempts < max_attempts:
        assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
        sat, total = check_assignment(clauses, assignment)
        if sat == total:
            data.append([assignment[i + 1] for i in range(n_vars)])
        attempts += 1

    # Second pass: accept near-satisfying if we don't have enough exact ones.
    # This is necessary for hard instances near the phase transition where
    # exact solutions are exponentially rare.
    if len(data) < n_samples:
        while len(data) < n_samples and attempts < max_attempts * 2:
            assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
            sat, total = check_assignment(clauses, assignment)
            if sat >= total - 1:
                data.append([assignment[i + 1] for i in range(n_vars)])
            attempts += 1

    return clauses, np.array(data[:n_samples], dtype=np.float32)


def graph_coloring_to_sat(n_nodes: int, n_colors: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Encode a graph coloring problem as a SAT instance.

    Each node-color pair (node i, color c) becomes a Boolean variable.
    Variable numbering: var(i, c) = i * n_colors + c + 1 (1-indexed).

    Clauses encode two types of constraints:
      1. At-least-one: each node must have at least one color.
         For node i: (x_{i,0} OR x_{i,1} OR ... OR x_{i,k-1})
      2. Not-both: adjacent nodes cannot share a color.
         For edge (i,j), color c: (NOT x_{i,c} OR NOT x_{j,c})

    For 3-coloring, the at-least-one clauses have exactly 3 literals
    (matching 3-SAT structure). For other color counts, the clauses
    have different widths, but the not-both clauses are always 2-literal.

    Args:
        n_nodes: Number of graph nodes.
        n_colors: Number of colors available.
        edges: List of (node_i, node_j) edges, 0-indexed.

    Returns:
        List of clauses, each a list of signed integers (SAT literal format).
    """
    def var(node: int, color: int) -> int:
        return node * n_colors + color + 1

    clauses = []

    # At-least-one color per node.
    for i in range(n_nodes):
        clause = [var(i, c) for c in range(n_colors)]
        # For 3-coloring this is exactly a 3-literal clause.
        clauses.append(clause)

    # Adjacent nodes cannot share a color.
    for i, j in edges:
        for c in range(n_colors):
            clauses.append([-var(i, c), -var(j, c)])

    return clauses


def generate_graph_coloring_data(
    n_vars: int,
    n_samples: int,
    seed: int = 42,
) -> tuple[list, np.ndarray]:
    """Generate satisfying assignments for a graph 3-coloring SAT instance.

    Creates a random graph with n_vars/3 nodes (since each node needs 3 color
    variables for a 3-coloring encoding) and encodes the coloring constraint
    as SAT. Then searches for valid colorings via random assignment.

    Args:
        n_vars: Total number of Boolean variables (must be divisible by 3).
        n_samples: Target number of satisfying assignments.
        seed: Random seed.

    Returns:
        (clauses, data) where data has shape (n_found, n_vars) float32.
    """
    from experiment_39_thrml_sat import check_assignment

    n_colors = 3
    n_nodes = n_vars // n_colors  # 5 nodes for n_vars=15

    # Generate a random sparse graph (each node connected to ~2 others).
    rng = np.random.default_rng(seed)
    edges = set()
    for i in range(n_nodes):
        n_neighbors = rng.integers(1, min(3, n_nodes))
        targets = rng.choice([j for j in range(n_nodes) if j != i], size=n_neighbors, replace=False)
        for j in targets:
            edge = (min(i, j), max(i, j))
            edges.add(edge)
    edges = sorted(edges)

    clauses = graph_coloring_to_sat(n_nodes, n_colors, edges)
    total_vars = n_nodes * n_colors  # Should equal n_vars.

    # Find satisfying assignments by random search.
    data = []
    attempts = 0
    max_attempts = n_samples * 5000
    while len(data) < n_samples and attempts < max_attempts:
        assignment = {i + 1: bool(rng.choice([True, False])) for i in range(total_vars)}
        sat, total = check_assignment(clauses, assignment)
        if sat == total:
            data.append([assignment[i + 1] for i in range(total_vars)])
        attempts += 1

    # Fallback: near-satisfying.
    if len(data) < n_samples:
        while len(data) < n_samples and attempts < max_attempts * 2:
            assignment = {i + 1: bool(rng.choice([True, False])) for i in range(total_vars)}
            sat, total = check_assignment(clauses, assignment)
            if sat >= total - 1:
                data.append([assignment[i + 1] for i in range(total_vars)])
            attempts += 1

    return clauses, np.array(data[:n_samples], dtype=np.float32)


# --- CD training (reused from Exp 50 with minor adaptations) ---

def train_ising_cd(
    data: np.ndarray,
    n_epochs: int = 100,
    lr: float = 0.05,
    beta: float = 2.0,
    cd_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Train an Ising model via Contrastive Divergence on binary data.

    Positive phase computes statistics from the training data (satisfying
    assignments). Negative phase samples from the current model using
    parallel Gibbs. The gradient pushes the model distribution toward the
    data distribution.

    See Exp 50 docstring for full derivation.

    Args:
        data: Training samples, shape (n_samples, n_vars), float {0,1}.
        n_epochs: Number of CD updates.
        lr: Learning rate for parameter updates.
        beta: Inverse temperature (higher = sharper energy landscape).
        cd_steps: Gibbs steps per negative phase sample (CD-k).

    Returns:
        (biases, coupling_matrix).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import ParallelIsingSampler

    n_samples, n_vars = data.shape

    # Initialize parameters to small random values.
    rng = np.random.default_rng(42)
    biases = np.zeros(n_vars, dtype=np.float32)
    J = rng.normal(0, 0.01, (n_vars, n_vars)).astype(np.float32)
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    # Positive phase statistics (from data) -- computed once since data is fixed.
    data_jax = jnp.array(data)
    spins_data = 2.0 * data_jax - 1.0  # Map {0,1} -> {-1,+1}.
    pos_bias_moments = jnp.mean(spins_data, axis=0)
    pos_weight_moments = jnp.mean(
        jnp.einsum("bi,bj->bij", spins_data, spins_data), axis=0
    )

    sampler = ParallelIsingSampler(
        n_warmup=cd_steps * 10,
        n_samples=n_samples,
        steps_per_sample=cd_steps,
        schedule=None,
        use_checkerboard=True,
    )

    for epoch in range(n_epochs):
        key = jrandom.PRNGKey(epoch)

        # Negative phase: sample from current model.
        b_jax = jnp.array(biases)
        J_jax = jnp.array(J)
        model_samples = sampler.sample(key, b_jax, J_jax, beta=beta)

        spins_model = 2.0 * model_samples.astype(jnp.float32) - 1.0
        neg_bias_moments = jnp.mean(spins_model, axis=0)
        neg_weight_moments = jnp.mean(
            jnp.einsum("bi,bj->bij", spins_model, spins_model), axis=0
        )

        # CD gradient update.
        grad_b = -beta * (pos_bias_moments - neg_bias_moments)
        grad_J = -beta * (pos_weight_moments - neg_weight_moments)

        biases -= lr * np.array(grad_b)
        J -= lr * np.array(grad_J)
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)

    return biases, J


# --- Evaluation ---

def evaluate_model_on_domain(
    biases: np.ndarray,
    J: np.ndarray,
    clauses: list,
    n_vars: int,
    beta: float = 2.0,
    n_eval_samples: int = 100,
) -> dict:
    """Generate samples from a trained Ising model and check SAT quality on a given domain.

    This is the core transfer test: take a model trained on domain A and
    evaluate how well its samples satisfy clauses from domain B.

    Args:
        biases: Learned bias vector, shape (n_vars,).
        J: Learned coupling matrix, shape (n_vars, n_vars).
        clauses: SAT clauses to evaluate against.
        n_vars: Number of Boolean variables.
        beta: Inverse temperature for sampling.
        n_eval_samples: Number of samples to generate and evaluate.

    Returns:
        Dict with mean_pct (average SAT%), best_pct (best single sample),
        and n_perfect (samples satisfying all clauses).
    """
    import jax.numpy as jnp
    import jax.random as jrandom
    from carnot.samplers.parallel_ising import ParallelIsingSampler
    from experiment_39_thrml_sat import check_assignment

    sampler = ParallelIsingSampler(
        n_warmup=500,
        n_samples=n_eval_samples,
        steps_per_sample=20,
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(12345),
        jnp.array(biases),
        jnp.array(J),
        beta=beta,
    )

    n_clauses = len(clauses)
    sat_counts = []
    for s_idx in range(samples.shape[0]):
        assignment = {v + 1: bool(samples[s_idx, v]) for v in range(n_vars)}
        sat, total = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    return {
        "mean_pct": float(np.mean(sat_counts)) / n_clauses * 100,
        "best_pct": float(max(sat_counts)) / n_clauses * 100,
        "n_perfect": sum(1 for s in sat_counts if s == n_clauses),
    }


def random_baseline(clauses: list, n_vars: int, n_samples: int = 100, seed: int = 42) -> dict:
    """Compute SAT% for uniformly random assignments (the baseline to beat).

    Args:
        clauses: SAT clauses to evaluate.
        n_vars: Number of Boolean variables.
        n_samples: How many random assignments to try.
        seed: Random seed.

    Returns:
        Dict with mean_pct, best_pct, n_perfect (same format as evaluate_model_on_domain).
    """
    from experiment_39_thrml_sat import check_assignment

    rng = np.random.default_rng(seed)
    n_clauses = len(clauses)
    sat_counts = []
    for _ in range(n_samples):
        assignment = {i + 1: bool(rng.choice([True, False])) for i in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    return {
        "mean_pct": float(np.mean(sat_counts)) / n_clauses * 100,
        "best_pct": float(max(sat_counts)) / n_clauses * 100,
        "n_perfect": sum(1 for s in sat_counts if s == n_clauses),
    }


def main() -> int:
    import jax

    print("=" * 70)
    print("EXPERIMENT 52: Cross-Domain Transfer of Learned Ising Models")
    print("  Q: Do Ising couplings trained on one SAT structure generalize?")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 70)

    start = time.time()

    n_vars = 15
    n_data = 200  # Training samples per domain.
    n_epochs = 100

    # ---- Step 1: Generate three SAT domains ----
    print("\n--- Step 1: Generate three SAT domains (n_vars=15) ---")

    # Domain A: Random 3-SAT at phase transition ratio 4.26.
    # This is the hardest random regime -- few satisfying assignments, so the
    # model must learn tight structural constraints.
    print("  (a) Random 3-SAT, ratio=4.26 (phase transition)...")
    clauses_a, data_a = generate_random_3sat_data(n_vars, ratio=4.26, n_samples=n_data, seed=100)
    print(f"      {len(clauses_a)} clauses, {data_a.shape[0]} training samples found")

    # Domain B: Random 3-SAT at ratio 3.0 (underconstrained).
    # Many satisfying assignments exist -- the model learns a looser distribution.
    print("  (b) Random 3-SAT, ratio=3.0 (underconstrained)...")
    clauses_b, data_b = generate_random_3sat_data(n_vars, ratio=3.0, n_samples=n_data, seed=200)
    print(f"      {len(clauses_b)} clauses, {data_b.shape[0]} training samples found")

    # Domain C: Graph 3-coloring encoded as SAT.
    # Non-random clause structure: at-least-one clauses (positive literals) plus
    # not-both clauses (pairs of negative literals). Very different from random 3-SAT.
    print("  (c) Graph 3-coloring as SAT (structured)...")
    clauses_c, data_c = generate_graph_coloring_data(n_vars, n_samples=n_data, seed=300)
    print(f"      {len(clauses_c)} clauses, {data_c.shape[0]} training samples found")

    domains = [
        ("Phase-trans(4.26)", clauses_a, data_a),
        ("Under-const(3.0)", clauses_b, data_b),
        ("Graph-color", clauses_c, data_c),
    ]

    # Check we have enough data for each domain.
    for name, clauses, data in domains:
        if data.shape[0] < 20:
            print(f"  WARNING: Only {data.shape[0]} samples for {name}. Results may be unreliable.")

    # ---- Step 2: Train one Ising model per domain ----
    print(f"\n--- Step 2: Train Ising models ({n_epochs} epochs CD-1 each) ---")
    models = []
    for name, clauses, data in domains:
        print(f"  Training on {name}...")
        biases, J = train_ising_cd(data, n_epochs=n_epochs, lr=0.05, beta=2.0, cd_steps=1)
        models.append((biases, J))

    # ---- Step 3: Evaluate each model on all domains ----
    print("\n--- Step 3: Evaluate each model on all three domains ---")

    # transfer_matrix[i][j] = result of model_i evaluated on domain_j.
    transfer_matrix = []
    for i, (train_name, _, _) in enumerate(domains):
        row = []
        biases, J = models[i]
        for j, (test_name, test_clauses, _) in enumerate(domains):
            result = evaluate_model_on_domain(biases, J, test_clauses, n_vars, beta=2.0)
            tag = "in-domain" if i == j else "TRANSFER"
            print(f"    Train={train_name} -> Test={test_name}: "
                  f"mean={result['mean_pct']:.1f}% best={result['best_pct']:.1f}% "
                  f"perfect={result['n_perfect']} [{tag}]")
            row.append(result)
        transfer_matrix.append(row)

    # ---- Step 4: Random baselines for each domain ----
    print("\n--- Step 4: Random baselines ---")
    baselines = []
    for name, clauses, _ in domains:
        bl = random_baseline(clauses, n_vars, n_samples=100, seed=999)
        print(f"    Random on {name}: mean={bl['mean_pct']:.1f}% best={bl['best_pct']:.1f}%")
        baselines.append(bl)

    # ---- Step 5: Print transfer matrix ----
    elapsed = time.time() - start
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"EXPERIMENT 52 RESULTS ({elapsed:.0f}s)")
    print(sep)

    # Short names for the matrix header.
    short_names = ["PT(4.26)", "UC(3.0)", "GrColor"]

    # Mean SAT% transfer matrix.
    print("\n  TRANSFER MATRIX (mean SAT%)")
    print(f"  {'Train \\ Test':>18s}", end="")
    for sn in short_names:
        print(f"  {sn:>10s}", end="")
    print(f"  {'Random':>10s}")
    print("  " + "-" * (18 + 11 * (len(short_names) + 1)))

    for i, (train_name, _, _) in enumerate(domains):
        print(f"  {short_names[i]:>18s}", end="")
        for j in range(len(domains)):
            val = transfer_matrix[i][j]["mean_pct"]
            marker = " *" if i == j else "  "
            print(f"  {val:>8.1f}{marker}", end="")
        print(f"  {baselines[i]['mean_pct']:>8.1f}  ")
    print("  (* = in-domain)")

    # Best SAT% transfer matrix.
    print("\n  TRANSFER MATRIX (best SAT%)")
    print(f"  {'Train \\ Test':>18s}", end="")
    for sn in short_names:
        print(f"  {sn:>10s}", end="")
    print(f"  {'Random':>10s}")
    print("  " + "-" * (18 + 11 * (len(short_names) + 1)))

    for i, (train_name, _, _) in enumerate(domains):
        print(f"  {short_names[i]:>18s}", end="")
        for j in range(len(domains)):
            val = transfer_matrix[i][j]["best_pct"]
            marker = " *" if i == j else "  "
            print(f"  {val:>8.1f}{marker}", end="")
        print(f"  {baselines[i]['best_pct']:>8.1f}  ")
    print("  (* = in-domain)")

    # ---- Step 6: Analysis and verdict ----
    print(f"\n--- Analysis ---")

    # For each model, compare in-domain vs cross-domain performance.
    in_domain_scores = []
    cross_domain_scores = []
    for i in range(len(domains)):
        in_domain_scores.append(transfer_matrix[i][i]["mean_pct"])
        for j in range(len(domains)):
            if i != j:
                cross_domain_scores.append(transfer_matrix[i][j]["mean_pct"])

    avg_in_domain = np.mean(in_domain_scores)
    avg_cross_domain = np.mean(cross_domain_scores)
    avg_random = np.mean([b["mean_pct"] for b in baselines])

    print(f"  Avg in-domain SAT%:    {avg_in_domain:.1f}%")
    print(f"  Avg cross-domain SAT%: {avg_cross_domain:.1f}%")
    print(f"  Avg random baseline:   {avg_random:.1f}%")
    print(f"  In-domain advantage:   {avg_in_domain - avg_cross_domain:+.1f}%")
    print(f"  Cross-domain vs random:{avg_cross_domain - avg_random:+.1f}%")

    # Check: does any cross-domain pair beat random?
    cross_beats_random = 0
    cross_total = 0
    for i in range(len(domains)):
        for j in range(len(domains)):
            if i != j:
                cross_total += 1
                if transfer_matrix[i][j]["mean_pct"] > baselines[j]["mean_pct"] + 1.0:
                    cross_beats_random += 1

    in_beats_random = sum(
        1 for i in range(len(domains))
        if transfer_matrix[i][i]["mean_pct"] > baselines[i]["mean_pct"] + 1.0
    )

    print(f"\n  In-domain models beating random (+1%):    {in_beats_random}/{len(domains)}")
    print(f"  Cross-domain models beating random (+1%): {cross_beats_random}/{cross_total}")

    # Verdict.
    print(f"\n{sep}")
    if avg_cross_domain > avg_random + 3.0 and cross_beats_random >= cross_total // 2:
        print("  VERDICT: Ising models TRANSFER across SAT domains!")
        print(f"  Cross-domain ({avg_cross_domain:.1f}%) substantially beats random ({avg_random:.1f}%).")
        print("  Structural constraints encoded in couplings generalize.")
    elif avg_cross_domain > avg_random + 1.0:
        print("  VERDICT: PARTIAL transfer -- cross-domain slightly above random.")
        print(f"  Cross-domain ({avg_cross_domain:.1f}%) vs random ({avg_random:.1f}%).")
        if avg_in_domain > avg_cross_domain + 3.0:
            print("  In-domain is significantly better: couplings are partly instance-specific.")
        else:
            print("  But in-domain advantage is small: structure is shared.")
    else:
        print("  VERDICT: NO transfer -- Ising couplings are domain-specific.")
        print(f"  Cross-domain ({avg_cross_domain:.1f}%) is near random ({avg_random:.1f}%).")
        if avg_in_domain > avg_random + 3.0:
            print("  In-domain works, but learned couplings don't generalize.")
        else:
            print("  Even in-domain is weak -- may need more data or epochs.")
    print(sep)

    return 0


if __name__ == "__main__":
    sys.exit(main())
