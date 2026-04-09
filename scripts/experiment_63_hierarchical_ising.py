#!/usr/bin/env python3
"""Experiment 63: Hierarchical Ising — block-structured coupling for large SAT.

**Why hierarchical?**
    Flat Ising models (Exps 60, 61) treat all N^2 couplings the same way.
    But real SAT instances have STRUCTURE: clauses connect nearby variables
    more often than distant ones (especially in structured/industrial SAT),
    and even random 3-SAT has a sparse clause graph where most variables
    only interact with a small neighborhood.

    Hierarchical Ising exploits this by grouping variables into blocks of
    size B (e.g., 50) and decomposing the coupling matrix into:
    1. **Intra-block couplings**: dense B×B matrices within each block.
       These capture fine-grained correlations among nearby variables.
    2. **Inter-block couplings**: sparse connections between blocks.
       Only variable pairs that share a clause across block boundaries
       get a coupling — the rest are zero.

    This decomposition has several advantages:
    - **Parameter efficiency**: Instead of N^2 params, we have
      n_blocks * B^2 (intra) + sparse_inter_edges (inter). For 1000 vars
      with B=50, that's 20 * 2500 = 50K intra + ~few K inter, vs 500K flat.
    - **Training speed**: Intra-block couplings are small dense matrices
      that fit in cache. Inter-block couplings are sparse and few.
    - **Sampling quality**: The two-level Gibbs sampler (inner: parallel
      within blocks, outer: update inter-block messages) naturally matches
      the hierarchical structure.
    - **Scalability**: Adding more variables just adds more blocks — the
      per-block cost stays constant at O(B^2).

**Sampling strategy:**
    1. Outer loop: compute "effective fields" from inter-block couplings
       (coarse-grained messages between blocks).
    2. Inner loop: run parallel Gibbs WITHIN each block using its dense
       intra-block couplings + the effective fields from step 1.
    3. Anneal beta from low (exploration) to high (exploitation).
    This is analogous to belief propagation in graphical models: blocks
    are "super-nodes" and inter-block edges are "messages."

**Training strategy:**
    1. First train intra-block couplings via CD on each block independently.
       Each block has only B^2 ~ 2500 params, so CD converges fast.
    2. Then train inter-block couplings via CD with L1 regularization.
       Only the sparse cross-block edges are updated; everything else
       is frozen. L1 keeps the inter-block connections sparse.

**Benchmark:**
    Compare hierarchical vs flat-sparse (Exp 61) vs flat-dense (Exp 60)
    vs random baseline on 200/500/1000-variable random 3-SAT instances.
    Report: accuracy (% clauses satisfied), wall time, memory (param count).

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_63_hierarchical_ising.py
"""

from __future__ import annotations

import math
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from experiment_39_thrml_sat import random_3sat, sat_to_ising, check_assignment
from carnot.samplers.parallel_ising import (
    ParallelIsingSampler,
    AnnealingSchedule,
    sat_to_coupling_matrix,
)


# ---------------------------------------------------------------------------
# Data structures for hierarchical block decomposition
# ---------------------------------------------------------------------------


class HierarchicalBlock:
    """One block in the hierarchical decomposition.

    **What this represents:**
        A contiguous group of SAT variables (e.g., variables 0-49 for block 0
        with block_size=50). Each block stores its own dense intra-block
        coupling matrix and a list of sparse inter-block edges connecting
        its variables to variables in OTHER blocks.

    Attributes:
        block_id: Index of this block (0-based).
        var_indices: Global variable indices belonging to this block.
        intra_J: Dense coupling matrix, shape (block_size, block_size).
            intra_J[i, j] is the coupling between var_indices[i] and
            var_indices[j]. Zero diagonal.
        intra_biases: Bias vector for this block, shape (block_size,).
        inter_edges: List of (local_i, global_j, weight) triples.
            local_i is the index within this block (0 to block_size-1).
            global_j is the global variable index in another block.
            weight is the coupling strength (initially from clause graph).
    """

    def __init__(self, block_id: int, var_indices: list[int], n_vars: int):
        self.block_id = block_id
        self.var_indices = var_indices
        self.block_size = len(var_indices)
        # Map from global var index to local index within this block.
        self.global_to_local = {g: l for l, g in enumerate(var_indices)}
        # Dense intra-block couplings (initialized to zero, filled later).
        self.intra_J = np.zeros(
            (self.block_size, self.block_size), dtype=np.float32
        )
        self.intra_biases = np.zeros(self.block_size, dtype=np.float32)
        # Sparse inter-block edges: (local_idx, other_global_idx, weight).
        self.inter_edges: list[tuple[int, int, float]] = []


def hierarchical_encode(
    clauses: list[list[int]],
    n_vars: int,
    block_size: int = 50,
) -> list[HierarchicalBlock]:
    """Decompose SAT clauses into a hierarchical block structure.

    **How the decomposition works:**
        1. Variables are split into contiguous blocks of block_size.
           Block k contains variables [k*block_size, (k+1)*block_size).
        2. For each clause, extract the variable pairs. If both variables
           are in the same block, add the coupling to intra_J. If they're
           in different blocks, add to inter_edges of both blocks.
        3. Biases from sat_to_ising are distributed to each block.

    **Why contiguous blocks?**
        For random 3-SAT, the variable ordering is arbitrary, so contiguous
        blocks are as good as any partition. For structured SAT (e.g., from
        circuit verification), variables with nearby indices often appear in
        the same clauses, making contiguous blocks a natural choice. A future
        experiment could try graph-based partitioning (e.g., METIS) to
        minimize cross-block edges.

    Args:
        clauses: SAT clauses from random_3sat.
        n_vars: Number of variables.
        block_size: Number of variables per block.

    Returns:
        List of HierarchicalBlock objects, one per block.
    """
    n_blocks = math.ceil(n_vars / block_size)

    # Create blocks with contiguous variable ranges.
    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = min(start + block_size, n_vars)
        var_indices = list(range(start, end))
        blocks.append(HierarchicalBlock(b, var_indices, n_vars))

    # Helper: which block does a global variable belong to?
    def var_to_block(v: int) -> int:
        return v // block_size

    # Get biases from the hand-coded Ising mapping and distribute to blocks.
    biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)
    biases_np = np.array(biases_vec, dtype=np.float32)
    for b_obj in blocks:
        for local_i, global_i in enumerate(b_obj.var_indices):
            b_obj.intra_biases[local_i] = biases_np[global_i]

    # Process each clause: classify variable pairs as intra- or inter-block.
    for clause in clauses:
        # Extract 0-based variable indices.
        var_indices = [abs(lit) - 1 for lit in clause]
        # Determine coupling signs from literal polarities. If both literals
        # in a pair have the same sign (both positive or both negative), the
        # coupling is positive (ferromagnetic, encourage agreement). If
        # opposite signs, the coupling is negative (antiferromagnetic).
        signs = [1.0 if lit > 0 else -1.0 for lit in clause]

        # Add pairwise couplings for all variable pairs in this clause.
        for a in range(len(var_indices)):
            for b_idx in range(a + 1, len(var_indices)):
                vi, vj = var_indices[a], var_indices[b_idx]
                if vi == vj:
                    continue
                # Coupling weight: same formula as sat_to_ising.
                weight = (1.0 / 3.0) * signs[a] * signs[b_idx]
                bi, bj = var_to_block(vi), var_to_block(vj)

                if bi == bj:
                    # Same block: add to dense intra-block coupling.
                    block = blocks[bi]
                    li = block.global_to_local[vi]
                    lj = block.global_to_local[vj]
                    block.intra_J[li, lj] += weight
                    block.intra_J[lj, li] += weight
                else:
                    # Different blocks: add to sparse inter-block edges.
                    block_i = blocks[bi]
                    block_j = blocks[bj]
                    li = block_i.global_to_local[vi]
                    lj = block_j.global_to_local[vj]
                    block_i.inter_edges.append((li, vj, weight))
                    block_j.inter_edges.append((lj, vi, weight))

    # Report decomposition statistics.
    total_intra_params = sum(
        b.block_size * (b.block_size - 1) // 2 for b in blocks
    )
    total_inter_edges = sum(len(b.inter_edges) for b in blocks) // 2
    total_flat_params = n_vars * (n_vars - 1) // 2
    print(
        f"    Hierarchical: {n_blocks} blocks of ~{block_size}, "
        f"intra={total_intra_params} params, "
        f"inter={total_inter_edges} edges, "
        f"total={total_intra_params + total_inter_edges} "
        f"(vs flat {total_flat_params}, "
        f"{(total_intra_params + total_inter_edges) / total_flat_params * 100:.1f}%)"
    )

    return blocks


def hierarchical_sample(
    blocks: list[HierarchicalBlock],
    n_vars: int,
    n_samples: int = 200,
    n_outer_steps: int = 50,
    n_inner_steps: int = 10,
    beta_init: float = 0.1,
    beta_final: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Two-level hierarchical Gibbs sampler.

    **How it works, step by step:**
        1. Initialize all spins randomly.
        2. For each outer step (coarse grain):
           a. Compute the current beta using linear annealing.
           b. For each block, compute "effective fields" from inter-block
              couplings: for each variable in the block, sum up
              weight * spin_value for all inter-block neighbors.
           c. Add these effective fields to the block's intra-block biases
              to get the total local field for each spin.
           d. Run n_inner_steps of parallel Gibbs within the block using
              the combined biases and intra-block couplings.
        3. After annealing, collect n_samples by repeating the inner/outer
           loop at the final temperature.

    **Why two levels?**
        In a flat sampler, each sweep costs O(N^2) for the matrix-vector
        product J @ s. In the hierarchical sampler, each inner sweep costs
        O(B^2) per block = O(N * B) total (since there are N/B blocks).
        The outer step costs O(inter_edges) which is sparse. So total cost
        per sweep is O(N * B + inter_edges) vs O(N^2) flat.

    Args:
        blocks: Hierarchical block decomposition from hierarchical_encode.
        n_vars: Total number of variables.
        n_samples: Number of samples to collect.
        n_outer_steps: Number of outer (annealing) steps.
        n_inner_steps: Number of inner Gibbs steps per block per outer step.
        beta_init: Starting inverse temperature.
        beta_final: Final inverse temperature.
        seed: Random seed.

    Returns:
        Array of shape (n_samples, n_vars), float32 in {0, 1}.
    """
    rng = np.random.default_rng(seed)

    # Initialize global spin state randomly.
    spins = rng.choice([0.0, 1.0], size=n_vars).astype(np.float32)

    def _run_sweeps(spins: np.ndarray, n_steps: int, beta: float) -> np.ndarray:
        """Run n_steps of hierarchical sweeps at fixed beta.

        Each sweep:
          1. Compute inter-block effective fields for each block.
          2. Run parallel Gibbs within each block.
        """
        for _ in range(n_steps):
            for block in blocks:
                bs = block.block_size
                # Compute effective field from inter-block neighbors.
                # This is the "message" from other blocks: for each spin in
                # this block, sum weight * neighbor_spin for all inter-block
                # connections.
                effective_bias = block.intra_biases.copy()
                for local_i, global_j, weight in block.inter_edges:
                    effective_bias[local_i] += weight * spins[global_j]

                # Extract current block spins.
                block_spins = spins[block.var_indices].copy()

                # Run parallel Gibbs within the block.
                # For each spin i in the block:
                #   h_i = effective_bias[i] + sum_j intra_J[i,j] * s_j
                #   P(s_i=1) = sigmoid(2 * beta * h_i)
                h = effective_bias + block.intra_J @ block_spins
                probs = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
                new_spins = (rng.random(bs) < probs).astype(np.float32)

                # Write updated spins back to global state.
                for local_i, global_i in enumerate(block.var_indices):
                    spins[global_i] = new_spins[local_i]
        return spins

    # --- Phase 1: Annealing warmup ---
    for step in range(n_outer_steps):
        frac = step / max(n_outer_steps - 1, 1)
        beta = beta_init + frac * (beta_final - beta_init)
        spins = _run_sweeps(spins, n_inner_steps, beta)

    # --- Phase 2: Collect samples at final temperature ---
    samples = []
    for _ in range(n_samples):
        spins = _run_sweeps(spins, n_inner_steps, beta_final)
        samples.append(spins.copy())

    return np.array(samples, dtype=np.float32)


def hierarchical_cd_train(
    blocks: list[HierarchicalBlock],
    n_vars: int,
    data: np.ndarray,
    n_epochs_intra: int = 100,
    n_epochs_inter: int = 50,
    lr: float = 0.01,
    beta: float = 2.0,
    l1_lambda_inter: float = 0.01,
) -> list[HierarchicalBlock]:
    """Two-phase CD training for hierarchical Ising model.

    **Phase 1: Intra-block training.**
        For each block independently, run CD using only the block's variables.
        This is fast because each block has only B^2 parameters and B training
        dimensions. We compute positive-phase statistics from the data
        (restricted to this block's variables) and negative-phase statistics
        from parallel Gibbs within the block.

    **Phase 2: Inter-block training.**
        Freeze intra-block couplings. For each inter-block edge, compute the
        CD gradient using the global data and hierarchical sampler. Apply L1
        regularization to keep inter-block connections sparse. Only edges
        that already exist in the clause graph are updated (hard sparsity
        from the initial decomposition).

    **Why train in two phases?**
        Intra-block couplings are dense and numerous (B^2 per block) but
        operate on small matrices. Training them first gives each block a
        good local model. Inter-block couplings are sparse and few, but
        operate on the full global state. Training them second, with intra
        frozen, avoids the instability of optimizing all parameters at once.
        This is analogous to the "pre-training then fine-tuning" strategy
        in deep learning.

    Args:
        blocks: Hierarchical block decomposition.
        n_vars: Total number of variables.
        data: Training data, shape (n_samples, n_vars), float {0,1}.
        n_epochs_intra: Epochs for intra-block CD.
        n_epochs_inter: Epochs for inter-block CD.
        lr: Learning rate.
        beta: Inverse temperature for CD sampling.
        l1_lambda_inter: L1 regularization strength on inter-block couplings.

    Returns:
        The blocks list with updated intra_J and inter_edges.
    """
    n_samples = data.shape[0]
    rng = np.random.default_rng(42)

    # =========================
    # Phase 1: Intra-block CD
    # =========================
    print(f"    Phase 1: Intra-block CD ({n_epochs_intra} epochs per block)...")
    for block in blocks:
        bs = block.block_size
        # Extract this block's data columns.
        block_data = data[:, block.var_indices]  # (n_samples, block_size)
        # Map {0,1} -> {-1,+1} for Ising convention.
        spins_data = 2.0 * block_data - 1.0

        # Positive-phase statistics (from data).
        pos_bias = np.mean(spins_data, axis=0)
        pos_corr = np.mean(
            np.einsum("bi,bj->bij", spins_data, spins_data), axis=0
        )

        # Initialize intra_J with small random values (keep existing from
        # clause structure as a warm start).
        block.intra_J += rng.normal(0, 0.001, (bs, bs)).astype(np.float32)
        block.intra_J = (block.intra_J + block.intra_J.T) / 2.0
        np.fill_diagonal(block.intra_J, 0.0)

        for epoch in range(n_epochs_intra):
            # Negative phase: Gibbs sample within this block.
            # Start from random state each epoch (CD-k with fresh init).
            model_spins = rng.choice(
                [-1.0, 1.0], size=(min(n_samples, 500), bs)
            ).astype(np.float32)
            for _ in range(5):  # 5 Gibbs steps.
                h = block.intra_biases + model_spins @ block.intra_J.T
                probs = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
                model_spins = (
                    2.0 * (rng.random(model_spins.shape) < probs).astype(
                        np.float32
                    )
                    - 1.0
                )

            neg_bias = np.mean(model_spins, axis=0)
            neg_corr = np.mean(
                np.einsum("bi,bj->bij", model_spins, model_spins), axis=0
            )

            # CD gradient.
            grad_J = -beta * (pos_corr - neg_corr)
            grad_b = -beta * (pos_bias - neg_bias)

            # Update.
            block.intra_J -= lr * grad_J
            block.intra_biases -= lr * grad_b
            block.intra_J = (block.intra_J + block.intra_J.T) / 2.0
            np.fill_diagonal(block.intra_J, 0.0)

        if block.block_id % 5 == 0 or block.block_id == len(blocks) - 1:
            recon = float(np.mean((pos_bias - neg_bias) ** 2))
            print(
                f"      Block {block.block_id}: final recon={recon:.6f}"
            )

    # =========================
    # Phase 2: Inter-block CD
    # =========================
    print(
        f"    Phase 2: Inter-block CD ({n_epochs_inter} epochs, "
        f"L1={l1_lambda_inter})..."
    )
    # Map {0,1} -> {-1,+1} for correlation computation.
    spins_data = 2.0 * data - 1.0

    # Compute positive-phase pairwise correlations for inter-block edges.
    # We only need correlations for variable pairs that have inter-block edges.
    inter_edge_set: set[tuple[int, int]] = set()
    for block in blocks:
        for local_i, global_j, _ in block.inter_edges:
            global_i = block.var_indices[local_i]
            key = (min(global_i, global_j), max(global_i, global_j))
            inter_edge_set.add(key)

    # Precompute positive correlations for these edges.
    pos_corr_inter: dict[tuple[int, int], float] = {}
    for gi, gj in inter_edge_set:
        pos_corr_inter[(gi, gj)] = float(
            np.mean(spins_data[:, gi] * spins_data[:, gj])
        )

    for epoch in range(n_epochs_inter):
        # Negative phase: sample from current hierarchical model.
        # Use a small batch for efficiency.
        neg_samples_list = []
        for _ in range(min(n_samples, 200)):
            # Start from a random data point (persistent CD).
            idx = rng.integers(n_samples)
            state = data[idx].copy()
            # Run a few hierarchical sweeps.
            for block in blocks:
                block_spins = state[block.var_indices]
                effective_bias = block.intra_biases.copy()
                for local_i, global_j, weight in block.inter_edges:
                    effective_bias[local_i] += weight * state[global_j]
                h = effective_bias + block.intra_J @ block_spins
                probs = 1.0 / (1.0 + np.exp(-2.0 * beta * h))
                new_spins = (rng.random(block.block_size) < probs).astype(
                    np.float32
                )
                for local_i, global_i in enumerate(block.var_indices):
                    state[global_i] = new_spins[local_i]
            neg_samples_list.append(state)

        neg_data = np.array(neg_samples_list, dtype=np.float32)
        neg_spins = 2.0 * neg_data - 1.0

        # Compute negative correlations for inter-block edges.
        neg_corr_inter: dict[tuple[int, int], float] = {}
        for gi, gj in inter_edge_set:
            neg_corr_inter[(gi, gj)] = float(
                np.mean(neg_spins[:, gi] * neg_spins[:, gj])
            )

        # Update inter-block edge weights.
        for block in blocks:
            for e_idx in range(len(block.inter_edges)):
                local_i, global_j, weight = block.inter_edges[e_idx]
                global_i = block.var_indices[local_i]
                key = (min(global_i, global_j), max(global_i, global_j))
                pos_c = pos_corr_inter[key]
                neg_c = neg_corr_inter[key]
                grad = -beta * (pos_c - neg_c)
                # L1 regularization: push small weights toward zero.
                l1_grad = l1_lambda_inter * np.sign(weight)
                new_weight = weight - lr * (grad + l1_grad)
                block.inter_edges[e_idx] = (local_i, global_j, new_weight)

        if epoch % 10 == 0 or epoch == n_epochs_inter - 1:
            # Report average absolute inter-block weight.
            all_weights = [
                abs(w)
                for block in blocks
                for _, _, w in block.inter_edges
            ]
            mean_w = np.mean(all_weights) if all_weights else 0.0
            print(f"      Epoch {epoch}: mean |inter_weight|={mean_w:.4f}")

    return blocks


def blocks_to_flat(
    blocks: list[HierarchicalBlock],
    n_vars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a flat (biases, J) from hierarchical blocks.

    **Why reconstruct?**
        So we can use the same evaluate_method() / sampler as the flat
        baselines for a fair apples-to-apples comparison.

    Args:
        blocks: Hierarchical blocks with trained couplings.
        n_vars: Total number of variables.

    Returns:
        (biases, J) where biases has shape (n_vars,) and J has shape
        (n_vars, n_vars), symmetric with zero diagonal.
    """
    biases = np.zeros(n_vars, dtype=np.float32)
    J = np.zeros((n_vars, n_vars), dtype=np.float32)

    for block in blocks:
        # Copy intra-block couplings.
        for li, gi in enumerate(block.var_indices):
            biases[gi] = block.intra_biases[li]
            for lj, gj in enumerate(block.var_indices):
                J[gi, gj] = block.intra_J[li, lj]

        # Copy inter-block couplings.
        for local_i, global_j, weight in block.inter_edges:
            global_i = block.var_indices[local_i]
            J[global_i, global_j] = weight

    # Ensure symmetry and zero diagonal.
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)

    return biases, J


# ---------------------------------------------------------------------------
# Evaluation helpers (reused from Exp 61 pattern)
# ---------------------------------------------------------------------------


def generate_training_data(
    clauses: list[list[int]],
    n_vars: int,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate near-satisfying assignments via hand-coded Ising + parallel sampler.

    Same bootstrap strategy as Exps 60/61: use the hand-coded SAT-to-Ising
    mapping to generate high-quality training data, then train a hierarchical
    model that may sample better than the hand-coded model.

    Args:
        clauses: SAT clauses.
        n_vars: Number of variables.
        n_samples: Desired number of training samples.
        seed: Random seed.

    Returns:
        Array of shape (n_collected, n_vars), float32 in {0, 1}.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    biases_vec, weights_vec, _ = sat_to_ising(clauses, n_vars)
    biases, J_mat = sat_to_coupling_matrix(biases_vec, weights_vec, n_vars)

    oversample_factor = 2
    sampler = ParallelIsingSampler(
        n_warmup=min(500, n_vars * 5),
        n_samples=min(n_samples * oversample_factor, 4000),
        steps_per_sample=max(10, min(n_vars // 10, 50)),
        schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
        use_checkerboard=True,
    )

    key = jrandom.PRNGKey(seed)
    samples = sampler.sample(key, biases, J_mat, beta=10.0)
    samples_np = np.asarray(samples)

    n_clauses = len(clauses)
    scored = []
    for i in range(samples_np.shape[0]):
        assignment = {v + 1: bool(samples_np[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        scored.append((sat, i))

    scored.sort(key=lambda x: -x[0])
    top_indices = [idx for _, idx in scored[:n_samples]]
    data = samples_np[top_indices].astype(np.float32)

    top_sats = [s for s, _ in scored[:n_samples]]
    mean_pct = np.mean(top_sats) / n_clauses * 100
    best_pct = max(top_sats) / n_clauses * 100
    print(
        f"    Training data: {len(data)} samples, "
        f"mean {mean_pct:.1f}% sat, best {best_pct:.1f}% sat"
    )

    return data


def evaluate_model(
    biases: np.ndarray,
    J: np.ndarray,
    clauses: list[list[int]],
    n_vars: int,
    beta: float = 10.0,
    n_eval_samples: int = 200,
    label: str = "",
) -> dict:
    """Evaluate an Ising model by sampling and checking SAT satisfaction.

    Draws samples from the flat (biases, J) model at the given inverse
    temperature, checks what fraction of clauses each sample satisfies.

    Args:
        biases: Bias vector, shape (n_vars,).
        J: Coupling matrix, shape (n_vars, n_vars).
        clauses: SAT clauses.
        n_vars: Number of variables.
        beta: Inverse temperature.
        n_eval_samples: Number of samples.
        label: Display name.

    Returns:
        Dict with best_pct, mean_pct, n_perfect, best_sat, n_clauses.
    """
    import jax.numpy as jnp
    import jax.random as jrandom

    sampler = ParallelIsingSampler(
        n_warmup=min(500, n_vars * 5),
        n_samples=min(n_eval_samples, 200),
        steps_per_sample=max(10, min(n_vars // 10, 50)),
        schedule=AnnealingSchedule(beta_init=0.5, beta_final=beta),
        use_checkerboard=True,
    )

    samples = sampler.sample(
        jrandom.PRNGKey(12345),
        jnp.array(biases, dtype=jnp.float32),
        jnp.array(J, dtype=jnp.float32),
        beta=beta,
    )
    samples_np = np.asarray(samples)

    n_clauses = len(clauses)
    sat_counts = []
    for i in range(samples_np.shape[0]):
        assignment = {v + 1: bool(samples_np[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100
    n_perfect = sum(1 for s in sat_counts if s == n_clauses)

    if label:
        print(
            f"    {label:20s}: mean={mean_pct:.1f}%, "
            f"best={best}/{n_clauses} ({best_pct:.1f}%), "
            f"perfect={n_perfect}/{n_eval_samples}"
        )

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": n_perfect,
    }


def evaluate_hierarchical_sampler(
    blocks: list[HierarchicalBlock],
    n_vars: int,
    clauses: list[list[int]],
    n_eval_samples: int = 200,
    label: str = "",
) -> dict:
    """Evaluate the hierarchical sampler directly (not via flat reconstruction).

    This tests the hierarchical sampling strategy itself, which is the
    main contribution of this experiment. The hierarchical sampler may
    produce different (potentially better) samples than the flat sampler
    using the same couplings, because the two-level sweep order naturally
    respects the block structure.

    Args:
        blocks: Trained hierarchical blocks.
        n_vars: Total number of variables.
        clauses: SAT clauses.
        n_eval_samples: Number of samples.
        label: Display name.

    Returns:
        Dict with best_pct, mean_pct, n_perfect.
    """
    samples = hierarchical_sample(
        blocks,
        n_vars,
        n_samples=n_eval_samples,
        n_outer_steps=100,
        n_inner_steps=10,
        beta_init=0.5,
        beta_final=10.0,
        seed=12345,
    )

    n_clauses = len(clauses)
    sat_counts = []
    for i in range(samples.shape[0]):
        assignment = {v + 1: bool(samples[i, v]) for v in range(n_vars)}
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100
    n_perfect = sum(1 for s in sat_counts if s == n_clauses)

    if label:
        print(
            f"    {label:20s}: mean={mean_pct:.1f}%, "
            f"best={best}/{n_clauses} ({best_pct:.1f}%), "
            f"perfect={n_perfect}/{n_eval_samples}"
        )

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": n_perfect,
    }


def random_baseline(
    clauses: list[list[int]],
    n_vars: int,
    n_tries: int = 500,
) -> dict:
    """Evaluate random assignment baseline.

    For 3-SAT, random assignments satisfy ~7/8 of clauses (~87.5%).
    """
    rng_base = np.random.default_rng(42)
    n_clauses = len(clauses)
    sat_counts = []
    for _ in range(n_tries):
        assignment = {
            i + 1: bool(rng_base.choice([True, False])) for i in range(n_vars)
        }
        sat, _ = check_assignment(clauses, assignment)
        sat_counts.append(sat)

    best = max(sat_counts)
    mean_pct = np.mean(sat_counts) / n_clauses * 100
    best_pct = best / n_clauses * 100

    print(
        f"    {'Random':20s}: mean={mean_pct:.1f}%, "
        f"best={best}/{n_clauses} ({best_pct:.1f}%)"
    )

    return {
        "best_sat": best,
        "n_clauses": n_clauses,
        "best_pct": best_pct,
        "mean_pct": mean_pct,
        "n_perfect": 0,
    }


def build_sparse_mask_and_train(
    clauses: list[list[int]],
    n_vars: int,
    data: np.ndarray,
    lr: float,
    n_epochs: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train flat sparse Ising (Exp 61 style) and return (biases, J, time).

    Replicates Exp 61's approach: build a clause-graph edge mask, then
    run CD with the mask applied to gradients so only clause-graph edges
    are learned.

    Args:
        clauses: SAT clauses.
        n_vars: Number of variables.
        data: Training data.
        lr: Learning rate.
        n_epochs: Number of training epochs.

    Returns:
        Tuple of (biases, J, training_time_seconds).
    """
    from experiment_61_sparse_ising import (
        build_clause_edge_mask,
        train_ising_cd_sparse,
    )

    mask = build_clause_edge_mask(clauses, n_vars)
    t0 = time.time()
    biases, J, _ = train_ising_cd_sparse(
        data,
        edge_mask=mask,
        n_epochs=n_epochs,
        lr=lr,
        beta=2.0,
        cd_steps=1,
        l1_lambda=0.001,
    )
    elapsed = time.time() - t0
    return biases, J, elapsed


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def main() -> int:
    import jax

    print("=" * 72)
    print("EXPERIMENT 63: Hierarchical Ising — Block-Structured Coupling")
    print("  Group spins into blocks, learn intra-block (dense) + inter-block")
    print("  (sparse) couplings separately. Two-level Gibbs sampler.")
    print(f"  JAX backend: {jax.default_backend()}")
    print("=" * 72)

    start = time.time()
    block_size = 50

    # Problem sizes: 200 (4 blocks), 500 (10 blocks), 1000 (20 blocks).
    # alpha=3.5 (below phase transition) so satisfying assignments exist.
    test_cases = [
        (200, int(200 * 3.5)),
        (500, int(500 * 3.5)),
        (1000, int(1000 * 3.5)),
    ]

    all_results = []

    for n_vars, n_clauses in test_cases:
        n_blocks = math.ceil(n_vars / block_size)
        n_flat_params = n_vars * (n_vars - 1) // 2
        sep = "-" * 72
        print(f"\n{sep}")
        print(
            f"  {n_vars} vars, {n_clauses} clauses, {n_blocks} blocks of "
            f"{block_size} (flat: {n_flat_params} params)"
        )
        print(sep)

        clauses = random_3sat(n_vars, n_clauses, seed=42)

        # --- Step 1: Generate training data ---
        n_train_samples = min(2000, max(500, n_vars * 2))
        print(f"\n  [1/6] Generating training data ({n_train_samples} samples)...")
        t0 = time.time()
        data = generate_training_data(clauses, n_vars, n_train_samples, seed=42)
        data_time = time.time() - t0
        print(f"    Data generation: {data_time:.1f}s")

        if data.shape[0] < 100:
            print(f"    WARNING: Only {data.shape[0]} samples. Skipping.")
            continue

        # Scale learning rate for problem size.
        lr = 0.05 / (n_vars / 50)
        n_epochs = max(100, 200 - n_vars // 10)

        # --- Step 2: Hierarchical encode + train ---
        print(f"\n  [2/6] Hierarchical encode + CD train...")
        t0 = time.time()
        blocks = hierarchical_encode(clauses, n_vars, block_size=block_size)
        # Scale epochs: intra is fast (per-block), inter needs fewer.
        intra_epochs = max(50, n_epochs // 2)
        inter_epochs = max(30, n_epochs // 4)
        blocks = hierarchical_cd_train(
            blocks,
            n_vars,
            data,
            n_epochs_intra=intra_epochs,
            n_epochs_inter=inter_epochs,
            lr=lr,
            beta=2.0,
            l1_lambda_inter=0.01,
        )
        hier_time = time.time() - t0
        print(f"    Hierarchical training: {hier_time:.1f}s")

        # Reconstruct flat model for comparison.
        biases_hier, J_hier = blocks_to_flat(blocks, n_vars)
        hier_n_params = sum(
            b.block_size * (b.block_size - 1) // 2 for b in blocks
        ) + sum(len(b.inter_edges) for b in blocks) // 2

        # --- Step 3: Flat sparse CD (Exp 61 style) ---
        print(f"\n  [3/6] Flat sparse CD (Exp 61 style, {n_epochs} epochs)...")
        biases_sparse, J_sparse, sparse_time = build_sparse_mask_and_train(
            clauses, n_vars, data, lr=lr, n_epochs=n_epochs
        )
        print(f"    Flat sparse training: {sparse_time:.1f}s")

        # --- Step 4: Flat dense CD ---
        print(f"\n  [4/6] Flat dense CD ({max(50, n_epochs // 2)} epochs)...")
        from experiment_61_sparse_ising import train_ising_cd_sparse

        dense_epochs = max(50, n_epochs // 2)
        t0 = time.time()
        biases_dense, J_dense, _ = train_ising_cd_sparse(
            data,
            edge_mask=None,
            n_epochs=dense_epochs,
            lr=lr,
            beta=2.0,
            cd_steps=1,
            l1_lambda=0.001,
        )
        dense_time = time.time() - t0
        print(f"    Flat dense training: {dense_time:.1f}s")

        # --- Step 5: Hand-coded Ising (oracle) ---
        biases_hc_vec, weights_hc_vec, _ = sat_to_ising(clauses, n_vars)
        biases_hc, J_hc = sat_to_coupling_matrix(
            biases_hc_vec, weights_hc_vec, n_vars
        )
        biases_hc_np = np.array(biases_hc)
        J_hc_np = np.array(J_hc)

        # --- Step 5: Evaluate all methods ---
        print(f"\n  [5/6] Evaluating all methods...")

        # Hierarchical sampler (native two-level).
        res_hier_native = evaluate_hierarchical_sampler(
            blocks, n_vars, clauses, n_eval_samples=200,
            label="Hier (native)",
        )

        # Hierarchical couplings with flat sampler.
        res_hier_flat = evaluate_model(
            biases_hier, J_hier, clauses, n_vars,
            label="Hier (flat eval)",
        )

        # Flat sparse.
        res_sparse = evaluate_model(
            biases_sparse, J_sparse, clauses, n_vars,
            label="Flat sparse",
        )

        # Flat dense.
        res_dense = evaluate_model(
            biases_dense, J_dense, clauses, n_vars,
            label="Flat dense",
        )

        # Hand-coded.
        res_hc = evaluate_model(
            biases_hc_np, J_hc_np, clauses, n_vars,
            label="Hand-coded",
        )

        # Random.
        res_random = random_baseline(clauses, n_vars)

        # --- Step 6: Memory measurement ---
        # Approximate memory as parameter count * 4 bytes (float32).
        hier_mem_bytes = hier_n_params * 4
        sparse_n_params = int(np.count_nonzero(J_sparse)) // 2
        sparse_mem_bytes = sparse_n_params * 4
        dense_mem_bytes = n_flat_params * 4

        all_results.append({
            "n_vars": n_vars,
            "n_clauses": n_clauses,
            "n_blocks": n_blocks,
            "n_flat_params": n_flat_params,
            "hier_n_params": hier_n_params,
            "sparse_n_params": sparse_n_params,
            "data_time": data_time,
            "hier_time": hier_time,
            "sparse_time": sparse_time,
            "dense_time": dense_time,
            "hier_mem_kb": hier_mem_bytes / 1024,
            "sparse_mem_kb": sparse_mem_bytes / 1024,
            "dense_mem_kb": dense_mem_bytes / 1024,
            "hier_native": res_hier_native,
            "hier_flat": res_hier_flat,
            "sparse": res_sparse,
            "dense": res_dense,
            "handcoded": res_hc,
            "random": res_random,
        })

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.time() - start
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"EXPERIMENT 63 RESULTS ({elapsed:.0f}s)")
    print(sep)

    # --- Scaling table: vars × method × (accuracy, time, memory) ---
    print(f"\n  SCALING TABLE: Mean % Clauses Satisfied")
    print(
        f"  {'Vars':>6s} {'Hier(N)':>9s} {'Hier(F)':>9s} {'Sparse':>9s} "
        f"{'Dense':>9s} {'HC':>9s} {'Random':>9s}"
    )
    print(f"  {'-' * 60}")
    for r in all_results:
        print(
            f"  {r['n_vars']:>6d} "
            f"{r['hier_native']['mean_pct']:>8.1f}% "
            f"{r['hier_flat']['mean_pct']:>8.1f}% "
            f"{r['sparse']['mean_pct']:>8.1f}% "
            f"{r['dense']['mean_pct']:>8.1f}% "
            f"{r['handcoded']['mean_pct']:>8.1f}% "
            f"{r['random']['mean_pct']:>8.1f}%"
        )

    print(f"\n  SCALING TABLE: Best % Clauses Satisfied")
    print(
        f"  {'Vars':>6s} {'Hier(N)':>9s} {'Hier(F)':>9s} {'Sparse':>9s} "
        f"{'Dense':>9s} {'HC':>9s}"
    )
    print(f"  {'-' * 52}")
    for r in all_results:
        print(
            f"  {r['n_vars']:>6d} "
            f"{r['hier_native']['best_pct']:>8.1f}% "
            f"{r['hier_flat']['best_pct']:>8.1f}% "
            f"{r['sparse']['best_pct']:>8.1f}% "
            f"{r['dense']['best_pct']:>8.1f}% "
            f"{r['handcoded']['best_pct']:>8.1f}%"
        )

    print(f"\n  SCALING TABLE: Training Time (seconds)")
    print(
        f"  {'Vars':>6s} {'Data':>8s} {'Hier':>8s} {'Sparse':>8s} {'Dense':>8s}"
    )
    print(f"  {'-' * 38}")
    for r in all_results:
        print(
            f"  {r['n_vars']:>6d} "
            f"{r['data_time']:>7.1f}s "
            f"{r['hier_time']:>7.1f}s "
            f"{r['sparse_time']:>7.1f}s "
            f"{r['dense_time']:>7.1f}s"
        )

    print(f"\n  SCALING TABLE: Memory (Parameter Count + KB)")
    print(
        f"  {'Vars':>6s} {'Hier':>12s} {'Sparse':>12s} {'Dense':>12s}"
    )
    print(f"  {'-' * 44}")
    for r in all_results:
        print(
            f"  {r['n_vars']:>6d} "
            f"{r['hier_n_params']:>6d} ({r['hier_mem_kb']:>5.1f}K) "
            f"{r['sparse_n_params']:>6d} ({r['sparse_mem_kb']:>5.1f}K) "
            f"{r['n_flat_params']:>6d} ({r['dense_mem_kb']:>5.1f}K)"
        )

    # --- Verdicts ---
    if all_results:
        print(f"\n  ANALYSIS:")

        # Hierarchical vs flat sparse.
        for r in all_results:
            hier_over_sparse = (
                r["hier_native"]["mean_pct"] - r["sparse"]["mean_pct"]
            )
            hier_over_random = (
                r["hier_native"]["mean_pct"] - r["random"]["mean_pct"]
            )
            speedup = (
                r["sparse_time"] / r["hier_time"]
                if r["hier_time"] > 0
                else 0
            )
            print(
                f"    {r['n_vars']} vars: "
                f"hier vs sparse: {hier_over_sparse:+.1f}%, "
                f"hier vs random: {hier_over_random:+.1f}%, "
                f"training speedup: {speedup:.1f}x"
            )

        # Overall verdict.
        mean_hier_over_sparse = np.mean([
            r["hier_native"]["mean_pct"] - r["sparse"]["mean_pct"]
            for r in all_results
        ])
        mean_hier_over_random = np.mean([
            r["hier_native"]["mean_pct"] - r["random"]["mean_pct"]
            for r in all_results
        ])

        if mean_hier_over_sparse > 2:
            print(
                f"\n  HIERARCHY: Hierarchical outperforms flat sparse "
                f"by {mean_hier_over_sparse:+.1f}% mean"
            )
        elif mean_hier_over_sparse > -1:
            print(
                f"\n  HIERARCHY: Hierarchical matches flat sparse "
                f"({mean_hier_over_sparse:+.1f}%) with fewer params"
            )
        else:
            print(
                f"\n  HIERARCHY: Flat sparse outperforms hierarchical "
                f"by {-mean_hier_over_sparse:.1f}%"
            )

        if mean_hier_over_random > 5:
            print(
                f"  VERDICT: Hierarchical Ising produces useful couplings "
                f"({mean_hier_over_random:+.1f}% over random)"
            )
        elif mean_hier_over_random > 1:
            print(
                f"  VERDICT: Modest improvement from hierarchical "
                f"({mean_hier_over_random:+.1f}% over random)"
            )
        else:
            print(
                f"  VERDICT: Hierarchical does not significantly beat random "
                f"({mean_hier_over_random:+.1f}%)"
            )

        # Scaling trend.
        hier_advantages = [
            r["hier_native"]["mean_pct"] - r["sparse"]["mean_pct"]
            for r in all_results
        ]
        if len(hier_advantages) >= 2:
            trend = hier_advantages[-1] - hier_advantages[0]
            if trend > 2:
                print(
                    f"  SCALING: Hierarchical advantage GROWS with size "
                    f"({trend:+.1f}%)"
                )
            elif trend < -2:
                print(
                    f"  SCALING: Hierarchical advantage SHRINKS with size "
                    f"({trend:+.1f}%)"
                )
            else:
                print(
                    f"  SCALING: Hierarchical advantage stable "
                    f"({trend:+.1f}%)"
                )

    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
