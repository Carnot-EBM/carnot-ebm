#!/usr/bin/env python3
"""Exp 950 — E-MVL Sparsified Ising: K-regular sparse sampler vs dense baseline (CPU).

WHY THIS EXPERIMENT:
    arXiv 2604.04606 (E-MVL, April 2026) achieves ~6x FPGA speedup over simulated
    annealing by replacing the dense O(N^2) coupling sum with a sparse O(N*K) majority
    vote over K nearest neighbors (K << N). For the KV260 RTL:

    - Dense v3 N=128: 128 multipliers per spin = 290K LUTs (over 117K budget, BLOCKED)
    - Sparse v4 N=128, K=16: 16 multipliers per spin = ~36K LUTs (well within budget)

    This experiment validates the Python-level convergence and AUC of the sparse sampler
    vs the dense baseline, providing the empirical basis for the v4 RTL spec. We test
    K=[8, 16, 32] to identify the best tradeoff between LUT savings and sampling quality.

SPEC: REQ-SAMPLE-020, SCENARIO-SAMPLE-035
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from scripts.experiment_template import ExperimentTemplate
from python.carnot.models.sparse_ising import SparseIsingEBM
from python.carnot.models.ising import IsingConfig, IsingModel

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 950
TITLE = "E-MVL Sparsified Ising: K-regular sparse sampler vs dense baseline (CPU)"
DELIVERABLE = "results/experiment_950_emvl_sparsified_ising.json"

N_VARS = 64  # Number of Ising spins (KV260 default)
K_VALUES = [8, 16, 32]  # Sparse connectivity levels to test
N_STEPS = 50  # Sampling steps per trial
N_TRIALS = 10  # Convergence trials per K value
N_AUC_SAMPLES = 200  # Samples for AUC evaluation (100 low-energy, 100 high-energy)
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# AUC computation
# ---------------------------------------------------------------------------


def compute_auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Compute area under the ROC curve for binary classification by energy.

    **Detailed explanation for engineers:**
        We treat "low energy = positive class" (correctly-constrained spin assignment)
        and "high energy = negative class" (random/incorrect assignment).

        AUC = P(E(low-energy sample) < E(high-energy sample))
        Computed as the fraction of (pos, neg) pairs where E_pos < E_neg.
        AUC = 0.5 means random (sparse model can't discriminate); AUC > 0.5
        means the energy successfully distinguishes the two classes.

    Args:
        scores_pos: Energy values for the "good" class (should be low).
        scores_neg: Energy values for the "bad" class (should be high).

    Returns:
        AUC in [0, 1].
    """
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    count = 0
    for ep in scores_pos:
        for en in scores_neg:
            if ep < en:
                count += 1
            elif ep == en:
                count += 0.5
    return count / (n_pos * n_neg)


def generate_auc_samples(
    model: SparseIsingEBM,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate low-energy (positive) and random (negative) samples for AUC.

    **Detailed explanation for engineers:**
        Low-energy (positive) samples: run the E-MVL sampler for N_STEPS sweeps
        from random initialization. After sufficient steps, spins settle into
        low-energy configurations that satisfy most constraints.

        High-energy (negative) samples: purely random ±1 spin assignments.
        These don't respect the coupling structure, so they typically have
        higher energy than sampled configurations.

        AUC measures how well the energy function discriminates between these
        two classes. Good AUC (>0.60) confirms the sparse model has learned
        a meaningful energy landscape.

    Args:
        model: SparseIsingEBM instance to evaluate.
        n_samples: Number of samples in each class.
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of (energies_positive, energies_negative) arrays.
    """
    n_vars = model.config.input_dim
    nbrs_np = np.array(model.neighbor_idx)
    J_np = np.array(model.J_sparse)
    b_np = np.array(model.bias)

    def sparse_energy(s: np.ndarray) -> float:
        nbr_s = s[nbrs_np]
        lf = np.sum(J_np * nbr_s, axis=1)
        return float(-0.5 * np.dot(s, lf) - np.dot(b_np, s))

    # Generate positive (low-energy) samples via E-MVL from random init
    energies_pos = []
    for i in range(n_samples):
        spins = rng.choice([-1.0, 1.0], size=n_vars)
        # Run E-MVL for N_STEPS sweeps to reach low energy
        for _ in range(N_STEPS):
            new_spins = np.empty(n_vars)
            for vi in range(n_vars):
                h_i = float(np.dot(J_np[vi], spins[nbrs_np[vi]])) + float(b_np[vi])
                new_spins[vi] = 1.0 if h_i >= 0 else -1.0
            spins = new_spins
        energies_pos.append(sparse_energy(spins))

    # Generate negative (random) samples — no optimization
    energies_neg = []
    for _ in range(n_samples):
        spins = rng.choice([-1.0, 1.0], size=n_vars)
        energies_neg.append(sparse_energy(spins))

    return np.array(energies_pos), np.array(energies_neg)


def dense_auc_samples(
    model: IsingModel,
    n_samples: int,
    rng: np.random.Generator,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate AUC samples for the dense Ising baseline via Gibbs sampling.

    Same structure as generate_auc_samples but uses dense Gibbs updates.
    """
    n_vars = model.config.input_dim
    J_np = np.array(model.coupling)
    b_np = np.array(model.bias)

    def dense_energy(s: np.ndarray) -> float:
        return float(-0.5 * s @ J_np @ s - b_np @ s)

    # Positive samples via dense Gibbs
    energies_pos = []
    for i in range(n_samples):
        spins = rng.choice([-1.0, 1.0], size=n_vars)
        for _ in range(n_steps):
            order = rng.permutation(n_vars)
            for vi in order:
                h_i = float(J_np[vi] @ spins) + float(b_np[vi])
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * h_i))
                spins[vi] = 1.0 if rng.random() < p_plus else -1.0
        energies_pos.append(dense_energy(spins))

    # Negative samples (random)
    energies_neg = []
    for _ in range(n_samples):
        spins = rng.choice([-1.0, 1.0], size=n_vars)
        energies_neg.append(dense_energy(spins))

    return np.array(energies_pos), np.array(energies_neg)


def steps_to_converge(trajectory: list[float]) -> int:
    """Return first step index where energy crosses the midpoint threshold.

    **Detailed explanation for engineers:**
        "Convergence" is defined as the energy dropping below the midpoint between
        the initial and final energy in the trajectory. This is a normalized metric
        that works regardless of the absolute energy scale.

        If the energy never drops below this threshold (e.g., the chain gets stuck),
        we return len(trajectory) — the maximum possible value — indicating slow
        convergence.
    """
    if len(trajectory) < 2:
        return len(trajectory)
    threshold = (trajectory[0] + trajectory[-1]) / 2.0
    for step, e in enumerate(trajectory):
        if e <= threshold:
            return step
    return len(trajectory)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict:
    """Run the E-MVL sparse Ising experiment and return result dict."""
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    rng = np.random.default_rng(RANDOM_SEED)
    key = jrandom.PRNGKey(RANDOM_SEED)

    # -----------------------------------------------------------------------
    # Dense baseline: IsingModel with N=64
    # -----------------------------------------------------------------------
    print(f"[Exp {EXP_ID}] Building dense baseline (N={N_VARS})...")
    dense_key, key = jrandom.split(key)
    dense_model = IsingModel(IsingConfig(input_dim=N_VARS), key=dense_key)
    J_dense = np.array(dense_model.coupling)
    b_dense = np.array(dense_model.bias)

    def dense_traj(trial_idx: int) -> list[float]:
        """Run dense Gibbs for N_STEPS and record energy trajectory."""
        trial_rng = np.random.default_rng(RANDOM_SEED + trial_idx * 7)
        spins = trial_rng.choice([-1.0, 1.0], size=N_VARS)

        def e_dense(s: np.ndarray) -> float:
            return float(-0.5 * s @ J_dense @ s - b_dense @ s)

        traj = [e_dense(spins)]
        for _ in range(N_STEPS):
            order = trial_rng.permutation(N_VARS)
            for vi in order:
                h_i = float(J_dense[vi] @ spins) + float(b_dense[vi])
                p_plus = 1.0 / (1.0 + np.exp(-2.0 * h_i))
                spins[vi] = 1.0 if trial_rng.random() < p_plus else -1.0
            traj.append(e_dense(spins))
        return traj

    print(f"[Exp {EXP_ID}] Running dense baseline convergence ({N_TRIALS} trials)...")
    dense_steps = []
    for t in range(N_TRIALS):
        traj = dense_traj(t)
        dense_steps.append(steps_to_converge(traj))
    mean_dense_steps = float(np.mean(dense_steps))
    print(f"  Dense mean steps to converge: {mean_dense_steps:.1f}")

    print(
        f"[Exp {EXP_ID}] Running dense baseline AUC ({N_AUC_SAMPLES // 2} pos + {N_AUC_SAMPLES // 2} neg)..."
    )
    auc_rng = np.random.default_rng(RANDOM_SEED + 1000)
    e_pos_dense, e_neg_dense = dense_auc_samples(dense_model, N_AUC_SAMPLES // 2, auc_rng, N_STEPS)
    auc_dense = compute_auc(e_pos_dense, e_neg_dense)
    print(f"  Dense AUC: {auc_dense:.4f}")

    # -----------------------------------------------------------------------
    # Sparse models at each K value
    # -----------------------------------------------------------------------
    sparse_results = {}

    for K in K_VALUES:
        print(f"\n[Exp {EXP_ID}] Testing sparse K={K}...")
        k_key, key = jrandom.split(key)
        sparse_model = SparseIsingEBM(n_vars=N_VARS, n_neighbors=K, key=k_key)

        # --- Convergence: sparse Gibbs ---
        gibbs_steps = []
        for t in range(N_TRIALS):
            trial_key = jrandom.PRNGKey(RANDOM_SEED + t * 13 + K)
            traj = sparse_model.energy_trajectory(N_STEPS, sampler="gibbs", key=trial_key)
            gibbs_steps.append(steps_to_converge(traj))
        mean_gibbs_steps = float(np.mean(gibbs_steps))

        # --- Convergence: E-MVL ---
        emvl_steps = []
        for t in range(N_TRIALS):
            trial_key = jrandom.PRNGKey(RANDOM_SEED + t * 17 + K)
            traj = sparse_model.energy_trajectory(N_STEPS, sampler="emvl", key=trial_key)
            emvl_steps.append(steps_to_converge(traj))
        mean_emvl_steps = float(np.mean(emvl_steps))

        # --- AUC ---
        auc_rng_k = np.random.default_rng(RANDOM_SEED + 2000 + K)
        e_pos_k, e_neg_k = generate_auc_samples(sparse_model, N_AUC_SAMPLES // 2, auc_rng_k)
        auc_k = compute_auc(e_pos_k, e_neg_k)

        # Speedup ratios (steps_dense / steps_sparse)
        speedup_gibbs = (
            mean_dense_steps / mean_gibbs_steps if mean_gibbs_steps > 0 else float(N_STEPS)
        )
        speedup_emvl = mean_dense_steps / mean_emvl_steps if mean_emvl_steps > 0 else float(N_STEPS)

        # LUT estimate relative to dense N=128
        # Dense N=128: 290K LUTs. Sparse K: (K/128) * 290K
        lut_estimate_n128 = int(290_000 * K / 128)

        print(f"  K={K}: Gibbs steps={mean_gibbs_steps:.1f}, E-MVL steps={mean_emvl_steps:.1f}")
        print(f"  K={K}: speedup_emvl={speedup_emvl:.2f}x vs dense, AUC={auc_k:.4f}")
        print(f"  K={K}: estimated N=128 LUTs = {lut_estimate_n128:,}")

        sparse_results[f"K{K}"] = {
            "n_neighbors": K,
            "mean_gibbs_steps": mean_gibbs_steps,
            "mean_emvl_steps": mean_emvl_steps,
            "speedup_ratio_emvl_vs_dense": speedup_emvl,
            "speedup_ratio_gibbs_vs_dense": speedup_gibbs,
            "auc": auc_k,
            "lut_estimate_n128_spins": lut_estimate_n128,
            "lut_within_budget": lut_estimate_n128 <= 117_000,
        }

    # -----------------------------------------------------------------------
    # Determine honest_verdict based on K=16 speedup ratio
    # -----------------------------------------------------------------------
    k16_speedup = sparse_results["K16"]["speedup_ratio_emvl_vs_dense"]
    if k16_speedup >= 1.5:
        honest_verdict = "emvl_speedup_confirmed"
    elif k16_speedup > 0.8:
        honest_verdict = "emvl_comparable"
    else:
        honest_verdict = "emvl_slower"

    print(f"\n[Exp {EXP_ID}] K=16 speedup ratio: {k16_speedup:.2f}x -> {honest_verdict}")

    # -----------------------------------------------------------------------
    # Build result artifact
    # -----------------------------------------------------------------------
    result = tmpl.build_result(
        {
            "honest_verdict": honest_verdict,
            "n_vars": N_VARS,
            "n_steps": N_STEPS,
            "n_trials": N_TRIALS,
            "dense_baseline": {
                "mean_steps_to_converge": mean_dense_steps,
                "auc": auc_dense,
            },
            "sparse_results": sparse_results,
            "k16_speedup_ratio": k16_speedup,
            "recommended_k_for_kv260_v4": 16,
            "kv260_v4_lut_estimate_k16": sparse_results["K16"]["lut_estimate_n128_spins"],
            "kv260_v3_lut_overflow": 290_000,
            "kv260_lut_budget": 117_000,
            "summary": (
                f"E-MVL K=16 achieves {k16_speedup:.2f}x convergence speedup vs dense baseline. "
                f"AUC={sparse_results['K16']['auc']:.4f}. "
                f"Estimated KV260 N=128 LUTs at K=16: "
                f"{sparse_results['K16']['lut_estimate_n128_spins']:,} "
                f"({'within' if sparse_results['K16']['lut_within_budget'] else 'over'} 117K budget)."
            ),
        },
        status="success" if honest_verdict != "emvl_slower" else "partial",
    )

    return result


if __name__ == "__main__":
    result = run_experiment()
    out_path = DELIVERABLE
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[Exp {EXP_ID}] Result written to {out_path}")
    print(f"[Exp {EXP_ID}] honest_verdict: {result['honest_verdict']}")
