"""Experiment 435a — Phase 3 seed: continuous EBM vs Ising minimum recovery.

**Researcher summary:**
    Trains a tiny continuous-valued EBM on a 10-variable constraint problem and
    compares its energy minimum against the Ising simulated-annealing baseline.
    CPU-only, <30 min.  This is the first concrete seed for Phase 3 (Kona parity).

**What this experiment does:**
    1. Builds a random 10-variable Ising model with sparse couplings (~30% density).
    2. Runs simulated annealing to find an approximate ground state (discrete {-1,+1}).
    3. Reuses the same J/h to build a ContinuousEBM and runs gradient descent with
       tanh squashing to find an approximate continuous minimum.
    4. Compares the two minimisers via L2 distance and sign agreement.
    5. Writes a structured artifact to results/experiment_435a_kona_continuous_energy.json.

**Why does this matter?**
    Kona (Phase 3 North Star) requires non-autoregressive inference over a continuous
    latent space.  For that architecture to be sound, the energy function must be
    consistent across the discrete↔continuous boundary.  If gradient descent and
    simulated annealing agree on the same 10-variable problem, the energy landscape is
    "trustworthy" — not an artefact of the discrete spin domain.

Spec: REQ-KONA-001, SCENARIO-KONA-001, SCENARIO-KONA-002
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Apply environment autofix first — detects GPU and injects CARNOT_FORCE_LIVE if needed.
# This is CPU-only but we call it as a belt-and-suspenders measure per project convention.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

from carnot.phase3.continuous_ebm import (
    build_kona_artifact,
    compare_minima,
    fit_continuous_ebm,
    sample_continuous,
)

RESULTS_PATH = Path("results/experiment_435a_kona_continuous_energy.json")
N_VARS = 10
COUPLING_DENSITY = 0.3
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_sparse_ising(
    n: int = N_VARS,
    density: float = COUPLING_DENSITY,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random n-variable Ising coupling matrix J and bias h.

    Uses uniform random entries masked to achieve the target density.  The
    coupling is symmetrised to enforce J = J^T (required by the Ising energy).

    Args:
        n: Number of variables.
        density: Fraction of off-diagonal entries that are non-zero.
        seed: Random seed for reproducibility.

    Returns:
        Tuple (J, h) where J has shape (n, n) and h has shape (n,).
    """
    rng = np.random.default_rng(seed)
    # Upper-triangular sparsity mask, symmetrised
    mask = rng.random((n, n)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    J_raw = rng.uniform(-1.0, 1.0, (n, n)) * mask
    J = (J_raw + J_raw.T) / 2.0  # exact symmetry
    h = rng.uniform(-0.5, 0.5, n)
    return J, h


def simulated_annealing(
    J: np.ndarray,
    h: np.ndarray,
    n_steps: int = 10_000,
    T_start: float = 2.0,
    T_end: float = 0.01,
    seed: int = 1,
) -> tuple[np.ndarray, float]:
    """Simulated annealing on the Ising energy E(x) = -0.5*x^T*J*x - h^T*x.

    Uses single-spin Metropolis-Hastings flips with a geometric cooling schedule.
    Variables are kept in {-1, +1} (discrete spins).

    Args:
        J: Symmetric coupling matrix of shape (n, n).
        h: Bias vector of shape (n,).
        n_steps: Total number of flip proposals.
        T_start: Starting temperature (high = exploratory).
        T_end: Final temperature (low = greedy).
        seed: Random seed.

    Returns:
        Tuple (best_state, best_energy) where best_state is a numpy array of
        {-1, +1} values and best_energy is the scalar energy at that state.
    """
    rng = np.random.default_rng(seed)
    n = J.shape[0]
    state = rng.choice(np.array([-1.0, 1.0]), size=n)
    best = state.copy()
    best_e = float(-0.5 * state @ J @ state - h @ state)

    for step in range(n_steps):
        # Geometric cooling: T = T_start * (T_end/T_start)^(step/n_steps)
        T = T_start * (T_end / T_start) ** (step / n_steps)
        i = int(rng.integers(n))
        # Energy change for flipping spin i: ΔE = 2*s_i*(J_i·s + h_i)
        delta = 2.0 * state[i] * (float(J[i] @ state) + float(h[i]))
        if delta < 0.0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
            state[i] = -state[i]
        e = float(-0.5 * state @ J @ state - h @ state)
        if e < best_e:
            best_e = e
            best = state.copy()

    return best, best_e


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 435a and write the deliverable JSON artifact."""
    t0 = time.time()

    print("=== Exp 435a: Phase 3 seed — continuous EBM vs Ising ===")
    print(f"  n_vars={N_VARS}, density={COUPLING_DENSITY}, seed={SEED}")

    # 1. Build the Ising problem
    J, h = build_sparse_ising(N_VARS, COUPLING_DENSITY, SEED)
    n_nonzero = int(np.sum(J != 0))
    print(f"  Built {N_VARS}-var Ising: {n_nonzero} non-zero couplings")

    # 2. Ising simulated annealing
    print("  Running simulated annealing (Ising)...")
    ising_state, ising_energy = simulated_annealing(J, h, n_steps=10_000, seed=1)
    print(f"    Ising ground-state energy: {ising_energy:.6f}")
    print(f"    Ising state: {ising_state}")

    # 3. Fit ContinuousEBM from the same J/h
    print("  Fitting ContinuousEBM from Ising parameters...")

    class _IsingProxy:
        """Minimal proxy so fit_continuous_ebm can read coupling/bias."""

        def __init__(self, coupling: np.ndarray, bias: np.ndarray) -> None:
            self.coupling = coupling
            self.bias = bias

    proxy = _IsingProxy(J, h)
    cont_model = fit_continuous_ebm(proxy)

    # 4. Gradient descent on the continuous EBM
    print("  Running gradient descent (continuous EBM)...")
    cont_state = sample_continuous(cont_model, n_steps=2000, lr=0.02, seed=0)
    cont_energy = float(
        -0.5 * cont_state @ J @ cont_state - h @ cont_state
    )
    print(f"    Continuous minimiser energy: {cont_energy:.6f}")
    print(f"    Continuous state: {np.round(cont_state, 3)}")

    # 5. Compare
    comparison = compare_minima(ising_state, cont_state)
    print(
        f"  Comparison: L2={comparison['l2_distance']:.4f}, "
        f"sign_agreement={comparison['sign_agreement']:.3f}"
    )

    # 6. Build artifact
    duration_s = time.time() - t0
    artifact = build_kona_artifact(
        comparison,
        extra={
            "experiment": "435a",
            "phase": "phase3_seed",
            "n_vars": N_VARS,
            "coupling_density": COUPLING_DENSITY,
            "ising_energy": ising_energy,
            "continuous_energy": cont_energy,
            "ising_state": ising_state.tolist(),
            "continuous_state": cont_state.tolist(),
            "duration_s": round(duration_s, 3),
            "note": (
                "Phase 3 seed experiment (NOT production). "
                "Validates discrete↔continuous energy landscape consistency."
            ),
        },
    )

    verdict = artifact["honest_verdict"]
    print(f"\n  Honest verdict: {verdict}")

    # 7. Write results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"  Written: {RESULTS_PATH}")
    print(f"  Duration: {duration_s:.1f}s")


if __name__ == "__main__":
    main()
