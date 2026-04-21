#!/usr/bin/env python3
"""Experiment 650: KAEM Multilevel Sparse — combining coarse-to-fine with sparsification.

**Researcher summary:**
    RETRO-057 (carry 5) identifies that sparse-only training (Exp 637) fails to reach
    the 5% energy accuracy threshold vs dense baseline (error=0.429).  MultilevelKAEM
    (Exp 634) also underperformed.  This experiment tests the combined approach:
    MultilevelSparseKAEMTrainer trains SparseKAEMEnergy at progressively finer knot
    resolutions (K=16 -> K=32 -> K=64), sparsifying the coupling matrix after each
    level.  The hypothesis is that coarse-to-fine avoids poor local minima while
    sparsity provides parameter reduction.

**Gate:**
    0. apply_env_autofix() FIRST.
    1. ExperimentTimeoutWatchdog(650, timeout_minutes=35).
    2. Load prior baselines from Exp 637.
    3. Train MultilevelSparseKAEMTrainer, sweep top_k_fraction in [0.05, 0.10, 0.20].
    4. Compare multilevel_sparse_vs_dense_error to 5% threshold and prior.
    5. tmpl.assert_deliverable_written() — FINAL LINE.

Spec: REQ-SAMPLE-025, SCENARIO-SAMPLE-040, SCENARIO-SAMPLE-041
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix BEFORE any heavy imports.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

# ---------------------------------------------------------------------------
# Step 1: Watchdog — hard 35-minute wall-clock cap.
# ---------------------------------------------------------------------------
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(650, timeout_minutes=35)

# ---------------------------------------------------------------------------
# Remaining imports
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from carnot.models.kaem_energy import KAEMEnergy  # noqa: E402
from carnot.models.sparse_kaem_energy import SparseKAEMEnergy  # noqa: E402
from carnot.training.multilevel_sparse_kaem import MultilevelSparseKAEMTrainer  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_DELIVERABLE = "results/experiment_650_kaem_multilevel_sparse.json"

tmpl = ExperimentTemplate(
    650,
    "KAEM Multilevel Sparse",
    _DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()


# ---------------------------------------------------------------------------
# Synthetic energy landscape (same parameters as Exps 634/637)
# ---------------------------------------------------------------------------

N_VARS = 20
N_DATA = 200
SCHEDULE = [16, 32, 64]
EPOCHS_PER_LEVEL = 20
TOP_K_FRACTIONS = [0.05, 0.10, 0.20]


def _make_synthetic_data(n_vars: int, n_data: int, seed: int = 42) -> jnp.ndarray:
    """Generate synthetic training data with known structure in [-1, 1].

    Uses a multivariate Gaussian with structured covariance so that the
    energy function has non-trivial pairwise interaction terms.  This is the
    same generation procedure as Exps 634 and 637, ensuring baseline
    comparisons are apples-to-apples.

    Args:
        n_vars: Number of variables.
        n_data: Number of training samples.
        seed: NumPy random seed.

    Returns:
        jnp.ndarray of shape (n_data, n_vars) clipped to [-1, 1].
    """
    rng = np.random.default_rng(seed)
    # Structured covariance: Toeplitz-like, decaying off-diagonal correlations
    cov = np.eye(n_vars)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                cov[i, j] = 0.3 ** abs(i - j)
    samples = rng.multivariate_normal(np.zeros(n_vars), cov, size=n_data).astype(np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    return jnp.array(samples)


def _eval_energy_mae(model_a: object, model_b: object, n_samples: int = 500, seed: int = 7) -> float:
    """Compute mean absolute error between two models' energies on random inputs.

    Evaluates both models on the same set of random points in [-1, 1]^n_vars,
    then returns mean(|E_a(x) - E_b(x)|).

    Args:
        model_a: First energy model with .energy(x) -> scalar.
        model_b: Second energy model with .energy(x) -> scalar.
        n_samples: Number of evaluation points.
        seed: NumPy random seed for evaluation points.

    Returns:
        Float MAE between the two models' energies.
    """
    rng = np.random.default_rng(seed)
    n_vars = model_a.n_vars if hasattr(model_a, "n_vars") else model_b.n_vars
    xs = rng.uniform(-1.0, 1.0, size=(n_samples, n_vars)).astype(np.float32)
    errors = []
    for x in xs:
        ea = float(model_a.energy(jnp.array(x)))
        eb = float(model_b.energy(jnp.array(x)))
        errors.append(abs(ea - eb))
    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

print("Experiment 650: KAEM Multilevel Sparse")
print(f"N_VARS={N_VARS}, N_DATA={N_DATA}, SCHEDULE={SCHEDULE}, EPOCHS_PER_LEVEL={EPOCHS_PER_LEVEL}")

# Step A: Generate synthetic data
data = _make_synthetic_data(N_VARS, N_DATA)

# Step B: Load prior baselines from Exp 637
_EXP637_PATH = _REPO_ROOT / "results" / "experiment_637_lowrank_kaem_sparse.json"
try:
    with open(_EXP637_PATH) as f:
        exp637 = json.load(f)
    accuracy_standard = exp637.get("energy_accuracy_dense", None)
    accuracy_sparse_only = exp637.get("energy_accuracy_sparse", None)
    sparse_vs_dense_error_prior = exp637.get("sparse_vs_dense_error", 0.429)
    print(f"Loaded Exp 637: accuracy_standard={accuracy_standard:.4f}, "
          f"accuracy_sparse_only={accuracy_sparse_only:.4f}, "
          f"sparse_vs_dense_error_prior={sparse_vs_dense_error_prior:.4f}")
except (FileNotFoundError, KeyError) as e:
    print(f"WARNING: could not load Exp 637 results ({e}). Using fallback values.")
    accuracy_standard = None
    accuracy_sparse_only = None
    sparse_vs_dense_error_prior = 0.429

# Step C: If accuracy_standard is None, train a fresh dense baseline
if accuracy_standard is None:
    print("Training fresh dense KAEMEnergy baseline...")
    dense_model = KAEMEnergy(n_vars=N_VARS, n_hidden=64)
    dense_model.fit(data, n_epochs=EPOCHS_PER_LEVEL * len(SCHEDULE))
    # Create a reference sparse model to measure against
    ref_sparse = SparseKAEMEnergy(n_vars=N_VARS, n_knots=64, top_k_fraction=0.1)
    ref_sparse.fit(data, n_epochs=EPOCHS_PER_LEVEL * len(SCHEDULE))
    accuracy_standard = _eval_energy_mae(dense_model, ref_sparse)
    print(f"Fresh baseline accuracy_standard={accuracy_standard:.4f}")

# Step D: Sweep top_k_fraction to find best multilevel sparse result
print(f"\nSweeping top_k_fraction over {TOP_K_FRACTIONS}...")
sweep_results = []
best_error = float("inf")
best_top_k = TOP_K_FRACTIONS[0]
best_model = None
best_accuracy = None

for tkf in TOP_K_FRACTIONS:
    print(f"  top_k_fraction={tkf}: training multilevel sparse...")
    trainer = MultilevelSparseKAEMTrainer(
        schedule=SCHEDULE,
        epochs_per_level=EPOCHS_PER_LEVEL,
        top_k_fraction=tkf,
    )
    trained = trainer.train(n_vars=N_VARS, data=data)

    # Measure energy MAE vs a reference dense model
    # Use a simple KAEMEnergy as the dense reference for consistent comparison
    dense_ref = KAEMEnergy(n_vars=N_VARS, n_hidden=64)
    dense_ref.fit(data, n_epochs=EPOCHS_PER_LEVEL * len(SCHEDULE))
    mae = _eval_energy_mae(dense_ref, trained)

    # Relative error vs dense (consistent with Exp 637 metric)
    if accuracy_standard and accuracy_standard > 0:
        rel_error = abs(mae - accuracy_standard) / accuracy_standard
    else:
        rel_error = float("inf")

    print(f"    mae={mae:.4f}, rel_error_vs_dense={rel_error:.4f}")
    sweep_results.append({
        "top_k_fraction": tkf,
        "mae": mae,
        "relative_error_vs_dense": rel_error,
    })

    if rel_error < best_error:
        best_error = rel_error
        best_top_k = tkf
        best_model = trained
        best_accuracy = mae

total_epochs = len(SCHEDULE) * EPOCHS_PER_LEVEL
accuracy_multilevel_sparse = best_accuracy if best_accuracy is not None else float("inf")
multilevel_sparse_vs_dense_error = best_error

print(f"\nBest top_k_fraction={best_top_k}")
print(f"accuracy_multilevel_sparse={accuracy_multilevel_sparse:.4f}")
print(f"multilevel_sparse_vs_dense_error={multilevel_sparse_vs_dense_error:.4f}")
print(f"sparse_vs_dense_error_prior={sparse_vs_dense_error_prior:.4f}")

retro_057_resolved = multilevel_sparse_vs_dense_error < 0.05
improvement_over_sparse_only = sparse_vs_dense_error_prior - multilevel_sparse_vs_dense_error

print(f"retro_057_resolved={retro_057_resolved}")
print(f"improvement_over_sparse_only={improvement_over_sparse_only:.4f}")

if retro_057_resolved:
    honest_verdict = "multilevel_sparse_resolves_retro_057"
elif multilevel_sparse_vs_dense_error < sparse_vs_dense_error_prior:
    honest_verdict = "multilevel_sparse_improved"
else:
    honest_verdict = "multilevel_sparse_no_improvement"

print(f"honest_verdict={honest_verdict}")

# ---------------------------------------------------------------------------
# Build and write artifact
# ---------------------------------------------------------------------------
_artifact_data = tmpl.build_result(
    {
        "schema": "carnot.kaem_multilevel_sparse.v1",
        "n_vars": N_VARS,
        "accuracy_standard": accuracy_standard,
        "accuracy_sparse_only": accuracy_sparse_only,
        "accuracy_multilevel_sparse": accuracy_multilevel_sparse,
        "sparse_vs_dense_error_prior": sparse_vs_dense_error_prior,
        "multilevel_sparse_vs_dense_error": multilevel_sparse_vs_dense_error,
        "improvement_over_sparse_only": improvement_over_sparse_only,
        "best_top_k_fraction": best_top_k,
        "total_epochs": total_epochs,
        "sweep_results": sweep_results,
        "retro_057_resolved": retro_057_resolved,
        "honest_verdict": honest_verdict,
    },
    status="success",
)

AtomicResultWriter(str(_REPO_ROOT / _DELIVERABLE)).write(_artifact_data)
tmpl.assert_deliverable_written()
