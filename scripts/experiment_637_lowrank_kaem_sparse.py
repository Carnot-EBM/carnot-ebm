#!/usr/bin/env python3
"""Experiment 637: LowRankKAEM Sparse Redesign — SparseKAEMEnergy vs Dense vs LowRank baseline.

**Context (RETRO-057):**
    LowRankKAEMEnergy at small k compresses the energy function by SVD projection.
    This loses many small eigenvalues that collectively matter (carry count >= 4 in RETRO-057).
    The affine calibration approach (Exp 532/559) was exhausted.

    New approach: keep all univariate terms (full-rank marginals), but sparsify only
    the pairwise coupling matrix — retain only top-K interactions per variable.
    This should lose fewer significant energy terms while achieving similar parameter savings.

**Design:**
    - Dense KAEMEnergy baseline: KAEMEnergy(n_hidden=64), trained via MultilevelKAEMTrainer
    - SparseKAEMEnergy(top_k_fraction=0.1): same training budget
    - Sweep top_k_fraction in [0.05, 0.10, 0.20, 0.50] to find accuracy vs sparsity tradeoff
    - Compare to LowRankKAEM prior result from results/experiment_532_*.json if present
    - Report retro_057_resolved = (sparse_vs_dense_error < 0.05)

Spec: REQ-SAMPLE-021, REQ-SAMPLE-022, SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import logging

import jax.numpy as jnp
import numpy as np

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.models.kaem_energy import KAEMEnergy
from carnot.models.sparse_kaem_energy import SparseKAEMEnergy
from carnot.training.multilevel_kan_trainer import MultilevelKAEMTrainer
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Synthetic energy landscape helpers (same as Exp 634 for reproducibility)
# ---------------------------------------------------------------------------


def ground_truth_energy(x: np.ndarray) -> float:
    """Compute sum_i [sin(3*x_i) + x_i^2] — same ground truth as Exp 634.

    This is a smooth, non-trivial multivariate energy with known analytic
    structure.  Using the same ground truth as Exp 634 makes the two experiments
    directly comparable (same training and test distributions).

    Parameters
    ----------
    x : np.ndarray
        Sample vector of shape (n_vars,), values in [-1, 1].

    Returns
    -------
    float
        Scalar ground-truth energy.
    """
    return float(np.sum(np.sin(3.0 * x) + x**2))


def generate_data(n_samples: int, n_vars: int, rng: np.random.Generator) -> np.ndarray:
    """Generate training data from ground-truth Boltzmann distribution.

    Uses rejection sampling from a uniform proposal, accepting with probability
    proportional to exp(-E(x)).  Same procedure as Exp 634.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.
    n_vars : int
        Dimensionality.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, n_vars), values in [-1, 1].
    """
    n_candidates = n_samples * 20
    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, n_vars)).astype(np.float32)
    energies = np.array([ground_truth_energy(candidates[i]) for i in range(n_candidates)])
    energies_shifted = energies - np.min(energies)
    probs = np.exp(-energies_shifted)
    probs = probs / probs.sum()
    indices = rng.choice(n_candidates, size=n_samples, replace=False, p=probs)
    return candidates[indices]


def compute_energy_mae(model: "KAEMEnergy | SparseKAEMEnergy", test_x: np.ndarray) -> float:
    """Compute mean absolute error between model energy and ground truth.

    Both energies are centred on their mean before computing MAE so we
    measure shape accuracy, not absolute offset (EBMs are defined up to
    an additive constant).

    Parameters
    ----------
    model : KAEMEnergy or SparseKAEMEnergy
        Trained model with an energy(x) method.
    test_x : np.ndarray
        Test points, shape (n_test, n_vars).

    Returns
    -------
    float
        Mean absolute error between model and ground-truth energies.
    """
    n_test = len(test_x)
    model_energies = np.array(
        [float(model.energy(jnp.array(test_x[i]))) for i in range(n_test)]
    )
    gt_energies = np.array([ground_truth_energy(test_x[i]) for i in range(n_test)])
    model_energies -= model_energies.mean()
    gt_energies -= gt_energies.mean()
    return float(np.mean(np.abs(model_energies - gt_energies)))


def _load_lowrank_baseline() -> "float | None":
    """Load LowRankKAEM energy error from Exp 532 result file if it exists.

    Returns None if the result file is missing (graceful degradation — the
    comparison is informational and not required for the retro resolution check).

    Returns
    -------
    float | None
        Prior lowrank energy error, or None if unavailable.
    """
    # Try a few candidate file names from the exp 532 series
    candidates = [
        _REPO_ROOT / "results" / "experiment_532_low_rank_kaem.json",
        _REPO_ROOT / "results" / "experiment_532_lowrank_kaem.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                # Try common field names used in the Exp 532 / Exp 559 series
                for field in ["energy_mad_normalized", "energy_error", "lowrank_energy_error"]:
                    if field in data:
                        return float(data[field])
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 637: SparseKAEMEnergy vs dense baseline, RETRO-057 resolution check."""
    apply_env_autofix()
    ExperimentTimeoutWatchdog(637, timeout_minutes=35)

    tmpl = ExperimentTemplate(
        637,
        "LowRankKAEM Sparse Redesign",
        "results/experiment_637_lowrank_kaem_sparse.json",
        requires_gpu=False,
    )
    tmpl.setup()

    n_vars = 20
    n_train = 300
    n_test = 500
    rng = np.random.default_rng(42)

    _log.info("Generating synthetic energy landscape: %d vars, %d train, %d test", n_vars, n_train, n_test)
    train_data = generate_data(n_train, n_vars, rng)
    test_x = rng.uniform(-1.0, 1.0, size=(n_test, n_vars)).astype(np.float32)
    train_jax = jnp.array(train_data)

    # ------------------------------------------------------------------
    # Dense KAEMEnergy baseline via MultilevelKAEMTrainer (same as Exp 634)
    # Using smaller schedule to keep runtime within watchdog limit
    # ------------------------------------------------------------------
    _log.info("Dense baseline: MultilevelKAEMTrainer(schedule=[16,32,64], epochs_per_level=10)")
    trainer = MultilevelKAEMTrainer(schedule=[16, 32, 64], epochs_per_level=10)
    dense_model = trainer.train(n_vars=n_vars, data=train_jax)
    energy_accuracy_dense = compute_energy_mae(dense_model, test_x)
    _log.info("Dense baseline MAE: %.6f", energy_accuracy_dense)

    # ------------------------------------------------------------------
    # SparseKAEMEnergy at default top_k_fraction=0.10
    # Training: same total epoch budget (30 epochs for fair comparison)
    # ------------------------------------------------------------------
    _log.info("SparseKAEMEnergy(top_k_fraction=0.10): training 30 epochs")
    sparse_model = SparseKAEMEnergy(n_vars=n_vars, n_knots=64, top_k_fraction=0.10)
    sparse_model.fit(train_jax, n_epochs=30)
    energy_accuracy_sparse = compute_energy_mae(sparse_model, test_x)
    _log.info("Sparse MAE (fraction=0.10): %.6f", energy_accuracy_sparse)

    # ------------------------------------------------------------------
    # Sweep top_k_fraction to find best fraction where error < 5%
    # ------------------------------------------------------------------
    fractions_to_try = [0.05, 0.10, 0.20, 0.50]
    sweep_results: list[dict] = []
    best_fraction = fractions_to_try[0]
    best_fraction_mae = None

    for frac in fractions_to_try:
        _log.info("Sweep fraction=%.2f", frac)
        m = SparseKAEMEnergy(n_vars=n_vars, n_knots=64, top_k_fraction=frac)
        m.fit(train_jax, n_epochs=30)
        mae = compute_energy_mae(m, test_x)
        # relative error vs dense baseline
        if energy_accuracy_dense > 1e-10:
            rel_err = abs(mae - energy_accuracy_dense) / energy_accuracy_dense
        else:
            rel_err = abs(mae - energy_accuracy_dense)
        _log.info("  fraction=%.2f MAE=%.6f rel_err=%.4f", frac, mae, rel_err)
        sweep_results.append({
            "top_k_fraction": frac,
            "mae": float(mae),
            "relative_error_vs_dense": float(rel_err),
            "within_5pct": bool(rel_err < 0.05),
        })
        # Track the maximum fraction that still achieves < 5% relative error
        if rel_err < 0.05:
            best_fraction = frac
            best_fraction_mae = mae

    # Fall back to default (0.10) if none passed
    if best_fraction_mae is None:
        # Use the fraction with smallest rel_err
        best_entry = min(sweep_results, key=lambda r: r["relative_error_vs_dense"])
        best_fraction = best_entry["top_k_fraction"]
        best_fraction_mae = best_entry["mae"]

    # ------------------------------------------------------------------
    # RETRO-057 resolution check
    # ------------------------------------------------------------------
    if energy_accuracy_dense > 1e-10:
        sparse_vs_dense_error = abs(energy_accuracy_sparse - energy_accuracy_dense) / energy_accuracy_dense
    else:
        sparse_vs_dense_error = abs(energy_accuracy_sparse - energy_accuracy_dense)

    retro_057_resolved = bool(sparse_vs_dense_error < 0.05)

    if retro_057_resolved:
        honest_verdict = "sparse_resolves_retro_057"
    elif sparse_vs_dense_error < 0.10:
        honest_verdict = "sparse_improved_not_resolved"
    else:
        honest_verdict = "sparse_no_improvement"

    _log.info(
        "sparse_vs_dense_error=%.4f, retro_057_resolved=%s, verdict=%s",
        sparse_vs_dense_error,
        retro_057_resolved,
        honest_verdict,
    )

    # ------------------------------------------------------------------
    # LowRank baseline comparison (Exp 532 result if available)
    # ------------------------------------------------------------------
    lowrank_baseline_error = _load_lowrank_baseline()
    if lowrank_baseline_error is not None:
        _log.info("LowRank baseline error (Exp 532): %.6f", lowrank_baseline_error)
    else:
        _log.info("LowRank baseline result not found; omitting comparison")

    # ------------------------------------------------------------------
    # Build and write artifact
    # ------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "schema": "carnot.lowrank_kaem_sparse.v1",
            "n_vars": n_vars,
            "energy_accuracy_dense": float(energy_accuracy_dense),
            "energy_accuracy_sparse": float(energy_accuracy_sparse),
            "sparse_vs_dense_error": float(sparse_vs_dense_error),
            "best_top_k_fraction": float(best_fraction),
            "retro_057_resolved": retro_057_resolved,
            "lowrank_baseline_error": lowrank_baseline_error,
            "honest_verdict": honest_verdict,
            "sweep_results": sweep_results,
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_637_lowrank_kaem_sparse.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", out_path)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
