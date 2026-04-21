#!/usr/bin/env python3
"""Experiment 634: Multilevel KAN KAEMEnergy — knot refinement schedule benchmark.

**Context (arXiv 2603.04827, March 2026):**
    Multilevel training starts at coarse knot resolution (K=16), trains to
    convergence, then interpolates weights analytically to K=32, K=64, K=128.
    Each coarser level finds the global energy basin that would be hard to find
    at fine resolution, giving the finer level a warm start near the optimum.

    KAEMEnergy (Exp 447) starts training directly at K=256 (n_hidden default),
    which may be over-parameterised for early training steps.  This experiment
    benchmarks whether starting coarse and refining improves accuracy.

**Benchmark design:**
    - Synthetic ground truth: 20-variable energy = sum_i [sin(3*x_i) + x_i^2]
    - Standard baseline: KAEMEnergy(n_hidden=128) trained for 80 epochs (same total budget)
    - Multilevel: schedule=[16,32,64,128], epochs_per_level=20 (4*20 = 80 epochs total)
    - Accuracy metric: mean absolute error between model energy and ground truth
      on 500 held-out test points (lower = better)
    - accuracy_improvement > 0.01 → multilevel_wins; else multilevel_no_improvement

Spec: REQ-SAMPLE-038, SCENARIO-SAMPLE-063, SCENARIO-SAMPLE-064
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
import jax.random as jrandom
import numpy as np

from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.models.kaem_energy import KAEMEnergy
from carnot.training.multilevel_kan_trainer import MultilevelKAEMTrainer
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ground-truth energy for benchmark
# ---------------------------------------------------------------------------


def ground_truth_energy(x: np.ndarray) -> float:
    """Compute sum_i [sin(3*x_i) + x_i^2] for a 1D sample vector.

    This is a non-trivial but smooth multivariate energy that has known
    analytic structure: each variable contributes independently with a
    sinusoidal modulation on top of a quadratic well.  A good model should
    learn the per-variable marginal shape accurately.

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
    """Generate training data by sampling from the ground-truth energy via MCMC.

    We use rejection sampling relative to a uniform proposal: draw x ~ U[-1,1]^n,
    accept with probability proportional to exp(-E(x)).  This produces samples
    from the correct Boltzmann distribution for benchmark training.

    Because we want training to be fast for the benchmark, we use a simplified
    approach: draw random points in [-1,1]^n_vars weighted by exp(-E(x)).

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
    # Rejection sampling from the Boltzmann distribution
    # Draw candidates from uniform, accept proportional to exp(-E)
    accepted = []
    n_candidates = n_samples * 20  # oversample to ensure enough acceptances
    candidates = rng.uniform(-1.0, 1.0, size=(n_candidates, n_vars)).astype(np.float32)
    energies = np.array([ground_truth_energy(candidates[i]) for i in range(n_candidates)])
    # Normalise energies for numerical stability
    energies_shifted = energies - np.min(energies)
    probs = np.exp(-energies_shifted)
    probs = probs / probs.sum()
    # Weighted sample without replacement
    indices = rng.choice(n_candidates, size=n_samples, replace=False, p=probs)
    accepted = candidates[indices]
    return accepted


def compute_energy_mae(model: KAEMEnergy, test_x: np.ndarray) -> float:
    """Compute mean absolute error between model energy and ground truth.

    For each test point, we compute |E_model(x) - E_gt(x)| and average.
    Lower MAE = better accuracy.  Both energies are shift-free (relative),
    so we first center both on their mean to measure shape accuracy.

    Parameters
    ----------
    model : KAEMEnergy
        Trained model to evaluate.
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

    # Center both energy scales (energy-based models are defined up to an additive constant)
    model_energies -= model_energies.mean()
    gt_energies -= gt_energies.mean()

    return float(np.mean(np.abs(model_energies - gt_energies)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 634: multilevel KAN KAEMEnergy benchmark."""
    apply_env_autofix()
    ExperimentTimeoutWatchdog(634, timeout_minutes=30)

    tmpl = ExperimentTemplate(
        634,
        "Multilevel KAN KAEMEnergy",
        "results/experiment_634_multilevel_kan_kaem.json",
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
    # Standard training baseline: K=128, 80 epochs
    # ------------------------------------------------------------------
    _log.info("Standard training: KAEMEnergy(n_hidden=128) for 80 epochs")
    std_model = KAEMEnergy(n_vars=n_vars, n_hidden=128)
    std_model.fit(train_jax, n_epochs=80)
    accuracy_standard = compute_energy_mae(std_model, test_x)
    _log.info("Standard accuracy (MAE): %.6f", accuracy_standard)

    # ------------------------------------------------------------------
    # Multilevel training: schedule=[16,32,64,128], 20 epochs per level
    # ------------------------------------------------------------------
    _log.info("Multilevel training: schedule=[16,32,64,128], epochs_per_level=20")
    trainer = MultilevelKAEMTrainer(schedule=[16, 32, 64, 128], epochs_per_level=20)
    ml_model = trainer.train(n_vars=n_vars, data=train_jax)
    accuracy_multilevel = compute_energy_mae(ml_model, test_x)
    total_epochs_multilevel = len(trainer.schedule) * trainer.epochs_per_level
    _log.info("Multilevel accuracy (MAE): %.6f (total epochs: %d)", accuracy_multilevel, total_epochs_multilevel)

    # ------------------------------------------------------------------
    # Comparison metrics
    # ------------------------------------------------------------------
    # accuracy_improvement: positive means multilevel is BETTER (lower MAE)
    # We use (standard - multilevel) / abs(standard) so positive = improvement
    if abs(accuracy_standard) > 1e-10:
        accuracy_improvement = (accuracy_standard - accuracy_multilevel) / abs(accuracy_standard)
    else:
        accuracy_improvement = 0.0

    epoch_reduction = (80 - total_epochs_multilevel) / 80  # fraction of epochs saved

    multilevel_faster = total_epochs_multilevel < 80
    honest_verdict = (
        "multilevel_wins" if accuracy_improvement > 0.01 else "multilevel_no_improvement"
    )

    _log.info(
        "accuracy_improvement=%.4f, epoch_reduction=%.4f, verdict=%s",
        accuracy_improvement,
        epoch_reduction,
        honest_verdict,
    )

    artifact = tmpl.build_result(
        {
            "n_vars": n_vars,
            "standard_n_knots": 128,
            "accuracy_standard": float(accuracy_standard),
            "accuracy_multilevel": float(accuracy_multilevel),
            "accuracy_improvement": float(accuracy_improvement),
            "epoch_reduction": float(epoch_reduction),
            "total_epochs_multilevel": int(total_epochs_multilevel),
            "multilevel_faster": multilevel_faster,
            "honest_verdict": honest_verdict,
        },
        status="success",
    )

    out_path = _REPO_ROOT / "results" / "experiment_634_multilevel_kan_kaem.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Artifact written to %s", out_path)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
