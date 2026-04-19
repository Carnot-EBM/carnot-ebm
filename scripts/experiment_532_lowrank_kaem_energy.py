#!/usr/bin/env python3
"""Experiment 532: LowRankKAEMEnergy — SVD-based dimensionality reduction before spline computation.

**Researcher summary:**
    arXiv 2604.04384 (April 2026) demonstrates that transformer logit matrices are empirically
    low-rank: 90% of total variance is captured by only 2-11 singular components.  KAEMEnergy
    (Exp 447) operates on the full-dimensional energy space.  A rank-k KAEMEnergy with k=11
    should achieve 90%+ AUROC at 10-100x fewer KAN spline computations than the full-rank model.

    This experiment validates the low-rank hypothesis by training LowRankKAEMEnergy at
    k in [2, 4, 8, 11, 20, 50] on synthetic logit-like data and measuring AUROC and
    evaluation speed vs the full-rank baseline.

**Expected outcome:**
    - At k=11: AUROC within 95% of full-rank baseline, with speedup >= 2x.
    - At k=50 (full rank for n_vars=50): matches full-rank baseline exactly.
    - optimal_k (minimum k for 95% AUROC recovery) <= 11 confirms the paper's claim.

**Outputs:**
    results/experiment_532_lowrank_kaem_energy.json

Spec: REQ-SAMPLE-027, REQ-SAMPLE-028, SCENARIO-SAMPLE-041, SCENARIO-SAMPLE-042, SCENARIO-SAMPLE-043
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any CUDA import (RETRO-022 fix)
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json
import logging
import time

import jax.numpy as jnp
import numpy as np

from carnot.models.kaem_energy import KAEMEnergy
from carnot.models.lowrank_kaem import LowRankKAEMEnergy
from carnot.pipeline.deliverable_guard import DeliverableGuard
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 532
EXP_TITLE = "LowRankKAEMEnergy — SVD dimensionality reduction for KAN fast-path (arXiv 2604.04384)"
DELIVERABLE = "results/experiment_532_lowrank_kaem_energy.json"
N_TRAIN = 500
N_VARS = 50
TRUE_RANK = 11
N_HELD_OUT = 100
N_EVAL_REPEATS = 200  # repeats for timing measurement
K_VALUES = [2, 4, 8, 11, 20, 50]


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_low_rank_logit_data(
    n_samples: int,
    n_vars: int,
    true_rank: int,
    noise_scale: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic logit-like data with known rank-true_rank structure.

    The signal lives in a true_rank-dimensional subspace (mimicking the low-rank
    structure observed in transformer logit matrices by arXiv 2604.04384).
    A small Gaussian noise term is added to prevent perfectly zero residual variance.

    Data is normalised to [-1, 1] to match KAEM spline domain.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((n_vars, true_rank)).astype(np.float32)
    coef = rng.standard_normal((n_samples, true_rank)).astype(np.float32)
    signal = coef @ basis.T  # (n_samples, n_vars)
    noise = rng.standard_normal((n_samples, n_vars)).astype(np.float32) * noise_scale
    data = signal + noise
    mx = float(np.max(np.abs(data))) + 1e-6
    return (data / mx).astype(np.float32)


def _compute_auroc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """Compute AUROC given scores for positive and negative classes.

    Uses the Wilcoxon-Mann-Whitney U statistic: for KAEM, lower energy = positive class.
    We negate the energy so that higher score = more likely positive.
    """
    n_pos = len(scores_pos)
    n_neg = len(scores_neg)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Count pairs where positive has lower energy (higher -energy score) than negative
    count = 0
    for p in scores_pos:
        count += int(np.sum(p < scores_neg))
        count += int(np.sum(p == scores_neg)) // 2
    return float(count) / (n_pos * n_neg)


def _evaluate_model(model, x_pos: np.ndarray, x_neg: np.ndarray) -> tuple[float, float]:
    """Return (auroc, eval_ms_per_sample) for a fitted model.

    Uses negated energy as the score: lower energy = positive class.
    Timing is measured over N_EVAL_REPEATS energy calls to get stable ms/sample.
    """
    # Compute energy scores
    energies_pos = np.array([float(model.energy(jnp.array(x))) for x in x_pos])
    energies_neg = np.array([float(model.energy(jnp.array(x))) for x in x_neg])

    # AUROC: lower energy should predict positive class
    auroc = _compute_auroc(energies_pos, energies_neg)

    # Timing: measure energy() on held-out samples
    x_sample = jnp.array(x_pos[0])
    # Warm up (JAX tracing)
    _ = model.energy(x_sample)
    t0 = time.perf_counter()
    for _ in range(N_EVAL_REPEATS):
        model.energy(x_sample)
    eval_ms = (time.perf_counter() - t0) * 1000.0 / N_EVAL_REPEATS

    return auroc, eval_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=25)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    guard = DeliverableGuard(str(_REPO_ROOT / DELIVERABLE))

    _log.info("Exp %d: generating synthetic low-rank logit data (n_vars=%d, rank=%d)",
              EXP_ID, N_VARS, TRUE_RANK)

    # Generate train/held-out data with rank-11 signal structure
    rng = np.random.default_rng(42)
    all_data = _make_low_rank_logit_data(
        n_samples=N_TRAIN + N_HELD_OUT * 2,
        n_vars=N_VARS,
        true_rank=TRUE_RANK,
        noise_scale=0.02,
        seed=42,
    )
    train_data = all_data[:N_TRAIN]
    # Positive class: high-norm samples (above median norm = "valid" distribution)
    held_out = all_data[N_TRAIN:]
    norms = np.linalg.norm(held_out, axis=1)
    median_norm = float(np.median(norms))
    x_pos = held_out[norms >= median_norm][:N_HELD_OUT]  # (100, 50)
    x_neg = held_out[norms < median_norm][:N_HELD_OUT]   # (100, 50)
    # Ensure we have enough samples
    n_pos = min(len(x_pos), N_HELD_OUT)
    n_neg = min(len(x_neg), N_HELD_OUT)
    x_pos = x_pos[:n_pos]
    x_neg = x_neg[:n_neg]

    train_jax = jnp.array(train_data)

    _log.info("Training full-rank KAEMEnergy baseline (n_vars=%d)...", N_VARS)
    full_model = KAEMEnergy(n_vars=N_VARS, n_hidden=16)
    full_model.fit(train_jax, n_epochs=50)
    full_auroc, full_eval_ms = _evaluate_model(full_model, x_pos, x_neg)
    _log.info("Full-rank AUROC=%.4f, eval_ms=%.4f", full_auroc, full_eval_ms)

    # Run low-rank experiments at each k
    results_by_k = []
    optimal_k = None
    speedup_at_optimal_k = None

    for k in K_VALUES:
        _log.info("Training LowRankKAEMEnergy k=%d...", k)
        lr_model = LowRankKAEMEnergy(n_vars=N_VARS, k=k)
        lr_model.fit(train_jax, n_epochs=50)
        auroc, eval_ms = _evaluate_model(lr_model, x_pos, x_neg)

        auroc_vs_full = (auroc / full_auroc * 100.0) if full_auroc > 0 else 0.0
        speedup = full_eval_ms / eval_ms if eval_ms > 0 else float("inf")

        _log.info("k=%d: AUROC=%.4f (%.1f%% of full-rank), eval_ms=%.4f, speedup=%.2fx",
                  k, auroc, auroc_vs_full, eval_ms, speedup)

        results_by_k.append({
            "k": k,
            "auroc": float(auroc),
            "eval_ms": float(eval_ms),
            "auroc_vs_fullrank_pct": float(auroc_vs_full),
            "speedup": float(speedup),
        })

        # Track optimal_k: minimum k where AUROC >= 95% of full-rank
        if optimal_k is None and auroc >= 0.95 * full_auroc:
            optimal_k = k
            speedup_at_optimal_k = speedup

    if optimal_k is None:
        optimal_k = K_VALUES[-1]
        speedup_at_optimal_k = results_by_k[-1]["speedup"]

    lowrank_viable = optimal_k <= 11
    honest_verdict = "lowrank_viable" if lowrank_viable else "no_compression"

    artifact = tmpl.build_result(
        {
            "schema": "carnot.lowrank_kaem.v1",
            "full_rank_auroc": float(full_auroc),
            "full_rank_eval_ms": float(full_eval_ms),
            "results_by_k": results_by_k,
            "optimal_k": int(optimal_k),
            "speedup_at_optimal_k": float(speedup_at_optimal_k),
            "lowrank_viable": bool(lowrank_viable),
            "honest_verdict": honest_verdict,
            "n_train": N_TRAIN,
            "n_vars": N_VARS,
            "true_rank": TRUE_RANK,
        },
        status="success",
    )

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Wrote %s", output_path)
    _log.info("honest_verdict=%s, optimal_k=%d, speedup_at_optimal_k=%.2fx",
              honest_verdict, optimal_k, speedup_at_optimal_k)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
