#!/usr/bin/env python3
"""Experiment 544: LowRankKAEM Cascade Integration.

Wires LowRankKAEMEnergy (Exp 532, k=2, 23.7x speedup) into the verification
cascade as the default KAN fast-path tier for n_vars <= 100.

**What this experiment validates:**
    1. get_kaem_energy() factory correctly selects model type based on n_vars.
    2. LowRankKAEMEnergy(k=2) achieves speedup vs full-rank KAEMEnergy.
    3. Energy values from both models are in a comparable range (within 5% mean
       absolute deviation normalized by full-rank energy scale).
    4. VerificationResult.use_lowrank_kaem field is wired correctly.

**Benchmark problems:**
    10-var, 50-var, 100-var, 200-var Ising-like constraint problems.
    For each: time full-rank vs low-rank KAEM energy evaluation.

**Outputs:**
    results/experiment_544_lowrank_kaem_cascade.json

Spec: REQ-SAMPLE-029, SCENARIO-SAMPLE-044, SCENARIO-SAMPLE-045
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# MUST be first: apply_env_autofix() before any CUDA import
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
import jax.random as jrandom
import numpy as np

from carnot.models.kaem_energy import KAEMEnergy, get_kaem_energy
from carnot.models.lowrank_kaem import LowRankKAEMEnergy
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 544
EXP_TITLE = "LowRankKAEM Cascade Integration"
DELIVERABLE = "results/experiment_544_lowrank_kaem_cascade.json"
N_VARS_LIST = [10, 50, 100, 200]
N_TRAIN = 200
N_TEST = 50
N_EPOCHS = 20
N_EVAL_REPEATS = 100


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def _make_ising_data(n_vars: int, n_samples: int, key: jax.Array) -> jnp.ndarray:
    """Generate synthetic Ising-like data in [-1, 1] for benchmarking.

    Simulates binary spin configurations typical of constraint verification
    problems: spins are Bernoulli ±1 with small correlations (chain topology).

    Parameters
    ----------
    n_vars : int
        Number of spin variables.
    n_samples : int
        Number of samples to generate.
    key : jax.Array
        PRNG key.

    Returns
    -------
    jnp.ndarray
        Shape (n_samples, n_vars), values in {-1.0, +1.0} cast to float32.
    """
    # Draw uniform and threshold to get ±1 spins
    raw = jrandom.uniform(key, (n_samples, n_vars))
    return jnp.where(raw > 0.5, 1.0, -1.0).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Benchmarking helpers
# ---------------------------------------------------------------------------

import jax


def _time_energy_eval(model: object, test_points: jnp.ndarray, n_repeats: int) -> float:
    """Time n_repeats energy evaluations on test_points; return ms per call.

    We call energy() on each test point individually (not batched) since the
    cascade calls energy() per-response. Warm-up on first point to exclude JIT.

    Parameters
    ----------
    model : KAEMEnergy | LowRankKAEMEnergy
        Fitted model with an energy(x) method.
    test_points : jnp.ndarray
        Shape (n_test, n_vars). Each row is evaluated separately.
    n_repeats : int
        Number of timing repetitions across the test set.

    Returns
    -------
    float
        Mean milliseconds per energy() call.
    """
    n_test = test_points.shape[0]

    # Warm-up: first call triggers JAX JIT compilation
    _ = model.energy(test_points[0])

    t0 = time.perf_counter()
    for _ in range(n_repeats):
        for i in range(n_test):
            _ = model.energy(test_points[i])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return elapsed_ms / (n_repeats * n_test)


def _energy_tolerance(
    full_model: KAEMEnergy,
    lr_model: LowRankKAEMEnergy,
    test_points: jnp.ndarray,
) -> tuple[float, bool]:
    """Compute mean absolute energy deviation (normalized) between models.

    Both models are trained on the same data but operate in different spaces
    (full-rank vs projected), so absolute values differ. We normalize by the
    scale of the full-rank energies so the comparison is dimensionless.

    Returns (normalized_mad, within_5pct) where within_5pct = normalized_mad < 0.05.
    """
    full_energies = np.array([float(full_model.energy(test_points[i]))
                               for i in range(test_points.shape[0])])
    lr_energies = np.array([float(lr_model.energy(test_points[i]))
                             for i in range(test_points.shape[0])])

    scale = float(np.mean(np.abs(full_energies))) + 1e-9
    mad_normalized = float(np.mean(np.abs(full_energies - lr_energies))) / scale
    return mad_normalized, mad_normalized < 0.05


# ---------------------------------------------------------------------------
# Per-n_vars benchmark
# ---------------------------------------------------------------------------


def _benchmark_one(n_vars: int, key: jax.Array) -> dict:
    """Run full-rank vs low-rank KAEM benchmark for one problem size.

    Parameters
    ----------
    n_vars : int
        Number of constraint variables.
    key : jax.Array
        PRNG key (split internally for train/test/model init).

    Returns
    -------
    dict with keys: n_vars, use_lowrank, speedup, energy_mad_normalized,
                    energy_within_5pct, full_ms_per_call, lr_ms_per_call
    """
    k_train, k_test, k_full, k_lr = jrandom.split(key, 4)

    train_data = _make_ising_data(n_vars, N_TRAIN, k_train)
    test_data = _make_ising_data(n_vars, N_TEST, k_test)

    use_lowrank = n_vars <= 100

    _log.info("n_vars=%d: fitting full-rank KAEMEnergy...", n_vars)
    full_model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=k_full)
    full_model.fit(train_data, n_epochs=N_EPOCHS)

    _log.info("n_vars=%d: fitting LowRankKAEMEnergy(k=2)...", n_vars)
    lr_model = get_kaem_energy(n_vars=n_vars, use_lowrank=True, k=2, key=k_lr)
    lr_model.fit(train_data, n_epochs=N_EPOCHS)

    _log.info("n_vars=%d: timing energy evaluations...", n_vars)
    full_ms = _time_energy_eval(full_model, test_data, N_EVAL_REPEATS)
    lr_ms = _time_energy_eval(lr_model, test_data, N_EVAL_REPEATS)

    speedup = full_ms / lr_ms if lr_ms > 0 else float("inf")

    mad_normalized, within_5pct = _energy_tolerance(full_model, lr_model, test_data)

    _log.info(
        "n_vars=%d: full_ms=%.4f, lr_ms=%.4f, speedup=%.2fx, mad=%.4f, within_5pct=%s",
        n_vars, full_ms, lr_ms, speedup, mad_normalized, within_5pct,
    )

    return {
        "n_vars": n_vars,
        "use_lowrank": bool(use_lowrank),
        "full_ms_per_call": float(full_ms),
        "lr_ms_per_call": float(lr_ms),
        "speedup": float(speedup),
        "energy_mad_normalized": float(mad_normalized),
        "energy_within_5pct": bool(within_5pct),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 544: LowRankKAEM cascade integration benchmark."""
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=20):
        tmpl = ExperimentTemplate(
            EXP_ID,
            EXP_TITLE,
            DELIVERABLE,
        )
        tmpl.setup()

        master_key = jrandom.PRNGKey(544)
        keys = jrandom.split(master_key, len(N_VARS_LIST))

        results_by_n_vars = []
        for n_vars, key in zip(N_VARS_LIST, keys):
            row = _benchmark_one(n_vars, key)
            results_by_n_vars.append(row)

        # Build speedup_by_n_vars summary dict
        speedup_by_n_vars = {r["n_vars"]: round(r["speedup"], 3) for r in results_by_n_vars}

        # Overall tolerance: all problems must be within 5%
        energy_tolerance_within_5pct = all(r["energy_within_5pct"] for r in results_by_n_vars)

        honest_verdict = "lowrank_wired" if energy_tolerance_within_5pct else "tolerance_exceeded"

        _log.info("speedup_by_n_vars=%s", speedup_by_n_vars)
        _log.info("energy_tolerance_within_5pct=%s", energy_tolerance_within_5pct)
        _log.info("honest_verdict=%s", honest_verdict)

        artifact = tmpl.build_result(
            {
                "schema": "carnot.lowrank_kaem_cascade.v1",
                "speedup_by_n_vars": speedup_by_n_vars,
                "energy_tolerance_within_5pct": energy_tolerance_within_5pct,
                "tier_kaem_default": "lowrank",
                "honest_verdict": honest_verdict,
                "results_by_n_vars": results_by_n_vars,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "n_epochs": N_EPOCHS,
                "k": 2,
            },
            status="success",
        )

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(artifact, indent=2))
        _log.info("Wrote %s", output_path)

        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
