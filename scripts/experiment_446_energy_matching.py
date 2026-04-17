#!/usr/bin/env python3
"""Experiment 446: Langevin Dynamics + Energy Matching for ContinuousEBM.

**Researcher summary:**
    Exp 435a showed gradient descent alone achieves L2=2.69 vs Ising ground state
    (partial_match only).  This experiment applies two new algorithms:

    1. Langevin dynamics (arXiv 2506.15121 — Generative Thermodynamic Computing):
       x_{t+1} = x_t - lr * grad_E(x_t) + noise * sqrt(2*lr) * eps_t
       Thermal noise escapes local minima that trap gradient descent.

    2. Energy Matching normalised gradient flow (arXiv 2504.10612, NeurIPS 2025):
       x = x - step * grad_E / ||grad_E||
       Constant-speed convergence regardless of energy landscape curvature.

    Target: L2 < 0.5 for at least one sampler (vs baseline L2=2.69).

**Why this is Phase 3 seed work:**
    Langevin dynamics is the core sampling mechanism for Kona-style continuous
    reasoning.  If we can show that Langevin finds a closer approximation to the
    Ising ground state than gradient descent, it validates the foundation for
    non-autoregressive inference over continuous latent spaces.

CPU-only.  No GPU required.  Timeout: 25 minutes.

Outputs: results/experiment_446_energy_matching.json

Spec: REQ-KONA-002, REQ-KONA-003, SCENARIO-KONA-003, SCENARIO-KONA-004,
      SCENARIO-KONA-005
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() FIRST — resolves RETRO-022 env propagation issue.
# Must be called before any other carnot import to ensure CARNOT_FORCE_LIVE is set.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import numpy as np

# Add repo root to path for experiment_template import
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.phase3.continuous_ebm import (  # noqa: E402
    ContinuousEBM,
    compare_samplers,
    fit_continuous_ebm,
    sample_continuous,
)
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.samplers.parallel_ising import AnnealingSchedule, parallel_sample_states  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 446
TITLE = "Langevin Dynamics + Energy Matching for ContinuousEBM (Exp 446)"
RESULT_PATH = "results/experiment_446_energy_matching.json"
TIMEOUT_MINUTES = 25

# Problem parameters — MUST match Exp 435a for fair comparison
N_VARS = 10
COUPLING_DENSITY = 0.3
ISING_SEED = 42

# Evaluation parameters
N_TRIALS = 20  # independent trials per sampler

# Exp 435a baseline for honest comparison
BASELINE_L2 = 2.6943395932596865


# ---------------------------------------------------------------------------
# Build Ising model (same as Exp 435a)
# ---------------------------------------------------------------------------


def _build_sparse_ising(n: int, density: float, seed: int):
    """Build a random n-variable Ising model — same construction as Exp 435a.

    Returns a simple namespace with .coupling and .bias attributes so
    fit_continuous_ebm() can consume it.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    mask = np.triu(mask, k=1)
    mask = mask | mask.T
    J_raw = rng.uniform(-1.0, 1.0, (n, n)) * mask
    J = (J_raw + J_raw.T) / 2.0
    h = rng.uniform(-0.5, 0.5, n)

    class _Ising:
        coupling = J
        bias = h

    return _Ising()


# ---------------------------------------------------------------------------
# Ising ground state via simulated annealing
# ---------------------------------------------------------------------------


def _simulated_annealing_ground_state(ising, seed: int = 1) -> np.ndarray:
    """Discrete ground state via simulated annealing — same as Exp 435a."""
    rng = np.random.default_rng(seed)
    n = ising.coupling.shape[0]
    J, h = ising.coupling, ising.bias
    state = rng.choice([-1.0, 1.0], size=n)
    best = state.copy()
    best_e = float(-0.5 * state @ J @ state - h @ state)
    n_steps = 10_000
    for step in range(n_steps):
        T = 2.0 * (0.01 / 2.0) ** (step / n_steps)
        i = int(rng.integers(n))
        delta = 2.0 * state[i] * (J[i] @ state + h[i])
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            state[i] = -state[i]
        e = float(-0.5 * state @ J @ state - h @ state)
        if e < best_e:
            best_e = e
            best = state.copy()
    _log.info("Ising ground state energy: %.4f  state: %s", best_e, best.tolist())
    return best


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 446: compare gradient descent, Langevin, and Energy Matching."""

    # Step 2: Watchdog — hard 25-minute wall-clock cap
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / RESULT_PATH),
    )
    watchdog.start()

    # Step 3: ExperimentTemplate — CPU mode (no GPU required)
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=RESULT_PATH,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    try:
        # Step 4: Build 10-variable sparse Ising (same seed as Exp 435a)
        _log.info("Building 10-variable sparse Ising (seed=%d, density=%.2f)", ISING_SEED, COUPLING_DENSITY)
        ising = _build_sparse_ising(N_VARS, COUPLING_DENSITY, ISING_SEED)

        # Step 5: Sample Ising ground state (discrete reference)
        _log.info("Sampling Ising ground state via simulated annealing...")
        ising_ground_state = _simulated_annealing_ground_state(ising, seed=1)
        ising_energy = float(-0.5 * ising_ground_state @ ising.coupling @ ising_ground_state
                             - ising.bias @ ising_ground_state)

        # Step 6: Fit ContinuousEBM from Ising J/h
        model = fit_continuous_ebm(ising)
        _log.info("ContinuousEBM fitted: %d variables", model.variables)

        # Step 7: Run compare_samplers (gradient descent, Langevin, Energy Matching)
        _log.info("Running compare_samplers with %d trials per sampler...", N_TRIALS)
        sampler_results = compare_samplers(model, ising_ground_state, n_trials=N_TRIALS)

        # Step 8: Extract per-sampler results
        gd = sampler_results["gradient_descent"]
        lan = sampler_results["langevin"]
        em = sampler_results["energy_matching"]
        best_sampler = sampler_results["best_sampler"]

        _log.info(
            "gradient_descent: mean_l2=%.4f std=%.4f sign=%.3f",
            gd["mean_l2"], gd["std_l2"], gd["mean_sign_agreement"],
        )
        _log.info(
            "langevin:         mean_l2=%.4f std=%.4f sign=%.3f",
            lan["mean_l2"], lan["std_l2"], lan["mean_sign_agreement"],
        )
        _log.info(
            "energy_matching:  mean_l2=%.4f std=%.4f sign=%.3f",
            em["mean_l2"], em["std_l2"], em["mean_sign_agreement"],
        )
        _log.info("best_sampler: %s", best_sampler)

        # Step 9: Compute honest verdict
        langevin_l2 = lan["mean_l2"]
        energy_matching_l2 = em["mean_l2"]
        best_l2 = min(langevin_l2, energy_matching_l2)

        if best_l2 < 0.5:
            honest_verdict = "continuous_improved"
        elif best_l2 < 1.0:
            honest_verdict = "partial_improvement"
        else:
            honest_verdict = "no_improvement"

        _log.info(
            "honest_verdict: %s  (baseline_l2=%.4f, best_new_l2=%.4f)",
            honest_verdict, BASELINE_L2, best_l2,
        )

        # Step 10: Build artifact
        artifact = tmpl.build_result(
            {
                "schema": "carnot.energy_matching.v1",
                "baseline_l2": BASELINE_L2,
                "baseline_experiment": "435a",
                "n_vars": N_VARS,
                "coupling_density": COUPLING_DENSITY,
                "ising_seed": ISING_SEED,
                "n_trials": N_TRIALS,
                "ising_energy": ising_energy,
                "ising_ground_state": ising_ground_state.tolist(),
                "gradient_descent": gd,
                "langevin": lan,
                "energy_matching": em,
                "langevin_l2": langevin_l2,
                "energy_matching_l2": energy_matching_l2,
                "best_sampler": best_sampler,
                "honest_verdict": honest_verdict,
                "phase": "phase3_seed",
                "note": (
                    "Exp 446: Langevin dynamics + Energy Matching vs gradient descent baseline. "
                    "Algorithms: Langevin (arXiv 2506.15121), Energy Matching (arXiv 2504.10612). "
                    "CPU-only, Phase 3 seed work."
                ),
                "env_autofix": {
                    "gpu_detected": _autofix.gpu_detected,
                    "auto_fix_applied": _autofix.auto_fix_applied,
                },
            },
            status="success",
        )

        # Write result
        output_path = _REPO_ROOT / RESULT_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)
        _log.info("Result written to %s", output_path)

    finally:
        watchdog.stop()


if __name__ == "__main__":
    main()
