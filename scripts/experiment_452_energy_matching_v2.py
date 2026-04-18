#!/usr/bin/env python3
"""Experiment 452: Energy Matching v2 — RETRO-030 Closure (Atomic Write + File Verification).

**Researcher summary (RETRO-030):**
    Exp 446 ran the Langevin + Energy Matching comparison for ContinuousEBM and exited
    with status 0, but produced no result file.  Root cause: an exception occurred
    after ``open()`` but before ``json.dump()`` completed, leaving nothing on disk.
    The watchdog did not catch this because it checked exit code, not file existence.

    This experiment re-runs the identical energy matching logic with three safeguards:
    1. ``AtomicResultWriter`` — writes to ``.tmp`` then ``os.rename()``, so the
       final path is either absent or complete JSON, never a partial write.
    2. ``writer.verify_exists()`` called after write — raises ``RuntimeError`` if the
       file is still absent (catches filesystem-level failures that pass os.rename).
    3. ``ExperimentTimeoutWatchdog`` as a context manager — the watchdog always stops
       cleanly even if an exception propagates from the body.

**Phase 3 tracking:**
    If ``results/experiment_446_energy_matching.json`` is present, the energy loss
    from Exp 446 is loaded as a baseline and ``phase3_improvement`` is computed.
    If absent (Exp 446 never wrote its result — RETRO-030), improvement is null.

CPU-only.  No GPU required.  Timeout: 30 minutes.

Outputs: results/experiment_452_energy_matching_v2.json

Spec: REQ-INFRA-031, REQ-INFRA-032, REQ-KONA-002, REQ-KONA-003,
      SCENARIO-INFRA-039, SCENARIO-INFRA-040,
      SCENARIO-KONA-003, SCENARIO-KONA-004, SCENARIO-KONA-005
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() FIRST — belt-and-suspenders RETRO-022 fix.
# Must come before any other carnot import so CARNOT_FORCE_LIVE propagates.
# ---------------------------------------------------------------------------
from carnot.pipeline.env_autofix import apply_env_autofix

_autofix = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------
import json

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402

from carnot.phase3.continuous_ebm import (  # noqa: E402
    ContinuousEBM,
    compare_samplers,
    fit_continuous_ebm,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

EXP_ID = 452
TITLE = "Energy Matching v2 — RETRO-030 Closure (Atomic Write)"
RESULT_PATH = "results/experiment_452_energy_matching_v2.json"
EXP_446_RESULT_PATH = "results/experiment_446_energy_matching.json"
TIMEOUT_MINUTES = 30

# Problem parameters — MUST match Exp 446 / 435a for fair comparison
N_VARS = 10
COUPLING_DENSITY = 0.3
ISING_SEED = 42

# Evaluation parameters
N_TRIALS = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sparse_ising(n: int, density: float, seed: int):
    """Build a random n-variable Ising model — same construction as Exp 435a/446.

    Why the same seed and density: reproducibility.  Exp 435a established the
    baseline L2=2.69; Exp 446 and this experiment must use identical problem
    parameters so that sampler comparisons are valid.
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


def _simulated_annealing_ground_state(ising, seed: int = 1) -> np.ndarray:
    """Discrete ground state via simulated annealing — same as Exp 435a/446.

    Why simulated annealing: it provides a deterministic discrete reference state
    that the continuous samplers are measured against.  Keeping the algorithm and
    seed identical ensures the L2 comparisons are apples-to-apples.
    """
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


def _load_exp446_baseline() -> float | None:
    """Load the energy_matching_l2 from Exp 446 result, or None if unavailable.

    Why this can be None: Exp 446 never wrote its result file (RETRO-030).  This
    function handles that case gracefully so Exp 452 can complete even without the
    baseline, producing phase3_improvement=null in the artifact.
    """
    path = _REPO_ROOT / EXP_446_RESULT_PATH
    if not path.exists():
        _log.info("Exp 446 result not found at %s — phase3_improvement will be null", path)
        return None
    try:
        data = json.loads(path.read_text())
        em_l2 = data.get("energy_matching_l2") or data.get("energy_matching", {}).get("mean_l2")
        if em_l2 is not None:
            _log.info("Loaded Exp 446 energy_matching_l2=%.4f as Phase 3 baseline", em_l2)
        return float(em_l2) if em_l2 is not None else None
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        _log.warning("Could not parse Exp 446 result (%s) — phase3_improvement will be null", exc)
        return None


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 452: energy matching v2 with atomic write and file verification."""

    output_path = _REPO_ROOT / RESULT_PATH

    # Pre-run check: note whether file already exists (for logging only).
    pre_exists = output_path.exists()
    _log.info("Pre-run check: result file %s (%s)", output_path, "exists" if pre_exists else "absent")

    # Step 2: Watchdog as context manager — hard 30-minute wall-clock cap.
    # The context manager ensures watchdog.stop() is called even on exceptions.
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(output_path),
    )

    # Step 3: ExperimentTemplate — CPU mode (no GPU required).
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=RESULT_PATH,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    watchdog.start()
    try:
        # Step 4: Build Ising model (same seed as 435a/446).
        _log.info(
            "Building %d-variable sparse Ising (seed=%d, density=%.2f)",
            N_VARS, ISING_SEED, COUPLING_DENSITY,
        )
        ising = _build_sparse_ising(N_VARS, COUPLING_DENSITY, ISING_SEED)

        # Step 5: Discrete reference state via simulated annealing.
        _log.info("Sampling Ising ground state via simulated annealing...")
        ising_ground_state = _simulated_annealing_ground_state(ising, seed=1)
        ising_energy = float(
            -0.5 * ising_ground_state @ ising.coupling @ ising_ground_state
            - ising.bias @ ising_ground_state
        )

        # Step 6: Fit ContinuousEBM.
        model = fit_continuous_ebm(ising)
        _log.info("ContinuousEBM fitted: %d variables", model.variables)

        # Step 7: Run compare_samplers (gradient descent, Langevin, Energy Matching).
        _log.info("Running compare_samplers with %d trials per sampler...", N_TRIALS)
        sampler_results = compare_samplers(model, ising_ground_state, n_trials=N_TRIALS)

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

        # Step 8: Phase 3 improvement tracking.
        exp446_baseline = _load_exp446_baseline()
        energy_loss = em["mean_l2"]
        if exp446_baseline is not None:
            phase3_improvement = float(exp446_baseline - energy_loss)
            _log.info(
                "Phase 3 improvement: %.4f (Exp 446 baseline=%.4f, this run=%.4f)",
                phase3_improvement, exp446_baseline, energy_loss,
            )
        else:
            phase3_improvement = None
            _log.info("Phase 3 improvement: null (Exp 446 result unavailable — RETRO-030)")

        # Step 9: Honest verdict.
        best_l2 = min(lan["mean_l2"], em["mean_l2"])
        if best_l2 < 0.5:
            honest_verdict = "retro_030_closed"
        elif best_l2 < 1.0:
            honest_verdict = "retro_030_closed_partial_improvement"
        else:
            honest_verdict = "retro_030_closed_no_improvement"

        # Step 10: Build artifact.
        artifact = tmpl.build_result(
            {
                "schema": "carnot.energy_matching.v2",
                "atomic_write": True,
                "retro_030_resolved": True,
                "n_vars": N_VARS,
                "coupling_density": COUPLING_DENSITY,
                "ising_seed": ISING_SEED,
                "n_trials": N_TRIALS,
                "ising_energy": ising_energy,
                "ising_ground_state": ising_ground_state.tolist(),
                "gradient_descent": gd,
                "langevin": lan,
                "energy_matching": em,
                "energy_loss": energy_loss,
                "phase3_improvement": phase3_improvement,
                "best_sampler": best_sampler,
                "honest_verdict": honest_verdict,
                "phase": "phase3_seed",
                "env_autofix": {
                    "gpu_detected": _autofix.gpu_detected,
                    "auto_fix_applied": _autofix.auto_fix_applied,
                },
                "note": (
                    "Exp 452: RETRO-030 closure.  Re-runs Exp 446 energy matching logic "
                    "with AtomicResultWriter (write-to-tmp + rename) and verify_exists() "
                    "assertion.  CPU-only.  Phase 3 seed work."
                ),
            },
            status="success",
        )

        # Step 11: Write result atomically (RETRO-030 fix — REQ-INFRA-031).
        writer = AtomicResultWriter(str(output_path))
        writer.write(artifact)
        _log.info("Atomic write complete: %s", output_path)

        # Step 12: Verify file exists (REQ-INFRA-032).
        # A missing file here means os.rename() succeeded but the FS did not persist
        # it — extremely rare, but the check ensures the experiment self-reports failure
        # rather than silently succeeding with no deliverable.
        if not writer.verify_exists():
            raise RuntimeError(
                f"RETRO-030 guard: result file not found after atomic write: {output_path}. "
                "This is a filesystem-level failure — the rename appeared to succeed "
                "but the file is absent.  Experiment 452 is BLOCKED."
            )

        _log.info("verify_exists() passed — RETRO-030 resolved.  Result at %s", output_path)

    finally:
        watchdog.stop()


if __name__ == "__main__":
    main()
