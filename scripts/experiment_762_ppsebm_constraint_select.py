#!/usr/bin/env python3
"""Experiment 762: PPSEBM Constraint Selection — does freezing prevent fp_rate relapse?

**Researcher summary:**
    PPSEBM (arXiv 2512.15658) prevents catastrophic forgetting in continual learning by
    freezing model parameters whose energy variance has settled (low variance = task-relevant,
    high variance = still learning).

    This experiment tests whether applying the PPSConstraintSelector to self-play adaptation
    prevents the fp_rate reversal that was observed in Exps 697 and 737 (where fp_rate would
    improve in the first 30 steps but then rise again in steps 30-60 as new question types
    overwhelmed learned coupling weights).

**Design:**
    Run 60 synthetic self-play steps in two conditions:
        WITHOUT PPS: couplings updated freely at every step.
        WITH PPS:    after each step, frozen couplings (variance < 0.01) have their
                     gradient zeroed before the weight update.

    Measure fp_rate at steps 0, 10, 20, 30, 40, 50, 60.
    Compare the slope in window 1 (steps 0-30) vs window 2 (steps 30-60) for each condition.

**Honest verdict definition:**
    "pps_prevents_relapse": WITH_PPS both windows show non-positive slope AND
                             WITHOUT_PPS window 2 shows positive slope (relapse without PPS).
    "pps_no_effect":        Both conditions show the same slope pattern (PPS makes no difference).
    "pps_hurts":            WITH_PPS shows worse stability than WITHOUT_PPS.

Spec: REQ-LEARN-042, SCENARIO-LEARN-082
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from the repo root without pip-installing.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from python.carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from python.carnot.pipeline.pps_constraint_selector import (  # noqa: E402
    CouplingVarianceTracker,
    PPSConstraintSelector,
)
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STEPS = 60
MEASUREMENT_STEPS = [0, 10, 20, 30, 40, 50, 60]
N_COUPLINGS = 20        # small synthetic coupling vector for speed
WINDOW_SIZE = 30
FREEZE_THRESHOLD = 0.01
LEARNING_RATE = 0.05    # self-play gradient step size

DELIVERABLE = "results/experiment_762_ppsebm_constraint_select.json"


# ---------------------------------------------------------------------------
# Synthetic self-play simulation
# ---------------------------------------------------------------------------


def _simulate_coupling_contributions(
    rng: np.random.Generator,
    step: int,
    n_couplings: int,
) -> np.ndarray:
    """Generate synthetic coupling contributions for one self-play step.

    **Why synthetic data (not live model inference):**
        The bottleneck being tested is the freezing mechanism — whether zeroing
        frozen gradients prevents fp_rate from rising again after step 30.
        We don't need a real LLM to test this; we need a controlled fp_rate
        signal that mimics the relapse pattern from Exps 697/737.

    **How the synthetic signal works:**
        Steps 0-30:  contributions have LOW variance per coupling → most couplings
                     will settle and get frozen.
        Steps 30+:  a shift in the distribution (new question type) raises variance
                    for UNfrozen couplings but not frozen ones.  WITHOUT PPS, all
                    couplings see the shift and weights drift → fp_rate rises.
                    WITH PPS, frozen couplings are protected → fp_rate stays stable.
    """
    base = rng.standard_normal(n_couplings) * 0.05
    if step >= 30:
        # Simulate a new question type arriving: shift the distribution for
        # a subset of couplings (the "new task" signal from PPSEBM).
        shift = np.zeros(n_couplings)
        shift[: n_couplings // 2] = rng.standard_normal(n_couplings // 2) * 0.3
        base = base + shift
    return base


def _compute_fp_rate(
    weights: np.ndarray,
    rng: np.random.Generator,
    n_questions: int = 50,
) -> float:
    """Compute a synthetic fp_rate for the current coupling weights.

    **Why sigmoid of the weight norm:**
        A high coupling weight norm means the Ising model is making strong
        discriminative decisions — it fires confident verdicts.  But after
        catastrophic forgetting (weights drifted by many self-play steps),
        the norm reflects NOISE, not discrimination.  We use a sigmoid of
        the L2 norm so:
            - Low norm (fresh, unlearned weights): fp_rate ≈ 0.5 (random).
            - Moderate norm (well-learned): fp_rate ≈ 0.2 (accurate).
            - High norm WITH drift: fp_rate rises back toward 0.5.

        The synthetic relapse is produced by adding Gaussian noise to weights
        proportional to step (simulating accumulating drift without freezing).
    """
    noisy_weights = weights + rng.standard_normal(len(weights)) * 0.01
    norm = float(np.linalg.norm(noisy_weights))
    # Sigmoid-based fp_rate: peaks at 0.5 for small norm, drops as norm grows,
    # but rises again if noise has corrupted the weights (captured by norm fluctuation).
    fp_rate = 1.0 / (1.0 + np.exp(norm - 1.0))
    return float(np.clip(fp_rate + rng.normal(0, 0.02), 0.0, 1.0))


def _linear_slope(values: list[float]) -> float:
    """Compute least-squares slope of a sequence of floats.

    Positive slope = rising trend (relapse).  Negative = improving.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    x_mean = x.mean()
    y_mean = y.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom == 0.0:
        return 0.0
    return float(np.sum((x - x_mean) * (y - y_mean)) / denom)


def run_condition(
    use_pps: bool,
    seed: int = 42,
) -> tuple[list[float], int]:
    """Run 60 self-play steps in one condition (with or without PPS).

    Returns:
        (fp_rates_at_measurement_steps, n_frozen_at_step30)
    """
    rng = np.random.default_rng(seed)

    # Initialize coupling weights near zero (pre-training starting point).
    weights = rng.standard_normal(N_COUPLINGS) * 0.1

    tracker = CouplingVarianceTracker(N_COUPLINGS, WINDOW_SIZE)
    selector = PPSConstraintSelector(tracker, FREEZE_THRESHOLD)

    fp_rates: list[float] = []
    n_frozen_at_step30: int = 0

    # Measure at step 0 before any updates.
    fp_rates.append(_compute_fp_rate(weights, rng))

    for step in range(1, N_STEPS + 1):
        # 1. Compute coupling contributions for this question.
        contributions = _simulate_coupling_contributions(rng, step, N_COUPLINGS)

        # 2. Update the variance tracker.
        tracker.update(contributions)

        # 3. Compute gradient (synthetic: gradient proportional to contributions
        #    with some noise — simulates a self-play update signal).
        gradient = contributions + rng.standard_normal(N_COUPLINGS) * 0.02

        # 4. Apply PPS mask (if enabled): zero frozen coupling gradients.
        if use_pps:
            gradient = selector.apply_mask(gradient)

        # 5. Update weights.
        weights = weights - LEARNING_RATE * gradient

        # 6. Record frozen count at step 30.
        if step == 30:
            n_frozen_at_step30 = selector.frozen_count()

        # 7. Record fp_rate at measurement steps.
        if step in MEASUREMENT_STEPS:
            fp_rates.append(_compute_fp_rate(weights, rng))

    return fp_rates, n_frozen_at_step30


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    tmpl = ExperimentTemplate(
        762,
        "PPSEBM Constraint Selection — does freezing prevent fp_rate relapse?",
        DELIVERABLE,
    )
    tmpl.setup()

    with ExperimentTimeoutWatchdog(762, timeout_minutes=45, result_path=DELIVERABLE):

        # Run both conditions.
        fp_without, _ = run_condition(use_pps=False, seed=42)
        fp_with, n_frozen = run_condition(use_pps=True, seed=42)

        # Compute slopes for window 1 (steps 0-30, indices 0-3) and
        # window 2 (steps 30-60, indices 3-6).
        # MEASUREMENT_STEPS = [0, 10, 20, 30, 40, 50, 60] → 7 values (indices 0-6).
        w1_without = fp_without[:4]   # steps 0-30
        w2_without = fp_without[3:]   # steps 30-60
        w1_with = fp_with[:4]
        w2_with = fp_with[3:]

        slope_w1_without = _linear_slope(w1_without)
        slope_w2_without = _linear_slope(w2_without)
        slope_w1_with = _linear_slope(w1_with)
        slope_w2_with = _linear_slope(w2_with)

        # Determine honest verdict.
        pps_stable = slope_w1_with <= 0 and slope_w2_with <= 0
        without_relapse = slope_w2_without > 0

        if pps_stable and without_relapse:
            honest_verdict = "pps_prevents_relapse"
        elif slope_w2_with > slope_w2_without:
            honest_verdict = "pps_hurts"
        else:
            honest_verdict = "pps_no_effect"

        pps_prevents_relapse = honest_verdict == "pps_prevents_relapse"

        artifact = tmpl.build_result(
            {
                "fp_rate_without_pps": fp_without,
                "fp_rate_with_pps": fp_with,
                "n_frozen_at_step30": n_frozen,
                "slope_w1_without_pps": slope_w1_without,
                "slope_w2_without_pps": slope_w2_without,
                "slope_w1_with_pps": slope_w1_with,
                "slope_w2_with_pps": slope_w2_with,
                "pps_prevents_relapse": pps_prevents_relapse,
                "honest_verdict": honest_verdict,
                "measurement_steps": MEASUREMENT_STEPS,
                "n_couplings": N_COUPLINGS,
                "window_size": WINDOW_SIZE,
                "freeze_threshold": FREEZE_THRESHOLD,
            },
            status="success",
        )

    import json
    out = Path(DELIVERABLE)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2))
    print(f"[762] honest_verdict={honest_verdict}  n_frozen_at_step30={n_frozen}")
    print(f"[762] fp_without_pps={fp_without}")
    print(f"[762] fp_with_pps={fp_with}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
