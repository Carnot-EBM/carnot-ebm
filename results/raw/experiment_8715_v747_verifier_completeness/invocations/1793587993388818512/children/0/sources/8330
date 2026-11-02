#!/usr/bin/env python3
"""Exp 909: Lagrange forgetting curve — measure impact on weight entropy and Tier 3 rejection rate.

**Researcher summary:**
    Exp 866 / .66 confirmed FR-11 Tier 1 Lagrange adaptive weight updates work.
    However, when violation rate drops (later questions), previously accumulated
    weights never decay, causing weight staleness: a handful of early-phase
    constraints accumulate all the weight (entropy collapses), and the energy
    function becomes insensitive to currently-violated constraints.  The Tier 3
    sampler rejects more candidates because the stale high-weight constraints
    dominate the acceptance criterion even for questions where those constraints
    do not apply.

    This experiment quantifies the benefit of exponential weight forgetting
    (already implemented in LagrangeAdaptiveUpdater via forgetting_lambda) by
    running a 100-step synthetic violation sequence — 50 steps at 70% violation
    rate then 50 steps at 10% violation rate — and comparing:

        1. Baseline:      forgetting_lambda=0 (no decay, accumulate indefinitely)
        2. Decay variant: forgetting_lambda=0.05 (half-life ≈ 14 steps)

    Metrics at every 10-step checkpoint:
        - weight_entropy (Shannon entropy of weight distribution)
        - max_weight, mean_weight
        - tier3_rejection_rate (fraction of constraints with weight > threshold,
          simulating the fraction of candidate samples that trip the Tier 3 gate)

    Verdict criteria:
        - "forgetting_curve_improves_entropy" if signed_entropy_improvement > 0.5
        - "marginal_improvement"              if signed_entropy_improvement > 0
        - "no_improvement"                    otherwise

Spec: REQ-SELF-007, SCENARIO-SELF-007
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap sys.path so the module resolves when run from the repo root.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.lagrange_updater import LagrangeAdaptiveUpdater  # noqa: E402

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
EXP_ID = 909
TITLE = "Lagrange Forgetting Curve — Weight Entropy and Tier 3 Rejection Rate"
DELIVERABLE = "results/experiment_909_lagrange_forgetting_curve.json"

N_STEPS = 100
HIGH_VIOLATION_STEPS = 50  # steps 0-49: high violation rate
HIGH_VIOLATION_PROB = 0.7
LOW_VIOLATION_PROB = 0.1
SEED = 42

# Checkpoint interval (every 10 steps)
CHECKPOINT_INTERVAL = 10

# Tier 3 rejection threshold: constraints with weight > this are "active rejectors"
TIER3_WEIGHT_THRESHOLD = 1.5

FORGETTING_LAMBDA = 0.05  # decay variant: exp(-0.05) per tick ≈ 0.951


def _build_violation_sequence(n_steps: int, seed: int) -> list[bool]:
    """Generate a 100-step boolean violation sequence with a high-to-low shift.

    Steps 0-49: violation probability = HIGH_VIOLATION_PROB (0.7).
    Steps 50-99: violation probability = LOW_VIOLATION_PROB (0.1).

    Using a single fixed seed ensures baseline and decay variants see the exact
    same violation events, so all differences are attributable to the forgetting
    curve alone.
    """
    rng = random.Random(seed)
    seq = []
    for i in range(n_steps):
        prob = HIGH_VIOLATION_PROB if i < HIGH_VIOLATION_STEPS else LOW_VIOLATION_PROB
        seq.append(rng.random() < prob)
    return seq


def _run_simulation(
    violation_sequence: list[bool],
    forgetting_lambda: float,
) -> dict:
    """Simulate the Lagrange updater over the violation sequence.

    Each step:
        1. update("constraint_0", violated) — single constraint for simplicity.
        2. tick(1) — apply decay (or no-op when lambda=0).
        3. Record metrics at every CHECKPOINT_INTERVAL steps.

    Returns a dict with per-checkpoint metrics and final summary values.
    """
    updater = LagrangeAdaptiveUpdater(
        weight_init=1.0,
        weight_lr=0.1,
        forgetting_lambda=forgetting_lambda,
        replay_threshold=0.8,
        precision_min_violation_rate=0.1,
    )

    checkpoints = []

    for step, violated in enumerate(violation_sequence):
        updater.update("constraint_0", violated=violated)
        updater.tick(1)

        if (step + 1) % CHECKPOINT_INTERVAL == 0:
            weights = list(updater.constraint_weights.values())
            if weights:
                max_w = max(weights)
                mean_w = sum(weights) / len(weights)
                # Tier 3 rejection rate: fraction of active constraints above threshold.
                rejection_rate = sum(1 for w in weights if w > TIER3_WEIGHT_THRESHOLD) / len(
                    weights
                )
            else:
                # All constraints expired (possible at high lambda).
                max_w = 0.0
                mean_w = 0.0
                rejection_rate = 0.0

            entropy = updater.weight_entropy

            checkpoints.append(
                {
                    "step": step + 1,
                    "weight_entropy": round(entropy, 6),
                    "max_weight": round(max_w, 6),
                    "mean_weight": round(mean_w, 6),
                    "tier3_rejection_rate": round(rejection_rate, 6),
                    "n_active_constraints": updater.n_constraints,
                }
            )

    final_entropy = checkpoints[-1]["weight_entropy"] if checkpoints else 0.0
    final_rejection_rate = checkpoints[-1]["tier3_rejection_rate"] if checkpoints else 0.0

    return {
        "forgetting_lambda": forgetting_lambda,
        "checkpoints": checkpoints,
        "entropy_at_step_100": final_entropy,
        "tier3_rejection_rate_at_step_100": final_rejection_rate,
        "updater_summary": updater.summary(),
    }


def main() -> None:
    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # Build the shared violation sequence (same seed for both variants).
    violation_sequence = _build_violation_sequence(N_STEPS, SEED)

    # Run baseline (no decay) and decay variant.
    baseline_result = _run_simulation(violation_sequence, forgetting_lambda=0.0)
    decay_result = _run_simulation(violation_sequence, forgetting_lambda=FORGETTING_LAMBDA)

    entropy_no_decay = baseline_result["entropy_at_step_100"]
    entropy_with_decay = decay_result["entropy_at_step_100"]
    signed_entropy_improvement = entropy_with_decay - entropy_no_decay

    tier3_no_decay = baseline_result["tier3_rejection_rate_at_step_100"]
    tier3_with_decay = decay_result["tier3_rejection_rate_at_step_100"]
    tier3_improvement = tier3_no_decay - tier3_with_decay  # positive = decay helps

    if signed_entropy_improvement > 0.5:
        honest_verdict = "forgetting_curve_improves_entropy"
    elif signed_entropy_improvement > 0:
        honest_verdict = "marginal_improvement"
    else:
        honest_verdict = "no_improvement"

    payload = {
        "honest_verdict": honest_verdict,
        "entropy_collapse_without_decay": round(entropy_no_decay, 6),
        "entropy_sustained_with_decay": round(entropy_with_decay, 6),
        "signed_entropy_improvement": round(signed_entropy_improvement, 6),
        "tier3_rejection_rate_no_decay": round(tier3_no_decay, 6),
        "tier3_rejection_rate_with_decay": round(tier3_with_decay, 6),
        "tier3_rejection_improvement": round(tier3_improvement, 6),
        "forgetting_lambda_used": FORGETTING_LAMBDA,
        "n_steps": N_STEPS,
        "high_violation_steps": HIGH_VIOLATION_STEPS,
        "high_violation_prob": HIGH_VIOLATION_PROB,
        "low_violation_prob": LOW_VIOLATION_PROB,
        "seed": SEED,
        "baseline_checkpoints": baseline_result["checkpoints"],
        "decay_checkpoints": decay_result["checkpoints"],
        "baseline_updater_summary": baseline_result["updater_summary"],
        "decay_updater_summary": decay_result["updater_summary"],
    }

    artifact = tmpl.build_result(payload, status="success")

    output_path = Path(DELIVERABLE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(artifact, f, indent=2)

    print(f"honest_verdict:              {honest_verdict}")
    print(f"entropy (no decay):          {entropy_no_decay:.4f}")
    print(f"entropy (with decay=0.05):   {entropy_with_decay:.4f}")
    print(f"signed_entropy_improvement:  {signed_entropy_improvement:.4f}")
    print(f"tier3 rejection (no decay):  {tier3_no_decay:.4f}")
    print(f"tier3 rejection (w/ decay):  {tier3_with_decay:.4f}")
    print(f"Deliverable written: {output_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
