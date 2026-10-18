#!/usr/bin/env python3
"""Experiment 918: Lagrange Forgetting Multi-Constraint Rerun.

**Why this experiment exists (root cause of prior failure):**
    Exp 909 (Lagrange forgetting curve) measured signed_entropy_improvement = 0.0
    because the corpus had only 1 constraint type with a constant violation probability
    of 1.0. With p=1.0 always, -p*log(p) = 0 by definition: entropy is zero from step 0
    and cannot improve. The result was labeled "no_improvement" but the algorithm never
    had a chance to demonstrate anything — it was a degenerate test design, not an
    algorithm failure.

**What is different in this experiment:**
    8 heterogeneous constraint types with distinct violation probabilities. In Phase 1
    (steps 1-100) each constraint fires at its high_viol_prob; in Phase 2 (steps 101-200)
    each fires at its low_viol_prob. This creates a non-uniform weight distribution where
    entropy is non-zero from the start (~log(8) = 2.08 nats when weights are equal) and
    where the forgetting curve (decay_rate=0.95 per step, i.e. lambda=-ln(0.95)≈0.0513)
    has a measurable effect compared to the no-decay baseline.

**What we measure:**
    At each 20-step interval: weight_entropy for both the baseline (no decay) and the
    decay updater. At step 200: signed_entropy_improvement = entropy_decay - entropy_baseline.
    A positive value confirms that exponential forgetting preserves weight diversity better
    than the no-forgetting baseline, validating the FOREVER-curve hypothesis.

**Failure mode we guard against:**
    If entropy_at_step_1_baseline < 0.1, the corpus is still degenerate and we emit
    honest_verdict='degenerate_again_retire', which adds this attempt to the retire list.

Prior failure: Exp 909, verdict: no_improvement, root cause: single-constraint degenerate corpus.
Addressed by: 8-constraint heterogeneous corpus with varying violation rates.
retire_if_same_verdict: false (root cause identified and directly fixed).

Spec: REQ-SELF-007, SCENARIO-SELF-007
"""

import math
import random
import sys
from pathlib import Path

# Allow running from repo root or scripts/ directly.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from python.carnot.pipeline.lagrange_updater import LagrangeAdaptiveUpdater  # noqa: E402

# ---------------------------------------------------------------------------
# Corpus definition — 8 constraint types with heterogeneous violation rates
# ---------------------------------------------------------------------------

# Each entry: name -> (high_viol_prob used in steps 1-100, low_viol_prob used in 101-200).
# The spread across probabilities ensures weight_entropy is non-degenerate from step 1.
CONSTRAINTS: dict[str, tuple[float, float]] = {
    "arithmetic_carry": (0.80, 0.05),
    "sign_check": (0.70, 0.10),
    "unit_consistency": (0.60, 0.15),
    "comparison_direction": (0.50, 0.20),
    "equality_check": (0.40, 0.25),
    "range_check": (0.30, 0.30),
    "step_coherence": (0.20, 0.50),
    "modular_arithmetic": (0.10, 0.70),
}

TOTAL_STEPS = 200
PHASE_BOUNDARY = 100  # steps 1-100 use high_viol_prob; 101-200 use low_viol_prob
RECORD_INTERVAL = 20  # record entropy every 20 steps

# decay_rate=0.95 per step means forgetting_lambda = -ln(0.95) ≈ 0.0513
DECAY_RATE = 0.95
FORGETTING_LAMBDA = -math.log(DECAY_RATE)  # ≈ 0.0513


def _make_updater(forgetting_lambda: float) -> LagrangeAdaptiveUpdater:
    """Create a LagrangeAdaptiveUpdater with the given forgetting lambda.

    Initialised with all 8 constraints registered at weight_init=1.0 so that
    entropy at step 0 is well-defined (equal weights -> maximum entropy).
    """
    updater = LagrangeAdaptiveUpdater(
        weight_init=1.0,
        weight_lr=0.1,
        forgetting_lambda=forgetting_lambda,
        replay_threshold=0.95,  # high threshold — we want natural decay, not replay rescue
        precision_min_violation_rate=0.1,
    )
    # Pre-register all 8 constraints so they exist at step 0 with equal weights.
    for cid in CONSTRAINTS:
        updater.update(cid, violated=False)
    return updater


def run_simulation(rng: random.Random) -> dict:
    """Run both baseline (no decay) and decay updaters for TOTAL_STEPS steps.

    At each RECORD_INTERVAL: snapshot weight_entropy, max_weight, mean_weight for both.

    Returns a dict with the interval log and final entropy values for verdict computation.
    """
    baseline = _make_updater(forgetting_lambda=0.0)  # no forgetting
    decay_upd = _make_updater(forgetting_lambda=FORGETTING_LAMBDA)

    interval_log: list[dict] = []

    for step in range(1, TOTAL_STEPS + 1):
        # Determine phase.
        high_phase = step <= PHASE_BOUNDARY

        # Feed each constraint to both updaters.
        for cid, (high_prob, low_prob) in CONSTRAINTS.items():
            viol_prob = high_prob if high_phase else low_prob
            violated = rng.random() < viol_prob
            baseline.update(cid, violated=violated)
            decay_upd.update(cid, violated=violated)

        # Apply forgetting tick (no-op for baseline since lambda=0).
        baseline.tick(step=1)
        decay_upd.tick(step=1)

        # Record at interval boundaries.
        if step % RECORD_INTERVAL == 0:
            weights_b = list(baseline.constraint_weights.values())
            weights_d = list(decay_upd.constraint_weights.values())
            interval_log.append(
                {
                    "step": step,
                    "entropy_baseline": baseline.weight_entropy,
                    "entropy_decay": decay_upd.weight_entropy,
                    "max_weight_baseline": max(weights_b) if weights_b else 0.0,
                    "mean_weight_baseline": sum(weights_b) / len(weights_b) if weights_b else 0.0,
                    "max_weight_decay": max(weights_d) if weights_d else 0.0,
                    "mean_weight_decay": sum(weights_d) / len(weights_d) if weights_d else 0.0,
                    "n_active_baseline": baseline.n_constraints,
                    "n_active_decay": decay_upd.n_constraints,
                }
            )

    # Entropy after the first interval — used to verify corpus is non-degenerate.
    entropy_step1 = interval_log[0]["entropy_baseline"] if interval_log else 0.0
    entropy_final_baseline = interval_log[-1]["entropy_baseline"] if interval_log else 0.0
    entropy_final_decay = interval_log[-1]["entropy_decay"] if interval_log else 0.0

    return {
        "interval_log": interval_log,
        "entropy_at_first_interval_baseline": entropy_step1,
        "entropy_baseline_at_200": entropy_final_baseline,
        "entropy_decay_at_200": entropy_final_decay,
        "signed_entropy_improvement": entropy_final_decay - entropy_final_baseline,
        "baseline_summary": baseline.summary(),
        "decay_summary": decay_upd.summary(),
    }


def compute_verdict(sim_result: dict) -> str:
    """Map simulation metrics to an honest_verdict string.

    Non-degenerate check first: if the corpus itself is still degenerate
    (entropy < 0.1 at step 20), retire immediately.
    """
    entropy_step1 = sim_result["entropy_at_first_interval_baseline"]
    signed_improvement = sim_result["signed_entropy_improvement"]

    if entropy_step1 < 0.1:
        # Corpus is still degenerate — this design has the same flaw as Exp 909.
        return "degenerate_again_retire"
    if signed_improvement > 0.5:
        return "forgetting_curve_improves_entropy"
    if signed_improvement > 0.0:
        return "marginal_improvement"
    return "no_improvement"


def main() -> None:
    """Run Exp 918 end-to-end and write the deliverable artifact."""
    tmpl = ExperimentTemplate(
        exp_id=918,
        title="Lagrange Forgetting Multi-Constraint — RETRO-LAGRANGE-ENTROPY-DEGENERATE Fix",
        deliverable="results/experiment_918_lagrange_forgetting_multi_constraint.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Fixed seed for reproducibility; the corpus heterogeneity ensures non-degenerate
    # entropy regardless of seed, but fixing the seed makes the run deterministic.
    rng = random.Random(918)

    sim = run_simulation(rng)
    verdict = compute_verdict(sim)

    prior_exp = 909
    prior_verdict = "no_improvement"
    prior_root_cause = "Single-constraint corpus with p=1.0 always; entropy = 0 by construction."
    fix_applied = (
        "8 heterogeneous constraint types with violation rates spanning 0.05–0.80, "
        "producing non-zero entropy from step 1 (~log(8) = 2.08 nats at equal weights)."
    )

    artifact = tmpl.build_result(
        {
            "n_constraints": len(CONSTRAINTS),
            "constraint_names": list(CONSTRAINTS.keys()),
            "total_steps": TOTAL_STEPS,
            "phase_boundary": PHASE_BOUNDARY,
            "forgetting_lambda": FORGETTING_LAMBDA,
            "decay_rate": DECAY_RATE,
            "entropy_at_first_interval_baseline": sim["entropy_at_first_interval_baseline"],
            "entropy_baseline_at_200": sim["entropy_baseline_at_200"],
            "entropy_decay_at_200": sim["entropy_decay_at_200"],
            "signed_entropy_improvement": sim["signed_entropy_improvement"],
            "interval_log": sim["interval_log"],
            "baseline_summary": sim["baseline_summary"],
            "decay_summary": sim["decay_summary"],
            "honest_verdict": verdict,
            "prior_failure": {
                "experiment_id": f"exp{prior_exp}_lagrange_forgetting_curve",
                "verdict": prior_verdict,
                "root_cause": prior_root_cause,
                "fix_applied": fix_applied,
                "retire_if_same_verdict": False,
            },
        },
        status="success",
    )

    print(f"Honest verdict: {verdict}")
    print(f"Entropy at step 20 (baseline): {sim['entropy_at_first_interval_baseline']:.4f}")
    print(f"Entropy at step 200 (baseline): {sim['entropy_baseline_at_200']:.4f}")
    print(f"Entropy at step 200 (decay):    {sim['entropy_decay_at_200']:.4f}")
    print(f"signed_entropy_improvement:     {sim['signed_entropy_improvement']:.4f}")

    import json

    out_path = Path(tmpl.deliverable)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Wrote: {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
