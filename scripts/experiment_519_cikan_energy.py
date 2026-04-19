#!/usr/bin/env python3
"""Experiment 519: CIKANEnergy vs KAEMEnergy — near-boundary AUROC advantage.

**Researcher summary:**
    Validates that CIKANEnergy (Constraint-Informed KAN, arXiv 2412.03710) outperforms
    KAEMEnergy on near-boundary examples when the constraint boundary is known.

    Setup: 500 samples from a distribution with a hard boundary at x=0.
    Correct side: x > 0 (label=1). Violated side: x < 0 (label=0).
    Train on 400 samples; evaluate AUROC on 100 held-out for near-boundary
    (|x|<0.2) and far-from-boundary (|x|>0.5) subsets.

    Prediction: CIKANEnergy's extra knots near x=0 give sharper energy gradients
    at the boundary, yielding higher AUROC on near-boundary examples vs KAEMEnergy.

**CPU-only. Always produces a result JSON.**

Spec: REQ-SAMPLE-025, REQ-SAMPLE-026,
      SCENARIO-SAMPLE-038, SCENARIO-SAMPLE-039, SCENARIO-SAMPLE-040
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from carnot.pipeline.env_autofix import apply_env_autofix

_env_fix = apply_env_autofix()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.deliverable_guard import DeliverableGuard  # noqa: E402

EXP_ID = 519
TITLE = "CIKANEnergy vs KAEMEnergy — near-boundary AUROC advantage"
DELIVERABLE = "results/experiment_519_cikan_energy.json"
TIMEOUT_MINUTES = 25

N_TOTAL = 500
N_TRAIN = 400
N_TEST = 100
NEAR_BOUNDARY_THRESHOLD = 0.2
FAR_BOUNDARY_THRESHOLD = 0.5
BOUNDARY_POSITION = 0.0


def _auroc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC for binary classification.

    Scores are model predictions (higher = more likely positive/correct).
    Labels are 0 (violated) or 1 (correct).
    Returns area under ROC curve in [0, 1].
    """
    if len(scores) < 2 or len(set(labels)) < 2:
        return 0.5

    paired = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0
    for _, label in paired:
        if label == 1:
            tp += 1
        else:
            fp += 1
            auc += tp  # trapezoid rectangle
        prev_fp = fp
        prev_tp = tp
    return auc / (n_pos * n_neg)


def main() -> None:
    """Run Experiment 519: CIKANEnergy vs KAEMEnergy near-boundary AUROC."""
    import jax.numpy as jnp
    import numpy as np

    from carnot.models.kaem_energy import KAEMEnergy
    from carnot.models.cikan_energy import CIKANEnergy, ConstraintBoundary

    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=EXP_ID,
        timeout_minutes=TIMEOUT_MINUTES,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    tmpl = ExperimentTemplate(
        exp_id=EXP_ID,
        title=TITLE,
        deliverable=DELIVERABLE,
        requires_gpu=False,
        repo_root=_REPO_ROOT,
    )
    tmpl.setup()

    _guard = DeliverableGuard(str(_REPO_ROOT / DELIVERABLE))

    with watchdog:
        _log.info("Experiment %d starting — CPU-only CIKANEnergy AUROC benchmark", EXP_ID)

        rng = np.random.default_rng(42)

        # -----------------------------------------------------------------
        # Generate samples from distribution with boundary at x=0
        # Correct side: x > 0 (sampled from Uniform(0, 1))
        # Violated side: x < 0 (sampled from Uniform(-1, 0))
        # Training uses only correct-side samples; evaluation tests AUROC
        # of the trained energy function at separating the two sides.
        # -----------------------------------------------------------------
        correct_samples = rng.uniform(0.0, 1.0, (N_TOTAL,)).astype(np.float32)
        violated_samples = rng.uniform(-1.0, 0.0, (N_TOTAL,)).astype(np.float32)

        # Training data: first N_TRAIN correct samples (the distribution to learn)
        train_data = jnp.array(correct_samples[:N_TRAIN].reshape(-1, 1))

        # Held-out test set: last N_TEST correct + N_TEST violated
        test_correct = correct_samples[N_TRAIN:N_TRAIN + N_TEST]
        test_violated = violated_samples[:N_TEST]

        # Near-boundary test: |x| < NEAR_BOUNDARY_THRESHOLD
        near_correct = test_correct[np.abs(test_correct) < NEAR_BOUNDARY_THRESHOLD]
        near_violated = test_violated[np.abs(test_violated) < NEAR_BOUNDARY_THRESHOLD]

        # Far-from-boundary test: |x| > FAR_BOUNDARY_THRESHOLD
        far_correct = test_correct[np.abs(test_correct) > FAR_BOUNDARY_THRESHOLD]
        far_violated = test_violated[np.abs(test_violated) > FAR_BOUNDARY_THRESHOLD]

        _log.info(
            "Near-boundary test set: %d correct, %d violated",
            len(near_correct), len(near_violated),
        )
        _log.info(
            "Far-from-boundary test set: %d correct, %d violated",
            len(far_correct), len(far_violated),
        )

        # -----------------------------------------------------------------
        # Train baseline: KAEMEnergy (uniform knots)
        # -----------------------------------------------------------------
        _log.info("Training KAEMEnergy baseline...")
        baseline = KAEMEnergy(n_vars=1, n_hidden=8)
        baseline.fit(train_data, n_epochs=50)

        # -----------------------------------------------------------------
        # Train CIKANEnergy with boundary at x=0
        # -----------------------------------------------------------------
        _log.info("Training CIKANEnergy with boundary at x=0...")
        cikan = CIKANEnergy(n_vars=1, n_hidden=8)
        cikan.fit_with_constraints(
            train_data,
            boundaries=[ConstraintBoundary(BOUNDARY_POSITION)],
            n_epochs=50,
        )

        # -----------------------------------------------------------------
        # Evaluate AUROC
        # Energy-based scoring: lower energy = more likely correct (label=1).
        # We negate energy so higher score = more likely correct.
        # -----------------------------------------------------------------

        def score_samples(model: KAEMEnergy, xs: np.ndarray) -> list[float]:
            """Return negated energy scores for each sample (higher = more correct)."""
            scores = []
            for x in xs:
                e = float(model.energy(jnp.array([x])))
                scores.append(-e)
            return scores

        def compute_auroc(model: KAEMEnergy, correct: np.ndarray, violated: np.ndarray) -> float:
            """Compute AUROC distinguishing correct (label=1) from violated (label=0)."""
            all_xs = np.concatenate([correct, violated])
            all_labels = [1] * len(correct) + [0] * len(violated)
            scores = score_samples(model, all_xs)
            return _auroc(scores, all_labels)

        baseline_auroc_near = compute_auroc(baseline, near_correct, near_violated)
        cikan_auroc_near = compute_auroc(cikan, near_correct, near_violated)
        baseline_auroc_far = compute_auroc(baseline, far_correct, far_violated)
        cikan_auroc_far = compute_auroc(cikan, far_correct, far_violated)

        _log.info(
            "Near-boundary AUROC — baseline=%.3f, cikan=%.3f",
            baseline_auroc_near, cikan_auroc_near,
        )
        _log.info(
            "Far AUROC — baseline=%.3f, cikan=%.3f",
            baseline_auroc_far, cikan_auroc_far,
        )

        cikan_advantage = bool(cikan_auroc_near > baseline_auroc_near)
        honest_verdict = "cikan_advantage" if cikan_advantage else "no_advantage"

        _log.info("honest_verdict=%s", honest_verdict)

        # -----------------------------------------------------------------
        # Write artifact
        # -----------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.cikan_energy.v1",
                "baseline_auroc_near_boundary": float(baseline_auroc_near),
                "cikan_auroc_near_boundary": float(cikan_auroc_near),
                "baseline_auroc_far": float(baseline_auroc_far),
                "cikan_auroc_far": float(cikan_auroc_far),
                "cikan_advantage": cikan_advantage,
                "honest_verdict": honest_verdict,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "boundary_position": BOUNDARY_POSITION,
                "near_boundary_threshold": NEAR_BOUNDARY_THRESHOLD,
                "far_boundary_threshold": FAR_BOUNDARY_THRESHOLD,
                "near_correct_count": int(len(near_correct)),
                "near_violated_count": int(len(near_violated)),
                "far_correct_count": int(len(far_correct)),
                "far_violated_count": int(len(far_violated)),
                "env_fix": {
                    "gpu_detected": _env_fix.gpu_detected,
                    "auto_fix_applied": _env_fix.auto_fix_applied,
                },
            },
            status="success",
        )

        output_path = _REPO_ROOT / DELIVERABLE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(artifact, f, indent=2)

        _log.info("Artifact written to %s", output_path)

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
