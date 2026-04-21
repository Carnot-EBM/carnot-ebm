#!/usr/bin/env python3
"""Experiment 658: SpecGuard Step Verifier — LPBV + ABGV hallucination detection.

**Context:**
    arXiv 2604.15244 introduces SpecGuard, a spec-based guardrail system that uses
    two signals computed from the generation forward pass to reject hallucinated
    reasoning steps sub-millisecond:

        LPBV (Log-Probability-Based Verification): steps where the model assigns
        low log-prob to its own output indicate uncertainty or hallucination.

        ABGV (Attention-Based Grounding Verification): steps where attention weights
        do not concentrate on specification-relevant tokens indicate the model is
        not grounded in the problem constraints.

    This experiment evaluates SpecGuardVerifier on live_pairs_578.json (578 labelled
    question-response pairs) and measures AUROC to determine whether the verifier
    meets the target of >= 0.70 AUC for deployment as Tier 0f in ThreeTierPipeline.

**Success criteria:**
    specguard_auc >= 0.70  ->  honest_verdict = 'specguard_tier_0f_viable'
    specguard_auc <  0.70  ->  honest_verdict = 'specguard_below_threshold'

**CPU-safe:**
    No GPU required.  SpecGuardVerifier uses text heuristics when logprobs and
    attention weights are absent (as they are in the pre-collected live_pairs_578.json).

Spec: REQ-VERIFY-152, REQ-VERIFY-153, REQ-VERIFY-154
SCENARIO-VERIFY-206, SCENARIO-VERIFY-207, SCENARIO-VERIFY-208
"""

import json
import os
import sys

# env_autofix MUST be first — injects CARNOT_FORCE_LIVE=1 if a GPU is detected.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.specguard_verifier import SpecGuardVerifier  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 658
TITLE = "SpecGuard Step Verifier"
DELIVERABLE = "results/experiment_658_specguard_verifier.json"
LIVE_PAIRS_FILE = os.path.join(_REPO_ROOT, "results", "live_pairs_578.json")
AUC_TARGET = 0.70

# ---------------------------------------------------------------------------
# Watchdog (arms immediately — guards the entire script lifetime)
# ---------------------------------------------------------------------------

_watchdog = ExperimentTimeoutWatchdog(
    EXP_ID, timeout_minutes=30, result_path=DELIVERABLE
)
_watchdog.start()

# ---------------------------------------------------------------------------
# Template setup
# ---------------------------------------------------------------------------

tmpl = ExperimentTemplate(
    exp_id=EXP_ID,
    title=TITLE,
    deliverable=DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

_DELIVERABLE_PATH = os.path.join(_REPO_ROOT, DELIVERABLE)
os.makedirs(os.path.dirname(_DELIVERABLE_PATH), exist_ok=True)

# ---------------------------------------------------------------------------
# Load live pairs
# ---------------------------------------------------------------------------

with open(LIVE_PAIRS_FILE) as f:
    live_pairs = json.load(f)

# ---------------------------------------------------------------------------
# Score each pair with SpecGuardVerifier
# ---------------------------------------------------------------------------

verifier = SpecGuardVerifier()

# Collect (label, score) pairs for AUROC computation.
# label = 1 when is_correct=False (hallucinated / wrong).
# We want high detection_score to correlate with incorrect responses.
labels: list[int] = []
scores: list[float] = []

tp = tn = fp = fn = 0

for pair in live_pairs:
    response = pair.get("response", "")
    is_correct = pair.get("is_correct", True)

    score = verifier.detection_score(response)
    predicted_hallucinated = score >= 0.5
    true_incorrect = not is_correct

    labels.append(1 if true_incorrect else 0)
    scores.append(score)

    if predicted_hallucinated and true_incorrect:
        tp += 1
    elif predicted_hallucinated and not true_incorrect:
        fp += 1
    elif not predicted_hallucinated and true_incorrect:
        fn += 1
    else:
        tn += 1

# ---------------------------------------------------------------------------
# Compute AUROC via trapezoidal rule (no sklearn dependency)
# ---------------------------------------------------------------------------

def _compute_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC from parallel label and score lists.

    Uses the trapezoidal rule over all unique score thresholds.  This is
    equivalent to the Wilcoxon-Mann-Whitney statistic: the probability that
    a randomly chosen positive (hallucinated) pair scores higher than a
    randomly chosen negative (correct) pair.

    Parameters
    ----------
    labels : list[int]
        1 for hallucinated/incorrect, 0 for correct.
    scores : list[float]
        Detection scores aligned with labels.

    Returns
    -------
    float
        AUROC in [0, 1].  0.5 = random classifier.
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending; ties broken with label ascending.
    paired = sorted(zip(scores, labels), key=lambda x: (-x[0], x[1]))

    tpr_points: list[float] = [0.0]
    fpr_points: list[float] = [0.0]
    tp_acc = 0
    fp_acc = 0

    for _score, label in paired:
        if label == 1:
            tp_acc += 1
        else:
            fp_acc += 1
        tpr_points.append(tp_acc / n_pos)
        fpr_points.append(fp_acc / n_neg)

    # Trapezoidal rule: sum of trapezoids under the ROC curve.
    auc = 0.0
    for i in range(1, len(tpr_points)):
        auc += (fpr_points[i] - fpr_points[i - 1]) * (tpr_points[i] + tpr_points[i - 1]) / 2.0
    return auc


specguard_auc = _compute_auroc(labels, scores)
tier_0f_viable = specguard_auc >= AUC_TARGET
n_pairs = len(live_pairs)

honest_verdict = (
    "specguard_tier_0f_viable" if tier_0f_viable else "specguard_below_threshold"
)

print(f"[Exp {EXP_ID}] n_pairs={n_pairs}, specguard_auc={specguard_auc:.4f}, "
      f"tier_0f_viable={tier_0f_viable}, verdict={honest_verdict}")
print(f"[Exp {EXP_ID}] TP={tp} FP={fp} TN={tn} FN={fn}")

# ---------------------------------------------------------------------------
# Build and write artifact
# ---------------------------------------------------------------------------

artifact = tmpl.build_result(
    {
        "schema": "carnot.specguard_verifier.v1",
        "n_pairs": n_pairs,
        "specguard_auc": specguard_auc,
        "tier_0f_viable": tier_0f_viable,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auc_target": AUC_TARGET,
        "arxiv_ref": "2604.15244",
        "honest_verdict": honest_verdict,
    },
    status="success",
)

with open(_DELIVERABLE_PATH, "w") as f:
    json.dump(artifact, f, indent=2)

_watchdog.stop()
tmpl.assert_deliverable_written()
