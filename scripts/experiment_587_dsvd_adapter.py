#!/usr/bin/env python3
"""Experiment 587: DSVD Adapter — hidden-state linear probe for mid-generation violation detection.

**Researcher summary:**
    arXiv 2503.03149 (DSVD) shows that hidden-state features at step boundaries in a
    transformer's residual stream carry real-time violation signal during generation —
    not just post-hoc.  Carnot's current pipeline does purely post-hoc verify-repair
    (generate full response, then check).  DSVD opens a Tier 2.5 slot in the cascade:
    between EORM (Tier 2, energy scoring) and CoACEExtractor (Tier 3, symbolic execution).

    This experiment validates a CPU-only approximation of DSVD:
      - Extract four cheap text features from each CoT step (length, n_numbers,
        n_operators, char_entropy).
      - Project to a 64-dimensional space via a fixed random matrix
        (Johnson-Lindenstrauss approximation of hidden-state projection).
      - Fit a logistic-regression probe on 80% of the 132-pair FOVER corpus steps.
      - Evaluate AUC-ROC on the 20% held-out split.
      - Compare to CoACEExtractor v1 baseline AUC on the same steps.

    Gate: tier_2_5_viable = dsvd_auc > 0.60.

**Honest reporting:**
    honest_verdict is one of:
      'tier_2_5_viable'           — DSVD AUC > 0.60; Tier 2.5 insertion recommended
      'tier_2_5_below_threshold'  — DSVD AUC ≤ 0.60; not viable at current fidelity

Spec: REQ-VERIFY-118, SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be first — it patches JAX_PLATFORMS and ROCm env vars before
# any JAX import.  Moving it later causes "No GPU/TPU found" crashes on ROCm hosts.
from carnot.pipeline.env_autofix import apply_env_autofix

apply_env_autofix()

import json
import os
import random
from collections import defaultdict
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID: int = 587
TITLE: str = "DSVD Adapter"
DELIVERABLE: str = "results/experiment_587_dsvd_adapter.json"
CORPUS_PATH: str = "results/fover_corpus_v2.json"
SCHEMA_NAME: str = "carnot.dsvd_adapter.v1"

TRAIN_FRAC: float = 0.80
VIOLATION_THRESHOLD: float = 0.5
DSVD_VIABLE_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Watchdog — kills the process after 20 minutes to prevent runaway experiments.
# ---------------------------------------------------------------------------

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

_watchdog = ExperimentTimeoutWatchdog(EXPERIMENT_ID, timeout_minutes=20)

# ---------------------------------------------------------------------------
# Experiment template (provides setup, checkpointing, build_result, assert_deliverable_written)
# ---------------------------------------------------------------------------

from scripts.experiment_template import ExperimentTemplate  # noqa: E402

tmpl = ExperimentTemplate(
    exp_id=EXPERIMENT_ID,
    title=TITLE,
    deliverable=DELIVERABLE,
    requires_gpu=False,
)
tmpl.setup()

# ---------------------------------------------------------------------------
# Pipeline imports (after env_autofix and template setup)
# ---------------------------------------------------------------------------

from carnot.extraction.coace_extractor import CoACEExtractor  # noqa: E402
from carnot.pipeline.dsvd_adapter import DSVDAdapter, DSVDLinearProbe  # noqa: E402

# ---------------------------------------------------------------------------
# AUC-ROC implementation — no sklearn dependency required.
# ---------------------------------------------------------------------------


def _compute_auc_roc(y_true: list[float], y_score: list[float]) -> float:
    """Compute AUC-ROC using the trapezoidal rule.

    Args:
        y_true: Binary labels where 1.0 = positive class (violation).
        y_score: Predicted probability scores in [0, 1].

    Returns:
        AUC value in [0, 1].  Returns 0.5 for degenerate inputs (all-same labels).
    """
    # Build (threshold, tp_rate, fp_rate) curve by sweeping thresholds.
    n = len(y_true)
    if n == 0:
        return 0.5

    n_pos = sum(1 for y in y_true if y == 1.0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate

    # Sort by descending score.
    paired = sorted(zip(y_score, y_true), key=lambda x: -x[0])

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0
    for score, label in paired:
        if label == 1.0:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    # Trapezoidal integration.
    auc = 0.0
    for i in range(1, len(fpr_list)):
        d_fpr = fpr_list[i] - fpr_list[i - 1]
        auc += d_fpr * (tpr_list[i - 1] + tpr_list[i]) / 2.0
    return float(auc)


# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------


def load_corpus(path: str) -> list[dict[str, Any]]:
    """Load the FOVER corpus from JSON.  Raises FileNotFoundError if absent."""
    corpus_path = Path(path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {path}")
    with open(corpus_path) as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Step extraction
# ---------------------------------------------------------------------------


def extract_steps(corpus: list[dict[str, Any]]) -> tuple[list[str], list[float], list[str]]:
    """Extract (step_text, violation_label, question_id) triples from corpus.

    Each step inherits its chain's correctness label.
    violation_label = 1.0 when is_correct=False (the step is from an incorrect chain).

    Args:
        corpus: List of FOVER corpus entries.

    Returns:
        Tuple of (step_texts, labels, question_ids).
    """
    step_texts: list[str] = []
    labels: list[float] = []
    question_ids: list[str] = []

    for entry in corpus:
        qid = str(entry.get("question", ""))
        is_correct = bool(entry.get("is_correct", True))
        label = 0.0 if is_correct else 1.0
        for step in entry.get("cot_steps", []):
            text = step.get("step_text", "")
            if text.strip():
                step_texts.append(text)
                labels.append(label)
                question_ids.append(qid)

    return step_texts, labels, question_ids


# ---------------------------------------------------------------------------
# Train/val split by question_id (prevent data leakage across chains)
# ---------------------------------------------------------------------------


def split_by_question(
    step_texts: list[str],
    labels: list[float],
    question_ids: list[str],
    train_frac: float = 0.80,
    seed: int = 42,
) -> tuple[list[str], list[float], list[str], list[float]]:
    """Split steps into train/val sets, keeping all steps from one question in the same split.

    Splitting by question_id prevents the probe from overfitting to question-level
    features rather than step-level violation signals.

    Args:
        step_texts: All step texts.
        labels: Corresponding violation labels.
        question_ids: Question ID for each step.
        train_frac: Fraction of questions to use for training.
        seed: RNG seed for reproducibility.

    Returns:
        (train_texts, train_labels, val_texts, val_labels)
    """
    unique_qids = sorted(set(question_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_qids)

    n_train = max(1, int(len(unique_qids) * train_frac))
    train_qids = set(unique_qids[:n_train])

    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    for text, label, qid in zip(step_texts, labels, question_ids):
        if qid in train_qids:
            train_texts.append(text)
            train_labels.append(label)
        else:
            val_texts.append(text)
            val_labels.append(label)

    return train_texts, train_labels, val_texts, val_labels


# ---------------------------------------------------------------------------
# CoACE baseline AUC on validation steps
# ---------------------------------------------------------------------------


def coace_v1_auc(val_texts: list[str], val_labels: list[float]) -> float:
    """Compute CoACEExtractor v1 AUC-ROC on validation steps.

    CoACE extracts and executes arithmetic equations, returning n_violations per step.
    We use n_violations as the score (more violations → higher violation probability).
    Steps with n_violations > 0 are flagged as violations.

    The AUC is computed against val_labels (1.0 = violation chain).

    Returns:
        AUC-ROC float in [0, 1].
    """
    extractor = CoACEExtractor(tolerance=1e-6, min_confidence=0.5)
    scores: list[float] = []
    for text in val_texts:
        result = extractor.extract(text)
        # Use n_violations as a continuous score so AUC captures ranking quality.
        scores.append(float(result.n_violations))
    return _compute_auc_roc(val_labels, scores)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run the DSVD adapter experiment and return the result artifact dict."""
    # 1. Load corpus.
    corpus = load_corpus(CORPUS_PATH)

    # 2. Extract steps.
    step_texts, labels, question_ids = extract_steps(corpus)

    # 3. Split by question_id.
    train_texts, train_labels, val_texts, val_labels = split_by_question(
        step_texts, labels, question_ids, train_frac=TRAIN_FRAC
    )

    n_train = len(train_texts)
    n_val = len(val_texts)

    # 4. Fit DSVDLinearProbe.
    probe = DSVDLinearProbe(hidden_dim=64)
    probe.fit(train_texts, train_labels)

    # 5. Predict on validation steps.
    adapter = DSVDAdapter(probe, violation_threshold=VIOLATION_THRESHOLD)
    val_scores = [probe.predict(t) for t in val_texts]

    # 6. Compute DSVD AUC-ROC.
    dsvd_auc = _compute_auc_roc(val_labels, val_scores)

    # 7. CoACE v1 baseline AUC on same validation steps.
    coace_auc = coace_v1_auc(val_texts, val_labels)

    # 8. Viability gate.
    tier_2_5_viable = dsvd_auc > DSVD_VIABLE_THRESHOLD
    honest_verdict = "tier_2_5_viable" if tier_2_5_viable else "tier_2_5_below_threshold"

    # 9. Build artifact.
    data = {
        "schema": SCHEMA_NAME,
        "n_train_steps": n_train,
        "n_val_steps": n_val,
        "dsvd_auc": round(dsvd_auc, 4),
        "coace_v1_auc": round(coace_auc, 4),
        "tier_2_5_viable": tier_2_5_viable,
        "honest_verdict": honest_verdict,
    }

    return tmpl.build_result(data, status="success", decision_class="verify")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    artifact = run_experiment()

    deliverable_path = Path(DELIVERABLE)
    deliverable_path.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable_path, "w") as fh:
        json.dump(artifact, fh, indent=2)

    print(f"Deliverable written: {deliverable_path}")
    print(f"  dsvd_auc={artifact['dsvd_auc']}")
    print(f"  coace_v1_auc={artifact['coace_v1_auc']}")
    print(f"  tier_2_5_viable={artifact['tier_2_5_viable']}")
    print(f"  honest_verdict={artifact['honest_verdict']}")

    tmpl.assert_deliverable_written()
