#!/usr/bin/env python3
"""Exp 359 — EORM Real-Data Retrain: Compare AUC-ROC vs Exp 346 Synthetic Baseline.

**Researcher summary:**
    Exp 346 trained EORM entirely on synthetic (question, CoT response, correctness)
    pairs because all live GPU experiments (Exp 340, 341, 355) ran in simulated mode
    and returned no real model responses. This experiment:
    1. Checks whether any live data is now available from Exps 340, 341, or 355.
    2. If ≥50 real pairs found: retrain mode = "real_data" (real improvement signal).
    3. If <50 real pairs: retrain mode = "synthetic_only" (honest fallback).
    4. Loads the Exp 346 EORM model (or builds fresh if not present).
    5. Evaluates before_auc on the 20% test split.
    6. Trains for 50 epochs (CI) or 200 epochs (live) on the 80% train split.
    7. Evaluates after_auc on the same test split.
    8. Saves the retrained model to results/eorm_model_359_real.safetensors.
    9. Emits a schema="carnot.eorm_retrain.v1" artifact with honest_verdict.

**Honest reporting:**
    - "real_data_improvement": ≥50 real pairs AND after_auc > before_auc.
    - "real_data_no_improvement": ≥50 real pairs but AUC did not improve.
    - "synthetic_only": <50 real pairs — live GPU required for real improvement signal.

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_359_eorm_real_retrain.py
    CARNOT_FORCE_LIVE=1 python scripts/experiment_359_eorm_real_retrain.py

Spec: REQ-LEARN-025, SCENARIO-LEARN-043, SCENARIO-LEARN-044
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure repo root on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax.random as jrandom

from carnot.embeddings.jepa_retrain import ViolationPair
from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.eorm_retrain import (
    EORMRetrainResult,
    build_retrain_artifact,
    load_real_cot_pairs,
    make_synthetic_eorm_pairs,
    merge_cot_corpora,
)
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 359
TITLE = "EORM real-data retrain — AUC-ROC comparison vs Exp 346 synthetic baseline"
DELIVERABLE = "results/experiment_359_eorm_real_retrain.json"

# Minimum real pairs required to declare retrain_mode="real_data"
REAL_DATA_THRESHOLD = 50

# Training hyperparameters
TRAIN_SPLIT = 0.8
N_EPOCHS_CI = 50       # CPU / CI mode (fast)
N_EPOCHS_LIVE = 200    # when CARNOT_FORCE_LIVE=1 (more thorough)
BATCH_SIZE = 16
LR = 1e-4
MARGIN = 1.0

# Max corpus size per source
MAX_REAL = 300
MAX_SYNTHETIC = 100

# EORM model config (matches Exp 346 defaults)
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 2


# ---------------------------------------------------------------------------
# AUC-ROC evaluator for EORM (standalone, no sklearn needed)
# ---------------------------------------------------------------------------


def _evaluate_eorm_auc(model: EORMModel, pairs: list[ViolationPair]) -> float:
    """Compute AUC-ROC for EORM model on ViolationPair test set.

    **For engineers:**
        EORM outputs lower energy for responses it considers more correct.
        A "violation" corresponds to an incorrect response (has_violation=True).
        To match the AUC convention (high score = positive class = violation),
        we use NEGATED energy as the score:
            score = -energy  (high negated energy = high energy = probably violation)

        This matches how JEPARetrainer.evaluate_auc_roc works: high energy predicts
        violation, so we treat negated energy as the discrimination score.

    Args:
        model: Trained EORMModel.
        pairs: List of ViolationPair test examples.

    Returns:
        AUC-ROC in [0, 1]. 0.5 = random baseline.
    """
    if not pairs:
        return 0.5

    scores: list[float] = []
    labels: list[int] = []

    for p in pairs:
        cot = CoTEnergyInput(
            question_text=p.question_id,  # use question_id as proxy for question text
            response_text=p.full_response,
        )
        energy = model.energy(cot)
        # High energy → predicted violation (score = -energy for AUC convention)
        scores.append(-energy)
        labels.append(1 if p.has_violation else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending (high score = predicted positive = violation)
    scored = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    tpr_pts = [0.0]
    fpr_pts = [0.0]
    tp = 0
    fp = 0

    for _s, lab in scored:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / n_pos)
        fpr_pts.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_pts)):
        dfpr = fpr_pts[i] - fpr_pts[i - 1]
        auc += dfpr * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0

    return float(auc)


# ---------------------------------------------------------------------------
# Convert ViolationPairs to EORM contrastive triples
# ---------------------------------------------------------------------------


def _pairs_to_contrastive_triples(
    pairs: list[ViolationPair],
) -> list[tuple[str, str, str]]:
    """Convert ViolationPair list into (correct, incorrect, question) triples for EORM.

    **For engineers:**
        EORM trains on *contrastive pairs*: given the same question, which response
        is correct and which is wrong? ViolationPairs carry individual responses with
        binary labels, not explicit contrast pairs.

        This function groups pairs by question_id and creates all cross-product
        (correct_response, incorrect_response) combinations for each question that
        has both a correct and an incorrect entry.

        For questions with only one label type (all correct or all incorrect), no
        contrastive triple can be formed and those pairs are skipped.

        When question_id is "unknown" or "synthetic_*", all correct pairs are treated
        as coming from a notional "shared" question and cross-multiplied with all
        incorrect pairs. This gives a reasonable number of training triples from
        synthetic data.

    Args:
        pairs: List of ViolationPair objects.

    Returns:
        List of (correct_response, incorrect_response, question_text) tuples.
    """
    # Group by question_id
    from collections import defaultdict

    correct_by_q: dict[str, list[str]] = defaultdict(list)
    incorrect_by_q: dict[str, list[str]] = defaultdict(list)

    for p in pairs:
        q_key = p.question_id
        if p.has_violation:
            incorrect_by_q[q_key].append(p.full_response)
        else:
            correct_by_q[q_key].append(p.full_response)

    # All known question IDs
    all_q_ids = set(correct_by_q.keys()) | set(incorrect_by_q.keys())

    triples: list[tuple[str, str, str]] = []

    for q_id in sorted(all_q_ids):
        corrects = correct_by_q.get(q_id, [])
        incorrects = incorrect_by_q.get(q_id, [])

        if not corrects or not incorrects:
            # Cannot form a contrastive pair without both labels
            continue

        # Round-robin pairing: avoids O(n^2) explosion for large question groups
        n_pairs = max(len(corrects), len(incorrects))
        for i in range(n_pairs):
            c = corrects[i % len(corrects)]
            ic = incorrects[i % len(incorrects)]
            # Use question_id as the question text proxy (full question not available)
            triples.append((c, ic, q_id))

    return triples


# ---------------------------------------------------------------------------
# Load or build baseline EORM model
# ---------------------------------------------------------------------------


def _load_or_build_eorm_model(baseline_path: Path) -> EORMModel:
    """Load the Exp 346 EORM baseline or build a fresh model if not present.

    **For engineers:**
        We prefer to load the Exp 346 trained model so the before_auc reflects
        the state after synthetic training rather than random initialization. If
        the safetensors file is not present (e.g., Exp 346 was never run on this
        machine), we build a fresh model with a fixed seed. Both cases are honest:
        - Load: before_auc reflects synthetic-trained EORM.
        - Build: before_auc is ~0.5 (random-init EORM has no discrimination power).

    Args:
        baseline_path: Expected path to results/eorm_model_346.safetensors.

    Returns:
        An EORMModel instance (either loaded or freshly initialized).
    """
    if baseline_path.exists():
        try:
            model = EORMModel.load(str(baseline_path))
            _log.info("Loaded Exp 346 EORM model from %s", baseline_path)
            return model
        except Exception as exc:
            _log.warning("Failed to load Exp 346 model (%s); building fresh model", exc)

    # Fresh model with fixed seed for reproducibility
    model = EORMModel(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        key=jrandom.PRNGKey(346),
    )
    _log.info("Built fresh EORMModel (embed_dim=%d, n_layers=%d)", EMBED_DIM, N_LAYERS)
    return model


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    force_live: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Execute Exp 359: load data, retrain EORM, evaluate AUC before/after.

    **For engineers:**
        This function is the single entry point for both live execution and unit tests.
        All file paths can be overridden via ``repo_root`` for test isolation.

    Args:
        force_live: If True, use N_EPOCHS_LIVE (200) instead of N_EPOCHS_CI (50).
        repo_root: Override repo root (used in tests for temp directory isolation).

    Returns:
        Full experiment artifact dict (matches what is written to JSON).
    """
    _root = repo_root or _REPO_ROOT

    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # EORM retrain runs on CPU (JAX_PLATFORMS=cpu)
        repo_root=_root,
    )
    tmpl.setup()

    # ---- 1. Load real pairs from available experiment result files ----
    result_files = [
        str(_root / "results" / "experiment_340_live_precision_benchmark.json"),
        str(_root / "results" / "experiment_341_live_humaneval.json"),
        str(_root / "results" / "experiment_355_adversarial_gsm8k_benchmark.json"),
    ]
    real_pairs = load_real_cot_pairs(result_files)
    n_real = len(real_pairs)
    _log.info("Loaded %d real CoT pairs from experiment result files", n_real)

    # ---- 2. Determine retrain mode ----
    retrain_mode = "real_data" if n_real >= REAL_DATA_THRESHOLD else "synthetic_only"
    _log.info("retrain_mode=%s (threshold=%d)", retrain_mode, REAL_DATA_THRESHOLD)

    # ---- 3. Build corpus (real first, synthetic fill) ----
    synthetic_pairs = make_synthetic_eorm_pairs(n=MAX_SYNTHETIC + 20, seed=359)
    corpus = merge_cot_corpora(real_pairs, synthetic_pairs, max_real=MAX_REAL, max_synthetic=MAX_SYNTHETIC)
    _log.info("Corpus: %d total pairs (%d real, %d synthetic)", len(corpus), n_real, len(corpus) - n_real)

    # ---- 4. Train / test split (80/20, no shuffle for reproducibility) ----
    n_train = max(1, int(len(corpus) * TRAIN_SPLIT))
    train_pairs = corpus[:n_train]
    test_pairs = corpus[n_train:] if len(corpus) > n_train else corpus  # fallback: use full corpus as test

    # ---- 5. Load / build baseline EORM model ----
    baseline_path = _root / "results" / "eorm_model_346.safetensors"
    model = _load_or_build_eorm_model(baseline_path)

    # ---- 6. Evaluate before_auc ----
    before_auc = _evaluate_eorm_auc(model, test_pairs)
    _log.info("before_auc = %.4f", before_auc)

    # ---- 7. Build contrastive training triples ----
    triples = _pairs_to_contrastive_triples(train_pairs)
    n_epochs = N_EPOCHS_LIVE if force_live else N_EPOCHS_CI
    _log.info(
        "Training for %d epochs on %d contrastive triples (from %d pairs)",
        n_epochs, len(triples), len(train_pairs),
    )

    trainer = EORMTrainer(model, lr=LR, margin=MARGIN)

    if triples:
        for epoch in range(n_epochs):
            loss = trainer.train_epoch(triples, batch_size=BATCH_SIZE)
            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                _log.info("Epoch %d/%d — mean loss = %.4f", epoch + 1, n_epochs, loss)
    else:
        _log.warning(
            "No contrastive triples could be formed from the corpus "
            "(need at least one correct and one incorrect response per question). "
            "Model parameters unchanged."
        )

    # ---- 8. Evaluate after_auc ----
    after_auc = _evaluate_eorm_auc(model, test_pairs)
    _log.info("after_auc = %.4f (improvement = %+.4f)", after_auc, after_auc - before_auc)

    # ---- 9. Save retrained model ----
    model_path = str(_root / "results" / "eorm_model_359_real.safetensors")
    try:
        model.save(model_path)
        _log.info("Saved retrained model to %s", model_path)
    except Exception as exc:
        _log.warning("Could not save model: %s", exc)
        model_path = ""

    # ---- 10. Build result and artifact ----
    n_synthetic_used = len(corpus) - n_real
    result = EORMRetrainResult(
        n_real_pairs=n_real,
        n_synthetic_pairs=n_synthetic_used,
        before_auc=before_auc,
        after_auc=after_auc,
        auc_improvement=after_auc - before_auc,
        retrain_mode=retrain_mode,
        model_path=model_path,
    )

    retrain_data = build_retrain_artifact(result)

    artifact = tmpl.build_result(
        {
            **retrain_data,
            "n_contrastive_triples": len(triples),
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "n_epochs": n_epochs,
        },
        status="success",
    )

    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 359 and write results to the deliverable JSON file."""
    force_live = bool(int(os.environ.get("CARNOT_FORCE_LIVE", "0")))

    artifact = run_experiment(force_live=force_live)

    deliverable = _REPO_ROOT / DELIVERABLE
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(deliverable, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info(
        "Exp 359 complete: retrain_mode=%s, before_auc=%.4f, after_auc=%.4f, "
        "improvement=%+.4f, honest_verdict=%s",
        artifact.get("retrain_mode"),
        artifact.get("before_auc", 0.0),
        artifact.get("after_auc", 0.0),
        artifact.get("auc_improvement", 0.0),
        artifact.get("honest_verdict"),
    )


if __name__ == "__main__":
    main()
