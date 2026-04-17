#!/usr/bin/env python3
"""Exp 443 — EORM + JEPA Retrain on Live FOVER Pairs (RETRO-024 Closure Attempt, Milestone 8).

**Researcher summary:**
    RETRO-024 has been open for 8 consecutive milestones because all previous EORM/JEPA
    retrains used synthetic-only data (AUC ≈ 0.5 baseline — indistinguishable from random).

    Exp 442 changed this: it ran FOVERAnnotator on 300 live GPU CoT responses from Exp 439
    and produced 57 real labeled steps (30 correct, 27 incorrect) in
    ``results/fover_labeled_steps_live.json``.  57 real pairs far exceeds the threshold of 20.

    This experiment:
    1. Loads the Exp 442 live annotation result and labeled pairs.
    2. Falls back to synthetic corpus if n_labeled < 20 (honest reporting).
    3. Evaluates EORM ``before_auc`` on a held-out 20% test split.
    4. Retrains EORM for 150 epochs on contrastive triples from real pairs.
    5. Also retrains JEPA predictor on (step_prefix, violation_flag) pairs.
    6. Evaluates ``after_auc`` on the same held-out split.
    7. Saves ``results/eorm_443_live.safetensors`` and ``results/jepa_443_live.safetensors``.
    8. Emits ``schema='carnot.eorm_jepa_retrain.v3'`` artifact with honest_verdict.
    9. Sets ``retro_024_closed=True`` iff ``honest_verdict='real_data_improvement'``.

**Honest reporting (compute_retrain_verdict_v2):**
    - ``'real_data_improvement'``   : ≥20 real pairs AND after_auc > before_auc → RETRO-024 CLOSED
    - ``'real_data_no_improvement'``: ≥20 real pairs but AUC did not improve
    - ``'real_data_insufficient'``  : <20 real pairs (synthetic fallback used)
    - ``'synthetic_only'``          : source != 'live'

**Usage:**
    JAX_PLATFORMS=cpu python scripts/experiment_443_eorm_jepa_live_retrain.py
    CARNOT_FORCE_LIVE=1 python scripts/experiment_443_eorm_jepa_live_retrain.py

Spec: REQ-LEARN-036, SCENARIO-LEARN-064, SCENARIO-LEARN-065
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Ensure repo root + python/ + scripts/ are importable
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

import jax.random as jrandom
import numpy as np

from carnot.embeddings.jepa_energy import ContextPredictionEnergy, JEPAEnergyConfig
from carnot.embeddings.jepa_retrain import JEPARetrainer, ViolationPair, _make_synthetic_pairs
from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from carnot.models.eorm_retrain import make_synthetic_eorm_pairs
from carnot.pipeline.env_autofix import apply_env_autofix
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog
from carnot.pipeline.fover_eorm_retrain import (
    compute_retrain_verdict_v2,
    load_fover_pairs,
)
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_ID = 443
TITLE = "EORM + JEPA retrain on live FOVER pairs (RETRO-024 closure attempt, milestone 8)"
DELIVERABLE = "results/experiment_443_eorm_jepa_live_retrain.json"

# Minimum real FOVER pairs required for retrain_mode='real_data'
REAL_PAIR_THRESHOLD = 20

# Training hyperparameters (150 epochs, up from 100 in Exp 431, for more convergence)
TRAIN_SPLIT = 0.8
N_EPOCHS = 150
CHECKPOINT_EVERY = 50
BATCH_SIZE = 16
LR = 1e-4
MARGIN = 1.0

# EORM model config (matches Exp 346/359/431 for fair AUC comparison)
EMBED_DIM = 128
N_HEADS = 4
N_LAYERS = 2

# JEPA model config
JEPA_EMBED_DIM = 64
JEPA_HIDDEN_DIMS = [64, 32]

# Synthetic fallback sizes when real pairs < REAL_PAIR_THRESHOLD
SYNTHETIC_EORM_N = 120
SYNTHETIC_JEPA_N = 50


# ---------------------------------------------------------------------------
# AUC-ROC evaluator for EORM (same as Exp 431)
# ---------------------------------------------------------------------------


def _evaluate_eorm_auc(model: EORMModel, pairs: list[ViolationPair]) -> float:
    """Compute AUC-ROC for EORM model on a ViolationPair test set.

    EORM outputs lower energy for responses it considers correct; higher energy predicts
    violation.  We negate energy so that the ROC curve uses high score = predicted violation.

    Args:
        model: Trained EORMModel instance.
        pairs: ViolationPair test examples.

    Returns:
        AUC-ROC in [0, 1]. 0.5 = random baseline.
    """
    if not pairs:
        return 0.5

    scores: list[float] = []
    labels: list[int] = []

    for p in pairs:
        cot = CoTEnergyInput(
            question_text=p.question_id,
            response_text=p.full_response,
        )
        energy = model.energy(cot)
        scores.append(-energy)
        labels.append(1 if p.has_violation else 0)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    scored = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    tpr_pts = [0.0]
    fpr_pts = [0.0]
    tp = fp = 0

    for _s, lab in scored:
        if lab == 1:
            tp += 1
        else:
            fp += 1
        tpr_pts.append(tp / n_pos)
        fpr_pts.append(fp / n_neg)

    auc = 0.0
    for i in range(1, len(fpr_pts)):
        dfpr = fpr_pts[i] - fpr_pts[i - 1]
        auc += dfpr * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0

    return float(auc)


# ---------------------------------------------------------------------------
# Convert FOVER pairs to ViolationPairs for JEPA training
# ---------------------------------------------------------------------------


def _fover_pairs_to_violation_pairs(fover_pairs: list[dict]) -> list[ViolationPair]:
    """Map FOVER labeled steps to ViolationPair objects for JEPARetrainer.

    Each FOVER step:
    - label='incorrect' → has_violation=True
    - label='correct'   → has_violation=False

    Full step_text is used as both partial_response and full_response — JEPA violation
    detection does not need a prefix split at this stage.

    Args:
        fover_pairs: FOVER pair dicts from load_fover_pairs().

    Returns:
        List of ViolationPair objects.
    """
    result: list[ViolationPair] = []
    for p in fover_pairs:
        result.append(
            ViolationPair(
                partial_response=p["step_text"],
                full_response=p["step_text"],
                has_violation=(p["label"] == "incorrect"),
                model_id="fover_live_443",
                question_id=p["question_id"],
            )
        )
    return result


# ---------------------------------------------------------------------------
# Save JEPA model as safetensors
# ---------------------------------------------------------------------------


def _save_jepa_model(model: ContextPredictionEnergy, path: str) -> None:
    """Save ContextPredictionEnergy parameters as a safetensors file.

    Flattens the parameter dict to numpy arrays keyed as 'layer_N_weight',
    'layer_N_bias', 'output_weight', 'output_bias'.

    Args:
        model: ContextPredictionEnergy instance to save.
        path: Destination file path (should end in .safetensors).
    """
    from safetensors.numpy import save_file

    np_flat: dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(model.layers):
        np_flat[f"layer_{i}_weight"] = np.array(w)
        np_flat[f"layer_{i}_bias"] = np.array(b)
    np_flat["output_weight"] = np.array(model.output_weight)
    np_flat["output_bias"] = np.array([model.output_bias], dtype=np.float32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file as _save  # noqa: F811
    _save(np_flat, path)


# ---------------------------------------------------------------------------
# Load or build baseline EORM model
# ---------------------------------------------------------------------------


def _load_or_build_eorm_model(root: Path) -> EORMModel:
    """Load the most recently trained EORM, or fall back to fresh random init.

    Preference order (most recent first):
    1. results/eorm_431_real.safetensors  (Exp 431 live-ish retrain)
    2. results/eorm_model_359_real.safetensors  (Exp 359 retrain)
    3. results/eorm_model_346.safetensors       (Exp 346 synthetic baseline)
    4. Fresh random init with seed 443

    Using the most recently trained model gives a realistic before_auc rather than
    always starting from a random baseline.
    """
    for candidate in [
        root / "results" / "eorm_431_real.safetensors",
        root / "results" / "eorm_model_359_real.safetensors",
        root / "results" / "eorm_model_346.safetensors",
    ]:
        if candidate.exists():
            try:
                model = EORMModel.load(str(candidate))
                _log.info("Loaded EORM baseline from %s", candidate)
                return model
            except Exception as exc:
                _log.warning("Could not load %s (%s); trying next", candidate, exc)

    model = EORMModel(
        embed_dim=EMBED_DIM,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        key=jrandom.PRNGKey(443),
    )
    _log.info("Built fresh EORMModel (embed_dim=%d, n_layers=%d)", EMBED_DIM, N_LAYERS)
    return model


# ---------------------------------------------------------------------------
# Build EORM contrastive triples from ViolationPairs
# ---------------------------------------------------------------------------


def _build_eorm_triples(
    violation_pairs: list[ViolationPair],
) -> list[tuple[str, str, str]]:
    """Build (correct_response, incorrect_response, question_id) triples for EORMTrainer.

    Groups ViolationPairs by question_id and round-robin matches correct vs incorrect
    responses for the same question.  Cross-question matching is avoided — that would
    create false contrastive signal (penalizing a correct answer from question A when
    compared to an incorrect answer from question B).

    Args:
        violation_pairs: ViolationPair list with has_violation labels.

    Returns:
        List of (correct_response, incorrect_response, question_text) triples.
    """
    from collections import defaultdict

    _SHARED_POOL = "_synthetic_pool"
    correct_by_q: dict[str, list[str]] = defaultdict(list)
    incorrect_by_q: dict[str, list[str]] = defaultdict(list)

    for vp in violation_pairs:
        q_id = vp.question_id
        if q_id == "unknown" or q_id.startswith("synthetic_"):
            key = _SHARED_POOL
        else:
            key = q_id

        if vp.has_violation:
            incorrect_by_q[key].append(vp.full_response)
        else:
            correct_by_q[key].append(vp.full_response)

    all_keys = set(correct_by_q.keys()) | set(incorrect_by_q.keys())
    triples: list[tuple[str, str, str]] = []

    for key in sorted(all_keys):
        corrects = correct_by_q.get(key, [])
        incorrects = incorrect_by_q.get(key, [])
        if not corrects or not incorrects:
            continue
        n = max(len(corrects), len(incorrects))
        for i in range(n):
            c = corrects[i % len(corrects)]
            ic = incorrects[i % len(incorrects)]
            triples.append((c, ic, key))

    return triples


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def run_experiment(
    *,
    force_live: bool = False,
    repo_root: Path | None = None,
) -> dict:
    """Execute Exp 443: EORM + JEPA retrain on live FOVER pairs from Exp 442.

    All file paths can be overridden via repo_root for test isolation.

    Args:
        force_live: If True, behave as if CARNOT_FORCE_LIVE=1 (informational).
        repo_root: Override repo root (used in tests to isolate file I/O).

    Returns:
        Full experiment artifact dict (also written to DELIVERABLE JSON).
    """
    _root = repo_root or _REPO_ROOT

    # Step 1: apply_env_autofix FIRST (belt-and-suspenders for GPU env vars)
    autofix = apply_env_autofix()
    _log.info(
        "env_autofix: gpu_detected=%s auto_fix_applied=%s",
        autofix.gpu_detected, autofix.auto_fix_applied,
    )

    tmpl = ExperimentTemplate(
        EXPERIMENT_ID,
        TITLE,
        DELIVERABLE,
        requires_gpu=False,  # EORM/JEPA retrain is CPU-only
        repo_root=_root,
    )
    tmpl.setup()

    # Step 2: Load Exp 442 annotation result to get source metadata
    ann_path = _root / "results" / "experiment_442_fover_live_annotation.json"
    source = "synthetic"
    n_labeled_from_442 = 0
    if ann_path.exists():
        try:
            with open(ann_path) as f:
                ann_data = json.load(f)
            source = str(ann_data.get("source", "synthetic"))
            n_labeled_from_442 = int(ann_data.get("n_labeled", 0))
            _log.info(
                "Loaded Exp 442 annotation: source=%s n_labeled=%d",
                source, n_labeled_from_442,
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            _log.warning("Could not load Exp 442 annotation (%s); defaulting to synthetic", exc)
    else:
        _log.info("Exp 442 annotation not found at %s; defaulting to synthetic", ann_path)

    # Step 3: Load FOVER labeled pairs from live file
    fover_live_path = str(_root / "results" / "fover_labeled_steps_live.json")
    fover_pairs = load_fover_pairs(fover_live_path)
    n_real = len(fover_pairs)
    _log.info("Loaded %d FOVER real pairs from %s", n_real, fover_live_path)

    # Step 4: Determine retrain mode and build corpus
    if source == "live" and n_real >= REAL_PAIR_THRESHOLD:
        retrain_mode = "real_data"
        violation_pairs = _fover_pairs_to_violation_pairs(fover_pairs)
        _log.info("Using %d real live FOVER pairs for EORM/JEPA retrain", n_real)
    else:
        retrain_mode = "synthetic_only"
        _log.info(
            "Real pairs insufficient (n=%d, threshold=%d, source=%s); using synthetic fallback",
            n_real, REAL_PAIR_THRESHOLD, source,
        )
        synthetic_eorm = make_synthetic_eorm_pairs(n=SYNTHETIC_EORM_N, seed=443)
        synthetic_jepa = _make_synthetic_pairs(n=SYNTHETIC_JEPA_N, seed=443)
        violation_pairs = list(synthetic_eorm) + list(synthetic_jepa)

    # Train/test split (80/20, no shuffle for reproducibility)
    n_train = max(1, int(len(violation_pairs) * TRAIN_SPLIT))
    train_vp = violation_pairs[:n_train]
    test_vp = violation_pairs[n_train:] if len(violation_pairs) > n_train else violation_pairs

    # Step 5: Load/build EORM baseline and evaluate before_auc
    eorm_model = _load_or_build_eorm_model(_root)
    before_auc = _evaluate_eorm_auc(eorm_model, test_vp)
    _log.info("EORM before_auc = %.4f", before_auc)

    # Step 6: Build contrastive triples and retrain EORM (150 epochs)
    triples = _build_eorm_triples(train_vp)
    _log.info(
        "Training EORM for %d epochs on %d contrastive triples",
        N_EPOCHS, len(triples),
    )

    eorm_trainer = EORMTrainer(eorm_model, lr=LR, margin=MARGIN)
    epoch_losses: list[float] = []

    if triples:
        for epoch in range(N_EPOCHS):
            loss = eorm_trainer.train_epoch(triples, batch_size=BATCH_SIZE)
            epoch_losses.append(loss)
            if (epoch + 1) % CHECKPOINT_EVERY == 0:
                _log.info("Epoch %d/%d — mean loss = %.4f", epoch + 1, N_EPOCHS, loss)
                tmpl.checkpoint_save(
                    {"epoch": epoch + 1, "loss": round(loss, 6), "n_triples": len(triples)},
                    step=epoch + 1,
                )
    else:
        _log.warning(
            "No contrastive triples formed (need correct+incorrect per question). "
            "EORM parameters unchanged."
        )

    # Step 7: Evaluate after_auc
    after_auc = _evaluate_eorm_auc(eorm_model, test_vp)
    _log.info("EORM after_auc = %.4f (delta = %+.4f)", after_auc, after_auc - before_auc)

    # Step 8: Retrain JEPA on FOVER violation pairs
    jepa_config = JEPAEnergyConfig(embed_dim=JEPA_EMBED_DIM, hidden_dims=JEPA_HIDDEN_DIMS)
    jepa_model = ContextPredictionEnergy(config=jepa_config, key=jrandom.PRNGKey(443))
    jepa_retrainer = JEPARetrainer(jepa_model, lr=LR)

    jepa_before_auc = jepa_retrainer.evaluate_auc_roc(test_vp)
    _log.info("JEPA before_auc = %.4f", jepa_before_auc)

    if train_vp:
        jepa_epochs = max(10, N_EPOCHS // 5)
        for _epoch in range(jepa_epochs):
            jepa_retrainer.train_epoch(train_vp, batch_size=BATCH_SIZE)

    jepa_after_auc = jepa_retrainer.evaluate_auc_roc(test_vp)
    _log.info("JEPA after_auc = %.4f (delta = %+.4f)", jepa_after_auc, jepa_after_auc - jepa_before_auc)

    # Step 9: Save models
    eorm_path = str(_root / "results" / "eorm_443_live.safetensors")
    jepa_path = str(_root / "results" / "jepa_443_live.safetensors")

    try:
        eorm_model.save(eorm_path)
        _log.info("Saved EORM model to %s", eorm_path)
    except Exception as exc:
        _log.warning("Could not save EORM model: %s", exc)
        eorm_path = ""

    try:
        _save_jepa_model(jepa_model, jepa_path)
        _log.info("Saved JEPA model to %s", jepa_path)
    except Exception as exc:
        _log.warning("Could not save JEPA model: %s", exc)
        jepa_path = ""

    # Step 10: Compute honest verdict and build artifact
    honest_verdict = compute_retrain_verdict_v2(before_auc, after_auc, n_real, source)
    retro_024_closed = honest_verdict == "real_data_improvement"

    _log.info(
        "honest_verdict=%s retro_024_closed=%s",
        honest_verdict, retro_024_closed,
    )

    artifact = tmpl.build_result(
        {
            "schema": "carnot.eorm_jepa_retrain.v3",
            "retrain_mode": retrain_mode,
            "n_real_pairs": n_real,
            "source": source,
            "before_auc": round(before_auc, 6),
            "after_auc": round(after_auc, 6),
            "auc_improvement": round(after_auc - before_auc, 6),
            "honest_verdict": honest_verdict,
            "retro_024_closed": retro_024_closed,
            "eorm_model_path": eorm_path,
            "jepa_before_auc": round(jepa_before_auc, 6),
            "jepa_after_auc": round(jepa_after_auc, 6),
            "jepa_model_path": jepa_path,
            "n_contrastive_triples": len(triples),
            "n_train_pairs": len(train_vp),
            "n_test_pairs": len(test_vp),
            "n_eorm_epochs": N_EPOCHS,
            "n_labeled_from_442": n_labeled_from_442,
        },
        status="success",
    )

    return artifact


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 443 with watchdog, write results to deliverable JSON."""
    force_live = bool(int(os.environ.get("CARNOT_FORCE_LIVE", "0")))

    watchdog = ExperimentTimeoutWatchdog(
        EXPERIMENT_ID,
        timeout_minutes=45,
        result_path=str(_REPO_ROOT / DELIVERABLE),
    )

    with watchdog:
        artifact = run_experiment(force_live=force_live)

    deliverable = _REPO_ROOT / DELIVERABLE
    deliverable.parent.mkdir(parents=True, exist_ok=True)
    with open(deliverable, "w") as f:
        json.dump(artifact, f, indent=2)

    _log.info(
        "Exp 443 complete. honest_verdict=%s retro_024_closed=%s -> %s",
        artifact.get("honest_verdict"),
        artifact.get("retro_024_closed"),
        deliverable,
    )


if __name__ == "__main__":
    main()
