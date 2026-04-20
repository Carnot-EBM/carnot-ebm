#!/usr/bin/env python3
"""Experiment 608: NUP Probe v6 — CAPO Calibration-Aware Retrain on Live Corpus v4.

**Researcher summary:**
    NUP Probe v5 (Exp 599) achieved AUC=0.739 on 37 GRPO pairs — below the 0.80
    Tier 0c deployment threshold.  Two root causes identified in RETRO-049:

    1. Training corpus too small and synthetic-heavy (37 pairs, many GRPO-generated).
       Fix: use the full fover_corpus_v4 (300 live GPU pairs from Exps 578/579/602).

    2. Contrastive-only loss can overfit to extreme gaps (AUC=1.0 on training data)
       while generalising poorly to held-out data — the same pattern seen in JEPA v11.
       Fix: add CAPO calibration regularisation (arXiv 2604.12632) to penalise
       overconfident predictions on borderline pairs.

    Gate: v6_val_auc >= 0.80 → tier_0c_deployable = True → retro_049_resolved = True.

**Architecture (unchanged from v4):**
    - Character bigram bag-of-features embedding (energy_dim=32 buckets, hashed)
    - Single linear layer: features -> energy scalar
    - Energy function: E(step) = dot(weights, encode(step)) + bias
    - Higher energy = more likely incorrect (hallucinated)

**Training changes from v5 to v6:**
    - Loss: CAPOCalibrationLoss (margin=1.0, lambda_cal=0.1) replaces ContrastivePairLoss
    - Corpus: fover_corpus_v4.json (300 live pairs) instead of 37 GRPO pairs
    - Epochs: 200 with best-AUC checkpoint saved every 50 epochs
    - Split: 80/20 train/val stratified by is_correct label

Spec: REQ-VERIFY-140, REQ-VERIFY-141,
      SCENARIO-VERIFY-171, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
"""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX/GPU import
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.capo_calibration import CAPOCalibrationLoss  # noqa: E402
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

EXP_ID = 608
EXP_TITLE = "NUP Probe v6 CAPO Retrain"
DELIVERABLE = "results/experiment_608_nup_probe_v6.json"

CORPUS_V4_PATH = _REPO_ROOT / "results" / "fover_corpus_v4.json"
LIVE_578_PATH = _REPO_ROOT / "results" / "live_pairs_578.json"
LIVE_579_PATH = _REPO_ROOT / "results" / "live_pairs_579.json"
LIVE_602_PATH = _REPO_ROOT / "results" / "live_pairs_602.json"
V6_MODEL_PATH = _REPO_ROOT / "results" / "nup_probe_v6.safetensors"

ENERGY_DIM = 32
N_EPOCHS = 200
CHECKPOINT_EVERY = 50
LEARNING_RATE = 1e-3
MARGIN = 1.0
LAMBDA_CAL = 0.1
TRAIN_FRAC = 0.8
RANDOM_SEED = 42
V5_AUC = 0.739


# ---------------------------------------------------------------------------
# Embedding helpers — identical to NUPProbeV4 for compatibility
# ---------------------------------------------------------------------------


def _encode(step_text: str, energy_dim: int = ENERGY_DIM) -> List[float]:
    """Embed a CoT step as a normalised character-bigram feature vector.

    Uses the same polynomial hash as NUPProbeV4 so embeddings are compatible.
    Character bigrams are hashed into `energy_dim` buckets and L2-normalised.
    """
    if len(step_text) < 2:
        return [0.0] * energy_dim

    bigram_counts: Counter[str] = Counter()
    for i in range(len(step_text) - 1):
        bigram_counts[step_text[i : i + 2]] += 1

    vec = [0.0] * energy_dim
    for bigram, count in bigram_counts.items():
        h = (ord(bigram[0]) * 31 + ord(bigram[1])) % energy_dim
        vec[h] += float(count)

    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0.0:
        vec = [x / norm for x in vec]
    return vec


def _score(
    step_text: str,
    weights: List[float],
    bias: float,
    energy_dim: int = ENERGY_DIM,
) -> float:
    """Compute energy score for a step: dot(weights, encode(step)) + bias."""
    features = _encode(step_text, energy_dim)
    return sum(w * f for w, f in zip(weights, features)) + bias


# ---------------------------------------------------------------------------
# AUC computation — same as NUPProbeV4.evaluate_auc
# ---------------------------------------------------------------------------


def _compute_auc(
    correct_steps: List[str],
    incorrect_steps: List[str],
    weights: List[float],
    bias: float,
) -> float:
    """Compute AUROC treating incorrect steps as the positive class (label=1).

    Higher energy should correlate with incorrect steps.  AUROC=1.0 means
    perfect discrimination; AUROC=0.5 is chance.
    """
    n_pos = len(incorrect_steps)
    n_neg = len(correct_steps)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    scored: List[Tuple[float, bool]] = []
    for s in correct_steps:
        scored.append((_score(s, weights, bias), False))
    for s in incorrect_steps:
        scored.append((_score(s, weights, bias), True))

    scored.sort(key=lambda x: x[0], reverse=True)

    tp = 0
    fp = 0
    auc = 0.0
    prev_fpr = 0.0
    prev_tpr = 0.0

    for _, is_incorrect in scored:
        if is_incorrect:
            tp += 1
        else:
            fp += 1
        fpr = fp / n_neg
        tpr = tp / n_pos
        if fpr > prev_fpr:
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
        prev_fpr = fpr
        prev_tpr = tpr

    return float(min(1.0, max(0.0, auc)))


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _load_corpus() -> List[Dict]:
    """Load live pairs from fover_corpus_v4.json (preferred) or merged fallback sources.

    fover_corpus_v4.json is the canonical merged corpus written by Exp 602.
    If unavailable, we merge live_pairs_578.json + live_pairs_579.json + live_pairs_602.json.

    Each entry has at minimum: {'response': str, 'is_correct': bool}.
    """
    if CORPUS_V4_PATH.exists():
        raw = json.loads(CORPUS_V4_PATH.read_text())
        # fover_corpus_v4 wraps pairs under a 'pairs' key
        if isinstance(raw, dict) and "pairs" in raw:
            return raw["pairs"]
        if isinstance(raw, list):
            return raw
        _log.warning("Unexpected fover_corpus_v4.json structure; trying fallback sources.")

    _log.info("fover_corpus_v4.json not found; merging fallback live pair sources.")
    entries: List[Dict] = []
    for path in (LIVE_578_PATH, LIVE_579_PATH, LIVE_602_PATH):
        if path.exists():
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                entries.extend(raw)
            _log.info("  Loaded %d entries from %s", len(raw) if isinstance(raw, list) else 0, path.name)

    return entries


# ---------------------------------------------------------------------------
# Train/val split (stratified by is_correct)
# ---------------------------------------------------------------------------


def _stratified_split(
    entries: List[Dict],
    train_frac: float = TRAIN_FRAC,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Dict], List[Dict]]:
    """80/20 split stratified by is_correct so both splits have balanced labels."""
    rng = random.Random(seed)
    correct = [e for e in entries if e.get("is_correct", True)]
    incorrect = [e for e in entries if not e.get("is_correct", True)]

    def split_class(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_frac))
        return shuffled[:n_train], shuffled[n_train:]

    c_train, c_val = split_class(correct)
    i_train, i_val = split_class(incorrect)
    train = c_train + i_train
    val = c_val + i_val
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# ---------------------------------------------------------------------------
# Training loop with CAPO loss
# ---------------------------------------------------------------------------


def _train_nup_probe_v6(
    train_entries: List[Dict],
    val_entries: List[Dict],
    tmpl: ExperimentTemplate,
) -> Tuple[List[float], float, float, List[float]]:
    """Train NUP Probe v6 with CAPO calibration loss.

    Returns (weights, bias, best_val_auc, loss_history).
    Checkpoints every CHECKPOINT_EVERY epochs; keeps best weights by val_auc.
    """
    rng = random.Random(RANDOM_SEED)
    weights: List[float] = [(rng.random() - 0.5) * 0.01 for _ in range(ENERGY_DIM)]
    bias: float = 0.0

    best_weights = weights[:]
    best_bias = bias
    best_val_auc = 0.0

    loss_fn = CAPOCalibrationLoss(lambda_cal=LAMBDA_CAL, margin=MARGIN)

    # Pre-encode all training steps (avoids re-encoding on every epoch)
    correct_enc = [_encode(e["response"]) for e in train_entries if e.get("is_correct", True)]
    incorrect_enc = [_encode(e["response"]) for e in train_entries if not e.get("is_correct", True)]

    # Build val step lists for AUC evaluation
    val_correct_steps = [e["response"] for e in val_entries if e.get("is_correct", True)]
    val_incorrect_steps = [e["response"] for e in val_entries if not e.get("is_correct", True)]

    loss_history: List[float] = []

    for epoch in range(1, N_EPOCHS + 1):
        epoch_loss = 0.0
        n_pairs = 0

        for c_enc in correct_enc:
            for i_enc in incorrect_enc:
                e_correct = sum(w * f for w, f in zip(weights, c_enc)) + bias
                e_incorrect = sum(w * f for w, f in zip(weights, i_enc)) + bias

                # Compute CAPO gradients analytically
                # The loss per pair:
                #   margin_loss = max(0, margin - (e_incorrect - e_correct))
                #   diff = e_correct - e_incorrect  (for calibration term)
                #   cal_active = |diff| < 0.3
                #   cal_loss = (diff + 0.5)^2 if cal_active else 0
                #   total = margin_loss + lambda_cal * cal_loss

                gap = e_incorrect - e_correct
                margin_loss = max(0.0, MARGIN - gap)

                diff = e_correct - e_incorrect
                cal_active = abs(diff) < CAPOCalibrationLoss._CALIBRATION_THRESHOLD
                cal_loss = (diff + 0.5) ** 2 if cal_active else 0.0

                pair_loss = margin_loss + LAMBDA_CAL * cal_loss
                epoch_loss += pair_loss
                n_pairs += 1

                # Gradient of margin_loss wrt weights (hinge):
                #   d(margin_loss)/dw = -(i_enc - c_enc) when margin_loss > 0 else 0
                # Gradient of cal_loss wrt weights:
                #   d((diff+0.5)^2)/dw = 2*(diff+0.5) * d(diff)/dw
                #   diff = e_correct - e_incorrect = dot(w, c_enc) - dot(w, i_enc)
                #   d(diff)/dw_j = c_enc[j] - i_enc[j]
                #   So: d(cal_loss)/dw_j = 2*(diff+0.5)*(c_enc[j] - i_enc[j])

                for j in range(ENERGY_DIM):
                    grad = 0.0
                    if margin_loss > 0.0:
                        # hinge gradient: push E(incorrect) up, E(correct) down
                        grad += -(i_enc[j] - c_enc[j])
                    if cal_active:
                        grad += LAMBDA_CAL * 2.0 * (diff + 0.5) * (c_enc[j] - i_enc[j])
                    weights[j] -= LEARNING_RATE * grad

                # Gradient wrt bias:
                #   d(margin_loss)/db = -1 when margin_loss > 0 (both energies shift)
                #   Cal term: d(diff)/db = 0 (bias cancels in diff = e_correct - e_incorrect)
                bias_grad = 0.0
                if margin_loss > 0.0:
                    bias_grad = -1.0
                bias -= LEARNING_RATE * bias_grad

        mean_loss = epoch_loss / n_pairs if n_pairs > 0 else 0.0
        loss_history.append(mean_loss)

        if epoch % CHECKPOINT_EVERY == 0 or epoch == N_EPOCHS:
            val_auc = _compute_auc(val_correct_steps, val_incorrect_steps, weights, bias)
            _log.info("Epoch %d/%d  loss=%.4f  val_auc=%.4f", epoch, N_EPOCHS, mean_loss, val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_weights = weights[:]
                best_bias = bias
            tmpl.checkpoint_save(
                {"weights": best_weights, "bias": best_bias, "best_val_auc": best_val_auc},
                step=epoch,
            )

    return best_weights, best_bias, best_val_auc, loss_history


# ---------------------------------------------------------------------------
# Safetensors save (minimal, no external deps needed — plain JSON fallback)
# ---------------------------------------------------------------------------


def _save_model(weights: List[float], bias: float, path: Path) -> None:
    """Save model weights as safetensors.  Falls back to JSON if safetensors unavailable.

    safetensors is the standard format used across Carnot EBM models.
    JSON fallback prevents the save from blocking the deliverable in CI environments
    that don't have the safetensors Python package installed.
    """
    try:
        import numpy as np
        from safetensors.numpy import save_file

        tensors = {
            "weights": np.array(weights, dtype=np.float32),
            "bias": np.array([bias], dtype=np.float32),
        }
        save_file(tensors, str(path))
        _log.info("Model saved to %s (safetensors)", path)
    except ImportError:
        _log.warning("safetensors not available; saving as JSON fallback.")
        json_path = path.with_suffix(".json")
        json_path.write_text(json.dumps({"weights": weights, "bias": bias}))
        _log.info("Model saved to %s (JSON fallback)", json_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run NUP Probe v6 CAPO retrain and write deliverable."""
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=40)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # Load corpus
    entries = _load_corpus()
    _log.info("Loaded %d corpus entries.", len(entries))
    n_live_pairs = len(entries)

    if n_live_pairs == 0:
        artifact = tmpl.build_result(
            {
                "n_live_pairs": 0,
                "v5_auc": V5_AUC,
                "v6_val_auc": 0.0,
                "capo_applied": True,
                "lambda_cal": LAMBDA_CAL,
                "model_saved": False,
                "tier_0c_deployable": False,
                "retro_049_resolved": False,
                "honest_verdict": "no_corpus_data",
            },
            status="blocked",
        )
        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))
        writer.write(artifact)
        tmpl.assert_deliverable_written()
        return

    # Stratified 80/20 split
    train_entries, val_entries = _stratified_split(entries)
    train_pairs = len(train_entries)
    val_pairs = len(val_entries)
    _log.info("Split: %d train, %d val.", train_pairs, val_pairs)

    # Train v6
    best_weights, best_bias, v6_val_auc, loss_history = _train_nup_probe_v6(
        train_entries, val_entries, tmpl
    )

    _log.info("v5_auc=%.4f  v6_val_auc=%.4f", V5_AUC, v6_val_auc)

    # Save model if threshold met
    model_saved = v6_val_auc >= 0.80
    if model_saved:
        _save_model(best_weights, best_bias, V6_MODEL_PATH)

    # Determine verdict
    if v6_val_auc >= 0.80:
        honest_verdict = "nup_v6_tier0c_ready"
    elif v6_val_auc > V5_AUC:
        honest_verdict = "nup_v6_improved"
    else:
        honest_verdict = "no_improvement"

    artifact = tmpl.build_result(
        {
            "n_live_pairs": n_live_pairs,
            "train_pairs": train_pairs,
            "val_pairs": val_pairs,
            "v5_auc": V5_AUC,
            "v6_val_auc": float(v6_val_auc),
            "capo_applied": True,
            "lambda_cal": LAMBDA_CAL,
            "model_saved": model_saved,
            "tier_0c_deployable": model_saved,
            "retro_049_resolved": model_saved,
            "honest_verdict": honest_verdict,
        },
        status="success",
        schema="carnot.nup_probe_v6.v1",
    )

    writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))
    writer.write(artifact)
    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
