#!/usr/bin/env python3
"""Experiment 646: JEPA v14 Platt Temperature Scaling.

**Researcher summary:**
    JEPA v14 (Exp 631) achieved excellent OOD AUC=0.912 but ECE=0.132, above the
    calibration target of < 0.10.  The model discriminates correctly but its
    confidence scores are overconfident.

    This experiment applies Platt temperature scaling (Guo et al., 2017): a single
    scalar T is fitted on a 20% calibration split of fover_corpus_v5_oracle.json to
    minimise NLL.  The remaining 80% is used for evaluation.  Temperature scaling
    preserves AUC (ordering is unchanged) while reducing ECE by softening the
    sigmoid outputs.

    Expected outcome: ECE drops from 0.132 to < 0.10 (50-70% reduction typical).
    AUC should remain >= 0.892 (within 0.02 of baseline 0.912).

    Gate: calibration_target_met = (ece_after < 0.10).

Spec: REQ-VERIFY-144, SCENARIO-VERIFY-190, SCENARIO-VERIFY-191
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX import.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

from scripts.experiment_template import ExperimentTemplate  # noqa: E402
from carnot.training.platt_scaler import PlattScaler  # noqa: E402

EXP_ID = 646
EXP_TITLE = "JEPA v14 Platt Scaling"
DELIVERABLE = "results/experiment_646_jepa_v14_platt.json"

ORACLE_CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v5_oracle.json"
V14_MODEL_PATH = _REPO_ROOT / "results" / "jepa_v14_oracle.npz"
V13_MODEL_PATH = _REPO_ROOT / "results" / "jepa_v13_capo.npz"
SCALER_OUT_PATH = _REPO_ROOT / "results" / "jepa_v14_platt_T.json"

EMBED_DIM = 128
SEED = 42
CAL_FRAC = 0.20  # 20% calibration, 80% eval

V14_OOD_AUC_BASELINE = 0.912
V14_ECE_BASELINE = 0.132


# ---------------------------------------------------------------------------
# Embed + score helpers (same projection as Exp 631/618/607 — must match)
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Deterministic random-projection text embedder (identical to Exp 631).

    Must use the same seed and projection matrix as the JEPA v14 training run
    so that loaded weights operate on the same embedding space.
    """
    key = jrandom.PRNGKey(seed)
    proj = jrandom.normal(key, (256, embed_dim)) / np.sqrt(embed_dim)

    def embed_fn(text: str) -> jnp.ndarray:
        if not text:
            return jnp.zeros(embed_dim, dtype=jnp.float32)
        char_indices = jnp.array([ord(c) % 256 for c in text[:512]], dtype=jnp.int32)
        vecs = proj[char_indices]
        return jnp.mean(vecs, axis=0).astype(jnp.float32)

    return embed_fn


def _score(params: dict, emb: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: embedding -> scalar energy.  SiLU activation matches v14."""
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return (params["w2"] @ h + params["b2"])[0]


def _load_model(model_path: Path) -> dict:
    """Load npz weights and reconstruct params dict.

    The npz format stores each leaf as a separate array named after its key.
    This is the same serialisation format used in Exp 618 and 631.

    Args:
        model_path: Path to .npz weights file.

    Returns:
        params dict with keys w1, b1, w2, b2 as jnp arrays.
    """
    data = np.load(str(model_path))
    return {k: jnp.array(data[k]) for k in data.files}


def _compute_logits(params: dict, embed_fn, pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Compute raw logits and binary labels for a list of corpus pairs.

    Each pair is scored as energy = _score(params, embed(response + question)).
    Labels: 1.0 if is_correct=False (incorrect response), 0.0 if is_correct=True.

    Args:
        params:    Loaded JEPA model weights.
        embed_fn:  Text embedding function (same projection as training).
        pairs:     List of dicts with 'response', 'question', 'is_correct'.

    Returns:
        (logits, labels) as float32 numpy arrays of shape (N,).
    """
    logits = []
    labels = []
    for entry in pairs:
        text = (entry.get("response", "") or "") + " " + (entry.get("question", "") or "")
        emb = embed_fn(text)
        logit = float(_score(params, emb))
        label = 0.0 if entry.get("is_correct", True) else 1.0
        logits.append(logit)
        labels.append(label)
    return np.array(logits, dtype=np.float32), np.array(labels, dtype=np.float32)


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC using the trapezoidal rule (no sklearn dependency).

    For binary AUROC: sort by decreasing score, compute TPR/FPR at each threshold.
    Returns AUC in [0, 1].  AUC=0.5 is random; AUC=1.0 is perfect.

    Args:
        scores: Model scores (higher = more likely incorrect).
        labels: Binary labels (1=incorrect, 0=correct).

    Returns:
        float AUROC in [0, 1].
    """
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    sorted_idx = np.argsort(-scores)
    labels_sorted = labels[sorted_idx]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend (0, 0) for trapezoidal integration.
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    return float(np.trapz(tpr, fpr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run JEPA v14 Platt temperature scaling calibration."""
    _watchdog = ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30)

    tmpl = ExperimentTemplate(
        EXP_ID,
        EXP_TITLE,
        DELIVERABLE,
        requires_gpu=False,
    )
    tmpl.setup()

    # --- Load JEPA model weights (v14 preferred, v13 fallback) ---------------
    if V14_MODEL_PATH.exists():
        _log.info("Loading JEPA v14 from %s", V14_MODEL_PATH)
        params = _load_model(V14_MODEL_PATH)
        model_source = "jepa_v14_oracle"
    elif V13_MODEL_PATH.exists():
        _log.info("v14 not found; falling back to JEPA v13 from %s", V13_MODEL_PATH)
        params = _load_model(V13_MODEL_PATH)
        model_source = "jepa_v13_capo_fallback"
    else:
        artifact = tmpl.build_result(
            {"model_source": "none", "error": "Neither v14 nor v13 weights found"},
            status="blocked",
        )
        Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))
        tmpl.assert_deliverable_written()
        return

    embed_fn = _make_embed_fn()

    # --- Load oracle corpus ---------------------------------------------------
    _log.info("Loading oracle corpus from %s", ORACLE_CORPUS_PATH)
    chains = json.loads(ORACLE_CORPUS_PATH.read_text())
    _log.info("Corpus: %d chains", len(chains))

    # Flatten to pairs (one per chain — use model_response + question).
    pairs = [
        {
            "response": c.get("model_response", ""),
            "question": c.get("question", ""),
            "is_correct": c.get("is_correct", True),
        }
        for c in chains
    ]

    # 20% calibration / 80% evaluation split (deterministic by index).
    n_cal = max(1, int(len(pairs) * CAL_FRAC))
    cal_pairs = pairs[:n_cal]
    val_pairs = pairs[n_cal:]
    _log.info("Split: %d cal / %d val", len(cal_pairs), len(val_pairs))

    # --- Compute raw logits ---------------------------------------------------
    cal_logits, cal_labels = _compute_logits(params, embed_fn, cal_pairs)
    val_logits, val_labels = _compute_logits(params, embed_fn, val_pairs)

    # --- Baseline ECE (before Platt) -----------------------------------------
    scaler_pre = PlattScaler(init_temperature=1.0)
    raw_probs_val = np.array(jax.nn.sigmoid(jnp.array(val_logits)))
    ece_before = scaler_pre.compute_ece(jnp.array(raw_probs_val), jnp.array(val_labels))
    auc_before = _auroc(val_logits, val_labels)
    _log.info("Baseline — ECE=%.4f, AUC=%.4f", ece_before, auc_before)

    # --- Fit Platt scaler on calibration split --------------------------------
    scaler = PlattScaler(init_temperature=1.0)
    T_optimal = scaler.fit(jnp.array(cal_logits), jnp.array(cal_labels))
    _log.info("Fitted T=%.4f", T_optimal)

    # --- Evaluate on validation split ----------------------------------------
    cal_probs_val = scaler.calibrate(jnp.array(val_logits))
    ece_after = scaler.compute_ece(cal_probs_val, jnp.array(val_labels))
    auc_after = _auroc(np.array(cal_probs_val), val_labels)
    _log.info("After Platt — ECE=%.4f, AUC=%.4f", ece_after, auc_after)

    ece_reduction_pct = 100.0 * (ece_before - ece_after) / max(ece_before, 1e-9)

    # --- Save scaler ---------------------------------------------------------
    SCALER_OUT_PATH.write_text(json.dumps({"temperature": T_optimal}, indent=2))
    _log.info("Saved scaler to %s", SCALER_OUT_PATH)

    # --- Determine verdict ---------------------------------------------------
    if ece_after < 0.10:
        honest_verdict = "platt_calibrated"
    elif ece_after < V14_ECE_BASELINE:
        honest_verdict = "platt_improved_not_calibrated"
    else:
        honest_verdict = "platt_no_improvement"

    calibration_target_met = bool(ece_after < 0.10)

    # --- Build and write artifact --------------------------------------------
    artifact = tmpl.build_result(
        {
            "schema": "carnot.jepa_v14_platt.v1",
            "model_source": model_source,
            "n_cal": len(cal_pairs),
            "n_val": len(val_pairs),
            "v14_ece_before": float(ece_before),
            "ece_after": float(ece_after),
            "T_optimal": float(T_optimal),
            "ece_reduction_pct": float(ece_reduction_pct),
            "v14_auc_before": float(auc_before),
            "auc_after": float(auc_after),
            "calibration_target_met": calibration_target_met,
            "scaler_saved": str(SCALER_OUT_PATH),
            "honest_verdict": honest_verdict,
        },
        status="success",
        decision_class="verify",
    )
    Path(DELIVERABLE).write_text(json.dumps(artifact, indent=2))
    _log.info(
        "Done — ECE %.4f -> %.4f (%.1f%% reduction), AUC %.4f -> %.4f, verdict=%s",
        ece_before, ece_after, ece_reduction_pct, auc_before, auc_after, honest_verdict,
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
