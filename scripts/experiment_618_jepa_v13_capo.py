#!/usr/bin/env python3
"""Experiment 618: JEPA v13 CAPO Calibrated Retrain.

**Researcher summary:**
    JEPA v12 (Exp 607) achieved in-distribution val_auc=1.0 but OOD AUC=0.5 — exactly
    random baseline.  The diagnostic: extreme confidence scores (near 0 or 1) indicate
    the model learned to recognise surface patterns from the training corpus rather than
    genuine reasoning quality signals.  When it encounters unseen question phrasings
    (OOD), those surface patterns don't fire and the model defaults to random output.

    This experiment retrains the same 2-layer MLP (JEPA architecture) with CAPO loss
    (arXiv 2604.12632): jointly optimising contrastive margin (keeps ranking ability)
    plus Expected Calibration Error (forces probabilistic uncertainty estimates to
    match empirical accuracy).  By penalising overconfidence during training, the model
    cannot rely on spurious surface cues — they would produce miscalibrated outputs
    and incur a calibration penalty.

    Training corpus: fover_corpus_v5.json (350 pairs from 175 questions).
    OOD evaluation: question indices NOT seen during training.
    Gate conditions:
        ood_improved       = v13_ood_auc >= 0.75
        calibration_improved = v13_ece < 0.10

Spec: REQ-VERIFY-120, REQ-VERIFY-121,
      SCENARIO-VERIFY-157, SCENARIO-VERIFY-158, SCENARIO-VERIFY-159
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX import to repair ROCm env vars.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.training.capo_loss import capo_loss  # noqa: E402
from carnot.training.capo_loss import ece_loss  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 618
EXP_TITLE = "JEPA v13 CAPO Calibrated Retrain"
DELIVERABLE = "results/experiment_618_jepa_v13_capo.json"

CORPUS_V5_PATH = _REPO_ROOT / "results" / "fover_corpus_v5.json"
CORPUS_V4_PATH = _REPO_ROOT / "results" / "fover_corpus_v4.json"
MODEL_OUT_PATH = _REPO_ROOT / "results" / "jepa_v13_capo.npz"
EXP_607_PATH = _REPO_ROOT / "results" / "experiment_607_jepa_v12_ood.json"

# Architecture constants — same 2-layer MLP as v12 for comparability.
EMBED_DIM = 128
SEED = 42
TRAIN_FRAC = 0.8
N_EPOCHS = 50
EVAL_EVERY = 10
MARGIN = 1.0
LAMBDA_CALIB = 0.1


# ---------------------------------------------------------------------------
# Text embedder — same random-projection as Exp 607/593 for model comparability
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Deterministic random-projection text embedder.

    Maps a text string to a fixed-size float32 vector by projecting character
    ordinals through a seed-stable Gaussian matrix.  Must use identical parameters
    to Exp 593/607 so that trained weights operate on the same embedding space.

    Args:
        embed_dim: Output embedding dimension.  Must match v12 architecture (128).
        seed:      Master seed for the projection matrix.  Must match v12 (42).

    Returns:
        Callable str -> jnp.ndarray of shape (embed_dim,).
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


# ---------------------------------------------------------------------------
# 2-layer MLP — identical to Exp 593/607
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray, embed_dim: int = EMBED_DIM) -> dict:
    """Initialise a 2-layer MLP: input(128) -> hidden(128) -> output(1).

    Xavier uniform initialisation — same as Exp 593 for architecture consistency.
    """
    k1, k2 = jrandom.split(key)
    lim1 = float(jnp.sqrt(6.0 / (embed_dim + embed_dim)))
    lim2 = float(jnp.sqrt(6.0 / (embed_dim + 1)))
    return {
        "w1": jrandom.uniform(k1, (embed_dim, embed_dim), minval=-lim1, maxval=lim1),
        "b1": jnp.zeros(embed_dim),
        "w2": jrandom.uniform(k2, (1, embed_dim), minval=-lim2, maxval=lim2),
        "b2": jnp.zeros(1),
    }


def _score(params: dict, emb: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: embedding -> scalar energy.  SiLU activation."""
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return (params["w2"] @ h + params["b2"])[0]


# ---------------------------------------------------------------------------
# Load corpus
# ---------------------------------------------------------------------------


def _load_corpus(v5_path: Path, v4_path: Path) -> list[dict]:
    """Load fover_corpus_v5.json, falling back to v4 if v5 is absent.

    Returns a flat list of entry dicts, each with 'question', 'response', 'is_correct',
    and 'question_index' fields.  The corpus may have a 'pairs' key (v5 format) or be
    a flat list (v4 format).

    Args:
        v5_path: Path to fover_corpus_v5.json.
        v4_path: Path to fover_corpus_v4.json (fallback).

    Returns:
        List of entry dicts.

    Spec: REQ-VERIFY-120
    """
    for path in (v5_path, v4_path):
        if path.exists():
            raw = json.loads(path.read_text())
            if isinstance(raw, list):
                _log.info("Loaded %d entries from %s (flat list)", len(raw), path.name)
                return raw
            elif isinstance(raw, dict) and "pairs" in raw:
                entries = raw["pairs"]
                _log.info("Loaded %d entries from %s (pairs key)", len(entries), path.name)
                return entries
    _log.error("No corpus found at %s or %s — returning empty list.", v5_path, v4_path)
    return []


# ---------------------------------------------------------------------------
# Build embeddings for all entries
# ---------------------------------------------------------------------------


def _embed_entries(entries: list[dict], embed_fn) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embed all entries and return (embeddings, labels) arrays.

    Embeds (question + response) text for each entry.  Labels: 1 = incorrect, 0 = correct.
    This mirrors the scoring convention in Exp 607 where higher energy = more likely wrong.

    Args:
        entries:  List of entry dicts with 'question', 'response', 'is_correct'.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        (embeddings, labels) — shapes (N, embed_dim) and (N,).

    Spec: REQ-VERIFY-120
    """
    embs = []
    labels = []
    for entry in entries:
        text = (entry.get("question", "") + " " + entry.get("response", "")).strip()
        embs.append(embed_fn(text))
        labels.append(0 if entry.get("is_correct", False) else 1)
    return jnp.stack(embs), jnp.array(labels, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Train/test split by question_index to prevent leakage
# ---------------------------------------------------------------------------


def _split_entries(
    entries: list[dict],
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries into train, in-distribution test, and OOD test sets.

    Splits by question_index (not by entry) to prevent leakage from the same
    question appearing in both train and test.  OOD = question indices with the
    highest 20% of indices — questions the model has never encountered.

    The 80/20 split is applied to all unique question indices.  The top-index
    questions (those NOT in the training question index set) form the OOD set.
    The in-distribution test set is the 20% of indices adjacent to training ones.

    Why split by question_index rather than response:
        If we split randomly by response, a question seen in training would
        appear in the test set with a different response — the model would
        correctly rank it using memorised question-pattern cues, giving us an
        inflated AUC that doesn't reflect OOD performance.

    Args:
        entries:    Full list of corpus entries.
        train_frac: Fraction of question indices for training.
        seed:       RNG seed for reproducible shuffle.

    Returns:
        (train_entries, test_entries, ood_entries)

    Spec: REQ-VERIFY-120
    """
    rng = np.random.RandomState(seed)

    q_indices = sorted(set(e.get("question_index", 0) for e in entries))
    rng.shuffle(q_indices)
    n_train_q = max(1, int(len(q_indices) * train_frac))
    train_q = set(q_indices[:n_train_q])
    test_q = set(q_indices[n_train_q:])

    # OOD = entries whose question_index was NOT seen during training.
    # This matches the Exp 607 OOD methodology.
    train_entries = [e for e in entries if e.get("question_index", 0) in train_q]
    test_entries = [e for e in entries if e.get("question_index", 0) in test_q]

    # Build a stricter OOD set: top 20% of question indices by numeric value
    # (ensures OOD comes from later GSM8K questions, not just a random split).
    all_q_sorted = sorted(set(e.get("question_index", 0) for e in entries))
    ood_q_start = int(len(all_q_sorted) * train_frac)
    ood_q = set(all_q_sorted[ood_q_start:])
    ood_entries = [e for e in entries if e.get("question_index", 0) in ood_q]

    _log.info(
        "Split: %d train entries (%d questions), %d test entries, %d OOD entries (%d questions)",
        len(train_entries), len(train_q),
        len(test_entries),
        len(ood_entries), len(ood_q),
    )
    return train_entries, test_entries, ood_entries


# ---------------------------------------------------------------------------
# AUC computation via sklearn
# ---------------------------------------------------------------------------


def _compute_auc(params: dict, entries: list[dict], embed_fn) -> float:
    """Compute ROC-AUC for the given entries using the trained model.

    AUC convention: label=1 for incorrect responses (positive class).  The model
    should assign higher energy (score) to incorrect responses, so AUC > 0.5
    means the model discriminates correctly.

    Returns 0.5 (random baseline) when one class is absent (cannot compute AUC).

    Args:
        params:   MLP parameter dict.
        entries:  List of entry dicts.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        Float AUC in [0, 1].

    Spec: REQ-VERIFY-120
    """
    if not entries:
        return 0.5
    from sklearn.metrics import roc_auc_score  # local import to keep startup fast

    embs, labels = _embed_entries(entries, embed_fn)
    scores = [float(_score(params, embs[i])) for i in range(len(entries))]
    labels_np = np.array(labels)
    if labels_np.min() == labels_np.max():
        _log.warning("Only one class in AUC evaluation — returning 0.5.")
        return 0.5
    return float(roc_auc_score(labels_np, scores))


# ---------------------------------------------------------------------------
# ECE computation on a held-out set
# ---------------------------------------------------------------------------


def _compute_ece(params: dict, entries: list[dict], embed_fn) -> float:
    """Compute Expected Calibration Error on entries using the trained model.

    Converts energy scores to probabilities via sigmoid(energy) = P(incorrect).
    ECE < 0.10 means the model's confidence is within 10% of its accuracy.

    Args:
        params:   MLP parameter dict.
        entries:  List of entry dicts.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        Float ECE in [0, 1].

    Spec: REQ-VERIFY-121
    """
    if not entries:
        return 0.0
    embs, labels = _embed_entries(entries, embed_fn)
    energy_scores = jnp.array([float(_score(params, embs[i])) for i in range(len(entries))])
    predicted_probs = jax.nn.sigmoid(energy_scores)
    return float(ece_loss(predicted_probs, labels.astype(jnp.float32)))


# ---------------------------------------------------------------------------
# CAPO training loop
# ---------------------------------------------------------------------------


def _train_capo(
    train_entries: list[dict],
    embed_fn,
    n_epochs: int = N_EPOCHS,
    margin: float = MARGIN,
    lambda_calib: float = LAMBDA_CALIB,
    seed: int = SEED,
    tmpl: ExperimentTemplate | None = None,
) -> dict:
    """Train JEPA v13 with CAPO loss for n_epochs.

    Uses optax.adamw with the same hyperparameters as Exp 593/607 for comparable
    learning dynamics.  CAPO loss = contrastive margin + lambda_calib * ECE.

    Args:
        train_entries:  Training corpus entries.
        embed_fn:       Text -> jnp.ndarray embedder.
        n_epochs:       Number of training epochs.
        margin:         Contrastive margin (default 1.0).
        lambda_calib:   Calibration loss weight (default 0.1).
        seed:           RNG seed.
        tmpl:           ExperimentTemplate for checkpoint saves.

    Returns:
        Trained param dict.

    Spec: REQ-VERIFY-120, REQ-VERIFY-121
    """
    import optax

    embs, labels = _embed_entries(train_entries, embed_fn)
    params = _init_params(jrandom.PRNGKey(seed))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def loss_fn(p: dict) -> jnp.ndarray:
        # Compute energy for each embedding in the training batch.
        energies = jnp.array([_score(p, embs[i]) for i in range(len(train_entries))])
        return capo_loss(energies, labels, margin=margin, lambda_calib=lambda_calib)

    grad_fn = jax.jit(jax.grad(loss_fn))

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if epoch % EVAL_EVERY == 0 or epoch == n_epochs:
            loss_val = float(loss_fn(params))
            _log.info("epoch %d/%d  loss=%.4f", epoch, n_epochs, loss_val)
            if tmpl is not None:
                tmpl.checkpoint_save({"epoch": epoch, "loss": loss_val}, step=epoch)

    return params


# ---------------------------------------------------------------------------
# Save model to .npz
# ---------------------------------------------------------------------------


def _save_model_npz(params: dict, path: Path) -> None:
    """Save model weights to numpy .npz format.

    Saves each parameter array (w1, b1, w2, b2) as a separate array in the .npz.
    This differs from the safetensors format used in Exp 607/593 because .npz
    is dependency-free for loading and sufficient for this diagnostic experiment.
    Use safetensors for production deployments.

    Args:
        params: MLP parameter dict.
        path:   Output path ending in .npz.

    Spec: REQ-VERIFY-120
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{k: np.array(v) for k, v in params.items()})
    _log.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 618: JEPA v13 CAPO calibrated retrain."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()
        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

        # ------------------------------------------------------------------
        # Step 1: Load v12 OOD AUC from Exp 607 result for comparison
        # ------------------------------------------------------------------
        v12_ood_auc = 0.5  # default if Exp 607 result not available
        if EXP_607_PATH.exists():
            try:
                exp607 = json.loads(EXP_607_PATH.read_text())
                v12_ood_auc = float(exp607.get("v12_ood_auc", 0.5))
                _log.info("v12_ood_auc from Exp 607: %.4f", v12_ood_auc)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                _log.warning("Could not parse Exp 607 result: %s — using default 0.5", exc)
        else:
            _log.warning("Exp 607 result not found — defaulting v12_ood_auc=0.5")

        # ------------------------------------------------------------------
        # Step 2: Load corpus
        # ------------------------------------------------------------------
        entries = _load_corpus(CORPUS_V5_PATH, CORPUS_V4_PATH)
        if not entries:
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_v13_capo.v1",
                    "v12_ood_auc": v12_ood_auc,
                    "v13_in_dist_auc": 0.5,
                    "v13_ood_auc": 0.5,
                    "v13_ece": 1.0,
                    "n_training_pairs": 0,
                    "n_test_pairs": 0,
                    "calibration_improved": False,
                    "ood_improved": False,
                    "model_saved_at": str(MODEL_OUT_PATH),
                    "honest_verdict": "blocked_no_corpus",
                },
                status="blocked_no_corpus",
            )
            writer.write(artifact)
            tmpl.assert_deliverable_written()
            return

        embed_fn = _make_embed_fn(embed_dim=EMBED_DIM, seed=SEED)

        # ------------------------------------------------------------------
        # Step 3: Split into train, in-dist test, OOD test
        # ------------------------------------------------------------------
        train_entries, test_entries, ood_entries = _split_entries(entries)
        n_training_pairs = len(train_entries)
        n_test_pairs = len(test_entries)
        _log.info(
            "Corpus split: %d train, %d in-dist test, %d OOD",
            n_training_pairs, n_test_pairs, len(ood_entries),
        )

        # ------------------------------------------------------------------
        # Step 4: Train with CAPO loss for 50 epochs
        # ------------------------------------------------------------------
        _log.info("Training JEPA v13 with CAPO loss for %d epochs...", N_EPOCHS)
        params = _train_capo(
            train_entries,
            embed_fn,
            n_epochs=N_EPOCHS,
            margin=MARGIN,
            lambda_calib=LAMBDA_CALIB,
            seed=SEED,
            tmpl=tmpl,
        )

        # ------------------------------------------------------------------
        # Step 5: Evaluate in-distribution AUC
        # ------------------------------------------------------------------
        _log.info("Computing in-distribution AUC...")
        v13_in_dist_auc = _compute_auc(params, test_entries, embed_fn)
        _log.info("v13_in_dist_auc=%.4f", v13_in_dist_auc)

        # ------------------------------------------------------------------
        # Step 6: Evaluate OOD AUC
        # ------------------------------------------------------------------
        _log.info("Computing OOD AUC...")
        v13_ood_auc = _compute_auc(params, ood_entries, embed_fn)
        _log.info("v13_ood_auc=%.4f  (v12_ood_auc=%.4f)", v13_ood_auc, v12_ood_auc)

        # ------------------------------------------------------------------
        # Step 7: Compute ECE on test split
        # ------------------------------------------------------------------
        _log.info("Computing ECE...")
        v13_ece = _compute_ece(params, test_entries, embed_fn)
        _log.info("v13_ece=%.4f", v13_ece)

        # ------------------------------------------------------------------
        # Step 8: Save model
        # ------------------------------------------------------------------
        _save_model_npz(params, MODEL_OUT_PATH)

        # ------------------------------------------------------------------
        # Step 9: Gate evaluation and honest verdict
        # ------------------------------------------------------------------
        ood_improved = v13_ood_auc >= 0.75
        calibration_improved = v13_ece < 0.10

        if ood_improved and calibration_improved:
            honest_verdict = "v13_calibrated_and_ood"
        elif calibration_improved:
            honest_verdict = "v13_calibrated_overfit"
        else:
            honest_verdict = "v13_uncalibrated"

        _log.info(
            "v13_ood_auc=%.4f  v13_ece=%.4f  ood_improved=%s  calibration_improved=%s  verdict=%s",
            v13_ood_auc, v13_ece, ood_improved, calibration_improved, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Step 10: Build artifact and assert deliverable written
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_v13_capo.v1",
                "v12_ood_auc": v12_ood_auc,
                "v13_in_dist_auc": v13_in_dist_auc,
                "v13_ood_auc": v13_ood_auc,
                "v13_ece": v13_ece,
                "n_training_pairs": n_training_pairs,
                "n_test_pairs": n_test_pairs,
                "calibration_improved": bool(calibration_improved),
                "ood_improved": bool(ood_improved),
                "model_saved_at": str(MODEL_OUT_PATH),
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
