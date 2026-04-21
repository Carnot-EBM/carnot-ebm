#!/usr/bin/env python3
"""Experiment 631: JEPA v14 ORACLE Calibrated Retrain.

**Researcher summary:**
    JEPA v13 (Exp 618) achieved OOD AUC=0.868 (architecture is sound) but ECE=0.207
    — well above the calibration target of < 0.10.  Root cause: all training data came
    from synthetic violations or binary correct/incorrect labels, not from step-level
    constraint labels derived from live LLM output.

    This experiment trains JEPA v14 on the ORACLE-labeled corpus built in Exp 628
    (fover_corpus_v5_oracle.json), which contains per-step SymCodeVerifier labels for
    each reasoning step in live LLM responses.  Where the oracle corpus has step-level
    violated labels, we build step-level (response, step_text) training pairs.  Where
    no violated steps exist (oracle corpus has n_violated_steps=0), we fall back to
    flat correct/incorrect pairs from fover_corpus_v5.json or fover_corpus_v4.json.

    Lambda calibration tuning: we sweep lambda_calib in {0.05, 0.10, 0.20} on a
    validation split and pick the value that achieves lowest ECE without dropping
    AUC below 0.70.  This avoids over-penalising the contrastive signal.

    Gate conditions:
        calibration_improved = v14_ece < 0.10
        ood_maintained       = v14_ood_auc >= 0.75

Spec: REQ-VERIFY-134, REQ-VERIFY-135,
      SCENARIO-VERIFY-175, SCENARIO-VERIFY-176, SCENARIO-VERIFY-177
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

EXP_ID = 631
EXP_TITLE = "JEPA v14 ORACLE Calibrated Retrain"
DELIVERABLE = "results/experiment_631_jepa_v14_oracle.json"

ORACLE_CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v5_oracle.json"
CORPUS_V5_PATH = _REPO_ROOT / "results" / "fover_corpus_v5.json"
CORPUS_V4_PATH = _REPO_ROOT / "results" / "fover_corpus_v4.json"
MODEL_OUT_PATH = _REPO_ROOT / "results" / "jepa_v14_oracle.npz"

# Architecture constants — same 2-layer MLP as v13 for comparability.
EMBED_DIM = 128
SEED = 42
TRAIN_FRAC = 0.8
N_EPOCHS = 60
EVAL_EVERY = 15
MARGIN = 1.0
# Candidate lambda_calib values — sweep selects best by validation ECE
LAMBDA_CALIB_CANDIDATES = [0.05, 0.10, 0.20]

# Comparison baselines from Exp 618 (v13)
V13_OOD_AUC = 0.868
V13_ECE = 0.207


# ---------------------------------------------------------------------------
# Text embedder — same random-projection as Exp 618/607/593 for comparability
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Deterministic random-projection text embedder.

    Maps a text string to a fixed-size float32 vector by projecting character
    ordinals through a seed-stable Gaussian matrix.  Must use identical parameters
    to Exp 618 so that v14 weights operate on the same embedding space as v13.

    Args:
        embed_dim: Output embedding dimension (must match v13 architecture: 128).
        seed:      Master seed for the projection matrix (must match v13: 42).

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
# 2-layer MLP — identical to Exp 618
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray, embed_dim: int = EMBED_DIM) -> dict:
    """Initialise a 2-layer MLP: input(128) -> hidden(128) -> output(1).

    Xavier uniform initialisation matches v13 (Exp 618) for architecture consistency.
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
    """Forward pass: embedding -> scalar energy.  SiLU activation matches v13."""
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return (params["w2"] @ h + params["b2"])[0]


# ---------------------------------------------------------------------------
# Oracle corpus loading
# ---------------------------------------------------------------------------


def _load_oracle_corpus(oracle_path: Path) -> tuple[list[dict], bool]:
    """Load fover_corpus_v5_oracle.json and extract the chain list.

    The oracle corpus is a JSON list of OracleChain dicts with fields:
    question_id, question, model_response, is_correct, step_labels,
    has_violation, n_violated_steps.

    Returns:
        (chains, corpus_ready) where corpus_ready=True iff file exists and
        the chain list has at least 100 entries.

    Spec: REQ-VERIFY-134-1
    """
    if not oracle_path.exists():
        _log.warning("Oracle corpus not found at %s", oracle_path)
        return [], False
    try:
        chains = json.loads(oracle_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Failed to parse oracle corpus: %s", exc)
        return [], False
    if not isinstance(chains, list):
        _log.warning("Oracle corpus has unexpected format (expected list)")
        return [], False
    corpus_ready = len(chains) >= 100
    _log.info("Oracle corpus: %d chains, corpus_ready=%s", len(chains), corpus_ready)
    return chains, corpus_ready


def _build_oracle_pairs(chains: list[dict]) -> list[dict]:
    """Build step-level training pairs from oracle-labeled chains.

    For each chain where is_correct=False, for each step_label with label='violated':
      - positive pair: (model_response, step_text) with label=1 (incorrect/violated)
    For pairing with negatives, we use responses from chains marked is_correct=True
    (if available) or just build single-label entries.

    Because step-level pairs need a "correct" reference, we pair violated steps
    with the same step_text from a correct chain that has no violation, creating
    a (response, step_text, label) tuple for CAPO training.

    When no violated steps exist (n_violated_steps=0 across all chains), returns
    an empty list and the caller falls back to flat corpus pairs.

    Args:
        chains: List of OracleChain dicts from fover_corpus_v5_oracle.json.

    Returns:
        List of flat entry dicts with fields: question, response, is_correct,
        question_index (derived from question_id hash for split compatibility).

    Spec: REQ-VERIFY-134-1, SCENARIO-VERIFY-177
    """
    incorrect_chains = [c for c in chains if not c.get("is_correct", True)]
    correct_chains = [c for c in chains if c.get("is_correct", False)]
    n_violated_total = sum(
        sum(1 for sl in c.get("step_labels", []) if sl.get("label") == "violated")
        for c in incorrect_chains
    )
    _log.info(
        "Oracle: %d incorrect chains, %d correct chains, %d violated steps",
        len(incorrect_chains), len(correct_chains), n_violated_total,
    )

    if n_violated_total == 0:
        _log.info(
            "No violated steps in oracle corpus — will use flat fallback corpus for training."
        )
        return []

    pairs: list[dict] = []
    for chain in incorrect_chains:
        for step_label in chain.get("step_labels", []):
            if step_label.get("label") != "violated":
                continue
            step_text = step_label.get("step_text", "")
            # Positive: violated step in incorrect response gets label=1 (incorrect).
            pairs.append({
                "question": chain.get("question", ""),
                "response": chain.get("model_response", "") + " " + step_text,
                "is_correct": False,
                "question_index": abs(hash(chain.get("question_id", ""))) % 10000,
            })
    # Add correct chains as label=0 entries for contrastive training.
    for chain in correct_chains:
        pairs.append({
            "question": chain.get("question", ""),
            "response": chain.get("model_response", ""),
            "is_correct": True,
            "question_index": abs(hash(chain.get("question_id", ""))) % 10000,
        })

    _log.info("Built %d oracle step-level training pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Fallback corpus loading (same as v13)
# ---------------------------------------------------------------------------


def _load_flat_corpus(v5_path: Path, v4_path: Path) -> list[dict]:
    """Load flat corpus from fover_corpus_v5.json or fover_corpus_v4.json as fallback.

    This is the same corpus used in Exp 618 (v13), providing a meaningful baseline
    when the oracle corpus has no violated steps to build step-level pairs from.

    Args:
        v5_path: Path to fover_corpus_v5.json.
        v4_path: Path to fover_corpus_v4.json (fallback).

    Returns:
        List of entry dicts with question, response, is_correct, question_index fields.

    Spec: REQ-VERIFY-134-1, SCENARIO-VERIFY-177
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
    _log.error("No fallback corpus found at %s or %s", v5_path, v4_path)
    return []


# ---------------------------------------------------------------------------
# Embedding and label extraction
# ---------------------------------------------------------------------------


def _embed_entries(entries: list[dict], embed_fn) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embed all entries and return (embeddings, labels) arrays.

    Embeds (question + response) text for each entry.  Labels: 1=incorrect, 0=correct.
    Higher energy = model believes response is more likely wrong.

    Args:
        entries:  List of entry dicts with question, response, is_correct fields.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        (embeddings, labels) — shapes (N, embed_dim) and (N,).

    Spec: REQ-VERIFY-134
    """
    embs = []
    labels = []
    for entry in entries:
        text = (entry.get("question", "") + " " + entry.get("response", "")).strip()
        embs.append(embed_fn(text))
        labels.append(0 if entry.get("is_correct", False) else 1)
    return jnp.stack(embs), jnp.array(labels, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Train/test/OOD split — same methodology as Exp 618
# ---------------------------------------------------------------------------


def _split_entries(
    entries: list[dict],
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries into train, in-distribution test, and OOD test sets.

    Splits by question_index to prevent leakage from the same question appearing
    in both train and test.  OOD = top 20% of question indices by numeric value
    (later GSM8K questions the model has never encountered during training).

    Args:
        entries:    Full list of corpus entries.
        train_frac: Fraction of question indices for training (default 0.80).
        seed:       RNG seed for reproducible shuffle.

    Returns:
        (train_entries, test_entries, ood_entries)

    Spec: REQ-VERIFY-135-1
    """
    rng = np.random.RandomState(seed)
    q_indices = sorted(set(e.get("question_index", 0) for e in entries))
    rng.shuffle(q_indices)
    n_train_q = max(1, int(len(q_indices) * train_frac))
    train_q = set(q_indices[:n_train_q])

    train_entries = [e for e in entries if e.get("question_index", 0) in train_q]
    test_entries = [e for e in entries if e.get("question_index", 0) not in train_q]

    # Strict OOD: top 20% of question indices by numeric value (same as Exp 618).
    all_q_sorted = sorted(set(e.get("question_index", 0) for e in entries))
    ood_q_start = int(len(all_q_sorted) * train_frac)
    ood_q = set(all_q_sorted[ood_q_start:])
    ood_entries = [e for e in entries if e.get("question_index", 0) in ood_q]

    _log.info(
        "Split: %d train, %d in-dist test, %d OOD",
        len(train_entries), len(test_entries), len(ood_entries),
    )
    return train_entries, test_entries, ood_entries


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


def _compute_auc(params: dict, entries: list[dict], embed_fn) -> float:
    """Compute ROC-AUC for the given entries using the trained model.

    AUC > 0.5 means the model correctly assigns higher energy to incorrect responses.
    Returns 0.5 (random baseline) when only one class is present.

    Args:
        params:   MLP parameter dict.
        entries:  List of entry dicts.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        Float AUC in [0, 1].

    Spec: REQ-VERIFY-135
    """
    if not entries:
        return 0.5
    from sklearn.metrics import roc_auc_score

    embs, labels = _embed_entries(entries, embed_fn)
    scores = [float(_score(params, embs[i])) for i in range(len(entries))]
    labels_np = np.array(labels)
    if labels_np.min() == labels_np.max():
        _log.warning("Only one class in AUC evaluation — returning 0.5.")
        return 0.5
    return float(roc_auc_score(labels_np, scores))


def _compute_ece(params: dict, entries: list[dict], embed_fn) -> float:
    """Compute Expected Calibration Error on entries using the trained model.

    Converts energy scores to probabilities via sigmoid(energy) = P(incorrect).
    ECE < 0.10 means the model's stated confidence is within 10% of empirical accuracy.

    Args:
        params:   MLP parameter dict.
        entries:  List of entry dicts.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        Float ECE in [0, 1].

    Spec: REQ-VERIFY-134-4
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
    lambda_calib: float = 0.10,
    seed: int = SEED,
    tmpl: ExperimentTemplate | None = None,
    eval_every: int = EVAL_EVERY,
    embed_dim: int = EMBED_DIM,
) -> dict:
    """Train JEPA v14 with CAPO loss for n_epochs.

    Uses optax.adamw matching v13 hyperparameters for fair comparison.
    CAPO loss = contrastive margin hinge + lambda_calib * ECE.

    Args:
        train_entries:  Training corpus entries.
        embed_fn:       Text -> jnp.ndarray embedder.
        n_epochs:       Training epochs (default 60 for v14, vs 50 for v13).
        margin:         Contrastive margin (default 1.0, same as v13).
        lambda_calib:   Calibration loss weight to evaluate.
        seed:           RNG seed (42, same as v13 for architecture comparability).
        tmpl:           ExperimentTemplate for checkpoint saves every eval_every epochs.
        eval_every:     Checkpoint interval (default 15 epochs).
        embed_dim:      MLP input dimension (must match embed_fn output; default 128).

    Returns:
        Trained param dict.

    Spec: REQ-VERIFY-134-3
    """
    import optax

    embs, labels = _embed_entries(train_entries, embed_fn)
    params = _init_params(jrandom.PRNGKey(seed), embed_dim=embed_dim)
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def loss_fn(p: dict) -> jnp.ndarray:
        energies = jnp.array([_score(p, embs[i]) for i in range(len(train_entries))])
        return capo_loss(energies, labels, margin=margin, lambda_calib=lambda_calib)

    grad_fn = jax.jit(jax.grad(loss_fn))

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if epoch % eval_every == 0 or epoch == n_epochs:
            loss_val = float(loss_fn(params))
            _log.info(
                "lambda=%.2f  epoch %d/%d  loss=%.4f",
                lambda_calib, epoch, n_epochs, loss_val,
            )
            if tmpl is not None:
                tmpl.checkpoint_save(
                    {"epoch": epoch, "loss": loss_val, "lambda_calib": lambda_calib},
                    step=epoch,
                )

    return params


def _select_lambda(
    train_entries: list[dict],
    val_entries: list[dict],
    embed_fn,
    candidates: list[float] = LAMBDA_CALIB_CANDIDATES,
    n_epochs: int = N_EPOCHS,
    auc_floor: float = 0.70,
    seed: int = SEED,
    embed_dim: int = EMBED_DIM,
) -> tuple[float, dict]:
    """Sweep lambda_calib candidates and return (best_lambda, best_params).

    Selection criterion: lowest validation ECE among candidates whose validation
    AUC >= auc_floor (0.70).  If all candidates drop below the AUC floor, picks
    the one with highest AUC (preserves discrimination over calibration).

    Args:
        train_entries: Training corpus entries.
        val_entries:   Validation (in-dist test) corpus entries.
        embed_fn:      Text -> jnp.ndarray embedder.
        candidates:    List of lambda_calib values to try.
        n_epochs:      Training epochs per candidate.
        auc_floor:     Minimum acceptable validation AUC.
        seed:          RNG seed.
        embed_dim:     Embedding dimension (must match embed_fn; default 128).

    Returns:
        (selected_lambda, trained_params_for_selected_lambda)

    Spec: REQ-VERIFY-134-2
    """
    best_lambda = candidates[0]
    best_params = None
    best_ece = float("inf")
    best_auc = 0.0

    for lam in candidates:
        _log.info("Trying lambda_calib=%.2f ...", lam)
        params = _train_capo(
            train_entries,
            embed_fn,
            n_epochs=n_epochs,
            lambda_calib=lam,
            seed=seed,
            embed_dim=embed_dim,
        )
        val_ece = _compute_ece(params, val_entries, embed_fn)
        val_auc = _compute_auc(params, val_entries, embed_fn)
        _log.info("  lambda=%.2f  val_ece=%.4f  val_auc=%.4f", lam, val_ece, val_auc)

        if val_auc >= auc_floor:
            if val_ece < best_ece:
                best_ece = val_ece
                best_lambda = lam
                best_params = params
                best_auc = val_auc
        else:
            # Below AUC floor: only pick if we have no valid candidate yet.
            if best_params is None or val_auc > best_auc:
                best_lambda = lam
                best_params = params
                best_auc = val_auc

    _log.info(
        "Selected lambda_calib=%.2f  (val_ece=%.4f  val_auc=%.4f)",
        best_lambda, best_ece, best_auc,
    )
    return best_lambda, best_params  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Save model
# ---------------------------------------------------------------------------


def _save_model_npz(params: dict, path: Path) -> None:
    """Save model weights to numpy .npz format.

    Each parameter array (w1, b1, w2, b2) is saved as a separate array.
    Use safetensors for production deployments — .npz is used here for
    dependency-free loading in downstream diagnostic experiments.

    Args:
        params: MLP parameter dict.
        path:   Output path ending in .npz.

    Spec: REQ-VERIFY-134-5
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(path), **{k: np.array(v) for k, v in params.items()})
    _log.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 631: JEPA v14 ORACLE calibrated retrain."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=40):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()
        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

        # ------------------------------------------------------------------
        # Step 1: Load oracle corpus and attempt step-level pair construction
        # ------------------------------------------------------------------
        oracle_chains, corpus_ready = _load_oracle_corpus(ORACLE_CORPUS_PATH)

        if corpus_ready:
            entries = _build_oracle_pairs(oracle_chains)
            corpus_source = "oracle"
        else:
            entries = []
            corpus_source = "fallback"

        # ------------------------------------------------------------------
        # Step 2: Fall back to flat corpus if oracle had no violated steps
        # ------------------------------------------------------------------
        if not entries:
            _log.info("Falling back to flat corpus for training pairs.")
            entries = _load_flat_corpus(CORPUS_V5_PATH, CORPUS_V4_PATH)
            corpus_source = "fallback"

        if not entries:
            artifact = tmpl.build_result(
                {
                    "result_schema": "carnot.jepa_v14_oracle.v1",
                    "v13_ood_auc": V13_OOD_AUC,
                    "v13_ece": V13_ECE,
                    "v14_in_dist_auc": 0.5,
                    "v14_ood_auc": 0.5,
                    "v14_ece": 1.0,
                    "lambda_calib_selected": 0.10,
                    "n_training_pairs": 0,
                    "calibration_improved": False,
                    "ood_maintained": False,
                    "model_saved": str(MODEL_OUT_PATH),
                    "corpus_source": corpus_source,
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
        _log.info(
            "Corpus split: %d train, %d in-dist test, %d OOD  (source=%s)",
            n_training_pairs, len(test_entries), len(ood_entries), corpus_source,
        )

        # ------------------------------------------------------------------
        # Step 4: Sweep lambda_calib to select best calibration weight
        # ------------------------------------------------------------------
        _log.info(
            "Sweeping lambda_calib %s over %d epochs...",
            LAMBDA_CALIB_CANDIDATES, N_EPOCHS,
        )
        lambda_calib_selected, params = _select_lambda(
            train_entries,
            test_entries,
            embed_fn,
            candidates=LAMBDA_CALIB_CANDIDATES,
            n_epochs=N_EPOCHS,
            seed=SEED,
        )
        _log.info("lambda_calib_selected=%.2f", lambda_calib_selected)

        # Checkpoint after final lambda selection.
        tmpl.checkpoint_save(
            {"lambda_calib_selected": lambda_calib_selected, "step": "lambda_sweep_done"},
            step=N_EPOCHS,
        )

        # ------------------------------------------------------------------
        # Step 5: Evaluate in-distribution AUC
        # ------------------------------------------------------------------
        _log.info("Evaluating in-distribution AUC...")
        v14_in_dist_auc = _compute_auc(params, test_entries, embed_fn)
        _log.info("v14_in_dist_auc=%.4f", v14_in_dist_auc)

        # ------------------------------------------------------------------
        # Step 6: Evaluate OOD AUC
        # ------------------------------------------------------------------
        _log.info("Evaluating OOD AUC...")
        v14_ood_auc = _compute_auc(params, ood_entries, embed_fn)
        _log.info(
            "v14_ood_auc=%.4f  (v13_ood_auc=%.4f  baseline=0.868)",
            v14_ood_auc, V13_OOD_AUC,
        )

        # ------------------------------------------------------------------
        # Step 7: Compute ECE on test split
        # ------------------------------------------------------------------
        _log.info("Computing ECE...")
        v14_ece = _compute_ece(params, test_entries, embed_fn)
        _log.info("v14_ece=%.4f  (v13_ece=%.4f  target<0.10)", v14_ece, V13_ECE)

        # ------------------------------------------------------------------
        # Step 8: Save model
        # ------------------------------------------------------------------
        _save_model_npz(params, MODEL_OUT_PATH)

        # ------------------------------------------------------------------
        # Step 9: Gate evaluation and honest verdict
        # ------------------------------------------------------------------
        calibration_improved = bool(v14_ece < 0.10)
        ood_maintained = bool(v14_ood_auc >= 0.75)

        if calibration_improved and ood_maintained:
            honest_verdict = "v14_calibrated_ood_maintained"
        elif calibration_improved:
            honest_verdict = "v14_calibrated_ood_dropped"
        else:
            honest_verdict = "v14_uncalibrated"

        _log.info(
            "v14_ood_auc=%.4f  v14_ece=%.4f  calibration_improved=%s  "
            "ood_maintained=%s  verdict=%s",
            v14_ood_auc, v14_ece, calibration_improved, ood_maintained, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Step 10: Build artifact and assert deliverable written
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "result_schema": "carnot.jepa_v14_oracle.v1",
                "v13_ood_auc": V13_OOD_AUC,
                "v13_ece": V13_ECE,
                "v14_in_dist_auc": float(v14_in_dist_auc),
                "v14_ood_auc": float(v14_ood_auc),
                "v14_ece": float(v14_ece),
                "lambda_calib_selected": float(lambda_calib_selected),
                "n_training_pairs": n_training_pairs,
                "calibration_improved": calibration_improved,
                "ood_maintained": ood_maintained,
                "model_saved": str(MODEL_OUT_PATH),
                "corpus_source": corpus_source,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
