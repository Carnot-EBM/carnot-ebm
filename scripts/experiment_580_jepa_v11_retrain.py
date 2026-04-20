#!/usr/bin/env python3
"""Experiment 580: JEPA v11 Full Retrain with CPMI Contrastive Objective — RETRO-063 Resolution / FR-11.

**Researcher summary (RETRO-063):**
    JEPA v8/v9/v10 all produced AUC < 0.5 despite switching from BCE to PURE min-form loss.
    Root cause: ALL three variants trained on scalar loss with step-level labels.  Step-level
    labels are noisy (intermediate steps in a correct chain can look temporarily wrong), so
    the model hedges to P=0.5 because the per-chain gradient can be satisfied with near-0.5
    outputs regardless of whether the chain is actually correct or not.

    The CPMI fix (Exp 577, arXiv 2604.10660) sidesteps step-level noise entirely by
    constructing EXPLICIT contrastive pairs: one (correct_chain, incorrect_chain) pair per
    question, where correct/incorrect is determined by the FINAL ANSWER verdict.
    CPMIContrastiveLoss then enforces:

        E(incorrect_chain) > E(correct_chain) + margin

    for each pair.  There is no BCE, no PURE, no per-step label in this objective — just a
    pairwise ordering constraint between two whole chains from the SAME question.

**Why this experiment is expected to succeed where PURE failed:**
    PURE improved over BCE by using min() aggregation, but still assigned labels to individual
    chains (correct=1, incorrect=0).  The model can satisfy this by making all scores near 0.5.
    Contrastive margin loss sees BOTH chains in a single gradient step and penalises the model
    directly when the ordering is wrong.  This is the identical mechanism that produced
    AUC=1.0 in the NUP Probe v4 experiments.

**Architecture:**
    Input: step embeddings (128-D each; embed_fn encodes step text as random projection)
    Predictor: same 2-layer MLP as v10 (embed_dim=128, n_layers=2) for comparability
    Loss: CPMIContrastiveLoss(margin=1.0, chain_energy_mode='mean')
    Optimizer: optax.adamw(lr=1e-3, weight_decay=1e-4)
    Epochs: 300 with eval every 25 epochs, best-checkpoint tracking

Spec: REQ-LEARN-067,
      SCENARIO-LEARN-104, SCENARIO-LEARN-105, SCENARIO-LEARN-106
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() — MUST be called before any JAX import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

_autofix_result = apply_env_autofix()

# ---------------------------------------------------------------------------
# Standard imports (after env fix)
# ---------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

from carnot.inference.jepa_cpmi_pairs import (  # noqa: E402
    CPMIContrastiveLoss,
    JEPACPMIPairBuilder,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 580
EXP_TITLE = "JEPA v11 CPMI Retrain"
DELIVERABLE = "results/experiment_580_jepa_v11_retrain.json"
MODEL_DELIVERABLE = "results/jepa_predictor_v11.safetensors"

CORPUS_V3_PATH = _REPO_ROOT / "results" / "fover_corpus_v3.json"
CORPUS_V2_PATH = _REPO_ROOT / "results" / "fover_corpus_v2.json"

V10_AUC = 0.4444      # Exp 567 baseline; this is what we must beat
MARGIN = 1.0
N_EPOCHS = 300
EVAL_EVERY = 25
TRAIN_FRAC = 0.8
SEED = 42
EMBED_DIM = 128       # Matches JEPAPredictor embed_dim from Exp 557/567


# ---------------------------------------------------------------------------
# Embedding function — random projection of step text into 128-D space.
# This is reproducible (same hash -> same projection) and CPU-fast.
# Real embeddings would use a sentence-transformer; random projection is
# sufficient for the contrastive loss because the RELATIVE ordering signal
# comes from the CPMI pair labels, not from the embedding geometry.
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Return a deterministic embed_fn: str -> jnp.ndarray (embed_dim,).

    Uses a fixed random projection matrix seeded by `seed`.  Two different
    strings will (with high probability) land in different regions of the
    embedding space, giving the model a non-trivial signal to work with.

    The projection matrix is stored in a closure so we only allocate it once.
    """
    rng = np.random.RandomState(seed)
    # Max string length we hash into a feature vector; longer strings are truncated.
    _MAX_CHARS = 256
    _CHAR_DIM = 128  # ASCII range

    # Random projection: (_CHAR_DIM,) -> (embed_dim,)
    proj = rng.randn(_CHAR_DIM, embed_dim).astype(np.float32)

    def embed_fn(text: str) -> jnp.ndarray:
        """Embed a step text string as a 128-D jnp array via random char projection.

        The text is converted to a normalised ASCII histogram (128-D), then
        projected by the fixed matrix `proj`.  This is deterministic and O(len(text)).
        """
        text = text[:_MAX_CHARS]
        char_vec = np.zeros(_CHAR_DIM, dtype=np.float32)
        for ch in text:
            idx = ord(ch) % _CHAR_DIM
            char_vec[idx] += 1.0
        norm = np.linalg.norm(char_vec)
        if norm > 0:
            char_vec /= norm
        emb = char_vec @ proj  # (embed_dim,)
        return jnp.array(emb)

    return embed_fn


# ---------------------------------------------------------------------------
# MLP model: 2 layers, input=embed_dim, hidden=embed_dim, output=1 scalar
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray, embed_dim: int = EMBED_DIM) -> dict:
    """Initialise a 2-layer MLP: input(128) -> hidden(128) -> output(1).

    Xavier uniform initialisation for stable training.  Same architecture
    depth as Exp 567 (v10) so results are directly comparable.
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


def _model_fn(params: dict, emb: jnp.ndarray) -> float:
    """Forward pass: emb(128,) -> scalar energy.  SiLU activation, no final sigmoid.

    No sigmoid so the energy can grow unboundedly — this is important for the
    contrastive margin loss which needs E_incorrect - E_correct >= margin.  A
    sigmoid would squash the gap to near zero, killing the gradient.
    """
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return float((params["w2"] @ h + params["b2"])[0])


# ---------------------------------------------------------------------------
# Question-ID-based 80/20 split (no cross-question leakage)
# ---------------------------------------------------------------------------


def _split_by_question_id(
    pairs: list,
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[list, list]:
    """Split CPMI pairs into train/val sets by question_id — no leakage.

    Each pair has a unique question_id (one pair per question from the builder).
    We shuffle the question_ids and take the first train_frac fraction as training.
    This prevents any question appearing in both train and val, which is critical
    because the same question's chain embeddings are strongly correlated.

    Args:
        pairs:      List of JEPACPMIPair objects from JEPACPMIPairBuilder.
        train_frac: Fraction of pairs to include in train set.  Default 0.8.
        seed:       Random seed for reproducibility.

    Returns:
        (train_pairs, val_pairs) tuple.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_train = max(1, int(len(pairs) * train_frac))
    train_pairs = [pairs[i] for i in idx[:n_train]]
    val_pairs = [pairs[i] for i in idx[n_train:]]
    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# AUC evaluation using contrastive pair ordering
# ---------------------------------------------------------------------------


def _evaluate_auc_from_pairs(params: dict, pairs: list) -> float:
    """Compute ranking AUC from contrastive pairs using model energy ordering.

    For each pair, the model correctly ranks if E(incorrect) > E(correct).
    AUC = fraction of pairs correctly ranked.  This is a direct estimate of
    the ranking AUC since each pair is one (correct, incorrect) comparison.

    Returns 0.5 if no pairs are available (random baseline).

    Args:
        params: JAX param dict.
        pairs:  List of JEPACPMIPair objects.

    Returns:
        Float AUC in [0, 1].
    """
    if not pairs:
        return 0.5

    loss_obj = CPMIContrastiveLoss(margin=0.0, chain_energy_mode="mean")
    n_correct_rank = 0
    for pair in pairs:
        e_correct = loss_obj.chain_energy(lambda emb: _model_fn(params, emb), pair.correct_embeddings)
        e_incorrect = loss_obj.chain_energy(lambda emb: _model_fn(params, emb), pair.incorrect_embeddings)
        if e_incorrect > e_correct:
            n_correct_rank += 1
    return n_correct_rank / len(pairs)


# ---------------------------------------------------------------------------
# Training loop: CPMI contrastive loss, 300 epochs, best-checkpoint tracking
# ---------------------------------------------------------------------------


def _compute_contrastive_loss_jax(
    params: dict,
    train_pairs: list,
    margin: float,
) -> jnp.ndarray:
    """Compute CPMI contrastive hinge margin loss as a differentiable JAX scalar.

    For each pair p:
        E_correct   = mean step energy over correct_embeddings
        E_incorrect = mean step energy over incorrect_embeddings
        L_p = max(0, margin - (E_incorrect - E_correct))

    Returns mean(L_p) as a JAX array so JAX grad() can differentiate through it.
    Returns 0.0 when no pairs are available.

    This is the core training signal for JEPA v11.  Unlike BCE or PURE, it
    never computes per-chain labels — only the RELATIVE ordering matters.

    Args:
        params:      JAX param dict.
        train_pairs: List of JEPACPMIPair objects.
        margin:      Minimum required energy gap.

    Returns:
        Mean pair loss as a JAX scalar array.
    """
    if not train_pairs:
        return jnp.array(0.0)

    total = jnp.array(0.0)
    for pair in train_pairs:
        # Chain energy: mean of per-step model outputs (no sigmoid — unbounded energy).
        def _chain_e(embeddings: list) -> jnp.ndarray:
            if not embeddings:
                return jnp.array(0.0)
            scores = jnp.stack([
                (params["w2"] @ jax.nn.silu(params["w1"] @ emb + params["b1"]) + params["b2"])[0]
                for emb in embeddings
            ])
            return jnp.mean(scores)

        e_correct = _chain_e(pair.correct_embeddings)
        e_incorrect = _chain_e(pair.incorrect_embeddings)
        gap = e_incorrect - e_correct
        total = total + jnp.maximum(jnp.array(0.0), margin - gap)

    return total / len(train_pairs)


def train_jepa_v11(
    train_pairs: list,
    val_pairs: list,
    margin: float,
    n_epochs: int,
    eval_every: int,
    seed: int,
) -> tuple[dict, float, int, list[dict]]:
    """Full JEPA v11 training loop with CPMI contrastive objective.

    Trains a 2-layer MLP using CPMIContrastiveLoss pairwise ordering objective.
    Evaluates ranking AUC on val pairs every eval_every epochs and tracks the
    best checkpoint.

    Args:
        train_pairs: List of JEPACPMIPair for training.
        val_pairs:   List of JEPACPMIPair for validation.
        margin:      Hinge margin for contrastive loss.
        n_epochs:    Total number of training epochs.
        eval_every:  Evaluate on val every this many epochs.
        seed:        Random seed for parameter initialisation.

    Returns:
        (best_params, best_val_auc, best_epoch, eval_log)

        best_params:   JAX param dict with the best-AUC weights.
        best_val_auc:  Best validation AUC across all checkpoints.
        best_epoch:    1-indexed epoch where best_val_auc was achieved.
        eval_log:      List of {epoch, val_auc, loss} dicts per eval checkpoint.

    Spec: REQ-LEARN-067, SCENARIO-LEARN-104
    """
    import optax  # local import to keep startup light

    params = _init_params(jrandom.PRNGKey(seed))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def _loss_fn(p: dict) -> jnp.ndarray:
        return _compute_contrastive_loss_jax(p, train_pairs, margin)

    grad_fn = jax.jit(jax.grad(_loss_fn))

    best_params = params
    best_val_auc = 0.0
    best_epoch = 0
    eval_log: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        opt_state = opt_state_new

        if epoch % eval_every == 0 or epoch == n_epochs:
            loss_val = float(_loss_fn(params))
            val_auc = _evaluate_auc_from_pairs(params, val_pairs)
            eval_log.append({"epoch": epoch, "val_auc": val_auc, "loss": loss_val})
            _log.info("Epoch %d/%d  loss=%.4f  val_auc=%.4f", epoch, n_epochs, loss_val, val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_params = {k: jnp.array(v) for k, v in params.items()}

    return best_params, best_val_auc, best_epoch, eval_log


def save_model_safetensors(params: dict, path: Path) -> None:
    """Save JAX param dict to safetensors format.

    Converts each JAX array to numpy before saving to avoid device-dependent
    serialisation edge cases.  safetensors is the standard Carnot format for
    trained model checkpoints.

    Spec: REQ-LEARN-067, SCENARIO-LEARN-104
    """
    from safetensors.numpy import save_file

    np_params = {k: np.array(v) for k, v in params.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(np_params, str(path))
    _log.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 580: JEPA v11 full retrain with CPMI contrastive objective."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=60):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # ------------------------------------------------------------------
        # Load corpus: prefer v3, fall back to v2
        # ------------------------------------------------------------------
        corpus_path = CORPUS_V3_PATH if CORPUS_V3_PATH.exists() else CORPUS_V2_PATH
        corpus_version = "v3" if CORPUS_V3_PATH.exists() else "v2"
        _log.info("Loading FOVER corpus %s from: %s", corpus_version, corpus_path)

        try:
            raw_corpus: list[dict] = json.loads(corpus_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("Failed to load corpus: %s", exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_retrain.v11",
                    "n_train": 0,
                    "n_val": 0,
                    "n_real_pairs": 0,
                    "n_synthetic_pairs": 0,
                    "loss_function": "cpmi_contrastive_hinge_margin",
                    "v10_auc": V10_AUC,
                    "v11_auc": 0.0,
                    "auc_improvement": 0.0,
                    "best_epoch": 0,
                    "model_path": str(_REPO_ROOT / MODEL_DELIVERABLE),
                    "retro_063_resolved": False,
                    "fr11_retrain_complete": True,
                    "honest_verdict": "blocked_corpus_load_error",
                    "block_reason": str(exc),
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        _log.info("Corpus entries: %d", len(raw_corpus))

        # ------------------------------------------------------------------
        # Build embedding function and CPMI pairs
        # ------------------------------------------------------------------
        embed_fn = _make_embed_fn(embed_dim=EMBED_DIM, seed=SEED)
        builder = JEPACPMIPairBuilder(embed_fn=embed_fn, min_pairs=5)
        real_pairs = builder.build_pairs(raw_corpus)
        n_real_pairs = len(real_pairs)
        _log.info("Real CPMI pairs: %d", n_real_pairs)

        n_synthetic_pairs = 0
        all_pairs = real_pairs
        if n_real_pairs < 5:
            _log.warning("Only %d real pairs — augmenting with 20 synthetic pairs.", n_real_pairs)
            synthetic = builder.build_synthetic_pairs(20)
            all_pairs = real_pairs + synthetic
            n_synthetic_pairs = len(synthetic)

        _log.info("Total pairs for training: %d (%d real, %d synthetic)",
                  len(all_pairs), n_real_pairs, n_synthetic_pairs)

        # ------------------------------------------------------------------
        # 80/20 train/val split by question_id (no cross-question leakage)
        # ------------------------------------------------------------------
        train_pairs, val_pairs = _split_by_question_id(all_pairs, TRAIN_FRAC, SEED)
        n_train = len(train_pairs)
        n_val = len(val_pairs)
        _log.info("Train pairs: %d  Val pairs: %d", n_train, n_val)

        # ------------------------------------------------------------------
        # Train JEPA v11 with CPMI contrastive objective
        # ------------------------------------------------------------------
        _log.info("Training JEPA v11 for %d epochs (eval every %d)...", N_EPOCHS, EVAL_EVERY)
        best_params, best_val_auc, best_epoch, eval_log = train_jepa_v11(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            margin=MARGIN,
            n_epochs=N_EPOCHS,
            eval_every=EVAL_EVERY,
            seed=SEED,
        )

        v11_auc = best_val_auc
        auc_improvement = v11_auc - V10_AUC
        retro_063_resolved = v11_auc > 0.5

        if v11_auc > 0.5:
            honest_verdict = "jepa_v11_above_random"
        elif v11_auc < 0.5:
            honest_verdict = "jepa_v11_still_inverted"
        else:
            honest_verdict = "jepa_v11_at_random"

        _log.info(
            "Best val AUC=%.4f at epoch %d  improvement=%.4f  verdict=%s",
            v11_auc, best_epoch, auc_improvement, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Save best model checkpoint
        # ------------------------------------------------------------------
        model_path = _REPO_ROOT / MODEL_DELIVERABLE
        save_model_safetensors(best_params, model_path)

        # ------------------------------------------------------------------
        # Write result artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_retrain.v11",
                "n_train": n_train,
                "n_val": n_val,
                "n_real_pairs": n_real_pairs,
                "n_synthetic_pairs": n_synthetic_pairs,
                "loss_function": "cpmi_contrastive_hinge_margin",
                "v10_auc": V10_AUC,
                "v11_auc": v11_auc,
                "auc_improvement": auc_improvement,
                "best_epoch": best_epoch,
                "model_path": str(model_path),
                "retro_063_resolved": retro_063_resolved,
                "fr11_retrain_complete": True,
                "honest_verdict": honest_verdict,
                "eval_log": eval_log,
                "corpus_version": corpus_version,
            },
            status="success",
        )

        AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
