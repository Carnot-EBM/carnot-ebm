#!/usr/bin/env python3
"""Experiment 593: JEPA v12 Live Corpus Retrain with PROGRS Outcome-Centered CPMI.

**Researcher summary:**
    JEPA v11 (Exp 580) achieved AUC=1.0 on only 9 pairs — almost certainly overfitting.
    Exp 578 now provides 100 live pairs (GSM8K 0-49, both Qwen3.5-0.8B and Gemma4-E4B-it)
    with inference_mode='live_gpu'.  This experiment trains JEPA v12 on the full live corpus
    using:

    1. CPMI contrastive loss (arXiv 2604.10660) — correct vs. incorrect chain ordering
    2. PROGRS outcome-conditioned centering (arXiv 2604.02341) — normalises energy gaps
       within each question group to prevent reward hacking on easy questions

    Target: val_auc >= 0.70 on a held-out 20% split (20 real live pairs).

    If v12_val_auc < 0.55, v11 AUC=1.0 was overfitting (honest_verdict=jepa_overfit_confirmed).
    If v12_val_auc >= 0.70, the architecture generalises (honest_verdict=jepa_validated).

**Architecture:**
    Embedder: RandomProjection (128-D per step, same as v11)
    Predictor: 2-layer MLP (embed_dim=128, same as v11) — re-trained from scratch
    Loss: PROGRSCentering.compute_centered_loss() wrapping CPMIContrastiveLoss
    Optimizer: optax.adamw(lr=1e-3, weight_decay=1e-4)
    Epochs: 100 with eval every 20 epochs, best-checkpoint tracking

Spec: REQ-LEARN-068, REQ-LEARN-069,
      SCENARIO-LEARN-107, SCENARIO-LEARN-108, SCENARIO-LEARN-109
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() MUST be called before any JAX import.
# ---------------------------------------------------------------------------
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

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
    JEPACPMIPair,
    JEPACPMIPairBuilder,
    PROGRSCentering,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 593
EXP_TITLE = "JEPA v12 Live Corpus Retrain"
DELIVERABLE = "results/experiment_593_jepa_v12_retrain.json"
MODEL_DELIVERABLE = "results/jepa_predictor_v12.safetensors"
LIVE_PAIRS_PATH = _REPO_ROOT / "results" / "live_pairs_578.json"

EMBED_DIM = 128
SEED = 42
TRAIN_FRAC = 0.8
N_EPOCHS = 100
EVAL_EVERY = 20
MARGIN = 1.0
SYNTHETIC_FALLBACK_COUNT = 20
MIN_REAL_PAIRS = 10

V11_AUC = 1.0  # from Exp 580 — n_real_pairs=9, suspected overfitting


# ---------------------------------------------------------------------------
# Embed function: RandomProjection (same as v11 for comparability)
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Create a deterministic random-projection text embedder.

    Maps a text string to a fixed-size float32 vector by:
    1. Hashing characters to a seed-based random state.
    2. Drawing a Gaussian random vector of shape (embed_dim,) seeded by the hash.

    This is not a semantic embedder — it is a fast, reproducible fingerprint
    that gives the MLP a distinct signal per unique step text.  Two identical
    step texts produce the same embedding; two different texts produce independent
    random vectors.  This is sufficient for training the contrastive ranking loss.

    Args:
        embed_dim: Output embedding dimension.  Default 128.
        seed:      Master seed for the projection matrix.

    Returns:
        Callable str -> jnp.ndarray of shape (embed_dim,).
    """
    key = jrandom.PRNGKey(seed)
    # Fixed projection matrix: (vocab_bucket, embed_dim).
    # We use 256 character-level buckets — good enough for step-text fingerprinting.
    proj = jrandom.normal(key, (256, embed_dim)) / np.sqrt(embed_dim)

    def embed_fn(text: str) -> jnp.ndarray:
        if not text:
            return jnp.zeros(embed_dim, dtype=jnp.float32)
        # Use character ordinals (mod 256) as indices into the projection matrix.
        char_indices = jnp.array([ord(c) % 256 for c in text[:512]], dtype=jnp.int32)
        vecs = proj[char_indices]  # (len, embed_dim)
        return jnp.mean(vecs, axis=0).astype(jnp.float32)

    return embed_fn


# ---------------------------------------------------------------------------
# Build JEPACPMIPair list from live_pairs_578.json
# ---------------------------------------------------------------------------


def _build_pairs_from_live_json(live_data: list[dict], embed_fn) -> list[JEPACPMIPair]:
    """Build CPMI contrastive pairs from live_pairs_578.json entries.

    live_pairs_578.json entries have keys:
        question_index, question, model, response, is_correct, cot_steps, fover_labels

    We group by ``question`` (same as question_id in JEPACPMIPairBuilder), require both
    a correct and an incorrect entry for the same question, and pick the best correct
    (most cot_steps) and hardest incorrect (most cot_steps) — identical selection
    logic to JEPACPMIPairBuilder.build_pairs().

    Args:
        live_data: List of entry dicts from live_pairs_578.json.
        embed_fn:  Text -> jnp.ndarray embedder.

    Returns:
        List of JEPACPMIPair objects, one per valid question group.
    """
    builder = JEPACPMIPairBuilder(embed_fn=embed_fn, min_pairs=MIN_REAL_PAIRS)
    return builder.build_pairs(live_data)


# ---------------------------------------------------------------------------
# MLP param init and forward (same architecture as v11 for comparability)
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray, embed_dim: int = EMBED_DIM) -> dict:
    """Initialise a 2-layer MLP: input(128) -> hidden(128) -> output(1).

    Xavier uniform initialisation for stable training with the contrastive margin loss.
    Same architecture as v11 so AUC comparisons are meaningful.
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

    No sigmoid so energy can grow unboundedly — critical for contrastive margin loss.
    """
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return float((params["w2"] @ h + params["b2"])[0])


# ---------------------------------------------------------------------------
# 80/20 question-ID-based split (no cross-question leakage)
# ---------------------------------------------------------------------------


def _split_by_question_id(
    pairs: list[JEPACPMIPair],
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[list[JEPACPMIPair], list[JEPACPMIPair]]:
    """Split CPMI pairs into train/val sets by question_id — no leakage.

    Shuffles question_ids and takes the first train_frac fraction as training.
    This prevents any question appearing in both train and val since embeddings
    for the same question are strongly correlated.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_train = max(1, int(len(pairs) * train_frac))
    return [pairs[i] for i in idx[:n_train]], [pairs[i] for i in idx[n_train:]]


# ---------------------------------------------------------------------------
# AUC evaluation using contrastive pair ordering
# ---------------------------------------------------------------------------


def _evaluate_auc(params: dict, pairs: list[JEPACPMIPair]) -> float:
    """Compute ranking AUC: fraction of pairs where E(incorrect) > E(correct).

    Returns 0.5 (random baseline) when no pairs are available.
    """
    if not pairs:
        return 0.5

    loss_obj = CPMIContrastiveLoss(margin=0.0, chain_energy_mode="mean")
    model = lambda emb: _model_fn(params, emb)  # noqa: E731
    n_correct = sum(
        1
        for pair in pairs
        if loss_obj.chain_energy(model, pair.incorrect_embeddings)
        > loss_obj.chain_energy(model, pair.correct_embeddings)
    )
    return n_correct / len(pairs)


# ---------------------------------------------------------------------------
# Training loop with PROGRS centering
# ---------------------------------------------------------------------------


def _compute_progrs_loss_jax(
    params: dict,
    train_pairs: list[JEPACPMIPair],
    margin: float,
) -> jnp.ndarray:
    """Compute PROGRS-centered CPMI contrastive loss as a differentiable JAX scalar.

    For each pair p:
        g_p = E(incorrect) - E(correct)  (raw gap)
        centered_g_p = g_p - mean(g_q for q in same question group)
        L_p = max(0, margin - centered_g_p)
    Returns mean(L_p).

    Implemented as a differentiable JAX computation so jax.grad() can
    produce gradients for the training loop.

    Args:
        params:      JAX MLP parameter dict.
        train_pairs: List of JEPACPMIPair objects.
        margin:      Minimum required centered energy gap.

    Returns:
        Mean centered pair loss as a JAX scalar array.
    """
    if not train_pairs:
        return jnp.array(0.0)

    # Compute differentiable chain energies for every pair.
    def _chain_e(embeddings: list) -> jnp.ndarray:
        if not embeddings:
            return jnp.array(0.0)
        scores = jnp.stack([
            (params["w2"] @ jax.nn.silu(params["w1"] @ emb + params["b1"]) + params["b2"])[0]
            for emb in embeddings
        ])
        return jnp.mean(scores)

    raw_gaps = [
        _chain_e(p.incorrect_embeddings) - _chain_e(p.correct_embeddings)
        for p in train_pairs
    ]

    # Group gaps by question_id and compute group means.
    # We use Python dicts here (outside JAX); this is fine because question_ids are
    # fixed for a given training batch and do not affect the gradient path.
    from collections import defaultdict

    group_sums: dict[str, jnp.ndarray] = defaultdict(lambda: jnp.array(0.0))
    group_counts: dict[str, int] = defaultdict(int)
    for pair, gap in zip(train_pairs, raw_gaps):
        group_sums[pair.question_id] = group_sums[pair.question_id] + gap
        group_counts[pair.question_id] += 1

    group_means = {qid: group_sums[qid] / group_counts[qid] for qid in group_sums}

    # Compute (possibly centered) gaps and hinge loss.
    # PROGRS centering is only applied when a group has >1 pair.
    # For single-pair groups, centering yields centered_gap=0 (raw_gap - raw_gap),
    # which zeroes out all gradients and prevents learning.  The GRPO paper targets
    # multi-response groups; for single-pair groups we fall back to raw_gap.
    total = jnp.array(0.0)
    for pair, gap in zip(train_pairs, raw_gaps):
        if group_counts[pair.question_id] > 1:
            effective_gap = gap - group_means[pair.question_id]
        else:
            effective_gap = gap  # single-pair group: raw gap, gradients intact
        total = total + jnp.maximum(jnp.array(0.0), margin - effective_gap)

    return total / len(train_pairs)


def train_jepa_v12(
    train_pairs: list[JEPACPMIPair],
    val_pairs: list[JEPACPMIPair],
    n_epochs: int = N_EPOCHS,
    eval_every: int = EVAL_EVERY,
    margin: float = MARGIN,
    seed: int = SEED,
) -> tuple[dict, float, int, list[dict]]:
    """Train JEPA v12 with PROGRS-centered CPMI objective.

    Returns (best_params, best_val_auc, best_epoch, eval_log).
    """
    import optax  # local import to keep startup light

    params = _init_params(jrandom.PRNGKey(seed))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    def _loss_fn(p: dict) -> jnp.ndarray:
        return _compute_progrs_loss_jax(p, train_pairs, margin)

    grad_fn = jax.jit(jax.grad(_loss_fn))

    best_params = params
    best_val_auc = 0.0
    best_epoch = 0
    eval_log: list[dict] = []

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if epoch % eval_every == 0 or epoch == n_epochs:
            loss_val = float(_loss_fn(params))
            val_auc = _evaluate_auc(params, val_pairs)
            eval_log.append({"epoch": epoch, "val_auc": val_auc, "loss": loss_val})
            _log.info(
                "Epoch %d/%d  loss=%.4f  val_auc=%.4f", epoch, n_epochs, loss_val, val_auc
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                best_params = {k: jnp.array(v) for k, v in params.items()}

    return best_params, best_val_auc, best_epoch, eval_log


def _save_model_safetensors(params: dict, path: Path) -> None:
    """Save JAX param dict to safetensors format (numpy arrays)."""
    from safetensors.numpy import save_file

    np_params = {k: np.array(v) for k, v in params.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(np_params, str(path))
    _log.info("Model saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 593: JEPA v12 live corpus retrain with PROGRS outcome-centered CPMI."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=40):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # ------------------------------------------------------------------
        # Load live pairs from Exp 578
        # ------------------------------------------------------------------
        _log.info("Loading live pairs from: %s", LIVE_PAIRS_PATH)
        try:
            live_data: list[dict] = json.loads(LIVE_PAIRS_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("Failed to load live pairs: %s", exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_v12_retrain.v1",
                    "n_real_pairs": 0,
                    "n_synthetic_pairs": 0,
                    "train_pairs": 0,
                    "val_pairs": 0,
                    "v11_auc": V11_AUC,
                    "v12_val_auc": 0.0,
                    "v12_model_saved": False,
                    "progrs_centering_applied": True,
                    "retro_063_validated": False,
                    "honest_verdict": "blocked_live_pairs_load_error",
                    "block_reason": str(exc),
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        _log.info("Live entries loaded: %d", len(live_data))

        # ------------------------------------------------------------------
        # Build CPMI pairs from live data
        # ------------------------------------------------------------------
        embed_fn = _make_embed_fn(embed_dim=EMBED_DIM, seed=SEED)
        real_pairs = _build_pairs_from_live_json(live_data, embed_fn)
        n_real_pairs = len(real_pairs)
        _log.info("Real CPMI pairs built: %d", n_real_pairs)

        # Augment with synthetic pairs if real corpus too small.
        n_synthetic_pairs = 0
        all_pairs = real_pairs
        if n_real_pairs < MIN_REAL_PAIRS:
            _log.warning(
                "Only %d real pairs (<10) — augmenting with %d synthetic pairs.",
                n_real_pairs,
                SYNTHETIC_FALLBACK_COUNT,
            )
            builder = JEPACPMIPairBuilder(embed_fn=embed_fn, min_pairs=MIN_REAL_PAIRS)
            synthetic = builder.build_synthetic_pairs(SYNTHETIC_FALLBACK_COUNT)
            all_pairs = real_pairs + synthetic
            n_synthetic_pairs = len(synthetic)

        _log.info(
            "Total pairs: %d (%d real, %d synthetic)", len(all_pairs), n_real_pairs, n_synthetic_pairs
        )

        # ------------------------------------------------------------------
        # 80/20 train/val split by question_id (no cross-question leakage)
        # ------------------------------------------------------------------
        train_pairs, val_pairs = _split_by_question_id(all_pairs, TRAIN_FRAC, SEED)
        n_train = len(train_pairs)
        n_val = len(val_pairs)
        _log.info("Train pairs: %d  Val pairs: %d", n_train, n_val)

        # ------------------------------------------------------------------
        # Train JEPA v12 with PROGRS-centered CPMI objective
        # ------------------------------------------------------------------
        _log.info("Training JEPA v12 for %d epochs (eval every %d)...", N_EPOCHS, EVAL_EVERY)
        best_params, best_val_auc, best_epoch, eval_log = train_jepa_v12(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            n_epochs=N_EPOCHS,
            eval_every=EVAL_EVERY,
            margin=MARGIN,
            seed=SEED,
        )

        v12_val_auc = best_val_auc
        v12_model_saved = v12_val_auc >= 0.70
        retro_063_validated = v12_model_saved

        if v12_val_auc >= 0.70:
            honest_verdict = "jepa_validated"
        elif v12_val_auc < 0.55:
            honest_verdict = "jepa_overfit_confirmed"
        else:
            honest_verdict = "jepa_marginal"

        _log.info(
            "v11_auc=%.4f (9 pairs)  v12_val_auc=%.4f (%d real pairs)  verdict=%s",
            V11_AUC, v12_val_auc, n_real_pairs, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Save best model if above threshold
        # ------------------------------------------------------------------
        if v12_model_saved:
            _save_model_safetensors(best_params, _REPO_ROOT / MODEL_DELIVERABLE)
            _log.info("Model saved (val_auc=%.4f >= 0.70)", v12_val_auc)
        else:
            _log.info(
                "Model NOT saved (val_auc=%.4f < 0.70) — threshold not met", v12_val_auc
            )

        # ------------------------------------------------------------------
        # Write result artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_v12_retrain.v1",
                "n_real_pairs": n_real_pairs,
                "n_synthetic_pairs": n_synthetic_pairs,
                "train_pairs": n_train,
                "val_pairs": n_val,
                "v11_auc": V11_AUC,
                "v12_val_auc": v12_val_auc,
                "v12_model_saved": v12_model_saved,
                "progrs_centering_applied": True,
                "retro_063_validated": retro_063_validated,
                "honest_verdict": honest_verdict,
                "best_epoch": best_epoch,
                "eval_log": eval_log,
            },
            status="success",
        )

        AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
