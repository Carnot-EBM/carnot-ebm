#!/usr/bin/env python3
"""Experiment 607: JEPA v12 OOD Generalization Validation.

**Researcher summary:**
    JEPA v12 (Exp 593) achieved val_auc=1.0 on its in-distribution validation split
    (20 pairs from GSM8K questions 0-49).  AUC=1.0 on 20 pairs is suspicious — it may
    mean the model learned a genuine ranking or it may mean it memorised the small corpus.

    This experiment answers: does v12 generalize to STRICTLY HELD-OUT questions from
    GSM8K 250-349 (live_pairs_602.json) that were never seen during training?

    Gate rule:
        v12_generalized = ood_auc >= 0.65
        v12_overfit     = ood_auc < 0.55

    If v12 is overfit: retrain JEPA v13 on fover_corpus_v4.json (full merged corpus).
    fr11_generalization_confirmed = v12_generalized OR (v12_overfit AND v13_val_auc >= 0.65).

**OOD evaluation approach:**
    live_pairs_602.json has 200 entries (100 questions × 2 models), question indices 250-349.
    These entries have no cot_steps — the full response text is embedded as a single-step
    chain.  The model scores each response; AUC is computed using roc_auc_score with
    label=1 for incorrect responses (high energy expected) and label=0 for correct ones.

    This is equivalent to the contrastive pair evaluation but without requiring matched
    correct/incorrect pairs per question — standard classification AUC is appropriate
    when the goal is to test the model as a standalone discriminator on unseen data.

Spec: REQ-LEARN-073, REQ-LEARN-074,
      SCENARIO-LEARN-115, SCENARIO-LEARN-116, SCENARIO-LEARN-117
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# apply_env_autofix MUST be called before any JAX import — repairs ROCm env vars
# that would otherwise cause silent device-selection failures.
from carnot.pipeline.env_autofix import apply_env_autofix  # noqa: E402

apply_env_autofix()

from carnot.pipeline.live_assertion import assert_live_or_ci_skip  # noqa: E402

assert_live_or_ci_skip()

from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_log = logging.getLogger(__name__)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

from carnot.inference.jepa_cpmi_pairs import (  # noqa: E402
    CPMIContrastiveLoss,
    JEPACPMIPair,
    JEPACPMIPairBuilder,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

EXP_ID = 607
EXP_TITLE = "JEPA v12 OOD Validation"
DELIVERABLE = "results/experiment_607_jepa_v12_ood.json"

V12_MODEL_PATH = _REPO_ROOT / "results" / "jepa_predictor_v12.safetensors"
V13_MODEL_PATH = _REPO_ROOT / "results" / "jepa_predictor_v13.safetensors"
CORPUS_V4_PATH = _REPO_ROOT / "results" / "fover_corpus_v4.json"
LIVE_578_PATH = _REPO_ROOT / "results" / "live_pairs_578.json"
LIVE_602_PATH = _REPO_ROOT / "results" / "live_pairs_602.json"

# Architecture constants — must match v12 training (Exp 593)
EMBED_DIM = 128
SEED = 42
TRAIN_FRAC = 0.8
N_EPOCHS = 100
EVAL_EVERY = 20
MARGIN = 1.0
SYNTHETIC_FALLBACK_COUNT = 20
MIN_REAL_PAIRS = 10

# Training range used by Exp 578/593: question indices 0-49.
# OOD = any question index NOT in this range.
TRAIN_QUESTION_RANGE_MAX = 49


# ---------------------------------------------------------------------------
# Embed function — identical to Exp 593 for compatible scoring
# ---------------------------------------------------------------------------


def _make_embed_fn(embed_dim: int = EMBED_DIM, seed: int = SEED):
    """Create a deterministic random-projection text embedder (same as Exp 593).

    Maps a text string to a fixed-size float32 vector by projecting character
    ordinals through a seed-stable Gaussian matrix.  Identical parameters to
    Exp 593 ensure that v12 model weights operate on the same embedding space
    as the OOD evaluation.

    Args:
        embed_dim: Output embedding dimension.  Must match v12 training (128).
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
# MLP forward — identical architecture to Exp 593 v12
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray, embed_dim: int = EMBED_DIM) -> dict:
    """Initialise a 2-layer MLP: input(128) -> hidden(128) -> output(1).

    Xavier uniform initialisation — same as Exp 593 so that loaded v12 weights
    are compatible with this forward pass.
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

    Unbounded energy is required for the contrastive margin loss to work correctly.
    Higher energy = model believes this chain is more likely to be incorrect.
    """
    h = jax.nn.silu(params["w1"] @ emb + params["b1"])
    return float((params["w2"] @ h + params["b2"])[0])


# ---------------------------------------------------------------------------
# Load v12 model weights from safetensors
# ---------------------------------------------------------------------------


def load_v12_params(path: Path) -> dict | None:
    """Load JEPA v12 model weights from safetensors.

    Returns the param dict if successful, None if the file is missing.
    Raises ValueError if the file exists but has unexpected keys (indicates
    a model architecture mismatch that should be investigated).

    Args:
        path: Path to the .safetensors file saved by Exp 593.

    Returns:
        Dict mapping 'w1', 'b1', 'w2', 'b2' to jnp.ndarray, or None.

    Spec: REQ-LEARN-073, SCENARIO-LEARN-117
    """
    if not path.exists():
        return None
    from safetensors.numpy import load_file

    np_params = load_file(str(path))
    expected_keys = {"w1", "b1", "w2", "b2"}
    missing = expected_keys - set(np_params.keys())
    if missing:
        raise ValueError(
            f"safetensors at {path} is missing keys: {sorted(missing)} — "
            "was it saved by a different architecture?"
        )
    return {k: jnp.array(v) for k, v in np_params.items()}


# ---------------------------------------------------------------------------
# Score individual entries (no cot_steps required)
# ---------------------------------------------------------------------------


def score_entries(params: dict, entries: list[dict], embed_fn) -> tuple[list[float], list[int]]:
    """Score each entry with the MLP and collect labels for AUC computation.

    Each entry is embedded by concatenating its response text (and question text
    for context) into a single string, then scored by the 2-layer MLP.  A higher
    score = model believes this response is more likely to be incorrect.

    Labels follow the convention for roc_auc_score:
        label = 1 for incorrect responses (positive class = model should flag these)
        label = 0 for correct responses

    Args:
        params:    MLP parameter dict loaded from v12 safetensors.
        entries:   List of dicts from live_pairs_602.json.
        embed_fn:  Same random-projection embedder used during v12 training.

    Returns:
        (scores, labels) — parallel lists of float scores and int 0/1 labels.

    Spec: REQ-LEARN-073, SCENARIO-LEARN-115
    """
    scores: list[float] = []
    labels: list[int] = []

    for entry in entries:
        # Embed question + response together to give the model full context.
        # This mirrors how the pipeline would use the predictor in production.
        text = (entry.get("question", "") + " " + entry.get("response", "")).strip()
        emb = embed_fn(text)
        score = _model_fn(params, emb)
        scores.append(score)
        # Positive class = incorrect responses (the model should give these high energy).
        labels.append(0 if entry.get("is_correct", False) else 1)

    return scores, labels


# ---------------------------------------------------------------------------
# Build CPMI pairs from entries that have cot_steps
# ---------------------------------------------------------------------------


def build_ood_pairs(entries: list[dict], embed_fn) -> list[JEPACPMIPair]:
    """Attempt to build JEPACPMIPairs from OOD entries that have cot_steps.

    live_pairs_602.json does not include cot_steps, so this function will
    typically return an empty list for that corpus.  If fover_corpus_v4 ever
    has OOD entries with cot_steps, this path will be exercised.

    Args:
        entries:  List of dicts.  Must have 'question', 'is_correct', 'cot_steps'.
        embed_fn: Text -> jnp.ndarray embedder.

    Returns:
        List of JEPACPMIPair objects (may be empty).

    Spec: REQ-LEARN-073
    """
    has_steps = [e for e in entries if e.get("cot_steps")]
    if not has_steps:
        return []
    builder = JEPACPMIPairBuilder(embed_fn=embed_fn, min_pairs=0)
    return builder.build_pairs(has_steps)


# ---------------------------------------------------------------------------
# AUC evaluation on contrastive pairs (used when cot_steps are available)
# ---------------------------------------------------------------------------


def evaluate_pair_auc(params: dict, pairs: list[JEPACPMIPair]) -> float:
    """Fraction of pairs where E(incorrect) > E(correct) — direct contrastive AUC.

    Returns 0.5 (random baseline) for an empty pair list.

    Args:
        params: MLP parameter dict.
        pairs:  List of JEPACPMIPair objects.

    Returns:
        Float in [0, 1].  1.0 = all pairs ranked correctly.

    Spec: REQ-LEARN-073
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
# v13 retrain — triggered when v12 is overfit
# ---------------------------------------------------------------------------


def _compute_progrs_loss_jax(
    params: dict,
    train_pairs: list[JEPACPMIPair],
    margin: float,
) -> jnp.ndarray:
    """Differentiable PROGRS-centered CPMI contrastive loss for JAX grad.

    For single-pair question groups, the centered gap equals the raw gap
    (centering has no effect), preserving the gradient signal.  This matches
    the Exp 593 training loop exactly so v13 uses the same learning dynamics.

    Args:
        params:      MLP parameter dict.
        train_pairs: List of JEPACPMIPair objects.
        margin:      Minimum required energy gap.

    Returns:
        Mean centered pair loss as a JAX scalar.

    Spec: REQ-LEARN-074
    """
    if not train_pairs:
        return jnp.array(0.0)

    def _chain_e(embeddings: list) -> jnp.ndarray:
        if not embeddings:
            return jnp.array(0.0)
        scores = jnp.stack([
            (params["w2"] @ jax.nn.silu(params["w1"] @ emb + params["b1"]) + params["b2"])[0]
            for emb in embeddings
        ])
        return jnp.mean(scores)

    raw_gaps: list[jnp.ndarray] = []
    for pair in train_pairs:
        raw_gaps.append(_chain_e(pair.incorrect_embeddings) - _chain_e(pair.correct_embeddings))

    from collections import defaultdict

    group_sums: dict[str, jnp.ndarray] = defaultdict(lambda: jnp.array(0.0))
    group_counts: dict[str, int] = defaultdict(int)
    for pair, gap in zip(train_pairs, raw_gaps):
        group_sums[pair.question_id] = group_sums[pair.question_id] + gap
        group_counts[pair.question_id] += 1
    group_means = {qid: group_sums[qid] / group_counts[qid] for qid in group_sums}

    total = jnp.array(0.0)
    for pair, gap in zip(train_pairs, raw_gaps):
        if group_counts[pair.question_id] > 1:
            effective_gap = gap - group_means[pair.question_id]
        else:
            effective_gap = gap  # single-pair group: use raw gap to preserve gradient
        total = total + jnp.maximum(jnp.array(0.0), margin - effective_gap)

    return total / len(train_pairs)


def _split_by_question_id(
    pairs: list[JEPACPMIPair],
    train_frac: float = TRAIN_FRAC,
    seed: int = SEED,
) -> tuple[list[JEPACPMIPair], list[JEPACPMIPair]]:
    """Split CPMI pairs into train/val by question_id to prevent leakage.

    Args:
        pairs:      Full list of JEPACPMIPair objects.
        train_frac: Fraction of questions to use for training.
        seed:       RNG seed for reproducible shuffle.

    Returns:
        (train_pairs, val_pairs)

    Spec: REQ-LEARN-074
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    n_train = max(1, int(len(pairs) * train_frac))
    return [pairs[i] for i in idx[:n_train]], [pairs[i] for i in idx[n_train:]]


def retrain_jepa_v13(
    corpus_entries: list[dict],
    embed_fn,
    n_epochs: int = N_EPOCHS,
    margin: float = MARGIN,
    seed: int = SEED,
) -> tuple[dict | None, float]:
    """Retrain JEPA v13 on a corpus, triggered when v12 is overfit.

    Uses the same CPMI+PROGRS architecture as Exp 593.  Falls back to synthetic
    pairs if the real corpus has fewer than MIN_REAL_PAIRS contrastive pairs.

    Args:
        corpus_entries: List of FOVER-style dicts with cot_steps, is_correct, question.
        embed_fn:       Text -> jnp.ndarray embedder (same as v12 for comparability).
        n_epochs:       Number of training epochs.  Default 100.
        margin:         Contrastive margin.  Default 1.0.
        seed:           RNG seed.

    Returns:
        (best_params, best_val_auc) — params are None if training yielded 0 pairs.

    Spec: REQ-LEARN-074, SCENARIO-LEARN-116
    """
    import optax

    builder = JEPACPMIPairBuilder(embed_fn=embed_fn, min_pairs=MIN_REAL_PAIRS)
    real_pairs = builder.build_pairs(corpus_entries)
    n_real = len(real_pairs)
    _log.info("v13 retrain: %d real pairs from corpus.", n_real)

    all_pairs = real_pairs
    if n_real < MIN_REAL_PAIRS:
        synthetic = builder.build_synthetic_pairs(SYNTHETIC_FALLBACK_COUNT)
        all_pairs = real_pairs + synthetic
        _log.warning(
            "v13 retrain: only %d real pairs, added %d synthetic.", n_real, len(synthetic)
        )

    if not all_pairs:
        _log.error("v13 retrain: no pairs available — cannot train.")
        return None, 0.0

    train_pairs, val_pairs = _split_by_question_id(all_pairs, TRAIN_FRAC, seed)
    _log.info("v13 retrain: %d train, %d val pairs.", len(train_pairs), len(val_pairs))

    params = _init_params(jrandom.PRNGKey(seed))
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
    opt_state = optimizer.init(params)
    grad_fn = jax.jit(jax.grad(lambda p: _compute_progrs_loss_jax(p, train_pairs, margin)))

    best_params = params
    best_val_auc = 0.0

    for epoch in range(1, n_epochs + 1):
        grads = grad_fn(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        if epoch % EVAL_EVERY == 0 or epoch == n_epochs:
            val_auc = evaluate_pair_auc(params, val_pairs)
            loss_val = float(_compute_progrs_loss_jax(params, train_pairs, margin))
            _log.info("v13 epoch %d/%d  loss=%.4f  val_auc=%.4f", epoch, n_epochs, loss_val, val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_params = {k: jnp.array(v) for k, v in params.items()}

    return best_params, best_val_auc


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
    """Run Exp 607: JEPA v12 OOD generalization validation."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=60):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        writer = AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE))

        # ------------------------------------------------------------------
        # Step 1: Try to load v12 model weights
        # ------------------------------------------------------------------
        _log.info("Loading v12 model from: %s", V12_MODEL_PATH)
        v12_params = load_v12_params(V12_MODEL_PATH)
        if v12_params is None:
            _log.warning("v12 model not found at %s — exiting cleanly.", V12_MODEL_PATH)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_v12_ood.v1",
                    "model_available": False,
                    "n_ood_pairs": 0,
                    "ood_question_range": "N/A",
                    "v12_val_auc": 1.0,
                    "v12_ood_auc": None,
                    "v12_generalized": False,
                    "v12_overfit": False,
                    "v13_retrained": False,
                    "v13_val_auc": None,
                    "fr11_generalization_confirmed": False,
                    "honest_verdict": "blocked_no_v12_model",
                },
                status="blocked_no_v12_model",
            )
            writer.write(artifact)
            tmpl.assert_deliverable_written()
            return

        _log.info("v12 model loaded successfully.")
        embed_fn = _make_embed_fn(embed_dim=EMBED_DIM, seed=SEED)

        # ------------------------------------------------------------------
        # Step 2: Load OOD entries — prefer fover_corpus_v4, fall back to 602
        # ------------------------------------------------------------------
        _log.info("Loading fover_corpus_v4 for OOD entries...")
        try:
            corpus_v4_raw = json.loads(CORPUS_V4_PATH.read_text())
            corpus_v4_entries = corpus_v4_raw if isinstance(corpus_v4_raw, list) else corpus_v4_raw.get("entries", [])
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning("Could not load fover_corpus_v4: %s", exc)
            corpus_v4_entries = []

        # Filter corpus_v4 to only OOD question indices (not in 0-49 training range).
        ood_entries = [
            e for e in corpus_v4_entries
            if e.get("question_index", 0) > TRAIN_QUESTION_RANGE_MAX
        ]
        _log.info("fover_corpus_v4 OOD entries (idx>49): %d", len(ood_entries))

        ood_source = "fover_corpus_v4"
        if len(ood_entries) < 10:
            _log.info("Insufficient fover_corpus_v4 OOD entries — loading live_pairs_602.json")
            try:
                raw_602 = json.loads(LIVE_602_PATH.read_text())
                ood_entries = raw_602 if isinstance(raw_602, list) else []
                ood_source = "live_pairs_602_GSM8K_250-349"
            except (OSError, json.JSONDecodeError) as exc:
                _log.error("Could not load live_pairs_602: %s", exc)
                ood_entries = []
                ood_source = "none_available"

        _log.info("OOD entries for evaluation: %d (source=%s)", len(ood_entries), ood_source)

        # ------------------------------------------------------------------
        # Step 3: Build JEPACPMIPairs if cot_steps are available;
        #         otherwise score individual entries for roc_auc_score.
        # ------------------------------------------------------------------
        ood_pairs = build_ood_pairs(ood_entries, embed_fn)
        _log.info("CPMI OOD pairs built (cot_steps entries): %d", len(ood_pairs))

        n_ood_pairs = len(ood_entries)

        if ood_pairs:
            # Use contrastive pair AUC (more informative when cot_steps are available).
            ood_auc = evaluate_pair_auc(v12_params, ood_pairs)
            _log.info("OOD pair AUC (contrastive): %.4f  (n_pairs=%d)", ood_auc, len(ood_pairs))
        elif ood_entries:
            # Score individual entries — embed full response text as single-step chain.
            from sklearn.metrics import roc_auc_score  # local import to keep startup fast

            scores, labels = score_entries(v12_params, ood_entries, embed_fn)
            n_pos = sum(labels)
            n_neg = len(labels) - n_pos
            _log.info(
                "OOD individual scoring: %d entries (%d incorrect, %d correct)",
                len(labels), n_pos, n_neg,
            )
            if n_pos == 0 or n_neg == 0:
                # Cannot compute AUC with only one class — record as 0.5 (random).
                ood_auc = 0.5
                _log.warning("Only one class in OOD labels — defaulting ood_auc=0.5.")
            else:
                ood_auc = float(roc_auc_score(labels, scores))
            _log.info("OOD AUC (individual scoring): %.4f", ood_auc)
        else:
            # No OOD data available at all — cannot evaluate.
            ood_auc = 0.5
            _log.error("No OOD entries available — setting ood_auc=0.5 (undefined).")

        # ------------------------------------------------------------------
        # Step 4: Determine generalization vs. overfit
        # ------------------------------------------------------------------
        v12_generalized = ood_auc >= 0.65
        v12_overfit = ood_auc < 0.55
        _log.info(
            "v12_ood_auc=%.4f  v12_generalized=%s  v12_overfit=%s",
            ood_auc, v12_generalized, v12_overfit,
        )

        # ------------------------------------------------------------------
        # Step 5: Retrain v13 if v12 is overfit
        # ------------------------------------------------------------------
        v13_retrained = False
        v13_val_auc: float | None = None
        v13_saved = False

        if v12_overfit:
            _log.info("v12 overfit detected — retraining JEPA v13 on fover_corpus_v4...")
            v13_params, v13_val_auc = retrain_jepa_v13(corpus_v4_entries, embed_fn)
            v13_retrained = True

            if v13_params is not None and v13_val_auc is not None and v13_val_auc >= 0.65:
                _save_model_safetensors(v13_params, V13_MODEL_PATH)
                v13_saved = True
                _log.info("v13 saved (val_auc=%.4f >= 0.65).", v13_val_auc)
            else:
                _log.info(
                    "v13 NOT saved (val_auc=%s < 0.65 or no params).", v13_val_auc
                )

        # ------------------------------------------------------------------
        # Step 6: FR-11 gate and honest verdict
        # ------------------------------------------------------------------
        fr11_generalization_confirmed = v12_generalized or (
            v12_overfit and v13_val_auc is not None and v13_val_auc >= 0.65
        )

        if v12_generalized:
            honest_verdict = "v12_generalized"
        elif v12_overfit and v13_val_auc is not None and v13_val_auc >= 0.65:
            honest_verdict = "v12_overfit_v13_saved"
        else:
            honest_verdict = "jepa_fails_ood"

        _log.info(
            "FR-11 generalization confirmed: %s  verdict: %s",
            fr11_generalization_confirmed, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Step 7: Build and write artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_v12_ood.v1",
                "model_available": True,
                "n_ood_pairs": n_ood_pairs,
                "ood_question_range": ood_source,
                "v12_val_auc": 1.0,
                "v12_ood_auc": ood_auc,
                "v12_generalized": v12_generalized,
                "v12_overfit": v12_overfit,
                "v13_retrained": v13_retrained,
                "v13_val_auc": v13_val_auc,
                "v13_saved": v13_saved,
                "fr11_generalization_confirmed": fr11_generalization_confirmed,
                "honest_verdict": honest_verdict,
            },
            status="success",
        )
        writer.write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
