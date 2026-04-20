#!/usr/bin/env python3
"""Experiment 566: JEPA PURE MinForm Loss — RETRO-060 Fix Validation.

**Researcher summary (RETRO-060):**
    Exp 557 (JEPA v9) produced AUC=0.4286 on the 132-pair FOVER corpus — below random
    baseline — for the SECOND consecutive retrain.  Root cause: binary BCE loss lets the
    model hedge toward P=0.5 everywhere, producing near-zero gradient and no useful signal.

    Fix (arXiv 2504.15275, PURE PRM objective):
        score(chain) = min(step_score(t) for t in chain)
        loss = mean(max(0, margin - (min_score_incorrect - min_score_correct)))

    A chain with even one bad step gets a strong low signal.  A chain with all good steps
    gets a strong high signal.  This enforces a hard contrastive margin — exactly the
    signal that NUP Probe v4 used to achieve AUC=1.0.

    CPU-only — no GPU needed.

**Pipeline:**
    0. apply_env_autofix()
    1. ExperimentTimeoutWatchdog(566, 30)
    2. ExperimentTemplate(566, ...)
    3. Load 132-pair corpus from results/fover_corpus_v2.json
    4. Build correct_chains and incorrect_chains via pairs_to_pure_chains
    5. Train small JEPA MLP for 100 epochs using PUREMinFormLoss
    6. Evaluate AUC on held-out 20% split
    7. Compare against Exp 557 baseline AUC=0.4286
    8. Write artifact with schema='carnot.jepa_pure_loss.v1'
    9. tmpl.assert_deliverable_written()   — FINAL LINE

Spec: REQ-LEARN-061, REQ-LEARN-062,
      SCENARIO-LEARN-095, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: apply_env_autofix() — must be called before any JAX import.
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
from collections import Counter  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.random as jrandom  # noqa: E402
import numpy as np  # noqa: E402

from carnot.inference.jepa_pure_loss import (  # noqa: E402
    JEPAChainScore,
    PUREMinFormLoss,
    pairs_to_pure_chains,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 566
EXP_TITLE = "JEPA PURE MinForm Loss"
DELIVERABLE = "results/experiment_566_jepa_pure_margin.json"
CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v2.json"
OLD_AUC = 0.4286          # Exp 557 baseline
MARGIN = 1.0
N_EPOCHS = 100
TRAIN_FRAC = 0.8
SEED = 566
FEAT_DIM = 4              # [frac_correct, frac_incorrect, frac_nv, norm_n_steps]
HIDDEN_DIM = 16


# ---------------------------------------------------------------------------
# Feature extraction (same as Exp 557 — compatibility with existing corpus)
# ---------------------------------------------------------------------------


def _entry_to_features(entry: dict) -> jnp.ndarray:
    """Convert a FOVER corpus dict to a 4-D feature vector.

    **For engineers:**
        We reuse the same feature encoding as Exp 557 (_corpus_entry_to_features) so
        that the PURE loss experiment trains on the exact same feature space and results
        are directly comparable.  Features are:
            [frac_correct, frac_incorrect, frac_not_verifiable, norm_n_steps]
        where norm_n_steps = min(1.0, n_steps / 20).
    """
    ctypes = entry.get("constraint_types", [])
    n = len(ctypes) if ctypes else 0
    if n == 0:
        return jnp.zeros(FEAT_DIM)
    frac_correct = sum(1 for t in ctypes if t == "correct") / n
    frac_incorrect = sum(1 for t in ctypes if t == "incorrect") / n
    frac_nv = sum(1 for t in ctypes if t == "not_verifiable") / n
    norm_n = min(1.0, n / 20.0)
    return jnp.array([frac_correct, frac_incorrect, frac_nv, norm_n], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Tiny MLP — same architecture as Exp 557 (input=4, hidden=16, output=1)
# ---------------------------------------------------------------------------


def _init_params(key: jnp.ndarray) -> dict:
    """Initialise a 2-layer MLP: input(4) -> hidden(16) -> output(1)."""
    k1, k2 = jrandom.split(key)
    lim1 = float(jnp.sqrt(6.0 / (FEAT_DIM + HIDDEN_DIM)))
    lim2 = float(jnp.sqrt(6.0 / (HIDDEN_DIM + 1)))
    return {
        "w1": jrandom.uniform(k1, (HIDDEN_DIM, FEAT_DIM), minval=-lim1, maxval=lim1),
        "b1": jnp.zeros(HIDDEN_DIM),
        "w2": jrandom.uniform(k2, (1, HIDDEN_DIM), minval=-lim2, maxval=lim2),
        "b2": jnp.zeros(1),
    }


def _forward(params: dict, x: jnp.ndarray) -> float:
    """Forward pass: x -> scalar score in (-inf, +inf)."""
    h = jax.nn.silu(params["w1"] @ x + params["b1"])
    out = params["w2"] @ h + params["b2"]
    return float(out[0])


# ---------------------------------------------------------------------------
# Training loop with PURE min-form loss
# ---------------------------------------------------------------------------


def _train_pure(
    train_entries: list[dict],
    margin: float,
    n_epochs: int,
    seed: int,
) -> tuple[dict, list[float]]:
    """Train the MLP using PUREMinFormLoss for n_epochs.

    **For engineers:**
        We use a simple gradient-descent loop via JAX's jit+grad.  The contrastive
        loss is computed per-batch (all correct vs all incorrect within the training set).
        We return the final params and a list of per-epoch loss values.

        Step scores are computed by running each entry's feature vector through the
        current params (treating each entry as a 1-step chain — the per-step granularity
        that PUREMinFormLoss expects).  The step_score IS the scalar model output; the
        min_score of a 1-step chain equals that score.
    """
    import optax  # local import to keep startup light

    params = _init_params(jrandom.PRNGKey(seed))
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)
    loss_fn = PUREMinFormLoss(margin=margin)
    epoch_losses: list[float] = []

    def _score_entries(p: dict, entries: list[dict]) -> list[JEPAChainScore]:
        """Score a list of entries and return JEPAChainScore objects."""
        chains: list[JEPAChainScore] = []
        for e in entries:
            feat = _entry_to_features(e)
            score = float(jax.nn.sigmoid(
                p["w2"] @ jax.nn.silu(p["w1"] @ feat + p["b1"]) + p["b2"]
            )[0])
            chains.append(
                JEPAChainScore(
                    chain_id=f"{e.get('question','')[:30]}/{e.get('model_id','')}",
                    step_scores=[score],
                    min_score=score,
                    is_correct=bool(e.get("is_correct", False)),
                )
            )
        return chains

    correct_entries = [e for e in train_entries if e.get("is_correct", False)]
    incorrect_entries = [e for e in train_entries if not e.get("is_correct", False)]

    def _compute_pure_loss_value(p: dict) -> jnp.ndarray:
        """Compute contrastive margin loss as a JAX scalar for grad."""
        # Score all entries with this params snapshot.
        pairs_loss = jnp.array(0.0)
        n_pairs = 0
        for ce in correct_entries:
            feat_c = _entry_to_features(ce)
            score_c = jax.nn.sigmoid(
                p["w2"] @ jax.nn.silu(p["w1"] @ feat_c + p["b1"]) + p["b2"]
            )[0]
            for we in incorrect_entries:
                feat_w = _entry_to_features(we)
                score_w = jax.nn.sigmoid(
                    p["w2"] @ jax.nn.silu(p["w1"] @ feat_w + p["b1"]) + p["b2"]
                )[0]
                # gap = score_wrong - score_correct; want incorrect > correct by >= margin
                gap = score_w - score_c
                pairs_loss = pairs_loss + jnp.maximum(jnp.array(0.0), margin - gap)
                n_pairs += 1
        if n_pairs == 0:
            return jnp.array(0.0)
        return pairs_loss / n_pairs

    grad_fn = jax.jit(jax.grad(_compute_pure_loss_value))

    for epoch in range(n_epochs):
        grads = grad_fn(params)
        updates, opt_state_new = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        opt_state = opt_state_new

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            loss_val = float(_compute_pure_loss_value(params))
            epoch_losses.append(loss_val)
            _log.info("Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, loss_val)

    return params, epoch_losses


# ---------------------------------------------------------------------------
# AUC evaluation
# ---------------------------------------------------------------------------


def _evaluate_auc(params: dict, entries: list[dict]) -> float:
    """ROC-AUC on a list of entries using the trained MLP.

    **For engineers:**
        Each entry's 4-D feature vector is passed through the MLP; sigmoid(output)
        is the predicted correctness probability.  We use the same trapezoid AUC
        implementation as Exp 557 to ensure comparability.
    """
    from carnot.embeddings.jepa_energy import _auc_from_scores

    scores: list[float] = []
    labels: list[float] = []
    for e in entries:
        feat = _entry_to_features(e)
        score = float(jax.nn.sigmoid(
            params["w2"] @ jax.nn.silu(params["w1"] @ feat + params["b1"]) + params["b2"]
        )[0])
        scores.append(score)
        labels.append(float(bool(e.get("is_correct", False))))
    from carnot.embeddings.jepa_energy import _auc_from_scores  # re-import for clarity
    return _auc_from_scores(scores, labels)


# ---------------------------------------------------------------------------
# Stratified split (same logic as Exp 557)
# ---------------------------------------------------------------------------


def _stratified_split(
    entries: list[dict],
    train_frac: float = 0.8,
    seed: int = 566,
) -> tuple[list[dict], list[dict]]:
    """80/20 stratified split by majority constraint_type."""
    rng = np.random.RandomState(seed)

    def _majority(e: dict) -> str:
        ct = e.get("constraint_types", [])
        if not ct:
            return "unknown"
        return Counter(ct).most_common(1)[0][0]

    by_class: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        cls = _majority(e)
        by_class.setdefault(cls, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for idx_list in by_class.values():
        arr = np.array(idx_list)
        rng.shuffle(arr)
        n_train = max(1, int(len(arr) * train_frac))
        train_idx.extend(arr[:n_train].tolist())
        test_idx.extend(arr[n_train:].tolist())

    return [entries[i] for i in sorted(train_idx)], [entries[i] for i in sorted(test_idx)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 566: JEPA PURE MinForm Loss retrain and AUC comparison."""

    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):

        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # ------------------------------------------------------------------
        # Load corpus
        # ------------------------------------------------------------------
        _log.info("Loading FOVER corpus v2 from: %s", CORPUS_PATH)
        try:
            raw_corpus: list[dict] = json.loads(CORPUS_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("Failed to load corpus: %s", exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_pure_loss.v1",
                    "n_pairs": 0,
                    "loss_function": "pure_min_form",
                    "margin": MARGIN,
                    "train_auc": 0.5,
                    "val_auc": 0.5,
                    "old_auc": OLD_AUC,
                    "auc_improvement": 0.0,
                    "retro_060_partial": False,
                    "honest_verdict": "blocked_corpus_load_error",
                    "block_reason": str(exc),
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        n_corpus = len(raw_corpus) if isinstance(raw_corpus, list) else 0
        _log.info("Corpus entries: %d", n_corpus)

        # ------------------------------------------------------------------
        # Split and train
        # ------------------------------------------------------------------
        train_entries, test_entries = _stratified_split(raw_corpus, TRAIN_FRAC, SEED)
        _log.info("Train: %d  Test: %d", len(train_entries), len(test_entries))

        params, epoch_losses = _train_pure(train_entries, MARGIN, N_EPOCHS, SEED)

        # ------------------------------------------------------------------
        # Evaluate
        # ------------------------------------------------------------------
        train_auc = _evaluate_auc(params, train_entries)
        val_auc = _evaluate_auc(params, test_entries)
        _log.info("Train AUC=%.4f  Val AUC=%.4f", train_auc, val_auc)

        # ------------------------------------------------------------------
        # pairs_to_pure_chains counts (informational)
        # ------------------------------------------------------------------

        class _DictEntry:
            """Adapter to make raw dicts quack like FOVERCorpusEntry for pairs_to_pure_chains."""
            def __init__(self, d: dict) -> None:
                self.question = str(d.get("question", ""))
                self.model_id = str(d.get("model_id", "unknown"))
                self.is_correct = bool(d.get("is_correct", False))
                self.cot_steps = d.get("cot_steps", [])
                self.response = str(d.get("response", ""))

        adapted = [_DictEntry(e) for e in raw_corpus]
        embed_fn = lambda text: jnp.array([float(len(text)) / 1000.0])  # trivial proxy
        correct_chains, incorrect_chains = pairs_to_pure_chains(adapted, embed_fn)

        n_pairs = len(correct_chains) + len(incorrect_chains)
        _log.info("Correct chains: %d  Incorrect chains: %d", len(correct_chains), len(incorrect_chains))

        # ------------------------------------------------------------------
        # Honest verdict
        # ------------------------------------------------------------------
        auc_improvement = val_auc - OLD_AUC
        retro_060_partial = val_auc > 0.5

        if val_auc > 0.5:
            honest_verdict = "loss_redesign_success"
        elif val_auc > OLD_AUC:
            honest_verdict = "loss_redesign_partial"
        else:
            honest_verdict = "loss_redesign_no_improvement"

        _log.info(
            "AUC improvement=%.4f  retro_060_partial=%s  verdict=%s",
            auc_improvement, retro_060_partial, honest_verdict,
        )

        # ------------------------------------------------------------------
        # Write artifact
        # ------------------------------------------------------------------
        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_pure_loss.v1",
                "n_pairs": n_pairs,
                "loss_function": "pure_min_form",
                "loss_function_old": "binary_ce",
                "margin": MARGIN,
                "train_auc": train_auc,
                "val_auc": val_auc,
                "old_auc": OLD_AUC,
                "auc_improvement": auc_improvement,
                "retro_060_partial": retro_060_partial,
                "honest_verdict": honest_verdict,
                "n_correct_chains": len(correct_chains),
                "n_incorrect_chains": len(incorrect_chains),
                "n_train": len(train_entries),
                "n_test": len(test_entries),
                "epochs_trained": N_EPOCHS,
                "final_epoch_loss": epoch_losses[-1] if epoch_losses else None,
            },
            status="success",
        )

        AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
