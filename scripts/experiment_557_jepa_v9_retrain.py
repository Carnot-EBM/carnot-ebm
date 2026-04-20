#!/usr/bin/env python3
"""Experiment 557: JEPA v9 Retrain with LeWorldModel Objective on Diverse FOVER Corpus.

**Researcher summary:**
    Exp 543 (JEPA v8) produced AUC=0.444 — below random — because the corpus had only
    24 pairs with 88% carry violations (constraint_type_entropy ~0 bits).  This is
    RETRO-056.  The fix has two parts:
    1. Diverse corpus from Exp 553 (fover_corpus_v2.json): 132 entries with
       constraint_type_entropy ~1.5 bits.
    2. LeWorldModel two-term objective (arXiv 2603.19312): L_total = L_pred + lambda_kl * KL.
       The Gaussian KL term prevents latent collapse without curriculum scheduling.

    This experiment (v9) combines both fixes and measures whether AUC recovers to >= 0.8.

**Pipeline:**
    0. subprocess kill -9 (harmless zombie reap)
    1. apply_env_autofix()                         — normalise env before any CUDA import
    2. ExperimentTimeoutWatchdog(557, 30)          — hard 30-minute cap
    3. ExperimentTemplate(557, ...)                — scaffolding + deliverable path
    4. Load fover_corpus_v2.json                   — gate if n_labeled < 100
    5. Filter: constraint_type_entropy >= 1.0 bits — gate if entropy too low
    6. 80/20 stratified split by constraint_type
    7. train_leworldmodel(train_pairs, lambda_kl=0.01) — 200 epochs, MSE + KL
    8. Evaluate AUC on held-out test set
    9. Save jepa_predictor_557_real.safetensors
   10. Build artifact with schema='carnot.jepa_retrain.v9'
   11. tmpl.assert_deliverable_written()            — FINAL LINE

Spec: REQ-LEARN-047,
      SCENARIO-LEARN-076, SCENARIO-LEARN-077, SCENARIO-LEARN-078
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Step 0: Kill zombie PIDs FIRST — before any CUDA import.
# ---------------------------------------------------------------------------
import subprocess

subprocess.run(["kill", "-9"], capture_output=True)  # no specific PIDs; harmless call

# ---------------------------------------------------------------------------
# Step 1: apply_env_autofix() — must be called before any CUDA/JAX import.
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
import math  # noqa: E402
from collections import Counter  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402

from carnot.embeddings.jepa_energy import (  # noqa: E402
    _leworldmodel_forward,
    train_leworldmodel,
)
from carnot.pipeline.atomic_writer import AtomicResultWriter  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from scripts.experiment_template import ExperimentTemplate  # noqa: E402

_log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 557
EXP_TITLE = "JEPA v9 Retrain Diverse Corpus"
DELIVERABLE = "results/experiment_557_jepa_v9_retrain.json"
CORPUS_PATH = _REPO_ROOT / "results" / "fover_corpus_v2.json"
MODEL_OUTPUT = _REPO_ROOT / "results" / "jepa_predictor_557_real.safetensors"
BEFORE_AUC = 0.444          # Exp 543 v8 result
MIN_LABELED = 100           # gate: need at least 100 corpus entries
MIN_ENTROPY = 1.0           # gate: need at least 1.0 bits constraint_type_entropy
RETRO_AUC_GATE = 0.800      # retro_056_closed threshold
LAMBDA_KL = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_entropy(entries: list[dict]) -> float:
    """Shannon entropy over constraint_type distribution across all corpus steps."""
    counter: Counter[str] = Counter()
    for e in entries:
        for t in e.get("constraint_types", []):
            counter[t] += 1
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum(
        (c / total) * math.log2(c / total) for c in counter.values() if c > 0
    )


def _stratified_split(
    entries: list[dict], train_frac: float = 0.8, seed: int = 557
) -> tuple[list[dict], list[dict]]:
    """Stratified 80/20 split by constraint_type majority label.

    **For engineers:**
        We stratify on the majority constraint_type of each entry so that both
        train and test sets have representative proportions of each type.  A plain
        random shuffle could put all 'correct' entries in one split.
    """
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

    train_entries = [entries[i] for i in sorted(train_idx)]
    test_entries = [entries[i] for i in sorted(test_idx)]
    return train_entries, test_entries


def _evaluate_auc_on_set(
    params: dict, test_entries: list[dict]
) -> float:
    """Compute ROC-AUC on the test set using trained LeWorldModel params.

    **For engineers:**
        Converts each test entry to a 4-D feature vector, runs the forward pass,
        applies sigmoid to get a score in (0,1), and computes AUC via the trapezoid
        rule.  Returns 0.5 on degenerate test sets (all one class).
    """
    from carnot.embeddings.jepa_energy import _auc_from_scores, _corpus_entry_to_features

    features = [_corpus_entry_to_features(e) for e in test_entries]
    labels = [float(bool(e.get("is_correct", False))) for e in test_entries]

    scores = []
    for feat in features:
        mu, _ = _leworldmodel_forward(params, feat)
        scores.append(float(jax.nn.sigmoid(mu)[0]))

    return _auc_from_scores(scores, labels)


def _save_predictor(params: dict, path: Path) -> None:
    """Save LeWorldModel params as a safetensors file.

    **For engineers:**
        Converts JAX arrays to numpy for safetensors serialisation.  The file can
        be loaded back with safetensors.numpy.load_file() and the params dict can
        be used directly in _leworldmodel_forward().
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: np.array(v) for k, v in params.items()}
    save_file(tensors, str(path))
    _log.info("Saved predictor to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 557: JEPA v9 retrain with LeWorldModel objective on diverse corpus."""

    # Step 2: hard timeout guard
    with ExperimentTimeoutWatchdog(EXP_ID, timeout_minutes=30):

        # Step 3: ExperimentTemplate scaffolding
        tmpl = ExperimentTemplate(
            exp_id=EXP_ID,
            title=EXP_TITLE,
            deliverable=str(_REPO_ROOT / DELIVERABLE),
            requires_gpu=False,
        )
        tmpl.setup()

        # Step 4: Load and gate on corpus size
        _log.info("Loading FOVER corpus v2 from: %s", CORPUS_PATH)
        try:
            raw_corpus: list[dict] = json.loads(CORPUS_PATH.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _log.error("Failed to load corpus: %s", exc)
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_retrain.v9",
                    "inference_mode": "blocked",
                    "corpus_size": 0,
                    "corpus_entropy": 0.0,
                    "lambda_kl": LAMBDA_KL,
                    "before_auc": BEFORE_AUC,
                    "final_auc": BEFORE_AUC,
                    "auc_vs_random": BEFORE_AUC - 0.5,
                    "training_stable": False,
                    "retro_056_closed": False,
                    "honest_verdict": "blocked_corpus_load_error",
                    "block_reason": str(exc),
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        n_labeled = len(raw_corpus) if isinstance(raw_corpus, list) else 0
        _log.info("Corpus entries loaded: %d", n_labeled)

        if n_labeled < MIN_LABELED:
            _log.warning(
                "Corpus has %d entries, need >= %d — writing blocked artifact",
                n_labeled, MIN_LABELED,
            )
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_retrain.v9",
                    "inference_mode": "blocked",
                    "corpus_size": n_labeled,
                    "corpus_entropy": 0.0,
                    "lambda_kl": LAMBDA_KL,
                    "before_auc": BEFORE_AUC,
                    "final_auc": BEFORE_AUC,
                    "auc_vs_random": BEFORE_AUC - 0.5,
                    "training_stable": False,
                    "retro_056_closed": False,
                    "honest_verdict": "blocked_insufficient_pairs",
                    "block_reason": f"n_labeled={n_labeled} < {MIN_LABELED}",
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        # Step 5: Check constraint_type_entropy gate
        entropy_before = _compute_entropy(raw_corpus)
        _log.info("Corpus constraint_type_entropy: %.4f bits", entropy_before)

        # Filter entries that have at least one constraint_type label
        labeled_entries = [e for e in raw_corpus if e.get("constraint_types")]
        entropy_after = _compute_entropy(labeled_entries)
        _log.info("Entropy after filtering unlabeled: %.4f bits (n=%d)", entropy_after, len(labeled_entries))

        if entropy_after < MIN_ENTROPY:
            _log.warning(
                "Entropy %.4f bits < %.1f bits minimum — writing blocked artifact",
                entropy_after, MIN_ENTROPY,
            )
            artifact = tmpl.build_result(
                {
                    "schema": "carnot.jepa_retrain.v9",
                    "inference_mode": "blocked",
                    "corpus_size": n_labeled,
                    "corpus_entropy": entropy_after,
                    "lambda_kl": LAMBDA_KL,
                    "before_auc": BEFORE_AUC,
                    "final_auc": BEFORE_AUC,
                    "auc_vs_random": BEFORE_AUC - 0.5,
                    "training_stable": False,
                    "retro_056_closed": False,
                    "honest_verdict": "blocked_low_entropy",
                    "block_reason": f"entropy={entropy_after:.4f} < {MIN_ENTROPY}",
                },
                status="blocked",
            )
            AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
            tmpl.assert_deliverable_written()
            return

        # Use labeled entries for training (those with constraint_type data)
        # Fall back to full corpus if fewer than MIN_LABELED labeled entries remain
        training_corpus = labeled_entries if len(labeled_entries) >= MIN_LABELED else raw_corpus
        corpus_entropy = entropy_after if len(labeled_entries) >= MIN_LABELED else entropy_before
        _log.info("Training corpus size: %d, entropy: %.4f bits", len(training_corpus), corpus_entropy)

        # Step 6: Stratified 80/20 split by constraint_type
        train_entries, test_entries = _stratified_split(training_corpus)
        _log.info("Train: %d, Test: %d", len(train_entries), len(test_entries))

        # Step 7: Train with LeWorldModel objective
        _log.info("Training JEPA v9 with LeWorldModel objective (lambda_kl=%.3f, 200 epochs)...", LAMBDA_KL)
        training_history = train_leworldmodel(train_entries, lambda_kl=LAMBDA_KL)
        _log.info("Training complete. Final epoch loss: %.4f", training_history[-1][1] if training_history else float("nan"))

        # Extract final trained params for AUC evaluation and saving
        # Re-run training to get final params back (train_leworldmodel returns history only)
        # We need to get the params — rebuild them via a second training call on the same data
        # OR we can evaluate AUC directly using the training history AUC (last epoch)
        # Since train_leworldmodel reports AUC on train set, we need held-out test AUC.

        # For the held-out test AUC, re-instantiate and retrain to get params.
        # This is a small model (200 epochs) — retraining is fast (< 5s).
        import optax  # noqa: PLC0415
        import jax.random as jrandom  # noqa: PLC0415
        from carnot.embeddings.jepa_energy import (  # noqa: PLC0415
            _leworldmodel_init_params,
            _leworldmodel_loss,
            _corpus_entry_to_features,
        )

        X_train = jnp.stack([_corpus_entry_to_features(e) for e in train_entries])
        y_train = jnp.array([float(bool(e.get("is_correct", False))) for e in train_entries], dtype=jnp.float32)
        X_test = jnp.stack([_corpus_entry_to_features(e) for e in test_entries])
        y_test_labels = [float(bool(e.get("is_correct", False))) for e in test_entries]

        params = _leworldmodel_init_params(jrandom.PRNGKey(557))
        opt = optax.adamw(learning_rate=1e-3, weight_decay=1e-4)
        opt_state = opt.init(params)

        def _batch_loss(p: dict):
            def _single(xi, yi):
                return _leworldmodel_loss(p, xi, yi, LAMBDA_KL)
            totals, preds, kls = jax.vmap(_single)(X_train, y_train)
            return jnp.mean(totals), (jnp.mean(preds), jnp.mean(kls))

        for _ in range(200):
            (_, _), grads = jax.value_and_grad(_batch_loss, has_aux=True)(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        # Step 8: Evaluate AUC on held-out test set
        from carnot.embeddings.jepa_energy import _auc_from_scores  # noqa: PLC0415

        test_scores = []
        for feat in X_test:
            mu, _ = _leworldmodel_forward(params, feat)
            test_scores.append(float(jax.nn.sigmoid(mu)[0]))

        final_auc = _auc_from_scores(test_scores, y_test_labels)
        train_auc = training_history[-1][4] if training_history else 0.5
        _log.info("Test AUC: %.4f, Train AUC (last epoch): %.4f", final_auc, train_auc)

        # Step 9: Save predictor
        _save_predictor(params, MODEL_OUTPUT)

        # Step 10: Determine honest_verdict
        training_stable = final_auc > 0.5
        retro_056_closed = final_auc >= RETRO_AUC_GATE

        if final_auc >= RETRO_AUC_GATE:
            honest_verdict = "jepa_recovered"
        elif final_auc > 0.5:
            honest_verdict = "jepa_partial"
        else:
            honest_verdict = "jepa_still_inverted"

        artifact = tmpl.build_result(
            {
                "schema": "carnot.jepa_retrain.v9",
                "inference_mode": "real_data",
                "corpus_size": n_labeled,
                "corpus_entropy": float(corpus_entropy),
                "lambda_kl": LAMBDA_KL,
                "before_auc": BEFORE_AUC,
                "final_auc": float(final_auc),
                "auc_vs_random": float(final_auc) - 0.5,
                "training_stable": training_stable,
                "retro_056_closed": retro_056_closed,
                "honest_verdict": honest_verdict,
                "n_train": len(train_entries),
                "n_test": len(test_entries),
                "epochs_trained": len(training_history),
                "final_train_auc": float(train_auc),
                "model_path": str(MODEL_OUTPUT),
            },
            status="success",
        )
        AtomicResultWriter(str(_REPO_ROOT / DELIVERABLE)).write(artifact)
        tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
