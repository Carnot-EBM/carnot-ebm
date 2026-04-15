#!/usr/bin/env python3
"""Experiment 346: EORM CoT Energy Reward Model — Training and Evaluation.

**Researcher summary:**
    Trains an EORM-style energy reward model (arXiv 2505.14999) on live
    benchmark (question, response, correctness) pairs from Exp 340.
    Evaluates ranking quality via AUC-ROC on a held-out split.
    Falls back to synthetic data when live pairs are unavailable.

**Detailed explanation for engineers:**
    This experiment:

    1. **Loads training data** from Exp 340's live precision benchmark.
       Each pair in the training set is (correct_response, incorrect_response,
       question) — a "contrastive pair" where we know one response is right
       and one is wrong.

       If Exp 340 does not contain actual (response, correctness) pairs (it was
       marked "partial"), we generate 100 synthetic pairs using a simple
       arithmetic template.  The artifact is labelled ``inference_mode=simulated``
       in that case so downstream analysis knows it is not live GPU data.

    2. **Trains EORMModel** for 10 epochs (CI) or 50 epochs (live GPU) on the
       training split using contrastive hinge loss.

    3. **Evaluates** on the held-out test split by computing AUC-ROC:
       for each test pair, the model should score the correct response lower
       (better) than the incorrect one.  AUC = 1.0 means perfect ranking.

    4. **Saves** the trained model to results/eorm_model_346.safetensors.

    5. **Compares** vs the JEPA gate (Tier 3 predictor from Exp 308/309) using
       the JEPA gate's best_TP_rate as a reference signal.  Note: JEPA and EORM
       solve different sub-problems (partial response gate vs full-response rank)
       so this comparison is directional, not apples-to-apples.

    6. **Writes artifact** to results/experiment_346_eorm_training.json with
       schema "carnot.eorm.v1".

Usage::

    JAX_PLATFORMS=cpu python scripts/experiment_346_eorm_training.py

    # With live GPU (ROCm):
    python scripts/experiment_346_eorm_training.py

Spec: REQ-LEARN-022, REQ-LEARN-023,
      SCENARIO-LEARN-038, SCENARIO-LEARN-039, SCENARIO-LEARN-040
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root path wiring — must happen before any carnot imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import jax.random as jrandom

from carnot.models.eorm import CoTEnergyInput, EORMModel, EORMTrainer
from scripts.experiment_template import ExperimentTemplate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP_ID = 346
TITLE = "EORM CoT energy reward model — training and AUC-ROC evaluation"
DELIVERABLE = "results/experiment_346_eorm_training.json"
MODEL_SAVE_PATH = "results/eorm_model_346.safetensors"

# Epoch counts: fewer for CI (no GPU), more for live GPU training
CI_EPOCHS = 10
LIVE_EPOCHS = 50

# Train/test split ratio
TEST_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_exp340_pairs() -> list[tuple[str, str, str]] | None:
    """Try to load (correct_response, incorrect_response, question) pairs from Exp 340.

    **For engineers:**
        Exp 340 records a "live precision benchmark" that scores GSM8K questions
        against two models.  We look for a ``responses`` list in the result
        artifact.  Each item should have keys: ``question``, ``response``,
        ``correct`` (bool).

        We pair up correct and incorrect responses for the same question to
        form contrastive training examples.

        Returns None if the file is absent or does not contain usable pairs.
        The caller should fall back to synthetic data in that case.

    Returns:
        List of (correct_response, incorrect_response, question) triples,
        or None if data is unavailable.
    """
    path = _REPO_ROOT / "results" / "experiment_340_live_precision_benchmark.json"
    if not path.exists():
        _log.warning("Exp 340 result not found: %s", path)
        return None

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log.warning("Could not parse Exp 340 result: %s", exc)
        return None

    # Exp 340 was marked "partial" with no actual response pairs in the artifact
    responses_raw = data.get("responses") or data.get("results") or []
    if not isinstance(responses_raw, list) or not responses_raw:
        _log.info("Exp 340 has no response list — falling back to synthetic data.")
        return None

    # Group by question
    by_question: dict[str, list[dict]] = {}
    for item in responses_raw:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        by_question.setdefault(q, []).append(item)

    pairs: list[tuple[str, str, str]] = []
    for question, items in by_question.items():
        corrects = [it["response"] for it in items if it.get("correct") is True]
        incorrects = [it["response"] for it in items if it.get("correct") is False]
        for c in corrects:
            for ic in incorrects:
                pairs.append((str(c), str(ic), question))

    if not pairs:
        _log.info("Exp 340 has no usable contrastive pairs — falling back to synthetic.")
        return None

    _log.info("Loaded %d contrastive pairs from Exp 340.", len(pairs))
    return pairs


def _generate_synthetic_pairs(n: int = 100) -> list[tuple[str, str, str]]:
    """Generate synthetic (correct, incorrect, question) pairs for CI/offline use.

    **For engineers:**
        When live benchmark data is unavailable, we create deterministic
        arithmetic problems.  For each problem a×b:
        - Correct response: "The answer is <a*b>."
        - Incorrect response: "The answer is <a*b + delta>."
          where delta is a small nonzero offset (1–9).

        These synthetic pairs test that the training pipeline runs end-to-end
        without requiring any GPU or external data.  Results from synthetic
        training are clearly labelled ``inference_mode="simulated"`` in the
        artifact so they are excluded from headline claims.

    Args:
        n: Number of pairs to generate.

    Returns:
        List of (correct_response, incorrect_response, question) triples.
    """
    rng = random.Random(42)
    pairs: list[tuple[str, str, str]] = []
    for i in range(n):
        a = rng.randint(2, 50)
        b = rng.randint(2, 20)
        answer = a * b
        delta = rng.randint(1, 9)
        question = f"What is {a} times {b}?"
        correct = (
            f"Step 1: {a} times {b}. "
            f"Step 2: Multiply to get {answer}. "
            f"The answer is {answer}."
        )
        incorrect = (
            f"Step 1: Trying {a} times {b}. "
            f"Step 2: I think the answer is {answer + delta}. "
            f"The answer is {answer + delta}."
        )
        pairs.append((correct, incorrect, question))
    return pairs


# ---------------------------------------------------------------------------
# AUC-ROC computation
# ---------------------------------------------------------------------------

def _compute_auc_roc(
    model: EORMModel,
    test_pairs: list[tuple[str, str, str]],
) -> float:
    """Compute AUC-ROC for the model on test (correct, incorrect, question) pairs.

    **For engineers:**
        For each test pair we compute:
            score = E(incorrect) - E(correct)

        A positive score means the model correctly ranks the incorrect response
        higher (worse) than the correct one.  AUC-ROC is the probability that
        the model correctly ranks a random (correct, incorrect) pair.

        Calculation: AUC = #{pairs where score > 0} / total_pairs.
        (Ties count as 0.5.)  This is equivalent to the Wilcoxon-Mann-Whitney
        rank-sum statistic, which equals the trapezoidal AUC for binary classifiers.

    Args:
        model: Trained EORMModel.
        test_pairs: List of (correct_response, incorrect_response, question).

    Returns:
        AUC-ROC in [0, 1].  0.5 = random, 1.0 = perfect.
    """
    if not test_pairs:
        return 0.5

    n_correct = 0
    n_tie = 0
    for correct_resp, incorrect_resp, question in test_pairs:
        e_correct = model.energy(
            CoTEnergyInput(question_text=question, response_text=correct_resp)
        )
        e_incorrect = model.energy(
            CoTEnergyInput(question_text=question, response_text=incorrect_resp)
        )
        diff = e_incorrect - e_correct
        if diff > 0:
            n_correct += 1
        elif diff == 0:
            n_tie += 1

    return (n_correct + 0.5 * n_tie) / len(test_pairs)


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run Experiment 346: EORM training and evaluation."""
    tmpl = ExperimentTemplate(EXP_ID, TITLE, DELIVERABLE)
    tmpl.setup()

    # ------------------------------------------------------------------ #
    # 1. Load or generate training data                                   #
    # ------------------------------------------------------------------ #
    live_pairs = _load_exp340_pairs()
    if live_pairs is not None:
        all_pairs = live_pairs
        inference_mode = "live_benchmark"
        _log.info("Using %d live pairs from Exp 340.", len(all_pairs))
    else:
        all_pairs = _generate_synthetic_pairs(100)
        inference_mode = "simulated"
        _log.info("Using %d synthetic pairs (no live data).", len(all_pairs))

    # Shuffle with fixed seed for reproducibility
    rng = random.Random(42)
    rng.shuffle(all_pairs)

    # Train / test split
    n_test = max(1, int(len(all_pairs) * TEST_FRACTION))
    test_pairs = all_pairs[:n_test]
    train_pairs = all_pairs[n_test:]

    _log.info(
        "Split: %d train pairs, %d test pairs.", len(train_pairs), len(test_pairs)
    )

    # ------------------------------------------------------------------ #
    # 2. Initialise model and trainer                                     #
    # ------------------------------------------------------------------ #
    force_live = os.environ.get("CARNOT_FORCE_LIVE", "0") == "1"
    n_epochs = LIVE_EPOCHS if force_live else CI_EPOCHS

    # Use embed_dim=128 (default) when live; smaller for faster CI
    embed_dim = 128 if force_live else 64
    n_layers = 2
    n_heads = 4

    _log.info(
        "Initialising EORMModel: embed_dim=%d, n_layers=%d, n_heads=%d, epochs=%d",
        embed_dim, n_layers, n_heads, n_epochs,
    )

    model = EORMModel(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=256,
        vocab_size=2048,
        key=jrandom.PRNGKey(42),
    )
    trainer = EORMTrainer(model, lr=1e-3, margin=1.0)

    n_params = model.n_params
    _log.info("EORMModel parameter count: %d", n_params)

    # ------------------------------------------------------------------ #
    # 3. Evaluate before training (baseline AUC)                         #
    # ------------------------------------------------------------------ #
    auc_before = _compute_auc_roc(model, test_pairs)
    _log.info("AUC before training: %.4f", auc_before)

    # ------------------------------------------------------------------ #
    # 4. Train                                                            #
    # ------------------------------------------------------------------ #
    epoch_losses: list[float] = []
    for epoch in range(n_epochs):
        loss = trainer.train_epoch(train_pairs, batch_size=16)
        epoch_losses.append(loss)
        if epoch % max(1, n_epochs // 5) == 0:
            _log.info("Epoch %d/%d  loss=%.4f", epoch + 1, n_epochs, loss)

        # Checkpoint every 5 epochs (prevents data loss on conductor interrupts)
        if (epoch + 1) % 5 == 0:
            tmpl.checkpoint_save(
                {"epoch_losses": epoch_losses, "epoch": epoch + 1},
                step=epoch + 1,
            )

    mean_loss_final = epoch_losses[-1] if epoch_losses else 0.0
    _log.info("Final epoch loss: %.4f", mean_loss_final)

    # ------------------------------------------------------------------ #
    # 5. Evaluate after training                                          #
    # ------------------------------------------------------------------ #
    auc_roc = _compute_auc_roc(model, test_pairs)
    _log.info("AUC-ROC on test set: %.4f", auc_roc)

    # Load JEPA benchmark reference (Exp 308) for comparison
    jepa_ref_path = _REPO_ROOT / "results" / "experiment_308_jepa_gate_benchmark.json"
    vs_jepa_auc: float | None = None
    if jepa_ref_path.exists():
        try:
            jepa_data = json.loads(jepa_ref_path.read_text())
            # Exp 308 reports best_TP_rate (true positive rate at chosen threshold)
            # Not a true AUC, but the best available comparison signal
            vs_jepa_auc = float(jepa_data.get("best_TP_rate", 0.0))
            _log.info("JEPA gate best_TP_rate (reference): %.4f", vs_jepa_auc)
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    # 6. Save model                                                       #
    # ------------------------------------------------------------------ #
    model_path = _REPO_ROOT / MODEL_SAVE_PATH
    model.save(model_path)
    _log.info("Saved trained model to %s", model_path)

    # ------------------------------------------------------------------ #
    # 7. Build and write artifact                                         #
    # ------------------------------------------------------------------ #
    artifact = tmpl.build_result(
        {
            "schema": "carnot.eorm.v1",
            "inference_mode": inference_mode,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
            "n_epochs": n_epochs,
            "embed_dim": embed_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_params": n_params,
            "training_mode": "live_gpu" if force_live else "ci_cpu",
            "auc_roc": round(auc_roc, 4),
            "auc_before_training": round(auc_before, 4),
            "mean_loss_final": round(mean_loss_final, 6),
            "epoch_losses": [round(l, 6) for l in epoch_losses],
            "vs_jepa_tp_rate": vs_jepa_auc,
            "model_path": MODEL_SAVE_PATH,
        },
        status="success",
    )

    output_path = _REPO_ROOT / DELIVERABLE
    output_path.write_text(json.dumps(artifact, indent=2))
    _log.info("Wrote artifact to %s", output_path)
    _log.info(
        "Done — AUC-ROC=%.4f  n_params=%d  n_train=%d  n_test=%d",
        auc_roc, n_params, len(train_pairs), len(test_pairs),
    )


if __name__ == "__main__":
    main()
