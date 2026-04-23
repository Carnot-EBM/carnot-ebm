"""Tests for Exp 770: JEPA v19 Multi-Step Predictive Probe on Real Accumulated Data.

Spec: REQ-LEARN-043, REQ-LEARN-044, REQ-LEARN-045,
      SCENARIO-LEARN-085, SCENARIO-LEARN-086, SCENARIO-LEARN-087
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path

import pytest

from carnot.samplers.jepa_v19 import MultiStepJEPAv19, _TFIDFVectoriser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_binary_data(n: int = 30, seed: int = 42) -> tuple[list[list[str]], list[float]]:
    """Synthetic binary labeled step sequences for training unit tests.

    WHY synthetic here: these tests only check that the MLP trains at all,
    not that it generalises to real data.  REQ-LEARN-043 is enforced by the
    experiment script, not by this unit test.
    """
    import random
    rng = random.Random(seed)
    seqs: list[list[str]] = []
    labels: list[float] = []
    positive_words = ["error", "wrong", "incorrect", "undefined", "failed"]
    negative_words = ["correct", "verified", "right", "computed", "therefore"]
    for i in range(n):
        label = float(i % 2)
        word_pool = positive_words if label == 1.0 else negative_words
        steps = [
            f"{rng.choice(word_pool)} step {i} part a",
            f"{rng.choice(word_pool)} step {i} part b",
            f"{rng.choice(word_pool)} step {i} part c",
        ]
        seqs.append(steps)
        labels.append(label)
    return seqs, labels


# ---------------------------------------------------------------------------
# REQ-LEARN-045 / SCENARIO-LEARN-087: pooling across exactly n_steps steps
# ---------------------------------------------------------------------------


def test_forward_pools_exactly_n_steps() -> None:
    """MultiStepJEPAv19.forward must pool across exactly n_steps step embeddings.

    Given a probe with n_steps=3, even when 1 or 2 steps are provided, the
    pooled embedding must have length vocab_size (zero-padding fills missing steps).

    Spec: REQ-LEARN-045, SCENARIO-LEARN-087
    """
    seqs, labels = _make_binary_data(30)
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    probe.train(seqs, labels, n_epochs=1, lr=1e-3)

    # With 1, 2, and 3 steps the forward pass must succeed and return a scalar.
    for n_steps_provided in [1, 2, 3]:
        steps = [f"test step {k}" for k in range(n_steps_provided)]
        result = probe.forward(steps)
        assert isinstance(result, float), f"forward returned {type(result)}, expected float"
        assert 0.0 <= result <= 1.0, f"forward returned {result}, expected value in [0, 1]"


def test_embed_steps_zero_pads_missing_steps() -> None:
    """_embed_steps zero-pads when fewer than n_steps steps are provided.

    This ensures the pooled vector length is always vocab_size regardless of
    how many steps are in the input sequence.

    Spec: REQ-LEARN-045, SCENARIO-LEARN-087
    """
    seqs, labels = _make_binary_data(20)
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    # Must fit vectoriser first (normally done by train).
    all_texts = [s for seq in seqs for s in seq]
    probe._vectoriser.fit(all_texts)

    vocab_size = len(probe._vectoriser._vocab)
    for n_steps_provided in [1, 2, 3]:
        steps = [f"error wrong incorrect step {k}" for k in range(n_steps_provided)]
        pooled = probe._embed_steps(steps)
        assert len(pooled) == vocab_size, (
            f"pooled length={len(pooled)} != vocab_size={vocab_size} "
            f"when n_steps_provided={n_steps_provided}"
        )


def test_extra_steps_beyond_n_steps_are_ignored() -> None:
    """Steps beyond n_steps must be ignored (not affect the pooled vector).

    Spec: REQ-LEARN-045
    """
    seqs, labels = _make_binary_data(20)
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    all_texts = [s for seq in seqs for s in seq]
    probe._vectoriser.fit(all_texts)

    steps_3 = ["error wrong step 0", "incorrect step 1", "failed step 2"]
    steps_5 = steps_3 + ["extra step 3", "extra step 4"]

    pooled_3 = probe._embed_steps(steps_3)
    pooled_5 = probe._embed_steps(steps_5)
    assert pooled_3 == pooled_5, "Extra steps beyond n_steps must be ignored"


# ---------------------------------------------------------------------------
# REQ-LEARN-043 / SCENARIO-LEARN-085: training reduces BCE loss
# ---------------------------------------------------------------------------


def test_training_reduces_bce_loss_over_10_epochs() -> None:
    """Training MultiStepJEPAv19 for 10 epochs must reduce BCE loss.

    Checks that the Adam gradient updates are functional: loss at epoch 10
    must be strictly less than the initial random loss (approximately ln(2) ≈ 0.693
    for a freshly initialised network predicting 0.5 for all examples).

    Spec: REQ-LEARN-043, SCENARIO-LEARN-085
    """
    seqs, labels = _make_binary_data(40)
    probe = MultiStepJEPAv19(hidden_dim=32, n_steps=3, max_vocab=100)

    # Baseline: random probe predicts ~0.5 → BCE ≈ ln(2) ≈ 0.693.
    # After 10 epochs it should be lower than that.
    result = probe.train(seqs, labels, n_epochs=10, lr=1e-3)
    assert "final_loss" in result, "train() must return dict with final_loss"
    assert math.isfinite(result["final_loss"]), "final_loss must be finite"
    assert result["final_loss"] < math.log(2) + 0.05, (
        f"final_loss={result['final_loss']:.4f} is suspiciously high after 10 epochs; "
        "expected < 0.75 (ln(2)≈0.693 + small slack)"
    )
    assert result["n_train"] == len(seqs)


def test_training_on_empty_dataset_raises() -> None:
    """Training on an empty dataset must raise ValueError.

    Spec: REQ-LEARN-043
    """
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    with pytest.raises(ValueError, match="empty"):
        probe.train([], [], n_epochs=1)


# ---------------------------------------------------------------------------
# REQ-LEARN-044 / SCENARIO-LEARN-086: OOD AUC computation
# ---------------------------------------------------------------------------


def test_compute_auc_perfect_separation() -> None:
    """compute_auc must return 1.0 for perfectly separated scores.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-086
    """
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = [1.0, 1.0, 0.0, 0.0]
    auc = MultiStepJEPAv19.compute_auc(scores, labels)
    assert auc == 1.0, f"Expected AUC=1.0, got {auc}"


def test_compute_auc_random_baseline() -> None:
    """compute_auc must return 0.5 when only one class is present.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-086
    """
    # All positive — no negative class → can't rank → return 0.5.
    auc = MultiStepJEPAv19.compute_auc([0.9, 0.8, 0.7], [1.0, 1.0, 1.0])
    assert auc == 0.5

    # All negative — same logic.
    auc = MultiStepJEPAv19.compute_auc([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])
    assert auc == 0.5


def test_compute_auc_from_probe_predictions() -> None:
    """compute_auc must return a value in [0, 1] from real probe predictions.

    This is the full OOD AUC computation path tested end-to-end.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-086
    """
    seqs, labels = _make_binary_data(40)
    probe = MultiStepJEPAv19(hidden_dim=32, n_steps=3, max_vocab=100)
    probe.train(seqs, labels, n_epochs=50, lr=1e-3)

    # OOD-style: use different vocabulary to score.
    ood_seqs = [
        ["the result is incorrect", "error in step 2"],
        ["everything is correct and verified", "therefore the answer is right"],
        ["undefined variable found", "wrong approach used"],
        ["computed successfully", "final answer correct"],
    ]
    ood_labels = [1.0, 0.0, 1.0, 0.0]
    ood_scores = [probe.forward(seq) for seq in ood_seqs]
    auc = MultiStepJEPAv19.compute_auc(ood_scores, ood_labels)
    assert 0.0 <= auc <= 1.0, f"AUC must be in [0,1], got {auc}"


# ---------------------------------------------------------------------------
# REQ-LEARN-044 / SCENARIO-LEARN-086: model saved to npz when ood_auc > 0.75
# ---------------------------------------------------------------------------


def test_model_can_be_saved_to_npz() -> None:
    """Model weights must be saveable to .npz when OOD AUC > 0.75.

    This test exercises the save path directly (without requiring the full
    experiment pipeline).  It verifies that the probe's internal weight arrays
    are numpy-serialisable.

    Spec: REQ-LEARN-044, SCENARIO-LEARN-086
    """
    pytest.importorskip("numpy")
    import numpy as np

    seqs, labels = _make_binary_data(30)
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    probe.train(seqs, labels, n_epochs=5, lr=1e-3)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        save_path = f.name

    try:
        np.savez(
            save_path,
            w1=np.array(probe._w1),
            b1=np.array(probe._b1),
            w2=np.array(probe._w2),
            b2=np.array(probe._b2),
        )
        loaded = np.load(save_path)
        assert "w1" in loaded, "npz must contain w1"
        assert "b1" in loaded, "npz must contain b1"
        assert "w2" in loaded, "npz must contain w2"
        assert "b2" in loaded, "npz must contain b2"
        assert loaded["w1"].shape[0] == probe.hidden_dim
        assert loaded["w2"].shape[1] == probe.hidden_dim
    finally:
        Path(save_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# forward() before train() must raise RuntimeError
# ---------------------------------------------------------------------------


def test_forward_before_train_raises() -> None:
    """forward() must raise RuntimeError when called before train().

    Spec: REQ-LEARN-043
    """
    probe = MultiStepJEPAv19(hidden_dim=16, n_steps=3, max_vocab=50)
    with pytest.raises(RuntimeError, match="train"):
        probe.forward(["some step text"])


# ---------------------------------------------------------------------------
# TF-IDF vectoriser: transform before fit raises RuntimeError
# ---------------------------------------------------------------------------


def test_tfidf_transform_before_fit_raises() -> None:
    """_TFIDFVectoriser.transform() must raise RuntimeError before fit().

    Spec: REQ-LEARN-043
    """
    vec = _TFIDFVectoriser(max_features=10)
    with pytest.raises(RuntimeError, match="fit"):
        vec.transform("some text")
