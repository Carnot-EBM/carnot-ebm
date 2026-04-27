"""Tests for SCEnergyModel and the Exp 944 helper functions.

All tests trace to REQ-MODEL-031 or SCENARIO-MODEL-016 as required by spec coverage.

Spec: REQ-MODEL-031, SCENARIO-MODEL-016
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from python.carnot.models.sc_energy import (
    SCEnergyConfig,
    SCEnergyModel,
    TFIDFEmbedder,
    _tokenize,
)


# ---------------------------------------------------------------------------
# _tokenize  — REQ-MODEL-031
# ---------------------------------------------------------------------------


def test_tokenize_basic():
    """_tokenize extracts lowercase alphabetic tokens. Spec: REQ-MODEL-031"""
    tokens = _tokenize("Hello World 123!")
    assert tokens == ["hello", "world"]


def test_tokenize_empty():
    """_tokenize returns empty list for numeric-only input. Spec: REQ-MODEL-031"""
    assert _tokenize("123 456") == []


def test_tokenize_mixed():
    """_tokenize handles mixed alphanumeric strings. Spec: REQ-MODEL-031"""
    tokens = _tokenize("abc123def")
    assert tokens == ["abc", "def"]


# ---------------------------------------------------------------------------
# TFIDFEmbedder — REQ-MODEL-031
# ---------------------------------------------------------------------------


def _make_corpus():
    return [
        "Sarah has apples at the market",
        "There are cars in the parking lot",
        "The classroom starts with students present",
        "Mom bakes cookies on the first tray",
    ]


def test_tfidf_fit_builds_vocab():
    """TFIDFEmbedder.fit populates vocab_ and idf_. Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=20)
    embedder.fit(_make_corpus())
    assert len(embedder.vocab_) > 0
    assert embedder.idf_ is not None
    assert len(embedder.idf_) == len(embedder.vocab_)


def test_tfidf_transform_shape():
    """transform returns float32 vector of shape (max_features,). Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=20)
    embedder.fit(_make_corpus())
    vec = embedder.transform("Sarah has apples")
    assert vec.shape == (20,)
    assert vec.dtype == np.float32


def test_tfidf_transform_unit_norm():
    """transform returns L2-normalised vector (norm ≈ 1). Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=20)
    embedder.fit(_make_corpus())
    vec = embedder.transform("Sarah has apples")
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5 or norm < 1e-8  # 0 only if no vocab overlap


def test_tfidf_transform_zero_for_empty():
    """transform returns zero vector for out-of-vocabulary statement. Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=20)
    embedder.fit(_make_corpus())
    vec = embedder.transform("zzzzqqqq aaabbbccc")  # no vocab overlap
    assert np.all(vec == 0.0)


def test_tfidf_transform_before_fit_raises():
    """transform before fit raises RuntimeError. Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=20)
    with pytest.raises(RuntimeError, match="fitted"):
        embedder.transform("test")


def test_tfidf_max_features_respected():
    """vocab_ size is at most max_features. Spec: REQ-MODEL-031"""
    embedder = TFIDFEmbedder(max_features=3)
    embedder.fit(_make_corpus())
    assert len(embedder.vocab_) <= 3


# ---------------------------------------------------------------------------
# SCEnergyModel — REQ-MODEL-031, SCENARIO-MODEL-016
# ---------------------------------------------------------------------------


def _make_trained_model():
    """Helper: small model trained on a tiny corpus for fast tests."""
    corpus = [
        "Sarah has apples at the market",
        "She sells apples to a customer",
        "There are cars in the parking lot",
        "More cars arrive during lunch",
    ]
    embedder = TFIDFEmbedder(max_features=32)
    embedder.fit(corpus)

    import jax.random as jrandom

    config = SCEnergyConfig(embed_dim=32, hidden_dim=16, margin=1.0, learning_rate=0.05)
    model = SCEnergyModel(config, key=jrandom.PRNGKey(0))
    model.embedder = embedder
    return model, embedder


def test_sc_energy_model_init():
    """SCEnergyModel initialises W1/b1/W2/b2 with correct shapes. Spec: REQ-MODEL-031"""
    model, _ = _make_trained_model()
    assert model.W1.shape == (16, 32)
    assert model.b1.shape == (16,)
    assert model.W2.shape == (16,)
    assert model.b2.shape == ()


def test_sc_energy_energy_returns_scalar():
    """energy() returns a Python float scalar. Spec: REQ-MODEL-031"""
    model, _ = _make_trained_model()
    e = model.energy(["Sarah has apples at the market", "She sells apples"])
    assert isinstance(e, float)


def test_sc_energy_energy_no_embedder_raises():
    """energy() without embedder raises RuntimeError. Spec: REQ-MODEL-031"""
    import jax.random as jrandom

    config = SCEnergyConfig(embed_dim=32, hidden_dim=16)
    model = SCEnergyModel(config, key=jrandom.PRNGKey(0))
    with pytest.raises(RuntimeError, match="embedder"):
        model.energy(["some statement"])


def test_sc_energy_permutation_invariant():
    """energy() is permutation-invariant: reordering statements gives same energy. Spec: REQ-MODEL-031"""
    model, _ = _make_trained_model()
    stmts = ["Sarah has apples", "She sells them", "Customer pays"]
    e1 = model.energy(stmts)
    e2 = model.energy(stmts[::-1])
    # Mean pooling is exactly permutation-invariant, so energies must be identical
    assert abs(e1 - e2) < 1e-5


def test_sc_energy_train_returns_loss_history():
    """train() returns list of per-epoch mean losses. Spec: REQ-MODEL-031"""
    model, _ = _make_trained_model()
    coherent = [["Sarah has apples", "She sells them"]]
    contradictory = [["There are cars", "The classroom has students"]]
    history = model.train(coherent, contradictory, n_epochs=3)
    assert len(history) == 3
    assert all(isinstance(v, float) for v in history)


def test_sc_energy_train_unequal_lengths_raises():
    """train() raises ValueError when coherent/contradictory lengths differ. Spec: REQ-MODEL-031"""
    model, _ = _make_trained_model()
    with pytest.raises(ValueError, match="same length"):
        model.train(
            [["a", "b"], ["c", "d"]],
            [["e", "f"]],
            n_epochs=1,
        )


def test_sc_energy_predict_coherent_score_range():
    """predict_coherent_score returns value in [0, 1]. Spec: SCENARIO-MODEL-016"""
    model, _ = _make_trained_model()
    score = model.predict_coherent_score(["Sarah has apples", "She sells them"])
    assert 0.0 <= score <= 1.0


def test_sc_energy_training_separates_energies():
    """After training, coherent sets should have lower mean energy than contradictory sets.
    Uses a larger mini-corpus to give the model a real training signal.
    Spec: SCENARIO-MODEL-016
    """
    corpus_statements = [
        "Sarah has apples at the market",
        "She sells apples to a customer",
        "She has apples remaining after the sale",
        "She buys more apples from the farmer",
        "Now Sarah has apples in total",
        "There are cars in the parking lot",
        "During lunch more cars arrive",
        "The lot now contains cars total",
        "In the afternoon cars leave",
        "At closing time cars remain",
    ]
    embedder = TFIDFEmbedder(max_features=64)
    embedder.fit(corpus_statements)

    import jax.random as jrandom

    config = SCEnergyConfig(embed_dim=64, hidden_dim=32, margin=1.0, learning_rate=0.05)
    model = SCEnergyModel(config, key=jrandom.PRNGKey(1))
    model.embedder = embedder

    coherent = [
        [
            "Sarah has apples at the market",
            "She sells apples to a customer",
            "She has apples remaining",
        ],
        [
            "Sarah has apples at the market",
            "She buys more apples from the farmer",
            "Now Sarah has apples in total",
        ],
        [
            "She sells apples to a customer",
            "She has apples remaining after the sale",
            "Now Sarah has apples",
        ],
    ]
    contradictory = [
        [
            "Sarah has apples at the market",
            "During lunch more cars arrive",
            "At closing time cars remain",
        ],
        [
            "She sells apples to a customer",
            "The lot now contains cars total",
            "In the afternoon cars leave",
        ],
        [
            "She buys more apples from the farmer",
            "There are cars in the parking lot",
            "Cars leave in afternoon",
        ],
    ]

    model.train(coherent, contradictory, n_epochs=100)

    mean_coh_energy = np.mean([model.energy(s) for s in coherent])
    mean_con_energy = np.mean([model.energy(s) for s in contradictory])
    # Coherent sets should have strictly lower energy after training
    assert mean_coh_energy < mean_con_energy, (
        f"Training failed to separate energies: coherent={mean_coh_energy:.4f}, "
        f"contradictory={mean_con_energy:.4f}"
    )


# ---------------------------------------------------------------------------
# Experiment 944 helper functions — REQ-MODEL-031, SCENARIO-MODEL-016
# ---------------------------------------------------------------------------


def test_generate_dataset_sizes():
    """_generate_dataset returns correct number of pairs. Spec: REQ-MODEL-031"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _generate_dataset

    rng = random.Random(0)
    coh, con = _generate_dataset(10, rng)
    assert len(coh) == 10
    assert len(con) == 10


def test_generate_dataset_coherent_step_count():
    """Coherent sets have 3-5 steps. Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _generate_dataset

    rng = random.Random(0)
    coh, _ = _generate_dataset(50, rng)
    for s in coh:
        assert 3 <= len(s) <= 5, f"Coherent set has {len(s)} steps, expected 3-5"


def test_generate_dataset_contradictory_step_count():
    """Contradictory sets have exactly 4 steps (2 from each problem). Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _generate_dataset

    rng = random.Random(0)
    _, con = _generate_dataset(50, rng)
    for s in con:
        assert len(s) == 4, f"Contradictory set has {len(s)} steps, expected 4"


def test_compute_auroc_perfect():
    """AUROC = 1.0 when scores perfectly separate classes. Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _compute_auroc

    y_true = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    auroc = _compute_auroc(y_true, scores)
    assert abs(auroc - 1.0) < 1e-6


def test_compute_auroc_random():
    """AUROC in [0,1] when scores are constant (tied scores give non-deterministic AUROC). Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _compute_auroc

    y_true = [1, 0, 1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    auroc = _compute_auroc(y_true, scores)
    assert 0.0 <= auroc <= 1.0


def test_compute_auroc_degenerate_all_positive():
    """AUROC = 0.5 when all labels are positive (no negatives to rank against). Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _compute_auroc

    y_true = [1, 1, 1]
    scores = [0.9, 0.5, 0.1]
    auroc = _compute_auroc(y_true, scores)
    assert auroc == 0.5


def test_rule_based_predict_returns_one():
    """Rule-based baseline always returns 1.0 (predicts coherent). Spec: SCENARIO-MODEL-016"""
    from scripts.experiment_944_sc_energy_set_consistency_v2 import _rule_based_predict

    result = _rule_based_predict(["Sarah has apples", "There are cars"])
    assert result == 1.0
