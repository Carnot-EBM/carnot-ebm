"""Tests for Exp 763: Dual-Pathway MoP Probe (arXiv 2601.07422).

Spec: REQ-PROBE-010, REQ-PROBE-011, SCENARIO-PROBE-020, SCENARIO-PROBE-021
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.pipeline.dual_pathway_probe import (
    AnswerAnchoredProbe,
    GateNetwork,
    MixtureOfProbes,
    QuestionAnchoredProbe,
    _MLP,
    _TFIDFEmbedder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _minimal_labeled_steps(n: int = 12) -> list[dict]:
    """Return n minimal labeled steps for unit tests."""
    steps = []
    for i in range(n):
        label = "incorrect" if i % 2 == 0 else "correct"
        steps.append(
            {
                "question_context": f"What is {i} plus {i}?",
                "step_text": f"Step result: {i * 2}." if i % 2 != 0 else f"Step result: {i * 3}.",
                "label": label,
            }
        )
    return steps


# ---------------------------------------------------------------------------
# REQ-PROBE-010: QuestionAnchoredProbe forward pass
# ---------------------------------------------------------------------------


def test_question_anchored_probe_forward_returns_scalar() -> None:
    """QuestionAnchoredProbe.forward() must return a scalar float in [0, 1].

    Spec: REQ-PROBE-010, REQ-PROBE-010-1
    """
    steps = _minimal_labeled_steps(10)
    q_probe = QuestionAnchoredProbe(hidden_dim=128, output_dim=1)
    a_probe = AnswerAnchoredProbe(hidden_dim=128, output_dim=1)
    gate = GateNetwork(input_dim=2, output_dim=1)
    mop = MixtureOfProbes(q_probe, a_probe, gate)
    mop.train(steps, n_epochs=1, lr=1e-3)

    # After training, the question probe's _mlp must be set and return a scalar.
    dummy_emb = np.zeros(128, dtype=np.float32)
    score = q_probe.forward(dummy_emb)

    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Expected score in [0, 1], got {score}"


# ---------------------------------------------------------------------------
# REQ-PROBE-010: GateNetwork combines two scalar inputs
# ---------------------------------------------------------------------------


def test_gate_network_combines_two_scalars() -> None:
    """GateNetwork.forward() must accept two scalar floats and return a float in [0, 1].

    Spec: REQ-PROBE-010, REQ-PROBE-010-3
    """
    steps = _minimal_labeled_steps(10)
    q_probe = QuestionAnchoredProbe(hidden_dim=128, output_dim=1)
    a_probe = AnswerAnchoredProbe(hidden_dim=128, output_dim=1)
    gate = GateNetwork(input_dim=2, output_dim=1)
    mop = MixtureOfProbes(q_probe, a_probe, gate)
    mop.train(steps, n_epochs=1, lr=1e-3)

    result = gate.forward(0.3, 0.7)

    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Expected in [0, 1], got {result}"


# ---------------------------------------------------------------------------
# REQ-PROBE-010, SCENARIO-PROBE-020: MixtureOfProbes.train() runs for 10 epochs
# ---------------------------------------------------------------------------


def test_mixture_of_probes_train_runs_without_error() -> None:
    """MixtureOfProbes.train() must complete for 10 epochs and return a loss dict.

    Spec: REQ-PROBE-010, SCENARIO-PROBE-020
    """
    steps = _minimal_labeled_steps(12)
    q_probe = QuestionAnchoredProbe(hidden_dim=128, output_dim=1)
    a_probe = AnswerAnchoredProbe(hidden_dim=128, output_dim=1)
    gate = GateNetwork(input_dim=2, output_dim=1)
    mop = MixtureOfProbes(q_probe, a_probe, gate)

    result = mop.train(steps, n_epochs=10, lr=1e-3)

    assert "final_loss" in result, "train() must return dict with 'final_loss'"
    assert isinstance(result["final_loss"], float), "final_loss must be float"
    assert result["final_loss"] >= 0.0, "Loss must be non-negative"
    assert result["n_train"] == 12, f"Expected n_train=12, got {result['n_train']}"


# ---------------------------------------------------------------------------
# REQ-PROBE-011, SCENARIO-PROBE-021: AUROC computation
# ---------------------------------------------------------------------------


def test_auroc_computation_from_predictions_and_labels() -> None:
    """MixtureOfProbes.evaluate_auroc() must return a float in [0, 1].

    Spec: REQ-PROBE-011, SCENARIO-PROBE-021
    """
    # Perfect classifier: all positives score higher than all negatives.
    perfect_scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    perfect_labels = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    auroc = MixtureOfProbes.evaluate_auroc(perfect_scores, perfect_labels)
    assert auroc == pytest.approx(1.0), f"Perfect classifier AUROC must be 1.0, got {auroc}"

    # Random classifier: interspersed scores.
    random_scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    random_labels = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    auroc_rand = MixtureOfProbes.evaluate_auroc(random_scores, random_labels)
    assert auroc_rand == pytest.approx(0.5), f"Tied classifier AUROC must be 0.5, got {auroc_rand}"

    # Degenerate: only one class — should return 0.5 (no discrimination possible).
    auroc_degen = MixtureOfProbes.evaluate_auroc([0.9, 0.8], [1.0, 1.0])
    assert auroc_degen == pytest.approx(0.5), f"Single-class AUROC must be 0.5, got {auroc_degen}"


# ---------------------------------------------------------------------------
# REQ-PROBE-010: _TFIDFEmbedder produces consistent output dim
# ---------------------------------------------------------------------------


def test_tfidf_embedder_output_dim() -> None:
    """_TFIDFEmbedder must return a vector of shape (max_features,) after fit.

    Spec: REQ-PROBE-010 (embedding proxy used in both sub-probes)
    """
    # Use a corpus that has >= 16 unique words to fill the vocabulary.
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "one two three four five six seven eight nine ten",
        "alpha beta gamma delta epsilon zeta eta theta",
    ]
    embedder = _TFIDFEmbedder(max_features=16)
    embedder.fit(corpus)
    vec = embedder.embed("the fox")
    assert vec.shape == (16,), f"Expected shape (16,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"


# ---------------------------------------------------------------------------
# REQ-PROBE-010: _MLP forward pass returns scalar in [0, 1]
# ---------------------------------------------------------------------------


def test_mlp_forward_returns_scalar_in_unit_interval() -> None:
    """_MLP.forward() must return a Python float in [0.0, 1.0].

    Spec: REQ-PROBE-010 (used by all three sub-components)
    """
    rng = np.random.default_rng(0)
    w1 = rng.standard_normal((8, 4)).astype(np.float32)
    b1 = rng.standard_normal(8).astype(np.float32)
    w2 = rng.standard_normal((1, 8)).astype(np.float32)
    b2 = rng.standard_normal(1).astype(np.float32)
    mlp = _MLP(w1, b1, w2, b2)

    x = rng.standard_normal(4).astype(np.float32)
    result = mlp.forward(x)

    assert isinstance(result, float), f"Expected float, got {type(result)}"
    assert 0.0 <= result <= 1.0, f"Expected in [0, 1], got {result}"


# ---------------------------------------------------------------------------
# REQ-PROBE-011: predict() returns a float after training
# ---------------------------------------------------------------------------


def test_mixture_of_probes_predict_returns_float() -> None:
    """MixtureOfProbes.predict() must return a float in [0, 1] after training.

    Spec: REQ-PROBE-011
    """
    steps = _minimal_labeled_steps(10)
    mop = MixtureOfProbes(
        QuestionAnchoredProbe(hidden_dim=128),
        AnswerAnchoredProbe(hidden_dim=128),
        GateNetwork(input_dim=2),
    )
    mop.train(steps, n_epochs=5, lr=1e-3)

    score = mop.predict("What is 3 + 4?", "Step: 3 + 4 = 7.")
    assert isinstance(score, float), f"Expected float, got {type(score)}"
    assert 0.0 <= score <= 1.0, f"Expected in [0, 1], got {score}"


# ---------------------------------------------------------------------------
# REQ-PROBE-010: untrained probe raises RuntimeError
# ---------------------------------------------------------------------------


def test_untrained_probe_raises_runtime_error() -> None:
    """Calling forward() on an untrained probe must raise RuntimeError.

    Spec: REQ-PROBE-010-1
    """
    q_probe = QuestionAnchoredProbe(hidden_dim=128)
    with pytest.raises(RuntimeError, match="train"):
        q_probe.forward(np.zeros(128, dtype=np.float32))


def test_untrained_gate_raises_runtime_error() -> None:
    """Calling forward() on an untrained gate must raise RuntimeError.

    Spec: REQ-PROBE-010-3
    """
    gate = GateNetwork(input_dim=2)
    with pytest.raises(RuntimeError, match="train"):
        gate.forward(0.5, 0.5)


def test_predict_before_train_raises_runtime_error() -> None:
    """MixtureOfProbes.predict() before train() must raise RuntimeError.

    Spec: REQ-PROBE-011
    """
    mop = MixtureOfProbes(
        QuestionAnchoredProbe(hidden_dim=128),
        AnswerAnchoredProbe(hidden_dim=128),
        GateNetwork(input_dim=2),
    )
    with pytest.raises(RuntimeError, match="train"):
        mop.predict("q", "a")
