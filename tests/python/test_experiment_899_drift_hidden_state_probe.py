"""Tests for Experiment 899: DRIFTProbe hidden-state representational drift.

These tests verify the DRIFTProbe implementation (python/carnot/probes/drift_probe.py)
and its wiring into ThreeTierPipeline.verify_extended().

All tests run without a real LLM — the model load path is designed to fail gracefully
when transformers/weights are absent (CI-safe), returning zero signatures of the correct
shape.  The probe logic (linear classifier) is tested with synthetic drift signatures
injected directly rather than going through the full forward pass.

Spec traces: REQ-TIER0-009, SCENARIO-TIER0-009
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def drift_probe_default():
    """DRIFTProbe with default probe_layers=[4, 8, 12, 16]."""
    from carnot.probes.drift_probe import DRIFTProbe

    return DRIFTProbe(model_name="nonexistent/model", probe_layers=[4, 8, 12, 16])


@pytest.fixture
def fitted_drift_probe():
    """DRIFTProbe with linear_probe fitted on synthetic zero signatures.

    Because the model will not load (nonexistent ID), extract_drift_signature()
    returns zeros.  We fit the probe directly on synthetic pairs so the linear
    classifier is tested even when no LLM is present.
    """
    from carnot.probes.drift_probe import DRIFTProbe

    probe = DRIFTProbe(model_name="nonexistent/model", probe_layers=[4, 8, 12, 16])
    # Inject synthetic pairs with a mix of labels.
    # The probe's fit() calls extract_drift_signature() which returns zeros
    # when model load fails.  We then override linear_probe with one fitted
    # on hand-crafted signatures so we can test predict_proba deterministically.
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    # 20 hallucinating (label 1): higher drift values
    # 20 truthful (label 0): lower drift values
    X_hallu = rng.uniform(0.5, 1.5, (20, 3)).astype(np.float32)
    X_truth = rng.uniform(0.0, 0.3, (20, 3)).astype(np.float32)
    X = np.vstack([X_hallu, X_truth])
    y = np.array([1] * 20 + [0] * 20)
    probe.linear_probe = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
    return probe


# ---------------------------------------------------------------------------
# REQ-TIER0-009-1: extract_drift_signature returns correct shape
# ---------------------------------------------------------------------------


def test_extract_drift_signature_shape_default_layers(drift_probe_default):
    """extract_drift_signature returns shape (3,) for probe_layers=[4,8,12,16].

    Spec: REQ-TIER0-009-1, SCENARIO-TIER0-009
    """
    # Model won't load (nonexistent ID) → CI-safe zero signature.
    sig = drift_probe_default.extract_drift_signature("Hello world")
    assert sig.shape == (3,), f"Expected shape (3,), got {sig.shape}"


def test_extract_drift_signature_shape_custom_layers():
    """extract_drift_signature shape = len(probe_layers)-1 for any probe_layers.

    Spec: REQ-TIER0-009-1
    """
    from carnot.probes.drift_probe import DRIFTProbe

    probe = DRIFTProbe(model_name="nonexistent/model", probe_layers=[2, 4, 6, 8, 10])
    sig = probe.extract_drift_signature("test")
    assert sig.shape == (4,), f"Expected shape (4,), got {sig.shape}"


def test_extract_drift_signature_values_in_valid_range(drift_probe_default):
    """extract_drift_signature values lie in [0, 2] (drift = 1 - cossim, clamped).

    Spec: REQ-TIER0-009-1
    """
    sig = drift_probe_default.extract_drift_signature("Some response text")
    assert np.all(sig >= 0.0), f"Negative drift values found: {sig}"
    assert np.all(sig <= 2.0), f"Drift values exceed 2.0: {sig}"


def test_extract_drift_signature_returns_zeros_when_model_absent(drift_probe_default):
    """extract_drift_signature returns zeros when model load fails (CI safety).

    Spec: REQ-TIER0-009-6
    """
    sig = drift_probe_default.extract_drift_signature("test text")
    assert sig.dtype in (np.float32, np.float64)
    # Without a real model, all values should be 0.0 (safe fallback).
    assert np.allclose(sig, 0.0)


# ---------------------------------------------------------------------------
# REQ-TIER0-009-2: fit() trains probe
# ---------------------------------------------------------------------------


def test_fit_trains_linear_probe(drift_probe_default):
    """fit() sets linear_probe to a fitted LogisticRegression instance.

    Spec: REQ-TIER0-009-2, SCENARIO-TIER0-009
    """
    pairs = [{"text": f"text_{i}", "label": i % 2} for i in range(10)]
    drift_probe_default.fit(pairs)
    assert drift_probe_default.linear_probe is not None


def test_fit_probe_has_coef(drift_probe_default):
    """After fit(), linear_probe.coef_ has shape (1, n_features).

    Spec: REQ-TIER0-009-2
    """
    pairs = [{"text": f"text_{i}", "label": i % 2} for i in range(20)]
    drift_probe_default.fit(pairs)
    coef = drift_probe_default.linear_probe.coef_
    n_pairs = len(drift_probe_default.probe_layers) - 1
    assert coef.shape == (1, n_pairs), f"Expected coef shape (1, {n_pairs}), got {coef.shape}"


# ---------------------------------------------------------------------------
# REQ-TIER0-009-3: predict_proba returns float in [0, 1]
# ---------------------------------------------------------------------------


def test_predict_proba_range(fitted_drift_probe):
    """predict_proba returns a float in [0, 1].

    Spec: REQ-TIER0-009-3, SCENARIO-TIER0-009
    """
    prob = fitted_drift_probe.predict_proba("Some response")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0, f"predict_proba out of range: {prob}"


def test_predict_proba_before_fit_returns_half():
    """predict_proba returns 0.5 when probe has not been fitted.

    Spec: REQ-TIER0-009-3
    """
    from carnot.probes.drift_probe import DRIFTProbe

    probe = DRIFTProbe(model_name="nonexistent/model")
    assert probe.predict_proba("test") == 0.5


# ---------------------------------------------------------------------------
# REQ-TIER0-009-4: is_representationally_drifted threshold logic
# ---------------------------------------------------------------------------


def test_is_representationally_drifted_high_prob(fitted_drift_probe):
    """is_representationally_drifted returns True when proba > threshold.

    Spec: REQ-TIER0-009-4
    """
    # Patch predict_proba to return a fixed high value.
    fitted_drift_probe.predict_proba = lambda _text: 0.8
    assert fitted_drift_probe.is_representationally_drifted("text", threshold=0.6) is True


def test_is_representationally_drifted_low_prob(fitted_drift_probe):
    """is_representationally_drifted returns False when proba <= threshold.

    Spec: REQ-TIER0-009-4
    """
    fitted_drift_probe.predict_proba = lambda _text: 0.4
    assert fitted_drift_probe.is_representationally_drifted("text", threshold=0.6) is False


# ---------------------------------------------------------------------------
# Cosine similarity helper tests
# ---------------------------------------------------------------------------


def test_cosine_similarity_zero_vector():
    """_cosine_similarity_vectors returns 1.0 for zero vectors (no drift).

    This documents the invariant that zero-norm vectors produce no drift contribution.
    Spec: REQ-TIER0-009-1
    """
    from carnot.probes.drift_probe import _cosine_similarity_vectors

    a = np.zeros(4, dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert _cosine_similarity_vectors(a, b) == 1.0


def test_cosine_similarity_orthogonal():
    """_cosine_similarity_vectors returns 0.0 for orthogonal vectors.

    Spec: REQ-TIER0-009-1
    """
    from carnot.probes.drift_probe import _cosine_similarity_vectors

    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    val = _cosine_similarity_vectors(a, b)
    assert abs(val) < 1e-6, f"Expected ~0.0, got {val}"


def test_cosine_similarity_identical():
    """_cosine_similarity_vectors returns 1.0 for identical vectors (zero drift).

    Spec: REQ-TIER0-009-1
    """
    from carnot.probes.drift_probe import _cosine_similarity_vectors

    a = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    assert abs(_cosine_similarity_vectors(a, a) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# REQ-TIER0-009-5: ThreeTierPipeline wiring
# ---------------------------------------------------------------------------


def test_wire_drift_probe_sets_attribute():
    """wire_drift_probe() attaches the probe to the pipeline instance.

    Spec: REQ-TIER0-009-5, SCENARIO-TIER0-009
    """
    from unittest.mock import MagicMock

    from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
    from carnot.probes.drift_probe import DRIFTProbe

    pipeline = ThreeTierPipeline(
        sink_probe=MagicMock(),
        eorm_model=MagicMock(),
        ising_pipeline=lambda r, q: (True, 0.1),
    )
    probe = DRIFTProbe(model_name="nonexistent/model")
    pipeline.wire_drift_probe(probe)
    assert pipeline.drift_probe is probe


def test_verify_extended_includes_drift_field_when_probe_wired():
    """verify_extended() includes is_representationally_drifted when drift probe is wired.

    Spec: REQ-TIER0-009-5, SCENARIO-TIER0-009
    """
    import jax.numpy as jnp
    from unittest.mock import MagicMock

    from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
    from carnot.probes.drift_probe import DRIFTProbe

    # Build a minimal working pipeline.
    sink_probe = MagicMock()
    sink_probe.score.return_value = MagicMock(mean_sink_score=0.1)

    eorm_model = MagicMock()
    eorm_model.energy.return_value = 0.9  # above threshold → goes to Ising

    pipeline = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_model,
        ising_pipeline=lambda r, q: (True, 0.1),
        sink_threshold=0.3,
        eorm_threshold=0.5,
    )

    # Wire a drift probe with predict_proba always returning 0.3 (below threshold).
    probe = DRIFTProbe(model_name="nonexistent/model")
    probe.predict_proba = lambda _text: 0.3  # type: ignore[method-assign]
    pipeline.wire_drift_probe(probe)

    result = pipeline.verify_extended("The answer is 42.", question="What is 6*7?")
    assert "is_representationally_drifted" in result
    assert result["is_representationally_drifted"] is False


def test_verify_extended_drift_false_when_probe_not_wired():
    """verify_extended() includes is_representationally_drifted=False when no drift probe.

    Spec: REQ-TIER0-009-5
    """
    from unittest.mock import MagicMock

    from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline

    sink_probe = MagicMock()
    sink_probe.score.return_value = MagicMock(mean_sink_score=0.1)

    eorm_model = MagicMock()
    eorm_model.energy.return_value = 0.9

    pipeline = ThreeTierPipeline(
        sink_probe=sink_probe,
        eorm_model=eorm_model,
        ising_pipeline=lambda r, q: (True, 0.1),
    )

    result = pipeline.verify_extended("Some text.")
    # Field should be present with default False even without probe wired.
    assert "is_representationally_drifted" in result
    assert result["is_representationally_drifted"] is False
