"""Tests for multi-layer hallucination probing.

**Researcher summary:**
    Verifies the multi-layer probing module that trains per-layer EBM probes
    to find which transformer layer best separates correct from hallucinated
    activations. Tests cover probe training, per-layer evaluation, best-layer
    selection, and results formatting.

**Detailed explanation for engineers:**
    Tests use synthetic activations with different separability at different
    layers to verify that the probing correctly identifies the most separable
    layer. No real models are loaded — all activations are pre-generated.

Spec coverage: REQ-INFER-016
"""

from __future__ import annotations

import numpy as np
from carnot.embeddings.layer_probing import (
    LayerProbeResult,
    MultiLayerProbeResults,
    probe_all_layers,
    train_layer_probe,
)

# --- LayerProbeResult tests ---


def test_layer_probe_result_fields() -> None:
    """LayerProbeResult stores all probe metrics.

    Spec: REQ-INFER-016
    """
    result = LayerProbeResult(
        layer_index=12,
        train_accuracy=0.85,
        test_accuracy=0.78,
        energy_gap=1.5,
        n_train=800,
        n_test=200,
    )
    assert result.layer_index == 12
    assert result.train_accuracy == 0.85
    assert result.test_accuracy == 0.78
    assert result.energy_gap == 1.5


def test_multi_layer_results_summary() -> None:
    """Summary output includes all layers and marks the best.

    Spec: REQ-INFER-016
    """
    results = MultiLayerProbeResults(
        layer_results=[
            LayerProbeResult(0, 0.60, 0.55, 0.1, 100, 25),
            LayerProbeResult(1, 0.75, 0.70, 0.5, 100, 25),
            LayerProbeResult(2, 0.80, 0.65, 0.3, 100, 25),
        ],
        best_layer=1,
        best_test_accuracy=0.70,
        model_name="test-model",
    )
    summary = results.summary()
    assert "test-model" in summary
    assert "BEST" in summary
    assert "70.0%" in summary


# --- train_layer_probe tests ---


def test_train_probe_separable_data() -> None:
    """Probe achieves >60% on well-separated clusters.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)
    correct = rng.standard_normal((100, 16)).astype(np.float32) + 3.0
    wrong = rng.standard_normal((100, 16)).astype(np.float32) - 3.0

    train_acc, test_acc, gap = train_layer_probe(
        correct, wrong, hidden_dim=16, n_epochs=100, lr=0.01,
    )

    assert train_acc > 0.6
    assert test_acc > 0.55
    assert gap > 0


def test_train_probe_random_data_near_chance() -> None:
    """Probe on random (unseparable) data stays near 50%.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)
    correct = rng.standard_normal((100, 16)).astype(np.float32)
    wrong = rng.standard_normal((100, 16)).astype(np.float32)

    train_acc, test_acc, gap = train_layer_probe(
        correct, wrong, hidden_dim=16, n_epochs=50, lr=0.005,
    )

    # Should be near chance (40-60%)
    assert 0.35 < test_acc < 0.65


def test_train_probe_insufficient_data() -> None:
    """Probe with < 10 samples returns 50% / 0.0 gap.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)
    correct = rng.standard_normal((5, 16)).astype(np.float32)
    wrong = rng.standard_normal((5, 16)).astype(np.float32)

    train_acc, test_acc, gap = train_layer_probe(
        correct, wrong, hidden_dim=16,
    )

    assert train_acc == 0.5
    assert test_acc == 0.5
    assert gap == 0.0


# --- probe_all_layers tests ---


def test_probe_all_layers_finds_best() -> None:
    """Probing identifies the layer with most separable activations.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)

    # Layer 0: not separable (random)
    # Layer 1: well separable (offset=3)
    # Layer 2: slightly separable (offset=0.5)
    correct_acts = {
        0: rng.standard_normal((80, 16)).astype(np.float32),
        1: rng.standard_normal((80, 16)).astype(np.float32) + 3.0,
        2: rng.standard_normal((80, 16)).astype(np.float32) + 0.5,
    }
    wrong_acts = {
        0: rng.standard_normal((80, 16)).astype(np.float32),
        1: rng.standard_normal((80, 16)).astype(np.float32) - 3.0,
        2: rng.standard_normal((80, 16)).astype(np.float32) - 0.5,
    }

    results = probe_all_layers(
        correct_acts, wrong_acts,
        hidden_dim=16, n_epochs=100, lr=0.01,
        model_name="test-model",
    )

    assert len(results.layer_results) == 3
    # Layer 1 should be best (highest separation), layer 0 near chance
    assert results.best_layer == 1
    assert results.best_test_accuracy > 0.6
    # Layer 0 should be near chance
    layer0_acc = next(r for r in results.layer_results if r.layer_index == 0).test_accuracy
    assert layer0_acc < results.best_test_accuracy


def test_probe_subset_of_layers() -> None:
    """Can probe only a subset of layers.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)
    correct_acts = {
        0: rng.standard_normal((50, 8)).astype(np.float32) + 2.0,
        1: rng.standard_normal((50, 8)).astype(np.float32) + 2.0,
        2: rng.standard_normal((50, 8)).astype(np.float32) + 2.0,
    }
    wrong_acts = {
        0: rng.standard_normal((50, 8)).astype(np.float32) - 2.0,
        1: rng.standard_normal((50, 8)).astype(np.float32) - 2.0,
        2: rng.standard_normal((50, 8)).astype(np.float32) - 2.0,
    }

    results = probe_all_layers(
        correct_acts, wrong_acts,
        hidden_dim=8, n_epochs=50,
        layers=[0, 2],  # Skip layer 1
    )

    assert len(results.layer_results) == 2
    assert all(r.layer_index in [0, 2] for r in results.layer_results)


def test_probe_empty_layers() -> None:
    """Probing with no matching layers returns default results.

    Spec: REQ-INFER-016
    """
    results = probe_all_layers(
        {}, {},
        hidden_dim=8,
    )

    assert len(results.layer_results) == 0
    assert results.best_test_accuracy == 0.5


def test_probe_mismatched_layers_skipped() -> None:
    """Layers present in correct but not wrong are skipped.

    Spec: REQ-INFER-016
    """
    rng = np.random.default_rng(42)
    correct_acts = {
        0: rng.standard_normal((50, 8)).astype(np.float32),
        1: rng.standard_normal((50, 8)).astype(np.float32),
    }
    wrong_acts = {
        0: rng.standard_normal((50, 8)).astype(np.float32),
        # Layer 1 missing from wrong
    }

    results = probe_all_layers(correct_acts, wrong_acts, hidden_dim=8, n_epochs=50)
    assert len(results.layer_results) == 1
    assert results.layer_results[0].layer_index == 0


# --- Package export tests ---


def test_exports_from_embeddings_package() -> None:
    """All new symbols are exported from carnot.embeddings.

    Spec: REQ-INFER-016
    """
    from carnot.embeddings import (
        LayerProbeResult,
        MultiLayerProbeResults,
        probe_all_layers,
        train_layer_probe,
    )
    assert LayerProbeResult is not None
    assert MultiLayerProbeResults is not None
    assert callable(probe_all_layers)
    assert callable(train_layer_probe)
