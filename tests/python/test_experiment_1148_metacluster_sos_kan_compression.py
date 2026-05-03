"""Tests for Exp 1148 MetaCluster-style SOSKANEnergyV3 compression.

Spec: REQ-KAN-1148, SCENARIO-KAN-1148
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.models.sos_kan import SOSKANEnergyV3  # noqa: E402
from carnot.models.sos_kan_metacluster import (  # noqa: E402
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    classify_metacluster_verdict,
    collect_sos_kan_coefficient_vectors,
    compress_sos_kan_v3,
    inspect_sos_kan_v3_coefficients,
    reconstruct_sos_kan_v3,
)


def _toy_model() -> SOSKANEnergyV3:
    model = SOSKANEnergyV3(n_splines=4, rank=2, n_features=3, hidden_dim=4, seed=1148)
    rng = np.random.default_rng(1148)
    model.W1[:] = rng.normal(0.0, 0.2, model.W1.shape)
    model.b1[:] = rng.normal(0.0, 0.05, model.b1.shape)
    model.W2[:] = rng.normal(0.0, 0.2, model.W2.shape)
    model.b2[:] = rng.normal(0.0, 0.05, model.b2.shape)
    model.c[:] = rng.normal(0.0, 0.01, model.c.shape)
    return model


def test_inspect_exp1128_sos_kan_structure() -> None:
    """REQ-KAN-1148: inspect basis and coefficient-vector structure."""
    model = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)

    info = inspect_sos_kan_v3_coefficients(model)

    assert info["n_kan_basis_functions"] == 8
    assert info["coefficients_per_spline"] == 4
    assert info["hidden_dim"] == 16
    assert info["trainable_parameter_count"] == 1699
    assert info["coefficient_vector_count"] == 227
    assert info["max_coefficient_vector_width"] == 16


def test_collect_vectors_pads_all_parameter_blocks() -> None:
    """REQ-KAN-1148: collect coefficient vectors across all learned blocks."""
    model = SOSKANEnergyV3(n_splines=8, rank=4, n_features=3, hidden_dim=16, seed=1121)

    matrix, blocks = collect_sos_kan_coefficient_vectors(model)

    assert matrix.shape == (227, 16)
    assert [block.name for block in blocks] == ["W1", "b1", "W2", "b2", "c"]
    assert [block.vector_width for block in blocks] == [3, 1, 16, 1, 1]
    assert sum(block.vector_count for block in blocks) == matrix.shape[0]


def test_compress_and_reconstruct_preserves_shapes_and_finite_energies() -> None:
    """SCENARIO-KAN-1148: compressed codebook reconstructs a usable model."""
    model = _toy_model()
    X = np.random.default_rng(99).uniform(-1.0, 1.0, (12, model.n_features))

    payload = compress_sos_kan_v3(model, n_centroids=8, random_state=1148)
    restored = reconstruct_sos_kan_v3(payload)

    assert payload.n_centroids == 8
    assert restored.W1.shape == model.W1.shape
    assert restored.b1.shape == model.b1.shape
    assert restored.W2.shape == model.W2.shape
    assert restored.b2.shape == model.b2.shape
    assert restored.c.shape == model.c.shape
    assert payload.size_compressed_bytes < payload.size_original_bytes
    assert payload.size_reduction_factor == pytest.approx(
        payload.size_original_bytes / payload.size_compressed_bytes
    )

    energies = np.array([restored.energy(row) for row in X])
    assert energies.shape == (len(X),)
    assert np.isfinite(energies).all()


def test_compression_uses_integer_indices_and_float32_centroids() -> None:
    """REQ-KAN-1148: store centroids plus compact integer indices."""
    payload = compress_sos_kan_v3(_toy_model(), n_centroids=8, random_state=1148)

    assert payload.centroids.dtype == np.float32
    assert payload.indices.dtype == np.uint8
    assert payload.centroids.shape[0] == 8
    assert payload.indices.ndim == 1
    assert payload.indices.size == payload.vector_count
    assert len(payload.packed_indices) < payload.indices.nbytes


def test_partial_compression_keeps_sensitive_small_blocks_exact() -> None:
    """REQ-KAN-1148: codebook compression can target dominant coefficient blocks."""
    model = _toy_model()

    payload = compress_sos_kan_v3(
        model,
        n_centroids=8,
        random_state=1148,
        block_names=("W2", "b2"),
    )
    restored = reconstruct_sos_kan_v3(payload)

    assert payload.vector_count == model.W2.shape[0] + model.b2.shape[0]
    assert set(payload.uncompressed_arrays) == {"W1", "b1", "c"}
    np.testing.assert_allclose(restored.W1, model.W1)
    np.testing.assert_allclose(restored.b1, model.b1)
    np.testing.assert_allclose(restored.c, model.c)
    assert payload.size_compressed_bytes == (
        payload.centroids.nbytes
        + len(payload.packed_indices)
        + sum(array.nbytes for array in payload.uncompressed_arrays.values())
    )


def test_compression_rejects_more_centroids_than_vectors() -> None:
    """REQ-KAN-1148: invalid codebook requests fail explicitly."""
    model = SOSKANEnergyV3(n_splines=2, rank=1, n_features=1, hidden_dim=1, seed=1)

    with pytest.raises(ValueError, match="n_centroids"):
        compress_sos_kan_v3(model, n_centroids=16, random_state=1)


@pytest.mark.parametrize(
    ("auroc_compressed", "size_reduction_factor", "expected"),
    [
        (0.9802, 5.1, "compressed_within_02_auroc_5x_smaller"),
        (0.9600, 6.0, "compressed_auroc_degraded"),
        (0.9802, 4.9, "compression_ratio_below_5x"),
        (0.9902, 1.0, "checkpoint_not_found"),
    ],
)
def test_verdict_classifier(
    auroc_compressed: float, size_reduction_factor: float, expected: str
) -> None:
    """REQ-KAN-1148: honest verdict follows AUROC, ratio, and checkpoint gates."""
    verdict = classify_metacluster_verdict(
        checkpoint_found=expected != "checkpoint_not_found",
        auroc_original=0.9902,
        auroc_compressed=auroc_compressed,
        size_reduction_factor=size_reduction_factor,
    )

    assert verdict == expected


def test_artifact_required_fields_schema() -> None:
    """REQ-KAN-1148: artifact carries the required result fields."""
    artifact = {
        "sos_kan_compressed": True,
        "auroc_original": 0.9902,
        "auroc_compressed": 0.9802,
        "auroc_drop": 0.01,
        "auroc_drop_within_02": True,
        "size_original_bytes": 1000,
        "size_compressed_bytes": 100,
        "size_reduction_factor": 10.0,
        "n_centroids": 32,
        "energy_correlation": 0.99,
        "honest_verdict": "compressed_within_02_auroc_5x_smaller",
    }

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert not artifact_has_required_fields({"sos_kan_compressed": True})


def test_existing_exp1128_artifact_records_reference_auroc() -> None:
    """REQ-KAN-1148: exp1148 is anchored to the exp1128 AUROC reference."""
    path = _PROJECT_ROOT / "results" / "experiment_1128_sos_kan_root_cause_k5_fix.json"
    payload = json.loads(path.read_text())

    assert payload["sos_kan_individual_auroc_after"] == 0.9902
