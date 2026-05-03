"""Tests for Exp 1199 KANtize-style SOSKANEnergyV3 quantization.

Spec: REQ-KAN-1199, SCENARIO-KAN-1199
"""

from __future__ import annotations

import math
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.models"]:
    if _pkg not in sys.modules:
        _module = types.ModuleType(_pkg)
        _module.__path__ = [str(_PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _module.__package__ = _pkg
        sys.modules[_pkg] = _module

from carnot.models.sos_kan import SOSKANEnergyV3  # noqa: E402
from carnot.models.sos_kan_quantization import (  # noqa: E402
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    classify_kantize_verdict,
    clone_sos_kan_v3,
    endpoint_row_mask,
    estimate_sos_kan_v3_size_bytes,
    evaluate_sos_kan_v3_auroc,
    export_quantized_sos_kan_v3_safetensors,
    load_sos_kan_v3_npz,
    measure_per_sample_latency_ms,
    quantize_sos_kan_v3_spline_weights,
    quantize_to_grid,
)
from scripts import experiment_1199_kantize_soskan_4bit_quantization as exp1199  # noqa: E402


def _toy_model() -> SOSKANEnergyV3:
    """Return a small deterministic model with non-grid spline values."""
    model = SOSKANEnergyV3(n_splines=4, rank=2, n_features=2, hidden_dim=3, seed=1199)
    rng = np.random.default_rng(1199)
    model.W1[:] = rng.normal(0.0, 0.2, model.W1.shape)
    model.b1[:] = rng.normal(0.0, 0.05, model.b1.shape)
    model.W2[:] = 0.37
    model.b2[:] = 0.37
    model.c[:] = rng.normal(0.0, 0.01, model.c.shape)
    return model


def test_quantize_to_grid_rounds_to_requested_interval() -> None:
    """REQ-KAN-1199: values round to nearest 1/(2^bits - 1) grid interval."""
    values = np.array([0.0, 0.37, -0.37, 1.0], dtype=np.float64)

    q4 = quantize_to_grid(values, bits=4)
    q8 = quantize_to_grid(values, bits=8)

    np.testing.assert_allclose(q4, [0.0, 6 / 15, -6 / 15, 1.0])
    np.testing.assert_allclose(q8, [0.0, 94 / 255, -94 / 255, 1.0])
    with pytest.raises(ValueError, match="bits"):
        quantize_to_grid(values, bits=0)


def test_endpoint_row_mask_marks_first_and_last_spline_rows() -> None:
    """REQ-KAN-1199: endpoint rows are first/last control point per feature spline."""
    model = SOSKANEnergyV3(n_splines=4, rank=2, n_features=2, hidden_dim=3, seed=1)

    mask = endpoint_row_mask(model)

    expected = []
    for feature in range(model.n_features):
        for spline in range(model.n_splines):
            for _rank in range(model.rank):
                expected.append(spline in (0, model.n_splines - 1))
    assert mask.tolist() == expected
    assert int(mask.sum()) == model.n_features * 2 * model.rank


def test_quantization_uses_doubled_endpoint_precision_and_keeps_head_exact() -> None:
    """SCENARIO-KAN-1199: 4-bit interiors and 8-bit endpoints are applied separately."""
    model = _toy_model()
    original = clone_sos_kan_v3(model)
    mask = endpoint_row_mask(model)

    result = quantize_sos_kan_v3_spline_weights(model, bits=4)
    quantized = result.model

    interior_value = 6 / 15
    endpoint_value = 94 / 255
    np.testing.assert_allclose(quantized.W2[~mask], interior_value)
    np.testing.assert_allclose(quantized.W2[mask], endpoint_value)
    np.testing.assert_allclose(quantized.b2[~mask], interior_value)
    np.testing.assert_allclose(quantized.b2[mask], endpoint_value)
    np.testing.assert_allclose(quantized.W1, original.W1)
    np.testing.assert_allclose(quantized.b1, original.b1)
    np.testing.assert_allclose(quantized.c, original.c)

    # The source model is not mutated by quantization.
    np.testing.assert_allclose(model.W2, original.W2)
    np.testing.assert_allclose(model.b2, original.b2)
    assert result.bits == 4
    assert result.endpoint_bits == 8
    assert result.quantized_parameter_names == ("W2", "b2")
    with pytest.raises(ValueError, match="bits"):
        quantize_sos_kan_v3_spline_weights(model, bits=3)
    with pytest.raises(ValueError, match="endpoint_bits"):
        quantize_sos_kan_v3_spline_weights(model, bits=4, endpoint_bits=2)


def test_size_estimate_uses_packed_spline_bits_and_float32_head() -> None:
    """REQ-KAN-1199: model size estimates account for packed spline precision."""
    model = _toy_model()
    mask = endpoint_row_mask(model)

    full = estimate_sos_kan_v3_size_bytes(model)
    q8 = estimate_sos_kan_v3_size_bytes(model, bits=8)
    q4 = estimate_sos_kan_v3_size_bytes(model, bits=4)

    learned_arrays = (model.W1, model.b1, model.W2, model.b2, model.c)
    assert full == sum(array.size for array in learned_arrays) * 4

    endpoint_values = int(mask.sum()) * (model.hidden_dim + 1)
    interior_values = int((~mask).sum()) * (model.hidden_dim + 1)
    exact_head_values = model.W1.size + model.b1.size + model.c.size
    expected_q4 = math.ceil(
        (exact_head_values * 32 + endpoint_values * 8 + interior_values * 4) / 8
    )
    expected_q8 = math.ceil(
        (exact_head_values * 32 + endpoint_values * 16 + interior_values * 8) / 8
    )

    assert q4 == expected_q4
    assert q8 == expected_q8
    assert q4 < q8 < full
    with pytest.raises(ValueError, match="bits"):
        estimate_sos_kan_v3_size_bytes(model, bits=3)


def test_evaluation_and_latency_helpers() -> None:
    """REQ-KAN-1199: AUROC and latency helpers evaluate SOS-KAN energy calls."""
    model = _toy_model()
    X = np.random.default_rng(7).uniform(-1.0, 1.0, (6, model.n_features))
    y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    auroc = evaluate_sos_kan_v3_auroc(model, X, y)
    latency = measure_per_sample_latency_ms(model, X, n_samples=5)

    assert 0.0 <= auroc <= 1.0
    assert latency >= 0.0
    with pytest.raises(ValueError, match="n_samples"):
        measure_per_sample_latency_ms(model, X, n_samples=0)
    with pytest.raises(ValueError, match="at least one"):
        measure_per_sample_latency_ms(model, X[:0], n_samples=1)


def test_quantized_safetensors_export_contains_metadata_and_arrays(tmp_path: Path) -> None:
    """SCENARIO-KAN-1199: 4-bit export writes safetensors checkpoint metadata."""
    result = quantize_sos_kan_v3_spline_weights(_toy_model(), bits=4)
    path = tmp_path / "soskan_v3_q4.safetensors"

    written = export_quantized_sos_kan_v3_safetensors(
        result.model,
        path,
        bits=4,
        source_metadata={"source": "unit-test"},
    )

    from safetensors import safe_open

    assert written == path
    with safe_open(str(path), framework="np") as handle:
        keys = set(handle.keys())
        metadata = handle.metadata()
        stored_w2 = handle.get_tensor("W2")

    assert {"W1", "b1", "W2", "b2", "c"} <= keys
    assert metadata["schema"] == "carnot.soskan.v3.kantize.safetensors.v1"
    assert metadata["spec"] == "REQ-KAN-1199"
    assert metadata["quantization_bits"] == "4"
    assert metadata["endpoint_bits"] == "8"
    assert metadata["source"] == "unit-test"
    np.testing.assert_allclose(stored_w2, result.model.W2.astype(np.float32))
    with pytest.raises(ValueError, match="bits"):
        export_quantized_sos_kan_v3_safetensors(result.model, tmp_path / "bad.safetensors", bits=3)

    # Also cover the no-extra-metadata branch.
    no_meta_path = export_quantized_sos_kan_v3_safetensors(
        result.model,
        tmp_path / "soskan_v3_q8.safetensors",
        bits=8,
    )
    assert no_meta_path.exists()


@pytest.mark.parametrize(
    ("auroc_4bit", "auroc_8bit", "expected"),
    [
        (0.9700, 0.9800, "4bit_auroc_above_threshold"),
        (0.9699, 0.9800, "4bit_auroc_below_threshold_8bit_viable"),
        (0.9600, 0.9699, "quantization_failed"),
    ],
)
def test_verdict_classifier(auroc_4bit: float, auroc_8bit: float, expected: str) -> None:
    """REQ-KAN-1199: honest verdict follows 4-bit and 8-bit AUROC thresholds."""
    assert classify_kantize_verdict(auroc_4bit, auroc_8bit) == expected


def test_artifact_schema_requires_requested_fields() -> None:
    """REQ-KAN-1199: result artifact carries all required deployment fields."""
    artifact = {
        "soskan_full_precision_auroc": 0.9902,
        "soskan_8bit_auroc": 0.9901,
        "soskan_4bit_auroc": 0.9701,
        "soskan_full_precision_size_mb": 1.0,
        "soskan_8bit_size_mb": 0.3,
        "soskan_4bit_size_mb": 0.2,
        "soskan_4bit_inference_latency_ms": 0.01,
        "kantize_4bit_checkpoint_path": "results/soskan_q4.safetensors",
        "kantize_auroc_maintained_above_0p97": True,
        "edge_deployment_ready": True,
        "honest_verdict": "4bit_auroc_above_threshold",
    }

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert not artifact_has_required_fields({"honest_verdict": "quantization_failed"})


def test_exp1130_checkpoint_loader_and_runner_paths() -> None:
    """REQ-KAN-1199: existing SOSKANEnergyV3 checkpoint can be loaded when present."""
    checkpoint = _PROJECT_ROOT / "results" / "experiment_1130_soskan_energy_v3_retrained.npz"
    if checkpoint.exists():
        model, metadata = load_sos_kan_v3_npz(checkpoint)
        assert model.n_splines == int(metadata["n_splines"])
        assert model.rank == int(metadata["rank"])
        assert model.n_features == int(metadata["n_features"])
        assert model.W2.shape[0] == model.n_features * model.n_splines * model.rank

    assert exp1199.OUTPUT_PATH.name == "experiment_1199_kantize_soskan_4bit_quantization.json"
    assert exp1199.CHECKPOINT_PATH.name == "experiment_1130_soskan_energy_v3_retrained.npz"


def test_checkpoint_loader_rejects_missing_and_mismatched_arrays(tmp_path: Path) -> None:
    """REQ-KAN-1199: malformed SOS-KAN checkpoints fail with clear errors."""
    missing = tmp_path / "missing_w2.npz"
    np.savez(
        missing,
        W1=np.zeros((1, 1)),
        b1=np.zeros(1),
        b2=np.zeros(1),
        c=np.zeros(1),
        metadata=json.dumps(
            {"n_splines": 2, "rank": 1, "n_features": 1, "hidden_dim": 1, "seed": 1}
        ),
    )
    with pytest.raises(ValueError, match="missing W2"):
        load_sos_kan_v3_npz(missing)

    mismatched = tmp_path / "mismatched_w1.npz"
    np.savez(
        mismatched,
        W1=np.zeros((2, 2)),
        b1=np.zeros(1),
        W2=np.zeros((2, 1)),
        b2=np.zeros(2),
        c=np.zeros(1),
        metadata=json.dumps(
            {"n_splines": 2, "rank": 1, "n_features": 1, "hidden_dim": 1, "seed": 1}
        ),
    )
    with pytest.raises(ValueError, match="W1 shape"):
        load_sos_kan_v3_npz(mismatched)
