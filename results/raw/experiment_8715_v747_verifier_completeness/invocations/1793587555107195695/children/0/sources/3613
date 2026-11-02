"""KANtize-style quantization helpers for ``SOSKANEnergyV3``.

The Exp 1199 quantizer targets the spline-control rows in the
``SOSKANEnergyV3`` neural-Gram head.  Interior spline rows are rounded to a
fixed low-bit grid, while the first and last spline rows get doubled precision
because endpoint perturbations have outsized influence on spline shape.

Spec: REQ-KAN-1199, SCENARIO-KAN-1199
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from carnot.eval.metrics import auroc as canonical_auroc
from carnot.models.sos_kan import SOSKANEnergyV3

REQUIRED_ARTIFACT_FIELDS = {
    "soskan_full_precision_auroc",
    "soskan_8bit_auroc",
    "soskan_4bit_auroc",
    "soskan_full_precision_size_mb",
    "soskan_8bit_size_mb",
    "soskan_4bit_size_mb",
    "soskan_4bit_inference_latency_ms",
    "kantize_4bit_checkpoint_path",
    "kantize_auroc_maintained_above_0p97",
    "edge_deployment_ready",
    "honest_verdict",
}

_VALID_VERDICTS = {
    "4bit_auroc_above_threshold",
    "4bit_auroc_below_threshold_8bit_viable",
    "quantization_failed",
}

_LEARNED_ARRAY_NAMES = ("W1", "b1", "W2", "b2", "c")
_SPLINE_ARRAY_NAMES = ("W2", "b2")


@dataclass(frozen=True)
class SOSKANQuantizationResult:
    """A quantized SOSKANEnergyV3 instance plus the precision policy used."""

    model: SOSKANEnergyV3
    bits: int
    endpoint_bits: int
    quantized_parameter_names: tuple[str, ...]


def model_architecture(model: SOSKANEnergyV3) -> dict[str, int]:
    """Return the constructor fields required to recreate ``model``."""
    return {
        "n_splines": int(model.n_splines),
        "rank": int(model.rank),
        "n_features": int(model.n_features),
        "hidden_dim": int(model.hidden_dim),
    }


def learned_arrays(model: SOSKANEnergyV3) -> dict[str, np.ndarray]:
    """Return SOSKANEnergyV3 learned arrays in checkpoint order."""
    return {name: np.asarray(getattr(model, name)) for name in _LEARNED_ARRAY_NAMES}


def clone_sos_kan_v3(model: SOSKANEnergyV3) -> SOSKANEnergyV3:
    """Clone a SOSKANEnergyV3 by architecture and learned arrays."""
    clone = SOSKANEnergyV3(**model_architecture(model))
    for name, array in learned_arrays(model).items():
        setattr(clone, name, np.asarray(array, dtype=np.float64).copy())
    return clone


def quantize_to_grid(values: np.ndarray, bits: int) -> np.ndarray:
    """Round values to nearest ``1/(2^bits - 1)`` fixed grid interval.

    The operation intentionally does not clamp values.  The current
    SOSKANEnergyV3 weights are signed neural-head coefficients rather than
    probabilities, and the user-requested KANtize approximation is grid
    rounding, not affine min/max clipping.

    Spec: REQ-KAN-1199
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    scale = float((1 << bits) - 1)
    return np.round(np.asarray(values, dtype=np.float64) * scale) / scale


def endpoint_row_mask(model: SOSKANEnergyV3) -> np.ndarray:
    """Return a mask over W2/b2 output rows that correspond to spline endpoints.

    ``SOSKANEnergyV3`` reshapes the MLP output as
    ``(n_features, n_splines, rank)``.  Rows where ``spline`` is 0 or
    ``n_splines - 1`` are the first/last control points and receive doubled
    quantization precision.

    Spec: REQ-KAN-1199
    """
    mask = np.zeros(model.n_features * model.n_splines * model.rank, dtype=bool)
    row = 0
    for _feature in range(model.n_features):
        for spline in range(model.n_splines):
            is_endpoint = spline in (0, model.n_splines - 1)
            for _rank in range(model.rank):
                mask[row] = is_endpoint
                row += 1
    return mask


def quantize_sos_kan_v3_spline_weights(
    model: SOSKANEnergyV3,
    bits: int,
    endpoint_bits: int | None = None,
) -> SOSKANQuantizationResult:
    """Quantize SOSKANEnergyV3 spline-control rows and keep head parameters exact.

    ``W2`` and ``b2`` are the learned arrays that directly produce the
    per-feature, per-spline low-rank factors.  ``W1``, ``b1``, and ``c`` are not
    spline-control arrays and are copied exactly.

    Spec: REQ-KAN-1199, SCENARIO-KAN-1199
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8 for Exp 1199, got {bits}")
    endpoint_bits = bits * 2 if endpoint_bits is None else int(endpoint_bits)
    if endpoint_bits < bits:
        raise ValueError("endpoint_bits must be >= bits")

    quantized = clone_sos_kan_v3(model)
    mask = endpoint_row_mask(model)

    quantized.W2[~mask] = quantize_to_grid(quantized.W2[~mask], bits)
    quantized.W2[mask] = quantize_to_grid(quantized.W2[mask], endpoint_bits)
    quantized.b2[~mask] = quantize_to_grid(quantized.b2[~mask], bits)
    quantized.b2[mask] = quantize_to_grid(quantized.b2[mask], endpoint_bits)

    return SOSKANQuantizationResult(
        model=quantized,
        bits=bits,
        endpoint_bits=endpoint_bits,
        quantized_parameter_names=_SPLINE_ARRAY_NAMES,
    )


def estimate_sos_kan_v3_size_bytes(model: SOSKANEnergyV3, bits: int | None = None) -> int:
    """Estimate deployable parameter bytes for full or packed quantized storage.

    ``bits=None`` reports a float32 full-precision baseline.  Quantized reports
    packed W2/b2 spline rows plus float32 W1/b1/c head parameters.

    Spec: REQ-KAN-1199
    """
    arrays = learned_arrays(model)
    if bits is None:
        return int(sum(array.size for array in arrays.values()) * 4)
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    endpoint_bits = bits * 2
    mask = endpoint_row_mask(model)
    exact_head_values = arrays["W1"].size + arrays["b1"].size + arrays["c"].size
    endpoint_values = int(mask.sum()) * (model.hidden_dim + 1)
    interior_values = int((~mask).sum()) * (model.hidden_dim + 1)
    total_bits = exact_head_values * 32 + endpoint_values * endpoint_bits + interior_values * bits
    return int(math.ceil(total_bits / 8))


def evaluate_sos_kan_v3_auroc(model: SOSKANEnergyV3, X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate SOSKANEnergyV3 AUROC with lower energy mapped to higher score."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    energies = np.array([model.energy(row) for row in X], dtype=np.float64)
    return float(canonical_auroc(y, -energies))


def measure_per_sample_latency_ms(
    model: SOSKANEnergyV3,
    X: np.ndarray,
    n_samples: int = 100,
) -> float:
    """Measure average per-sample latency over ``n_samples`` energy calls."""
    import time

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    X = np.asarray(X, dtype=np.float64)
    if len(X) == 0:
        raise ValueError("X must contain at least one sample")
    reps = np.resize(X, (n_samples, X.shape[1]))

    # One warmup call keeps import/cache effects out of the timing loop.
    model.energy(reps[0])
    start = time.perf_counter()
    for row in reps:
        model.energy(row)
    elapsed = time.perf_counter() - start
    return float((elapsed / n_samples) * 1000.0)


def load_sos_kan_v3_npz(path: str | Path) -> tuple[SOSKANEnergyV3, dict[str, Any]]:
    """Load a SOSKANEnergyV3 checkpoint from the NumPy format used by Exp 1130."""
    checkpoint = Path(path)
    archive = np.load(checkpoint, allow_pickle=False)
    metadata_raw = str(archive["metadata"]) if "metadata" in archive.files else "{}"
    metadata = json.loads(metadata_raw)
    architecture = {
        "n_splines": int(metadata.get("n_splines", 8)),
        "rank": int(metadata.get("rank", 8)),
        "n_features": int(metadata.get("n_features", 16)),
        "hidden_dim": int(metadata.get("hidden_dim", 32)),
        "seed": int(metadata.get("seed", 42)),
    }
    model = SOSKANEnergyV3(**architecture)
    for name in _LEARNED_ARRAY_NAMES:
        if name not in archive.files:
            raise ValueError(f"checkpoint missing {name}")
        value = np.asarray(archive[name], dtype=np.float64)
        expected_shape = getattr(model, name).shape
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape {value.shape} != expected {expected_shape}")
        setattr(model, name, value.copy())
    return model, metadata


def export_quantized_sos_kan_v3_safetensors(
    model: SOSKANEnergyV3,
    path: str | Path,
    bits: int,
    source_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a quantized SOSKANEnergyV3 safetensors checkpoint.

    Safetensors does not define native 4-bit NumPy tensors, so the checkpoint
    stores the rounded arrays as float32 along with explicit quantization
    metadata.  The reported Exp 1199 size fields use the packed-bit deployment
    estimate from ``estimate_sos_kan_v3_size_bytes``.

    Spec: REQ-KAN-1199, SCENARIO-KAN-1199
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import save_file

    tensors = {
        name: np.ascontiguousarray(array.astype(np.float32))
        for name, array in learned_arrays(model).items()
    }
    metadata = {
        "schema": "carnot.soskan.v3.kantize.safetensors.v1",
        "spec": "REQ-KAN-1199",
        "quantization_bits": str(bits),
        "endpoint_bits": str(bits * 2),
        "architecture": json.dumps(model_architecture(model), sort_keys=True),
        "quantized_parameter_names": json.dumps(list(_SPLINE_ARRAY_NAMES)),
    }
    if source_metadata:
        metadata.update({str(key): str(value) for key, value in source_metadata.items()})
    save_file(tensors, str(destination), metadata=metadata)
    return destination


def classify_kantize_verdict(
    auroc_4bit: float,
    auroc_8bit: float,
    threshold: float = 0.97,
) -> str:
    """Return the approved Exp 1199 honest verdict string."""
    if auroc_4bit >= threshold:
        return "4bit_auroc_above_threshold"
    if auroc_8bit >= threshold:
        return "4bit_auroc_below_threshold_8bit_viable"
    return "quantization_failed"


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an Exp 1199 artifact satisfies the required schema."""
    return (
        REQUIRED_ARTIFACT_FIELDS <= set(artifact)
        and artifact.get("honest_verdict") in _VALID_VERDICTS
    )
