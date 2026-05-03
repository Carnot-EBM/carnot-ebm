"""MetaCluster-style centroid compression for SOSKANEnergyV3.

The SOSKANEnergyV3 head stores many small learned coefficient vectors.  For the
exp1128 production shape, the largest block is ``W2``: 96 rows that map the
hidden layer to per-feature spline factors.  This module treats every learned
row vector, including bias scalars as one-column vectors, as a candidate for a
shared K-means codebook.  The compressed payload stores one float32 centroid
table plus one integer centroid index per vector, then reconstructs dense
SOSKANEnergyV3 weights by expanding indices back to centroids.

Spec: REQ-KAN-1148, SCENARIO-KAN-1148
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from carnot.models.sos_kan import SOSKANEnergyV3

REQUIRED_ARTIFACT_FIELDS = {
    "sos_kan_compressed",
    "auroc_original",
    "auroc_compressed",
    "auroc_drop",
    "auroc_drop_within_02",
    "size_original_bytes",
    "size_compressed_bytes",
    "size_reduction_factor",
    "n_centroids",
    "energy_correlation",
    "honest_verdict",
}

_VERDICTS = {
    "compressed_within_02_auroc_5x_smaller",
    "compressed_auroc_degraded",
    "compression_ratio_below_5x",
    "checkpoint_not_found",
}

_LEARNED_ARRAY_NAMES = ("W1", "b1", "W2", "b2", "c")


@dataclass(frozen=True)
class CoefficientBlock:
    """One learned SOSKANEnergyV3 array represented as row coefficient vectors."""

    name: str
    shape: tuple[int, ...]
    vector_count: int
    vector_width: int
    start: int
    end: int


@dataclass(frozen=True)
class SOSKANV3Codebook:
    """Compressed SOSKANEnergyV3 payload backed by centroids and indices."""

    architecture: dict[str, int]
    blocks: tuple[CoefficientBlock, ...]
    centroids: np.ndarray
    indices: np.ndarray
    packed_indices: bytes
    uncompressed_arrays: dict[str, np.ndarray]
    size_original_bytes: int

    @property
    def n_centroids(self) -> int:
        """Return the number of centroids in the shared codebook."""
        return int(self.centroids.shape[0])

    @property
    def vector_count(self) -> int:
        """Return the number of coefficient vectors encoded by ``indices``."""
        return int(self.indices.size)

    @property
    def size_compressed_bytes(self) -> int:
        """Return the byte count of the deployable codebook payload."""
        uncompressed_bytes = sum(array.nbytes for array in self.uncompressed_arrays.values())
        return int(self.centroids.nbytes + len(self.packed_indices) + uncompressed_bytes)

    @property
    def size_reduction_factor(self) -> float:
        """Return original dense parameter bytes divided by codebook bytes."""
        return float(self.size_original_bytes / max(self.size_compressed_bytes, 1))


def _learned_arrays(model: SOSKANEnergyV3) -> dict[str, np.ndarray]:
    """Return the learned dense arrays that are stored in a checkpoint."""
    return {
        name: np.asarray(getattr(model, name), dtype=np.float64) for name in _LEARNED_ARRAY_NAMES
    }


def _array_to_vectors(array: np.ndarray) -> np.ndarray:
    """Flatten one learned array into row vectors suitable for K-means."""
    if array.ndim == 1:
        return array.reshape(array.shape[0], 1)
    return array.reshape(array.shape[0], int(np.prod(array.shape[1:])))


def collect_sos_kan_coefficient_vectors(
    model: SOSKANEnergyV3,
    array_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[CoefficientBlock, ...]]:
    """Collect all learned SOSKANEnergyV3 coefficient vectors into one matrix.

    K-means needs a single vector width, while SOSKANEnergyV3 has blocks with
    different row widths.  We therefore right-pad shorter rows with zeros and
    record the original width in ``CoefficientBlock`` so reconstruction can crop
    each row before reshaping it back to the source array.
    """
    arrays = _learned_arrays(model)
    selected_names = _LEARNED_ARRAY_NAMES if array_names is None else tuple(array_names)
    raw_vectors = {name: _array_to_vectors(arrays[name]) for name in selected_names}
    max_width = max(vectors.shape[1] for vectors in raw_vectors.values())

    padded_blocks: list[np.ndarray] = []
    blocks: list[CoefficientBlock] = []
    start = 0
    for name in selected_names:
        vectors = raw_vectors[name]
        padded = np.zeros((vectors.shape[0], max_width), dtype=np.float32)
        padded[:, : vectors.shape[1]] = vectors.astype(np.float32)
        end = start + vectors.shape[0]
        blocks.append(
            CoefficientBlock(
                name=name,
                shape=tuple(int(dim) for dim in arrays[name].shape),
                vector_count=int(vectors.shape[0]),
                vector_width=int(vectors.shape[1]),
                start=start,
                end=end,
            )
        )
        padded_blocks.append(padded)
        start = end

    return np.vstack(padded_blocks), tuple(blocks)


def _pack_indices(indices: np.ndarray, n_centroids: int) -> bytes:
    """Pack codebook indices into the minimum fixed-width bit representation."""
    bit_width = max(1, int(math.ceil(math.log2(max(n_centroids, 2)))))
    shifts = np.arange(bit_width - 1, -1, -1, dtype=np.uint8)
    bits = ((indices.astype(np.uint32)[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits).tobytes()


def inspect_sos_kan_v3_coefficients(model: SOSKANEnergyV3) -> dict[str, Any]:
    """Return human-readable structure details for exp1148 reporting."""
    vectors, blocks = collect_sos_kan_coefficient_vectors(model)
    arrays = _learned_arrays(model)
    return {
        "n_kan_basis_functions": int(model.n_splines),
        "coefficients_per_spline": int(model.rank),
        "n_features": int(model.n_features),
        "hidden_dim": int(model.hidden_dim),
        "coefficient_vector_count": int(vectors.shape[0]),
        "max_coefficient_vector_width": int(vectors.shape[1]),
        "trainable_parameter_count": int(sum(array.size for array in arrays.values())),
        "parameter_bytes_float64": int(sum(array.nbytes for array in arrays.values())),
        "parameter_blocks": [
            {
                "name": block.name,
                "shape": list(block.shape),
                "vector_count": block.vector_count,
                "vector_width": block.vector_width,
            }
            for block in blocks
        ],
    }


def compress_sos_kan_v3(
    model: SOSKANEnergyV3,
    n_centroids: int = 32,
    random_state: int = 1148,
    block_names: tuple[str, ...] | None = None,
) -> SOSKANV3Codebook:
    """Compress learned SOSKANEnergyV3 vectors with sklearn KMeans.

    The codebook deliberately uses float32 centroids and uint8 indices.  This is
    the deployment-oriented storage format, while reconstruction casts back to
    float64 because the current NumPy SOSKANEnergyV3 implementation computes in
    float64.
    """
    compressed_names = _LEARNED_ARRAY_NAMES if block_names is None else tuple(block_names)
    vectors, blocks = collect_sos_kan_coefficient_vectors(model, compressed_names)
    if n_centroids < 1 or n_centroids > 256 or n_centroids > len(vectors):
        raise ValueError(f"n_centroids must be in [1, min(256, {len(vectors)})], got {n_centroids}")

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_centroids, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(vectors)
    arrays = _learned_arrays(model)
    uncompressed_arrays = {
        name: arrays[name].copy() for name in _LEARNED_ARRAY_NAMES if name not in compressed_names
    }
    architecture = {
        "n_splines": int(model.n_splines),
        "rank": int(model.rank),
        "n_features": int(model.n_features),
        "hidden_dim": int(model.hidden_dim),
    }
    return SOSKANV3Codebook(
        architecture=architecture,
        blocks=blocks,
        centroids=kmeans.cluster_centers_.astype(np.float32),
        indices=labels.astype(np.uint8),
        packed_indices=_pack_indices(labels.astype(np.uint8), n_centroids),
        uncompressed_arrays=uncompressed_arrays,
        size_original_bytes=int(sum(array.nbytes for array in arrays.values())),
    )


def reconstruct_sos_kan_v3(payload: SOSKANV3Codebook) -> SOSKANEnergyV3:
    """Expand a centroid payload back into a dense SOSKANEnergyV3 instance."""
    model = SOSKANEnergyV3(**payload.architecture)
    expanded = payload.centroids[payload.indices].astype(np.float64)

    for name, array in payload.uncompressed_arrays.items():
        setattr(model, name, array.astype(np.float64).copy())

    for block in payload.blocks:
        rows = expanded[block.start : block.end, : block.vector_width]
        setattr(model, block.name, rows.reshape(block.shape).astype(np.float64))

    return model


def classify_metacluster_verdict(
    checkpoint_found: bool,
    auroc_original: float,
    auroc_compressed: float,
    size_reduction_factor: float,
) -> str:
    """Classify the exp1148 honest verdict from the measured gates."""
    if not checkpoint_found:
        return "checkpoint_not_found"
    if auroc_original - auroc_compressed > 0.02:
        return "compressed_auroc_degraded"
    if size_reduction_factor < 5.0:
        return "compression_ratio_below_5x"
    return "compressed_within_02_auroc_5x_smaller"


def artifact_has_required_fields(artifact: dict[str, Any]) -> bool:
    """Return whether an exp1148 artifact includes the required schema fields."""
    return REQUIRED_ARTIFACT_FIELDS <= set(artifact) and artifact.get("honest_verdict") in _VERDICTS
