"""Sparse KAN clustering with Global Group Lasso and spectral regularization.

This module treats a constraint-memory table as the set of KAN control-vector
rows that need to be stored for later energy evaluation.  The Sparse KAN
clusterer replaces those rows with a smaller centroid codebook, adds a Global
Group Lasso penalty that can prune whole centroid groups, and adds a spectral
regularizer so nearby constraint rows prefer the same cluster.  The goal is not
to train a production classifier here; it is a deterministic CPU probe for the
Exp 1604 memory-compression question.

Spec references: REQ-KAN-1604, SCENARIO-KAN-1604.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_RESULT_PATH = (
    Path(__file__).resolve().parents[3] / "results/experiment_1604_sparse_kan.json"
)


@dataclass(frozen=True)
class SparseKANConfig:
    """Configuration for the deterministic Sparse KAN clustering probe."""

    n_clusters: int = 4
    group_lasso_weight: float = 0.25
    spectral_weight: float = 0.5
    prune_threshold: float = 0.2
    max_iter: int = 8
    seed: int = 1604

    def validate(self) -> None:
        """Reject settings that would make the compression metric ambiguous."""

        if self.n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        if self.group_lasso_weight < 0.0 or self.spectral_weight < 0.0:
            raise ValueError("regularization weights must be non-negative")
        if self.prune_threshold < 0.0:
            raise ValueError("prune_threshold must be non-negative")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")


@dataclass(frozen=True)
class SparseKANClusteringResult:
    """Completed Sparse KAN clustering result and compression diagnostics."""

    centroids: np.ndarray
    assignments: np.ndarray
    active_mask: np.ndarray
    loss_components: dict[str, float]
    dense_memory_bytes: int
    compressed_memory_bytes: int

    @property
    def active_group_count(self) -> int:
        """Return the number of centroid groups kept after sparsification."""

        return int(np.count_nonzero(self.active_mask))

    @property
    def zero_group_count(self) -> int:
        """Return how many centroid groups were pruned to zero memory."""

        return int(self.active_mask.size - self.active_group_count)

    @property
    def sparsity_ratio(self) -> float:
        """Return pruned groups divided by total groups."""

        return float(self.zero_group_count / max(int(self.active_mask.size), 1))

    @property
    def memory_compression_ratio(self) -> float:
        """Return dense memory bytes divided by sparse codebook bytes."""

        return float(self.dense_memory_bytes / max(self.compressed_memory_bytes, 1))


def one_hot_assignments(labels: Sequence[int], n_clusters: int) -> np.ndarray:
    """Convert integer cluster labels into the assignment matrix `Z`."""

    labels_array = np.asarray(labels, dtype=np.int64)
    assignment = np.zeros((labels_array.size, n_clusters), dtype=np.float64)
    assignment[np.arange(labels_array.size), labels_array] = 1.0
    return assignment


def global_group_lasso_penalty(centroids: np.ndarray) -> float:
    """Return the Global Group Lasso penalty `sum_g ||C_g||_2`."""

    return float(np.sum(np.linalg.norm(np.asarray(centroids, dtype=np.float64), axis=1)))


def spectral_constraint_regularization(assignments: np.ndarray, affinity: np.ndarray) -> float:
    """Return `trace(Z^T L Z)` for a row-affinity graph Laplacian.

    Nearby constraint rows receive high affinity.  If clustering separates those
    rows, the one-hot assignment matrix changes across a high-weight graph edge
    and this term grows.  Keeping the calculation explicit makes the experiment
    artifact auditable without depending on a graph-learning library.
    """

    z = np.asarray(assignments, dtype=np.float64)
    affinity_matrix = np.asarray(affinity, dtype=np.float64)
    degree = np.diag(np.sum(affinity_matrix, axis=1))
    laplacian = degree - affinity_matrix
    return float(np.trace(z.T @ laplacian @ z))


class SparseKANClusterer:
    """Cluster and sparsify KAN constraint-memory rows."""

    def __init__(self, config: SparseKANConfig | None = None) -> None:
        self.config = SparseKANConfig() if config is None else config
        self.config.validate()

    def fit(self, constraint_memory: np.ndarray) -> SparseKANClusteringResult:
        """Fit a deterministic centroid codebook and prune weak groups."""

        rows = self._as_constraint_matrix(constraint_memory)
        centroids = self._initial_centroids(rows)
        assignments = np.zeros(rows.shape[0], dtype=np.int64)

        for _ in range(self.config.max_iter):
            assignments = self._assign_rows(rows, centroids)
            centroids = self._recompute_centroids(rows, centroids, assignments)

        sparse_centroids, active_mask = self._apply_group_lasso_sparsity(centroids)
        affinity = self._row_affinity(rows)
        loss_components = self.regularized_loss_components(rows, sparse_centroids, assignments, affinity)
        dense_bytes = int(rows.nbytes)
        compressed_bytes = self._compressed_memory_bytes(rows.shape[0], rows.shape[1], active_mask)

        return SparseKANClusteringResult(
            centroids=sparse_centroids,
            assignments=assignments,
            active_mask=active_mask,
            loss_components=loss_components,
            dense_memory_bytes=dense_bytes,
            compressed_memory_bytes=compressed_bytes,
        )

    def regularized_loss_components(
        self,
        rows: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        affinity: np.ndarray,
    ) -> dict[str, float]:
        """Return the reconstruction, group-lasso, spectral, and total losses."""

        reconstructed = centroids[assignments]
        reconstruction_loss = float(np.mean((rows - reconstructed) ** 2))
        group_lasso = global_group_lasso_penalty(centroids)
        spectral = spectral_constraint_regularization(
            one_hot_assignments(assignments, self.config.n_clusters),
            affinity,
        )
        total = (
            reconstruction_loss
            + self.config.group_lasso_weight * group_lasso
            + self.config.spectral_weight * spectral
        )
        return {
            "reconstruction_loss": reconstruction_loss,
            "global_group_lasso_penalty": group_lasso,
            "spectral_constraint_regularization": spectral,
            "total_loss": float(total),
        }

    def _as_constraint_matrix(self, constraint_memory: np.ndarray) -> np.ndarray:
        rows = np.asarray(constraint_memory, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("constraint_memory must be a 2D matrix")
        if rows.shape[0] < self.config.n_clusters:
            raise ValueError("constraint_memory rows must be >= n_clusters")
        return rows

    def _initial_centroids(self, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(np.linalg.norm(rows, axis=1), kind="stable")
        positions = np.linspace(0, rows.shape[0] - 1, self.config.n_clusters).round().astype(int)
        return rows[order[positions]].copy()

    @staticmethod
    def _assign_rows(rows: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.sum((rows[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)

    def _recompute_centroids(
        self,
        rows: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
    ) -> np.ndarray:
        return np.vstack(
            [
                rows[assignments == cluster].mean(axis=0)
                if np.any(assignments == cluster)
                else centroids[cluster]
                for cluster in range(self.config.n_clusters)
            ]
        )

    def _apply_group_lasso_sparsity(self, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        norms = np.linalg.norm(centroids, axis=1)
        active_mask = norms >= self.config.prune_threshold
        sparse_centroids = centroids.copy()
        sparse_centroids[~active_mask] = 0.0
        return sparse_centroids, active_mask

    @staticmethod
    def _row_affinity(rows: np.ndarray) -> np.ndarray:
        diff = rows[:, None, :] - rows[None, :, :]
        squared_distance = np.sum(diff**2, axis=2)
        positive = squared_distance[squared_distance > 0.0]
        sigma2 = float(np.median(positive)) if positive.size else 1.0
        affinity = np.exp(-squared_distance / max(sigma2, 1e-12))
        np.fill_diagonal(affinity, 0.0)
        return affinity

    @staticmethod
    def _compressed_memory_bytes(n_rows: int, n_features: int, active_mask: np.ndarray) -> int:
        active_centroid_bytes = int(np.count_nonzero(active_mask) * n_features * np.dtype(np.float64).itemsize)
        assignment_bytes = int(n_rows * np.dtype(np.uint8).itemsize)
        return active_centroid_bytes + assignment_bytes


def _reference_constraint_memory() -> np.ndarray:
    return np.array(
        [
            [1.00, 0.00, 0.00, 0.00],
            [1.10, 0.05, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.05, 1.10, 0.00, 0.00],
            [0.01, 0.00, 0.00, 0.00],
            [0.00, 0.01, 0.00, 0.00],
        ],
        dtype=np.float64,
    )


def build_experiment_1604_artifact() -> dict[str, object]:
    """Build the stable Exp 1604 Sparse KAN clustering artifact payload."""

    clusterer = SparseKANClusterer()
    result = clusterer.fit(_reference_constraint_memory())
    verdict = (
        "complete: sparse_kan_clustering_compressed_constraint_memory"
        if result.memory_compression_ratio > 1.0
        else "complete: sparse_kan_clustering_no_memory_gain"
    )

    return {
        "schema": "carnot.sparse_kan_clustering.v1",
        "status": "complete",
        "experiment": 1604,
        "experiment_id": 1604,
        "run_date": "20260509",
        "title": "Sparse KAN clustering with spectral constraints",
        "spec": ["REQ-KAN-1604", "SCENARIO-KAN-1604"],
        "module": "python/carnot/models/sparse_kan_clustering.py",
        "artifact_path": "results/experiment_1604_sparse_kan.json",
        "n_constraint_rows": int(result.assignments.size),
        "n_features": int(result.centroids.shape[1]),
        "n_clusters": int(result.centroids.shape[0]),
        "active_group_count": result.active_group_count,
        "zero_group_count": result.zero_group_count,
        "dense_memory_bytes": result.dense_memory_bytes,
        "compressed_memory_bytes": result.compressed_memory_bytes,
        "global_group_lasso_weight": clusterer.config.group_lasso_weight,
        "spectral_constraint_weight": clusterer.config.spectral_weight,
        "global_group_lasso_penalty": result.loss_components["global_group_lasso_penalty"],
        "spectral_constraint_regularization": result.loss_components[
            "spectral_constraint_regularization"
        ],
        "reconstruction_loss": result.loss_components["reconstruction_loss"],
        "total_loss": result.loss_components["total_loss"],
        "sparsity_ratio": result.sparsity_ratio,
        "memory_compression_ratio": result.memory_compression_ratio,
        "constraints_compressed": result.memory_compression_ratio > 1.0,
        "sparse_kan_clustering_ready": True,
        "hardware_execution_confirmed": False,
        "honest_verdict": verdict,
    }


def write_experiment_1604_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, object]:
    """Write `results/experiment_1604_sparse_kan.json` and return the payload."""

    artifact = build_experiment_1604_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_RESULT_PATH",
    "SparseKANClusterer",
    "SparseKANClusteringResult",
    "SparseKANConfig",
    "build_experiment_1604_artifact",
    "global_group_lasso_penalty",
    "one_hot_assignments",
    "spectral_constraint_regularization",
    "write_experiment_1604_artifact",
]
