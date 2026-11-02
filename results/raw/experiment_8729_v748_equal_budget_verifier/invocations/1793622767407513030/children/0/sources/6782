#!/usr/bin/env python3
"""Exp 1648 Sparse KAN spectral constraint grouping.

Spec: REQ-KAN-1648, SCENARIO-KAN-1648.

The experiment keeps the Tier 4 adaptive-landscape probe deterministic and
CPU-only.  It first builds a small constraint-memory matrix with repeated local
landscape structures, then uses a graph Laplacian eigenvector ordering to form
spectral groups before applying Sparse KAN-style centroid memory accounting.
The direct `compression_ratio` field is recorded for conductor comparisons.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from carnot.models.sparse_kan_clustering import (
    SparseKANClusterer,
    SparseKANConfig,
    global_group_lasso_penalty,
    one_hot_assignments,
    spectral_constraint_regularization,
)

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1648_sparse_kan.json"
EXPERIMENT_ID = 1648
RUN_DATE = "20260509"
SCHEMA = "carnot.spectral_sparse_kan.v1"
TITLE = "Sparse KANs with spectral constraints for manifold compression"
SPEC_TRACES = ["REQ-KAN-1648", "SCENARIO-KAN-1648"]
TIER = "FR-11 Tier 4"
DEFAULT_N_GROUPS = 3

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema",
    "status",
    "experiment_id",
    "spec_traces",
    "n_constraint_rows",
    "n_spectral_groups",
    "spectral_gap",
    "spectral_grouping_penalty",
    "dense_memory_bytes",
    "compressed_memory_bytes",
    "compression_ratio",
    "honest_verdict",
)


@dataclass(frozen=True)
class SpectralGroupingResult:
    """Rows grouped by a Laplacian spectral embedding."""

    labels: np.ndarray
    embedding: np.ndarray
    affinity: np.ndarray
    eigenvalues: np.ndarray
    spectral_gap: float
    spectral_grouping_penalty: float


@dataclass(frozen=True)
class SpectralSparseKANResult:
    """Sparse KAN compression metrics after spectral row grouping."""

    grouping: SpectralGroupingResult
    centroids: np.ndarray
    active_mask: np.ndarray
    dense_memory_bytes: int
    compressed_memory_bytes: int
    sparse_kan_memory_compression_ratio: float
    global_group_lasso_penalty: float

    @property
    def active_group_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def zero_group_count(self) -> int:
        return int(self.active_mask.size - self.active_group_count)

    @property
    def compression_ratio(self) -> float:
        return float(self.dense_memory_bytes / max(self.compressed_memory_bytes, 1))

    @property
    def spectral_gap(self) -> float:
        return self.grouping.spectral_gap

    @property
    def spectral_grouping_penalty(self) -> float:
        return self.grouping.spectral_grouping_penalty


def build_tier4_adaptive_landscape() -> np.ndarray:
    """Return deterministic Tier 4 rows with repeated local KAN structures."""

    return np.array(
        [
            [1.00, 0.05, 0.00, 0.00, 0.90, 0.10],
            [1.05, 0.02, 0.00, 0.00, 0.85, 0.12],
            [0.95, 0.08, 0.00, 0.00, 0.92, 0.08],
            [0.00, 1.00, 0.06, 0.00, 0.15, 0.80],
            [0.02, 1.05, 0.03, 0.00, 0.12, 0.82],
            [0.00, 0.95, 0.08, 0.00, 0.18, 0.78],
            [0.00, 0.00, 1.00, 0.05, 0.65, 0.35],
            [0.00, 0.02, 1.08, 0.03, 0.63, 0.38],
            [0.00, 0.00, 0.94, 0.08, 0.68, 0.32],
        ],
        dtype=np.float64,
    )


def spectral_group_rows(rows: np.ndarray, *, n_groups: int = DEFAULT_N_GROUPS) -> SpectralGroupingResult:
    """Assign rows to groups using graph Laplacian eigenvector ordering."""

    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("rows must be a 2D matrix")
    if not 1 <= n_groups <= matrix.shape[0]:
        raise ValueError("n_groups must be between 1 and number of rows")

    affinity = _row_affinity(matrix)
    degree = np.diag(np.sum(affinity, axis=1))
    laplacian = degree - affinity
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    embedding = eigenvectors[:, :n_groups]
    order_key = eigenvectors[:, 1] if n_groups > 1 and eigenvectors.shape[1] > 1 else eigenvectors[:, 0]
    labels = np.empty(matrix.shape[0], dtype=np.int64)
    for group_index, row_indices in enumerate(np.array_split(np.argsort(order_key, kind="stable"), n_groups)):
        labels[row_indices] = group_index

    spectral_gap = _spectral_gap(eigenvalues, n_groups)
    penalty = spectral_constraint_regularization(one_hot_assignments(labels, n_groups), affinity)
    return SpectralGroupingResult(
        labels=labels,
        embedding=embedding,
        affinity=affinity,
        eigenvalues=eigenvalues,
        spectral_gap=spectral_gap,
        spectral_grouping_penalty=penalty,
    )


def compress_with_spectral_sparse_kan(
    rows: np.ndarray,
    *,
    n_groups: int = DEFAULT_N_GROUPS,
    sparse_config: SparseKANConfig | None = None,
) -> SpectralSparseKANResult:
    """Compress spectral row groups while reusing Sparse KAN memory accounting."""

    matrix = np.asarray(rows, dtype=np.float64)
    grouping = spectral_group_rows(matrix, n_groups=n_groups)
    centroids = _centroids_from_labels(matrix, grouping.labels, n_groups)
    config = sparse_config or SparseKANConfig(
        n_clusters=n_groups,
        group_lasso_weight=0.25,
        spectral_weight=0.5,
        prune_threshold=0.05,
        max_iter=6,
        seed=EXPERIMENT_ID,
    )
    sparse_result = SparseKANClusterer(config).fit(matrix)
    active_mask = np.linalg.norm(centroids, axis=1) >= config.prune_threshold
    dense_bytes = int(matrix.nbytes)
    compressed_bytes = _compressed_memory_bytes(matrix.shape[0], matrix.shape[1], active_mask)
    return SpectralSparseKANResult(
        grouping=grouping,
        centroids=centroids,
        active_mask=active_mask,
        dense_memory_bytes=dense_bytes,
        compressed_memory_bytes=compressed_bytes,
        sparse_kan_memory_compression_ratio=sparse_result.memory_compression_ratio,
        global_group_lasso_penalty=global_group_lasso_penalty(centroids),
    )


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
    rows: np.ndarray | None = None,
    n_groups: int = DEFAULT_N_GROUPS,
) -> JsonDict:
    """Build the deterministic Exp 1648 artifact without writing it."""

    matrix = build_tier4_adaptive_landscape() if rows is None else np.asarray(rows, dtype=np.float64)
    result = compress_with_spectral_sparse_kan(matrix, n_groups=n_groups)
    status = "complete" if result.compression_ratio > 1.0 and math.isfinite(result.spectral_gap) else "blocked"
    verdict = (
        "complete: spectral_sparse_kan_compressed_tier4_landscape"
        if status == "complete"
        else "blocked: spectral_sparse_kan_no_memory_gain"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": status,
        "experiment": "1648_sparse_kan",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "title": TITLE,
        "spec_traces": list(SPEC_TRACES),
        "tier": TIER,
        "module": "scripts/experiment_1648_sparse_kan.py",
        "artifact_path": "results/experiment_1648_sparse_kan.json",
        "n_constraint_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "n_spectral_groups": int(n_groups),
        "active_group_count": result.active_group_count,
        "zero_group_count": result.zero_group_count,
        "spectral_gap": result.spectral_gap,
        "spectral_grouping_penalty": result.spectral_grouping_penalty,
        "dense_memory_bytes": result.dense_memory_bytes,
        "compressed_memory_bytes": result.compressed_memory_bytes,
        "compression_ratio": result.compression_ratio,
        "sparse_kan_memory_compression_ratio": result.sparse_kan_memory_compression_ratio,
        "global_group_lasso_penalty": result.global_group_lasso_penalty,
        "spectral_groups": result.grouping.labels.astype(int).tolist(),
        "spectral_sparse_kan_ready": status == "complete",
        "hardware_execution_confirmed": False,
        "tests_run": list(tests_run or []),
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: JsonDict) -> None:
    """Validate the schema fields that downstream conductor logic depends on."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    expected_ratio = float(
        artifact["dense_memory_bytes"] / max(int(artifact["compressed_memory_bytes"]), 1)
    )
    if not math.isclose(float(artifact["compression_ratio"]), expected_ratio, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("compression_ratio must equal dense_memory_bytes / compressed_memory_bytes")
    if artifact["status"] == "complete" and float(artifact["compression_ratio"]) <= 1.0:
        raise AssertionError("compression_ratio must be > 1.0 for complete artifacts")
    if artifact["spec_traces"] != SPEC_TRACES:
        raise AssertionError("spec_traces must cite REQ-KAN-1648 and SCENARIO-KAN-1648")


def run_experiment(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Write `results/experiment_1648_sparse_kan.json` and return its payload."""

    artifact = build_artifact(run_date=run_date, tests_run=tests_run)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Exp 1648 runner."""

    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-date", default=RUN_DATE)
    args = parser.parse_args(argv)
    artifact = run_experiment(output_path=args.output, run_date=args.run_date)
    print(f"wrote={args.output}")
    print(f"compression_ratio={artifact['compression_ratio']:.6f}")
    return 0


def _row_affinity(rows: np.ndarray) -> np.ndarray:
    diff = rows[:, None, :] - rows[None, :, :]
    squared_distance = np.sum(diff**2, axis=2)
    positive = squared_distance[squared_distance > 0.0]
    sigma2 = float(np.median(positive)) if positive.size else 1.0
    affinity = np.exp(-squared_distance / max(sigma2, 1e-12))
    np.fill_diagonal(affinity, 0.0)
    return affinity


def _spectral_gap(eigenvalues: np.ndarray, n_groups: int) -> float:
    left = min(max(n_groups - 1, 0), eigenvalues.size - 1)
    right = min(n_groups, eigenvalues.size - 1)
    return float(max(eigenvalues[right] - eigenvalues[left], 0.0))


def _centroids_from_labels(rows: np.ndarray, labels: np.ndarray, n_groups: int) -> np.ndarray:
    return np.vstack([rows[labels == group].mean(axis=0) for group in range(n_groups)])


def _compressed_memory_bytes(n_rows: int, n_features: int, active_mask: np.ndarray) -> int:
    centroid_bytes = int(np.count_nonzero(active_mask) * n_features * np.dtype(np.float64).itemsize)
    assignment_bytes = int(n_rows * np.dtype(np.uint8).itemsize)
    return centroid_bytes + assignment_bytes


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
