"""Tests for Sparse KAN clustering with spectral constraints.

Spec traces: REQ-KAN-1604, SCENARIO-KAN-1604
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.models.sparse_kan_clustering import (
    SparseKANClusterer,
    SparseKANConfig,
    build_experiment_1604_artifact,
    global_group_lasso_penalty,
    one_hot_assignments,
    spectral_constraint_regularization,
    write_experiment_1604_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kan_1604_spec_anchor_exists() -> None:
    """REQ-KAN-1604, SCENARIO-KAN-1604: sparse KAN work is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1604" in spec
    assert "SCENARIO-KAN-1604" in spec
    assert "results/experiment_1604_sparse_kan.json" in spec
    assert "Global Group Lasso" in spec


def test_req_kan_1604_global_group_lasso_and_spectral_terms() -> None:
    """REQ-KAN-1604: the loss exposes group-lasso and spectral terms."""

    centroids = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    assert global_group_lasso_penalty(centroids) == pytest.approx(6.0)

    affinity = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    same_cluster = one_hot_assignments([0, 0, 1], n_clusters=2)
    split_cluster = one_hot_assignments([0, 1, 1], n_clusters=2)

    assert spectral_constraint_regularization(same_cluster, affinity) == pytest.approx(0.0)
    assert spectral_constraint_regularization(split_cluster, affinity) == pytest.approx(2.0)


def test_req_kan_1604_clusterer_fit_sparsifies_constraint_memory() -> None:
    """REQ-KAN-1604: fitting prunes low-norm groups and measures sparsity."""

    constraint_memory = np.array(
        [
            [1.00, 0.00, 0.00],
            [1.10, 0.05, 0.00],
            [0.00, 1.00, 0.00],
            [0.05, 1.10, 0.00],
            [0.01, 0.00, 0.00],
            [0.00, 0.01, 0.00],
        ],
        dtype=np.float64,
    )
    clusterer = SparseKANClusterer(
        SparseKANConfig(
            n_clusters=4,
            group_lasso_weight=0.25,
            spectral_weight=0.5,
            prune_threshold=0.2,
            max_iter=8,
            seed=1604,
        )
    )

    result = clusterer.fit(constraint_memory)

    assert result.centroids.shape == (4, 3)
    assert result.assignments.shape == (6,)
    assert result.active_group_count < 4
    assert result.zero_group_count == 4 - result.active_group_count
    assert result.sparsity_ratio == pytest.approx(result.zero_group_count / 4)
    assert result.memory_compression_ratio > 1.0
    assert result.loss_components["global_group_lasso_penalty"] > 0.0
    assert result.loss_components["spectral_constraint_regularization"] >= 0.0
    assert result.loss_components["total_loss"] >= result.loss_components["reconstruction_loss"]
    assert np.all(np.isfinite(result.centroids))


def test_req_kan_1604_validation_rejects_malformed_inputs() -> None:
    """REQ-KAN-1604: sparse KAN clustering validates config and matrix shapes."""

    with pytest.raises(ValueError, match="n_clusters"):
        SparseKANConfig(n_clusters=0).validate()
    with pytest.raises(ValueError, match="weights"):
        SparseKANConfig(group_lasso_weight=-0.1).validate()
    with pytest.raises(ValueError, match="prune_threshold"):
        SparseKANConfig(prune_threshold=-0.1).validate()
    with pytest.raises(ValueError, match="max_iter"):
        SparseKANConfig(max_iter=0).validate()
    with pytest.raises(ValueError, match="2D"):
        SparseKANClusterer(SparseKANConfig(n_clusters=2)).fit(np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="rows"):
        SparseKANClusterer(SparseKANConfig(n_clusters=3)).fit(np.ones((2, 2)))


def test_scenario_kan_1604_builds_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1604: the sparse KAN compression artifact is complete."""

    artifact = build_experiment_1604_artifact()

    required_fields = {
        "schema",
        "status",
        "experiment",
        "experiment_id",
        "run_date",
        "spec",
        "module",
        "artifact_path",
        "sparse_kan_clustering_ready",
        "global_group_lasso_penalty",
        "spectral_constraint_regularization",
        "sparsity_ratio",
        "memory_compression_ratio",
        "constraints_compressed",
        "honest_verdict",
    }
    assert required_fields <= set(artifact)
    assert artifact["schema"] == "carnot.sparse_kan_clustering.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1604
    assert artifact["spec"] == ["REQ-KAN-1604", "SCENARIO-KAN-1604"]
    assert artifact["sparse_kan_clustering_ready"] is True
    assert artifact["sparsity_ratio"] > 0.0
    assert artifact["memory_compression_ratio"] > 1.0
    assert artifact["constraints_compressed"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    output_path = tmp_path / "experiment_1604_sparse_kan.json"
    written = write_experiment_1604_artifact(output_path)

    assert written == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact
