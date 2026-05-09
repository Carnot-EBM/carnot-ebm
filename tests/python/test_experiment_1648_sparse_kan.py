"""Tests for Exp 1648 Sparse KAN spectral constraint grouping.

Spec: REQ-KAN-1648, SCENARIO-KAN-1648.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import experiment_1648_sparse_kan as exp


def test_req_kan_1648_spectral_grouping_keeps_nearby_rows_together() -> None:
    """REQ-KAN-1648: spectral grouping uses Laplacian structure before compression."""

    rows = exp.build_tier4_adaptive_landscape()
    grouping = exp.spectral_group_rows(rows, n_groups=3)

    assert grouping.labels.shape == (rows.shape[0],)
    assert grouping.embedding.shape == (rows.shape[0], 3)
    assert grouping.affinity.shape == (rows.shape[0], rows.shape[0])
    assert grouping.spectral_gap > 0.0
    assert grouping.spectral_grouping_penalty >= 0.0
    assert grouping.labels[0] == grouping.labels[1]
    assert grouping.labels[4] == grouping.labels[5]
    assert len(set(grouping.labels.tolist())) == 3


def test_req_kan_1648_grouped_sparse_kan_records_exact_compression_ratio() -> None:
    """REQ-KAN-1648: grouped Sparse KAN compression exposes direct compression_ratio."""

    rows = exp.build_tier4_adaptive_landscape()
    result = exp.compress_with_spectral_sparse_kan(rows, n_groups=3)

    assert result.dense_memory_bytes == rows.nbytes
    assert result.compressed_memory_bytes < result.dense_memory_bytes
    assert result.compression_ratio == pytest.approx(
        result.dense_memory_bytes / result.compressed_memory_bytes
    )
    assert result.sparse_kan_memory_compression_ratio > 1.0
    assert result.spectral_grouping_penalty >= 0.0
    assert result.active_group_count >= 1


def test_scenario_kan_1648_artifact_and_runner_write_required_json(tmp_path: Path) -> None:
    """SCENARIO-KAN-1648: the runner writes the completed compression artifact."""

    output_path = tmp_path / "experiment_1648_sparse_kan.json"
    artifact = exp.run_experiment(
        output_path=output_path,
        run_date="20260509",
        tests_run=["test_scenario_kan_1648_artifact_and_runner_write_required_json"],
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))

    assert persisted == artifact
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert artifact["schema"] == "carnot.spectral_sparse_kan.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1648
    assert artifact["spec_traces"] == ["REQ-KAN-1648", "SCENARIO-KAN-1648"]
    assert artifact["tier"] == "FR-11 Tier 4"
    assert artifact["compression_ratio"] == pytest.approx(
        artifact["dense_memory_bytes"] / artifact["compressed_memory_bytes"]
    )
    assert artifact["compression_ratio"] > 1.0
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_kan_1648_validation_and_input_guards(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """REQ-KAN-1648: validation catches schema drift and invalid grouping inputs."""

    with pytest.raises(ValueError, match="2D matrix"):
        exp.spectral_group_rows(np.array([1.0, 2.0]), n_groups=2)
    with pytest.raises(ValueError, match="between 1 and number of rows"):
        exp.spectral_group_rows(np.ones((2, 2)), n_groups=3)

    artifact = exp.build_artifact()
    missing = dict(artifact)
    del missing["compression_ratio"]
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact(missing)
    with pytest.raises(AssertionError, match="compression_ratio"):
        exp.validate_artifact(dict(artifact, compression_ratio=0.0))
    with pytest.raises(AssertionError, match="compression_ratio must be > 1.0"):
        exp.validate_artifact(
            dict(
                artifact,
                compressed_memory_bytes=artifact["dense_memory_bytes"],
                compression_ratio=1.0,
            )
        )
    with pytest.raises(AssertionError, match="spec_traces"):
        exp.validate_artifact(dict(artifact, spec_traces=[]))

    output_path = tmp_path / "cli_experiment_1648_sparse_kan.json"
    rc = exp.main(["--output", str(output_path), "--run-date", "20260509"])
    assert rc == 0
    assert "compression_ratio=" in capsys.readouterr().out
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "complete"
