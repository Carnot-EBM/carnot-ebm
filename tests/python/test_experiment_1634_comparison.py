"""Tests for Exp 1634 Pi-Net vs T-SKM comparison.

Spec refs: REQ-KONA-038, SCENARIO-KONA-038.
"""

import json
from pathlib import Path

from scripts.experiment_1634_comparison import (
    DEFAULT_OUTPUT_PATH,
    EXPERIMENT_ID,
    SCHEMA,
    SPEC_REFS,
    build_artifact,
    run_comparison,
    run_experiment,
    validate_artifact,
)


def test_run_comparison() -> None:
    """Test the raw comparison function."""
    result = run_comparison(iterations=1)
    assert isinstance(result["pinet_faster_than_tskm"], bool)
    assert isinstance(result["latency_diff"], float)
    assert isinstance(result["skm_avg_latency"], float)
    assert isinstance(result["pinet_avg_latency"], float)


def test_build_artifact() -> None:
    """Test artifact generation and validation."""
    tests = ["test_build_artifact"]
    artifact = build_artifact(tests_run=tests)

    assert artifact["status"] == "complete"
    assert artifact["schema"] == SCHEMA
    assert artifact["experiment_id"] == EXPERIMENT_ID
    assert artifact["spec_refs"] == SPEC_REFS
    assert "pinet_faster_than_tskm" in artifact
    assert "latency_diff" in artifact
    assert "honest_verdict" in artifact
    assert artifact["tests_run"] == tests

    validate_artifact(artifact)


def test_run_experiment_writes_json(tmp_path: Path) -> None:
    """Test that the entrypoint writes the required JSON artifact."""
    output_path = tmp_path / "results" / "experiment_1634_pinet_vs_tskm.json"
    result = run_experiment(output_path=output_path, tests_run=["test_run_experiment_writes_json"])

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved["schema"] == SCHEMA
    assert saved["experiment_id"] == EXPERIMENT_ID
    assert saved["status"] == "complete"
    assert saved["tests_run"] == ["test_run_experiment_writes_json"]
    assert saved["honest_verdict"] == result["honest_verdict"]
