"""Tests for Exp 2257 ODAR routing on real Tier 0 probe outputs.

Spec: REQ-ODAR-2257, SCENARIO-ODAR-2257
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from carnot.reporting.odar_real_benchmark import (
    DEFAULT_N_CORPUS,
    build_reasoning_corpus,
    run_benchmark,
)


def test_build_reasoning_corpus_is_balanced_100_examples() -> None:
    """REQ-ODAR-2257: the real-probe ODAR benchmark uses exactly 100 examples."""
    corpus = build_reasoning_corpus()

    assert len(corpus) == DEFAULT_N_CORPUS == 100
    assert sum(case.expected_correct for case in corpus) == 50
    assert sum(not case.expected_correct for case in corpus) == 50
    assert all("+" in case.response and "=" in case.response for case in corpus)


def test_run_benchmark_writes_validated_artifact(tmp_path) -> None:
    """SCENARIO-ODAR-2257: real Tier 0 probe EFE clears the ODAR gates."""
    output_path = tmp_path / "experiment_2257_odar_real_benchmark.json"

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [
            sys.executable,
            "scripts/experiment_2257_odar_real_benchmark.py",
            "--output-path",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    artifact = json.loads(output_path.read_text())
    assert artifact["n_corpus"] == 100
    assert artifact["odar_real_validated"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["compute_reduction_pct"] >= 25.0
    assert artifact["routing_overhead_ms"] <= 5.0
    assert artifact["accuracy_delta"] >= -2.0
    assert artifact["fast_path_fraction"] >= 0.4
    assert artifact["external_llm_calls"] == 0
    assert "odar_router_imported" in artifact["preconditions_checked"]
    assert "real_tier0_probe_outputs_collected" in artifact["preconditions_checked"]


def test_missing_odar_module_writes_blocked_artifact(tmp_path) -> None:
    """REQ-ODAR-2257: missing ODAR router reports blocked_odar_missing."""
    output_path = tmp_path / "blocked.json"

    artifact = run_benchmark(
        output_path=output_path,
        odar_path=tmp_path / "missing_odar_router.py",
        run_date="20260517",
    )

    assert artifact["honest_verdict"].startswith("blocked_odar_missing:")
    assert artifact["odar_real_validated"] is False
    assert artifact["n_corpus"] == 0
    assert artifact["compute_reduction_pct"] == 0.0
    assert artifact["routing_overhead_ms"] == 0.0
    assert "odar_router_import_failed" in artifact["preconditions_checked"]
