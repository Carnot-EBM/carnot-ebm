"""Tests for Exp 1599 KANELÉ hardware LUT-complexity accounting.

Spec refs: REQ-KAN-1599, SCENARIO-KAN-1599.
"""

import json
from pathlib import Path

from carnot.analysis.kanele_audit_1599 import compute_kan_metrics, run_kanele_audit


def test_compute_kan_metrics() -> None:
    """Verify the estimation of RM, BOP, and NABS."""
    metrics = compute_kan_metrics(n_inputs=32, k_splines=16)
    # basis_evaluations = 32 * 16 = 512
    # rm = 512
    # bop = 512 * 8 = 4096
    # index_add_shift = 64
    # interpolation_add_sub = 1024
    # accumulation_adds = 32 * 15 = 480
    # nabs = 64 + 1024 + 480 = 1568
    assert metrics["rm_per_inference"] == 512
    assert metrics["bop_per_inference"] == 4096
    assert metrics["nabs_per_inference"] == 1568


def test_run_kanele_audit(tmp_path: Path) -> None:
    """Verify that the SCENARIO-KAN-1599 audit artifact is generated correctly."""
    artifact_path = tmp_path / "experiment_1599_kanele_audit.json"
    artifact = run_kanele_audit(deliverable_path=artifact_path)

    assert artifact_path.exists()

    loaded = json.loads(artifact_path.read_text())
    assert loaded["experiment"] == 1599
    assert loaded["hardware_execution_confirmed"] is False
    assert loaded["rm_per_inference"] == 512
    assert loaded["bop_per_inference"] == 4096
    assert loaded["nabs_per_inference"] == 1568
    assert "complete_no_synthesis" in loaded["honest_verdict"]
