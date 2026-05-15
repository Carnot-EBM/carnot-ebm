import json
from pathlib import Path
import sys
import pytest

# Ensure scripts directory is in path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.kanele_hardware_accounting as kha

def test_compute_kan_metrics():
    metrics = kha.compute_kan_metrics(n_inputs=64, k_splines=16)
    assert metrics["rm_per_inference"] == 64 * 16
    assert metrics["bop_per_inference"] == 64 * 16 * 8
    
def test_run_cem_optimizer():
    final_energy = kha.run_cem_optimizer()
    assert isinstance(final_energy, float)

def test_write_accounting_report(tmp_path):
    metrics = {
        "rm_per_inference": 1024,
        "bop_per_inference": 8192,
        "nabs_per_inference": 3136,
        "total_luts_estimate": 11776
    }
    report_path = tmp_path / "report.md"
    kha.write_accounting_report(metrics, report_path)
    
    content = report_path.read_text()
    assert "CEM n=64 Hardware Accounting Report" in content
    assert "RM (Routing Muxes) per inference: 1024" in content
    assert "BOP (Bit Operations) per inference: 8192" in content

def test_main(tmp_path, monkeypatch):
    # Mock PROJECT_ROOT so it writes to tmp_path
    monkeypatch.setattr(kha, "PROJECT_ROOT", tmp_path)
    
    kha.main()
    
    report_path = tmp_path / "docs" / "research-notes" / "cem_n64_hardware_accounting.md"
    assert report_path.exists()
    
    artifact_path = tmp_path / "results" / "experiment_1730.json"
    assert artifact_path.exists()
    
    artifact = json.loads(artifact_path.read_text())
    assert artifact["experiment_id"] == "1730"
    assert artifact["status"] == "complete"
    assert artifact["hardware_execution_claim"] is False
    assert artifact["honest_verdict"].startswith("complete")
    assert "metrics" in artifact
