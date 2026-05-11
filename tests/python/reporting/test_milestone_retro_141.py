"""Test milestone .141 retrospective reporting.

References:
- REQ-REPORT-1824
- SCENARIO-REPORT-1824
"""

import json
from pathlib import Path

from carnot.reporting import milestone_retro_141


def test_milestone_retro_141_generation(tmp_path: Path):
    """Test generating the Phase 18 Final Evaluation Retrospective.
    
    Validates REQ-REPORT-1824.
    """
    # Create fake results directory and source artifacts
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Fake 1814 (blocked)
    (results_dir / "experiment_1814_dual_gpu_profiling.json").write_text(json.dumps({
        "status": "blocked",
        "honest_verdict": "blocked_gate_check_failed"
    }))
    
    # Fake 1818 (complete)
    (results_dir / "experiment_1818_vr_scaling.json").write_text(json.dumps({
        "status": "complete",
        "honest_verdict": "complete: SOTA verify-repair scaling evaluated"
    }))
    
    # Fake 1820 (complete, distillation)
    (results_dir / "experiment_1820_moe_distill.json").write_text(json.dumps({
        "status": "complete",
        "honest_verdict": "distillation_logged",
        "distillation_loss": 0.09
    }))
    
    # Fake 1822 (complete, RTL)
    (results_dir / "experiment_1822_rtl_synth.json").write_text(json.dumps({
        "status": "complete",
        "honest_verdict": "yosys_synthesis_clean"
    }))
    
    # Fake 1823 (complete, final eval)
    (results_dir / "experiment_1823_final_eval.json").write_text(json.dumps({
        "status": "complete",
        "honest_verdict": "complete: Phase 18 final evaluation completed",
        "self_learning_delta": 0.04
    }))
    
    # Missing 1816
    # Invalid JSON for 1819
    (results_dir / "experiment_1819_kan_latency.json").write_text("{ invalid_json ]")

    out_path = results_dir / "experiment_1824_retro.json"
    
    artifact = milestone_retro_141.run(results_dir=results_dir, out_path=out_path)
    
    assert artifact["experiment"] == 1824
    assert artifact["milestone"] == "2026.05.141"
    assert "top_3_gaps" in artifact
    assert len(artifact["top_3_gaps"]) == 3
    assert artifact["honest_verdict"] == "milestone_complete"
    assert "hardware_integration_results" in artifact
    assert "online_distillation_metrics" in artifact
    
    # Verify file was written
    assert out_path.exists()
    written_data = json.loads(out_path.read_text(encoding="utf-8"))
    assert written_data["experiment"] == 1824

def test_milestone_retro_141_defaults(monkeypatch, tmp_path: Path):
    """Test the default arguments."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # Patch REQ_ROOT to point to our tmp_path parent
    monkeypatch.setattr(milestone_retro_141, "REPO_ROOT", tmp_path)
    
    artifact = milestone_retro_141.run()
    assert artifact["experiment"] == 1824
    assert (results_dir / "experiment_1824_retro.json").exists()
