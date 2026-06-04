"""
Test for Experiment 3781 EDLM Feasibility Scoping
"""
import sys
import json
from pathlib import Path

# Add scripts dir to path to import the script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from experiment_3781_edlm_next_thesis_feasibility_scoping import generate_feasibility_artifact, main

def test_generate_feasibility_artifact():
    artifact = generate_feasibility_artifact()
    assert "honest_verdict" in artifact
    assert artifact["honest_verdict"] == "complete: edlm_feasibility_scoped_residual_corrector_not_blocked_by_either_negative_minimal_kill_gate_designed_operator_decision_surface_loop_does_not_commit"
    assert "inference_substrate" in artifact
    assert "edlm_mechanism_summary" in artifact
    assert "why_not_blocked_by_energy_negatives" in artifact
    assert "prerequisites" in artifact
    assert "minimal_kill_gate_design" in artifact
    assert "compute_estimate_gpu_hours" in artifact
    assert "operator_decision_framing" in artifact
    assert "loop_does_not_commit" in artifact
    assert artifact["loop_does_not_commit"] is True
    assert "random_seed" in artifact
    assert artifact["random_seed"] == 3781
    assert "reproducibility_checksum" in artifact
    assert "duration_s" in artifact
    assert artifact["duration_s"] >= 0

def test_main(tmp_path, monkeypatch):
    import experiment_3781_edlm_next_thesis_feasibility_scoping
    
    # Patch Path to use tmp_path for 'results' directory
    original_path = Path
    
    class MockPath(type(Path())):
        def __new__(cls, *args, **kwargs):
            if args and args[0] == "results":
                return original_path(tmp_path) / "results"
            return original_path(*args, **kwargs)
            
    monkeypatch.setattr(experiment_3781_edlm_next_thesis_feasibility_scoping, "Path", MockPath)
    
    main()
    
    output_file = tmp_path / "results" / "experiment_3781_edlm_next_thesis_feasibility_scoping.json"
    assert output_file.exists()
    
    with open(output_file, "r") as f:
        data = json.load(f)
        
    assert data["honest_verdict"] == "complete: edlm_feasibility_scoped_residual_corrector_not_blocked_by_either_negative_minimal_kill_gate_designed_operator_decision_surface_loop_does_not_commit"
