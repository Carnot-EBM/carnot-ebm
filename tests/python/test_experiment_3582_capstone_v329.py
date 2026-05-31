import json
import os
from pathlib import Path

def test_experiment_3582():
    # Run the script
    exit_code = os.system("python3 scripts/experiment_3582_capstone_v329.py")
    assert exit_code == 0
    
    # Check the output artifact
    results_dir = Path("results")
    out_path = results_dir / "experiment_3582_capstone_v329.json"
    assert out_path.exists()
    
    with open(out_path) as f:
        data = json.load(f)
        
    assert "complete: capstone_v329_verifier_value_math_only_domain_bound_code_False_facts_False_paper_ready_true" in data["honest_verdict"]
    assert data["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert data["code_generalizes"] is False
    assert data["facts_generalize"] is False
    assert data["second_pair_of_eyes_lift_real"] is True
    assert data["verifier_value_scope"] == "math_only_domain_bound"
    assert data["paper_ready"] is True
    
    assert isinstance(data["cited_upstream_artifacts"], list)
    # Check that 3574 is excluded
    assert not any("3574" in artifact for artifact in data["cited_upstream_artifacts"])
    assert any("3573" in artifact for artifact in data["cited_upstream_artifacts"])
    assert any("3575" in artifact for artifact in data["cited_upstream_artifacts"])
    assert any("3576" in artifact for artifact in data["cited_upstream_artifacts"])
    
    assert "random_seed" in data
    assert "reproducibility_checksum" in data
    assert "duration_s" in data
    assert isinstance(data["duration_s"], float)
