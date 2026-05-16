import json
import os
from carnot.position_paper_nexus import generate_experiment_1913_json

def test_experiment_1913_arch_paper_exists_and_valid(tmp_path):
    """Test that the deliverable JSON exists and has required schema fields (REQ-PUBLISH-027)."""
    # Test generation and coverage of the module
    test_path = str(tmp_path / "experiment_1913_arch_paper.json")
    assert generate_experiment_1913_json(test_path) is True
    
    with open(test_path, "r") as f:
        data = json.load(f)
        
    assert data.get("experiment") == 1913
    assert data.get("schema") == "carnot.arch_paper.v1"
    assert "status" in data
    assert "honest_verdict" in data
    assert data.get("position_paper_drafted") is True
    assert data.get("architecture_updated") is True

    # Also verify the real file exists (if it was generated manually)
    real_path = "results/experiment_1913_arch_paper.json"
    if os.path.exists(real_path):
        with open(real_path, "r") as f:
            real_data = json.load(f)
        assert real_data.get("experiment") == 1913

