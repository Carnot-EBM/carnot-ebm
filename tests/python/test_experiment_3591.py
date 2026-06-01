import json
from pathlib import Path
import sys

# Ensure the scripts directory is in path for imports if needed
sys.path.append(str(Path(__file__).parent.parent.parent / "scripts"))

# Import the script
import experiment_3591_cross_domain_synthesis_v2

def test_generate_synthesis(tmp_path, monkeypatch):
    # Mock the results directory so we don't overwrite real data during tests
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    
    # We monkeypatch the output path
    original_generate = experiment_3591_cross_domain_synthesis_v2.generate_synthesis
    
    def mock_write(path_str):
        # Always write to our tmp_path
        pass

    # Actually, the easiest way to test is to let it write, but change the current working directory
    monkeypatch.chdir(tmp_path)
    
    # Create fake upstream artifacts
    (tmp_path / "results").mkdir(exist_ok=True)
    fake_json = json.dumps({"value": "fake"})
    
    (tmp_path / "results" / "experiment_3584_diagnose_329_null_positive_control.json").write_text(fake_json)
    
    # Create 3587 with factual signal
    (tmp_path / "results" / "experiment_3587_retrieval_nli_factual_grounding_verifier.json").write_text(json.dumps({
        "grounding_adds_factual_signal": {"value": True},
        "ensemble_with_grounding_auroc": {"value": 0.54},
        "confidence_baseline_auroc": {"value": 0.50}
    }))
    
    experiment_3591_cross_domain_synthesis_v2.generate_synthesis()
    
    out_file = tmp_path / "results" / "experiment_3591_cross_domain_synthesis_v2.json"
    assert out_file.exists()
    
    data = json.loads(out_file.read_text())
    assert data["honest_verdict"]["value"].startswith("complete: cross_domain_synthesis_v2")
    assert "math" in data["corrected_generalization_table"]["value"]
    assert data["verifier_value_generalizes"]["value"] == "math_only_earned"
