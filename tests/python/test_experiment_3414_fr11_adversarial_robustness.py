import json
from pathlib import Path
import sys

# Ensure the scripts directory is importable
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from scripts.experiment_3414_fr11_adversarial_robustness import run_experiment

def test_experiment_3414_output():
    """REQ-INFER-3414, SCENARIO-INFER-3414-001: Stress test for FR-11 with NUP phase transition and Latent Spills."""
    # Run the script
    run_experiment()
    
    # Check output JSON
    out_path = ROOT / "results" / "experiment_3414_fr11_adversarial_robustness.json"
    assert out_path.exists()
    
    data = json.loads(out_path.read_text())
    assert data["status"] == "success"
    assert "honest_verdict" in data
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in data["model_specs"]
    assert data["accuracy"] > 0
    assert data["calibration_limit"] > 0
