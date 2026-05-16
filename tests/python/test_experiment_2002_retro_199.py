import json
import os
import sys
from pathlib import Path

# Add the project root to sys.path to allow importing from scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.experiment_2002_retro_199 import generate_retro

def test_experiment_2002_retro_artifact():
    """SCENARIO-RETRO-199: Generates the 2026.05.199 milestone retrospective."""
    artifact_path = Path("results/operational_retro_2026_05_199.json")
    
    # Run the function to generate it
    if artifact_path.exists():
        artifact_path.unlink()
        
    generate_retro()
    
    assert artifact_path.exists(), "Retro artifact must exist"
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "carnot.operational_retro.v64"
    assert data["experiment"] == 2002
    assert data["honest_verdict"].startswith("terminal_")
    
    summary = data.get("summary", "")
    assert "execution wall time" in summary.lower() or "wall time" in summary.lower()
    assert "bottleneck" in summary.lower()
    assert "gec" in summary.lower()
    assert "clara-v" in summary.lower()
