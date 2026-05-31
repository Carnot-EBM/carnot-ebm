import json
import os
import sys
import pytest
from pathlib import Path

# Add the scripts directory to the path so we can import it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import experiment_3539_archive_v325_activate_v326

def test_experiment_3539_outputs(tmp_path, monkeypatch):
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    # Create results directory
    (tmp_path / "results").mkdir()
    
    experiment_3539_archive_v325_activate_v326.main()
    
    # Verify experiment JSON
    exp_file = tmp_path / "results/experiment_3539_archive_v325_activate_v326.json"
    assert exp_file.exists()
    exp_data = json.loads(exp_file.read_text())
    
    assert exp_data["schema"] == "carnot.operational_retro.v67"
    assert exp_data["experiment"] == 3539
    assert exp_data["honest_verdict"].startswith("complete:")
    assert exp_data["archive_v325_activate_v326_ready"] is True
    assert exp_data["random_seed"] == 20260601
    assert "RESCUE the graph-coloring positive cleanly" in exp_data["top_forward_gap"]
    assert "external run pending = sole unmet gate" in exp_data["g2_status"]
    
    # Verify retro JSON
    retro_file = tmp_path / "results/operational_retro_2026_05_325.json"
    assert retro_file.exists()
    retro_data = json.loads(retro_file.read_text())
    assert retro_data["schema"] == "carnot.operational_retro.v67"

