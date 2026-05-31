import json
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import it
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import experiment_3561_archive_v327_activate_v328

def test_experiment_3561_outputs(tmp_path, monkeypatch):
    """
    Tests that the script creates the expected JSON deliverables for archiving
    milestone .327 and activating .328.
    """
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)
    
    experiment_3561_archive_v327_activate_v328.main()
    
    # Verify experiment JSON
    exp_file = tmp_path / "results/experiment_3561_archive_v327_activate_v328.json"
    assert exp_file.exists()
    exp_data = json.loads(exp_file.read_text())
    
    assert exp_data["schema"] == "carnot.operational_retro.v67"
    assert exp_data["experiment"] == 3561
    assert exp_data["honest_verdict"].startswith("complete:")
    assert exp_data["archive_v327_activate_v328_ready"] is True
    assert exp_data["random_seed"] == 20260601
    assert "CONSOLIDATE: generalize the Route-1 positive to a SECOND discriminating CSP" in exp_data["top_forward_gap"]
    assert "external run pending = sole unmet gate" in exp_data["g2_status"]
    
    # Verify retro JSON
    retro_file = tmp_path / "results/operational_retro_2026_05_327.json"
    assert retro_file.exists()
    retro_data = json.loads(retro_file.read_text())
    assert retro_data["schema"] == "carnot.operational_retro.v67"
