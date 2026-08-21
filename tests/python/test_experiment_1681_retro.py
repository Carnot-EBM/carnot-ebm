import os
import json
import runpy
from pathlib import Path

from carnot.experiment_artifacts import resolve_experiment_artifact_path


def test_experiment_1681_retro_generates_valid_json():
    # REQ-PIPELINE-1681, SCENARIO-PIPELINE-1681
    script_path = os.path.join(os.path.dirname(__file__), "../../scripts/experiment_1681_retro.py")

    # Run the script
    runpy.run_path(script_path, run_name="__main__")

    # Verify JSON output
    out_file = resolve_experiment_artifact_path(
        "results/experiment_1681_retro.json",
        root=Path(__file__).resolve().parents[2],
    )
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert data["experiment"] == "1681_milestone_retro"
    assert data["schema"] == "milestone_retro_v2"
    assert "criteria_met" in data
    assert data["criteria_total"] == 4
    assert "experiment_honest_verdicts" in data
