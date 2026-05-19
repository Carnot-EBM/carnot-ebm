import json
from pathlib import Path
from carnot.experiment_2453_gatemate_ising_synthesis_v2 import GateMateIsingSynthesisV2Artifact

def test_experiment_2453_artifact():
    artifact_path = Path("results/experiment_2453_gatemate_ising_synthesis_v2.json")
    assert artifact_path.exists(), "Artifact file must exist"
    
    artifact = GateMateIsingSynthesisV2Artifact.load(str(artifact_path))
    
    assert artifact.honest_verdict.startswith("terminal_")
    assert artifact.synthesis_completed is True
    assert artifact.pnr_completed is True
    assert artifact.gatemate_bitstream_flashed is True
    assert "130/40960" in artifact.lut_utilization
    assert "Passively cooled" in artifact.thermal_note
    assert "Yosys 0.64+149" in artifact.yosys_version
    assert artifact.duration_s > 30.0
    assert len(artifact.preconditions_checked) == 4
