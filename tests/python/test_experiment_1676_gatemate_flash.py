import json
from pathlib import Path
from carnot.experiment_1676_gatemate_flash import GateMateFlashArtifact

def test_gatemate_flash_artifact_schema():
    artifact_path = Path("results/experiment_1676_gatemate_flash.json")
    assert artifact_path.exists(), "Artifact file must exist"
    
    # Load and validate schema
    artifact = GateMateFlashArtifact.load(str(artifact_path))
    
    assert artifact.schema == "carnot.gatemate_flash.v1"
    assert "GateMateA1-EVB-2M" in artifact.board
    assert artifact.idcode_verified == "0x20000001"
    assert "yosys_version" in artifact.toolchain
    assert "nextpnr_version" in artifact.toolchain
    assert "openFPGALoader_version" in artifact.toolchain
    assert "chtype -map t:CC_LUT3 CC_LUT4" in artifact.yosys_invocation
    
    assert artifact.synthesis_completed is True
    assert artifact.pnr_completed is True
    assert artifact.lut_utilization < 0.30
    assert artifact.flash_succeeded is True
    assert artifact.bitstream_size_bytes > 0
    assert artifact.acceptance_gate_passed is True
    assert artifact.honest_verdict.startswith("success:") or artifact.honest_verdict.startswith("shipped:") or artifact.honest_verdict.startswith("complete:")
