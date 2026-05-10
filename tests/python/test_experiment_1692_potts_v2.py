import json
import subprocess
from pathlib import Path

def test_experiment_1692_potts_v2_rtl_exists():
    """REQ-POTTS-007-1: The top-level RTL SHALL be located at rtl/potts_machine_v2.v."""
    rtl_path = Path("rtl/potts_machine_v2.v")
    assert rtl_path.exists(), "rtl/potts_machine_v2.v must exist"
    content = rtl_path.read_text()
    assert "module potts_machine_v2" in content
    assert "always @(posedge clk" in content, "Must use standard synchronous design constraints (REQ-POTTS-007-2)"
    
    # Optional iverilog syntax check if available
    iverilog_path = subprocess.run(["which", "iverilog"], capture_output=True, text=True).stdout.strip()
    if iverilog_path:
        res = subprocess.run([iverilog_path, "-t", "null", str(rtl_path)], capture_output=True, text=True)
        assert res.returncode == 0, f"iverilog syntax error:\n{res.stderr}"

def test_experiment_1692_potts_v2_artifact():
    """REQ-POTTS-007-3: The task SHALL write results/experiment_1692_potts_export.json."""
    artifact_path = Path("results/experiment_1692_potts_export.json")
    assert artifact_path.exists(), "results/experiment_1692_potts_export.json must exist"
    data = json.loads(artifact_path.read_text())
    assert "status" in data
    assert data["status"] == "success"
