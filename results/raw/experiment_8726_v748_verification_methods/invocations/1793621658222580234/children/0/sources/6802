"""
Experiment 1704: KV260 Vivado Synthesis for the q=3 Potts machine.
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

def run_experiment():
    """Attempt Vivado synthesis for potts_sampler_v1.v and write artifact."""
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "experiment_1704_kv260.json"

    rtl_path = Path("hardware/kv260/potts_sampler_v1.v")
    if not rtl_path.exists():
        logger.error(f"RTL file not found: {rtl_path}")
        return

    # Check if Vivado is available
    try:
        result = subprocess.run(["which", "vivado"], capture_output=True, text=True)
        vivado_available = result.returncode == 0
    except FileNotFoundError:
        vivado_available = False

    artifact = {
        "vivado_available": vivado_available,
        "synthesis_success": False,
        "performance": None,
        "resource_utilization": None,
        "honest_verdict": "vivado_not_installed"
    }

    if vivado_available:
        try:
            tcl_script = Path("synth_potts_1704.tcl")
            tcl_script.write_text(f"read_verilog {rtl_path}\nsynth_design -top potts_sampler_v1\n")
            res = subprocess.run(["vivado", "-mode", "batch", "-source", "synth_potts_1704.tcl"], capture_output=True, text=True)
            if res.returncode == 0:
                artifact["synthesis_success"] = True
                artifact["honest_verdict"] = "vivado_synthesis_successful"
                artifact["performance"] = "unknown"
                artifact["resource_utilization"] = "unknown"
            else:
                artifact["honest_verdict"] = "vivado_synthesis_failed"
            if tcl_script.exists():
                tcl_script.unlink()
        except Exception as e:
            artifact["honest_verdict"] = f"error_{e}"

    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)

    logger.info(f"Artifact written to {artifact_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiment()
