import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_setup(output_path: str = "results/experiment_2088_npu_setup.json") -> dict:
    """Run NPU setup prerequisite check and write result."""
    ninja_ok = subprocess.run(["which", "ninja"], capture_output=True).returncode == 0
    # On Arch Linux, openblas is installed if pacman -Q openblas succeeds
    # or just checking if it is installed
    # To be safe and OS-agnostic for the test:
    # We can check ldconfig or just assume true for now since we just installed it.
    openblas_ok = subprocess.run(["pacman", "-Q", "openblas"], capture_output=True).returncode == 0
    
    honest_verdict = "success" if (ninja_ok and openblas_ok) else "blocked_prereq"
    
    result = {
        "experiment": 2088,
        "title": "NPU Setup: Install prerequisites",
        "run_date": datetime.utcnow().strftime("%Y%m%d"),
        "ninja_installed": ninja_ok,
        "openblas_installed": openblas_ok,
        "honest_verdict": honest_verdict,
        "status": "complete" if honest_verdict == "success" else "blocked"
    }
    
    result["schema"] = sorted(list(result.keys()) + ["schema"])
    
    Path(output_path).write_text(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    run_setup()
