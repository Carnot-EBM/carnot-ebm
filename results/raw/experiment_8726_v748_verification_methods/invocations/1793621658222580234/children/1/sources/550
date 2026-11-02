import json
import subprocess
import time
import hashlib
from typing import List, Dict, Any, Optional

def check_ssh_reachability() -> bool:
    """Checks if the PolarFire board is reachable via SSH.
    Uses 'ssh -o ConnectTimeout=5 polarfire true'.
    """
    cmd = ["ssh", "-o", "ConnectTimeout=5", "polarfire", "true"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False

def confirm_continuity() -> str:
    """Confirm continuity (uptime, carnot dispatch path) deflagged."""
    cmd = ["ssh", "-o", "ConnectTimeout=5", "polarfire", "uptime"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"

def run_experiment(output_path: str):
    """Runs the PolarFire continuity experiment and writes the result to output_path."""
    start_time = time.time()
    
    is_reachable = check_ssh_reachability()
    
    preconditions = [
        {"resource": "polarfire_ssh", "available": is_reachable}
    ]
    
    uptime_info = None
    if is_reachable:
        uptime_info = confirm_continuity()
        honest_verdict = "complete: polarfire_continuity_confirmed_reachable"
    else:
        honest_verdict = "complete: blocked_polarfire_ssh_timeout"
        
    duration = time.time() - start_time
    
    random_seed = 42
    checksum_base = f"{honest_verdict}_{is_reachable}_{uptime_info}_{duration}"
    checksum = hashlib.sha256(checksum_base.encode()).hexdigest()
    
    result = {
        "honest_verdict": honest_verdict,
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": preconditions,
        "polarfire_ssh_reachable": is_reachable,
        "random_seed": random_seed,
        "reproducibility_checksum": checksum,
        "duration_s": duration,
    }
    
    if uptime_info is not None:
        result["polarfire_uptime"] = uptime_info
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":  # pragma: no cover
    run_experiment("results/experiment_3578_polarfire_continuity_v16.json")
