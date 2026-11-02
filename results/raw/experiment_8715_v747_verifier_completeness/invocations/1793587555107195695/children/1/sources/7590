"""
Experiment 3608: PolarFire Continuity Check v18.
Checks if the PolarFire SoC board is reachable via SSH.
Records uptime and carnot dispatch path if reachable.
"""

import json
import os
import subprocess
import time
import hashlib

RESULTS_FILE = "results/experiment_3608_polarfire_continuity_v18.json"

def get_reproducibility_checksum():
    """Simple drift detection checksum based on this file's contents."""
    try:
        with open(__file__, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "unknown"

def run_experiment():
    start_time = time.time()
    
    # Precondition: check SSH reachability
    ssh_cmd = ["ssh", "-o", "ConnectTimeout=5", "polarfire", "true"]
    print(f"Running precondition check: {' '.join(ssh_cmd)}")
    
    result = subprocess.run(ssh_cmd, capture_output=True, text=True)
    ssh_reachable = result.returncode == 0
    
    artifact = {
        "inference_substrate": "hardware_smoke",
        "preconditions_checked": {
            "ssh_reachability_command": " ".join(ssh_cmd),
            "exit_code": result.returncode
        },
        "polarfire_ssh_reachable": ssh_reachable,
        "random_seed": 42,  # Determinism precondition
        "reproducibility_checksum": get_reproducibility_checksum()
    }
    
    if ssh_reachable:
        artifact["honest_verdict"] = "complete: polarfire_continuity_confirmed_reachable"
        
        # Gather continuity evidence
        try:
            uptime_res = subprocess.run(["ssh", "polarfire", "uptime"], capture_output=True, text=True)
            if uptime_res.returncode == 0:
                artifact["polarfire_uptime"] = uptime_res.stdout.strip()
                
            dispatch_res = subprocess.run(["ssh", "polarfire", "which carnot"], capture_output=True, text=True)
            if dispatch_res.returncode == 0:
                artifact["polarfire_dispatch_path"] = dispatch_res.stdout.strip()
            elif dispatch_res.returncode == 1:
                # 'which' returns 1 if not found
                artifact["polarfire_dispatch_path"] = "not_found"
                
        except Exception as e:
            print(f"Error gathering continuity evidence: {e}")
            
    else:
        artifact["honest_verdict"] = "complete: blocked_polarfire_ssh_timeout"
        artifact["preconditions_checked"]["stderr"] = result.stderr.strip()
        
    end_time = time.time()
    artifact["duration_s"] = round(end_time - start_time, 2)
    
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(artifact, f, indent=2)
        
    print(f"Wrote artifact to {RESULTS_FILE}")
    print(f"Verdict: {artifact['honest_verdict']}")

if __name__ == "__main__":
    run_experiment()
