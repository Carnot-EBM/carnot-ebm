"""Experiment 2064 Audit Module."""
import os
import json
import subprocess
from typing import Dict, Any

def audit_deliverables(results_dir: str = "results") -> Dict[str, bool]:
    """Check the presence of deliverables from 2053 to 2063."""
    required_exps = [
        2053, 2054, 2055, 2056, 2057, 2058, 2059, 2060, 2061, 2062, 2063
    ]
    status = {str(exp): False for exp in required_exps}
    if not os.path.exists(results_dir):
        return status
    
    files = os.listdir(results_dir)
    for exp in required_exps:
        prefix = f"experiment_{exp}_"
        if any(f.startswith(prefix) and f.endswith(".json") for f in files):
            status[str(exp)] = True
    return status

def verify_verifier_e2e() -> bool:
    """Verify E2E tests pass for the new verifier architecture."""
    try:
        # In this project, running pytest tests/python -q is the standard command
        # as indicated by the user prompt "Run: .venv/bin/pytest tests/python -q"
        res = subprocess.run(
            [".venv/bin/pytest", "tests/python", "-q"],
            capture_output=True,
            text=True
        )
        return res.returncode == 0
    except Exception:
        return False

def run_experiment_2064_audit(results_dir: str = "results", out_file: str = "results/experiment_2064_audit.json") -> Dict[str, Any]:
    """Run the audit and save to JSON."""
    status = audit_deliverables(results_dir)
    tests_passed = verify_verifier_e2e()
    missing = [exp for exp, present in status.items() if not present]
    
    result = {
        "experiment": 2064,
        "deliverables_status": status,
        "missing_deliverables": missing,
        "e2e_tests_passed": tests_passed,
        "audit_passed": len(missing) == 0 and tests_passed,
    }
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
        
    return result

if __name__ == "__main__":
    run_experiment_2064_audit()
