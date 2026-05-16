"""Experiment 2052: KAN Z3 Verification."""
import json
import datetime
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from carnot.kan_z3 import verify_zero_false_accepts

def run_experiment():
    exp2051_path = _PROJECT_ROOT / "results" / "experiment_2051_kan_milp_verification.json"
    precondition_met = False
    if exp2051_path.exists():
        data = json.loads(exp2051_path.read_text())
        if data.get("status") == "complete" or data.get("honest_verdict", "").startswith("complete"):
            precondition_met = True
        else:
            print("Warning: exp2051 was not successful.")
    else:
        print("Warning: exp2051 results not found.")
        
    passed = verify_zero_false_accepts()
    
    # We produce the artifact
    artifact = {
        "schema": "carnot.kan_z3.v1",
        "experiment": 2052,
        "run_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "acceptance_gate_passed": passed,
        "acceptance_gate_criteria": "Z3 verification passes with zero false accepts.",
        "honest_verdict": f"complete: Z3 verification finished, passed={passed}"
    }
    
    out_path = _PROJECT_ROOT / "results" / "experiment_2052_kan_z3.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote artifact to {out_path}")

if __name__ == "__main__":
    run_experiment()
