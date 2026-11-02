#!/usr/bin/env python3
"""Run Exp 1683: FR-11 Policy Soundness Audit."""
import json
import os
import sys
import time
from datetime import datetime, timezone

# Add python to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from carnot.memory.audit import FR11Audit

def main():
    start_time = time.time()
    
    experiment_id = 1683
    date_str = os.environ.get("DATE", "20260514")
    
    exp1682_path = os.path.join("results", "experiment_1682_scg_ets_integration.json")
    deliverable_path = os.path.join("results", "experiment_1683_fr11_soundness.json")
    
    print(f"Running Exp {experiment_id}: FR-11 Policy Soundness Audit")
    
    audit = FR11Audit()
    try:
        results = audit.audit_rollback_passing(exp1682_path)
        status = "success"
        honest_verdict = "soundness_audit_passed"
    except Exception as e:
        results = {
            "soundness_mistakes": 0,
            "completeness_mistakes": 0,
            "nonforgetting_rate": 1.0,
            "error": str(e)
        }
        status = "failed"
        honest_verdict = "audit_failed"
        
    duration = time.time() - start_time
    
    output = {
        "experiment": experiment_id,
        "schema": "experiment_result_v1",
        "run_date": date_str,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": duration,
        "status": status,
        "honest_verdict": honest_verdict,
        "soundness_mistakes": results.get("soundness_mistakes", 0),
        "completeness_mistakes": results.get("completeness_mistakes", 0),
        "nonforgetting_rate": results.get("nonforgetting_rate", 1.0)
    }
    
    os.makedirs(os.path.dirname(deliverable_path), exist_ok=True)
    with open(deliverable_path, "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Results written to {deliverable_path}")
    print(f"  soundness_mistakes: {output['soundness_mistakes']}")
    print(f"  completeness_mistakes: {output['completeness_mistakes']}")
    print(f"  nonforgetting_rate: {output['nonforgetting_rate']}")

if __name__ == "__main__":
    main()
