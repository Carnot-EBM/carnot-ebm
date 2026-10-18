"""Experiment 2115: NEXUS Framework Grounding.

Bridges Carnot's symbolic verifiers (Z3) into the ActFocus reward trace.
Decouples physical feasibility from safety specifications during the CSL feedback loop.

Spec: REQ-NEXUS-2115, SCENARIO-NEXUS-2115
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from carnot.pipeline.nexus_grounding import NexusGroundingVerifier

DELIVERABLE = "results/experiment_2115_nexus.json"

def run_experiment() -> dict:
    nexus = NexusGroundingVerifier()
    
    # ActFocus reward trace with safe arithmetic
    safe_trace = "I will proceed with the action. The load is 50. 50 + 50 = 100, which is within bounds."
    safe_eval = nexus.evaluate_pre_action_safety(safe_trace, {"max_load": "100"})
    
    # ActFocus reward trace with unsafe arithmetic
    unsafe_trace = "I will proceed. The load is 50. 50 + 50 = 150, which is within bounds."
    unsafe_eval = nexus.evaluate_pre_action_safety(unsafe_trace, {"max_load": "100"})
    
    return {
        "experiment_id": "2115",
        "schema": "carnot.nexus.v1",
        "description": "NEXUS framework grounding pre-action defense into ActFocus reward trace.",
        "results": {
            "safe_trace": {
                "is_safe": safe_eval.is_safe,
                "risk_score": safe_eval.risk_score,
                "feasibility_decoupled": safe_eval.feasibility_decoupled
            },
            "unsafe_trace": {
                "is_safe": unsafe_eval.is_safe,
                "risk_score": unsafe_eval.risk_score,
                "feasibility_decoupled": unsafe_eval.feasibility_decoupled,
                "violations": unsafe_eval.violations
            }
        }
    }

def main():
    result = run_experiment()
    os.makedirs(os.path.dirname(DELIVERABLE), exist_ok=True)
    with open(DELIVERABLE, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Artifact written to {DELIVERABLE}")

if __name__ == "__main__":
    main()
