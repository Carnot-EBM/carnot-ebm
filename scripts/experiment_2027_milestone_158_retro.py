#!/usr/bin/env python3
"""
Generate Milestone 158 Retrospective based on pre-retro audit.
"""

import json
import os
from pathlib import Path

def generate_retro(project_root: str) -> dict:
    """Generate the milestone 158 retrospective based on the pre-retro audit."""
    root = Path(project_root)
    results_dir = root / "results"
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    
    if not pre_retro_file.exists():
        return {
            "schema": "carnot.milestone_retro.v1",
            "milestone": "2026.05.158",
            "experiment_id": 2027,
            "status": "failure",
            "seal_success": False,
            "stkan_success": False,
            "recommendations": ["Ensure pre-retro artifact is generated."],
            "retro_complete": False,
            "honest_verdict": "Pre-retro artifact missing"
        }
        
    try:
        with pre_retro_file.open("r") as f:
            pre_retro = json.load(f)
    except Exception as e:
        return {
            "schema": "carnot.milestone_retro.v1",
            "milestone": "2026.05.158",
            "experiment_id": 2027,
            "status": "failure",
            "seal_success": False,
            "stkan_success": False,
            "recommendations": [f"Fix unreadable pre-retro artifact: {e}"],
            "retro_complete": False,
            "honest_verdict": "Pre-retro artifact unreadable"
        }
        
    seal_success = pre_retro.get("seal_tasks_completed", False)
    stkan_success = pre_retro.get("stkan_tasks_completed", False)
    
    recommendations = []
    if seal_success and stkan_success:
        recommendations.append("Both SEAL and STKAN goals were met. Proceed to next milestone.")
        verdict = "Milestone .158 retrospective complete. Both SEAL and STKAN succeeded."
    elif not seal_success and not stkan_success:
        recommendations.append("Investigate GATE_BLOCK in SEAL loop and failures in STKAN prototype before attempting to proceed.")
        verdict = "Milestone .158 retrospective complete. Both SEAL and STKAN failed."
    else:
        if not seal_success:
            recommendations.append("SEAL tasks failed. Investigate continuous learning loop gate blocks.")
        if not stkan_success:
            recommendations.append("STKAN tasks failed. Debug spatio-temporal constraint model errors.")
        verdict = "Milestone .158 retrospective complete. Partial success."
        
    result = {
        "schema": "carnot.milestone_retro.v1",
        "milestone": "2026.05.158",
        "experiment_id": 2027,
        "status": "complete",
        "seal_success": seal_success,
        "stkan_success": stkan_success,
        "recommendations": recommendations,
        "retro_complete": True,
        "honest_verdict": verdict
    }
    
    return result

def main():
    """Execute the retrospective generation and write the JSON deliverable."""
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    result = generate_retro(project_root)
    
    out_file = Path(project_root) / "results" / "experiment_2027_milestone_158_retro.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
        
if __name__ == "__main__":
    main()
