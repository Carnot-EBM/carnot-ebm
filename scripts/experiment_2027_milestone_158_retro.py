"""Exp 2027: Milestone .158 Retrospective."""
import json
import os
from pathlib import Path


def generate_retro(project_root: str) -> dict:
    """Generate the milestone 158 retrospective based on the pre-retro audit."""
    results_dir = Path(project_root) / "results"
    pre_retro_file = results_dir / "experiment_2026_milestone_158_pre_retro.json"
    
    seal_success = False
    stkan_success = False
    status = "failure"
    
    if pre_retro_file.exists():
        try:
            with open(pre_retro_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            seal_success = data.get("seal_tasks_completed", False)
            stkan_success = data.get("stkan_tasks_completed", False)
            status = data.get("status", "failure")
            
        except json.JSONDecodeError:
            pass
            
    # Formulate recommendations based on the outcome
    recommendations = []
    if not seal_success:
        recommendations.append("SEAL learning loop requires debugging and stabilization before next phase.")
    if not stkan_success:
        recommendations.append("STKAN prototype failed to complete; investigate constraints and model boundaries.")
    if seal_success and stkan_success:
        recommendations.append("Proceed to next milestone; SEAL and STKAN are stable.")
        
    honest_verdict = f"Milestone .158 Retrospective: SEAL success={seal_success}, STKAN success={stkan_success}. "
    if seal_success and stkan_success:
        honest_verdict += "All key capabilities validated."
    else:
        honest_verdict += "Failed capabilities must be addressed in the next cycle."
        
    return {
        "schema": "carnot.milestone_retro.v1",
        "milestone": "2026.05.158",
        "experiment_id": 2027,
        "status": status,
        "seal_success": seal_success,
        "stkan_success": stkan_success,
        "recommendations": recommendations,
        "retro_complete": True,
        "honest_verdict": honest_verdict
    }


def main() -> None:
    """Execute the retrospective generation and write the JSON deliverable."""
    root = os.environ.get("PROJECT_ROOT", ".")
    res = generate_retro(root)
    out_path = Path(root) / "results" / "experiment_2027_milestone_158_retro.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
