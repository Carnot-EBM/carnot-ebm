"""Exp 2026: Milestone .158 Pre-Retro Audit."""
import json
import os
import re
from pathlib import Path


def audit_milestone_158(project_root: str) -> dict:
    """Audit conductor log to verify SEAL generation and STKAN tasks completed."""
    log_path = Path(project_root) / "ops" / "conductor-log.md"
    
    seal_completed = False
    stkan_completed = False
    seal_status = "UNKNOWN"
    stkan_status = "UNKNOWN"
    
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Check for Exp 2021 (SEAL) and Exp 2024 (STKAN) statuses
            seal_matches = re.findall(r"Exp 2021: SEAL Self-Adaptive Learning[^|]*\|\s*([^|]+?)\s*\|", content)
            if seal_matches:
                seal_status = seal_matches[-1].strip()
                if seal_status == "OK":
                    seal_completed = True
                    
            stkan_matches = re.findall(r"Exp 2024: STKAN Spatio-Temporal Constraint[^|]*\|\s*([^|]+?)\s*\|", content)
            if stkan_matches:
                stkan_status = stkan_matches[-1].strip()
                if stkan_status == "OK":
                    stkan_completed = True
                    
    success = seal_completed and stkan_completed
    
    verdict = f"Audit complete. SEAL status: {seal_status}. STKAN status: {stkan_status}. "
    if success:
        verdict += "Both tasks completed successfully."
    else:
        verdict += "Tasks did not complete."
        
    return {
        "experiment": 2026,
        "status": "success" if success else "failure",
        "seal_tasks_completed": seal_completed,
        "stkan_tasks_completed": stkan_completed,
        "seal_final_status": seal_status,
        "stkan_final_status": stkan_status,
        "honest_verdict": verdict.strip()
    }


def main() -> None:
    """Execute the audit and write the JSON deliverable."""
    root = os.environ.get("PROJECT_ROOT", ".")
    res = audit_milestone_158(root)
    out_path = Path(root) / "results" / "experiment_2026_milestone_158_pre_retro.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
