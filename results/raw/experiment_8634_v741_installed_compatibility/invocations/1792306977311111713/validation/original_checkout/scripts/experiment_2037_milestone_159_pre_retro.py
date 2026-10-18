"""Exp 2037: Milestone .159 Pre-Retro Audit."""
import json
import os
import re
from pathlib import Path


def audit_milestone_159(project_root: str) -> dict:
    """Audit conductor log to verify EBRM, KAN, and GEC tasks completed."""
    log_path = Path(project_root) / "ops" / "conductor-log.md"
    
    ebrm_completed = False
    kan_completed = False
    gec_completed = False
    
    ebrm_status = "UNKNOWN"
    kan_status = "UNKNOWN"
    gec_status = "UNKNOWN"
    
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Check for EBRM (Exp 2031)
            ebrm_matches = re.findall(r"Exp 2031: Continuous Latent EBRM[^|]*\|\s*([^|]+?)\s*\|", content)
            if ebrm_matches:
                ebrm_status = ebrm_matches[-1].strip()
                if ebrm_status == "OK":
                    ebrm_completed = True
                    
            # Check for KAN (Exp 2033)
            kan_matches = re.findall(r"Exp 2033: KAN Piecewise Affine[^|]*\|\s*([^|]+?)\s*\|", content)
            if kan_matches:
                kan_status = kan_matches[-1].strip()
                if kan_status == "OK":
                    kan_completed = True
                    
            # Check for GEC (Exp 2035)
            gec_matches = re.findall(r"Exp 2035: GEC Epsilon-Constraint[^|]*\|\s*([^|]+?)\s*\|", content)
            if gec_matches:
                gec_status = gec_matches[-1].strip()
                if gec_status == "OK":
                    gec_completed = True
                    
    success = ebrm_completed and kan_completed and gec_completed
    
    verdict = f"Audit complete. EBRM status: {ebrm_status}. KAN status: {kan_status}. GEC status: {gec_status}. "
    if success:
        verdict += "All tasks completed successfully."
    else:
        verdict += "Tasks did not complete."
        
    return {
        "experiment": 2037,
        "status": "success" if success else "failure",
        "ebrm_tasks_completed": ebrm_completed,
        "kan_tasks_completed": kan_completed,
        "gec_tasks_completed": gec_completed,
        "ebrm_final_status": ebrm_status,
        "kan_final_status": kan_status,
        "gec_final_status": gec_status,
        "honest_verdict": verdict.strip()
    }


def main() -> None:
    """Execute the audit and write the JSON deliverable."""
    root = os.environ.get("PROJECT_ROOT", ".")
    res = audit_milestone_159(root)
    out_path = Path(root) / "results" / "experiment_2037_milestone_159_pre_retro.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
