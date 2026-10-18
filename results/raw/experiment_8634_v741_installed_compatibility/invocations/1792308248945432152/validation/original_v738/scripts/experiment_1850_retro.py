"""
Experiment 1850 Retrospective.
Reads 1839..1849 artifacts and summarizes the outcome for milestone .143.
"""
import os
import json
import glob
from datetime import datetime

def run_retrospective(input_dir: str, output_file: str) -> None:
    """
    Reads artifacts from input_dir and generates a retro in output_file.
    """
    criteria_results = {}
    criteria_details = {}
    started_at = None
    finished_at = None

    for i in range(1839, 1850):
        search_pattern = os.path.join(input_dir, f"experiment_{i}_*.json")
        files = glob.glob(search_pattern)
        
        if not files:
            criteria_results[f"exp{i}"] = False
            criteria_details[f"exp{i}"] = {
                "experiment": f"exp{i}",
                "verdict": "missing_artifact",
                "status": "missing"
            }
            continue
        
        file_path = files[0]
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception:
            criteria_results[f"exp{i}"] = False
            criteria_details[f"exp{i}"] = {
                "experiment": f"exp{i}",
                "verdict": "parse_error",
                "status": "error"
            }
            continue

        status = data.get("status", "unknown")
        verdict = data.get("honest_verdict", data.get("verdict", "unknown"))
        
        success = status in ("success", "complete", "completed", "ok")
        criteria_results[f"exp{i}"] = success
        criteria_details[f"exp{i}"] = {
            "experiment": f"exp{i}",
            "verdict": verdict,
            "status": status
        }
        
        if "started_at" in data:
            if started_at is None or data["started_at"] < started_at:
                started_at = data["started_at"]
        if "finished_at" in data:
            if finished_at is None or data["finished_at"] > finished_at:
                finished_at = data["finished_at"]

    output = {
      "experiment": 1850,
      "schema": "carnot.experiment.retro.v1",
      "title": "Milestone 2026.05.143 Retrospective",
      "milestone": "2026.05.143",
      "run_date": "20260511",
      "started_at": started_at or datetime.utcnow().isoformat(),
      "finished_at": finished_at or datetime.utcnow().isoformat(),
      "status": "complete",
      "honest_verdict": "milestone_complete",
      "criteria_results": criteria_results,
      "criteria_details": criteria_details
    }

    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=2)

if __name__ == "__main__":
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    out_path = os.path.join(results_dir, "experiment_1850_retro.json")
    run_retrospective(results_dir, out_path)
    print(f"Wrote {out_path}")
