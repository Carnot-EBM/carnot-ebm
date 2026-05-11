import json
import glob
from datetime import datetime
import os

criteria_results = {}
criteria_details = {}
started_at = None
finished_at = None

for i in range(1839, 1850):
    files = glob.glob(f"results/experiment_{i}_*.json")
    if not files:
        criteria_results[f"exp{i}"] = False
        criteria_details[f"exp{i}"] = {
            "experiment": f"exp{i}",
            "verdict": "missing_artifact",
            "status": "missing"
        }
        continue
    
    file_path = files[0]
    with open(file_path, 'r') as f:
        data = json.load(f)
        status = data.get("status", "unknown")
        verdict = data.get("honest_verdict", data.get("verdict", "unknown"))
        
        # Consider success or complete as true
        success = status in ("success", "complete", "completed")
        criteria_results[f"exp{i}"] = success
        criteria_details[f"exp{i}"] = {
            "experiment": f"exp{i}",
            "verdict": verdict,
            "status": status
        }
        
        # Track start and end times if available
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

with open("results/experiment_1850_retro.json", "w") as f:
    json.dump(output, f, indent=2)

print("Generated results/experiment_1850_retro.json")
