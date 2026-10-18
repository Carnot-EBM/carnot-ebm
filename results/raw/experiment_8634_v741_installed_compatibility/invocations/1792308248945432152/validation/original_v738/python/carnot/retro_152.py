import json
import glob
import os

def generate_retro(output_path: str, results_dir: str = "results"):
    files = glob.glob(os.path.join(results_dir, "experiment_19*.json"))
    
    valid_files = []
    for f in files:
        basename = os.path.basename(f)
        parts = basename.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            exp_num = int(parts[1])
            if 1944 <= exp_num <= 1954:
                valid_files.append(f)
    
    completed = 0
    blocked = 0
    failed = 0
    
    for f in valid_files:
        with open(f, 'r') as fh:
            data = json.load(fh)
            
        status_str = ""
        if "honest_verdict" in data and data["honest_verdict"]:
            status_str += str(data["honest_verdict"]).lower()
        if "status" in data and data["status"]:
            status_str += " " + str(data["status"]).lower()
        if "result" in data and data["result"]:
            status_str += " " + str(data["result"]).lower()
            
        if "blocked" in status_str:
            blocked += 1
        elif "fail" in status_str:
            failed += 1
        else:
            completed += 1
            
    recommendations = [
        "Address gate check failures proactively.",
        "Refine tasks related to blocked components."
    ]
    
    artifact = {
        "schema": "carnot.milestone_retro.v1",
        "milestone": 152,
        "completed_task_count": completed,
        "blocked_task_count": blocked,
        "failed_task_count": failed,
        "recommendations": recommendations
    }
    
    with open(output_path, 'w') as fh:
        json.dump(artifact, fh, indent=2)

if __name__ == "__main__":  # pragma: no cover
    generate_retro("results/experiment_1955_milestone_152_retro.json")
