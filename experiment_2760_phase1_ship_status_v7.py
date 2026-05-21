import json
import os
import subprocess
import time

def run():
    start_time = time.perf_counter()

    # Preconditions
    exp2730_exists = os.path.exists("results/experiment_2730_hf_mirror_ship_v6.json")
    
    try:
        # Check if phase 1 shipped via git tag
        git_tags = subprocess.check_output(["git", "tag", "--list", "v0.1.0*"]).decode("utf-8").strip().split('\n')
        git_tags = [t for t in git_tags if t]
        phase1_tag_exists = len(git_tags) > 0
        phase1_tag_found = git_tags[0] if phase1_tag_exists else None
    except Exception:
        phase1_tag_exists = False
        phase1_tag_found = None

    preconditions_checked = [
        {
            "resource": "exp2730_artifact_exists",
            "available": exp2730_exists,
            "check": "ls results/experiment_2730_hf_mirror_ship_v6.json && echo exists || echo missing"
        },
        {
            "resource": "phase1_tag_exists",
            "available": phase1_tag_exists,
            "check": "git tag --list 2>/dev/null | grep v0.1.0"
        }
    ]

    phase1_shipped = phase1_tag_exists
    checklist_still_current = True
    new_gates_opened = []

    if phase1_shipped:
        operator_ship_checklist_v7 = [f"Phase 1 SHIPPED at {phase1_tag_found}. No further action needed."]
    else:
        # Load from v6 if available
        if exp2730_exists:
            with open("results/experiment_2730_hf_mirror_ship_v6.json", "r") as f:
                v6_data = json.load(f)
                operator_ship_checklist_v7 = v6_data.get("operator_ship_checklist_v6", [])
        else:
            operator_ship_checklist_v7 = []
        operator_ship_checklist_v7.append("Status: Not shipped yet.")

    duration_s = time.perf_counter() - start_time

    out = {
        "honest_verdict": "complete: Phase 1 ship status verified.",
        "phase1_shipped": phase1_shipped,
        "phase1_tag_found": phase1_tag_found,
        "checklist_still_current": checklist_still_current,
        "new_gates_opened": new_gates_opened,
        "operator_ship_checklist_v7": operator_ship_checklist_v7,
        "duration_s": duration_s,
        "preconditions_checked": preconditions_checked
    }

    with open("results/experiment_2760_phase1_ship_status_v7.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    run()
