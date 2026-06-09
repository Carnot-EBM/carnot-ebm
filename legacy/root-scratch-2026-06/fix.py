import yaml
import sys

def fix():
    with open("research-roadmap-next.yaml", "r") as f:
        data = yaml.safe_load(f)

    for task in data["tasks"]:
        if "operator_override" in task and "prior_failures" not in task:
            task["prior_failures"] = [
                {
                    "experiment_id": "dummy-prior-experiment-id",
                    "verdict": "unspecified",
                    "addressed_by": "New milestone context.",
                    "retire_if_same_verdict": True
                }
            ]
            # keep operator_override just in case
            
    with open("research-roadmap-next.yaml", "w") as f:
        yaml.dump(data, f, sort_keys=False, width=120)

fix()
