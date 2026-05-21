import json
import os
from carnot.phase1_ship_gate import evaluate_gate

if __name__ == "__main__":
    result = evaluate_gate()
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2431_phase1_ship_gate_v4.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Deliverable written to results/experiment_2431_phase1_ship_gate_v4.json")
