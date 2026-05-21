import json
from pathlib import Path
from carnot.models.pinet_layer import evaluate_toy_projection_cases

def main():
    summary = evaluate_toy_projection_cases()
    complete = (
        float(summary["projection_error"]) <= 1e-5
        and int(summary["convergence_steps"]) <= 64
        and bool(summary["differentiable_projection"])
    )
    
    artifact = {
        "experiment_id": 2109,
        "projection_success": complete,
        "honest_verdict": "pinet_layer_projection_complete" if complete else "pinet_layer_projection_blocked",
        "details": summary
    }
    
    out_path = Path("results/experiment_2109_pinet_rescue.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
