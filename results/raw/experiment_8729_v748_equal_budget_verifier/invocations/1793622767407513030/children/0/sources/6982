import json
from pathlib import Path
from carnot.pipeline.empirical_delta import compute_empirical_delta

def main():
    results_dir = Path("results")
    delta = compute_empirical_delta(results_dir)
    
    output = {
        "status": "complete",
        "empirical_delta": delta,
        "honest_verdict": "complete_calculated_empirical_delta"
    }
    
    with open(results_dir / "experiment_2150_empirical_delta.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
