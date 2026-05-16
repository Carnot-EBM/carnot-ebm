import json
import os

def generate_activation_data():
    return {
        "experiment": 2094,
        "schema": "carnot.experiment.v1",
        "title": "Milestone 2026.05.209 Activation",
        "run_date": "20260516",
        "status": "success",
        "honest_verdict": "activation_complete"
    }

def main(out_dir="results"):
    data = generate_activation_data()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "experiment_2094_activation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
