import json
import glob
from pathlib import Path

def parse_experiment_files(results_dir: str):
    """Parse experiment JSON files and aggregate honest_verdict."""
    files = glob.glob(f"{results_dir}/experiment_*.json")
    valid_files = []
    for f in files:
        name = Path(f).name
        parts = name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            exp_id = int(parts[1])
            if 1759 <= exp_id <= 1769:
                valid_files.append(f)
    
    results = {}
    for file in sorted(valid_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                results[Path(file).name] = data.get('honest_verdict', 'unknown_verdict')
        except Exception:
            results[Path(file).name] = 'error_parsing'
            
    return results

def generate_retro(results_dir: str = "results"):
    """Generate the retrospective artifact for experiment 1770."""
    results = parse_experiment_files(results_dir)
    retro = {
        "experiment": 1770,
        "schema": "carnot.retrospective.v1",
        "title": "Phase 4 Operations Retrospective",
        "run_date": "20260511",
        "status": "success",
        "honest_verdict": "phase_4_operations_aggregated",
        "aggregated_results": results
    }

    out_path = Path(results_dir) / "experiment_1770_retro.json"
    with open(out_path, "w") as f:
        json.dump(retro, f, indent=2)

if __name__ == "__main__":
    generate_retro()
