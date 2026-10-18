import json
import glob
from pathlib import Path

def parse_experiment_files(results_dir: str):
    """Parse experiment JSON files and aggregate honest_verdict and metrics."""
    files = glob.glob(f"{results_dir}/experiment_*.json")
    valid_files = []
    for f in files:
        name = Path(f).name
        parts = name.split('_')
        if len(parts) >= 2 and parts[1].isdigit():
            exp_id = int(parts[1])
            if 1799 <= exp_id <= 1810:
                valid_files.append(f)
    
    results = {}
    for file in sorted(valid_files):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                
                # Extract requested fields
                honest_verdict = data.get('honest_verdict', 'unknown_verdict')
                
                metrics = {}
                for key in ['accuracy_delta', 'dpo_improvement_pp']:
                    if key in data:
                        metrics[key] = data[key]
                        
                results[Path(file).name] = {
                    'honest_verdict': honest_verdict,
                    'metrics': metrics
                }
        except Exception:
            results[Path(file).name] = {
                'honest_verdict': 'error_parsing',
                'metrics': {}
            }
            
    return results

def generate_retro(results_dir: str = "results"):
    """Generate the retrospective artifact for experiment 1811."""
    results = parse_experiment_files(results_dir)
    
    retro = {
        "experiment": 1811,
        "schema": "carnot.retrospective.v1",
        "title": "Phase-16 Finding Summary",
        "run_date": "2026-05-11",
        "status": "success",
        "honest_verdict": "phase_16_aggregated",
        "aggregated_results": results
    }

    out_path = Path(results_dir) / "experiment_1811_retro.json"
    with open(out_path, "w") as f:
        json.dump(retro, f, indent=2)

if __name__ == "__main__":
    generate_retro()
