import json
import glob
import os
from pathlib import Path

def generate_retro(input_dir, output_path):
    """
    Generate Phase 4 retrospective from experiment 1785 to 1797 JSONs.
    """
    input_path = Path(input_dir)
    results = {}
    
    # We want to match experiment_1785_*.json through experiment_1797_*.json
    for i in range(1785, 1798):
        # Allow any suffix to match existing files like experiment_1785_ebft_pipeline.json
        # and also our mock files in tests
        pattern = str(input_path / f"experiment_{i}_*.json")
        matches = glob.glob(pattern)
        for match in matches:
            filename = os.path.basename(match)
            try:
                with open(match, "r") as f:
                    data = json.load(f)
                    results[filename] = data.get("honest_verdict", "Missing honest_verdict")
            except Exception as e:
                results[filename] = f"Error reading: {str(e)}"
    
    overall_verdict = "Phase 4 Retrospective Complete. Collected data from experiments 1785-1797."
    
    retro_data = {
        "milestone": "2026.05.138",
        "honest_verdict": results,
        "overall_verdict": overall_verdict
    }
    
    with open(output_path, "w") as f:
        json.dump(retro_data, f, indent=2)

if __name__ == "__main__":
    generate_retro("results", "results/experiment_1798_retro.json")
