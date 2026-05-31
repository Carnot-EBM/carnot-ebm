import sys
import os

# Add the python directory to sys.path so we can import carnot
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from carnot.experiment_3577_kv260_continuity_v16 import run_experiment

if __name__ == "__main__":
    output_path = "results/experiment_3577_kv260_continuity_v16.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    run_experiment(output_path)
