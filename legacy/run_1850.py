import json
from pathlib import Path
from carnot.samplers.thrml_parity_1850 import run_parity_n128

def main():
    result = run_parity_n128(seed=1850, n_samples=100)
    out_path = Path("results/experiment_1850_thrml_parity_n128.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
