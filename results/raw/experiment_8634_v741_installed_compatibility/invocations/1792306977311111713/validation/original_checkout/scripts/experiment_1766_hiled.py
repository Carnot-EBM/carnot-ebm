"""Run the HILED decoder prototype and generate the experiment deliverable."""

import json
import os
import sys
from pathlib import Path

# Add python dir to path so we can import carnot
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.inference.hiled_decoder import HiledDecoder

def main():
    decoder = HiledDecoder(simulator_axi_endpoint="axi://localhost:9000", max_steps=10)
    final_energy = decoder.minimize_energy(initial_state=[1, 1, 1, 1])
    
    result = {
        "status": "success",
        "experiment": "1766",
        "algorithm": "hiled_decoder",
        "simulator_endpoint": decoder.simulator_axi_endpoint,
        "steps_polled": decoder.steps_polled,
        "final_energy": final_energy,
        "honest_verdict": "hiled_decoder_implemented"
    }
    
    out_path = Path(__file__).parent.parent / "results" / "experiment_1766_hiled.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Wrote artifact to {out_path}")

if __name__ == "__main__":
    main()
