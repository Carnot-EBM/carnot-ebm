"""Benchmark the HILED hardware setup against the software fallback.

Spec: REQ-HW-055, SCENARIO-HW-055
"""

import json
import os
import sys
import time
from pathlib import Path

# Add python dir to path so we can import carnot
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from carnot.inference.hiled_decoder import HiledDecoder

def main():
    decoder = HiledDecoder(simulator_axi_endpoint="axi://localhost:9000", max_steps=50)
    initial_state = [1] * 100
    
    # Run hardware prototype
    start_hw = time.perf_counter()
    energy_hw = decoder.minimize_energy(initial_state=initial_state)
    end_hw = time.perf_counter()
    latency_hw = end_hw - start_hw
    
    # Run software fallback
    start_sw = time.perf_counter()
    energy_sw = decoder.minimize_energy_software(initial_state=initial_state)
    end_sw = time.perf_counter()
    latency_sw = end_sw - start_sw
    
    result = {
        "status": "success",
        "experiment": "1781",
        "algorithm": "hiled_benchmark",
        "simulator_endpoint": decoder.simulator_axi_endpoint,
        "energy_hardware": energy_hw,
        "energy_software": energy_sw,
        "latency_hardware_s": latency_hw,
        "latency_software_s": latency_sw,
        "speedup_factor": latency_sw / latency_hw if latency_hw > 0 else 0,
        "honest_verdict": "benchmark_completed"
    }
    
    out_path = Path(__file__).parent.parent / "results" / "experiment_1781_hiled_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Wrote benchmark results to {out_path}")

if __name__ == "__main__":
    main()
