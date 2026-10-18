#!/usr/bin/env python3
"""Run the experiment for EBT Gemma Integration (Phase 1).

Spec: REQ-EBT-2031
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from carnot.models.boltzmann.ebt_wrapper import EBTWrapper, MODEL_SPECS

def main():
    """Execute the wrapper script and save results to JSON."""
    start_time = time.time()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    wrapper = EBTWrapper(MODEL_SPECS[0])
    initial_trace = ["The sky is blue."]
    candidates = ["This is a contradiction.", "Thus, we can see it."]
    best_candidate, min_energy = wrapper.energy_guided_decoding(initial_trace, candidates)

    duration_s = int(time.time() - start_time)
    finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output = {
        "experiment": 2031,
        "schema": "carnot.experiment.v1",
        "title": "Phase 1: Integrate Gladstone EBT objective with Gemma-4-31B",
        "run_date": datetime.now(timezone.utc).strftime("%Y%m%d"),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "status": "success",
        "best_candidate": best_candidate,
        "min_energy": min_energy
    }

    # Ensure results directory exists
    Path("results").mkdir(exist_ok=True)
    
    with open("results/experiment_2031.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
