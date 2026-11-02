#!/usr/bin/env python3
"""Experiment 1970: THRML Hookup for Energy Initialization.

Spec: REQ-SAMPLE-041
"""
import time
import numpy as np
from scripts.experiment_template import ExperimentTemplate
from carnot.samplers.thrml_init import thrml_energy_init

def main():
    tmpl = ExperimentTemplate(
        exp_id=1970,
        title="THRML hookup for energy initialization",
        deliverable="results/experiment_1970_thrml_hookup.json",
        requires_gpu=False,
    )
    tmpl.setup()

    biases = np.random.randn(10)
    couplings = np.random.randn(10, 10)
    # Symmetrize couplings
    couplings = (couplings + couplings.T) / 2
    np.fill_diagonal(couplings, 0)
    n_samples = 100

    start_time = time.perf_counter()
    offsets = thrml_energy_init(
        biases=biases,
        couplings=couplings,
        n_samples=n_samples,
        n_steps=10,
        beta=1.0,
        seed=tmpl.random_seed
    )
    end_time = time.perf_counter()

    exec_time_s = end_time - start_time
    
    # Store the results. Since offsets is a numpy array, convert to list.
    offsets_list = offsets.tolist() if hasattr(offsets, 'tolist') else list(offsets)
    
    artifact = tmpl.build_result(
        data={
            "offsets": offsets_list,
            "exec_time_s": exec_time_s,
            "n_samples": n_samples,
            "n_steps": 10,
            "beta": 1.0,
            "mean_offset": float(np.mean(offsets)) if hasattr(offsets, 'mean') else 0.0
        },
        status="success",
        code_files=[__file__, "python/carnot/samplers/thrml_init.py"]
    )
    
    with open(tmpl.deliverable, "w") as f:
        import json
        json.dump(artifact, f, indent=2)

    tmpl.assert_deliverable_written()

if __name__ == "__main__":
    main()
