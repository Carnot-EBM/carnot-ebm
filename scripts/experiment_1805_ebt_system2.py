#!/usr/bin/env python3
"""Experiment 1805: Standalone EBT System 2 Loop."""

import sys
from pathlib import Path

# Add python directory and project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root_dir / "python"))
sys.path.insert(0, str(root_dir))

from scripts.experiment_template import ExperimentTemplate
from carnot.ebt_system2 import EBTSystem2Loop

def main():
    tmpl = ExperimentTemplate(
        exp_id=1805,
        title="Standalone EBT System 2 Loop",
        deliverable="results/experiment_1805_ebt_system2.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Step 1: Initialize the sampler
    loop = EBTSystem2Loop(model_hf_id="unsloth/gemma-4-26B-A4B-it-GGUF")
    
    # Step 2: Iterate over prediction candidates
    with tmpl.phase("optimization_loop"):
        results = loop.optimize_candidates(["initial prediction"], max_steps=5)

    # Step 3: Write summary
    artifact = tmpl.build_result(
        results,
        status="success",
        code_files=[__file__, str(Path(__file__).resolve().parents[1] / "python" / "carnot" / "ebt_system2.py")]
    )

    tmpl.assert_deliverable_written()

if __name__ == "__main__":
    main()
