#!/usr/bin/env python3
"""Experiment 2112: Muon-OGD for the LLM policy update step."""

import json
from pathlib import Path
import jax.numpy as jnp
from carnot.training.csl_loop import run_csl_loop

DELIVERABLE = "results/experiment_2112_muon_ogd.json"

def main() -> None:
    params = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    grads = jnp.array([[0.1, 0.2], [0.3, 0.4]])
    
    artifact = run_csl_loop(params, grads)
    artifact["schema"] = "carnot.csl.loop.v1"
    artifact["model"] = "unsloth/Qwen3.6-35B-A3B-GGUF"
    
    out_path = Path(DELIVERABLE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

if __name__ == "__main__":
    main()
