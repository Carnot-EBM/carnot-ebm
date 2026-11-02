#!/usr/bin/env python3
"""Exp 1628 EBRM Latent Trace Scoring.

Spec: REQ-VERIFY-1628, SCENARIO-VERIFY-1628.
"""

import json
from pathlib import Path
from typing import Any

import jax

from carnot.verify.ebrm_scoring import score_latent_trace

JsonDict = dict[str, Any]
RUN_DATE = "20260509"
DEFAULT_ARTIFACT_PATH = Path("results/experiment_1628_ebrm_scoring.json")


def evaluate_scoring() -> float:
    """Evaluate accuracy (percentage where good trace score > bad trace score)."""
    key = jax.random.PRNGKey(1628)
    
    correct = 0
    total = 100
    
    for _ in range(total):
        k1, k2, key = jax.random.split(key, 3)
        
        # Good trace: matching polarities
        good = jax.random.normal(k1, (5, 10))
        good = good.at[:, 0].set(1.0)
        
        # Bad trace: clashing polarities
        bad = jax.random.normal(k2, (5, 10))
        bad = bad.at[0:2, 0].set(1.0)
        bad = bad.at[2:, 0].set(-1.0)
        
        score_g = score_latent_trace(good)
        score_b = score_latent_trace(bad)
        
        if score_g > score_b:
            correct += 1
            
    return float(correct / total)


def run_experiment(output_path: Path | str = DEFAULT_ARTIFACT_PATH) -> JsonDict:
    """Run experiment and return artifact dict."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    
    accuracy = evaluate_scoring()
    
    artifact: JsonDict = {
        "status": "complete",
        "experiment_id": 1628,
        "run_date": RUN_DATE,
        "scoring_accuracy": accuracy,
        "honest_verdict": "ebrm_scoring_distinguishes_traces" if accuracy > 0.8 else "scoring_failed"
    }
    
    output.write_text(json.dumps(artifact, indent=2))
    return artifact


if __name__ == "__main__":  # pragma: no cover
    run_experiment()
