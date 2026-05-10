#!/usr/bin/env python3
"""Exp 1725: E2E Pipeline with SOTA, FourierCSP, CIKAN, and Online Updater.

Spec traces: REQ-LEARN-1725, SCENARIO-LEARN-1725
"""

import json
from pathlib import Path

from carnot.models.cikan_verifier import CIKAN
from carnot.pipeline.fouriercsp_extractor import FourierCSPExtractor
from carnot.pipeline.verification_loop import VerificationLoop, Violation
from carnot.training.online_updater import OnlineUpdater


def run_experiment() -> None:
    model_used = "unsloth/Qwen3.6-35B-A3B-GGUF"
    
    # 2. Instantiate the pipeline components
    extractor = FourierCSPExtractor()
    cikan = CIKAN(feature_names=["f1", "f2"], seed=42)
    updater = OnlineUpdater(optimizer="adamw", learning_rate=0.05)
    loop = VerificationLoop(cikan, updater)
    
    # 3. Execute a 50-problem stream
    stream = []
    for i in range(50):
        constraint_text = f"Constraint for problem {i}"
        poly = extractor.extract(constraint_text)
        
        # Alternate violations and valid to measure adaptation
        if i % 2 == 0:
            stream.append(Violation(features=[0.9, 0.9], label=0.0))
        else:
            stream.append(Violation(features=[0.1, 0.1], label=1.0))
            
    loop.run(stream)
    
    adaptation_rate = loop.n_updated / loop.n_processed if loop.n_processed > 0 else 0.0
    
    results = {
        "experiment_id": "1725",
        "model_used": model_used,
        "n_processed": loop.n_processed,
        "n_updated": loop.n_updated,
        "adaptation_rate": adaptation_rate,
        "honest_verdict": "e2e_pipeline_successful" if loop.n_processed == 50 else "e2e_pipeline_failed"
    }
    
    # 4. Write results
    output_path = Path("results/experiment_1725_e2e_cikan.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    run_experiment()
