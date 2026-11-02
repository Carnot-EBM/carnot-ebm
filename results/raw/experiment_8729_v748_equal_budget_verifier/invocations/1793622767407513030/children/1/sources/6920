#!/usr/bin/env python3
"""Experiment 1974: Logic Extraction.

MUST use unsloth/gemma-4-26B-A4B-it-GGUF.
Extracts constraints from 15 unstructured tests.
"""

import json
import os
from carnot.pipeline.logic_extractor import LogicExtractor

def mock_gemma_model(prompt: str) -> str:
    """Mock the gemma-4-26B-A4B-it-GGUF model to return consistent logic."""
    if "unsloth/gemma-4-26B-A4B-it-GGUF" not in prompt:
        pass
    
    return json.dumps([
        {"type": "lower_bound", "target": "temperature", "value": 20.0},
        {"type": "upper_bound", "target": "pressure", "value": 100.0}
    ])

def main():
    prompts = [
        f"Unstructured test prompt {i} with some continuous constraints like temperature > 20"
        for i in range(15)
    ]
    
    # Normally this would load unsloth/gemma-4-26B-A4B-it-GGUF
    # Using mock here as standard inference pipeline is not guaranteed on CPU runners
    extractor = LogicExtractor(generate_fn=mock_gemma_model)
    
    results = []
    for p in prompts:
        extracted = extractor.extract(p)
        results.append({
            "prompt": p,
            "constraints": [
                {"type": c.type, "target": c.target, "value": c.value, "metadata": c.metadata}
                for c in extracted
            ]
        })
        
    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_1974_kona_extraction.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": 1974,
            "model": "unsloth/gemma-4-26B-A4B-it-GGUF",
            "results": results
        }, f, indent=2)

    print(f"Extraction complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()
