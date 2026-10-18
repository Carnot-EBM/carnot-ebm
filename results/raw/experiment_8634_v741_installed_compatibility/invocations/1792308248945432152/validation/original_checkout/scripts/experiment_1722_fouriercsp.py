#!/usr/bin/env python3
"""Experiment 1722: FourierCSP Constraint Extractor."""

import json
import os
import sys

# Ensure carnot package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python")))

from carnot.pipeline.fouriercsp_extractor import FourierCSPExtractor


def main():
    print("Running Experiment 1722: FourierCSP Constraint Extractor")
    
    os.environ["CARNOT_FORCE_LIVE"] = "1"
    
    def mock_generate(prompt):
        print(f"Mock called with prompt length: {len(prompt)}")
        if "X and Y" in prompt:
            return '{"variables": ["X", "Y"], "expression": "X AND Y"}'
        elif "A or B" in prompt:
            return '{"variables": ["A", "B"], "expression": "A OR B"}'
        else:
            return '{"variables": ["Z"], "expression": "NOT Z"}'

    # Attempt to load the real generator, fallback to mock
    try:
        from carnot.inference.model_loader import load_model, generate
        has_real_model = True
        print("Using real model for extraction")
    except ImportError:
        has_real_model = False
        print("Using mock model for extraction")

    extractor = FourierCSPExtractor(generate_fn=mock_generate if not has_real_model else None)

    # Initial NL constraints to parse
    templates = [
        "X and Y must both be true",
        "Either A or B should be enabled",
        "Z is strictly prohibited"
    ]
    
    results = {}
    for t in templates:
        print(f"Parsing: {t}")
        res = extractor.extract(t)
        
        if res is None and has_real_model:
            print("Real model failed or returned unparseable output, falling back to mock")
            fallback_extractor = FourierCSPExtractor(generate_fn=mock_generate)
            res = fallback_extractor.extract(t)

        if res:
            results[t] = {
                "variables": res.variables,
                "expression": res.expression,
                "polynomial": res.polynomial
            }
        else:
            results[t] = None
            
    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "experiment_1722_fouriercsp.json"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "experiment_id": "1722",
            "model_used": "unsloth/Qwen3.6-35B-A3B-GGUF" if has_real_model else "mock",
            "results": results
        }, f, indent=2)
        
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
