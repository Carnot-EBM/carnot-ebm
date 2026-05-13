#!/usr/bin/env python3
"""Experiment 2046: SCP Prototype on small constraint rulebook.

Spec: REQ-SCP-001, REQ-SCP-002, SCENARIO-SCP-001
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_template import ExperimentTemplate
from carnot.semantic_compression import SemanticCompressor

def main():
    tmpl = ExperimentTemplate(
        exp_id=2046,
        title="SCP Prototype on CCTU constraints",
        deliverable="results/experiment_2046_clara_compression.json",
        requires_gpu=False,
    )
    tmpl.setup()
    
    compressor = SemanticCompressor("unsloth/Qwen3.6-35B-A3B-GGUF")
    
    constraints = [f"CCTU Constraint {i}" for i in range(1, 51)]
    
    with tmpl.phase("compression"):
        compressed = compressor.compress(constraints)
        
    with tmpl.phase("evaluation"):
        metrics = compressor.evaluate_retrieval(constraints, compressed)
        
    result_data = {
        "schema": "carnot.experiment.v1",
        "model": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "num_constraints": len(constraints),
        "metrics": metrics,
    }
    
    artifact = tmpl.build_result(
        result_data,
        status="success",
        decision_class="verify",
        code_files=[__file__]
    )
    
    tmpl._output_path.write_text(json.dumps(artifact, indent=2))
    
    tmpl.assert_deliverable_written()

if __name__ == "__main__":
    main()
