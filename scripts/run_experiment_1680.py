#!/usr/bin/env python3
"""Run experiment 1680: SCG-MEM structural enforcer."""

import os
import json
from carnot.memory.scg_adapter import ScgAdapter

def main():
    print("Initializing SCG-MEM adapter...")
    adapter = ScgAdapter()
    
    # Generated memory embeddings (mocking continuous FR-11 stream)
    raw_traces = [
        {
            "trace_id": "trace-001",
            "memory_embedding": [0.12, 0.34, -0.56],
            "cognitive_context": "FR-11 self-learning promotion loop",
            "utility_score": 0.85
        },
        {
            "trace_id": "trace-002",
            "memory_embedding": "invalid_array_type",
            "cognitive_context": "Should be filtered out",
            "utility_score": 0.1
        },
        {
            "trace_id": "trace-003",
            "memory_embedding": [-0.1, 0.9, 0.4],
            "cognitive_context": "Valid schema trace",
            "utility_score": 0.92
        },
        {
            "trace_id": "trace-004",
            "memory_embedding": [0.5, 0.5],
            # Missing cognitive_context, should be filtered
        }
    ]
    
    valid_traces = adapter.process_embeddings(raw_traces)
    
    output_path = "results/experiment_1680_scg_mem.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    deliverable = {
        "experiment": "1680",
        "description": "SCG-MEM structural enforcer for FR-11",
        "total_traces_processed": len(raw_traces),
        "valid_traces_retained": len(valid_traces),
        "traces": valid_traces,
        "schema_enforced": adapter.schema
    }
    
    with open(output_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"Successfully processed traces and wrote output to {output_path}")

if __name__ == "__main__":
    main()
