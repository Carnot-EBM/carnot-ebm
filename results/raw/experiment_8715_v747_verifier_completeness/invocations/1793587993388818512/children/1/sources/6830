#!/usr/bin/env python3
"""Experiment 1774: CEM Logic Decomposition.

Spec: REQ-CEM-003
"""

import json
from pathlib import Path
from carnot.verify.plan_graph_energy_adapter import convert_cctu_traces_to_plan_graphs
from carnot.cem.decomposition import decompose_plan_graph

def run_experiment():
    graphs = convert_cctu_traces_to_plan_graphs(trace_limit=20)
    total_subsets = 0
    
    for graph in graphs:
        landscapes = decompose_plan_graph(graph)
        total_subsets += len(landscapes)
        
    artifact = {
        "schema": "carnot.cem.decomposition.v1",
        "num_subsets": total_subsets,
        "graphs_processed": len(graphs)
    }
    
    out_path = Path("results/experiment_1774_cem_decomposition.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"Generated {out_path} with {total_subsets} subsets")

if __name__ == "__main__":
    run_experiment()
