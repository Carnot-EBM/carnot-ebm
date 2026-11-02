#!/usr/bin/env python3
"""Experiment 1647: RKAN Lean 4 Formal Specification Exporter."""

import json
from pathlib import Path

from carnot.models.rkan import RationalKANEnergyFunction
from carnot.models.rkan_lean_export import export_rkan_to_lean

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def run_experiment():
    model = RationalKANEnergyFunction(
        input_dim=2,
        edge_control_points={
            (0, 1): ["1/2", "3/4", "1"],
        },
        bias_control_points=[
            ["0", "1/2", "1"],
            ["-1", "0", "1"],
        ],
    )
    
    lean_spec = export_rkan_to_lean(model)
    
    lean_out_path = _repo_root() / "results" / "experiment_1647_rkan_spec.lean"
    lean_out_path.parent.mkdir(parents=True, exist_ok=True)
    lean_out_path.write_text(lean_spec)
    
    artifact = {
        "schema": "carnot.rkan_lean_export.v1",
        "status": "complete",
        "experiment": 1647,
        "experiment_id": 1647,
        "run_date": "20260509",
        "spec": ["REQ-KAN-1647", "SCENARIO-KAN-1647"],
        "export_success": True,
        "lean_spec_length": len(lean_spec),
        "lean_spec_path": "results/experiment_1647_rkan_spec.lean",
        "honest_verdict": "complete: rkan_lean_export_successful",
    }
    
    json_path = _repo_root() / "results" / "experiment_1647_rkan.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    
    print(f"Wrote {json_path}")
    print(f"Wrote {lean_out_path}")

if __name__ == "__main__":
    run_experiment()
