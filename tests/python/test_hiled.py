import os
import json
import pytest
from carnot.samplers.hiled import HiledSimulator

def test_hiled_simulator_req_1845():
    """
    Test that REQ-SAMPLE-1845 and SCENARIO-SAMPLE-1845 are satisfied.
    """
    simulator = HiledSimulator(target="KV260")
    assert simulator.target == "KV260"
    
    # Mock initial state
    initial_state = [1, 0, 1, 1]
    
    # Execute pipeline
    final_state = simulator.execute_pipeline(initial_state, num_steps=5)
    
    # State should remain structurally the same (mock logic)
    assert final_state == initial_state
    
    # Metrics should be updated
    assert simulator.metrics["pipeline_invocations"] == 1
    assert simulator.metrics["simulated_energy_minimized"] is True
    assert simulator.metrics["latency_ms"] > 0.0
    
    # Save deliverable
    deliverable_path = "results/experiment_1845_hiled.json"
    
    # Ensure directory exists just in case
    os.makedirs(os.path.dirname(deliverable_path), exist_ok=True)
    
    # Clean up existing if it exists
    if os.path.exists(deliverable_path):
        os.remove(deliverable_path)
        
    simulator.save_deliverable(deliverable_path)
    
    assert os.path.exists(deliverable_path)
    
    with open(deliverable_path, "r") as f:
        data = json.load(f)
        
    assert data["target"] == "KV260"
    assert data["simulated_energy_minimized"] is True
    assert "honest_verdict" in data
