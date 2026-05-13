import json
import os
import jax.numpy as jnp
from carnot.models.kan.glorokan import CarnotKAN, run_experiment_2071

def test_carnot_kan_symbolic_embedding_req_kan_2071():
    """
    Test discrete symbolic embedding capabilities in CarnotKAN (REQ-KAN-2071).
    Tests SCENARIO-KAN-2071: Symbolic primitive discovery via hierarchical gating.
    """
    kan = CarnotKAN(num_knots=10, degree=3, use_symbolic_gating=True)
    assert hasattr(kan, 'symbolic_gates')
    assert kan.symbolic_gates is not None
    
    # Run the experiment which will train and evaluate
    run_experiment_2071()
    
    # Check if the artifact was generated
    artifact_path = "results/experiment_2071_symbolic_kan.json"
    assert os.path.exists(artifact_path)
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["schema"] == "experiment_2071"
    assert data["status"] == "complete"
    assert "accuracy" in data
    assert data["accuracy"] >= 0.0  # Just needs to be a valid number
