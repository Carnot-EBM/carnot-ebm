import json
import numpy as np
import os
from carnot.models.carnot_kan.cikan_layer import CIKANLayer

def main():
    # Exp 1808: Symbolic Regression on CIKAN Layer
    
    # 1. Feed a known logical/algebraic dataset
    # We create a dataset where the ground truth is Z = X + Y
    layer = CIKANLayer(input_dim=3, n_nodes=2, seed=42)
    
    # Create 100 samples
    rng = np.random.RandomState(123)
    xs = rng.rand(100, 2)
    
    xs_correct = np.zeros((100, 3))
    xs_correct[:, 0] = xs[:, 0]
    xs_correct[:, 1] = xs[:, 1]
    xs_correct[:, 2] = xs[:, 0] + xs[:, 1]
    
    xs_incorrect = np.zeros((100, 3))
    xs_incorrect[:, 0] = xs[:, 0]
    xs_incorrect[:, 1] = xs[:, 1]
    xs_incorrect[:, 2] = xs[:, 0] + xs[:, 1] + rng.rand(100) * 0.5 + 0.5
    
    # Fit the layer
    layer.fit(xs_correct, xs_incorrect, n_epochs=50)
    
    # 2. Extract the symbolic representation
    representations = layer.extract_symbolic_representation()
    
    # Check if the correct equation (ADD) is extracted among the nodes
    found_correct = any("ADD" in rep for rep in representations)
    equation_match_accuracy = 1.0 if found_correct else 0.0
    
    # 3. Log equation_match_accuracy
    output_path = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1808_symbolic.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "equation_match_accuracy": equation_match_accuracy,
            "representations": representations,
            "status": "success",
            "schema_version": "1.0",
            "experiment_id": "1808"
        }, f, indent=4)
        
if __name__ == "__main__":
    main()
