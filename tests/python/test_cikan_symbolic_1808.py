# tests/python/test_cikan_symbolic_1808.py
import numpy as np
from carnot.models.carnot_kan.cikan_layer import CIKANLayer

def test_cikan_symbolic_extraction():
    # REQ-KAN-1808, SCENARIO-KAN-1808
    layer = CIKANLayer(input_dim=3, n_nodes=1, seed=0)
    
    # Fake dataset for Z = X + Y
    # Feature 0 is x, 1 is y, 2 is z
    xs = np.random.RandomState(42).rand(20, 2)
    xs_correct = np.zeros((20, 3))
    xs_correct[:, :2] = xs
    xs_correct[:, 2] = xs[:, 0] + xs[:, 1]
    
    xs_incorrect = np.zeros((20, 3))
    xs_incorrect[:, :2] = xs
    xs_incorrect[:, 2] = xs[:, 0] + xs[:, 1] + 1.0
    
    layer.fit(xs_correct, xs_incorrect, n_epochs=2)
    reps = layer.extract_symbolic_representation()
    assert len(reps) == 1
    assert "checks" in reps[0]
