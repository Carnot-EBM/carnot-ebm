import os
import sys

# ensure the train_kan script is importable
sys.path.insert(0, os.path.abspath('.'))
from train_kan import generate_data, expand_features

def test_generate_data():
    z, y = generate_data(n_samples=100)
    assert len(z) == 100
    assert len(y) == 100
    assert set(y).issubset({0, 1})

def test_expand_features():
    z, _ = generate_data(n_samples=10)
    X3 = expand_features(z, 3)
    assert X3.shape == (10, 3)
    
    X5 = expand_features(z, 5)
    assert X5.shape == (10, 5)
