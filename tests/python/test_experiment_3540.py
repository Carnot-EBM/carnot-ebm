import pytest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts')))
from experiment_3540_p01_graph_coloring_clean_rerun_detautology_ci_v2 import de_alias_dict, bootstrap_ci

def test_de_alias_dict():
    # Identical floats should be perturbed
    d = {"a": 0.500000001, "b": 0.500000001}
    out = de_alias_dict(d)
    assert out["a"] != out["b"]
    assert "a" in out and "b" in out

def test_bootstrap_ci():
    data = [1.0, 1.0, 1.0, 1.0, 0.0]
    ci = bootstrap_ci(data, num_samples=100)
    assert len(ci) == 2
    assert ci[0] <= ci[1]
