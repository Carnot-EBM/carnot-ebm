"""Tests for HardNet Graph Coloring experiment.

Spec: REQ-HARDNET-2086, SCENARIO-HARDNET-2086
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiment_2088_hardnet_graph_coloring import HardNetColoringModel, run_experiment
import jax
import jax.numpy as jnp

def test_hardnet_coloring_model_bounds():
    """Test that the model outputs are strictly within bounds.
    
    Spec: REQ-HARDNET-2086, SCENARIO-HARDNET-2086
    """
    model = HardNetColoringModel(n_nodes=4, n_colors=2)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (10, 4))
    params = model.init(key, x)
    out = model.apply(params, x)
    
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 1.0)

def test_experiment_runs_with_zero_violations():
    """Test that the experiment completes and has 0 violations.
    
    Spec: REQ-HARDNET-2086, SCENARIO-HARDNET-2086
    """
    result = run_experiment()
    assert result["violations"] == 0
    assert result["zero_false_accepts_verified"] is True
    jax.clear_caches()
    try:
        jax.clear_backends()
    except AttributeError:
        pass
    import gc
    gc.collect()
