import jax
import jax.numpy as jnp
from carnot.phase3.pem_composition import build_graph_coloring_energy, PEMCompositionSolver

def test_build_graph_coloring_energy():
    edges = [(0, 1)]
    num_nodes = 2
    energy = build_graph_coloring_energy(edges, num_nodes)
    
    # Test valid state (different colors, norm 1)
    x_valid = jnp.array([[1.0, 0.0], [-1.0, 0.0]])
    e_valid = energy(x_valid)
    assert e_valid == 0.0
    
    # Test invalid state (same color)
    x_invalid = jnp.array([[1.0, 0.0], [1.0, 0.0]])
    e_invalid = energy(x_invalid)
    assert e_invalid > 1.0

def test_pem_composition_solver():
    edges = [(0, 1)]
    num_nodes = 2
    energy = build_graph_coloring_energy(edges, num_nodes)
    
    solver = PEMCompositionSolver(energy, lr=0.05, steps=500)
    
    key = jax.random.PRNGKey(0)
    # Start in a slightly noisy state
    x_init = jax.random.normal(key, (num_nodes, 2))
    
    x_final, final_energy = solver.solve(x_init, key)
    
    # Assert energy is minimized significantly
    assert final_energy < 0.5
    # Test shape is preserved
    assert x_final.shape == (num_nodes, 2)
