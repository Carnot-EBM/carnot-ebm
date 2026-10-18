import jax
import jax.numpy as jnp
from typing import Sequence, Tuple
from carnot.phase3.compositional_energy import CompositionalEnergy
from carnot.phase3.pem_optimizer import PEMOptimizer

def build_graph_coloring_energy(edges: Sequence[Tuple[int, int]], num_nodes: int, node_dim: int = 2) -> CompositionalEnergy:
    """
    Builds local energy models for a graph coloring problem.
    Each edge is a local constraint.
    Nodes are mapped to node_dim dimensional vectors.
    """
    potentials = []
    
    # Unit length constraint for each node
    def build_norm_potential(node_idx):
        def norm_potential(x):
            v = x[node_idx]
            return (jnp.sum(v**2) - 1.0)**2
        return norm_potential
    
    for i in range(num_nodes):
        potentials.append(build_norm_potential(i))
        
    # Edge constraint: dot product should be as negative as possible or <= -0.5
    # To encourage differing colors.
    def build_edge_potential(u, v):
        def edge_potential(x):
            vec_u = x[u]
            vec_v = x[v]
            dot = jnp.dot(vec_u, vec_v)
            return jnp.maximum(0.0, dot + 0.5)**2
        return edge_potential
        
    for u, v in edges:
        potentials.append(build_edge_potential(u, v))
        
    return CompositionalEnergy(potentials)

class PEMCompositionSolver:
    """
    Solves composed energy landscapes using Parallel Energy Minimization.
    """
    def __init__(self, energy: CompositionalEnergy, lr: float = 0.05, steps: int = 1000):
        self.energy = energy
        self.optimizer = PEMOptimizer(energy, learning_rate=lr, noise_scale=0.01)
        self.steps = steps
        
    def solve(self, x_init: jnp.ndarray, key: jax.Array) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x_final, _ = self.optimizer.optimize(x_init, key, self.steps)
        final_energy = self.energy(x_final)
        return x_final, final_energy
