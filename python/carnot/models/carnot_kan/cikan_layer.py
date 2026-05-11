# python/carnot/models/carnot_kan/cikan_layer.py
import jax.numpy as jnp
import numpy as np
from carnot.models.symbolic_kan import SymbolicKANModel, SymbolicKANConfig

class CIKANLayer:
    """Constraint-Informed KAN layer for symbolic regression."""
    def __init__(self, input_dim: int, n_nodes: int = 2, seed: int = 42):
        self.config = SymbolicKANConfig(input_dim=input_dim, n_nodes=n_nodes, label_update_interval=5)
        self.model = SymbolicKANModel(self.config, seed=seed)

    def fit(self, xs_correct: np.ndarray, xs_incorrect: np.ndarray, n_epochs: int = 50):
        self.model.train(xs_correct, xs_incorrect, n_epochs=n_epochs)

    def extract_symbolic_representation(self) -> list[str]:
        """Extracts human-readable symbolic representations of the learned nodes."""
        representations = []
        for i in range(self.config.n_nodes):
            desc = self.model.describe_node(i)
            representations.append(desc)
        return representations
