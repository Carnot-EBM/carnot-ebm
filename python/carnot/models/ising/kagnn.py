"""Kolmogorov-Arnold Graph Neural Network (KAGNN) Verifier.

Spec references: REQ-KAN-2034, SCENARIO-KAN-2034.

This module provides a KAGNN layer that uses Symbolic-KAN routing logic
to evaluate edge constraints between adjacent nodes in a graph. For Graph Coloring,
adjacent nodes must have different values. The verifier assigns lower energy to
valid colorings and higher energy to invalid ones.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from carnot.models.kan.symbolic_kan import (
    SymbolicKANConfig,
    SymbolicKANParams,
    SymbolicRoutingLayer,
)


@dataclass(frozen=True)
class KAGNNConfig:
    """Configuration for KAGNN Verifier."""

    node_dim: int = 1
    hidden_dim: int = 2
    n_routes: int = 2
    primitives: tuple[str, ...] = ("identity", "square", "sin", "abs")
    temperature: float = 1.0


class KAGNNVerifier:
    """KAGNN Verifier using Symbolic-KAN splines on edges.
    
    Computes energy for Graph Coloring instances. The energy is the sum
    over all edges of the SymbolicKAN penalty between the node features.
    """

    def __init__(
        self,
        config: KAGNNConfig,
        key: jax.Array | None = None,
        params: SymbolicKANParams | None = None,
    ) -> None:
        self.config = config
        
        # We use a SymbolicRoutingLayer to model the penalty function between two nodes
        # The input to the routing layer is the concatenation (or difference) of two node features.
        # Here we use the concatenation, so input_dim is 2 * node_dim.
        sym_config = SymbolicKANConfig(
            input_dim=config.node_dim * 2,
            n_routes=config.n_routes,
            primitives=config.primitives,
            temperature=config.temperature,
        )
        self.routing_layer = SymbolicRoutingLayer(sym_config, key=key, params=params)

    def edge_energy(
        self, 
        node_u: jax.Array, 
        node_v: jax.Array, 
        hard: bool = False,
        params: SymbolicKANParams | None = None,
    ) -> jax.Array:
        """Evaluate constraint penalty for a single edge."""
        # Concatenate features of u and v
        edge_features = jnp.concatenate([node_u, node_v], axis=-1)
        return self.routing_layer.forward(edge_features, hard=hard, params=params)

    def forward(
        self, 
        node_features: jax.Array, 
        edge_indices: jax.Array, 
        hard: bool = False,
        params: SymbolicKANParams | None = None,
    ) -> jax.Array:
        """Evaluate the total energy of the graph.
        
        Args:
            node_features: Tensor of shape `(num_nodes, node_dim)`.
            edge_indices: Tensor of shape `(num_edges, 2)`.
            hard: Whether to use hard routing.
            params: Optional model parameters.
            
        Returns:
            Scalar total energy.
        """
        node_features = jnp.asarray(node_features, dtype=jnp.float32)
        edge_indices = jnp.asarray(edge_indices, dtype=jnp.int32)
        
        if edge_indices.size == 0:
            return jnp.array(0.0, dtype=jnp.float32)
        
        u = node_features[edge_indices[:, 0]]
        v = node_features[edge_indices[:, 1]]
        
        # Evaluate all edges in batch
        edge_features = jnp.concatenate([u, v], axis=-1)
        energies = self.routing_layer.forward(edge_features, hard=hard, params=params)
        
        return jnp.sum(energies)
