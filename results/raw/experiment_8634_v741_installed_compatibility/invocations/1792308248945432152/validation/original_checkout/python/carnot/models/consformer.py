"""ConsFormer-style iterative neural refinement loop.

**Researcher summary:**
    Implements a ConsFormer-style Transformer that iteratively refines
    variable assignments to solve Constraint Satisfaction Problems (CSPs).

**Detailed explanation for engineers:**
    This module contains the `ConsFormerRefiner` Flax module and a
    `refinement_loop` utility. The Transformer uses the graph's adjacency
    matrix as an attention mask (or bias) so that variables attend to
    their constrained neighbors.

Spec: REQ-MODEL-1934
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

class ConsFormerRefiner(nn.Module):
    """Transformer for refining CSP assignments.

    Args:
        d_model: Hidden dimension.
        num_heads: Number of attention heads.
        num_layers: Number of Transformer blocks.
    """
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2

    @nn.compact
    def __call__(self, x: jax.Array, adj_matrix: jax.Array) -> jax.Array:
        """Forward pass.

        Args:
            x: Current assignments, shape (num_vars, 1).
            adj_matrix: Adjacency matrix, shape (num_vars, num_vars).
                Used as an attention mask (1 = edge, 0 = no edge).

        Returns:
            Update to assignments, shape (num_vars,).
        """
        # Embed scalar variables into d_model
        h = nn.Dense(self.d_model)(x)
        
        # Create attention mask from adjacency matrix
        # For standard dot product attention, mask should be boolean (True = allow)
        # We allow self-attention (add identity)
        num_vars = x.shape[0]
        mask = (adj_matrix + jnp.eye(num_vars)) > 0

        for _ in range(self.num_layers):
            # Attention block
            h_norm = nn.LayerNorm()(h)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
            )(h_norm, h_norm, mask=mask)
            h = h + attn_out
            
            # FFN block
            h_norm2 = nn.LayerNorm()(h)
            ffn_out = nn.Dense(self.d_model * 4)(h_norm2)
            ffn_out = nn.relu(ffn_out)
            ffn_out = nn.Dense(self.d_model)(ffn_out)
            h = h + ffn_out

        # Output projection back to 1D update
        update = nn.Dense(1)(h)
        return jnp.squeeze(update, axis=-1)

def refinement_loop(
    params: dict,
    model: nn.Module,
    init_x: jax.Array,
    adj_matrix: jax.Array,
    num_steps: int = 10,
    step_size: float = 0.1,
) -> tuple[jax.Array, list[jax.Array]]:
    """Iteratively refine assignments using the ConsFormer model.

    Args:
        params: Model parameters.
        model: ConsFormerRefiner instance.
        init_x: Initial assignments, shape (num_vars,).
        adj_matrix: Adjacency matrix, shape (num_vars, num_vars).
        num_steps: Number of refinement steps.
        step_size: Step size for applying updates.

    Returns:
        Final assignments and trajectory.
    """
    def step_fn(x, _):
        # Model expects (num_vars, 1)
        x_in = jnp.expand_dims(x, axis=-1)
        update = model.apply(params, x_in, adj_matrix)
        next_x = x + step_size * update
        return next_x, next_x

    final_x, trajectory = jax.lax.scan(step_fn, init_x, None, length=num_steps)
    return final_x, trajectory
