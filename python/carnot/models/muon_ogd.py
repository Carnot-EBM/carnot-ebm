"""Muon-OGD Spectral Orthogonal Gradient Projection optimizer wrapper.

**Researcher summary:**
    Muon-OGD (arXiv 2604.14818) uses spectral orthogonal gradient projections to
    protect previously learned knowledge during LLM continual learning.
    This wrapper modifies an Optax optimizer to project gradients orthogonally
    against a running memory matrix of prior task gradients.

Spec: REQ-LEARN-1827, SCENARIO-LEARN-1827
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import optax


class MuonOGDState(NamedTuple):
    """State for the Muon-OGD optimizer wrapper.

    Attributes:
        inner_state: The state of the inner Optax optimizer.
        memory_matrix: Running matrix of prior orthogonalized gradient unit vectors.
        memory_idx: The current number of vectors stored in the memory matrix.
    """

    inner_state: optax.OptState
    memory_matrix: jax.Array
    memory_idx: jax.Array


def muon_ogd_wrapper(
    inner_optimizer: optax.GradientTransformation, max_memory_size: int
) -> optax.GradientTransformation:
    """Wrapper that applies Muon-OGD spectral orthogonal gradient projection.

    Args:
        inner_optimizer: The Optax optimizer to wrap.
        max_memory_size: Maximum number of prior gradient directions to store.

    Returns:
        An optax.GradientTransformation that projects gradients orthogonally
        to all prior stored gradients before applying the inner optimizer.
    """

    def init_fn(params: Any) -> MuonOGDState:
        inner_state = inner_optimizer.init(params)
        flat_params, _ = ravel_pytree(params)
        param_dim = flat_params.shape[0]
        # Pre-allocate memory matrix with zeros
        memory_matrix = jnp.zeros((max_memory_size, param_dim))
        return MuonOGDState(
            inner_state=inner_state,
            memory_matrix=memory_matrix,
            memory_idx=jnp.array(0, dtype=jnp.int32),
        )

    def update_fn(
        updates: optax.Updates, state: MuonOGDState, params: optax.Params | None = None
    ) -> tuple[optax.Updates, MuonOGDState]:
        flat_updates, unflatten_fn = ravel_pytree(updates)

        # Calculate dot products with all memory vectors
        dots = jnp.dot(state.memory_matrix, flat_updates)

        # Mask out vectors beyond memory_idx
        mask = jnp.arange(max_memory_size) < state.memory_idx
        dots = jnp.where(mask, dots, 0.0)

        # Calculate the orthogonal projection
        proj = jnp.dot(dots, state.memory_matrix)
        orthogonal_update = flat_updates - proj

        # Calculate norm of the orthogonal update
        v_norm = jnp.linalg.norm(orthogonal_update)

        # Compute normalized vector (avoid division by zero)
        v_unit = jnp.where(
            v_norm > 1e-8,
            orthogonal_update / jnp.maximum(v_norm, 1e-12),
            jnp.zeros_like(orthogonal_update),
        )

        # Only add to memory if we have space and it's a new orthogonal direction
        can_add = jnp.logical_and(state.memory_idx < max_memory_size, v_norm > 1e-8)

        new_memory_matrix = jnp.where(
            can_add, state.memory_matrix.at[state.memory_idx].set(v_unit), state.memory_matrix
        )
        new_memory_idx = state.memory_idx + jnp.where(can_add, 1, 0)

        unflattened_updates = unflatten_fn(orthogonal_update)

        inner_updates, new_inner_state = inner_optimizer.update(
            unflattened_updates, state.inner_state, params
        )

        new_state = MuonOGDState(
            inner_state=new_inner_state, memory_matrix=new_memory_matrix, memory_idx=new_memory_idx
        )
        return inner_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
