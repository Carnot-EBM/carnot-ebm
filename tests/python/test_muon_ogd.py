"""Tests for the Muon-OGD optimizer wrapper.

Spec: REQ-LEARN-1827, SCENARIO-LEARN-1827
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from carnot.models.muon_ogd import muon_ogd_wrapper, MuonOGDState


def test_muon_ogd_wrapper_orthogonal_projection():
    """Test that the muon_ogd_wrapper projects gradients orthogonally.

    Traces to: REQ-LEARN-1827, SCENARIO-LEARN-1827
    """
    inner_opt = optax.sgd(learning_rate=0.1)
    max_memory = 3
    opt = muon_ogd_wrapper(inner_opt, max_memory_size=max_memory)

    # 3 parameters
    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    state = opt.init(params)

    assert isinstance(state, MuonOGDState)
    assert state.memory_matrix.shape == (3, 3)
    assert state.memory_idx == 0

    # Step 1: gradient along x-axis
    grad1 = {"w": jnp.array([1.0, 0.0, 0.0])}
    updates1, state = opt.update(grad1, state, params)

    # The first gradient should be applied as is (scaled by lr)
    assert jnp.allclose(updates1["w"], jnp.array([-0.1, 0.0, 0.0]))
    assert state.memory_idx == 1
    # memory_matrix[0] should be the normalized gradient [1.0, 0.0, 0.0]
    assert jnp.allclose(state.memory_matrix[0], jnp.array([1.0, 0.0, 0.0]))

    # Step 2: gradient with x and y components
    grad2 = {"w": jnp.array([0.5, 1.0, 0.0])}
    updates2, state = opt.update(grad2, state, params)

    # The x component should be projected out, leaving only y component
    assert jnp.allclose(updates2["w"], jnp.array([0.0, -0.1, 0.0]))
    assert state.memory_idx == 2
    assert jnp.allclose(state.memory_matrix[1], jnp.array([0.0, 1.0, 0.0]))

    # Step 3: gradient along all axes
    grad3 = {"w": jnp.array([0.5, 0.5, 1.0])}
    updates3, state = opt.update(grad3, state, params)

    # The x and y components should be projected out, leaving only z component
    assert jnp.allclose(updates3["w"], jnp.array([0.0, 0.0, -0.1]))
    assert state.memory_idx == 3
    assert jnp.allclose(state.memory_matrix[2], jnp.array([0.0, 0.0, 1.0]))

    # Step 4: buffer full, gradient along all axes
    grad4 = {"w": jnp.array([1.0, 1.0, 1.0])}
    updates4, state = opt.update(grad4, state, params)

    # The buffer is full and spans the space, so update should be zero
    assert jnp.allclose(updates4["w"], jnp.array([0.0, 0.0, 0.0]), atol=1e-7)
    assert state.memory_idx == 3  # doesn't increment past max_memory


def test_muon_ogd_wrapper_zero_norm():
    """Test behavior when the orthogonal update has zero norm."""
    inner_opt = optax.sgd(learning_rate=0.1)
    opt = muon_ogd_wrapper(inner_opt, max_memory_size=2)

    params = {"w": jnp.array([1.0, 2.0])}
    state = opt.init(params)

    # Step 1: valid gradient
    grad1 = {"w": jnp.array([1.0, 0.0])}
    updates1, state = opt.update(grad1, state, params)
    assert state.memory_idx == 1

    # Step 2: parallel gradient (will be projected to zero)
    grad2 = {"w": jnp.array([0.5, 0.0])}
    updates2, state = opt.update(grad2, state, params)

    # Memory index shouldn't increment because orthogonal norm is 0
    assert state.memory_idx == 1
    assert jnp.allclose(updates2["w"], jnp.array([0.0, 0.0]))
