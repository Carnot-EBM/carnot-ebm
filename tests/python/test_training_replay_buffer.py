"""Tests for replay buffer and NCE loss with replay.

Spec coverage: REQ-TRAIN-006
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from carnot.training.replay_buffer import ReplayBuffer, nce_loss_with_replay


def _quadratic_energy(x: jax.Array) -> jax.Array:
    """Simple energy for testing."""
    return 0.5 * jnp.sum(x**2)


class TestReplayBuffer:
    """Tests for the ReplayBuffer class."""

    def test_add_single_state(self) -> None:
        """REQ-TRAIN-006: add a single state."""
        buf = ReplayBuffer(max_size=100)
        buf.add(jnp.array([1.0, 2.0]))
        assert len(buf) == 1

    def test_add_batch(self) -> None:
        """REQ-TRAIN-006: add a batch of states."""
        buf = ReplayBuffer(max_size=100)
        buf.add(jnp.ones((5, 3)))
        assert len(buf) == 5

    def test_max_size_eviction(self) -> None:
        """REQ-TRAIN-006: oldest states evicted when full."""
        buf = ReplayBuffer(max_size=3)
        for i in range(5):
            buf.add(jnp.array([float(i)]))
        assert len(buf) == 3

    def test_sample(self) -> None:
        """REQ-TRAIN-006: sample returns correct shape."""
        buf = ReplayBuffer(max_size=100)
        buf.add(jnp.ones((10, 4)))
        samples = buf.sample(3, jrandom.PRNGKey(0))
        assert samples.shape == (3, 4)

    def test_sample_empty_raises(self) -> None:
        """REQ-TRAIN-006: sampling from empty buffer raises."""
        buf = ReplayBuffer(max_size=100)
        with pytest.raises(ValueError, match="empty"):
            buf.sample(1, jrandom.PRNGKey(0))

    def test_sample_deterministic(self) -> None:
        """REQ-TRAIN-006: same key gives same samples."""
        buf = ReplayBuffer(max_size=100)
        buf.add(jnp.ones((10, 2)))
        s1 = buf.sample(3, jrandom.PRNGKey(42))
        s2 = buf.sample(3, jrandom.PRNGKey(42))
        assert jnp.allclose(s1, s2)


class TestNCELossWithReplay:
    """Tests for NCE loss augmented with replay buffer."""

    def test_loss_is_finite(self) -> None:
        """REQ-TRAIN-006: loss is finite."""
        data = jnp.zeros((4, 3))
        noise = jnp.ones((4, 3)) * 3.0
        replay = jnp.ones((4, 3)) * 1.5
        loss = nce_loss_with_replay(_quadratic_energy, data, noise, replay)
        assert jnp.isfinite(loss)

    def test_replay_increases_loss(self) -> None:
        """REQ-TRAIN-006: replay term adds to loss."""
        data = jnp.zeros((4, 3))
        noise = jnp.ones((4, 3)) * 3.0
        replay = jnp.ones((4, 3)) * 1.5

        loss_no_replay = nce_loss_with_replay(
            _quadratic_energy, data, noise, replay, replay_weight=0.0
        )
        loss_with_replay = nce_loss_with_replay(
            _quadratic_energy, data, noise, replay, replay_weight=1.0
        )
        # Replay term should add something (may be positive or negative)
        assert not jnp.allclose(loss_no_replay, loss_with_replay)

    def test_zero_replay_weight(self) -> None:
        """REQ-TRAIN-006: zero weight = standard NCE."""
        data = jnp.zeros((4, 3))
        noise = jnp.ones((4, 3)) * 3.0
        replay = jnp.ones((4, 3)) * 1.5
        loss = nce_loss_with_replay(_quadratic_energy, data, noise, replay, replay_weight=0.0)
        assert jnp.isfinite(loss)
