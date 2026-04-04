"""Replay buffer storing optimization trajectories as hard negatives.

**Researcher summary:**
    Implements a replay buffer for EBM training that stores intermediate states
    from repair/optimization trajectories and uses them as additional negative
    examples during NCE training. This shapes the energy landscape along entire
    repair paths, not just near the training data.

**Detailed explanation for engineers:**
    During standard NCE training, the energy landscape is only well-shaped near
    the training data (low energy) and the noise distribution (high energy). The
    vast space between data and noise is uncharted — the energy function can have
    arbitrary shape there, including spurious minima that trap samplers.

    **The problem:**
    When a sampler or repair algorithm tries to move from a random starting point
    toward the data, it passes through this uncharted region. If the energy
    landscape has valleys or flat spots there, the sampler gets stuck.

    **The solution: replay buffer**
    During optimization training (see optimization_training.py), the inner gradient
    descent produces a trajectory of intermediate states: y_hat_0, y_hat_1, ...,
    y_hat_N. These intermediate states are exactly the points where the energy
    landscape needs to be well-shaped.

    The replay buffer stores these intermediate states and feeds them back as
    additional negative examples during NCE training:
    - Data samples get LOW energy (as usual)
    - Noise samples get HIGH energy (as usual)
    - Replay samples also get HIGH energy (new!)

    This "fills in" the energy landscape between data and noise, ensuring the
    energy monotonically increases as you move away from the data along any
    repair trajectory.

    **Implementation:**
    The buffer is a simple FIFO (first-in, first-out) ring buffer. When it
    reaches max_size, the oldest entries are evicted. Sampling is uniform
    random — no prioritization (though prioritized replay would be a natural
    extension).

    **Reference:** arxiv 2507.02092, Section 3.2.

Spec: REQ-TRAIN-006
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

# Type alias for energy functions used in the NCE loss computation.
EnergyCallable = Callable[[jax.Array], jax.Array]


@dataclass
class ReplayBuffer:
    """Fixed-size buffer storing states from optimization trajectories.

    **Researcher summary:**
        FIFO ring buffer with uniform random sampling. Stores JAX arrays
        from repair trajectories for use as hard negatives during training.

    **Detailed explanation for engineers:**
        This is a straightforward data structure: a list that caps at max_size
        by evicting the oldest entries. The key design choice is simplicity —
        no prioritization, no weighting, no GPU storage optimization. This is
        adequate for moderate-scale experiments; production systems might want
        a GPU-resident circular buffer with prioritized sampling.

        **Why a dataclass?**
        Using @dataclass gives us __init__, __repr__, and __eq__ for free.
        The _buffer field uses field(default_factory=list) to ensure each
        instance gets its own empty list (a common Python gotcha with mutable
        default arguments).

    Attributes:
        max_size: Maximum number of states to store. When exceeded, the oldest
            states are evicted (FIFO). Default: 10000.

    Spec: REQ-TRAIN-006
    """

    max_size: int = 10000
    _buffer: list[jax.Array] = field(default_factory=list)

    def add(self, states: jax.Array) -> None:
        """Add states from a repair trajectory to the buffer.

        **Detailed explanation for engineers:**
            Accepts a batch of states (2-D array where each row is one state)
            or a single state (1-D array). Each state is appended individually.
            If the buffer exceeds max_size, the oldest entries are removed
            from the front of the list.

            **Performance note:** Python list.pop(0) is O(n) because it shifts
            all elements. For large buffers (>100k), consider using
            collections.deque which has O(1) popleft. For our typical use
            case (10k states), the difference is negligible.

        Args:
            states: A single state (1-D array, shape (dim,)) or a batch of
                states (2-D array, shape (n_states, dim)).

        Spec: REQ-TRAIN-006
        """
        if states.ndim == 1:
            # Single state — wrap in a list for uniform handling.
            self._buffer.append(states)
        else:
            # Batch of states — add each row individually.
            for i in range(states.shape[0]):
                self._buffer.append(states[i])

        # Evict oldest entries if we exceeded max_size.
        # This maintains the FIFO (first-in, first-out) invariant.
        while len(self._buffer) > self.max_size:
            self._buffer.pop(0)

    def sample(self, n: int, key: jax.Array) -> jax.Array:
        """Sample n states uniformly at random from the buffer.

        **Detailed explanation for engineers:**
            Uses JAX's PRNG to generate random indices into the buffer, then
            stacks the selected states into a batch array. Sampling is with
            replacement (the same state can appear multiple times in the
            returned batch). This is standard practice for replay buffers
            and simplifies the implementation.

            **Why not jax.random.choice?**
            jax.random.choice works on JAX arrays, but our buffer is a Python
            list. We generate random integer indices and index into the list
            directly. The result is then stacked into a JAX array.

        Args:
            n: Number of states to sample.
            key: JAX PRNG key for reproducible random sampling.

        Returns:
            A batch of sampled states, shape (n, dim).

        Raises:
            ValueError: If the buffer is empty (nothing to sample from).

        Spec: REQ-TRAIN-006
        """
        if len(self._buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        # Generate random indices in [0, buffer_size).
        indices = jax.random.randint(key, shape=(n,), minval=0, maxval=len(self._buffer))

        # Gather the selected states and stack into a batch array.
        # We convert indices to Python ints for list indexing.
        selected = [self._buffer[int(idx)] for idx in indices]
        return jnp.stack(selected)

    def __len__(self) -> int:
        """Return the current number of states in the buffer.

        Spec: REQ-TRAIN-006
        """
        return len(self._buffer)


def nce_loss_with_replay(
    energy_fn: EnergyCallable,
    data_batch: jax.Array,
    noise_batch: jax.Array,
    replay_batch: jax.Array,
    replay_weight: float = 0.5,
) -> jax.Array:
    """NCE loss with additional replay buffer negatives.

    **Researcher summary:**
        L = nce_base(data, noise) + replay_weight * mean(log_sigmoid(E(replay)))
        The replay term pushes energy up along repair trajectories.

    **Detailed explanation for engineers:**
        This extends the standard NCE loss with an extra term for replay buffer
        samples. The standard NCE loss has two parts:

        1. **Data term:** Push energy DOWN on real data
           -mean(log sigmoid(-E(x_data)))

        2. **Noise term:** Push energy UP on random noise
           -mean(log sigmoid(E(x_noise)))

        We add a third term:

        3. **Replay term:** Push energy UP on replay buffer samples
           -replay_weight * mean(log sigmoid(E(x_replay)))

        The replay samples are intermediate states from previous optimization
        trajectories. By pushing their energy up, we ensure the energy landscape
        slopes downward along the path from random starts to the data — exactly
        what gradient-based repair needs.

        **Why a separate weight?**
        The replay term is auxiliary — it should guide the landscape shape
        without overwhelming the primary data/noise signal. A weight of 0.5
        means the replay term contributes half as much as the noise term.

        **Numerical stability:**
        Like standard NCE, we use jax.nn.log_sigmoid for numerically stable
        computation of log(sigmoid(z)).

    Args:
        energy_fn: Callable that takes a 1-D array and returns scalar energy.
        data_batch: Real data samples, shape (batch_size, dim).
        noise_batch: Noise samples from known distribution, shape (n_noise, dim).
        replay_batch: States sampled from the replay buffer, shape (n_replay, dim).
        replay_weight: Weight for the replay term relative to the noise term.
            Default: 0.5.

    Returns:
        Scalar loss: NCE loss plus weighted replay term.

    For example::

        import jax
        import jax.numpy as jnp

        def energy(x):
            return 0.5 * jnp.sum(x ** 2)

        data = jnp.zeros((8, 4))
        noise = jnp.ones((8, 4)) * 5.0
        replay = jnp.ones((8, 4)) * 2.5  # intermediate states
        loss = nce_loss_with_replay(energy, data, noise, replay)

    Spec: REQ-TRAIN-006
    """
    # Vectorize the energy function over the batch dimension.
    # jax.vmap transforms a single-sample function into a batched one.
    data_energies = jax.vmap(energy_fn)(data_batch)
    noise_energies = jax.vmap(energy_fn)(noise_batch)
    replay_energies = jax.vmap(energy_fn)(replay_batch)

    # Data term: push energy DOWN on real data.
    # When E(data) is low, sigmoid(-E) is close to 1, log sigmoid is close to 0.
    data_term = -jnp.mean(jax.nn.log_sigmoid(-data_energies))

    # Noise term: push energy UP on random noise.
    # When E(noise) is high, sigmoid(E) is close to 1, log sigmoid is close to 0.
    noise_term = -jnp.mean(jax.nn.log_sigmoid(noise_energies))

    # Replay term: push energy UP on replay buffer samples.
    # Same logic as noise term, but weighted separately so the replay signal
    # does not overwhelm the primary NCE training signal.
    replay_term = -jnp.mean(jax.nn.log_sigmoid(replay_energies))

    return data_term + noise_term + replay_weight * replay_term
