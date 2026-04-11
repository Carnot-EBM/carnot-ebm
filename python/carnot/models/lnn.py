"""Liquid Constraint Model — coupling matrix evolves via MLP-parameterized ODE.

**Researcher summary:**
    Unlike static EBMs (Ising, Gibbs) with fixed coupling J, this model
    evolves J via a learned ODE: dJ/dt = f(J, observation), where f is a
    small 2-layer MLP. Each call to step(observation) advances J by one
    Euler step. Energy is computed from the *current* (adapted) J. This
    enables the model to strengthen or weaken constraint coupling as new
    facts arrive across a multi-step agent reasoning chain.

**Why "Liquid"?**
    Liquid Neural Networks (Hasani et al. 2021) use input-dependent dynamics
    so the network's behaviour adapts at inference time. Here, the coupling
    matrix J plays the role of the adaptive state: the MLP determines how
    each new observation shifts the constraint strengths.

**The ODE and its Euler discretisation:**

    Continuous form (one scalar element):
        dJ_ij/dt = f_ij(observation)     where f is an MLP

    Euler step (per call to step()):
        J_{t+1} = J_t + dt * MLP(observation)

    Symmetry is enforced after each step so the energy remains quadratic:
        J = (J + J^T) / 2

**Energy formula:**
    E(state) = -0.5 * state^T J state - b^T state

    Identical to the Ising formula (REQ-TIER-001), but with a J that shifts
    after each agent step rather than staying fixed at training time.

**Training goal:**
    Learn MLP weights so that, across a sequence of (observation, label)
    pairs, the model assigns lower energy to positive examples (label=+1)
    and higher energy to negative examples (label=-1).

    Loss per step: label * E(observation)
    Total loss:    mean over sequence steps

    Trained via `jax.value_and_grad` with full backpropagation through the
    sequence of Euler steps (BPTT-style unrolling via Python for-loops).

**Relationship to existing Carnot architecture:**
    - Implements EnergyFunction protocol (satisfies REQ-CORE-002).
    - Inherits AutoGradMixin: grad_energy and energy_batch are free.
    - Can replace IsingModel in ConstraintStateMachine for adaptive checking.
    - Complements LNNConstraintModel (lnn_constraint.py) which evolves a
      *hidden state* h; this model evolves the *coupling matrix* J directly.

Spec: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


@dataclass
class LiquidConstraintConfig:
    """Configuration for the Liquid Constraint Model.

    **Detailed explanation for engineers:**
        These hyperparameters are chosen before construction and cannot be
        changed afterwards. They control the input dimension, the capacity
        of the MLP that drives coupling evolution, the Euler step size, and
        how the initial coupling matrix is seeded.

    Attributes:
        input_dim: Dimension of observation vectors and the constraint
            state vector. The coupling matrix J has shape
            (input_dim, input_dim). Default 8.
        mlp_hidden_dim: Number of hidden units in the 2-layer MLP that
            maps observations to coupling updates dJ. Larger values give
            richer dynamics at the cost of more parameters. Default 16.
        dt: Euler integration step size for the ODE. Smaller values give
            finer-grained evolution but require more steps to shift J
            significantly. Default 0.1.
        coupling_init: How to initialise the coupling matrix J_0.
            - "xavier_uniform" (default): random values scaled so that
              the energy magnitude is stable regardless of input_dim.
            - "zeros": all-zero J (no pairwise interactions at start).

    For example::

        config = LiquidConstraintConfig(input_dim=8, mlp_hidden_dim=16)
        config.validate()  # raises ValueError if invalid

    Spec: REQ-CORE-001
    """

    input_dim: int = 8
    mlp_hidden_dim: int = 16
    dt: float = 0.1
    coupling_init: str = "xavier_uniform"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises:
            ValueError: If input_dim <= 0, mlp_hidden_dim <= 0, or dt <= 0.

        Spec: SCENARIO-CORE-001
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.mlp_hidden_dim <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")


class LiquidConstraintModel(AutoGradMixin):
    """Energy-Based Model with MLP-driven coupling matrix evolution.

    **Researcher summary:**
        Adaptive EBM where J evolves via dJ/dt = MLP(observation). Each
        step(obs) applies one Euler step. energy(state) uses the current J.
        Train MLP weights so that label * E(obs) is minimised across sequences.

    **Detailed explanation for engineers:**

        **Parameters (two groups):**

        *Initial coupling (frozen after training — the "prior"):*
        - J0: shape (input_dim, input_dim), symmetric — the starting
          point for J before any steps are taken. reset() returns to J0.
        - b0: shape (input_dim,) — initial bias vector.

        *MLP weights (trained — the "dynamics"):*
        - W1, b1: first linear layer, maps (input_dim,) -> (mlp_hidden,)
          with tanh activation.
        - W2_J, b2_J: second linear layer for coupling updates, maps
          (mlp_hidden,) -> (input_dim**2,). Reshaped to (d, d) and
          symmetrised to produce the coupling increment dJ.
        - W2_b, b2_b: second linear layer for bias updates, maps
          (mlp_hidden,) -> (input_dim,) to produce increment db.

        **Usage pattern in an agent pipeline:**

            model = LiquidConstraintModel(LiquidConstraintConfig(input_dim=16))

            # At each reasoning step, incorporate the new observation:
            model.step(embed(step_output))
            violation_energy = model.energy(embed(step_output))

            # Between independent agent sessions:
            model.reset()

        **Why BPTT for training?**
        The coupling J evolves sequentially through the steps of a reasoning
        chain. Backpropagation through time (unrolled Python for-loops inside
        jax.value_and_grad) gives exact gradients through this sequential
        evolution, correctly attributing later errors to early MLP decisions.

    Spec: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
    """

    def __init__(
        self,
        config: LiquidConstraintConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Initialise model parameters.

        **Detailed explanation for engineers:**
            Validates config, then initialises J0 and b0 (the starting
            coupling) plus the six MLP parameter matrices. All mutable state
            (J, b) starts equal to J0, b0 and is updated in-place by step().

            MLP output weights use a very small initialisation (0.01× scale)
            so that the initial coupling evolves slowly, maintaining energy
            stability from the first call to step().

        Args:
            config: LiquidConstraintConfig specifying architecture.
            key: JAX PRNG key. If None, uses seed 0 for reproducibility.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        d = config.input_dim
        h = config.mlp_hidden_dim

        k1, k2, k3, k4, k5 = jrandom.split(key, 5)

        # ── Initial coupling matrix J0 ───────────────────────────────────
        if config.coupling_init == "xavier_uniform":
            # Xavier uniform keeps energy magnitude stable: scale by sqrt(6/(d+d))
            lim = jnp.sqrt(6.0 / (d + d))
            j = jrandom.uniform(k1, (d, d), minval=-lim, maxval=lim)
        elif config.coupling_init == "zeros":
            # No pairwise interactions at initialisation — blank slate.
            j = jnp.zeros((d, d))
        else:
            raise ValueError(f"Unknown initializer: {config.coupling_init!r}")

        # Enforce symmetry: J_ij = J_ji so energy x^T J x is well-defined.
        self.J0: jax.Array = (j + j.T) / 2.0
        self.b0: jax.Array = jnp.zeros(d)

        # Mutable copies that evolve via step()
        self.J: jax.Array = self.J0
        self.b: jax.Array = self.b0

        # ── MLP layer 1: input -> hidden ─────────────────────────────────
        # Xavier init keeps gradients flowing through the tanh activation.
        lim1 = jnp.sqrt(6.0 / (d + h))
        self.W1: jax.Array = jrandom.uniform(k2, (h, d), minval=-lim1, maxval=lim1)
        self.b1: jax.Array = jnp.zeros(h)

        # ── MLP layer 2a: hidden -> coupling update dJ ───────────────────
        # Small init (0.01×) keeps coupling increments tiny at the start,
        # preventing energy blow-up before training has shaped the dynamics.
        lim2j = jnp.sqrt(6.0 / (h + d * d)) * 0.01
        self.W2_J: jax.Array = jrandom.uniform(k3, (d * d, h), minval=-lim2j, maxval=lim2j)
        self.b2_J: jax.Array = jnp.zeros(d * d)

        # ── MLP layer 2b: hidden -> bias update db ───────────────────────
        lim2b = jnp.sqrt(6.0 / (h + d)) * 0.01
        self.W2_b: jax.Array = jrandom.uniform(k4, (d, h), minval=-lim2b, maxval=lim2b)
        self.b2_b: jax.Array = jnp.zeros(d)

    # ── Pure functional MLP apply (used in jax.value_and_grad training) ──

    def _mlp_apply(
        self,
        obs: jax.Array,
        params: tuple[jax.Array, ...],
    ) -> tuple[jax.Array, jax.Array]:
        """Apply MLP to observation and return (dJ, db) — purely functional.

        **Detailed explanation for engineers:**
            This is a pure function (no side effects, all state passed
            explicitly as `params`) so that jax.value_and_grad can
            differentiate through it during training.

            Architecture:
                hidden = tanh(W1 @ obs + b1)          shape: (mlp_hidden,)
                dJ_flat = W2_J @ hidden + b2_J         shape: (d*d,)
                dJ = reshape(dJ_flat, d, d)             shape: (d, d)
                dJ = (dJ + dJ^T) / 2                   symmetrised
                db  = W2_b @ hidden + b2_b             shape: (d,)

        Args:
            obs: Observation vector, shape (input_dim,).
            params: Tuple (W1, b1, W2_J, b2_J, W2_b, b2_b) of MLP weights.

        Returns:
            (dJ, db): Coupling increment (d, d) and bias increment (d,).

        Spec: REQ-CORE-001
        """
        W1, b1, W2_J, b2_J, W2_b, b2_b = params
        d = self.config.input_dim
        # Hidden layer with tanh: keeps activations in [-1, 1]
        hidden = jnp.tanh(W1 @ obs + b1)
        # Flat coupling update, reshaped and symmetrised
        dJ_flat = W2_J @ hidden + b2_J
        dJ = dJ_flat.reshape(d, d)
        dJ = (dJ + dJ.T) / 2.0
        # Bias update
        db = W2_b @ hidden + b2_b
        return dJ, db

    def step(self, observation: jax.Array) -> None:
        """Advance the coupling matrix one Euler step based on observation.

        **Researcher summary:**
            J_{t+1} = J_t + dt * MLP(observation), then symmetrised.
            b_{t+1} = b_t + dt * MLP_b(observation).

        **Detailed explanation for engineers:**
            Each reasoning step in an agent chain introduces new information
            via `observation`. step() feeds that observation through the MLP
            to get (dJ, db), then applies one Euler step of the ODE to J and b.

            Symmetrisation after the step keeps J symmetric — required because
            the energy formula assumes J_ij = J_ji.

        Args:
            observation: New observation to incorporate, shape (input_dim,).
                Typically an embedding of the current agent turn's output.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        params = (self.W1, self.b1, self.W2_J, self.b2_J, self.W2_b, self.b2_b)
        dJ, db = self._mlp_apply(observation, params)
        # Euler step for J, then re-symmetrise
        J_new = self.J + self.config.dt * dJ
        self.J = (J_new + J_new.T) / 2.0
        # Euler step for bias
        self.b = self.b + self.config.dt * db

    def energy(self, state: jax.Array) -> jax.Array:
        """Compute scalar energy E(state) = -0.5 * state^T J state - b^T state.

        **Researcher summary:**
            Ising-style quadratic energy using the *current* (adapted) J.
            Lower energy → state is more consistent with accumulated context.

        **Detailed explanation for engineers:**
            This is identical in form to IsingModel.energy(), but J is not
            fixed — it has been shifted by prior calls to step(). So the same
            state x will yield a different energy before and after seeing new
            observations, reflecting updated constraint strengths.

            The energy is computed from self.J and self.b as they stand at
            call time. It does NOT advance J — that is step()'s responsibility.

        Args:
            state: Input configuration, shape (input_dim,).

        Returns:
            Scalar energy (0-D JAX array).

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        return -0.5 * state @ self.J @ state - self.b @ state

    def reset(self) -> None:
        """Restore J and b to their initial (pre-step) values.

        **Detailed explanation for engineers:**
            Reverts self.J to self.J0 and self.b to self.b0, as if no
            step() calls had been made. Call this between independent agent
            sessions to avoid context leakage from one reasoning chain into
            the next.

            Note: MLP weights (W1, b1, etc.) are NOT reset — only the
            mutable coupling state is restored.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        self.J = self.J0
        self.b = self.b0

    def train(
        self,
        observations: jax.Array,
        labels: jax.Array,
        n_epochs: int = 50,
        lr: float = 0.01,
    ) -> list[float]:
        """Train MLP weights via BPTT on energy-based classification sequences.

        **Researcher summary:**
            Each sequence drives J through multiple Euler steps. At each step,
            loss += label * E(obs). Backpropagation unrolls through all steps
            to give exact gradients for MLP weights.

        **Detailed explanation for engineers:**
            The training signal is:
                loss = mean over sequences of: sum_{t} label_t * E_t(obs_t)

            - label_t = +1: we want low energy at step t (obs is "correct")
            - label_t = -1: we want high energy at step t (obs is "incorrect")

            The computation is:
            1. Start from J0, b0 (initial coupling).
            2. For each step t in a sequence: run MLP(obs_t) → dJ, db → update J.
            3. Compute E_t = -0.5 obs_t^T J_t obs_t - b_t^T obs_t.
            4. Accumulate label_t * E_t.
            5. After all steps and sequences: backprop with jax.value_and_grad.
            6. Update MLP weights with gradient descent.

            The Python for-loops are unrolled by JAX during tracing, giving
            exact gradients (BPTT) through the full temporal sequence.

            After training, J and b remain at their post-training current state.
            Call reset() if you want to start a new agent session from J0, b0.

        Args:
            observations: shape (n_sequences, seq_len, input_dim). Each
                [i, t, :] is the embedding for sequence i at step t.
            labels: shape (n_sequences, seq_len). Values +1.0 or -1.0.
                +1 → this observation should have low energy.
                -1 → this observation should have high energy.
            n_epochs: Number of gradient descent steps over the full dataset.
            lr: Learning rate for gradient descent on MLP weights.

        Returns:
            List of per-epoch total loss values for convergence monitoring.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        J0 = self.J0
        b0 = self.b0
        dt = self.config.dt
        d = self.config.input_dim
        n_seqs = int(observations.shape[0])
        seq_len = int(observations.shape[1])

        def sequence_loss(
            params: tuple[jax.Array, ...],
            obs_seqs: jax.Array,
            lbl_seqs: jax.Array,
        ) -> jax.Array:
            """Pure functional loss — differentiable w.r.t. params.

            **How the for-loops work with JAX:**
                Python for-loops are traced (unrolled) by JAX. Each iteration
                creates nodes in the computation graph. This gives exact
                gradients at the cost of longer compile time for long sequences.
                The loop bounds (n_seqs, seq_len) must be Python ints — they are,
                since they come from array.shape which returns a concrete tuple.
            """
            W1, b1, W2_J, b2_J, W2_b, b2_b = params
            total = jnp.array(0.0)
            for i in range(n_seqs):
                # Each sequence starts from the initial (prior) coupling.
                J = J0
                b_vec = b0
                for t in range(seq_len):
                    obs = obs_seqs[i, t]
                    # MLP forward: observation → (dJ, db)
                    hidden = jnp.tanh(W1 @ obs + b1)
                    dJ_flat = W2_J @ hidden + b2_J
                    dJ = dJ_flat.reshape(d, d)
                    dJ = (dJ + dJ.T) / 2.0
                    db = W2_b @ hidden + b2_b
                    # Euler step
                    J = J + dt * dJ
                    J = (J + J.T) / 2.0
                    b_vec = b_vec + dt * db
                    # Energy-based classification loss for this step
                    e = -0.5 * obs @ J @ obs - b_vec @ obs
                    total = total + lbl_seqs[i, t] * e
            return total / n_seqs

        # Pack current MLP weights into a pytree for jax.value_and_grad
        params: tuple[jax.Array, ...] = (
            self.W1, self.b1, self.W2_J, self.b2_J, self.W2_b, self.b2_b
        )

        losses: list[float] = []
        for _ in range(n_epochs):
            loss_val, grads = jax.value_and_grad(sequence_loss)(
                params, observations, labels
            )
            # Gradient descent: move params in the direction that lowers loss
            params = tuple(p - lr * g for p, g in zip(params, grads))
            losses.append(float(loss_val))

        # Store updated MLP weights back onto self
        self.W1, self.b1, self.W2_J, self.b2_J, self.W2_b, self.b2_b = params
        return losses

    @property
    def input_dim(self) -> int:
        """Number of input dimensions (size of the state/observation vector)."""
        return self.config.input_dim


__all__ = ["LiquidConstraintConfig", "LiquidConstraintModel"]
