"""Liquid Neural Network (LNN) Constraint Model — adaptive EBM for agentic reasoning.

**Researcher summary:**
    Implements an adaptive Energy-Based Model based on Liquid Time-Constant Networks
    (LTCN). Unlike static Ising models with fixed coupling J, the LNN constraint model
    learns to *update* its coupling strengths in response to new observations via a
    gated continuous-time hidden state. This enables the constraint model to adapt as
    an agent reasons through a multi-step chain — new facts shift which constraints
    matter most.

**The Liquid Time-Constant Network (LTCN) formulation:**

    Standard recurrent networks evolve a hidden state h with a fixed time constant
    (e.g., LSTM gates). LTCNs use *input-dependent* time constants:

        τ(x) = τ_base / (1 + |W_gate · x|)

    The hidden state then evolves as:

        dh/dt = (1/τ(x)) × (−h + W_rec · tanh(h) + W_in · x + b)

    In the discrete Euler approximation (one step per call):

        h_new = h + dt × (1/τ(x)) × (−h + W_rec · tanh(h) + W_in · x + b)

    The energy is computed from the evolved hidden state:

        E(x) = −0.5 × h^T · J_eff · h − b_eff^T · tanh(h)

    where J_eff and b_eff are fixed parameters learned via CD training.

**Why does this help for agentic reasoning?**

    A static Ising model has a fixed energy landscape — it cannot "learn" from
    earlier steps in a reasoning chain. If step 1 establishes "x > 5", the Ising
    model cannot use this to make step 3's constraint "x > 4" more salient.

    The LNN's hidden state accumulates context across steps via adapt(observation).
    Each call to adapt() runs one forward pass of the LTCN, updating h to reflect
    the new observation. The energy landscape shifts in response. This means:
    - Early steps that establish key facts "tune" the network's sensitivity
    - Later contradictions produce larger energy spikes (easier to detect)
    - Errors introduced at steps 2-4 create asymmetric energy signatures

**Relationship to existing Carnot architecture:**
    - Implements EnergyFunction protocol (energy, energy_batch, grad_energy, input_dim)
    - Inherits AutoGradMixin for automatic grad_energy and energy_batch derivation
    - Designed to replace IsingModel in agentic pipelines (ConstraintState, propagate())
    - Training uses Contrastive Divergence (CD-1) which is simpler than DSM for
      recurrent models where the gradient must flow through the hidden state dynamics

**Target use case (Research-Program Goal #8):**
    Qwen3.5-0.8B or gemma-4-E4B-it produces a 5-step reasoning chain. At each step,
    the LNN is adapted with the step's output embedding. The energy of each step's
    output measures constraint violation — higher energy = more suspicious reasoning.

Spec: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


@dataclass
class LNNConstraintConfig:
    """Configuration for the LNN Constraint Model.

    **Detailed explanation for engineers:**
        Specifies the dimensions, liquid time-constant parameters, and initialization
        strategy for the adaptive constraint EBM. These hyperparameters are fixed
        after construction.

    Attributes:
        input_dim: Dimensionality of the input observation vector (e.g., embedding size).
        hidden_dim: Number of hidden units in the LTCN. Larger hidden_dim gives more
            expressive constraint representations but slower adaptation.
        tau_base: Base time constant for the liquid dynamics. Controls how quickly the
            hidden state responds to new observations. Smaller = faster adaptation.
            Default 1.0 is a balanced starting point.
        dt: Euler integration step size for discretizing dh/dt. Default 0.1 is stable
            for typical inputs in [-1, 1].
        coupling_init: How to initialize the energy output coupling matrix J_eff.
            "xavier_uniform" (default) or "zeros".

    For example::

        config = LNNConstraintConfig(input_dim=16, hidden_dim=8)
        config.validate()

    Spec: REQ-CORE-001
    """

    input_dim: int = 32
    hidden_dim: int = 16
    tau_base: float = 1.0
    dt: float = 0.1
    coupling_init: str = "xavier_uniform"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises ValueError if dimensions are non-positive or time constants are invalid.

        Spec: SCENARIO-CORE-001
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if self.tau_base <= 0.0:
            raise ValueError("tau_base must be > 0")
        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")


class LNNConstraintModel(AutoGradMixin):
    """Adaptive Energy-Based Model using Liquid Time-Constant Networks.

    **Researcher summary:**
        LTCN-based EBM where hidden state h evolves with input-dependent time
        constant τ(x). Energy is computed from h. adapt(obs) updates h one step,
        accumulating reasoning context across agent turns.

    **Detailed explanation for engineers:**

        This model has two sets of parameters:

        **LTCN dynamics parameters** (govern how h evolves):
        - W_in: (hidden_dim, input_dim) — maps observation x into the hidden state
        - W_rec: (hidden_dim, hidden_dim) — recurrent connections within h
        - W_gate: (hidden_dim, input_dim) — computes input-dependent time constant
        - b_dyn: (hidden_dim,) — dynamics bias

        **Energy output parameters** (compute energy from h):
        - J_eff: (hidden_dim, hidden_dim) — coupling matrix (symmetric) over hidden units
        - b_eff: (hidden_dim,) — energy bias over hidden units

        **How adaptation works:**
        1. Compute time constant: τ(x) = τ_base / (1 + sum|W_gate · x|)
           - Input-dependent: if x is "surprising", τ is small → fast update
           - If x is "routine", τ is large → slow update (stability)
        2. Compute dynamics: Δh = (1/τ) × (-h + W_rec·tanh(h) + W_in·x + b_dyn)
        3. Update: h_new = h + dt × Δh  (Euler step)

        **How energy is computed:**
        Given the current hidden state h (after adaptation):
            E(x) = −0.5 × tanh(h)^T J_eff tanh(h) − b_eff^T tanh(h)

        This is analogous to the Ising energy but over the *evolved hidden state*
        rather than directly over x. The tanh squashing prevents energy blowup.

    For example::

        import jax.numpy as jnp
        model = LNNConstraintModel(LNNConstraintConfig(input_dim=8, hidden_dim=4))

        # Static energy (no adaptation)
        x = jnp.ones(8)
        e1 = model.energy(x)

        # Adapt to an observation, then compute energy again
        model.adapt(jnp.array([1.0] * 8))
        e2 = model.energy(x)
        # e2 will differ from e1 because the hidden state has changed

        # Reset to initial state
        model.reset()
        assert model.energy(x) == e1

    Spec: REQ-CORE-001, REQ-CORE-002, SCENARIO-CORE-001
    """

    def __init__(
        self,
        config: LNNConstraintConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Create LNN Constraint Model with initialized parameters.

        **Detailed explanation for engineers:**
            Validates config, then initializes the LTCN dynamics parameters and
            energy output parameters. The hidden state h is initialized to zeros and
            is updated in-place by adapt().

        Args:
            config: LNNConstraintConfig specifying dimensions and hyperparameters.
            key: JAX PRNG key for parameter initialization. If None, uses seed 0.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        d_in = config.input_dim
        d_h = config.hidden_dim

        # Split the PRNG key into four sub-keys for independent initialization
        # of each parameter matrix.
        k1, k2, k3, k4 = jrandom.split(key, 4)

        # --- LTCN dynamics parameters ---

        # W_in: maps input x (d_in,) into hidden space (d_h,)
        # Xavier uniform: scale by sqrt(6 / (fan_in + fan_out))
        lim_in = jnp.sqrt(6.0 / (d_in + d_h))
        self.W_in: jax.Array = jrandom.uniform(k1, (d_h, d_in), minval=-lim_in, maxval=lim_in)

        # W_rec: recurrent connections within hidden state (d_h, d_h)
        # Initialized with small values to start near stable fixed point
        lim_rec = jnp.sqrt(6.0 / (d_h + d_h)) * 0.1
        self.W_rec: jax.Array = jrandom.uniform(k2, (d_h, d_h), minval=-lim_rec, maxval=lim_rec)

        # W_gate: computes input-dependent gate for time constant (d_h, d_in)
        # Uses small initialization so initial τ ≈ τ_base
        lim_gate = jnp.sqrt(6.0 / (d_in + d_h)) * 0.1
        self.W_gate: jax.Array = jrandom.uniform(k3, (d_h, d_in), minval=-lim_gate, maxval=lim_gate)

        # b_dyn: dynamics bias (d_h,), zero-initialized
        self.b_dyn: jax.Array = jnp.zeros(d_h)

        # --- Energy output parameters ---

        # J_eff: symmetric coupling over hidden units (d_h, d_h)
        if config.coupling_init == "xavier_uniform":
            lim_j = jnp.sqrt(6.0 / (d_h + d_h))
            j = jrandom.uniform(k4, (d_h, d_h), minval=-lim_j, maxval=lim_j)
        elif config.coupling_init == "zeros":
            j = jnp.zeros((d_h, d_h))
        else:
            raise ValueError(f"Unknown initializer: {config.coupling_init}")

        # Enforce symmetry so energy is well-defined (x^T J x = x^T J^T x)
        self.J_eff: jax.Array = (j + j.T) / 2.0
        self.b_eff: jax.Array = jnp.zeros(d_h)

        # --- Mutable hidden state ---
        # Initialized to zeros. Updated by adapt(). Reset by reset().
        self._h: jax.Array = jnp.zeros(d_h)
        # Store the initial hidden state for reset()
        self._h0: jax.Array = jnp.zeros(d_h)

    def _ltcn_step(self, h: jax.Array, x: jax.Array) -> jax.Array:
        """Perform one Euler step of the LTCN dynamics.

        **Detailed explanation for engineers:**
            This is the core of the Liquid Time-Constant Network. It computes:

            1. gate_activation = |W_gate · x|  — how "active" the gate is
            2. τ(x) = τ_base / (1 + mean(gate_activation))
               - Large gate activation → small τ → fast update
               - Small gate activation → large τ → slow update (stability)
            3. Δh = (1/τ) × (-h + W_rec·tanh(h) + W_in·x + b_dyn)
               - "-h" is the decay term: without input, h drifts to 0
               - "W_rec·tanh(h)" is the recurrent memory: past state influences current
               - "W_in·x" is the input injection
               - "b_dyn" is the learned bias
            4. h_new = h + dt × Δh  (Euler integration)

        This implements the continuous-time ODE:
            τ(x) × dh/dt = -h + W_rec·tanh(h) + W_in·x + b_dyn

        Args:
            h: Current hidden state, shape (hidden_dim,).
            x: Current input observation, shape (input_dim,).

        Returns:
            Updated hidden state h_new, shape (hidden_dim,).

        Spec: REQ-CORE-001
        """
        # Step 1: Compute input-dependent time constant
        # gate_act has shape (hidden_dim,) — each hidden unit has its own gate
        gate_act = jnp.abs(self.W_gate @ x)
        # τ(x) is a scalar: base time constant scaled down by gate activity.
        # Adding 1 prevents division by zero when gate_act is all zeros.
        tau = self.config.tau_base / (1.0 + jnp.mean(gate_act))

        # Step 2: Compute the dynamics Δh
        # Recurrent contribution: tanh ensures hidden values stay bounded in [-1, 1]
        recurrent = self.W_rec @ jnp.tanh(h)
        # Input contribution: project observation x into hidden space
        input_contrib = self.W_in @ x
        # Combined: the "target" toward which h is being pulled (before scaling by 1/τ)
        delta_h = (1.0 / tau) * (-h + recurrent + input_contrib + self.b_dyn)

        # Step 3: Euler integration step
        h_new = h + self.config.dt * delta_h
        return h_new

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy from the *current* hidden state.

        **Researcher summary:**
            E(x) = -0.5 * tanh(h)^T J_eff tanh(h) - b_eff^T tanh(h), where h is
            the current LTCN hidden state (updated by adapt() calls).

        **Detailed explanation for engineers:**
            Unlike the Ising model which computes energy directly from the input x,
            the LNN computes energy from the evolved hidden state h. The input x
            is used during energy computation to run one forward step of the LTCN
            and compute the energy from the resulting hidden state.

            Why tanh(h)?  The tanh activation squashes h into [-1, 1], ensuring
            the energy quadratic term stays numerically well-behaved regardless of
            how large the hidden state grows during adaptation.

            The formula is analogous to Ising: E(x) = -0.5 x^T J x - b^T x,
            but with tanh(h) substituted for x, and with h incorporating
            context from prior adapt() calls.

        Args:
            x: Input configuration, shape (input_dim,). Used to compute energy
               by running the LTCN one step forward from the current hidden state.
               Note: this does NOT mutate self._h; it only uses a transient step.

        Returns:
            Scalar energy (0-D JAX array).

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        # Compute energy from a one-step forward pass of LTCN starting from current h.
        # We do NOT update self._h here — energy() is a pure function of (x, self._h).
        # Updating happens exclusively in adapt().
        h_for_energy = self._ltcn_step(self._h, x)
        h_tanh = jnp.tanh(h_for_energy)
        # Ising-style quadratic energy over hidden space
        coupling_term = -0.5 * h_tanh @ self.J_eff @ h_tanh
        bias_term = -self.b_eff @ h_tanh
        return coupling_term + bias_term

    def adapt(self, observation: jax.Array) -> None:
        """Update the hidden state in response to a new observation.

        **Researcher summary:**
            Runs one Euler step of the LTCN ODE using the observation as input.
            Updates self._h in-place. Call this after each reasoning step to
            accumulate context for subsequent energy evaluations.

        **Detailed explanation for engineers:**
            This is the key method that makes the LNN model *adaptive*. Each call
            to adapt() evolves the hidden state h by one LTCN step, encoding
            information from the observation into the model's "memory".

            After adapt(), subsequent calls to energy() will see the updated h,
            producing different energy values that reflect the accumulated context.

            **Usage in agentic reasoning:**
            ```
            model = LNNConstraintModel(config)
            for step in reasoning_chain:
                embedding = embed(step.output_text)
                model.adapt(embedding)       # encode this step's context
                e = model.energy(embedding)  # evaluate constraint violation
            ```

        Args:
            observation: New observation to incorporate, shape (input_dim,).
                Typically an embedding of the current reasoning step's output.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        # Run one LTCN step and update the hidden state in-place
        self._h = self._ltcn_step(self._h, observation)

    def reset(self) -> None:
        """Reset hidden state to initial value (all zeros).

        **Detailed explanation for engineers:**
            Restores self._h to the zero vector it had at construction time.
            After reset(), energy() and adapt() behave exactly as if the model
            were freshly constructed (given the same parameters).

            Call this between independent reasoning chains (e.g., between
            different LLM generations) to avoid context leakage.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        self._h = self._h0.copy()

    def train_cd(
        self,
        data: jax.Array,
        n_epochs: int = 50,
        lr: float = 0.01,
        cd_steps: int = 1,
        noise_std: float = 0.1,
        key: jax.Array | None = None,
    ) -> list[float]:
        """Train the energy output parameters via Contrastive Divergence (CD-1).

        **Researcher summary:**
            CD-k training: positive phase uses data samples (h evolved from data),
            negative phase uses short MCMC chains from perturbed data (Langevin).
            Updates J_eff and b_eff only — LTCN dynamics parameters are frozen
            during this phase to keep the adaptation structure stable.

        **Detailed explanation for engineers:**
            Contrastive Divergence (Hinton 2002) approximates the gradient of the
            log-likelihood by comparing the model's energy at two points:

            1. **Positive phase**: Data points from the training set. These are
               configurations we *want* the model to assign low energy to.
               Energy is computed from the hidden state evolved from these inputs.

            2. **Negative phase**: "Fantasy" samples generated by running a short
               Markov chain (here: Gaussian perturbation = CD-1 approximation).
               These represent configurations the model currently assigns low energy
               to. We want to push their energy *up*.

            The parameter update rule:
                ΔJ_eff ∝ E_data[tanh(h_+) ⊗ tanh(h_+)] - E_fantasy[tanh(h_-) ⊗ tanh(h_-)]
                Δb_eff ∝ E_data[tanh(h_+)] - E_fantasy[tanh(h_-)]

            In plain English: make data configurations have lower energy, make
            fantasy configurations have higher energy.

            **Why CD instead of DSM here?**
            DSM requires differentiating through the energy gradient, which is
            tractable for the Ising model but becomes complex for the LTCN where
            energy is computed from an evolved hidden state. CD-1 is simpler and
            works well for the output-layer parameters J_eff, b_eff.

            **What's being trained:**
            Only J_eff and b_eff are updated — the dynamics parameters (W_in,
            W_rec, W_gate, b_dyn) remain fixed. This "two-stage" approach keeps
            the adaptation structure stable while fitting the energy function to data.

        Args:
            data: Training data of shape (n_samples, input_dim). Typically
                embeddings of correct/well-formed reasoning steps.
            n_epochs: Number of full passes over the training data.
            lr: Learning rate for gradient ascent/descent on CD loss.
            cd_steps: Number of MCMC steps for negative phase (1 = CD-1).
            noise_std: Standard deviation of Gaussian noise used to generate
                negative samples. Larger values explore more broadly.
            key: JAX PRNG key. If None, uses seed 42.

        Returns:
            List of per-epoch mean CD loss values for convergence monitoring.

        Spec: REQ-CORE-001, SCENARIO-CORE-001
        """
        if key is None:
            key = jrandom.PRNGKey(42)

        losses: list[float] = []
        n_samples = data.shape[0]

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            # --- Positive phase: compute hidden states from data ---
            # For each data point, run one LTCN step to get the hidden state,
            # then extract tanh(h) for the energy statistics.
            pos_h_list = []
            for i in range(n_samples):
                h_pos = self._ltcn_step(self._h0, data[i])
                pos_h_list.append(jnp.tanh(h_pos))
            # Stack into (n_samples, hidden_dim)
            pos_h = jnp.stack(pos_h_list, axis=0)

            # --- Negative phase: generate fantasy samples via Gaussian perturbation ---
            # CD-1 approximation: perturb data with Gaussian noise, then run LTCN
            key, subkey = jrandom.split(key)
            noise = jrandom.normal(subkey, shape=data.shape) * noise_std
            neg_data = data + noise

            neg_h_list = []
            for i in range(n_samples):
                h_neg = self._ltcn_step(self._h0, neg_data[i])
                neg_h_list.append(jnp.tanh(h_neg))
            neg_h = jnp.stack(neg_h_list, axis=0)

            # --- CD gradient for J_eff ---
            # Positive statistics: average outer product of tanh(h_pos) with itself
            # Shape: (n_samples, hidden_dim, hidden_dim)
            pos_outer = jax.vmap(lambda h: jnp.outer(h, h))(pos_h)
            neg_outer = jax.vmap(lambda h: jnp.outer(h, h))(neg_h)
            # CD gradient: decrease energy at data, increase at fantasy
            # Since E = -0.5 h^T J h, dE/dJ = -0.5 * outer(h, h)
            # CD update: J += lr * (E_data[outer] - E_fantasy[outer]) * 0.5
            dJ = jnp.mean(pos_outer - neg_outer, axis=0)
            self.J_eff = self.J_eff + lr * 0.5 * dJ
            # Maintain symmetry throughout training
            self.J_eff = (self.J_eff + self.J_eff.T) / 2.0

            # --- CD gradient for b_eff ---
            # Since E = -b^T tanh(h), dE/db = -tanh(h)
            # CD update: b += lr * (mean(pos_h) - mean(neg_h))
            db = jnp.mean(pos_h - neg_h, axis=0)
            self.b_eff = self.b_eff + lr * db

            # --- Compute loss for monitoring ---
            # CD loss = mean positive energy - mean negative energy.
            # For convergence: this should decrease (or stabilize near 0).
            pos_energies = jnp.array([
                self.energy(data[i]) for i in range(min(n_samples, 8))
            ])
            neg_energies = jnp.array([
                self.energy(neg_data[i]) for i in range(min(n_samples, 8))
            ])
            loss = float(jnp.mean(pos_energies) - jnp.mean(neg_energies))
            epoch_loss = loss
            losses.append(epoch_loss)

        return losses

    @property
    def input_dim(self) -> int:
        """Number of input dimensions for this model."""
        return self.config.input_dim

    @property
    def hidden_state(self) -> jax.Array:
        """Current hidden state h, shape (hidden_dim,). Read-only view."""
        return self._h


__all__ = ["LNNConstraintConfig", "LNNConstraintModel"]
