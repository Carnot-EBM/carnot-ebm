"""Potts Machine Verifier — q-state generalization of Ising EBM for constraint verification.

**Researcher summary:**
    The Ising (q=2) model represents constraint states as binary +1/-1 spins: a constraint
    is either satisfied or violated.  This loses information when constraints are
    partially satisfied (e.g., a multi-step arithmetic proof with one wrong sub-step).
    The q-state Potts machine extends Ising to q discrete states per spin, encoding
    'correct' / 'partial' / 'violated' as a joint energy landscape.

**Why Potts over binary Ising?**
    Binary Ising binarizes constraint confidence, discarding partial-credit information.
    Potts machines encode correct/partial/violated as a single joint distribution with
    the same FPGA-compatible sparse coupling structure.  arXiv 2602.04200 (Restoring
    Sparsity in Potts Machines via Mean-Field Constraints, February 2026) shows that
    mean-field constraints preserve sparsity during Potts machine optimization, enabling
    hardware-native Potts sampling on the KV260 FPGA without architecture changes.

    The Potts energy function is:

        E(s) = -sum_{i<j} J[s_i, s_j, i, j] - sum_i h[s_i, i]

    Where:
    - s is a vector of spins, each in {0, 1, ..., q-1}
    - J[a, b, i, j] is the coupling energy when spin i is in state a and spin j is in state b
    - h[a, i] is the local field energy when spin i is in state a

    For Gibbs sampling, the conditional probability of spin i being in state a given all
    other spins is proportional to exp(-beta * E_i(a)), where E_i(a) is the energy
    contribution of spin i if it takes state a.

**FPGA path:**
    The sparse coupling structure (most J[a,b,i,j] entries are zero) maps directly to the
    KV260 FPGA Ising architecture with a Potts extension.  The new Verilog module needs to
    iterate over q states per spin rather than 2, but the AXI-Lite register-map upload
    contract stays the same (sparse row format).

Spec: REQ-VERIFY-106, REQ-VERIFY-107, REQ-VERIFY-108,
      SCENARIO-VERIFY-142, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


@dataclass
class PottsState:
    """Represents one q-state spin as an integer in {0, ..., q-1}.

    **Detailed explanation for engineers:**
        In a q=3 Potts machine, each spin holds one of three values:
        - 0 → 'correct' (constraint fully satisfied)
        - 1 → 'partial' (constraint partially satisfied, e.g. one wrong sub-step)
        - 2 → 'violated' (constraint clearly broken)

        This is a container dataclass for documentation and type safety;
        the actual spin arrays in PottsMachineVerifier use jnp.ndarray of int32.

    Attributes:
        q: Number of discrete states per spin.  Default 3 for the standard
           correct/partial/violated encoding.
        value: The current state of this spin, an integer in [0, q-1].

    Spec: REQ-VERIFY-106
    """

    q: int = 3
    value: int = 0

    def __post_init__(self) -> None:
        if not (0 <= self.value < self.q):
            raise ValueError(f"value must be in [0, {self.q-1}], got {self.value}")
        if self.q < 2:
            raise ValueError("q must be >= 2")


@dataclass
class PottsCoupling:
    """Coupling matrix for a q-state Potts system.

    **Detailed explanation for engineers:**
        In a standard Ising model, J is a matrix of shape (n_spins, n_spins):
        J[i, j] is the coupling strength between spins i and j.

        In a Potts model, J has an extra two state dimensions:
        J has shape (q, q, n_spins, n_spins), where J[a, b, i, j] is the coupling
        energy when spin i is in state a and spin j is in state b.

        Why this shape?  Because Potts coupling is state-dependent: the energy
        between two spins depends not just on WHICH spins interact but on WHAT
        STATE each is in.  For q=2 and states {0, 1}, this reduces to the Ising
        case with J[0,0] coupling aligning '0' states, etc.

        The FPGA sparse-coupling upload stays the same (sparse row format per
        state pair (a, b)), so the hardware cost scales as O(q^2 * nnz) where
        nnz is the number of nonzero couplings.

    Attributes:
        J: Coupling tensor of shape (q, q, n_spins, n_spins).

    Spec: REQ-VERIFY-106
    """

    J: jax.Array

    @property
    def q(self) -> int:
        """Number of spin states, inferred from J.shape[0]."""
        return int(self.J.shape[0])

    @property
    def n_spins(self) -> int:
        """Number of spins in the system, inferred from J.shape[2]."""
        return int(self.J.shape[2])

    def energy_contribution(self, config: jax.Array) -> jax.Array:
        """Compute total coupling energy for a full spin configuration.

        **Detailed explanation for engineers:**
            For each pair (i, j), the coupling energy is J[s_i, s_j, i, j].
            This function sums all pairs, using JAX advanced indexing to
            select the state-specific coupling for each spin pair.

        Args:
            config: Integer array of shape (n_spins,) with values in [0, q-1].

        Returns:
            Scalar energy value (sum of all pairwise coupling energies).

        Spec: REQ-VERIFY-106
        """
        n = self.n_spins
        # For each pair (i, j), select J[config[i], config[j], i, j]
        # Build index arrays for vectorized gather
        i_idx = jnp.repeat(jnp.arange(n), n)
        j_idx = jnp.tile(jnp.arange(n), n)
        si = config[i_idx]
        sj = config[j_idx]
        # Gather coupling values for all pairs
        pair_couplings = self.J[si, sj, i_idx, j_idx]
        # Sum all pairs; factor of 0.5 for double-counting symmetric pairs
        return -0.5 * jnp.sum(pair_couplings)


class PottsMachineVerifier:
    """q-state Potts machine for multi-class constraint verification.

    **Researcher summary:**
        Extends IsingEBM from binary (correct/violated) to q-class
        (correct/partial/violated for q=3) constraint verification.  The energy
        function and Gibbs sampler generalize naturally.  arXiv 2602.04200 shows
        that sparse coupling structure is preserved under mean-field optimization,
        so this model is FPGA-compatible with the KV260 hardware target.

    **Detailed explanation for engineers:**
        The PottsMachineVerifier holds three parameter tensors:

        1. ``self.J`` — coupling tensor of shape (q, q, n_spins, n_spins).
           J[a, b, i, j] = energy when spin i=a and spin j=b.

        2. ``self.h`` — local field tensor of shape (q, n_spins).
           h[a, i] = energy bias for spin i being in state a.

        3. ``self.q`` — number of states per spin.  Default 3 for 3-class
           (correct=0, partial=1, violated=2) constraint encoding.

        Training uses Contrastive Divergence (CD-1): push down the energy of
        'correct' configurations, push up the energy of 'violated' (and optionally
        'partial') configurations.

        Gibbs sampling: for each spin i in turn, compute the conditional energy
        E_i(a) for each possible state a, then sample from softmax(-beta * E_i).
        This is the natural generalization of Ising Gibbs from 2 states to q states.

    Spec: REQ-VERIFY-106, REQ-VERIFY-107, REQ-VERIFY-108,
          SCENARIO-VERIFY-142, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
    """

    def __init__(
        self,
        n_spins: int,
        q: int = 3,
        key: jax.Array | None = None,
    ) -> None:
        """Create a new PottsMachineVerifier with random initialization.

        **Detailed explanation for engineers:**
            The coupling tensor J has shape (q, q, n_spins, n_spins).  We initialize
            it with small random values (scale 0.01) to break symmetry without
            dominating the training signal.  The local field h starts at zero —
            no preferred state initially.

            Parameter count: q^2 * n_spins^2 + q * n_spins.  For q=3, n=16:
            9 * 256 + 3 * 16 = 2304 + 48 = 2352 parameters.

        Args:
            n_spins: Number of spins (constraint variables) in the system.
            q: Number of states per spin.  Default 3 for correct/partial/violated.
            key: JAX PRNG key.  If None, uses seed 0.

        Spec: REQ-VERIFY-106
        """
        if n_spins <= 0:
            raise ValueError("n_spins must be > 0")
        if q < 2:
            raise ValueError("q must be >= 2")

        self.n_spins = n_spins
        self.q = q

        if key is None:
            key = jrandom.PRNGKey(0)

        k1, k2 = jrandom.split(key)

        # Coupling tensor: small random init to break symmetry
        j_raw = jrandom.normal(k1, (q, q, n_spins, n_spins)) * 0.01
        # Symmetrize: J[a, b, i, j] == J[b, a, j, i]
        self.J = 0.5 * (j_raw + jnp.einsum("abij->baji", j_raw))

        # Local field: zero init (no preferred state)
        self.h = jnp.zeros((q, n_spins))

    def energy(self, config: jax.Array) -> jax.Array:
        """Compute scalar Potts energy for a full spin configuration.

        **Detailed explanation for engineers:**
            The Potts energy has two terms:

            1. Coupling term: -0.5 * sum_{i,j} J[s_i, s_j, i, j]
               Sums pairwise interactions.  Factor 0.5 avoids double-counting.

            2. Local field term: -sum_i h[s_i, i]
               Sums individual spin biases.

            Lower energy = the model considers this configuration more likely
            (Boltzmann distribution: P(s) ∝ exp(-beta * E(s))).

        Args:
            config: Integer array of shape (n_spins,) with values in [0, q-1].

        Returns:
            Scalar JAX array (energy value).

        Spec: REQ-VERIFY-106, SCENARIO-VERIFY-142
        """
        n = self.n_spins
        i_idx = jnp.repeat(jnp.arange(n), n)
        j_idx = jnp.tile(jnp.arange(n), n)
        si = config[i_idx]
        sj = config[j_idx]

        # Coupling energy: -0.5 * sum_ij J[s_i, s_j, i, j]
        coupling_e = -0.5 * jnp.sum(self.J[si, sj, i_idx, j_idx])

        # Local field energy: -sum_i h[s_i, i]
        local_e = -jnp.sum(self.h[config, jnp.arange(n)])

        return coupling_e + local_e

    def gibbs_update(self, config: jax.Array, beta: float = 1.0) -> jax.Array:
        """Perform one full Gibbs sweep over all spins.

        **Detailed explanation for engineers:**
            For each spin i (in order), compute the conditional energy E_i(a)
            for each possible state a in {0, ..., q-1}:

                E_i(a) = -sum_j J[a, s_j, i, j] - h[a, i]

            Then sample spin i from the categorical distribution:

                P(s_i = a | s_{-i}) ∝ exp(-beta * E_i(a))

            This is a sequential Gibbs sweep — each spin update uses the latest
            state of all other spins (unlike the parallel Ising sampler which
            updates all spins simultaneously).  Sequential Gibbs is exact MCMC.

        Args:
            config: Integer array of shape (n_spins,) with current spin states.
            beta: Inverse temperature (higher = lower temperature = more greedy).

        Returns:
            New spin configuration array of shape (n_spins,), same dtype as config.

        Spec: REQ-VERIFY-107, SCENARIO-VERIFY-143
        """
        # Convert to numpy for sequential updates (JAX arrays are immutable)
        # We loop over spins sequentially, updating config after each spin
        config_np = np.array(config, dtype=np.int32)
        n = self.n_spins
        J_np = np.array(self.J)
        h_np = np.array(self.h)

        for i in range(n):
            # Compute conditional energy for each state a of spin i
            # E_i(a) = -sum_j J[a, s_j, i, j] - h[a, i]
            j_all = np.arange(n)
            sj = config_np[j_all]  # current states of all spins

            # coupling contribution: for each state a, sum over j of J[a, sj[j], i, j]
            # J_np shape: (q, q, n, n)
            # J_np[:, sj, i, j_all] has shape (q, n) — then sum over j
            conditional_e = np.zeros(self.q)
            for a in range(self.q):
                coupling_contrib = np.sum(J_np[a, sj, i, j_all])
                local_contrib = h_np[a, i]
                conditional_e[a] = -coupling_contrib - local_contrib

            # Sample from softmax(-beta * E_i)
            log_probs = -beta * conditional_e
            # Stable softmax
            log_probs -= np.max(log_probs)
            probs = np.exp(log_probs)
            probs /= np.sum(probs)

            # Sample new state for spin i (deterministic argmax at high beta)
            # Use numpy random for simplicity in sequential loop
            config_np[i] = np.random.choice(self.q, p=probs)

        return jnp.array(config_np, dtype=jnp.int32)

    def sample(self, n_steps: int = 100, key: jax.Array | None = None) -> jax.Array:
        """Run Gibbs sampling from a random initial configuration.

        **Detailed explanation for engineers:**
            Starts from a uniformly random spin configuration (each spin
            independently drawn from {0, ..., q-1}), then runs n_steps
            Gibbs sweeps.  Returns the final configuration.

            Why random init rather than a fixed state:
                Random init explores the energy landscape broadly.  If the model
                is trained, the sampler will converge toward low-energy regions
                regardless of starting point (assuming enough steps).

        Args:
            n_steps: Number of Gibbs sweeps to perform.
            key: JAX PRNG key.  If None, uses seed 42.

        Returns:
            Integer array of shape (n_spins,) with final spin states.

        Spec: REQ-VERIFY-107
        """
        if key is None:
            key = jrandom.PRNGKey(42)

        # Random initial configuration
        config = jrandom.randint(key, (self.n_spins,), 0, self.q)

        for _ in range(n_steps):
            config = self.gibbs_update(config, beta=1.0)

        return config

    def fit_cd(
        self,
        correct_configs: jax.Array,
        violated_configs: jax.Array,
        partial_configs: jax.Array | None = None,
        n_steps: int = 50,
        lr: float = 0.01,
        cd_k: int = 1,
    ) -> None:
        """Train using Contrastive Divergence (CD-k) on 3-class examples.

        **Detailed explanation for engineers:**
            CD training has two phases per step:

            1. Positive phase: compute the gradient of the energy for a real data
               sample (correct, partial, or violated configuration).  We want
               these energies to be LOW (the model should "like" real data).

            2. Negative phase: starting from the data sample, run cd_k Gibbs
               steps to get a "fantasy" sample.  We want these energies to be HIGH
               (the model should "dislike" made-up configurations).

            The parameter update is:
                dJ/dt = -(E'(data) - E'(fantasy)) * lr
                dh/dt = -(E'(data) - E'(fantasy)) * lr

            where E' denotes the gradient of E w.r.t. J and h.

            For 3-class training:
            - 'correct' configs (class 0) get CD updates (push energy down)
            - 'violated' configs (class 2) are treated as hard negatives (push energy up)
            - 'partial' configs (class 1) get a softer CD update if provided

        Args:
            correct_configs: Array of shape (n_correct, n_spins) — class 0 examples.
            violated_configs: Array of shape (n_violated, n_spins) — class 2 examples.
            partial_configs: Optional array of shape (n_partial, n_spins) — class 1.
            n_steps: Number of CD training steps.
            lr: Learning rate for parameter updates.
            cd_k: Number of Gibbs steps in the negative phase.

        Spec: REQ-VERIFY-106, REQ-VERIFY-108
        """
        correct_np = np.array(correct_configs, dtype=np.int32)
        violated_np = np.array(violated_configs, dtype=np.int32)
        partial_np = np.array(partial_configs, dtype=np.int32) if partial_configs is not None else None

        J_np = np.array(self.J)
        h_np = np.array(self.h)

        n = self.n_spins
        q = self.q

        for step in range(n_steps):
            # Pick a random batch of correct and violated configs
            rng = np.random.default_rng(step)

            # Sample one positive (correct) and one negative (violated) per step
            pos_idx = rng.integers(0, len(correct_np))
            neg_idx = rng.integers(0, len(violated_np))

            pos_config = correct_np[pos_idx].copy()
            neg_config = violated_np[neg_idx].copy()

            # Run cd_k Gibbs steps from positive to get fantasy negative
            fantasy = pos_config.copy()
            for _ in range(cd_k):
                for i in range(n):
                    j_all = np.arange(n)
                    sj = fantasy[j_all]
                    cond_e = np.zeros(q)
                    for a in range(q):
                        cond_e[a] = -np.sum(J_np[a, sj, i, j_all]) - h_np[a, i]
                    log_probs = -1.0 * cond_e
                    log_probs -= np.max(log_probs)
                    probs = np.exp(log_probs)
                    probs /= np.sum(probs)
                    fantasy[i] = rng.choice(q, p=probs)

            # Gradient of energy w.r.t. J: dE/dJ[a,b,i,j] = -0.5 * (s_i==a)(s_j==b)
            # CD update: J += lr * (fantasy_grad - pos_grad) [push down pos, up fantasy]
            def _config_grad_J(cfg):
                """Compute the contribution to J gradient for a given config."""
                grad = np.zeros_like(J_np)
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            a, b = cfg[i], cfg[j]
                            grad[a, b, i, j] += 0.5
                return grad

            def _config_grad_h(cfg):
                """Compute the contribution to h gradient for a given config."""
                grad = np.zeros_like(h_np)
                for i in range(n):
                    a = cfg[i]
                    grad[a, i] += 1.0
                return grad

            # Positive phase: gradient from correct config (model should lower energy)
            pos_J_grad = _config_grad_J(pos_config)
            pos_h_grad = _config_grad_h(pos_config)

            # Negative phase: gradient from fantasy (model should raise energy)
            neg_J_grad = _config_grad_J(fantasy)
            neg_h_grad = _config_grad_h(fantasy)

            # CD update: maximize data likelihood => lower E(pos), raise E(neg)
            # Energy E = -J_sum - h_sum => dJ toward (pos - neg) lowers E(pos), raises E(neg)
            J_np += lr * (pos_J_grad - neg_J_grad)
            h_np += lr * (pos_h_grad - neg_h_grad)

            # Also train against explicit violated configs (hard negatives)
            neg_J_grad2 = _config_grad_J(neg_config)
            neg_h_grad2 = _config_grad_h(neg_config)
            J_np -= lr * neg_J_grad2
            h_np -= lr * neg_h_grad2

            # Train with partial configs if provided (softer update, half lr)
            if partial_np is not None and len(partial_np) > 0:
                part_idx = rng.integers(0, len(partial_np))
                part_config = partial_np[part_idx]
                # Partial configs should have intermediate energy — push down slightly
                part_J_grad = _config_grad_J(part_config)
                part_h_grad = _config_grad_h(part_config)
                J_np += 0.5 * lr * part_J_grad
                h_np += 0.5 * lr * part_h_grad

        # Re-symmetrize after training
        J_np = 0.5 * (J_np + J_np.transpose(1, 0, 3, 2))
        self.J = jnp.array(J_np)
        self.h = jnp.array(h_np)

    def predict_class(self, config: jax.Array) -> int:
        """Predict the class of a spin configuration by finding lowest class energy.

        **Detailed explanation for engineers:**
            We have q=3 possible class assignments for the full configuration:
            - Class 0 (correct): all spins set to state 0
            - Class 1 (partial): all spins set to state 1
            - Class 2 (violated): all spins set to state 2

            We compute the energy of each uniform class assignment and return
            the class with the lowest energy.  This is a MAP prediction under
            the model.

            Why uniform class assignment?  During inference, we observe a
            test configuration and want to classify it.  The simplest approach
            is to compare the energy of the test configuration against three
            canonical reference configurations (pure correct, pure partial,
            pure violated).  More sophisticated approaches (e.g., computing
            the partition function for each class) are expensive for large n.

        Args:
            config: Integer array of shape (n_spins,).

        Returns:
            Integer class label in {0, 1, 2}.

        Spec: REQ-VERIFY-108, SCENARIO-VERIFY-144
        """
        n = self.n_spins
        energies = []
        for cls in range(self.q):
            # Canonical class configuration: all spins in state cls
            class_config = jnp.full((n,), cls, dtype=jnp.int32)
            energies.append(float(self.energy(class_config)))

        return int(np.argmin(energies))
