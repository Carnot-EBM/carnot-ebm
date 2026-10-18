"""Z1 SDK-aligned DTM (Denoising Thermodynamic Model) stub.

**Researcher summary:**
    Wraps the THRML Ising/Gibbs simulator to expose the *continuous* DTM
    sampling signature expected by the Extropic Z1 SDK.  The existing
    ``DtmBackend`` in ``dtm.py`` already handles the discrete (boolean)
    ``SamplerBackend`` protocol.  This module adds ``sample_thermodynamic``,
    which returns float-valued continuous states in [0, 1] instead of booleans,
    matching the Z1's denoising trajectory contract.

**Why a separate stub rather than modifying DtmBackend:**
    The Z1 SDK expects a *different* call signature — it feeds a noisy
    continuous state and receives a denoised continuous state.  Bolting this
    onto the existing ``DtmBackend`` would conflate two distinct contracts:
    the discrete Ising/Gibbs protocol (``SamplerBackend``) and the continuous
    DTM protocol.  Keeping them separate lets callers import exactly what they
    need and makes the Z1 migration path a simple backend swap.

**No hardware execution:** this module is simulator-only.  All computations
    run on CPU via JAX.  No Extropic Z1/XTR-0/TSU hardware is invoked.

Spec: REQ-SAMPLE-066, SCENARIO-SAMPLE-094
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


@dataclass
class DtmStub:
    """Continuous DTM stub aligned to the Extropic Z1 SDK interface.

    **Researcher summary:**
        Simulates one denoising thermodynamic step in continuous [0, 1] space.
        The Z1 hardware runs physical Langevin dynamics to denoise a noisy
        spin-glass state; this stub replicates that trajectory on CPU using
        a fixed Euler–Maruyama integrator, giving the same *interface* even
        though the physics are simulated.

    **Why Langevin here instead of Gibbs:**
        Gibbs sampling is discrete — it flips spins one at a time.  The Z1
        uses a *continuous* thermal process where each spin is a real-valued
        variable drifting toward ±1.  Langevin dynamics (gradient descent +
        Gaussian noise) is the simplest continuous analogue.

    Attributes:
        seed: Random seed for reproducibility.
        backend_name: Fixed to ``"dtm-stub-z1"`` for Z1 alignment audits.

    Spec: REQ-SAMPLE-066
    """

    seed: int = 42
    _key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key = jrandom.PRNGKey(self.seed)

    @property
    def backend_name(self) -> str:
        """Return ``"dtm-stub-z1"`` — distinguishes this from the plain DTM backend.

        Spec: REQ-SAMPLE-066-2
        """
        return "dtm-stub-z1"

    def _next_key(self) -> jax.Array:
        """Advance and return a fresh JAX PRNG subkey."""
        self._key, subkey = jrandom.split(self._key)
        return subkey

    # ------------------------------------------------------------------
    # Continuous DTM interface — the Z1-aligned addition
    # ------------------------------------------------------------------

    def sample_thermodynamic(
        self,
        noisy_state: np.ndarray,
        beta: float,
        n_denoising_steps: int = 10,
    ) -> np.ndarray:
        """Denoise a continuous spin state using Langevin dynamics.

        **Researcher summary:**
            Accepts a noisy continuous state in [0, 1], applies
            ``n_denoising_steps`` of Euler–Maruyama integration at inverse
            temperature ``beta``, and returns the denoised continuous state
            (still in [0, 1]).  This is the signature the Z1 SDK expects for
            DTM-mode sampling.

        **Why the output is float, not bool:**
            The Z1 hardware outputs marginal probabilities (soft spin values)
            before a final hard-threshold read-out.  Preserving the continuous
            representation lets downstream code apply its own threshold or feed
            the soft values into a gradient computation.

        Args:
            noisy_state: Float32 array of shape ``(n_samples, n_spins)`` with
                values in [0, 1] representing a noisy continuous spin state.
            beta: Inverse temperature.  Higher values push spins toward the
                ground state faster but increase the risk of trapping in local
                minima (exactly as on real Z1 hardware).
            n_denoising_steps: Number of Euler–Maruyama steps.  Default 10 is
                enough for small toy problems; scale up for larger Ising graphs.

        Returns:
            Float32 ndarray of shape ``(n_samples, n_spins)`` with values
            clipped to [0, 1].  Each entry is a soft spin marginal.

        Spec: REQ-SAMPLE-066-1
        """
        noisy_state = np.asarray(noisy_state, dtype=np.float32)
        n_samples, n_spins = noisy_state.shape

        # Centre the [0, 1] state in [-1, +1] for energy computation,
        # then de-centre at the end.  This matches the Ising ±1 convention
        # used in biases/couplings throughout Carnot.
        x = jnp.asarray(2.0 * noisy_state - 1.0, dtype=jnp.float32)

        dt = 0.01  # step size — small enough for stability without annealing

        for _ in range(n_denoising_steps):
            # Langevin gradient: dE/dx = -x  (pure noise energy; no bias or
            # coupling here because dtm_stub is a *shape-only* stub that does
            # not carry problem-specific Ising parameters).  A downstream
            # caller that has biases/couplings should subclass or extend this.
            grad_e = -x  # pushes x toward ±1 (the Ising ground states)
            noise = jrandom.normal(self._next_key(), shape=x.shape)
            x = x - grad_e * dt + jnp.sqrt(2.0 * dt / beta) * noise
            x = jnp.clip(x, -1.0, 1.0)

        # Map back to [0, 1] to match the Z1 SDK's probability-of-+1 output.
        result = (x + 1.0) / 2.0
        return np.asarray(result, dtype=np.float32)

    # ------------------------------------------------------------------
    # Discrete SamplerBackend protocol — kept for compatibility with
    # existing Carnot pipelines that call sample() / minimize_energy().
    # ------------------------------------------------------------------

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw boolean Ising samples by thresholding the thermodynamic output.

        **Researcher summary:**
            Feeds a uniform random noisy state into ``sample_thermodynamic``
            and thresholds the continuous output at 0.5 to produce the boolean
            spins required by the ``SamplerBackend`` protocol.

        Spec: REQ-SAMPLE-066-3
        """
        beta = float(config.get("beta", 1.0))
        n_steps = int(config.get("steps", 10))
        n_spins = biases.shape[0]
        rng = np.random.default_rng(self.seed)
        noisy_state = rng.uniform(0.0, 1.0, size=(n_samples, n_spins)).astype(np.float32)
        continuous = self.sample_thermodynamic(noisy_state, beta=beta, n_denoising_steps=n_steps)
        return (continuous > 0.5).astype(bool)

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Boolean low-energy configurations via thermodynamic annealing.

        **Researcher summary:**
            Runs ``sample_thermodynamic`` at high beta (low temperature) for
            more steps to bias toward lower-energy states, then thresholds.

        Spec: REQ-SAMPLE-066-3
        """
        return self.sample(biases, couplings, n_samples, {"beta": beta, "steps": n_steps})
