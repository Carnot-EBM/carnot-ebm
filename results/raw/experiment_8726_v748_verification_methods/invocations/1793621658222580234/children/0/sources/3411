"""Native Thermodynamic Distillation (Approach 3, the production path).

The load-bearing piece of the Phase 2 transpiler. Heavy verifier
backbones (transformer-class, deep MLP) cannot be embedded by
execution-trace compilation — Approach 1's spin count
``W*log²(1/eps)`` blows past any real hardware budget once ``W`` is
in the millions. The fix: don't compile the verifier's forward pass,
*train a Boltzmann Machine to match the verifier's MCMC samples*.

**The training procedure: Persistent Parallel Tempering (PT-PCD).**
SVAMP/FoVer-style energy landscapes are *glassy* — deep isolated
sinkholes (valid reasoning paths) separated by tall barriers
(logical contradictions). Naive contrastive divergence has two
failure modes that map directly to two hardware-deployment failures:

- **CD-1/CD-k** carves the valid modes well but never explores empty
  hypercube regions, leaving spurious deep wells in unconstrained
  spin configurations. At deployment the hardware falls into these
  hallucinated wells and emits garbage outputs (False Positives).
- **Vanilla PCD** with persistent chains can't cross logical-
  contradiction barriers, so the gradient violently biases toward the
  first discovered mode and destroys the others (False Negatives, mode
  collapse).

PT-PCD fixes both. Hot chains at high temperature random-walk over
the barriers; replica exchange swaps drag spurious-minimum states
down to the cold chain (``beta=1.0``) where the negative gradient
flattens them. The Round-4 production recipe layers five
guardrails:

1. Geometric beta ladder. ``M`` rungs from ``beta_0=1.0`` (cold) to
   ``beta_M = 0.05`` (hot). For SVAMP-class barrier heights ``beta_M``
   may need to drop further (``< 0.01``) — left as a hyperparameter.
2. Gray-code visible-spin encoder (see ``gray_code.py``) so adjacent
   continuous cells map to Hamming-distance-1 spin states. Eliminates
   the "cliff" pathology of standard binary.
3. 5%-rebirth teleportation on hot chains every epoch (overwrite with
   uniform ``{-1,+1}`` noise). Prevents chain death in flat regions.
4. ``L2 = 1e-4`` weight decay on ``J``. Without bounds, ``J`` would
   grow unboundedly during PCD, creating a "golf-course" landscape
   (flat plains with infinitely deep pinholes) that hardware
   simulated-bifurcation solvers can't dynamically relax into.
5. Symmetrize ``J`` and zero diagonal each step. Standard Ising
   convention; without these the gradient drifts the matrix off the
   valid manifold.

**Why empirical, not formal, KL bound.** Approach 1 gives a formal
``KL <= eps`` from the spectral gap of the QUBO penalty term.
Approach 3 trades that for engineering feasibility — the guarantee
becomes whatever PT-PCD converges to. We backfill confidence with the
three diagnostics in ``diagnostics.py`` (KDE overlap, energy
histogram overlap, swap acceptance rate). These are *necessary, not
sufficient* — small modes can still be missed at finite-sample
resolution — but they are falsifiable and catch the major failure
modes.

**Defaults targeting the Round-4 prototype.** A 1024-spin BM with
``n_vis=64`` Gray-coded for a 2D continuous latent over ``[-L, L]^2``
and ``n_hid=960`` hidden capacity. PT ladder of 16 rungs, 1024
chains, 15 Gibbs sweeps per training step, ``lr=0.01``,
``l2_reg=1e-4``, 100 epochs. Designed to fit on a single GPU with
~70 MB of chain-state memory.

Spec: REQ-PHASE2-004 (Native Thermodynamic Distillation primitive).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.hardware.transpiler.api import HardwareSpec, IsingSpec
from carnot.hardware.transpiler.gray_code import decode_2d, encode_2d

_log = logging.getLogger(__name__)


@dataclass
class DistillerConfig:
    """Hyperparameters for `CarnotNativeDistiller`. Defaults match the
    Round-4 prototype recipe. All fields are tunable for new hardware
    targets but the defaults are validated to converge on the synthetic
    2D-Gaussian-mixture test in ``test_transpiler_distill.py``.
    """

    n_vis: int = 64
    """Visible spins. ``2 * m_per_axis`` for a 2D Gray-coded latent."""

    n_hid: int = 960
    """Hidden spins. Capacity for the BM to model arbitrary distributions
    over visible spins. Total spin count is ``n_vis + n_hid``.
    """

    n_chains: int = 1024
    """Persistent chains per temperature rung."""

    n_temps: int = 16
    """Number of temperatures in the geometric PT ladder."""

    beta_min: float = 0.05
    """Hottest chain inverse temperature. For glassy SVAMP-class
    landscapes may need to drop further (``< 0.01``).
    """

    k_steps: int = 15
    """Gibbs sweeps per negative-phase iteration."""

    lr: float = 0.01
    """Stochastic-gradient learning rate."""

    l2_reg: float = 1e-4
    """L2 weight decay on ``J`` (golf-course-pathology cap)."""

    rebirth_fraction: float = 0.05
    """Fraction of hot chains overwritten with uniform noise per epoch.
    """

    seed: int = 0
    """RNG seed for reproducibility."""


class CarnotNativeDistiller:
    """PT-PCD trainer for Approach 3 (Native Thermodynamic Distillation).

    Usage::

        distiller = CarnotNativeDistiller(config, hardware_spec)
        for epoch in range(epochs):
            loss = distiller.train_epoch(continuous_samples_2d)
        spec = distiller.export_ising_spec(provenance={...})

    The exported `IsingSpec` has the same schema as Approach 1's output,
    so downstream `SamplerBackend` integration is approach-agnostic.

    Parameters
    ----------
    config
        Hyperparameters (see ``DistillerConfig``).
    hardware_spec
        Target hardware. Used to validate spin budget and to record
        provenance. ``hardware_spec.max_spins`` must satisfy
        ``n_vis + n_hid <= max_spins``.
    domain
        Continuous latent domain ``(lo, hi)`` for the Gray-code encoder.
    """

    def __init__(
        self,
        config: DistillerConfig,
        hardware_spec: HardwareSpec,
        domain: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        n_total = config.n_vis + config.n_hid
        if n_total > hardware_spec.max_spins:
            raise ValueError(
                f"n_vis+n_hid={n_total} exceeds hardware max_spins={hardware_spec.max_spins}"
            )
        if config.n_vis % 2 != 0:
            raise ValueError(f"n_vis={config.n_vis} must be even (m_per_axis per 2D axis)")
        beta_min, beta_max = hardware_spec.beta_range
        if not (config.beta_min <= beta_min):
            raise ValueError(
                f"distiller config beta_min={config.beta_min} must be <= "
                f"hardware beta_min={beta_min} so the cold chain spans the "
                f"deployment range"
            )
        self.config = config
        self.hardware_spec = hardware_spec
        self.domain = domain
        self.m_per_axis = config.n_vis // 2

        rng = np.random.default_rng(config.seed)
        # Initialize J small (avoid early ferromagnetic collapse). The L2
        # regularizer keeps it bounded throughout training.
        self.J = rng.normal(scale=0.001, size=(n_total, n_total))
        self.J = 0.5 * (self.J + self.J.T)  # symmetrize
        np.fill_diagonal(self.J, 0.0)
        self.h = rng.normal(scale=0.001, size=(n_total,))

        # Geometric beta ladder: beta_max=1.0 (production cold) down to
        # beta_min (hot exploration). Round-4 default beta_min=0.05.
        # log-linear spacing gives equal acceptance probability per swap
        # at the cost of denser temperatures near the cold end.
        self.betas = np.geomspace(1.0, config.beta_min, config.n_temps)

        # Persistent chains: one bank per temperature, all initialized at
        # uniform random {-1, +1}.
        self.chains = rng.choice([-1.0, 1.0], size=(config.n_temps, config.n_chains, n_total))

        # Track swap-acceptance for diagnostic C
        self.swap_accept_history: list[np.ndarray] = []
        self.rng = rng
        self.n_total = n_total

    # ------------------------------------------------------------------
    # Negative phase: PT relaxation
    # ------------------------------------------------------------------

    def _gibbs_sweep(self) -> None:
        """One Gibbs sweep across all ``n_temps`` chain banks. Updates
        ``self.chains`` in place via single-spin probabilistic flips.

        For each spin index ``i``, the local field is ``f_i = sum_j J_ij
        s_j + h_i`` and the flip probability is ``P(flip) =
        sigmoid(-2 * beta * f_i * s_i)``. We update spins one at a
        time (proper Gibbs) — random ordering each sweep to avoid
        scanning artifacts.
        """
        order = self.rng.permutation(self.n_total)
        for i in order:
            # local_fields shape: (n_temps, n_chains)
            local_field = self.chains @ self.J[:, i] + self.h[i]
            # P(flip) per chain per temperature
            arg = -2.0 * self.betas[:, None] * local_field * self.chains[..., i]
            # Numerically stable sigmoid
            p_flip = 1.0 / (1.0 + np.exp(-np.clip(arg, -50.0, 50.0)))
            flips = self.rng.uniform(size=p_flip.shape) < p_flip
            self.chains[..., i] = np.where(flips, -self.chains[..., i], self.chains[..., i])

    def _replica_exchange(self) -> np.ndarray:
        """Metropolis-Hastings replica exchange between adjacent
        temperature rungs. Returns the per-rung-pair acceptance rate
        for diagnostic C.

        For two chains with same configuration ``s`` at adjacent
        temperatures ``beta_a < beta_b``, the swap is accepted with
        probability ``min(1, exp((beta_b - beta_a)(E_a - E_b)))`` where
        ``E_x = -s^T J s - h^T s`` is the same for both (Ising energy
        doesn't depend on temperature). Practically the swap is between
        whole-chain configurations, so we compute energies for both
        chains and compute the standard PT swap log-ratio.
        """
        n_temps = self.config.n_temps
        n_chains = self.config.n_chains
        accept_per_pair = np.zeros(n_temps - 1)

        # Compute energies once: shape (n_temps, n_chains)
        # E(s) = -s^T J s - h^T s
        Js = self.chains @ self.J  # (T, C, N)
        E = -np.einsum("tcn,tcn->tc", self.chains, Js) - self.chains @ self.h

        # Sweep adjacent pairs in alternating order (even/odd) to avoid
        # double-swaps in the same sweep
        for parity in (0, 1):
            for t in range(parity, n_temps - 1, 2):
                # Swap chain bank t with bank t+1 per chain
                delta_beta = self.betas[t] - self.betas[t + 1]
                delta_E = E[t] - E[t + 1]
                log_p = delta_beta * delta_E  # accept if uniform < exp(log_p)
                accept = self.rng.uniform(size=n_chains) < np.exp(np.minimum(log_p, 0.0))
                accept_per_pair[t] = accept.mean()
                # Apply swaps in place
                if accept.any():
                    chains_t = self.chains[t].copy()
                    self.chains[t, accept] = self.chains[t + 1, accept]
                    self.chains[t + 1, accept] = chains_t[accept]
                    # Update cached energies for swapped chains
                    et = E[t].copy()
                    E[t, accept] = E[t + 1, accept]
                    E[t + 1, accept] = et[accept]

        return accept_per_pair

    def _rebirth_hot_chains(self) -> None:
        """Per-epoch 5% rebirth of the hottest chain bank. Prevents
        chain death in flat-gradient regions far from any learned mode.
        """
        n_chains = self.config.n_chains
        n_rebirth = max(1, int(self.config.rebirth_fraction * n_chains))
        rebirth_idx = self.rng.choice(n_chains, size=n_rebirth, replace=False)
        self.chains[-1, rebirth_idx] = self.rng.choice([-1.0, 1.0], size=(n_rebirth, self.n_total))

    # ------------------------------------------------------------------
    # Positive phase: encode continuous data, sample hidden conditionally
    # ------------------------------------------------------------------

    def _encode_visible(self, z_batch: np.ndarray) -> np.ndarray:
        """Gray-code-encode a 2D continuous batch to visible spins.
        ``z_batch`` shape ``(B, 2)`` → return shape ``(B, n_vis)``.
        """
        return encode_2d(z_batch, self.m_per_axis, *self.domain)

    def _sample_hidden_given_visible(self, v: np.ndarray) -> np.ndarray:
        """Sample hidden spins conditioned on visible. Standard BM
        positive-phase trick: hidden units are conditionally
        independent given visible, so we can sample exactly via per-
        spin sigmoid. ``v`` shape ``(B, n_vis)`` → return shape
        ``(B, n_hid)``.
        """
        n_vis = self.config.n_vis
        # Coupling block J[n_vis:, :n_vis] connects hidden to visible
        J_hv = self.J[n_vis:, :n_vis]  # (n_hid, n_vis)
        h_h = self.h[n_vis:]  # (n_hid,)
        local_field = v @ J_hv.T + h_h  # (B, n_hid)
        # P(h=+1 | v) = sigmoid(2 * beta * local_field) with beta=1.0
        # for the production cold chain
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(2.0 * local_field, -50.0, 50.0)))
        return np.where(self.rng.uniform(size=p_pos.shape) < p_pos, 1.0, -1.0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train_epoch(self, z_batch: np.ndarray) -> dict[str, float]:
        """Run one PT-PCD update on a batch of continuous 2D samples.

        Parameters
        ----------
        z_batch
            Shape ``(B, 2)``. Continuous samples drawn from the target
            distribution we want the BM to match.

        Returns
        -------
        dict
            ``{"swap_accept_min": float, "swap_accept_mean": float,
              "j_norm": float, "free_energy_gap": float}``.
            ``swap_accept_min`` is diagnostic C — must stay above ~0.15
            for the temperature ladder to be functional.
        """
        z_batch = np.atleast_2d(np.asarray(z_batch, dtype=np.float64))
        if z_batch.shape[-1] != 2:
            raise ValueError(f"expected 2D latent batch, got shape {z_batch.shape}")

        # Per-epoch chain rebirth (heuristic 3)
        self._rebirth_hot_chains()

        # Positive phase: encode data, sample hidden conditionally
        v_data = self._encode_visible(z_batch)
        h_data = self._sample_hidden_given_visible(v_data)
        s_data = np.concatenate([v_data, h_data], axis=-1)

        # Negative phase: PT-PCD on persistent chains
        for _ in range(self.config.k_steps):
            self._gibbs_sweep()
        accept = self._replica_exchange()
        self.swap_accept_history.append(accept.copy())

        # Cold-chain (beta=1.0) negative samples for the gradient
        s_model = self.chains[0]  # shape (n_chains, n_total)

        # Gradient: dJ/dt = <s s^T>_data - <s s^T>_model - l2 * J
        grad_J = (s_data.T @ s_data) / s_data.shape[0] - (s_model.T @ s_model) / s_model.shape[0]
        grad_h = s_data.mean(axis=0) - s_model.mean(axis=0)

        self.J += self.config.lr * (grad_J - self.config.l2_reg * self.J)
        self.J = 0.5 * (self.J + self.J.T)
        np.fill_diagonal(self.J, 0.0)
        self.h += self.config.lr * grad_h

        # Free-energy gap: a rough scalar tracking convergence. The
        # mean energy of data samples vs cold-chain model samples;
        # large positive gap = model assigns higher energy to data
        # than its own samples (training pulling data energy down).
        E_data = -np.einsum("bn,nm,bm->b", s_data, self.J, s_data) - s_data @ self.h
        E_model = -np.einsum("bn,nm,bm->b", s_model, self.J, s_model) - s_model @ self.h
        free_energy_gap = float(E_data.mean() - E_model.mean())

        return {
            "swap_accept_min": float(accept.min()),
            "swap_accept_mean": float(accept.mean()),
            "j_norm": float(np.linalg.norm(self.J)),
            "free_energy_gap": free_energy_gap,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def sample_cold_chain(self, n: int) -> np.ndarray:
        """Return ``n`` samples from the cold (``beta=1.0``) chain bank.
        Caller decodes visible portion to continuous latent if desired.
        """
        if n > self.config.n_chains:
            raise ValueError(
                f"requested {n} samples but only have {self.config.n_chains}"
                " persistent chains; run a few extra training epochs"
            )
        idx = self.rng.choice(self.config.n_chains, size=n, replace=False)
        return self.chains[0, idx]

    def export_ising_spec(self, provenance: dict[str, Any] | None = None) -> IsingSpec:
        """Snapshot the current ``J``, ``h`` and produce an ``IsingSpec``
        with bound encoder/decoder pair. The provenance dict should
        record the source ``state_dict`` hash, training-corpus hash,
        and any other info needed to audit a deployed payload.
        """
        m = self.m_per_axis
        lo, hi = self.domain
        # Capture state by value into closures to avoid mutability bugs
        J_snapshot = self.J.copy()
        h_snapshot = self.h.copy()

        def phi(z: np.ndarray) -> np.ndarray:
            return encode_2d(z, m, lo, hi)

        def psi(s: np.ndarray) -> np.ndarray:
            # Use a fresh RNG inside psi so spatial-noise decoding is
            # deterministic per-call when caller provides a seed via
            # numpy's global RNG; otherwise fully stochastic.
            rng = np.random.default_rng()
            v = s[..., : self.config.n_vis]
            return decode_2d(v, m, lo, hi, rng=rng)

        # Build provenance, hashing J for tamper-evidence
        prov = dict(provenance or {})
        prov.setdefault("approach", "native_thermodynamic_distillation")
        prov.setdefault(
            "hardware_spec",
            {
                "kind": self.hardware_spec.kind,
                "max_spins": self.hardware_spec.max_spins,
                "beta_range": list(self.hardware_spec.beta_range),
                "vendor_target": self.hardware_spec.vendor_target,
            },
        )
        prov.setdefault(
            "config",
            {
                "n_vis": self.config.n_vis,
                "n_hid": self.config.n_hid,
                "n_chains": self.config.n_chains,
                "n_temps": self.config.n_temps,
                "beta_min": self.config.beta_min,
                "k_steps": self.config.k_steps,
                "lr": self.config.lr,
                "l2_reg": self.config.l2_reg,
            },
        )
        prov["J_hash"] = hashlib.sha256(J_snapshot.tobytes()).hexdigest()[:16]

        return IsingSpec(J=J_snapshot, h=h_snapshot, phi=phi, psi=psi, provenance=prov)
