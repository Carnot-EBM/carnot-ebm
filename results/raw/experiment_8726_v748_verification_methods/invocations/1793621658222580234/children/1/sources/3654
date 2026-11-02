"""Continuous latent trace generator for EBFT training.

Generates reasoning trajectories through the latent space of a ContinuousEBM
using Langevin dynamics. These trajectories are the "latent reasoning traces"
that EBFT's feature matching objective operates on.

WHY latent traces instead of token-level supervision?
    Token-level loss forces the model to reproduce exact token sequences, which
    bakes in the teacher's verbosity and phrasing.  Latent-trace matching instead
    asks: "do the model's internal states traverse a similar region of latent space
    as the expert?"  This is softer — the model can find its own path to the
    same answer — while still providing a learning signal grounded in the energy
    landscape rather than surface form.

HOW the energy function grounds the traces:
    The ContinuousEBM assigns a scalar energy E(x) to each latent state x.
    Langevin dynamics produces trajectories that mix between energy basins,
    so a trace is a sequence of latent states that "reasons" from a noisy
    initial state toward a lower-energy region.  An expert trace comes from a
    well-trained or reference EBM; a rollout trace comes from the current
    (possibly undertrained) EBM.  EBFT matches their feature statistics.

Spec: REQ-TRAIN-007, REQ-KONA-002
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LatentTrace:
    """A single continuous latent reasoning trace.

    A trace is a sequence of latent states produced by iterative energy
    minimisation.  Each state x_t lives in (-1, 1)^d due to tanh squashing
    (the same domain used by ContinuousEBM).

    Attributes:
        states: Array of shape (n_steps, latent_dim) — the trajectory.
        energies: Scalar energy E(x_t) at each step, shape (n_steps,).
        seed: Random seed used for reproducibility.
    """

    states: np.ndarray
    energies: np.ndarray
    seed: int

    def features(self) -> np.ndarray:
        """Extract feature statistics from the trace for EBFT matching.

        Returns a concatenation of:
          - Mean latent state across time (latent_dim,): captures the
            "average position" of the trace in latent space.
          - Std latent state across time (latent_dim,): captures how much
            the trace explores vs stays put.
          - Mean energy (1,): where in the energy landscape the trace lives.

        These three statistics are the minimal sufficient statistics for
        EBFT feature matching — they characterise WHAT region the trace
        visits and HOW MUCH it moves around.
        """
        mean_state = np.mean(self.states, axis=0)
        std_state = np.std(self.states, axis=0)
        mean_energy = np.array([np.mean(self.energies)])
        return np.concatenate([mean_state, std_state, mean_energy])


@dataclass
class LatentGenerator:
    """Generates continuous latent traces from a ContinuousEBM via Langevin dynamics.

    Acts as the "rollout engine" for EBFT training: given an energy landscape,
    it produces trajectories that the EBFT objective can compare against expert
    traces.  The generator is stateless — each call to ``generate`` uses only
    the EBM coupling/bias and the provided seed.

    Attributes:
        n_steps: Number of Langevin steps per trace (trajectory length).
        lr: Langevin step size.  Smaller → more stable, slower convergence.
        noise_scale: Base noise magnitude (annealed via cosine schedule).
        record_interval: Record a latent state every this many steps.
            Defaults to 1 (record every step), giving traces of length n_steps.
    """

    n_steps: int = 200
    lr: float = 0.01
    noise_scale: float = 0.1
    record_interval: int = 1

    def generate(
        self,
        coupling: np.ndarray,
        bias: np.ndarray,
        seed: int = 0,
    ) -> LatentTrace:
        """Generate a single Langevin trajectory through the EBM energy landscape.

        Steps:
            1. Sample an initial state from N(0, I) — broad coverage.
            2. Run n_steps of annealed Langevin dynamics, recording states at
               each record_interval.
            3. Return a LatentTrace with the recorded states and their energies.

        WHY cosine noise annealing?
            Starting with high noise allows the chain to explore across energy
            barriers early (broad exploration); cooling toward zero noise late in
            the trajectory biases it toward a local energy minimum (exploitation).
            This mirrors how simulated annealing produces valid solutions rather
            than random walks.

        Args:
            coupling: Symmetric coupling matrix J of shape (d, d).
            bias: Bias vector h of shape (d,).
            seed: Random seed for reproducibility.

        Returns:
            LatentTrace with recorded states and their energies.
        """
        rng = np.random.default_rng(seed)
        J = np.asarray(coupling, dtype=np.float64)
        h = np.asarray(bias, dtype=np.float64)
        d = h.shape[0]

        x = rng.standard_normal(d)
        noise_std = self.noise_scale * np.sqrt(2.0 * self.lr)

        recorded_states: list[np.ndarray] = []
        recorded_energies: list[float] = []

        for t in range(self.n_steps):
            # Gradient of E(x) = -0.5 x^T J x - h^T x → dE/dx = -J x - h
            grad = -J @ x - h
            # Cosine-annealed noise: high early (exploration), low late (exploitation)
            temp_factor = 0.5 * (1.0 + np.cos(np.pi * t / max(self.n_steps - 1, 1)))
            noise = noise_std * temp_factor * rng.standard_normal(d)
            x = np.tanh(x - self.lr * grad + noise)

            if t % self.record_interval == 0:
                energy = float(-0.5 * x @ J @ x - h @ x)
                recorded_states.append(x.copy())
                recorded_energies.append(energy)

        states = np.stack(recorded_states, axis=0)  # (T, d)
        energies = np.array(recorded_energies)  # (T,)
        return LatentTrace(states=states, energies=energies, seed=seed)

    def generate_batch(
        self,
        coupling: np.ndarray,
        bias: np.ndarray,
        n_traces: int,
        base_seed: int = 0,
    ) -> list[LatentTrace]:
        """Generate a batch of independent Langevin traces.

        Each trace uses a distinct seed (base_seed + i) so traces are
        independent samples from the energy landscape.

        Args:
            coupling: Coupling matrix J, shape (d, d).
            bias: Bias vector h, shape (d,).
            n_traces: Number of independent traces to generate.
            base_seed: Seed for the first trace; subsequent seeds are base_seed+i.

        Returns:
            List of n_traces LatentTrace objects.
        """
        return [
            self.generate(coupling, bias, seed=base_seed + i)
            for i in range(n_traces)
        ]

    def feature_matrix(
        self,
        traces: list[LatentTrace],
    ) -> np.ndarray:
        """Stack feature vectors from a batch of traces into a matrix.

        Each row is the feature vector of one trace (see LatentTrace.features).
        This is the input to the EBFT feature matching objective.

        Args:
            traces: List of LatentTrace objects.

        Returns:
            Array of shape (n_traces, feature_dim) where feature_dim =
            2 * latent_dim + 1.
        """
        return np.stack([t.features() for t in traces], axis=0)
