"""Dikin-Langevin sampler for polytope-constrained sampling.

Spec: REQ-SAMPLE-2429, SCENARIO-SAMPLE-2429.
"""

from __future__ import annotations

import numpy as np


class DikinLangevinSampler:
    """CPU Dikin-Langevin sampler.

    Step: q += dt*(-grad_V + sqrt(2/dt)*M(q)^{-1/2}*noise)
    where M(q) = diag(1/delta(q_i)^2) and delta(q_i) = min(q_i - (-1), 1 - q_i).
    """

    def __init__(self, kT: float = 1.0, dt: float = 0.01, n_steps: int = 1000, random_seed: int = 42) -> None:
        self.kT = kT
        self.dt = dt
        self.n_steps = n_steps
        self.random_seed = random_seed

    def sample(self, grad_energy_fn, init_q, project_fn) -> np.ndarray:
        q = np.array(init_q, dtype=float, copy=True)
        rng = np.random.default_rng(self.random_seed)

        for _ in range(self.n_steps):
            # delta(q_i) = min(q_i - (-1), 1 - q_i)
            delta = np.minimum(q + 1.0, 1.0 - q)
            delta = np.clip(delta, 1e-8, None)

            grad_V = grad_energy_fn(q)
            noise = rng.normal(size=q.shape)

            # Step: q += dt*(-grad_V + sqrt(2/dt)*M(q)^{-1/2}*noise)
            # M(q)^{-1/2} = delta
            # We add kT for temperature scaling if needed, though prompt implies kT=1 implicitly.
            # We'll use np.sqrt(2 * self.kT / self.dt) to match the prompt's noise scaling.
            step = self.dt * (-grad_V + np.sqrt(2.0 * self.kT / self.dt) * delta * noise)
            q = q + step

            # Project to box [-1, 1]
            q = project_fn(q)

        return q
