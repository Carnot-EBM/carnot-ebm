from __future__ import annotations

import numpy as np
from typing import Any, Callable

class KineticLangevinSampler:
    """Underdamped (second-order) Langevin sampler using BAOAB splitting.
    
    BAOAB step:
      B: v = v - (dt/2) * grad U(x)
      A: x = x + (dt/2) * v
      O: v = c1 * v + c2 * R
      A: x = x + (dt/2) * v
      B: v = v - (dt/2) * grad U(x)
      
    Used for sampling on constrained distributions, providing faster mixing.
    """

    def __init__(
        self,
        gamma: float = 1.0,
        kT: float = 1.0,
        dt: float = 0.01,
        n_steps: int = 1000,
        random_seed: int = 42,
    ) -> None:
        self.gamma = float(gamma)
        self.kT = float(kT)
        self.dt = float(dt)
        self.n_steps = int(n_steps)
        self.random_seed = int(random_seed)

    def sample(
        self,
        grad_energy_fn: Callable[[np.ndarray], np.ndarray],
        init_x: np.ndarray,
        project_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """Run BAOAB sampling.
        
        Args:
            grad_energy_fn: Function computing the gradient of the energy.
            init_x: Initial position array (can be batched).
            project_fn: Optional function to project state onto constraints.
        """
        rng = np.random.default_rng(self.random_seed)
        x = np.asarray(init_x, dtype=float).copy()
        
        if project_fn is not None:
            x = project_fn(x)
            
        v = rng.normal(0.0, np.sqrt(self.kT), size=x.shape)
        
        c1 = np.exp(-self.gamma * self.dt)
        c2 = np.sqrt(self.kT * (1.0 - c1**2))
        dt_half = self.dt / 2.0
        
        for _ in range(self.n_steps):
            # B
            grad = grad_energy_fn(x)
            v = v - dt_half * grad
            
            # A
            x = x + dt_half * v
            if project_fn is not None:
                x = project_fn(x)
                
            # O
            noise = rng.normal(0.0, 1.0, size=x.shape)
            v = c1 * v + c2 * noise
            
            # A
            x = x + dt_half * v
            if project_fn is not None:
                x = project_fn(x)
                
            # B
            grad = grad_energy_fn(x)
            v = v - dt_half * grad
            
        return x
