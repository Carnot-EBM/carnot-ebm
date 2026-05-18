import numpy as np
from typing import Callable

class DEPSGLDSampler:
    def __init__(self, kT: float = 1.0, dt: float = 0.01, n_steps: int = 1000, random_seed: int = 42):
        self.kT = kT
        self.dt = dt
        self.n_steps = n_steps
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def sample(self, grad_energy_fn: Callable[[np.ndarray], np.ndarray], init_x: np.ndarray, project_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        x = np.array(init_x, dtype=float)
        noise_scale = np.sqrt(2 * self.dt * self.kT)
        
        for _ in range(self.n_steps):
            grad = grad_energy_fn(x)
            noise = self.rng.normal(0, 1, size=x.shape)
            x = x - self.dt * grad + noise_scale * noise
            x = project_fn(x)
            
        return x
