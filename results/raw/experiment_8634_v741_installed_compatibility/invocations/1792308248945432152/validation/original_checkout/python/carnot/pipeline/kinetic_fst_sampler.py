import math
import random
import numpy as np
from typing import Callable, List, Optional

from carnot.samplers.kinetic_langevin import KineticLangevinSampler

class KineticLangevinFSTSampler:
    """FST Sampler that uses Kinetic Langevin (BAOAB) to generate spin states."""
    
    def __init__(
        self,
        energy_fn: Callable[[np.ndarray], float],
        grad_energy_fn: Callable[[np.ndarray], np.ndarray],
        n_spins: int = 16,
        gamma: float = 1.0,
        kT: float = 1.0,
        dt: float = 0.01,
        seed: int = 42
    ):
        self.energy_fn = energy_fn
        self.grad_energy_fn = grad_energy_fn
        self.n_spins = n_spins
        self.gamma = gamma
        self.kT = kT
        self.dt = dt
        self.seed = seed

    def filter_tokens(
        self, prompt: str, tokens: List[str], random_state=None, entry_idx: int = 0
    ) -> List[str]:
        if random_state is None:
            random_state = random.Random(self.seed)

        accepted_tokens = []
        
        for t, token in enumerate(tokens):
            rng = np.random.RandomState(seed=self.seed + entry_idx * 100 + t)
            init_x = rng.randn(self.n_spins)
            
            sampler_before = KineticLangevinSampler(
                gamma=self.gamma, kT=self.kT, dt=self.dt, n_steps=10, 
                random_seed=self.seed + entry_idx * 100 + t
            )
            sampler_after = KineticLangevinSampler(
                gamma=self.gamma, kT=self.kT, dt=self.dt, n_steps=11, 
                random_seed=self.seed + entry_idx * 100 + t
            )
            
            spin_before = sampler_before.sample(
                grad_energy_fn=self.grad_energy_fn, init_x=init_x
            )
            spin_after = sampler_after.sample(
                grad_energy_fn=self.grad_energy_fn, init_x=init_x
            )
            
            energy_before = self.energy_fn(spin_before)
            energy_after = self.energy_fn(spin_after)
            
            accept_prob = min(1.0, math.exp(-(energy_after - energy_before)))
            
            if random_state.random() < accept_prob:
                accepted_tokens.append(token)

        return accepted_tokens
