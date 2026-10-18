import math
import random
import numpy as np

class MCHFSTFilter:
    def __init__(self, energy_fn, n_spins: int = 128):
        self.energy_fn = energy_fn
        self.n_spins = n_spins

    def accept_prob(self, energy_before: float, energy_after: float) -> float:
        return min(1.0, math.exp(-(energy_after - energy_before)))

    def filter_tokens(self, prompt: str, tokens: list[str], random_state=None, entry_idx: int = 0) -> list[str]:
        if random_state is None:
            random_state = random.Random(42)

        accepted_tokens = []
        
        for t, token in enumerate(tokens):
            rng = np.random.RandomState(seed=42 + entry_idx * 100 + t)
            spin_before = rng.randn(self.n_spins)
            spin_after = spin_before.copy()
            spin_after[t % self.n_spins] *= -1
            
            energy_before = self.energy_fn(spin_before)
            energy_after = self.energy_fn(spin_after)
            
            prob = self.accept_prob(energy_before, energy_after)
            
            if random_state.random() < prob:
                accepted_tokens.append(token)

        return accepted_tokens
