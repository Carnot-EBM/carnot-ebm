import math
import random

class MCHFSTFilter:
    def __init__(self, energy_fn):
        self.energy_fn = energy_fn

    def accept_prob(self, energy_before: float, energy_after: float) -> float:
        return min(1.0, math.exp(-(energy_after - energy_before)))

    def filter_tokens(self, prompt: str, tokens: list[str], random_state=None) -> list[str]:
        if random_state is None:
            random_state = random.Random(42)

        accepted_tokens = []
        current_text = prompt
        current_energy = self.energy_fn(current_text)

        for token in tokens:
            proposed_text = current_text + token
            proposed_energy = self.energy_fn(proposed_text)

            prob = self.accept_prob(current_energy, proposed_energy)
            if random_state.random() < prob:
                accepted_tokens.append(token)
                current_text = proposed_text
                current_energy = proposed_energy

        return accepted_tokens
