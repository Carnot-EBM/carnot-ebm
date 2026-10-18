import numpy as np
from typing import List

class TTCController:
    """
    Dynamic budget controller for generative decoding.
    Monitors PREM energy variance and dynamically expands beam/MCMC search budgets.
    """
    def __init__(self, base_budget: int = 10, max_budget: int = 100, scaling_factor: float = 10.0):
        self.base_budget = base_budget
        self.max_budget = max_budget
        self.scaling_factor = scaling_factor

    def get_budget(self, energy_history: List[float]) -> int:
        """
        Compute budget based on variance of energy history.
        """
        if len(energy_history) < 2:
            return self.base_budget
            
        variance = float(np.var(energy_history))
        
        # Scale budget based on variance
        additional_budget = int(np.round(variance * self.scaling_factor))
        new_budget = self.base_budget + additional_budget
        
        # Clamp to max_budget
        return min(max(self.base_budget, new_budget), self.max_budget)

def run_experiment_2150() -> None:
    import json
    import os
    
    controller = TTCController()
    low_var_budget = controller.get_budget([1.0, 1.05, 0.95, 1.0])
    high_var_budget = controller.get_budget([1.0, 5.0, -2.0, 10.0])
    
    results = {
        "status": "success",
        "experiment_id": 2150,
        "base_budget": controller.base_budget,
        "max_budget": controller.max_budget,
        "scaling_factor": controller.scaling_factor,
        "low_var_budget": low_var_budget,
        "high_var_budget": high_var_budget,
        "honest_verdict": "Dynamic budget controller successfully implemented, scaling TTC based on PREM energy variance."
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_2150_ttc_controller.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_experiment_2150()
