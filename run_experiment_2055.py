import json
import time
import jax.numpy as jnp
from carnot.models.compositional_energy import CompositionalEnergyMinimizer

def run():
    print("Running CompositionalEnergyMinimizer experiment...")
    
    # 10-step pathfinding problem
    def boundary_start(x):
        return (x[0] - 0.0)**2
        
    def boundary_end(x):
        return (x[9] - 9.0)**2
        
    def path_step(i):
        def _step(x):
            return (x[i] - x[i-1] - 1.0)**2
        return _step
        
    sub_energies = [boundary_start, boundary_end] + [path_step(i) for i in range(1, 10)]
    
    minimizer = CompositionalEnergyMinimizer(sub_energies, learning_rate=0.1)
    
    # Initial state (zeros)
    init_state = jnp.zeros(10)
    
    final_state, energy_history = minimizer.minimize(init_state, steps=500)
    
    # Measure drop per step for first 10 steps
    drops = []
    for i in range(1, len(energy_history)):
        drops.append(round(float(energy_history[i-1] - energy_history[i]), 4))
        
    avg_drop = sum(drops[:10]) / 10.0 if len(drops) >= 10 else 0.0
    
    deliverable = {
      "experiment": 2055,
      "experiment_id": "2055",
      "name": "Compositional Energy Minimization",
      "result": "success",
      "schema": "carnot.comp_energy.v1",
      "spec_refs": [
        "REQ-VERIFY-2055",
        "SCENARIO-VERIFY-2055"
      ],
      "metrics": {
        "energy_drops": drops[:10],
        "final_energy": round(float(energy_history[-1]), 4),
        "avg_drop_first_10": round(float(avg_drop), 4)
      }
    }
    
    with open("results/experiment_2055_comp_energy.json", "w") as f:
        json.dump(deliverable, f, indent=2)
    print("Wrote results/experiment_2055_comp_energy.json")

if __name__ == "__main__":
    run()
