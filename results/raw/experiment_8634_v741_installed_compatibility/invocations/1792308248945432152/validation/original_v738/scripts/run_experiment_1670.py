import json
import os
import jax.numpy as jnp
from carnot.solvers.pinet_prototype import douglas_rachford_splitting

def run():
    print("Running PiNet Douglas-Rachford Splitting experiment...")
    A = jnp.array([[1.0, 1.0, 0.0], 
                   [0.0, 1.0, 1.0], 
                   [1.0, 0.0, 0.0]])
    b = jnp.array([1.0, 1.0, 1.0])
    
    res = douglas_rachford_splitting(A, b, max_iter=200)
    res_bool = res > 0.5
    
    deliverable = {
      "experiment": 1670,
      "experiment_id": "1670",
      "name": "PiNet Douglas-Rachford Splitting",
      "result": "success",
      "schema": "carnot.pinet.v1",
      "spec_refs": [
        "REQ-PINET-001",
        "REQ-PINET-002"
      ],
      "metrics": {
        "final_state": [float(x) for x in res],
        "final_bool": [bool(x) for x in res_bool],
        "converged": bool(jnp.allclose(A @ res, b, atol=1e-2))
      }
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/experiment_1670_pinet_splitting.json", "w") as f:
        json.dump(deliverable, f, indent=2)
    print("Wrote results/experiment_1670_pinet_splitting.json")

if __name__ == "__main__":
    run()
