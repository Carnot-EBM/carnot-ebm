"""Exp 1771 SKM-style iterative projection baseline for Carnot's linear constraints.

Spec: REQ-VERIFY-1771, SCENARIO-VERIFY-1771.
"""

import json
from pathlib import Path

import numpy as np

from carnot.verify.skm_projection import LinearConstraintSystem, project_skm_randomized

ARTIFACT_PATH = Path("results/experiment_1771_tskm_projection.json")

def generate_synthetic_cctu_constraints(seed: int = 42) -> list[LinearConstraintSystem]:
    """Generate 50 synthetic linear systems mimicking CCTU constraints.
    
    Each system is a simple bounding box or a random hyperplane set
    that is guaranteed to be feasible.
    """
    rng = np.random.default_rng(seed)
    systems = []
    for i in range(50):
        # A simple feasible system: a random point is the center,
        # we generate random normal hyperplanes and bound them such that
        # the center is feasible, then perturb bounds randomly outward.
        center = rng.normal(size=5)
        matrix = rng.normal(size=(10, 5))
        bounds = matrix @ center + np.abs(rng.normal(size=10))
        
        system = LinearConstraintSystem(
            matrix=matrix,
            bounds=bounds,
            names=tuple(f"cctu_mock_row_{j}" for j in range(10)),
        )
        systems.append(system)
    return systems

def test_skm_randomized_projection_experiment_1771():
    systems = generate_synthetic_cctu_constraints(seed=1771)
    
    zero_violations = []
    iterations_taken = []
    
    for system in systems:
        # start from origin
        start = np.zeros(5)
        res = project_skm_randomized(system, start, max_iterations=5000)
        
        zero_violations.append(res.converged and res.max_constraint_violation <= 1e-9)
        iterations_taken.append(res.iterations)
        
    zero_violations_achieved = all(zero_violations)
    mean_projection_steps = float(np.mean(iterations_taken))
    
    artifact = {
        "schema": "carnot.tskm_projection.v1",
        "experiment": 1771,
        "run_date": "20260515",
        "zero_violations_achieved": zero_violations_achieved,
        "mean_projection_steps": mean_projection_steps,
        "honest_verdict": "terminal: complete baseline projection achieved strictly zero violations" if zero_violations_achieved else "terminal: failed to achieve zero violations",
    }
    
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_PATH, "w") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
        
    assert zero_violations_achieved
    assert artifact["schema"] == "carnot.tskm_projection.v1"
