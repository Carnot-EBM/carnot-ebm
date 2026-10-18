"""KAN4CBC MILP Z3 Verification.

Spec references: REQ-KAN-2083, SCENARIO-KAN-2083.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import z3

def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[4]

def verify_milp_kan_robustness(
    input_lb: float, input_ub: float, epsilon: float
) -> dict[str, object]:
    """Verify a simple robustness property of a MILP KAN using Z3.
    
    A toy MILP KAN piecewise linear edge: y = 2x if x > 0 else 0.5x
    Property: If x is within epsilon of 0.5, y is within bounds.
    """
    start_time = time.time()
    
    solver = z3.Solver()
    
    x = z3.Real('x')
    y = z3.Real('y')
    
    # Domain constraints on x
    solver.add(x >= input_lb)
    solver.add(x <= input_ub)
    
    # MILP KAN piecewise linear edge
    solver.add(
        z3.If(x > 0, y == 2.0 * x, y == 0.5 * x)
    )
    
    # Robustness property: for x in [0.5 - epsilon, 0.5 + epsilon],
    # y should be in [2.0 * (0.5 - epsilon), 2.0 * (0.5 + epsilon)]
    
    solver.push()
    solver.add(x >= 0.5 - epsilon)
    solver.add(x <= 0.5 + epsilon)
    
    # We want to verify that the property HOLDS. 
    # To verify a property, we assert its NEGATION and check for UNSAT.
    property_holds = (y >= 2.0 * (0.5 - epsilon)) & (y <= 2.0 * (0.5 + epsilon))
    solver.add(z3.Not(property_holds))
    
    result = solver.check()
    
    # If UNSAT, the property holds for all valid x.
    is_robust = (result == z3.unsat)
    solver.pop()
    
    execution_time = time.time() - start_time
    
    return {
        "status": "complete",
        "is_robust": is_robust,
        "execution_time_s": execution_time,
    }

def build_experiment_2083_artifact(run_date: str = "20260513") -> dict[str, object]:
    """Build the stable Exp 2083 KAN4CBC artifact payload."""
    result = verify_milp_kan_robustness(input_lb=-1.0, input_ub=1.0, epsilon=0.1)
    
    return {
        "schema": "carnot.kan4cbc.experiment_2083.v1",
        "status": "complete",
        "experiment_id": 2083,
        "run_date": run_date,
        "spec_traces": ["REQ-KAN-2083", "SCENARIO-KAN-2083"],
        "module": "python/carnot/models/kan/kan4cbc.py",
        "artifact_path": "results/experiment_2083_kan4cbc.json",
        "honest_verdict": "complete: kan4cbc_smt_robustness_verification",
        "is_robust": bool(result["is_robust"]),
        "execution_time_s": float(result["execution_time_s"]),  # type: ignore
    }

def write_experiment_2083_artifact(
    output_path: str | Path | None = None,
    run_date: str = "20260513",
) -> dict[str, object]:
    """Write the stable Exp 2083 artifact to disk."""
    if output_path is None:
        output_path = _repo_root() / "results/experiment_2083_kan4cbc.json"
    artifact = build_experiment_2083_artifact(run_date=run_date)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

__all__ = [
    "verify_milp_kan_robustness",
    "build_experiment_2083_artifact",
    "write_experiment_2083_artifact",
]
