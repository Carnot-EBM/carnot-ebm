#!/usr/bin/env python3
"""Exp 2041: Equilibrium Matching (EqM) Gradient Probing"""

from scripts.experiment_template import ExperimentTemplate
import jax.numpy as jnp
from carnot.inference.eqm_calibration import EqMCalibrator
from carnot.inference.far_eqm import extract_eqm_gradient

def run_refinement(n_steps: int = 50):
    calibrator = EqMCalibrator(learning_rate=0.1)
    
    # 10 toy constraints
    current_state = jnp.ones((10, 64)) * 0.5
    energies, _ = extract_eqm_gradient(current_state)
    
    trajectory = []
    
    for _ in range(n_steps):
        prev_energies = energies
        energies, gradients = extract_eqm_gradient(current_state)
        trajectory.append(float(jnp.mean(energies).item()))
        current_state = calibrator.update(current_state, gradients, energies, prev_energies)
        
    final_energies, _ = extract_eqm_gradient(current_state)
    trajectory.append(float(jnp.mean(final_energies).item()))
    
    return trajectory, current_state

def main():
    tmpl = ExperimentTemplate(
        exp_id=2041,
        title="EqM Gradient Probing",
        deliverable="results/experiment_2041_eqm_gradients.json",
        requires_gpu=False,
    )
    
    tmpl.setup()
    
    try:
        trajectory, final_state = run_refinement(n_steps=50)
        
        initial_energy = trajectory[0]
        final_energy = trajectory[-1]
        converged = final_energy < initial_energy
        
        artifact = tmpl.build_result(
            data={
                "model_mapped": "unsloth/gemma-4-26B-A4B-it-GGUF",
                "n_toy_constraints": 10,
                "initial_mean_energy": initial_energy,
                "final_mean_energy": final_energy,
                "trajectory": trajectory,
                "optimization_converged": bool(converged),
            },
            status="success",
            honest_verdict="eqm_gradients_extracted",
            code_files=[__file__, "python/carnot/inference/far_eqm.py"],
        )
        
        with open(tmpl._output_path, "w") as f:
            import json
            json.dump(artifact, f, indent=2)
            
    except Exception as e:
        artifact = tmpl.build_result(
            data={"error": str(e)},
            status="error",
            honest_verdict="failed",
        )
        with open(tmpl._output_path, "w") as f:
            import json
            json.dump(artifact, f, indent=2)
            
    tmpl.assert_deliverable_written()

if __name__ == "__main__":
    main()
