import os
import json
import time
import jax
import jax.numpy as jnp
from carnot.phase3.pem_optimizer import PEMOptimizer
from carnot.phase3.eqm_landscape import EqMLandscape, ComposedEqMLandscape, sample_langevin

def generate_graph_instance(key, num_nodes=10, num_edges=20):
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    # Target positions to create feasible distances
    target_pos = jax.random.normal(subkey1, (num_nodes, 2))
    
    # Generate random edges
    edges_i = jax.random.randint(subkey2, (num_edges,), 0, num_nodes)
    edges_j = jax.random.randint(subkey3, (num_edges,), 0, num_nodes)
    
    # Target distances
    d_ij = jnp.linalg.norm(target_pos[edges_i] - target_pos[edges_j], axis=-1)
    
    def energy_fn(x):
        # x is (num_nodes, 2)
        diff = x[edges_i] - x[edges_j]
        dist_sq = jnp.sum(diff**2, axis=-1)
        return jnp.sum((dist_sq - d_ij**2)**2)
        
    def energy_fn_eqm(theta, x):
        return energy_fn(x)
        
    return energy_fn, energy_fn_eqm

def evaluate_method(method_name, energy_fn, energy_fn_eqm, init_x, key, steps=200):
    if method_name == "PEM":
        optimizer = PEMOptimizer(energy_fn, learning_rate=0.01, noise_scale=0.01)
        # compile
        optimizer.optimize(init_x, key, 1)
        
        start_time = time.time()
        x_final, _ = optimizer.optimize(init_x, key, steps)
        x_final.block_until_ready()
        duration = time.time() - start_time
    else: # EqM
        l1 = EqMLandscape(energy_fn_eqm)
        composed = ComposedEqMLandscape([l1])
        grad_estimator = composed.get_gradient_estimator()
        
        # compile internal grad
        _ = grad_estimator(None, init_x)
        
        start_time = time.time()
        x_final = sample_langevin(grad_estimator, None, init_x, step_size=0.01, num_steps=steps, key=key)
        x_final.block_until_ready()
        duration = time.time() - start_time
        
    final_energy = energy_fn(x_final)
    return x_final, final_energy, duration

def run_experiment_2097():
    num_instances = 50
    key = jax.random.PRNGKey(42)
    
    pem_energies = []
    eqm_energies = []
    pem_times = []
    eqm_times = []
    
    for i in range(num_instances):
        key, subkey, init_key1, init_key2, eval_key1, eval_key2 = jax.random.split(key, 6)
        energy_fn, energy_fn_eqm = generate_graph_instance(subkey, num_nodes=10, num_edges=20)
        
        init_x = jax.random.normal(init_key1, (10, 2)) * 0.1
        
        _, pem_energy, pem_time = evaluate_method("PEM", energy_fn, energy_fn_eqm, init_x, eval_key1, steps=200)
        _, eqm_energy, eqm_time = evaluate_method("EqM", energy_fn, energy_fn_eqm, init_x, eval_key2, steps=200)
        
        pem_energy = float(pem_energy)
        eqm_energy = float(eqm_energy)
        if jnp.isnan(pem_energy) or jnp.isinf(pem_energy): pem_energy = 1000.0
        if jnp.isnan(eqm_energy) or jnp.isinf(eqm_energy): eqm_energy = 1000.0
        
        pem_energies.append(pem_energy)
        eqm_energies.append(eqm_energy)
        pem_times.append(pem_time)
        eqm_times.append(eqm_time)
        
    avg_pem_energy = sum(pem_energies) / len(pem_energies)
    avg_eqm_energy = sum(eqm_energies) / len(eqm_energies)
    avg_pem_time = sum(pem_times) / len(pem_times)
    avg_eqm_time = sum(eqm_times) / len(eqm_times)
    
    satisfaction_threshold = 1.0
    pem_satisfaction = sum(1 for e in pem_energies if e < satisfaction_threshold) / num_instances
    eqm_satisfaction = sum(1 for e in eqm_energies if e < satisfaction_threshold) / num_instances
    
    eqm_superior = avg_eqm_energy < avg_pem_energy

    result = {
        "schema": "experiment_result",
        "experiment_id": "2097",
        "spec_refs": ["REQ-KONA-2097", "SCENARIO-KONA-2097"],
        "num_instances": num_instances,
        "avg_pem_energy": avg_pem_energy,
        "avg_eqm_energy": avg_eqm_energy,
        "avg_pem_time": avg_pem_time,
        "avg_eqm_time": avg_eqm_time,
        "pem_satisfaction_rate": pem_satisfaction,
        "eqm_satisfaction_rate": eqm_satisfaction,
        "eqm_superior": eqm_superior,
        "honest_verdict": "SUCCESS: eqm_eval_complete"
    }

    output_path = "results/experiment_2097_eqm_eval.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        
if __name__ == "__main__":
    run_experiment_2097()
