"""Experiment 1934: ConsFormer Refinement Loop.

Evaluates the ConsFormer-style iterative neural refinement loop against
deterministic solver baselines (gradient descent) on Graph Coloring.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from carnot.models.consformer import ConsFormerRefiner, refinement_loop
from carnot.verify.graph_coloring import build_coloring_energy

REPO_ROOT = Path(__file__).parent.parent
RESULT_PATH = REPO_ROOT / "results" / "experiment_1934_consformer_refinement.json"

def get_graph():
    # Simple graph: a cycle of 5 nodes
    n_nodes = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    n_colors = 3
    
    adj_matrix = jnp.zeros((n_nodes, n_nodes))
    for u, v in edges:
        adj_matrix = adj_matrix.at[u, v].set(1.0)
        adj_matrix = adj_matrix.at[v, u].set(1.0)
        
    return edges, n_nodes, n_colors, adj_matrix

def run_experiment():
    edges, n_nodes, n_colors, adj_matrix = get_graph()
    energy_fn = build_coloring_energy(edges, n_nodes, n_colors)
    
    # 1. Deterministic baseline: Gradient Descent on Energy
    def deterministic_loss(x):
        return energy_fn.energy(x)
        
    grad_fn = jax.grad(deterministic_loss)
    
    init_x = jnp.array([0.1, 1.1, 2.1, 0.1, 1.1])  # Initial guess
    
    # Run deterministic GD
    det_x = init_x
    det_energy_traj = []
    det_steps = 100
    det_lr = 0.05
    for _ in range(det_steps):
        det_energy_traj.append(float(deterministic_loss(det_x)))
        g = grad_fn(det_x)
        det_x = det_x - det_lr * g
    
    det_final_energy = float(deterministic_loss(det_x))
    
    # 2. ConsFormer self-supervised
    # Train the ConsFormer to output refinement steps that minimize energy
    model = ConsFormerRefiner(d_model=32, num_heads=2, num_layers=2)
    key = jax.random.PRNGKey(42)
    
    # Dummy input to init
    dummy_x = jnp.zeros((n_nodes, 1))
    params = model.init(key, dummy_x, adj_matrix)
    
    # Loss: Run refinement loop, evaluate energy of final state
    num_refine_steps = 10
    def consformer_loss(p):
        final_x, _ = refinement_loop(
            p, model, init_x, adj_matrix, num_steps=num_refine_steps, step_size=0.1
        )
        return energy_fn.energy(final_x)
        
    cons_grad_fn = jax.value_and_grad(consformer_loss)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(params)
    
    train_steps = 200
    cons_energy_traj = []
    for _ in range(train_steps):
        loss_val, grads = cons_grad_fn(params)
        cons_energy_traj.append(float(loss_val))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
    # Evaluate final
    final_x_cons, _ = refinement_loop(
        params, model, init_x, adj_matrix, num_steps=num_refine_steps, step_size=0.1
    )
    cons_final_energy = float(energy_fn.energy(final_x_cons))
    
    # Calculate feasibility (energy < threshold)
    threshold = 0.05
    det_feasible = det_final_energy < threshold
    cons_feasible = cons_final_energy < threshold
    
    results = {
        "schema": "carnot.consformer_refinement.v1",
        "title": "ConsFormer vs Deterministic GD on Graph Coloring",
        "n_nodes": n_nodes,
        "n_edges": len(edges),
        "deterministic_baseline": {
            "final_energy": det_final_energy,
            "feasible": det_feasible,
            "trajectory": det_energy_traj,
        },
        "consformer_refinement": {
            "final_energy": cons_final_energy,
            "feasible": cons_feasible,
            "trajectory": cons_energy_traj,
        },
        "objective_gap": det_final_energy - cons_final_energy,
    }
    
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Results written to {RESULT_PATH}")

if __name__ == "__main__":
    run_experiment()
