import json
import os
import jax
import jax.numpy as jnp
import optax
from typing import Any

from carnot.training.ebft_loss import EBFTLoss

def main():
    # 1. Setup a dummy verifier
    # We use a simple verifiable property: sequence should match a target pattern
    # Let's say energy = sum((x - target)^2)
    target_pattern = jnp.array([1.0, -1.0, 1.0, -1.0])
    
    def dummy_verifier(params: jnp.ndarray, seqs: jnp.ndarray) -> jnp.ndarray:
        # params can be a scaling factor or bias
        diff = seqs - target_pattern
        return jnp.sum(params[0] * diff**2 + params[1], axis=-1)

    # 2. Setup simulated dataset
    # Expert sequences: close to target
    expert_seqs = jnp.array([
        [0.9, -0.9, 0.9, -0.9],
        [1.1, -1.1, 1.1, -1.1],
        [1.0, -1.0, 1.0, -1.0]
    ])
    
    # Rollout sequences: random or bad sequences
    rollout_seqs = jnp.array([
        [0.0, 0.0, 0.0, 0.0],
        [-1.0, 1.0, -1.0, 1.0],
        [0.5, 0.5, -0.5, -0.5]
    ])
    
    # 3. Initialize loss and parameters
    loss_fn = EBFTLoss(dummy_verifier)
    
    # Initial parameters for the energy function (verifier)
    # We want to learn parameters such that expert energy is low, rollout energy is high
    # Actually, in EBFT, we usually train the *model* (which generates rollouts).
    # But REQ-TRAIN-007 says: "The objective calculation supports gradient flow for differentiable energy functions."
    # Wait, "minimizes expert sequence energy while maximizing rollout sequence energy." 
    # If we train the energy function (verifier), we minimize expert energy and maximize rollout energy.
    params = jnp.array([1.0, 0.0]) 
    
    optimizer = optax.adam(learning_rate=0.1)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(p, state, experts, rollouts):
        def compute_loss(p_inner):
            return loss_fn(p_inner, experts, rollouts)
            
        loss_val, grads = jax.value_and_grad(compute_loss)(p)
        updates, new_state = optimizer.update(grads, state, p)
        new_p = optax.apply_updates(p, updates)
        return new_p, new_state, loss_val
        
    # 4. Micro-training loop for 10 epochs
    epochs = 10
    loss_curve = []
    
    for epoch in range(epochs):
        params, opt_state, loss_val = train_step(params, opt_state, expert_seqs, rollout_seqs)
        loss_curve.append(float(loss_val))
        
    # 5. Log to deliverable JSON
    deliverable = {
        "experiment_id": "2058",
        "task": "EBFT Scaffold",
        "epochs": epochs,
        "loss_curve": loss_curve,
        "final_params": params.tolist(),
        "status": "success",
        "note": "Scaffolded EBFT objective function using our new verifiers."
    }
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/experiment_2058_ebft_scaffold.json"
    with open(out_path, "w") as f:
        json.dump(deliverable, f, indent=2)
        
    print(f"Successfully wrote deliverable to {out_path}")

if __name__ == "__main__":
    main()
