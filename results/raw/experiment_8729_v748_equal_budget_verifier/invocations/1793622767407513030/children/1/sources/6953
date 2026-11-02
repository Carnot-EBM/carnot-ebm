#!/usr/bin/env python3
"""Experiment 2066: EBRM Training Loop via Contrastive Divergence.

Spec: REQ-LEARN-2066
"""

import json
import os
import sys

import jax
import jax.numpy as jnp
import optax

from carnot.training.ebrm_trainer import EBRMMLP, EBRMTrainer


def main():
    print("Running Experiment 2066: EBRM Training Loop...")
    
    # Initialize model and optimizer
    model = EBRMMLP(hidden_dim=32)
    optimizer = optax.adam(1e-3)
    trainer = EBRMTrainer(model, optimizer)
    
    # Setup data shapes and keys
    key = jax.random.PRNGKey(42)
    input_shape = (100, 16)  # Synthetic trace shape
    
    key, state_key = jax.random.split(key)
    state = trainer.create_train_state(state_key, input_shape)
    
    # Generate synthetic traces
    key, pos_key, neg_key = jax.random.split(key, 3)
    positive_traces = jax.random.normal(pos_key, input_shape) * 0.5 + 1.0  # offset for difference
    negative_traces = jax.random.normal(neg_key, input_shape)
    
    # Train using Contrastive Divergence
    epochs = 10
    print(f"Training for {epochs} epochs...")
    state, losses = trainer.train(state, positive_traces, negative_traces, epochs=epochs)
    
    print(f"Final training loss: {losses[-1]:.4f}")
    
    # Save the results
    deliverable_path = "results/experiment_2066_ebrm_training.json"
    os.makedirs(os.path.dirname(deliverable_path), exist_ok=True)
    
    result = {
        "status": "complete",
        "ebrm_trained": True,
        "epochs": epochs,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "honest_verdict": "complete_ebrm_cd_trained",
        "experiment_id": "2066",
        "model_hidden_dim": 32
    }
    
    with open(deliverable_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Wrote deliverable to {deliverable_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
