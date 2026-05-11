"""Experiment 1750: Evaluate Symbolic-KAN constraint accuracy vs CIKAN.

Spec references: REQ-KAN-1750, SCENARIO-KAN-1750.
"""

import json
from pathlib import Path
import jax
import jax.numpy as jnp
import optax
import numpy as np

from carnot.models.cikan_verifier import CIKAN
from carnot.models.kan.symbolic_kan import SymbolicKANConfig, SymbolicRoutingLayer, SymbolicKANParams

class MockConstraint:
    def __init__(self, vars, expr, poly):
        self.variables = vars
        self.expression = expr
        self.polynomial = poly

def generate_toy_dataset():
    """Generate toy dataset for logic rules."""
    np.random.seed(42)
    # Generate random points in [0, 1]
    X = np.random.uniform(0, 1, (200, 2))
    # AND logic rule: target is 1 if both features are > 0.5
    Y = ((X[:, 0] > 0.5) & (X[:, 1] > 0.5)).astype(np.float32)
    return X, Y

def train_cikan(X, Y):
    constraint = MockConstraint(["f1", "f2"], "(f1 AND f2)", "f1*f2")
    cikan = CIKAN(
        feature_names=["f1", "f2"],
        constraints=[constraint],
        n_knots=5,
        learning_rate=0.1,
        seed=42
    )
    # Fit CIKAN
    cikan.fit(X, Y, epochs=100)
    preds = cikan.predict(X)
    accuracy = float(np.mean(preds == Y))
    return accuracy

def extract_params(p):
    return {
        "projection_weights": p.projection_weights,
        "projection_bias": p.projection_bias,
        "route_logits": p.route_logits,
        "route_scales": p.route_scales,
        "output_bias": p.output_bias
    }

def repack_params(d):
    return SymbolicKANParams(
        projection_weights=d["projection_weights"],
        projection_bias=d["projection_bias"],
        route_logits=d["route_logits"],
        route_scales=d["route_scales"],
        output_bias=d["output_bias"]
    )

def train_symbolic_kan(X, Y):
    X_jax = jnp.array(X, dtype=jnp.float32)
    Y_jax = jnp.array(Y, dtype=jnp.float32)
    
    config = SymbolicKANConfig(input_dim=2, n_routes=2, primitives=("identity", "square", "sin", "abs"))
    layer = SymbolicRoutingLayer(config)
    params_dict = extract_params(layer.params)
    
    optimizer = optax.adam(0.1)
    opt_state = optimizer.init(params_dict)
    
    @jax.jit
    def loss_fn(p_dict, x, y):
        p = repack_params(p_dict)
        energy = layer.forward(x, params=p, hard=False)
        # Binary cross entropy
        probs = jax.nn.sigmoid(-energy)
        probs = jnp.clip(probs, 1e-7, 1.0 - 1e-7)
        loss = -jnp.mean(y * jnp.log(probs) + (1.0 - y) * jnp.log(1.0 - probs))
        return loss

    @jax.jit
    def step(p_dict, state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(p_dict, x, y)
        updates, state = optimizer.update(grads, state)
        p_dict = optax.apply_updates(p_dict, updates)
        return p_dict, state, loss
        
    for _ in range(100):
        params_dict, opt_state, loss = step(params_dict, opt_state, X_jax, Y_jax)
        
    final_params = repack_params(params_dict)
    final_energy = layer.forward(X_jax, params=final_params, hard=True)
    preds = (jax.nn.sigmoid(-final_energy) >= 0.5).astype(jnp.float32)
    accuracy = float(jnp.mean(preds == Y_jax))
    return float(accuracy)

def run_experiment(output_path: str = "/home/ianblenke/github.com/ianblenke/carnot/results/experiment_1750_symbolic_eval.json"):
    X, Y = generate_toy_dataset()
    cikan_acc = train_cikan(X, Y)
    skan_acc = train_symbolic_kan(X, Y)
    
    result = {
        "schema": "carnot.experiment_1750.v1",
        "status": "complete",
        "dataset": "logic_rules_AND",
        "cikan_accuracy": cikan_acc,
        "symbolic_kan_accuracy": skan_acc,
        "constraint_preservation_rate": {
            "cikan": cikan_acc,
            "symbolic_kan": skan_acc
        }
    }
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        
    return result

if __name__ == "__main__":
    run_experiment()
