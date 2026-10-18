"""Extended JEPA Violation Predictor for FR-11.

**Researcher summary:**
    Tier 3 JEPA predictor extended with IsingVerifier energy and logprob variance.
    Input dimension is 258.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

DOMAINS: list[str] = ["arithmetic", "code", "logic"]
EMBED_DIM: int = 258  # 256 existing + 2 new features
HIDDEN1: int = 64
HIDDEN2: int = 32
N_DOMAINS: int = len(DOMAINS)


def _init_params(key: jax.Array) -> dict[str, jax.Array]:
    k1, k2, k3 = jax.random.split(key, 3)

    w1 = jax.random.normal(k1, (EMBED_DIM, HIDDEN1)) * np.sqrt(2.0 / EMBED_DIM)
    w2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * np.sqrt(2.0 / HIDDEN1)
    w3 = jax.random.normal(k3, (HIDDEN2, N_DOMAINS)) * np.sqrt(2.0 / HIDDEN2)

    return {
        "w1": w1.astype(jnp.float32),
        "b1": jnp.zeros((HIDDEN1,), dtype=jnp.float32),
        "w2": w2.astype(jnp.float32),
        "b2": jnp.zeros((HIDDEN2,), dtype=jnp.float32),
        "w3": w3.astype(jnp.float32),
        "b3": jnp.zeros((N_DOMAINS,), dtype=jnp.float32),
    }


def _forward(params: dict[str, jax.Array], x: jax.Array) -> jax.Array:
    h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    logits = h2 @ params["w3"] + params["b3"]
    return logits


def _bce_loss(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    logits = _forward(params, x)
    per_element = optax.sigmoid_binary_cross_entropy(logits, y)
    return jnp.mean(per_element)


_grad_loss = jax.jit(jax.value_and_grad(_bce_loss))


class FR11ExtendedJEPA:
    """MLP that predicts per-domain constraint violations with extended features."""

    input_dim: int = EMBED_DIM
    domains: list[str] = DOMAINS

    def __init__(self, seed: int = 0) -> None:
        key = jax.random.PRNGKey(seed)
        self._params: dict[str, jax.Array] = _init_params(key)

    def predict(self, partial_embedding: jnp.ndarray) -> dict[str, float]:
        x = jnp.asarray(partial_embedding, dtype=jnp.float32)
        logits = _forward(self._params, x)
        probs = jax.nn.sigmoid(logits)
        return {domain: float(probs[i]) for i, domain in enumerate(DOMAINS)}

    def energy(self, x: jax.Array) -> jax.Array:
        logits = _forward(self._params, x)
        return jnp.mean(jax.nn.sigmoid(logits))

    def train(
        self,
        pairs: list[dict[str, Any]],
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Train the model."""
        import sklearn.metrics
        
        # Prepare data
        X_list = []
        Y_list = []
        for p in pairs:
            # Expected format: embedding has length 258
            X_list.append(p["embedding"])
            Y_list.append([p.get(f"violated_{d}", False) for d in DOMAINS])

        X = jnp.array(X_list, dtype=jnp.float32)
        Y = jnp.array(Y_list, dtype=jnp.float32)

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self._params)

        @jax.jit
        def step(params, opt_state, x_batch, y_batch):
            loss, grads = _grad_loss(params, x_batch, y_batch)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        n_samples = len(X)
        for epoch in range(n_epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = X[batch_idx]
                y_batch = Y[batch_idx]
                self._params, opt_state, loss = step(self._params, opt_state, x_batch, y_batch)
                
        # Calculate AUC
        logits = _forward(self._params, X)
        probs = jax.nn.sigmoid(logits)
        
        aucs = {}
        for i, d in enumerate(DOMAINS):
            y_true = np.array(Y[:, i])
            y_pred = np.array(probs[:, i])
            if len(np.unique(y_true)) > 1:
                aucs[d] = float(sklearn.metrics.roc_auc_score(y_true, y_pred))
                
        return {
            "macro_auroc": float(np.mean(list(aucs.values()))) if aucs else 0.5
        }
