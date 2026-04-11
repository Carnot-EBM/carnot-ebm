"""JEPA Violation Predictor — Tier 3 predictive verification mechanism.

**Researcher summary:**
    Trains a small MLP on (partial_response_embedding, final_violated) pairs
    collected by Exp 143. Predicts which constraint domains are likely to be
    violated before the response is complete, enabling early-exit or beam-reranking
    at low latency (MLP forward pass << EBM full verification).

**Detailed explanation for engineers:**
    The JEPA (Joint Embedding Predictive Architecture) paradigm in the Carnot
    pipeline works in three tiers:

    - Tier 1: Full EBM verification (high accuracy, high cost — run at the end)
    - Tier 2: Constraint projection guided decoding (mid-cost, per-clause)
    - Tier 3: *This module.* A lightweight MLP that takes a partial-response
      embedding (just the first 10–90% of tokens) and predicts which constraint
      domains (arithmetic, code, logic) are likely to be violated in the final
      response. If any domain probability exceeds a threshold, the generation
      can be stopped early, retried, or flagged.

    **Architecture:**
        Input: embedding vector of shape (256,) — same RandomProjection as Exp 143
        Layer 1: Linear(256 → 64), ReLU
        Layer 2: Linear(64 → 32), ReLU
        Layer 3: Linear(32 → N_domains) — raw logits per domain
        Output: sigmoid(logits) → per-domain violation probabilities in [0, 1]

    **Domains tracked (N_domains = 3):**
        - "arithmetic": Numeric computation errors (e.g., 2+2=5)
        - "code": Syntax or logic errors in generated code
        - "logic": Logical contradictions or invalid inferences

    **EnergyFunction protocol compatibility:**
        The predictor also satisfies the ``EnergyFunction`` protocol so that
        it can be dropped into any sampler or trainer that expects an EBM:
            - ``energy(x)`` = mean violation probability across all domains
              (high energy = high risk of violation = the model "dislikes" x)
            - ``energy_batch(xs)`` = vectorised version
            - ``grad_energy(x)`` = JAX autodiff of energy w.r.t. x
            - ``input_dim`` = 256

    **Training:**
        Binary cross-entropy independently per domain (multi-label, not
        multi-class — a response can violate arithmetic AND code at once).
        Optimizer: Adam, 50 epochs, lr=1e-3.

    **Serialization:**
        Parameters saved as a single ``.safetensors`` file (6 numpy arrays:
        w1, b1, w2, b2, w3, b3). No JSON sidecar needed — architecture is
        fixed by the DOMAINS list.

Spec: REQ-VERIFY-003, REQ-JEPA-001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from safetensors.numpy import load_file, save_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAINS: list[str] = ["arithmetic", "code", "logic"]
"""Constraint domain names, in order. The MLP output index corresponds to each."""

EMBED_DIM: int = 256
"""Input embedding dimension — must match the RandomProjection used in Exp 143."""

HIDDEN1: int = 64
"""First hidden layer width."""

HIDDEN2: int = 32
"""Second hidden layer width."""

N_DOMAINS: int = len(DOMAINS)
"""Number of output heads (one per constraint domain)."""


# ---------------------------------------------------------------------------
# MLP parameter helpers (pure functions, no class state)
# ---------------------------------------------------------------------------


def _init_params(key: jax.Array) -> dict[str, jax.Array]:
    """Initialise MLP parameters with He (Kaiming) normal initialisation.

    **Detailed explanation for engineers:**
        He initialisation scales random weights by ``sqrt(2 / fan_in)`` which
        keeps the variance of activations roughly constant through ReLU layers.
        Without this, deep networks suffer from vanishing or exploding gradients.
        Each weight matrix ``wN`` has shape (in_features, out_features); each
        bias vector ``bN`` is zeros of shape (out_features,).

    Args:
        key: JAX PRNG key for reproducible initialisation.

    Returns:
        Dict of parameter arrays: w1, b1, w2, b2, w3, b3.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Layer 1: (EMBED_DIM, HIDDEN1) — He init with fan_in = EMBED_DIM
    w1 = jax.random.normal(k1, (EMBED_DIM, HIDDEN1)) * np.sqrt(2.0 / EMBED_DIM)
    # Layer 2: (HIDDEN1, HIDDEN2) — He init with fan_in = HIDDEN1
    w2 = jax.random.normal(k2, (HIDDEN1, HIDDEN2)) * np.sqrt(2.0 / HIDDEN1)
    # Layer 3: (HIDDEN2, N_DOMAINS) — small init for final logits
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
    """Forward pass: embedding → raw logits per domain.

    **Detailed explanation for engineers:**
        Implements: x → ReLU(x @ w1 + b1) → ReLU(h1 @ w2 + b2) → h2 @ w3 + b3
        Returns raw logits (not probabilities). Apply ``jax.nn.sigmoid`` to get
        per-domain violation probabilities in [0, 1].

    Args:
        params: Dict of parameter arrays from ``_init_params``.
        x: Input embedding, shape (EMBED_DIM,) or (batch, EMBED_DIM).

    Returns:
        Raw logits of shape (N_DOMAINS,) or (batch, N_DOMAINS).
    """
    h1 = jax.nn.relu(x @ params["w1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["w2"] + params["b2"])
    logits = h2 @ params["w3"] + params["b3"]
    return logits


def _bce_loss(params: dict[str, jax.Array], x: jax.Array, y: jax.Array) -> jax.Array:
    """Mean binary cross-entropy loss across all domains and samples.

    **Detailed explanation for engineers:**
        Binary cross-entropy (BCE) is the standard loss for multi-label
        classification where each output is an independent Bernoulli variable.
        For a single domain with logit ``z`` and label ``y ∈ {0, 1}``:
            BCE(z, y) = -y * log(σ(z)) - (1-y) * log(1 - σ(z))
        where σ is the sigmoid function. JAX's ``optax.sigmoid_binary_cross_entropy``
        implements this numerically stably (avoids log(0) via log-sum-exp trick).

        We average across all (sample, domain) pairs so that the loss is
        independent of batch size and number of domains.

    Args:
        params: Model parameter dict.
        x: Batch of embeddings, shape (batch, EMBED_DIM).
        y: Batch of binary labels, shape (batch, N_DOMAINS). 0 = no violation.

    Returns:
        Scalar mean BCE loss.
    """
    logits = _forward(params, x)  # (batch, N_DOMAINS)
    per_element = optax.sigmoid_binary_cross_entropy(logits, y)  # (batch, N_DOMAINS)
    return jnp.mean(per_element)


# JIT-compiled gradient function — compiled once on first call.
_grad_loss = jax.jit(jax.value_and_grad(_bce_loss))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class JEPAViolationPredictor:
    """MLP that predicts per-domain constraint violations from partial embeddings.

    **Researcher summary:**
        Tier 3 JEPA predictor: given a RandomProjection embedding of a partial
        response, outputs P(violation | domain) for each of arithmetic, code,
        and logic constraints. Trained on Exp 143 data. Can gate Tier 1 EBM
        evaluation based on is_high_risk() threshold.

    **Detailed explanation for engineers:**
        This class wraps a simple 3-layer MLP (256→64→32→3) and provides:

        - ``train(pairs)``: Fit the MLP to (embedding, labels) pairs with Adam.
        - ``predict(x)``: Return per-domain violation probabilities as a dict.
        - ``is_high_risk(x, threshold)``: Boolean gate for early stopping.
        - ``save(path)`` / ``load(path)``: Persist parameters to safetensors.
        - ``energy(x)`` / ``energy_batch(xs)`` / ``grad_energy(x)``: EBM protocol.

        The ``energy`` function returns the mean sigmoid output across all domains
        (a scalar in [0, 1]), so ``energy ≈ 0`` means the partial response looks
        safe, and ``energy ≈ 1`` means high violation risk.

    Example::

        predictor = JEPAViolationPredictor()
        log = predictor.train(pairs)
        probs = predictor.predict(embedding)  # {"arithmetic": 0.82, ...}
        if predictor.is_high_risk(embedding):
            restart_generation()

    Spec: REQ-VERIFY-003, REQ-JEPA-001
    """

    # --- public attributes ---
    input_dim: int = EMBED_DIM
    """Input dimension: 256. Fixed by the Exp 143 RandomProjection embeddings."""

    domains: list[str] = DOMAINS
    """Constraint domain names in output index order."""

    def __init__(self, seed: int = 0) -> None:
        """Initialise with random parameters.

        Args:
            seed: Integer PRNG seed for reproducible weight initialisation.
        """
        key = jax.random.PRNGKey(seed)
        self._params: dict[str, jax.Array] = _init_params(key)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(self, partial_embedding: jnp.ndarray) -> dict[str, float]:
        """Predict per-domain violation probabilities from a partial embedding.

        **Detailed explanation for engineers:**
            Runs the MLP forward pass and applies sigmoid to convert raw logits
            to probabilities. The result is a dict so that callers can query
            specific domains without remembering index order.

        Args:
            partial_embedding: 1-D array of shape (EMBED_DIM,) = (256,).
                Produced by ``RandomProjectionEmbedding.encode(partial_text)``.

        Returns:
            Dict mapping domain name → float probability in [0, 1].
            E.g. {"arithmetic": 0.73, "code": 0.12, "logic": 0.08}

        Spec: REQ-JEPA-001, SCENARIO-JEPA-001
        """
        x = jnp.asarray(partial_embedding, dtype=jnp.float32)
        logits = _forward(self._params, x)
        probs = jax.nn.sigmoid(logits)
        return {domain: float(probs[i]) for i, domain in enumerate(DOMAINS)}

    def is_high_risk(
        self,
        partial_embedding: jnp.ndarray,
        threshold: float = 0.5,
    ) -> bool:
        """Return True if any domain violation probability exceeds threshold.

        **Detailed explanation for engineers:**
            This is the fast-path gate: call this *before* running the expensive
            Tier 1 EBM verification. If it returns True, the current generation
            is likely to produce a constraint violation and should be retried or
            flagged.

            **False negatives are costly** (we miss a real violation). The default
            threshold=0.5 is conservative (more false positives = more retries,
            but fewer missed violations). Lower the threshold for stricter gating.

        Args:
            partial_embedding: 1-D array of shape (256,).
            threshold: Violation probability above which we flag as high-risk.
                Default 0.5.

        Returns:
            True if max(predict(x).values()) >= threshold, else False.

        Spec: REQ-JEPA-001, SCENARIO-JEPA-002
        """
        probs = self.predict(partial_embedding)
        return max(probs.values()) >= threshold

    # ------------------------------------------------------------------
    # EnergyFunction protocol
    # ------------------------------------------------------------------

    def energy(self, x: jax.Array) -> jax.Array:
        """EBM energy = mean per-domain violation probability (scalar in [0,1]).

        **Detailed explanation for engineers:**
            Maps a 256-D embedding to a scalar "energy" that represents how
            risky the partial response looks. By convention in EBMs, lower energy
            = more likely (normal/safe) configuration. So:
                - energy ≈ 0.0 → the response looks constraint-satisfying
                - energy ≈ 1.0 → the response looks like it will violate constraints

            This is compatible with the ``EnergyFunction`` protocol so that
            the predictor can be plugged into Carnot samplers and trainers.

        Args:
            x: 1-D JAX array of shape (EMBED_DIM,).

        Returns:
            Scalar JAX array — mean violation probability across all domains.

        Spec: REQ-CORE-002, REQ-JEPA-001
        """
        logits = _forward(self._params, x)
        return jnp.mean(jax.nn.sigmoid(logits))

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Batched energy: shape (batch,) vector of mean violation probabilities.

        Args:
            xs: 2-D JAX array of shape (batch_size, EMBED_DIM).

        Returns:
            1-D JAX array of shape (batch_size,).

        Spec: REQ-CORE-002
        """
        logits = _forward(self._params, xs)  # (batch, N_DOMAINS)
        return jnp.mean(jax.nn.sigmoid(logits), axis=-1)  # (batch,)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy w.r.t. x via JAX autodiff.

        **Detailed explanation for engineers:**
            This is the gradient used by Langevin/HMC samplers to move x in
            the direction of decreasing energy (increasing constraint-safety).
            Computed automatically by ``jax.grad`` — no manual math required.

        Args:
            x: 1-D JAX array of shape (EMBED_DIM,).

        Returns:
            Gradient array of shape (EMBED_DIM,).

        Spec: REQ-CORE-002
        """
        return jax.grad(self.energy)(x)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        pairs: list[dict[str, Any]],
        n_epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Fit the MLP on (embedding, per-domain labels) pairs.

        **Detailed explanation for engineers:**
            Training procedure:
            1. Parse pairs into X (embeddings) and Y (binary label matrix).
            2. Stratified 80/20 train/val split on ``any_violated``.
            3. Run ``n_epochs`` epochs of mini-batch Adam gradient descent.
            4. Each epoch: shuffle training set, iterate over mini-batches,
               apply gradient update.
            5. After each epoch: record train loss and val loss.
            6. Return training log with losses + AUROC metrics on val set.

            **Why stratified split?**
            With 268/800 positives (~33%), a random split could give a val set
            that's mostly negative — making AUROC numerically unstable. Stratified
            split ensures both splits have ~33% positives.

            **Adam optimizer:**
            Adam (Adaptive Moment Estimation) adapts the learning rate for each
            parameter based on estimates of the first and second moments of the
            gradient. It's robust to noisy gradients and typically converges faster
            than plain SGD for small MLPs.

        Args:
            pairs: List of dicts from ``jepa_training_pairs.json``. Each dict must
                have keys: ``embedding`` (list of 256 floats) and per-domain boolean
                labels: ``violated_arithmetic``, ``violated_code``, ``violated_logic``.
            n_epochs: Number of full passes through the training set. Default 50.
            lr: Adam learning rate. Default 1e-3.
            batch_size: Mini-batch size. Default 64.
            val_fraction: Fraction of data to hold out for validation. Default 0.2.
            seed: Random seed for train/val split and epoch shuffling.

        Returns:
            Dict with keys:
                - ``train_losses``: list of float, one per epoch.
                - ``val_losses``: list of float, one per epoch.
                - ``val_auroc_per_domain``: dict mapping domain → AUROC on val set.
                - ``macro_auroc``: float, mean AUROC across domains.
                - ``precision_at_05``: dict mapping domain → precision at threshold 0.5.
                - ``recall_at_05``: dict mapping domain → recall at threshold 0.5.
                - ``n_train``: int, number of training samples.
                - ``n_val``: int, number of validation samples.

        Spec: REQ-JEPA-001, SCENARIO-JEPA-003
        """
        from sklearn.metrics import roc_auc_score

        # ---- 1. Parse pairs into numpy arrays ----
        X = np.array([p["embedding"] for p in pairs], dtype=np.float32)
        Y = np.array(
            [
                [
                    float(p.get("violated_arithmetic", False)),
                    float(p.get("violated_code", False)),
                    float(p.get("violated_logic", False)),
                ]
                for p in pairs
            ],
            dtype=np.float32,
        )
        any_violated = (Y.sum(axis=1) > 0).astype(int)

        # ---- 2. Stratified 80/20 split ----
        rng = np.random.RandomState(seed)
        pos_idx = np.where(any_violated == 1)[0]
        neg_idx = np.where(any_violated == 0)[0]

        n_pos_val = max(1, int(len(pos_idx) * val_fraction))
        n_neg_val = max(1, int(len(neg_idx) * val_fraction))

        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
        train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]

        # ---- 3. Set up Adam optimizer ----
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self._params)

        # JIT-compile the update step for speed.
        @jax.jit
        def _step(
            params: dict[str, jax.Array],
            state: Any,
            x_batch: jax.Array,
            y_batch: jax.Array,
        ) -> tuple[dict[str, jax.Array], Any, jax.Array]:
            """One Adam gradient step on a mini-batch.

            Args:
                params: Current model parameters.
                state: Optimizer state (Adam moment estimates).
                x_batch: Batch embeddings, shape (batch, EMBED_DIM).
                y_batch: Batch labels, shape (batch, N_DOMAINS).

            Returns:
                Tuple of (updated_params, updated_state, batch_loss).
            """
            loss, grads = _grad_loss(params, x_batch, y_batch)
            updates, new_state = optimizer.update(grads, state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, loss

        n_train = len(X_train)
        train_losses: list[float] = []
        val_losses: list[float] = []

        # ---- 4. Training loop ----
        for epoch in range(n_epochs):
            # Shuffle training set each epoch.
            perm = rng.permutation(n_train)
            X_shuf = X_train[perm]
            Y_shuf = Y_train[perm]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                x_b = jnp.asarray(X_shuf[start:end])
                y_b = jnp.asarray(Y_shuf[start:end])
                self._params, opt_state, batch_loss = _step(
                    self._params, opt_state, x_b, y_b
                )
                epoch_loss += float(batch_loss)
                n_batches += 1

            train_losses.append(epoch_loss / max(n_batches, 1))

            # Validation loss (no gradient).
            val_loss = float(_bce_loss(self._params, jnp.asarray(X_val), jnp.asarray(Y_val)))
            val_losses.append(val_loss)

        # ---- 5. Compute validation metrics ----
        val_logits = np.array(_forward(self._params, jnp.asarray(X_val)))
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))  # sigmoid

        auroc_per_domain: dict[str, float] = {}
        precision_at_05: dict[str, float] = {}
        recall_at_05: dict[str, float] = {}

        for i, domain in enumerate(DOMAINS):
            y_true = Y_val[:, i]
            y_prob = val_probs[:, i]
            y_pred = (y_prob >= 0.5).astype(float)

            # AUROC — if only one class present, report 0.5 (undefined).
            if len(np.unique(y_true)) < 2:
                auroc_per_domain[domain] = 0.5
            else:
                auroc_per_domain[domain] = float(roc_auc_score(y_true, y_prob))

            # Precision = TP / (TP + FP); recall = TP / (TP + FN).
            tp = float(((y_pred == 1) & (y_true == 1)).sum())
            fp = float(((y_pred == 1) & (y_true == 0)).sum())
            fn = float(((y_pred == 0) & (y_true == 1)).sum())

            precision_at_05[domain] = tp / max(tp + fp, 1.0)
            recall_at_05[domain] = tp / max(tp + fn, 1.0)

        macro_auroc = float(np.mean(list(auroc_per_domain.values())))

        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_auroc_per_domain": auroc_per_domain,
            "macro_auroc": macro_auroc,
            "precision_at_05": precision_at_05,
            "recall_at_05": recall_at_05,
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model parameters to a safetensors file.

        **Detailed explanation for engineers:**
            Converts each JAX array to NumPy (required by safetensors), then
            writes all six parameter tensors (w1, b1, w2, b2, w3, b3) into a
            single ``.safetensors`` binary file. The format is:
                - A JSON header listing tensor names, dtypes, shapes, and offsets
                - Raw binary data for each tensor

            To reconstruct the model: create a new ``JEPAViolationPredictor()``
            and call ``load(path)`` — the architecture constants (EMBED_DIM,
            HIDDEN1, HIDDEN2, N_DOMAINS) are fixed in this module, so no config
            sidecar is needed.

        Args:
            path: File path to write (e.g. "results/jepa_predictor.safetensors").
                Parent directory must exist.

        Spec: REQ-JEPA-001
        """
        np_params = {k: np.array(v, dtype=np.float32) for k, v in self._params.items()}
        save_file(np_params, path)

    def load(self, path: str) -> None:
        """Load model parameters from a safetensors file (in-place).

        **Detailed explanation for engineers:**
            Reads the safetensors file and converts each NumPy array back to a
            JAX array. Replaces self._params in-place, so any previously trained
            weights are discarded. Call on a freshly constructed predictor to
            restore a checkpoint.

        Args:
            path: Path to an existing ``.safetensors`` file written by ``save()``.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is missing required parameter keys.

        Spec: REQ-JEPA-001
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"No safetensors file at: {path}")
        raw = load_file(path)
        required = {"w1", "b1", "w2", "b2", "w3", "b3"}
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"safetensors file missing keys: {sorted(missing)}")
        self._params = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in raw.items()}
