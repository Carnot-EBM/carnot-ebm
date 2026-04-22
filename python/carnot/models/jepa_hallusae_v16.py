"""JEPAHalluSAEv16 — JEPA v16 retrained with HalluSAE sparse features as input.

**What this module does and why:**
    JEPA v16 (Exp 698) used raw hash-based text embeddings (dim=256). Those embeddings
    encode surface-level token statistics but not semantic arithmetic structure.

    HalluSAE (Exp 687, arXiv 2604.16430) trains a sparse auto-encoder on text features
    that encode arithmetic structure (digit counts, COMPUTE: lines, operator sequences).
    The sparse AE's hidden activations are *monosemantic* — each active dimension has a
    single interpretable meaning.  arXiv 2604.16430 shows that using sparse AE features
    as input to a downstream classifier improves AUC by 5-10pp over raw hidden states.

    JEPAHalluSAEv16 replaces the raw embedding lookup in JEPAv16.score() with a two-step
    encode pipeline:
        1. extract_text_features(text)  → dim-134 arithmetic feature vector
        2. sae(features)                → (reconstructed, sparse_hidden), dim=512

    The sparse_hidden (dim=512, exactly one non-zero per sample due to top-1 masking) is
    used as the MLP input instead of the raw 256-D text embedding.  This lets the JEPA MLP
    learn to separate correct from incorrect steps based on which single SAE feature
    "explains" each step — a much more semantically stable basis than raw trigram hashes.

    The SAE weights are *frozen* during JEPA training.  This is the standard transfer-
    learning pattern: freeze the pretrained feature extractor, fine-tune only the head.
    Any gradient that would have flowed through the SAE is stopped at the boundary.

**Spec:** REQ-LEARN-055, SCENARIO-LEARN-090, SCENARIO-LEARN-091
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from carnot.models.hallusal_sparse_ae import SparseAutoEncoder, extract_text_features


class JEPAHalluSAEv16:
    """JEPA v16 MLP re-ranker retrained on frozen HalluSAE sparse features.

    **Architecture overview:**
        Input: text string
            → extract_text_features()     : (134,) float32 arithmetic feature vector
            → frozen SparseAutoEncoder    : (512,) float32 sparse hidden (top-1 non-zero)
            → MLP(512 → 128 → 64 → 1)    : scalar correctness score in [0, 1]

        The MLP is wider than v16 (256→64→32→1) because the SAE sparse code (512-D) is
        sparser and needs more capacity in the first layer to learn which of the 512
        features indicates correctness.

    **Frozen SAE rationale:**
        If we allowed gradients to flow through the SAE, the SAE weights would drift away
        from the monosemantic feature directions learned in Exp 687.  Freezing ensures the
        sparse code retains its interpretability: the JEPA MLP learns to read the SAE's
        already-learned dictionary, not to alter it.

    REQ-LEARN-055-1, REQ-LEARN-055-2, REQ-LEARN-055-3
    """

    # SAE hidden dimension (must match SparseAutoEncoder.hidden_dim).
    SAE_DIM: int = 512

    def __init__(
        self,
        sae: SparseAutoEncoder,
        sae_params: dict[str, Any],
        seed: int = 42,
    ) -> None:
        """Initialise with a frozen SparseAutoEncoder and random MLP weights.

        Args:
            sae:        Flax SparseAutoEncoder module definition (not trained here).
            sae_params: Trained Flax parameter dict for sae (frozen — never updated).
            seed:       Random seed for reproducible MLP weight initialisation.

        Why params separate from module:
            Flax separates module definition from parameters.  The module defines the
            computation graph; sae_params holds the concrete float arrays.  By keeping
            them separate here, we make the "freeze by not calling optimizer.step on them"
            pattern explicit — there is simply no optimizer for sae_params.
        """
        self.sae = sae
        self.sae_params = sae_params  # frozen — never mutated after construction

        rng = np.random.default_rng(seed)

        # MLP: SAE_DIM(512) → 128 → 64 → 1
        # He initialisation: scale = sqrt(2 / fan_in) keeps gradients well-conditioned
        # through ReLU activations at the start of training.
        self._W1 = rng.standard_normal((self.SAE_DIM, 128)).astype(np.float32) * np.sqrt(2.0 / self.SAE_DIM)
        self._b1 = np.zeros(128, dtype=np.float32)
        self._W2 = rng.standard_normal((128, 64)).astype(np.float32) * np.sqrt(2.0 / 128)
        self._b2 = np.zeros(64, dtype=np.float32)
        self._W3 = rng.standard_normal((64, 1)).astype(np.float32) * np.sqrt(2.0 / 64)
        self._b3 = np.zeros(1, dtype=np.float32)

    def encode(self, text: str) -> np.ndarray:
        """Encode a text string to a sparse SAE feature vector.

        This is the key difference from JEPAv16.encode (which uses raw hash embeddings).
        By routing through the SAE we get a semantically meaningful sparse code where
        each dimension corresponds to one monosemantic hallucination feature.

        Args:
            text: Input reasoning step text.

        Returns:
            1-D numpy float32 array of shape (512,).
            Exactly one dimension is non-zero (top-1 sparsity from SparseAutoEncoder).

        REQ-LEARN-055-3, SCENARIO-LEARN-090
        """
        # Step 1: extract arithmetic features (134-D vector capturing digit counts,
        # COMPUTE: lines, trigram hashes, etc.)
        text_features = extract_text_features(text)  # jnp.ndarray shape (134,)

        # Step 2: run the frozen SAE forward pass.  We call model.apply() with the
        # frozen params — no gradient tracking, no parameter mutation.
        # The SAE returns (x_recon, h_sparse); we only need h_sparse.
        _x_recon, h_sparse = self.sae.apply(self.sae_params, text_features[None])  # batch dim

        # h_sparse shape: (1, 512).  Remove batch dim and convert to numpy for MLP.
        return np.asarray(jax.device_get(h_sparse[0]), dtype=np.float32)

    # ------------------------------------------------------------------
    # Internal helpers (same as JEPAv16 for consistency)
    # ------------------------------------------------------------------

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def _forward(self, x: np.ndarray) -> float:
        """Forward pass: SAE sparse vector → scalar correctness score in [0, 1].

        Args:
            x: 1-D numpy float32 array of shape (SAE_DIM,).

        Returns:
            Scalar float in [0, 1]. Higher = more likely correct.
        """
        h1 = self._relu(x @ self._W1 + self._b1)
        h2 = self._relu(h1 @ self._W2 + self._b2)
        return float(self._sigmoid(h2 @ self._W3 + self._b3)[0])

    def train(
        self,
        texts: list[str],
        labels: list[float],
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> dict[str, Any]:
        """Train the MLP re-ranker on SAE-encoded text features.

        **How this differs from JEPAv16.train():**
            JEPAv16 builds InfoNCE triplets and trains via contrastive loss.
            JEPAHalluSAEv16 uses binary cross-entropy directly on (text, label) pairs.
            The SAE features are pre-computed for the whole corpus before the training
            loop starts (calling self.encode() once per text) — this avoids rerunning
            the SAE forward pass on every epoch, which would be wasteful since the SAE
            is frozen and always produces the same output for a given text.

        Args:
            texts:    List of reasoning step texts.
            labels:   Corresponding binary labels: 1.0 = correct, 0.0 = incorrect.
            n_epochs: Number of training epochs. Default 200.
            lr:       Learning rate. Default 1e-3.

        Returns:
            Dict with keys: train_losses (list of float), n_train_pairs (int),
            sae_sparsity_rate (float — fraction of SAE dimensions that were non-zero
            across all training texts; should be close to 1/512 ≈ 0.002 for top-1).

        REQ-LEARN-055-2
        """
        # Pre-compute SAE encodings for all texts (frozen SAE → deterministic output).
        X_list = [self.encode(t) for t in texts]
        X = np.stack(X_list, axis=0)  # (N, 512)
        Y = np.array(labels, dtype=np.float32).reshape(-1, 1)  # (N, 1)
        N = len(X)

        # Compute sparsity rate: fraction of feature dimensions that are non-zero.
        # For perfect top-1 sparsity, this should equal 1 / SAE_DIM ≈ 0.002.
        sae_sparsity_rate = float(np.mean(X != 0.0))

        train_losses: list[float] = []
        rng = np.random.default_rng(0)

        for _ in range(n_epochs):
            perm = rng.permutation(N)
            X_shuf = X[perm]
            Y_shuf = Y[perm]

            h1 = self._relu(X_shuf @ self._W1 + self._b1)  # (N, 128)
            h2 = self._relu(h1 @ self._W2 + self._b2)       # (N, 64)
            logits = h2 @ self._W3 + self._b3               # (N, 1)
            preds = self._sigmoid(logits)

            eps = 1e-7
            bce = -np.mean(Y_shuf * np.log(preds + eps) + (1 - Y_shuf) * np.log(1 - preds + eps))
            train_losses.append(float(bce))

            # Analytic BCE-through-sigmoid gradient: d_loss/d_logit = (p - y) / N
            d_logit = (preds - Y_shuf) / N

            dW3 = h2.T @ d_logit
            db3 = d_logit.sum(axis=0)
            d_h2 = d_logit @ self._W3.T
            d_h2_pre = d_h2 * (h2 > 0).astype(np.float32)

            dW2 = h1.T @ d_h2_pre
            db2 = d_h2_pre.sum(axis=0)
            d_h1 = d_h2_pre @ self._W2.T
            d_h1_pre = d_h1 * (h1 > 0).astype(np.float32)

            dW1 = X_shuf.T @ d_h1_pre
            db1 = d_h1_pre.sum(axis=0)

            self._W3 -= lr * dW3
            self._b3 -= lr * db3
            self._W2 -= lr * dW2
            self._b2 -= lr * db2
            self._W1 -= lr * dW1
            self._b1 -= lr * db1

        return {
            "train_losses": train_losses,
            "n_train_pairs": N,
            "sae_sparsity_rate": sae_sparsity_rate,
        }

    def score(self, text: str) -> float:
        """Score a reasoning step text: returns P(correct) in [0, 1].

        Args:
            text: Input reasoning step text (same format as training texts).

        Returns:
            Float in [0, 1]. Higher = more likely correct.

        REQ-LEARN-055-3
        """
        return self._forward(self.encode(text))

    def save(self, path: str) -> None:
        """Save MLP weights to a numpy .npz file.

        Note: SAE weights are NOT saved here.  They should be loaded from the Exp 687
        result independently.  This file only stores the JEPA MLP re-ranker weights.

        Args:
            path: Output path (e.g. "results/jepa_hallusae_v16.npz").
        """
        np.savez(
            path,
            W1=self._W1, b1=self._b1,
            W2=self._W2, b2=self._b2,
            W3=self._W3, b3=self._b3,
        )

    def load(self, path: str) -> None:
        """Load MLP weights from a numpy .npz file (in-place).

        Args:
            path: Path to .npz file previously written by save().
        """
        data = np.load(path)
        self._W1 = data["W1"]
        self._b1 = data["b1"]
        self._W2 = data["W2"]
        self._b2 = data["b2"]
        self._W3 = data["W3"]
        self._b3 = data["b3"]
