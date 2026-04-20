"""OTV Verifier — One-Token Verification LoRA head for near-zero-cost LLM output checking.

**Why this module exists (arXiv 2603.01025 — OTV):**
    EORM (Tier 2) is a 55M-parameter transformer encoder that scores (question, response)
    pairs to estimate correctness.  It's effective but adds ~10 ms latency per call.  OTV
    (One-Token Verifier) replaces that encoder with a single LoRA linear projection applied
    to the *last hidden state* of the LLM that generated the response — the same hidden
    state is already computed during generation, so the OTV head adds near-zero wall-clock
    cost.

    The key insight from arXiv 2603.01025: the last token's hidden state contains rich
    self-assessment signal.  A single linear layer trained to predict correctness from that
    hidden state achieves AUC comparable to much larger discriminators, at a fraction of
    the compute.

**How it works:**
    1. During generation, the LLM produces a hidden state vector of shape (embed_dim,).
    2. OTVVerifier.score(hidden_state) applies W @ hidden_state → scalar → sigmoid.
    3. The resulting verification_score ∈ [0, 1] where 1 = predicted correct.
    4. OTVVerifier.train() fits W by minimising binary cross-entropy on labeled pairs.

**Relationship to EORM (Tier 2):**
    OTV sits at the same tier position as EORM but uses the generation's own hidden states
    rather than a separate encoder forward pass.  If OTV live AUC >= 0.80, it can replace
    EORM as Tier 2, saving 55M parameters and ~10 ms/call.

**CI safety:**
    train() calls assert_live_or_ci_skip() which enforces CARNOT_FORCE_LIVE when a live
    GPU is present but allows training to proceed in CI (CARNOT_IS_CI=1) and on CPU-only
    machines.  score() and predict() are always safe to call.

Spec: REQ-VERIFY-120, SCENARIO-VERIFY-160, SCENARIO-VERIFY-161, SCENARIO-VERIFY-162
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from carnot.pipeline.live_assertion import assert_live_or_ci_skip


@dataclass
class OTVVerificationToken:
    """Result from scoring a single hidden state with the OTV linear projection.

    Fields:
        token_logit: Raw pre-sigmoid output of the linear projection (W @ h + b).
            Positive values → predicted correct; negative → predicted incorrect.
        verification_score: Sigmoid of token_logit, in [0, 1].
            Values closer to 1 indicate the model predicts the response is correct.
        is_correct_pred: Boolean threshold decision at 0.5 cutoff.
            True when verification_score >= 0.5.
    """

    token_logit: float
    verification_score: float
    is_correct_pred: bool


class OTVVerifier:
    """One-Token Verifier — linear projection on LLM last hidden state for correctness prediction.

    Architecture:
        W: ndarray of shape (1, embed_dim) — the LoRA-style linear projection.
        b: float — scalar bias term.

    The projection maps a hidden state vector of shape (embed_dim,) to a scalar logit,
    which is passed through sigmoid to produce a verification score.

    Args:
        embed_dim: Dimensionality of the LLM hidden state vectors this verifier will
            be applied to.  Must match the hidden dimension of the generating model.
            Default 128 matches the synthetic stub hidden states used in Exp 592.
    """

    def __init__(self, embed_dim: int = 128) -> None:
        self.embed_dim = embed_dim
        # Linear layer: shape (1, embed_dim) — one output logit.
        # Initialised to zero so the untrained model predicts 0.5 for all inputs.
        self._W: np.ndarray = np.zeros((1, embed_dim), dtype=np.float32)
        self._b: float = 0.0

    def score(self, hidden_state: jnp.ndarray) -> float:
        """Apply linear projection → sigmoid and return the verification score.

        Args:
            hidden_state: JAX array of shape (embed_dim,).  This is the last token's
                hidden state from the generating LLM, already computed during generation.

        Returns:
            verification_score in [0, 1].  Higher = more likely correct.
        """
        h = np.array(hidden_state, dtype=np.float32).reshape(-1)
        logit = float((self._W @ h).item()) + self._b
        # Clamp logit to [-30, 30] to avoid overflow in exp.
        logit_clamped = max(-30.0, min(30.0, logit))
        return float(1.0 / (1.0 + np.exp(-logit_clamped)))

    def train(
        self,
        pairs: list[tuple[jnp.ndarray, bool]],
        n_epochs: int = 50,
    ) -> None:
        """Fit the linear projection on (hidden_state, is_correct) pairs.

        Uses plain SGD on binary cross-entropy loss.  assert_live_or_ci_skip()
        is called first — in CI, training is skipped entirely because hidden
        states from real models are not available.

        Args:
            pairs: List of (hidden_state, is_correct) tuples.
                hidden_state: jnp.ndarray of shape (embed_dim,).
                is_correct: True if the response was correct, False otherwise.
            n_epochs: Number of SGD passes over the full training set.  Default 50
                provides adequate convergence for small datasets (n < 200).
        """
        assert_live_or_ci_skip()

        if not pairs:
            return

        X = np.stack([np.array(h, dtype=np.float32).reshape(-1) for h, _ in pairs])  # (n, embed_dim)
        y = np.array([1.0 if correct else 0.0 for _, correct in pairs], dtype=np.float32)  # (n,)

        W = self._W.copy()  # (1, embed_dim)
        b = self._b
        lr = 0.01

        for _ in range(n_epochs):
            logits = (X @ W.T).ravel() + b  # (n,)
            probs = 1.0 / (1.0 + np.exp(-logits.clip(-30, 30)))  # (n,)
            err = probs - y  # (n,)
            grad_W = (err[:, None] * X).mean(axis=0, keepdims=True)  # (1, embed_dim)
            grad_b = float(err.mean())
            W -= lr * grad_W
            b -= lr * grad_b

        self._W = W
        self._b = b

    def predict(self, hidden_state: jnp.ndarray) -> OTVVerificationToken:
        """Score a hidden state and return a structured OTVVerificationToken.

        Args:
            hidden_state: JAX array of shape (embed_dim,).

        Returns:
            OTVVerificationToken with token_logit, verification_score, is_correct_pred.
        """
        h = np.array(hidden_state, dtype=np.float32).reshape(-1)
        logit = float((self._W @ h).item()) + self._b
        logit_clamped = max(-30.0, min(30.0, logit))
        vscore = float(1.0 / (1.0 + np.exp(-logit_clamped)))
        return OTVVerificationToken(
            token_logit=logit,
            verification_score=vscore,
            is_correct_pred=vscore >= 0.5,
        )
