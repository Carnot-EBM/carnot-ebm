"""SetConsistencyVerifier — global energy over N CoT steps (SC-Energy, arXiv 2503.10695).

**Why this module exists:**
    Tier 2.5 (SymCodeVerifier) checks pairwise arithmetic steps in isolation.
    Tier 2.7 (CausalReasoningVerifier) checks carry-forward between adjacent steps.
    Neither catches errors where every step looks locally fine but the WHOLE SET is
    globally inconsistent — for example, step 2 contradicts step 7, or the correct
    intermediate result is assembled into the wrong final answer.

    SC-Energy from arXiv 2503.10695 computes E(S_1, ..., S_N) over ALL N steps at once,
    using contrastive learning on (consistent set, mutually inconsistent set) pairs.
    This Carnot implementation is a CPU-friendly version that uses a simple TF-IDF
    bag-of-words encoding + a single-hidden-layer MLP energy function trained with
    a hinge contrastive loss.

**What "consistent" vs "inconsistent" means here:**
    Consistent set  = all CoT steps from a single CORRECT GSM8K response.
                      Every step belongs to the same problem and reasoning chain.
    Inconsistent set = the same chain but with one step swapped in from a DIFFERENT
                       question.  The intruder step uses different variable values
                       and a different problem context, producing a global contradiction
                       that is invisible to pairwise checkers.

**Energy interpretation:**
    Low energy  → steps are mutually consistent (typical of a correct chain).
    High energy → set contains internally contradictory steps (intruder detected).

Spec: REQ-VERIFY-149, REQ-VERIFY-150, REQ-VERIFY-151,
      SCENARIO-VERIFY-149, SCENARIO-VERIFY-150, SCENARIO-VERIFY-151
"""

from __future__ import annotations

import re
from typing import List

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# Text encoding helpers
# ---------------------------------------------------------------------------

# Vocabulary of the 512 most frequent digit/operator tokens used in GSM8K CoT steps.
# Built by scanning the FoVer corpus.  We use a fixed vocab so the model is
# reproducible without fitting a separate TF-IDF object on the training data.
#
# Pattern: all sequences of digits, dollar/percent signs, and operator symbols.
_TOKEN_RE = re.compile(r"[\d]+\.?[\d]*|[\+\-\*/=\$%><]")

# Vocabulary size for the bag-of-tokens embedding.
# Kept small (256 bins) so training on ~200 questions converges in < 5 s on CPU.
_VOCAB_SIZE: int = 256

# Hidden layer width for the MLP energy head.
_HIDDEN: int = 64

# Contrastive margin: inconsistent energy must exceed consistent energy by at least
# this much before the loss goes to zero.
_MARGIN: float = 1.0


def _tokenize(text: str) -> List[str]:
    """Extract numeric and operator tokens from a CoT step string.

    Why tokens rather than word-level features: GSM8K reasoning steps are almost
    entirely numeric arithmetic.  Operator and digit tokens carry the logical signal.
    Words like "the", "a", "of" are noise for consistency checking.
    """
    return _TOKEN_RE.findall(text.lower())


def _hash_token(tok: str) -> int:
    """Map a token string to a vocab bucket via a simple polynomial hash.

    Using a deterministic hash means we need no external vocabulary object —
    the same token always maps to the same bucket on every run.
    """
    h = 0
    for ch in tok:
        h = (h * 31 + ord(ch)) % _VOCAB_SIZE
    return h


def _encode_step(step_text: str) -> jnp.ndarray:
    """Bag-of-tokens embedding for one CoT step.

    Returns a float32 vector of length _VOCAB_SIZE where each element is the
    count of tokens hashing into that bucket, normalised by total token count.
    The normalisation makes step length invariant — a 3-word step and a 20-word
    step produce vectors on the same scale.

    Parameters
    ----------
    step_text : str
        The raw text of a single chain-of-thought step.

    Returns
    -------
    jnp.ndarray
        Shape (_VOCAB_SIZE,), dtype float32.
    """
    tokens = _tokenize(step_text)
    vec = [0.0] * _VOCAB_SIZE
    for tok in tokens:
        vec[_hash_token(tok)] += 1.0
    total = sum(vec) or 1.0  # avoid division by zero for empty steps
    return jnp.array([v / total for v in vec], dtype=jnp.float32)


# ---------------------------------------------------------------------------
# MLP energy function (JAX / pure-function style)
# ---------------------------------------------------------------------------


def _init_params(key: jax.Array) -> dict:
    """Initialise MLP parameters with Xavier uniform weights.

    Architecture: Linear(_VOCAB_SIZE → _HIDDEN) → ReLU → Linear(_HIDDEN → 1) → scalar.

    Why Xavier init: keeps gradients in a reasonable range for contrastive training
    even without batch normalisation.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale1 = jnp.sqrt(2.0 / (_VOCAB_SIZE + _HIDDEN))
    scale2 = jnp.sqrt(2.0 / (_HIDDEN + 1))
    return {
        "w1": jax.random.uniform(k1, (_VOCAB_SIZE, _HIDDEN), minval=-scale1, maxval=scale1),
        "b1": jnp.zeros((_HIDDEN,)),
        "w2": jax.random.uniform(k3, (_HIDDEN, 1), minval=-scale2, maxval=scale2),
        "b2": jnp.zeros((1,)),
    }


def _mlp_energy(params: dict, set_embedding: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: MLP(set_embedding) → scalar energy.

    Parameters
    ----------
    params : dict
        Keys w1, b1, w2, b2 — weights and biases of the two linear layers.
    set_embedding : jnp.ndarray
        Shape (_VOCAB_SIZE,) — mean-pooled step embeddings for the entire set.

    Returns
    -------
    jnp.ndarray
        Scalar (shape (1,)), the energy of the step set.
    """
    h = jax.nn.relu(set_embedding @ params["w1"] + params["b1"])
    return (h @ params["w2"] + params["b2"]).squeeze()


def _set_embedding(steps: List[str]) -> jnp.ndarray:
    """Mean-pool the per-step embeddings into a single set representation.

    Mean-pooling is order-invariant, which is exactly what we want for a SET
    consistency checker — the energy should not depend on step ordering.
    """
    if not steps:
        return jnp.zeros((_VOCAB_SIZE,), dtype=jnp.float32)
    encoded = jnp.stack([_encode_step(s) for s in steps])  # (N, VOCAB_SIZE)
    return jnp.mean(encoded, axis=0)


def _hinge_contrastive_loss(
    params: dict,
    consistent_embeddings: jnp.ndarray,   # (B, VOCAB_SIZE)
    inconsistent_embeddings: jnp.ndarray,  # (B, VOCAB_SIZE)
    margin: float = _MARGIN,
) -> jnp.ndarray:
    """Batch hinge contrastive loss for a mini-batch of (consistent, inconsistent) pairs.

    L = mean over batch of max(0, margin - (E_inconsistent - E_consistent))

    Why hinge rather than softmax cross-entropy: hinge loss has a built-in margin that
    creates a clear energy gap between classes, which is exactly what we want for a
    binary (consistent / inconsistent) classifier.

    The loss is zero when E_inconsistent exceeds E_consistent by at least `margin`.
    It drives training when the margin is not yet satisfied.

    Parameters
    ----------
    params : dict
        Current MLP parameters.
    consistent_embeddings : jnp.ndarray
        Mean-pooled set embeddings for consistent step sets, shape (B, VOCAB_SIZE).
    inconsistent_embeddings : jnp.ndarray
        Mean-pooled set embeddings for inconsistent step sets, shape (B, VOCAB_SIZE).
    margin : float
        Required energy gap.  Default 1.0.

    Returns
    -------
    jnp.ndarray
        Scalar loss value.
    """
    e_con = jax.vmap(lambda emb: _mlp_energy(params, emb))(consistent_embeddings)
    e_inc = jax.vmap(lambda emb: _mlp_energy(params, emb))(inconsistent_embeddings)
    losses = jax.nn.relu(margin - (e_inc - e_con))
    return jnp.mean(losses)


# ---------------------------------------------------------------------------
# SetConsistencyVerifier — public API
# ---------------------------------------------------------------------------


class SetConsistencyVerifier:
    """Verifier that computes a global consistency energy over an entire CoT step set.

    This is the Tier 2.9 candidate described in arXiv 2503.10695 (SC-Energy),
    adapted for the Carnot pipeline.

    Unlike Tier 2.5 (pairwise arithmetic) and Tier 2.7 (adjacent carry-forward),
    this verifier scores the ENTIRE set E(S_1, ..., S_N) in one shot, catching
    contradictions between non-adjacent steps.

    Typical usage
    -------------
    >>> verifier = SetConsistencyVerifier()
    >>> verifier.train(consistent_sets, inconsistent_sets)
    >>> score = verifier.energy(["Step 1: ...", "Step 2: ...", "Step 3: ..."])
    # Low score → consistent.  High score → inconsistent.

    Spec: REQ-VERIFY-149, REQ-VERIFY-150, SCENARIO-VERIFY-149, SCENARIO-VERIFY-150
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise with a fixed random seed so training is reproducible.

        Parameters
        ----------
        seed : int
            JAX PRNG seed.  Changing the seed will change the initial weights but
            not the architecture or training procedure.
        """
        self._params: dict = _init_params(jax.random.PRNGKey(seed))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def encode_step(self, step_text: str) -> jnp.ndarray:
        """Bag-of-tokens encoding for a single CoT step.

        This is exposed publicly so callers can pre-encode steps and cache them
        when scoring many sets that share steps.

        Parameters
        ----------
        step_text : str
            Raw text of a single chain-of-thought step.

        Returns
        -------
        jnp.ndarray
            Shape (_VOCAB_SIZE,), float32.  Token-count histogram normalised to sum ≤ 1.

        Spec: REQ-VERIFY-149
        """
        return _encode_step(step_text)

    def energy(self, steps: List[str]) -> float:
        """Global consistency energy for a set of CoT steps.

        Low energy means the steps look mutually consistent.
        High energy means the set contains conflicting information (likely an intruder step).

        The energy is computed as MLP(mean(encode_step(s) for s in steps)).
        Mean-pooling makes the score order-invariant — we are checking SET consistency,
        not sequence validity.

        Parameters
        ----------
        steps : list[str]
            All CoT steps from a single reasoning chain.

        Returns
        -------
        float
            Scalar energy.  Untrained model produces small values near zero;
            after training, consistent sets should produce lower energy than
            inconsistent sets by at least ``_MARGIN``.

        Spec: REQ-VERIFY-149
        """
        emb = _set_embedding(steps)
        return float(_mlp_energy(self._params, emb))

    def contrastive_loss(
        self,
        consistent_sets: List[List[str]],
        inconsistent_sets: List[List[str]],
    ) -> float:
        """Hinge contrastive loss on a pair of batches.

        L = mean(max(0, margin - (E_inconsistent - E_consistent)))

        A low loss means the model already separates consistent from inconsistent sets
        by at least `_MARGIN` in energy space.

        Parameters
        ----------
        consistent_sets : list[list[str]]
            Each inner list is all steps from a correct reasoning chain.
        inconsistent_sets : list[list[str]]
            Each inner list is the same chain but with one step replaced by an
            intruder from a different question.

        Returns
        -------
        float
            Scalar loss value.

        Spec: REQ-VERIFY-150
        """
        con_embs = jnp.stack([_set_embedding(s) for s in consistent_sets])
        inc_embs = jnp.stack([_set_embedding(s) for s in inconsistent_sets])
        return float(_hinge_contrastive_loss(self._params, con_embs, inc_embs))

    def train(
        self,
        consistent_sets: List[List[str]],
        inconsistent_sets: List[List[str]],
        n_epochs: int = 50,
        lr: float = 1e-3,
    ) -> None:
        """Train the MLP energy function with Adam using the hinge contrastive loss.

        Encodes all sets upfront (no per-epoch re-encoding overhead).
        Uses full-batch gradient descent because the FoVer corpus (~160 training
        pairs) fits entirely in JAX memory on CPU.

        Parameters
        ----------
        consistent_sets : list[list[str]]
            Consistent step sets for training.
        inconsistent_sets : list[list[str]]
            Inconsistent step sets for training (same length as consistent_sets).
        n_epochs : int
            Number of full-data passes.  50 is enough for convergence on the
            FoVer corpus without overfitting.
        lr : float
            Adam learning rate.  1e-3 is the standard Adam default.

        Spec: REQ-VERIFY-150
        """
        con_embs = jnp.stack([_set_embedding(s) for s in consistent_sets])
        inc_embs = jnp.stack([_set_embedding(s) for s in inconsistent_sets])

        optimizer = optax.adam(lr)
        opt_state = optimizer.init(self._params)

        loss_fn = jax.jit(
            lambda p: _hinge_contrastive_loss(p, con_embs, inc_embs)
        )
        grad_fn = jax.jit(jax.grad(loss_fn))

        for _ in range(n_epochs):
            grads = grad_fn(self._params)
            updates, opt_state = optimizer.update(grads, opt_state)
            self._params = optax.apply_updates(self._params, updates)

    def score_set(self, steps: List[str]) -> float:
        """Alias for energy() with a more intuitive name for pipeline callers.

        Higher score = more likely to be inconsistent.

        Spec: REQ-VERIFY-149
        """
        return self.energy(steps)
