"""SC-Energy set-level consistency model — JAX implementation.

**Researcher summary:**
    Implements the SC-Energy architecture from arXiv 2503.10695. Instead of scoring
    individual statements, this model assigns a scalar energy to an entire *set*
    of statements (e.g., the steps in a reasoning chain). Coherent sets get low energy;
    contradictory sets get high energy. Trained with a contrastive loss.

**Detailed explanation for engineers:**
    Traditional EBMs score individual data points (vectors). SC-Energy extends that to
    variable-size *sets* of natural-language statements. The challenge: the set may
    contain different numbers of statements, and the energy must be
    *permutation-invariant* (the order of the statements must not matter).

    **Architecture pipeline:**
        1. Embed each statement into a fixed-size vector using TF-IDF (a
           bag-of-words approach that counts how distinctive each word is
           across the corpus). Each statement becomes a sparse vector over the
           vocabulary.
        2. Mean-pool all statement vectors into a single fixed-size vector.
           Mean-pooling is permutation-invariant: shuffling the statements
           does not change the average.
        3. Pass the pooled vector through a 2-layer MLP that outputs a scalar.
           This is the energy E(set).

    **Training objective (contrastive loss):**
        For each pair (coherent_set, contradictory_set):
            loss = max(0, margin - (E(contradictory) - E(coherent)))
        This pushes E(coherent) << E(contradictory), meaning coherent sets
        have lower energy (are more "natural") than contradictory sets.
        The margin hyperparameter controls how large the energy gap must be.

    **Why TF-IDF instead of sentence transformers?**
        Sentence transformers (e.g., BGE-small) would produce better embeddings
        but require a ~100 MB model download and GPU memory. TF-IDF is fully
        local, requires no downloads, has no GPU dependency, and runs in
        milliseconds. It is adequate for detecting cross-problem contradictions
        because the vocabulary overlap between two different GSM8K problems is
        very low — TF-IDF naturally captures this distributional mismatch.

    **What is a GSM8K contradiction?**
        GSM8K is a dataset of grade-school math word problems with step-by-step
        solutions. A "coherent set" is 3-5 consecutive solution steps from the
        SAME problem (same quantities, same logical thread). A "contradictory set"
        mixes steps from TWO different problems — they reference different
        quantities and different logical contexts, creating implicit contradictions.

Spec: REQ-MODEL-031, SCENARIO-MODEL-016
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


# ---------------------------------------------------------------------------
# TF-IDF embedding (pure numpy — no heavy dependencies)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphabetic tokens.

    **Why so simple?**
    For cross-problem contradiction detection we only need to distinguish
    vocabulary from Problem A vs Problem B. Simple word-level tokenization
    captures that difference without requiring a tokenizer dependency.
    """
    import re

    return re.findall(r"[a-zA-Z]+", text.lower())


class TFIDFEmbedder:
    """Fit a TF-IDF vocabulary on a corpus and embed statements.

    **What is TF-IDF?**
        TF = Term Frequency: how often a word appears in THIS statement.
        IDF = Inverse Document Frequency: log(N / df) where N is the corpus
        size and df is how many statements contain the word. Rare words get
        high IDF weight; common words like "the" get near-zero weight.
        TF-IDF = TF * IDF. Words that are distinctive to a particular
        statement get high scores; stop-words get suppressed.

    This is a standard sklearn-style interface but implemented without sklearn
    to avoid adding a hard dependency. The vocabulary is capped at `max_features`
    most-frequent terms to keep the embedding dimension manageable.

    Spec: REQ-MODEL-031
    """

    def __init__(self, max_features: int = 512) -> None:
        """Create an unfitted TFIDFEmbedder.

        Args:
            max_features: Maximum vocabulary size. Larger = richer embeddings
                but higher memory and compute cost. 512 is sufficient for the
                cross-problem discrimination task.
        """
        self.max_features = max_features
        self.vocab_: dict[str, int] = {}
        self.idf_: np.ndarray | None = None

    def fit(self, statements: list[str]) -> "TFIDFEmbedder":
        """Learn vocabulary and IDF weights from a corpus of statements.

        Args:
            statements: All statements in the training corpus.

        Returns:
            self (fluent interface: allows embedder.fit(stmts).transform(x))
        """
        # Count document frequency for each term
        df: dict[str, int] = {}
        token_lists = [_tokenize(s) for s in statements]
        for tokens in token_lists:
            for word in set(tokens):  # set() avoids counting a word twice per doc
                df[word] = df.get(word, 0) + 1

        # Keep only the top max_features most common terms
        sorted_terms = sorted(df.keys(), key=lambda w: -df[w])[: self.max_features]
        self.vocab_ = {w: i for i, w in enumerate(sorted_terms)}

        # Compute IDF: log((N + 1) / (df + 1)) + 1 (sklearn smoothing)
        # Adding 1 to numerator and denominator prevents division by zero
        # and avoids zero IDF for universal terms.
        n = len(statements)
        # Always allocate max_features entries so the output vector has a
        # fixed, predictable dimension regardless of actual vocab size.
        idf = np.zeros(self.max_features, dtype=np.float32)
        for word, idx in self.vocab_.items():
            idf[idx] = np.log((n + 1) / (df[word] + 1)) + 1.0
        self.idf_ = idf
        return self

    def transform(self, statement: str) -> np.ndarray:
        """Embed a single statement as a TF-IDF vector.

        Args:
            statement: A single natural-language statement string.

        Returns:
            A float32 numpy array of shape (max_features,). L2-normalized
            so that the embedding magnitude is 1.0 regardless of statement
            length.

        Spec: SCENARIO-MODEL-016
        """
        if self.idf_ is None:
            raise RuntimeError("TFIDFEmbedder must be fitted before transform()")

        tokens = _tokenize(statement)
        # Allocate exactly max_features so the output shape is always predictable
        tf = np.zeros(self.max_features, dtype=np.float32)
        for word in tokens:
            if word in self.vocab_:
                tf[self.vocab_[word]] += 1.0

        vec = tf * self.idf_
        # L2-normalize to unit sphere so statement length doesn't dominate
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec


# ---------------------------------------------------------------------------
# SC-Energy MLP model (JAX)
# ---------------------------------------------------------------------------


@dataclass
class SCEnergyConfig:
    """Configuration for the SC-Energy model.

    Attributes:
        embed_dim: Dimension of the TF-IDF embedding vector (= TFIDFEmbedder.max_features).
        hidden_dim: Number of hidden units in the 2-layer MLP.
        margin: Contrastive margin. The model learns to push
            E(contradictory) - E(coherent) > margin.
        learning_rate: Step size for gradient descent during training.

    Spec: REQ-MODEL-031
    """

    embed_dim: int = 512
    hidden_dim: int = 64
    margin: float = 1.0
    learning_rate: float = 0.01


class SCEnergyModel:
    """Set-level Energy-Based Model for statement consistency scoring.

    **Researcher summary:**
        Implements SC-Energy (arXiv 2503.10695). Input: list of statements.
        Output: scalar energy (low = coherent, high = contradictory).

    **Detailed explanation for engineers:**
        This model takes a variable-length list of natural-language statements
        and returns a single scalar energy value. It works as follows:

        1. Each statement is embedded with a pre-fitted TFIDFEmbedder into
           a vector of shape (embed_dim,).
        2. All statement vectors are mean-pooled into a single (embed_dim,) vector.
           Mean pooling is permutation-invariant: the model sees the same
           aggregate regardless of statement ordering.
        3. The pooled vector passes through a 2-layer MLP:
               h = tanh(W1 @ pooled + b1)   # shape: (hidden_dim,)
               energy = W2 @ h + b2          # shape: scalar
           tanh keeps activations bounded, preventing energy explosion during training.

        **Training (contrastive loss):**
        You provide pairs (coherent_set, contradictory_set). The loss is:
            L = max(0, margin - (E(contradictory) - E(coherent)))
        If E(contradictory) > E(coherent) + margin, the pair is already well-separated
        and contributes 0 to the loss. Otherwise, the gradient pushes the gap wider.

    Spec: REQ-MODEL-031, SCENARIO-MODEL-016
    """

    def __init__(self, config: SCEnergyConfig, key: jax.Array | None = None) -> None:
        """Initialize the SC-Energy model with random weights.

        Args:
            config: SCEnergyConfig specifying architecture and training hyperparameters.
            key: JAX PRNG key. If None, uses seed 0.

        Spec: REQ-MODEL-031
        """
        self.config = config
        if key is None:
            key = jrandom.PRNGKey(0)

        k1, k2, k3, k4 = jrandom.split(key, 4)

        # Xavier-uniform initialization for both MLP layers.
        # Xavier init keeps the variance of pre-activations constant across layers,
        # preventing vanishing or exploding gradients in deeper networks.
        d, h = config.embed_dim, config.hidden_dim
        lim1 = jnp.sqrt(6.0 / (d + h))
        lim2 = jnp.sqrt(6.0 / (h + 1))

        # Layer 1: (embed_dim -> hidden_dim)
        self.W1 = jrandom.uniform(k1, (h, d), minval=-lim1, maxval=lim1)
        self.b1 = jnp.zeros(h)
        # Layer 2: (hidden_dim -> 1 scalar)
        self.W2 = jrandom.uniform(k2, (h,), minval=-lim2, maxval=lim2)
        self.b2 = jnp.zeros(())

        # Store embedder reference (must be fitted before calling energy())
        self.embedder: TFIDFEmbedder | None = None

    def _mlp_energy(
        self,
        pooled: jax.Array,
        W1: jax.Array,
        b1: jax.Array,
        W2: jax.Array,
        b2: jax.Array,
    ) -> jax.Array:
        """Compute MLP energy from a pooled embedding vector.

        **Why a separate function taking explicit params?**
        JAX's gradient functions (jax.grad) differentiate with respect to
        Python function arguments, not object attributes. By passing W1/b1/W2/b2
        explicitly, we can call jax.grad(..., argnums=(1,2,3,4)) to get gradients
        with respect to all four weight arrays in one call.

        Args:
            pooled: Mean-pooled embedding, shape (embed_dim,).
            W1, b1, W2, b2: MLP parameters.

        Returns:
            Scalar energy value.

        Spec: REQ-MODEL-031
        """
        h = jnp.tanh(W1 @ pooled + b1)  # (hidden_dim,)
        return jnp.dot(W2, h) + b2  # scalar

    def _embed_set(self, statements: Sequence[str]) -> jax.Array:
        """Embed a list of statements and mean-pool into a single vector.

        Args:
            statements: List of natural-language statement strings.

        Returns:
            JAX float32 array of shape (embed_dim,).
        """
        if self.embedder is None:
            raise RuntimeError("SCEnergyModel.embedder must be set before embedding")
        vecs = np.stack([self.embedder.transform(s) for s in statements], axis=0)
        return jnp.array(vecs.mean(axis=0))

    def energy(self, statements: Sequence[str]) -> float:
        """Compute the set-level energy for a list of statements.

        Lower energy = more coherent set.
        Higher energy = more contradictory set.

        Args:
            statements: List of natural-language statements (reasoning chain steps).

        Returns:
            Scalar float energy value.

        Spec: REQ-MODEL-031, SCENARIO-MODEL-016
        """
        pooled = self._embed_set(statements)
        return float(self._mlp_energy(pooled, self.W1, self.b1, self.W2, self.b2))

    def train(
        self,
        coherent_sets: list[list[str]],
        contradictory_sets: list[list[str]],
        n_epochs: int = 50,
    ) -> list[float]:
        """Train with contrastive loss on (coherent, contradictory) pairs.

        **How training works:**
            For each epoch we iterate over all pairs and compute the gradient
            of the contrastive loss with respect to W1, b1, W2, b2. We then
            subtract lr * gradient from each parameter (gradient descent).

        Args:
            coherent_sets: List of sets; each is a list of statements from the
                same coherent reasoning chain.
            contradictory_sets: List of sets; each mixes steps from different
                reasoning chains (guaranteed contradiction).
            n_epochs: Number of full passes through the training data.

        Returns:
            List of per-epoch mean loss values (for diagnostics).

        Spec: REQ-MODEL-031
        """
        if len(coherent_sets) != len(contradictory_sets):
            raise ValueError("coherent_sets and contradictory_sets must be same length")

        margin = self.config.margin
        lr = self.config.learning_rate

        def contrastive_loss(
            W1: jax.Array,
            b1: jax.Array,
            W2: jax.Array,
            b2: jax.Array,
            pooled_coh: jax.Array,
            pooled_con: jax.Array,
        ) -> jax.Array:
            """Contrastive loss for one (coherent, contradictory) pair.

            loss = max(0, margin - (E_contra - E_coh))
            If E_contra already exceeds E_coh by more than margin, loss = 0.
            """
            e_coh = self._mlp_energy(pooled_coh, W1, b1, W2, b2)
            e_con = self._mlp_energy(pooled_con, W1, b1, W2, b2)
            return jnp.maximum(0.0, margin - (e_con - e_coh))

        # Pre-compute all pooled embeddings (done outside the training loop
        # so we don't re-tokenize on every epoch)
        pooled_cohs = [self._embed_set(s) for s in coherent_sets]
        pooled_cons = [self._embed_set(s) for s in contradictory_sets]

        grad_fn = jax.grad(contrastive_loss, argnums=(0, 1, 2, 3))

        loss_history = []
        for _ in range(n_epochs):
            epoch_loss = 0.0
            for pc, pn in zip(pooled_cohs, pooled_cons):
                loss_val = float(contrastive_loss(self.W1, self.b1, self.W2, self.b2, pc, pn))
                epoch_loss += loss_val
                if loss_val > 0.0:
                    # Only compute and apply gradients when the loss is non-zero
                    # (hinge loss: gradient is zero when pair is already separated)
                    gW1, gb1, gW2, gb2 = grad_fn(self.W1, self.b1, self.W2, self.b2, pc, pn)
                    self.W1 = self.W1 - lr * gW1
                    self.b1 = self.b1 - lr * gb1
                    self.W2 = self.W2 - lr * gW2
                    self.b2 = self.b2 - lr * gb2
            loss_history.append(epoch_loss / max(len(pooled_cohs), 1))

        return loss_history

    def predict_coherent_score(self, statements: Sequence[str]) -> float:
        """Return a coherence score in [0, 1]: higher = more coherent.

        This inverts and normalizes the raw energy into a probability-like score
        using a sigmoid: score = sigmoid(-energy) = 1 / (1 + exp(energy)).

        Args:
            statements: List of natural-language statement strings.

        Returns:
            Float in [0, 1]. Higher means more coherent.

        Spec: SCENARIO-MODEL-016
        """
        e = self.energy(statements)
        return float(1.0 / (1.0 + np.exp(e)))
