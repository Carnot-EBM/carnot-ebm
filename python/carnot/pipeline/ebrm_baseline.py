"""EBRM baseline implementation — Energy-Based Reward Model from arXiv 2504.13134.

**Researcher summary:**
    The EBRM paper (arXiv 2504.13134, April 2025) introduces an energy-based reward model
    that treats noisy reward signals as a distribution over (response, reward) pairs.
    Instead of a discriminative classifier that outputs a single score, EBRM defines an
    energy function E(response, reward) and trains it so that correct responses with high
    rewards have low energy, while incorrect responses with low rewards have high energy.

    This module implements EBRM as a comparison baseline against Carnot's EORM (Energy-based
    Outcome/step-level Reward Model).  Key architectural difference:
    - EBRM (this file): operates at the response level — one energy score per whole response.
    - EORM (eorm_model.py): operates at the step level — one energy score per reasoning step.

    If EORM outperforms EBRM on FoVer step-level tasks, that is publishable evidence that
    step-level granularity is the right architectural choice for Carnot.

**Training approach (margin loss):**
    For each training pair (text, label):
    - label=1 (correct): treat reward=1.0 — we want low energy for (text, 1.0).
    - label=0 (incorrect): treat reward=0.0 — we want high energy for (text, 0.0).
    Margin loss = max(0, margin - (E_negative - E_positive)).
    This pushes E(incorrect, 0.0) > E(correct, 1.0) by at least `margin`.

REQ-EBRM-001, REQ-EBRM-002
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EBRMEnergy:
    """2-layer MLP energy function over (response_features, reward_scalar) pairs.

    The energy E(x, r) is a scalar that measures how "unlikely" it is to observe
    response x paired with reward r.  Low energy = likely/correct; high energy = unlikely/wrong.

    Architecture:
        Input: TF-IDF embedding (dim=feature_dim) concatenated with reward scalar (dim=1)
               => total input dim = feature_dim + 1
        Hidden layer: Linear(feature_dim+1, hidden_dim) -> ReLU
        Output layer: Linear(hidden_dim, 1) -> scalar energy

    Parameters are stored as numpy arrays for zero-dependency training (no JAX/PyTorch needed
    for this small baseline model; the comparison validity does not require GPU power).

    REQ-EBRM-001: energy() MUST accept concatenated [response_features, reward_scalar].
    """

    def __init__(self, feature_dim: int = 128, hidden_dim: int = 64) -> None:
        """Initialise the energy MLP with small random weights.

        Args:
            feature_dim: Dimensionality of the TF-IDF response embedding.
            hidden_dim:  Number of hidden units in the single hidden layer.
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        input_dim = feature_dim + 1  # +1 for the scalar reward

        rng = np.random.default_rng(42)
        # Xavier-style initialisation: scale = sqrt(2 / fan_in)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

        # TF-IDF vectoriser — fitted during training, reused at inference.
        self.vectorizer: TfidfVectorizer | None = None

    def energy(self, response_features: np.ndarray, reward_scalar: float) -> float:
        """Compute the scalar energy E(response_features, reward_scalar).

        Low energy means the model considers this (response, reward) pair likely.
        High energy means unlikely / inconsistent.

        Args:
            response_features: 1-D float array of length feature_dim (TF-IDF embedding).
            reward_scalar:      Scalar reward in [0, 1].  1.0 = correct; 0.0 = incorrect.

        Returns:
            Scalar energy value (float).

        Why concatenation: EBRM (arXiv 2504.13134 §3) conditions the energy on both the
        response and the reward together.  Concatenation is the simplest joint representation
        that lets the MLP learn interactions between text quality and reward signal.
        """
        x = np.concatenate([response_features, [reward_scalar]], axis=0).astype(np.float32)
        h = np.maximum(0.0, x @ self.W1 + self.b1)  # ReLU hidden layer
        e = (h @ self.W2 + self.b2)[0]               # scalar output
        return float(e)

    def log_prob(self, response_features: np.ndarray, reward_scalar: float) -> float:
        """Return unnormalized log-probability = -energy(response_features, reward_scalar).

        Higher log_prob means the model considers this (response, reward) pair more likely.
        This is the standard EBM convention: p(x) ∝ exp(-E(x)).
        """
        return -self.energy(response_features, reward_scalar)

    def score(self, response_text: str) -> float:
        """Score a response text: return log_prob under reward=1.0 (correct hypothesis).

        This implements the EBRM scoring convention: a high score means the model
        assigns high probability to (response, correct) — i.e., the response looks correct.

        Args:
            response_text: Raw text of the reasoning step or response to score.

        Returns:
            Unnormalized log-probability (float).  Higher = model thinks this is correct.
        """
        if self.vectorizer is None:
            raise RuntimeError("EBRMEnergy.score() called before training. Call EBRMTrainer.train() first.")
        features = self.vectorizer.transform([response_text]).toarray()[0].astype(np.float32)
        # Truncate or pad to feature_dim (TF-IDF vocab may differ from feature_dim)
        features = _resize_vector(features, self.feature_dim)
        return self.log_prob(features, 1.0)


class EBRMTrainer:
    """Train EBRMEnergy using margin loss on (text, label) pairs.

    Margin loss (per batch):
        E_pos = energy(correct_text, reward=1.0)
        E_neg = energy(incorrect_text, reward=0.0)
        loss  = mean(max(0, margin - (E_neg - E_pos)))

    The loss pushes incorrect examples to higher energy than correct ones by at
    least `margin`.  This is analogous to a contrastive energy objective.

    REQ-EBRM-001: Train with NLL-equivalent margin loss on FoVer v2 split.
    """

    def __init__(self, model: EBRMEnergy, margin: float = 1.0) -> None:
        """Initialise trainer.

        Args:
            model:  EBRMEnergy instance to train (modified in-place).
            margin: Minimum energy gap between negative and positive samples.
                    Margin=1.0 follows the contrastive EBM convention.
        """
        self.model = model
        self.margin = margin

    def train(
        self,
        step_texts: list[str],
        labels: list[int],
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> None:
        """Train the energy model via gradient descent with margin loss.

        Args:
            step_texts: List of reasoning step texts (one per sample).
            labels:     Parallel list of labels: 1=correct, 0=incorrect.
            n_epochs:   Number of full passes over the training data.
            lr:         SGD learning rate (Adam-style not used to keep this
                        dependency-free; plain SGD is sufficient for the baseline).

        Why TF-IDF: EBRM in the paper uses a pretrained LM encoder.  For a fair
        apples-to-apples baseline against EORM (which uses TF-IDF), we use the
        same TF-IDF representation.  This isolates the architectural difference
        (response-level vs step-level energy) from the representation difference.
        """
        # Fit TF-IDF on all texts (same preprocessing as EORM uses).
        self.model.vectorizer = TfidfVectorizer(max_features=self.model.feature_dim)
        X_raw = self.model.vectorizer.fit_transform(step_texts).toarray().astype(np.float32)
        # Resize each row to feature_dim (vocab may be < max_features for small corpora).
        X = np.stack([_resize_vector(row, self.model.feature_dim) for row in X_raw])
        y = np.array(labels, dtype=np.int32)

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            # Degenerate dataset — no gradient signal; skip training.
            return

        rng = np.random.default_rng(0)

        for _ in range(n_epochs):
            # Sample pairs (pos, neg) for contrastive training.
            n_pairs = min(len(pos_idx), len(neg_idx))
            pos_sample = rng.choice(pos_idx, size=n_pairs, replace=False)
            neg_sample = rng.choice(neg_idx, size=n_pairs, replace=False)

            for pi, ni in zip(pos_sample, neg_sample):
                x_pos = X[pi]
                x_neg = X[ni]

                e_pos = self.model.energy(x_pos, 1.0)
                e_neg = self.model.energy(x_neg, 0.0)

                gap = e_neg - e_pos
                if gap >= self.margin:
                    continue  # Constraint satisfied; no gradient needed.

                # Compute gradients via finite differences (keeps this dependency-free).
                # d(loss)/d(E_pos) = +1  (we want E_pos to decrease → raise gap)
                # d(loss)/d(E_neg) = -1  (we want E_neg to increase → raise gap)
                self._gradient_step(x_pos, 1.0, direction=+1.0, lr=lr)
                self._gradient_step(x_neg, 0.0, direction=-1.0, lr=lr)

    def _gradient_step(
        self,
        features: np.ndarray,
        reward: float,
        direction: float,
        lr: float,
    ) -> None:
        """Apply one gradient step to minimise (direction * energy(features, reward)).

        direction=+1 → minimise energy (push energy down for this sample).
        direction=-1 → maximise energy (push energy up for this sample).

        Gradients are computed analytically through the 2-layer MLP.
        """
        x = np.concatenate([features, [reward]], axis=0).astype(np.float32)
        h_pre = x @ self.model.W1 + self.model.b1   # (hidden_dim,)
        h = np.maximum(0.0, h_pre)                   # ReLU
        # e = h @ W2 + b2  (scalar)

        # Backprop: d(loss) = direction * 1.0
        d_e = direction * 1.0

        # Gradients for W2, b2
        dW2 = np.outer(h, [d_e])          # (hidden_dim, 1)
        db2 = np.array([d_e])

        # Backprop through ReLU
        d_h = (self.model.W2 @ [d_e])     # (hidden_dim,) — chain rule
        d_h_pre = d_h * (h_pre > 0).astype(np.float32)  # ReLU mask

        # Gradients for W1, b1
        dW1 = np.outer(x, d_h_pre)        # (input_dim, hidden_dim)
        db1 = d_h_pre

        # SGD update (direction already baked into d_e sign)
        self.model.W1 -= lr * dW1
        self.model.b1 -= lr * db1
        self.model.W2 -= lr * dW2
        self.model.b2 -= lr * db2

    def predict(self, step_text: str) -> float:
        """Return sigmoid-scaled score for AUC computation.

        Higher score = model thinks this step is correct.
        Sigmoid maps the unbounded log_prob into [0, 1] for sklearn's roc_auc_score.
        """
        raw = self.model.score(step_text)
        return float(1.0 / (1.0 + np.exp(-raw)))  # sigmoid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resize_vector(v: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize 1-D vector to target_dim by truncating or zero-padding.

    Why: TF-IDF max_features may not be reached when the corpus is small (fewer
    unique tokens than max_features).  This ensures the energy MLP always receives
    a fixed-length input regardless of corpus size.
    """
    current = v.shape[0]
    if current >= target_dim:
        return v[:target_dim]
    pad = np.zeros(target_dim - current, dtype=v.dtype)
    return np.concatenate([v, pad])
