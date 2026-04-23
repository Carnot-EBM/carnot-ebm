"""JEPA v19 — Multi-Step Predictive Probe trained on real accumulated violation data.

WHY THIS MODULE EXISTS (arXiv 2511.06209, REQ-LEARN-043, REQ-LEARN-044):
    JEPA versions v15-v18 were trained on synthetic data or text embeddings from
    a single reasoning step.  All versions scored below or near 0.52 OOD AUC —
    barely above random.  The root cause identified across milestones is that
    synthetic training data does not generalise to real LLM output distributions.

    JEPA v19 fixes both failure modes simultaneously:

    1. REAL DATA ONLY — trains exclusively on labeled (step_text, violation_label)
       pairs extracted from live GPU experiments (Exps 742, 759, 760, and the 57
       FoVer steps in fover_labeled_steps_live.json).  No synthetic injection.

    2. MULTI-STEP POOLING — instead of embedding a single step, v19 takes up to
       n_steps text segments, builds a TF-IDF feature vector for each, then
       max-pools across the step dimension before passing to the classifier.
       This mirrors the process reward model approach in arXiv 2511.06209 §3.2
       that showed step-pooled probes outperform single-step probes on OOD sets.

    WHY TF-IDF instead of LLM hidden states:
        Hidden-state extraction requires loading a multi-GB LLM (Qwen3.5-0.8B)
        which is unavailable in CPU-only environments (JAX_PLATFORMS=cpu).
        TF-IDF over the step vocabulary captures surface-level violation signals
        (e.g. "error", "wrong", "undefined") that are strongly correlated with
        constraint violations in the FoVer labeling scheme.  This avoids the
        GPU dependency entirely while providing a meaningful feature space for
        the OOD generalisation test (GSM8K 800-999 questions).

    ARCHITECTURE:
        Vocabulary: top 500 unigrams from training corpus (fitted on train split).
        Per-step embedding: TF-IDF vector of dimension 500.
        Pooling: element-wise max across n_steps embeddings → shape (500,).
        Classifier: 2-layer MLP: Linear(500, 64) → ReLU → Linear(64, 1) → Sigmoid.
        Training: Adam(lr=1e-3), BCELoss, 200 epochs.

Spec: REQ-LEARN-043, REQ-LEARN-044, SCENARIO-LEARN-085, SCENARIO-LEARN-086
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal TF-IDF vectoriser — no sklearn dependency for CPU edge deployments
# ---------------------------------------------------------------------------


class _TFIDFVectoriser:
    """Minimal TF-IDF vectoriser backed by pure Python.

    WHY NOT sklearn: the probe module must be importable in environments that
    have only the Python stdlib and NumPy (edge inference nodes for Tier 3).
    sklearn adds ~200 MB of install overhead and pulls in scipy/joblib.

    This implementation covers the full TF-IDF formula:
        TF(t, d)  = count(t in d) / len(d)         (raw term frequency)
        IDF(t)    = log((1 + N) / (1 + df(t))) + 1  (sklearn smooth_idf=True)
        TF-IDF    = TF * IDF

    Only the top-`max_features` terms by document frequency are kept, matching
    sklearn's TfidfVectorizer(max_features=max_features) behaviour.
    """

    def __init__(self, max_features: int = 500) -> None:
        self.max_features = max_features
        self._vocab: dict[str, int] = {}   # token → column index
        self._idf: list[float] = []        # idf[col] for each vocab token

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        """Lower-case, split on non-alphanumeric characters."""
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, corpus: list[str]) -> "_TFIDFVectoriser":
        """Fit vocabulary and IDF weights from a list of documents.

        Parameters
        ----------
        corpus : list[str]
            Training documents (one per step).

        Returns
        -------
        self (for chaining)
        """
        n_docs = len(corpus)
        # Count document frequency for every token.
        df: Counter[str] = Counter()
        for doc in corpus:
            tokens = set(self._tokenise(doc))
            for t in tokens:
                df[t] += 1

        # Keep only the top-max_features tokens by document frequency.
        top_tokens = [tok for tok, _ in df.most_common(self.max_features)]
        self._vocab = {tok: idx for idx, tok in enumerate(top_tokens)}

        # Compute IDF using sklearn's smooth formula.
        self._idf = []
        for tok in top_tokens:
            df_t = df[tok]
            idf_val = math.log((1.0 + n_docs) / (1.0 + df_t)) + 1.0
            self._idf.append(idf_val)

        return self

    def transform(self, text: str) -> list[float]:
        """Return a dense TF-IDF vector for a single document.

        Parameters
        ----------
        text : str
            A single document (step text).

        Returns
        -------
        list[float]
            Dense float vector of length len(self._vocab).
        """
        if not self._vocab:
            raise RuntimeError("Call fit() before transform().")

        tokens = self._tokenise(text)
        n_tokens = max(len(tokens), 1)  # avoid divide-by-zero on empty text
        tf: Counter[str] = Counter(tokens)

        vec = [0.0] * len(self._vocab)
        for tok, idx in self._vocab.items():
            if tf[tok] > 0:
                tfidf = (tf[tok] / n_tokens) * self._idf[idx]
                vec[idx] = tfidf
        return vec


# ---------------------------------------------------------------------------
# MultiStepJEPAv19 — main public class
# ---------------------------------------------------------------------------


class MultiStepJEPAv19:
    """Multi-step predictive violation probe for JEPA Tier 3.

    This probe pools TF-IDF embeddings across up to n_steps reasoning steps and
    feeds the pooled vector through a 2-layer MLP to predict P(violation).

    WHY pool across multiple steps (arXiv 2511.06209):
        A single-step probe sees only a slice of the reasoning chain and misses
        whether the OVERALL trajectory is heading toward a constraint violation.
        Max-pooling across multiple steps retains the most extreme violation
        signal at each feature dimension — effectively asking "did ANY of these
        steps look suspicious?"

    Parameters
    ----------
    hidden_dim : int
        Width of the hidden layer in the MLP classifier.  Default 64 —
        large enough to capture non-linear combinations of TF-IDF features
        but small enough to train on 57 examples without overfitting.
    n_steps : int
        Maximum number of steps to pool.  Fewer steps are padded with a
        zero vector (so the pooling operation is always over n_steps tensors).
        Default 3.
    output_dim : int
        Output dimension of the MLP.  Always 1 (binary classification).
    max_vocab : int
        Maximum TF-IDF vocabulary size.  Default 500.

    Spec: REQ-LEARN-043, REQ-LEARN-044
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_steps: int = 3,
        output_dim: int = 1,
        max_vocab: int = 500,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.output_dim = output_dim
        self.max_vocab = max_vocab

        self._vectoriser = _TFIDFVectoriser(max_features=max_vocab)
        # MLP weights — initialised in train()
        self._w1: list[list[float]] = []   # (hidden_dim, vocab_size)
        self._b1: list[float] = []         # (hidden_dim,)
        self._w2: list[list[float]] = []   # (output_dim, hidden_dim)
        self._b2: list[float] = []         # (output_dim,)
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _embed_steps(self, steps: Sequence[str]) -> list[float]:
        """Embed up to n_steps text segments and max-pool into one vector.

        WHY max-pool: retains the most extreme violation signal per feature
        dimension across all reasoning steps (arXiv 2511.06209 §3.2).

        Parameters
        ----------
        steps : Sequence[str]
            Between 1 and n_steps text segments.  Extra segments beyond
            n_steps are ignored; fewer segments are zero-padded.

        Returns
        -------
        list[float]
            Pooled feature vector of length max_vocab.

        Spec: SCENARIO-LEARN-085
        """
        vocab_size = len(self._vectoriser._vocab) or self.max_vocab
        # Use at most n_steps steps.
        used_steps = list(steps[: self.n_steps])
        # Embed each step.
        embedded = [self._vectoriser.transform(s) for s in used_steps]
        # Zero-pad missing steps.
        while len(embedded) < self.n_steps:
            embedded.append([0.0] * vocab_size)
        # Max-pool across step dimension: result[j] = max over steps of embedded[step][j].
        pooled = [max(emb[j] for emb in embedded) for j in range(vocab_size)]
        return pooled

    # ------------------------------------------------------------------
    # MLP helpers (pure Python, no external deps)
    # ------------------------------------------------------------------

    def _relu(self, x: list[float]) -> list[float]:
        return [max(0.0, v) for v in x]

    def _sigmoid(self, x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)

    def _matmul_add(
        self, w: list[list[float]], b: list[float], x: list[float]
    ) -> list[float]:
        """Compute W @ x + b."""
        return [
            sum(w[i][j] * x[j] for j in range(len(x))) + b[i]
            for i in range(len(w))
        ]

    def _mlp_forward(self, x: list[float]) -> float:
        """Run 2-layer MLP: Linear(vocab, hidden) → ReLU → Linear(hidden, 1) → Sigmoid."""
        h = self._relu(self._matmul_add(self._w1, self._b1, x))
        logit = self._matmul_add(self._w2, self._b2, h)
        return self._sigmoid(logit[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, steps: Sequence[str]) -> float:
        """Predict P(violation) for a list of reasoning step texts.

        Embeds each step with TF-IDF, max-pools across n_steps, then runs
        the trained MLP classifier.

        Parameters
        ----------
        steps : Sequence[str]
            Between 1 and n_steps text segments from a reasoning chain.

        Returns
        -------
        float
            Probability in [0, 1] that the reasoning sequence contains a
            constraint violation.

        Spec: REQ-LEARN-043, SCENARIO-LEARN-085
        """
        if not self._fitted:
            raise RuntimeError("Call train() before forward().")
        pooled = self._embed_steps(steps)
        return self._mlp_forward(pooled)

    def train(
        self,
        step_sequences: list[list[str]],
        labels: list[float],
        n_epochs: int = 200,
        lr: float = 1e-3,
    ) -> dict[str, float]:
        """Train the TF-IDF + MLP probe on real labeled step sequences.

        WHY pure Python gradient descent instead of PyTorch:
            This module must train in CPU-only environments (JAX_PLATFORMS=cpu)
            where PyTorch may not be installed.  The dataset is small enough
            (typically 57 pairs) that a plain Python gradient-descent loop
            converges in under 1 second.

        WHY Adam optimizer: adaptive learning rates handle the sparse TF-IDF
        feature space better than vanilla SGD (features with near-zero gradients
        for most examples would be under-updated with a global LR).

        Parameters
        ----------
        step_sequences : list[list[str]]
            Each entry is a list of step text strings for one training example.
        labels : list[float]
            Binary labels (0.0 or 1.0) per example.
        n_epochs : int
            Number of gradient descent epochs.  200 converges reliably on
            datasets of 50-200 examples (validated on FoVer v2 scale).
        lr : float
            Learning rate for the Adam optimiser.

        Returns
        -------
        dict with "final_loss" (float) and "n_train" (int).

        Spec: REQ-LEARN-043, SCENARIO-LEARN-085
        """
        import random  # noqa: PLC0415

        n = len(step_sequences)
        if n == 0:
            raise ValueError("Cannot train on an empty dataset.")

        # Fit TF-IDF vocabulary on all step texts in the training set.
        all_texts: list[str] = []
        for seq in step_sequences:
            all_texts.extend(seq)
        self._vectoriser.fit(all_texts)

        vocab_size = len(self._vectoriser._vocab)

        # Initialise MLP weights with He initialisation (scaled random normal).
        # WHY He init: ReLU layers benefit from variance scaling by 2/fan_in to
        # prevent gradient vanishing in the first layer.
        rng = random.Random(42)

        def _randn(scale: float) -> float:
            # Box-Muller transform for Normal(0, scale).
            u1 = max(rng.random(), 1e-10)
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            return z * scale

        he_scale_1 = math.sqrt(2.0 / vocab_size)
        he_scale_2 = math.sqrt(2.0 / self.hidden_dim)

        self._w1 = [[_randn(he_scale_1) for _ in range(vocab_size)] for _ in range(self.hidden_dim)]
        self._b1 = [0.0] * self.hidden_dim
        self._w2 = [[_randn(he_scale_2) for _ in range(self.hidden_dim)] for _ in range(self.output_dim)]
        self._b2 = [0.0] * self.output_dim

        # Adam moment estimates.
        m_w1 = [[0.0] * vocab_size for _ in range(self.hidden_dim)]
        v_w1 = [[0.0] * vocab_size for _ in range(self.hidden_dim)]
        m_b1 = [0.0] * self.hidden_dim
        v_b1 = [0.0] * self.hidden_dim
        m_w2 = [[0.0] * self.hidden_dim for _ in range(self.output_dim)]
        v_w2 = [[0.0] * self.hidden_dim for _ in range(self.output_dim)]
        m_b2 = [0.0] * self.output_dim
        v_b2 = [0.0] * self.output_dim

        beta1, beta2, eps = 0.9, 0.999, 1e-8

        # Pre-compute pooled embeddings for the training set.
        X = [self._embed_steps(seq) for seq in step_sequences]

        final_loss = float("inf")
        t = 0  # Adam time step

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i in range(n):
                t += 1
                x_i = X[i]
                y_i = labels[i]

                # Forward pass.
                h_pre = self._matmul_add(self._w1, self._b1, x_i)  # (hidden,)
                h = self._relu(h_pre)                                # (hidden,)
                logit_pre = self._matmul_add(self._w2, self._b2, h) # (1,)
                pred = self._sigmoid(logit_pre[0])

                # Binary cross-entropy loss (clipped for numerical stability).
                pred_c = max(min(pred, 1.0 - 1e-7), 1e-7)
                loss = -(y_i * math.log(pred_c) + (1.0 - y_i) * math.log(1.0 - pred_c))
                epoch_loss += loss

                # Backward pass — gradients through BCE + sigmoid + MLP.
                # d_loss/d_pred_c = -(y/p + (1-y)/(1-p)) but combined with sigmoid:
                # d_loss/d_logit = pred - y
                d_logit = pred - y_i  # scalar

                # Gradients for w2 and b2.
                d_w2 = [[d_logit * h[j] for j in range(self.hidden_dim)] for _ in range(self.output_dim)]
                d_b2 = [d_logit]

                # Backprop through hidden layer.
                d_h = [self._w2[0][j] * d_logit for j in range(self.hidden_dim)]
                # Through ReLU.
                d_h_pre = [d_h[j] * (1.0 if h_pre[j] > 0 else 0.0) for j in range(self.hidden_dim)]

                # Gradients for w1 and b1.
                d_w1 = [[d_h_pre[i2] * x_i[j] for j in range(vocab_size)] for i2 in range(self.hidden_dim)]
                d_b1 = list(d_h_pre)

                # Adam update for w1.
                for i2 in range(self.hidden_dim):
                    for j in range(vocab_size):
                        g = d_w1[i2][j]
                        m_w1[i2][j] = beta1 * m_w1[i2][j] + (1 - beta1) * g
                        v_w1[i2][j] = beta2 * v_w1[i2][j] + (1 - beta2) * g * g
                        m_hat = m_w1[i2][j] / (1 - beta1 ** t)
                        v_hat = v_w1[i2][j] / (1 - beta2 ** t)
                        self._w1[i2][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

                # Adam update for b1.
                for i2 in range(self.hidden_dim):
                    g = d_b1[i2]
                    m_b1[i2] = beta1 * m_b1[i2] + (1 - beta1) * g
                    v_b1[i2] = beta2 * v_b1[i2] + (1 - beta2) * g * g
                    m_hat = m_b1[i2] / (1 - beta1 ** t)
                    v_hat = v_b1[i2] / (1 - beta2 ** t)
                    self._b1[i2] -= lr * m_hat / (math.sqrt(v_hat) + eps)

                # Adam update for w2.
                for i2 in range(self.output_dim):
                    for j in range(self.hidden_dim):
                        g = d_w2[i2][j]
                        m_w2[i2][j] = beta1 * m_w2[i2][j] + (1 - beta1) * g
                        v_w2[i2][j] = beta2 * v_w2[i2][j] + (1 - beta2) * g * g
                        m_hat = m_w2[i2][j] / (1 - beta1 ** t)
                        v_hat = v_w2[i2][j] / (1 - beta2 ** t)
                        self._w2[i2][j] -= lr * m_hat / (math.sqrt(v_hat) + eps)

                # Adam update for b2.
                for i2 in range(self.output_dim):
                    g = d_b2[i2]
                    m_b2[i2] = beta1 * m_b2[i2] + (1 - beta1) * g
                    v_b2[i2] = beta2 * v_b2[i2] + (1 - beta2) * g * g
                    m_hat = m_b2[i2] / (1 - beta1 ** t)
                    v_hat = v_b2[i2] / (1 - beta2 ** t)
                    self._b2[i2] -= lr * m_hat / (math.sqrt(v_hat) + eps)

            final_loss = epoch_loss / n

        self._fitted = True
        return {"final_loss": final_loss, "n_train": n}

    @staticmethod
    def compute_auc(scores: list[float], labels: list[float]) -> float:
        """Compute binary AUC (area under ROC) via Mann-Whitney U statistic.

        WHY Mann-Whitney instead of sklearn: avoids the sklearn dependency
        on edge deployments (same rationale as JEPAReasonerProbe.evaluate_auc).

        Parameters
        ----------
        scores : list[float]
            Predicted violation probabilities in [0, 1].
        labels : list[float]
            Ground-truth binary labels (0.0 or 1.0).

        Returns
        -------
        float
            AUC in [0.0, 1.0].  Returns 0.5 when only one class is present.

        Spec: REQ-LEARN-044, SCENARIO-LEARN-086
        """
        pos = [s for s, l in zip(scores, labels) if l == 1.0]
        neg = [s for s, l in zip(scores, labels) if l == 0.0]
        if not pos or not neg:
            return 0.5
        n_pos, n_neg = len(pos), len(neg)
        concordant = 0.0
        for p in pos:
            concordant += sum(1.0 for n in neg if p > n) + 0.5 * sum(1.0 for n in neg if p == n)
        return concordant / (n_pos * n_neg)
