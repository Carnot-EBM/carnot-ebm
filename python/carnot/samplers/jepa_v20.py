"""JEPA v20 — Multi-Step Predictive Probe with Class-Weight Balancing.

WHY THIS MODULE EXISTS (REQ-LEARN-052, REQ-LEARN-053):
    JEPA v19 (Exp 770) achieved OOD AUC=0.5667 on GSM8K 800-999 — below the 0.75
    gate for Tier 3.5 deployment.  Root cause: only 57 training pairs (RETRO-JEPA-OOD-V19).

    JEPA v20 addresses this with two improvements over v19:

    1. EXPANDED CORPUS — trained on EDU-PRM uncertainty-selected steps from
       ``fover_edu_prm_selected.json`` (Exp 782), pooled with the original
       57-pair FoVer corpus when available.  EDU-PRM selects the top 30% of
       steps by bootstrap prediction variance — the hardest, most informative
       examples for the classifier.

    2. CLASS-WEIGHT BALANCING — the EDU-PRM selected corpus may be imbalanced.
       Weighting positive (incorrect) examples by ``n_negative / n_positive``
       in the BCE loss prevents the classifier from collapsing to the majority
       class.  This is the standard correction for class imbalance described
       in sklearn's documentation and used routinely in process reward models.

    CHANGED HYPERPARAMETERS vs v19:
        n_epochs: 200 → 300  (more training for smaller datasets)
        lr: 1e-3 → 5e-4      (gentler learning rate stabilises training on <30 examples)

    UNCHANGED: architecture (TF-IDF + 2-layer MLP), n_steps=3, max_vocab=500.

Spec: REQ-LEARN-052, REQ-LEARN-053, SCENARIO-LEARN-096, SCENARIO-LEARN-097
"""

from __future__ import annotations

import math
from typing import Sequence

from carnot.samplers.jepa_v19 import MultiStepJEPAv19


class MultiStepJEPAv20(MultiStepJEPAv19):
    """Multi-step JEPA probe with class-weight-balanced BCE loss.

    Inherits the full TF-IDF + 2-layer MLP architecture from MultiStepJEPAv19.
    The only behavioural change is in ``train()``: each positive (violation=1)
    example's loss contribution is multiplied by ``n_negative / n_positive`` to
    correct for label imbalance in the EDU-PRM selected corpus.

    WHY inherit rather than copy: the architecture is identical; only the loss
    weighting changes.  Inheritance keeps the diff minimal so code review can
    focus on the one change that matters.

    Parameters
    ----------
    hidden_dim : int
        MLP hidden layer width.  Default 64 (same as v19).
    n_steps : int
        Max steps to pool.  Default 3 (same as v19).
    output_dim : int
        Always 1 (binary).
    max_vocab : int
        TF-IDF vocabulary size.  Default 500 (same as v19).

    Spec: REQ-LEARN-052, SCENARIO-LEARN-097
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_steps: int = 3,
        output_dim: int = 1,
        max_vocab: int = 500,
    ) -> None:
        super().__init__(
            hidden_dim=hidden_dim,
            n_steps=n_steps,
            output_dim=output_dim,
            max_vocab=max_vocab,
        )

    def train(
        self,
        step_sequences: list[list[str]],
        labels: list[float],
        n_epochs: int = 300,
        lr: float = 5e-4,
    ) -> dict:
        """Train with class-weight-balanced BCE loss.

        Computes ``weight_positive = n_negative / n_positive`` from the training
        labels and multiplies each positive example's BCE loss contribution by
        that weight before the Adam gradient update.  When the corpus is perfectly
        balanced (n_positive == n_negative), weight_positive == 1.0 and training
        is identical to v19.

        WHY 300 epochs: v19 used 200 epochs on 57 examples.  With a smaller EDU-PRM
        corpus (~18-30 items) and a lower LR (5e-4), 300 epochs keeps the total
        gradient steps comparable while giving the optimizer more passes per example.

        WHY lr=5e-4: Adam with lr=1e-3 on <20 examples overshoots in the early
        epochs when the loss landscape is flat (sparse TF-IDF features).  Halving
        the LR reduces oscillation without significantly increasing wall time.

        Parameters
        ----------
        step_sequences : list[list[str]]
            Each entry is a list of step text strings for one training example.
        labels : list[float]
            Binary labels (0.0 = correct, 1.0 = violation).
        n_epochs : int
            Number of gradient descent epochs.  Default 300 (increased from v19's 200).
        lr : float
            Learning rate for Adam.  Default 5e-4 (tuned for small datasets).

        Returns
        -------
        dict with "final_loss" (float), "n_train" (int), "weight_positive" (float),
        "class_weight_used" (bool).

        Spec: REQ-LEARN-052, SCENARIO-LEARN-097
        """
        import random  # noqa: PLC0415

        n = len(step_sequences)
        if n == 0:
            raise ValueError("Cannot train on an empty dataset.")

        # Compute class weights BEFORE fitting vocabulary so the weight is
        # available in the first epoch.  weight_positive = n_neg / n_pos.
        n_positive = sum(1 for l in labels if l == 1.0)
        n_negative = n - n_positive

        # Avoid division by zero when one class is absent (degenerate dataset).
        if n_positive == 0 or n_negative == 0:
            weight_positive = 1.0
        else:
            weight_positive = n_negative / n_positive

        class_weight_used = weight_positive != 1.0 or (n_positive > 0 and n_negative > 0)

        # Fit TF-IDF vocabulary on all step texts.
        all_texts: list[str] = []
        for seq in step_sequences:
            all_texts.extend(seq)
        self._vectoriser.fit(all_texts)

        vocab_size = len(self._vectoriser._vocab)

        # He initialisation (same as v19).
        rng = random.Random(42)

        def _randn(scale: float) -> float:
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

        # Pre-compute pooled TF-IDF embeddings for all training examples.
        X = [self._embed_steps(seq) for seq in step_sequences]

        final_loss = float("inf")
        t = 0

        for _epoch in range(n_epochs):
            epoch_loss = 0.0
            for i in range(n):
                t += 1
                x_i = X[i]
                y_i = labels[i]

                # Per-example class weight: positive examples are up-weighted.
                # WHY here and not as a separate array: keeping the weight local
                # to the loss computation makes the gradient derivation explicit.
                sample_weight = weight_positive if y_i == 1.0 else 1.0

                # Forward pass.
                h_pre = self._matmul_add(self._w1, self._b1, x_i)
                h = self._relu(h_pre)
                logit_pre = self._matmul_add(self._w2, self._b2, h)
                pred = self._sigmoid(logit_pre[0])

                # Weighted BCE loss.
                pred_c = max(min(pred, 1.0 - 1e-7), 1e-7)
                raw_loss = -(y_i * math.log(pred_c) + (1.0 - y_i) * math.log(1.0 - pred_c))
                epoch_loss += sample_weight * raw_loss

                # Gradient of weighted BCE w.r.t. logit:
                # d(weighted_loss)/d_logit = sample_weight * (pred - y)
                d_logit = sample_weight * (pred - y_i)

                # Gradients for w2 and b2.
                d_w2 = [[d_logit * h[j] for j in range(self.hidden_dim)] for _ in range(self.output_dim)]
                d_b2 = [d_logit]

                # Backprop through hidden layer.
                d_h = [self._w2[0][j] * d_logit for j in range(self.hidden_dim)]
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
        return {
            "final_loss": final_loss,
            "n_train": n,
            "weight_positive": weight_positive,
            "class_weight_used": class_weight_used,
        }
