"""infonce_loss — InfoNCE contrastive loss for JEPA v16.

**Researcher summary:**
    JEPA v15 used PUREMinFormLoss (a margin loss over min-step scores). Exp 693 diagnosed this
    as the root cause of OOD AUC=0.4751 (below random): the formal-minimisation term inverts the
    gradient direction on OOD inputs whose formal structure differs from training. InfoNCE has no
    such term. It simply asks "can the model distinguish the anchor's positive partner from N-1
    negatives?" — a discrimination objective that generalises across distributions.

**Why InfoNCE and not cross-entropy?**
    Cross-entropy assigns a loss to each sample independently. InfoNCE creates relative pressure:
    the model must push (anchor, positive) embeddings together AND push all (anchor, negative)
    pairs apart *simultaneously* within each mini-batch. This contrastive pressure is what drives
    better representation geometry and, in turn, better OOD AUC.

**Formula (standard SimCLR / MoCo variant):**
    For anchor a, positive p, and negatives {n_1, ..., n_K}:

        L = -log( exp(sim(a, p) / T) / (exp(sim(a, p) / T) + sum_k exp(sim(a, n_k) / T)) )

    where sim(u, v) = dot(u, v) / (||u|| * ||v||) is cosine similarity and T is temperature.

    When T is small (e.g. 0.07), the softmax is sharper — the model is penalised heavily for
    putting any negative close to the anchor. When T is large, the gradient signal is diffuse.
    T=0.07 is the standard choice from SimCLR / CLIP.

Spec: REQ-LEARN-053, SCENARIO-LEARN-087
"""

from __future__ import annotations

import numpy as np


class InfoNCELoss:
    """InfoNCE contrastive loss for JEPA v16 chain-score training.

    **Detailed explanation for engineers:**
        The training data consists of (anchor, positive, negatives) triplets. For JEPA v16:
        - anchor: embedding of a question's context (or the first step of a correct chain)
        - positive: embedding of a correct chain
        - negatives: embeddings of incorrect chains for the same question

        The loss is the negative log of the softmax probability assigned to the positive sample
        among all (positive + negative) candidates. Because the denominator sums over all
        candidates simultaneously, the gradient from one negative affects all others — this
        "global contrast" is what makes InfoNCE representations generalise to OOD inputs.

    Attributes:
        temperature: Softmax sharpness parameter T. Default 0.07 (standard SimCLR value).
                     Lower T = sharper distribution = stronger OOD discrimination.

    Spec: REQ-LEARN-053, REQ-LEARN-053-2, SCENARIO-LEARN-087
    """

    def __init__(self, temperature: float = 0.07) -> None:
        """Initialise with the given temperature.

        Args:
            temperature: Softmax temperature. Must be > 0.
                         Default 0.07 follows the SimCLR / CLIP standard.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def _cosine_sim(self, u: np.ndarray, v: np.ndarray) -> float:
        """Cosine similarity between two 1-D vectors.

        **Why cosine and not dot product?**
            Dot product conflates magnitude with direction. Two embeddings with large norms
            can have a large dot product even if they point in completely different directions.
            Cosine similarity normalises out magnitude, so the model learns directional
            alignment rather than scale — more robust to the embedding norm drift that plagues
            long training runs.

        Args:
            u: 1-D numpy array.
            v: 1-D numpy array of the same shape as u.

        Returns:
            Scalar cosine similarity in [-1, 1].
        """
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u < 1e-12 or norm_v < 1e-12:
            return 0.0
        return float(np.dot(u, v) / (norm_u * norm_v))

    def compute(
        self,
        anchor: np.ndarray,
        positive: np.ndarray,
        negatives: list[np.ndarray],
    ) -> float:
        """Compute InfoNCE loss for a single (anchor, positive, negatives) triplet.

        **Detailed explanation for engineers:**
            1. Compute sim(anchor, positive) → sim_pos.
            2. Compute sim(anchor, n_i) for each negative → [sim_neg_1, ..., sim_neg_K].
            3. Divide all similarities by temperature T (sharpens the softmax).
            4. Compute softmax over [sim_pos/T, sim_neg_1/T, ..., sim_neg_K/T].
            5. Loss = -log(softmax[0]) — the negative log probability of the positive.

            If there are no negatives, returns 0.0 (nothing to discriminate against —
            equivalent to a batch size of 1, which produces zero gradient).

        Args:
            anchor:    1-D numpy array, the query embedding.
            positive:  1-D numpy array, the correct chain embedding.
            negatives: List of 1-D numpy arrays, each an incorrect chain embedding.

        Returns:
            Scalar float loss in [0, +inf). Lower is better.

        Spec: REQ-LEARN-053, SCENARIO-LEARN-087
        """
        if len(negatives) == 0:
            return 0.0

        # Similarities scaled by temperature.
        sim_pos = self._cosine_sim(anchor, positive) / self.temperature
        sim_negs = [self._cosine_sim(anchor, neg) / self.temperature for neg in negatives]

        # All logits: [positive, neg_1, ..., neg_K]
        logits = np.array([sim_pos] + sim_negs, dtype=np.float64)

        # Numerically stable log-softmax: subtract max before exp to avoid overflow.
        logits_shifted = logits - logits.max()
        log_sum_exp = np.log(np.sum(np.exp(logits_shifted)))
        # Loss = -log_softmax[0] = -(logits[0] - log_sum_exp(logits))
        loss = -(logits_shifted[0] - log_sum_exp)
        return float(loss)

    def batch_loss(
        self,
        anchors: list[np.ndarray],
        positives: list[np.ndarray],
        negatives_list: list[list[np.ndarray]],
    ) -> float:
        """Compute mean InfoNCE loss over a batch of triplets.

        Args:
            anchors:        List of anchor embeddings (one per training example).
            positives:      List of positive embeddings (one per training example).
            negatives_list: List of negative-lists (one list per training example).

        Returns:
            Mean scalar loss across all triplets. 0.0 if batch is empty.

        Spec: REQ-LEARN-053
        """
        if not anchors:
            return 0.0
        losses = [
            self.compute(a, p, ns)
            for a, p, ns in zip(anchors, positives, negatives_list)
        ]
        return float(np.mean(losses))
