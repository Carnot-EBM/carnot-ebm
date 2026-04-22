"""JEPA v18 — LambdaRank listwise ranking loss with ActPRM uncertainty weighting.

WHY THIS MODULE EXISTS (RETRO-CRITICAL — v15/v16/v17 all below-random):
    JEPA v15, v16, v17 used pairwise losses (contrastive, BCE, RankNet) and produced
    OOD AUC of 0.4751, 0.4759, 0.4819 — all below random chance (0.5).  The root
    cause was that pairwise methods compare exactly two steps at a time.  When the
    training corpus has limited diversity (FoVer v1: 200 examples, ~2 steps/question),
    many pairs have nearly identical quality, producing near-zero gradients.  The model
    converges numerically but learns nothing discriminative.

    LambdaRank (Burges 2006) fixes this by optimising NDCG directly over the FULL step
    sequence for each question simultaneously.  The gradient (lambda_ij) for each pair
    is weighted by delta_NDCG — the change in NDCG that would result from swapping the
    pair's ranks.  Pairs that matter most to the ranking get the largest gradients.
    Pairs near the decision boundary dominate; trivially separated pairs contribute
    little.  This focuses learning effort on the genuinely hard cases.

    ActPRM uncertainty weighting (arXiv 2504.10559) further focuses training by
    up-weighting examples where Z3 and PDDL labels disagree.  These are the steps
    where formal verification is ambiguous — the hardest and most informative examples.
    Down-weighting easy (unambiguous) examples prevents the model from over-fitting
    to trivially correct steps that carry no ranking signal.

WHY NUMPY ONLY (no JAX/torch for the v18 implementation):
    The first three JEPA versions failed in part because the GPU-dependent encoders
    (Qwen3.5-0.8B hidden states) were not available during pre-flight tests, causing
    silent simulation fallbacks.  v18 uses a bag-of-words feature extractor that runs
    identically on any machine — no GPU required.  The encoder can be swapped for
    Qwen hidden states when the GPU is available; the ranking head and loss function
    stay the same.

Spec: REQ-VER-028, REQ-VER-029, SCENARIO-VER-035, SCENARIO-VER-036
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Feature extractor — bag-of-words over character n-grams
# ---------------------------------------------------------------------------

# Vocabulary size for the bag-of-words encoder.  1024 buckets are enough to
# distinguish arithmetic steps (numbers, operators, step markers) without
# requiring a tokeniser or model download.
_VOCAB_SIZE = 1024


def _featurize(step_text: str, vocab_size: int = _VOCAB_SIZE) -> np.ndarray:
    """Convert a step string to a normalised bag-of-words feature vector.

    WHY character n-grams: arithmetic steps contain numbers and operators.
    Character 3-grams capture patterns like "= 8." or "42." more reliably
    than word-level features when numbers vary between questions.  Hash-based
    bucketing means no vocabulary file is needed.

    Parameters
    ----------
    step_text : str
        The raw text of a reasoning step (e.g. "First, 3 + 5 = 8.").
    vocab_size : int
        Number of hash buckets.  Must match the model's input dimension.

    Returns
    -------
    np.ndarray
        Float32 feature vector of shape ``(vocab_size,)``, L2-normalised.
    """
    vec = np.zeros(vocab_size, dtype=np.float32)
    text = step_text.lower()
    for i in range(len(text) - 2):
        trigram = text[i : i + 3]
        bucket = hash(trigram) % vocab_size
        vec[bucket] += 1.0
    # Also include individual characters and bigrams for short steps
    for i in range(len(text)):
        bucket = hash(text[i]) % vocab_size
        vec[bucket] += 0.5
    for i in range(len(text) - 1):
        bigram = text[i : i + 2]
        bucket = hash(bigram) % vocab_size
        vec[bucket] += 0.75
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def _ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int | None = None) -> float:
    """Compute NDCG@k for a single query group.

    WHY NDCG: LambdaRank optimises the expected change in NDCG for each
    pairwise swap (Burges 2006).  NDCG is position-sensitive — ranking a
    relevant step at position 1 is worth more than ranking it at position 3.
    This makes LambdaRank's gradients focus on mistakes at the top of the
    ranking, which is exactly what matters for step selection in CoT chains.

    Parameters
    ----------
    labels : np.ndarray
        Relevance labels (0 = incorrect step, 1 = correct step).
    scores : np.ndarray
        Model-predicted scores (higher = more likely correct).
    k : int | None
        Rank cutoff.  None means use all positions.

    Returns
    -------
    float
        NDCG@k in [0, 1].
    """
    n = len(labels)
    if k is None:
        k = n
    order = np.argsort(-scores)
    ranked_labels = labels[order[:k]]
    ideal_labels = np.sort(labels)[::-1][:k]

    def dcg(rel: np.ndarray) -> float:
        return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))

    ideal = dcg(ideal_labels)
    if ideal == 0.0:
        return 1.0  # all labels are 0 → perfect ranking by definition
    return dcg(ranked_labels) / ideal


# ---------------------------------------------------------------------------
# LambdaRank loss
# ---------------------------------------------------------------------------


def lambda_rank_loss(
    scores: np.ndarray,
    labels: np.ndarray,
    example_weights: np.ndarray | None = None,
) -> Tuple[float, np.ndarray]:
    """Compute LambdaRank loss and per-item gradient for a single query group.

    LambdaRank (Burges 2006) directly optimises NDCG by computing a signed
    gradient (lambda_i) for each step i that represents how much the step's
    score should change to improve the group's NDCG.  The gradient for step i
    is the sum of lambda_ij over all steps j in the group, where:

        lambda_ij = delta_NDCG_ij * |sigma_ij * (1 - sigma_ij)|  (for i relevant, j irrelevant)

    and sigma_ij = sigmoid(score_i - score_j).

    WHY |delta_NDCG| as the weight: the magnitude of NDCG change caused by
    swapping two ranks tells us how important that correction is.  A swap near
    the top of the list (e.g., rank 1↔2) has a large delta_NDCG and thus drives
    a large gradient update.  A swap at rank 100↔101 contributes almost nothing.

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores, shape ``(n,)``.  Higher = more relevant.
    labels : np.ndarray
        Binary relevance labels, shape ``(n,)``.  1 = correct, 0 = incorrect.
    example_weights : np.ndarray | None
        Per-item ActPRM uncertainty weights, shape ``(n,)``.  When provided,
        lambda_ij is multiplied by ``(weight_i + weight_j) / 2``.

    Returns
    -------
    (loss, lambdas) : (float, np.ndarray)
        ``loss`` — scalar LambdaRank cross-entropy loss.
        ``lambdas`` — gradient for each step's score, shape ``(n,)``.
        Positive lambda_i means "increase score_i"; negative means "decrease".
    """
    n = len(scores)
    if n == 0:
        return 0.0, np.zeros(0, dtype=np.float32)

    if example_weights is None:
        example_weights = np.ones(n, dtype=np.float32)

    # Current NDCG before any swap
    base_ndcg = _ndcg_at_k(labels, scores)

    lambdas = np.zeros(n, dtype=np.float32)
    total_loss = 0.0

    for i in range(n):
        for j in range(n):
            if labels[i] <= labels[j]:
                # Only compute lambda for pairs where i is more relevant than j
                continue
            # labels[i] > labels[j]: i should rank above j
            score_diff = float(scores[i]) - float(scores[j])
            sigma = 1.0 / (1.0 + math.exp(-score_diff))  # sigmoid(s_i - s_j)

            # delta_NDCG: how much NDCG would change if we swapped ranks of i and j
            swapped_scores = scores.copy()
            swapped_scores[i], swapped_scores[j] = scores[j], scores[i]
            swapped_ndcg = _ndcg_at_k(labels, swapped_scores)
            delta_ndcg = abs(base_ndcg - swapped_ndcg)

            # Weight by ActPRM uncertainty (average of the two step weights)
            w = (float(example_weights[i]) + float(example_weights[j])) / 2.0

            # LambdaRank loss contribution: -log(sigma) weighted by delta_NDCG
            # When sigma → 1 (i ranked above j as desired), loss → 0
            log_sigma = math.log(sigma + 1e-10)
            total_loss += -delta_ndcg * w * log_sigma

            # Lambda (gradient signal): scale by derivative of log-loss w.r.t. score_diff
            # d(-log(sigma)) / d(score_diff) = -(1 - sigma) * sigma / sigma = -(1-sigma)
            # But for LambdaRank we use: lambda_ij = delta_NDCG * w * (1 - sigma)
            # positive for i (should be ranked higher), negative for j
            lam = delta_ndcg * w * (1.0 - sigma)
            lambdas[i] += lam   # push i's score up
            lambdas[j] -= lam   # push j's score down

    return total_loss, lambdas


# ---------------------------------------------------------------------------
# ActPRM uncertainty weighting
# ---------------------------------------------------------------------------


def actprm_weight(z3_label: bool | None, pddl_label: bool | None) -> float:
    """Compute ActPRM uncertainty weight for a training example.

    ActPRM (arXiv 2504.10559) weights training examples by label disagreement:
    when Z3 and PDDL formal verifiers agree on a step's correctness, the example
    is unambiguous and carries little training signal.  When they disagree, the
    step is genuinely hard — exactly the case we want the model to learn from.

    Formula:
        agreement_score = 1.0 if z3_label == pddl_label else 0.0
        weight = 1.0 - agreement_score + 0.1   # floor at 0.1 so no example is ignored

    WHY a floor of 0.1: completely zeroing out agreed examples would ignore 90%+ of
    FoVer v2 (which has high Z3/PDDL agreement on arithmetic steps).  The 0.1 floor
    ensures all examples contribute a small amount, but disagreed examples get 11×
    more gradient signal.

    Parameters
    ----------
    z3_label : bool | None
        Whether Z3 verified the step as correct.  None means "not available".
    pddl_label : bool | None
        Whether PDDL verified the step as correct.  None means "not available".

    Returns
    -------
    float
        Weight in [0.1, 1.1].  Higher weight = more uncertain = harder example.
    """
    if z3_label is None or pddl_label is None:
        # If only one verifier has a label, treat as uncertain (moderate weight)
        return 0.6
    agreement_score = 1.0 if (z3_label == pddl_label) else 0.0
    return 1.0 - agreement_score + 0.1


# ---------------------------------------------------------------------------
# 2-layer MLP ranking head
# ---------------------------------------------------------------------------


class JEPALambdaRankV18:
    """JEPA v18 ranking model: LambdaRank loss + ActPRM uncertainty weighting.

    Architecture:
        1. Encoder: ``_featurize()`` — character n-gram bag-of-words (1024-dim).
           In production this would be replaced by Qwen3.5-0.8B layer-16 hidden
           states (1024-dim), but the bag-of-words encoder produces features of the
           same dimensionality and lets the model run on any machine without a GPU.
        2. Ranking head: 2-layer MLP → scalar energy score.
           Layer 1: Linear(feature_dim, hidden_dim) + ReLU
           Layer 2: Linear(hidden_dim, hidden_dim) + ReLU
           Layer 3: Linear(hidden_dim, 1) — output scalar score

    Training: Adam optimiser (lr=1e-4) over 50 epochs.  Each epoch processes all
    query groups (one question = one group).  LambdaRank loss is computed per group;
    gradients are accumulated and applied once per epoch.

    Evaluation: OOD AUC on held-out questions.  AUC = AUROC over all pairwise
    (correct_step, incorrect_step) comparisons from the eval corpus.

    Parameters
    ----------
    feature_dim : int
        Dimensionality of the input feature vector.  Must match ``_featurize()``
        output (default 1024).
    hidden_dim : int
        Width of each hidden layer.
    """

    def __init__(self, feature_dim: int = _VOCAB_SIZE, hidden_dim: int = 64) -> None:
        rng = np.random.default_rng(42)  # deterministic init for reproducibility
        scale = 0.1

        # Layer 1 weights and biases
        self.W1 = rng.normal(0, scale, (hidden_dim, feature_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2 weights and biases
        self.W2 = rng.normal(0, scale, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

        # Output layer weights and bias
        self.W3 = rng.normal(0, scale, (1, hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(1, dtype=np.float32)

        # Adam optimiser state (moment estimates)
        self._adam_state: dict = {}
        self._adam_t = 0  # step counter

    def _forward(self, x: np.ndarray) -> Tuple[float, dict]:
        """Forward pass: feature vector → scalar score, caching activations.

        Parameters
        ----------
        x : np.ndarray
            Feature vector, shape ``(feature_dim,)``.

        Returns
        -------
        (score, cache) : (float, dict)
            ``score`` is the scalar output.
            ``cache`` holds intermediate activations needed for backprop.
        """
        h1 = np.maximum(0.0, self.W1 @ x + self.b1)    # ReLU
        h2 = np.maximum(0.0, self.W2 @ h1 + self.b2)   # ReLU
        score = float((self.W3 @ h2 + self.b3)[0])
        return score, {"x": x, "h1": h1, "h2": h2}

    def _backward(self, cache: dict, d_score: float) -> dict:
        """Backward pass: compute parameter gradients from score gradient.

        Parameters
        ----------
        cache : dict
            Activations from ``_forward``.
        d_score : float
            Gradient of the loss w.r.t. the output score.

        Returns
        -------
        dict
            Gradients for W1, b1, W2, b2, W3, b3.
        """
        x, h1, h2 = cache["x"], cache["h1"], cache["h2"]

        d_h2 = self.W3.T * d_score          # shape (hidden_dim,)
        d_h2 = d_h2.squeeze()
        d_h2_relu = d_h2 * (h2 > 0)         # ReLU backward

        dW3 = np.outer(np.array([d_score]), h2)
        db3 = np.array([d_score], dtype=np.float32)

        d_h1 = self.W2.T @ d_h2_relu
        d_h1_relu = d_h1 * (h1 > 0)         # ReLU backward

        dW2 = np.outer(d_h2_relu, h1)
        db2 = d_h2_relu.copy()

        dW1 = np.outer(d_h1_relu, x)
        db1 = d_h1_relu.copy()

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def _adam_update(self, param_name: str, param: np.ndarray, grad: np.ndarray,
                     lr: float, beta1: float = 0.9, beta2: float = 0.999,
                     eps: float = 1e-8) -> np.ndarray:
        """Apply one Adam update step to a parameter.

        WHY Adam over plain SGD: Adam adapts the learning rate per-parameter
        using first and second moment estimates.  This is critical for LambdaRank
        because the loss landscape is non-uniform — parameters connected to the
        top-ranked steps get very large gradients while others get near-zero
        gradients in every batch.  Plain SGD oscillates; Adam converges reliably.
        """
        if param_name not in self._adam_state:
            self._adam_state[param_name] = {
                "m": np.zeros_like(param),
                "v": np.zeros_like(param),
            }
        state = self._adam_state[param_name]
        state["m"] = beta1 * state["m"] + (1 - beta1) * grad
        state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2
        m_hat = state["m"] / (1 - beta1 ** self._adam_t)
        v_hat = state["v"] / (1 - beta2 ** self._adam_t)
        return param - lr * m_hat / (np.sqrt(v_hat) + eps)

    def predict_score(self, step_text: str) -> float:
        """Return the model's quality score for a single step.

        Higher score means the model believes the step is more likely correct.

        Parameters
        ----------
        step_text : str
            Raw text of the reasoning step.

        Returns
        -------
        float
            Scalar quality score.
        """
        x = _featurize(step_text)
        score, _ = self._forward(x)
        return score

    def train(
        self,
        groups: List[dict],
        n_epochs: int = 50,
        lr: float = 1e-4,
    ) -> List[float]:
        """Train the model on a list of query groups using LambdaRank.

        Each group represents one question (query) and contains a list of steps
        with their relevance labels and ActPRM weights.

        Parameters
        ----------
        groups : list of dict
            Each dict has:
            - "steps": list of {"text": str, "label": int (1=correct, 0=incorrect),
                                "z3_label": bool|None, "pddl_label": bool|None}

        n_epochs : int
            Number of full passes over all query groups.
        lr : float
            Adam learning rate.

        Returns
        -------
        list of float
            Training loss per epoch.
        """
        epoch_losses: List[float] = []

        for epoch in range(n_epochs):
            self._adam_t += 1
            epoch_loss = 0.0
            n_groups = 0

            # Accumulate parameter gradients across all groups
            accum_grads: dict = {}

            for group in groups:
                steps = group["steps"]
                if len(steps) < 2:
                    continue  # need at least 2 steps for ranking

                # Check that the group has both positive and negative examples
                labels_arr = np.array([s["label"] for s in steps], dtype=np.float32)
                if labels_arr.max() == labels_arr.min():
                    continue  # all same label — no ranking signal

                # Featurize all steps and run forward pass
                features = [_featurize(s["text"]) for s in steps]
                scores = np.array([self._forward(x)[0] for x in features], dtype=np.float32)
                caches = [self._forward(x)[1] for x in features]

                # ActPRM weights for each step
                weights = np.array([
                    actprm_weight(s.get("z3_label"), s.get("pddl_label"))
                    for s in steps
                ], dtype=np.float32)

                # LambdaRank loss and per-step score gradients
                loss, lambdas = lambda_rank_loss(scores, labels_arr, weights)
                epoch_loss += loss
                n_groups += 1

                # Backprop: for each step, use lambda_i as d_loss/d_score_i
                for idx, (cache, lam) in enumerate(zip(caches, lambdas)):
                    # Negate lambda because lambdas point in the "increase score" direction
                    # and we want to minimise loss (i.e., move in the -gradient direction)
                    d_score = -float(lam)
                    grads = self._backward(cache, d_score)
                    for k, g in grads.items():
                        accum_grads[k] = accum_grads.get(k, 0.0) + g

            # Apply accumulated gradients
            if n_groups > 0:
                scale = 1.0 / n_groups
                self.W1 = self._adam_update("W1", self.W1, accum_grads.get("W1", 0.0) * scale, lr)
                self.b1 = self._adam_update("b1", self.b1, accum_grads.get("b1", 0.0) * scale, lr)
                self.W2 = self._adam_update("W2", self.W2, accum_grads.get("W2", 0.0) * scale, lr)
                self.b2 = self._adam_update("b2", self.b2, accum_grads.get("b2", 0.0) * scale, lr)
                self.W3 = self._adam_update("W3", self.W3, accum_grads.get("W3", 0.0) * scale, lr)
                self.b3 = self._adam_update("b3", self.b3, accum_grads.get("b3", 0.0) * scale, lr)

            epoch_losses.append(epoch_loss / max(1, n_groups))

        return epoch_losses

    def evaluate_auc(self, eval_groups: List[dict]) -> float:
        """Evaluate OOD AUC on a list of held-out query groups.

        AUC = AUROC over all (correct_step, incorrect_step) pairs from the eval
        corpus.  For each such pair, a "correct prediction" is: score(correct) >
        score(incorrect).

        WHY pairwise AUC: this mirrors the metric used by v15/v16/v17 so results
        are comparable.  AUC = 0.5 means the model is no better than random.
        AUC > 0.5 means the model has learned something discriminative.

        Parameters
        ----------
        eval_groups : list of dict
            Same format as ``train()``.

        Returns
        -------
        float
            OOD AUC in [0, 1].
        """
        correct_scores: List[float] = []
        incorrect_scores: List[float] = []

        for group in eval_groups:
            steps = group["steps"]
            for s in steps:
                score = self.predict_score(s["text"])
                if s["label"] == 1:
                    correct_scores.append(score)
                else:
                    incorrect_scores.append(score)

        if not correct_scores or not incorrect_scores:
            return 0.5  # degenerate case: no discrimination possible

        # Compute AUC via counting concordant pairs (no sklearn needed)
        n_correct = len(correct_scores)
        n_incorrect = len(incorrect_scores)
        n_concordant = 0
        n_tied = 0
        for cs in correct_scores:
            for is_ in incorrect_scores:
                if cs > is_:
                    n_concordant += 1
                elif cs == is_:
                    n_tied += 1

        total_pairs = n_correct * n_incorrect
        return (n_concordant + 0.5 * n_tied) / total_pairs
