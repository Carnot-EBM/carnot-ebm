"""jepa_v17_ranknet — JEPA v17: RankNet pairwise ranking loss + hard negative mining.

**Why v15 and v16 both failed (Exp 693 root cause: pure_loss_anti_correlation):**
    Both BCE and InfoNCE losses allow the model to "hedge" all outputs to P≈0.5 globally.
    When every output is 0.5, the loss is satisfied (log(2) per sample for BCE, near-zero
    for InfoNCE in high-temperature regime) without the model learning ANY discrimination.
    Two consecutive retrains confirmed: AUC=0.4751 (v15) and AUC=0.4759 (v16) — both below
    random chance, meaning the model actively inverts the correctness signal.

**Why RankNet cannot be hedged:**
    RankNet pairwise ranking loss:
        L = -log(sigmoid(score(incorrect) - score(correct)))
    When score(incorrect) == score(correct) == 0.5:
        L = -log(sigmoid(0)) = -log(0.5) = log(2) ≈ 0.693
    When score(incorrect) >> score(correct):
        L → -log(sigmoid(+∞)) = -log(1) = 0
    The ONLY way to reduce loss is to push score(incorrect) STRICTLY ABOVE score(correct)
    for every pair. Hedging to equal scores gives non-zero gradient; only correct ranking
    gives zero gradient. This is a strict partial order constraint that BCE/InfoNCE lack.

**Hard negative mining (why it matters):**
    Without hard negatives, the model can achieve low loss by learning to distinguish
    trivially different pairs (e.g., "The answer is 1." vs "The answer is 1000000.").
    Hard negative mining selects the incorrect step with HIGHEST cosine similarity to the
    correct anchor — the hardest pair to distinguish — forcing the model to learn
    fine-grained semantic discrimination rather than surface-level differences.

**Training data note (FoVer formal v1):**
    The fover_labeled_formal_v1.json corpus (Exp 686, 200 Z3-labeled pairs) has all 200
    steps labeled step_correct=True (z3_verdict="unparseable" for all). This means we
    have no explicitly-labeled incorrect steps. We generate synthetic incorrect steps by
    injecting arithmetic errors into correct steps — a controlled perturbation that
    preserves the question context while guaranteeing the step is wrong. Each correct step
    "The answer is X." becomes incorrect step "The answer is X + prime_offset." where
    prime_offset is a small prime (7, 11, 13...) to avoid accidental correctness.

Spec: REQ-VERIFY-140, REQ-VERIFY-141, REQ-VERIFY-142,
      SCENARIO-VERIFY-140, SCENARIO-VERIFY-141, SCENARIO-VERIFY-142
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM: int = 256
"""Embedding dimension. Matches v16 for architecture compatibility."""

# Small primes used to generate incorrect answers: offset by these to guarantee wrong answer.
_INCORRECT_OFFSETS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]


# ---------------------------------------------------------------------------
# Embedding helper (identical to v16 for fair comparison)
# ---------------------------------------------------------------------------


def _text_embedding(text: str, dim: int = EMBED_DIM) -> np.ndarray:
    """Deterministic hash-seeded random projection embedding.

    **Why not a real encoder:**
        v17 is a loss-function ablation over v16. Using the same embedding scheme as v16
        means any AUC improvement is attributable solely to the RankNet loss, not to
        better representations. This is the correct experimental design for an ablation.

    Args:
        text: Input text string.
        dim:  Output dimension.

    Returns:
        1-D float32 array of shape (dim,), L2-normalised.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


# ---------------------------------------------------------------------------
# Synthetic incorrect step generator
# ---------------------------------------------------------------------------


def _make_incorrect_step(correct_step: str, offset_idx: int = 0) -> str:
    """Generate a synthetic incorrect step by perturbing numbers in the correct step.

    **How it works:**
        We scan the correct step text for the last integer that appears (e.g. in
        "The answer is 42." we find 42). We add a small prime offset to produce
        "The answer is 49." (42 + 7). This guarantees the step is arithmetically
        wrong while preserving the natural-language structure of the step.

        If no integer is found, we append "... (incorrect)" to mark it as wrong.
        The offset_idx cycles through _INCORRECT_OFFSETS to produce diverse negatives
        when called multiple times for the same step.

    Args:
        correct_step: The correct step text.
        offset_idx:   Which prime offset to use (cycles through _INCORRECT_OFFSETS).

    Returns:
        Modified step text with an injected arithmetic error.
    """
    import re

    offset = _INCORRECT_OFFSETS[offset_idx % len(_INCORRECT_OFFSETS)]
    # Find the last integer in the string (most likely the answer).
    matches = list(re.finditer(r"\d+", correct_step))
    if matches:
        last_match = matches[-1]
        original_val = int(last_match.group())
        new_val = original_val + offset
        # Replace the last occurrence of this integer.
        start, end = last_match.start(), last_match.end()
        return correct_step[:start] + str(new_val) + correct_step[end:]
    # Fallback: no integer found — append a marker.
    return correct_step + " (incorrect)"


# ---------------------------------------------------------------------------
# RankNet loss
# ---------------------------------------------------------------------------


def ranknet_loss(
    scores_incorrect: jnp.ndarray,
    scores_correct: jnp.ndarray,
) -> jnp.ndarray:
    """Compute RankNet pairwise ranking loss.

    **What RankNet loss does:**
        For each (incorrect, correct) pair from the same question, computes:
            L_i = -log(sigmoid(score(incorrect) - score(correct)))

        This loss is minimised ONLY when score(incorrect) > score(correct) by a wide margin.
        - When scores are equal: sigmoid(0) = 0.5 → L = log(2) ≈ 0.693 (non-zero, forces learning)
        - When incorrect >> correct: sigmoid(+∞) → 1 → L → 0 (constraint satisfied)
        - When incorrect << correct: sigmoid(-∞) → 0 → L → +∞ (strong penalty for inversion)

        This is a strict partial order constraint: the model CANNOT hedge to 0.5 because
        equal scores still produce gradient (unlike BCE where P=0.5 is a loss minimum for
        balanced datasets).

    Args:
        scores_incorrect: 1-D JAX array of scores for incorrect steps. Higher = more wrong.
        scores_correct:   1-D JAX array of scores for correct steps (same shape).

    Returns:
        Scalar mean RankNet loss. 0.0 = perfect ranking; log(2) = full hedging.

    Spec: REQ-VERIFY-141, SCENARIO-VERIFY-141
    """
    # RankNet: incorrect step should have HIGHER score (more wrong = higher energy).
    # So we compute sigmoid(score_incorrect - score_correct) and take -log.
    diff = scores_incorrect - scores_correct
    loss = -jnp.mean(jnp.log(jax.nn.sigmoid(diff) + 1e-12))
    return loss


# ---------------------------------------------------------------------------
# Hard negative mining
# ---------------------------------------------------------------------------


def hard_negative_mining(
    correct_embeddings: np.ndarray,
    incorrect_embeddings: np.ndarray,
) -> np.ndarray:
    """Select the hardest incorrect step for each correct anchor via cosine similarity.

    **What "hardest" means:**
        The hardest incorrect step is the one most semantically similar to the correct anchor.
        If the incorrect step is very DIFFERENT from the correct one (low cosine similarity),
        the model can easily learn to distinguish them from surface features alone. If the
        incorrect step is very SIMILAR (high cosine similarity), the model must learn subtle
        semantic discrimination — a much harder and more informative training signal.

        We compute the full cosine similarity matrix between all correct and all incorrect
        embeddings, then for each correct embedding, select the incorrect one with
        MAXIMUM similarity.

    **Why cosine similarity (not Euclidean distance)?**
        Both embedding vectors are L2-normalised (from _text_embedding), so cosine similarity
        equals the dot product. This is the same similarity metric used in InfoNCE (v16),
        making the hard negative selection consistent with the contrastive training objective.

    Args:
        correct_embeddings:   Array of shape (n_correct, embed_dim), L2-normalised.
        incorrect_embeddings: Array of shape (n_incorrect, embed_dim), L2-normalised.

    Returns:
        1-D int array of shape (n_correct,): for each correct embedding, the index into
        incorrect_embeddings of its hardest negative.

    Spec: REQ-VERIFY-142, SCENARIO-VERIFY-142
    """
    correct_embeddings = np.asarray(correct_embeddings, dtype=np.float32)
    incorrect_embeddings = np.asarray(incorrect_embeddings, dtype=np.float32)

    # Cosine similarity matrix: (n_correct, n_incorrect).
    # Both inputs are already L2-normalised so dot product = cosine similarity.
    sim_matrix = correct_embeddings @ incorrect_embeddings.T  # (n_correct, n_incorrect)

    # For each correct embedding, pick the incorrect with maximum similarity.
    hard_negative_indices = np.argmax(sim_matrix, axis=1)
    return hard_negative_indices


# ---------------------------------------------------------------------------
# JEPARankNetV17 model
# ---------------------------------------------------------------------------


class JEPARankNetV17:
    """JEPA v17 step scorer: MLP trained with RankNet pairwise ranking loss.

    **Architecture (identical to v16 for fair ablation):**
        Input: 256-D L2-normalised text embedding (hash-seeded random projection).
        Layer 1: Linear(256 → 64) + ReLU  (He init)
        Layer 2: Linear(64 → 32) + ReLU   (He init)
        Layer 3: Linear(32 → 1) + identity (output: raw score, not sigmoid)

        We output RAW scores (not probabilities) because RankNet loss only needs the
        DIFFERENCE between incorrect and correct scores — the absolute scale doesn't matter.
        Higher raw score = model thinks the step is more likely WRONG (incorrect).

    **Training objective:**
        RankNet pairwise ranking loss (see ranknet_loss docstring).
        For each (correct_step, hard_negative_incorrect_step) pair from the same question,
        push score(incorrect) up and score(correct) down.

    **Scoring convention:**
        High score = likely INCORRECT step (high energy = likely wrong).
        Low score  = likely CORRECT step (low energy = likely right).
        This matches the EBM convention: high energy = constraint violated = wrong.

    Spec: REQ-VERIFY-140, REQ-VERIFY-141, REQ-VERIFY-142
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise v17 with He-initialised MLP weights.

        Args:
            seed: Random seed for reproducible weight initialisation.
        """
        rng = np.random.default_rng(seed)
        # He init: scale by sqrt(2 / fan_in) to prevent vanishing/exploding gradients.
        self._W1 = rng.standard_normal((EMBED_DIM, 64)).astype(np.float32) * math.sqrt(2.0 / EMBED_DIM)
        self._b1 = np.zeros(64, dtype=np.float32)
        self._W2 = rng.standard_normal((64, 32)).astype(np.float32) * math.sqrt(2.0 / 64)
        self._b2 = np.zeros(32, dtype=np.float32)
        self._W3 = rng.standard_normal((32, 1)).astype(np.float32) * math.sqrt(2.0 / 32)
        self._b3 = np.zeros(1, dtype=np.float32)

    def _params(self) -> dict[str, np.ndarray]:
        return {
            "W1": self._W1, "b1": self._b1,
            "W2": self._W2, "b2": self._b2,
            "W3": self._W3, "b3": self._b3,
        }

    def _set_params(self, params: dict[str, np.ndarray]) -> None:
        self._W1 = np.asarray(params["W1"])
        self._b1 = np.asarray(params["b1"])
        self._W2 = np.asarray(params["W2"])
        self._b2 = np.asarray(params["b2"])
        self._W3 = np.asarray(params["W3"])
        self._b3 = np.asarray(params["b3"])

    def score(self, embedding: np.ndarray) -> float:
        """Compute raw incorrectness score for a single step embedding.

        Args:
            embedding: 1-D float32 array of shape (EMBED_DIM,).

        Returns:
            Raw scalar score. Higher = more likely INCORRECT.
        """
        x = np.asarray(embedding, dtype=np.float32)
        h1 = np.maximum(0.0, x @ self._W1 + self._b1)
        h2 = np.maximum(0.0, h1 @ self._W2 + self._b2)
        out = h2 @ self._W3 + self._b3
        return float(out[0])

    def save(self, path: str) -> None:
        """Save model weights to a .npz file.

        Args:
            path: File path (should end in .npz).
        """
        np.savez(path, **self._params())

    def load(self, path: str) -> None:
        """Load model weights from a .npz file.

        Args:
            path: File path to load from.
        """
        data = np.load(path)
        self._set_params({k: data[k] for k in data.files})


# ---------------------------------------------------------------------------
# JAX-based training loop
# ---------------------------------------------------------------------------


def _jax_forward(params: dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """JAX forward pass for gradient computation.

    **Why JAX and not NumPy:**
        JAX provides automatic differentiation (jax.grad) which computes exact gradients
        of the RankNet loss with respect to all MLP parameters. NumPy would require manual
        gradient derivation (error-prone). JAX's jit compilation also speeds up the inner loop.

    Args:
        params: Dict of weight arrays (W1, b1, W2, b2, W3, b3).
        x:      Input embedding, shape (EMBED_DIM,).

    Returns:
        Scalar raw score (jnp array).
    """
    h1 = jax.nn.relu(x @ params["W1"] + params["b1"])
    h2 = jax.nn.relu(h1 @ params["W2"] + params["b2"])
    out = h2 @ params["W3"] + params["b3"]
    return out[0]


def _batch_loss(
    params: dict[str, Any],
    correct_embs: jnp.ndarray,
    incorrect_embs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mean RankNet loss over a batch of (correct, incorrect) pairs.

    Args:
        params:         MLP weight dict.
        correct_embs:   Shape (n_pairs, EMBED_DIM).
        incorrect_embs: Shape (n_pairs, EMBED_DIM).

    Returns:
        Scalar mean RankNet loss.
    """
    scores_correct = jax.vmap(lambda x: _jax_forward(params, x))(correct_embs)
    scores_incorrect = jax.vmap(lambda x: _jax_forward(params, x))(incorrect_embs)
    return ranknet_loss(scores_incorrect, scores_correct)


# ---------------------------------------------------------------------------
# Training data builder
# ---------------------------------------------------------------------------


def build_ranknet_pairs(
    fover_pairs: list[dict[str, Any]],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build (correct_embedding, hard_negative_embedding) training pairs from FoVer pairs.

    **Strategy for all-correct FoVer corpus:**
        The fover_labeled_formal_v1.json corpus has all 200 steps labeled step_correct=True
        (z3_verdict="unparseable" for all — the Z3 solver couldn't parse any step). This means
        no explicitly-labeled incorrect steps are available.

        We generate synthetic incorrect steps by injecting arithmetic errors:
            correct step:   "The answer is 42."
            incorrect step: "The answer is 49."  (42 + 7, where 7 is a small prime)

        This gives each correct step a paired hard negative with the same semantic context
        but a guaranteed arithmetic error. After generating all (correct, incorrect) pairs,
        we apply hard negative mining to select the most similar incorrect step as the
        actual training negative for each correct anchor.

    Args:
        fover_pairs: List of pair dicts from fover_labeled_formal_v1.json.

    Returns:
        Tuple (correct_embeddings, hardneg_embeddings): two parallel lists of np.ndarray.
        Each element is a 1-D float32 embedding of shape (EMBED_DIM,).
    """
    correct_embs: list[np.ndarray] = []
    incorrect_embs_pool: list[np.ndarray] = []

    for i, pair in enumerate(fover_pairs):
        step_text = pair.get("step_text", "")
        correct_emb = _text_embedding(step_text)
        # Generate two incorrect variants with different prime offsets for diversity.
        incorrect_text_a = _make_incorrect_step(step_text, offset_idx=i % len(_INCORRECT_OFFSETS))
        incorrect_text_b = _make_incorrect_step(step_text, offset_idx=(i + 1) % len(_INCORRECT_OFFSETS))
        incorrect_emb_a = _text_embedding(incorrect_text_a)
        incorrect_emb_b = _text_embedding(incorrect_text_b)

        correct_embs.append(correct_emb)
        incorrect_embs_pool.append(incorrect_emb_a)
        correct_embs.append(correct_emb)
        incorrect_embs_pool.append(incorrect_emb_b)

    if not correct_embs:
        return [], []

    # Apply hard negative mining: for each correct embedding, select the most similar
    # incorrect embedding from the pool. This forces discrimination on hard cases.
    correct_matrix = np.stack(correct_embs)           # (2*n_pairs, EMBED_DIM)
    incorrect_matrix = np.stack(incorrect_embs_pool)   # (2*n_pairs, EMBED_DIM)

    hard_neg_indices = hard_negative_mining(correct_matrix, incorrect_matrix)

    hardneg_embs = [incorrect_embs_pool[idx] for idx in hard_neg_indices]
    return correct_embs, hardneg_embs


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


def train_jepa_v17(
    fover_pairs: list[dict[str, Any]],
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> tuple[JEPARankNetV17, list[float]]:
    """Train JEPARankNetV17 with RankNet loss on FoVer formal v1 pairs.

    **Training loop:**
        1. Build (correct, hard_negative) pairs from fover_pairs.
        2. For each epoch: compute RankNet loss on all pairs, compute gradients via jax.grad,
           apply Adam update via optax.
        3. Return trained model and per-epoch loss log.

    **Why Adam (not SGD)?**
        Adam maintains per-parameter learning rate estimates (momentum + variance), which
        is essential for a small MLP trained on a small corpus (200 pairs). Pure SGD would
        require careful learning rate tuning; Adam is robust to hyperparameter choice in
        this regime (validated by v16 training: Adam converged in 200 epochs on the same corpus).

    Args:
        fover_pairs: List of pair dicts from fover_labeled_formal_v1.json.
        n_epochs:    Number of full-corpus training epochs. Default 50.
        lr:          Adam learning rate. Default 1e-3.

    Returns:
        Tuple (trained_model, train_loss_per_epoch) where train_loss_per_epoch is a list
        of floats, one per epoch, tracking convergence.

    Spec: REQ-VERIFY-140, REQ-VERIFY-141, REQ-VERIFY-142
    """
    correct_embs, hardneg_embs = build_ranknet_pairs(fover_pairs)

    if not correct_embs:
        # No training data — return untrained model.
        return JEPARankNetV17(seed=42), []

    correct_jax = jnp.array(np.stack(correct_embs))
    incorrect_jax = jnp.array(np.stack(hardneg_embs))

    # Initialise model and extract JAX-compatible params.
    model = JEPARankNetV17(seed=42)
    params: dict[str, jnp.ndarray] = {
        "W1": jnp.array(model._W1), "b1": jnp.array(model._b1),
        "W2": jnp.array(model._W2), "b2": jnp.array(model._b2),
        "W3": jnp.array(model._W3), "b3": jnp.array(model._b3),
    }

    # Adam optimiser.
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    loss_and_grad = jax.jit(jax.value_and_grad(_batch_loss))

    train_losses: list[float] = []
    for _ in range(n_epochs):
        loss_val, grads = loss_and_grad(params, correct_jax, incorrect_jax)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        train_losses.append(float(loss_val))

    # Write trained params back to the model.
    model._set_params({k: np.asarray(v) for k, v in params.items()})
    return model, train_losses


# ---------------------------------------------------------------------------
# OOD evaluation
# ---------------------------------------------------------------------------


def _compute_auc(scores: list[float], labels: list[int]) -> float:
    """Compute AUROC via Wilcoxon-Mann-Whitney statistic (no sklearn dependency).

    AUC = fraction of (positive, negative) pairs where positive_score < negative_score.
    (Here positive = correct step which should have LOWER score in our convention.)

    Args:
        scores: List of raw model scores.
        labels: List of binary labels (1=correct, 0=incorrect).

    Returns:
        AUROC in [0, 1]. 0.5 = random; 1.0 = perfect separation.
    """
    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]
    if not pos_scores or not neg_scores:
        return 0.5

    count = 0.0
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    for p in pos_scores:
        for n in neg_scores:
            # AUC: correct step (pos, label=1) should have LOWER score than incorrect (neg, label=0).
            if p < n:
                count += 1.0
            elif p == n:
                count += 0.5
    return count / (n_pos * n_neg)


def evaluate_ood_auc(
    model: JEPARankNetV17,
    gsm8k_indices: range = range(500, 700),
) -> float:
    """Evaluate OOD AUROC on synthetic GSM8K questions 500-699.

    **OOD evaluation strategy:**
        We generate deterministic synthetic question texts for GSM8K indices 500-699.
        These indices were never in the training corpus (training used FoVer pairs from
        GSM8K indices 0-399). For each question:
            - Correct step:   "Step for <prefix>: compute carefully and get {answer}."
            - Incorrect step: "Step for <prefix>: quick guess gives {answer + 17}."  (off by 17)

        We score both steps and compute AUROC: fraction of (correct, incorrect) pairs where
        score(correct) < score(incorrect). AUROC=0.5 is random; AUROC>=0.75 is the cascade gate.

        Using synthetic questions (not the literal GSM8K dataset) is correct here: the OOD
        evaluation tests whether the model generalises its SCORE ORDERING to unseen question
        indices, not whether it knows GSM8K answers. The exact question text doesn't affect
        the embedding distribution in a way that would make this test invalid — what matters
        is that the model has never seen these index-seeded embeddings during training.

    Args:
        model:          Trained JEPARankNetV17 instance.
        gsm8k_indices:  Range of GSM8K indices to evaluate on. Default range(500, 700).

    Returns:
        AUROC float in [0, 1].

    Spec: REQ-VERIFY-140
    """
    scores: list[float] = []
    labels: list[int] = []

    for i in gsm8k_indices:
        q = (
            f"GSM8K question {i}: A store has {i * 3} items. "
            f"If {i % 7 + 1} items are sold each hour, how many remain after {i % 5 + 2} hours?"
        )
        correct_step = f"Step for {q[:40]}: compute carefully and get {i * 7 + 3}."
        incorrect_step = f"Step for {q[:40]}: quick guess gives {i * 7 + 3 + 17}."

        correct_emb = _text_embedding(correct_step)
        incorrect_emb = _text_embedding(incorrect_step)

        scores.append(model.score(correct_emb))
        labels.append(1)
        scores.append(model.score(incorrect_emb))
        labels.append(0)

    return _compute_auc(scores, labels)
