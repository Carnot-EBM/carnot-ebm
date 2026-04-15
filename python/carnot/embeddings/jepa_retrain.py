"""JEPA predictor retraining on real live-GPU violation pairs.

**Researcher summary:**
    Exp 340 produced real LLM responses on 200 GSM8K questions with ground-truth
    correctness labels. This module uses that data to retrain the JEPA predictor
    (ContextPredictionEnergy) on (partial_response, has_violation) pairs, enabling
    *preemptive* verification: predict whether the FINAL response will contain a
    constraint violation from only the FIRST N tokens, so the expensive Ising check
    can be triggered early (or skipped entirely) rather than always running after
    generation completes.

**Why partial prefixes?**
    The JEPA gate sits between token-generation and verification. It must decide
    early — ideally after 50% of the response is generated — whether to queue the
    full Ising check. Training on (prefix, violation_flag) pairs directly teaches
    the model to recognise "this response is heading toward a violation" from early
    signals in the token stream.

**Architecture:**
    1. Each response is word-tokenized and split at ``prefix_fraction`` (default 0.5).
    2. The partial prefix and full response are embedded into JEPA embedding space
       via simple mean-pool of per-word character code vectors (fast, no GPU needed
       for the embedding step itself).
    3. Binary CE loss: treat high energy as "predicts violation = True".
    4. SGD update on model parameters, one mini-batch at a time.
    5. AUC-ROC measures discrimination quality before and after retraining.

Spec: REQ-LEARN-024, SCENARIO-LEARN-041, SCENARIO-LEARN-042
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jrandom

if TYPE_CHECKING:
    from carnot.embeddings.jepa_energy import ContextPredictionEnergy


# ---------------------------------------------------------------------------
# ViolationPair dataclass
# ---------------------------------------------------------------------------


@dataclass
class ViolationPair:
    """One training example for JEPA real-data retrain.

    **Researcher summary:**
        Holds the prefix (first N tokens) of a model response alongside the
        ground-truth flag indicating whether the FULL response violates a constraint.
        The JEPA model is trained to predict ``has_violation`` from ``partial_response``.

    **Detailed explanation:**
        ``partial_response`` is obtained by word-tokenizing the full response and
        keeping only the first ``round(prefix_fraction * n_words)`` words. This
        simulates what the JEPA gate would observe mid-generation.

        ``has_violation`` is the complement of the ground-truth correctness label
        from Exp 340: if the model's answer was wrong, the response contains a
        violation of the "correct reasoning" constraint.

    Attributes:
        partial_response: First N words of the response (space-joined), where N is
            determined by ``prefix_fraction`` in ``extract_violation_pairs``.
        full_response: Complete model response string.
        has_violation: True when the response contains a constraint violation
            (i.e., the model answered incorrectly).
        model_id: Identifier of the generating LLM (e.g. "gemma4-e4b-it").
        question_id: Identifier of the source question (e.g. "gsm8k_q042").

    Spec: REQ-LEARN-024-1
    """

    partial_response: str
    full_response: str
    has_violation: bool
    model_id: str
    question_id: str


# ---------------------------------------------------------------------------
# extract_violation_pairs
# ---------------------------------------------------------------------------


def extract_violation_pairs(
    live_results: dict | None,
    prefix_fraction: float = 0.5,
) -> list[ViolationPair]:
    """Build ViolationPair list from Exp 340 live results or synthetic fallback.

    **Researcher summary:**
        If live results are available, split each response at ``prefix_fraction``
        and tag it with the correctness label. If not (CI or pre-GPU), return 50
        deterministic synthetic pairs so all downstream code is exercised.

    **Detailed explanation for engineers:**
        Live results dict expected structure (Exp 340 format):

        ```json
        {
          "responses": [
            {
              "question_id": "gsm8k_q001",
              "model_id": "gemma4-e4b-it",
              "response": "Janet sells 16 eggs ...",
              "correct": true
            },
            ...
          ]
        }
        ```

        Word tokenization is a simple ``str.split()`` — no sentencepiece or BPE.
        This is intentional: the JEPA predictor receives character-level structure
        signals, not subword tokens, so word splitting is the right granularity.

        The prefix split uses ``max(1, round(prefix_fraction * n_words))`` to ensure
        at least one word is always present in the prefix even for very short responses.

        **Synthetic fallback:**
        When ``live_results`` is None or has an empty or missing ``"responses"`` list,
        50 synthetic pairs are generated with a fixed seed (42) for reproducibility.
        Half the pairs are marked as violations; the other half are not. This lets CI
        verify coverage of all code paths without requiring a live GPU run.

    Args:
        live_results: Dict from Exp 340 JSON (must have ``"responses"`` list), or
            ``None`` to trigger the synthetic fallback.
        prefix_fraction: Fraction of words to keep for the partial prefix.
            Must be in (0, 1]. Default 0.5 (first half).

    Returns:
        List of ViolationPair objects. Length equals the number of responses in
        ``live_results``, or 50 if the synthetic fallback is used.

    Spec: REQ-LEARN-024-2, REQ-LEARN-024-3, SCENARIO-LEARN-041, SCENARIO-LEARN-042
    """
    if prefix_fraction <= 0 or prefix_fraction > 1:
        raise ValueError(f"prefix_fraction must be in (0, 1], got {prefix_fraction}")

    # ---- Synthetic fallback (CI-safe, no GPU data required) ----
    responses = None
    if live_results is not None:
        responses = live_results.get("responses") or []

    if not responses:
        return _make_synthetic_pairs(n=50, seed=42)

    # ---- Real live data path ----
    pairs: list[ViolationPair] = []
    for entry in responses:
        full_response: str = entry.get("response", "") or ""
        words = full_response.split()
        n_words = len(words)
        prefix_len = max(1, round(prefix_fraction * n_words))
        partial = " ".join(words[:prefix_len])

        correct: bool = bool(entry.get("correct", False))
        has_violation: bool = not correct

        pairs.append(
            ViolationPair(
                partial_response=partial,
                full_response=full_response,
                has_violation=has_violation,
                model_id=str(entry.get("model_id", "unknown")),
                question_id=str(entry.get("question_id", "unknown")),
            )
        )

    return pairs


def _make_synthetic_pairs(n: int = 50, seed: int = 42) -> list[ViolationPair]:
    """Generate deterministic synthetic ViolationPairs for CI and unit tests.

    **For engineers:**
        Uses a simple LCG-style index arithmetic (not JAX, to avoid import cost at
        module load time) to produce varied synthetic responses. The first ``n // 2``
        pairs are violations; the rest are non-violations. This balanced split
        ensures the retrainer and AUC evaluator encounter both classes in any test.

    Args:
        n: Number of pairs to generate. Default 50.
        seed: Integer seed for determinism. Default 42.

    Returns:
        List of exactly ``n`` ViolationPair objects.

    Spec: REQ-LEARN-024-3
    """
    pairs: list[ViolationPair] = []
    half = n // 2

    for i in range(n):
        # Deterministic pseudo-random text based on index and seed
        idx = (seed * 31 + i * 17) % 997
        words = [f"word{(idx + j * 7) % 100}" for j in range(10 + (i % 5))]
        full = " ".join(words)
        partial = " ".join(words[: max(1, len(words) // 2)])
        has_viol = i < half

        pairs.append(
            ViolationPair(
                partial_response=partial,
                full_response=full,
                has_violation=has_viol,
                model_id="synthetic_model",
                question_id=f"synthetic_q{i:03d}",
            )
        )

    return pairs


# ---------------------------------------------------------------------------
# Text → embedding helper
# ---------------------------------------------------------------------------


def _text_to_embedding(text: str, embed_dim: int = 64) -> jax.Array:
    """Convert a text string to a fixed-size embedding vector via character-code mean pooling.

    **For engineers:**
        This is a fast, dependency-free embedding that requires no model load.
        Each word is encoded as its mean character unicode code point, normalised
        by 128 to keep values in [0, ~1). The per-word scalars are then projected
        to ``embed_dim`` dimensions by tiling and applying a simple sinusoidal
        positional weighting.

        This is NOT a quality text embedding — it is a lightweight, deterministic
        feature extractor that provides enough signal for the JEPA retrainer to
        learn from real data in unit-test and CI settings. In production, swap
        this out for a proper sentence embedding (e.g., ``all-MiniLM-L6-v2``).

    Args:
        text: Input string (may be empty).
        embed_dim: Output dimensionality. Must be >= 1.

    Returns:
        JAX array of shape (embed_dim,).
    """
    if not text:
        return jnp.zeros(embed_dim)

    words = text.split()

    # Compute per-word scalar: mean char code / 128
    # str.split() on a non-empty string always yields at least one non-empty token,
    # so we iterate directly without guarding for empty words.
    word_scalars: list[float] = [
        sum(ord(c) for c in w) / (len(w) * 128.0) for w in words
    ]

    # Aggregate: mean of per-word scalars
    mean_val = sum(word_scalars) / len(word_scalars)

    # Project to embed_dim with sinusoidal spread (gives each dimension a distinct signal)
    dims = jnp.arange(embed_dim, dtype=jnp.float32)
    # Even dims: cosine; odd dims: sine. Frequency varies across dims.
    freqs = (dims + 1.0) * math.pi / embed_dim
    emb = jnp.where(dims % 2 == 0, jnp.cos(freqs * mean_val), jnp.sin(freqs * mean_val))

    return emb


# ---------------------------------------------------------------------------
# JEPARetrainer
# ---------------------------------------------------------------------------


class JEPARetrainer:
    """Retrain a ContextPredictionEnergy JEPA model on real violation pairs.

    **Researcher summary:**
        Wraps a ContextPredictionEnergy model with a binary-CE loss and a simple
        SGD update loop. Given a list of ViolationPair objects, converts each
        partial_response to an embedding pair (context=partial, prediction=full)
        and trains the model to output HIGH energy when has_violation=True and
        LOW energy when has_violation=False.

    **Why binary cross-entropy on energy?**
        The JEPA model outputs a scalar energy. We interpret high energy as a signal
        that the prediction is incoherent (i.e., heading toward a violation).
        Binary CE loss with sigmoid on the negated energy provides a smooth, bounded
        gradient signal:

            p(violation) = sigmoid(energy)
            loss = -y * log(p) - (1-y) * log(1-p)

        where y=1 for violation, y=0 for non-violation.

    **Why not NCE here?**
        NCE was designed for the contrastive case where we have explicit noise pairs.
        Here we have binary labels from ground truth, so supervised BCE is more
        direct and better calibrated.

    Args:
        jepa_model: A ContextPredictionEnergy instance to retrain (mutated in-place).
        lr: SGD learning rate. Default 1e-4.
        embed_dim: Embedding dimensionality fed to the JEPA model. Must match
            ``jepa_model.config.embed_dim``.

    Spec: REQ-LEARN-024-4, REQ-LEARN-024-5, REQ-LEARN-024-6, REQ-LEARN-024-7
    """

    def __init__(
        self,
        jepa_model: "ContextPredictionEnergy",
        lr: float = 1e-4,
    ) -> None:
        self.model = jepa_model
        self.lr = lr
        self._embed_dim: int = jepa_model.config.embed_dim

    # ------------------------------------------------------------------
    # binary_ce_loss
    # ------------------------------------------------------------------

    def binary_ce_loss(
        self,
        predicted_energy: float,
        has_violation: bool,
    ) -> float:
        """Compute binary cross-entropy loss for one (energy, label) pair.

        **For engineers:**
            Treats ``sigmoid(predicted_energy)`` as p(violation).
            Loss = -y * log(sigmoid(E)) - (1-y) * log(1 - sigmoid(E))
            where y = 1.0 if has_violation else 0.0.

            A small epsilon (1e-7) prevents log(0).

        Args:
            predicted_energy: Scalar energy output from the JEPA model.
            has_violation: Ground-truth label (True = violation present).

        Returns:
            Scalar BCE loss.

        Spec: REQ-LEARN-024-5
        """
        eps = 1e-7
        e = float(predicted_energy)
        y = 1.0 if has_violation else 0.0
        p = 1.0 / (1.0 + math.exp(-e))  # sigmoid
        p = max(eps, min(1.0 - eps, p))
        return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))

    # ------------------------------------------------------------------
    # train_epoch
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        pairs: list[ViolationPair],
        batch_size: int = 8,
    ) -> float:
        """Train for one epoch over the violation pairs and return the mean loss.

        **For engineers:**
            Converts each ViolationPair to a (context_emb, pred_emb) embedding pair,
            computes binary CE loss, and performs a JAX-based SGD update on model
            parameters. Mini-batches are processed sequentially (no shuffling —
            for reproducibility in tests).

            The parameter update mirrors the approach in ``train_jepa_energy``:
            extract parameters as explicit arrays, use jax.value_and_grad on a
            pure function, then write back updated values.

        Args:
            pairs: List of ViolationPair training examples.
            batch_size: Mini-batch size. Default 8.

        Returns:
            Mean BCE loss over all pairs in the epoch.

        Spec: REQ-LEARN-024-6
        """
        if not pairs:
            return 0.0

        total_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(pairs), batch_size):
            batch = pairs[batch_start : batch_start + batch_size]
            batch_loss = self._train_batch(batch)
            total_loss += batch_loss * len(batch)
            n_batches += 1

        return total_loss / len(pairs)

    def _train_batch(self, batch: list[ViolationPair]) -> float:
        """Update model parameters on one mini-batch; return mean batch loss.

        **For engineers:**
            Builds context and prediction embedding arrays for the batch, computes
            a batched BCE loss, and performs a single SGD step. All JAX operations
            are on CPU (respects JAX_PLATFORMS=cpu).

        Spec: REQ-LEARN-024-6
        """
        from carnot.models.gibbs import _apply_activation  # noqa: PLC0415

        activation = self.model.config.activation

        # Build embedding arrays for the batch
        ctx_embs = jnp.stack(
            [_text_to_embedding(p.partial_response, self._embed_dim) for p in batch]
        )  # (B, embed_dim)
        pred_embs = jnp.stack(
            [_text_to_embedding(p.full_response, self._embed_dim) for p in batch]
        )  # (B, embed_dim)
        pairs_arr = jnp.concatenate([ctx_embs, pred_embs], axis=1)  # (B, 2*embed_dim)
        labels = jnp.array([1.0 if p.has_violation else 0.0 for p in batch])  # (B,)

        # Extract parameters for jax.grad
        weights = [w for w, _b in self.model.layers]
        biases = [b for _w, b in self.model.layers]

        def _loss_fn(
            layer_weights: list[jax.Array],
            layer_biases: list[jax.Array],
            output_weight: jax.Array,
            output_bias: jax.Array,
        ) -> jax.Array:
            """Pure function: batched BCE loss over the current mini-batch."""
            eps = 1e-7

            def _energy_single(x: jax.Array) -> jax.Array:
                h = x
                for w, b in zip(layer_weights, layer_biases):
                    h = _apply_activation(w @ h + b, activation)
                return output_weight @ h + output_bias

            energies = jax.vmap(_energy_single)(pairs_arr)  # (B,)
            p_viol = jax.nn.sigmoid(energies)  # (B,) probability of violation
            p_viol = jnp.clip(p_viol, eps, 1.0 - eps)
            bce = -(labels * jnp.log(p_viol) + (1.0 - labels) * jnp.log(1.0 - p_viol))
            return jnp.mean(bce)

        loss_val, grads = jax.value_and_grad(_loss_fn, argnums=(0, 1, 2, 3))(
            weights, biases, self.model.output_weight, jnp.array(self.model.output_bias)
        )

        grad_weights, grad_biases, grad_ow, grad_ob = grads

        # SGD update
        new_layers = []
        for i in range(len(self.model.layers)):
            new_w = weights[i] - self.lr * grad_weights[i]
            new_b = biases[i] - self.lr * grad_biases[i]
            new_layers.append((new_w, new_b))
        self.model.layers = new_layers
        self.model.output_weight = self.model.output_weight - self.lr * grad_ow
        self.model.output_bias = float(
            jnp.array(self.model.output_bias) - self.lr * grad_ob
        )

        return float(loss_val)

    # ------------------------------------------------------------------
    # evaluate_auc_roc
    # ------------------------------------------------------------------

    def evaluate_auc_roc(self, pairs: list[ViolationPair]) -> float:
        """Compute AUC-ROC of the model's violation predictions on the given pairs.

        **For engineers:**
            AUC-ROC (Area Under the Receiver Operating Characteristic Curve) measures
            how well the model discriminates violations from non-violations, regardless
            of threshold. AUC=0.5 is random; AUC=1.0 is perfect.

            We use a pure-NumPy trapezoidal approximation to avoid a hard sklearn
            dependency. The procedure:
            1. Compute energy for every pair.
            2. Sort by energy (descending — high energy = predicted violation).
            3. Walk the threshold from high to low, accumulating TPR and FPR.
            4. Compute area via the trapezoid rule.

            Edge cases:
            - All pairs have the same label → AUC is undefined; return 0.5.
            - Zero pairs → return 0.5.

        Args:
            pairs: List of ViolationPair test examples.

        Returns:
            AUC-ROC in [0, 1]. 0.5 = random baseline.

        Spec: REQ-LEARN-024-7
        """
        if not pairs:
            return 0.5

        energies: list[float] = []
        labels: list[int] = []
        for p in pairs:
            ctx_emb = _text_to_embedding(p.partial_response, self._embed_dim)
            pred_emb = _text_to_embedding(p.full_response, self._embed_dim)
            e = float(self.model.energy_pair(ctx_emb, pred_emb))
            energies.append(e)
            labels.append(1 if p.has_violation else 0)

        # Check for degenerate cases
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Sort by energy descending (high energy = predicts violation)
        scored = sorted(zip(energies, labels), key=lambda x: x[0], reverse=True)

        tpr_pts = [0.0]
        fpr_pts = [0.0]
        tp = 0
        fp = 0

        for _e, lab in scored:
            if lab == 1:
                tp += 1
            else:
                fp += 1
            tpr_pts.append(tp / n_pos)
            fpr_pts.append(fp / n_neg)

        # Trapezoidal AUC
        auc = 0.0
        for i in range(1, len(fpr_pts)):
            dfpr = fpr_pts[i] - fpr_pts[i - 1]
            auc += dfpr * (tpr_pts[i] + tpr_pts[i - 1]) / 2.0

        return float(auc)


# ---------------------------------------------------------------------------
# build_retrain_artifact
# ---------------------------------------------------------------------------


def build_retrain_artifact(
    before_auc: float,
    after_auc: float,
    n_pairs: int,
) -> dict:
    """Build a summary artifact for the JEPA real-data retrain experiment.

    **For engineers:**
        This is the metadata dict that gets merged into the ExperimentTemplate
        artifact. It captures the key metrics for comparing JEPA predictor
        quality before and after retraining on live violation pairs.

        ``auc_improvement`` is signed: positive means the retrain improved
        discrimination; negative means it degraded. Both values are reported
        honestly to support downstream analysis.

    Args:
        before_auc: AUC-ROC on the test split BEFORE retraining.
        after_auc: AUC-ROC on the test split AFTER retraining.
        n_pairs: Total number of (partial_response, has_violation) pairs used.

    Returns:
        Dict with keys: ``before_auc``, ``after_auc``, ``auc_improvement``,
        ``n_pairs``, ``schema_version``.

    Spec: REQ-LEARN-024-8
    """
    return {
        "before_auc": round(float(before_auc), 6),
        "after_auc": round(float(after_auc), 6),
        "auc_improvement": round(float(after_auc) - float(before_auc), 6),
        "n_pairs": int(n_pairs),
        "schema_version": "carnot.jepa_retrain.v1",
    }
