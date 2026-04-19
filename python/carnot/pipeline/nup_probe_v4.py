"""NUPProbeV4 — Contrastive-trained hallucination probe maximising energy gap.

**Why NUPProbeV4 exists (RETRO-049):**
    NUPProbe v1 (AUC=0.600) used character entropy.
    NUPProbe v2 (AUC=0.600) added Bayesian semantic entropy — same AUC, no improvement.
    NUPProbe v3 (AUC=0.400) added CLAP cross-layer attention features — worse.

    The root cause identified in RETRO-049: all three versions used binary cross-entropy
    (BCE) as their training objective.  BCE trains a classifier to separate a boundary
    between correct and incorrect steps.  But Carnot's verification signal is NOT a
    boundary — it is an ENERGY GAP: E(incorrect) >> E(correct).  BCE is indifferent to
    the magnitude of that gap; it only cares about which side of the boundary each step
    falls on.  When correct and incorrect steps happen to embed similarly (which is common
    for subtle hallucinations), BCE fails because it cannot distinguish them by boundary
    alone.

**Why contrastive training is the fix:**
    A contrastive (margin-based) loss directly optimises the quantity Carnot cares about:
    the energy gap between incorrect and correct steps.  Specifically, for a pair
    (correct step c, incorrect step i):

        loss(c, i) = max(0, margin - (E(i) - E(c)))

    This is zero when E(i) - E(c) >= margin (the gap is already large enough).
    This is positive when the gap is too small, proportional to the deficit.
    Minimising this loss pushes E(i) UP and E(c) DOWN until the gap is at least `margin`.

    This is the learning objective that directly matches the EBM verification invariant:
    the energy function IS the verification signal, and we need E(incorrect) >> E(correct).
    BCE cannot produce this because it is agnostic to the absolute values of the scores;
    contrastive loss is not.

**Architecture:**
    - TF-IDF-style character n-gram bag-of-features embedding (dimension = energy_dim)
    - Single linear layer: features -> energy scalar
    - Trained with ContrastivePairLoss over all (correct, incorrect) pairs
    - No GPU required — all operations in standard Python/NumPy

    The embedding is intentionally simple.  Exp 503 showed that feature enrichment is
    NOT the bottleneck; the training objective is.  We deliberately keep the embedding
    light and focus the improvement on the loss.

Spec: REQ-VERIFY-109, REQ-VERIFY-110,
      SCENARIO-VERIFY-143, SCENARIO-VERIFY-144, SCENARIO-VERIFY-145
"""

from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# ContrastivePairLoss
# ---------------------------------------------------------------------------


class ContrastivePairLoss:
    """Margin-based contrastive loss that maximises the energy gap between incorrect and correct steps.

    **Why this loss instead of binary cross-entropy:**
        BCE asks: "can we separate correct from incorrect with a boundary?"
        This loss asks: "is E(incorrect) - E(correct) at least `margin`?"

        For EBM verification, the second question is the right one.  The energy function
        IS the verification signal.  We need incorrect steps to have strictly higher
        energy than correct steps, with a guaranteed minimum gap.  BCE does not enforce
        this gap; contrastive loss does.

    **How the margin works:**
        `margin` is the minimum acceptable energy gap.  Recommended: 1.0.
        - If E(i) - E(c) >= 1.0: loss = 0 (no gradient, constraint satisfied).
        - If E(i) - E(c) = 0: loss = 1.0 (maximum push when energies are equal).
        - If E(i) - E(c) = -0.5: loss = 1.5 (worse than equal; loss exceeds margin).

    Args:
        margin: Minimum required energy gap E(incorrect) - E(correct).  Default 1.0.

    Spec: REQ-VERIFY-109, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
    """

    def __init__(self, margin: float = 1.0) -> None:
        self.margin = margin

    def loss(self, energy_incorrect: float, energy_correct: float) -> float:
        """Compute contrastive loss for a single (incorrect, correct) pair.

        **Detailed explanation:**
            Returns max(0, margin - (energy_incorrect - energy_correct)).
            This is hinge loss on the energy gap.  A gap of exactly `margin` yields
            loss=0; any smaller gap yields positive loss proportional to the deficit.

        Args:
            energy_incorrect: Energy assigned to the incorrect (hallucinated) step.
            energy_correct:   Energy assigned to the correct step.

        Returns:
            Non-negative float.  Zero when the gap is sufficient; positive otherwise.

        Spec: REQ-VERIFY-109, SCENARIO-VERIFY-143, SCENARIO-VERIFY-144
        """
        gap = energy_incorrect - energy_correct
        return max(0.0, self.margin - gap)

    def batch_loss(
        self,
        incorrect_energies: List[float],
        correct_energies: List[float],
    ) -> float:
        """Compute mean contrastive loss over a batch of (incorrect, correct) pairs.

        **Detailed explanation:**
            Pairs are matched positionally: incorrect_energies[i] is paired with
            correct_energies[i].  The mean is taken so the loss magnitude is
            independent of batch size.

        Args:
            incorrect_energies: Energy scores for incorrect steps (length N).
            correct_energies:   Energy scores for correct steps (length N).

        Returns:
            Mean loss across all N pairs.  0.0 if lists are empty.

        Spec: REQ-VERIFY-109
        """
        if not incorrect_energies or not correct_energies:
            return 0.0
        n = min(len(incorrect_energies), len(correct_energies))
        total = sum(
            self.loss(incorrect_energies[i], correct_energies[i])
            for i in range(n)
        )
        return total / n


# ---------------------------------------------------------------------------
# NUPProbeV4
# ---------------------------------------------------------------------------


class NUPProbeV4:
    """NUP Probe v4 — contrastive-trained energy probe for CoT step verification.

    **Why this version:**
        See module docstring.  The key change: we abandon BCE in favour of contrastive
        margin loss.  This forces the probe's energy function to assign systematically
        higher energy to incorrect steps, not merely to classify them across a boundary.

    **Embedding approach:**
        Character bigrams extracted from the step text, hashed into `energy_dim` buckets
        (like a feature hashing trick).  Normalised to unit L2 norm.  The output is a
        1D vector of length energy_dim.

        Why bigrams not tokens:
        - No tokeniser required → no external dependencies
        - Bigrams capture local character patterns that correlate with structural
          correctness (arithmetic symbols, numeric sequences, logical connectives)
        - Compatible with the TF-IDF insight from v1: formulaic text has narrow bigram
          distributions; freeform prose has broad ones

    **Energy function:**
        E(step) = dot(weights, encode(step)) + bias
        Weights are initialised to small random values and updated by gradient descent
        via the contrastive loss.  Gradient is computed analytically (no autodiff library
        needed).

    Args:
        energy_dim:   Dimension of the feature embedding.  Default 32.
        margin:       Margin for ContrastivePairLoss.  Default 1.0.
        learning_rate: SGD learning rate.  Default 0.01.
        random_seed:  Seed for weight initialisation.  Default 42.

    Spec: REQ-VERIFY-109, REQ-VERIFY-110,
          SCENARIO-VERIFY-143, SCENARIO-VERIFY-144, SCENARIO-VERIFY-145
    """

    def __init__(
        self,
        energy_dim: int = 32,
        margin: float = 1.0,
        learning_rate: float = 0.01,
        random_seed: int = 42,
    ) -> None:
        self.energy_dim = energy_dim
        self.margin = margin
        self.learning_rate = learning_rate
        self._rng = random.Random(random_seed)
        self._loss_fn = ContrastivePairLoss(margin=margin)

        # Weights and bias initialised to small random values.
        # Small init is important: we want the contrastive loss to drive the
        # weight geometry, not the initialisation.
        self._weights: List[float] = [
            (self._rng.random() - 0.5) * 0.01
            for _ in range(energy_dim)
        ]
        self._bias: float = 0.0

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def encode(self, step_text: str) -> List[float]:
        """Embed a CoT step as a normalised character-bigram feature vector.

        **Detailed explanation:**
            1. Extract all consecutive character pairs (bigrams) from step_text.
            2. Count each bigram (character n-gram frequency).
            3. Hash each bigram into one of `energy_dim` buckets using a simple
               polynomial hash (avoids the need for a bigram vocabulary).
            4. Normalise the resulting vector to unit L2 norm.

            Empty or single-character strings return a zero vector.

            Why normalise: ensures that step length does not dominate the energy score.
            A long correct step and a short correct step should have similar energies
            if their bigram distributions are similar.

        Args:
            step_text: The CoT step text to encode.

        Returns:
            List of floats of length `energy_dim` with L2 norm ~= 1.0.

        Spec: REQ-VERIFY-109
        """
        if len(step_text) < 2:
            return [0.0] * self.energy_dim

        # Count bigrams
        bigram_counts: Counter[str] = Counter()
        for i in range(len(step_text) - 1):
            bigram_counts[step_text[i : i + 2]] += 1

        # Hash into energy_dim buckets
        vec = [0.0] * self.energy_dim
        for bigram, count in bigram_counts.items():
            # Polynomial hash: stable, fast, no external deps
            h = (ord(bigram[0]) * 31 + ord(bigram[1])) % self.energy_dim
            vec[h] += float(count)

        # L2 normalise
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            vec = [x / norm for x in vec]
        return vec

    # ------------------------------------------------------------------
    # Energy scoring
    # ------------------------------------------------------------------

    def score(self, step_text: str) -> float:
        """Compute the energy score for a CoT step.

        **Detailed explanation:**
            Returns dot(weights, encode(step_text)) + bias.
            Higher energy = more likely to be an incorrect (hallucinated) step.
            Lower energy = more likely to be a correct step.

            After contrastive training, the weights will have been pushed so that
            incorrect steps consistently produce higher energy than correct steps
            with a gap of at least `margin`.

        Args:
            step_text: CoT step text.

        Returns:
            Float energy score.  Higher = more likely incorrect.

        Spec: REQ-VERIFY-109
        """
        features = self.encode(step_text)
        return sum(w * f for w, f in zip(self._weights, features)) + self._bias

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_contrastive(
        self,
        correct_steps: List[str],
        incorrect_steps: List[str],
        n_epochs: int = 50,
    ) -> Dict:
        """Train the probe using contrastive margin loss.

        **Detailed explanation:**
            For each epoch:
            1. Enumerate all (correct, incorrect) pairs (Cartesian product).
            2. Compute energy for each step in the pair.
            3. Compute the contrastive loss and its gradient.
            4. Update weights and bias with SGD.

            Gradient of the hinge loss wrt weights:
                If loss > 0 (gap is too small):
                    dL/dw = -(encode(incorrect) - encode(correct))
                    dL/db = -1.0
                    (we want to increase E(incorrect) - E(correct), so we move
                    weights toward encode(incorrect) and away from encode(correct))
                If loss == 0 (gap is sufficient):
                    dL/dw = 0, dL/db = 0 (hinge: no gradient when satisfied)

            The final AUC is computed on the full dataset.

        Args:
            correct_steps:   List of CoT step texts that are factually correct.
            incorrect_steps: List of CoT step texts that are hallucinated/incorrect.
            n_epochs:        Number of full passes over all pairs.

        Returns:
            Dict with keys:
                'converged': bool — True when final mean batch loss < 0.05 or
                             when the loss stopped decreasing across epochs.
                'final_loss': float — mean contrastive loss at the last epoch.
                'final_auc':  float — AUROC on the full dataset after training.
                'loss_history': list[float] — mean loss at each epoch.

        Spec: REQ-VERIFY-109, SCENARIO-VERIFY-145
        """
        if not correct_steps or not incorrect_steps:
            return {
                "converged": False,
                "final_loss": float("inf"),
                "final_auc": 0.5,
                "loss_history": [],
            }

        # Pre-compute embeddings (save work inside the epoch loop)
        correct_enc = [self.encode(s) for s in correct_steps]
        incorrect_enc = [self.encode(s) for s in incorrect_steps]

        loss_history: List[float] = []

        for _epoch in range(n_epochs):
            epoch_loss = 0.0
            n_pairs = 0

            for c_enc in correct_enc:
                for i_enc in incorrect_enc:
                    e_correct = (
                        sum(w * f for w, f in zip(self._weights, c_enc)) + self._bias
                    )
                    e_incorrect = (
                        sum(w * f for w, f in zip(self._weights, i_enc)) + self._bias
                    )
                    pair_loss = self._loss_fn.loss(e_incorrect, e_correct)
                    epoch_loss += pair_loss
                    n_pairs += 1

                    if pair_loss > 0.0:
                        # Hinge gradient: push E(incorrect) up, E(correct) down
                        # dL/dw_j = -(i_enc[j] - c_enc[j])
                        for j in range(self.energy_dim):
                            self._weights[j] -= (
                                self.learning_rate * -(i_enc[j] - c_enc[j])
                            )
                        self._bias -= self.learning_rate * -1.0

            mean_loss = epoch_loss / n_pairs if n_pairs > 0 else 0.0
            loss_history.append(mean_loss)

        final_loss = loss_history[-1] if loss_history else float("inf")

        # Converged if loss is near zero or loss didn't change much at the end
        converged = bool(
            final_loss < 0.05
            or (
                len(loss_history) >= 5
                and abs(loss_history[-1] - loss_history[-5]) < 0.01
            )
        )

        final_auc = self.evaluate_auc(correct_steps, incorrect_steps)

        return {
            "converged": converged,
            "final_loss": final_loss,
            "final_auc": final_auc,
            "loss_history": loss_history,
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_auc(
        self,
        correct_steps: List[str],
        incorrect_steps: List[str],
    ) -> float:
        """Compute AUROC of energy scores: correct steps should have lower energy.

        **Detailed explanation:**
            A perfect probe assigns lower energy to correct steps and higher energy
            to incorrect steps.  AUROC = 1.0 means perfect separation.

            We treat "incorrect" as the positive class (label=1) and "correct" as
            the negative class (label=0).  Higher energy = more likely positive (incorrect).

            AUROC is computed via the standard trapezoidal method on the ROC curve.

        Args:
            correct_steps:   CoT steps known to be correct.
            incorrect_steps: CoT steps known to be incorrect (hallucinated).

        Returns:
            Float in [0.0, 1.0].  0.5 = chance; 1.0 = perfect discrimination.

        Spec: REQ-VERIFY-110, SCENARIO-VERIFY-145
        """
        n_pos = len(incorrect_steps)  # positives = incorrect
        n_neg = len(correct_steps)    # negatives = correct

        if n_pos == 0 or n_neg == 0:
            return 0.5

        # Build scored list: (energy, is_incorrect)
        scored: List[Tuple[float, bool]] = []
        for s in correct_steps:
            scored.append((self.score(s), False))
        for s in incorrect_steps:
            scored.append((self.score(s), True))

        # Sort descending by energy (high energy = predicted incorrect)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build ROC curve and compute AUC via trapezoidal rule
        tp = 0
        fp = 0
        auc = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0

        for _, is_incorrect in scored:
            if is_incorrect:
                tp += 1
            else:
                fp += 1
            fpr = fp / n_neg
            tpr = tp / n_pos
            # Trapezoid: area under the step from prev to current
            if fpr > prev_fpr:
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
            prev_fpr = fpr
            prev_tpr = tpr

        return float(min(1.0, max(0.0, auc)))
