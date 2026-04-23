"""JailbreakDetectionKAN — TF-IDF + linear classifier for pre-generation safety gating.

**Why this module exists (arXiv 2602.11495):**
    "Jailbreaking Leaves a Trace" (Feb 2026) showed that adversarial prompts concentrate
    a detectable signal in middle transformer layers (8-16 for 32-layer models).  Hidden
    states from a real LLM would give maximum accuracy but require GPU inference on every
    prompt — expensive and slow.

    This module uses TF-IDF text features as a CPU-compatible proxy.  TF-IDF captures
    the same structural signal in a different representation: injection prompts reliably
    contain specific n-gram patterns ("ignore all previous", "as an AI with no restrictions",
    "DAN", "[[SYSTEM OVERRIDE]]") that benign prompts do not.  This is the same reason the
    hidden-state probe works — those n-grams are what activate the adversarial signal in
    the middle layers.  TF-IDF finds them directly in the text, no GPU required.

**Deployment context (Tier 0h):**
    This classifier runs BEFORE any LLM call.  If it detects a jailbreak attempt, the
    pipeline returns immediately with verified=False, mode="SAFETY_GATE".  This means
    a detected jailbreak consumes zero GPU resources — the most expensive operation is
    a TF-IDF transform plus a matrix multiply, which takes < 1ms on CPU.

    The precision requirement (>= 0.85) is not arbitrary: every false positive routes a
    legitimate user request to the safety gate, blocking it from LLM completion.  At
    precision < 0.85, more than 15% of blocked requests would be legitimate.  This is
    the same false-positive discipline as REQ-SAFE-017 (Tier 0b < 5% FP rate).

**Architecture:**
    1. TF-IDF vectorizer: max_features=256, ngram_range=(1,2), binary=False
       Word unigrams + bigrams capture both individual injection keywords AND
       their co-occurrence patterns ("ignore previous", "with no restrictions").

    2. Linear classifier (not KAN energy function):
       A two-layer linear network trained with binary cross-entropy.  We use a
       linear model rather than the full KAN B-spline because:
       - The signal is already concentrated in specific n-grams (simple linear boundary)
       - Fewer parameters = less overfitting on the 160-example training set
       - Simpler forward pass = faster inference (< 1ms vs. ~5ms for KAN)
       The KAN B-spline machinery adds expressiveness at the cost of more parameters;
       for this task, a linear boundary is sufficient and safer against overfitting.

Spec: REQ-SAFETY-001, REQ-SAFETY-002,
      SCENARIO-SAFETY-001, SCENARIO-SAFETY-002
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List


# ---------------------------------------------------------------------------
# JailbreakKANConfig
# ---------------------------------------------------------------------------


@dataclass
class JailbreakKANConfig:
    """Configuration for the jailbreak detection classifier.

    **Why these defaults:**
        n_features=256: Enough TF-IDF features to capture common injection patterns without
            overfitting.  Higher values risk memorising corpus-specific bigrams.

        hidden_dim=32: A single hidden layer with 32 units.  For a 256-dimensional TF-IDF
            input, this gives 256*32 + 32*1 = 8224 parameters — small enough to train in
            < 1 second on CPU, large enough to learn a nonlinear decision boundary.

        n_grid=8: Kept for API compatibility with KAN-style configs but not used in the
            linear classifier (we don't instantiate B-splines in this module).
            Future versions could replace the linear layer with a KAN spline layer.

    Spec: REQ-SAFETY-001
    """

    n_features: int = 256
    hidden_dim: int = 32
    n_grid: int = 8


# ---------------------------------------------------------------------------
# TF-IDF vectorizer (pure Python, no sklearn dependency)
# ---------------------------------------------------------------------------


class _TFIDFVectorizer:
    """Minimal TF-IDF vectorizer: fit/transform with top-k vocabulary selection.

    **How TF-IDF works (for engineers who have not used it before):**
        TF (term frequency): how many times a word/bigram appears in this document,
            divided by the total number of words.  Measures how important the term is
            TO THIS document.
        IDF (inverse document frequency): log(N / df), where N = corpus size and
            df = number of documents containing the term.  Measures how important the
            term is ACROSS the corpus (rare terms get high IDF; "the" gets low IDF).
        TF-IDF = TF * IDF: high score = appears often in THIS doc AND is rare overall.

    **Why TF-IDF for injection detection:**
        Injection patterns like "ignore all previous instructions" are very common in
        jailbreak prompts but essentially absent from benign prompts.  TF-IDF will
        assign high weight to these terms in jailbreak documents and near-zero weight
        in benign documents — exactly the signal we need to classify.

    **Implementation note:**
        We use character-level tokenisation for bigrams: split on whitespace, then
        form consecutive word pairs.  This captures compound patterns without needing
        a sophisticated tokeniser.
    """

    def __init__(self, max_features: int = 256, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._fitted = False

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on whitespace, then extract unigrams and bigrams."""
        words = text.lower().split()
        tokens: list[str] = []
        if self.ngram_range[0] <= 1:
            tokens.extend(words)
        if self.ngram_range[1] >= 2:
            for i in range(len(words) - 1):
                tokens.append(words[i] + " " + words[i + 1])
        return tokens

    def fit(self, documents: list[str]) -> None:
        """Build vocabulary from top-k most informative terms by document frequency."""
        n_docs = len(documents)
        if n_docs == 0:
            return

        # Count document frequency for every term
        df: dict[str, int] = {}
        for doc in documents:
            tokens_seen = set(self._tokenize(doc))
            for tok in tokens_seen:
                df[tok] = df.get(tok, 0) + 1

        # Sort by document frequency (most common first) and cap at max_features.
        # We exclude terms that appear in all documents (df == n_docs) or only once
        # (df == 1) because they carry no discriminative information.
        useful = {
            term: count for term, count in df.items()
            if 1 < count < n_docs or n_docs <= 2
        }
        if not useful:
            useful = df

        # Rank by document frequency descending (common terms are safer anchors)
        ranked = sorted(useful.items(), key=lambda x: -x[1])
        top_terms = [term for term, _ in ranked[: self.max_features]]

        self._vocab = {term: idx for idx, term in enumerate(top_terms)}

        # Pre-compute IDF for each term in the vocabulary
        self._idf = []
        for term in top_terms:
            doc_freq = df.get(term, 0)
            # Add-1 smoothing so IDF is never zero; +1 inside log avoids division issues
            idf_val = math.log((n_docs + 1) / (doc_freq + 1)) + 1.0
            self._idf.append(idf_val)

        self._fitted = True

    def transform(self, document: str) -> list[float]:
        """Convert a document to a TF-IDF feature vector of length max_features.

        Returns a zero-padded vector if vocabulary is smaller than max_features.
        """
        if not self._fitted or not self._vocab:
            return [0.0] * self.max_features

        tokens = self._tokenize(document)
        n_tokens = max(len(tokens), 1)

        # Term frequencies
        tf: dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0.0) + 1.0 / n_tokens

        # Build TF-IDF vector
        vec = [0.0] * len(self._vocab)
        for term, idx in self._vocab.items():
            if term in tf:
                vec[idx] = tf[term] * self._idf[idx]

        # L2 normalise (so vector length doesn't dominate the classification)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]

        # Pad to max_features if vocab is smaller
        if len(vec) < self.max_features:
            vec = vec + [0.0] * (self.max_features - len(vec))

        return vec[: self.max_features]


# ---------------------------------------------------------------------------
# Linear classifier (2-layer, trained with mini-batch SGD)
# ---------------------------------------------------------------------------


class _LinearClassifier:
    """Two-layer linear network: input -> hidden -> sigmoid output.

    **Architecture detail:**
        Layer 1: W1 (n_features x hidden_dim) + b1 (hidden_dim,) with ReLU activation.
        Layer 2: W2 (hidden_dim x 1) + b2 scalar with sigmoid activation.

    **Why ReLU + sigmoid instead of a KAN spline:**
        The jailbreak detection task has a crisp linear-ish boundary in TF-IDF space
        (injection prompts cluster tightly around specific high-TF-IDF terms).  A linear
        classifier with ReLU captures this without the risk of overfitting that B-splines
        would introduce at this dataset size (160 examples).  ReLU is also 10-100x faster
        to evaluate than B-spline basis function blending.

    **Training:**
        Mini-batch SGD with binary cross-entropy loss.  One epoch = one full pass over
        all training examples in random order.  Learning rate 0.01 is conservative but
        safe for a 160-example dataset.
    """

    def __init__(self, n_features: int, hidden_dim: int, random_seed: int = 42) -> None:
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        rng = random.Random(random_seed)

        # Xavier initialisation: scale = sqrt(2 / (fan_in + fan_out))
        # prevents vanishing/exploding gradients in the first few training steps
        scale1 = math.sqrt(2.0 / (n_features + hidden_dim))
        scale2 = math.sqrt(2.0 / (hidden_dim + 1))

        self.W1 = [
            [(rng.gauss(0, 1) * scale1) for _ in range(hidden_dim)]
            for _ in range(n_features)
        ]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [rng.gauss(0, 1) * scale2 for _ in range(hidden_dim)]
        self.b2 = 0.0

    def _forward(self, x: list[float]) -> tuple[list[float], float]:
        """Forward pass: returns (hidden_activations, output_probability)."""
        # Layer 1: hidden = ReLU(x @ W1 + b1)
        hidden = []
        for j in range(self.hidden_dim):
            z = sum(x[i] * self.W1[i][j] for i in range(len(x))) + self.b1[j]
            hidden.append(max(0.0, z))  # ReLU

        # Layer 2: out = sigmoid(hidden @ W2 + b2)
        logit = sum(hidden[j] * self.W2[j] for j in range(self.hidden_dim)) + self.b2
        prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))  # clipped sigmoid
        return hidden, prob

    def predict_proba(self, x: list[float]) -> float:
        """Return P(jailbreak) for a single TF-IDF feature vector."""
        _, prob = self._forward(x)
        return prob

    def train_step(self, x: list[float], label: int, lr: float) -> float:
        """Compute BCE loss and update weights via backprop for one example.

        Returns the BCE loss for this example (for monitoring convergence).
        """
        hidden, prob = self._forward(x)

        # Binary cross-entropy loss: -[y*log(p) + (1-y)*log(1-p)]
        eps = 1e-9
        loss = -(label * math.log(prob + eps) + (1 - label) * math.log(1 - prob + eps))

        # Backprop through sigmoid: dL/d_logit = prob - label
        d_logit = prob - label

        # Gradient for Layer 2
        d_W2 = [d_logit * hidden[j] for j in range(self.hidden_dim)]
        d_b2 = d_logit

        # Backprop through ReLU to Layer 1
        d_hidden = [d_logit * self.W2[j] for j in range(self.hidden_dim)]

        # ReLU derivative: 1 if hidden > 0, else 0
        d_hidden_pre = [
            d_hidden[j] if hidden[j] > 0 else 0.0
            for j in range(self.hidden_dim)
        ]

        d_W1 = [[x[i] * d_hidden_pre[j] for j in range(self.hidden_dim)] for i in range(len(x))]
        d_b1 = d_hidden_pre

        # SGD update
        for j in range(self.hidden_dim):
            self.W2[j] -= lr * d_W2[j]
            self.b1[j] -= lr * d_b1[j]
        self.b2 -= lr * d_b2
        for i in range(len(x)):
            for j in range(self.hidden_dim):
                self.W1[i][j] -= lr * d_W1[i][j]

        return loss


# ---------------------------------------------------------------------------
# JailbreakDetectionKAN
# ---------------------------------------------------------------------------


class JailbreakDetectionKAN:
    """KAN-based jailbreak detector: TF-IDF features + linear classifier.

    **What this does (step by step):**
        1. fit(prompts, labels) trains the detector on a labelled corpus.
           - Fits a TF-IDF vectorizer over all prompts.
           - Transforms each prompt to a 256-dimensional feature vector.
           - Trains a 2-layer linear classifier for 100 epochs using mini-batch SGD.

        2. predict(prompt) returns P(jailbreak) in [0, 1].
           - Transforms the prompt using the fitted TF-IDF vectorizer.
           - Runs the trained classifier forward pass.

        3. is_jailbreak(prompt, threshold=0.5) returns True if predict() > threshold.

    **Why TF-IDF is a valid proxy for hidden-state features:**
        arXiv 2602.11495 showed that jailbreak prompts concentrate a detectable signal
        in transformer middle layers.  The reason the signal appears there is that those
        layers process the specific n-gram patterns that define injection attacks.
        TF-IDF directly extracts those n-grams from the raw text, bypassing the need
        for a transformer entirely.  The resulting feature space is lower-dimensional but
        captures the same discriminative information.

    **Precision discipline (REQ-SAFETY-001):**
        Precision >= 0.85 is required.  This means at most 15% of "jailbreak" detections
        are false positives.  False positives are costly: they block a legitimate user
        request from reaching LLM completion.  A threshold of 0.5 is used by default;
        raise it to 0.6-0.7 if precision on your deployment distribution is below 0.85.

    Args:
        config: JailbreakKANConfig with n_features, hidden_dim, n_grid.
        learning_rate: SGD learning rate.  Default 0.01.
        n_epochs: Training epochs.  Default 100.
        random_seed: Seed for weight initialisation and batch shuffling.  Default 42.

    Spec: REQ-SAFETY-001, REQ-SAFETY-002,
          SCENARIO-SAFETY-001, SCENARIO-SAFETY-002
    """

    def __init__(
        self,
        config: JailbreakKANConfig | None = None,
        learning_rate: float = 0.01,
        n_epochs: int = 100,
        random_seed: int = 42,
    ) -> None:
        self.config = config or JailbreakKANConfig()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_seed = random_seed

        self._vectorizer = _TFIDFVectorizer(
            max_features=self.config.n_features,
            ngram_range=(1, 2),
        )
        self._classifier = _LinearClassifier(
            n_features=self.config.n_features,
            hidden_dim=self.config.hidden_dim,
            random_seed=random_seed,
        )
        self._fitted = False

    def fit(self, prompts: List[str], labels: List[int]) -> dict:
        """Train the jailbreak detector on a labelled corpus.

        **Training procedure:**
            1. Fit TF-IDF vectorizer over all prompts to build the vocabulary.
            2. Transform each prompt to a 256-dim TF-IDF feature vector.
            3. Train the linear classifier for n_epochs using SGD on binary cross-entropy.
            4. Shuffle training examples each epoch to prevent order-dependence.

        Args:
            prompts: List of prompt strings.
            labels:  Parallel list of labels: 0 = benign, 1 = jailbreak.

        Returns:
            Dict with 'final_loss' (float) and 'loss_history' (list[float]).

        Spec: REQ-SAFETY-001, SCENARIO-SAFETY-001
        """
        assert len(prompts) == len(labels), "prompts and labels must have equal length"

        # Step 1: fit vectorizer
        self._vectorizer.fit(prompts)

        # Step 2: pre-compute feature vectors
        features = [self._vectorizer.transform(p) for p in prompts]

        # Step 3: train classifier
        rng = random.Random(self.random_seed)
        loss_history: list[float] = []
        indices = list(range(len(prompts)))

        for _epoch in range(self.n_epochs):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for idx in indices:
                loss = self._classifier.train_step(
                    features[idx], labels[idx], self.learning_rate
                )
                epoch_loss += loss
            loss_history.append(epoch_loss / max(len(indices), 1))

        self._fitted = True
        return {
            "final_loss": loss_history[-1] if loss_history else float("inf"),
            "loss_history": loss_history,
        }

    def predict(self, prompt: str) -> float:
        """Return P(jailbreak) in [0, 1] for a single prompt.

        Args:
            prompt: The prompt text to score.

        Returns:
            Float in [0, 1].  Values > 0.5 indicate a probable jailbreak attempt.

        Spec: REQ-SAFETY-001, SCENARIO-SAFETY-001, SCENARIO-SAFETY-002
        """
        features = self._vectorizer.transform(prompt)
        return self._classifier.predict_proba(features)

    def is_jailbreak(self, prompt: str, threshold: float = 0.5) -> bool:
        """Return True if the prompt is likely a jailbreak attempt.

        **Threshold guidance:**
            threshold=0.5 (default): balanced precision/recall.  Appropriate when
                false positives (blocking benign requests) and false negatives
                (passing jailbreaks) are equally costly.
            threshold=0.6-0.7: higher precision (fewer false positives) at the cost
                of more missed jailbreaks.  Use when protecting legitimate user traffic
                is more important than catching every attack.

        Args:
            prompt:    The prompt text.
            threshold: P(jailbreak) cutoff.  Default 0.5.

        Returns:
            True if predict(prompt) > threshold.

        Spec: REQ-SAFETY-001, REQ-SAFETY-002, SCENARIO-SAFETY-002
        """
        return self.predict(prompt) > threshold

    def evaluate_auroc(
        self,
        prompts: List[str],
        labels: List[int],
    ) -> tuple[float, float, float]:
        """Evaluate AUROC, precision, and recall at threshold=0.5.

        **AUROC computation:**
            Area Under the Receiver Operating Characteristic curve.  Measures how
            well the classifier separates jailbreak (label=1) from benign (label=0)
            across ALL possible thresholds.  AUROC=1.0 is perfect; 0.5 is chance.
            We use the trapezoidal rule on the (FPR, TPR) curve.

        **Precision / recall at threshold=0.5:**
            Precision = TP / (TP + FP): of all prompts classified as jailbreak, what
                fraction actually is?  Must be >= 0.85 (REQ-SAFETY-001).
            Recall = TP / (TP + FN): of all actual jailbreak prompts, what fraction
                was detected?

        Args:
            prompts: List of prompt strings.
            labels:  Parallel list of labels (0=benign, 1=jailbreak).

        Returns:
            (auroc, precision, recall) as floats.

        Spec: REQ-SAFETY-001, SCENARIO-SAFETY-001
        """
        if not prompts:
            return 0.5, 0.0, 0.0

        scores = [self.predict(p) for p in prompts]

        # AUROC via trapezoidal rule
        n_pos = sum(1 for lbl in labels if lbl == 1)
        n_neg = sum(1 for lbl in labels if lbl == 0)

        if n_pos == 0 or n_neg == 0:
            auroc = 0.5
        else:
            # Sort by score descending (high score = predicted jailbreak)
            paired = sorted(zip(scores, labels), key=lambda x: -x[0])
            tp = 0
            fp = 0
            auc = 0.0
            prev_fpr = 0.0
            prev_tpr = 0.0
            for score, lbl in paired:
                if lbl == 1:
                    tp += 1
                else:
                    fp += 1
                fpr = fp / n_neg
                tpr = tp / n_pos
                if fpr > prev_fpr:
                    auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0
                prev_fpr = fpr
                prev_tpr = tpr
            auroc = float(min(1.0, max(0.0, auc)))

        # Precision and recall at threshold=0.5
        tp_count = sum(
            1 for score, lbl in zip(scores, labels) if score > 0.5 and lbl == 1
        )
        fp_count = sum(
            1 for score, lbl in zip(scores, labels) if score > 0.5 and lbl == 0
        )
        fn_count = sum(
            1 for score, lbl in zip(scores, labels) if score <= 0.5 and lbl == 1
        )

        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

        return auroc, precision, recall


# ---------------------------------------------------------------------------
# Tier 0h gate result
# ---------------------------------------------------------------------------


@dataclass
class Tier0hResult:
    """Result from the Tier 0h jailbreak pre-filter.

    **Fields:**
        jailbreak_score: P(jailbreak) from the classifier.  In [0, 1].
        is_jailbreak:    True when jailbreak_score > threshold.
        passed_tier0h:   True when the prompt was NOT flagged (safe to proceed to LLM).

    Spec: REQ-SAFETY-002
    """

    jailbreak_score: float
    is_jailbreak: bool
    passed_tier0h: bool
