"""EDU-PRM Step Selector — entropy-driven uncertainty selection for FoVer training corpus.

WHY THIS MODULE EXISTS (arXiv 2503.22233, REQ-LEARN-050, REQ-LEARN-051):
    JEPA v19 (Exp 770) trained on ALL 57 FoVer steps uniformly, including many
    trivially correct steps that added no discriminative signal.  The result was
    poor OOD generalisation.

    EDU-PRM (arXiv 2503.22233 §3) demonstrates that only ~1.5% of training data
    is needed when those examples are selected by *prediction variance*: steps
    near the classifier decision boundary (high variance across bootstrap samples)
    carry the most information, while clear-cut correct/incorrect steps add little.

    This module selects the top 30% highest-variance steps from the pooled FoVer
    corpus so that JEPA v20 training concentrates on hard, discriminative examples.

HOW IT WORKS:
    1. Fit N_BOOTSTRAP=10 TF-IDF + LogisticRegression classifiers, each on a
       bootstrap resample (sample with replacement, same size as corpus).
    2. For each step, collect predictions from all 10 classifiers.
    3. Compute variance of predictions per step — high variance = near boundary.
    4. Select the top 30% of steps by variance (selection_pct=0.30).

    WHY TF-IDF + LogisticRegression:
        CPU-only environments (JAX_PLATFORMS=cpu) cannot load multi-GB LLMs.
        TF-IDF captures surface-level violation signals ("error", "wrong",
        "undefined") that strongly correlate with FoVer labels.  LogisticRegression
        provides calibrated probability outputs, making variance a reliable proxy
        for boundary proximity.

Spec: REQ-LEARN-050, REQ-LEARN-051, SCENARIO-LEARN-094, SCENARIO-LEARN-095
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Minimal TF-IDF vectoriser (copied from jepa_v19 to avoid circular import)
# ---------------------------------------------------------------------------


class _TFIDFVec:
    """Lightweight TF-IDF vectoriser — stdlib + no-sklearn.

    Identical algorithm to jepa_v19._TFIDFVectoriser; kept separate so this
    module has no import dependency on the sampler layer.
    """

    def __init__(self, max_features: int = 128) -> None:
        self.max_features = max_features
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []

    @staticmethod
    def _tok(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def fit(self, corpus: list[str]) -> "_TFIDFVec":
        n = len(corpus)
        df: Counter[str] = Counter()
        for doc in corpus:
            df.update(set(self._tok(doc)))
        top = sorted(df.keys(), key=lambda t: -df[t])[: self.max_features]
        self._vocab = {t: i for i, t in enumerate(top)}
        self._idf = [
            math.log((1 + n) / (1 + df[t])) + 1 for t in top
        ]
        return self

    def transform(self, text: str) -> list[float]:
        tokens = self._tok(text)
        tf: Counter[str] = Counter(tokens)
        n = max(len(tokens), 1)
        vec = [0.0] * len(self._vocab)
        for t, idx in self._vocab.items():
            vec[idx] = (tf[t] / n) * self._idf[idx]
        return vec


# ---------------------------------------------------------------------------
# Minimal logistic regression (pure stdlib+math)
# ---------------------------------------------------------------------------


class _LogisticRegression:
    """Single-feature-weight logistic regression trained with gradient descent.

    WHY NO SKLEARN: keeps the module importable in minimal CPU environments.
    20 epochs with lr=0.5 converges reliably on TF-IDF features of this size.
    """

    def __init__(self, n_features: int, lr: float = 0.5, n_epochs: int = 20) -> None:
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr = lr
        self.n_epochs = n_epochs

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _predict_prob(self, x: list[float]) -> float:
        z = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        return self._sigmoid(z)

    def fit(self, X: list[list[float]], y: list[int]) -> "_LogisticRegression":
        n, d = len(X), len(self.w)
        for _ in range(self.n_epochs):
            dw = [0.0] * d
            db = 0.0
            for xi, yi in zip(X, y):
                p = self._predict_prob(xi)
                err = p - yi
                for j in range(d):
                    dw[j] += err * xi[j]
                db += err
            self.w = [self.w[j] - self.lr * dw[j] / n for j in range(d)]
            self.b -= self.lr * db / n
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [self._predict_prob(x) for x in X]


# ---------------------------------------------------------------------------
# EDU-PRM configuration and selector
# ---------------------------------------------------------------------------


@dataclass
class EDUPRMConfig:
    """Configuration for EDUPRMStepSelector.

    Attributes:
        n_bootstrap: Number of bootstrap classifiers to train.  10 is the
            value used in arXiv 2503.22233; more gives smoother variance
            estimates but increases runtime linearly.
        selection_pct: Fraction of steps to keep (top by variance).  0.30
            selects the 30% most uncertain steps.
        max_features: TF-IDF vocabulary size.  128 captures the key violation
            vocabulary without overfitting to corpus-specific n-grams.
        random_seed: Seed for reproducibility of bootstrap resamples.
    """

    n_bootstrap: int = 10
    selection_pct: float = 0.30
    max_features: int = 128
    random_seed: int = 42


class EDUPRMStepSelector:
    """Select high-uncertainty FoVer training steps via bootstrap variance.

    WHY BOOTSTRAP VARIANCE (arXiv 2503.22233 §3):
        A step is "hard" when different classifiers trained on different subsets
        of the corpus disagree about its label.  High disagreement = high variance
        in prediction probabilities.  These boundary-adjacent steps carry the most
        discriminative signal for training.  Clear-cut correct/incorrect steps
        (low variance = all classifiers agree) contribute little and can be dropped
        without hurting model quality — EDU-PRM showed full-corpus performance
        with just 1.5% of data selected this way.

    Usage:
        selector = EDUPRMStepSelector(EDUPRMConfig())
        indices = selector.select(step_texts, labels)
        score = selector.diversity_score([labels[i] for i in indices])
    """

    def __init__(self, config: EDUPRMConfig | None = None) -> None:
        self.config = config or EDUPRMConfig()
        # _bootstrap_preds[b][i] = predicted probability for step i under model b
        self._bootstrap_preds: list[list[float]] = []
        self._variances: list[float] = []

    def fit_bootstrap(self, step_texts: list[str], labels: list[int]) -> None:
        """Train N_BOOTSTRAP classifiers on bootstrap resamples; record predictions.

        WHY RESAMPLING WITH REPLACEMENT:
            Bootstrap resampling introduces variance in training data that reveals
            which examples are genuinely hard vs. which are easy flukes.  Steps
            that the model is uncertain about will show high variance in predicted
            probability across the N bootstrap classifiers.

        Args:
            step_texts: Raw text for each reasoning step.
            labels: Binary label (1=correct, 0=incorrect) for each step.
        """
        rng = random.Random(self.config.random_seed)
        n = len(step_texts)

        # Fit shared vocabulary on full corpus so all bootstrap classifiers
        # share the same feature space — only training *weights* vary.
        vec = _TFIDFVec(max_features=self.config.max_features)
        vec.fit(step_texts)
        X_full = [vec.transform(t) for t in step_texts]
        n_features = len(X_full[0]) if X_full else 0

        self._bootstrap_preds = []
        for _ in range(self.config.n_bootstrap):
            # Resample training set with replacement (same size as corpus).
            indices = [rng.randint(0, n - 1) for _ in range(n)]
            X_boot = [X_full[i] for i in indices]
            y_boot = [labels[i] for i in indices]

            clf = _LogisticRegression(n_features=n_features)
            clf.fit(X_boot, y_boot)

            # Record predictions on the FULL corpus (not just the bootstrap sample).
            preds = clf.predict_proba(X_full)
            self._bootstrap_preds.append(preds)

        # Compute per-step variance across the N bootstrap predictions.
        self._variances = []
        for i in range(n):
            preds_i = [self._bootstrap_preds[b][i] for b in range(self.config.n_bootstrap)]
            mean = sum(preds_i) / len(preds_i)
            var = sum((p - mean) ** 2 for p in preds_i) / len(preds_i)
            self._variances.append(var)

    def select(self, step_texts: list[str], labels: list[int]) -> list[int]:
        """Return indices of the top selection_pct steps by bootstrap variance.

        WHY SORT THEN SLICE:
            Sorting by descending variance and taking the top fraction is the
            exact selection rule from EDU-PRM §3.1.  The threshold adapts to
            any corpus size, always returning exactly ceil(n * selection_pct)
            steps.

        Args:
            step_texts: Raw text for each reasoning step.
            labels: Binary label (1=correct, 0=incorrect) for each step.

        Returns:
            List of selected indices, sorted by descending variance.
        """
        if not step_texts:
            return []

        self.fit_bootstrap(step_texts, labels)

        n = len(step_texts)
        k = max(1, math.ceil(n * self.config.selection_pct))

        # Sort by descending variance; return original indices.
        ranked = sorted(range(n), key=lambda i: -self._variances[i])
        return ranked[:k]

    def diversity_score(self, selected_labels: list[int]) -> float:
        """Return the fraction of selected labels that are positive (label=1).

        WHY THIS METRIC:
            An ideal selection (balanced hard examples) should have diversity_score
            near 0.5 — roughly equal numbers of hard-correct and hard-incorrect steps.
            A score near 0 or 1 indicates the selector is only picking one class,
            which would produce a biased training set.  Comparing against uniform
            selection diversity reveals whether EDU-PRM selection improves balance.

        Args:
            selected_labels: Binary labels for the selected steps only.

        Returns:
            Fraction in [0, 1] of labels equal to 1.  Returns 0.0 for empty input.
        """
        if not selected_labels:
            return 0.0
        return sum(1 for lbl in selected_labels if lbl == 1) / len(selected_labels)
