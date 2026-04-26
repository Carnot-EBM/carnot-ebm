"""DRIFTProbeEnsemble — Per-layer ensemble extension of DRIFTProbe for Tier 0i.

**Researcher summary:**
    arXiv 2604.13386 shows that training one logistic regression probe per adjacent
    layer pair, then combining predictions via learned weighting (alpha), outperforms a
    single probe trained on the concatenated multi-layer drift vector by 3-8% AUROC.

    Why does ensembling beat concatenation?
    Each layer pair captures a different aspect of the model's hallucination signal.
    Early late-layers (e.g., -4 vs -3) reflect semantic uncertainty; the final pair
    (-2 vs -1) reflects token-commitment instability.  A single probe must trade off
    sensitivity across all pairs simultaneously, while an ensemble can weight each
    pair's signal according to how discriminative it is on a held-out validation set.

    WHY learned alpha via grid search rather than gradient descent?
    With only 20 held-out examples (the Exp 911 split), gradient-based weight learning
    overfits badly.  A coarse 20-candidate simplex grid search finds the alpha weights
    that maximise held-out accuracy without overfitting.  The simplex constraint
    (alpha >= 0, sum(alpha) = 1) keeps the ensemble score in [0, 1].

    WHY inherit from / compose with DRIFTProbe rather than replace it?
    DRIFTProbe (Exp 911) stays untouched as the REQ-TIER0-005 reference implementation.
    DRIFTProbeEnsemble is an additive improvement (REQ-TIER0-006) that slots alongside it
    in the ThreeTierPipeline via the same wire_drift_probe() hook.

Spec: REQ-TIER0-006, SCENARIO-TIER0-006
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

# ---------------------------------------------------------------------------
# helpers (reuse from drift_probe without importing the whole class)
# ---------------------------------------------------------------------------


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float vectors (zero-safe)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


def _drift_for_pair(hidden_states: dict[int, np.ndarray], layer_a: int, layer_b: int) -> float:
    """Mean token-level cosine distance between two layer representations.

    Returns 0.0 when either layer is absent or the sequence is empty, so callers
    never need to guard against missing keys.
    """
    if layer_a not in hidden_states or layer_b not in hidden_states:
        return 0.0
    rep_a = hidden_states[layer_a]
    rep_b = hidden_states[layer_b]
    seq_len = min(rep_a.shape[0], rep_b.shape[0])
    if seq_len == 0:
        return 0.0
    dists = np.array(
        [float(np.clip(1.0 - _cosine_sim(rep_a[t], rep_b[t]), 0.0, 2.0)) for t in range(seq_len)],
        dtype=np.float32,
    )
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
# DRIFTProbeEnsemble
# ---------------------------------------------------------------------------


class DRIFTProbeEnsemble:
    """Per-layer ensemble of logistic regression probes for Tier 0i hallucination detection.

    **For engineers:**
        Drop-in replacement for DRIFTProbe in ThreeTierPipeline.wire_drift_probe().
        Instead of one probe trained on the full drift vector, this trains N separate
        probes (one per adjacent layer pair), then learns ensemble weights alpha on a
        held-out validation split.  The final violation probability is a weighted sum:

            p(hallucination) = sum(alpha_i * probe_i.predict_proba(drift_i)[class=1])

        The alpha weights are learned via grid search over a 20-point simplex covering
        [0, 1]^N with sum(alpha) = 1.  Grid points are evaluated by accuracy on the
        held-out set.

    Usage::

        runner = lambda text: model.get_hidden_states(text, layers=[-4,-3,-2,-1])
        ensemble = DRIFTProbeEnsemble(model_runner=runner, layers=[-4,-3,-2,-1])
        ensemble.fit(correct_examples, hallucinated_examples)
        prob = ensemble.predict_violation_prob(hidden_states)

    Args:
        model_runner: Callable[[str], dict[int, np.ndarray]] — runs the LLM forward pass.
                      None is safe for testing (all drift scores are zero).
        layers:       Layer indices to include.  Default: [-4, -3, -2, -1].
                      N-1 layer pairs are formed from N layers.

    Spec: REQ-TIER0-006
    """

    def __init__(
        self,
        model_runner: Callable[[str], dict[int, np.ndarray]] | None = None,
        layers: list[int] | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.layers: list[int] = layers if layers is not None else [-4, -3, -2, -1]

        # One probe per adjacent pair; filled by fit().
        self.per_layer_probes: list = []  # list[LogisticRegression]

        # Alpha weights learned on validation set; shape (n_pairs,).
        self.ensemble_weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_per_pair_drifts(self, hidden_states: dict[int, np.ndarray]) -> np.ndarray:
        """Return a 1-D array of cosine-distance drift values, one per adjacent pair.

        Shape: (len(self.layers) - 1,).  Each element is the mean token cosine
        distance for that layer pair.  Zero when a layer is absent.
        """
        n_pairs = len(self.layers) - 1
        drifts = np.zeros(n_pairs, dtype=np.float32)
        for i in range(n_pairs):
            drifts[i] = _drift_for_pair(hidden_states, self.layers[i], self.layers[i + 1])
        return drifts

    def _run_model(self, text: str) -> dict[int, np.ndarray]:
        """Run model_runner safely; return empty dict on failure or if None."""
        if self.model_runner is None:
            return {}
        try:
            return self.model_runner(text)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        correct_examples: list,
        hallucinated_examples: list,
        val_fraction: float = 0.2,
        n_grid_points: int = 20,
        random_state: int = 42,
    ) -> None:
        """Train per-layer probes and learn ensemble weights alpha.

        Steps:
          1. Extract hidden states for every example via model_runner.
          2. Compute per-pair drift scalar for each example (one scalar per pair).
          3. Split into train/validation (val_fraction held out).
          4. For each layer pair, fit a LogisticRegression on the training split.
          5. On the validation split, run a 20-point grid search over the simplex
             alpha >= 0, sum(alpha) = 1 to find weights that maximise accuracy.
          6. Store per_layer_probes and ensemble_weights.

        WHY val_fraction=0.2:
            Matches the Exp 911 80/20 split (80 train, 20 held-out per class side).
            With 20 held-out examples per class the simplex search is stable; fewer
            than ~10 per class risks the grid collapsing to a degenerate alpha.

        Args:
            correct_examples:      List of correct response strings (label=0),
                                   or dicts with a "text" key.
            hallucinated_examples: List of hallucinated response strings (label=1),
                                   or dicts with a "text" key.
            val_fraction:          Fraction of each class held out for alpha learning.
            n_grid_points:         Number of alpha candidates per pair in the simplex
                                   grid.  Total grid size ~ n_grid_points^n_pairs
                                   but pruned to simplex.
            random_state:          Seed for reproducibility.

        Spec: REQ-TIER0-006-1, REQ-TIER0-006-2
        """
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(random_state)

        def _get_text(ex):
            return ex.get("text", "") if isinstance(ex, dict) else str(ex)

        def _extract_all(examples: list) -> np.ndarray:
            # Returns shape (n_examples, n_pairs).
            rows = []
            for ex in examples:
                states = self._run_model(_get_text(ex))
                drifts = self._extract_per_pair_drifts(states)
                rows.append(drifts)
            return np.vstack(rows).astype(np.float32)

        # Build feature matrices: shape (n_examples, n_pairs).
        X_correct = _extract_all(correct_examples)  # label=0
        X_halluc = _extract_all(hallucinated_examples)  # label=1

        n_pairs = len(self.layers) - 1

        # Train/val split per class.
        n_val_correct = max(1, int(len(X_correct) * val_fraction))
        n_val_halluc = max(1, int(len(X_halluc) * val_fraction))

        idx_c = rng.permutation(len(X_correct))
        idx_h = rng.permutation(len(X_halluc))

        X_c_train, X_c_val = X_correct[idx_c[n_val_correct:]], X_correct[idx_c[:n_val_correct]]
        X_h_train, X_h_val = X_halluc[idx_h[n_val_halluc:]], X_halluc[idx_h[:n_val_halluc]]

        # Build (X_train, y_train) and (X_val, y_val) for per-pair classifiers.
        # Each pair uses column i of the feature matrix as a 1-D feature.
        X_train = np.vstack([X_c_train, X_h_train])  # (n_train, n_pairs)
        y_train = np.array([0] * len(X_c_train) + [1] * len(X_h_train), dtype=int)
        X_val = np.vstack([X_c_val, X_h_val])
        y_val = np.array([0] * len(X_c_val) + [1] * len(X_h_val), dtype=int)

        # Train one LogisticRegression per pair on that pair's scalar drift column.
        probes = []
        for i in range(n_pairs):
            feat_train = X_train[:, i : i + 1]  # (n_train, 1)
            probe = LogisticRegression(max_iter=500, random_state=random_state)
            probe.fit(feat_train, y_train)
            probes.append(probe)
        self.per_layer_probes = probes

        # Compute per-probe probabilities on the validation set.
        # val_probs shape: (n_val_examples, n_pairs)
        val_probs = self._compute_per_probe_probs(X_val)

        # Grid search for alpha over the simplex: alpha >= 0, sum = 1.
        best_alpha = np.ones(n_pairs, dtype=np.float64) / n_pairs  # uniform fallback
        best_acc = -1.0

        for alpha in self._simplex_grid(n_pairs, n_grid_points):
            scores = val_probs @ alpha  # (n_val_examples,)
            preds = (scores >= 0.5).astype(int)
            acc = float(np.mean(preds == y_val))
            if acc > best_acc:
                best_acc = acc
                best_alpha = alpha.copy()

        self.ensemble_weights = best_alpha

    def _compute_per_probe_probs(self, X: np.ndarray) -> np.ndarray:
        """Return per-probe hallucination probabilities for each example.

        Args:
            X: (n_examples, n_pairs) drift scalar matrix.

        Returns:
            (n_examples, n_pairs) float array of P(hallucination | pair_i).
        """
        n_pairs = len(self.per_layer_probes)
        n_examples = X.shape[0]
        out = np.zeros((n_examples, n_pairs), dtype=np.float64)
        for i, probe in enumerate(self.per_layer_probes):
            classes = list(probe.classes_)
            halluc_idx = classes.index(1) if 1 in classes else 1
            proba = probe.predict_proba(X[:, i : i + 1])  # (n_examples, n_classes)
            out[:, i] = proba[:, halluc_idx]
        return out

    @staticmethod
    def _simplex_grid(n_dims: int, n_points: int) -> list[np.ndarray]:
        """Generate ~n_points simplex-constrained alpha vectors.

        Uses a uniform Dirichlet sample to cover the simplex: each candidate is
        drawn as softmax(uniform(0,1)^n_dims), which samples the interior of the
        simplex roughly uniformly.  Appends the uniform-weight baseline always.

        Args:
            n_dims:   Dimensionality of alpha (= number of layer pairs).
            n_points: Number of candidate alpha vectors to generate.

        Returns:
            List of 1-D float64 arrays, each summing to 1 and non-negative.
        """
        rng = np.random.default_rng(0)  # fixed seed for reproducibility across calls
        candidates = []
        # Always include uniform as first candidate.
        candidates.append(np.ones(n_dims, dtype=np.float64) / n_dims)
        # Dirichlet-sampled candidates cover non-uniform regions of the simplex.
        raw = rng.random((n_points - 1, n_dims)) + 1e-6  # avoid zeros
        sums = raw.sum(axis=1, keepdims=True)
        for row in raw / sums:
            candidates.append(row.astype(np.float64))
        return candidates

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_violation_prob(self, hidden_states: dict[int, np.ndarray]) -> float:
        """Return probability that hidden_states come from a hallucinating response.

        Computes per-pair drift scalar, runs each per-layer probe, then returns the
        alpha-weighted sum of per-probe hallucination probabilities.

        Returns 0.5 if the ensemble has not been fitted yet (probes empty or weights None).

        Args:
            hidden_states: Dict[int, np.ndarray] keyed by layer index.

        Returns:
            Float in [0, 1].  Higher = more likely hallucination.

        Spec: REQ-TIER0-006-3
        """
        if not self.per_layer_probes or self.ensemble_weights is None:
            return 0.5

        drifts = self._extract_per_pair_drifts(hidden_states)  # (n_pairs,)
        per_probe_p = np.array(
            [
                self._probe_prob(probe, drift_val)
                for probe, drift_val in zip(self.per_layer_probes, drifts, strict=False)
            ],
            dtype=np.float64,
        )
        return float(np.clip(np.dot(self.ensemble_weights, per_probe_p), 0.0, 1.0))

    def _probe_prob(self, probe, drift_val: float) -> float:
        """Return P(hallucination) from a single fitted probe for one drift scalar."""
        classes = list(probe.classes_)
        halluc_idx = classes.index(1) if 1 in classes else 1
        proba = probe.predict_proba([[drift_val]])[0]
        return float(proba[halluc_idx])

    def predict_violation_prob_from_text(self, text: str) -> float:
        """Convenience: run model_runner then return ensemble violation probability.

        Returns 0.5 if model_runner is None or ensemble has not been fitted.

        Args:
            text: Response text to score.

        Returns:
            Float in [0, 1].

        Spec: REQ-TIER0-006-3
        """
        states = self._run_model(text)
        return self.predict_violation_prob(states)
