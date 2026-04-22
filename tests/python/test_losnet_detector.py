"""Tests for python/carnot/pipeline/losnet_detector.py.

Coverage target: 100% of losnet_detector.py

Spec: REQ-VERIFY-153, REQ-VERIFY-154,
      SCENARIO-VERIFY-202, SCENARIO-VERIFY-203, SCENARIO-VERIFY-204
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.losnet_detector import (
    LOSNetClassifier,
    LOSNetFeatures,
    build_losnet_artifact,
    extract_losnet_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _uniform_sequence(n_steps: int, k: int = 5) -> list[list[float]]:
    """Return n_steps uniform probability vectors of length k."""
    return [[1.0 / k] * k for _ in range(n_steps)]


def _peaked_sequence(n_steps: int, k: int = 5) -> list[list[float]]:
    """Return n_steps peaked probability vectors: 0.9 on first token, rest uniform."""
    probs = [0.9] + [0.1 / (k - 1)] * (k - 1)
    return [list(probs) for _ in range(n_steps)]


def _increasing_entropy_sequence(n_steps: int) -> list[list[float]]:
    """Return a sequence where entropy increases: starts peaked, ends uniform."""
    result = []
    for t in range(n_steps):
        # interpolate from peaked (t=0) to uniform (t=n_steps-1)
        alpha = t / max(n_steps - 1, 1)
        k = 5
        p_max = 0.9 * (1 - alpha) + (1.0 / k) * alpha
        p_other = (1.0 - p_max) / (k - 1)
        result.append([p_max] + [p_other] * (k - 1))
    return result


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-202: extract_losnet_features basic correctness
# ---------------------------------------------------------------------------


class TestExtractLosnetFeatures:
    """REQ-VERIFY-153: extract_losnet_features must return a valid LOSNetFeatures."""

    def test_uniform_sequence_entropy_is_log_k(self):
        """SCENARIO-VERIFY-202: uniform distribution over k tokens → entropy = log(k)."""
        k = 5
        seqs = _uniform_sequence(n_steps=4, k=k)
        feat = extract_losnet_features(seqs, top_k=k)

        assert feat.n_steps == 4
        expected_h = math.log(k)
        for h in feat.sequence_entropy:
            assert abs(h - expected_h) < 1e-6, f"Expected H={expected_h:.4f}, got {h:.4f}"

    def test_uniform_sequence_zero_variance(self):
        """Uniform sequence has zero entropy variance (constant entropy)."""
        seqs = _uniform_sequence(n_steps=6, k=4)
        feat = extract_losnet_features(seqs)
        assert feat.entropy_variance < 1e-6

    def test_uniform_sequence_zero_trend(self):
        """Uniform sequence has zero entropy trend (flat trajectory)."""
        seqs = _uniform_sequence(n_steps=6, k=4)
        feat = extract_losnet_features(seqs)
        assert abs(feat.entropy_trend) < 1e-6

    def test_increasing_entropy_positive_trend(self):
        """SCENARIO-VERIFY-202: increasing entropy sequence has positive trend."""
        seqs = _increasing_entropy_sequence(n_steps=10)
        feat = extract_losnet_features(seqs, top_k=5)
        assert feat.entropy_trend > 0.0

    def test_peaked_sequence_low_entropy(self):
        """Peaked distribution produces lower entropy than uniform."""
        seqs_uniform = _uniform_sequence(n_steps=4, k=5)
        seqs_peaked = _peaked_sequence(n_steps=4, k=5)
        feat_u = extract_losnet_features(seqs_uniform, top_k=5)
        feat_p = extract_losnet_features(seqs_peaked, top_k=5)
        assert feat_u.sequence_entropy[0] > feat_p.sequence_entropy[0]

    def test_top_k_truncation(self):
        """When input has more entries than top_k, only top_k are kept."""
        # 20-entry distributions
        seqs = [[0.05] * 20 for _ in range(3)]
        feat = extract_losnet_features(seqs, top_k=5)
        # After truncation, each step has 5 entries normalised.
        assert all(len(p) == 5 for p in feat.top_k_probs)

    def test_single_step_sequence(self):
        """Single step: n_steps=1, entropy_trend=0, variance=0."""
        seqs = [[0.5, 0.3, 0.2]]
        feat = extract_losnet_features(seqs, top_k=3)
        assert feat.n_steps == 1
        assert feat.entropy_trend == 0.0
        assert feat.entropy_variance == 0.0

    def test_empty_sequence(self):
        """Empty input returns a zero-filled LOSNetFeatures (graceful degradation)."""
        feat = extract_losnet_features([])
        assert feat.n_steps == 0
        assert feat.sequence_entropy == []
        assert feat.entropy_variance == 0.0
        assert feat.entropy_trend == 0.0

    def test_all_zero_probs_handled(self):
        """All-zero probability vector is treated as uniform (no crash)."""
        seqs = [[0.0, 0.0, 0.0]]
        feat = extract_losnet_features(seqs, top_k=3)
        assert feat.n_steps == 1
        assert len(feat.sequence_entropy) == 1
        assert feat.sequence_entropy[0] >= 0.0

    def test_fields_present(self):
        """LOSNetFeatures has all required fields."""
        feat = extract_losnet_features([[0.5, 0.5]])
        assert hasattr(feat, "top_k_probs")
        assert hasattr(feat, "n_steps")
        assert hasattr(feat, "sequence_entropy")
        assert hasattr(feat, "entropy_variance")
        assert hasattr(feat, "entropy_trend")


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-203: LOSNetClassifier.train correctness
# ---------------------------------------------------------------------------


class TestLOSNetClassifierTrain:
    """REQ-VERIFY-153: LOSNetClassifier must train on positive/negative pairs."""

    def _make_positive(self) -> LOSNetFeatures:
        """High-variance, positive-trend features (hallucination pattern)."""
        seqs = _increasing_entropy_sequence(n_steps=10)
        return extract_losnet_features(seqs, top_k=5)

    def _make_negative(self) -> LOSNetFeatures:
        """Low-variance, flat features (correct output pattern)."""
        seqs = _peaked_sequence(n_steps=10, k=5)
        return extract_losnet_features(seqs, top_k=5)

    def test_untrained_returns_half(self):
        """Untrained classifier always returns 0.5 (maximum uncertainty)."""
        clf = LOSNetClassifier()
        feat = self._make_positive()
        assert clf.score(feat) == 0.5

    def test_trained_flag_set(self):
        """After training, _trained is True."""
        clf = LOSNetClassifier()
        clf.train([self._make_positive()], [self._make_negative()])
        assert clf._trained is True

    def test_positive_scores_higher_than_negative(self):
        """SCENARIO-VERIFY-203: after training, hallucination patterns score higher."""
        positives = [self._make_positive() for _ in range(10)]
        negatives = [self._make_negative() for _ in range(10)]
        clf = LOSNetClassifier()
        clf.train(positives, negatives)

        pos_score = clf.score(self._make_positive())
        neg_score = clf.score(self._make_negative())
        assert pos_score > neg_score, (
            f"Expected pos_score ({pos_score:.4f}) > neg_score ({neg_score:.4f})"
        )

    def test_train_empty_positive(self):
        """Training with negatives only: classifier trains on 0-label examples, no crash."""
        clf = LOSNetClassifier()
        clf.train([], [self._make_negative()])
        # Training did run (n=1 negative), so _trained is True and score != 0.5.
        assert clf._trained is True
        s = clf.score(self._make_negative())
        assert 0.0 <= s <= 1.0

    def test_train_empty_both(self):
        """Training with zero examples: no crash, untrained state preserved."""
        clf = LOSNetClassifier()
        clf.train([], [])
        assert not clf._trained

    def test_score_in_unit_interval(self):
        """score() output is always in [0, 1]."""
        positives = [self._make_positive() for _ in range(5)]
        negatives = [self._make_negative() for _ in range(5)]
        clf = LOSNetClassifier()
        clf.train(positives, negatives)
        for feat in positives + negatives:
            s = clf.score(feat)
            assert 0.0 <= s <= 1.0, f"score out of range: {s}"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-204: score() returns P(hallucination)
# ---------------------------------------------------------------------------


class TestLOSNetClassifierScore:
    """REQ-VERIFY-154: score() must return probability in [0, 1]."""

    def test_score_is_float(self):
        """score() returns a Python float."""
        clf = LOSNetClassifier()
        feat = extract_losnet_features([[0.5, 0.5]])
        result = clf.score(feat)
        assert isinstance(result, float)

    def test_sigmoid_boundary_negative_z(self):
        """Internal _sigmoid: very negative z → ~0."""
        clf = LOSNetClassifier()
        assert clf._sigmoid(-100.0) < 0.01

    def test_sigmoid_boundary_positive_z(self):
        """Internal _sigmoid: very positive z → ~1."""
        clf = LOSNetClassifier()
        assert clf._sigmoid(100.0) > 0.99

    def test_sigmoid_zero(self):
        """Internal _sigmoid: z=0 → 0.5."""
        clf = LOSNetClassifier()
        assert abs(clf._sigmoid(0.0) - 0.5) < 1e-9

    def test_extract_vector_length(self):
        """_extract_vector always returns exactly 3 elements."""
        feat = extract_losnet_features([[0.5, 0.5], [0.3, 0.7]])
        vec = LOSNetClassifier._extract_vector(feat)
        assert len(vec) == 3

    def test_extract_vector_empty_features(self):
        """_extract_vector on empty sequence returns [0, 0, 0]."""
        feat = extract_losnet_features([])
        vec = LOSNetClassifier._extract_vector(feat)
        assert vec == [0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# build_losnet_artifact tests
# ---------------------------------------------------------------------------


class TestBuildLosnetArtifact:
    """Tests for the build_losnet_artifact helper."""

    def test_required_keys_present(self):
        """Artifact contains all required keys."""
        art = build_losnet_artifact(
            auc=0.80,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert "model" in art
        assert "auc_losnet" in art
        assert "auc_spilled_energy_baseline" in art
        assert "honest_verdict" in art
        assert "n_train_pairs" in art
        assert "n_eval_pairs" in art

    def test_tier0h_viable_verdict(self):
        """Artifact with AUC >= 0.75 has honest_verdict='tier0h_viable'."""
        art = build_losnet_artifact(
            auc=0.80,
            vs_spilled_energy_auc=0.60,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert art["honest_verdict"] == "tier0h_viable"

    def test_below_threshold_verdict(self):
        """Artifact with AUC < 0.75 has honest_verdict='below_threshold'."""
        art = build_losnet_artifact(
            auc=0.60,
            vs_spilled_energy_auc=0.55,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="below_threshold",
        )
        assert art["honest_verdict"] == "below_threshold"

    def test_feature_importances_included(self):
        """Feature importances are included when provided."""
        fi = {"entropy_variance": 0.5, "entropy_trend": 0.3, "max_entropy": 0.2}
        art = build_losnet_artifact(
            auc=0.78,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
            feature_importances=fi,
        )
        assert art["feature_importances"] == fi

    def test_feature_importances_absent_by_default(self):
        """feature_importances is not in artifact when not provided."""
        art = build_losnet_artifact(
            auc=0.78,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert "feature_importances" not in art

    def test_auc_delta_computed(self):
        """auc_delta = auc - vs_spilled_energy_auc (rounded to 4 decimal places)."""
        art = build_losnet_artifact(
            auc=0.80,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert abs(art["auc_delta"] - 0.15) < 1e-4

    def test_n_parameters_is_four(self):
        """Linear classifier has exactly 4 parameters (3 weights + 1 bias)."""
        art = build_losnet_artifact(
            auc=0.78,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert art["n_parameters"] == 4

    def test_paper_reference(self):
        """Artifact cites arXiv 2503.14043."""
        art = build_losnet_artifact(
            auc=0.78,
            vs_spilled_energy_auc=0.65,
            n_train_pairs=40,
            n_eval_pairs=10,
            honest_verdict="tier0h_viable",
        )
        assert "2503.14043" in art["paper"]
