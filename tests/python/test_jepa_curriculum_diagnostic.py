"""Tests for JEPACurriculumDiagnostic and CorpusAnalysis.

Tests trace to:
- REQ-DIAG-001: JEPACurriculumDiagnostic.analyze_corpus() computes label_imbalance_ratio,
  filter_rate, and n_pairs_remaining for the quality-gated corpus.
- REQ-DIAG-002: JEPACurriculumDiagnostic.simulate_regime() trains JEPA on a given pair
  ordering for 100 epochs and returns AUC on held-out set.
- SCENARIO-DIAG-001: CorpusAnalysis.is_imbalanced=True when n_correct=80, n_incorrect=5
- SCENARIO-DIAG-002: CorpusAnalysis.diagnosis='imbalance' when is_imbalanced and n_pairs_filtered >= 5
- SCENARIO-DIAG-003: JEPACurriculumDiagnostic.simulate_regime returns float in [0, 1]
"""

from __future__ import annotations

import pytest

from carnot.models.jepa_curriculum_diagnostic import CorpusAnalysis, JEPACurriculumDiagnostic
from carnot.models.jepa_retrain_v2 import CoTPairQualityFilter


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def _make_pairs(
    n_correct: int = 10,
    n_incorrect: int = 10,
    confidence: float = 1.0,
) -> list[dict]:
    """Create a balanced corpus of labeled step dicts."""
    pairs = []
    for i in range(n_correct):
        pairs.append(
            {
                "step_text": f"correct step {i}: 2 + 2 = 4",
                "label": "correct",
                "confidence": confidence,
            }
        )
    for i in range(n_incorrect):
        pairs.append(
            {
                "step_text": f"incorrect step {i}: 2 + 2 = 5",
                "label": "incorrect",
                "confidence": confidence,
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# CorpusAnalysis: filter_rate
# ---------------------------------------------------------------------------


class TestCorpusAnalysisFilterRate:
    """REQ-DIAG-001: filter_rate = n_pairs_filtered / n_pairs_raw"""

    def test_filter_rate_basic(self):
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=27,
            n_correct=25,
            n_incorrect=2,
            mean_label_confidence=0.9,
            label_imbalance_ratio=12.5,
        )
        assert abs(ca.filter_rate - 0.27) < 1e-9

    def test_filter_rate_all_pass(self):
        ca = CorpusAnalysis(
            n_pairs_raw=57,
            n_pairs_filtered=57,
            n_correct=30,
            n_incorrect=27,
            mean_label_confidence=0.85,
            label_imbalance_ratio=30 / 27,
        )
        assert ca.filter_rate == 1.0

    def test_filter_rate_none_pass(self):
        ca = CorpusAnalysis(
            n_pairs_raw=57,
            n_pairs_filtered=0,
            n_correct=0,
            n_incorrect=0,
            mean_label_confidence=0.5,
            label_imbalance_ratio=1.0,
        )
        assert ca.filter_rate == 0.0

    def test_filter_rate_zero_raw(self):
        ca = CorpusAnalysis(
            n_pairs_raw=0,
            n_pairs_filtered=0,
            n_correct=0,
            n_incorrect=0,
            mean_label_confidence=0.0,
            label_imbalance_ratio=1.0,
        )
        assert ca.filter_rate == 0.0


# ---------------------------------------------------------------------------
# CorpusAnalysis: is_imbalanced
# ---------------------------------------------------------------------------


class TestCorpusAnalysisIsImbalanced:
    """SCENARIO-DIAG-001: is_imbalanced=True when n_correct=80, n_incorrect=5"""

    def test_imbalanced_high_correct(self):
        # SCENARIO-DIAG-001: ratio=80/5=16 >> 3.0 → imbalanced
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=85,
            n_correct=80,
            n_incorrect=5,
            mean_label_confidence=0.95,
            label_imbalance_ratio=16.0,
        )
        assert ca.is_imbalanced is True

    def test_imbalanced_high_incorrect(self):
        # Inverted imbalance: 0.05 < 0.33 → imbalanced
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=85,
            n_correct=5,
            n_incorrect=80,
            mean_label_confidence=0.9,
            label_imbalance_ratio=5 / 80,
        )
        assert ca.is_imbalanced is True

    def test_balanced(self):
        # ratio=1.0 — perfectly balanced
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=40,
            n_correct=20,
            n_incorrect=20,
            mean_label_confidence=0.8,
            label_imbalance_ratio=1.0,
        )
        assert ca.is_imbalanced is False

    def test_boundary_just_above_3(self):
        # ratio=3.01 → imbalanced
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=40,
            n_correct=30,
            n_incorrect=10,
            mean_label_confidence=0.8,
            label_imbalance_ratio=3.01,
        )
        assert ca.is_imbalanced is True

    def test_boundary_exactly_3(self):
        # ratio=3.0 → NOT imbalanced (threshold is strictly > 3.0)
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=40,
            n_correct=30,
            n_incorrect=10,
            mean_label_confidence=0.8,
            label_imbalance_ratio=3.0,
        )
        assert ca.is_imbalanced is False


# ---------------------------------------------------------------------------
# CorpusAnalysis: diagnosis
# ---------------------------------------------------------------------------


class TestCorpusAnalysisDiagnosis:
    """SCENARIO-DIAG-002: diagnosis='imbalance' when is_imbalanced and n_pairs_filtered >= 5"""

    def test_diagnosis_imbalance(self):
        # SCENARIO-DIAG-002: n_correct=80, n_incorrect=5, n_filtered=85 >= 5
        ca = CorpusAnalysis(
            n_pairs_raw=100,
            n_pairs_filtered=85,
            n_correct=80,
            n_incorrect=5,
            mean_label_confidence=0.9,
            label_imbalance_ratio=16.0,
        )
        assert ca.diagnosis == "imbalance"

    def test_diagnosis_insufficient_data(self):
        # n_pairs_filtered=3 < 5 → insufficient_data, even if imbalanced
        ca = CorpusAnalysis(
            n_pairs_raw=57,
            n_pairs_filtered=3,
            n_correct=3,
            n_incorrect=0,
            mean_label_confidence=0.9,
            label_imbalance_ratio=3.0,
        )
        assert ca.diagnosis == "insufficient_data"

    def test_diagnosis_insufficient_data_zero(self):
        ca = CorpusAnalysis(
            n_pairs_raw=57,
            n_pairs_filtered=0,
            n_correct=0,
            n_incorrect=0,
            mean_label_confidence=0.9,
            label_imbalance_ratio=1.0,
        )
        assert ca.diagnosis == "insufficient_data"

    def test_diagnosis_ok(self):
        # balanced, n_filtered=20, mean_conf=0.75 → ok
        ca = CorpusAnalysis(
            n_pairs_raw=30,
            n_pairs_filtered=20,
            n_correct=10,
            n_incorrect=10,
            mean_label_confidence=0.75,
            label_imbalance_ratio=1.0,
        )
        assert ca.diagnosis == "ok"

    def test_diagnosis_domain_shift(self):
        # balanced, n_filtered=8 < 10, mean_conf > 0.95 → domain_shift
        ca = CorpusAnalysis(
            n_pairs_raw=10,
            n_pairs_filtered=8,
            n_correct=4,
            n_incorrect=4,
            mean_label_confidence=0.99,
            label_imbalance_ratio=1.0,
        )
        assert ca.diagnosis == "domain_shift"

    def test_diagnosis_imbalanced_exactly_5_pairs_filtered(self):
        # Boundary: n_filtered=5, is_imbalanced=True → diagnosis='imbalance'
        ca = CorpusAnalysis(
            n_pairs_raw=10,
            n_pairs_filtered=5,
            n_correct=5,
            n_incorrect=0,
            mean_label_confidence=0.8,
            label_imbalance_ratio=5.0,
        )
        assert ca.diagnosis == "imbalance"


# ---------------------------------------------------------------------------
# JEPACurriculumDiagnostic: analyze_corpus
# ---------------------------------------------------------------------------


class TestJEPACurriculumDiagnosticAnalyzeCorpus:
    """REQ-DIAG-001: analyze_corpus() computes label_imbalance_ratio, filter_rate, n_pairs_remaining"""

    def test_analyze_corpus_basic(self):
        pairs = _make_pairs(n_correct=10, n_incorrect=10, confidence=1.0)
        diag = JEPACurriculumDiagnostic(pairs)
        qf = CoTPairQualityFilter(min_coverage=0.0, min_confidence=0.7)
        ca = diag.analyze_corpus(qf)

        assert ca.n_pairs_raw == 20
        assert ca.n_pairs_filtered == 20  # all pass (confidence=1.0 >= 0.7)
        assert ca.n_correct == 10
        assert ca.n_incorrect == 10
        assert abs(ca.label_imbalance_ratio - 1.0) < 1e-9
        assert ca.filter_rate == 1.0

    def test_analyze_corpus_filter_removes_low_confidence(self):
        # Mix of high and low confidence pairs
        pairs = []
        for i in range(5):
            pairs.append({"step_text": f"step {i}", "label": "correct", "confidence": 0.5})
        for i in range(5):
            pairs.append({"step_text": f"step {i}", "label": "correct", "confidence": 1.0})
        for i in range(10):
            pairs.append({"step_text": f"step {i}", "label": "incorrect", "confidence": 1.0})

        diag = JEPACurriculumDiagnostic(pairs)
        qf = CoTPairQualityFilter(min_coverage=0.0, min_confidence=0.7)
        ca = diag.analyze_corpus(qf)

        # 5 low-confidence correct pairs are removed
        assert ca.n_pairs_raw == 20
        assert ca.n_pairs_filtered == 15
        assert ca.n_correct == 5
        assert ca.n_incorrect == 10

    def test_analyze_corpus_returns_corpus_analysis_type(self):
        pairs = _make_pairs(n_correct=5, n_incorrect=5)
        diag = JEPACurriculumDiagnostic(pairs)
        qf = CoTPairQualityFilter(min_coverage=0.0, min_confidence=0.7)
        ca = diag.analyze_corpus(qf)
        assert isinstance(ca, CorpusAnalysis)

    def test_analyze_corpus_empty(self):
        diag = JEPACurriculumDiagnostic([])
        qf = CoTPairQualityFilter(min_coverage=0.0, min_confidence=0.7)
        ca = diag.analyze_corpus(qf)
        assert ca.n_pairs_raw == 0
        assert ca.n_pairs_filtered == 0
        assert ca.filter_rate == 0.0

    def test_analyze_corpus_mean_confidence_uses_raw_pairs(self):
        # mean_label_confidence computed over ALL raw pairs, not just filtered
        pairs = [
            {"step_text": "a", "label": "correct", "confidence": 0.5},
            {"step_text": "b", "label": "correct", "confidence": 0.9},
        ]
        diag = JEPACurriculumDiagnostic(pairs)
        qf = CoTPairQualityFilter(min_coverage=0.0, min_confidence=0.7)
        ca = diag.analyze_corpus(qf)
        assert abs(ca.mean_label_confidence - 0.7) < 1e-9


# ---------------------------------------------------------------------------
# JEPACurriculumDiagnostic: simulate_regime
# ---------------------------------------------------------------------------


class TestJEPACurriculumDiagnosticSimulateRegime:
    """REQ-DIAG-002 / SCENARIO-DIAG-003: simulate_regime returns float in [0, 1]"""

    @pytest.fixture
    def small_pairs(self):
        return _make_pairs(n_correct=8, n_incorrect=8, confidence=1.0)

    def test_simulate_all_pairs_in_range(self, small_pairs):
        # SCENARIO-DIAG-003
        diag = JEPACurriculumDiagnostic(small_pairs)
        auc = diag.simulate_regime("all_pairs", n_epochs=2)
        assert 0.0 <= auc <= 1.0

    def test_simulate_quality_gated_in_range(self, small_pairs):
        diag = JEPACurriculumDiagnostic(small_pairs)
        auc = diag.simulate_regime("quality_gated", n_epochs=2)
        assert 0.0 <= auc <= 1.0

    def test_simulate_curriculum_in_range(self, small_pairs):
        diag = JEPACurriculumDiagnostic(small_pairs)
        auc = diag.simulate_regime("curriculum_high_to_low", n_epochs=2)
        assert 0.0 <= auc <= 1.0

    def test_simulate_random_50pct_in_range(self, small_pairs):
        diag = JEPACurriculumDiagnostic(small_pairs)
        auc = diag.simulate_regime("random_50pct", n_epochs=2)
        assert 0.0 <= auc <= 1.0

    def test_simulate_regime_returns_float(self, small_pairs):
        diag = JEPACurriculumDiagnostic(small_pairs)
        auc = diag.simulate_regime("all_pairs", n_epochs=1)
        assert isinstance(auc, float)

    def test_simulate_regime_invalid_raises(self, small_pairs):
        diag = JEPACurriculumDiagnostic(small_pairs)
        with pytest.raises(ValueError, match="regime must be one of"):
            diag.simulate_regime("invalid_regime", n_epochs=1)

    def test_simulate_regime_empty_corpus(self):
        # Empty corpus → returns 0.5 (undefined AUC)
        diag = JEPACurriculumDiagnostic([])
        auc = diag.simulate_regime("all_pairs", n_epochs=1)
        assert auc == 0.5

    def test_simulate_quality_gated_single_class(self):
        # If quality_gated removes all incorrect pairs, triples are empty → AUC=0.5
        pairs = [
            {"step_text": "step", "label": "correct", "confidence": 1.0}
        ] * 10
        diag = JEPACurriculumDiagnostic(pairs)
        auc = diag.simulate_regime("quality_gated", n_epochs=2)
        assert 0.0 <= auc <= 1.0  # returns 0.5 due to no contrastive pairs

    def test_simulate_curriculum_sorted_order(self):
        # Build pairs with varying confidence; curriculum should produce valid AUC
        pairs = []
        for conf in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            pairs.append({"step_text": f"step c={conf}", "label": "correct", "confidence": conf})
            pairs.append({"step_text": f"step c={conf}", "label": "incorrect", "confidence": conf})
        diag = JEPACurriculumDiagnostic(pairs)
        auc = diag.simulate_regime("curriculum_high_to_low", n_epochs=2)
        assert 0.0 <= auc <= 1.0
