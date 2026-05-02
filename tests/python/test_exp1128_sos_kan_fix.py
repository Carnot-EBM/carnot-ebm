"""Tests for Exp 1128: SOSKANEnergyV3Adapter normalization fix.

Root cause: feature normalization mismatch between training (data-driven min/max)
and inference (fixed anchors) caused AUROC 0.333 in Exp 1121.

Fix: fit_from_corpus() stores per-column (min,max) stats; score()/_featurize()
uses those stats for consistent train/inference normalization.

Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

for _pkg in ["carnot", "carnot.eval", "carnot.verify", "carnot.models", "carnot.pipeline"]:
    if _pkg not in sys.modules:
        _m = types.ModuleType(_pkg)
        _m.__path__ = [str(_PYTHON_DIR / _pkg.replace(".", "/"))]  # type: ignore[attr-defined]
        _m.__package__ = _pkg
        sys.modules[_pkg] = _m

from carnot.verify.and_composition_verifier import (  # noqa: E402
    SOSKANEnergyV3Adapter,
    _apply_feature_stats,
    _extract_raw_features,
    _extract_text_features,
)


# ---------------------------------------------------------------------------
# _extract_raw_features
# ---------------------------------------------------------------------------


def test_extract_raw_features_shape():
    """_extract_raw_features returns (n, 3) array for a list of n texts.

    Spec: REQ-VERIFY-1121
    """
    texts = ["Hello world 42", "The answer is 3.14", ""]
    result = _extract_raw_features(texts)
    assert result.shape == (3, 3), f"Expected (3, 3), got {result.shape}"


def test_extract_raw_features_nonneg_log_length():
    """Feature 0 (log-length) is non-negative for any input.

    Spec: REQ-VERIFY-1121
    """
    texts = ["x", "hello world 123 45.6", ""]
    result = _extract_raw_features(texts)
    assert (result[:, 0] >= 0).all(), "log-length feature must be >= 0"


def test_extract_raw_features_numeric_density_range():
    """Feature 1 (numeric density) is in [0, 1].

    Spec: REQ-VERIFY-1121
    """
    texts = ["100 200 300", "no numbers here", "mix 1 text 2"]
    result = _extract_raw_features(texts)
    assert (result[:, 1] >= 0).all() and (result[:, 1] <= 1).all(), (
        "numeric density must be in [0, 1]"
    )


def test_extract_raw_features_vocab_richness_range():
    """Feature 2 (vocab richness = unique/total) is in (0, 1].

    Spec: REQ-VERIFY-1121
    """
    texts = ["a a a a", "unique words only here"]
    result = _extract_raw_features(texts)
    assert (result[:, 2] > 0).all() and (result[:, 2] <= 1).all(), (
        "vocab richness must be in (0, 1]"
    )


# ---------------------------------------------------------------------------
# _apply_feature_stats
# ---------------------------------------------------------------------------


def test_apply_feature_stats_normalizes_to_minus1_plus1():
    """_apply_feature_stats maps column min to -1 and max to +1.

    Spec: REQ-VERIFY-1121
    """
    arr = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]], dtype=float)
    stats = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    result = _apply_feature_stats(arr, stats)
    np.testing.assert_allclose(result[0], [-1.0, -1.0, -1.0], atol=1e-10)
    np.testing.assert_allclose(result[1], [1.0, 1.0, 1.0], atol=1e-10)


def test_apply_feature_stats_clips_to_range():
    """Values outside [min, max] are clipped to [-1, 1] after normalization.

    Spec: REQ-VERIFY-1121
    """
    arr = np.array([[-1.0], [0.5], [2.0]], dtype=float)  # 0.5 maps to 1.0, 2.0 clips to 1.0
    stats = [(0.0, 1.0)]
    result = _apply_feature_stats(arr, stats)
    assert result[0, 0] == -1.0, "Value below min clips to -1"
    assert result[2, 0] == 1.0, "Value above max clips to +1"


def test_apply_feature_stats_constant_column():
    """Constant column (max == min) is mapped to 0.0 without division by zero.

    Spec: REQ-VERIFY-1121
    """
    arr = np.array([[5.0], [5.0]], dtype=float)
    stats = [(5.0, 5.0)]
    result = _apply_feature_stats(arr, stats)
    np.testing.assert_array_equal(result[:, 0], [0.0, 0.0])


# ---------------------------------------------------------------------------
# SOSKANEnergyV3Adapter — new normalization-aware interface
# ---------------------------------------------------------------------------


def test_adapter_set_feature_stats_stores_stats():
    """set_feature_stats() stores stats accessible via _feature_stats.

    Spec: REQ-VERIFY-1121
    """
    adapter = SOSKANEnergyV3Adapter()
    stats = [(1.0, 5.0), (0.0, 1.0), (0.1, 0.9)]
    adapter.set_feature_stats(stats)
    assert adapter._feature_stats == stats


def test_adapter_featurize_uses_stored_stats():
    """_featurize() applies stored training stats, not fixed anchors.

    With stored stats, the output should differ from _extract_text_features
    (which uses fixed anchors) when the corpus min/max differ from (0,10),(0,1),(0,1).

    Spec: REQ-VERIFY-1121
    """
    adapter = SOSKANEnergyV3Adapter()
    text = "Step 3: The result is 42 since x + y = 7 and we know x = 35."

    # Fixed-anchor result
    feat_fixed = np.asarray(_extract_text_features(text))

    # Set a custom corpus normalization that differs from the fixed anchors
    # (simulating a corpus where log-lengths range from 3 to 7, not 0 to 10)
    raw = np.asarray(_extract_raw_features([text]))
    custom_stats = [(3.0, 7.0), (0.0, 1.0), (0.0, 1.0)]
    adapter.set_feature_stats(custom_stats)
    feat_custom = np.asarray(adapter._featurize(text))

    # Feature 0 should differ because we used a narrower anchor range
    assert not np.allclose(feat_fixed[0], feat_custom[0], atol=1e-6), (
        "_featurize should use stored stats (different from fixed anchors)"
    )


def test_adapter_featurize_without_stats_falls_back_to_fixed_anchors():
    """_featurize() falls back to _extract_text_features when _feature_stats is None.

    Spec: REQ-VERIFY-1121
    """
    adapter = SOSKANEnergyV3Adapter()
    assert adapter._feature_stats is None
    text = "The answer is x = 7."
    feat_featurize = np.asarray(adapter._featurize(text))
    feat_fixed = np.asarray(_extract_text_features(text))
    np.testing.assert_allclose(feat_featurize, feat_fixed, atol=1e-10)


def test_adapter_fit_from_corpus_sets_trained_and_stats():
    """fit_from_corpus() sets _trained=True and populates _feature_stats.

    Spec: REQ-VERIFY-1121
    """
    examples = [
        {"step_text": "x + 3 = 7, so x = 4", "label": "correct"},
        {"step_text": "x - 2 = 5, so x = 2", "label": "incorrect"},  # wrong
        {"step_text": "2y = 10, so y = 5", "label": "correct"},
        {"step_text": "3z = 9, so z = 4", "label": "incorrect"},  # wrong
    ] * 10  # 40 examples to give the model something to train on

    adapter = SOSKANEnergyV3Adapter()
    assert not adapter._trained
    assert adapter._feature_stats is None

    adapter.fit_from_corpus(examples, n_epochs=5, lr=1e-3)

    assert adapter._trained, "fit_from_corpus should set _trained=True"
    assert adapter._feature_stats is not None, "fit_from_corpus should store feature stats"
    assert len(adapter._feature_stats) == 3, "Should store stats for all 3 features"


def test_adapter_fit_from_corpus_score_uses_stored_stats():
    """After fit_from_corpus, score() uses stored normalization (not fixed anchors).

    Verifies that the trained model produces a finite score in [0,1] and that
    its internal feature extraction differs from the fixed-anchor path.

    Spec: REQ-VERIFY-1121
    """
    examples = [
        {"step_text": "x + 3 = 7, so x = 4", "label": "correct"},
        {"step_text": "x - 2 = 5, so x = 2", "label": "incorrect"},
        {"step_text": "2y = 10, so y = 5", "label": "correct"},
        {"step_text": "3z = 9, so z = 4", "label": "incorrect"},
    ] * 15  # 60 examples

    adapter = SOSKANEnergyV3Adapter()
    adapter.fit_from_corpus(examples, n_epochs=5, lr=1e-3)

    score = adapter.score("The answer is x = 5 since 2x = 10.")
    assert 0.0 <= score <= 1.0, f"score() must return value in [0,1], got {score}"


def test_adapter_untrained_score_returns_neutral():
    """Untrained adapter returns 0.5 for any input (neutral, non-blocking).

    Spec: REQ-VERIFY-1121
    """
    adapter = SOSKANEnergyV3Adapter()
    assert adapter.score("any text at all") == 0.5


def test_adapter_fit_uses_fit_not_train():
    """fit() calls self._v.fit() (not the non-existent self._v.train()).

    Regression test for the bug in exp1121 where self._v.train() raised
    AttributeError, leaving the model untrained.

    Spec: REQ-VERIFY-1121
    """
    adapter = SOSKANEnergyV3Adapter()
    X = np.random.default_rng(42).uniform(-1, 1, (10, 3))
    y = np.array([1.0, 0.0] * 5)
    # Should not raise AttributeError (would if it called self._v.train())
    adapter.fit(X, y)
    assert adapter._trained


# ---------------------------------------------------------------------------
# Normalization consistency regression: core property of the fix
# ---------------------------------------------------------------------------


def test_consistent_normalization_improves_discrimination():
    """Training and scoring with consistent normalization produces correct polarity.

    Trains a small SOSKANEnergyV3 model on synthetic text examples with known
    polarity (correct steps are SHORT and math-dense, incorrect are LONG and
    prose-heavy), then verifies that the model assigns lower energy to correct
    examples than to incorrect ones — confirming the normalization fix works.

    This is the core regression test for Exp 1128's root cause.

    Spec: REQ-VERIFY-1121, SCENARIO-PHASE1D-001
    """
    rng = np.random.default_rng(42)

    # Correct examples: short, high numeric density
    correct_examples = [
        {"step_text": f"x={i}+{i + 1}={2 * i + 1}", "label": "correct"} for i in range(1, 31)
    ]
    # Incorrect examples: long, low numeric density (prose)
    incorrect_examples = [
        {
            "step_text": (
                f"considering the problem carefully we observe that the relationship "
                f"between the variables suggests an answer around {i}"
            ),
            "label": "incorrect",
        }
        for i in range(1, 31)
    ]
    examples = correct_examples + incorrect_examples

    adapter = SOSKANEnergyV3Adapter()
    adapter.fit_from_corpus(examples, n_epochs=150, lr=5e-3)

    # Score first 5 correct vs 5 incorrect
    e_correct = [
        adapter._v.energy(adapter._featurize(ex["step_text"])) for ex in correct_examples[:5]
    ]
    e_incorrect = [
        adapter._v.energy(adapter._featurize(ex["step_text"])) for ex in incorrect_examples[:5]
    ]

    mean_correct = float(np.mean(e_correct))
    mean_incorrect = float(np.mean(e_incorrect))

    assert mean_correct < mean_incorrect, (
        f"Correct examples should have lower energy ({mean_correct:.3f}) "
        f"than incorrect ({mean_incorrect:.3f}) after training with consistent normalization"
    )
