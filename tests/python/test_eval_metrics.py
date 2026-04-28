"""Tests for canonical metrics in `carnot.eval.metrics`.

Spec: REQ-EVAL-001, REQ-EVAL-002, REQ-EVAL-003.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.eval.metrics import auroc, f1_score, precision_recall


# REQ-EVAL-001
def test_auroc_perfect_separator() -> None:
    """Catches the 2026-04-28 inverted-AUROC bug that returned 0.0 here."""
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


# REQ-EVAL-001
def test_auroc_anti_correlated() -> None:
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(0.0)


# REQ-EVAL-001
def test_auroc_single_class_returns_05() -> None:
    assert auroc([1, 1, 1], [0.1, 0.2, 0.3]) == 0.5
    assert auroc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.5


# REQ-EVAL-001
def test_auroc_known_mixed_case() -> None:
    """y=[0,0,1,0,1,1], s=[0.1,0.4,0.35,0.8,0.7,0.6]; concordant=5/9."""
    assert auroc([0, 0, 1, 0, 1, 1], [0.1, 0.4, 0.35, 0.8, 0.7, 0.6]) == pytest.approx(5 / 9)


# REQ-EVAL-001
def test_auroc_input_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        auroc([0, 1], [0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="1D"):
        auroc(np.array([[0, 1]]), np.array([[0.1, 0.2]]))


# REQ-EVAL-002
def test_auroc_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = int(rng.integers(2, 200))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.normal(size=n)
        a = auroc(y, s)
        assert 0.0 <= a <= 1.0, f"trial {trial}: AUROC={a} outside [0,1]"


# REQ-EVAL-002
def test_auroc_symmetry_with_flipped_labels() -> None:
    """auroc(y, s) + auroc(1-y, s) == 1.0. Catches sign errors immediately."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        n = int(rng.integers(10, 100))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.normal(size=n)
        assert auroc(y, s) + auroc(1 - y, s) == pytest.approx(1.0, abs=1e-9)


# REQ-EVAL-002
def test_auroc_monotone_score_invariance() -> None:
    rng = np.random.default_rng(7)
    for _ in range(10):
        n = int(rng.integers(20, 100))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.normal(size=n)
        for transform in [
            lambda x: x * 3.0 + 5.0,
            lambda x: np.exp(x),
            lambda x: np.cbrt(x),
        ]:
            assert auroc(y, s) == pytest.approx(auroc(y, transform(s)), abs=1e-9)


# REQ-EVAL-002
def test_auroc_score_negation_inverts() -> None:
    rng = np.random.default_rng(123)
    for _ in range(10):
        n = int(rng.integers(10, 100))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.normal(size=n)
        assert auroc(y, s) + auroc(y, -s) == pytest.approx(1.0, abs=1e-9)


# REQ-EVAL-003
def test_auroc_matches_sklearn() -> None:
    """50-trial cross-validation against sklearn — gold-standard reference."""
    pytest.importorskip("sklearn", reason="sklearn is a test-only dependency")
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2026)
    for trial in range(50):
        n = int(rng.integers(10, 200))
        y = rng.integers(0, 2, size=n)
        if y.sum() == 0 or y.sum() == n:
            continue
        s = rng.normal(size=n)
        ours = auroc(y, s)
        theirs = float(roc_auc_score(y, s))
        assert ours == pytest.approx(theirs, abs=1e-9), (
            f"trial {trial}: carnot AUROC {ours} vs sklearn {theirs}"
        )


# REQ-EVAL-001
def test_precision_recall_perfect() -> None:
    p, r = precision_recall([0, 0, 1, 1], [0, 0, 1, 1])
    assert p == 1.0
    assert r == 1.0


# REQ-EVAL-001
def test_precision_recall_no_positives_predicted() -> None:
    p, r = precision_recall([0, 0, 1, 1], [0, 0, 0, 0])
    assert p == 0.0
    assert r == 0.0


# REQ-EVAL-001
def test_precision_recall_partial_overlap() -> None:
    """tp=2, fp=1, fn=1 → precision=recall=2/3."""
    p, r = precision_recall([0, 0, 1, 1, 1], [0, 1, 1, 1, 0])
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(2 / 3)


# REQ-EVAL-001
def test_f1_score_harmonic_mean() -> None:
    assert f1_score([0, 0, 1, 1, 1], [0, 1, 1, 1, 0]) == pytest.approx(2 / 3)


# REQ-EVAL-001
def test_f1_score_zero_when_undefined() -> None:
    assert f1_score([0, 0, 1, 1], [0, 0, 0, 0]) == 0.0
