"""Tests for AdversarialV4Result (Exp 504 — GSM-Symbolic Adversarial v4).

Spec: REQ-BENCH-049, REQ-BENCH-050, REQ-BENCH-051,
      SCENARIO-BENCH-068, SCENARIO-BENCH-069, SCENARIO-BENCH-070
"""

from __future__ import annotations

import pytest

from carnot.pipeline.adversarial_v4_result import AdversarialV4Result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def result_robust():
    """Baseline drops 15pp, pipeline drops only 8pp — Carnot is more robust.

    standard_drop_baseline = 0.80 - 0.65 = 0.15
    standard_drop_pipeline = 0.78 - 0.70 = 0.08
    robustness_delta        = 0.15 - 0.08 = 0.07 > 0 → carnot_more_robust=True

    Spec: SCENARIO-BENCH-068
    """
    return AdversarialV4Result(
        model_id="TestModel",
        standard_baseline=0.80,
        standard_pipeline=0.78,
        adversarial_baseline=0.65,
        adversarial_pipeline=0.70,
        n=100,
    )


@pytest.fixture
def result_not_robust():
    """Baseline drops 10pp, pipeline drops 12pp — Carnot is LESS robust.

    standard_drop_baseline = 0.80 - 0.70 = 0.10
    standard_drop_pipeline = 0.78 - 0.66 = 0.12
    robustness_delta        = 0.10 - 0.12 = -0.02 < 0 → carnot_more_robust=False
    """
    return AdversarialV4Result(
        model_id="WeakModel",
        standard_baseline=0.80,
        standard_pipeline=0.78,
        adversarial_baseline=0.70,
        adversarial_pipeline=0.66,
        n=50,
    )


@pytest.fixture
def result_equal_drop():
    """Both baseline and pipeline drop by exactly the same amount.

    robustness_delta = 0 → carnot_more_robust=False (strict >).
    """
    return AdversarialV4Result(
        model_id="EqualModel",
        standard_baseline=0.80,
        standard_pipeline=0.80,
        adversarial_baseline=0.70,
        adversarial_pipeline=0.70,
        n=100,
    )


# ---------------------------------------------------------------------------
# Tests: standard_improvement
# ---------------------------------------------------------------------------


def test_standard_improvement_positive(result_robust):
    """REQ-BENCH-050: standard_improvement = standard_pipeline - standard_baseline."""
    # 0.78 - 0.80 = -0.02 (pipeline slightly worse on standard — uncommon but valid)
    assert abs(result_robust.standard_improvement - (-0.02)) < 1e-9


def test_standard_improvement_zero():
    """standard_improvement is zero when pipeline equals baseline on standard questions."""
    r = AdversarialV4Result(
        model_id="M", standard_baseline=0.75, standard_pipeline=0.75,
        adversarial_baseline=0.60, adversarial_pipeline=0.65, n=100,
    )
    assert r.standard_improvement == pytest.approx(0.0)


def test_standard_improvement_positive_value():
    """standard_improvement is positive when pipeline beats baseline on standard questions."""
    r = AdversarialV4Result(
        model_id="M", standard_baseline=0.70, standard_pipeline=0.80,
        adversarial_baseline=0.60, adversarial_pipeline=0.65, n=100,
    )
    assert r.standard_improvement == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Tests: adversarial_improvement
# ---------------------------------------------------------------------------


def test_adversarial_improvement_positive(result_robust):
    """REQ-BENCH-050: adversarial_improvement = adversarial_pipeline - adversarial_baseline."""
    # 0.70 - 0.65 = 0.05
    assert abs(result_robust.adversarial_improvement - 0.05) < 1e-9


def test_adversarial_improvement_negative(result_not_robust):
    """adversarial_improvement can be negative (pipeline hurts on adversarial)."""
    # 0.66 - 0.70 = -0.04
    assert result_not_robust.adversarial_improvement == pytest.approx(-0.04)


# ---------------------------------------------------------------------------
# Tests: standard_drop_baseline
# ---------------------------------------------------------------------------


def test_standard_drop_baseline(result_robust):
    """REQ-BENCH-049: standard_drop_baseline = standard_baseline - adversarial_baseline."""
    # 0.80 - 0.65 = 0.15
    assert abs(result_robust.standard_drop_baseline - 0.15) < 1e-9


def test_standard_drop_baseline_zero():
    """Drop is zero when LLM is equally accurate on standard and adversarial."""
    r = AdversarialV4Result(
        model_id="M", standard_baseline=0.80, standard_pipeline=0.80,
        adversarial_baseline=0.80, adversarial_pipeline=0.80, n=100,
    )
    assert r.standard_drop_baseline == pytest.approx(0.0)


def test_standard_drop_baseline_negative():
    """Drop can be negative if adversarial questions are accidentally easier (honest reporting)."""
    r = AdversarialV4Result(
        model_id="M", standard_baseline=0.60, standard_pipeline=0.70,
        adversarial_baseline=0.70, adversarial_pipeline=0.75, n=100,
    )
    assert r.standard_drop_baseline == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Tests: standard_drop_pipeline
# ---------------------------------------------------------------------------


def test_standard_drop_pipeline(result_robust):
    """REQ-BENCH-049: standard_drop_pipeline = standard_pipeline - adversarial_pipeline."""
    # 0.78 - 0.70 = 0.08
    assert abs(result_robust.standard_drop_pipeline - 0.08) < 1e-9


def test_standard_drop_pipeline_zero():
    """Pipeline drop is zero when pipeline accuracy is equal on standard and adversarial."""
    r = AdversarialV4Result(
        model_id="M", standard_baseline=0.80, standard_pipeline=0.75,
        adversarial_baseline=0.70, adversarial_pipeline=0.75, n=100,
    )
    # standard_pipeline == adversarial_pipeline → drop is 0
    assert r.standard_drop_pipeline == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: robustness_delta
# ---------------------------------------------------------------------------


def test_robustness_delta_positive(result_robust):
    """REQ-BENCH-049: robustness_delta > 0 when pipeline drop < baseline drop.

    SCENARIO-BENCH-068: baseline drops 15pp, pipeline drops 8pp → delta = 7pp.
    """
    # 0.15 - 0.08 = 0.07
    assert abs(result_robust.robustness_delta - 0.07) < 1e-9


def test_robustness_delta_negative(result_not_robust):
    """robustness_delta < 0 when pipeline drops MORE than baseline under adversarial."""
    # 0.10 - 0.12 = -0.02
    assert result_not_robust.robustness_delta == pytest.approx(-0.02)


def test_robustness_delta_zero(result_equal_drop):
    """robustness_delta == 0 when both drops are identical."""
    assert result_equal_drop.robustness_delta == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: carnot_more_robust
# ---------------------------------------------------------------------------


def test_carnot_more_robust_true(result_robust):
    """REQ-BENCH-049: carnot_more_robust=True when baseline drops 15pp and pipeline 8pp.

    SCENARIO-BENCH-069: the RETRO-039 robustness claim holds.
    """
    assert result_robust.carnot_more_robust is True


def test_carnot_more_robust_false_negative_delta(result_not_robust):
    """carnot_more_robust=False when robustness_delta <= 0."""
    assert result_not_robust.carnot_more_robust is False


def test_carnot_more_robust_false_zero_delta(result_equal_drop):
    """carnot_more_robust=False when robustness_delta == 0 (strict >)."""
    assert result_equal_drop.carnot_more_robust is False


# ---------------------------------------------------------------------------
# Tests: to_dict
# ---------------------------------------------------------------------------


def test_to_dict_contains_all_fields(result_robust):
    """REQ-BENCH-051: to_dict() must contain all required fields.

    SCENARIO-BENCH-070: the artifact is JSON-serializable and complete.
    """
    d = result_robust.to_dict()
    required_keys = {
        "model_id", "n",
        "standard_baseline", "standard_pipeline",
        "adversarial_baseline", "adversarial_pipeline",
        "standard_improvement", "adversarial_improvement",
        "standard_drop_baseline", "standard_drop_pipeline",
        "robustness_delta", "carnot_more_robust",
    }
    assert required_keys.issubset(d.keys())


def test_to_dict_values_match_properties(result_robust):
    """to_dict() values match the corresponding computed properties."""
    d = result_robust.to_dict()
    assert d["model_id"] == result_robust.model_id
    assert d["n"] == result_robust.n
    assert d["standard_baseline"] == result_robust.standard_baseline
    assert d["adversarial_baseline"] == result_robust.adversarial_baseline
    assert d["standard_improvement"] == pytest.approx(result_robust.standard_improvement)
    assert d["adversarial_improvement"] == pytest.approx(result_robust.adversarial_improvement)
    assert d["standard_drop_baseline"] == pytest.approx(result_robust.standard_drop_baseline)
    assert d["standard_drop_pipeline"] == pytest.approx(result_robust.standard_drop_pipeline)
    assert d["robustness_delta"] == pytest.approx(result_robust.robustness_delta)
    assert d["carnot_more_robust"] == result_robust.carnot_more_robust


def test_to_dict_is_json_serializable(result_robust):
    """to_dict() output can be serialized to JSON without error."""
    import json

    d = result_robust.to_dict()
    serialized = json.dumps(d)
    restored = json.loads(serialized)
    assert restored["model_id"] == "TestModel"
    assert restored["carnot_more_robust"] is True


def test_to_dict_not_robust(result_not_robust):
    """to_dict() correctly records carnot_more_robust=False for a non-robust result."""
    d = result_not_robust.to_dict()
    assert d["carnot_more_robust"] is False
    assert d["robustness_delta"] == pytest.approx(-0.02)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_perfect_pipeline_robustness():
    """When pipeline has zero adversarial drop, robustness_delta equals baseline drop."""
    r = AdversarialV4Result(
        model_id="Perfect",
        standard_baseline=0.80, standard_pipeline=0.80,
        adversarial_baseline=0.60, adversarial_pipeline=0.80,
        n=200,
    )
    # standard_drop_baseline = 0.80 - 0.60 = 0.20
    # standard_drop_pipeline = 0.80 - 0.80 = 0.00
    # robustness_delta = 0.20
    assert r.standard_drop_baseline == pytest.approx(0.20)
    assert r.standard_drop_pipeline == pytest.approx(0.00)
    assert r.robustness_delta == pytest.approx(0.20)
    assert r.carnot_more_robust is True


def test_zero_n_valid():
    """n=0 is valid structurally (result arithmetic still works)."""
    r = AdversarialV4Result(
        model_id="Empty",
        standard_baseline=0.0, standard_pipeline=0.0,
        adversarial_baseline=0.0, adversarial_pipeline=0.0,
        n=0,
    )
    assert r.robustness_delta == pytest.approx(0.0)
    assert r.carnot_more_robust is False


def test_model_id_preserved():
    """model_id is stored as-is and returned by to_dict()."""
    r = AdversarialV4Result(
        model_id="Gemma4-Q4KM",
        standard_baseline=0.75, standard_pipeline=0.77,
        adversarial_baseline=0.68, adversarial_pipeline=0.73,
        n=100,
    )
    assert r.model_id == "Gemma4-Q4KM"
    assert r.to_dict()["model_id"] == "Gemma4-Q4KM"
