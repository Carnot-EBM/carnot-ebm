"""Tests for SpilledEnergyDetector — per-token logit-discrepancy hallucination signal.

Covers:
  - SpilledEnergyToken dataclass construction
  - compute_detector_spilled_energy: uniform vs peaked logit distributions
  - compute_detector_spilled_energy: temperature scaling
  - SpilledEnergyDetectorResult dataclass fields
  - SpilledEnergyDetector.score(): uniform logits → should_verify=True
  - SpilledEnergyDetector.score(): peaked logits → should_verify=False
  - SpilledEnergyDetector.score(): 1-D input handled correctly
  - SpilledEnergyDetector.score(): high_spill_fraction triggers should_verify
  - SpilledEnergyDetector.score(): per_token list length matches input
  - SpilledEnergyDetector.score_from_text(): returns valid result
  - SpilledEnergyDetector.score_from_text(): deterministic
  - SpilledEnergyDetector.score_from_text(): different texts → different results
  - SpilledEnergyDetector.score_from_text(): per_token is empty
  - SpilledEnergyDetector invalid constructor args
  - compute_detector_spilled_energy always >= 0
  - SpilledEnergyDetectorResult: max_spilled >= mean_spilled
  - carnot.pipeline exports new classes

Spec: REQ-VERIFY-092, REQ-VERIFY-093
SCENARIO-VERIFY-123, SCENARIO-VERIFY-124, SCENARIO-VERIFY-125
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from carnot.pipeline.spilled_energy import (
    SpilledEnergyDetector,
    SpilledEnergyDetectorResult,
    SpilledEnergyToken,
    compute_detector_spilled_energy,
)


# ---------------------------------------------------------------------------
# SpilledEnergyToken
# ---------------------------------------------------------------------------


def test_spilled_energy_token_construction() -> None:
    """SpilledEnergyToken stores position, token_id, spilled_energy correctly.

    Spec: REQ-VERIFY-092
    """
    tok = SpilledEnergyToken(position=3, token_id=42, spilled_energy=1.5)
    assert tok.position == 3
    assert tok.token_id == 42
    assert tok.spilled_energy == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# compute_detector_spilled_energy
# ---------------------------------------------------------------------------


def test_compute_detector_spilled_energy_uniform_is_high() -> None:
    """All-zero logits (uniform distribution) → spilled ≈ log(vocab_size).

    A uniform distribution is maximally uncertain. Spilled energy should be
    close to log(V) nats where V is the vocabulary size.

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-123
    """
    vocab_size = 100
    logits = jnp.zeros(vocab_size)
    se = compute_detector_spilled_energy(logits, temperature=1.0)
    expected = math.log(vocab_size)  # log(100) ≈ 4.605
    assert se == pytest.approx(expected, rel=1e-3)


def test_compute_detector_spilled_energy_peaked_is_low() -> None:
    """One very large logit (peaked distribution) → spilled ≈ 0.

    A near-one-hot distribution is maximally confident. Spilled energy should
    be very close to 0.

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-124
    """
    vocab_size = 100
    logits = jnp.zeros(vocab_size).at[0].set(100.0)  # token 0 dominates
    se = compute_detector_spilled_energy(logits, temperature=1.0)
    # With logit=100 for one token and 0 for all others, softmax gives p≈1 for token 0
    # Spilled energy = entropy of this near-one-hot distribution ≈ 0
    assert se < 0.01


def test_compute_detector_spilled_energy_temperature_scaling() -> None:
    """Very high temperature → spilled energy approaches log(vocab_size).

    The formula: spilled = log(sum exp(logit_j/T)) - sum p_j * logit_j
    As T → ∞, softmax(logits/T) → uniform(1/V), so softmax approaches uniform,
    and spilled approaches log(V) (maximum entropy).

    This test verifies the asymptotic behavior at very high T for peaked logits.

    Spec: REQ-VERIFY-092
    """
    vocab_size = 100
    # Moderately peaked logits: one dominant value
    logits = jnp.zeros(vocab_size).at[0].set(10.0)

    # Very high temperature → near-uniform distribution → spilled approaches log(V)
    se_high_t = compute_detector_spilled_energy(logits, temperature=100.0)
    expected_max = math.log(vocab_size)  # log(100) ≈ 4.605
    # At T=100, softmax is nearly uniform → spilled ≈ log(V)
    assert se_high_t == pytest.approx(expected_max, rel=0.05)

    # At T=1.0 with this logit, softmax is peaked → spilled is low
    se_t1 = compute_detector_spilled_energy(logits, temperature=1.0)
    assert se_t1 < 0.1  # very peaked → very low entropy → low spilled energy


def test_compute_detector_spilled_energy_nonnegative() -> None:
    """Spilled energy is always >= 0 for any logit inputs.

    Mathematically: spilled_energy = T * H(softmax(logits/T)) >= 0 since entropy >= 0.

    Spec: REQ-VERIFY-092
    """
    rng = np.random.default_rng(42)
    for _ in range(10):
        logits = jnp.array(rng.normal(size=50).astype(np.float32))
        se = compute_detector_spilled_energy(logits)
        assert se >= 0.0


# ---------------------------------------------------------------------------
# SpilledEnergyDetectorResult
# ---------------------------------------------------------------------------


def test_spilled_energy_detector_result_fields() -> None:
    """SpilledEnergyDetectorResult has all required fields with correct types.

    Spec: REQ-VERIFY-092
    """
    result = SpilledEnergyDetectorResult(
        mean_spilled=1.5,
        max_spilled=3.0,
        high_spill_fraction=0.3,
        should_verify=True,
        per_token=[],
    )
    assert result.mean_spilled == pytest.approx(1.5)
    assert result.max_spilled == pytest.approx(3.0)
    assert result.high_spill_fraction == pytest.approx(0.3)
    assert result.should_verify is True
    assert isinstance(result.per_token, list)


# ---------------------------------------------------------------------------
# SpilledEnergyDetector.score()
# ---------------------------------------------------------------------------


def test_spilled_energy_detector_score_uniform() -> None:
    """Uniform logits → should_verify=True with low fraction threshold.

    All-zero logits give uniform distribution → high spilled energy → should_verify.

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-123
    """
    vocab_size = 100
    T = 5
    # All-zero logits: uniform distribution at every token position
    logits = jnp.zeros((T, vocab_size))
    # Use a very low fraction threshold so even one uncertain token triggers
    detector = SpilledEnergyDetector(spill_threshold=2.0, high_spill_fraction_threshold=0.1)
    result = detector.score(logits)

    # With uniform distribution, spilled energy ≈ log(100) ≈ 4.6 >> 2.0 threshold
    assert result.mean_spilled > 2.0
    assert result.high_spill_fraction == pytest.approx(1.0)  # all tokens are high-spill
    assert result.should_verify is True


def test_spilled_energy_detector_score_peaked() -> None:
    """Peaked logits, spill_threshold=2.0 → should_verify=False (confident model).

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-124
    """
    vocab_size = 100
    T = 5
    # One dominant logit at position 0 for every token in the sequence
    logits_row = jnp.zeros(vocab_size).at[0].set(100.0)
    logits = jnp.stack([logits_row] * T)
    detector = SpilledEnergyDetector(spill_threshold=2.0, high_spill_fraction_threshold=0.2)
    result = detector.score(logits)

    # Near-one-hot distribution → spilled energy ≈ 0 << 2.0 threshold
    assert result.max_spilled < 0.01
    assert result.high_spill_fraction == pytest.approx(0.0)
    assert result.should_verify is False


def test_spilled_energy_detector_score_1d_input() -> None:
    """Single-token 1-D logit array works (treated as T=1 sequence).

    Spec: REQ-VERIFY-092
    """
    vocab_size = 50
    logits = jnp.zeros(vocab_size)  # shape (V,)
    detector = SpilledEnergyDetector()
    result = detector.score(logits)

    assert isinstance(result, SpilledEnergyDetectorResult)
    assert len(result.per_token) == 1
    assert result.per_token[0].position == 0


def test_spilled_energy_detector_score_high_spill_fraction() -> None:
    """Many uncertain tokens → should_verify=True.

    Create a mix: most tokens have uniform logits (high spill), a few are peaked.

    Spec: REQ-VERIFY-092, SCENARIO-VERIFY-123
    """
    vocab_size = 20
    T = 10
    # 8 uncertain (uniform) + 2 confident (peaked)
    uniform_row = jnp.zeros(vocab_size)
    peaked_row = jnp.zeros(vocab_size).at[0].set(100.0)
    rows = [uniform_row] * 8 + [peaked_row] * 2
    logits = jnp.stack(rows)

    detector = SpilledEnergyDetector(spill_threshold=2.0, high_spill_fraction_threshold=0.2)
    result = detector.score(logits)

    # 8/10 = 0.8 fraction of tokens are high-spill → should_verify=True
    assert result.high_spill_fraction == pytest.approx(0.8)
    assert result.should_verify is True


def test_spilled_energy_detector_score_per_token_records() -> None:
    """per_token list has correct length and valid SpilledEnergyToken entries.

    Spec: REQ-VERIFY-092
    """
    vocab_size = 30
    T = 7
    logits = jnp.zeros((T, vocab_size))
    detector = SpilledEnergyDetector()
    result = detector.score(logits)

    assert len(result.per_token) == T
    for i, tok in enumerate(result.per_token):
        assert isinstance(tok, SpilledEnergyToken)
        assert tok.position == i
        assert tok.spilled_energy >= 0.0


def test_spilled_energy_detector_result_mean_max_relationship() -> None:
    """max_spilled >= mean_spilled always holds.

    Spec: REQ-VERIFY-092
    """
    vocab_size = 50
    T = 6
    # Mix of uniform and peaked tokens
    rows = [jnp.zeros(vocab_size)] * 4 + [jnp.zeros(vocab_size).at[0].set(100.0)] * 2
    logits = jnp.stack(rows)
    detector = SpilledEnergyDetector()
    result = detector.score(logits)

    assert result.max_spilled >= result.mean_spilled


# ---------------------------------------------------------------------------
# SpilledEnergyDetector.score_from_text()
# ---------------------------------------------------------------------------


def test_spilled_energy_detector_score_from_text_valid() -> None:
    """score_from_text returns a valid SpilledEnergyDetectorResult.

    Spec: REQ-VERIFY-093, SCENARIO-VERIFY-125
    """
    detector = SpilledEnergyDetector()
    result = detector.score_from_text("The capital of France is Paris.")

    assert isinstance(result, SpilledEnergyDetectorResult)
    assert isinstance(result.should_verify, bool)
    assert result.mean_spilled >= 0.0
    assert result.max_spilled >= 0.0
    assert 0.0 <= result.high_spill_fraction <= 1.0


def test_spilled_energy_detector_score_from_text_deterministic() -> None:
    """Same text → same result every time (deterministic hash-based proxy).

    Spec: REQ-VERIFY-093, SCENARIO-VERIFY-125
    """
    detector = SpilledEnergyDetector()
    text = "The quick brown fox jumps over the lazy dog."

    result1 = detector.score_from_text(text)
    result2 = detector.score_from_text(text)
    result3 = detector.score_from_text(text)

    assert result1.mean_spilled == pytest.approx(result2.mean_spilled)
    assert result1.max_spilled == pytest.approx(result2.max_spilled)
    assert result1.high_spill_fraction == pytest.approx(result2.high_spill_fraction)
    assert result1.should_verify == result2.should_verify
    assert result2.mean_spilled == pytest.approx(result3.mean_spilled)


def test_spilled_energy_detector_score_from_text_different_texts() -> None:
    """Different texts typically produce different proxy spilled energy values.

    Spec: REQ-VERIFY-093
    """
    detector = SpilledEnergyDetector()
    result_a = detector.score_from_text("Paris is the capital of France.")
    result_b = detector.score_from_text("Quantum mechanics describes subatomic particles.")

    # Different inputs should produce different hash-derived values
    # (not guaranteed for every pair, but almost certain for these two)
    assert result_a.mean_spilled != pytest.approx(result_b.mean_spilled) or \
           result_a.max_spilled != pytest.approx(result_b.max_spilled)


def test_spilled_energy_detector_score_from_text_empty_per_token() -> None:
    """per_token list is empty in text mode (no real token records available).

    Spec: REQ-VERIFY-093, SCENARIO-VERIFY-125
    """
    detector = SpilledEnergyDetector()
    result = detector.score_from_text("Some response text.")
    assert result.per_token == []


# ---------------------------------------------------------------------------
# Invalid constructor arguments
# ---------------------------------------------------------------------------


def test_spilled_energy_detector_invalid_spill_threshold() -> None:
    """ValueError raised for spill_threshold <= 0.

    Spec: REQ-VERIFY-092
    """
    with pytest.raises(ValueError, match="spill_threshold must be > 0"):
        SpilledEnergyDetector(spill_threshold=0.0)

    with pytest.raises(ValueError, match="spill_threshold must be > 0"):
        SpilledEnergyDetector(spill_threshold=-1.0)


def test_spilled_energy_detector_invalid_fraction_threshold() -> None:
    """ValueError raised for high_spill_fraction_threshold outside (0, 1).

    Spec: REQ-VERIFY-092
    """
    with pytest.raises(ValueError, match="high_spill_fraction_threshold must be in"):
        SpilledEnergyDetector(high_spill_fraction_threshold=0.0)

    with pytest.raises(ValueError, match="high_spill_fraction_threshold must be in"):
        SpilledEnergyDetector(high_spill_fraction_threshold=1.0)

    with pytest.raises(ValueError, match="high_spill_fraction_threshold must be in"):
        SpilledEnergyDetector(high_spill_fraction_threshold=1.5)


# ---------------------------------------------------------------------------
# Pipeline exports
# ---------------------------------------------------------------------------


def test_pipeline_exports_new_classes() -> None:
    """SpilledEnergyDetector and related classes are exported from carnot.pipeline.

    Spec: REQ-VERIFY-092, REQ-VERIFY-093
    """
    from carnot.pipeline import (  # noqa: F401
        SpilledEnergyDetector,
        SpilledEnergyDetectorResult,
        SpilledEnergyToken,
        compute_detector_spilled_energy,
    )

    # Verify they are the correct classes
    assert SpilledEnergyDetector is not None
    assert SpilledEnergyDetectorResult is not None
    assert SpilledEnergyToken is not None
    assert callable(compute_detector_spilled_energy)
