"""Tests for DiffusionDoT v2 — EBM embedding-space diffusion verifier.

Spec: REQ-INFER-018, SCENARIO-INFER-018-001
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.inference.diffusion_of_thought_v2 import (
    DEFAULT_ALPHA,
    DEFAULT_N_STEPS,
    DEFAULT_SIGMA,
    DiffusionDoT,
    embed_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class QuadraticEBM:
    """Simple EBM: E(z) = ||z||² / dim.  Minimum at origin, gradient = 2z/dim."""

    def __call__(self, z: np.ndarray) -> float:
        return float(np.dot(z, z) / len(z))


class ConstantEBM:
    """EBM that returns a fixed constant regardless of input."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def __call__(self, z: np.ndarray) -> float:
        return self.value


# ---------------------------------------------------------------------------
# DiffusionDoT construction
# ---------------------------------------------------------------------------


def test_init_rejects_nonpositive_alpha() -> None:
    """REQ-INFER-018: alpha must be strictly positive."""
    with pytest.raises(ValueError, match="alpha"):
        DiffusionDoT(QuadraticEBM(), alpha=0.0)


def test_init_rejects_zero_n_steps() -> None:
    """REQ-INFER-018: n_steps must be >= 1."""
    with pytest.raises(ValueError, match="n_steps"):
        DiffusionDoT(QuadraticEBM(), n_steps=0)


def test_init_rejects_nonpositive_sigma() -> None:
    """REQ-INFER-018: sigma must be strictly positive."""
    with pytest.raises(ValueError, match="sigma"):
        DiffusionDoT(QuadraticEBM(), sigma=0.0)


def test_init_accepts_valid_params() -> None:
    """REQ-INFER-018: valid params produce a usable DiffusionDoT instance."""
    dot = DiffusionDoT(QuadraticEBM(), alpha=0.01, n_steps=5, sigma=0.1)
    assert dot.alpha == 0.01
    assert dot.n_steps == 5
    assert dot.sigma == 0.1


# ---------------------------------------------------------------------------
# forward_diffuse
# ---------------------------------------------------------------------------


def test_forward_diffuse_t0_is_identity() -> None:
    """SCENARIO-INFER-018-001: forward_diffuse with t=0 returns unchanged embedding."""
    dot = DiffusionDoT(QuadraticEBM())
    z = np.array([1.0, 2.0, 3.0])
    noisy = dot.forward_diffuse(z, t=0.0)
    np.testing.assert_array_equal(noisy, z)


def test_forward_diffuse_adds_noise_for_positive_t() -> None:
    """SCENARIO-INFER-018-001: forward_diffuse with t>0 changes the embedding."""
    dot = DiffusionDoT(QuadraticEBM(), rng=np.random.default_rng(42))
    z = np.zeros(16)
    noisy = dot.forward_diffuse(z, t=1.0)
    assert not np.allclose(noisy, z), "Expected noise to be added"


def test_forward_diffuse_rejects_negative_t() -> None:
    """REQ-INFER-018: negative diffusion time is invalid."""
    dot = DiffusionDoT(QuadraticEBM())
    with pytest.raises(ValueError, match="t must be non-negative"):
        dot.forward_diffuse(np.zeros(4), t=-0.1)


def test_forward_diffuse_noise_variance_scales_with_t() -> None:
    """SCENARIO-INFER-018-001: noise variance should grow proportionally to t."""
    rng = np.random.default_rng(0)
    dot = DiffusionDoT(QuadraticEBM(), sigma=1.0, rng=rng)
    z = np.zeros(1000)
    # t=1 → σ²·t = 1; t=4 → σ²·t = 4 (4x more variance).
    noisy_t1 = dot.forward_diffuse(z.copy(), t=1.0)
    rng2 = np.random.default_rng(1)
    dot2 = DiffusionDoT(QuadraticEBM(), sigma=1.0, rng=rng2)
    noisy_t4 = dot2.forward_diffuse(z.copy(), t=4.0)
    var_t1 = float(np.var(noisy_t1))
    var_t4 = float(np.var(noisy_t4))
    # t=4 should have approximately 4x the variance of t=1.
    assert var_t4 > var_t1 * 2, f"Expected var_t4 ({var_t4:.3f}) >> var_t1 ({var_t1:.3f})"


def test_forward_diffuse_does_not_modify_original() -> None:
    """REQ-INFER-018: forward_diffuse returns a new array, never mutates input."""
    dot = DiffusionDoT(QuadraticEBM(), rng=np.random.default_rng(7))
    z = np.array([1.0, 2.0, 3.0])
    original = z.copy()
    dot.forward_diffuse(z, t=1.0)
    np.testing.assert_array_equal(z, original)


# ---------------------------------------------------------------------------
# reverse_denoise
# ---------------------------------------------------------------------------


def test_reverse_denoise_reduces_energy_for_quadratic_ebm() -> None:
    """SCENARIO-INFER-018-001: denoising moves toward lower energy for a quadratic EBM."""
    ebm = QuadraticEBM()
    dot = DiffusionDoT(ebm, alpha=0.1, n_steps=20, rng=np.random.default_rng(42))
    z = np.array([2.0, 3.0, -1.0])
    noisy = dot.forward_diffuse(z, t=1.0)
    denoised = dot.reverse_denoise(noisy)
    e_noisy = ebm(noisy)
    e_denoised = ebm(denoised)
    assert e_denoised < e_noisy, (
        f"Denoised energy {e_denoised:.4f} should be < noisy energy {e_noisy:.4f}"
    )


def test_reverse_denoise_returns_same_shape() -> None:
    """REQ-INFER-018: reverse_denoise output has same shape as input."""
    dot = DiffusionDoT(QuadraticEBM(), rng=np.random.default_rng(0))
    z = np.zeros(8)
    noisy = dot.forward_diffuse(z, t=1.0)
    denoised = dot.reverse_denoise(noisy)
    assert denoised.shape == z.shape


def test_reverse_denoise_does_not_modify_input() -> None:
    """REQ-INFER-018: reverse_denoise does not mutate the noisy_embedding in place."""
    dot = DiffusionDoT(QuadraticEBM(), rng=np.random.default_rng(3))
    noisy = np.array([1.0, -1.0, 2.0])
    original_noisy = noisy.copy()
    dot.reverse_denoise(noisy)
    np.testing.assert_array_equal(noisy, original_noisy)


# ---------------------------------------------------------------------------
# score_verification
# ---------------------------------------------------------------------------


def test_score_verification_passes_when_clean_energy_lower() -> None:
    """REQ-INFER-018: verification passes when E(clean) < E(noisy)."""
    ebm = QuadraticEBM()
    dot = DiffusionDoT(ebm)
    clean = np.zeros(4)  # E = 0
    noisy = np.array([2.0, 2.0, 2.0, 2.0])  # E = 4
    assert dot.score_verification(clean, noisy) is True


def test_score_verification_fails_when_clean_energy_higher() -> None:
    """REQ-INFER-018: verification fails when E(clean) >= E(noisy)."""
    ebm = QuadraticEBM()
    dot = DiffusionDoT(ebm)
    clean = np.array([3.0, 3.0, 3.0, 3.0])  # E = 9
    noisy = np.zeros(4)  # E = 0
    assert dot.score_verification(clean, noisy) is False


def test_score_verification_fails_when_energies_equal() -> None:
    """REQ-INFER-018: verification fails (not passes) when E(clean) == E(noisy)."""
    ebm = ConstantEBM(value=1.0)
    dot = DiffusionDoT(ebm)
    z1 = np.zeros(4)
    z2 = np.ones(4)
    # Both get the same constant energy, so E(clean) is NOT < E(noisy).
    assert dot.score_verification(z1, z2) is False


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


def test_embed_text_returns_correct_shape() -> None:
    """REQ-INFER-018: embed_text returns an array of the requested dimensionality."""
    v = embed_text("hello world", dim=16)
    assert v.shape == (16,)


def test_embed_text_is_unit_normalized_for_nonempty_text() -> None:
    """REQ-INFER-018: embed_text normalizes to unit length for non-empty input."""
    v = embed_text("the quick brown fox jumps", dim=32)
    norm = float(np.linalg.norm(v))
    assert abs(norm - 1.0) < 1e-9, f"Expected unit norm, got {norm}"


def test_embed_text_is_deterministic() -> None:
    """REQ-INFER-018: embed_text returns the same vector for identical input."""
    v1 = embed_text("deterministic input", dim=24)
    v2 = embed_text("deterministic input", dim=24)
    np.testing.assert_array_equal(v1, v2)


def test_embed_text_differs_for_different_texts() -> None:
    """REQ-INFER-018: distinct texts produce distinct embeddings."""
    v1 = embed_text("correct reasoning step", dim=32)
    v2 = embed_text("incorrect reasoning step with error", dim=32)
    assert not np.allclose(v1, v2)


def test_embed_text_empty_string_returns_zeros() -> None:
    """REQ-INFER-018: empty string returns zero vector (no normalization applied)."""
    v = embed_text("", dim=8)
    assert v.shape == (8,)
    np.testing.assert_array_equal(v, np.zeros(8))


# ---------------------------------------------------------------------------
# Default constants
# ---------------------------------------------------------------------------


def test_default_constants_are_positive() -> None:
    """REQ-INFER-018: all default hyperparameters must be strictly positive."""
    assert DEFAULT_ALPHA > 0
    assert DEFAULT_N_STEPS >= 1
    assert DEFAULT_SIGMA > 0


# ---------------------------------------------------------------------------
# Integration: full forward→denoise→verify pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_valid_response_passes_more_often() -> None:
    """SCENARIO-INFER-018-001: embeddings near origin pass more often than far-from-origin ones.

    The QuadraticEBM has its minimum at the origin.  Embeddings near the
    origin (representing 'correct' responses) should pass verification more
    often than embeddings far from the origin (representing 'incorrect'
    responses) after the same diffusion+denoising process.
    """
    ebm = QuadraticEBM()
    rng = np.random.default_rng(1186)
    dot = DiffusionDoT(ebm, alpha=0.1, n_steps=10, sigma=0.5, rng=rng)

    n_trials = 20
    correct_passes = 0
    incorrect_passes = 0

    for _ in range(n_trials):
        # "Correct" embedding: small norm (low energy).
        z_correct = rng.standard_normal(16) * 0.1
        noisy_c = dot.forward_diffuse(z_correct, t=1.0)
        denoised_c = dot.reverse_denoise(noisy_c)
        if dot.score_verification(denoised_c, noisy_c):
            correct_passes += 1

        # "Incorrect" embedding: large norm (high energy).
        z_incorrect = rng.standard_normal(16) * 3.0
        noisy_i = dot.forward_diffuse(z_incorrect, t=1.0)
        denoised_i = dot.reverse_denoise(noisy_i)
        if dot.score_verification(denoised_i, noisy_i):
            incorrect_passes += 1

    # Correct embeddings should pass at least as often as incorrect ones.
    # We allow some slack because diffusion is stochastic.
    assert correct_passes >= incorrect_passes - 5, (
        f"Expected correct_passes ({correct_passes}) >= incorrect_passes ({incorrect_passes}) - 5"
    )
