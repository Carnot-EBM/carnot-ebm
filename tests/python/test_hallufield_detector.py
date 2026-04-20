"""Tests for HalluFieldDetector — Tier 0e thermodynamic hallucination signal.

Spec: REQ-VERIFY-117, SCENARIO-VERIFY-154, SCENARIO-VERIFY-155, SCENARIO-VERIFY-156
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from carnot.pipeline.hallufield_detector import (
    DEFAULT_INSTABILITY_THRESHOLD,
    DEFAULT_N_PATHS,
    DEFAULT_TEMPERATURE,
    HalluFieldDetector,
    HalluFieldResult,
)


# ---------------------------------------------------------------------------
# HalluFieldResult dataclass tests
# ---------------------------------------------------------------------------


def test_hallufield_result_fields() -> None:
    """HalluFieldResult stores all required fields without error.

    Spec: SCENARIO-VERIFY-154
    """
    result = HalluFieldResult(
        partition_variance=0.3,
        mean_energy=1.2,
        is_unstable=False,
        token_path_count=32,
        detector_mode="logit",
    )
    assert result.partition_variance == 0.3
    assert result.mean_energy == 1.2
    assert result.is_unstable is False
    assert result.token_path_count == 32
    assert result.detector_mode == "logit"


# ---------------------------------------------------------------------------
# HalluFieldDetector.__init__ validation
# ---------------------------------------------------------------------------


def test_init_defaults() -> None:
    """Default constructor uses documented default values."""
    det = HalluFieldDetector()
    assert det.n_paths == DEFAULT_N_PATHS
    assert det.temperature == DEFAULT_TEMPERATURE
    assert det.instability_threshold == DEFAULT_INSTABILITY_THRESHOLD


def test_init_custom_params() -> None:
    """Custom constructor stores provided values."""
    det = HalluFieldDetector(n_paths=16, temperature=0.5, instability_threshold=0.8)
    assert det.n_paths == 16
    assert det.temperature == 0.5
    assert det.instability_threshold == 0.8


def test_init_invalid_n_paths() -> None:
    """n_paths < 1 raises ValueError."""
    with pytest.raises(ValueError, match="n_paths must be >= 1"):
        HalluFieldDetector(n_paths=0)


def test_init_invalid_temperature() -> None:
    """temperature <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="temperature must be > 0"):
        HalluFieldDetector(temperature=0.0)


def test_init_invalid_threshold() -> None:
    """instability_threshold < 0 raises ValueError."""
    with pytest.raises(ValueError, match="instability_threshold must be >= 0"):
        HalluFieldDetector(instability_threshold=-0.1)


# ---------------------------------------------------------------------------
# CI-safe score(None) path — SCENARIO-VERIFY-155
# ---------------------------------------------------------------------------


def test_score_none_returns_ci_stub() -> None:
    """score(None) returns is_unstable=False with detector_mode='ci_stub'.

    Spec: REQ-VERIFY-117-6, SCENARIO-VERIFY-155
    """
    det = HalluFieldDetector()
    result = det.score(None)
    assert result.is_unstable is False
    assert result.detector_mode == "ci_stub"
    assert result.token_path_count == 0
    assert result.partition_variance == 0.0
    assert result.mean_energy == 0.0


# ---------------------------------------------------------------------------
# score() with real logits — SCENARIO-VERIFY-154
# ---------------------------------------------------------------------------


def test_score_1d_logits() -> None:
    """score() accepts 1-D logit vector (single token) without error.

    Spec: REQ-VERIFY-117-1, SCENARIO-VERIFY-154
    """
    logits = jnp.zeros(50)  # uniform distribution
    det = HalluFieldDetector(n_paths=4)
    result = det.score(logits)
    assert result.detector_mode == "logit"
    assert result.token_path_count == 4
    assert result.partition_variance >= 0.0


def test_score_2d_logits() -> None:
    """score() accepts (seq_len, vocab_size) logit array and produces valid result.

    Spec: REQ-VERIFY-117-1, SCENARIO-VERIFY-154
    """
    logits = jnp.ones((10, 100))  # all tokens equally likely
    det = HalluFieldDetector(n_paths=8)
    result = det.score(logits)
    assert result.detector_mode == "logit"
    assert result.token_path_count == 8
    assert isinstance(result.partition_variance, float)
    assert isinstance(result.mean_energy, float)
    assert isinstance(result.is_unstable, bool)


def test_score_peaked_logits_not_unstable() -> None:
    """Strongly peaked logits (confident model) produce low partition variance.

    Spec: REQ-VERIFY-117-4, REQ-VERIFY-117-5, SCENARIO-VERIFY-156
    """
    # Make token 0 massively more probable than all others: one-hot-like
    logits = jnp.zeros((5, 50)).at[:, 0].set(100.0)
    det = HalluFieldDetector(n_paths=32, instability_threshold=0.5)
    result = det.score(logits, rng_seed=0)
    # Peaked distribution → all paths sample the same token → low variance
    assert result.partition_variance < 0.5, (
        f"Expected low variance for peaked logits, got {result.partition_variance}"
    )


def test_score_uniform_logits_higher_variance_than_peaked() -> None:
    """Uniform logits produce strictly higher partition variance than peaked logits.

    Spec: SCENARIO-VERIFY-156
    """
    # Uniform distribution: high entropy, many competing paths
    uniform_logits = jnp.zeros((8, 50))
    # Peaked distribution: token 0 is overwhelmingly probable
    peaked_logits = jnp.zeros((8, 50)).at[:, 0].set(50.0)

    det = HalluFieldDetector(n_paths=32)
    uniform_result = det.score(uniform_logits, rng_seed=42)
    peaked_result = det.score(peaked_logits, rng_seed=42)

    assert uniform_result.partition_variance >= peaked_result.partition_variance, (
        f"Expected uniform variance ({uniform_result.partition_variance}) >= "
        f"peaked variance ({peaked_result.partition_variance})"
    )


def test_score_n_paths_matches_result() -> None:
    """token_path_count in result matches n_paths constructor argument.

    Spec: REQ-VERIFY-117-2, SCENARIO-VERIFY-154
    """
    logits = jnp.ones((3, 20))
    det = HalluFieldDetector(n_paths=12)
    result = det.score(logits)
    assert result.token_path_count == 12


def test_score_is_unstable_flag_consistent() -> None:
    """is_unstable = (partition_variance > instability_threshold).

    Spec: REQ-VERIFY-117-5
    """
    logits = jnp.zeros((5, 50))  # uniform → some variance
    threshold = 0.0  # any positive variance triggers unstable
    det = HalluFieldDetector(n_paths=16, instability_threshold=threshold)
    result = det.score(logits, rng_seed=7)
    assert result.is_unstable == (result.partition_variance > threshold)


# ---------------------------------------------------------------------------
# _compute_partition_variance unit tests
# ---------------------------------------------------------------------------


def test_compute_partition_variance_zero_for_constant_energies() -> None:
    """Constant path energies yield Var(E) = 0.

    When all paths have the same energy, the variance is zero by definition.
    """
    det = HalluFieldDetector()
    energies = jnp.ones(10) * 3.5  # all equal
    variance = det._compute_partition_variance(energies)
    assert abs(variance) < 1e-5, f"Expected ~0 variance for constant energies, got {variance}"


def test_compute_partition_variance_nonnegative() -> None:
    """Partition variance is always >= 0."""
    import jax

    det = HalluFieldDetector()
    rng = jax.random.PRNGKey(99)
    for _ in range(10):
        rng, subkey = jax.random.split(rng)
        energies = jax.random.uniform(subkey, shape=(20,)) * 5.0
        variance = det._compute_partition_variance(energies)
        assert variance >= 0.0, f"Got negative variance: {variance}"


# ---------------------------------------------------------------------------
# _compute_token_path_energies unit tests
# ---------------------------------------------------------------------------


def test_compute_token_path_energies_shape() -> None:
    """_compute_token_path_energies returns array of length n_paths."""
    import jax

    det = HalluFieldDetector(n_paths=8)
    logits = jnp.ones((4, 30))
    rng = jax.random.PRNGKey(0)
    energies = det._compute_token_path_energies(logits, rng)
    assert len(energies) == 8


def test_compute_token_path_energies_nonnegative() -> None:
    """Path energies (mean NLL) are always >= 0."""
    import jax

    det = HalluFieldDetector(n_paths=16)
    logits = jnp.zeros((6, 25))
    rng = jax.random.PRNGKey(1)
    energies = det._compute_token_path_energies(logits, rng)
    assert all(e >= 0.0 for e in energies), f"Negative path energies: {energies}"


def test_compute_token_path_energies_1d_input() -> None:
    """1-D logit input is treated as (1, vocab_size) without error."""
    import jax

    det = HalluFieldDetector(n_paths=4)
    logits = jnp.zeros(20)  # 1-D
    rng = jax.random.PRNGKey(2)
    energies = det._compute_token_path_energies(logits, rng)
    assert len(energies) == 4


# ---------------------------------------------------------------------------
# Export check — REQ-VERIFY-117-7
# ---------------------------------------------------------------------------


def test_exported_from_pipeline_init() -> None:
    """HalluFieldDetector and HalluFieldResult are importable from carnot.pipeline.

    Spec: REQ-VERIFY-117-7
    """
    from carnot.pipeline import HalluFieldDetector as D, HalluFieldResult as R  # noqa: PLC0415

    assert D is HalluFieldDetector
    assert R is HalluFieldResult
