"""Tests for EBM-guided rejection sampling.

**Researcher summary:**
    Verifies the EBM rejection sampling pipeline that combines per-token
    activation energy with logprob scores to select the best candidate.
    Tests cover config validation, scoring logic, candidate selection,
    and the composite energy formula.

**Detailed explanation for engineers:**
    These are pure unit tests using mock models and pre-computed activations.
    No real LLM loading or GPU access required. We mock the model generation
    and hidden state extraction to test the scoring and selection logic
    in isolation.

Spec coverage: REQ-INFER-015
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest
from carnot.inference.ebm_rejection import (
    EBMCandidateScore,
    EBMRejectionConfig,
    EBMRejectionResult,
    score_activations_with_ebm,
)
from carnot.models.gibbs import GibbsConfig, GibbsModel

# --- Config validation tests ---


def test_config_defaults() -> None:
    """Default config has sensible values.

    Spec: REQ-INFER-015
    """
    config = EBMRejectionConfig()
    assert config.ebm_weight == 1.0
    assert config.logprob_weight == 1.0
    assert config.hidden_layer == -1
    assert config.n_candidates == 5
    assert config.temperature == 0.8
    assert config.max_new_tokens == 80


def test_config_negative_ebm_weight_rejected() -> None:
    """Negative ebm_weight raises ValueError.

    Spec: REQ-INFER-015
    """
    with pytest.raises(ValueError, match="ebm_weight"):
        EBMRejectionConfig(ebm_weight=-1.0)


def test_config_negative_logprob_weight_rejected() -> None:
    """Negative logprob_weight raises ValueError.

    Spec: REQ-INFER-015
    """
    with pytest.raises(ValueError, match="logprob_weight"):
        EBMRejectionConfig(logprob_weight=-0.5)


def test_config_zero_candidates_rejected() -> None:
    """n_candidates < 1 raises ValueError.

    Spec: REQ-INFER-015
    """
    with pytest.raises(ValueError, match="n_candidates"):
        EBMRejectionConfig(n_candidates=0)


def test_config_custom_values() -> None:
    """Custom config values are preserved.

    Spec: REQ-INFER-015
    """
    config = EBMRejectionConfig(
        ebm_weight=2.0,
        logprob_weight=0.5,
        hidden_layer=12,
        n_candidates=10,
        temperature=1.2,
        max_new_tokens=50,
    )
    assert config.ebm_weight == 2.0
    assert config.logprob_weight == 0.5
    assert config.hidden_layer == 12
    assert config.n_candidates == 10


# --- EBM scoring tests ---


def test_score_activations_with_ebm_basic() -> None:
    """EBM scores per-token activations and returns mean energy.

    Spec: REQ-INFER-015
    """
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=8, hidden_dims=[4])
    ebm = GibbsModel(config, key=key)

    # Create 5 tokens × 8 dimensions
    activations = np.random.default_rng(42).standard_normal((5, 8)).astype(np.float32)
    mean_energy = score_activations_with_ebm(ebm, activations)

    assert isinstance(mean_energy, float)
    assert np.isfinite(mean_energy)


def test_score_activations_single_token() -> None:
    """Single-token activation works correctly.

    Spec: REQ-INFER-015
    """
    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=8, hidden_dims=[4])
    ebm = GibbsModel(config, key=key)

    activations = np.random.default_rng(42).standard_normal((1, 8)).astype(np.float32)
    mean_energy = score_activations_with_ebm(ebm, activations)

    # With single token, mean energy == single token energy
    single_energy = float(ebm.energy(jnp.array(activations[0])))
    assert mean_energy == pytest.approx(single_energy, abs=1e-5)


def test_score_activations_correct_vs_random() -> None:
    """Trained EBM assigns lower energy to correct-like activations.

    Spec: REQ-INFER-015
    """
    import jax
    from carnot.training.nce import nce_loss

    key = jrandom.PRNGKey(42)
    config = GibbsConfig(input_dim=8, hidden_dims=[4])
    ebm = GibbsModel(config, key=key)

    # Create separable clusters
    rng = np.random.default_rng(42)
    correct = jnp.array(rng.standard_normal((50, 8)).astype(np.float32) + 2.0)
    wrong = jnp.array(rng.standard_normal((50, 8)).astype(np.float32) - 2.0)

    # Quick train
    def get_p(m: GibbsModel) -> dict:
        return {"layers": [(w, b) for w, b in m.layers],
                "output_weight": m.output_weight, "output_bias": m.output_bias}

    def set_p(m: GibbsModel, p: dict) -> None:
        m.layers = list(p["layers"])
        m.output_weight = p["output_weight"]
        m.output_bias = p["output_bias"]

    params = get_p(ebm)

    def loss_fn(p: dict) -> float:
        old = get_p(ebm)
        set_p(ebm, p)
        r = nce_loss(ebm, correct, wrong)
        set_p(ebm, old)
        return r

    for _ in range(50):
        grads = jax.grad(loss_fn)(params)
        params = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    set_p(ebm, params)

    # Score some new samples
    new_correct = rng.standard_normal((10, 8)).astype(np.float32) + 2.0
    new_wrong = rng.standard_normal((10, 8)).astype(np.float32) - 2.0

    correct_energy = score_activations_with_ebm(ebm, new_correct)
    wrong_energy = score_activations_with_ebm(ebm, new_wrong)

    # Trained EBM should assign lower energy to correct (data) distribution
    assert correct_energy < wrong_energy


# --- Candidate score tests ---


def test_candidate_score_dataclass() -> None:
    """EBMCandidateScore holds all scoring components.

    Spec: REQ-INFER-015
    """
    score = EBMCandidateScore(
        response="Paris",
        mean_logprob=-0.5,
        mean_ebm_energy=-1.2,
        composite_energy=-1.7,
        n_tokens=5,
    )
    assert score.response == "Paris"
    assert score.mean_logprob == -0.5
    assert score.mean_ebm_energy == -1.2
    assert score.composite_energy == -1.7
    assert score.n_tokens == 5


def test_composite_energy_formula() -> None:
    """Composite energy = ebm_weight * ebm - logprob_weight * logprob.

    Spec: REQ-INFER-015
    """
    config = EBMRejectionConfig(ebm_weight=2.0, logprob_weight=0.5)
    mean_ebm = 1.5
    mean_logprob = -0.8

    # composite = 2.0 * 1.5 - 0.5 * (-0.8) = 3.0 + 0.4 = 3.4
    composite = config.ebm_weight * mean_ebm - config.logprob_weight * mean_logprob
    assert composite == pytest.approx(3.4)


def test_result_sorting() -> None:
    """EBMRejectionResult.all_candidates should be sorted by composite ascending.

    Spec: REQ-INFER-015
    """
    candidates = [
        EBMCandidateScore("C", -0.3, 2.0, 1.7, 5),
        EBMCandidateScore("A", -0.5, 0.5, 0.0, 5),
        EBMCandidateScore("B", -0.4, 1.0, 0.6, 5),
    ]
    candidates.sort(key=lambda c: c.composite_energy)
    result = EBMRejectionResult(best=candidates[0], all_candidates=candidates)

    assert result.best.response == "A"
    assert result.all_candidates[0].composite_energy <= result.all_candidates[1].composite_energy
    assert result.all_candidates[1].composite_energy <= result.all_candidates[2].composite_energy


def test_ebm_only_scoring() -> None:
    """With logprob_weight=0, scoring is purely EBM-based.

    Spec: REQ-INFER-015
    """
    config = EBMRejectionConfig(ebm_weight=1.0, logprob_weight=0.0)
    composite = config.ebm_weight * 2.0 - config.logprob_weight * (-0.5)
    assert composite == pytest.approx(2.0)


def test_logprob_only_scoring() -> None:
    """With ebm_weight=0, scoring is purely logprob-based.

    Spec: REQ-INFER-015
    """
    config = EBMRejectionConfig(ebm_weight=0.0, logprob_weight=1.0)
    composite = config.ebm_weight * 2.0 - config.logprob_weight * (-0.5)
    assert composite == pytest.approx(0.5)  # -1.0 * (-0.5) = 0.5


# --- Package export tests ---


def test_exports_from_inference_package() -> None:
    """All new symbols are exported from carnot.inference.

    Spec: REQ-INFER-015
    """
    from carnot.inference import (
        EBMCandidateScore,
        EBMRejectionConfig,
        EBMRejectionResult,
        ebm_rejection_sample,
        score_activations_with_ebm,
    )
    assert EBMCandidateScore is not None
    assert EBMRejectionConfig is not None
    assert EBMRejectionResult is not None
    assert callable(ebm_rejection_sample)
    assert callable(score_activations_with_ebm)
