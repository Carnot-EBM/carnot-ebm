"""Tests for Exp 774: Adaptive Bayesian PSV — variance-based early stopping.

Spec traces: REQ-SAMPLE-020, REQ-SAMPLE-021, SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031

Coverage target: 100% of python/carnot/pipeline/adaptive_psv_sampler.py
and the helpers in scripts/experiment_774_adaptive_bayesian_psv.py.
"""

from __future__ import annotations

import pytest

from python.carnot.pipeline.adaptive_psv_sampler import (
    AdaptivePSVSampler,
    AdaptiveSamplerConfig,
    AdaptiveSampleResult,
    _population_variance,
    compute_sample_reduction_fraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constant_energy_fn(value: float):
    """Return an energy function that always returns the same value.

    Used to produce zero-variance distributions (all samples identical) so the
    adaptive sampler stops at K_min.
    """
    def compute_energy(question: str, candidate: str) -> float:  # noqa: ARG001
        return value
    return compute_energy


def _make_alternating_energy_fn(low: float, high: float):
    """Return an energy function that alternates between low and high.

    Produces high variance across samples so the adaptive sampler should NOT
    stop early (variance stays above threshold).
    """
    calls = [0]

    def compute_energy(question: str, candidate: str) -> float:  # noqa: ARG001
        val = low if calls[0] % 2 == 0 else high
        calls[0] += 1
        return val

    return compute_energy


def _make_generate_fn():
    """Return a generate function that returns unique strings."""
    counter = [0]

    def generate(question: str) -> str:  # noqa: ARG001
        counter[0] += 1
        return f"candidate_{counter[0]}"

    return generate


# ---------------------------------------------------------------------------
# _population_variance tests
# ---------------------------------------------------------------------------


def test_population_variance_single_element():
    """Single element: variance is always 0.0 (no spread possible)."""
    assert _population_variance([0.5]) == 0.0


def test_population_variance_identical_elements():
    """Identical elements: variance is 0.0 regardless of value."""
    assert _population_variance([0.3, 0.3, 0.3]) == 0.0


def test_population_variance_two_distinct():
    """Two distinct values: variance is ((d/2)^2) * 2 / 2 = (d/2)^2."""
    # [0.0, 1.0] -> mean=0.5, var = ((0.5)^2 + (0.5)^2) / 2 = 0.25
    assert abs(_population_variance([0.0, 1.0]) - 0.25) < 1e-9


def test_population_variance_empty_list():
    """Empty list: returns 0.0 (degenerate, no elements)."""
    assert _population_variance([]) == 0.0


# ---------------------------------------------------------------------------
# AdaptiveSamplerConfig tests
# ---------------------------------------------------------------------------


def test_config_defaults():
    """Default config values match REQ-SAMPLE-020-1."""
    cfg = AdaptiveSamplerConfig()
    assert cfg.K_min == 2
    assert cfg.K_max == 8
    assert cfg.variance_threshold == 0.05


def test_config_custom():
    """Custom config values are stored correctly."""
    cfg = AdaptiveSamplerConfig(K_min=3, K_max=10, variance_threshold=0.10)
    assert cfg.K_min == 3
    assert cfg.K_max == 10
    assert cfg.variance_threshold == 0.10


# ---------------------------------------------------------------------------
# AdaptivePSVSampler — REQ-SAMPLE-020 / SCENARIO-SAMPLE-030
# ---------------------------------------------------------------------------


def test_sampler_stops_at_k_min_when_variance_below_threshold():
    """Adaptive sampler stops at K_min when energy is constant (variance == 0.0).

    REQ-SAMPLE-020, SCENARIO-SAMPLE-030:
    When all energy scores are identical, population variance = 0.0, which is
    below any positive variance_threshold.  The sampler should stop immediately
    after collecting K_min samples, never proceeding to K_max.
    """
    cfg = AdaptiveSamplerConfig(K_min=2, K_max=8, variance_threshold=0.05)
    sampler = AdaptivePSVSampler(
        generate_fn=_make_generate_fn(),
        compute_energy_fn=_make_constant_energy_fn(0.2),
        config=cfg,
    )
    result = sampler.sample_until_convergent("What is 3 + 4?")

    assert result.k_used == cfg.K_min, (
        f"Expected k_used={cfg.K_min} (stopped at K_min), got {result.k_used}"
    )
    assert result.stopped_early is True
    assert len(result.samples) == cfg.K_min
    assert len(result.energy_scores) == cfg.K_min


def test_sampler_never_stops_before_k_min():
    """Adaptive sampler never stops before K_min samples regardless of variance.

    REQ-SAMPLE-020-2:
    Even with constant zero-variance energy, the sampler must collect at least
    K_min samples before evaluating the stopping criterion.  This test uses
    K_min=3 to confirm the guard works beyond the default K_min=2.
    """
    cfg = AdaptiveSamplerConfig(K_min=3, K_max=8, variance_threshold=0.99)
    sampler = AdaptivePSVSampler(
        generate_fn=_make_generate_fn(),
        compute_energy_fn=_make_constant_energy_fn(0.2),
        config=cfg,
    )
    result = sampler.sample_until_convergent("What is 5 * 6?")

    # Variance == 0.0 < 0.99 threshold, but K_min=3 so must collect at least 3.
    assert result.k_used >= cfg.K_min, (
        f"k_used={result.k_used} is below K_min={cfg.K_min}"
    )


def test_sampler_uses_k_max_when_variance_stays_high():
    """Adaptive sampler uses all K_max samples when variance stays above threshold.

    REQ-SAMPLE-020, SCENARIO-SAMPLE-031:
    When energy alternates between 0.0 and 1.0, population variance stays near
    0.25 — well above any reasonable threshold.  The sampler must reach K_max.
    """
    cfg = AdaptiveSamplerConfig(K_min=2, K_max=8, variance_threshold=0.05)
    sampler = AdaptivePSVSampler(
        generate_fn=_make_generate_fn(),
        compute_energy_fn=_make_alternating_energy_fn(0.0, 1.0),
        config=cfg,
    )
    result = sampler.sample_until_convergent("What is 7 - 3?")

    assert result.k_used == cfg.K_max, (
        f"Expected k_used={cfg.K_max} (no early stop), got {result.k_used}"
    )
    assert result.stopped_early is False
    assert len(result.samples) == cfg.K_max


def test_sampler_result_best_sample_is_minimum_energy():
    """best_sample is the candidate with the lowest energy score.

    When energies are [0.8, 0.3, 0.9, ...], the best_sample corresponds to
    the second candidate generated (energy=0.3).
    """
    energies = [0.8, 0.3, 0.9, 0.1]
    energy_iter = iter(energies)

    def fixed_energy(question: str, candidate: str) -> float:  # noqa: ARG001
        return next(energy_iter)

    candidates = ["c0", "c1", "c2", "c3"]
    cand_iter = iter(candidates)

    def fixed_generate(question: str) -> str:  # noqa: ARG001
        return next(cand_iter)

    # Use high variance_threshold so sampler runs to K_max.
    cfg = AdaptiveSamplerConfig(K_min=2, K_max=4, variance_threshold=0.001)
    sampler = AdaptivePSVSampler(
        generate_fn=fixed_generate,
        compute_energy_fn=fixed_energy,
        config=cfg,
    )
    result = sampler.sample_until_convergent("question")

    assert result.best_energy == 0.1
    assert result.best_sample == "c3"


def test_sampler_default_config_used_when_none_passed():
    """Sampler uses AdaptiveSamplerConfig() defaults when config=None."""
    sampler = AdaptivePSVSampler(
        generate_fn=_make_generate_fn(),
        compute_energy_fn=_make_constant_energy_fn(0.5),
        config=None,
    )
    assert sampler.config.K_min == 2
    assert sampler.config.K_max == 8
    assert sampler.config.variance_threshold == 0.05


# ---------------------------------------------------------------------------
# compute_sample_reduction_fraction — REQ-SAMPLE-021
# ---------------------------------------------------------------------------


def test_sample_reduction_fraction_all_k_max():
    """When all questions use K_max samples, reduction is 0.0.

    REQ-SAMPLE-021-3.
    """
    k_used = [8, 8, 8, 8]
    result = compute_sample_reduction_fraction(k_used, K_max=8)
    assert result == 0.0


def test_sample_reduction_fraction_all_k_min():
    """When all questions stop at K_min=2 with K_max=8, reduction is 1 - 2/8 = 0.75.

    REQ-SAMPLE-021-4.
    """
    k_used = [2, 2, 2, 2]
    result = compute_sample_reduction_fraction(k_used, K_max=8)
    assert abs(result - 0.75) < 1e-9


def test_sample_reduction_fraction_mixed():
    """Mean of [2, 4, 6, 8] = 5.0; reduction = 1 - 5/8 = 0.375.

    REQ-SAMPLE-021-1.
    """
    k_used = [2, 4, 6, 8]
    result = compute_sample_reduction_fraction(k_used, K_max=8)
    assert abs(result - 0.375) < 1e-9


def test_sample_reduction_fraction_empty_list():
    """Empty k_used list returns 0.0 (no questions processed)."""
    result = compute_sample_reduction_fraction([], K_max=8)
    assert result == 0.0


def test_sample_reduction_fraction_zero_k_max():
    """K_max=0 returns 0.0 (degenerate guard)."""
    result = compute_sample_reduction_fraction([2, 4], K_max=0)
    assert result == 0.0


# ---------------------------------------------------------------------------
# AdaptiveSampleResult dataclass
# ---------------------------------------------------------------------------


def test_adaptive_sample_result_fields():
    """AdaptiveSampleResult stores all required fields correctly."""
    result = AdaptiveSampleResult(
        samples=["a", "b"],
        energy_scores=[0.3, 0.2],
        k_used=2,
        stopped_early=True,
        best_sample="b",
        best_energy=0.2,
    )
    assert result.k_used == 2
    assert result.stopped_early is True
    assert result.best_energy == 0.2
    assert result.best_sample == "b"


# ---------------------------------------------------------------------------
# Integration: full experiment logic helpers
# ---------------------------------------------------------------------------


def test_experiment_helpers_produce_consistent_questions():
    """_make_questions produces N questions with alternating labels."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.experiment_774_adaptive_bayesian_psv import _make_questions

    qs = _make_questions(seed=42, n=10)
    assert len(qs) == 10
    # Labels alternate True/False.
    for i, q in enumerate(qs):
        assert q["label"] == (i % 2 == 0)


def test_experiment_auroc_chance_level():
    """AUROC returns 0.5 when all scores are equal (chance level)."""
    from scripts.experiment_774_adaptive_bayesian_psv import _auroc

    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [True, False, True, False]
    result = _auroc(scores, labels)
    assert result == 0.5


def test_experiment_auroc_perfect_separation():
    """AUROC returns 1.0 when correct scores are all lower than incorrect scores."""
    from scripts.experiment_774_adaptive_bayesian_psv import _auroc

    scores = [0.1, 0.9, 0.2, 0.8]
    labels = [True, False, True, False]
    result = _auroc(scores, labels)
    assert result == 1.0


def test_experiment_auroc_empty_class_returns_chance():
    """AUROC returns 0.5 when one class is missing (degenerate)."""
    from scripts.experiment_774_adaptive_bayesian_psv import _auroc

    # All positive, no negatives.
    result = _auroc([0.1, 0.2], [True, True])
    assert result == 0.5


def test_run_fixed_k_returns_exactly_k_samples_per_question():
    """Fixed-K baseline draws exactly FIXED_K samples for each question."""
    from scripts.experiment_774_adaptive_bayesian_psv import (
        _make_questions,
        _make_generate_fn,
        _run_fixed_k,
    )

    call_count = [0]

    def counting_energy(q: str, c: str) -> float:  # noqa: ARG001
        call_count[0] += 1
        return 0.3

    questions = _make_questions(seed=99, n=5)
    _run_fixed_k(questions, _make_generate_fn(), counting_energy, k=4, seed=99)
    # 5 questions * 4 samples = 20 energy evaluations.
    assert call_count[0] == 20


def test_run_adaptive_reduction_goal():
    """Adaptive run achieves >= 30% sample reduction on low-variance energy function.

    REQ-SAMPLE-021: sample_reduction_fraction must be >= 0.30 to claim efficiency gain.
    With constant energy (variance=0.0), adaptive sampler stops at K_min=2 for ALL
    questions.  reduction = 1 - 2/8 = 0.75, well above the 0.30 target.
    """
    from scripts.experiment_774_adaptive_bayesian_psv import (
        _make_questions,
        _make_generate_fn,
        _run_adaptive,
    )

    questions = _make_questions(seed=42, n=10)
    config = AdaptiveSamplerConfig(K_min=2, K_max=8, variance_threshold=0.05)
    result = _run_adaptive(
        questions,
        _make_generate_fn(),
        _make_constant_energy_fn(0.2),
        config,
    )
    assert result["sample_reduction_fraction"] >= 0.30, (
        f"Expected >= 0.30, got {result['sample_reduction_fraction']}"
    )
