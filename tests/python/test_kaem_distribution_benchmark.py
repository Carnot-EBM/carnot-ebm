"""Tests for KAEMDistributionBenchmark and DistributionFamilyResult.

Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, REQ-SAMPLE-024,
      SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036, SCENARIO-SAMPLE-037
"""

from __future__ import annotations

import pytest

from carnot.models.kaem_distribution_benchmark import (
    DistributionFamilyResult,
    KAEMDistributionBenchmark,
)


# ---------------------------------------------------------------------------
# DistributionFamilyResult tests
# ---------------------------------------------------------------------------


class TestDistributionFamilyResult:
    """Tests for DistributionFamilyResult dataclass.

    Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036
    """

    def test_kaem_advantage_positive_when_kaem_wins(self):
        """SCENARIO-SAMPLE-035: kaem_advantage > 0 when kaem_mean_l2 < mcmc_mean_l2."""
        r = DistributionFamilyResult(
            family_name="gaussian_mixture",
            kaem_mean_l2=0.1,
            mcmc_mean_l2=0.5,
        )
        assert abs(r.kaem_advantage - 0.4) < 1e-6

    def test_kaem_wins_true_when_kaem_advantage_positive(self):
        """SCENARIO-SAMPLE-035: kaem_wins=True when kaem_advantage > 0."""
        r = DistributionFamilyResult(
            family_name="gaussian_mixture",
            kaem_mean_l2=0.1,
            mcmc_mean_l2=0.5,
        )
        assert r.kaem_wins is True

    def test_kaem_advantage_negative_when_mcmc_wins(self):
        """SCENARIO-SAMPLE-036: kaem_advantage < 0 when mcmc_mean_l2 < kaem_mean_l2."""
        r = DistributionFamilyResult(
            family_name="student_t",
            kaem_mean_l2=0.6,
            mcmc_mean_l2=0.3,
        )
        assert abs(r.kaem_advantage - (-0.3)) < 1e-6

    def test_kaem_wins_false_when_mcmc_wins(self):
        """SCENARIO-SAMPLE-036: kaem_wins=False when kaem_advantage <= 0."""
        r = DistributionFamilyResult(
            family_name="student_t",
            kaem_mean_l2=0.6,
            mcmc_mean_l2=0.3,
        )
        assert r.kaem_wins is False

    def test_kaem_wins_false_when_tied(self):
        """kaem_wins=False when kaem_advantage == 0 (tie goes to MCMC)."""
        r = DistributionFamilyResult(
            family_name="piecewise_uniform",
            kaem_mean_l2=0.5,
            mcmc_mean_l2=0.5,
        )
        assert r.kaem_wins is False

    def test_to_dict_contains_all_fields(self):
        """to_dict() returns dict with all required fields."""
        r = DistributionFamilyResult(
            family_name="gaussian_mixture",
            kaem_mean_l2=0.1,
            mcmc_mean_l2=0.5,
        )
        d = r.to_dict()
        assert d["family_name"] == "gaussian_mixture"
        assert abs(d["kaem_mean_l2"] - 0.1) < 1e-6
        assert abs(d["mcmc_mean_l2"] - 0.5) < 1e-6
        assert abs(d["kaem_advantage"] - 0.4) < 1e-6
        assert d["kaem_wins"] is True

    def test_to_dict_values_are_json_serializable(self):
        """to_dict() values are plain Python types, not numpy scalars."""
        r = DistributionFamilyResult(
            family_name="student_t",
            kaem_mean_l2=0.3,
            mcmc_mean_l2=0.7,
        )
        d = r.to_dict()
        assert isinstance(d["family_name"], str)
        assert isinstance(d["kaem_mean_l2"], float)
        assert isinstance(d["mcmc_mean_l2"], float)
        assert isinstance(d["kaem_advantage"], float)
        assert isinstance(d["kaem_wins"], bool)


# ---------------------------------------------------------------------------
# KAEMDistributionBenchmark constructor tests
# ---------------------------------------------------------------------------


class TestKAEMDistributionBenchmarkConstructor:
    """Tests for KAEMDistributionBenchmark constructor validation.

    Spec: REQ-SAMPLE-022
    """

    def test_default_params(self):
        """Default n_vars=10, n_samples=200."""
        b = KAEMDistributionBenchmark()
        assert b.n_vars == 10
        assert b.n_samples == 200

    def test_custom_params(self):
        """Custom n_vars and n_samples are stored correctly."""
        b = KAEMDistributionBenchmark(n_vars=5, n_samples=50)
        assert b.n_vars == 5
        assert b.n_samples == 50

    def test_n_vars_zero_raises(self):
        """n_vars=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_vars"):
            KAEMDistributionBenchmark(n_vars=0)

    def test_n_samples_one_raises(self):
        """n_samples=1 raises ValueError (need >= 2 for CDF comparison)."""
        with pytest.raises(ValueError, match="n_samples"):
            KAEMDistributionBenchmark(n_samples=1)


# ---------------------------------------------------------------------------
# KAEMDistributionBenchmark benchmark method tests
# ---------------------------------------------------------------------------


class TestKAEMDistributionBenchmarkMethods:
    """Tests for benchmark methods and best_family.

    Uses n_vars=2, n_samples=20 for speed in CI.

    Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, REQ-SAMPLE-024, SCENARIO-SAMPLE-037
    """

    @pytest.fixture
    def bench(self):
        return KAEMDistributionBenchmark(n_vars=2, n_samples=20)

    def test_benchmark_gaussian_mixture_returns_correct_type(self, bench):
        """benchmark_gaussian_mixture returns DistributionFamilyResult."""
        r = bench.benchmark_gaussian_mixture()
        assert isinstance(r, DistributionFamilyResult)

    def test_benchmark_gaussian_mixture_family_name(self, bench):
        """benchmark_gaussian_mixture sets family_name='gaussian_mixture'."""
        r = bench.benchmark_gaussian_mixture()
        assert r.family_name == "gaussian_mixture"

    def test_benchmark_gaussian_mixture_l2_positive(self, bench):
        """benchmark_gaussian_mixture produces non-negative l2 values."""
        r = bench.benchmark_gaussian_mixture()
        assert r.kaem_mean_l2 >= 0.0
        assert r.mcmc_mean_l2 >= 0.0

    def test_benchmark_student_t_returns_correct_type(self, bench):
        """benchmark_student_t returns DistributionFamilyResult."""
        r = bench.benchmark_student_t(nu=2.0)
        assert isinstance(r, DistributionFamilyResult)

    def test_benchmark_student_t_family_name(self, bench):
        """benchmark_student_t sets family_name='student_t'."""
        r = bench.benchmark_student_t()
        assert r.family_name == "student_t"

    def test_benchmark_student_t_l2_positive(self, bench):
        """benchmark_student_t produces non-negative l2 values."""
        r = bench.benchmark_student_t()
        assert r.kaem_mean_l2 >= 0.0
        assert r.mcmc_mean_l2 >= 0.0

    def test_benchmark_piecewise_uniform_returns_correct_type(self, bench):
        """benchmark_piecewise_uniform returns DistributionFamilyResult."""
        r = bench.benchmark_piecewise_uniform(n_pieces=3)
        assert isinstance(r, DistributionFamilyResult)

    def test_benchmark_piecewise_uniform_family_name(self, bench):
        """benchmark_piecewise_uniform sets family_name='piecewise_uniform'."""
        r = bench.benchmark_piecewise_uniform()
        assert r.family_name == "piecewise_uniform"

    def test_benchmark_piecewise_uniform_l2_positive(self, bench):
        """benchmark_piecewise_uniform produces non-negative l2 values."""
        r = bench.benchmark_piecewise_uniform()
        assert r.kaem_mean_l2 >= 0.0
        assert r.mcmc_mean_l2 >= 0.0


# ---------------------------------------------------------------------------
# best_family tests
# ---------------------------------------------------------------------------


class TestBestFamily:
    """Tests for KAEMDistributionBenchmark.best_family().

    Spec: REQ-SAMPLE-024, SCENARIO-SAMPLE-037
    """

    def test_best_family_returns_winner_when_one_wins(self):
        """best_family returns the name of the only winning family."""
        results = [
            DistributionFamilyResult("gaussian_mixture", kaem_mean_l2=0.1, mcmc_mean_l2=0.5),
            DistributionFamilyResult("student_t", kaem_mean_l2=0.6, mcmc_mean_l2=0.3),
            DistributionFamilyResult("piecewise_uniform", kaem_mean_l2=0.7, mcmc_mean_l2=0.4),
        ]
        bench = KAEMDistributionBenchmark(n_vars=2, n_samples=20)
        assert bench.best_family(results=results) == "gaussian_mixture"

    def test_best_family_returns_largest_advantage(self):
        """best_family returns family with largest kaem_advantage, not first win."""
        results = [
            DistributionFamilyResult("gaussian_mixture", kaem_mean_l2=0.3, mcmc_mean_l2=0.5),  # advantage=0.2
            DistributionFamilyResult("student_t", kaem_mean_l2=0.1, mcmc_mean_l2=0.8),          # advantage=0.7
            DistributionFamilyResult("piecewise_uniform", kaem_mean_l2=0.6, mcmc_mean_l2=0.4),  # advantage=-0.2
        ]
        bench = KAEMDistributionBenchmark(n_vars=2, n_samples=20)
        assert bench.best_family(results=results) == "student_t"

    def test_best_family_returns_none_when_all_mcmc_wins(self):
        """SCENARIO-SAMPLE-037: best_family returns 'none' when all kaem_wins=False.

        Spec: REQ-SAMPLE-024
        """
        results = [
            DistributionFamilyResult("gaussian_mixture", kaem_mean_l2=0.6, mcmc_mean_l2=0.3),
            DistributionFamilyResult("student_t", kaem_mean_l2=0.7, mcmc_mean_l2=0.4),
            DistributionFamilyResult("piecewise_uniform", kaem_mean_l2=0.8, mcmc_mean_l2=0.5),
        ]
        bench = KAEMDistributionBenchmark(n_vars=2, n_samples=20)
        assert bench.best_family(results=results) == "none"

    def test_best_family_runs_all_benchmarks_when_no_results(self):
        """best_family(results=None) runs benchmarks internally and returns str."""
        bench = KAEMDistributionBenchmark(n_vars=2, n_samples=10)
        result = bench.best_family()
        assert isinstance(result, str)
        # Must be one of the valid family names or 'none'
        valid = {"gaussian_mixture", "student_t", "piecewise_uniform", "none"}
        assert result in valid
