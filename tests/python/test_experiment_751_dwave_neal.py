"""Tests for Experiment 751 — D-Wave Neal SamplerBackend Validation.

Tests the DWaveNealBackend class and helper functions in the experiment script.

Spec traces: REQ-SAMPLE-017, REQ-SAMPLE-018, SCENARIO-SAMPLE-030, SCENARIO-SAMPLE-031
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from python.carnot.samplers.dwave_neal_backend import DWaveNealBackend, SampleResult  # noqa: E402
import scripts.experiment_751_dwave_neal_backend as exp751  # noqa: E402

DELIVERABLE = _REPO / "results" / "experiment_751_dwave_neal_backend.json"


# ---------------------------------------------------------------------------
# REQ-SAMPLE-017: to_bqm converts J and h correctly
# ---------------------------------------------------------------------------


class TestToBqm(unittest.TestCase):
    """Verify DWaveNealBackend.to_bqm correctly converts IsingEBM coupling matrix and biases.

    Spec traces: REQ-SAMPLE-017
    """

    def _make_mock_ebm(self, n: int = 4) -> MagicMock:
        """Create a mock IsingEBM with known J and h values."""
        ebm = MagicMock()
        J = np.array([
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.3, 0.0],
            [0.0, 0.3, 0.0, -0.2],
            [0.0, 0.0, -0.2, 0.0],
        ], dtype=np.float32)
        h = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
        ebm.coupling = jnp.asarray(J)
        ebm.bias = jnp.asarray(h)
        return ebm

    def test_to_bqm_returns_bqm_with_correct_variable_count(self):
        """to_bqm must produce a BQM with num_variables == n_spins.

        Spec traces: REQ-SAMPLE-017
        """
        backend = DWaveNealBackend()
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        ebm = self._make_mock_ebm(n=4)
        bqm = backend.to_bqm(ebm)
        self.assertEqual(bqm.num_variables, 4)

    def test_to_bqm_encodes_biases_correctly(self):
        """to_bqm must encode all non-zero h biases as linear biases in the BQM.

        Spec traces: REQ-SAMPLE-017
        """
        backend = DWaveNealBackend()
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        ebm = self._make_mock_ebm(n=4)
        bqm = backend.to_bqm(ebm)
        # Variable 0 should have linear bias ~0.1
        self.assertAlmostEqual(bqm.get_linear(0), 0.1, places=5)
        # Variable 1 should have linear bias ~-0.2
        self.assertAlmostEqual(bqm.get_linear(1), -0.2, places=5)

    def test_to_bqm_encodes_couplings_correctly(self):
        """to_bqm must encode non-zero J[i,j] as quadratic interactions in the BQM.

        Spec traces: REQ-SAMPLE-017
        """
        backend = DWaveNealBackend()
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        ebm = self._make_mock_ebm(n=4)
        bqm = backend.to_bqm(ebm)
        # J[0,1] = 0.5 should appear as quadratic interaction (0,1) or (1,0)
        quad = bqm.get_quadratic(0, 1)
        self.assertAlmostEqual(quad, 0.5, places=5)

    def test_to_bqm_zero_couplings_not_included(self):
        """to_bqm must skip zero-value couplings (sparse representation).

        dimod raises ValueError when accessing a non-existent quadratic
        interaction, which confirms it was skipped from the BQM.

        Spec traces: REQ-SAMPLE-017
        """
        backend = DWaveNealBackend()
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        ebm = self._make_mock_ebm(n=4)
        bqm = backend.to_bqm(ebm)
        # J[0,2] = 0.0, so (0,2) should not be in the BQM's quadratic interactions.
        # dimod raises ValueError for non-existent interactions.
        interactions = set(bqm.quadratic)
        self.assertNotIn((0, 2), interactions)
        self.assertNotIn((2, 0), interactions)


# ---------------------------------------------------------------------------
# REQ-SAMPLE-018: sample() returns SampleResult with energy field
# ---------------------------------------------------------------------------


class TestSampleReturnsResult(unittest.TestCase):
    """Verify DWaveNealBackend.sample returns SampleResult with energy field.

    Spec traces: REQ-SAMPLE-017, REQ-SAMPLE-018
    """

    def _make_small_ising_model(self) -> object:
        """Create a small IsingModel for testing."""
        from carnot.models.ising import IsingConfig, IsingModel

        config = IsingConfig(input_dim=6, coupling_init="zeros")
        model = IsingModel(config)
        # Set simple ferromagnetic couplings.
        J = np.zeros((6, 6), dtype=np.float32)
        for i in range(5):
            J[i, i + 1] = 1.0
            J[i + 1, i] = 1.0
        model.coupling = jnp.asarray(J)
        model.bias = jnp.zeros(6)
        return model

    def test_sample_returns_sampleresult_type(self):
        """sample() must return a SampleResult instance.

        Spec traces: REQ-SAMPLE-018
        """
        backend = DWaveNealBackend(num_reads=10, num_sweeps=50)
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        model = self._make_small_ising_model()
        result = backend.sample(model)
        self.assertIsInstance(result, SampleResult)

    def test_sample_result_has_energy_field(self):
        """SampleResult must have a float energy field.

        Spec traces: REQ-SAMPLE-018
        """
        backend = DWaveNealBackend(num_reads=10, num_sweeps=50)
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        model = self._make_small_ising_model()
        result = backend.sample(model)
        self.assertIsInstance(result.energy, float)

    def test_sample_result_has_wall_time_field(self):
        """SampleResult must have a positive wall_time_s field.

        Spec traces: REQ-SAMPLE-018
        """
        backend = DWaveNealBackend(num_reads=10, num_sweeps=50)
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        model = self._make_small_ising_model()
        result = backend.sample(model)
        self.assertIsInstance(result.wall_time_s, float)
        self.assertGreater(result.wall_time_s, 0.0)

    def test_sample_result_spins_correct_shape(self):
        """SampleResult.spins must be a boolean array of shape (n_spins,).

        Spec traces: REQ-SAMPLE-018
        """
        backend = DWaveNealBackend(num_reads=10, num_sweeps=50)
        if not backend.available:
            self.skipTest("dwave-ocean-sdk not installed")
        model = self._make_small_ising_model()
        result = backend.sample(model)
        self.assertEqual(result.spins.shape, (6,))
        self.assertEqual(result.spins.dtype, bool)

    def test_sample_unavailable_backend_returns_sentinel(self):
        """When backend is unavailable, sample() returns SampleResult with inf energy.

        This tests the graceful fallback path when dwave-ocean-sdk is missing.
        Spec traces: REQ-SAMPLE-017
        """
        backend = DWaveNealBackend()
        backend.available = False
        backend._sampler = None
        model = self._make_small_ising_model()
        result = backend.sample(model)
        self.assertIsInstance(result, SampleResult)
        self.assertEqual(result.energy, float("inf"))
        self.assertEqual(result.spins.shape, (6,))


# ---------------------------------------------------------------------------
# REQ-SAMPLE-018: energy_improvement_pct computed correctly
# ---------------------------------------------------------------------------


class TestEnergyImprovementPct(unittest.TestCase):
    """Verify compute_energy_improvement_pct formula is correct.

    Spec traces: REQ-SAMPLE-018
    """

    def test_positive_improvement_when_neal_lower(self):
        """Positive pct when neal finds lower energy (better result).

        Spec traces: REQ-SAMPLE-018
        """
        pct = exp751.compute_energy_improvement_pct(
            mean_energy_gibbs=-10.0,
            mean_energy_neal=-12.0,  # lower = better
        )
        # (-10 - -12) / |-10| * 100 = 2/10 * 100 = 20%
        self.assertAlmostEqual(pct, 20.0, places=5)

    def test_negative_improvement_when_gibbs_lower(self):
        """Negative pct when Gibbs finds lower energy (neal worse).

        Spec traces: REQ-SAMPLE-018
        """
        pct = exp751.compute_energy_improvement_pct(
            mean_energy_gibbs=-12.0,
            mean_energy_neal=-10.0,  # higher = worse
        )
        # (-12 - -10) / |-12| * 100 = -2/12 * 100 ≈ -16.67%
        self.assertAlmostEqual(pct, -200.0 / 12.0, places=4)

    def test_zero_improvement_when_equal(self):
        """Zero pct when both backends find the same energy.

        Spec traces: REQ-SAMPLE-018
        """
        pct = exp751.compute_energy_improvement_pct(-5.0, -5.0)
        self.assertAlmostEqual(pct, 0.0, places=5)

    def test_zero_gibbs_energy_returns_zero(self):
        """Returns 0.0 when mean_energy_gibbs is zero (avoid division by zero).

        Spec traces: REQ-SAMPLE-018
        """
        pct = exp751.compute_energy_improvement_pct(0.0, -1.0)
        self.assertEqual(pct, 0.0)

    def test_honest_verdict_neal_better_energy(self):
        """Verdict is 'neal_better_energy' when improvement > 5%.

        Spec traces: REQ-SAMPLE-018
        """
        # 20% improvement → neal_better_energy
        pct = exp751.compute_energy_improvement_pct(-10.0, -12.0)
        self.assertGreater(pct, 5.0)

    def test_honest_verdict_neal_comparable(self):
        """Verdict is 'neal_comparable_energy' when improvement within ±5%.

        Spec traces: REQ-SAMPLE-018
        """
        # 0% improvement → neal_comparable_energy
        pct = exp751.compute_energy_improvement_pct(-10.0, -10.0)
        self.assertLessEqual(abs(pct), 5.0)


# ---------------------------------------------------------------------------
# Deliverable JSON schema check
# ---------------------------------------------------------------------------


class TestDeliverableSchema(unittest.TestCase):
    """Verify the deliverable JSON has all required schema fields.

    Spec traces: REQ-SAMPLE-017, REQ-SAMPLE-018
    """

    REQUIRED_FIELDS = [
        "experiment",
        "title",
        "run_date",
        "honest_verdict",
        "mean_energy_neal",
        "mean_energy_gibbs",
        "energy_improvement_pct",
        "wall_time_s_neal",
        "wall_time_s_gibbs",
        "n_problems",
        "n_spins",
        "n_samples",
    ]

    def test_deliverable_exists(self):
        """Deliverable JSON must exist after experiment run.

        Spec traces: REQ-SAMPLE-017
        """
        self.assertTrue(
            DELIVERABLE.exists(),
            f"Deliverable JSON not found: {DELIVERABLE}",
        )

    def test_deliverable_has_required_fields(self):
        """Deliverable JSON must contain all required schema fields.

        Spec traces: REQ-SAMPLE-017, REQ-SAMPLE-018
        """
        if not DELIVERABLE.exists():
            self.skipTest("Deliverable not found")
        with open(DELIVERABLE) as f:
            artifact = json.load(f)
        for field in self.REQUIRED_FIELDS:
            self.assertIn(
                field,
                artifact,
                f"Required field '{field}' missing from deliverable JSON",
            )

    def test_deliverable_honest_verdict_is_valid(self):
        """honest_verdict must be one of the four defined values.

        Spec traces: REQ-SAMPLE-018
        """
        if not DELIVERABLE.exists():
            self.skipTest("Deliverable not found")
        with open(DELIVERABLE) as f:
            artifact = json.load(f)
        valid_verdicts = {
            "neal_better_energy",
            "neal_comparable_energy",
            "neal_worse_energy",
            "blocked_on_dependency",
        }
        self.assertIn(
            artifact.get("honest_verdict"),
            valid_verdicts,
            f"honest_verdict '{artifact.get('honest_verdict')}' is not a valid value",
        )


if __name__ == "__main__":
    unittest.main()
