"""Tests for Experiment 826 — PRM Cross-Domain Degradation Benchmark.

Spec: REQ-VERIFY-145, SCENARIO-VERIFY-174
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import experiment_826_prm_cross_domain_benchmark as exp826  # noqa: E402


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_CERTS_ALL_UNSAT_HIGH_ENERGY = [
    {
        "step_id": f"humaneval_step_{i}",
        "jepa_energy_delta": 0.6 + i * 0.01,
        "constraint_type": "code_logic",
        "z3_verdict": "unsat",
        "confidence_score": 0.7,
    }
    for i in range(10)
] + [
    {
        "step_id": f"arc_step_{i}",
        "jepa_energy_delta": 0.5 + i * 0.01,
        "constraint_type": "planning",
        "z3_verdict": "unsat",
        "confidence_score": 0.65,
    }
    for i in range(10)
]

_CERTS_MIXED = [
    # 10 corroborate (unsat + high energy)
    {
        "step_id": f"humaneval_step_{i}",
        "jepa_energy_delta": 0.6,
        "constraint_type": "code_logic",
        "z3_verdict": "unsat",
        "confidence_score": 0.7,
    }
    for i in range(10)
] + [
    # 10 do NOT corroborate (unsat but low/negative energy)
    {
        "step_id": f"arc_step_{i}",
        "jepa_energy_delta": -0.1,
        "constraint_type": "planning",
        "z3_verdict": "unsat",
        "confidence_score": 0.65,
    }
    for i in range(10)
]


# ---------------------------------------------------------------------------
# REQ-VERIFY-145: compute_degradation
# ---------------------------------------------------------------------------


class TestComputeDegradation(unittest.TestCase):
    """REQ-VERIFY-145: cross_domain_degradation = in_dist_auc - ood_auc."""

    def test_degradation_positive_when_ood_lower(self):
        """SCENARIO-VERIFY-174: degradation is positive when OOD AUC < in-dist AUC."""
        # in_dist=0.87, ood=0.76 → degradation=0.11
        result = exp826.compute_degradation(0.87, 0.76)
        self.assertAlmostEqual(result, 0.11, places=6)

    def test_degradation_zero_when_equal(self):
        """Degradation is exactly 0 when in-dist and OOD AUCs are identical."""
        result = exp826.compute_degradation(0.80, 0.80)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_degradation_negative_when_ood_higher(self):
        """Degradation can be negative if model performs BETTER on OOD domain."""
        result = exp826.compute_degradation(0.70, 0.80)
        self.assertAlmostEqual(result, -0.10, places=6)

    def test_degradation_uses_real_exp825_values(self):
        """Sanity check with actual Exp 824/825 values: in_dist=0.8705, humaneval=0.76."""
        deg = exp826.compute_degradation(0.8705357142857143, 0.76)
        self.assertGreater(deg, 0.10)
        self.assertLess(deg, 0.12)


# ---------------------------------------------------------------------------
# REQ-VERIFY-145: determine_honest_verdict — threshold logic
# ---------------------------------------------------------------------------


class TestDetermineHonestVerdict(unittest.TestCase):
    """REQ-VERIFY-145: honest_verdict boundary conditions at 0.08 / 0.09."""

    def test_data_unavailable_overrides_all(self):
        """SCENARIO-VERIFY-174: data_unavailable flag short-circuits numeric logic."""
        verdict = exp826.determine_honest_verdict(0.0, data_unavailable=True)
        self.assertEqual(verdict, "data_unavailable")

    def test_beats_baseline_exactly_at_0_08(self):
        """degradation_max == 0.08 → above_baseline (boundary is inclusive)."""
        verdict = exp826.determine_honest_verdict(0.08)
        self.assertEqual(verdict, "above_baseline")

    def test_beats_baseline_well_below(self):
        """degradation_max = 0.05 → above_baseline."""
        verdict = exp826.determine_honest_verdict(0.05)
        self.assertEqual(verdict, "above_baseline")

    def test_at_baseline_just_above_0_08(self):
        """degradation_max = 0.085 → at_baseline (within ±0.01 tolerance band)."""
        verdict = exp826.determine_honest_verdict(0.085)
        self.assertEqual(verdict, "at_baseline")

    def test_at_baseline_exactly_at_0_09(self):
        """degradation_max = 0.09 → at_baseline (abs(0.09-0.08)==0.01, within band)."""
        verdict = exp826.determine_honest_verdict(0.09)
        self.assertEqual(verdict, "at_baseline")

    def test_below_baseline_above_0_09(self):
        """degradation_max = 0.10 → below_baseline."""
        verdict = exp826.determine_honest_verdict(0.10)
        self.assertEqual(verdict, "below_baseline")

    def test_below_baseline_large_degradation(self):
        """degradation_max = 0.50 → below_baseline."""
        verdict = exp826.determine_honest_verdict(0.50)
        self.assertEqual(verdict, "below_baseline")


# ---------------------------------------------------------------------------
# REQ-VERIFY-145: compute_corroboration_rate
# ---------------------------------------------------------------------------


class TestComputeCorroborationRate(unittest.TestCase):
    """REQ-VERIFY-145: corroboration_rate measures JEPA ↔ Z3 agreement."""

    def test_all_corroborate(self):
        """All unsat + positive energy → corroboration_rate = 1.0."""
        rate = exp826.compute_corroboration_rate(_CERTS_ALL_UNSAT_HIGH_ENERGY)
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_half_corroborate(self):
        """10/20 corroborate → corroboration_rate = 0.5."""
        rate = exp826.compute_corroboration_rate(_CERTS_MIXED)
        self.assertAlmostEqual(rate, 0.5, places=6)

    def test_empty_certificates(self):
        """Empty certificate list → corroboration_rate = 0.0 (not division-by-zero)."""
        rate = exp826.compute_corroboration_rate([])
        self.assertEqual(rate, 0.0)

    def test_sat_low_energy_corroborates(self):
        """sat + negative/zero energy also counts as corroboration."""
        certs = [
            {
                "step_id": "s0",
                "jepa_energy_delta": -0.2,
                "z3_verdict": "sat",
                "constraint_type": "arithmetic",
                "confidence_score": 0.8,
            },
            {
                "step_id": "s1",
                "jepa_energy_delta": 0.5,
                "z3_verdict": "unsat",
                "constraint_type": "arithmetic",
                "confidence_score": 0.7,
            },
        ]
        rate = exp826.compute_corroboration_rate(certs)
        self.assertAlmostEqual(rate, 1.0, places=6)

    def test_sat_high_energy_does_not_corroborate(self):
        """sat + positive energy is a contradiction → does NOT corroborate."""
        certs = [
            {
                "step_id": "s0",
                "jepa_energy_delta": 0.8,
                "z3_verdict": "sat",
                "constraint_type": "arithmetic",
                "confidence_score": 0.8,
            },
        ]
        rate = exp826.compute_corroboration_rate(certs)
        self.assertAlmostEqual(rate, 0.0, places=6)


# ---------------------------------------------------------------------------
# Integration: full main() flow via mocked filesystem
# ---------------------------------------------------------------------------


class TestMainFlow(unittest.TestCase):
    """SCENARIO-VERIFY-174: main() reads Exp 824/825 and writes correct artifact."""

    def _make_exp825(self, **overrides):
        base = {
            "experiment": 825,
            "status": "success",
            "auc_gsm8k": 0.36,
            "auc_humaneval": 0.76,
            "auc_arc": 0.04,
            "overall_ood_auc": 0.4,
            "verification_certificates": _CERTS_ALL_UNSAT_HIGH_ENERGY,
        }
        base.update(overrides)
        return base

    def _make_exp824(self, **overrides):
        base = {
            "experiment": 824,
            "status": "success",
            "in_dist_auc": 0.8705357142857143,
        }
        base.update(overrides)
        return base

    def _run_main_with_mocks(self, exp824_data, exp825_data):
        """Run main() with temp-directory fixture files and a captured FakeTemplate.

        We write real JSON files to a temp dir and point _REPO_ROOT there so that
        the experiment script's open() calls succeed without mocking builtins.open
        (which would intercept writes too).
        """
        import tempfile

        written = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            results_dir = tmppath / "results"
            results_dir.mkdir()
            (results_dir / "experiment_824_jepa_v23_limo_corpus.json").write_text(
                json.dumps(exp824_data)
            )
            (results_dir / "experiment_825_jepa_v23_eval_fr11_tier3.json").write_text(
                json.dumps(exp825_data)
            )

            class FakeTemplate:
                def __init__(self, *a, **kw):
                    pass

                def setup(self):
                    pass

                def build_result(self, payload, **kwargs):
                    merged = {**payload, **kwargs}
                    written.update(merged)
                    return merged

                def assert_deliverable_written(self):
                    pass

            with (
                patch("experiment_826_prm_cross_domain_benchmark.ExperimentTemplate", FakeTemplate),
                patch(
                    "experiment_826_prm_cross_domain_benchmark.ExperimentTimeoutWatchdog",
                    MagicMock(),
                ),
                patch.object(exp826, "_REPO_ROOT", tmppath),
            ):
                exp826.main()

        return written

    def test_beats_baseline_verdict_when_degradation_low(self):
        """SCENARIO-VERIFY-174: low degradation → honest_verdict='above_baseline'."""
        # in_dist=0.87, humaneval=0.76 → deg=0.11 (below_baseline), arc=0.04 → deg=0.83 (below)
        # Let's override with values that beat baseline
        exp824 = self._make_exp824(in_dist_auc=0.80)
        exp825 = self._make_exp825(auc_humaneval=0.76, auc_arc=0.75)
        result = self._run_main_with_mocks(exp824, exp825)
        # deg_humaneval=0.04, deg_arc=0.05 → max=0.05 <= 0.08 → above_baseline
        self.assertEqual(result["honest_verdict"], "above_baseline")
        self.assertTrue(result["beats_baseline"])
        self.assertAlmostEqual(result["cross_domain_degradation_humaneval"], 0.04, places=4)

    def test_below_baseline_verdict_when_degradation_high(self):
        """SCENARIO-VERIFY-174: high degradation → honest_verdict='below_baseline'."""
        exp824 = self._make_exp824(in_dist_auc=0.8705357142857143)
        exp825 = self._make_exp825(auc_humaneval=0.76, auc_arc=0.04)
        result = self._run_main_with_mocks(exp824, exp825)
        # deg_arc = 0.8705 - 0.04 = 0.8305 >> 0.09 → below_baseline
        self.assertEqual(result["honest_verdict"], "below_baseline")
        self.assertFalse(result["beats_baseline"])
        self.assertIn("worst_domain", result)

    def test_corroboration_rate_in_artifact(self):
        """SCENARIO-VERIFY-174: corroboration_rate is written to artifact."""
        exp824 = self._make_exp824(in_dist_auc=0.80)
        exp825 = self._make_exp825(auc_humaneval=0.76, auc_arc=0.75)
        result = self._run_main_with_mocks(exp824, exp825)
        self.assertIn("corroboration_rate", result)
        self.assertGreaterEqual(result["corroboration_rate"], 0.0)
        self.assertLessEqual(result["corroboration_rate"], 1.0)

    def test_n_certificates_matches_input(self):
        """n_certificates in artifact equals the number of certs in Exp 825."""
        exp824 = self._make_exp824(in_dist_auc=0.80)
        exp825 = self._make_exp825(auc_humaneval=0.76, auc_arc=0.75)
        result = self._run_main_with_mocks(exp824, exp825)
        self.assertEqual(result["n_certificates"], len(_CERTS_ALL_UNSAT_HIGH_ENERGY))

    def test_published_baseline_constant_in_artifact(self):
        """published_baseline is always 0.08 in the artifact (traceable to arXiv 2506.00027)."""
        exp824 = self._make_exp824(in_dist_auc=0.80)
        exp825 = self._make_exp825(auc_humaneval=0.76, auc_arc=0.75)
        result = self._run_main_with_mocks(exp824, exp825)
        self.assertEqual(result["published_baseline"], 0.08)


if __name__ == "__main__":
    unittest.main()
