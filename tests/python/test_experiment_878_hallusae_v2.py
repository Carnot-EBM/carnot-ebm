"""Tests for Exp 878: HalluSAEGeometricProbeV2 — temporal velocity + acceleration features.

Spec: REQ-VERIFY-143, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORRECT_STEPS_SHORT = [
    "Let x equal 5. We need to compute 2 times x.",
    "2 times 5 equals 10.",
    "Therefore x times 2 equals 10.",
]

HALLUCINATED_STEPS_SHORT = [
    "Let x equal 5. We need to compute 2 times x.",
    "Therefore x equals 42 because the purple elephant decided so.",
    "Therefore x times 2 equals 84.",
]

# 25 correct + 25 hallucinated pairs (same dataset as Exp 863/878)
CORRECT_PAIRS = [
    [
        f"Let x equal {i}. We need to compute 2 times x.",
        f"2 times {i} equals {2 * i}.",
        f"Therefore x times 2 equals {2 * i}.",
    ]
    for i in range(1, 26)
]
HALLUCINATED_PAIRS = [
    [
        f"Let x equal {i}. We need to compute 2 times x.",
        "The cosmic rainbow signifies the unicorn was wrong about quantum dolphins.",
        "Therefore the answer is rainbow-flavored antimatter explosions.",
    ]
    for i in range(1, 26)
]


@pytest.fixture()
def reference_steps() -> list[str]:
    """All correct CoT steps flattened into reference list."""
    return [step for pair in CORRECT_PAIRS for step in pair]


@pytest.fixture()
def probe(reference_steps):
    """Untrained V2 probe fitted on reference_steps."""
    from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

    return HalluSAEGeometricProbeV2(reference_steps=reference_steps)


@pytest.fixture()
def trained_probe(reference_steps):
    """V2 probe trained on first 15 pairs from each class."""
    from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

    p = HalluSAEGeometricProbeV2(reference_steps=reference_steps)
    p.train_trajectory(
        pos_corpus=HALLUCINATED_PAIRS[:15],
        neg_corpus=CORRECT_PAIRS[:15],
    )
    return p


# ---------------------------------------------------------------------------
# REQ-VERIFY-143-1: compute_trajectory_features shape and values
# ---------------------------------------------------------------------------


class TestComputeTrajectoryFeatures:
    """REQ-VERIFY-143-1: trajectory feature vector shape and finite-difference semantics."""

    def test_output_shape(self, probe) -> None:
        """Feature vector must have shape (6,).

        Spec: REQ-VERIFY-143-1
        """
        energies = [0.1, 0.2, 0.3]
        feat = probe.compute_trajectory_features(energies)
        assert feat.shape == (6,), f"Expected (6,), got {feat.shape}"

    def test_output_dtype(self, probe) -> None:
        """Feature vector must have float64 dtype.

        Spec: REQ-VERIFY-143-1
        """
        feat = probe.compute_trajectory_features([0.5, 1.0, 1.5])
        assert feat.dtype == np.float64

    def test_energy_mean_correct(self, probe) -> None:
        """energy_mean (index 0) must equal the arithmetic mean of step_energies.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 2.0, 3.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[0] == pytest.approx(2.0)

    def test_energy_std_correct(self, probe) -> None:
        """energy_std (index 1) must equal np.std of step_energies.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 2.0, 3.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[1] == pytest.approx(np.std([1.0, 2.0, 3.0]))

    def test_peak_energy_correct(self, probe) -> None:
        """peak_energy (index 2) must equal max of step_energies.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 5.0, 2.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[2] == pytest.approx(5.0)

    def test_velocity_mean_correct(self, probe) -> None:
        """velocity_mean (index 3) must be mean of zero-padded first differences.

        velocity = [0, 1, 1] for energies [1, 2, 3].
        velocity_mean = (0 + 1 + 1) / 3 = 2/3.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 2.0, 3.0]
        feat = probe.compute_trajectory_features(energies)
        expected_velocity_mean = (0.0 + 1.0 + 1.0) / 3.0
        assert feat[3] == pytest.approx(expected_velocity_mean)

    def test_accel_mean_correct(self, probe) -> None:
        """accel_mean (index 4) must be mean of zero-zero-padded second differences.

        energies = [1, 2, 4].
        velocity = [0, 1, 2].
        accel = [0, 0, 1].
        accel_mean = 1/3.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 2.0, 4.0]
        feat = probe.compute_trajectory_features(energies)
        expected_accel_mean = (0.0 + 0.0 + 1.0) / 3.0
        assert feat[4] == pytest.approx(expected_accel_mean)

    def test_monotone_increase_fraction_correct(self, probe) -> None:
        """monotone_increase_fraction (index 5) = fraction of steps t>=1 where velocity>0.

        energies = [1, 3, 2, 4]: velocity = [0, 2, -1, 2], transitions t>=1 = [2, -1, 2].
        Fraction where > 0: 2 out of 3 = 2/3.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.0, 3.0, 2.0, 4.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[5] == pytest.approx(2.0 / 3.0)

    def test_single_step_returns_zeros_for_dynamic_features(self, probe) -> None:
        """Single step: velocity and accel are zero; fraction is 0.

        Spec: REQ-VERIFY-143-1
        """
        energies = [1.5]
        feat = probe.compute_trajectory_features(energies)
        assert feat[3] == pytest.approx(0.0)  # velocity_mean
        assert feat[4] == pytest.approx(0.0)  # accel_mean
        assert feat[5] == pytest.approx(0.0)  # monotone_increase_fraction

    def test_empty_energies_returns_zeros(self, probe) -> None:
        """Empty energy list must return zero feature vector without crashing.

        Spec: REQ-VERIFY-143-1
        """
        feat = probe.compute_trajectory_features([])
        assert feat.shape == (6,)
        assert np.all(feat == 0.0)

    def test_constant_energies_zero_velocity(self, probe) -> None:
        """Constant energy sequence must produce velocity_mean=0 and accel_mean=0.

        Spec: REQ-VERIFY-143-1
        """
        energies = [2.0, 2.0, 2.0, 2.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[3] == pytest.approx(0.0)  # velocity_mean
        assert feat[4] == pytest.approx(0.0)  # accel_mean
        assert feat[5] == pytest.approx(0.0)  # monotone_increase_fraction


# ---------------------------------------------------------------------------
# HalluSAEGeometricProbeV2 constructor and inheritance
# ---------------------------------------------------------------------------


class TestProbeV2Constructor:
    """REQ-VERIFY-143: V2 constructor sets feature_dim and initialises classifier=None."""

    def test_feature_dim_default(self, probe) -> None:
        """Default feature_dim must be 6.

        Spec: REQ-VERIFY-143
        """
        assert probe.feature_dim == 6

    def test_classifier_none_before_training(self, probe) -> None:
        """Classifier must be None before train_trajectory() is called.

        Spec: REQ-VERIFY-143-3
        """
        assert probe.classifier is None

    def test_inherits_geometric_energy(self, probe) -> None:
        """V2 must inherit working geometric_energy() from V1.

        Spec: REQ-VERIFY-143
        """
        energy = probe.geometric_energy(CORRECT_STEPS_SHORT)
        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_empty_reference_raises(self) -> None:
        """Constructor must raise ValueError for empty reference_steps.

        Spec: REQ-VERIFY-143
        """
        from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

        with pytest.raises(ValueError):
            HalluSAEGeometricProbeV2(reference_steps=[])

    def test_custom_feature_dim(self, reference_steps) -> None:
        """feature_dim parameter must be stored on the instance.

        Spec: REQ-VERIFY-143
        """
        from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

        p = HalluSAEGeometricProbeV2(reference_steps=reference_steps, feature_dim=6)
        assert p.feature_dim == 6


# ---------------------------------------------------------------------------
# train_trajectory and classifier state
# ---------------------------------------------------------------------------


class TestTrainTrajectory:
    """REQ-VERIFY-143-3: train_trajectory sets self.classifier."""

    def test_classifier_set_after_training(self, trained_probe) -> None:
        """classifier must not be None after train_trajectory().

        Spec: REQ-VERIFY-143-3
        """
        assert trained_probe.classifier is not None

    def test_classifier_has_coef(self, trained_probe) -> None:
        """Classifier must have coef_ of shape (1, feature_dim).

        Spec: REQ-VERIFY-143-3
        """
        coef = trained_probe.classifier.coef_
        assert coef.shape == (1, trained_probe.feature_dim)

    def test_empty_pos_corpus_raises(self, probe) -> None:
        """train_trajectory must raise ValueError when pos_corpus is empty.

        Spec: REQ-VERIFY-143-3
        """
        with pytest.raises(ValueError):
            probe.train_trajectory(pos_corpus=[], neg_corpus=CORRECT_PAIRS[:5])

    def test_empty_neg_corpus_raises(self, probe) -> None:
        """train_trajectory must raise ValueError when neg_corpus is empty.

        Spec: REQ-VERIFY-143-3
        """
        with pytest.raises(ValueError):
            probe.train_trajectory(pos_corpus=HALLUCINATED_PAIRS[:5], neg_corpus=[])


# ---------------------------------------------------------------------------
# detect_trajectory
# ---------------------------------------------------------------------------


class TestDetectTrajectory:
    """REQ-VERIFY-143-4: detect_trajectory output structure and runtime error guard."""

    def test_raises_before_training(self, probe) -> None:
        """detect_trajectory must raise RuntimeError before train_trajectory() is called.

        Spec: REQ-VERIFY-143-4
        """
        with pytest.raises(RuntimeError, match="train_trajectory"):
            probe.detect_trajectory(CORRECT_STEPS_SHORT)

    def test_output_keys(self, trained_probe) -> None:
        """detect_trajectory must return dict with is_unstable_v2, trajectory_auc, feature_importances.

        Spec: REQ-VERIFY-143-4
        """
        result = trained_probe.detect_trajectory(HALLUCINATED_STEPS_SHORT)
        assert "is_unstable_v2" in result
        assert "trajectory_auc" in result
        assert "feature_importances" in result

    def test_is_unstable_v2_is_bool(self, trained_probe) -> None:
        """is_unstable_v2 must be a Python bool.

        Spec: REQ-VERIFY-143-4
        """
        result = trained_probe.detect_trajectory(HALLUCINATED_STEPS_SHORT)
        assert isinstance(result["is_unstable_v2"], bool)

    def test_feature_importances_keys(self, trained_probe) -> None:
        """feature_importances must contain all 6 named features.

        Spec: REQ-VERIFY-143-4
        """
        result = trained_probe.detect_trajectory(CORRECT_STEPS_SHORT)
        expected_keys = {
            "energy_mean",
            "energy_std",
            "peak_energy",
            "velocity_mean",
            "accel_mean",
            "monotone_increase_fraction",
        }
        assert set(result["feature_importances"].keys()) == expected_keys


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-169: V2 AUC >= 0.65 on 50 synthetic CoT pairs
# ---------------------------------------------------------------------------


class TestAUCV2OnSyntheticPairs:
    """SCENARIO-VERIFY-169: V2 classifier AUC >= 0.65 on 50 synthetic CoT pairs."""

    def test_auc_v2_above_threshold(self, reference_steps) -> None:
        """V2 AUC on 10-pair held-out test set must be >= 0.65 to close RETRO.

        Spec: SCENARIO-VERIFY-169
        """
        from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

        probe = HalluSAEGeometricProbeV2(reference_steps=reference_steps)
        probe.train_trajectory(
            pos_corpus=HALLUCINATED_PAIRS[:15],
            neg_corpus=CORRECT_PAIRS[:15],
        )
        auc = probe.compute_trajectory_auc(
            pos_corpus=HALLUCINATED_PAIRS[15:],
            neg_corpus=CORRECT_PAIRS[15:],
        )
        assert auc >= 0.65, f"V2 AUC={auc:.4f} did not reach 0.65 threshold"

    def test_auc_exceeds_v1(self, reference_steps) -> None:
        """V2 AUC must exceed the V1 AUC of 0.6144.

        Spec: SCENARIO-VERIFY-169
        """
        from carnot.probes.hallusae_geometric_probe_v2 import HalluSAEGeometricProbeV2

        V1_AUC = 0.6144
        probe = HalluSAEGeometricProbeV2(reference_steps=reference_steps)
        probe.train_trajectory(
            pos_corpus=HALLUCINATED_PAIRS[:15],
            neg_corpus=CORRECT_PAIRS[:15],
        )
        auc = probe.compute_trajectory_auc(
            pos_corpus=HALLUCINATED_PAIRS[15:],
            neg_corpus=CORRECT_PAIRS[15:],
        )
        assert auc > V1_AUC, f"V2 AUC={auc:.4f} did not exceed V1 AUC={V1_AUC}"


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-170: hallucinating trajectories show positive accel
# ---------------------------------------------------------------------------


class TestTemporalSignature:
    """SCENARIO-VERIFY-170: hallucinating CoT trajectories must show positive accel_mean."""

    def test_hallu_accel_mean_positive(self, probe) -> None:
        """Hallucinating steps (energy increases) must produce positive accel_mean.

        We construct a purely monotone-increasing energy sequence to validate
        the acceleration feature in isolation.

        Spec: SCENARIO-VERIFY-170
        """
        # Monotone-increasing energy simulates a hallucinating chain.
        increasing_energies = [1.0, 2.0, 4.0, 7.0, 11.0]
        feat = probe.compute_trajectory_features(increasing_energies)
        # accel[t] = velocity[t] - velocity[t-1]:
        # velocity = [0, 1, 2, 3, 4], accel = [0, 0, 1, 1, 1]
        # accel_mean = 3/5 = 0.6 > 0
        assert feat[4] > 0.0, f"Expected positive accel_mean for increasing energies, got {feat[4]}"

    def test_correct_oscillating_accel_near_zero(self, probe) -> None:
        """Oscillating energy (correct CoT pattern) must produce near-zero accel_mean.

        Spec: SCENARIO-VERIFY-170
        """
        # Perfectly oscillating energy: +delta, -delta, +delta, ...
        oscillating_energies = [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        feat = probe.compute_trajectory_features(oscillating_energies)
        # velocity = [0, 1, -1, 1, -1, 1]
        # accel = [0, 0, -2, 2, -2, 2], accel_mean = 0
        assert abs(feat[4]) < 0.5, (
            f"Oscillating energy should have near-zero accel_mean, got {feat[4]}"
        )

    def test_monotone_increase_fraction_high_for_hallu(self, probe) -> None:
        """Monotonically increasing energies must produce monotone_increase_fraction = 1.0.

        Spec: SCENARIO-VERIFY-170
        """
        energies = [1.0, 2.0, 3.0, 4.0, 5.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[5] == pytest.approx(1.0)

    def test_monotone_increase_fraction_zero_for_decreasing(self, probe) -> None:
        """Monotonically decreasing energies must produce monotone_increase_fraction = 0.0.

        Spec: SCENARIO-VERIFY-170
        """
        energies = [5.0, 4.0, 3.0, 2.0, 1.0]
        feat = probe.compute_trajectory_features(energies)
        assert feat[5] == pytest.approx(0.0)
