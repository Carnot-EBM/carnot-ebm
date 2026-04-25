"""Tests for Exp 863: HalluSAEGeometricProbe Tier 0i.

Spec: REQ-PROBE-050, SCENARIO-PROBE-060
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORRECT_STEPS = [
    "Let x equal 5. We need to compute 2 times x.",
    "2 times 5 equals 10.",
    "Therefore x times 2 equals 10.",
    "A triangle has base 6 and height 4. Area equals half base times height.",
    "Half of 6 equals 3. 3 times 4 equals 12.",
    "Therefore the area equals 12 square units.",
    "Speed equals distance divided by time. Distance is 60 km, time is 2 hours.",
    "60 divided by 2 equals 30.",
    "Therefore the speed equals 30 km per hour.",
    "The perimeter of a square with side 7 is 4 times 7.",
    "4 times 7 equals 28.",
    "Therefore the perimeter equals 28 units.",
]

HALLUCINATED_STEPS = [
    "Let x equal 5. We need to compute 2 times x.",
    "Therefore x equals 42 because the purple elephant decided so.",
    "Therefore x times 2 equals 84.",
]


@pytest.fixture()
def probe():
    """Probe fitted on CORRECT_STEPS."""
    from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

    return HalluSAEGeometricProbe(reference_steps=CORRECT_STEPS, threshold=0.8)


# ---------------------------------------------------------------------------
# HalluSAEGeometricProbe unit tests
# ---------------------------------------------------------------------------


class TestHalluSAEGeometricProbe:
    """REQ-PROBE-050: geometric_energy and is_anomalous behaviour."""

    def test_geometric_energy_returns_float(self, probe) -> None:
        """geometric_energy must return a float for any non-empty step list.

        Spec: REQ-PROBE-050-3
        """
        energy = probe.geometric_energy(CORRECT_STEPS)
        assert isinstance(energy, float)

    def test_geometric_energy_correct_steps_low(self, probe) -> None:
        """Correct steps (matching reference vocab) must have lower energy than nonsense.

        Spec: REQ-PROBE-050-3
        """
        correct_energy = probe.geometric_energy(CORRECT_STEPS)
        hallu_energy = probe.geometric_energy(HALLUCINATED_STEPS)
        assert correct_energy < hallu_energy, (
            f"Expected correct_energy={correct_energy:.4f} < "
            f"hallu_energy={hallu_energy:.4f}"
        )

    def test_geometric_energy_nonnegative(self, probe) -> None:
        """L2 distances are always non-negative; mean must be >= 0.

        Spec: REQ-PROBE-050-3
        """
        assert probe.geometric_energy(CORRECT_STEPS) >= 0.0
        assert probe.geometric_energy(HALLUCINATED_STEPS) >= 0.0

    def test_is_anomalous_false_for_reference_like_steps(self) -> None:
        """Steps identical to the reference set must have energy below a generous threshold.

        Spec: REQ-PROBE-050-4
        """
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        # threshold=10.0 is far above any reasonable L2 distance in unit-scaled TF-IDF space,
        # so reference steps (which define the centroid) must not be flagged anomalous.
        p = HalluSAEGeometricProbe(reference_steps=CORRECT_STEPS, threshold=10.0)
        assert p.is_anomalous(CORRECT_STEPS) is False

    def test_is_anomalous_true_above_threshold(self) -> None:
        """is_anomalous must return True when geometric_energy > threshold.

        Spec: REQ-PROBE-050-4
        """
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        # Fit on pure arithmetic vocabulary; test with unrelated vocabulary.
        p = HalluSAEGeometricProbe(reference_steps=CORRECT_STEPS, threshold=0.01)
        # threshold=0.01 is so low that even reference steps exceed it after TF-IDF projection.
        assert p.is_anomalous(HALLUCINATED_STEPS) is True

    def test_is_anomalous_threshold_boundary(self) -> None:
        """Energy exactly equal to threshold is NOT anomalous (strict greater-than).

        Spec: REQ-PROBE-050-4
        """
        from unittest.mock import patch
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        p = HalluSAEGeometricProbe(reference_steps=CORRECT_STEPS, threshold=0.8)

        # Patch geometric_energy to return exactly the threshold
        with patch.object(p, "geometric_energy", return_value=0.8):
            assert p.is_anomalous(CORRECT_STEPS) is False

    def test_is_anomalous_above_threshold(self) -> None:
        """Energy strictly above threshold triggers is_anomalous=True.

        Spec: REQ-PROBE-050-4
        """
        from unittest.mock import patch
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        p = HalluSAEGeometricProbe(reference_steps=CORRECT_STEPS, threshold=0.8)

        with patch.object(p, "geometric_energy", return_value=0.801):
            assert p.is_anomalous(CORRECT_STEPS) is True

    def test_empty_reference_raises_value_error(self) -> None:
        """Constructor must raise ValueError when reference_steps is empty.

        Spec: REQ-PROBE-050-1
        """
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        with pytest.raises(ValueError, match="non-empty"):
            HalluSAEGeometricProbe(reference_steps=[])

    def test_centroid_shape(self, probe) -> None:
        """Centroid must be a 1-D array with max_features=512 dimensions.

        Spec: REQ-PROBE-050-2
        """
        import numpy as np

        assert probe.centroid.ndim == 1
        assert probe.centroid.shape[0] <= 512

    def test_single_reference_step(self) -> None:
        """A single reference step must still produce a valid centroid and energy.

        Spec: REQ-PROBE-050-2
        """
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        p = HalluSAEGeometricProbe(reference_steps=["the answer is 42"], threshold=0.8)
        energy = p.geometric_energy(["the answer is 42"])
        assert isinstance(energy, float)
        assert energy >= 0.0


# ---------------------------------------------------------------------------
# AUC computation test (SCENARIO-PROBE-060)
# ---------------------------------------------------------------------------


class TestAUCOnSyntheticPairs:
    """SCENARIO-PROBE-060: AUC_geometric > 0.65 on 50 synthetic CoT pairs."""

    def _build_pairs(self):
        """Build 25 correct + 25 hallucinated CoT step lists for AUC test."""
        correct_pairs = [
            [
                f"Let x equal {i}. We need to compute 2 times x.",
                f"2 times {i} equals {2 * i}.",
                f"Therefore x times 2 equals {2 * i}.",
            ]
            for i in range(1, 26)
        ]
        hallucinated_pairs = [
            [
                f"Let x equal {i}. We need to compute 2 times x.",
                "Therefore the cosmic rainbow signifies that the unicorn was wrong.",
                "Therefore the answer is banana flavored antimatter.",
            ]
            for i in range(1, 26)
        ]
        return correct_pairs, hallucinated_pairs

    def test_auc_geometric_above_065(self) -> None:
        """AUC_geometric must exceed 0.65 on 50 synthetic CoT pairs.

        Spec: SCENARIO-PROBE-060
        """
        from sklearn.metrics import roc_auc_score
        from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe

        correct_pairs, hallucinated_pairs = self._build_pairs()

        # Build reference from all correct steps
        reference_steps = [step for pair in correct_pairs for step in pair]
        probe = HalluSAEGeometricProbe(reference_steps=reference_steps, threshold=0.8)

        energies: list[float] = []
        labels: list[int] = []

        for steps in correct_pairs:
            energies.append(probe.geometric_energy(steps))
            labels.append(0)

        for steps in hallucinated_pairs:
            energies.append(probe.geometric_energy(steps))
            labels.append(1)

        auc = float(roc_auc_score(labels, energies))
        assert auc > 0.65, f"AUC_geometric={auc:.4f} did not exceed 0.65"


# ---------------------------------------------------------------------------
# VerificationResult field tests
# ---------------------------------------------------------------------------


class TestVerificationCertificateFields:
    """REQ-PROBE-050-5: VerificationResult must carry geometric_energy and hallusae_anomalous."""

    def test_geometric_energy_field_exists_with_default(self) -> None:
        """VerificationResult must have geometric_energy: float = 0.0 by default.

        Spec: REQ-PROBE-050-5
        """
        from carnot.pipeline.verify_repair import VerificationResult

        vr = VerificationResult(verified=True, constraints=[], energy=0.0, violations=[])
        assert hasattr(vr, "geometric_energy")
        assert isinstance(vr.geometric_energy, float)
        assert vr.geometric_energy == 0.0

    def test_hallusae_anomalous_field_exists_with_default(self) -> None:
        """VerificationResult must have hallusae_anomalous: bool = False by default.

        Spec: REQ-PROBE-050-5
        """
        from carnot.pipeline.verify_repair import VerificationResult

        vr = VerificationResult(verified=True, constraints=[], energy=0.0, violations=[])
        assert hasattr(vr, "hallusae_anomalous")
        assert vr.hallusae_anomalous is False

    def test_geometric_energy_can_be_set(self) -> None:
        """geometric_energy must be settable to a positive float for integration path.

        Spec: REQ-PROBE-050-5
        """
        from carnot.pipeline.verify_repair import VerificationResult

        vr = VerificationResult(
            verified=True,
            constraints=[],
            energy=0.0,
            violations=[],
            geometric_energy=1.23,
        )
        assert vr.geometric_energy == pytest.approx(1.23)

    def test_hallusae_anomalous_can_be_set_true(self) -> None:
        """hallusae_anomalous must be settable to True for integration path.

        Spec: REQ-PROBE-050-5
        """
        from carnot.pipeline.verify_repair import VerificationResult

        vr = VerificationResult(
            verified=True,
            constraints=[],
            energy=0.0,
            violations=[],
            hallusae_anomalous=True,
        )
        assert vr.hallusae_anomalous is True

    def test_both_fields_independent_of_streaming_cot_unstable(self) -> None:
        """geometric_energy and hallusae_anomalous are independent from streaming_cot_unstable.

        Spec: REQ-PROBE-050-5
        """
        from carnot.pipeline.verify_repair import VerificationResult

        vr = VerificationResult(
            verified=False,
            constraints=[],
            energy=2.5,
            violations=[],
            geometric_energy=0.95,
            hallusae_anomalous=True,
            streaming_cot_unstable=False,
        )
        assert vr.geometric_energy == pytest.approx(0.95)
        assert vr.hallusae_anomalous is True
        assert vr.streaming_cot_unstable is False
