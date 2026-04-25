"""Tests for Experiment 864 — FR-11 Tier 2 integration v5 relay.

All external dependencies (EORM, SinkProbe, Ising sampler) are mocked so the
test suite runs on CPU in CI without GPU or SOTA GGUFs.

Traces to: REQ-FR11-030, SCENARIO-FR11-040
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import targets
# ---------------------------------------------------------------------------

from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline
from carnot.probes.hallusae_geometric_probe import HalluSAEGeometricProbe
from carnot.probes.streaming_cot_detector import StreamingCoTHalluDetector
from carnot.samplers.lagrange_adaptive import LagrangeAdaptiveIsingConstraints

from scripts.experiment_864_fr11_tier2_integration_v5 import (
    _REFERENCE_STEPS,
    _DEFAULT_CONSTRAINTS,
    _MockEORMModel,
    _build_pipeline,
    _make_session_problems,
    run_relay,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_sink_probe():
    """A MagicMock with spec=SinkProbe that never triggers early exit."""
    from carnot.pipeline.sink_probe import SinkProbe

    probe = MagicMock(spec=SinkProbe)
    return probe


@pytest.fixture()
def mock_ising():
    """Stub Ising callable: True (verified) when 'Verified' in response."""

    def _ising(response: str, question: str) -> tuple[bool, float]:
        ok = "Verified" in response
        return ok, 0.1 if ok else 0.8

    return _ising


@pytest.fixture()
def base_pipeline(mock_sink_probe, mock_ising):
    """ThreeTierPipeline with no optional probes wired."""
    return ThreeTierPipeline(
        sink_probe=mock_sink_probe,
        eorm_model=_MockEORMModel(),
        ising_pipeline=mock_ising,
        eorm_threshold=0.5,
    )


@pytest.fixture()
def full_pipeline():
    """ThreeTierPipeline with all three FR-11 tiers wired via _build_pipeline()."""
    return _build_pipeline(_REFERENCE_STEPS)


# ---------------------------------------------------------------------------
# REQ-FR11-030-1: wire_tier_0g
# ---------------------------------------------------------------------------


class TestWireTier0g:
    """REQ-FR11-030-1: wire_tier_0g attaches StreamingCoTHalluDetector."""

    def test_wire_sets_detector(self, base_pipeline):
        """Wiring a detector stores it on the pipeline instance."""
        assert base_pipeline.streaming_cot_detector is None
        detector = StreamingCoTHalluDetector(eorm_model=_MockEORMModel())
        base_pipeline.wire_tier_0g(detector)
        assert base_pipeline.streaming_cot_detector is detector

    def test_verify_extended_without_wire_returns_false(self, base_pipeline):
        """When no Tier 0g detector is wired, streaming_cot_unstable defaults to False."""
        result = base_pipeline.verify_extended("Some response. Verified.")
        assert result["streaming_cot_unstable"] is False

    def test_verify_extended_with_wire_calls_detector(self, base_pipeline):
        """verify_extended() calls process_step() for each CoT step when wired."""
        detector = MagicMock(spec=StreamingCoTHalluDetector)
        detector.is_streaming_unstable.return_value = True
        base_pipeline.wire_tier_0g(detector)
        result = base_pipeline.verify_extended("Step one.\nStep two.")
        detector.reset.assert_called_once()
        assert detector.process_step.call_count == 2
        assert result["streaming_cot_unstable"] is True

    def test_detector_reset_between_calls(self, base_pipeline):
        """detector.reset() is called before processing each new response."""
        detector = MagicMock(spec=StreamingCoTHalluDetector)
        detector.is_streaming_unstable.return_value = False
        base_pipeline.wire_tier_0g(detector)
        base_pipeline.verify_extended("Response A.")
        base_pipeline.verify_extended("Response B.")
        assert detector.reset.call_count == 2


# ---------------------------------------------------------------------------
# REQ-FR11-030-2: wire_tier_0i
# ---------------------------------------------------------------------------


class TestWireTier0i:
    """REQ-FR11-030-2: wire_tier_0i attaches HalluSAEGeometricProbe."""

    def test_wire_sets_probe(self, base_pipeline):
        """Wiring a probe stores it on the pipeline instance."""
        assert base_pipeline.hallusae_probe is None
        probe = HalluSAEGeometricProbe(reference_steps=_REFERENCE_STEPS)
        base_pipeline.wire_tier_0i(probe)
        assert base_pipeline.hallusae_probe is probe

    def test_verify_extended_without_wire_returns_zero(self, base_pipeline):
        """When no Tier 0i probe is wired, geometric_energy=0.0 and hallusae_anomalous=False."""
        result = base_pipeline.verify_extended("Any response.")
        assert result["geometric_energy"] == 0.0
        assert result["hallusae_anomalous"] is False

    def test_verify_extended_with_wire_calls_probe(self, base_pipeline):
        """verify_extended() calls geometric_energy() and is_anomalous() when wired."""
        probe = MagicMock(spec=HalluSAEGeometricProbe)
        probe.geometric_energy.return_value = 1.5
        probe.is_anomalous.return_value = True
        base_pipeline.wire_tier_0i(probe)
        result = base_pipeline.verify_extended("Hallucinated step here.")
        probe.geometric_energy.assert_called_once()
        probe.is_anomalous.assert_called_once()
        assert result["geometric_energy"] == 1.5
        assert result["hallusae_anomalous"] is True


# ---------------------------------------------------------------------------
# REQ-FR11-030-3: wire_lagrange
# ---------------------------------------------------------------------------


class TestWireLagrange:
    """REQ-FR11-030-3: wire_lagrange attaches LagrangeAdaptiveIsingConstraints."""

    def test_wire_sets_adaptive(self, base_pipeline):
        """Wiring an adaptive instance stores it on the pipeline."""
        assert base_pipeline.lagrange_adaptive is None
        adaptive = LagrangeAdaptiveIsingConstraints(n_spins=8, n_constraints=4)
        base_pipeline.wire_lagrange(adaptive)
        assert base_pipeline.lagrange_adaptive is adaptive

    def test_run_lagrange_session_without_wire_returns_empty(self, base_pipeline):
        """run_lagrange_session() returns {} when no adaptive instance is wired."""
        result = base_pipeline.run_lagrange_session(_DEFAULT_CONSTRAINTS)
        assert result == {}

    def test_run_lagrange_session_delegates_to_adaptive(self, base_pipeline):
        """run_lagrange_session() delegates to lagrange_adaptive.run_session()."""
        adaptive = MagicMock(spec=LagrangeAdaptiveIsingConstraints)
        adaptive.run_session.return_value = {"violation_rate": 0.25, "lambdas": [1.1, 1.0, 1.0, 1.0]}
        base_pipeline.wire_lagrange(adaptive)
        result = base_pipeline.run_lagrange_session(_DEFAULT_CONSTRAINTS, n_sweeps=20, n_samples=3)
        adaptive.run_session.assert_called_once_with(_DEFAULT_CONSTRAINTS, n_sweeps=20, n_samples=3)
        assert result["violation_rate"] == 0.25

    def test_lambdas_update_across_sessions(self):
        """Lagrange lambdas grow across sessions when constraints are violated."""
        adaptive = LagrangeAdaptiveIsingConstraints(
            n_spins=8, n_constraints=4, lambda_init=1.0, lambda_lr=0.1
        )
        initial_lambdas = adaptive.lambdas.copy()
        adaptive.run_session(_DEFAULT_CONSTRAINTS, n_sweeps=20, n_samples=5)
        # Lambdas must be >= initial because they only grow (or stay the same on zero violation).
        assert np.all(adaptive.lambdas >= initial_lambdas)


# ---------------------------------------------------------------------------
# REQ-FR11-030-4: verify_extended return shape
# ---------------------------------------------------------------------------


class TestVerifyExtendedShape:
    """REQ-FR11-030-4: verify_extended returns dict with all required keys."""

    REQUIRED_KEYS = {
        "verified",
        "tier_used",
        "energy",
        "streaming_cot_unstable",
        "geometric_energy",
        "hallusae_anomalous",
    }

    def test_all_keys_present_unwired(self, base_pipeline):
        """All six keys present even when no probes are wired."""
        result = base_pipeline.verify_extended("Test response. Verified.")
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_all_keys_present_wired(self, full_pipeline):
        """All six keys present when all three tiers are wired."""
        result = full_pipeline.verify_extended(
            "Let n = 5. Then 2n = 10. Subtracting 3 gives 7. Verified."
        )
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_correct_response_verified_true(self, full_pipeline):
        """A correct-pattern response produces verified=True via EORM fast path."""
        result = full_pipeline.verify_extended("Step one. Subtracting gives result. Verified.")
        assert result["verified"] is True

    def test_hallucinated_response_verified_false(self, full_pipeline):
        """A hallucinated-pattern response produces verified=False."""
        result = full_pipeline.verify_extended("Let n = 5. Then 2n = 999 via magic. Banana.")
        assert result["verified"] is False


# ---------------------------------------------------------------------------
# REQ-FR11-030-5: defaults when probes are absent
# ---------------------------------------------------------------------------


class TestVerifyExtendedDefaults:
    """REQ-FR11-030-5: missing probes produce safe defaults."""

    def test_streaming_default_false(self, base_pipeline):
        assert base_pipeline.verify_extended("resp")["streaming_cot_unstable"] is False

    def test_geometric_energy_default_zero(self, base_pipeline):
        assert base_pipeline.verify_extended("resp")["geometric_energy"] == 0.0

    def test_hallusae_anomalous_default_false(self, base_pipeline):
        assert base_pipeline.verify_extended("resp")["hallusae_anomalous"] is False


# ---------------------------------------------------------------------------
# _split_cot_steps helper
# ---------------------------------------------------------------------------


class TestSplitCotSteps:
    """ThreeTierPipeline._split_cot_steps splits by newlines; never returns empty."""

    def test_multiline_response(self):
        steps = ThreeTierPipeline._split_cot_steps("Step one.\nStep two.\nStep three.")
        assert steps == ["Step one.", "Step two.", "Step three."]

    def test_single_line_returns_list_of_one(self):
        steps = ThreeTierPipeline._split_cot_steps("Only one line.")
        assert steps == ["Only one line."]

    def test_empty_string_returns_list_of_original(self):
        steps = ThreeTierPipeline._split_cot_steps("")
        assert steps == [""]

    def test_blank_lines_filtered(self):
        steps = ThreeTierPipeline._split_cot_steps("A.\n\nB.")
        assert steps == ["A.", "B."]


# ---------------------------------------------------------------------------
# SCENARIO-FR11-040: 5-session relay
# ---------------------------------------------------------------------------


class TestRunRelay:
    """SCENARIO-FR11-040: 5-session relay produces correct structure."""

    def test_relay_returns_five_sessions(self, full_pipeline):
        """session_aucs and session_violation_rates each have 5 entries."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        assert len(result["session_aucs"]) == 5
        assert len(result["session_violation_rates"]) == 5

    def test_auc_in_valid_range(self, full_pipeline):
        """Each session AUC is in [0, 1]."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        for auc in result["session_aucs"]:
            assert 0.0 <= auc <= 1.0

    def test_violation_rate_in_valid_range(self, full_pipeline):
        """Each session violation rate is in [0, 1]."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        for vr in result["session_violation_rates"]:
            assert 0.0 <= vr <= 1.0

    def test_delta_auc_computed_correctly(self, full_pipeline):
        """delta_auc_s1_to_s5 equals aucs[-1] - aucs[0]."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        expected = result["session_aucs"][-1] - result["session_aucs"][0]
        assert abs(result["delta_auc_s1_to_s5"] - expected) < 1e-9

    def test_delta_violations_computed_correctly(self, full_pipeline):
        """delta_violations_s1_to_s5 equals vr[0] - vr[-1]."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        expected = result["session_violation_rates"][0] - result["session_violation_rates"][-1]
        assert abs(result["delta_violations_s1_to_s5"] - expected) < 1e-9

    def test_tier2_relay_confirmed_logic(self, full_pipeline):
        """tier2_relay_confirmed is True iff delta_auc > 0 OR delta_violations > 0."""
        result = run_relay(full_pipeline, n_sessions=5, n_per_session=10)
        da = result["delta_auc_s1_to_s5"]
        dv = result["delta_violations_s1_to_s5"]
        expected = da > 0 or dv > 0
        assert result["tier2_relay_confirmed"] == expected


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


class TestMakeSessionProblems:
    """_make_session_problems generates balanced synthetic CoT problems."""

    def test_balanced_labels(self):
        responses, labels = _make_session_problems(n_correct=25, n_halluc=25, seed=0)
        assert sum(labels) == 25
        assert len(labels) == 50

    def test_correct_responses_contain_verified(self):
        responses, labels = _make_session_problems(n_correct=10, n_halluc=0, seed=0)
        for r in responses:
            assert "Verified" in r

    def test_halluc_responses_contain_banana(self):
        responses, labels = _make_session_problems(n_correct=0, n_halluc=10, seed=0)
        for r in responses:
            assert "banana" in r

    def test_seed_deterministic(self):
        r1, l1 = _make_session_problems(seed=42)
        r2, l2 = _make_session_problems(seed=42)
        assert r1 == r2 and l1 == l2

    def test_different_seeds_differ(self):
        r1, _ = _make_session_problems(seed=0)
        r2, _ = _make_session_problems(seed=99)
        assert r1 != r2


# ---------------------------------------------------------------------------
# Mock EORM model
# ---------------------------------------------------------------------------


class TestMockEORMModel:
    """_MockEORMModel returns correct energies for CI-safe testing."""

    def test_verified_response_low_energy(self):
        model = _MockEORMModel()
        inp = MagicMock()
        inp.response_text = "Step. Verified."
        assert model.energy(inp) == 0.1

    def test_hallucinated_response_high_energy(self):
        model = _MockEORMModel()
        inp = MagicMock()
        inp.response_text = "Banana magic."
        assert model.energy(inp) == 0.8


# ---------------------------------------------------------------------------
# _build_pipeline integration check
# ---------------------------------------------------------------------------


class TestBuildPipeline:
    """_build_pipeline() produces a fully wired ThreeTierPipeline."""

    def test_all_three_tiers_wired(self):
        pipeline = _build_pipeline(_REFERENCE_STEPS)
        assert pipeline.streaming_cot_detector is not None
        assert pipeline.hallusae_probe is not None
        assert pipeline.lagrange_adaptive is not None

    def test_pipeline_is_three_tier(self):
        pipeline = _build_pipeline(_REFERENCE_STEPS)
        assert isinstance(pipeline, ThreeTierPipeline)

    def test_lagrange_instance_type(self):
        pipeline = _build_pipeline(_REFERENCE_STEPS)
        assert isinstance(pipeline.lagrange_adaptive, LagrangeAdaptiveIsingConstraints)


# ---------------------------------------------------------------------------
# Deliverable JSON existence (integration smoke test)
# ---------------------------------------------------------------------------


class TestDeliverableJson:
    """The deliverable JSON must exist and contain all required fields."""

    DELIVERABLE = Path("results/experiment_864_fr11_tier2_integration_v5.json")

    REQUIRED_FIELDS = {
        "experiment",
        "title",
        "run_date",
        "status",
        "honest_verdict",
        "tier2_relay_confirmed",
        "session_aucs",
        "session_violation_rates",
        "delta_auc_s1_to_s5",
        "delta_violations_s1_to_s5",
        "tiers_integrated",
    }

    def test_deliverable_exists(self):
        assert self.DELIVERABLE.exists(), f"Missing deliverable: {self.DELIVERABLE}"

    def test_required_fields_present(self):
        data = json.loads(self.DELIVERABLE.read_text())
        missing = self.REQUIRED_FIELDS - set(data.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_tiers_integrated_correct(self):
        data = json.loads(self.DELIVERABLE.read_text())
        assert set(data["tiers_integrated"]) == {"0g", "0i", "3_lagrange"}

    def test_session_aucs_length(self):
        data = json.loads(self.DELIVERABLE.read_text())
        assert len(data["session_aucs"]) == 5

    def test_honest_verdict_valid(self):
        data = json.loads(self.DELIVERABLE.read_text())
        valid = {"fr11_tier2_confirmed", "fr11_tier2_no_improvement", "integration_partial"}
        assert data["honest_verdict"] in valid
