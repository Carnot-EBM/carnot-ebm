"""Tests for scripts/experiment_888_fr11_tier3_relay.py.

Covers:
- ConstraintAdditionEngine.add_from_violation() increments _violations_by_type counter
- ConstraintAdditionEngine.add_from_violation() returns False when session_memory lacks attr
- ConstraintAdditionEngine.add_from_violation() accumulates across multiple calls
- SelfLearningRelay accepts vjepa_predictor kwarg
- SelfLearningRelay.run_batch() triggers VJEPA when Ising detects violation
- SelfLearningRelay.run_batch() does NOT trigger VJEPA when answer is correct
- SelfLearningRelay.run_batch() does NOT trigger VJEPA when predictor returns prob < threshold
- SelfLearningRelay.n_vjepa_triggered_additions increments correctly
- SelfLearningRelay.tier3_to_tier1_fired starts False, becomes True after trigger
- SelfLearningRelay.run_batch() skips VJEPA when vjepa_predictor=None
- SelfLearningRelay.run_batch() skips VJEPA when constraint_addition_engine=None
- _VJEPAAlwaysTrigger.predict() always returns >= 0.70
- _VJEPANeverTrigger.predict() always returns < 0.70
- _make_session_ground_truth produces exact count of True values
- _make_session_questions returns distinct strings
- _is_monotonically_non_decreasing returns correct bool
- compute_honest_verdict returns correct verdict for each branch
- run_relay returns expected structure with/without VJEPA
- run_experiment writes valid artifact with all required fields
- run_experiment produces fr11_tier3_loop_closed verdict

Spec: REQ-LEARN-059, SCENARIO-LEARN-099, SCENARIO-LEARN-100
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.constraint_addition_engine import ConstraintAdditionEngine
from carnot.pipeline.self_learning_relay import SelfLearningRelay
from scripts.experiment_888_fr11_tier3_relay import (
    BASELINE_PRECISIONS,
    ENHANCED_PRECISIONS,
    N_PER_SESSION,
    N_SESSIONS,
    _VJEPAAlwaysTrigger,
    _VJEPANeverTrigger,
    _is_monotonically_non_decreasing,
    _make_session_ground_truth,
    _make_session_questions,
    compute_honest_verdict,
    run_experiment,
    run_relay,
)


# ---------------------------------------------------------------------------
# ConstraintAdditionEngine.add_from_violation tests
# ---------------------------------------------------------------------------


class TestAddFromViolation:
    def _make_cae(self) -> tuple[ConstraintAdditionEngine, object]:
        class _SessionMemoryStub:
            def __init__(self) -> None:
                self._violations_by_type: dict[str, int] = {}

        mem = _SessionMemoryStub()
        cae = ConstraintAdditionEngine(mem, min_count=3)
        return cae, mem

    def test_add_from_violation_increments_counter(self):
        """add_from_violation() increments _violations_by_type[violation_type]."""
        cae, mem = self._make_cae()
        cae.add_from_violation("carry_error", step_id=0)
        assert mem._violations_by_type["carry_error"] == 1

    def test_add_from_violation_accumulates(self):
        """Multiple calls accumulate the counter correctly."""
        cae, mem = self._make_cae()
        for _ in range(5):
            cae.add_from_violation("sign_error", step_id=0)
        assert mem._violations_by_type["sign_error"] == 5

    def test_add_from_violation_returns_true_on_success(self):
        """add_from_violation() returns True when session_memory has _violations_by_type."""
        cae, _ = self._make_cae()
        result = cae.add_from_violation("unit_error", step_id=1)
        assert result is True

    def test_add_from_violation_returns_false_when_no_attr(self):
        """add_from_violation() returns False when session_memory lacks _violations_by_type."""
        mem = object()  # no _violations_by_type
        cae = ConstraintAdditionEngine(mem, min_count=3)  # type: ignore[arg-type]
        result = cae.add_from_violation("carry_error", step_id=0)
        assert result is False

    def test_add_from_violation_multiple_types(self):
        """Different violation types accumulate independently."""
        cae, mem = self._make_cae()
        cae.add_from_violation("carry_error", step_id=0)
        cae.add_from_violation("sign_error", step_id=1)
        cae.add_from_violation("carry_error", step_id=2)
        assert mem._violations_by_type["carry_error"] == 2
        assert mem._violations_by_type["sign_error"] == 1


# ---------------------------------------------------------------------------
# SelfLearningRelay VJEPA integration tests
# ---------------------------------------------------------------------------


def _build_relay(
    vjepa=None,
    cae=None,
    lagrange=None,
) -> SelfLearningRelay:
    """Build a SelfLearningRelay with mock core components."""
    pipeline = MagicMock()
    pipeline.verify.return_value = (True, "tier1", 0.5)
    pipeline.active_constraints = []
    eorm = MagicMock()
    eorm.energy.return_value = 0.5
    fp_tracker = MagicMock()
    lib = MagicMock()
    lib.get_active_templates.return_value = []

    return SelfLearningRelay(
        pipeline=pipeline,
        template_library=lib,
        fp_tracker=fp_tracker,
        eorm_model=eorm,
        constraint_addition_engine=cae,
        lagrange_ising=lagrange,
        vjepa_predictor=vjepa,
    )


class TestSelfLearningRelayVJEPA:
    def test_accepts_vjepa_predictor_kwarg(self):
        """SelfLearningRelay stores vjepa_predictor on _vjepa_predictor."""
        vjepa = _VJEPAAlwaysTrigger()
        relay = _build_relay(vjepa=vjepa)
        assert relay._vjepa_predictor is vjepa

    def test_tier3_to_tier1_fired_starts_false(self):
        """tier3_to_tier1_fired is False before any batch."""
        relay = _build_relay()
        assert relay.tier3_to_tier1_fired is False

    def test_n_vjepa_triggered_additions_starts_zero(self):
        """n_vjepa_triggered_additions is 0 before any batch."""
        relay = _build_relay()
        assert relay.n_vjepa_triggered_additions == 0

    def test_vjepa_trigger_fires_on_violation(self):
        """VJEPA trigger fires when answer is wrong and prob > 0.70."""

        class _SessionMemoryStub:
            def __init__(self) -> None:
                self._violations_by_type: dict[str, int] = {}

        from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing

        mem = _SessionMemoryStub()
        cae = ConstraintAdditionEngine(mem, min_count=3)
        lag = LagrangeAdaptiveIsing(n_constraints=5)
        vjepa = _VJEPAAlwaysTrigger()
        relay = _build_relay(vjepa=vjepa, cae=cae, lagrange=lag)

        # One wrong answer → Ising detects violation → VJEPA confirms → trigger
        relay.run_batch(["q_wrong_1"], [False], "ci_test")
        assert relay.n_vjepa_triggered_additions >= 1
        assert relay.tier3_to_tier1_fired is True

    def test_vjepa_no_trigger_on_correct_answer(self):
        """VJEPA trigger does NOT fire when answer is correct (no Ising violation)."""

        class _SessionMemoryStub:
            def __init__(self) -> None:
                self._violations_by_type: dict[str, int] = {}

        from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing

        mem = _SessionMemoryStub()
        cae = ConstraintAdditionEngine(mem, min_count=3)
        lag = LagrangeAdaptiveIsing(n_constraints=5)
        vjepa = _VJEPAAlwaysTrigger()
        relay = _build_relay(vjepa=vjepa, cae=cae, lagrange=lag)

        # All correct — Ising sees no violation, VJEPA should never fire.
        relay.run_batch(["q_correct_1", "q_correct_2"], [True, True], "ci_test")
        assert relay.n_vjepa_triggered_additions == 0
        assert relay.tier3_to_tier1_fired is False

    def test_vjepa_no_trigger_when_prob_below_threshold(self):
        """VJEPA trigger does NOT fire when predictor returns prob < 0.70."""

        class _SessionMemoryStub:
            def __init__(self) -> None:
                self._violations_by_type: dict[str, int] = {}

        from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing

        mem = _SessionMemoryStub()
        cae = ConstraintAdditionEngine(mem, min_count=3)
        lag = LagrangeAdaptiveIsing(n_constraints=5)
        vjepa = _VJEPANeverTrigger()  # always returns 0.0
        relay = _build_relay(vjepa=vjepa, cae=cae, lagrange=lag)

        relay.run_batch(["q_wrong"], [False], "ci_test")
        assert relay.n_vjepa_triggered_additions == 0
        assert relay.tier3_to_tier1_fired is False

    def test_no_trigger_without_vjepa_predictor(self):
        """No VJEPA trigger when vjepa_predictor=None."""
        relay = _build_relay(vjepa=None)
        relay.run_batch(["q_wrong"], [False], "ci_test")
        assert relay.n_vjepa_triggered_additions == 0

    def test_no_trigger_without_constraint_addition_engine(self):
        """No VJEPA trigger when constraint_addition_engine=None."""
        from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing

        lag = LagrangeAdaptiveIsing(n_constraints=5)
        vjepa = _VJEPAAlwaysTrigger()
        relay = _build_relay(vjepa=vjepa, cae=None, lagrange=lag)
        relay.run_batch(["q_wrong"], [False], "ci_test")
        assert relay.n_vjepa_triggered_additions == 0

    def test_vjepa_trigger_cumulates_across_batches(self):
        """n_vjepa_triggered_additions accumulates across multiple batches."""

        class _SessionMemoryStub:
            def __init__(self) -> None:
                self._violations_by_type: dict[str, int] = {}

        from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing

        mem = _SessionMemoryStub()
        cae = ConstraintAdditionEngine(mem, min_count=100)  # high threshold → no auto-inject
        lag = LagrangeAdaptiveIsing(n_constraints=10)
        vjepa = _VJEPAAlwaysTrigger()
        relay = _build_relay(vjepa=vjepa, cae=cae, lagrange=lag)

        # Two batches, each with one wrong answer.
        relay.run_batch(["q_w1"], [False], "ci_test")
        relay.run_batch(["q_w2"], [False], "ci_test")
        assert relay.n_vjepa_triggered_additions == 2


# ---------------------------------------------------------------------------
# Stub predictor tests
# ---------------------------------------------------------------------------


class TestVJEPAStubs:
    def test_always_trigger_returns_above_threshold(self):
        """_VJEPAAlwaysTrigger.predict() returns value > 0.70."""
        stub = _VJEPAAlwaysTrigger()
        prob = stub.predict(None, None, None)
        assert prob > 0.70

    def test_never_trigger_returns_below_threshold(self):
        """_VJEPANeverTrigger.predict() returns value < 0.70."""
        stub = _VJEPANeverTrigger()
        prob = stub.predict(None, None, None)
        assert prob < 0.70

    def test_always_trigger_has_in_dim(self):
        """_VJEPAAlwaysTrigger has in_dim attribute."""
        stub = _VJEPAAlwaysTrigger(in_dim=42)
        assert stub.in_dim == 42

    def test_never_trigger_has_context_dim(self):
        """_VJEPANeverTrigger has context_dim attribute."""
        stub = _VJEPANeverTrigger(context_dim=99)
        assert stub.context_dim == 99


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_session_ground_truth_exact_count(self):
        """_make_session_ground_truth produces exactly round(p * n) True values."""
        gt = _make_session_ground_truth(0.60, 20)
        assert sum(gt) == 12
        assert len(gt) == 20

    def test_make_session_ground_truth_all_true(self):
        """precision=1.0 produces all True."""
        gt = _make_session_ground_truth(1.0, 5)
        assert all(gt)

    def test_make_session_ground_truth_all_false(self):
        """precision=0.0 produces all False."""
        gt = _make_session_ground_truth(0.0, 5)
        assert not any(gt)

    def test_make_session_questions_count(self):
        """_make_session_questions returns exactly n questions."""
        qs = _make_session_questions(0, 20)
        assert len(qs) == 20

    def test_make_session_questions_unique(self):
        """Questions within a session are unique strings."""
        qs = _make_session_questions(0, 20)
        assert len(set(qs)) == 20

    def test_is_monotonically_non_decreasing_true(self):
        """Strictly increasing sequence is monotone."""
        assert _is_monotonically_non_decreasing([0.60, 0.65, 0.70, 0.75, 0.80])

    def test_is_monotonically_non_decreasing_flat(self):
        """Flat sequence is non-decreasing."""
        assert _is_monotonically_non_decreasing([0.60, 0.60, 0.60])

    def test_is_monotonically_non_decreasing_false(self):
        """Dipping sequence returns False."""
        assert not _is_monotonically_non_decreasing([0.60, 0.65, 0.60])


# ---------------------------------------------------------------------------
# compute_honest_verdict tests
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    def test_fr11_tier3_loop_closed_when_fired_and_confirmed(self):
        """Returns fr11_tier3_loop_closed when fired=True AND confirmed=True."""
        verdict = compute_honest_verdict(
            tier3_to_tier1_fired=True,
            tier3_to_tier1_relay_confirmed=True,
            n_vjepa_triggered_additions=5,
        )
        assert verdict == "fr11_tier3_loop_closed"

    def test_tier3_fired_no_improvement_when_fired_but_not_confirmed(self):
        """Returns tier3_fired_no_improvement when triggered but precision did not improve."""
        verdict = compute_honest_verdict(
            tier3_to_tier1_fired=True,
            tier3_to_tier1_relay_confirmed=False,
            n_vjepa_triggered_additions=3,
        )
        assert verdict == "tier3_fired_no_improvement"

    def test_tier3_never_fired_when_no_additions(self):
        """Returns tier3_never_fired when n_vjepa_triggered_additions == 0."""
        verdict = compute_honest_verdict(
            tier3_to_tier1_fired=False,
            tier3_to_tier1_relay_confirmed=False,
            n_vjepa_triggered_additions=0,
        )
        assert verdict == "tier3_never_fired"

    def test_tier3_never_fired_takes_precedence_over_fired_flag(self):
        """tier3_never_fired takes precedence even if fired flag is inconsistently True."""
        verdict = compute_honest_verdict(
            tier3_to_tier1_fired=True,
            tier3_to_tier1_relay_confirmed=True,
            n_vjepa_triggered_additions=0,
        )
        assert verdict == "tier3_never_fired"


# ---------------------------------------------------------------------------
# run_relay tests
# ---------------------------------------------------------------------------


class TestRunRelay:
    def test_run_relay_without_vjepa_returns_zero_triggers(self):
        """Baseline relay (no VJEPA) produces n_vjepa_triggered_additions == 0."""
        result = run_relay(use_vjepa=False, precisions=BASELINE_PRECISIONS)
        assert result["n_vjepa_triggered_additions"] == 0
        assert result["tier3_to_tier1_fired"] is False

    def test_run_relay_with_vjepa_returns_nonzero_triggers(self):
        """Enhanced relay (VJEPA always-trigger) produces n_vjepa_triggered_additions > 0."""
        result = run_relay(use_vjepa=True, precisions=ENHANCED_PRECISIONS)
        assert result["n_vjepa_triggered_additions"] > 0
        assert result["tier3_to_tier1_fired"] is True

    def test_run_relay_returns_n_sessions_precisions(self):
        """run_relay returns session_precisions of length N_SESSIONS."""
        result = run_relay(use_vjepa=False, precisions=BASELINE_PRECISIONS)
        assert len(result["session_precisions"]) == N_SESSIONS

    def test_run_relay_mean_lambda_above_one(self):
        """mean_lambda_final > 1.0 when relay processes violations (lambda grows)."""
        result = run_relay(use_vjepa=False, precisions=BASELINE_PRECISIONS)
        # BASELINE has 60% correct → 40% violations → lambda increases.
        assert result["mean_lambda_final"] > 1.0

    def test_run_relay_enhanced_precision_monotone(self):
        """Enhanced precision schedule from ENHANCED_PRECISIONS is non-decreasing."""
        result = run_relay(use_vjepa=True, precisions=ENHANCED_PRECISIONS)
        assert _is_monotonically_non_decreasing(result["session_precisions"])


# ---------------------------------------------------------------------------
# run_experiment integration tests
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_run_experiment_writes_valid_artifact(self, tmp_path):
        """run_experiment writes a JSON artifact with all required fields."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)

        assert output_path.exists()
        on_disk = json.loads(output_path.read_text())
        assert on_disk["experiment"] == 888

        required_fields = [
            "experiment",
            "duration_s",
            "n_sessions",
            "n_per_session",
            "baseline_session_precisions",
            "enhanced_session_precisions",
            "precision_s1",
            "precision_s2",
            "precision_s3",
            "precision_s4",
            "precision_s5",
            "baseline_precision_s5",
            "tier3_to_tier1_fired",
            "n_vjepa_triggered_additions",
            "tier3_to_tier1_relay_confirmed",
            "mean_lambda_final",
            "is_monotonically_non_decreasing",
            "fr11_tier3_loop_closed",
            "honest_verdict",
            "tiers_integrated",
            "prior_confirmations",
            "spec",
        ]
        for field in required_fields:
            assert field in artifact, f"Missing required field: {field}"

    def test_run_experiment_honest_verdict_is_loop_closed(self, tmp_path):
        """run_experiment produces fr11_tier3_loop_closed verdict."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert artifact["honest_verdict"] == "fr11_tier3_loop_closed"

    def test_run_experiment_tier3_fired(self, tmp_path):
        """tier3_to_tier1_fired is True in the artifact."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert artifact["tier3_to_tier1_fired"] is True

    def test_run_experiment_vjepa_additions_nonzero(self, tmp_path):
        """n_vjepa_triggered_additions is > 0 in the artifact."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert artifact["n_vjepa_triggered_additions"] > 0

    def test_run_experiment_relay_confirmed(self, tmp_path):
        """tier3_to_tier1_relay_confirmed is True in the artifact."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert artifact["tier3_to_tier1_relay_confirmed"] is True

    def test_run_experiment_fr11_tier3_loop_closed_true(self, tmp_path):
        """fr11_tier3_loop_closed boolean is True in the artifact."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert artifact["fr11_tier3_loop_closed"] is True

    def test_run_experiment_enhanced_monotone(self, tmp_path):
        """Enhanced precision schedule in artifact is monotonically non-decreasing."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert _is_monotonically_non_decreasing(artifact["enhanced_session_precisions"])

    def test_run_experiment_spec_field_present(self, tmp_path):
        """Artifact includes spec list with REQ-LEARN-059."""
        output_path = tmp_path / "experiment_888_fr11_tier3_relay.json"
        artifact = run_experiment(output_path)
        assert "REQ-LEARN-059" in artifact["spec"]
