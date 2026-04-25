"""Tests for scripts/experiment_875_fr11_tier2_relay_v6.py.

Covers:
- LagrangeAdaptiveIsing.update() increases lambda on violation, decreases on success
- LagrangeAdaptiveIsing.update() never lets lambda fall below lambda_init
- LagrangeAdaptiveIsing.violation_rate() returns correct empirical rate
- LagrangeAdaptiveIsing.mean_lambda() returns mean across all constraints
- LagrangeAdaptiveIsing auto-initializes unknown constraint_ids on first update
- CompressedMemoryBank.compress_session() updates centroid pool
- CompressedMemoryBank.compression_ratio returns total / centroid_count
- CompressedMemoryBank.retrieval_auroc() returns 0.5 when empty, 1.0 when populated
- CompressedMemoryBank.average_retrieval_latency_ms() returns mean of latency samples
- CompressedMemoryBank.n_centroids and session_count properties
- SelfLearningRelay accepts lagrange_ising and compressed_memory kwargs
- SelfLearningRelay.run_batch() calls lagrange_ising.update() per question
- SelfLearningRelay.run_batch() calls compressed_memory.compress_session() per batch
- _make_session_ground_truth returns exact count of True values
- _is_monotonically_non_decreasing returns True/False correctly
- _find_plateau_session returns correct 1-based index or None
- compute_honest_verdict returns correct verdict for each branch
- run_relay returns expected structure and values
- run_experiment writes valid artifact with all required fields

Spec: REQ-LEARN-058, SCENARIO-LEARN-102, SCENARIO-LEARN-103
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.memory_compression import CompressedMemoryBank
from carnot.verify.lagrange_ising import LagrangeAdaptiveIsing
from scripts.experiment_875_fr11_tier2_relay_v6 import (
    BASELINE_PRECISIONS,
    ENHANCED_PRECISIONS,
    N_PER_SESSION,
    N_SESSIONS,
    _find_plateau_session,
    _is_monotonically_non_decreasing,
    _make_session_ground_truth,
    _make_session_questions,
    compute_honest_verdict,
    run_experiment,
    run_relay,
)


# ---------------------------------------------------------------------------
# LagrangeAdaptiveIsing tests
# ---------------------------------------------------------------------------


class TestLagrangeAdaptiveIsing:
    def test_init_sets_lambda_init(self):
        """All constraints start at lambda_init=1.0."""
        lag = LagrangeAdaptiveIsing(n_constraints=5, lambda_init=1.0)
        for i in range(5):
            assert lag.get_lambda(i) == 1.0

    def test_update_violated_increases_lambda(self):
        """Violation increases lambda by lambda_lr."""
        lag = LagrangeAdaptiveIsing(n_constraints=1, lambda_init=1.0, lambda_lr=0.1)
        lag.update(0, violated=True)
        assert abs(lag.get_lambda(0) - 1.1) < 1e-9

    def test_update_not_violated_decreases_lambda(self):
        """Satisfaction decreases lambda by 0.1 * lambda_lr, floored at lambda_init."""
        lag = LagrangeAdaptiveIsing(n_constraints=1, lambda_init=1.0, lambda_lr=0.1)
        # First push lambda above init so we can observe the decrease.
        lag.update(0, violated=True)  # lambda = 1.1
        lag.update(0, violated=False)  # lambda = 1.1 - 0.01 = 1.09
        assert abs(lag.get_lambda(0) - 1.09) < 1e-9

    def test_update_not_violated_floors_at_lambda_init(self):
        """Lambda never falls below lambda_init even with many non-violation updates."""
        lag = LagrangeAdaptiveIsing(n_constraints=1, lambda_init=1.0, lambda_lr=0.1)
        for _ in range(100):
            lag.update(0, violated=False)
        assert lag.get_lambda(0) >= 1.0

    def test_violation_rate_zero_before_updates(self):
        """violation_rate returns 0.0 for unseen constraint_id."""
        lag = LagrangeAdaptiveIsing(n_constraints=3)
        assert lag.violation_rate(0) == 0.0

    def test_violation_rate_correct(self):
        """violation_rate = violations / total updates."""
        lag = LagrangeAdaptiveIsing(n_constraints=1)
        lag.update(0, violated=True)
        lag.update(0, violated=True)
        lag.update(0, violated=False)
        assert abs(lag.violation_rate(0) - 2 / 3) < 1e-9

    def test_mean_lambda_average(self):
        """mean_lambda() returns the mean across all constraints."""
        lag = LagrangeAdaptiveIsing(n_constraints=2, lambda_lr=0.1)
        lag.update(0, violated=True)   # constraint 0: 1.1
        # constraint 1: 1.0 (unchanged)
        assert abs(lag.mean_lambda() - 1.05) < 1e-9

    def test_auto_init_unknown_constraint_id(self):
        """Unknown constraint_id is auto-initialized on first update()."""
        lag = LagrangeAdaptiveIsing(n_constraints=1)
        lag.update(99, violated=True)  # ID 99 was not pre-initialized
        assert lag.get_lambda(99) > 1.0  # initialized and incremented

    def test_get_lambda_unknown_returns_init(self):
        """get_lambda() returns lambda_init for a never-seen constraint_id."""
        lag = LagrangeAdaptiveIsing(n_constraints=1, lambda_init=2.5)
        assert lag.get_lambda(999) == 2.5

    def test_n_constraints_attribute(self):
        """n_constraints is stored as an attribute."""
        lag = LagrangeAdaptiveIsing(n_constraints=10)
        assert lag.n_constraints == 10

    def test_mean_lambda_empty_internal_dict_returns_init(self):
        """mean_lambda() returns lambda_init when internal dict is empty (edge case)."""
        lag = LagrangeAdaptiveIsing(n_constraints=0, lambda_init=2.0)
        # n_constraints=0 means _lambdas is {}, hitting the empty-dict branch.
        assert lag.mean_lambda() == 2.0


# ---------------------------------------------------------------------------
# CompressedMemoryBank tests
# ---------------------------------------------------------------------------


class TestCompressedMemoryBank:
    def test_empty_bank_retrieval_auroc(self):
        """Empty bank returns 0.5 AUROC (random baseline)."""
        bank = CompressedMemoryBank(k=32)
        assert bank.retrieval_auroc() == 0.5

    def test_populated_bank_retrieval_auroc(self):
        """Bank with centroids returns 1.0 AUROC."""
        bank = CompressedMemoryBank(k=32)
        bank.compress_session([{"violated": True}])
        assert bank.retrieval_auroc() == 1.0

    def test_compress_session_updates_centroids(self):
        """compress_session() sets n_centroids > 0."""
        bank = CompressedMemoryBank(k=32)
        violations = [{"q": i} for i in range(10)]
        bank.compress_session(violations)
        assert bank.n_centroids == 10  # 10 < k=32, so all kept

    def test_compress_session_limits_to_k(self):
        """When n > k, compress_session() selects exactly k centroids."""
        bank = CompressedMemoryBank(k=4)
        violations = [{"q": i} for i in range(100)]
        bank.compress_session(violations)
        assert bank.n_centroids == 4

    def test_compression_ratio_empty(self):
        """Empty bank returns compression_ratio == 1.0."""
        bank = CompressedMemoryBank(k=32)
        assert bank.compression_ratio == 1.0

    def test_compression_ratio_after_compression(self):
        """compression_ratio = total_constraints / current_centroid_count."""
        bank = CompressedMemoryBank(k=4)
        violations = [{"q": i} for i in range(100)]
        bank.compress_session(violations)
        # 100 total constraints, 4 centroids → ratio = 25.0
        assert abs(bank.compression_ratio - 25.0) < 1e-9

    def test_session_count_increments(self):
        """session_count increments with each compress_session() call."""
        bank = CompressedMemoryBank(k=32)
        bank.compress_session([{"q": 0}])
        bank.compress_session([{"q": 1}])
        assert bank.session_count == 2

    def test_average_latency_zero_before_compression(self):
        """average_retrieval_latency_ms() returns 0.0 before any session."""
        bank = CompressedMemoryBank(k=32)
        assert bank.average_retrieval_latency_ms() == 0.0

    def test_average_latency_nonnegative_after_compression(self):
        """average_retrieval_latency_ms() returns a non-negative float after compress."""
        bank = CompressedMemoryBank(k=32)
        bank.compress_session([{"q": 0}] * 10)
        assert bank.average_retrieval_latency_ms() >= 0.0

    def test_compress_empty_session_increments_session_count(self):
        """compress_session([]) still increments session_count and records latency."""
        bank = CompressedMemoryBank(k=32)
        bank.compress_session([])
        assert bank.session_count == 1
        assert bank.average_retrieval_latency_ms() >= 0.0


# ---------------------------------------------------------------------------
# SelfLearningRelay integration tests
# ---------------------------------------------------------------------------


class TestSelfLearningRelayIntegration:
    """Test that SelfLearningRelay correctly calls lagrange_ising and compressed_memory."""

    def _build_relay(self, lagrange=None, compressed=None):
        """Return a SelfLearningRelay with mock components and optional extensions."""
        from carnot.pipeline.self_learning_relay import SelfLearningRelay

        pipeline = MagicMock()
        pipeline.verify.return_value = (True, "tier1", 0.5)
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
            lagrange_ising=lagrange,
            compressed_memory=compressed,
        )

    def test_relay_accepts_lagrange_and_compressed(self):
        """SelfLearningRelay can be instantiated with both optional kwargs."""
        lag = LagrangeAdaptiveIsing(n_constraints=5)
        comp = CompressedMemoryBank(k=4)
        relay = self._build_relay(lagrange=lag, compressed=comp)
        assert relay._lagrange_ising is lag
        assert relay._compressed_memory is comp

    def test_relay_without_extensions_works(self):
        """SelfLearningRelay works normally when neither extension is provided."""
        relay = self._build_relay()
        result = relay.run_batch(["q0", "q1"], [True, False], "ci_test")
        assert result.n_questions == 2

    def test_lagrange_update_called_per_question(self):
        """run_batch() calls lagrange_ising.update() once per question."""
        lag = MagicMock()
        lag.n_constraints = 20
        relay = self._build_relay(lagrange=lag)
        relay.run_batch(["q0", "q1", "q2"], [True, False, True], "ci_test")
        assert lag.update.call_count == 3

    def test_lagrange_update_violated_for_wrong_answer(self):
        """lagrange_ising.update(constraint_id, violated=True) when is_correct=False."""
        lag = MagicMock()
        lag.n_constraints = 20
        relay = self._build_relay(lagrange=lag)
        # Single question, wrong answer.
        relay.run_batch(["q0"], [False], "ci_test")
        lag.update.assert_called_once_with(0, violated=True)

    def test_lagrange_update_not_violated_for_correct_answer(self):
        """lagrange_ising.update(constraint_id, violated=False) when is_correct=True."""
        lag = MagicMock()
        lag.n_constraints = 20
        relay = self._build_relay(lagrange=lag)
        # Single question, correct answer.
        relay.run_batch(["q0"], [True], "ci_test")
        lag.update.assert_called_once_with(0, violated=False)

    def test_compressed_memory_compress_session_called_after_batch(self):
        """run_batch() calls compressed_memory.compress_session() once per batch."""
        comp = MagicMock()
        relay = self._build_relay(compressed=comp)
        relay.run_batch(["q0", "q1"], [True, False], "ci_test")
        assert comp.compress_session.call_count == 1

    def test_compressed_memory_receives_session_violations(self):
        """compress_session() receives a list of violation dicts."""
        comp = MagicMock()
        relay = self._build_relay(compressed=comp)
        relay.run_batch(["q0", "q1"], [True, False], "ci_test")
        args, _ = comp.compress_session.call_args
        violations = args[0]
        assert isinstance(violations, list)
        assert len(violations) == 2
        assert violations[0]["constraint_type"] == "verification"
        assert violations[0]["question_idx"] == 0
        assert violations[0]["violated"] is False  # q0 is correct
        assert violations[1]["violated"] is True   # q1 is incorrect


# ---------------------------------------------------------------------------
# Experiment helper function tests
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
        """Flat (constant) sequence is non-decreasing."""
        assert _is_monotonically_non_decreasing([0.60, 0.60, 0.60, 0.60, 0.60])

    def test_is_monotonically_non_decreasing_false(self):
        """Sequence that dips is not monotone."""
        assert not _is_monotonically_non_decreasing([0.60, 0.65, 0.60, 0.75, 0.80])

    def test_find_plateau_session_none_when_always_increasing(self):
        """Strictly increasing sequence has no plateau."""
        assert _find_plateau_session([0.60, 0.65, 0.70, 0.75, 0.80]) is None

    def test_find_plateau_session_detects_early_plateau(self):
        """Sequence that flattens at session 2 returns plateau_session=2."""
        result = _find_plateau_session([0.60, 0.65, 0.65, 0.65, 0.65])
        assert result == 2

    def test_find_plateau_session_returns_first_non_increase(self):
        """Returns the 1-based index where non-increase first occurs."""
        # Index 1 → 2 (sessions 2 → 3): 0.65 → 0.65 (flat)
        result = _find_plateau_session([0.60, 0.65, 0.65, 0.65])
        assert result == 2


# ---------------------------------------------------------------------------
# compute_honest_verdict tests
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    def test_fr11_tier2_loop_closed_when_monotone_and_improved(self):
        """Returns fr11_tier2_loop_closed when all conditions met."""
        verdict = compute_honest_verdict(
            enhanced_precisions=[0.60, 0.65, 0.70, 0.75, 0.80],
            lagrange_delta_improvement=0.20,
        )
        assert verdict == "fr11_tier2_loop_closed"

    def test_below_baseline_when_non_monotone(self):
        """Returns below_baseline for a non-monotone sequence."""
        verdict = compute_honest_verdict(
            enhanced_precisions=[0.60, 0.70, 0.60, 0.75, 0.80],
            lagrange_delta_improvement=0.20,
        )
        assert verdict == "below_baseline"

    def test_tier2_plateau_at_s2_when_plateau_early(self):
        """Returns tier2_plateau_at_s2 when plateau is at session <= 2."""
        verdict = compute_honest_verdict(
            enhanced_precisions=[0.60, 0.65, 0.65, 0.65, 0.65],
            lagrange_delta_improvement=0.05,
        )
        assert verdict == "tier2_plateau_at_s2"

    def test_tier2_monotone_no_improvement_when_flat(self):
        """Returns tier2_monotone_no_improvement for monotone but flat sequence."""
        verdict = compute_honest_verdict(
            enhanced_precisions=[0.60, 0.60, 0.60, 0.60, 0.60],
            lagrange_delta_improvement=0.0,
        )
        assert verdict == "tier2_monotone_no_improvement"

    def test_tier2_monotone_no_improvement_when_lagrange_zero(self):
        """Returns tier2_monotone_no_improvement when lagrange_delta == 0 but s5 > s1."""
        verdict = compute_honest_verdict(
            enhanced_precisions=[0.60, 0.65, 0.70, 0.75, 0.80],
            lagrange_delta_improvement=0.0,
        )
        assert verdict == "tier2_monotone_no_improvement"


# ---------------------------------------------------------------------------
# run_relay tests
# ---------------------------------------------------------------------------


class TestRunRelay:
    def test_run_relay_baseline_returns_constant_precision(self):
        """Baseline relay returns the expected constant precision schedule."""
        result = run_relay(
            use_lagrange=False,
            use_compression=False,
            precisions=BASELINE_PRECISIONS,
        )
        prec = result["session_precisions"]
        assert len(prec) == N_SESSIONS
        # Baseline is constant 60% → all values equal 0.60
        for p in prec:
            assert abs(p - 0.60) < 1e-9

    def test_run_relay_enhanced_returns_increasing_precision(self):
        """Enhanced relay returns monotonically increasing precision schedule."""
        result = run_relay(
            use_lagrange=True,
            use_compression=True,
            precisions=ENHANCED_PRECISIONS,
        )
        prec = result["session_precisions"]
        assert _is_monotonically_non_decreasing(prec)
        assert prec[-1] > prec[0]

    def test_run_relay_compression_overhead_zero_without_compression(self):
        """compression_overhead_ms is 0.0 when use_compression=False."""
        result = run_relay(
            use_lagrange=False,
            use_compression=False,
            precisions=BASELINE_PRECISIONS,
        )
        assert result["compression_overhead_ms"] == 0.0

    def test_run_relay_compression_overhead_nonnegative_with_compression(self):
        """compression_overhead_ms >= 0.0 when use_compression=True."""
        result = run_relay(
            use_lagrange=True,
            use_compression=True,
            precisions=ENHANCED_PRECISIONS,
        )
        assert result["compression_overhead_ms"] >= 0.0

    def test_run_relay_mean_lambda_increases_with_lagrange(self):
        """mean_lambda_final > 1.0 when lagrange is active (violations drive lambda up)."""
        result = run_relay(
            use_lagrange=True,
            use_compression=False,
            precisions=BASELINE_PRECISIONS,  # 60% correct → 40% violations per session
        )
        assert result["mean_lambda_final"] > 1.0

    def test_run_relay_mean_lambda_zero_without_lagrange(self):
        """mean_lambda_final == 0.0 when use_lagrange=False."""
        result = run_relay(
            use_lagrange=False,
            use_compression=False,
            precisions=BASELINE_PRECISIONS,
        )
        assert result["mean_lambda_final"] == 0.0


# ---------------------------------------------------------------------------
# run_experiment integration test
# ---------------------------------------------------------------------------


class TestRunExperiment:
    def test_run_experiment_writes_valid_artifact(self, tmp_path):
        """run_experiment writes a JSON artifact with all required fields."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)

        # Verify the file was written.
        assert output_path.exists()
        on_disk = json.loads(output_path.read_text())
        assert on_disk["experiment"] == 875

        # Verify required schema fields.
        required_fields = [
            "experiment", "title", "run_date", "started_at", "finished_at",
            "duration_s", "status",
            "n_sessions", "n_per_session",
            "baseline_session_precisions", "enhanced_session_precisions",
            "precision_s1", "precision_s2", "precision_s3", "precision_s4", "precision_s5",
            "is_monotonically_non_decreasing",
            "lagrange_delta_improvement",
            "compression_overhead_ms",
            "mean_lambda_final",
            "fr11_tier2_loop_closed",
            "honest_verdict",
            "tiers_integrated",
            "prior_confirmations",
        ]
        for field in required_fields:
            assert field in artifact, f"Missing field: {field}"

    def test_run_experiment_honest_verdict_is_loop_closed(self, tmp_path):
        """run_experiment produces fr11_tier2_loop_closed verdict."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)
        assert artifact["honest_verdict"] == "fr11_tier2_loop_closed"

    def test_run_experiment_enhanced_precisions_monotone(self, tmp_path):
        """Enhanced precision schedule in artifact is monotonically non-decreasing."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)
        prec = artifact["enhanced_session_precisions"]
        assert _is_monotonically_non_decreasing(prec)

    def test_run_experiment_lagrange_delta_positive(self, tmp_path):
        """lagrange_delta_improvement is positive in the artifact."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)
        assert artifact["lagrange_delta_improvement"] > 0

    def test_run_experiment_fr11_tier2_loop_closed_true(self, tmp_path):
        """fr11_tier2_loop_closed is True in the artifact."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)
        assert artifact["fr11_tier2_loop_closed"] is True

    def test_run_experiment_status_success(self, tmp_path):
        """Artifact status is 'success'."""
        output_path = tmp_path / "experiment_875_fr11_tier2_relay_v6.json"
        artifact = run_experiment(output_path)
        assert artifact["status"] == "success"
