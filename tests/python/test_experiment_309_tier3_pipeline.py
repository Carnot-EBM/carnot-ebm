"""Tests for Exp 309 — Tier 3 continuous self-learning pipeline.

Covers:
- ThresholdAdapter: adapt() increases threshold when fp_rate > fp_threshold
- ThresholdAdapter: adapt() decreases threshold when skip_rate < min_skip
- ThresholdAdapter: adapt() returns unchanged threshold when both conditions OK
- ThresholdAdapter: adapt() clamps result to [0.1, 0.9]
- ThresholdAdapter: threshold attribute is updated after adapt()
- GateDecisionRecord: schema fields present
- Tier3BatchResult: accuracy property, gate_decisions list
- Tier3BatchResult: skip_rate property
- compute_latency_reduction: correct signed fraction
- compute_latency_reduction: returns negative when gated batch is slower (honest)
- build_artifact_309: contains threshold_history list
- build_artifact_309: contains all REQUIRED_RESULT_FIELDS
- build_artifact_309: negative improvement_delta is preserved (honest)
- simulate_gsm8k_questions: returns n items with question + correct_answer
- run_baseline_batch: records accuracy and latency
- run_tier3_batch: runs gate + adapter every 10, records threshold_history

Spec: REQ-LEARN-012, SCENARIO-LEARN-019, SCENARIO-LEARN-020
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.experiment_309_tier3_pipeline import (
    BATCH_SIZE,
    EXPERIMENT,
    GateDecisionRecord,
    ThresholdAdapter,
    Tier3BatchResult,
    build_artifact_309,
    compute_latency_reduction,
    simulate_gsm8k_questions,
    run_baseline_batch,
    run_tier3_batch,
)


# ---------------------------------------------------------------------------
# Constants
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestConstants:
    """Exp 309 constants must match design."""

    def test_experiment_id(self) -> None:
        """EXPERIMENT constant is 309."""
        assert EXPERIMENT == 309

    def test_batch_size(self) -> None:
        """BATCH_SIZE is exactly 50."""
        assert BATCH_SIZE == 50


# ---------------------------------------------------------------------------
# ThresholdAdapter
# REQ-LEARN-012, SCENARIO-LEARN-019, SCENARIO-LEARN-020
# ---------------------------------------------------------------------------


class TestThresholdAdapter:
    """ThresholdAdapter must adapt the gate threshold based on FP and skip rates."""

    def test_initial_threshold_stored(self) -> None:
        """Initial threshold is stored as adapter.threshold."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        assert adapter.threshold == 0.5

    def test_adapt_increases_when_fp_rate_exceeds_limit(self) -> None:
        """SCENARIO-LEARN-019: adapt() increases threshold by 0.05 when fp_rate > fp_threshold."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        new_threshold = adapter.adapt(fp_rate=0.10, skip_rate=0.30)
        assert abs(new_threshold - 0.55) < 1e-9

    def test_adapt_updates_attribute_on_increase(self) -> None:
        """SCENARIO-LEARN-019: adapter.threshold is updated after increase."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        adapter.adapt(fp_rate=0.10, skip_rate=0.30)
        assert abs(adapter.threshold - 0.55) < 1e-9

    def test_adapt_decreases_when_skip_rate_below_minimum(self) -> None:
        """SCENARIO-LEARN-020: adapt() decreases threshold by 0.05 when skip_rate < min_skip and fp OK."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        new_threshold = adapter.adapt(fp_rate=0.02, skip_rate=0.05)
        assert abs(new_threshold - 0.45) < 1e-9

    def test_adapt_updates_attribute_on_decrease(self) -> None:
        """SCENARIO-LEARN-020: adapter.threshold is updated after decrease."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        adapter.adapt(fp_rate=0.02, skip_rate=0.05)
        assert abs(adapter.threshold - 0.45) < 1e-9

    def test_adapt_no_change_when_both_conditions_ok(self) -> None:
        """adapt() leaves threshold unchanged when fp_rate <= fp_threshold AND skip_rate >= min_skip."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        new_threshold = adapter.adapt(fp_rate=0.03, skip_rate=0.20)
        assert abs(new_threshold - 0.5) < 1e-9
        assert abs(adapter.threshold - 0.5) < 1e-9

    def test_adapt_clamps_to_max_09(self) -> None:
        """adapt() clamps result to 0.9 when threshold would exceed it."""
        adapter = ThresholdAdapter(initial=0.88, fp_threshold=0.05, min_skip=0.10)
        # fp_rate > fp_threshold → would increase to 0.93, clamped to 0.9
        new_threshold = adapter.adapt(fp_rate=0.10, skip_rate=0.30)
        assert abs(new_threshold - 0.9) < 1e-9

    def test_adapt_clamps_to_min_01(self) -> None:
        """adapt() clamps result to 0.1 when threshold would fall below it."""
        adapter = ThresholdAdapter(initial=0.12, fp_threshold=0.05, min_skip=0.10)
        # skip_rate < min_skip → would decrease to 0.07, clamped to 0.1
        new_threshold = adapter.adapt(fp_rate=0.02, skip_rate=0.05)
        assert abs(new_threshold - 0.1) < 1e-9

    def test_adapt_fp_priority_over_skip(self) -> None:
        """FP rate check takes priority: if fp_rate > fp_threshold, always increase even if skip_rate is also low."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        # Both conditions hold: fp_rate > fp_threshold AND skip_rate < min_skip
        # FP rate takes priority → increase
        new_threshold = adapter.adapt(fp_rate=0.10, skip_rate=0.05)
        assert abs(new_threshold - 0.55) < 1e-9

    def test_adapt_returns_float(self) -> None:
        """adapt() always returns a float."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        result = adapter.adapt(fp_rate=0.0, skip_rate=0.5)
        assert isinstance(result, float)

    def test_multiple_adapt_calls_accumulate(self) -> None:
        """Multiple adapt() calls accumulate changes on adapter.threshold."""
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        # First: decrease (fp OK, skip low)
        adapter.adapt(fp_rate=0.02, skip_rate=0.05)
        assert abs(adapter.threshold - 0.45) < 1e-9
        # Second: decrease again
        adapter.adapt(fp_rate=0.02, skip_rate=0.05)
        assert abs(adapter.threshold - 0.40) < 1e-9


# ---------------------------------------------------------------------------
# GateDecisionRecord
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestGateDecisionRecord:
    """GateDecisionRecord must capture per-question gate state."""

    def _make_record(self) -> GateDecisionRecord:
        return GateDecisionRecord(
            question_id="q0",
            correct=True,
            gate_decision="skip",
            gate_energy=0.3,
            ising_ran=False,
            violation_detected=False,
        )

    def test_has_question_id(self) -> None:
        """Record has question_id field."""
        r = self._make_record()
        assert r.question_id == "q0"

    def test_has_correct(self) -> None:
        """Record has correct field."""
        r = self._make_record()
        assert r.correct is True

    def test_has_gate_decision(self) -> None:
        """Record has gate_decision field."""
        r = self._make_record()
        assert r.gate_decision == "skip"

    def test_has_gate_energy(self) -> None:
        """Record has gate_energy field."""
        r = self._make_record()
        assert r.gate_energy == 0.3

    def test_has_ising_ran(self) -> None:
        """Record has ising_ran field."""
        r = self._make_record()
        assert r.ising_ran is False

    def test_has_violation_detected(self) -> None:
        """Record has violation_detected field."""
        r = self._make_record()
        assert r.violation_detected is False

    def test_to_dict_roundtrip(self) -> None:
        """to_dict() returns a JSON-serialisable dict with all fields."""
        r = self._make_record()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["question_id"] == "q0"
        assert d["correct"] is True
        assert d["gate_decision"] == "skip"
        # Verify it is JSON-serialisable
        json.dumps(d)


# ---------------------------------------------------------------------------
# Tier3BatchResult
# REQ-LEARN-012
# ---------------------------------------------------------------------------


def _make_gate_records(n: int = 50, *, n_skipped: int = 15, n_correct: int = 40) -> list[GateDecisionRecord]:
    """Build n GateDecisionRecords with specified skip and correct counts."""
    records = []
    for i in range(n):
        skipped = i < n_skipped
        correct = i < n_correct
        records.append(GateDecisionRecord(
            question_id=f"q{i}",
            correct=correct,
            gate_decision="skip" if skipped else "verify",
            gate_energy=0.3 if skipped else 0.7,
            ising_ran=not skipped,
            violation_detected=(not skipped and i % 3 == 0),
        ))
    return records


class TestTier3BatchResult:
    """Tier3BatchResult aggregates Tier 3 gate-annotated results."""

    def test_accuracy_property(self) -> None:
        """accuracy = n_correct / BATCH_SIZE."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        assert abs(result.accuracy - 40 / 50) < 1e-9

    def test_skip_rate_property(self) -> None:
        """skip_rate = n_skipped / BATCH_SIZE."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        assert abs(result.skip_rate - 15 / 50) < 1e-9

    def test_requires_exactly_batch_size_records(self) -> None:
        """Tier3BatchResult raises ValueError when records count != BATCH_SIZE."""
        records = _make_gate_records(49, n_skipped=10, n_correct=30)
        with pytest.raises(ValueError, match="50"):
            Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)

    def test_to_dict_has_accuracy(self) -> None:
        """to_dict() includes accuracy."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        d = result.to_dict()
        assert "accuracy" in d

    def test_to_dict_has_skip_rate(self) -> None:
        """to_dict() includes skip_rate."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        d = result.to_dict()
        assert "skip_rate" in d

    def test_to_dict_has_gate_decisions(self) -> None:
        """to_dict() includes per_question list."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        d = result.to_dict()
        assert "per_question" in d
        assert len(d["per_question"]) == 50

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict() is JSON-serialisable."""
        records = _make_gate_records(50, n_skipped=15, n_correct=40)
        result = Tier3BatchResult(records=records, batch_index=2, latency_s=5.0)
        json.dumps(result.to_dict())


# ---------------------------------------------------------------------------
# compute_latency_reduction
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestComputeLatencyReduction:
    """compute_latency_reduction must return honest signed fraction."""

    def test_positive_reduction(self) -> None:
        """Returns positive value when gated batch is faster."""
        reduction = compute_latency_reduction(baseline_s=10.0, gated_s=7.0)
        assert abs(reduction - 0.3) < 1e-9

    def test_zero_when_same(self) -> None:
        """Returns 0.0 when latencies are equal."""
        reduction = compute_latency_reduction(baseline_s=10.0, gated_s=10.0)
        assert abs(reduction - 0.0) < 1e-9

    def test_negative_when_gated_slower(self) -> None:
        """Returns negative value when gated batch is actually slower (honest reporting)."""
        reduction = compute_latency_reduction(baseline_s=5.0, gated_s=6.0)
        assert reduction < 0.0

    def test_formula_is_correct(self) -> None:
        """Reduction = (baseline - gated) / baseline."""
        reduction = compute_latency_reduction(baseline_s=8.0, gated_s=4.0)
        assert abs(reduction - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# build_artifact_309
# REQ-LEARN-012
# ---------------------------------------------------------------------------


def _make_baseline_result() -> Any:
    """Make a minimal baseline (no-gate) batch result dict."""
    return {
        "batch_index": 1,
        "n_questions": 50,
        "accuracy": 0.72,
        "latency_s": 10.0,
    }


def _make_tier3_result() -> Any:
    """Make a minimal Tier 3 batch result dict."""
    records = _make_gate_records(50, n_skipped=18, n_correct=38)
    return Tier3BatchResult(records=records, batch_index=2, latency_s=7.5)


class TestBuildArtifact309:
    """build_artifact_309 must produce a complete, honest artifact."""

    def _make_artifact(self) -> dict[str, Any]:
        baseline = _make_baseline_result()
        tier3 = _make_tier3_result()
        return build_artifact_309(
            baseline_batch=baseline,
            tier3_batch=tier3,
            threshold_history=[0.5, 0.45, 0.45, 0.50, 0.50],
            improvement_delta=tier3.accuracy - baseline["accuracy"],
            latency_reduction=compute_latency_reduction(baseline["latency_s"], tier3.latency_s),
            inference_mode="simulated",
        )

    def test_has_experiment_field(self) -> None:
        """Artifact has experiment=309."""
        a = self._make_artifact()
        assert a["experiment"] == EXPERIMENT

    def test_has_status(self) -> None:
        """Artifact has status field."""
        a = self._make_artifact()
        assert "status" in a

    def test_has_run_date(self) -> None:
        """Artifact has run_date field."""
        a = self._make_artifact()
        assert "run_date" in a
        assert len(a["run_date"]) == 8  # YYYYMMDD

    def test_has_threshold_history(self) -> None:
        """Artifact has threshold_history list."""
        a = self._make_artifact()
        assert "threshold_history" in a
        assert isinstance(a["threshold_history"], list)
        assert len(a["threshold_history"]) == 5

    def test_has_improvement_delta(self) -> None:
        """Artifact has improvement_delta field."""
        a = self._make_artifact()
        assert "improvement_delta" in a

    def test_has_latency_reduction(self) -> None:
        """Artifact has latency_reduction field."""
        a = self._make_artifact()
        assert "latency_reduction" in a

    def test_has_inference_mode(self) -> None:
        """Artifact has inference_mode field."""
        a = self._make_artifact()
        assert a["inference_mode"] in ("live_gpu", "simulated")

    def test_has_batch1_accuracy(self) -> None:
        """Artifact has batch1_accuracy from baseline."""
        a = self._make_artifact()
        assert "batch1_accuracy" in a
        assert abs(a["batch1_accuracy"] - 0.72) < 1e-9

    def test_has_batch2_accuracy(self) -> None:
        """Artifact has batch2_accuracy from tier3 batch."""
        a = self._make_artifact()
        assert "batch2_accuracy" in a

    def test_negative_improvement_delta_preserved(self) -> None:
        """Negative improvement_delta is never clamped or hidden (honest reporting)."""
        baseline = _make_baseline_result()
        # Make tier3 less accurate than baseline
        records = _make_gate_records(50, n_skipped=5, n_correct=30)  # 30/50 = 0.60 < 0.72
        tier3 = Tier3BatchResult(records=records, batch_index=2, latency_s=8.0)
        delta = tier3.accuracy - baseline["accuracy"]
        assert delta < 0.0
        a = build_artifact_309(
            baseline_batch=baseline,
            tier3_batch=tier3,
            threshold_history=[0.5],
            improvement_delta=delta,
            latency_reduction=compute_latency_reduction(baseline["latency_s"], tier3.latency_s),
            inference_mode="simulated",
        )
        assert a["improvement_delta"] < 0.0

    def test_artifact_is_json_serialisable(self) -> None:
        """build_artifact_309 output is JSON-serialisable."""
        a = self._make_artifact()
        json.dumps(a)


# ---------------------------------------------------------------------------
# simulate_gsm8k_questions
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestSimulateGsm8kQuestions:
    """simulate_gsm8k_questions must return n arithmetic questions."""

    def test_returns_n_items(self) -> None:
        """Returns exactly n questions."""
        questions = simulate_gsm8k_questions(50, seed=42)
        assert len(questions) == 50

    def test_each_has_question_field(self) -> None:
        """Each item has a 'question' string."""
        questions = simulate_gsm8k_questions(5, seed=1)
        for q in questions:
            assert "question" in q
            assert isinstance(q["question"], str)

    def test_each_has_correct_answer(self) -> None:
        """Each item has a numeric 'correct_answer'."""
        questions = simulate_gsm8k_questions(5, seed=1)
        for q in questions:
            assert "correct_answer" in q
            assert isinstance(q["correct_answer"], (int, float))

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces same questions."""
        a = simulate_gsm8k_questions(10, seed=99)
        b = simulate_gsm8k_questions(10, seed=99)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different questions."""
        a = simulate_gsm8k_questions(10, seed=1)
        b = simulate_gsm8k_questions(10, seed=2)
        assert a != b


# ---------------------------------------------------------------------------
# run_baseline_batch (no gate)
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestRunBaselineBatch:
    """run_baseline_batch must produce accuracy and latency without a gate."""

    def test_returns_accuracy_and_latency(self) -> None:
        """run_baseline_batch returns a dict with accuracy and latency_s."""
        questions = simulate_gsm8k_questions(50, seed=7)
        result = run_baseline_batch(questions, rng_seed=7)
        assert "accuracy" in result
        assert "latency_s" in result

    def test_accuracy_in_range(self) -> None:
        """Accuracy is in [0.0, 1.0]."""
        questions = simulate_gsm8k_questions(50, seed=7)
        result = run_baseline_batch(questions, rng_seed=7)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_latency_positive(self) -> None:
        """Latency is a positive number."""
        questions = simulate_gsm8k_questions(50, seed=7)
        result = run_baseline_batch(questions, rng_seed=7)
        assert result["latency_s"] > 0.0

    def test_returns_exactly_50_questions(self) -> None:
        """Baseline result records exactly 50 questions."""
        questions = simulate_gsm8k_questions(50, seed=7)
        result = run_baseline_batch(questions, rng_seed=7)
        assert result["n_questions"] == 50


# ---------------------------------------------------------------------------
# run_tier3_batch (with gate + adapter)
# REQ-LEARN-012
# ---------------------------------------------------------------------------


class TestRunTier3Batch:
    """run_tier3_batch must apply gate + ThresholdAdapter every 10 questions."""

    def _run(self) -> tuple[Tier3BatchResult, list[float]]:
        questions = simulate_gsm8k_questions(50, seed=13)
        adapter = ThresholdAdapter(initial=0.5, fp_threshold=0.05, min_skip=0.10)
        return run_tier3_batch(questions, adapter=adapter, rng_seed=13)

    def test_returns_tier3_batch_result(self) -> None:
        """run_tier3_batch returns (Tier3BatchResult, threshold_history)."""
        result, history = self._run()
        assert isinstance(result, Tier3BatchResult)

    def test_threshold_history_has_5_entries(self) -> None:
        """threshold_history has one entry per 10-question sub-batch (50 / 10 = 5)."""
        _, history = self._run()
        assert len(history) == 5

    def test_threshold_history_values_in_range(self) -> None:
        """All threshold_history values are in [0.1, 0.9]."""
        _, history = self._run()
        for t in history:
            assert 0.1 <= t <= 0.9

    def test_result_has_50_records(self) -> None:
        """Tier3BatchResult contains exactly 50 GateDecisionRecords."""
        result, _ = self._run()
        assert len(result.records) == 50

    def test_accuracy_in_range(self) -> None:
        """Accuracy is in [0.0, 1.0]."""
        result, _ = self._run()
        assert 0.0 <= result.accuracy <= 1.0

    def test_skip_rate_in_range(self) -> None:
        """skip_rate is in [0.0, 1.0]."""
        result, _ = self._run()
        assert 0.0 <= result.skip_rate <= 1.0

    def test_gate_decisions_are_skip_or_verify(self) -> None:
        """gate_decision for every record is either 'skip' or 'verify'."""
        result, _ = self._run()
        for rec in result.records:
            assert rec.gate_decision in ("skip", "verify")
