"""Tests for Exp 318 — Four-Tier Continuous Self-Learning Relay Benchmark.

Covers:
- RelayBatchResult: batch_id, accuracy, n_questions, tiers_active, constraint_delta
- tiers_active list for each batch: ["tier1"] / ["tier1","tier2"] / ["tier1","tier2","tier3","z3"]
- RelayArtifact: batch1, batch2, batch3, improvement_1to2, improvement_1to3
- improvement computations: signed floats, never clamped
- constraint_delta: n_constraints_after - n_constraints_before per batch
- honest: any batch can have negative improvement; no fabrication
- artifact schema: experiment=318, schema="carnot.self_learning_relay.v1"
- BATCH_SIZE is exactly 33 per relay design
- simulate_gsm8k_questions produces n deterministic questions
- run_relay_batch: returns RelayBatchResult with correct fields
- build_relay_artifact: produces complete JSON-serializable artifact
- improvement_1to3 = batch3_accuracy - batch1_accuracy (signed)
- improvement_1to2 = batch2_accuracy - batch1_accuracy (signed)

Spec: REQ-LEARN-013, SCENARIO-LEARN-021, SCENARIO-LEARN-022
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from scripts.experiment_318_self_learning_relay import (
    BATCH_SIZE,
    EXPERIMENT,
    RelayBatchResult,
    build_relay_artifact,
    compute_relay_improvement,
    run_relay_batch,
    simulate_gsm8k_questions,
)


# ---------------------------------------------------------------------------
# Constants
# REQ-LEARN-013
# ---------------------------------------------------------------------------


class TestConstants:
    """Exp 318 constants must match relay design."""

    def test_experiment_id(self) -> None:
        """EXPERIMENT constant is 318."""
        assert EXPERIMENT == 318

    def test_batch_size(self) -> None:
        """BATCH_SIZE is exactly 33 per three-batch relay design."""
        assert BATCH_SIZE == 33


# ---------------------------------------------------------------------------
# RelayBatchResult schema
# REQ-LEARN-013, SCENARIO-LEARN-021
# ---------------------------------------------------------------------------


def _make_relay_batch(
    batch_id: int,
    n_correct: int,
    tiers_active: list[str],
    constraint_delta: int = 0,
) -> RelayBatchResult:
    """Helper: build a RelayBatchResult with n_correct correct answers."""
    # Build per-question results: first n_correct are correct
    per_question = [
        {"question_id": f"q{i}", "correct": i < n_correct}
        for i in range(BATCH_SIZE)
    ]
    return RelayBatchResult(
        batch_id=batch_id,
        n_questions=BATCH_SIZE,
        n_correct=n_correct,
        tiers_active=tiers_active,
        constraint_delta=constraint_delta,
        per_question=per_question,
    )


class TestRelayBatchResult:
    """RelayBatchResult must carry all relay design fields and compute accuracy correctly."""

    def test_batch_id_field(self) -> None:
        """REQ-LEARN-013: batch_id field is present and correct."""
        r = _make_relay_batch(1, 20, ["tier1"])
        assert r.batch_id == 1

    def test_n_questions_field(self) -> None:
        """REQ-LEARN-013: n_questions is BATCH_SIZE (33)."""
        r = _make_relay_batch(1, 20, ["tier1"])
        assert r.n_questions == BATCH_SIZE

    def test_accuracy_computation(self) -> None:
        """accuracy = n_correct / n_questions."""
        r = _make_relay_batch(1, 22, ["tier1"])
        assert abs(r.accuracy - 22 / BATCH_SIZE) < 1e-9

    def test_accuracy_zero(self) -> None:
        """accuracy is 0.0 when n_correct=0."""
        r = _make_relay_batch(1, 0, ["tier1"])
        assert r.accuracy == pytest.approx(0.0)

    def test_accuracy_perfect(self) -> None:
        """accuracy is 1.0 when all questions correct."""
        r = _make_relay_batch(1, BATCH_SIZE, ["tier1"])
        assert r.accuracy == pytest.approx(1.0)

    # SCENARIO-LEARN-021: tiers_active list per batch
    def test_batch1_tiers_active(self) -> None:
        """SCENARIO-LEARN-021: Batch 1 has tiers_active=['tier1']."""
        r = _make_relay_batch(1, 20, ["tier1"])
        assert r.tiers_active == ["tier1"]

    def test_batch2_tiers_active(self) -> None:
        """SCENARIO-LEARN-021: Batch 2 has tiers_active=['tier1','tier2']."""
        r = _make_relay_batch(2, 22, ["tier1", "tier2"])
        assert r.tiers_active == ["tier1", "tier2"]

    def test_batch3_tiers_active(self) -> None:
        """SCENARIO-LEARN-021: Batch 3 has tiers_active=['tier1','tier2','tier3','z3']."""
        r = _make_relay_batch(3, 24, ["tier1", "tier2", "tier3", "z3"])
        assert r.tiers_active == ["tier1", "tier2", "tier3", "z3"]

    def test_constraint_delta_field(self) -> None:
        """REQ-LEARN-013: constraint_delta = n_constraints_after - n_constraints_before."""
        r = _make_relay_batch(2, 22, ["tier1", "tier2"], constraint_delta=3)
        assert r.constraint_delta == 3

    def test_constraint_delta_zero(self) -> None:
        """REQ-LEARN-013: constraint_delta=0 is valid (no new constraints added)."""
        r = _make_relay_batch(1, 20, ["tier1"], constraint_delta=0)
        assert r.constraint_delta == 0

    def test_to_dict_has_all_fields(self) -> None:
        """RelayBatchResult.to_dict() emits all required schema fields."""
        r = _make_relay_batch(1, 20, ["tier1"])
        d = r.to_dict()
        assert "batch_id" in d
        assert "accuracy" in d
        assert "n_questions" in d
        assert "tiers_active" in d
        assert "constraint_delta" in d

    def test_to_dict_accuracy_matches_computation(self) -> None:
        """to_dict() accuracy matches computed accuracy property."""
        r = _make_relay_batch(2, 25, ["tier1", "tier2"])
        d = r.to_dict()
        assert abs(d["accuracy"] - 25 / BATCH_SIZE) < 1e-9

    def test_to_dict_is_json_serializable(self) -> None:
        """to_dict() output is JSON-serializable."""
        r = _make_relay_batch(3, 28, ["tier1", "tier2", "tier3", "z3"])
        json.dumps(r.to_dict())

    def test_to_dict_tiers_active_list(self) -> None:
        """to_dict() tiers_active is a list of strings."""
        r = _make_relay_batch(3, 28, ["tier1", "tier2", "tier3", "z3"])
        d = r.to_dict()
        assert isinstance(d["tiers_active"], list)
        assert all(isinstance(t, str) for t in d["tiers_active"])


# ---------------------------------------------------------------------------
# compute_relay_improvement
# REQ-LEARN-013, SCENARIO-LEARN-022
# ---------------------------------------------------------------------------


class TestComputeRelayImprovement:
    """compute_relay_improvement returns honest signed delta — never clamped."""

    def test_positive_improvement(self) -> None:
        """Delta is positive when batch_n > batch1."""
        assert compute_relay_improvement(0.6, 0.7) == pytest.approx(0.1)

    def test_negative_improvement_allowed(self) -> None:
        """SCENARIO-LEARN-022: negative improvement is reported honestly, not clamped."""
        delta = compute_relay_improvement(0.7, 0.6)
        assert delta < 0.0
        assert abs(delta - (-0.1)) < 1e-9

    def test_zero_improvement(self) -> None:
        """Delta is 0.0 when batch accuracies are equal."""
        assert compute_relay_improvement(0.5, 0.5) == pytest.approx(0.0)

    def test_exact_formula(self) -> None:
        """SCENARIO-LEARN-022: improvement = batch_n_accuracy - batch1_accuracy exactly."""
        batch1 = 0.60606060606  # 20/33
        batch3 = 0.72727272727  # 24/33
        assert compute_relay_improvement(batch1, batch3) == pytest.approx(batch3 - batch1)

    def test_full_range_positive(self) -> None:
        """Delta can be +1.0 (all wrong in batch1, all correct in batch_n)."""
        assert compute_relay_improvement(0.0, 1.0) == pytest.approx(1.0)

    def test_full_range_negative(self) -> None:
        """Delta can be -1.0 (all correct in batch1, all wrong in batch_n)."""
        assert compute_relay_improvement(1.0, 0.0) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# simulate_gsm8k_questions
# REQ-LEARN-013
# ---------------------------------------------------------------------------


class TestSimulateGsm8kQuestions:
    """simulate_gsm8k_questions produces deterministic synthetic GSM8K-style questions."""

    def test_produces_n_questions(self) -> None:
        """Produces exactly n questions."""
        questions = simulate_gsm8k_questions(n=99, seed=318)
        assert len(questions) == 99

    def test_produces_33_questions(self) -> None:
        """Produces exactly 33 questions for a single relay batch."""
        questions = simulate_gsm8k_questions(n=33, seed=0)
        assert len(questions) == 33

    def test_each_question_has_required_fields(self) -> None:
        """Each question dict has question, answer, correct_answer fields."""
        questions = simulate_gsm8k_questions(n=5, seed=0)
        for q in questions:
            assert "question" in q
            assert "answer" in q
            assert "correct_answer" in q

    def test_each_question_has_question_id(self) -> None:
        """Each question dict has a question_id field."""
        questions = simulate_gsm8k_questions(n=5, seed=0)
        for q in questions:
            assert "question_id" in q

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical questions."""
        q1 = simulate_gsm8k_questions(n=10, seed=318)
        q2 = simulate_gsm8k_questions(n=10, seed=318)
        assert q1 == q2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different questions."""
        q1 = simulate_gsm8k_questions(n=10, seed=318)
        q2 = simulate_gsm8k_questions(n=10, seed=999)
        assert q1 != q2


# ---------------------------------------------------------------------------
# run_relay_batch
# REQ-LEARN-013
# ---------------------------------------------------------------------------


class TestRunRelayBatch:
    """run_relay_batch returns a RelayBatchResult with exactly BATCH_SIZE records."""

    def _make_questions(self, n: int = BATCH_SIZE) -> list[dict[str, Any]]:
        return simulate_gsm8k_questions(n=n, seed=318)

    def test_returns_relay_batch_result(self) -> None:
        """run_relay_batch returns a RelayBatchResult."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=1,
            tiers_active=["tier1"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert isinstance(result, RelayBatchResult)

    def test_batch_id_propagated(self) -> None:
        """batch_id is propagated to RelayBatchResult."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=2,
            tiers_active=["tier1", "tier2"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert result.batch_id == 2

    def test_tiers_active_propagated(self) -> None:
        """tiers_active is propagated to RelayBatchResult."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=3,
            tiers_active=["tier1", "tier2", "tier3", "z3"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert result.tiers_active == ["tier1", "tier2", "tier3", "z3"]

    def test_n_questions_matches_input(self) -> None:
        """n_questions in result matches len(questions)."""
        questions = self._make_questions(BATCH_SIZE)
        result = run_relay_batch(
            questions=questions,
            batch_id=1,
            tiers_active=["tier1"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert result.n_questions == BATCH_SIZE

    def test_accuracy_in_range(self) -> None:
        """accuracy is in [0.0, 1.0]."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=1,
            tiers_active=["tier1"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert 0.0 <= result.accuracy <= 1.0

    def test_per_question_length_matches_batch(self) -> None:
        """per_question has exactly BATCH_SIZE entries."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=1,
            tiers_active=["tier1"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert len(result.per_question) == BATCH_SIZE

    def test_constraint_delta_zero_in_batch1(self) -> None:
        """constraint_delta=0 in batch 1 (warmup — no constraint addition)."""
        questions = self._make_questions()
        result = run_relay_batch(
            questions=questions,
            batch_id=1,
            tiers_active=["tier1"],
            pipeline=None,
            jepa_gate=None,
            z3_repair=None,
        )
        assert result.constraint_delta == 0


# ---------------------------------------------------------------------------
# build_relay_artifact
# REQ-LEARN-013, SCENARIO-LEARN-022
# ---------------------------------------------------------------------------


def _make_dummy_relay_batch(
    batch_id: int,
    n_correct: int,
    tiers_active: list[str],
    constraint_delta: int = 0,
) -> RelayBatchResult:
    per_question = [
        {"question_id": f"q{i}", "correct": i < n_correct}
        for i in range(BATCH_SIZE)
    ]
    return RelayBatchResult(
        batch_id=batch_id,
        n_questions=BATCH_SIZE,
        n_correct=n_correct,
        tiers_active=tiers_active,
        constraint_delta=constraint_delta,
        per_question=per_question,
    )


class TestBuildRelayArtifact:
    """build_relay_artifact must produce a complete, honest relay artifact."""

    def _make_artifact(
        self,
        b1_correct: int = 20,
        b2_correct: int = 22,
        b3_correct: int = 25,
    ) -> dict[str, Any]:
        batch1 = _make_dummy_relay_batch(1, b1_correct, ["tier1"])
        batch2 = _make_dummy_relay_batch(2, b2_correct, ["tier1", "tier2"])
        batch3 = _make_dummy_relay_batch(3, b3_correct, ["tier1", "tier2", "tier3", "z3"])
        return build_relay_artifact(
            batch1=batch1,
            batch2=batch2,
            batch3=batch3,
            inference_mode="simulated",
            jepa_skip_rate=0.45,
            z3_sat_rate=0.60,
        )

    def test_experiment_field_is_318(self) -> None:
        """Artifact has experiment=318."""
        a = self._make_artifact()
        assert a["experiment"] == 318

    def test_schema_field(self) -> None:
        """Artifact has schema='carnot.self_learning_relay.v1'."""
        a = self._make_artifact()
        assert a["schema"] == "carnot.self_learning_relay.v1"

    def test_has_run_date(self) -> None:
        """Artifact has run_date in YYYYMMDD format."""
        a = self._make_artifact()
        assert "run_date" in a
        assert len(a["run_date"]) == 8
        assert a["run_date"].isdigit()

    def test_has_inference_mode(self) -> None:
        """Artifact has inference_mode field."""
        a = self._make_artifact()
        assert a["inference_mode"] in ("live_gpu", "simulated")

    def test_has_batch1_accuracy(self) -> None:
        """Artifact has batch1_accuracy."""
        a = self._make_artifact(b1_correct=20)
        assert "batch1_accuracy" in a
        assert abs(a["batch1_accuracy"] - 20 / BATCH_SIZE) < 1e-6

    def test_has_batch2_accuracy(self) -> None:
        """Artifact has batch2_accuracy."""
        a = self._make_artifact(b2_correct=22)
        assert "batch2_accuracy" in a
        assert abs(a["batch2_accuracy"] - 22 / BATCH_SIZE) < 1e-6

    def test_has_batch3_accuracy(self) -> None:
        """Artifact has batch3_accuracy."""
        a = self._make_artifact(b3_correct=25)
        assert "batch3_accuracy" in a
        assert abs(a["batch3_accuracy"] - 25 / BATCH_SIZE) < 1e-6

    def test_improvement_1to2_positive(self) -> None:
        """improvement_1to2 > 0 when batch2 better than batch1."""
        a = self._make_artifact(b1_correct=20, b2_correct=25)
        assert a["improvement_1to2"] > 0.0

    def test_improvement_1to3_positive(self) -> None:
        """improvement_1to3 > 0 when batch3 better than batch1."""
        a = self._make_artifact(b1_correct=20, b3_correct=28)
        assert a["improvement_1to3"] > 0.0

    def test_improvement_1to2_negative_allowed(self) -> None:
        """SCENARIO-LEARN-022: negative improvement_1to2 is reported, not clamped."""
        a = self._make_artifact(b1_correct=28, b2_correct=20)
        assert a["improvement_1to2"] < 0.0

    def test_improvement_1to3_negative_allowed(self) -> None:
        """SCENARIO-LEARN-022: negative improvement_1to3 is reported, not clamped."""
        a = self._make_artifact(b1_correct=30, b3_correct=20)
        assert a["improvement_1to3"] < 0.0

    def test_improvement_1to3_formula(self) -> None:
        """SCENARIO-LEARN-022: improvement_1to3 = batch3_accuracy - batch1_accuracy exactly."""
        a = self._make_artifact(b1_correct=20, b3_correct=25)
        expected = 25 / BATCH_SIZE - 20 / BATCH_SIZE
        assert abs(a["improvement_1to3"] - expected) < 1e-6

    def test_improvement_1to2_formula(self) -> None:
        """improvement_1to2 = batch2_accuracy - batch1_accuracy exactly."""
        a = self._make_artifact(b1_correct=20, b2_correct=24)
        expected = 24 / BATCH_SIZE - 20 / BATCH_SIZE
        assert abs(a["improvement_1to2"] - expected) < 1e-6

    def test_has_jepa_skip_rate(self) -> None:
        """Artifact has jepa_skip_rate field."""
        a = self._make_artifact()
        assert "jepa_skip_rate" in a
        assert a["jepa_skip_rate"] == pytest.approx(0.45)

    def test_has_z3_sat_rate(self) -> None:
        """Artifact has z3_sat_rate field."""
        a = self._make_artifact()
        assert "z3_sat_rate" in a
        assert a["z3_sat_rate"] == pytest.approx(0.60)

    def test_has_batch1_nested(self) -> None:
        """Artifact has 'batch1' nested dict with tiers_active."""
        a = self._make_artifact()
        assert "batch1" in a
        assert a["batch1"]["tiers_active"] == ["tier1"]

    def test_has_batch2_nested(self) -> None:
        """Artifact has 'batch2' nested dict with tiers_active."""
        a = self._make_artifact()
        assert "batch2" in a
        assert a["batch2"]["tiers_active"] == ["tier1", "tier2"]

    def test_has_batch3_nested(self) -> None:
        """Artifact has 'batch3' nested dict with tiers_active."""
        a = self._make_artifact()
        assert "batch3" in a
        assert a["batch3"]["tiers_active"] == ["tier1", "tier2", "tier3", "z3"]

    def test_artifact_is_json_serializable(self) -> None:
        """build_relay_artifact output is JSON-serializable."""
        a = self._make_artifact()
        json.dumps(a)

    def test_has_title(self) -> None:
        """Artifact has title field."""
        a = self._make_artifact()
        assert "title" in a
        assert len(a["title"]) > 0

    def test_batch_n_questions_all_33(self) -> None:
        """All three batch nested dicts report n_questions=33."""
        a = self._make_artifact()
        assert a["batch1"]["n_questions"] == BATCH_SIZE
        assert a["batch2"]["n_questions"] == BATCH_SIZE
        assert a["batch3"]["n_questions"] == BATCH_SIZE

    def test_batch3_has_constraint_delta(self) -> None:
        """batch3 nested dict has constraint_delta field."""
        a = self._make_artifact()
        assert "constraint_delta" in a["batch3"]

    def test_constraint_delta_in_batch2(self) -> None:
        """batch2 nested dict has constraint_delta field."""
        batch1 = _make_dummy_relay_batch(1, 20, ["tier1"])
        batch2 = _make_dummy_relay_batch(2, 22, ["tier1", "tier2"], constraint_delta=2)
        batch3 = _make_dummy_relay_batch(3, 25, ["tier1", "tier2", "tier3", "z3"])
        a = build_relay_artifact(
            batch1=batch1,
            batch2=batch2,
            batch3=batch3,
            inference_mode="simulated",
            jepa_skip_rate=0.4,
            z3_sat_rate=0.5,
        )
        assert a["batch2"]["constraint_delta"] == 2
