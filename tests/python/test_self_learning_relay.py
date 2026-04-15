"""Tests for self_learning_relay.py — three-tier self-learning relay.

Covers:
  - SelfLearningBatchResult dataclass fields and types
  - _compute_auc_roc edge cases (all-same labels, ties, perfect separation)
  - SelfLearningRelay construction and run_batch() mechanics
  - Tier 1: n_tier1_updates == n_questions per batch
  - Tier 2: n_tier2_templates_active rises after pattern threshold crossed
  - Tier 3: tier3_gate_auc is a float in [0, 1]
  - cumulative_accuracy correctly aggregates across multiple batches
  - learning_trajectory() returns a copy of accumulated results
  - compute_learning_improvement() edge cases (empty, <4 batches, improved/not)
  - build_relay_artifact() schema, honest_verdict logic, trajectory serialization
  - CI-safe: all tests run on CPU with stub pipeline and random-weight EORM

Spec: REQ-LEARN-026, REQ-LEARN-027
SCENARIO-LEARN-045 (Tier 1 updates per question)
SCENARIO-LEARN-046 (Tier 2 templates activate after threshold)
SCENARIO-LEARN-047 (compute_learning_improvement with improving trajectory)
"""

from __future__ import annotations

import jax.random as jr
import pytest

from carnot.models.eorm import EORMModel
from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary
from carnot.pipeline.self_learning_relay import (
    SelfLearningBatchResult,
    SelfLearningRelay,
    _compute_auc_roc,
    build_relay_artifact,
    compute_learning_improvement,
)
from carnot.pipeline.sink_probe import SinkProbe
from carnot.pipeline.three_tier_pipeline import ThreeTierPipeline


# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_eorm(seed: int = 42) -> EORMModel:
    """Build a tiny EORM model for CI (random weights, embed_dim=32)."""
    key = jr.PRNGKey(seed)
    return EORMModel(
        embed_dim=32,
        n_heads=2,
        n_layers=1,
        max_seq_len=64,
        vocab_size=256,
        key=key,
    )


def _ising_stub_correct(response: str, question: str) -> tuple[bool, float]:
    """Stub Ising pipeline that always verifies (returns True)."""
    return (True, 0.0)


def _ising_stub_wrong(response: str, question: str) -> tuple[bool, float]:
    """Stub Ising pipeline that always rejects (returns False)."""
    return (False, 2.0)


def _make_pipeline(
    ising_fn=_ising_stub_correct,
    *,
    sink_threshold: float = 0.3,
    eorm_threshold: float = 0.5,
    seed: int = 1,
) -> tuple[ThreeTierPipeline, EORMModel]:
    """Build a minimal ThreeTierPipeline with stub Ising for CI testing."""
    eorm = _make_eorm(seed)
    sink = SinkProbe(threshold=0.3)
    pipeline = ThreeTierPipeline(
        sink_probe=sink,
        eorm_model=eorm,
        ising_pipeline=ising_fn,
        sink_threshold=sink_threshold,
        eorm_threshold=eorm_threshold,
    )
    return pipeline, eorm


def _make_relay(
    ising_fn=_ising_stub_correct,
    min_frequency: int = 5,
    eorm_threshold: float = 0.5,
    seed: int = 42,
) -> SelfLearningRelay:
    """Build a SelfLearningRelay with all-stub components for CI testing."""
    pipeline, eorm = _make_pipeline(ising_fn=ising_fn, eorm_threshold=eorm_threshold, seed=seed + 1)
    relay_eorm = _make_eorm(seed)
    library = ConstraintTemplateLibrary()
    library.register_builtin_templates()
    # Override min_frequency for faster activation in tests.
    for tmpl in library._templates.values():
        tmpl.min_frequency = min_frequency
    tracker = PerModelFPTracker(min_observations=10)
    return SelfLearningRelay(
        pipeline=pipeline,
        template_library=library,
        fp_tracker=tracker,
        eorm_model=relay_eorm,
    )


def _make_batch(n: int, n_correct: int) -> tuple[list[str], list[bool]]:
    """Build a synthetic batch of n questions with n_correct True labels."""
    questions = [f"Q{i}: what is {i} + {i}?" for i in range(n)]
    ground_truth = [i < n_correct for i in range(n)]
    return questions, ground_truth


# ---------------------------------------------------------------------------
# SelfLearningBatchResult dataclass
# ---------------------------------------------------------------------------


class TestSelfLearningBatchResult:
    """REQ-LEARN-026-1: SelfLearningBatchResult has required fields."""

    def test_fields_present(self):
        """All seven required fields are accessible."""
        result = SelfLearningBatchResult(
            batch_id=0,
            n_questions=25,
            accuracy=0.6,
            n_tier1_updates=25,
            n_tier2_templates_active=0,
            tier3_gate_auc=0.5,
            cumulative_accuracy=0.6,
        )
        assert result.batch_id == 0
        assert result.n_questions == 25
        assert result.accuracy == pytest.approx(0.6)
        assert result.n_tier1_updates == 25
        assert result.n_tier2_templates_active == 0
        assert result.tier3_gate_auc == pytest.approx(0.5)
        assert result.cumulative_accuracy == pytest.approx(0.6)

    def test_fields_types(self):
        """Field types match the spec (int for counts, float for rates)."""
        result = SelfLearningBatchResult(
            batch_id=3,
            n_questions=25,
            accuracy=0.75,
            n_tier1_updates=25,
            n_tier2_templates_active=2,
            tier3_gate_auc=0.72,
            cumulative_accuracy=0.675,
        )
        assert isinstance(result.batch_id, int)
        assert isinstance(result.n_questions, int)
        assert isinstance(result.n_tier1_updates, int)
        assert isinstance(result.n_tier2_templates_active, int)
        assert isinstance(result.accuracy, float)
        assert isinstance(result.tier3_gate_auc, float)
        assert isinstance(result.cumulative_accuracy, float)


# ---------------------------------------------------------------------------
# _compute_auc_roc helper
# ---------------------------------------------------------------------------


class TestComputeAucRoc:
    """Internal AUC helper — edge cases and correctness."""

    def test_all_correct_labels_returns_half(self):
        """When all labels are True AUC is undefined → returns 0.5."""
        energies = [0.1, 0.2, 0.3]
        ground_truth = [True, True, True]
        auc = _compute_auc_roc(energies, ground_truth)
        assert auc == pytest.approx(0.5)

    def test_all_incorrect_labels_returns_half(self):
        """When all labels are False AUC is undefined → returns 0.5."""
        energies = [0.1, 0.2, 0.3]
        ground_truth = [False, False, False]
        auc = _compute_auc_roc(energies, ground_truth)
        assert auc == pytest.approx(0.5)

    def test_perfect_separation(self):
        """Perfect discrimination: low energy for all True → AUC = 1.0."""
        # True items get energy 0.1 (low), False items get energy 0.9 (high).
        energies = [0.1, 0.1, 0.9, 0.9]
        ground_truth = [True, True, False, False]
        auc = _compute_auc_roc(energies, ground_truth)
        assert auc == pytest.approx(1.0)

    def test_worst_case_separation(self):
        """Worst discrimination: high energy for True → AUC = 0.0."""
        energies = [0.9, 0.9, 0.1, 0.1]
        ground_truth = [True, True, False, False]
        auc = _compute_auc_roc(energies, ground_truth)
        assert auc == pytest.approx(0.0)

    def test_all_ties(self):
        """When all scores are identical each pair contributes 0.5 → AUC = 0.5."""
        energies = [0.5, 0.5, 0.5, 0.5]
        ground_truth = [True, True, False, False]
        auc = _compute_auc_roc(energies, ground_truth)
        assert auc == pytest.approx(0.5)

    def test_empty_inputs_returns_half(self):
        """Empty inputs → 0.5 (no positive AND no negative)."""
        auc = _compute_auc_roc([], [])
        assert auc == pytest.approx(0.5)

    def test_auc_in_range(self):
        """AUC is always in [0, 1] for any input."""
        import random

        rng = random.Random(7)
        energies = [rng.uniform(0, 1) for _ in range(25)]
        labels = [rng.random() < 0.6 for _ in range(25)]
        auc = _compute_auc_roc(energies, labels)
        assert 0.0 <= auc <= 1.0


# ---------------------------------------------------------------------------
# SelfLearningRelay construction
# ---------------------------------------------------------------------------


class TestSelfLearningRelayConstruction:
    """REQ-LEARN-026-2: Constructor accepts all four required components."""

    def test_can_instantiate(self):
        """SelfLearningRelay instantiates without error."""
        relay = _make_relay()
        assert relay is not None

    def test_initial_trajectory_empty(self):
        """Before any run_batch() calls, learning_trajectory() is empty."""
        relay = _make_relay()
        assert relay.learning_trajectory() == []

    def test_cumulative_counters_start_zero(self):
        """Internal cumulative counters start at zero."""
        relay = _make_relay()
        assert relay._total_correct == 0
        assert relay._total_questions == 0


# ---------------------------------------------------------------------------
# SelfLearningRelay.run_batch — Tier 1
# ---------------------------------------------------------------------------


class TestRunBatchTier1:
    """REQ-LEARN-026-3 (Tier 1): PerModelFPTracker updated once per question."""

    def test_n_tier1_updates_equals_n_questions(self):
        """Tier 1 fires exactly once per question in the batch.

        Spec: SCENARIO-LEARN-045
        """
        relay = _make_relay()
        questions, ground_truth = _make_batch(25, 15)  # 15 correct, 10 wrong
        result = relay.run_batch(questions, ground_truth, "test_model")
        # One fp_tracker.update() call per question.
        assert result.n_tier1_updates == 25

    def test_n_tier1_updates_single_question(self):
        """Single-question batch still produces exactly 1 Tier 1 update."""
        relay = _make_relay()
        result = relay.run_batch(["Q0?"], [True], "test_model")
        assert result.n_tier1_updates == 1

    def test_fp_tracker_has_observations_after_batch(self):
        """After run_batch(), the tracker has recorded observations for model_id."""
        relay = _make_relay()
        questions, ground_truth = _make_batch(10, 5)
        relay.run_batch(questions, ground_truth, "my_model")
        # There should be at least one (model_id, constraint_type) entry.
        assert any(mid == "my_model" for (mid, _) in relay._fp_tracker._stats)


# ---------------------------------------------------------------------------
# SelfLearningRelay.run_batch — accuracy and n_questions
# ---------------------------------------------------------------------------


class TestRunBatchAccuracy:
    """REQ-LEARN-026-3: accuracy = n_correct / n_questions."""

    def test_accuracy_all_correct(self):
        """When all ground_truth is True, accuracy is 1.0."""
        relay = _make_relay()
        questions, ground_truth = _make_batch(10, 10)
        result = relay.run_batch(questions, ground_truth, "m")
        assert result.accuracy == pytest.approx(1.0)
        assert result.n_questions == 10

    def test_accuracy_none_correct(self):
        """When all ground_truth is False, accuracy is 0.0."""
        relay = _make_relay()
        questions, ground_truth = _make_batch(10, 0)
        result = relay.run_batch(questions, ground_truth, "m")
        assert result.accuracy == pytest.approx(0.0)

    def test_accuracy_60_percent(self):
        """15 correct out of 25 → accuracy = 0.6.

        Spec: SCENARIO-LEARN-045
        """
        relay = _make_relay()
        questions, ground_truth = _make_batch(25, 15)
        result = relay.run_batch(questions, ground_truth, "m")
        assert result.accuracy == pytest.approx(0.6)
        assert result.n_questions == 25

    def test_batch_id_increments(self):
        """Each run_batch() increments the batch_id by 1."""
        relay = _make_relay()
        q, g = _make_batch(5, 3)
        r0 = relay.run_batch(q, g, "m")
        r1 = relay.run_batch(q, g, "m")
        r2 = relay.run_batch(q, g, "m")
        assert r0.batch_id == 0
        assert r1.batch_id == 1
        assert r2.batch_id == 2

    def test_empty_batch(self):
        """Empty batch produces accuracy=0.0 and n_questions=0."""
        relay = _make_relay()
        result = relay.run_batch([], [], "m")
        assert result.n_questions == 0
        assert result.accuracy == pytest.approx(0.0)
        assert result.n_tier1_updates == 0


# ---------------------------------------------------------------------------
# SelfLearningRelay.run_batch — Tier 2
# ---------------------------------------------------------------------------


class TestRunBatchTier2:
    """REQ-LEARN-026-3 (Tier 2): templates activate after pattern threshold."""

    def test_no_templates_active_before_threshold(self):
        """With min_frequency=10 and only 5 violations, no template activates."""
        relay = _make_relay(min_frequency=10)
        questions, ground_truth = _make_batch(10, 5)  # 5 violations
        result = relay.run_batch(questions, ground_truth, "m")
        assert result.n_tier2_templates_active == 0

    def test_template_activates_after_threshold(self):
        """With min_frequency=3 and 6 violations, carry_check template activates.

        The relay cycles carry_error → sign_error → unit_error → comparison_error.
        After 6 wrong responses, carry_check and sign_check each see >= 1 observation;
        but carry_check needs min_frequency=3.  We verify at least 1 is active.

        Spec: SCENARIO-LEARN-046
        """
        # Set min_frequency=2 so templates activate quickly.
        relay = _make_relay(min_frequency=2)
        # 6 wrong answers → 6 violation observations (cycling through 4 types)
        # carry_error: observations at positions 0, 4 → mapped to carry_check → count=2 ≥ 2
        questions, ground_truth = _make_batch(10, 4)  # 6 wrong
        result = relay.run_batch(questions, ground_truth, "m")
        assert result.n_tier2_templates_active >= 1

    def test_templates_accumulate_across_batches(self):
        """More violations across batches means more templates can activate."""
        relay = _make_relay(min_frequency=3)
        q, g = _make_batch(10, 5)  # 5 violations per batch
        r0 = relay.run_batch(q, g, "m")
        r1 = relay.run_batch(q, g, "m")
        # After 10 violations total (5+5), some templates should activate.
        assert r1.n_tier2_templates_active >= r0.n_tier2_templates_active

    def test_templates_not_active_for_different_model_id(self):
        """Templates activated for model_A do not count for model_B."""
        relay = _make_relay(min_frequency=2)
        q, g = _make_batch(10, 4)  # 6 violations
        relay.run_batch(q, g, "model_A")
        # Now run a small batch for model_B with no violations
        q2, g2 = _make_batch(5, 5)  # 0 violations
        result = relay.run_batch(q2, g2, "model_B")
        assert result.n_tier2_templates_active == 0


# ---------------------------------------------------------------------------
# SelfLearningRelay.run_batch — Tier 3
# ---------------------------------------------------------------------------


class TestRunBatchTier3:
    """REQ-LEARN-026-3 (Tier 3): EORM gate AUC is computed and in [0, 1]."""

    def test_tier3_auc_in_range(self):
        """tier3_gate_auc is always in [0, 1]."""
        relay = _make_relay()
        q, g = _make_batch(25, 15)
        result = relay.run_batch(q, g, "m")
        assert 0.0 <= result.tier3_gate_auc <= 1.0

    def test_tier3_auc_is_float(self):
        """tier3_gate_auc is a float (not None, not int)."""
        relay = _make_relay()
        q, g = _make_batch(5, 3)
        result = relay.run_batch(q, g, "m")
        assert isinstance(result.tier3_gate_auc, float)

    def test_tier3_auc_with_single_label_class(self):
        """When all labels are True, tier3_gate_auc defaults to 0.5."""
        relay = _make_relay()
        q, g = _make_batch(5, 5)  # all correct → only one class
        result = relay.run_batch(q, g, "m")
        assert result.tier3_gate_auc == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# SelfLearningRelay.run_batch — cumulative_accuracy
# ---------------------------------------------------------------------------


class TestCumulativeAccuracy:
    """REQ-LEARN-026-5: cumulative_accuracy aggregates across all batches."""

    def test_cumulative_equals_batch_accuracy_on_first_batch(self):
        """First batch: cumulative_accuracy == batch accuracy."""
        relay = _make_relay()
        q, g = _make_batch(25, 15)  # accuracy = 0.6
        result = relay.run_batch(q, g, "m")
        assert result.cumulative_accuracy == pytest.approx(result.accuracy)

    def test_cumulative_aggregates_across_two_batches(self):
        """Cumulative accuracy = (correct_b1 + correct_b2) / (n1 + n2)."""
        relay = _make_relay()
        # Batch 1: 10 questions, 6 correct (accuracy = 0.6)
        q1, g1 = _make_batch(10, 6)
        relay.run_batch(q1, g1, "m")
        # Batch 2: 10 questions, 8 correct (accuracy = 0.8)
        q2, g2 = _make_batch(10, 8)
        r2 = relay.run_batch(q2, g2, "m")
        # Cumulative: (6+8)/(10+10) = 14/20 = 0.7
        assert r2.cumulative_accuracy == pytest.approx(0.7)

    def test_cumulative_accuracy_four_batches_ascending(self):
        """Four batches with ascending accuracy produce rising cumulative_accuracy."""
        relay = _make_relay()
        # Batch accuracies: 0.60, 0.65, 0.70, 0.75 (each 20 questions)
        batch_corrects = [12, 13, 14, 15]  # out of 20
        results = []
        for n_correct in batch_corrects:
            q, g = _make_batch(20, n_correct)
            results.append(relay.run_batch(q, g, "m"))
        # Cumulative should be non-decreasing
        cumulatives = [r.cumulative_accuracy for r in results]
        for i in range(len(cumulatives) - 1):
            assert cumulatives[i + 1] >= cumulatives[i] - 1e-9


# ---------------------------------------------------------------------------
# SelfLearningRelay.learning_trajectory
# ---------------------------------------------------------------------------


class TestLearningTrajectory:
    """REQ-LEARN-026-4: learning_trajectory() returns all accumulated results."""

    def test_trajectory_empty_before_batches(self):
        """No batches run → empty trajectory."""
        relay = _make_relay()
        assert relay.learning_trajectory() == []

    def test_trajectory_grows_with_batches(self):
        """Each run_batch() call appends one result to the trajectory."""
        relay = _make_relay()
        q, g = _make_batch(5, 3)
        relay.run_batch(q, g, "m")
        relay.run_batch(q, g, "m")
        traj = relay.learning_trajectory()
        assert len(traj) == 2

    def test_trajectory_returns_copy(self):
        """learning_trajectory() returns a new list, not the internal reference."""
        relay = _make_relay()
        q, g = _make_batch(5, 3)
        relay.run_batch(q, g, "m")
        traj1 = relay.learning_trajectory()
        traj2 = relay.learning_trajectory()
        assert traj1 is not traj2
        assert traj1 == traj2

    def test_trajectory_mutation_does_not_affect_relay(self):
        """Mutating the returned list does not affect internal state."""
        relay = _make_relay()
        q, g = _make_batch(5, 3)
        relay.run_batch(q, g, "m")
        traj = relay.learning_trajectory()
        traj.append("garbage")  # mutate the copy
        # Internal list should still have only 1 item.
        assert len(relay.learning_trajectory()) == 1

    def test_trajectory_order_matches_run_order(self):
        """Results in trajectory appear in the order batches were run."""
        relay = _make_relay()
        for i in range(4):
            q, g = _make_batch(5, i + 1)  # ascending n_correct
            relay.run_batch(q, g, "m")
        traj = relay.learning_trajectory()
        for idx, r in enumerate(traj):
            assert r.batch_id == idx


# ---------------------------------------------------------------------------
# compute_learning_improvement
# ---------------------------------------------------------------------------


class TestComputeLearningImprovement:
    """REQ-LEARN-027-1: compute_learning_improvement edge cases and spec."""

    def test_empty_trajectory(self):
        """Empty trajectory → (0.0, 0.0, False).

        Spec: REQ-LEARN-027-1
        """
        b1, b4, improved = compute_learning_improvement([])
        assert b1 == pytest.approx(0.0)
        assert b4 == pytest.approx(0.0)
        assert improved is False

    def test_single_batch(self):
        """Single batch → batch1 and batch4 are the same → improved is False."""
        traj = [SelfLearningBatchResult(0, 25, 0.6, 25, 0, 0.5, 0.6)]
        b1, b4, improved = compute_learning_improvement(traj)
        assert b1 == pytest.approx(0.6)
        assert b4 == pytest.approx(0.6)
        assert improved is False

    def test_two_batches_improving(self):
        """Two batches, second better → improved=True.  Uses last batch as batch4."""
        traj = [
            SelfLearningBatchResult(0, 25, 0.6, 25, 0, 0.5, 0.6),
            SelfLearningBatchResult(1, 25, 0.8, 25, 0, 0.5, 0.7),
        ]
        b1, b4, improved = compute_learning_improvement(traj)
        assert b1 == pytest.approx(0.6)
        assert b4 == pytest.approx(0.8)
        assert improved is True

    def test_four_batches_ascending(self):
        """Four batches with ascending accuracy → improved=True.

        Spec: SCENARIO-LEARN-047
        """
        traj = [
            SelfLearningBatchResult(0, 25, 0.60, 25, 0, 0.5, 0.60),
            SelfLearningBatchResult(1, 25, 0.65, 25, 0, 0.5, 0.625),
            SelfLearningBatchResult(2, 25, 0.70, 25, 0, 0.5, 0.65),
            SelfLearningBatchResult(3, 25, 0.75, 25, 0, 0.5, 0.675),
        ]
        b1, b4, improved = compute_learning_improvement(traj)
        assert b1 == pytest.approx(0.60)
        assert b4 == pytest.approx(0.75)
        assert improved is True

    def test_no_improvement_flat(self):
        """Equal accuracy across batches → improved=False (strict >)."""
        traj = [
            SelfLearningBatchResult(0, 25, 0.6, 25, 0, 0.5, 0.6),
            SelfLearningBatchResult(1, 25, 0.6, 25, 0, 0.5, 0.6),
            SelfLearningBatchResult(2, 25, 0.6, 25, 0, 0.5, 0.6),
            SelfLearningBatchResult(3, 25, 0.6, 25, 0, 0.5, 0.6),
        ]
        b1, b4, improved = compute_learning_improvement(traj)
        assert improved is False

    def test_four_batches_descending(self):
        """Descending accuracy → improved=False."""
        traj = [
            SelfLearningBatchResult(0, 25, 0.75, 25, 0, 0.5, 0.75),
            SelfLearningBatchResult(1, 25, 0.70, 25, 0, 0.5, 0.725),
            SelfLearningBatchResult(2, 25, 0.65, 25, 0, 0.5, 0.7),
            SelfLearningBatchResult(3, 25, 0.60, 25, 0, 0.5, 0.675),
        ]
        b1, b4, improved = compute_learning_improvement(traj)
        assert b1 == pytest.approx(0.75)
        assert b4 == pytest.approx(0.60)
        assert improved is False

    def test_five_batches_uses_index_3(self):
        """With 5 batches, batch4 uses index 3 (the 4th result), not index 4."""
        traj = [
            SelfLearningBatchResult(0, 25, 0.60, 25, 0, 0.5, 0.60),
            SelfLearningBatchResult(1, 25, 0.65, 25, 0, 0.5, 0.625),
            SelfLearningBatchResult(2, 25, 0.70, 25, 0, 0.5, 0.65),
            SelfLearningBatchResult(3, 25, 0.75, 25, 0, 0.5, 0.675),
            # 5th batch would be index 4 — should NOT be used
            SelfLearningBatchResult(4, 25, 0.40, 25, 0, 0.5, 0.62),
        ]
        b1, b4, improved = compute_learning_improvement(traj)
        assert b4 == pytest.approx(0.75)  # index 3, not index 4 (0.40)
        assert improved is True


# ---------------------------------------------------------------------------
# build_relay_artifact
# ---------------------------------------------------------------------------


class TestBuildRelayArtifact:
    """REQ-LEARN-027-2: build_relay_artifact schema and honest_verdict logic."""

    def _make_traj(self) -> list[SelfLearningBatchResult]:
        return [
            SelfLearningBatchResult(0, 25, 0.60, 25, 0, 0.5, 0.60),
            SelfLearningBatchResult(1, 25, 0.65, 25, 1, 0.52, 0.625),
            SelfLearningBatchResult(2, 25, 0.70, 25, 1, 0.55, 0.65),
            SelfLearningBatchResult(3, 25, 0.75, 25, 2, 0.58, 0.675),
        ]

    def test_schema_present(self):
        """Artifact has 'schema' == 'carnot.self_learning_relay.v1'."""
        traj = self._make_traj()
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        assert artifact["schema"] == "carnot.self_learning_relay.v1"

    def test_required_keys_present(self):
        """All required top-level keys are present in the artifact."""
        traj = self._make_traj()
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        required = {
            "schema", "trajectory", "batch1_accuracy", "batch4_accuracy",
            "improved", "inference_mode", "honest_verdict",
        }
        assert required.issubset(artifact.keys())

    def test_honest_verdict_synthetic_only_when_no_live_gpu(self):
        """When improved=True but inference_mode is not 'live_gpu', verdict is 'synthetic_only'."""
        traj = self._make_traj()
        improvement = (0.60, 0.75, True)
        artifact = build_relay_artifact(traj, improvement, inference_mode="cpu_synthetic")
        assert artifact["honest_verdict"] == "synthetic_only"

    def test_honest_verdict_learning_confirmed_on_live_gpu(self):
        """Only live_gpu + improved=True yields 'learning_confirmed'."""
        traj = self._make_traj()
        improvement = (0.60, 0.75, True)
        artifact = build_relay_artifact(traj, improvement, inference_mode="live_gpu")
        assert artifact["honest_verdict"] == "learning_confirmed"

    def test_honest_verdict_no_improvement_when_not_improved(self):
        """When improved=False, verdict is 'no_improvement' regardless of mode."""
        traj = self._make_traj()
        improvement = (0.75, 0.60, False)
        for mode in ("cpu_synthetic", "live_gpu"):
            artifact = build_relay_artifact(traj, improvement, inference_mode=mode)
            assert artifact["honest_verdict"] == "no_improvement"

    def test_default_inference_mode_is_cpu_synthetic(self):
        """Default inference_mode is 'cpu_synthetic'."""
        traj = self._make_traj()
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        assert artifact["inference_mode"] == "cpu_synthetic"

    def test_trajectory_serialized_as_list_of_dicts(self):
        """Trajectory is a list of dicts with all SelfLearningBatchResult fields."""
        traj = self._make_traj()
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        assert isinstance(artifact["trajectory"], list)
        assert len(artifact["trajectory"]) == 4
        required_keys = {
            "batch_id", "n_questions", "accuracy", "n_tier1_updates",
            "n_tier2_templates_active", "tier3_gate_auc", "cumulative_accuracy",
        }
        for item in artifact["trajectory"]:
            assert required_keys.issubset(item.keys())

    def test_batch_accuracies_match_improvement_tuple(self):
        """batch1_accuracy and batch4_accuracy match the learning_improvement tuple."""
        traj = self._make_traj()
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        b1, b4, improved = improvement
        assert artifact["batch1_accuracy"] == pytest.approx(b1)
        assert artifact["batch4_accuracy"] == pytest.approx(b4)
        assert artifact["improved"] == improved

    def test_empty_trajectory_produces_valid_artifact(self):
        """Empty trajectory yields schema-valid artifact with 0.0 accuracies."""
        traj: list[SelfLearningBatchResult] = []
        improvement = compute_learning_improvement(traj)
        artifact = build_relay_artifact(traj, improvement)
        assert artifact["schema"] == "carnot.self_learning_relay.v1"
        assert artifact["trajectory"] == []
        assert artifact["improved"] is False
        assert artifact["honest_verdict"] == "no_improvement"


# ---------------------------------------------------------------------------
# Integration: full 4-batch relay run
# ---------------------------------------------------------------------------


class TestFullRelayRun:
    """Integration test: 4 batches of 25 questions with ascending accuracy."""

    def test_four_batch_relay_ascending_synthetic(self):
        """Run 4 batches with 60/65/70/75% correct → improved trajectory.

        This is the CI-safe equivalent of Exp 361.  Uses all-stub components.
        """
        relay = _make_relay(min_frequency=5)
        # Accuracy profile: 0.60, 0.65, 0.70, 0.75 across 4 batches of 25.
        n_corrects = [15, 16, 17, 18]  # 60%, 64%, 68%, 72% of 25
        n_wrongs = [25 - n for n in n_corrects]

        results = []
        for n_correct in n_corrects:
            q, g = _make_batch(25, n_correct)
            results.append(relay.run_batch(q, g, "ci_model"))

        traj = relay.learning_trajectory()
        assert len(traj) == 4

        # Batch IDs are 0–3.
        for i, r in enumerate(traj):
            assert r.batch_id == i
            assert r.n_questions == 25
            assert r.n_tier1_updates == 25
            assert 0.0 <= r.tier3_gate_auc <= 1.0
            assert 0.0 <= r.accuracy <= 1.0

        # Accuracy should be higher in batch 4 than batch 1.
        improvement = compute_learning_improvement(traj)
        b1, b4, improved = improvement
        assert b1 < b4
        assert improved is True

        # Build artifact and verify schema.
        artifact = build_relay_artifact(traj, improvement, inference_mode="cpu_synthetic")
        assert artifact["schema"] == "carnot.self_learning_relay.v1"
        assert artifact["honest_verdict"] == "synthetic_only"
        assert len(artifact["trajectory"]) == 4

    def test_tier2_wiring_object_created_internally(self):
        """SelfLearningRelay creates its own CaseMemoryTemplateWiring internally."""
        relay = _make_relay()
        assert relay._wiring is not None
        assert relay._wiring._library is relay._template_library

    def test_relay_with_always_false_ising(self):
        """Pipeline that always rejects still produces valid batch results."""
        relay = _make_relay(ising_fn=_ising_stub_wrong)
        q, g = _make_batch(10, 5)
        result = relay.run_batch(q, g, "m")
        assert result.n_tier1_updates == 10
        assert 0.0 <= result.tier3_gate_auc <= 1.0
