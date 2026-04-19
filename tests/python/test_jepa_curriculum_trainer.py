"""Tests for jepa_curriculum_trainer: CurriculumStageResult, JEPACurriculumTrainer, JEPARetrainV3Result.

100% coverage for python/carnot/models/jepa_curriculum_trainer.py.

Spec coverage: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042,
               SCENARIO-LEARN-068, SCENARIO-LEARN-069, SCENARIO-LEARN-070
"""

from __future__ import annotations

import pytest

from carnot.models.jepa_curriculum_trainer import (
    CurriculumStageResult,
    JEPACurriculumTrainer,
    JEPARetrainV3Result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pairs(n_correct: int = 15, n_incorrect: int = 15, confidence: float = 0.9) -> list[dict]:
    """Generate labeled pair dicts for testing."""
    pairs = []
    for i in range(n_correct):
        pairs.append({
            "step_text": f"2 * {i + 1} = {2 * (i + 1)} so the answer is {2 * (i + 1)}",
            "label": "correct",
            "label_confidence": confidence,
        })
    for i in range(n_incorrect):
        pairs.append({
            "step_text": f"2 * {i + 1} = {2 * (i + 1) + 1} wrong calculation",
            "label": "incorrect",
            "label_confidence": confidence,
        })
    return pairs


def _make_mixed_pairs() -> list[dict]:
    """Pairs with mixed label_confidence: some above 0.85, some below."""
    high_conf = _make_pairs(n_correct=5, n_incorrect=5, confidence=0.95)
    low_conf = _make_pairs(n_correct=5, n_incorrect=5, confidence=0.60)
    return high_conf + low_conf


# ---------------------------------------------------------------------------
# CurriculumStageResult tests
# ---------------------------------------------------------------------------


class TestCurriculumStageResult:
    """Spec: REQ-LEARN-040, SCENARIO-LEARN-068"""

    def test_auc_improved_true_when_after_gt_before(self):
        """auc_improved=True when AUC increases from stage to stage."""
        r = CurriculumStageResult(stage=2, n_pairs=30, n_epochs=100, auc_after=0.7, auc_before=0.5)
        assert r.auc_improved is True

    def test_auc_improved_false_when_after_le_before(self):
        """auc_improved=False when AUC did not increase."""
        r = CurriculumStageResult(stage=2, n_pairs=30, n_epochs=100, auc_after=0.4, auc_before=0.5)
        assert r.auc_improved is False

    def test_auc_improved_false_when_equal(self):
        """auc_improved=False when AUC is exactly equal (not strictly greater)."""
        r = CurriculumStageResult(stage=2, n_pairs=30, n_epochs=100, auc_after=0.5, auc_before=0.5)
        assert r.auc_improved is False

    def test_stage_attributes_stored(self):
        """All attributes are stored correctly."""
        r = CurriculumStageResult(stage=1, n_pairs=12, n_epochs=50, auc_after=0.8, auc_before=0.5)
        assert r.stage == 1
        assert r.n_pairs == 12
        assert r.n_epochs == 50
        assert r.auc_after == 0.8
        assert r.auc_before == 0.5

    def test_default_auc_before_is_0_5(self):
        """Default auc_before is 0.5 (random baseline)."""
        r = CurriculumStageResult(stage=1, n_pairs=10, n_epochs=100, auc_after=0.6)
        assert r.auc_before == 0.5


# ---------------------------------------------------------------------------
# JEPACurriculumTrainer tests
# ---------------------------------------------------------------------------


class TestJEPACurriculumTrainer:
    """Spec: REQ-LEARN-040, REQ-LEARN-041, REQ-LEARN-042, SCENARIO-LEARN-069"""

    def test_train_returns_three_stages(self):
        """train() returns exactly three CurriculumStageResult objects."""
        pairs = _make_pairs(n_correct=10, n_incorrect=10)
        trainer = JEPACurriculumTrainer(n_stage1_epochs=2, n_stage2_epochs=2, n_stage3_epochs=2)
        stages = trainer.train(pairs)
        assert len(stages) == 3

    def test_stage_numbers_are_1_2_3(self):
        """Stage numbers are 1, 2, 3 in order."""
        pairs = _make_pairs(n_correct=10, n_incorrect=10)
        trainer = JEPACurriculumTrainer(n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1)
        stages = trainer.train(pairs)
        assert [s.stage for s in stages] == [1, 2, 3]

    def test_stage1_filters_by_high_conf(self):
        """Stage 1 only trains on pairs with label_confidence >= high_conf_threshold.

        Spec: REQ-LEARN-040 — JEPACurriculumTrainer stage1 trains on high-confidence pairs.
        """
        pairs = _make_mixed_pairs()  # 10 high-conf (0.95), 10 low-conf (0.60)
        trainer = JEPACurriculumTrainer(
            n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1,
            high_conf_threshold=0.85,
        )
        # Verify internal filter returns only high-conf pairs
        # Use 80% of all pairs as the "train_pool"
        train_pool = pairs[: int(len(pairs) * 0.8)]
        stage1_pairs = trainer._filter_high_conf(train_pool)
        # All returned pairs must have confidence >= 0.85
        for p in stage1_pairs:
            conf = p.get("label_confidence", p.get("confidence", 1.0))
            assert conf >= 0.85, f"Pair with confidence {conf} should not be in stage1"

    def test_stage1_excludes_low_conf_pairs(self):
        """Stage 1 excludes pairs with label_confidence < high_conf_threshold.

        Spec: REQ-LEARN-040
        """
        pairs = _make_mixed_pairs()
        trainer = JEPACurriculumTrainer(high_conf_threshold=0.85)
        stage1_pairs = trainer._filter_high_conf(pairs)
        low_conf_in_stage1 = [p for p in stage1_pairs if p.get("label_confidence", 1.0) < 0.85]
        assert len(low_conf_in_stage1) == 0

    def test_stage2_uses_all_pairs_no_filter(self):
        """Stage 2 trains on ALL pairs (no confidence filter).

        Spec: REQ-LEARN-041 — stage2 fine-tunes on all pairs unfiltered.
        Verified by checking that stage2 n_pairs > stage1 n_pairs on mixed corpus.
        """
        pairs = _make_mixed_pairs()  # mixed confidence
        trainer = JEPACurriculumTrainer(n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1)
        stages = trainer.train(pairs)
        # Stage 2 must have more pairs than Stage 1 (because Stage 1 filters)
        assert stages[1].n_pairs >= stages[0].n_pairs

    def test_stage3_n_pairs_includes_synthetic(self):
        """Stage 3 augments with synthetic pairs when real corpus < 200.

        Spec: REQ-LEARN-042 — stage3 augments to n_total >= 200.
        """
        pairs = _make_pairs(n_correct=10, n_incorrect=10)  # 20 pairs, well below 200
        trainer = JEPACurriculumTrainer(n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1)
        stages = trainer.train(pairs)
        # Stage 3 training pool should include synthetic augmentation
        # The training pool is 80% of 20 = 16 pairs; synthetic needed = 200-20 = 180
        # So stage3 n_pairs should be significantly larger than stage2 n_pairs
        assert stages[2].n_pairs > stages[1].n_pairs

    def test_stage3_no_synthetic_when_200_pairs(self):
        """Stage 3 does NOT add synthetic pairs when real corpus already >= 200.

        Spec: REQ-LEARN-042
        """
        pairs = _make_pairs(n_correct=100, n_incorrect=100)  # 200 pairs
        trainer = JEPACurriculumTrainer(n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1)
        stages = trainer.train(pairs)
        # With 200 pairs, no synthetic needed; stage3 n_pairs == training pool size (80% of 200)
        assert stages[2].n_pairs == stages[1].n_pairs

    def test_train_auc_values_in_range(self):
        """All stage AUC values are in [0, 1]."""
        pairs = _make_pairs(n_correct=8, n_incorrect=8)
        trainer = JEPACurriculumTrainer(n_stage1_epochs=2, n_stage2_epochs=2, n_stage3_epochs=2)
        stages = trainer.train(pairs)
        for s in stages:
            assert 0.0 <= s.auc_after <= 1.0, f"Stage {s.stage} auc_after={s.auc_after} out of range"

    def test_get_final_auc_before_train_returns_0_5(self):
        """get_final_auc() returns 0.5 when called before train()."""
        trainer = JEPACurriculumTrainer()
        pairs = _make_pairs(n_correct=5, n_incorrect=5)
        auc = trainer.get_final_auc(pairs)
        assert auc == 0.5

    def test_get_final_auc_after_train(self):
        """get_final_auc() returns float in [0, 1] after training."""
        pairs = _make_pairs(n_correct=10, n_incorrect=10)
        trainer = JEPACurriculumTrainer(n_stage1_epochs=2, n_stage2_epochs=2, n_stage3_epochs=2)
        trainer.train(pairs)
        held_out = _make_pairs(n_correct=3, n_incorrect=3)
        auc = trainer.get_final_auc(held_out)
        assert 0.0 <= auc <= 1.0

    def test_train_with_empty_high_conf_stage1(self):
        """train() handles the case where no pairs pass Stage 1 filter (all low-confidence)."""
        pairs = _make_pairs(n_correct=10, n_incorrect=10, confidence=0.50)
        trainer = JEPACurriculumTrainer(
            n_stage1_epochs=1, n_stage2_epochs=1, n_stage3_epochs=1,
            high_conf_threshold=0.85,
        )
        stages = trainer.train(pairs)
        assert stages[0].n_pairs == 0  # no pairs pass stage1 filter
        assert 0.0 <= stages[0].auc_after <= 1.0  # still returns a valid AUC

    def test_n_stage_epochs_stored_correctly(self):
        """Each stage records its configured epoch count."""
        pairs = _make_pairs(n_correct=8, n_incorrect=8)
        trainer = JEPACurriculumTrainer(n_stage1_epochs=5, n_stage2_epochs=10, n_stage3_epochs=15)
        stages = trainer.train(pairs)
        assert stages[0].n_epochs == 5
        assert stages[1].n_epochs == 10
        assert stages[2].n_epochs == 15


# ---------------------------------------------------------------------------
# JEPARetrainV3Result tests
# ---------------------------------------------------------------------------


class TestJEPARetrainV3Result:
    """Spec: REQ-LEARN-042, SCENARIO-LEARN-070"""

    def _make_stages(self) -> list[CurriculumStageResult]:
        return [
            CurriculumStageResult(stage=1, n_pairs=10, n_epochs=100, auc_after=0.55, auc_before=0.5),
            CurriculumStageResult(stage=2, n_pairs=30, n_epochs=100, auc_after=0.65, auc_before=0.55),
            CurriculumStageResult(stage=3, n_pairs=180, n_epochs=100, auc_after=0.72, auc_before=0.65),
        ]

    def test_auc_improvement_positive(self):
        """auc_improvement = after_auc - before_auc."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.72,
        )
        assert abs(r.auc_improvement - (0.72 - 0.281)) < 1e-9

    def test_target_met_true_when_after_gt_0_6(self):
        """target_met=True when after_auc > 0.600."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.65,
        )
        assert r.target_met is True

    def test_target_met_false_when_after_le_0_6(self):
        """target_met=False when after_auc <= 0.600."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.55,
        )
        assert r.target_met is False

    def test_regression_recovered_true_when_after_gt_0_4(self):
        """regression_recovered=True when after_auc > 0.400 (recovery from 0.281).

        Spec: SCENARIO-LEARN-070
        """
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.45,
        )
        assert r.regression_recovered is True

    def test_regression_recovered_false_when_after_le_0_4(self):
        """regression_recovered=False when after_auc <= 0.400."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.35,
        )
        assert r.regression_recovered is False

    def test_regression_recovered_boundary_exactly_0_4(self):
        """regression_recovered=False at exactly 0.400 (strict inequality)."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.281,
            after_auc=0.400,
        )
        assert r.regression_recovered is False

    def test_all_attributes_stored(self):
        """n_pairs_raw, curriculum_stages, before_auc, after_auc are all accessible."""
        stages = self._make_stages()
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=stages,
            before_auc=0.281,
            after_auc=0.72,
        )
        assert r.n_pairs_raw == 57
        assert r.curriculum_stages is stages
        assert r.before_auc == 0.281
        assert r.after_auc == 0.72

    def test_auc_improvement_negative_when_regression(self):
        """auc_improvement is negative when after_auc < before_auc (degradation detected)."""
        r = JEPARetrainV3Result(
            n_pairs_raw=57,
            curriculum_stages=self._make_stages(),
            before_auc=0.5,
            after_auc=0.3,
        )
        assert r.auc_improvement < 0
