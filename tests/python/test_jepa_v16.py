"""Tests for JEPA v16: InfoNCE loss, v16 training data, and cascade unblock logic.

Spec: REQ-LEARN-053, REQ-LEARN-054, SCENARIO-LEARN-087, SCENARIO-LEARN-088, SCENARIO-LEARN-089
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from python.carnot.models.infonce_loss import InfoNCELoss
from python.carnot.pipeline.jepa_v16 import (
    JEPAv16,
    _text_embedding,
    build_v16_training_data,
    EMBED_DIM,
)
from scripts.experiment_698_jepa_v16 import (
    _compute_auc,
    _platt_calibrate,
    update_cascade_block,
    _gsm8k_ood_questions,
    _build_ood_pairs,
)


# ---------------------------------------------------------------------------
# InfoNCELoss tests (SCENARIO-LEARN-087, REQ-LEARN-053)
# ---------------------------------------------------------------------------


class TestInfoNCELoss:
    """Tests for InfoNCELoss computation. Spec: REQ-LEARN-053, SCENARIO-LEARN-087."""

    def test_loss_positive_with_negatives(self):
        """InfoNCE loss is a positive float when negatives are present.
        Spec: REQ-LEARN-053, SCENARIO-LEARN-087.
        """
        loss_fn = InfoNCELoss(temperature=0.07)
        rng = np.random.default_rng(0)
        anchor = rng.standard_normal(EMBED_DIM).astype(np.float32)
        positive = rng.standard_normal(EMBED_DIM).astype(np.float32)
        negatives = [rng.standard_normal(EMBED_DIM).astype(np.float32) for _ in range(3)]
        loss = loss_fn.compute(anchor, positive, negatives)
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_loss_zero_without_negatives(self):
        """InfoNCE loss is 0.0 when no negatives are provided.
        Spec: REQ-LEARN-053.
        """
        loss_fn = InfoNCELoss(temperature=0.07)
        rng = np.random.default_rng(1)
        anchor = rng.standard_normal(EMBED_DIM).astype(np.float32)
        positive = rng.standard_normal(EMBED_DIM).astype(np.float32)
        assert loss_fn.compute(anchor, positive, []) == 0.0

    def test_perfect_separation_gives_low_loss(self):
        """When anchor and positive are identical and negatives are orthogonal, loss is low.
        Spec: REQ-LEARN-053, SCENARIO-LEARN-087.
        """
        loss_fn = InfoNCELoss(temperature=0.07)
        dim = 32
        anchor = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        positive = np.ones(dim, dtype=np.float32) / np.sqrt(dim)  # identical to anchor
        neg = -np.ones(dim, dtype=np.float32) / np.sqrt(dim)      # antipodal
        loss = loss_fn.compute(anchor, positive, [neg])
        # With temp=0.07, sim(a,p)=1.0/0.07≈14.3 and sim(a,neg)=-14.3 — softmax ≈ 1.0 → loss ≈ 0.
        assert loss < 0.01

    def test_batch_loss_empty(self):
        """batch_loss returns 0.0 for empty input. Spec: REQ-LEARN-053."""
        loss_fn = InfoNCELoss(temperature=0.07)
        assert loss_fn.batch_loss([], [], []) == 0.0

    def test_batch_loss_multiple_triplets(self):
        """batch_loss returns the mean of individual losses. Spec: REQ-LEARN-053."""
        loss_fn = InfoNCELoss(temperature=0.07)
        rng = np.random.default_rng(2)
        anchors = [rng.standard_normal(EMBED_DIM).astype(np.float32) for _ in range(4)]
        positives = [rng.standard_normal(EMBED_DIM).astype(np.float32) for _ in range(4)]
        negatives_list = [[rng.standard_normal(EMBED_DIM).astype(np.float32) for _ in range(2)] for _ in range(4)]
        batch = loss_fn.batch_loss(anchors, positives, negatives_list)
        individual = [loss_fn.compute(a, p, ns) for a, p, ns in zip(anchors, positives, negatives_list)]
        assert abs(batch - float(np.mean(individual))) < 1e-6

    def test_temperature_zero_raises(self):
        """temperature <= 0 raises ValueError. Spec: REQ-LEARN-053-2."""
        with pytest.raises(ValueError, match="temperature must be > 0"):
            InfoNCELoss(temperature=0.0)

    def test_default_temperature(self):
        """Default temperature is 0.07. Spec: REQ-LEARN-053-2."""
        loss_fn = InfoNCELoss()
        assert loss_fn.temperature == 0.07

    def test_cosine_sim_zero_vector(self):
        """Cosine similarity with zero vector returns 0.0 without error.
        Spec: REQ-LEARN-053.
        """
        loss_fn = InfoNCELoss(temperature=0.07)
        zero = np.zeros(EMBED_DIM, dtype=np.float32)
        normal = np.ones(EMBED_DIM, dtype=np.float32)
        assert loss_fn._cosine_sim(zero, normal) == 0.0


# ---------------------------------------------------------------------------
# Training data builder tests (SCENARIO-LEARN-088, REQ-LEARN-053-4)
# ---------------------------------------------------------------------------


class TestBuildV16TrainingData:
    """Tests for build_v16_training_data. Spec: REQ-LEARN-053-4, SCENARIO-LEARN-088."""

    def _make_pairs(self, n_questions: int = 5, steps_per_q: int = 4) -> list[dict]:
        """Build synthetic FoVer pairs for testing."""
        pairs = []
        for q_idx in range(n_questions):
            q = f"Question {q_idx}: how many apples?"
            for s_idx in range(steps_per_q):
                pairs.append({
                    "question": q,
                    "step_text": f"Step {s_idx} for Q{q_idx}: compute {s_idx * 2}.",
                    "step_correct": s_idx % 2 == 0,  # even steps = correct
                    "z3_verdict": "sat",
                })
        return pairs

    def test_returns_list_of_triplets(self):
        """build_v16_training_data returns a list of dicts with anchor/positive/negatives.
        Spec: REQ-LEARN-053-4, SCENARIO-LEARN-088.
        """
        pairs = self._make_pairs(n_questions=5, steps_per_q=4)
        triplets = build_v16_training_data(pairs)
        assert len(triplets) > 0
        for t in triplets:
            assert "anchor" in t
            assert "positive" in t
            assert "negatives" in t

    def test_anchor_shape(self):
        """Anchor embeddings have shape (EMBED_DIM,). Spec: REQ-LEARN-053-4."""
        pairs = self._make_pairs()
        triplets = build_v16_training_data(pairs)
        for t in triplets:
            assert t["anchor"].shape == (EMBED_DIM,)

    def test_at_least_200_pairs_from_fover(self):
        """With 200 fover pairs, triplet count is >= 200. Spec: SCENARIO-LEARN-088."""
        # 200 pairs across 200 questions, 1 step each (all correct — uses cross-question negatives).
        pairs = [
            {"question": f"Question {i}", "step_text": f"Step for Q{i}", "step_correct": True}
            for i in range(200)
        ]
        triplets = build_v16_training_data(pairs)
        assert len(triplets) >= 200

    def test_single_question_all_positive_skipped(self):
        """A single question with only correct steps and no cross-question negatives yields no triplets.
        Spec: REQ-LEARN-053.
        """
        # Only one question — no other questions to use as cross-question negatives.
        pairs = [
            {"question": "Q1", "step_text": "Step A", "step_correct": True},
            {"question": "Q1", "step_text": "Step B", "step_correct": True},
        ]
        triplets = build_v16_training_data(pairs)
        assert triplets == []

    def test_cross_question_negatives_used_when_no_incorrect_steps(self):
        """When all steps are correct, cross-question negatives enable triplet construction.
        Spec: REQ-LEARN-053, SCENARIO-LEARN-088.
        """
        pairs = [
            {"question": "Q1", "step_text": "Step A", "step_correct": True},
            {"question": "Q2", "step_text": "Step B", "step_correct": True},
        ]
        triplets = build_v16_training_data(pairs)
        # Q1's positive uses Q2's step as cross-question negative, and vice versa.
        assert len(triplets) > 0

    def test_skips_all_negative_questions(self):
        """Questions with only incorrect steps are skipped.
        Spec: REQ-LEARN-053.
        """
        pairs = [
            {"question": "Q1", "step_text": "Step A", "step_correct": False},
            {"question": "Q1", "step_text": "Step B", "step_correct": False},
        ]
        triplets = build_v16_training_data(pairs)
        assert triplets == []


# ---------------------------------------------------------------------------
# Cascade unblock tests (SCENARIO-LEARN-089, REQ-LEARN-053-5)
# ---------------------------------------------------------------------------


class TestCascadeUnblock:
    """Tests for update_cascade_block. Spec: REQ-LEARN-053-5, SCENARIO-LEARN-089."""

    def _make_manifest(self, with_jepa_block: bool = True) -> dict:
        manifest = {
            "excluded": [
                {"experiment_id": 308, "reason": "legacy"},
            ]
        }
        if with_jepa_block:
            manifest["excluded"].append({
                "experiment_id": "jepa_v15_cascade",
                "completed_milestone": "2026.04.53",
                "reason": "ood_auc_below_random_blocked_until_v16",
            })
        return manifest

    def test_removes_block_when_auc_target_met(self, tmp_path):
        """jepa_v15_cascade block is removed when v16_ood_auc >= 0.75.
        Spec: REQ-LEARN-053-5, SCENARIO-LEARN-089.
        """
        manifest_path = tmp_path / "conductor_exclusion_manifest.json"
        manifest_path.write_text(json.dumps(self._make_manifest(with_jepa_block=True)))
        result = update_cascade_block(manifest_path, v16_ood_auc=0.76)
        assert result is True
        updated = json.loads(manifest_path.read_text())
        ids = [str(e.get("experiment_id", "")) for e in updated["excluded"]]
        assert "jepa_v15_cascade" not in ids

    def test_does_not_remove_when_auc_below_target(self, tmp_path):
        """Block is NOT removed when v16_ood_auc < 0.75.
        Spec: REQ-LEARN-053-5.
        """
        manifest_path = tmp_path / "conductor_exclusion_manifest.json"
        manifest_path.write_text(json.dumps(self._make_manifest(with_jepa_block=True)))
        result = update_cascade_block(manifest_path, v16_ood_auc=0.74)
        assert result is False
        updated = json.loads(manifest_path.read_text())
        ids = [str(e.get("experiment_id", "")) for e in updated["excluded"]]
        assert "jepa_v15_cascade" in ids

    def test_returns_false_when_block_absent(self, tmp_path):
        """Returns False (not a double-remove) when block was already absent.
        Spec: REQ-LEARN-053-5.
        """
        manifest_path = tmp_path / "conductor_exclusion_manifest.json"
        manifest_path.write_text(json.dumps(self._make_manifest(with_jepa_block=False)))
        result = update_cascade_block(manifest_path, v16_ood_auc=0.80)
        assert result is False

    def test_returns_false_when_manifest_missing(self, tmp_path):
        """Returns False (gracefully) when manifest file does not exist.
        Spec: REQ-LEARN-053-5.
        """
        result = update_cascade_block(tmp_path / "nonexistent.json", v16_ood_auc=0.80)
        assert result is False


# ---------------------------------------------------------------------------
# JEPAv16 model tests (REQ-LEARN-053)
# ---------------------------------------------------------------------------


class TestJEPAv16:
    """Tests for the JEPAv16 model class. Spec: REQ-LEARN-053."""

    def test_score_in_unit_interval(self):
        """score() returns a float in [0, 1]. Spec: REQ-LEARN-053."""
        model = JEPAv16(seed=0)
        emb = np.random.default_rng(0).standard_normal(EMBED_DIM).astype(np.float32)
        s = model.score(emb)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_train_returns_log(self):
        """train() returns a dict with train_losses key. Spec: REQ-LEARN-053."""
        model = JEPAv16(seed=0)
        rng = np.random.default_rng(0)
        triplets = [{
            "anchor": rng.standard_normal(EMBED_DIM).astype(np.float32),
            "positive": rng.standard_normal(EMBED_DIM).astype(np.float32),
            "negatives": [rng.standard_normal(EMBED_DIM).astype(np.float32)],
        } for _ in range(5)]
        log = model.train(triplets, n_epochs=5)
        assert "train_losses" in log
        assert len(log["train_losses"]) == 5

    def test_save_and_load_roundtrip(self, tmp_path):
        """save/load roundtrip preserves scores. Spec: REQ-LEARN-053."""
        model = JEPAv16(seed=7)
        emb = np.ones(EMBED_DIM, dtype=np.float32)
        score_before = model.score(emb)
        path = str(tmp_path / "v16.npz")
        model.save(path)
        model2 = JEPAv16(seed=99)  # different seed
        model2.load(path)
        assert abs(model2.score(emb) - score_before) < 1e-6

    def test_load_missing_file_raises(self, tmp_path):
        """load() raises FileNotFoundError for missing path. Spec: REQ-LEARN-053."""
        model = JEPAv16()
        with pytest.raises(FileNotFoundError):
            model.load(str(tmp_path / "missing.npz"))

    def test_train_empty_triplets(self):
        """train() with empty triplets returns empty log without error. Spec: REQ-LEARN-053."""
        model = JEPAv16()
        log = model.train([])
        assert log["train_losses"] == []
        assert log["n_triplets"] == 0


# ---------------------------------------------------------------------------
# AUC and calibration helpers (REQ-LEARN-054)
# ---------------------------------------------------------------------------


class TestAUCAndCalibration:
    """Tests for _compute_auc and _platt_calibrate. Spec: REQ-LEARN-054."""

    def test_perfect_auc(self):
        """AUC = 1.0 when all positive scores exceed all negative scores.
        Spec: REQ-LEARN-054.
        """
        scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        labels = [1, 1, 1, 0, 0, 0]
        assert _compute_auc(scores, labels) == 1.0

    def test_random_auc(self):
        """AUC ≈ 0.5 for uniformly random scores. Spec: REQ-LEARN-054."""
        rng = np.random.default_rng(42)
        scores = list(rng.random(1000))
        labels = [1 if i % 2 == 0 else 0 for i in range(1000)]
        auc = _compute_auc(scores, labels)
        assert 0.4 < auc < 0.6

    def test_platt_returns_temperature_and_ece(self):
        """_platt_calibrate returns (temperature float, ece float). Spec: REQ-LEARN-054."""
        rng = np.random.default_rng(0)
        scores = list(rng.random(100))
        labels = [1 if s > 0.5 else 0 for s in scores]
        temp, ece = _platt_calibrate(scores, labels)
        assert isinstance(temp, float)
        assert isinstance(ece, float)
        assert temp > 0.0
        assert 0.0 <= ece <= 1.0

    def test_ece_perfect_calibration(self):
        """ECE is low when predicted probabilities match empirical frequencies.
        Spec: REQ-LEARN-054.
        """
        # Perfect calibration: score = label probability.
        rng = np.random.default_rng(1)
        true_probs = rng.uniform(0.1, 0.9, 200)
        labels = (rng.random(200) < true_probs).astype(int).tolist()
        scores = true_probs.tolist()
        _, ece = _platt_calibrate(scores, labels, n_steps=1000)
        # With actual calibrated inputs, ECE should be reasonably low.
        assert ece < 0.25  # loose bound — calibration depends on sample randomness


# ---------------------------------------------------------------------------
# OOD evaluation helpers
# ---------------------------------------------------------------------------


class TestOODHelpers:
    """Tests for OOD question generation and pair building."""

    def test_ood_questions_count(self):
        """_gsm8k_ood_questions returns 200 questions for indices 500-699."""
        qs = _gsm8k_ood_questions(500, 700)
        assert len(qs) == 200

    def test_ood_pairs_balanced(self):
        """_build_ood_pairs returns equal numbers of positive and negative samples."""
        qs = _gsm8k_ood_questions(500, 510)
        embeddings, labels = _build_ood_pairs(qs)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        assert n_pos == n_neg

    def test_ood_embedding_shape(self):
        """All OOD embeddings have shape (EMBED_DIM,)."""
        qs = _gsm8k_ood_questions(500, 505)
        embeddings, _ = _build_ood_pairs(qs)
        for emb in embeddings:
            assert emb.shape == (EMBED_DIM,)


# ---------------------------------------------------------------------------
# Text embedding helper
# ---------------------------------------------------------------------------


class TestTextEmbedding:
    """Tests for _text_embedding. Spec: REQ-LEARN-053."""

    def test_shape(self):
        """_text_embedding returns (EMBED_DIM,) array."""
        emb = _text_embedding("hello world")
        assert emb.shape == (EMBED_DIM,)

    def test_unit_norm(self):
        """_text_embedding returns L2-normalised vector."""
        emb = _text_embedding("test text for normalisation")
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_deterministic(self):
        """Same input always produces the same embedding."""
        emb1 = _text_embedding("deterministic test")
        emb2 = _text_embedding("deterministic test")
        np.testing.assert_array_equal(emb1, emb2)

    def test_different_inputs_differ(self):
        """Different inputs produce different embeddings (with very high probability)."""
        emb1 = _text_embedding("text A")
        emb2 = _text_embedding("text B")
        assert not np.allclose(emb1, emb2)
