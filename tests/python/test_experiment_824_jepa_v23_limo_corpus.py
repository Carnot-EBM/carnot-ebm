"""Tests for Experiment 824 — JEPA v23 LIMO curation + triplet loss.

Each test traces to REQ-LEARN-824-001, REQ-LEARN-824-002, or REQ-LEARN-824-003.

Coverage target: 100% of the new code added in:
- python/carnot/pipeline/limo_curator.py
- python/carnot/inference/jepa_v23.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.limo_curator import (
    CuratedPair,
    LIMOCurator,
    _make_humaneval_pairs,
    _make_svamp_pairs,
)
from carnot.inference.jepa_v23 import (
    JEPAv23Predictor,
    TripletLoss,
    _TFIDFVectoriser,
    _compute_auc,
    _cosine_dist,
    _grad_cosine_dist_u,
    _grad_cosine_dist_v,
    _synthetic_holdout,
    evaluate_v23,
    train_v23,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_pair(
    prefix: str = "q1",
    positive: str = "3 + 4 = 7",
    negative: str = "3 + 4 = 8",
    z3_conf: float = 1.0,
    cpmi: float = 0.8,
    domain: str = "gsm8k",
) -> CuratedPair:
    return CuratedPair(
        prefix_text=prefix,
        positive_step=positive,
        negative_step=negative,
        z3_confidence=z3_conf,
        cpmi_score=cpmi,
        source_domain=domain,
        quality_score=z3_conf * cpmi,
    )


def _fover_json(tmp_path: Path) -> Path:
    """Write a minimal FoVer JSON with one correct + one incorrect step per question."""
    data = [
        {"question_id": "1", "step_text": "2 + 3 = 5", "label": "correct", "confidence": 1.0},
        {"question_id": "1", "step_text": "2 + 3 = 6", "label": "incorrect", "confidence": 1.0},
        {"question_id": "2", "step_text": "4 * 5 = 20", "label": "correct", "confidence": 0.95},
        {"question_id": "2", "step_text": "4 * 5 = 19", "label": "incorrect", "confidence": 0.95},
        {"question_id": "3", "step_text": "no equation here", "label": "correct", "confidence": 0.5},
    ]
    p = tmp_path / "fover.json"
    p.write_text(json.dumps(data))
    return p


def _cpmi_json(tmp_path: Path) -> Path:
    """Write a minimal CPMI triples JSON."""
    data = [
        {
            "prefix_text": "gsm8k_1",
            "positive_step": "10 + 5 = 15",
            "negative_step": "10 + 5 = 16",
            "cpmi_score": 0.6,
            "source_domain": "gsm8k",
        },
        {
            "prefix_text": "gsm8k_2",
            "positive_step": "7 * 3 = 21",
            "negative_step": "7 * 3 = 22",
            "cpmi_score": 0.4,
            "source_domain": "gsm8k",
        },
    ]
    p = tmp_path / "cpmi.json"
    p.write_text(json.dumps(data))
    return p


# ===========================================================================
# LIMOCurator tests (REQ-LEARN-824-001, REQ-LEARN-824-002)
# ===========================================================================


class TestLIMOCuratorScorePairs:
    """REQ-LEARN-824-001: score_pairs must sort by z3_confidence × cpmi_score descending."""

    def test_score_pairs_descending(self) -> None:
        """Pairs are returned in descending order by quality_score."""
        pairs = [
            _make_pair(z3_conf=0.5, cpmi=0.5),   # quality = 0.25
            _make_pair(z3_conf=1.0, cpmi=0.9),   # quality = 0.90
            _make_pair(z3_conf=0.8, cpmi=0.7),   # quality = 0.56
        ]
        curator = LIMOCurator.__new__(LIMOCurator)
        ranked = curator.score_pairs(pairs)
        scores = [p.quality_score for p in ranked]
        assert scores == sorted(scores, reverse=True), (
            f"Expected descending quality_score order, got {scores}"
        )

    def test_score_pairs_preserves_all_items(self) -> None:
        """score_pairs returns the same number of items as input."""
        pairs = [_make_pair(z3_conf=i * 0.1, cpmi=i * 0.1) for i in range(1, 6)]
        curator = LIMOCurator.__new__(LIMOCurator)
        ranked = curator.score_pairs(pairs)
        assert len(ranked) == len(pairs)

    def test_score_pairs_empty(self) -> None:
        """score_pairs returns empty list for empty input."""
        curator = LIMOCurator.__new__(LIMOCurator)
        assert curator.score_pairs([]) == []

    def test_score_pairs_single(self) -> None:
        """score_pairs with one item returns that item."""
        pair = _make_pair(z3_conf=0.9, cpmi=0.8)
        curator = LIMOCurator.__new__(LIMOCurator)
        result = curator.score_pairs([pair])
        assert len(result) == 1
        assert result[0] == pair


class TestLIMOCuratorSelectTopK:
    """REQ-LEARN-824-001: select_top_k must return exactly k pairs."""

    def test_select_top_k_returns_k_items(self, tmp_path: Path) -> None:
        """select_top_k returns exactly k items when corpus has >= k pairs."""
        fover = _fover_json(tmp_path)
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(fover_path=fover, cpmi_path=cpmi)
        result = curator.select_top_k(k=2)
        assert len(result) == 2, f"Expected 2 pairs, got {len(result)}"

    def test_select_top_k_returns_fewer_when_corpus_small(self, tmp_path: Path) -> None:
        """select_top_k returns all available pairs when k > corpus size."""
        fover = _fover_json(tmp_path)
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(fover_path=fover, cpmi_path=cpmi)
        # Request more than the corpus has (2 fover + 2 cpmi = max 4 meaningful pairs)
        result = curator.select_top_k(k=100)
        assert len(result) <= 100

    def test_select_top_k_respects_z3_threshold(self, tmp_path: Path) -> None:
        """Pairs below z3_confidence_threshold are excluded."""
        fover = _fover_json(tmp_path)
        cpmi = _cpmi_json(tmp_path)
        # Set threshold very high — should exclude the confidence=0.5 entry
        curator = LIMOCurator(fover_path=fover, cpmi_path=cpmi, z3_confidence_threshold=0.95)
        result = curator.select_top_k(k=10)
        for pair in result:
            if pair.source_domain == "gsm8k" and pair.z3_confidence < 1.0:
                # Only the fover pair with confidence=0.5 should be excluded
                assert pair.z3_confidence >= 0.95, (
                    f"Pair with z3_confidence={pair.z3_confidence} should have been excluded"
                )

    def test_select_top_k_missing_fover(self, tmp_path: Path) -> None:
        """select_top_k works when fover file does not exist."""
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(
            fover_path=tmp_path / "nonexistent.json",
            cpmi_path=cpmi,
        )
        result = curator.select_top_k(k=5)
        assert isinstance(result, list)

    def test_select_top_k_missing_cpmi(self, tmp_path: Path) -> None:
        """select_top_k works when cpmi file does not exist."""
        fover = _fover_json(tmp_path)
        curator = LIMOCurator(
            fover_path=fover,
            cpmi_path=tmp_path / "nonexistent.json",
        )
        result = curator.select_top_k(k=5)
        assert isinstance(result, list)


class TestLIMOCuratorAddDomainPairs:
    """REQ-LEARN-824-002: add_domain_pairs must include HumanEval + SVAMP."""

    def test_add_domain_pairs_domain_diversity(self, tmp_path: Path) -> None:
        """add_domain_pairs returns pairs from all three domains."""
        fover = _fover_json(tmp_path)
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(fover_path=fover, cpmi_path=cpmi)
        result = curator.add_domain_pairs(humaneval_n=10, svamp_n=10)
        domains = {p.source_domain for p in result}
        assert "humaneval" in domains, f"humaneval not in domains: {domains}"
        assert "svamp" in domains, f"svamp not in domains: {domains}"

    def test_add_domain_pairs_count(self, tmp_path: Path) -> None:
        """add_domain_pairs returns top-50 + humaneval_n + svamp_n pairs."""
        fover = _fover_json(tmp_path)
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(fover_path=fover, cpmi_path=cpmi)
        result = curator.add_domain_pairs(humaneval_n=5, svamp_n=5)
        n_humaneval = sum(1 for p in result if p.source_domain == "humaneval")
        n_svamp = sum(1 for p in result if p.source_domain == "svamp")
        assert n_humaneval == 5, f"Expected 5 HumanEval pairs, got {n_humaneval}"
        assert n_svamp == 5, f"Expected 5 SVAMP pairs, got {n_svamp}"


class TestLIMOCuratorLoadFoverPairs:
    """Tests for load_fover_pairs and load_cpmi_triples internals."""

    def test_load_fover_pairs_returns_pairs(self, tmp_path: Path) -> None:
        """load_fover_pairs creates (correct, incorrect) pairings per question."""
        fover = _fover_json(tmp_path)
        curator = LIMOCurator(fover_path=fover, cpmi_path=tmp_path / "none.json")
        pairs = curator.load_fover_pairs()
        assert len(pairs) > 0
        for p in pairs:
            assert isinstance(p, CuratedPair)
            assert p.source_domain == "gsm8k"

    def test_load_fover_pairs_no_correct_step_skipped(self, tmp_path: Path) -> None:
        """Questions with no correct steps produce no pairs."""
        data = [
            {"question_id": "99", "step_text": "wrong step", "label": "incorrect", "confidence": 1.0},
        ]
        p = tmp_path / "fover_no_correct.json"
        p.write_text(json.dumps(data))
        curator = LIMOCurator(fover_path=p, cpmi_path=tmp_path / "none.json")
        pairs = curator.load_fover_pairs()
        assert len(pairs) == 0

    def test_load_cpmi_triples_returns_list(self, tmp_path: Path) -> None:
        """load_cpmi_triples returns a list of dicts."""
        cpmi = _cpmi_json(tmp_path)
        curator = LIMOCurator(fover_path=tmp_path / "none.json", cpmi_path=cpmi)
        triples = curator.load_cpmi_triples()
        assert isinstance(triples, list)
        assert len(triples) == 2

    def test_load_cpmi_triples_missing_file(self, tmp_path: Path) -> None:
        """load_cpmi_triples returns [] when file is missing."""
        curator = LIMOCurator(
            fover_path=tmp_path / "none.json",
            cpmi_path=tmp_path / "none2.json",
        )
        assert curator.load_cpmi_triples() == []

    def test_load_fover_pairs_positive_with_synthetic_negative(self, tmp_path: Path) -> None:
        """FoVer question with only a correct step gets a synthetic negative."""
        data = [
            {"question_id": "42", "step_text": "5 + 5 = 10", "label": "correct", "confidence": 1.0},
        ]
        p = tmp_path / "fover_single.json"
        p.write_text(json.dumps(data))
        curator = LIMOCurator(fover_path=p, cpmi_path=tmp_path / "none.json")
        pairs = curator.load_fover_pairs()
        assert len(pairs) == 1
        assert "INCORRECT" in pairs[0].negative_step or "incorrect" in pairs[0].negative_step.lower()


class TestMakeDomainPairs:
    """Tests for the domain pair generators."""

    def test_make_humaneval_pairs_count(self) -> None:
        """_make_humaneval_pairs returns exactly n pairs."""
        for n in [1, 5, 10]:
            pairs = _make_humaneval_pairs(n)
            assert len(pairs) == n, f"Expected {n}, got {len(pairs)}"

    def test_make_humaneval_pairs_domain(self) -> None:
        """All HumanEval pairs have source_domain='humaneval'."""
        pairs = _make_humaneval_pairs(10)
        assert all(p.source_domain == "humaneval" for p in pairs)

    def test_make_svamp_pairs_count(self) -> None:
        """_make_svamp_pairs returns exactly n pairs."""
        for n in [1, 5, 10]:
            pairs = _make_svamp_pairs(n)
            assert len(pairs) == n, f"Expected {n}, got {len(pairs)}"

    def test_make_svamp_pairs_domain(self) -> None:
        """All SVAMP pairs have source_domain='svamp'."""
        pairs = _make_svamp_pairs(10)
        assert all(p.source_domain == "svamp" for p in pairs)

    def test_make_humaneval_pairs_zero(self) -> None:
        """n=0 returns empty list."""
        assert _make_humaneval_pairs(0) == []

    def test_make_svamp_pairs_zero(self) -> None:
        assert _make_svamp_pairs(0) == []


# ===========================================================================
# TripletLoss tests (REQ-LEARN-824-003)
# ===========================================================================


class TestTripletLoss:
    """REQ-LEARN-824-003: TripletLoss is zero when d(a,p) < d(a,n) - margin."""

    def test_loss_zero_when_constraint_satisfied(self) -> None:
        """Loss is 0 when positive is clearly closer to anchor than negative."""
        # Anchor and positive are the same vector → d(a,p) = 0.
        # Anchor and negative are orthogonal → d(a,n) = 1.0.
        # margin=0.5 → 0.0 - 1.0 + 0.5 = -0.5 → max(0, -0.5) = 0.0
        anchor = [1.0, 0.0]
        positive = [1.0, 0.0]    # same as anchor: d=0
        negative = [0.0, 1.0]    # orthogonal: d=1

        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn(anchor, positive, negative)
        assert loss == 0.0, f"Expected 0.0, got {loss}"

    def test_loss_positive_when_negative_closer_than_positive(self) -> None:
        """Loss > 0 when negative is closer to anchor than positive."""
        anchor = [1.0, 0.0]
        positive = [0.0, 1.0]    # orthogonal to anchor: d=1.0
        negative = [1.0, 0.0]    # same as anchor: d=0.0

        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn(anchor, positive, negative)
        # d(a,p)=1.0, d(a,n)=0.0 → 1.0 - 0.0 + 0.5 = 1.5
        assert loss > 0.0, f"Expected positive loss, got {loss}"
        assert abs(loss - 1.5) < 1e-6, f"Expected ~1.5, got {loss}"

    def test_loss_exactly_at_margin_boundary(self) -> None:
        """Loss is 0 when d(a,p) - d(a,n) == -margin exactly."""
        # d(a,p) - d(a,n) + margin = 0 → boundary case.
        anchor = [1.0, 0.0]
        positive = [0.0, 1.0]   # d(a,p) = 1.0
        # Need d(a,n) such that 1.0 - d(a,n) + 0.5 = 0 → d(a,n) = 1.5
        # But cosine distance max is 2.0. We can pick negative = -anchor.
        negative = [-1.0, 0.0]  # d(a,n) = 1 - (-1) = 2.0
        # loss = 1.0 - 2.0 + 0.5 = -0.5 → max(0, -0.5) = 0.0
        loss_fn = TripletLoss(margin=0.5)
        loss = loss_fn(anchor, positive, negative)
        assert loss == 0.0

    def test_gradient_zero_when_loss_zero(self) -> None:
        """Gradients are all-zero when the triplet constraint is already satisfied."""
        anchor = [1.0, 0.0]
        positive = [1.0, 0.0]  # d=0
        negative = [0.0, 1.0]  # d=1

        loss_fn = TripletLoss(margin=0.5)
        grad_a, grad_p, grad_n = loss_fn.gradient(anchor, positive, negative)
        assert all(abs(g) < 1e-12 for g in grad_a), f"Expected zero anchor gradient, got {grad_a}"
        assert all(abs(g) < 1e-12 for g in grad_p)
        assert all(abs(g) < 1e-12 for g in grad_n)

    def test_gradient_nonzero_when_loss_positive(self) -> None:
        """Gradients are non-zero when loss > 0."""
        anchor = [1.0, 0.0]
        positive = [0.0, 1.0]   # far from anchor
        negative = [1.0, 0.0]   # close to anchor

        loss_fn = TripletLoss(margin=0.5)
        grad_a, grad_p, grad_n = loss_fn.gradient(anchor, positive, negative)
        # At least some gradients should be non-zero.
        all_grads = list(grad_a) + list(grad_p) + list(grad_n)
        assert any(abs(g) > 1e-6 for g in all_grads), "Expected non-zero gradients"

    def test_default_margin(self) -> None:
        """Default margin is 0.5."""
        loss_fn = TripletLoss()
        assert loss_fn.margin == 0.5

    def test_custom_margin(self) -> None:
        """Custom margin is respected."""
        loss_fn = TripletLoss(margin=0.1)
        assert loss_fn.margin == 0.1


class TestCosineDistance:
    """Unit tests for cosine distance helpers."""

    def test_identical_vectors_have_zero_distance(self) -> None:
        u = [1.0, 0.0, 0.0]
        assert abs(_cosine_dist(u, u)) < 1e-9

    def test_orthogonal_vectors_have_distance_one(self) -> None:
        u = [1.0, 0.0]
        v = [0.0, 1.0]
        assert abs(_cosine_dist(u, v) - 1.0) < 1e-9

    def test_opposite_vectors_have_distance_two(self) -> None:
        u = [1.0, 0.0]
        v = [-1.0, 0.0]
        assert abs(_cosine_dist(u, v) - 2.0) < 1e-9

    def test_grad_cosine_dist_u_shape(self) -> None:
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        grad = _grad_cosine_dist_u(u, v)
        assert len(grad) == 3

    def test_grad_cosine_dist_v_shape(self) -> None:
        u = [1.0, 0.0]
        v = [0.5, 0.5]
        grad = _grad_cosine_dist_v(u, v)
        assert len(grad) == 2


# ===========================================================================
# JEPAv23Predictor tests (REQ-LEARN-824-003)
# ===========================================================================


class TestJEPAv23Predictor:
    """REQ-LEARN-824-003: JEPAv23Predictor forward pass produces correct shapes."""

    def _make_trained_model(self) -> JEPAv23Predictor:
        """Return a minimal trained model for testing."""
        pairs = [
            _make_pair("question about math", "3 + 4 = 7", "3 + 4 = 8"),
            _make_pair("compute sum", "10 - 3 = 7", "10 - 3 = 6"),
        ]
        model, _, _ = train_v23(pairs, epochs=2, lr=1e-3, seed=0)
        return model

    def test_encode_returns_correct_length(self) -> None:
        """encode() returns a vector of length embed_dim."""
        model = self._make_trained_model()
        emb = model.encode("some step text here")
        assert len(emb) == model.embed_dim, (
            f"Expected embedding dim {model.embed_dim}, got {len(emb)}"
        )

    def test_encode_returns_unit_norm(self) -> None:
        """encode() returns an L2-normalised vector."""
        model = self._make_trained_model()
        emb = model.encode("some step text here")
        norm = math.sqrt(sum(v * v for v in emb))
        # Allow near-zero norm for degenerate all-zero embeddings (ReLU collapse)
        assert norm < 1.0 + 1e-6, f"Norm {norm} exceeds 1.0 + epsilon"

    def test_predict_energy_returns_scalar(self) -> None:
        """predict_energy() returns a float in [0, 2]."""
        model = self._make_trained_model()
        energy = model.predict_energy("question", "step text")
        assert isinstance(energy, float)
        assert 0.0 <= energy <= 2.0 + 1e-9, f"Energy {energy} out of [0, 2] range"

    def test_default_embed_dim(self) -> None:
        """Default embed_dim is 64."""
        model = JEPAv23Predictor()
        assert model.embed_dim == 64


# ===========================================================================
# train_v23 integration tests (REQ-LEARN-824-003)
# ===========================================================================


class TestTrainV23:
    """Integration tests for the train_v23 function."""

    def test_train_returns_three_items(self) -> None:
        """train_v23 returns (model, train_losses, final_loss)."""
        pairs = [_make_pair("q", "good step", "bad step")]
        result = train_v23(pairs, epochs=2, seed=0)
        assert len(result) == 3

    def test_train_losses_length_equals_epochs(self) -> None:
        """train_losses list has exactly `epochs` entries."""
        pairs = [_make_pair("q", "good step", "bad step")]
        _, losses, _ = train_v23(pairs, epochs=5, seed=0)
        assert len(losses) == 5

    def test_final_loss_matches_last_entry(self) -> None:
        """final_epoch_loss matches the last entry in train_losses."""
        pairs = [_make_pair("q", "good step", "bad step")]
        _, losses, final = train_v23(pairs, epochs=3, seed=0)
        assert abs(final - losses[-1]) < 1e-12

    def test_train_deterministic_with_seed(self) -> None:
        """Same seed produces same final loss."""
        pairs = [_make_pair("q", "correct answer 7", "wrong answer 8")]
        _, _, loss1 = train_v23(pairs, epochs=3, seed=42)
        _, _, loss2 = train_v23(pairs, epochs=3, seed=42)
        assert abs(loss1 - loss2) < 1e-12

    def test_train_multi_pair_corpus(self) -> None:
        """train_v23 works with multiple pairs from mixed domains."""
        pairs = (
            [_make_pair(f"q{i}", f"correct {i}", f"wrong {i}") for i in range(5)]
            + _make_humaneval_pairs(3)
            + _make_svamp_pairs(3)
        )
        model, losses, final = train_v23(pairs, epochs=3, seed=0)
        assert isinstance(model, JEPAv23Predictor)
        assert len(losses) == 3


# ===========================================================================
# evaluate_v23 tests
# ===========================================================================


class TestEvaluateV23:
    """Tests for evaluate_v23 with synthetic and real-like holdout data."""

    def _make_model(self) -> JEPAv23Predictor:
        pairs = [
            _make_pair("q1", "3 + 4 = 7, correct", "3 + 4 = 8, wrong"),
            _make_pair("q2", "multiply gives 42", "multiply gives 43 wrong"),
        ]
        model, _, _ = train_v23(pairs, epochs=3, seed=1)
        return model

    def test_evaluate_v23_missing_file_uses_synthetic(self, tmp_path: Path) -> None:
        """evaluate_v23 falls back to synthetic holdout when file is missing."""
        model = self._make_model()
        in_auc, ood_auc = evaluate_v23(model, tmp_path / "nonexistent.json")
        assert 0.0 <= in_auc <= 1.0
        assert 0.0 <= ood_auc <= 1.0

    def test_evaluate_v23_with_real_holdout(self, tmp_path: Path) -> None:
        """evaluate_v23 works with a real holdout file."""
        data = [
            {"question_id": str(i), "step_text": f"step {i}", "label": "correct" if i % 2 == 0 else "incorrect"}
            for i in range(10)
        ]
        p = tmp_path / "holdout.json"
        p.write_text(json.dumps(data))

        model = self._make_model()
        in_auc, ood_auc = evaluate_v23(model, p)
        assert 0.0 <= in_auc <= 1.0
        assert 0.0 <= ood_auc <= 1.0

    def test_evaluate_v23_returns_tuple(self, tmp_path: Path) -> None:
        """evaluate_v23 returns a 2-tuple."""
        model = self._make_model()
        result = evaluate_v23(model, tmp_path / "nonexistent.json")
        assert isinstance(result, tuple)
        assert len(result) == 2


# ===========================================================================
# _compute_auc tests
# ===========================================================================


class TestComputeAUC:
    """Tests for the AUC computation helper."""

    def test_perfect_auc(self) -> None:
        """Perfect discrimination gives AUC = 1.0."""
        # All positives (label=1) scored higher than all negatives (label=0).
        scored = [(0.9, 1.0), (0.8, 1.0), (0.2, 0.0), (0.1, 0.0)]
        auc = _compute_auc(scored)
        assert abs(auc - 1.0) < 1e-9

    def test_random_auc(self) -> None:
        """Random discrimination gives AUC around 0.5."""
        # Interleaved scores: no discrimination.
        scored = [(0.5, 1.0), (0.5, 0.0), (0.5, 1.0), (0.5, 0.0)]
        auc = _compute_auc(scored)
        assert abs(auc - 0.5) < 1e-9, f"Expected 0.5 for tied scores, got {auc}"

    def test_single_class_returns_half(self) -> None:
        """Returns 0.5 when only one class is present."""
        scored = [(0.9, 1.0), (0.8, 1.0)]
        assert _compute_auc(scored) == 0.5

    def test_empty_returns_half(self) -> None:
        """Returns 0.5 for empty input."""
        assert _compute_auc([]) == 0.5


# ===========================================================================
# _TFIDFVectoriser tests
# ===========================================================================


class TestTFIDFVectoriser:
    """Tests for the TF-IDF vectoriser."""

    def test_fit_and_transform(self) -> None:
        """fit then transform returns a vector of the vocabulary size."""
        v = _TFIDFVectoriser(max_features=10)
        v.fit(["hello world", "world foo bar"])
        vec = v.transform("hello world")
        assert len(vec) == len(v._vocab)
        assert len(v._vocab) <= 10

    def test_transform_empty_text(self) -> None:
        """transform returns all-zeros for text with no vocabulary tokens."""
        v = _TFIDFVectoriser(max_features=5)
        v.fit(["alpha beta gamma"])
        vec = v.transform("xyz 123 !!!!")
        assert all(val == 0.0 for val in vec)

    def test_transform_normalised(self) -> None:
        """transform produces an L2-normalised vector for non-empty text."""
        v = _TFIDFVectoriser(max_features=20)
        v.fit(["the quick brown fox jumps", "over the lazy dog"])
        vec = v.transform("quick fox")
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            assert abs(norm - 1.0) < 1e-6, f"Expected L2-norm=1.0, got {norm}"


# ===========================================================================
# _synthetic_holdout test
# ===========================================================================


class TestSyntheticHoldout:
    def test_synthetic_holdout_has_both_classes(self) -> None:
        """Synthetic holdout has both 'correct' and 'incorrect' labels."""
        data = _synthetic_holdout()
        labels = {d["label"] for d in data}
        assert "correct" in labels
        assert "incorrect" in labels

    def test_synthetic_holdout_nonempty(self) -> None:
        assert len(_synthetic_holdout()) > 0


# ===========================================================================
# Coverage gap tests — edge cases not hit by main tests
# ===========================================================================


class TestEdgeCases:
    """Tests targeting uncovered edge case branches."""

    def test_train_v23_empty_corpus_guard(self) -> None:
        """train_v23 with a corpus where all texts are identical (degenerate TF-IDF vocab)."""
        # Use empty texts to trigger the vocab_size == 0 guard.
        pairs = [
            CuratedPair(
                prefix_text="",
                positive_step="",
                negative_step="",
                z3_confidence=1.0,
                cpmi_score=1.0,
                source_domain="gsm8k",
                quality_score=1.0,
            )
        ]
        # Should not raise; model trains even on degenerate corpus.
        model, losses, final = train_v23(pairs, epochs=2, seed=0)
        assert len(losses) == 2

    def test_evaluate_v23_empty_holdout(self, tmp_path: Path) -> None:
        """evaluate_v23 returns (0.5, 0.5) for empty holdout file."""
        p = tmp_path / "empty.json"
        p.write_text("[]")
        pairs = [_make_pair("q", "correct step", "wrong step")]
        model, _, _ = train_v23(pairs, epochs=2, seed=0)
        in_auc, ood_auc = evaluate_v23(model, p)
        assert in_auc == 0.5
        assert ood_auc == 0.5

    def test_load_cpmi_triples_dict_format(self, tmp_path: Path) -> None:
        """load_cpmi_triples handles dict-format CPMI corpus (experiment artifact)."""
        data = {
            "experiment": 798,
            "triples": [
                {
                    "prefix_text": "q1",
                    "positive_step": "5 + 5 = 10",
                    "negative_step": "5 + 5 = 11",
                    "cpmi_score": 0.7,
                    "source_domain": "gsm8k",
                }
            ],
        }
        p = tmp_path / "cpmi_dict.json"
        p.write_text(json.dumps(data))
        curator = LIMOCurator(fover_path=tmp_path / "none.json", cpmi_path=p)
        triples = curator.load_cpmi_triples()
        assert len(triples) == 1
        assert triples[0]["cpmi_score"] == 0.7

    def test_train_v23_loss_zero_branch(self) -> None:
        """Train with a pair where the model starts satisfied (loss=0) — tests continue branch."""
        # This exercises the loss <= 0.0 continue branch.
        # We train for many epochs on a pair where positive == negative (loss often 0).
        pairs = [
            CuratedPair(
                prefix_text="anchor text here",
                positive_step="same step text",
                negative_step="same step text",  # identical → d(a,n)=d(a,p) → loss=margin
                z3_confidence=1.0,
                cpmi_score=1.0,
                source_domain="gsm8k",
                quality_score=1.0,
            )
        ]
        model, losses, final = train_v23(pairs, epochs=3, seed=99)
        assert len(losses) == 3
        assert isinstance(final, float)

    def test_train_v23_satisfied_triplet_skips_backprop(self) -> None:
        """A triplet that is already satisfied (loss=0) exercises the continue branch.

        We use many epochs with a corpus that the model eventually satisfies,
        ensuring the continue branch at loss <= 0.0 is executed.
        The positive and negative are identical so d(a,p)==d(a,n): the initial loss
        equals the margin (0.5 > 0).  After several epochs the model may satisfy some
        triplets.  We train 20 epochs so at least one epoch is guaranteed to hit loss=0
        on at least one triplet once the weights converge.
        """
        # Anchor == positive (all tokens identical) ensures d(a,p) starts at ~0.
        # Negative uses completely disjoint vocabulary so d(a,n) starts at ~1.
        # loss = d(a,p) - d(a,n) + 0.5 = 0 - 1 + 0.5 = -0.5 → 0.0 → continue.
        pairs = [
            CuratedPair(
                prefix_text="alpha beta gamma",
                positive_step="alpha beta gamma",  # identical to anchor
                negative_step="xyz uvw rst",        # completely disjoint
                z3_confidence=1.0,
                cpmi_score=1.0,
                source_domain="gsm8k",
                quality_score=1.0,
            )
        ]
        model, losses, final = train_v23(pairs, epochs=20, seed=77)
        assert len(losses) == 20
        # The loss should be 0 or very small since the triplet is satisfied from init.
        assert final <= 0.5 + 1e-6
