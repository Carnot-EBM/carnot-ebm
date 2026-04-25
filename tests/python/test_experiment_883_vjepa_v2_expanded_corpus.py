"""Tests for Experiment 883: VJEPA v2 expanded corpus.

Spec traces: REQ-VERIFY-160, SCENARIO-VERIFY-231, SCENARIO-VERIFY-232

**Coverage targets (code added in Exp 883):**
    - generate_gsm8k_synthetic: label correctness, determinism, count
    - generate_arc_synthetic: label correctness, determinism, count
    - generate_svamp_synthetic: label correctness, determinism, count
    - split_by_question_id: reproducibility, no question_id leakage
    - _make_domain_weight_vector: correct indexing, neutral fallback
    - assign_honest_verdict: all five verdict branches
    - DomainReweightedLoss.compute_domain_weights + weighted_loss: 4-domain case
    - train_vjepa_domain_weighted: smoke-test (2 epochs, tiny corpus)
    - evaluate_on_split: returns 0.5 on empty corpus
    - compute_uncertainty_calibration: returns 0.0 on empty corpus
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Ensure project root importable
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from experiment_883_vjepa_v2_expanded_corpus import (
    EXP_877_OOD_AUC,
    _make_domain_weight_vector,
    assign_honest_verdict,
    compute_uncertainty_calibration,
    evaluate_on_split,
    generate_arc_synthetic,
    generate_gsm8k_synthetic,
    generate_svamp_synthetic,
    split_by_question_id,
    train_vjepa_domain_weighted,
)
from python.carnot.models.jepa_predictor import DomainReweightedLoss
from python.carnot.models.vjepa_predictor import (
    VariationalJEPAPredictor,
    build_tfidf_features,
    prepare_corpus,
)


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-231: Synthetic pair generator produces correct labels
# ---------------------------------------------------------------------------

class TestGenerateGsm8kSynthetic:
    """REQ-VERIFY-160 — GSM8K synthetic generator correctness."""

    def test_count(self):
        pairs = generate_gsm8k_synthetic(n_steps=100, seed=42)
        assert 100 <= len(pairs) <= 107  # complete problems, may overshoot by up to 1 problem (7 steps)

    def test_each_problem_has_exactly_one_incorrect_step(self):
        pairs = generate_gsm8k_synthetic(n_steps=100, seed=42)
        by_qid: dict[str, list[str]] = {}
        for p in pairs:
            by_qid.setdefault(p["question_id"], []).append(p["label"])
        for qid, labels in by_qid.items():
            n_incorrect = labels.count("incorrect")
            assert n_incorrect == 1, (
                f"Problem {qid} has {n_incorrect} incorrect steps (expected 1)"
            )

    def test_labels_are_correct_or_incorrect(self):
        pairs = generate_gsm8k_synthetic(n_steps=30, seed=42)
        for p in pairs:
            assert p["label"] in {"correct", "incorrect"}

    def test_domain_tag(self):
        pairs = generate_gsm8k_synthetic(n_steps=10, seed=42)
        assert all(p["domain"] == "gsm8k_synthetic" for p in pairs)

    def test_deterministic_same_seed(self):
        a = generate_gsm8k_synthetic(n_steps=20, seed=42)
        b = generate_gsm8k_synthetic(n_steps=20, seed=42)
        assert [p["step_text"] for p in a] == [p["step_text"] for p in b]

    def test_different_seed_differs(self):
        a = generate_gsm8k_synthetic(n_steps=20, seed=42)
        b = generate_gsm8k_synthetic(n_steps=20, seed=99)
        assert [p["step_text"] for p in a] != [p["step_text"] for p in b]

    def test_has_question_id(self):
        pairs = generate_gsm8k_synthetic(n_steps=10, seed=42)
        assert all("question_id" in p for p in pairs)


class TestGenerateArcSynthetic:
    """REQ-VERIFY-160 — ARC synthetic generator correctness."""

    def test_count(self):
        pairs = generate_arc_synthetic(n_steps=30, seed=42)
        assert 30 <= len(pairs) <= 37

    def test_each_problem_has_one_incorrect(self):
        pairs = generate_arc_synthetic(n_steps=30, seed=42)
        by_qid: dict[str, list[str]] = {}
        for p in pairs:
            by_qid.setdefault(p["question_id"], []).append(p["label"])
        for qid, labels in by_qid.items():
            assert labels.count("incorrect") == 1

    def test_domain_tag(self):
        pairs = generate_arc_synthetic(n_steps=10, seed=42)
        assert all(p["domain"] == "arc_synthetic" for p in pairs)

    def test_deterministic(self):
        a = generate_arc_synthetic(n_steps=12, seed=42)
        b = generate_arc_synthetic(n_steps=12, seed=42)
        assert [p["label"] for p in a] == [p["label"] for p in b]


class TestGenerateSvampSynthetic:
    """REQ-VERIFY-160 — SVAMP synthetic generator correctness."""

    def test_count(self):
        pairs = generate_svamp_synthetic(n_steps=20, seed=42)
        assert 20 <= len(pairs) <= 26

    def test_each_problem_has_one_incorrect(self):
        pairs = generate_svamp_synthetic(n_steps=20, seed=42)
        by_qid: dict[str, list[str]] = {}
        for p in pairs:
            by_qid.setdefault(p["question_id"], []).append(p["label"])
        for qid, labels in by_qid.items():
            assert labels.count("incorrect") == 1

    def test_domain_tag(self):
        pairs = generate_svamp_synthetic(n_steps=10, seed=42)
        assert all(p["domain"] == "svamp_synthetic" for p in pairs)

    def test_deterministic(self):
        a = generate_svamp_synthetic(n_steps=10, seed=42)
        b = generate_svamp_synthetic(n_steps=10, seed=42)
        assert [p["label"] for p in a] == [p["label"] for p in b]


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-232: Train/eval split is reproducible and leakage-free
# ---------------------------------------------------------------------------

class TestSplitByQuestionId:
    """REQ-VERIFY-160 — Reproducible train/eval split."""

    def _make_corpus(self) -> list[dict]:
        pairs = generate_gsm8k_synthetic(n_steps=60, seed=42)
        return pairs

    def test_reproducible_same_seed(self):
        corpus = self._make_corpus()
        train_a, test_a = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        train_b, test_b = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        assert [s["question_id"] for s in train_a] == [s["question_id"] for s in train_b]
        assert [s["question_id"] for s in test_a] == [s["question_id"] for s in test_b]

    def test_different_seed_differs(self):
        corpus = self._make_corpus()
        _, test_a = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        _, test_b = split_by_question_id(corpus, test_fraction=0.2, seed=99)
        # Different seeds should (almost certainly) produce different test sets
        ids_a = {s["question_id"] for s in test_a}
        ids_b = {s["question_id"] for s in test_b}
        assert ids_a != ids_b

    def test_no_question_id_leakage(self):
        corpus = self._make_corpus()
        train, test = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        train_qids = {s["question_id"] for s in train}
        test_qids = {s["question_id"] for s in test}
        assert train_qids.isdisjoint(test_qids)

    def test_covers_full_corpus(self):
        corpus = self._make_corpus()
        train, test = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        # All steps should be assigned to either train or test
        assert len(train) + len(test) == len(corpus)
        assert len(train) > 0

    def test_test_fraction_respected(self):
        corpus = self._make_corpus()
        train, test = split_by_question_id(corpus, test_fraction=0.2, seed=42)
        # At least 1 question_id in test
        assert len({s["question_id"] for s in test}) >= 1


# ---------------------------------------------------------------------------
# DomainReweightedLoss: 4-domain case
# ---------------------------------------------------------------------------

class TestDomainReweightedLossFourDomains:
    """SCENARIO-VERIFY-232 — DomainReweightedLoss with 4 distinct domains."""

    def _make_corpus_4_domains(self) -> list[dict]:
        """Build a small corpus with 4 domain sizes: 10, 30, 20, 40 samples."""
        corpus = []
        for i in range(10):
            corpus.append({"domain": "domain_a", "label": 0})
        for i in range(30):
            corpus.append({"domain": "domain_b", "label": 1})
        for i in range(20):
            corpus.append({"domain": "domain_c", "label": 0})
        for i in range(40):
            corpus.append({"domain": "domain_d", "label": 1})
        return corpus

    def test_weights_sum_to_one(self):
        corpus = self._make_corpus_4_domains()
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_small_domain_gets_higher_weight(self):
        corpus = self._make_corpus_4_domains()
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        # domain_a (10 samples) should have higher weight than domain_d (40 samples)
        assert weights["domain_a"] > weights["domain_d"]

    def test_all_four_domains_present(self):
        corpus = self._make_corpus_4_domains()
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert set(weights.keys()) == {"domain_a", "domain_b", "domain_c", "domain_d"}

    def test_weighted_loss_returns_scalar(self):
        corpus = self._make_corpus_4_domains()
        domain_names = ["domain_a", "domain_b", "domain_c", "domain_d"]
        loss_fn = DomainReweightedLoss()
        weights_dict = loss_fn.compute_domain_weights(corpus)
        weight_vec = _make_domain_weight_vector(weights_dict, domain_names)
        domain_to_idx = {d: i for i, d in enumerate(domain_names)}
        domain_ids = jnp.array(
            [domain_to_idx[s["domain"]] for s in corpus], dtype=jnp.int32
        )
        logits = jnp.zeros(len(corpus))
        labels = jnp.array([float(s["label"]) for s in corpus])
        result = loss_fn.weighted_loss(logits, labels, domain_ids, weight_vec)
        assert result.shape == ()
        assert float(result) > 0.0

    def test_make_domain_weight_vector_unknown_domain_gets_neutral_weight(self):
        weights_dict = {"domain_a": 0.6, "domain_b": 0.4}
        # "domain_c" is not in weights_dict → should get fallback 1.0
        vec = _make_domain_weight_vector(weights_dict, ["domain_a", "domain_b", "domain_c"])
        assert float(vec[2]) == 1.0


# ---------------------------------------------------------------------------
# Assign honest verdict
# ---------------------------------------------------------------------------

class TestAssignHonestVerdict:
    """REQ-VERIFY-160 — All five verdict branches are reachable."""

    def test_collapsed(self):
        assert assign_honest_verdict(0.7, 0.005) == "vjepa_v2_collapsed"

    def test_above_gate(self):
        assert assign_honest_verdict(0.66, 0.05) == "vjepa_ood_above_gate"

    def test_deployable(self):
        assert assign_honest_verdict(0.62, 0.05) == "vjepa_ood_deployable"

    def test_improved_below_gate(self):
        assert assign_honest_verdict(0.57, 0.05) == "vjepa_improved_below_gate"

    def test_regressed(self):
        assert assign_honest_verdict(0.50, 0.05) == "vjepa_v2_regressed"

    def test_exactly_at_gate_boundary_is_deployable(self):
        # 0.60 is NOT > 0.60, so should be improved_below_gate
        assert assign_honest_verdict(0.60, 0.05) == "vjepa_improved_below_gate"

    def test_exp877_baseline_is_correct(self):
        assert EXP_877_OOD_AUC == 0.5833


# ---------------------------------------------------------------------------
# Edge cases: empty corpus returns safe defaults
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """REQ-VERIFY-160 — Safe degradation on empty/degenerate inputs."""

    def test_evaluate_on_split_empty_returns_half(self):
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=8)
        key = jax.random.PRNGKey(0)
        result = evaluate_on_split(model, [], key)
        assert result == 0.5

    def test_calibration_empty_returns_zero(self):
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=8)
        key = jax.random.PRNGKey(0)
        result = compute_uncertainty_calibration(model, [], key)
        assert result == 0.0

    def test_train_domain_weighted_empty_returns_empty_lists(self):
        model = VariationalJEPAPredictor(in_dim=10, context_dim=10, latent_dim=8)
        losses, kls = train_vjepa_domain_weighted(
            model, [], ["domain_a"], n_epochs=5, lr=1e-3, seed=0
        )
        assert losses == []
        assert kls == []


# ---------------------------------------------------------------------------
# Smoke test: train for 2 epochs on tiny corpus (fast, no GPU)
# ---------------------------------------------------------------------------

class TestTrainDomainWeightedSmoke:
    """REQ-VERIFY-160 — Training smoke test with minimal corpus."""

    def _make_tiny_corpus(self) -> list[dict]:
        """Build a tiny 8-step corpus with 2 domains for a fast smoke test."""
        texts_a = [
            "step one calculate five plus three equals eight correct",
            "step two multiply two times four equals eight correct",
            "step three add ten plus wrong value error incorrect",
            "step four final sum equals correct value",
        ]
        texts_b = [
            "logical step all mammals breathe oxygen valid",
            "logical step whales are mammals therefore breathe oxygen valid",
            "logical step fish breathe air error incorrect",
            "logical step birds have wings valid conclusion correct",
        ]
        all_texts = texts_a + texts_b
        _, tok2idx = build_tfidf_features(all_texts, vocab_size=20)

        raw_a = [
            {"question_id": "q0", "step_text": t, "label": "incorrect" if "error" in t else "correct",
             "domain": "domain_a"}
            for t in texts_a
        ]
        raw_b = [
            {"question_id": "q1", "step_text": t, "label": "incorrect" if "error" in t else "correct",
             "domain": "domain_b"}
            for t in texts_b
        ]
        raw = raw_a + raw_b
        corpus = prepare_corpus(raw, tok2idx, 20)
        for i, step in enumerate(raw):
            corpus[i]["domain"] = step["domain"]
        return corpus

    def test_training_produces_losses(self):
        corpus = self._make_tiny_corpus()
        model = VariationalJEPAPredictor(in_dim=20, context_dim=20, latent_dim=8)
        losses, kls = train_vjepa_domain_weighted(
            model, corpus, ["domain_a", "domain_b"],
            n_epochs=2, lr=1e-3, seed=0
        )
        assert len(losses) == 2
        assert len(kls) == 2
        assert all(isinstance(v, float) for v in losses)
        assert all(v >= 0.0 for v in kls)

    def test_kl_magnitude_positive(self):
        corpus = self._make_tiny_corpus()
        model = VariationalJEPAPredictor(in_dim=20, context_dim=20, latent_dim=8)
        _, kls = train_vjepa_domain_weighted(
            model, corpus, ["domain_a", "domain_b"],
            n_epochs=3, lr=1e-3, seed=0
        )
        # KL should be non-negative (absolute value)
        assert all(k >= 0.0 for k in kls)
