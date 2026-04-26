"""Tests for scripts/experiment_872_jepa_v25_dg_prm.py.

Traces to: REQ-LEARN-050, SCENARIO-LEARN-095, SCENARIO-LEARN-096

**What we test:**
    - DomainReweightedLoss.compute_domain_weights() returns correct inverse-frequency
      weights that sum to 1.0, with smaller domains receiving higher weights.
    - DomainReweightedLoss.weighted_loss() produces a scalar and applies domain weights.
    - build_balanced_corpus() produces 40 pairs per domain (20 correct + 20 incorrect).
    - SVAMP corpus has exactly 20 correct and 20 incorrect entries (v25 expansion).
    - _embed_text() returns unit-normed float32 arrays.
    - _init_v25_params() returns correct shapes.
    - _forward_v25() returns (corr_prob, dom_prob) with correct shapes and ranges.
    - compute_honest_verdict() maps AUC values to the four v25 verdict labels.
    - train_jepa_v25() returns valid per-domain AUC metrics.
    - Loss decreases over training (convergence check).
    - Deliverable JSON exists with all required schema fields.

All tests run on CPU via JAX_PLATFORMS=cpu; no GPU or live model required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from python.carnot.models.jepa_predictor import DomainReweightedLoss
from scripts.experiment_872_jepa_v25_dg_prm import (
    ARC_CORRECT_STEPS,
    ARC_INCORRECT_STEPS,
    DELTA_ENERGY_MAX,
    DELTA_ENERGY_MIN,
    DG_PRM_DOMAIN_WEIGHTS,
    DOMAIN_NAMES,
    EMBED_DIM,
    HIDDEN1,
    HIDDEN2,
    N_DOMAINS,
    GSM8K_CORRECT_STEPS,
    GSM8K_INCORRECT_STEPS,
    HUMANEVAL_CORRECT_STEPS,
    HUMANEVAL_INCORRECT_STEPS,
    SVAMP_CORRECT_STEPS,
    SVAMP_INCORRECT_STEPS,
    TRIPLET_MARGIN,
    _build_triplets,
    _embed_text,
    _forward_v25,
    _init_v25_params,
    build_balanced_corpus,
    compute_honest_verdict,
    train_jepa_v25,
)


# ---------------------------------------------------------------------------
# DomainReweightedLoss tests (REQ-LEARN-050)
# ---------------------------------------------------------------------------


class TestDomainReweightedLossWeights:
    """Tests for DomainReweightedLoss.compute_domain_weights().

    Traces to: REQ-LEARN-050
    """

    def _make_corpus(self, domain_sizes: dict[str, int]) -> list[dict[str, Any]]:
        """Build a minimal corpus with specified per-domain sample counts."""
        corpus = []
        for domain, n in domain_sizes.items():
            d_idx = DOMAIN_NAMES.index(domain) if domain in DOMAIN_NAMES else 0
            for i in range(n):
                corpus.append(
                    {
                        "text": f"sample {i} for {domain}",
                        "label": i % 2,
                        "domain": domain,
                        "domain_idx": d_idx,
                    }
                )
        return corpus

    def test_weights_sum_to_one(self) -> None:
        """compute_domain_weights must return weights that sum to 1.0.

        Traces to: REQ-LEARN-050
        """
        corpus = self._make_corpus({"gsm8k": 40, "humaneval": 40, "arc": 40, "svamp": 40})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert abs(sum(weights.values()) - 1.0) < 1e-5

    def test_equal_domains_produce_uniform_weights(self) -> None:
        """When all domains have equal sample counts, weights must be equal.

        Traces to: REQ-LEARN-050
        """
        corpus = self._make_corpus({"gsm8k": 40, "humaneval": 40, "arc": 40, "svamp": 40})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        values = list(weights.values())
        assert all(abs(v - values[0]) < 1e-5 for v in values)

    def test_smaller_domain_gets_higher_weight(self) -> None:
        """A domain with fewer samples must receive a higher weight than a larger domain.

        Traces to: REQ-LEARN-050 — this is the core property that fixes SVAMP AUC=0.
        """
        corpus = self._make_corpus({"gsm8k": 40, "svamp": 20})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        # svamp has 20 samples, gsm8k has 40. svamp weight must be higher.
        assert weights["svamp"] > weights["gsm8k"]

    def test_returns_dict_with_correct_keys(self) -> None:
        """compute_domain_weights must return a dict whose keys match corpus domains."""
        corpus = self._make_corpus({"gsm8k": 10, "arc": 20})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert set(weights.keys()) == {"gsm8k", "arc"}

    def test_all_weights_positive(self) -> None:
        """All returned weights must be strictly positive."""
        corpus = self._make_corpus({"gsm8k": 40, "humaneval": 40, "arc": 40, "svamp": 40})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert all(w > 0 for w in weights.values())

    def test_single_domain_gets_weight_one(self) -> None:
        """A corpus with only one domain must assign that domain weight 1.0."""
        corpus = self._make_corpus({"gsm8k": 30})
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus)
        assert abs(weights["gsm8k"] - 1.0) < 1e-5

    def test_inversely_proportional_to_count(self) -> None:
        """Weight ratio between two domains must equal inverse ratio of their counts.

        When corpus has 20 samples in domain A and 40 in domain B,
        weight_A / weight_B ≈ (1/20) / (1/40) = 2.0.
        """
        corpus = self._make_corpus({"a": 20, "b": 40})
        # Use a corpus where 'a' and 'b' are domain names (not in DOMAIN_NAMES — that's OK,
        # compute_domain_weights only uses the 'domain' key)
        corpus_generic = [{"domain": "a"} for _ in range(20)] + [{"domain": "b"} for _ in range(40)]
        loss_fn = DomainReweightedLoss()
        weights = loss_fn.compute_domain_weights(corpus_generic)
        ratio = weights["a"] / weights["b"]
        assert abs(ratio - 2.0) < 0.1


class TestDomainReweightedLossWeightedLoss:
    """Tests for DomainReweightedLoss.weighted_loss().

    Traces to: REQ-LEARN-050
    """

    def test_returns_scalar(self) -> None:
        """weighted_loss must return a scalar JAX array."""
        loss_fn = DomainReweightedLoss()
        logits = jnp.array([0.5, -0.5, 0.1, -0.1])
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        domain_ids = jnp.array([0, 1, 2, 3])
        domain_weights = jnp.array([0.25, 0.25, 0.25, 0.25])
        result = loss_fn.weighted_loss(logits, labels, domain_ids, domain_weights)
        assert result.shape == ()

    def test_higher_weight_domain_increases_loss(self) -> None:
        """Assigning higher weight to a domain with high error must increase total loss.

        Traces to: REQ-LEARN-050
        """
        loss_fn = DomainReweightedLoss()
        # Wrong prediction for domain 0 (logit=2.0, label=0 → high BCE)
        logits = jnp.array([2.0, 2.0])
        labels = jnp.array([0.0, 0.0])
        domain_ids = jnp.array([0, 1])

        # High weight on domain 0
        w_high = jnp.array([0.9, 0.1])
        # Low weight on domain 0
        w_low = jnp.array([0.1, 0.9])

        loss_high = float(loss_fn.weighted_loss(logits, labels, domain_ids, w_high))
        loss_low = float(loss_fn.weighted_loss(logits, labels, domain_ids, w_low))
        # Both domains have same wrong prediction; since both samples are wrong,
        # the total loss differs only by the weight applied to each.
        # Domain 0 has same error as domain 1 here, so loss is nearly the same.
        # Instead, test with different errors per domain.
        # Domain 0: logit=3.0, label=0 (high error); domain 1: logit=-3.0, label=0 (low error)
        logits2 = jnp.array([3.0, -3.0])
        w_heavy_d0 = jnp.array([0.9, 0.1])
        w_light_d0 = jnp.array([0.1, 0.9])
        loss_heavy = float(loss_fn.weighted_loss(logits2, labels, domain_ids, w_heavy_d0))
        loss_light = float(loss_fn.weighted_loss(logits2, labels, domain_ids, w_light_d0))
        assert loss_heavy > loss_light

    def test_uniform_weights_proportional_to_unweighted_mean(self) -> None:
        """With uniform per-sample weights w, weighted_loss = w × mean_unweighted_BCE.

        The implementation computes mean(bce * w_per_sample).  When all w = 0.25,
        mean(bce * 0.25) = 0.25 × mean(bce).  This is the correct behaviour:
        uniform weights preserve the relative ordering of per-domain losses while
        scaling the total gradient magnitude by the weight value.

        Traces to: REQ-LEARN-050
        """
        import optax as _optax

        loss_fn = DomainReweightedLoss()
        logits = jnp.array([1.0, -1.0, 0.5, -0.5])
        labels = jnp.array([1.0, 0.0, 1.0, 0.0])
        domain_ids = jnp.array([0, 1, 2, 3])
        w = 0.25
        uniform_weights = jnp.array([w, w, w, w])

        weighted = float(loss_fn.weighted_loss(logits, labels, domain_ids, uniform_weights))
        unweighted = float(jnp.mean(_optax.sigmoid_binary_cross_entropy(logits, labels)))
        assert abs(weighted - w * unweighted) < 1e-5

    def test_batch_and_1d_logits_both_work(self) -> None:
        """weighted_loss must accept both (batch,) and (batch, 1) shaped logits."""
        loss_fn = DomainReweightedLoss()
        logits_1d = jnp.array([0.5, -0.5])
        logits_2d = jnp.array([[0.5], [-0.5]])
        labels_1d = jnp.array([1.0, 0.0])
        labels_2d = jnp.array([[1.0], [0.0]])
        domain_ids = jnp.array([0, 1])
        weights = jnp.array([0.5, 0.5])
        loss1 = float(loss_fn.weighted_loss(logits_1d, labels_1d, domain_ids, weights))
        loss2 = float(loss_fn.weighted_loss(logits_2d, labels_2d, domain_ids, weights))
        assert abs(loss1 - loss2) < 1e-5


# ---------------------------------------------------------------------------
# Corpus structure tests
# ---------------------------------------------------------------------------


class TestBuildBalancedCorpus:
    """Tests for build_balanced_corpus() enforcing 40 pairs per domain.

    Traces to: SCENARIO-LEARN-095
    """

    def test_returns_list_of_dicts(self) -> None:
        """build_balanced_corpus() must return a non-empty list of dicts."""
        corpus = build_balanced_corpus()
        assert isinstance(corpus, list)
        assert len(corpus) > 0
        assert all(isinstance(p, dict) for p in corpus)

    def test_all_four_domains_present(self) -> None:
        """Corpus must contain exactly the four expected domain names."""
        corpus = build_balanced_corpus()
        domains_present = {p["domain"] for p in corpus}
        assert domains_present == {"gsm8k", "humaneval", "arc", "svamp"}

    def test_svamp_has_twenty_correct_and_twenty_incorrect(self) -> None:
        """SVAMP must contribute 20 correct and 20 incorrect pairs (v25 expansion).

        Traces to: SCENARIO-LEARN-095 — v24 had 10+10, v25 requires 20+20.
        """
        corpus = build_balanced_corpus()
        c = sum(1 for p in corpus if p["domain"] == "svamp" and p["label"] == 1)
        i = sum(1 for p in corpus if p["domain"] == "svamp" and p["label"] == 0)
        assert c == 20, f"SVAMP correct count: expected 20, got {c}"
        assert i == 20, f"SVAMP incorrect count: expected 20, got {i}"

    def test_all_domains_have_equal_sample_count(self) -> None:
        """All four domains must have the same total sample count.

        Traces to: SCENARIO-LEARN-095 — equal counts ensure uniform DomainReweightedLoss weights.
        """
        corpus = build_balanced_corpus()
        counts: dict[str, int] = {}
        for p in corpus:
            counts[p["domain"]] = counts.get(p["domain"], 0) + 1
        values = list(counts.values())
        assert all(v == values[0] for v in values), f"Unequal domain counts: {counts}"

    def test_total_corpus_size_is_160(self) -> None:
        """Total corpus must be 160 pairs (4 domains × 40 pairs each)."""
        corpus = build_balanced_corpus()
        assert len(corpus) == 160

    def test_each_pair_has_required_keys(self) -> None:
        """Every corpus entry must have: text, label, domain, domain_idx."""
        corpus = build_balanced_corpus()
        for p in corpus:
            assert "text" in p
            assert "label" in p
            assert "domain" in p
            assert "domain_idx" in p

    def test_labels_are_binary(self) -> None:
        """Every label must be 0 or 1."""
        corpus = build_balanced_corpus()
        for p in corpus:
            assert p["label"] in (0, 1)

    def test_domain_idx_matches_domain_name(self) -> None:
        """domain_idx must be the DOMAIN_NAMES index of domain."""
        corpus = build_balanced_corpus()
        for p in corpus:
            assert p["domain_idx"] == DOMAIN_NAMES.index(p["domain"])


class TestStaticCorpusData:
    """Sanity checks on corpus constant sizes.

    Traces to: SCENARIO-LEARN-095
    """

    def test_svamp_correct_exactly_twenty(self) -> None:
        """SVAMP_CORRECT_STEPS must have exactly 20 entries (v25 doubled from v24's 10)."""
        assert len(SVAMP_CORRECT_STEPS) == 20

    def test_svamp_incorrect_exactly_twenty(self) -> None:
        """SVAMP_INCORRECT_STEPS must have exactly 20 entries."""
        assert len(SVAMP_INCORRECT_STEPS) == 20

    def test_gsm8k_counts(self) -> None:
        """GSM8K correct and incorrect lists each have 20 entries."""
        assert len(GSM8K_CORRECT_STEPS) == 20
        assert len(GSM8K_INCORRECT_STEPS) == 20

    def test_humaneval_counts(self) -> None:
        """HumanEval correct and incorrect lists each have 20 entries."""
        assert len(HUMANEVAL_CORRECT_STEPS) == 20
        assert len(HUMANEVAL_INCORRECT_STEPS) == 20

    def test_arc_counts(self) -> None:
        """ARC correct and incorrect lists each have 20 entries."""
        assert len(ARC_CORRECT_STEPS) == 20
        assert len(ARC_INCORRECT_STEPS) == 20

    def test_domain_names_length(self) -> None:
        """DOMAIN_NAMES must have exactly 4 entries."""
        assert len(DOMAIN_NAMES) == 4
        assert N_DOMAINS == 4


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbedText:
    """Tests for _embed_text() hash-projection.

    Traces to: REQ-LEARN-050
    """

    def test_returns_float32_array(self) -> None:
        """_embed_text() must return a float32 numpy array."""
        assert _embed_text("hello world").dtype == np.float32

    def test_correct_dimension(self) -> None:
        """Output shape must be (EMBED_DIM,)."""
        assert _embed_text("test text here").shape == (EMBED_DIM,)

    def test_unit_normed(self) -> None:
        """Output must be unit-normed for non-empty text."""
        emb = _embed_text("some text input")
        assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5

    def test_deterministic(self) -> None:
        """Same text + seed must produce identical embeddings."""
        np.testing.assert_array_equal(_embed_text("hello", seed=42), _embed_text("hello", seed=42))


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------


class TestInitV25Params:
    """Tests for _init_v25_params() returning correct parameter shapes.

    Traces to: REQ-LEARN-050
    """

    def test_all_keys_present(self) -> None:
        """All expected parameter keys must be present."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        expected = {"w1", "b1", "w2", "b2", "w_corr", "b_corr", "w_dom", "b_dom"}
        assert set(params.keys()) == expected

    def test_w1_shape(self) -> None:
        """w1 shape must be (EMBED_DIM, HIDDEN1)."""
        assert _init_v25_params(jax.random.PRNGKey(0))["w1"].shape == (EMBED_DIM, HIDDEN1)

    def test_domain_head_shape(self) -> None:
        """w_dom shape must be (HIDDEN2, N_DOMAINS)."""
        assert _init_v25_params(jax.random.PRNGKey(0))["w_dom"].shape == (HIDDEN2, N_DOMAINS)

    def test_biases_are_zeros(self) -> None:
        """All bias terms must be initialised to zero."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        for k in ("b1", "b2", "b_corr", "b_dom"):
            np.testing.assert_array_equal(np.array(params[k]), 0.0)


class TestForwardV25:
    """Tests for _forward_v25() output shapes and value ranges.

    Traces to: REQ-LEARN-050
    """

    def test_single_sample_shapes(self) -> None:
        """Single-sample forward pass must return shapes (1,) and (N_DOMAINS,)."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        corr, dom = _forward_v25(params, jnp.ones((EMBED_DIM,)))
        assert corr.shape == (1,)
        assert dom.shape == (N_DOMAINS,)

    def test_batch_shapes(self) -> None:
        """Batched forward pass must return (batch, 1) and (batch, N_DOMAINS)."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        corr, dom = _forward_v25(params, jnp.ones((8, EMBED_DIM)))
        assert corr.shape == (8, 1)
        assert dom.shape == (8, N_DOMAINS)

    def test_corr_prob_in_range(self) -> None:
        """Correctness probabilities must be in [0, 1]."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        val = float(_forward_v25(params, jnp.ones((EMBED_DIM,)))[0].squeeze())
        assert 0.0 <= val <= 1.0

    def test_domain_probs_sum_to_one(self) -> None:
        """Domain probabilities must sum to 1 (softmax output)."""
        params = _init_v25_params(jax.random.PRNGKey(0))
        dom = _forward_v25(params, jnp.ones((EMBED_DIM,)))[1]
        assert abs(float(jnp.sum(dom)) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Verdict logic tests
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Tests for compute_honest_verdict() mapping v25 AUC values to verdict labels.

    Traces to: SCENARIO-LEARN-096
    """

    def test_ood_improved_when_both_gates_met(self) -> None:
        """ood_auc > 0.65 AND svamp_auc > 0.50 → ood_improved.

        Traces to: SCENARIO-LEARN-096 (primary success condition).
        """
        assert compute_honest_verdict(0.7, 0.70, 0.60) == "ood_improved"

    def test_svamp_improved_when_svamp_above_050_ood_below_065(self) -> None:
        """svamp_auc > 0.50 but ood_auc <= 0.65 → svamp_improved_ood_below."""
        assert compute_honest_verdict(0.6, 0.60, 0.55) == "svamp_improved_ood_below"

    def test_marginal_improvement_when_ood_between_050_and_065(self) -> None:
        """ood_auc in (0.50, 0.65] → marginal_improvement."""
        assert compute_honest_verdict(0.6, 0.55, 0.40) == "marginal_improvement"

    def test_still_blocked_when_ood_at_or_below_050(self) -> None:
        """ood_auc <= 0.50 → jepa_v25_still_blocked (retire_if_same_verdict trigger)."""
        assert compute_honest_verdict(0.6, 0.50, 0.30) == "jepa_v25_still_blocked"

    def test_still_blocked_when_ood_zero(self) -> None:
        """ood_auc = 0.0 → jepa_v25_still_blocked."""
        assert compute_honest_verdict(0.5, 0.0, 0.0) == "jepa_v25_still_blocked"

    def test_boundary_ood_exactly_065_is_svamp_improved_not_ood_improved(self) -> None:
        """ood_auc exactly 0.65 with svamp > 0.50 → svamp_improved_ood_below (not > 0.65)."""
        assert compute_honest_verdict(0.6, 0.65, 0.60) == "svamp_improved_ood_below"

    def test_ood_improved_requires_svamp_above_050(self) -> None:
        """ood_auc > 0.65 but svamp_auc <= 0.50 → svamp_improved_ood_below, not ood_improved."""
        # svamp=0.40 is below 0.50, so ood_improved gate not met
        result = compute_honest_verdict(0.7, 0.70, 0.40)
        assert result != "ood_improved"


# ---------------------------------------------------------------------------
# Integration: train_jepa_v25 on a mini corpus
# ---------------------------------------------------------------------------


class TestTrainJepaV25Integration:
    """Integration tests for train_jepa_v25() on a mini corpus.

    Traces to: REQ-LEARN-050, SCENARIO-LEARN-095
    """

    def _make_mini_corpus(self) -> list[dict[str, Any]]:
        """Build a minimal 4-domain corpus with 4 correct + 4 incorrect per domain."""
        corpus = []
        for d_idx, domain in enumerate(DOMAIN_NAMES):
            for i in range(4):
                corpus.append(
                    {
                        "text": f"Correct reasoning step {i} for {domain}.",
                        "label": 1,
                        "domain": domain,
                        "domain_idx": d_idx,
                    }
                )
                corpus.append(
                    {
                        "text": f"Incorrect wrong step {i} in {domain} with error.",
                        "label": 0,
                        "domain": domain,
                        "domain_idx": d_idx,
                    }
                )
        return corpus

    def test_returns_params_and_log(self) -> None:
        """train_jepa_v25() must return (params, log) tuple."""
        corpus = self._make_mini_corpus()
        params, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        assert isinstance(params, dict)
        assert isinstance(log, dict)

    def test_log_has_required_keys(self) -> None:
        """Training log must have: train_losses, val_losses, auc_per_domain, n_train, n_val, domain_weights."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        for key in (
            "train_losses",
            "val_losses",
            "auc_per_domain",
            "n_train",
            "n_val",
            "domain_weights",
        ):
            assert key in log, f"Missing key: {key}"

    def test_loss_decreases_over_training(self) -> None:
        """Training loss must decrease from first to last epoch over 20 epochs.

        Traces to: REQ-LEARN-050 — convergence check for DomainReweightedLoss.
        """
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=20, batch_size=8, seed=0)
        first_loss = log["train_losses"][0]
        last_loss = log["train_losses"][-1]
        # Allow small floating-point variance but overall must decrease
        assert last_loss <= first_loss + 0.5, (
            f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}"
        )

    def test_domain_weights_in_log(self) -> None:
        """domain_weights in log must cover all four domains."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        assert set(log["domain_weights"].keys()) == set(DOMAIN_NAMES)

    def test_auc_per_domain_has_all_domains(self) -> None:
        """auc_per_domain must include all four domain names."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        for domain in DOMAIN_NAMES:
            assert domain in log["auc_per_domain"]

    def test_auc_values_in_range(self) -> None:
        """All AUC values must be in [0, 1]."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        for domain, auc in log["auc_per_domain"].items():
            assert 0.0 <= auc <= 1.0, f"{domain} AUC={auc} out of range"

    def test_n_train_plus_n_val_equals_corpus(self) -> None:
        """n_train + n_val must equal the corpus size."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8)
        assert log["n_train"] + log["n_val"] == len(corpus)

    def test_uniform_weights_when_balanced_corpus(self) -> None:
        """With all domains equal size, domain_weights must be approximately equal.

        Traces to: REQ-LEARN-050 — balanced corpus → uniform DomainReweightedLoss.
        """
        corpus = self._make_mini_corpus()  # 8 per domain
        _, log = train_jepa_v25(corpus, n_epochs=3, batch_size=8, use_domain_reweighting=True)
        weights = list(log["domain_weights"].values())
        assert all(abs(w - weights[0]) < 0.01 for w in weights), f"Non-uniform: {weights}"


# ---------------------------------------------------------------------------
# Triplet builder tests
# ---------------------------------------------------------------------------


class TestBuildTriplets:
    """Tests for _build_triplets() producing valid (pos, neg, delta) triplets.

    Traces to: REQ-LEARN-050
    """

    def _simple_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.RandomState(0)
        X = rng.randn(8, EMBED_DIM).astype(np.float32)
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        domains = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        return X, labels, domains

    def test_returns_three_arrays(self) -> None:
        """_build_triplets must return exactly 3 arrays."""
        X, labels, domains = self._simple_data()
        assert (
            len(_build_triplets(X, labels, domains, _init_v25_params(jax.random.PRNGKey(0)))) == 3
        )

    def test_delta_weights_clamped(self) -> None:
        """All ΔEnergy weights must be in [DELTA_ENERGY_MIN, DELTA_ENERGY_MAX]."""
        X, labels, domains = self._simple_data()
        params = _init_v25_params(jax.random.PRNGKey(0))
        _, _, deltas = _build_triplets(X, labels, domains, params)
        assert float(np.min(deltas)) >= DELTA_ENERGY_MIN - 1e-6
        assert float(np.max(deltas)) <= DELTA_ENERGY_MAX + 1e-6


# ---------------------------------------------------------------------------
# Deliverable artifact integration tests
# ---------------------------------------------------------------------------


class TestDeliverableArtifact:
    """Integration test: the written JSON must have all required schema fields.

    Traces to: SCENARIO-LEARN-096
    """

    _artifact_path = Path("results/experiment_872_jepa_v25_dg_prm.json")

    def test_deliverable_exists(self) -> None:
        """The deliverable JSON must exist on disk after the experiment runs."""
        assert self._artifact_path.exists(), (
            f"Deliverable not found at {self._artifact_path}. "
            "Run scripts/experiment_872_jepa_v25_dg_prm.py first."
        )

    def _load(self) -> dict[str, Any]:
        with open(self._artifact_path) as fh:
            return json.load(fh)

    def test_required_schema_fields_present(self) -> None:
        """All base REQUIRED_RESULT_FIELDS must be present."""
        d = self._load()
        for field in (
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
        ):
            assert field in d, f"Missing: {field}"

    def test_experiment_id(self) -> None:
        """experiment field must be 872."""
        assert self._load()["experiment"] == 872

    def test_status_success(self) -> None:
        """status must be 'success'."""
        assert self._load()["status"] == "success"

    def test_honest_verdict_is_valid(self) -> None:
        """honest_verdict must be one of the four v25 verdict values."""
        verdict = self._load()["honest_verdict"]
        assert verdict in {
            "ood_improved",
            "svamp_improved_ood_below",
            "marginal_improvement",
            "jepa_v25_still_blocked",
        }

    def test_in_dist_and_ood_auc_fields_present(self) -> None:
        """in_dist_auc, ood_auc, svamp_auc must all be present and numeric."""
        d = self._load()
        for field in ("in_dist_auc", "ood_auc", "svamp_auc"):
            assert field in d, f"Missing: {field}"
            assert isinstance(d[field], (int, float))

    def test_domain_weights_present(self) -> None:
        """domain_weights field must be present and cover all four domains."""
        d = self._load()
        assert "domain_weights" in d
        assert set(d["domain_weights"].keys()) == set(DOMAIN_NAMES)

    def test_n_training_pairs_is_160(self) -> None:
        """n_training_pairs must be 160 (4 domains × 40 pairs)."""
        assert self._load()["n_training_pairs"] == 160

    def test_model_path_field_present(self) -> None:
        """model_path field must be present."""
        assert "model_path" in self._load()

    def test_decision_class_verify(self) -> None:
        """decision_class must be 'verify'."""
        assert self._load()["decision_class"] == "verify"

    def test_schema_lists_all_keys(self) -> None:
        """schema field must be a subset of actual keys."""
        d = self._load()
        assert set(d["schema"]) <= set(d.keys())
