"""Tests for scripts/experiment_834_jepa_v24_dg_prm.py.

Traces to: REQ-LEARN-047, REQ-LEARN-834-001, SCENARIO-LEARN-059, SCENARIO-LEARN-834-001

**What we test:**
    - build_balanced_corpus() enforces >= 10 ARC pairs (n_arc_pairs assertion).
    - Corpus contains exactly 4 domains: gsm8k, humaneval, arc, svamp.
    - ARC corpus has exactly 20 correct and 20 incorrect steps.
    - _embed_text() returns unit-normed float32 array of the right dimension.
    - _init_v24_params() returns all required keys with correct shapes.
    - _forward_v24() returns (corr_prob, dom_prob) with correct shapes and ranges.
    - compute_honest_verdict() correctly maps AUC values to verdict labels.
    - train_jepa_v24() returns valid per-domain AUC metrics on synthetic corpus.
    - Deliverable JSON exists with all required schema fields (if experiment ran).

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

from scripts.experiment_834_jepa_v24_dg_prm import (
    ARC_CORRECT_STEPS,
    ARC_INCORRECT_STEPS,
    DELTA_ENERGY_MAX,
    DELTA_ENERGY_MIN,
    DG_PRM_DOMAIN_WEIGHTS,
    DOMAIN_NAMES,
    DREAM_PRM_WEIGHTS,
    EMBED_DIM,
    HIDDEN1,
    HIDDEN2,
    N_DOMAINS,
    N_EPOCHS,
    GSM8K_CORRECT_STEPS,
    GSM8K_INCORRECT_STEPS,
    HUMANEVAL_CORRECT_STEPS,
    HUMANEVAL_INCORRECT_STEPS,
    SVAMP_CORRECT_STEPS,
    SVAMP_INCORRECT_STEPS,
    TRIPLET_MARGIN,
    _build_triplets,
    _embed_text,
    _forward_v24,
    _init_v24_params,
    build_balanced_corpus,
    compute_honest_verdict,
    train_jepa_v24,
)


# ---------------------------------------------------------------------------
# Corpus structure tests
# ---------------------------------------------------------------------------


class TestBuildBalancedCorpus:
    """Tests for build_balanced_corpus() enforcing domain balance.

    Traces to: REQ-LEARN-834-001, SCENARIO-LEARN-834-001
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

    def test_arc_pairs_count_at_least_ten(self) -> None:
        """ARC correct + incorrect examples must each be >= 10.

        This is the REQ-LEARN-834-001 invariant: JEPA v24 training MUST include
        at least 10 ARC pairs.  Zero ARC pairs caused the JEPA v23 AUC=0.04 failure.
        """
        corpus = build_balanced_corpus()
        arc_correct = sum(1 for p in corpus if p["domain"] == "arc" and p["label"] == 1)
        arc_incorrect = sum(1 for p in corpus if p["domain"] == "arc" and p["label"] == 0)
        assert arc_correct >= 10, f"Need >= 10 ARC correct, got {arc_correct}"
        assert arc_incorrect >= 10, f"Need >= 10 ARC incorrect, got {arc_incorrect}"

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

    def test_gsm8k_has_twenty_correct_and_incorrect(self) -> None:
        """GSM8K must contribute 20 correct and 20 incorrect pairs."""
        corpus = build_balanced_corpus()
        c = sum(1 for p in corpus if p["domain"] == "gsm8k" and p["label"] == 1)
        i = sum(1 for p in corpus if p["domain"] == "gsm8k" and p["label"] == 0)
        assert c == 20
        assert i == 20

    def test_svamp_has_ten_correct_and_incorrect(self) -> None:
        """SVAMP must contribute 10 correct and 10 incorrect pairs."""
        corpus = build_balanced_corpus()
        c = sum(1 for p in corpus if p["domain"] == "svamp" and p["label"] == 1)
        i = sum(1 for p in corpus if p["domain"] == "svamp" and p["label"] == 0)
        assert c == 10
        assert i == 10


class TestStaticCorpusData:
    """Sanity checks on embedded corpus constants.

    Traces to: REQ-LEARN-834-001
    """

    def test_arc_correct_exactly_twenty(self) -> None:
        """ARC_CORRECT_STEPS must have exactly 20 entries."""
        assert len(ARC_CORRECT_STEPS) == 20

    def test_arc_incorrect_exactly_twenty(self) -> None:
        """ARC_INCORRECT_STEPS must have exactly 20 entries."""
        assert len(ARC_INCORRECT_STEPS) == 20

    def test_gsm8k_counts(self) -> None:
        """GSM8K correct and incorrect lists each have 20 entries."""
        assert len(GSM8K_CORRECT_STEPS) == 20
        assert len(GSM8K_INCORRECT_STEPS) == 20

    def test_humaneval_counts(self) -> None:
        """HumanEval correct and incorrect lists each have 20 entries."""
        assert len(HUMANEVAL_CORRECT_STEPS) == 20
        assert len(HUMANEVAL_INCORRECT_STEPS) == 20

    def test_svamp_counts(self) -> None:
        """SVAMP correct and incorrect lists each have 10 entries."""
        assert len(SVAMP_CORRECT_STEPS) == 10
        assert len(SVAMP_INCORRECT_STEPS) == 10

    def test_arc_steps_are_strings(self) -> None:
        """All ARC step entries must be non-empty strings."""
        for step in ARC_CORRECT_STEPS + ARC_INCORRECT_STEPS:
            assert isinstance(step, str) and len(step) > 0

    def test_domain_names_length(self) -> None:
        """DOMAIN_NAMES must have exactly 4 entries."""
        assert len(DOMAIN_NAMES) == 4
        assert N_DOMAINS == 4


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------


class TestEmbedText:
    """Tests for the _embed_text() hash-projection function.

    Traces to: REQ-LEARN-047
    """

    def test_returns_float32_array(self) -> None:
        """_embed_text() must return a float32 numpy array."""
        emb = _embed_text("hello world")
        assert emb.dtype == np.float32

    def test_correct_dimension(self) -> None:
        """Output shape must be (EMBED_DIM,) = (256,)."""
        emb = _embed_text("test text here")
        assert emb.shape == (EMBED_DIM,)

    def test_unit_normed(self) -> None:
        """Output must be unit-normed (L2 norm ≈ 1.0) for non-empty text."""
        emb = _embed_text("some text input for embedding")
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self) -> None:
        """Same text with same seed must produce identical embeddings."""
        e1 = _embed_text("hello", seed=42)
        e2 = _embed_text("hello", seed=42)
        np.testing.assert_array_equal(e1, e2)

    def test_different_texts_differ(self) -> None:
        """Different texts must produce different embeddings."""
        e1 = _embed_text("correct arithmetic step", seed=42)
        e2 = _embed_text("incorrect reasoning fallacy", seed=42)
        assert not np.allclose(e1, e2)

    def test_custom_dim(self) -> None:
        """Custom dim parameter changes output shape."""
        emb = _embed_text("test", dim=64)
        assert emb.shape == (64,)


# ---------------------------------------------------------------------------
# Model architecture tests
# ---------------------------------------------------------------------------


class TestInitV24Params:
    """Tests for _init_v24_params() returning correct parameter shapes.

    Traces to: REQ-LEARN-834-001
    """

    def test_returns_dict(self) -> None:
        """_init_v24_params() must return a dict."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        assert isinstance(params, dict)

    def test_all_keys_present(self) -> None:
        """All expected parameter keys must be present."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        expected = {"w1", "b1", "w2", "b2", "w_corr", "b_corr", "w_dom", "b_dom"}
        assert set(params.keys()) == expected

    def test_w1_shape(self) -> None:
        """w1 shape must be (EMBED_DIM, HIDDEN1)."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        assert params["w1"].shape == (EMBED_DIM, HIDDEN1)

    def test_w2_shape(self) -> None:
        """w2 shape must be (HIDDEN1, HIDDEN2)."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        assert params["w2"].shape == (HIDDEN1, HIDDEN2)

    def test_correctness_head_shape(self) -> None:
        """w_corr shape must be (HIDDEN2, 1)."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        assert params["w_corr"].shape == (HIDDEN2, 1)

    def test_domain_head_shape(self) -> None:
        """w_dom shape must be (HIDDEN2, N_DOMAINS)."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        assert params["w_dom"].shape == (HIDDEN2, N_DOMAINS)

    def test_biases_are_zeros(self) -> None:
        """All bias terms must be initialised to zero."""
        key = jax.random.PRNGKey(0)
        params = _init_v24_params(key)
        for bias_key in ("b1", "b2", "b_corr", "b_dom"):
            np.testing.assert_array_equal(np.array(params[bias_key]), 0.0)


class TestForwardV24:
    """Tests for _forward_v24() output shapes and value ranges.

    Traces to: REQ-LEARN-834-001
    """

    def _make_params(self) -> dict[str, jax.Array]:
        return _init_v24_params(jax.random.PRNGKey(0))

    def test_single_sample_shapes(self) -> None:
        """Single-sample forward pass must return shapes (1,) and (N_DOMAINS,)."""
        params = self._make_params()
        x = jnp.ones((EMBED_DIM,))
        corr, dom = _forward_v24(params, x)
        assert corr.shape == (1,)
        assert dom.shape == (N_DOMAINS,)

    def test_batch_shapes(self) -> None:
        """Batched forward pass must return (batch, 1) and (batch, N_DOMAINS)."""
        params = self._make_params()
        x = jnp.ones((8, EMBED_DIM))
        corr, dom = _forward_v24(params, x)
        assert corr.shape == (8, 1)
        assert dom.shape == (8, N_DOMAINS)

    def test_corr_prob_in_range(self) -> None:
        """Correctness probabilities must be in [0, 1]."""
        params = self._make_params()
        x = jnp.ones((EMBED_DIM,))
        corr, _ = _forward_v24(params, x)
        val = float(corr.squeeze())
        assert 0.0 <= val <= 1.0

    def test_domain_probs_sum_to_one(self) -> None:
        """Domain probabilities must sum to 1 (softmax output)."""
        params = self._make_params()
        x = jnp.ones((EMBED_DIM,))
        _, dom = _forward_v24(params, x)
        assert abs(float(jnp.sum(dom)) - 1.0) < 1e-5

    def test_domain_probs_all_positive(self) -> None:
        """All domain probabilities must be > 0."""
        params = self._make_params()
        x = jnp.ones((EMBED_DIM,))
        _, dom = _forward_v24(params, x)
        assert float(jnp.min(dom)) > 0.0


# ---------------------------------------------------------------------------
# Triplet builder tests
# ---------------------------------------------------------------------------


class TestBuildTriplets:
    """Tests for _build_triplets() producing valid (pos, neg, delta) triplets.

    Traces to: REQ-LEARN-834-001
    """

    def _simple_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make a small balanced dataset with 2 domains, 4 pos + 4 neg."""
        rng = np.random.RandomState(0)
        X = rng.randn(8, EMBED_DIM).astype(np.float32)
        # labels: first 4 correct, next 4 incorrect
        labels = np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        # domains: first 4 = domain 0, next 4 = domain 1
        domains = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        return X, labels, domains

    def test_returns_three_arrays(self) -> None:
        """_build_triplets must return exactly 3 arrays."""
        X, labels, domains = self._simple_data()
        params = _init_v24_params(jax.random.PRNGKey(0))
        result = _build_triplets(X, labels, domains, params)
        assert len(result) == 3

    def test_pos_neg_same_length(self) -> None:
        """x_pos and x_neg must have the same number of triplets."""
        X, labels, domains = self._simple_data()
        params = _init_v24_params(jax.random.PRNGKey(0))
        x_pos, x_neg, deltas = _build_triplets(X, labels, domains, params)
        assert len(x_pos) == len(x_neg) == len(deltas)

    def test_delta_weights_clamped(self) -> None:
        """All ΔEnergy weights must be in [DELTA_ENERGY_MIN, DELTA_ENERGY_MAX]."""
        X, labels, domains = self._simple_data()
        params = _init_v24_params(jax.random.PRNGKey(0))
        _, _, deltas = _build_triplets(X, labels, domains, params)
        assert float(np.min(deltas)) >= DELTA_ENERGY_MIN - 1e-6
        assert float(np.max(deltas)) <= DELTA_ENERGY_MAX + 1e-6

    def test_embed_dim_preserved(self) -> None:
        """Triplet embeddings must have shape (n, EMBED_DIM)."""
        X, labels, domains = self._simple_data()
        params = _init_v24_params(jax.random.PRNGKey(0))
        x_pos, x_neg, _ = _build_triplets(X, labels, domains, params)
        assert x_pos.shape[1] == EMBED_DIM
        assert x_neg.shape[1] == EMBED_DIM


# ---------------------------------------------------------------------------
# Verdict logic tests
# ---------------------------------------------------------------------------


class TestComputeHonestVerdict:
    """Tests for compute_honest_verdict() mapping AUC values to verdict labels.

    Traces to: SCENARIO-LEARN-834-001
    """

    def test_all_above_055_gives_domain_balanced(self) -> None:
        """All per-domain AUCs > 0.55 → jepa_v24_domain_balanced."""
        verdict = compute_honest_verdict(
            auc_gsm8k=0.7,
            auc_humaneval=0.65,
            auc_arc=0.6,
            auc_svamp=0.68,
            overall_ood_auc=0.66,
            min_domain_auc=0.6,
        )
        assert verdict == "jepa_v24_domain_balanced"

    def test_high_overall_but_arc_below_gate_gives_improvement(self) -> None:
        """overall_ood_auc > 0.65 but min_domain_auc <= 0.55 → jepa_v24_improvement."""
        verdict = compute_honest_verdict(
            auc_gsm8k=0.85,
            auc_humaneval=0.80,
            auc_arc=0.45,
            auc_svamp=0.82,
            overall_ood_auc=0.73,
            min_domain_auc=0.45,
        )
        assert verdict == "jepa_v24_improvement"

    def test_arc_above_040_gives_arc_improved(self) -> None:
        """ARC AUC > 0.40 but overall ≤ 0.65 and min ≤ 0.55 → jepa_v24_arc_improved."""
        verdict = compute_honest_verdict(
            auc_gsm8k=0.55,
            auc_humaneval=0.52,
            auc_arc=0.45,
            auc_svamp=0.53,
            overall_ood_auc=0.51,
            min_domain_auc=0.45,
        )
        assert verdict == "jepa_v24_arc_improved"

    def test_arc_at_or_below_040_gives_still_unbalanced(self) -> None:
        """ARC AUC <= 0.40 → jepa_v24_still_unbalanced."""
        verdict = compute_honest_verdict(
            auc_gsm8k=0.70,
            auc_humaneval=0.65,
            auc_arc=0.30,
            auc_svamp=0.60,
            overall_ood_auc=0.56,
            min_domain_auc=0.30,
        )
        assert verdict == "jepa_v24_still_unbalanced"

    def test_arc_exactly_at_040_is_still_unbalanced(self) -> None:
        """ARC AUC exactly 0.40 → jepa_v24_still_unbalanced (> not >=)."""
        verdict = compute_honest_verdict(
            auc_gsm8k=0.7,
            auc_humaneval=0.7,
            auc_arc=0.40,
            auc_svamp=0.7,
            overall_ood_auc=0.625,
            min_domain_auc=0.40,
        )
        assert verdict == "jepa_v24_still_unbalanced"

    def test_all_at_boundary_055_is_still_unbalanced(self) -> None:
        """All AUCs exactly at 0.55 → jepa_v24_arc_improved (not domain_balanced, 0.55 not > 0.55)."""
        # min_domain_auc = 0.55, not > 0.55 so domain_balanced not triggered
        # overall_ood_auc = 0.55, not > 0.65 so improvement not triggered
        # auc_arc = 0.55 > 0.40 so arc_improved is triggered
        verdict = compute_honest_verdict(
            auc_gsm8k=0.55,
            auc_humaneval=0.55,
            auc_arc=0.55,
            auc_svamp=0.55,
            overall_ood_auc=0.55,
            min_domain_auc=0.55,
        )
        assert verdict == "jepa_v24_arc_improved"


# ---------------------------------------------------------------------------
# Domain weight constants tests
# ---------------------------------------------------------------------------


class TestDomainWeightConstants:
    """Tests for DG-PRM and DreamPRM weight constants.

    Traces to: REQ-LEARN-834-001
    """

    def test_dg_prm_arc_weight_at_least_three(self) -> None:
        """ARC domain weight in DG_PRM_DOMAIN_WEIGHTS must be >= 3.0.

        REQ-LEARN-834-001 mandates ARC weight >= 3.0 when ARC was absent from
        prior training data (which it was in JEPA v23).
        """
        assert DG_PRM_DOMAIN_WEIGHTS["arc"] >= 3.0

    def test_dg_prm_all_domains_present(self) -> None:
        """DG_PRM_DOMAIN_WEIGHTS must cover all four domains."""
        assert set(DG_PRM_DOMAIN_WEIGHTS.keys()) == {"gsm8k", "humaneval", "arc", "svamp"}

    def test_dream_prm_arc_weight_highest(self) -> None:
        """DREAM_PRM_WEIGHTS arc value must be the maximum (highest penalty domain)."""
        assert DREAM_PRM_WEIGHTS["arc"] == max(DREAM_PRM_WEIGHTS.values())

    def test_dream_prm_all_domains_present(self) -> None:
        """DREAM_PRM_WEIGHTS must cover all four domains."""
        assert set(DREAM_PRM_WEIGHTS.keys()) == {"gsm8k", "humaneval", "arc", "svamp"}

    def test_delta_energy_clamp_range(self) -> None:
        """DELTA_ENERGY_MIN < DELTA_ENERGY_MAX and both positive."""
        assert 0 < DELTA_ENERGY_MIN < DELTA_ENERGY_MAX

    def test_triplet_margin_positive(self) -> None:
        """TRIPLET_MARGIN must be positive."""
        assert TRIPLET_MARGIN > 0.0


# ---------------------------------------------------------------------------
# Integration: train_jepa_v24 on a small synthetic subset
# ---------------------------------------------------------------------------


class TestTrainJepaV24Integration:
    """Integration tests for train_jepa_v24() on a mini corpus.

    Runs a short training pass (10 epochs) to verify the API contract.
    Traces to: REQ-LEARN-047, SCENARIO-LEARN-059
    """

    def _make_mini_corpus(self) -> list[dict[str, Any]]:
        """Build a minimal 4-domain corpus with 4 correct + 4 incorrect per domain."""
        corpus = []
        for d_idx, domain in enumerate(DOMAIN_NAMES):
            for i in range(4):
                corpus.append(
                    {
                        "text": f"Correct step {i} for {domain} domain.",
                        "label": 1,
                        "domain": domain,
                        "domain_idx": d_idx,
                    }
                )
                corpus.append(
                    {
                        "text": f"Incorrect step {i} for {domain} domain errors.",
                        "label": 0,
                        "domain": domain,
                        "domain_idx": d_idx,
                    }
                )
        return corpus

    def test_returns_params_and_log(self) -> None:
        """train_jepa_v24() must return (params, log) tuple."""
        corpus = self._make_mini_corpus()
        params, log = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        assert isinstance(params, dict)
        assert isinstance(log, dict)

    def test_log_has_required_keys(self) -> None:
        """Training log must have: train_losses, val_losses, auc_per_domain, n_train, n_val."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        for key in ("train_losses", "val_losses", "auc_per_domain", "n_train", "n_val"):
            assert key in log, f"Missing key: {key}"

    def test_loss_lists_length_matches_epochs(self) -> None:
        """train_losses and val_losses must have one entry per epoch."""
        n_ep = 5
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v24(corpus, n_epochs=n_ep, batch_size=8)
        assert len(log["train_losses"]) == n_ep
        assert len(log["val_losses"]) == n_ep

    def test_auc_per_domain_has_all_domains(self) -> None:
        """auc_per_domain must include all four domain names (or as many as present in val)."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        # All four domains should appear since we have pairs from all four
        for domain in DOMAIN_NAMES:
            assert domain in log["auc_per_domain"]

    def test_auc_values_in_range(self) -> None:
        """All AUC values must be in [0, 1]."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        for domain, auc in log["auc_per_domain"].items():
            assert 0.0 <= auc <= 1.0, f"{domain} AUC={auc} out of range"

    def test_params_have_correct_keys(self) -> None:
        """Returned params dict must have all v24 parameter keys."""
        corpus = self._make_mini_corpus()
        params, _ = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        expected = {"w1", "b1", "w2", "b2", "w_corr", "b_corr", "w_dom", "b_dom"}
        assert set(params.keys()) == expected

    def test_n_train_plus_n_val_equals_corpus(self) -> None:
        """n_train + n_val must equal the corpus size."""
        corpus = self._make_mini_corpus()
        _, log = train_jepa_v24(corpus, n_epochs=5, batch_size=8)
        assert log["n_train"] + log["n_val"] == len(corpus)


# ---------------------------------------------------------------------------
# Deliverable artifact integration tests
# ---------------------------------------------------------------------------


class TestDeliverableArtifact:
    """Integration test: the written JSON must have all required schema fields.

    Traces to: REQ-LEARN-834-001, SCENARIO-LEARN-834-001
    """

    _artifact_path = Path("results/experiment_834_jepa_v24_dg_prm.json")

    def test_deliverable_exists(self) -> None:
        """The deliverable JSON must exist on disk after the experiment runs."""
        assert self._artifact_path.exists(), (
            f"Deliverable not found at {self._artifact_path}. "
            "Run scripts/experiment_834_jepa_v24_dg_prm.py first."
        )

    def _load(self) -> dict[str, Any]:
        with open(self._artifact_path) as fh:
            return json.load(fh)

    def test_required_schema_fields_present(self) -> None:
        """All REQUIRED_RESULT_FIELDS must be present in the artifact."""
        d = self._load()
        required = [
            "experiment",
            "schema",
            "run_date",
            "started_at",
            "finished_at",
            "duration_s",
            "status",
            "title",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"

    def test_experiment_id(self) -> None:
        """experiment field must be 834."""
        assert self._load()["experiment"] == 834

    def test_status_success(self) -> None:
        """status must be 'success'."""
        assert self._load()["status"] == "success"

    def test_honest_verdict_is_one_of_four(self) -> None:
        """honest_verdict must be one of the four recognised values."""
        verdict = self._load()["honest_verdict"]
        assert verdict in {
            "jepa_v24_domain_balanced",
            "jepa_v24_improvement",
            "jepa_v24_arc_improved",
            "jepa_v24_still_unbalanced",
        }

    def test_auc_fields_present(self) -> None:
        """All five AUC fields must be present and numeric."""
        d = self._load()
        for field in ("auc_gsm8k", "auc_humaneval", "auc_arc", "auc_svamp", "overall_ood_auc"):
            assert field in d, f"Missing AUC field: {field}"
            assert isinstance(d[field], (int, float))

    def test_min_domain_auc_field_present(self) -> None:
        """min_domain_auc must be present."""
        d = self._load()
        assert "min_domain_auc" in d

    def test_corpus_composition_has_all_domains(self) -> None:
        """corpus_composition must cover all four domains."""
        d = self._load()
        comp = d["corpus_composition"]
        assert set(comp.keys()) == {"gsm8k", "humaneval", "arc", "svamp"}

    def test_domain_weights_used_present(self) -> None:
        """domain_weights_used field must be present and have 4 domains."""
        d = self._load()
        assert "domain_weights_used" in d
        assert len(d["domain_weights_used"]) == 4

    def test_retro_jepa_ood_improving_is_bool(self) -> None:
        """retro_jepa_ood_improving must be a boolean."""
        d = self._load()
        assert isinstance(d["retro_jepa_ood_improving"], bool)

    def test_arc_auc_baseline_field(self) -> None:
        """arc_auc_v23_baseline must be 0.04 (the Exp 832 root-cause value)."""
        d = self._load()
        assert d["arc_auc_v23_baseline"] == pytest.approx(0.04)

    def test_decision_class_verify(self) -> None:
        """decision_class must be 'verify'."""
        assert self._load()["decision_class"] == "verify"

    def test_schema_lists_all_keys(self) -> None:
        """schema field must be a subset of actual keys.

        The invariant-checker appends `invariant_violations` after build_result()
        computes the schema field, so the schema may be a strict subset of the
        actual top-level keys.  We verify no schema key is MISSING from the artifact.
        """
        d = self._load()
        schema_keys = set(d["schema"])
        actual_keys = set(d.keys())
        # Every key in schema must exist in the artifact (no phantom keys)
        assert schema_keys <= actual_keys
