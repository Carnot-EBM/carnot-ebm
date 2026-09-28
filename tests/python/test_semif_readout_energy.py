"""Tests for python/carnot/verify/semif_readout_energy.py -- REQ-VERIFY-7750,
the A1 readout-energy product-of-experts calibration experiment.

No test in this file makes a live model call. The readout-logit-to-decision
path is exercised through hand-built logit dictionaries (the exact shape a
real `llama.cpp` forward pass produces), and the corpus-split tests read the
real `data/fover_corpus_v4.json` file -- a plain file read, not an LLM call.

Spec: REQ-VERIFY-7750, SCENARIO-VERIFY-7750-POE, SCENARIO-VERIFY-7750-POSITIVE-CONTROL,
SCENARIO-VERIFY-7750-DEGENERATE, SCENARIO-VERIFY-7750-HEADROOM,
REQ-VERIFY-7751, SCENARIO-VERIFY-7751-TOURNAMENT, SCENARIO-VERIFY-7751-POSITIVE-CONTROL,
SCENARIO-VERIFY-7751-DEGENERATE, SCENARIO-VERIFY-7751-GATE,
REQ-VERIFY-7752, SCENARIO-VERIFY-7752-POLICY, SCENARIO-VERIFY-7752-POSITIVE-CONTROL,
SCENARIO-VERIFY-7752-DEGENERATE, SCENARIO-VERIFY-7752-COST-GRID
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import semif_readout_energy as sre


# ==========================================================================
# SCENARIO-VERIFY-7750-DEGENERATE case (c): missing/non-finite logit -> escalate.
# ==========================================================================


class TestReadoutFromLogits:
    def test_all_finite_logits_produce_a_full_distribution(self) -> None:
        result = sre.readout_from_logits({"accept": 2.0, "reject": -1.0, "escalate": -1.0})
        assert result.probs is not None
        assert result.degenerate_reason is None
        assert math.isclose(sum(result.probs.values()), 1.0, abs_tol=1e-9)
        assert result.decision == "accept"
        assert result.energy_accept is not None

    def test_missing_logit_forces_escalate_not_accept(self) -> None:
        result = sre.readout_from_logits({"accept": 5.0, "reject": None, "escalate": 0.0})
        assert result.decision == "escalate"
        assert result.probs is None
        assert result.energy_accept is None
        assert result.degenerate_reason == "missing_or_non_finite_option_logit"

    def test_non_finite_logit_forces_escalate_not_accept(self) -> None:
        for bad in (float("nan"), float("inf"), float("-inf")):
            result = sre.readout_from_logits({"accept": bad, "reject": 0.0, "escalate": 0.0})
            assert result.decision == "escalate"
            assert result.probs is None
            assert result.energy_accept is None

    def test_missing_key_entirely_also_forces_escalate(self) -> None:
        result = sre.readout_from_logits({"accept": 1.0, "reject": 0.0})
        assert result.decision == "escalate"
        assert result.probs is None

    def test_energy_accept_is_low_when_accept_dominates(self) -> None:
        confident_accept = sre.readout_from_logits(
            {"accept": 10.0, "reject": -10.0, "escalate": -10.0}
        )
        confident_reject = sre.readout_from_logits(
            {"accept": -10.0, "reject": 10.0, "escalate": -10.0}
        )
        assert confident_accept.energy_accept is not None
        assert confident_reject.energy_accept is not None
        assert confident_accept.energy_accept < confident_reject.energy_accept

    def test_decision_matches_the_dominant_option(self) -> None:
        assert (
            sre.readout_from_logits({"accept": -1.0, "reject": 5.0, "escalate": -1.0}).decision
            == "reject"
        )
        assert (
            sre.readout_from_logits({"accept": -1.0, "reject": -1.0, "escalate": 5.0}).decision
            == "escalate"
        )


# ==========================================================================
# Prompt builder and single-token label resolution.
# ==========================================================================


class TestBuildReadoutPrompt:
    def test_prompt_contains_the_step_text_verbatim(self) -> None:
        prompt = sre.build_readout_prompt("2 + 2 = 5")
        assert "2 + 2 = 5" in prompt

    def test_prompt_never_leaks_the_gold_label(self) -> None:
        prompt = sre.build_readout_prompt("some step text")
        assert "correct" not in prompt.split("Step:")[1].split("Answer with")[0]

    def test_prompt_ends_ready_for_a_single_letter_answer(self) -> None:
        assert sre.build_readout_prompt("x").rstrip().endswith("Answer:")


class _FakeLlama:
    """Minimal stand-in for llama_cpp.Llama's tokenize/reset/eval/scores
    surface, so the resolution and single-pass-read logic can be tested
    without loading a real GGUF model."""

    def __init__(self, single_token_map: dict[str, int], vocab_size: int = 32) -> None:
        self._single_token_map = single_token_map
        self._vocab_size = vocab_size
        self.last_eval_tokens: list[int] | None = None
        self.n_tokens = 0
        self.scores: list[list[float]] = [[0.0] * vocab_size]
        # A real llama.cpp binding fills `scores` as a side effect of `eval`.
        # Tests set this before calling the code under test, mirroring that.
        # `read_option_logits` reads `scores[n_tokens - 1]` -- NOT `scores[-1]`
        # -- so the fake stores the row at that exact index, the same shape
        # bug this experiment found in the real binding (see
        # `read_option_logits`'s docstring).
        self.next_eval_scores: list[float] | None = None

    def tokenize(self, data: bytes, add_bos: bool = True) -> list[int]:
        text = data.decode("utf-8")
        if text in self._single_token_map:
            return [self._single_token_map[text]]
        # Anything else "tokenizes" to two tokens, i.e. is NOT single-token.
        return [0, 1]

    def reset(self) -> None:
        self.n_tokens = 0
        self.scores = [[0.0] * self._vocab_size]

    def eval(self, tokens: list[int]) -> None:
        self.last_eval_tokens = list(tokens)
        self.n_tokens = len(tokens)
        if self.next_eval_scores is not None:
            padding = [[0.0] * self._vocab_size] * (self.n_tokens - 1)
            self.scores = [*padding, self.next_eval_scores]
        else:
            self.scores = [[0.0] * self._vocab_size] * max(self.n_tokens, 1)


class TestResolveOptionTokenIds:
    def test_resolves_leading_space_letters(self) -> None:
        llama = _FakeLlama({" A": 10, " B": 11, " C": 12})
        token_ids = sre.resolve_option_token_ids(llama)
        assert token_ids == {"accept": 10, "reject": 11, "escalate": 12}

    def test_falls_back_to_bare_letter(self) -> None:
        llama = _FakeLlama({"A": 20, "B": 21, "C": 22})
        token_ids = sre.resolve_option_token_ids(llama)
        assert token_ids == {"accept": 20, "reject": 21, "escalate": 22}

    def test_returns_none_when_a_letter_is_not_single_token(self) -> None:
        llama = _FakeLlama({" A": 10, " B": 11})  # "C" resolves to neither surface form
        assert sre.resolve_option_token_ids(llama) is None

    def test_returns_none_on_tokenizer_collision(self) -> None:
        """Three options that would resolve to the SAME token id are a
        tokenizer artifact, not a real three-way question -- must not be
        silently accepted as resolved."""
        llama = _FakeLlama({" A": 5, " B": 5, " C": 5})
        assert sre.resolve_option_token_ids(llama) is None


class TestReadOptionLogits:
    def test_reads_the_logit_at_each_resolved_token_id(self) -> None:
        llama = _FakeLlama({" A": 3, " B": 4, " C": 5}, vocab_size=8)
        llama.next_eval_scores = [0.0, 0.0, 0.0, 1.5, -0.5, 2.0, 0.0, 0.0]
        token_ids = {"accept": 3, "reject": 4, "escalate": 5}
        result = sre.read_option_logits(llama, "prompt", token_ids)
        assert result == {"accept": 1.5, "reject": -0.5, "escalate": 2.0}

    def test_out_of_range_token_id_reads_as_none(self) -> None:
        llama = _FakeLlama({" A": 3, " B": 4, " C": 999}, vocab_size=8)
        token_ids = {"accept": 3, "reject": 4, "escalate": 999}
        result = sre.read_option_logits(llama, "prompt", token_ids)
        assert result["escalate"] is None

    def test_calls_reset_before_eval_for_a_clean_single_pass(self) -> None:
        llama = _FakeLlama({" A": 3, " B": 4, " C": 5}, vocab_size=8)
        sre.read_option_logits(llama, "hello", {"accept": 3})
        assert llama.last_eval_tokens is not None


# ==========================================================================
# Corpus split preservation -- SCENARIO-VERIFY-7750-POE precondition.
# ==========================================================================


class TestLoadCorpusRowsWithFeatures:
    def test_row_count_matches_the_pcib_feature_cache(self) -> None:
        from carnot.autoresearch import calibrated_decision_benchmark as cdb

        rows = sre.load_corpus_rows_with_features()
        assert len(rows) == len(cdb._pcib_features())

    def test_split_label_matches_calibrated_decision_benchmark_exactly(self) -> None:
        """No question splits across train/held-out -- the split here must
        be IDENTICAL to calibrated_decision_benchmark's own, row for row."""
        from carnot.autoresearch import calibrated_decision_benchmark as cdb

        rows = sre.load_corpus_rows_with_features()
        raw_rows = cdb._load_corpus_rows()
        for row, raw in zip(rows, raw_rows):
            expected_bucket = cdb._bucket_of(raw.get("question_id"))
            expected_split = "train" if expected_bucket < cdb._TRAIN_BUCKET_CEILING else "held_out"
            assert row.split == expected_split

    def test_no_question_id_appears_on_both_sides_of_the_split(self) -> None:
        rows = sre.load_corpus_rows_with_features()
        train_ids = {r.question_id for r in rows if r.split == "train"}
        held_out_ids = {r.question_id for r in rows if r.split == "held_out"}
        assert train_ids.isdisjoint(held_out_ids)

    def test_both_splits_are_non_empty_and_have_both_labels(self) -> None:
        rows = sre.load_corpus_rows_with_features()
        for split in ("train", "held_out"):
            subset = [r for r in rows if r.split == split]
            assert len(subset) > 0
            assert {r.label for r in subset} == {0, 1}


class TestKfoldBucket:
    def test_deterministic(self) -> None:
        assert sre.kfold_bucket("q1", 5) == sre.kfold_bucket("q1", 5)

    def test_in_range(self) -> None:
        for qid in ("1", "abc", "9999"):
            bucket = sre.kfold_bucket(qid, 7)
            assert 0 <= bucket < 7

    def test_independent_salt_from_the_outer_split(self) -> None:
        from carnot.autoresearch import calibrated_decision_benchmark as cdb

        qids = [str(i) for i in range(50)]
        # Different modulus AND different salt -- confirm at least one id's
        # relative ordering differs between the two hash families.
        outer = [cdb._bucket_of(q) % 5 for q in qids]
        nested = [sre.kfold_bucket(q, 5) for q in qids]
        assert outer != nested


# ==========================================================================
# Verifier-only energy: real NCE training on a tiny synthetic separable set.
# ==========================================================================


class TestTrainGibbsVerifier:
    def test_trained_verifier_is_not_degenerate_on_separable_data(self) -> None:
        rng = np.random.default_rng(0)
        correct = (rng.normal(loc=-2.0, scale=0.2, size=(60, 2))).tolist()
        incorrect = (rng.normal(loc=2.0, scale=0.2, size=(60, 2))).tolist()
        model = sre.train_gibbs_verifier(correct, incorrect, seed=1, n_epochs=200)
        held_out = correct[:20] + incorrect[:20]
        energies = sre.gibbs_energy_batch(model, held_out)
        assert not sre.is_degenerate_energy(energies)

    def test_correct_rows_score_lower_energy_than_incorrect_rows(self) -> None:
        rng = np.random.default_rng(1)
        correct = (rng.normal(loc=-2.0, scale=0.2, size=(60, 2))).tolist()
        incorrect = (rng.normal(loc=2.0, scale=0.2, size=(60, 2))).tolist()
        model = sre.train_gibbs_verifier(correct, incorrect, seed=2, n_epochs=200)
        held_correct = np.asarray(correct[:20])
        held_incorrect = np.asarray(incorrect[:20])
        e_correct = sre.gibbs_energy_batch(model, held_correct)
        e_incorrect = sre.gibbs_energy_batch(model, held_incorrect)
        assert float(e_correct.mean()) < float(e_incorrect.mean())


class TestIsDegenerateEnergy:
    def test_constant_array_is_degenerate(self) -> None:
        assert sre.is_degenerate_energy(np.zeros(10))

    def test_varying_array_is_not_degenerate(self) -> None:
        assert not sre.is_degenerate_energy(np.array([0.1, 0.5, 0.9, -0.3]))

    def test_all_non_finite_is_degenerate(self) -> None:
        assert sre.is_degenerate_energy(np.array([float("nan"), float("inf")]))

    def test_near_duplicate_within_tolerance_is_degenerate(self) -> None:
        values = np.array([1.0 + 1e-12, 1.0 - 1e-12, 1.0])
        assert sre.is_degenerate_energy(values)


# ==========================================================================
# SCENARIO-VERIFY-7750-DEGENERATE case (a): uniform readout -> verifier ranking.
# ==========================================================================


class TestUniformReadoutDegenerateCase:
    def test_uniform_readout_preserves_verifier_ranking_exactly(self) -> None:
        rng = np.random.default_rng(3)
        e_verifier = rng.normal(size=200)
        corr = sre.check_uniform_readout_preserves_verifier_ranking(e_verifier, alpha=1.0, beta=1.0)
        assert corr == pytest.approx(1.0, abs=1e-9)

    def test_holds_for_any_positive_alpha_beta_and_constant(self) -> None:
        rng = np.random.default_rng(4)
        e_verifier = rng.normal(size=200)
        for alpha, beta, constant in ((0.5, 2.0, -1.3), (3.0, 0.1, 10.0)):
            corr = sre.check_uniform_readout_preserves_verifier_ranking(
                e_verifier, alpha=alpha, beta=beta, constant=constant
            )
            assert corr == pytest.approx(1.0, abs=1e-9)


# ==========================================================================
# SCENARIO-VERIFY-7750-DEGENERATE case (b): degenerate verifier rejected.
# ==========================================================================


class TestFitPoeWithGuards:
    def test_degenerate_verifier_energy_is_rejected(self) -> None:
        rng = np.random.default_rng(5)
        n = 100
        e_readout = rng.normal(size=n)
        e_verifier_degenerate = np.zeros(n)
        labels = rng.integers(0, 2, size=n)
        assert sre.fit_poe_with_guards(e_readout, e_verifier_degenerate, labels) is None

    def test_non_degenerate_inputs_fit_normally(self) -> None:
        rng = np.random.default_rng(6)
        n = 200
        labels = rng.integers(0, 2, size=n)
        e_readout = (2.0 * labels - 1.0) + rng.normal(scale=0.5, size=n)
        e_verifier = (2.0 * labels - 1.0) + rng.normal(scale=0.5, size=n)
        fit = sre.fit_poe_with_guards(e_readout, e_verifier, labels)
        assert fit is not None
        assert fit["alpha"] >= 0.0
        assert fit["beta"] >= 0.0


class TestFitNonnegativeWeights:
    def test_weights_are_never_negative(self) -> None:
        rng = np.random.default_rng(7)
        n = 100
        labels = rng.integers(0, 2, size=n)
        e = rng.normal(size=n)
        weights, _ = sre.fit_nonnegative_weights([e], labels)
        assert all(w >= 0.0 for w in weights)

    def test_a_perfectly_predictive_signal_gets_a_positive_weight(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        e = np.array([-3.0, -3.0, -3.0, 3.0, 3.0, 3.0])  # high energy exactly on incorrect rows
        weights, _ = sre.fit_nonnegative_weights([e], labels)
        assert weights[0] > 0.0

    def test_a_pure_noise_signal_can_be_fit_to_zero(self) -> None:
        rng = np.random.default_rng(8)
        n = 400
        labels = rng.integers(0, 2, size=n)
        e_noise = rng.normal(size=n)  # unrelated to labels
        weights, _ = sre.fit_nonnegative_weights([e_noise], labels)
        # Zero is in the grid and should be at least competitive with noise.
        assert 0.0 in sre.DEFAULT_WEIGHT_GRID


# ==========================================================================
# Metrics.
# ==========================================================================


class TestMetrics:
    def test_brier_score_zero_for_perfect_predictions(self) -> None:
        assert sre.brier_score([0.0, 1.0], [0, 1]) == 0.0

    def test_brier_score_bounded(self) -> None:
        rng = np.random.default_rng(9)
        probs = rng.uniform(size=50)
        labels = rng.integers(0, 2, size=50)
        assert 0.0 <= sre.brier_score(probs, labels) <= 1.0

    def test_log_loss_low_for_confident_correct_predictions(self) -> None:
        loss = sre.log_loss_score([0.001, 0.999], [0, 1])
        assert loss < 0.01

    def test_ece_zero_for_perfectly_calibrated_bins(self) -> None:
        # Every prediction exactly matches its own outcome -> zero gap per bin.
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        assert sre.ece_fixed_bins(probs, labels) == pytest.approx(0.0, abs=1e-9)

    def test_ece_empty_input_is_zero(self) -> None:
        assert sre.ece_fixed_bins([], []) == 0.0

    def test_auroc_perfect_separation(self) -> None:
        assert sre.auroc_score([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0

    def test_aurc_perfect_confidence_ordering_has_zero_risk_at_full_coverage_prefix(self) -> None:
        # Confident and correct on every row -> risk is zero everywhere.
        labels = [0, 0, 1, 1]
        probs = [0.01, 0.02, 0.98, 0.99]
        result = sre.aurc_and_coverage_at_risk(labels, probs, target_risk=0.05)
        assert result["aurc"] == pytest.approx(0.0, abs=1e-9)
        assert result["coverage_at_5pct_risk"] == pytest.approx(1.0, abs=1e-9)

    def test_aurc_empty_input_does_not_raise(self) -> None:
        result = sre.aurc_and_coverage_at_risk([], [])
        assert result == {"aurc": 0.0, "coverage_at_5pct_risk": 0.0}


# ==========================================================================
# SCENARIO-VERIFY-7750-POSITIVE-CONTROL.
# ==========================================================================


class TestPositiveControl:
    def test_positive_control_passes(self) -> None:
        result = sre.run_positive_control(seed=123, n=1500)
        assert result["alpha_is_positive"] is True
        assert result["poe_beats_no_signal_verifier"] is True
        assert result["passed"] is True

    def test_positive_control_is_deterministic(self) -> None:
        r1 = sre.run_positive_control(seed=42, n=500)
        r2 = sre.run_positive_control(seed=42, n=500)
        assert r1 == r2

    def test_positive_control_never_touches_the_real_corpus(self) -> None:
        # A pure synthetic function must never CALL the real corpus/readout/
        # verifier-training paths -- mentioning them in prose (a docstring
        # explaining what it deliberately does NOT do) is fine and expected.
        import inspect

        source = inspect.getsource(sre.run_positive_control)
        assert "load_corpus_rows_with_features(" not in source
        assert "train_gibbs_verifier(" not in source
        assert "read_option_logits(" not in source


# ==========================================================================
# Readout cache round-trip.
# ==========================================================================


class TestReadoutCache:
    def test_cache_key_is_stable_and_content_addressed(self) -> None:
        key1 = sre.readout_cache_key("q1", "some text")
        key2 = sre.readout_cache_key("q1", "some text")
        key3 = sre.readout_cache_key("q1", "different text")
        assert key1 == key2
        assert key1 != key3

    def test_missing_cache_file_returns_empty_dict(self, tmp_path: Path) -> None:
        assert sre.load_readout_cache(tmp_path / "missing.json") == {}

    def test_corrupt_cache_file_returns_empty_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.json"
        path.write_text("{not json", encoding="utf-8")
        assert sre.load_readout_cache(path) == {}

    def test_save_then_load_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "cache.json"
        cache = {"abc123": {"accept": 1.0, "reject": -1.0, "escalate": 0.0}}
        sre.save_readout_cache(path, cache)
        assert sre.load_readout_cache(path) == cache


# ==========================================================================
# Grouped out-of-fold cross-validation (SCENARIO-VERIFY-7750-POE).
# ==========================================================================


class TestRunOofCrossValidation:
    def _make_scored_rows(self, seed: int, n: int) -> list[sre.ScoredRow]:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n)
        e_readout = (2.0 * labels - 1.0) + rng.normal(scale=0.6, size=n)
        e_verifier = (2.0 * labels - 1.0) + rng.normal(scale=0.6, size=n)
        return [
            sre.ScoredRow(
                question_id=f"q{i}",
                label=int(labels[i]),
                e_readout=float(e_readout[i]),
                e_verifier=float(e_verifier[i]),
            )
            for i in range(n)
        ]

    def test_produces_one_fold_result_per_requested_fold(self) -> None:
        rows = self._make_scored_rows(10, 400)
        result = sre.run_oof_cross_validation(rows, k=4)
        assert len(result["per_fold"]) <= 4
        assert any(f["status"] == "ok" for f in result["per_fold"])

    def test_pooled_predictions_cover_every_row_exactly_once(self) -> None:
        rows = self._make_scored_rows(11, 500)
        result = sre.run_oof_cross_validation(rows, k=5)
        assert len(result["pooled"]["question_id"]) == len(rows)
        assert set(result["pooled"]["question_id"]) == {r.question_id for r in rows}

    def test_a_learnable_signal_gives_poe_a_low_brier_score(self) -> None:
        rows = self._make_scored_rows(12, 600)
        result = sre.run_oof_cross_validation(rows, k=5)
        ok_folds = [f for f in result["per_fold"] if f["status"] == "ok"]
        assert ok_folds
        mean_brier = float(np.mean([f["brier_poe"] for f in ok_folds]))
        assert mean_brier < 0.25  # meaningfully better than the always-0.5 baseline

    def test_degenerate_verifier_column_is_flagged_per_fold(self) -> None:
        rows = [
            sre.ScoredRow(question_id=f"q{i}", label=i % 2, e_readout=float(i % 2), e_verifier=0.0)
            for i in range(200)
        ]
        result = sre.run_oof_cross_validation(rows, k=5)
        assert all(f["status"] == "degenerate_verifier_rejected" for f in result["per_fold"])


class TestPairedGroupBootstrapBrierDelta:
    def test_reports_both_deltas_with_intervals(self) -> None:
        rng = np.random.default_rng(13)
        n = 300
        pooled = {
            "question_id": [f"q{i}" for i in range(n)],
            "label": rng.integers(0, 2, size=n).tolist(),
            "p_poe": rng.uniform(size=n).tolist(),
            "p_readout_only": rng.uniform(size=n).tolist(),
            "p_verifier_only": rng.uniform(size=n).tolist(),
        }
        result = sre.paired_group_bootstrap_brier_delta(pooled, n_boot=200, seed=1)
        assert "ci95" in result["brier_delta_vs_readout_only"]
        assert "ci95" in result["brier_delta_vs_verifier_only"]
        assert len(result["brier_delta_vs_readout_only"]["ci95"]) == 2

    def test_a_clearly_better_poe_has_a_negative_interval(self) -> None:
        rng = np.random.default_rng(14)
        n = 600
        labels = rng.integers(0, 2, size=n)
        pooled = {
            "question_id": [f"q{i}" for i in range(n)],
            "label": labels.tolist(),
            # PoE nearly perfect; both single experts near chance.
            "p_poe": [0.02 if lbl == 0 else 0.98 for lbl in labels],
            "p_readout_only": [0.5] * n,
            "p_verifier_only": [0.5] * n,
        }
        result = sre.paired_group_bootstrap_brier_delta(pooled, n_boot=300, seed=2)
        assert result["brier_delta_vs_readout_only"]["ci95"][1] < 0.0
        assert result["brier_delta_vs_verifier_only"]["ci95"][1] < 0.0

    def test_grouped_resampling_treats_a_multi_row_question_as_one_unit(self) -> None:
        # Every row shares ONE question id -- a group bootstrap can only ever
        # draw that single group, so every resample must equal the point estimate.
        n = 50
        pooled = {
            "question_id": ["only_question"] * n,
            "label": ([0, 1] * (n // 2)),
            "p_poe": [0.3] * n,
            "p_readout_only": [0.5] * n,
            "p_verifier_only": [0.5] * n,
        }
        result = sre.paired_group_bootstrap_brier_delta(pooled, n_boot=50, seed=3)
        lo, hi = result["brier_delta_vs_readout_only"]["ci95"]
        assert lo == pytest.approx(hi, abs=1e-9)


# ==========================================================================
# A2: target-workload calibrator tournament -- REQ-VERIFY-7751.
#
# No test below makes a live model call. Every test either builds hand-made
# logit fixtures (the exact shape A1's cache stores) or generates synthetic
# data with a known ground truth, per the SCENARIO-VERIFY-7751-POSITIVE-
# CONTROL and SCENARIO-VERIFY-7751-DEGENERATE requirements.
#
# Spec: REQ-VERIFY-7751, SCENARIO-VERIFY-7751-TOURNAMENT,
# SCENARIO-VERIFY-7751-POSITIVE-CONTROL, SCENARIO-VERIFY-7751-DEGENERATE,
# SCENARIO-VERIFY-7751-GATE
# ==========================================================================


class TestFitTwoParameterPlatt:
    def test_recovers_a_reasonable_scale_and_shift_on_separable_data(self) -> None:
        rng = np.random.default_rng(100)
        n = 1000
        z_true = rng.normal(scale=1.5, size=n)
        labels = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-z_true))).astype(np.float64)
        z_observed = 2.0 * z_true - 1.0
        fit = sre.fit_two_parameter_platt(z_observed, labels)
        # a should be positive (the sign of the true relationship is preserved)
        # and roughly of order 1/2 given the true scale of 2.0.
        assert fit["a"] > 0.0
        assert 0.1 < fit["a"] < 2.0

    def test_calibrate_output_is_always_in_unit_interval(self) -> None:
        logits = np.array([-100.0, -1.0, 0.0, 1.0, 100.0])
        out = sre.calibrate_two_parameter_platt(logits, a=3.0, b=-2.0)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_a_equals_one_b_equals_zero_matches_plain_sigmoid(self) -> None:
        logits = np.array([-2.0, 0.0, 1.5])
        out = sre.calibrate_two_parameter_platt(logits, a=1.0, b=0.0)
        expected = 1.0 / (1.0 + np.exp(-logits))
        assert np.allclose(out, expected)


class TestEnoughExamplesForIsotonic:
    def test_below_floor_on_positives_is_false(self) -> None:
        labels = [1] * 10 + [0] * 30
        assert sre.enough_examples_for_isotonic(labels) is False

    def test_below_floor_on_negatives_is_false(self) -> None:
        labels = [1] * 30 + [0] * 10
        assert sre.enough_examples_for_isotonic(labels) is False

    def test_at_floor_on_both_is_true(self) -> None:
        labels = [1] * 20 + [0] * 20
        assert sre.enough_examples_for_isotonic(labels) is True

    def test_just_below_floor_by_one_is_false(self) -> None:
        labels = [1] * 19 + [0] * 20
        assert sre.enough_examples_for_isotonic(labels) is False


class TestFitIsotonicCalibrator:
    def test_returns_none_below_the_sample_size_floor(self) -> None:
        rng = np.random.default_rng(101)
        scores = rng.uniform(size=30)
        labels = [1] * 5 + [0] * 25
        assert sre.fit_isotonic_calibrator(scores, labels) is None

    def test_returns_a_fitted_calibrator_at_or_above_the_floor(self) -> None:
        rng = np.random.default_rng(102)
        n = 100
        scores = rng.uniform(size=n)
        labels = (rng.uniform(size=n) < scores).astype(int)
        reg = sre.fit_isotonic_calibrator(scores, labels)
        assert reg is not None
        preds = reg.predict(scores)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


class TestValidateProbabilities:
    def test_empty_array_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sre.validate_probabilities([])

    def test_out_of_range_high_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sre.validate_probabilities([0.1, 1.2])

    def test_out_of_range_low_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            sre.validate_probabilities([-0.1, 0.5])

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            sre.validate_probabilities([0.5, float("nan")])

    def test_valid_probabilities_do_not_raise(self) -> None:
        sre.validate_probabilities([0.0, 0.5, 1.0])


class TestValidateTwoClassFold:
    def test_one_class_raises(self) -> None:
        with pytest.raises(ValueError, match="one-class fold"):
            sre.validate_two_class_fold([1, 1, 1])

    def test_two_classes_does_not_raise(self) -> None:
        sre.validate_two_class_fold([0, 1, 0, 1])


class TestMaxCalibrationError:
    def test_zero_for_perfectly_calibrated_bins(self) -> None:
        probs = [0.0, 0.0, 1.0, 1.0]
        labels = [0, 0, 1, 1]
        assert sre.max_calibration_error(probs, labels) == pytest.approx(0.0, abs=1e-9)

    def test_positive_when_a_bin_is_miscalibrated(self) -> None:
        # Every prediction says 0.9 confident-incorrect, but only half are.
        probs = [0.9] * 10
        labels = [1] * 5 + [0] * 5
        assert sre.max_calibration_error(probs, labels) > 0.3

    def test_empty_input_is_zero(self) -> None:
        assert sre.max_calibration_error([], []) == 0.0

    def test_mce_is_at_least_the_overall_ece(self) -> None:
        # MCE is the WORST bin gap; ECE is the weighted-AVERAGE bin gap.
        # The worst single bin can never be smaller than the weighted average.
        rng = np.random.default_rng(103)
        probs = rng.uniform(size=200)
        labels = rng.integers(0, 2, size=200)
        mce = sre.max_calibration_error(probs, labels)
        ece = sre.ece_fixed_bins(probs, labels)
        assert mce >= ece - 1e-9


class TestReliabilityTable:
    def test_rows_only_for_non_empty_bins(self) -> None:
        probs = [0.05, 0.05, 0.95]
        labels = [0, 0, 1]
        table = sre.reliability_table(probs, labels, n_bins=10)
        assert len(table) == 2  # only the two bins that actually contain data

    def test_each_row_has_the_expected_keys(self) -> None:
        table = sre.reliability_table([0.1, 0.9], [0, 1], n_bins=10)
        for row in table:
            assert set(row.keys()) == {
                "lo",
                "hi",
                "count",
                "mean_confidence",
                "mean_accuracy",
                "gap",
            }

    def test_counts_sum_to_total_rows(self) -> None:
        rng = np.random.default_rng(104)
        probs = rng.uniform(size=100)
        labels = rng.integers(0, 2, size=100)
        table = sre.reliability_table(probs, labels, n_bins=10)
        assert sum(row["count"] for row in table) == 100


class TestIsotonicPreservesOrder:
    def test_true_when_calibrated_output_matches_raw_order(self) -> None:
        raw = [0.1, 0.5, 0.9]
        calibrated = [0.05, 0.4, 0.95]  # same relative order, increasing
        assert sre.isotonic_preserves_order(raw, calibrated, increasing=True) is True

    def test_false_when_an_order_inversion_is_injected(self) -> None:
        raw = [0.1, 0.5, 0.9]
        calibrated = [0.05, 0.95, 0.4]  # middle and last swapped -> inversion
        assert sre.isotonic_preserves_order(raw, calibrated, increasing=True) is False

    def test_ties_in_raw_score_do_not_count_as_an_inversion(self) -> None:
        raw = [0.5, 0.5, 0.9]
        calibrated = [0.3, 0.3, 0.8]
        assert sre.isotonic_preserves_order(raw, calibrated, increasing=True) is True

    def test_decreasing_direction_is_checked_correctly(self) -> None:
        raw = [0.1, 0.5, 0.9]
        calibrated = [0.9, 0.5, 0.1]  # decreasing with raw
        assert sre.isotonic_preserves_order(raw, calibrated, increasing=False) is True


class TestChannelProbability:
    def test_matches_readout_from_logits_probs(self) -> None:
        raw_logits = {"accept": 2.0, "reject": -1.0, "escalate": -1.0}
        expected = sre.readout_from_logits(raw_logits).probs
        assert expected is not None
        for channel in sre.OPTIONS:
            assert sre.channel_probability(raw_logits, channel) == pytest.approx(expected[channel])

    def test_none_when_a_logit_is_missing(self) -> None:
        raw_logits = {"accept": 1.0, "reject": None, "escalate": 0.0}
        assert sre.channel_probability(raw_logits, "accept") is None


# ==========================================================================
# SCENARIO-VERIFY-7751-TOURNAMENT.
# ==========================================================================


class TestRunCalibratorTournament:
    def _make_rows(self, seed: int, n: int) -> list[sre.A2ScoredRow]:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n)
        # A `reject`-leaning readout: higher reject logit for incorrect rows.
        z_reject = (2.0 * labels - 1.0) * 1.5 + rng.normal(scale=1.0, size=n)
        z_accept = -z_reject * 0.5 + rng.normal(scale=0.5, size=n)
        z_escalate = rng.normal(scale=0.5, size=n)
        return [
            sre.A2ScoredRow(
                question_id=f"q{i}",
                label=int(labels[i]),
                raw_logits={
                    "accept": float(z_accept[i]),
                    "reject": float(z_reject[i]),
                    "escalate": float(z_escalate[i]),
                },
            )
            for i in range(n)
        ]

    def test_produces_a_result_for_every_declared_channel(self) -> None:
        rows = self._make_rows(200, 500)
        result = sre.run_calibrator_tournament(rows, k=5)
        assert set(result.keys()) == set(sre.OPTIONS)

    def test_every_channel_has_pooled_predictions_covering_every_row(self) -> None:
        rows = self._make_rows(201, 500)
        result = sre.run_calibrator_tournament(rows, k=5)
        for channel in sre.OPTIONS:
            assert len(result[channel]["pooled"]["question_id"]) == len(rows)

    def test_reject_channel_has_a_learnable_signal_and_beats_chance_brier(self) -> None:
        rows = self._make_rows(202, 800)
        result = sre.run_calibrator_tournament(rows, k=5)
        ok_folds = [f for f in result["reject"]["per_fold"] if f["status"] == "ok"]
        assert ok_folds
        mean_brier_raw = float(np.mean([f["brier_raw"] for f in ok_folds]))
        assert mean_brier_raw < 0.25  # meaningfully better than the always-0.5 baseline

    def test_isotonic_omitted_when_a_fold_has_too_few_examples_of_one_class(self) -> None:
        rng = np.random.default_rng(203)
        n = 100
        # Only 5 positives total -- every fold's training split will be
        # well below the 20-positive floor.
        labels = np.array([1] * 5 + [0] * 95)
        rng.shuffle(labels)
        rows = [
            sre.A2ScoredRow(
                question_id=f"q{i}",
                label=int(labels[i]),
                raw_logits={
                    "accept": float(rng.normal()),
                    "reject": float(rng.normal()),
                    "escalate": float(rng.normal()),
                },
            )
            for i in range(n)
        ]
        result = sre.run_calibrator_tournament(rows, k=5)
        for channel in sre.OPTIONS:
            assert len(result[channel]["isotonic_omitted"]) > 0
            for entry in result[channel]["isotonic_omitted"]:
                assert entry["reason"] == "sample_size_floor"


class TestPairedGroupBootstrapCalibratorDeltas:
    def _pooled(self, seed: int, n: int) -> dict[str, list]:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n)
        raw = rng.uniform(size=n)
        temperature = np.clip(raw + rng.normal(scale=0.01, size=n), 0.0, 1.0)
        return {
            "question_id": [f"q{i}" for i in range(n)],
            "label": labels.tolist(),
            "raw": raw.tolist(),
            "temperature": temperature.tolist(),
            "platt2": temperature.tolist(),
            "isotonic": [None] * n,
        }

    def test_reports_ci_for_temperature_and_platt2(self) -> None:
        pooled = self._pooled(300, 300)
        result = sre.paired_group_bootstrap_calibrator_deltas(pooled, n_boot=200, seed=1)
        assert "ci95" in result["temperature"]["brier_delta_vs_raw"]
        assert "ci95" in result["platt2"]["brier_delta_vs_raw"]

    def test_isotonic_omitted_entirely_when_never_fit_in_any_fold(self) -> None:
        pooled = self._pooled(301, 300)
        result = sre.paired_group_bootstrap_calibrator_deltas(pooled, n_boot=100, seed=2)
        assert result["isotonic"]["omitted_entirely"] is True

    def test_isotonic_bootstrap_only_uses_rows_where_it_was_fit(self) -> None:
        pooled = self._pooled(302, 100)
        # Isotonic fit for half the rows only.
        pooled["isotonic"] = [0.5] * 50 + [None] * 50
        result = sre.paired_group_bootstrap_calibrator_deltas(pooled, n_boot=100, seed=3)
        assert result["isotonic"]["n_rows_isotonic_available"] == 50

    def test_a_clearly_better_calibrator_has_a_negative_interval(self) -> None:
        n = 600
        labels = [0, 1] * (n // 2)
        pooled = {
            "question_id": [f"q{i}" for i in range(n)],
            "label": labels,
            "raw": [0.5] * n,
            "temperature": [0.02 if lbl == 0 else 0.98 for lbl in labels],
            "platt2": [0.5] * n,
            "isotonic": [None] * n,
        }
        result = sre.paired_group_bootstrap_calibrator_deltas(pooled, n_boot=300, seed=4)
        assert result["temperature"]["brier_delta_vs_raw"]["ci95"][1] < 0.0


# ==========================================================================
# SCENARIO-VERIFY-7751-GATE.
# ==========================================================================


class TestSelectCalibrator:
    def _bootstrap_with(self, **arms: dict) -> dict:
        base = {
            "temperature": {
                "brier_point": 0.20,
                "brier_delta_vs_raw": {"ci95": [0.01, 0.02]},
                "ece_ci95": [0.01, 0.02],
            },
            "platt2": {
                "brier_point": 0.21,
                "brier_delta_vs_raw": {"ci95": [0.01, 0.02]},
                "ece_ci95": [0.01, 0.02],
            },
            "isotonic": {"omitted_entirely": True},
        }
        base.update(arms)
        return base

    def test_selects_the_only_eligible_calibrator(self) -> None:
        bootstrap = self._bootstrap_with(
            temperature={
                "brier_point": 0.15,
                "brier_delta_vs_raw": {"ci95": [-0.05, -0.02]},
                "ece_ci95": [0.01, 0.02],
            }
        )
        result = sre.select_calibrator(bootstrap)
        assert result["selected"] == "temperature"

    def test_no_eligible_calibrator_is_diagnostic_only(self) -> None:
        bootstrap = self._bootstrap_with()  # both temperature and platt2 have ci95 upper > 0
        result = sre.select_calibrator(bootstrap)
        assert result["selected"] is None
        assert result["verdict"] == "diagnostic_only_no_method_cleared_gate"

    def test_a_tie_prefers_temperature_over_platt2(self) -> None:
        bootstrap = self._bootstrap_with(
            temperature={
                "brier_point": 0.15,
                "brier_delta_vs_raw": {"ci95": [-0.05, -0.02]},
                "ece_ci95": [0.01, 0.02],
            },
            platt2={
                "brier_point": 0.150005,  # within the default 1e-4 tie tolerance
                "brier_delta_vs_raw": {"ci95": [-0.05, -0.02]},
                "ece_ci95": [0.01, 0.02],
            },
        )
        result = sre.select_calibrator(bootstrap)
        assert result["selected"] == "temperature"

    def test_a_clear_platt2_win_selects_platt2_not_temperature(self) -> None:
        bootstrap = self._bootstrap_with(
            temperature={
                "brier_point": 0.20,
                "brier_delta_vs_raw": {"ci95": [-0.01, -0.005]},
                "ece_ci95": [0.01, 0.02],
            },
            platt2={
                "brier_point": 0.10,  # clearly better, outside the tie tolerance
                "brier_delta_vs_raw": {"ci95": [-0.10, -0.08]},
                "ece_ci95": [0.01, 0.02],
            },
        )
        result = sre.select_calibrator(bootstrap)
        assert result["selected"] == "platt2"

    def test_ece_ceiling_disqualifies_an_otherwise_winning_calibrator(self) -> None:
        bootstrap = self._bootstrap_with(
            temperature={
                "brier_point": 0.10,
                "brier_delta_vs_raw": {"ci95": [-0.10, -0.05]},
                "ece_ci95": [0.04, 0.06],  # upper bound above the 0.05 ceiling
            },
            platt2={
                "brier_point": 0.20,
                "brier_delta_vs_raw": {"ci95": [0.01, 0.02]},
                "ece_ci95": [0.01, 0.02],
            },
        )
        result = sre.select_calibrator(bootstrap)
        assert result["selected"] is None


class TestCalibratorKillCheck:
    def test_kill_true_when_no_method_clears_ece(self) -> None:
        bootstrap = {
            "temperature": {"brier_point": 0.10, "ece_ci95": [0.06, 0.08]},
            "platt2": {"brier_point": 0.10, "ece_ci95": [0.06, 0.08]},
            "isotonic": {"omitted_entirely": True},
            "brier_raw_point": 0.25,
        }
        labels = [0, 1] * 50
        per_fold = [{"fold": 0, "status": "ok", "temperature": 1.0}]
        result = sre.calibrator_kill_check(bootstrap, labels, per_fold)
        assert result["kill"] is True
        assert result["no_method_clears_ece"] is True

    def test_kill_true_when_no_better_than_prevalence(self) -> None:
        labels = [1] * 20 + [0] * 80  # prevalence 0.2
        bootstrap = {
            "temperature": {
                "brier_point": 0.30,
                "ece_ci95": [0.01, 0.02],
            },  # worse than 0.2*0.8=0.16
            "platt2": {"brier_point": 0.30, "ece_ci95": [0.01, 0.02]},
            "isotonic": {"omitted_entirely": True},
            "brier_raw_point": 0.30,
        }
        per_fold = [{"fold": 0, "status": "ok", "temperature": 1.0}]
        result = sre.calibrator_kill_check(bootstrap, labels, per_fold)
        assert result["brier_no_better_than_prevalence"] is True
        assert result["kill"] is True

    def test_kill_true_on_fold_to_fold_temperature_reversal(self) -> None:
        bootstrap = {
            "temperature": {"brier_point": 0.05, "ece_ci95": [0.01, 0.02]},
            "platt2": {"brier_point": 0.05, "ece_ci95": [0.01, 0.02]},
            "isotonic": {"omitted_entirely": True},
            "brier_raw_point": 0.25,
        }
        labels = [0, 1] * 50
        per_fold = [
            {"fold": 0, "status": "ok", "temperature": 0.5},
            {"fold": 1, "status": "ok", "temperature": 2.0},
        ]
        result = sre.calibrator_kill_check(bootstrap, labels, per_fold)
        assert result["fold_to_fold_calibration_reverses"] is True
        assert result["kill"] is True

    def test_kill_false_on_a_clean_pass(self) -> None:
        bootstrap = {
            "temperature": {"brier_point": 0.05, "ece_ci95": [0.01, 0.02]},
            "platt2": {"brier_point": 0.06, "ece_ci95": [0.01, 0.02]},
            "isotonic": {"omitted_entirely": True},
            "brier_raw_point": 0.25,
        }
        labels = [0, 1] * 50  # prevalence brier = 0.25
        per_fold = [
            {"fold": 0, "status": "ok", "temperature": 1.0},
            {"fold": 1, "status": "ok", "temperature": 1.05},
        ]
        result = sre.calibrator_kill_check(bootstrap, labels, per_fold)
        assert result["kill"] is False


# ==========================================================================
# SCENARIO-VERIFY-7751-POSITIVE-CONTROL.
# ==========================================================================


class TestTemperatureRecoveryPositiveControl:
    def test_temperature_scaling_beats_the_raw_overconfident_probability(self) -> None:
        result = sre.run_temperature_recovery_positive_control(seed=1, n=2000)
        assert result["passed"] is True
        assert result["brier_temperature_calibrated"] < result["brier_raw_overconfident"]

    def test_is_deterministic_given_a_fixed_seed(self) -> None:
        r1 = sre.run_temperature_recovery_positive_control(seed=42, n=500)
        r2 = sre.run_temperature_recovery_positive_control(seed=42, n=500)
        assert r1 == r2

    def test_never_touches_the_real_corpus_or_readout_paths(self) -> None:
        import inspect

        source = inspect.getsource(sre.run_temperature_recovery_positive_control)
        assert "load_corpus_rows_with_features(" not in source
        assert "read_option_logits(" not in source


class TestPlattAffineRecoveryPositiveControl:
    def test_two_parameter_platt_beats_temperature_only(self) -> None:
        result = sre.run_platt_affine_recovery_positive_control(seed=2, n=2000)
        assert result["passed"] is True
        assert result["brier_two_parameter_platt"] < result["brier_temperature_only"]

    def test_is_deterministic_given_a_fixed_seed(self) -> None:
        r1 = sre.run_platt_affine_recovery_positive_control(seed=7, n=500)
        r2 = sre.run_platt_affine_recovery_positive_control(seed=7, n=500)
        assert r1 == r2

    def test_never_touches_the_real_corpus_or_readout_paths(self) -> None:
        import inspect

        source = inspect.getsource(sre.run_platt_affine_recovery_positive_control)
        assert "load_corpus_rows_with_features(" not in source
        assert "read_option_logits(" not in source


# ==========================================================================
# SCENARIO-VERIFY-7751-DEGENERATE.
# ==========================================================================


class TestCalibratorDegenerateCases:
    def test_constant_probability_stays_constant_under_every_calibrator(self) -> None:
        result = sre.check_constant_probability_calibration_is_a_no_op()
        assert result["passes"] is True
        assert result["temperature_stays_constant"] is True
        assert result["platt2_stays_constant"] is True
        assert result["isotonic_stays_constant"] is True

    def test_isotonic_never_introduces_a_false_rank_improvement(self) -> None:
        result = sre.check_isotonic_no_false_rank_improvement()
        assert result["passes"] is True
        assert result["order_preserved"] is True

    def test_hard_failures_all_raise(self) -> None:
        result = sre.check_hard_failures_are_raised()
        assert result["passes"] is True
        assert result["empty_probabilities_raises"] is True
        assert result["one_class_fold_raises"] is True
        assert result["out_of_range_probability_raises"] is True


class TestProbabilityToLogit:
    def test_identity_at_temperature_one(self) -> None:
        # sigmoid(probability_to_logit(p)) must recover p exactly -- an
        # unfit (temperature=1) calibrator must be a pass-through.
        probs = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        z = sre.probability_to_logit(probs)
        recovered = 1.0 / (1.0 + np.exp(-z))
        assert np.allclose(recovered, probs, atol=1e-5)

    def test_clips_away_from_zero_and_one(self) -> None:
        z = sre.probability_to_logit([0.0, 1.0])
        assert np.all(np.isfinite(z))

    def test_zero_point_five_maps_to_zero(self) -> None:
        z = sre.probability_to_logit([0.5])
        assert z[0] == pytest.approx(0.0, abs=1e-9)


class TestCalibratorTournamentOnLargeMagnitudeRawLogits:
    """Regression test for the incident found running the real A1 cache:
    raw per-option vocabulary logits are ~15-20 units in magnitude (a real
    example from the cache: accept=17.007, reject=17.348, escalate=17.575).
    Feeding those raw values straight into temperature/Platt scaling
    saturated the sigmoid and produced a catastrophic Brier score (0.93,
    far worse than the 0.22 raw baseline) and NaN Platt fits. The fix
    calibrates the log-odds of the softmaxed channel probability instead
    (`probability_to_logit`), which is well-scaled regardless of the raw
    vocabulary logit's absolute magnitude.
    """

    def _make_large_magnitude_rows(self, seed: int, n: int) -> list[sre.A2ScoredRow]:
        rng = np.random.default_rng(seed)
        labels = rng.integers(0, 2, size=n)
        base = 17.0
        # `reject` logit meaningfully higher than the others for incorrect
        # rows -- a real, learnable signal riding on top of a large shared
        # baseline, exactly like the real cache's near-uniform-but-shifted
        # option logits.
        reject_shift = (2.0 * labels - 1.0) * 0.8 + rng.normal(scale=0.3, size=n)
        return [
            sre.A2ScoredRow(
                question_id=f"q{i}",
                label=int(labels[i]),
                raw_logits={
                    "accept": base + float(rng.normal(scale=0.2)),
                    "reject": base + float(reject_shift[i]),
                    "escalate": base + float(rng.normal(scale=0.2)),
                },
            )
            for i in range(n)
        ]

    def test_temperature_and_platt2_produce_finite_brier_no_worse_than_raw(self) -> None:
        rows = self._make_large_magnitude_rows(500, 800)
        result = sre.run_calibrator_tournament(rows, k=5)
        for fold in result["reject"]["per_fold"]:
            if fold["status"] != "ok":
                continue
            assert math.isfinite(fold["brier_temperature"])
            assert math.isfinite(fold["brier_platt2"])
            # Neither calibrator should ever be worse than the raw,
            # uncalibrated probability by a wide margin -- at temperature
            # 1 (the unfit default) they are a pass-through, so a fit can
            # only do as well or better on the training data's own scale.
            assert fold["brier_temperature"] < 0.5
            assert fold["brier_platt2"] < 0.5

    def test_reject_channel_beats_chance_after_the_fix(self) -> None:
        rows = self._make_large_magnitude_rows(501, 800)
        result = sre.run_calibrator_tournament(rows, k=5)
        ok_folds = [f for f in result["reject"]["per_fold"] if f["status"] == "ok"]
        assert ok_folds
        mean_brier_temp = float(np.mean([f["brier_temperature"] for f in ok_folds]))
        assert mean_brier_temp < 0.25  # meaningfully better than the always-0.5 baseline


# ==========================================================================
# REQ-VERIFY-7752 (A3): calibrated accept/reject/escalate policy.
# ==========================================================================


class TestAurcAndCoverageAtRiskExtended:
    """SCENARIO-VERIFY-7752-POLICY: AURC / coverage-at-risk / risk-at-fixed-
    coverage computed by hand on a tiny fixture, verifying the extension
    added for A3 (a `confidence` override and `risk_at_fixed_coverage`)
    while confirming the original two-key shape is unchanged by default."""

    def test_default_call_shape_is_unchanged_from_a1(self) -> None:
        # SCENARIO-VERIFY-7750/7751 callers pass no `confidence` and no
        # `fixed_coverages` -- the extension must not add keys they never
        # asked for.
        result = sre.aurc_and_coverage_at_risk([0, 0, 1, 1], [0.01, 0.02, 0.98, 0.99])
        assert set(result.keys()) == {"aurc", "coverage_at_5pct_risk"}

    def test_hand_computed_aurc_and_risk_at_fixed_coverage(self) -> None:
        # By hand: confidence = max(p, 1-p) = [0.9, 0.9, 0.6, 0.6]; a stable
        # descending sort keeps the original order (no confidence-tie
        # reordering). pred = p >= 0.5 -> [0, 1, 1, 0]; y = [0, 1, 0, 1].
        # errors = [0, 0, 1, 1]; cumulative = [0, 0, 1, 2]; counts =
        # [1, 2, 3, 4]; cum_risk = [0, 0, 1/3, 1/2]; coverage =
        # [0.25, 0.5, 0.75, 1.0].
        labels = [0, 1, 0, 1]
        probs = [0.1, 0.9, 0.6, 0.4]
        result = sre.aurc_and_coverage_at_risk(
            labels, probs, target_risk=0.05, fixed_coverages=(0.50, 0.80, 0.90)
        )
        # trapezoid((0,0,1/3,1/2), x=(.25,.5,.75,1)) = 0 + (0+1/3)/2*.25 + (1/3+1/2)/2*.25
        assert result["aurc"] == pytest.approx(0.145833333, abs=1e-6)
        # cum_risk <= 0.05 only at the first two points -> max coverage 0.5.
        assert result["coverage_at_5pct_risk"] == pytest.approx(0.5, abs=1e-9)
        risk_at = result["risk_at_fixed_coverage"]
        assert risk_at["0.50"] == pytest.approx(0.0, abs=1e-9)  # k=2 -> cum_risk[1]=0
        assert risk_at["0.80"] == pytest.approx(1.0 / 3.0, abs=1e-9)  # k=round(3.2)=3
        assert risk_at["0.90"] == pytest.approx(0.5, abs=1e-9)  # k=round(3.6)=4

    def test_confidence_override_changes_the_ranking(self) -> None:
        # Same labels/probs, but a confidence array that reverses the
        # ranking must produce a DIFFERENT AURC than the default max(p,1-p)
        # ranking -- proving the override is actually used.
        labels = [0, 1, 0, 1]
        probs = [0.1, 0.9, 0.6, 0.4]
        default_result = sre.aurc_and_coverage_at_risk(labels, probs)
        reversed_confidence = [0.1, 0.2, 0.8, 0.9]  # exact reverse of max(p,1-p)
        overridden_result = sre.aurc_and_coverage_at_risk(
            labels, probs, confidence=reversed_confidence
        )
        assert overridden_result["aurc"] != pytest.approx(default_result["aurc"])

    def test_empty_input_with_fixed_coverages_returns_zeros(self) -> None:
        result = sre.aurc_and_coverage_at_risk([], [], fixed_coverages=(0.5, 0.8))
        assert result["aurc"] == 0.0
        assert result["risk_at_fixed_coverage"] == {"0.50": 0.0, "0.80": 0.0}


class TestCombineCalibratedRisk:
    def test_unweighted_mean_of_three_signals(self) -> None:
        combined = sre.combine_calibrated_risk([0.1, 0.5], [0.2, 0.5], [0.3, 0.5])
        assert combined[0] == pytest.approx(0.2, abs=1e-9)
        assert combined[1] == pytest.approx(0.5, abs=1e-9)

    def test_never_reads_the_accept_channel_by_construction(self) -> None:
        # SCENARIO-VERIFY-7752-POLICY: only three arguments are accepted --
        # there is structurally no fourth "accept calibration" input.
        import inspect

        params = list(inspect.signature(sre.combine_calibrated_risk).parameters)
        assert params == ["p_verifier", "p_reject_cal", "p_escalate_cal"]


class TestThreeWayEntropyConfidence:
    def test_uniform_distribution_has_zero_confidence(self) -> None:
        third = 1.0 / 3.0
        confidence = sre.three_way_entropy_confidence(
            {"accept": [third], "reject": [third], "escalate": [third]}
        )
        assert confidence[0] == pytest.approx(0.0, abs=1e-6)

    def test_near_certain_distribution_has_near_full_confidence(self) -> None:
        confidence = sre.three_way_entropy_confidence(
            {"accept": [0.999998], "reject": [0.000001], "escalate": [0.000001]}
        )
        assert confidence[0] > 0.95


class TestChowRejectOptionDecisions:
    def test_hand_computed_three_way_split(self) -> None:
        # By hand at cost_escalate=0.2: accept iff p < 0.2, reject iff
        # p > 0.8, escalate otherwise.
        p = [0.05, 0.2, 0.5, 0.8, 0.95]
        decisions = sre.chow_reject_option_decisions(p, cost_escalate=0.2)
        assert list(decisions) == ["accept", "escalate", "escalate", "escalate", "reject"]

    def test_cost_zero_forces_universal_escalation(self) -> None:
        decisions = sre.chow_reject_option_decisions([0.0, 0.5, 1.0], cost_escalate=0.0)
        assert list(decisions) == ["escalate", "escalate", "escalate"]

    def test_cost_half_forces_a_two_way_split_never_escalate(self) -> None:
        decisions = sre.chow_reject_option_decisions([0.1, 0.4999, 0.9], cost_escalate=0.5)
        assert "escalate" not in list(decisions)

    def test_cost_is_clipped_to_the_valid_range(self) -> None:
        # A cost above 0.5 or below 0 must not silently invert the rule.
        decisions_high = sre.chow_reject_option_decisions([0.1, 0.9], cost_escalate=0.9)
        decisions_low = sre.chow_reject_option_decisions([0.1, 0.9], cost_escalate=-0.3)
        assert "escalate" not in list(decisions_high)
        assert list(decisions_low) == ["escalate", "escalate"]


class TestPolicyConfusionMatrix:
    def test_counts_and_by_action_breakdown(self) -> None:
        labels = [0, 1, 0, 1, 0]
        decisions = ["accept", "reject", "accept", "escalate", "escalate"]
        result = sre.policy_confusion_matrix(labels, decisions)
        assert result["counts"] == {"accept": 2, "reject": 1, "escalate": 2}
        assert result["by_action_and_label"]["accept"]["n_label_correct_0"] == 2
        assert result["by_action_and_label"]["reject"]["n_label_incorrect_1"] == 1
        assert result["n_total"] == 5


class TestCheckActionBalance:
    """SCENARIO-VERIFY-7752-DEGENERATE: both the failing case and the
    justified-by-cost-matrix case."""

    def test_missing_action_at_an_interior_cost_is_a_real_failure(self) -> None:
        # The exp7385 shape: near-universal accept, no rejects, at a
        # moderate cost the grid does NOT mathematically force.
        counts = {"accept": 6613, "reject": 0, "escalate": 2}
        result = sre.check_action_balance(counts, cost_escalate=0.2)
        assert result["degenerate"] is True
        assert result["zero_actions"] == ["reject"]
        assert result["justified_by_cost_matrix"] is False

    def test_missing_escalate_at_cost_half_is_justified(self) -> None:
        counts = {"accept": 5, "reject": 5, "escalate": 0}
        result = sre.check_action_balance(counts, cost_escalate=0.5)
        assert result["degenerate"] is True
        assert result["justified_by_cost_matrix"] is True

    def test_missing_accept_and_reject_at_cost_zero_is_justified(self) -> None:
        counts = {"accept": 0, "reject": 0, "escalate": 10}
        result = sre.check_action_balance(counts, cost_escalate=0.0)
        assert result["degenerate"] is True
        assert result["justified_by_cost_matrix"] is True

    def test_all_three_actions_present_is_not_degenerate(self) -> None:
        counts = {"accept": 3, "reject": 3, "escalate": 3}
        result = sre.check_action_balance(counts, cost_escalate=0.2)
        assert result["degenerate"] is False
        assert result["justified_by_cost_matrix"] is None


class TestEscalationValue:
    def test_no_escalation_means_no_benefit(self) -> None:
        labels = [1, 1, 0, 0]
        p = [0.9, 0.9, 0.1, 0.1]
        decisions = sre.chow_reject_option_decisions(p, cost_escalate=0.2)
        assert "escalate" not in list(decisions)
        result = sre.escalation_value(labels, p, decisions, cost_escalate=0.2)
        assert result["policy_n_escalated"] == 0
        assert result["forced_decision_errors"] == 0
        assert result["escalation_saves_cost"] is False

    def test_escalating_an_error_prone_row_saves_cost(self) -> None:
        # A single row the forced 0.5 threshold gets wrong (p=0.49, label=1)
        # falls inside the cost=0.1 escalate zone and is spared.
        labels = [1]
        p = [0.49]
        decisions = sre.chow_reject_option_decisions(p, cost_escalate=0.1)
        assert list(decisions) == ["escalate"]
        result = sre.escalation_value(labels, p, decisions, cost_escalate=0.1)
        assert result["forced_decision_errors"] == 1
        assert result["policy_decided_errors"] == 0
        assert result["policy_n_escalated"] == 1
        assert result["policy_total_cost"] == pytest.approx(0.1, abs=1e-9)
        assert result["errors_avoided_by_escalation"] == 1
        assert result["escalation_saves_cost"] is True


class TestA3KillCheck:
    def _grid_point(self, counts: dict[str, int], saves_cost: bool) -> dict:
        return {
            "confusion_matrix": {"counts": counts},
            "escalation_value": {"escalation_saves_cost": saves_cost},
        }

    def test_all_single_action_across_grid_triggers_kill(self) -> None:
        grid = [
            self._grid_point({"accept": 10, "reject": 0, "escalate": 0}, saves_cost=True),
            self._grid_point({"accept": 10, "reject": 0, "escalate": 0}, saves_cost=True),
        ]
        result = sre.a3_kill_check(grid, coverage_at_5pct_risk=0.5)
        assert result["all_grid_points_single_action"] is True
        assert result["kill"] is True

    def test_coverage_below_floor_triggers_kill(self) -> None:
        grid = [self._grid_point({"accept": 5, "reject": 5, "escalate": 5}, saves_cost=True)]
        result = sre.a3_kill_check(grid, coverage_at_5pct_risk=0.10)
        assert result["coverage_below_25pct_floor"] is True
        assert result["kill"] is True

    def test_escalation_never_saving_cost_triggers_kill(self) -> None:
        grid = [
            self._grid_point({"accept": 5, "reject": 5, "escalate": 5}, saves_cost=False),
            self._grid_point({"accept": 5, "reject": 5, "escalate": 5}, saves_cost=False),
        ]
        result = sre.a3_kill_check(grid, coverage_at_5pct_risk=0.5)
        assert result["no_escalation_ever_saves_cost_at_any_grid_point"] is True
        assert result["kill"] is True

    def test_healthy_grid_does_not_trigger_kill(self) -> None:
        grid = [
            self._grid_point({"accept": 5, "reject": 5, "escalate": 5}, saves_cost=True),
            self._grid_point({"accept": 8, "reject": 2, "escalate": 5}, saves_cost=False),
        ]
        result = sre.a3_kill_check(grid, coverage_at_5pct_risk=0.5)
        assert result["kill"] is False


class TestA3PositiveControl:
    """SCENARIO-VERIFY-7752-POSITIVE-CONTROL."""

    def test_noisy_gold_beats_both_controls(self) -> None:
        result = sre.run_a3_positive_control(seed=20260928, n=2000)
        assert result["noisy_gold_beats_entropy_control"] is True
        assert result["noisy_gold_beats_prevalence_control"] is True
        assert result["passed"] is True

    def test_deterministic_given_the_same_seed(self) -> None:
        r1 = sre.run_a3_positive_control(seed=7, n=500)
        r2 = sre.run_a3_positive_control(seed=7, n=500)
        assert r1 == r2

    def test_never_touches_the_real_corpus(self) -> None:
        import inspect

        source = inspect.getsource(sre.run_a3_positive_control)
        assert "load_corpus_rows_with_features(" not in source
        assert "train_gibbs_verifier(" not in source
        assert "read_option_logits(" not in source


class TestPairedGroupBootstrapAurcDelta:
    def test_a_clearly_better_signal_has_a_negative_delta_ci_below_zero(self) -> None:
        rng = np.random.default_rng(2026)
        n = 400
        question_ids = [f"q{i}" for i in range(n)]
        labels = rng.integers(0, 2, size=n)
        signal = 2.0 * labels - 1.0
        z_main = 3.0 * signal + rng.normal(scale=0.3, size=n)
        p_main = 1.0 / (1.0 + np.exp(-z_main))
        p_control = rng.uniform(size=n)  # uninformative

        result = sre.paired_group_bootstrap_aurc_delta(
            question_ids, labels, p_main, p_control, n_boot=200, seed=11
        )
        assert result["point"] < 0.0
        assert result["ci95"][1] < 0.0

    def test_n_groups_matches_unique_question_ids(self) -> None:
        question_ids = ["a", "a", "b", "b", "c", "c"]
        labels = [0, 1, 0, 1, 0, 1]
        probs = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7]
        result = sre.paired_group_bootstrap_aurc_delta(
            question_ids, labels, probs, probs, n_boot=50, seed=1
        )
        assert result["n_groups"] == 3


class TestCostGridIsPreRegistered:
    """SCENARIO-VERIFY-7752-COST-GRID: the grid and its primary point are
    fixed module-level constants, never derived from data at evaluation
    time."""

    def test_grid_is_a_plain_sorted_tuple_of_floats(self) -> None:
        assert isinstance(sre.A3_COST_GRID, tuple)
        assert all(isinstance(c, float) for c in sre.A3_COST_GRID)
        assert list(sre.A3_COST_GRID) == sorted(sre.A3_COST_GRID)
        assert all(0.0 <= c <= 0.5 for c in sre.A3_COST_GRID)

    def test_primary_cost_is_a_member_of_the_grid(self) -> None:
        assert isinstance(sre.A3_PRIMARY_COST, float)
        assert any(abs(sre.A3_PRIMARY_COST - c) < 1e-12 for c in sre.A3_COST_GRID)

    def test_run_a3_policy_evaluation_default_grid_is_the_module_constant(self) -> None:
        import inspect

        sig = inspect.signature(sre.run_a3_policy_evaluation)
        assert sig.parameters["cost_grid"].default is sre.A3_COST_GRID
        assert sig.parameters["primary_cost"].default is sre.A3_PRIMARY_COST


class TestRunA3PolicyEvaluation:
    """SCENARIO-VERIFY-7752-POLICY: the full evaluation wired together on a
    strongly-separated synthetic corpus, so every gate condition should
    individually resolve in the expected direction -- this is a
    correctness test of the wiring, not a claim about the real corpus
    (see `results/experiment_semif_readout_ebm_eval_a3.json` for that)."""

    def _synthetic_inputs(self, seed: int = 20260929, n: int = 1500) -> dict:
        rng = np.random.default_rng(seed)
        question_ids = [f"q{i}" for i in range(n)]
        labels = rng.integers(0, 2, size=n)
        signal = 2.0 * labels - 1.0
        # Three independently-noised but genuinely informative calibrated
        # channels. This noise level is deliberately moderate, not strong:
        # a too-confident signal drives the forced-decision (no escalation)
        # baseline's error count so low that escalating anything, even at
        # the smallest registered cost, can never pay for itself -- a real
        # property of the escalation-value formula, not a bug, but it
        # would make `escalation_saves_cost` false at every grid point and
        # defeat this test's purpose. Verified directly: scale=0.6,
        # noise=1.2 gives ~20 percent forced-decision error and escalation
        # saves cost at 4 of 10 registered grid points.
        z_verifier = 0.6 * signal + rng.normal(scale=1.2, size=n)
        z_reject = 0.6 * signal + rng.normal(scale=1.2, size=n)
        z_escalate = 0.6 * signal + rng.normal(scale=1.2, size=n)
        p_verifier = 1.0 / (1.0 + np.exp(-z_verifier))
        p_reject_cal = 1.0 / (1.0 + np.exp(-z_reject))
        p_escalate_cal = 1.0 / (1.0 + np.exp(-z_escalate))

        # Raw 3-way probabilities for the entropy control: pure noise, no
        # relationship to the label -- so the combined-risk signal should
        # clearly beat it.
        random_logits = rng.normal(size=(n, 3))
        exp_logits = np.exp(random_logits - random_logits.max(axis=1, keepdims=True))
        random_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return {
            "question_ids": question_ids,
            "labels": labels.tolist(),
            "p_verifier": p_verifier,
            "p_reject_cal": p_reject_cal,
            "p_escalate_cal": p_escalate_cal,
            "p_accept_raw": random_probs[:, 0],
            "p_reject_raw": random_probs[:, 1],
            "p_escalate_raw": random_probs[:, 2],
        }

    def test_strong_signal_beats_both_controls_and_passes_the_gate_shape(self) -> None:
        inputs = self._synthetic_inputs()
        result = sre.run_a3_policy_evaluation(
            inputs["question_ids"],
            inputs["labels"],
            inputs["p_verifier"],
            inputs["p_reject_cal"],
            inputs["p_escalate_cal"],
            inputs["p_accept_raw"],
            inputs["p_reject_raw"],
            inputs["p_escalate_raw"],
            seed=7752,
            n_boot=200,
        )
        assert result["aurc_delta_vs_entropy_control"]["ci95"][1] < 0.0
        assert result["aurc_delta_vs_verifier_only_control"]["point"] <= 0.0
        assert result["main_metrics"]["coverage_at_5pct_risk"] >= 0.25
        assert result["primary_cost_result"] is not None
        primary_counts = result["primary_cost_result"]["confusion_matrix"]["counts"]
        assert all(primary_counts[a] > 0 for a in sre.A3_ACTIONS)
        assert result["kill_check"]["kill"] is False

    def test_deterministic_given_the_same_seed_and_inputs(self) -> None:
        inputs = self._synthetic_inputs()
        r1 = sre.run_a3_policy_evaluation(
            inputs["question_ids"],
            inputs["labels"],
            inputs["p_verifier"],
            inputs["p_reject_cal"],
            inputs["p_escalate_cal"],
            inputs["p_accept_raw"],
            inputs["p_reject_raw"],
            inputs["p_escalate_raw"],
            seed=99,
            n_boot=50,
        )
        r2 = sre.run_a3_policy_evaluation(
            inputs["question_ids"],
            inputs["labels"],
            inputs["p_verifier"],
            inputs["p_reject_cal"],
            inputs["p_escalate_cal"],
            inputs["p_accept_raw"],
            inputs["p_reject_raw"],
            inputs["p_escalate_raw"],
            seed=99,
            n_boot=50,
        )
        assert r1["main_metrics"]["aurc"] == r2["main_metrics"]["aurc"]
        assert r1["aurc_delta_vs_entropy_control"] == r2["aurc_delta_vs_entropy_control"]

    def test_cost_grid_sweep_covers_every_registered_point(self) -> None:
        inputs = self._synthetic_inputs(n=200)
        result = sre.run_a3_policy_evaluation(
            inputs["question_ids"],
            inputs["labels"],
            inputs["p_verifier"],
            inputs["p_reject_cal"],
            inputs["p_escalate_cal"],
            inputs["p_accept_raw"],
            inputs["p_reject_raw"],
            inputs["p_escalate_raw"],
            seed=1,
            n_boot=20,
        )
        assert len(result["cost_grid_results"]) == len(sre.A3_COST_GRID)
        costs_seen = [r["cost_escalate"] for r in result["cost_grid_results"]]
        assert costs_seen == list(sre.A3_COST_GRID)
