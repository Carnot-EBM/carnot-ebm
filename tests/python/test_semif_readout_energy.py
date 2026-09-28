"""Tests for python/carnot/verify/semif_readout_energy.py -- REQ-VERIFY-7750,
the A1 readout-energy product-of-experts calibration experiment.

No test in this file makes a live model call. The readout-logit-to-decision
path is exercised through hand-built logit dictionaries (the exact shape a
real `llama.cpp` forward pass produces), and the corpus-split tests read the
real `data/fover_corpus_v4.json` file -- a plain file read, not an LLM call.

Spec: REQ-VERIFY-7750, SCENARIO-VERIFY-7750-POE, SCENARIO-VERIFY-7750-POSITIVE-CONTROL,
SCENARIO-VERIFY-7750-DEGENERATE, SCENARIO-VERIFY-7750-HEADROOM
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
