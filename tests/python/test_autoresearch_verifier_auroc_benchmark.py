"""Tests for python/carnot/autoresearch/verifier_auroc_benchmark.py --
fitness target #2, closing the "no reusable AUROC harness exists yet" gap
named in autoresearch_conductor_round.py's own docstring.

Spec: REQ-AUTO-025
"""

from __future__ import annotations

from carnot.autoresearch import verifier_auroc_benchmark as vab


class TestBucketOf:
    def test_deterministic_across_calls(self) -> None:
        assert vab._bucket_of("156") == vab._bucket_of("156")

    def test_in_range(self) -> None:
        for qid in ("1", "2", "abc", "9999", None):
            bucket = vab._bucket_of(qid)
            assert 0 <= bucket < vab._BUCKET_MODULUS


class TestSplitCorpus:
    def test_missing_corpus_returns_empty_rows(self, monkeypatch, tmp_path) -> None:
        """REQ-AUTO-025: an unreadable corpus is unscoreable, not fatal."""
        monkeypatch.setattr(vab, "repo_path", lambda *_parts: tmp_path / "missing.json")
        vab._load_corpus_rows.cache_clear()
        try:
            assert vab._load_corpus_rows() == ()
        finally:
            vab._load_corpus_rows.cache_clear()

    def test_non_list_corpus_returns_empty_rows(self, monkeypatch, tmp_path) -> None:
        """REQ-AUTO-025: the corpus root must be a JSON list."""
        corpus_path = tmp_path / "not-a-list.json"
        corpus_path.write_text('{"row": "not a list"}', encoding="utf-8")
        monkeypatch.setattr(vab, "repo_path", lambda *_parts: corpus_path)
        vab._load_corpus_rows.cache_clear()
        try:
            assert vab._load_corpus_rows() == ()
        finally:
            vab._load_corpus_rows.cache_clear()

    def test_corrupt_json_corpus_returns_empty_rows(self, monkeypatch, tmp_path) -> None:
        """REQ-AUTO-025: corrupt JSON is unscoreable, not fatal."""
        corpus_path = tmp_path / "corrupt.json"
        corpus_path.write_text("{not-json", encoding="utf-8")
        monkeypatch.setattr(vab, "repo_path", lambda *_parts: corpus_path)
        vab._load_corpus_rows.cache_clear()
        try:
            assert vab._load_corpus_rows() == ()
        finally:
            vab._load_corpus_rows.cache_clear()

    def test_train_and_held_out_are_disjoint(self) -> None:
        train, held_out = vab._split_corpus()
        train_qids = {r.get("question_id") for r in train}
        held_out_qids = {r.get("question_id") for r in held_out}
        assert not (train_qids & held_out_qids)

    def test_both_splits_are_non_empty(self) -> None:
        train, held_out = vab._split_corpus()
        assert len(train) > 0
        assert len(held_out) > 0

    def test_held_out_has_both_classes(self) -> None:
        """The benchmark is meaningless if the held-out split has no
        positive (incorrect) or no negative (correct) examples."""
        _, held_out = vab._split_corpus()
        labels = {r.get("label") for r in held_out}
        assert "correct" in labels
        assert "incorrect" in labels

    def test_split_is_stable_across_calls(self) -> None:
        train1, held_out1 = vab._split_corpus()
        train2, held_out2 = vab._split_corpus()
        assert train1 == train2
        assert held_out1 == held_out2


class TestTrainRowsForPrompt:
    def test_returns_only_step_text_and_label(self) -> None:
        rows = vab.train_rows_for_prompt()
        assert rows
        for row in rows[:5]:
            assert set(row.keys()) == {"step_text", "label"}

    def test_matches_the_train_split(self) -> None:
        train, _ = vab._split_corpus()
        rows = vab.train_rows_for_prompt()
        assert len(rows) == len(train)


class TestBinaryAuroc:
    def test_perfect_separation_is_one(self) -> None:
        assert vab._binary_auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0

    def test_perfect_inversion_is_zero(self) -> None:
        assert vab._binary_auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0

    def test_ties_count_as_half_a_win(self) -> None:
        assert vab._binary_auroc([1, 0], [0.5, 0.5]) == 0.5

    def test_no_positive_examples_returns_none(self) -> None:
        assert vab._binary_auroc([0, 0, 0], [0.1, 0.2, 0.3]) is None

    def test_no_negative_examples_returns_none(self) -> None:
        assert vab._binary_auroc([1, 1, 1], [0.1, 0.2, 0.3]) is None


class TestValidateWeights:
    def test_valid_pair_of_floats(self) -> None:
        assert vab._validate_weights([0.5, 0.5]) == (0.5, 0.5)

    def test_accepts_negative_weights(self) -> None:
        assert vab._validate_weights([-1.0, 1.0]) == (-1.0, 1.0)

    def test_wrong_length_is_none(self) -> None:
        assert vab._validate_weights([0.5]) is None
        assert vab._validate_weights([0.5, 0.5, 0.5]) is None

    def test_non_list_is_none(self) -> None:
        assert vab._validate_weights("not a list") is None
        assert vab._validate_weights(None) is None

    def test_non_numeric_entries_are_none(self) -> None:
        assert vab._validate_weights(["a", "b"]) is None

    def test_nan_and_inf_are_none(self) -> None:
        assert vab._validate_weights([float("nan"), 0.0]) is None
        assert vab._validate_weights([float("inf"), 0.0]) is None
        assert vab._validate_weights([0.0, float("-inf")]) is None

    def test_out_of_range_is_none(self) -> None:
        assert vab._validate_weights([vab.MAX_ABS_WEIGHT + 1.0, 0.0]) is None

    def test_overflow_error_is_none_not_raised(self) -> None:
        """2026-09-16 adversarial review, REAL_BUG 3: `float(10**400)` raises
        OverflowError, not TypeError/ValueError -- confirm it is now caught."""
        assert vab._validate_weights([10**400, 0.0]) is None
        assert vab._validate_weights([0.0, 10**400]) is None

    def test_at_the_boundary_is_valid(self) -> None:
        assert vab._validate_weights([vab.MAX_ABS_WEIGHT, -vab.MAX_ABS_WEIGHT]) == (
            vab.MAX_ABS_WEIGHT,
            -vab.MAX_ABS_WEIGHT,
        )


class TestRecomputeVerifierAurocEnergy:
    def test_real_weights_recompute_a_real_energy_in_range(self) -> None:
        energy = vab.recompute_verifier_auroc_energy([0.5, 0.5])
        assert energy is not None
        assert 0.0 <= energy <= 1.0

    def test_malformed_state_returns_none(self) -> None:
        assert vab.recompute_verifier_auroc_energy(None) is None
        assert vab.recompute_verifier_auroc_energy("not a state") is None
        assert vab.recompute_verifier_auroc_energy([1.0]) is None

    def test_a_fabricated_number_cannot_influence_this_at_all(self) -> None:
        """There is no 'claimed AUROC' parameter -- only the weights matter,
        and the same weights always recompute to the same energy."""
        e1 = vab.recompute_verifier_auroc_energy([0.5, 0.5])
        e2 = vab.recompute_verifier_auroc_energy([0.5, 0.5])
        assert e1 == e2

    def test_different_weights_can_change_the_energy(self) -> None:
        """Real, measured headroom: the probe's own documented defaults
        (0.5, 0.5) score worse than a sign-flipped pair on this corpus."""
        default_energy = vab.recompute_verifier_auroc_energy([0.5, 0.5])
        flipped_energy = vab.recompute_verifier_auroc_energy([-1.0, 1.0])
        assert default_energy is not None
        assert flipped_energy is not None
        assert flipped_energy < default_energy

    def test_never_raises_on_pathological_input(self) -> None:
        # A hypothesis's output is untrusted arbitrary data -- confirm the
        # function degrades to None rather than raising for every shape of
        # bad input a real generator could plausibly produce.
        for bad in (
            {"not": "a list"},
            [1.0, 2.0, 3.0],
            [],
            [object(), object()],
            [10**400, 0.0],  # OverflowError on float() -- 2026-09-16 review
        ):
            assert vab.recompute_verifier_auroc_energy(bad) is None

    def test_degenerate_constant_scorer_is_rejected(self) -> None:
        """2026-09-16 adversarial review, REAL_BUG 4: weights (0.0, 0.0) score
        every held-out row identically, which _binary_auroc's own tie rule
        turns into an AUROC of exactly 0.5 -- better than this benchmark's
        worse-than-chance seed baseline, and so was wrongly accepted as a
        real 'improvement' that discriminates nothing. Must now be None."""
        assert vab.recompute_verifier_auroc_energy([0.0, 0.0]) is None

    def test_a_real_non_degenerate_pair_is_not_rejected(self) -> None:
        """Confirm the degenerate-scorer guard does not also reject a real,
        non-constant scorer -- it must be narrowly scoped to the tied case."""
        assert vab.recompute_verifier_auroc_energy([0.5, 0.5]) is not None
        assert vab.recompute_verifier_auroc_energy([-1.0, 1.0]) is not None

    def test_empty_held_out_split_returns_none(self, monkeypatch) -> None:
        """REQ-AUTO-025: an empty held-out split cannot produce an AUROC."""
        monkeypatch.setattr(vab, "_split_corpus", lambda: ((), ()))
        assert vab.recompute_verifier_auroc_energy([0.5, 0.5]) is None

    def test_probe_scoring_exception_returns_none(self, monkeypatch) -> None:
        """REQ-AUTO-025: one unscoreable row must not crash the round."""

        class RaisingProbe:
            def __init__(self, **_weights) -> None:
                pass

            def score(self, _step_text: str, _context: str) -> float:
                raise RuntimeError("unscoreable row")

        held_out = ({"label": "incorrect", "step_text": "bad row"},)
        monkeypatch.setattr(vab, "_split_corpus", lambda: ((), held_out))
        monkeypatch.setattr(vab, "PCIBProbe", RaisingProbe)
        assert vab.recompute_verifier_auroc_energy([0.5, 0.5]) is None

    def test_undefined_auroc_returns_none(self, monkeypatch) -> None:
        """REQ-AUTO-025: an undefined held-out AUROC is unscoreable."""

        class VaryingProbe:
            def __init__(self, **_weights) -> None:
                pass

            def score(self, step_text: str, _context: str) -> float:
                return float(step_text)

        held_out = (
            {"label": "incorrect", "step_text": "1"},
            {"label": "correct", "step_text": "2"},
        )
        monkeypatch.setattr(vab, "_split_corpus", lambda: ((), held_out))
        monkeypatch.setattr(vab, "PCIBProbe", VaryingProbe)
        monkeypatch.setattr(vab, "_binary_auroc", lambda _labels, _scores: None)
        assert vab.recompute_verifier_auroc_energy([0.5, 0.5]) is None


class TestMeasureDefaultWeightEnergy:
    def test_matches_direct_recompute(self) -> None:
        assert vab.measure_default_weight_energy() == vab.recompute_verifier_auroc_energy(
            [0.5, 0.5]
        )

    def test_is_a_real_measurement_not_a_placeholder(self) -> None:
        """Regression guard for the exact number this module's docstring
        claims -- if the corpus or PCIBProbe changes underneath this, the
        seed baseline should visibly move, not silently stay stale."""
        energy = vab.measure_default_weight_energy()
        assert energy > 0.5  # default weights score WORSE than chance here
