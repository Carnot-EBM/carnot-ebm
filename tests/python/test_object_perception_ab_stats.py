"""Tests for the LEVER #1 object-perception A/B measurement machinery
(REQ-ARC-WMTE-5830 / experiment_6018).

Two things are pinned here, and they are the two things that could silently invalidate the
measurement:

1. THE STATISTICS. The minimum reachable two-sided p, the exact sign test, and the exact
   sign-flip test. The failure mode these guard is the one CLAUDE.md names directly: calling
   a zero-discordance outcome "no significant difference" when no test was possible at all.
   `test_was_possible` must be False there, not a p of 1.0 wearing a passing face.

2. THE HELD-OUT SPLIT. `experiment_6018` claims its held-out set is the set of transitions
   the induction prompt did NOT show. That claim is checked against the REAL `induce_prompt`
   on REAL numpy grids -- no mock prompt, no reimplementation of `_transitions_block`'s
   `changed[:k-2] + noop[:2]` selection rule. If a future change to the prompt format breaks
   line-membership identification, the split would silently start reporting SHOWN
   transitions as held out (training accuracy relabelled as generalization), which is the
   exact defect experiment_6018 exists to remove from exp5831. These tests fail loudly then.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic import arc_executable_world_model as e3
from carnot.experiment_6018_object_perception_heldout_ab import (
    _paired_per_game,
    analyse,
    bootstrap_ci,
    build_preregistration,
    min_reachable_two_sided_p,
    sign_test_two_sided,
    signflip_exact_two_sided,
    split_shown_heldout,
    transition_prompt_line,
)


def _row(game: str, rep: int, arm: str, acc: float, *, measurable: bool = True) -> dict:
    return {
        "cell_id": f"{game}__r{rep}__{arm}",
        "game": game,
        "replicate": rep,
        "arm": arm,
        "stratum": "heldout",
        "induce_ok": True,
        "engine_loaded": True,
        "elapsed_s": 1.0,
        "prompt_chars": 100,
        "generated_tokens": 10,
        "heldout": {
            "measurable": measurable,
            "accuracy": acc,
            "cell_recall": acc,
            "change_fidelity": acc,
            "correct_changed_cells": 0,
            "spurious_changed_cells": 0,
            "n_changes_correct": 0,
            "n": 6,
        },
        "production_tail": {"measurable": False},
        "full_window": {"measurable": False},
    }


class TestMatchedPairingOnly:
    def test_a_replicate_missing_one_arm_is_dropped_from_the_pair(self) -> None:
        rows = [
            _row("aa", 0, "off", 0.1),
            _row("aa", 0, "on", 0.2),
            _row("aa", 1, "off", 0.9),  # no ON partner -> the whole replicate is dropped
        ]
        pg = _paired_per_game(rows, "heldout", "heldout", "accuracy")
        assert pg["aa"]["off"] == [0.1]
        assert pg["aa"]["on"] == [0.2]
        assert pg["aa"]["replicates"] == [0.0]

    def test_the_unmatched_cell_is_named_not_silently_averaged(self) -> None:
        rows = [
            _row("aa", 0, "off", 0.1),
            _row("aa", 0, "on", 0.2),
            _row("aa", 1, "off", 0.9),
        ]
        res = analyse(rows, {})
        assert res["n_unmatched_cells"] == 1
        assert res["unmatched_cells_excluded_from_pairing"] == ["aa__r1__off"]
        # and the surviving per-game mean must be the MATCHED one, not (0.1+0.9)/2
        assert res["primary"]["per_game_means"]["aa"]["off"] == 0.1

    def test_a_fully_matched_run_reports_zero_unmatched(self) -> None:
        rows = [
            _row(g, rep, arm, 0.0) for g in ("aa", "bb") for rep in (0, 1) for arm in ("off", "on")
        ]
        res = analyse(rows, {})
        assert res["n_unmatched_cells"] == 0
        assert res["primary"]["per_game_means"]["aa"]["n_matched_replicates"] == 2

    def test_an_unmeasurable_cell_cannot_partner_a_measurable_one(self) -> None:
        rows = [_row("aa", 0, "off", 0.1), _row("aa", 0, "on", 0.0, measurable=False)]
        pg = _paired_per_game(rows, "heldout", "heldout", "accuracy")
        assert pg == {}
        res = analyse(rows, {})
        assert res["n_unmatched_cells"] == 1


class TestMinReachableP:
    def test_zero_discordant_pairs_cannot_reach_any_significance(self) -> None:
        # THE load-bearing case: no discordance -> no test -> the floor is 1.0.
        assert min_reachable_two_sided_p(0) == 1.0

    def test_one_discordant_pair_still_cannot_reach_significance(self) -> None:
        assert min_reachable_two_sided_p(1) == 1.0

    def test_small_supports(self) -> None:
        assert min_reachable_two_sided_p(2) == 0.5
        assert min_reachable_two_sided_p(3) == 0.25
        assert min_reachable_two_sided_p(5) == 0.0625

    def test_fourteen_game_support_matches_two_to_the_minus_thirteen(self) -> None:
        assert min_reachable_two_sided_p(14) == 2.0 ** (-13)


class TestSignTest:
    def test_all_ties_reports_no_test_possible(self) -> None:
        out = sign_test_two_sided([0.0, 0.0, 0.0])
        assert out["test_was_possible"] is False
        assert out["n_discordant"] == 0
        assert out["n_ties"] == 3
        assert out["p_two_sided"] == 1.0
        assert out["min_reachable_two_sided_p_at_this_discordance"] == 1.0

    def test_unanimous_five_positive(self) -> None:
        out = sign_test_two_sided([0.1] * 5)
        assert out["n_positive"] == 5
        assert out["n_negative"] == 0
        assert out["p_two_sided"] == 0.0625
        assert out["test_was_possible"] is True

    def test_three_up_one_down(self) -> None:
        # d=4, max=3 -> 2 * (C(4,3)+C(4,4)) / 16 = 2 * 5/16 = 0.625
        out = sign_test_two_sided([1.0, 1.0, 1.0, -1.0])
        assert out["n_discordant"] == 4
        assert out["p_two_sided"] == 0.625

    def test_ties_are_dropped_from_the_denominator_but_still_counted(self) -> None:
        out = sign_test_two_sided([1.0, 1.0, 0.0, 0.0, 0.0])
        assert out["n_discordant"] == 2
        assert out["n_ties"] == 3
        assert out["p_two_sided"] == 0.5


class TestSignFlipTest:
    def test_no_nonzero_deltas_is_not_a_test(self) -> None:
        out = signflip_exact_two_sided([0.0, 0.0])
        assert out["test_was_possible"] is False
        assert out["n_nonzero"] == 0

    def test_three_equal_positive_deltas_enumerates_exactly(self) -> None:
        out = signflip_exact_two_sided([1.0, 1.0, 1.0])
        # Only all-plus and all-minus reach |sum| >= 3 out of 8 assignments.
        assert out["n_enumerated"] == 8
        assert out["p_two_sided"] == 0.25
        assert out["test_was_possible"] is True

    def test_single_delta_floor_is_one(self) -> None:
        out = signflip_exact_two_sided([0.5])
        assert out["p_two_sided"] == 1.0

    def test_oversized_support_is_declared_not_silently_skipped(self) -> None:
        out = signflip_exact_two_sided([1.0] * 30, max_n=20)
        assert out["test_was_possible"] is False
        assert out["p_two_sided"] is None
        assert "max_n" in out["note"]


class TestBootstrap:
    def test_constant_values_give_a_degenerate_interval(self) -> None:
        out = bootstrap_ci([0.25] * 8, seed=1, n_resamples=200)
        assert out["mean"] == 0.25
        assert out["lo"] == 0.25
        assert out["hi"] == 0.25

    def test_interval_brackets_the_mean_and_is_seed_reproducible(self) -> None:
        vals = [0.0, 0.1, -0.2, 0.4, -0.1, 0.05]
        a = bootstrap_ci(vals, seed=7, n_resamples=500)
        b = bootstrap_ci(vals, seed=7, n_resamples=500)
        assert a == b
        assert a["lo"] <= a["mean"] <= a["hi"]

    def test_empty_input_is_reported_as_no_data_not_as_zero(self) -> None:
        out = bootstrap_ci([], seed=1)
        assert out["mean"] is None
        assert out["n"] == 0


def _t(before: np.ndarray, after: np.ndarray, action: int) -> e3.Transition:
    return e3.Transition(
        grid=before, action=action, data=None, next_grid=after, level_before=0, level_after=0
    )


def _synthetic_window(n: int = 12) -> list[e3.Transition]:
    """n grid-CHANGING transitions with pairwise-distinct deltas, so every transition has a
    distinct rendered line and membership testing is unambiguous."""
    rows = []
    base = np.zeros((8, 8), dtype=int)
    for i in range(n):
        before = base.copy()
        before[0, 0] = i + 1
        after = before.copy()
        after[i % 8, (i * 3) % 8] = i + 2
        rows.append(_t(before, after, action=(i % 5) + 1))
    return rows


class TestHeldoutSplitAgainstTheRealPrompt:
    def test_default_k_shows_six_of_twelve_and_holds_out_the_rest(self) -> None:
        window = _synthetic_window(12)
        prompt = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        shown, held = split_shown_heldout(e3, "zz00", window, 1, prompt)
        assert len(shown) + len(held) == 12
        # k=8 -> changed[:6] + noop[:2]; this window is all-changing, so 6 are shown.
        assert len(shown) == 6
        assert len(held) == 6
        assert shown == [0, 1, 2, 3, 4, 5]

    def test_shown_set_is_NOT_a_positional_prefix_when_the_window_has_noops(self) -> None:
        """The discriminating case. `_transitions_block` samples `changed[:k-2] + noop[:2]`,
        so with no-ops interleaved the SHOWN set is not the first N transitions. A
        membership-by-index shortcut agrees with prompt-membership on an all-changing window
        and disagrees here -- without this case, replacing the real membership test with
        `i < 6` passes the whole file (verified by mutation)."""
        window = []
        base = np.zeros((8, 8), dtype=int)
        for i in range(12):
            before = base.copy()
            before[0, 0] = i + 1
            if i % 2 == 0:  # a genuine no-op at every even index
                window.append(_t(before, before.copy(), action=(i % 5) + 1))
            else:
                after = before.copy()
                after[i % 8, (i * 3) % 8] = i + 2
                window.append(_t(before, after, action=(i % 5) + 1))
        prompt = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        shown, held = split_shown_heldout(e3, "zz00", window, 1, prompt)
        assert shown != list(range(len(shown))), (
            "with no-ops present the shown set must not be a positional prefix, or this test "
            "cannot distinguish prompt-membership from an index heuristic"
        )
        # the 6 changing transitions are at odd indices; 2 no-ops (0, 2) are also shown
        assert 0 in shown and 2 in shown
        assert 4 in held and 6 in held
        for i in held:
            assert transition_prompt_line(e3, window[i]) not in prompt
        for i in shown:
            assert transition_prompt_line(e3, window[i]) in prompt

    def test_every_heldout_line_is_genuinely_absent_from_the_prompt(self) -> None:
        window = _synthetic_window(12)
        prompt = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        _shown, held = split_shown_heldout(e3, "zz00", window, 1, prompt)
        assert held, "the split must find something held out on a 12-transition window"
        for i in held:
            assert transition_prompt_line(e3, window[i]) not in prompt

    def test_every_shown_line_is_genuinely_present_in_the_prompt(self) -> None:
        window = _synthetic_window(12)
        prompt = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        shown, _held = split_shown_heldout(e3, "zz00", window, 1, prompt)
        for i in shown:
            assert transition_prompt_line(e3, window[i]) in prompt

    def test_short_window_holds_out_nothing_which_is_why_it_is_a_train_only_stratum(
        self,
    ) -> None:
        # THE exp5831 defect, pinned: a window the prompt shows entirely yields NO held-out
        # transition, so any accuracy on it is training accuracy.
        window = _synthetic_window(4)
        prompt = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        shown, held = split_shown_heldout(e3, "zz00", window, 1, prompt)
        assert len(shown) == 4
        assert held == []

    def test_object_block_changes_the_prompt_but_not_the_split(self, monkeypatch) -> None:
        window = _synthetic_window(12)
        monkeypatch.delenv("CARNOT_ARC_OBJECT_PERCEPTION", raising=False)
        p_off = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        monkeypatch.setenv("CARNOT_ARC_OBJECT_PERCEPTION", "1")
        p_on = e3.induce_prompt("zz00", window, 1, k=e3._induce_transitions_k())
        assert p_on != p_off
        assert p_on.startswith(p_off)
        assert "OBJECT STRUCTURE" in p_on
        assert "OBJECT STRUCTURE" not in p_off
        # The treatment must not move the train/heldout boundary, or the two arms would be
        # graded on different transition sets and the comparison would be meaningless.
        assert split_shown_heldout(e3, "zz00", window, 1, p_off) == split_shown_heldout(
            e3, "zz00", window, 1, p_on
        )


class TestPreregistration:
    def test_min_reachable_p_matches_the_declared_support(self) -> None:
        pre = build_preregistration()
        n = pre["n_primary_support_games"]
        assert n == len(pre["primary_support_games"])
        assert pre["min_reachable_two_sided_p_if_all_support_games_discordant"] == round(
            min_reachable_two_sided_p(n), 10
        )

    def test_floor_rule_and_train_only_stratum_are_declared(self) -> None:
        pre = build_preregistration()
        assert "unmeasurable_instrument_floor" in pre["floor_rule"]
        assert pre["train_only_stratum_games"]
        assert "TRAINING" in pre["train_only_stratum_reason"]
        # The excluded games must be named with reasons, never silently dropped.
        assert pre["excluded_games"]
        assert all(isinstance(v, str) and v for v in pre["excluded_games"].values())
