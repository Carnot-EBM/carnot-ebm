"""Tests for the 2026-07-27 gateway-Card-read correction and the checks added alongside it.

REQ-ARC-CARD-001 / SCENARIO-ARC-CARD-*: the gateway's charge is READ off its own `Card`, not
modelled as `offline_actions + resets - k`. These tests cover the pieces that decide whether a
future reader can trust that:

  * the Card reader tolerates every arcade shape it can be handed, and NEVER fabricates a number;
  * the independent invariant re-checker actually catches a corrupted cell (the mutation proofs);
  * the freshness lint survives BOTH shapes of `rows_sources` (a crash there blocks every commit);
  * the rebuild differ separates expected-on-rebuild leaves from measurement-bearing ones, which is
    the distinction a prior lane got wrong when it claimed "only git_head moved";
  * the p-floor walker used by the MAX_ACTIONS scope block derives its floor from the tests actually
    emitted rather than from a hardcoded literal.

Every test asserts. No test is skipped: none of them needs a GPU, a network, or a live board.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

# `arc_leaderboard_eval` and the per-level capture pull in the full ARC agent (and JAX through it),
# which costs ~600MB of RSS on first import. That trips the repo's memory watchdog at TEARDOWN, not
# because these tests leak but because the import is genuinely that big. `memory_watchdog_skip`
# exempts the TEARDOWN CHECK only -- every test below still runs and still asserts, so this is not a
# skipped test (CLAUDE.md forbids those). Same convention as
# tests/python/test_arc_scored_path_lever_harness.py.
pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------- the Card reader
def test_read_gateway_card_returns_empty_dict_when_there_is_no_scorecard_manager():
    """SCENARIO-ARC-CARD-001: no manager -> {} , so a consumer can tell 'never read' from 'zero'."""
    lb = _load("_lb_card", "scripts/arc_leaderboard_eval.py")

    class NoMgr:
        pass

    assert lb._read_gateway_card(NoMgr(), "vc33") == {}


def test_read_gateway_card_reads_the_LAST_play_row_not_the_first():
    """SCENARIO-ARC-CARD-002: the arcade's construction reset creates play 0 with zero actions; the
    agent's opening RESET creates play 1. Reading play 0 would report a charge of ZERO for every run.
    """
    lb = _load("_lb_card2", "scripts/arc_leaderboard_eval.py")

    class Card:
        def model_dump(self):
            return {
                "total_plays": 2,
                "actions": [0, 399],
                "resets": [0, 12],
                "actions_by_level": [[], [(1, 15), (2, 57)]],
            }

    class Scorecard:
        cards = {"vc33-abc": Card()}

    class Arc:
        scorecard_manager = type("M", (), {"scorecards": {"s": Scorecard()}})()

    got = lb._read_gateway_card(Arc(), "vc33")
    assert got["gateway_card_total_plays"] == 2
    assert got["gateway_card_play_index_read"] == 1
    assert got["gateway_card_actions"] == 399  # NOT 0 (play 0)
    assert got["gateway_card_resets"] == 12
    assert got["gateway_card_actions_by_level"] == [[1, 15], [2, 57]]


def test_read_gateway_card_reports_zero_plays_rather_than_inventing_a_charge():
    """SCENARIO-ARC-CARD-003: an unplayed card must not produce a charge of 0 that looks measured."""
    lb = _load("_lb_card3", "scripts/arc_leaderboard_eval.py")

    class Card:
        def model_dump(self):
            return {"total_plays": 0, "actions": [], "resets": [], "actions_by_level": []}

    class Scorecard:
        cards = {"vc33-abc": Card()}

    class Arc:
        scorecard_manager = type("M", (), {"scorecards": {"s": Scorecard()}})()

    got = lb._read_gateway_card(Arc(), "vc33")
    assert got == {"gateway_card_total_plays": 0}
    assert "gateway_card_actions" not in got


# --------------------------------------------------- the independent invariant re-checker
def _clean_cell() -> dict:
    return {
        "game": "vc33",
        "seed": 20260724,
        "levels": 2,
        "run_offline_actions": 387,
        "run_resets": 13,
        "run_frames": 400,
        "run_gateway_charged": 400,
        "segment_offline_actions": [15, 42],
        "segment_resets": [1, 1],
        "segment_gateway_charged": [16, 43],
        "tail_offline_actions": 330,
        "tail_resets": 11,
        "resets_in_completed_segments": 2,
        "resets_in_tail": 11,
        "efficiency_offline_precise": 2.0897311,
        "efficiency_gateway_charged_precise": 1.9955471,
        "card_actions": 399,
        "empty_frame_actions": 0,
        "observed_full_resets": 1,
        "consecutive_reset_pairs": 0,
    }


def test_the_recheck_passes_a_consistent_cell():
    """SCENARIO-ARC-CARD-010: the re-checker must not cry wolf on a clean cell."""
    cap = _load("_perlevel_cap", "scripts/arc_per_level_reset_attribution_capture.py")
    assert cap._recheck_cell(_clean_cell()) == []


def test_every_mutation_of_a_clean_cell_is_CAUGHT_by_the_recheck():
    """SCENARIO-ARC-CARD-011: the mutation proofs must actually catch their mutations.

    This is the test the prior lane's prose claimed and its artifact did not contain.
    """
    cap = _load("_perlevel_cap2", "scripts/arc_per_level_reset_attribution_capture.py")
    proofs = cap._mutation_proofs([_clean_cell()])
    assert proofs["ran"] is True
    assert proofs["baseline_violations_on_the_unmutated_cell"] == []
    assert proofs["n_mutations"] >= 10
    assert proofs["n_escaped"] == 0, proofs["escaped"]
    assert proofs["n_caught"] == proofs["n_mutations"]


def test_the_recheck_catches_a_gateway_score_that_beats_the_offline_score():
    """SCENARIO-ARC-CARD-012: charging MORE actions can never score HIGHER (the score is
    min((baseline/charged)**2*100, 115), monotonically decreasing in `charged`)."""
    cap = _load("_perlevel_cap3", "scripts/arc_per_level_reset_attribution_capture.py")
    bad = _clean_cell()
    bad["efficiency_gateway_charged_precise"] = 99.0
    assert "gateway_charged_score_exceeds_offline_score" in cap._recheck_cell(bad)


def test_the_recheck_catches_a_card_charge_above_offline_plus_resets():
    """SCENARIO-ARC-CARD-013: the Card can only ever charge <= offline + resets (free full resets,
    free post-death actions), so a larger value means the read or the join is wrong."""
    cap = _load("_perlevel_cap4", "scripts/arc_per_level_reset_attribution_capture.py")
    bad = _clean_cell()
    bad["card_actions"] = 5000
    assert "card_charge_exceeds_offline_plus_resets" in cap._recheck_cell(bad)


# ------------------------------------------------------- the freshness lint's field-shape fix
def test_freshness_lint_walks_BOTH_rows_sources_shapes():
    """SCENARIO-ARC-CARD-020: `rows_sources` is a dict-of-groups in most artifacts and a flat LIST in
    others. `.values()` on the list shape raised AttributeError and took the whole lint down, which
    blocks every commit while reporting nothing about staleness."""
    lint = _load("_freshlint", "scripts/artifact_freshness_lint.py")
    dict_shape = {"rows_sources": {"g1": [{"path": "a.json", "sha256": "x"}]}}
    list_shape = {"rows_sources": [{"path": "a.json", "sha256": "x"}]}
    assert lint._rows_source_entries(dict_shape) == [{"path": "a.json", "sha256": "x"}]
    assert lint._rows_source_entries(list_shape) == [{"path": "a.json", "sha256": "x"}]


def test_freshness_lint_tolerates_junk_rows_sources_without_raising():
    """SCENARIO-ARC-CARD-021: an unexpected shape must be SKIPPED, never fatal."""
    lint = _load("_freshlint2", "scripts/artifact_freshness_lint.py")
    for junk in ({"rows_sources": "nonsense"}, {"rows_sources": [None, 3]}, {}):
        assert lint._rows_source_entries(junk) == []


# ----------------------------------------------------------------- the rebuild differ
def test_rebuild_diff_separates_expected_leaves_from_measurement_bearing_ones():
    """SCENARIO-ARC-CARD-030: clocks, provenance fingerprints and the derived checksum move on ANY
    rebuild; a measurement number moving is a correction owed. A prior lane conflated the two and
    reported "only git_head moved" when five other leaf classes had also moved."""
    rd = _load("_rebuilddiff", "scripts/artifact_rebuild_diff.py")
    old = {
        "duration_s": 7.9,
        "run_date": "2026-07-26T00:00:00Z",
        "provenance": {"git_head": "aaa", "code": [{"path": "x.py", "sha256": "1", "bytes": 10}]},
        "reproducibility_checksum": "sha256:old",
        "median": 0.0369,
    }
    new = json.loads(json.dumps(old))
    new.update(
        {
            "duration_s": 10.1,
            "run_date": "2026-07-27T00:00:00Z",
            "reproducibility_checksum": "sha256:new",
        }
    )
    new["provenance"]["git_head"] = "bbb"
    new["provenance"]["code"][0]["sha256"] = "2"
    d = rd.diff(old, new)
    assert d["clean"] is True
    assert d["n_measurement_bearing"] == 0
    assert d["n_expected"] >= 5  # duration_s, run_date, checksum, git_head, code sha256

    new2 = json.loads(json.dumps(new))
    new2["median"] = 0.0181
    d2 = rd.diff(old, new2)
    assert d2["clean"] is False
    assert d2["n_measurement_bearing"] == 1
    assert d2["MEASUREMENT_BEARING"][0]["path"] == "median"


def test_rebuild_diff_flags_a_REMOVED_key_as_a_never_prune_violation():
    """SCENARIO-ARC-CARD-031: a rebuild that drops a key is a never-prune violation, not a diff."""
    rd = _load("_rebuilddiff2", "scripts/artifact_rebuild_diff.py")
    d = rd.diff({"a": 1, "b": 2}, {"a": 1})
    assert d["n_removed"] == 1 and d["clean"] is False


# ------------------------------------------------- the MAX_ACTIONS scope block's derived p-floor
def test_min_emitted_p_floor_is_derived_from_the_tests_actually_emitted():
    """SCENARIO-ARC-CARD-040: the scope block used to hardcode `n_seeds: 1` and compute its p-floor as
    2/2**n_games, producing 0.125 while the artifact's own headline reported p=0.0312 -- a false
    methodology violation manufactured by a stale literal."""
    ma = _load("_maxact", "scripts/analyze_arc_max_actions_answer.py")
    tree = {
        "a": {"min_reachable_two_sided_p_at_this_support": 0.5},
        "b": [{"nested": {"min_reachable_two_sided_p_at_this_support": 0.0312}}],
        "c": {"unrelated": 1},
    }
    assert sorted(ma._walk_p_floors(tree)) == [0.0312, 0.5]
    assert ma._min_emitted_p_floor(tree) == 0.0312
    assert ma._min_emitted_p_floor({"nothing": 1}) is None
