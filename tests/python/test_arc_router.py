"""Unit tests for the TRAINED ARC router (arc_router) — the learned 'which approach for which
game' model that replaces the hand-coded threshold and improves as the solve ledger grows.

These pin: the learned decision thresholds, prediction on clear cases, leave-one-out
GENERALISATION on a separable ledger (the honest 'does it generalise to an unseen game' test),
explore-vs-exploit by novelty, and the online ledger update.
"""
import json
import types

import numpy as np

from carnot.agentic import arc_router


def _entry(game, winner, cell_impact, bfs_expansions, action_type="click", spatial=True):
    return {"game": game, "winner": winner, "mask_hud": False,
            "features": {"cell_impact": cell_impact, "bfs_expansions": bfs_expansions,
                         "start_wrong_cells": cell_impact * 10, "start_wrong_regions": 5.0,
                         "solution_depth": 7.0, "action_type": action_type,
                         "spatial": spatial, "difficulty": "medium"},
            "outcomes": {}}


# A separable synthetic ledger mirroring the measured structure: small BFS-expansions ⇒ bfs;
# else low cell-impact ⇒ cell_count, high ⇒ region_count.
LEDGER = [
    _entry("a", "bfs", 200, 400), _entry("b", "bfs", 30, 300),
    _entry("c", "cell_count", 18, 1800), _entry("d", "cell_count", 22, 2000),
    _entry("e", "region_count", 70, 2200), _entry("f", "region_count", 90, 4000),
    _entry("g", "region_count", 55, 1900),
]


def test_learns_thresholds_between_classes():
    th = arc_router._learn_thresholds(LEDGER)
    # headroom between bfs-winners (max 400) and heuristic-winners (min 1800)
    assert 400 < th["headroom"] < 1800
    # impact between cell-winners (max 22) and region-winners (min 55)
    assert 22 < th["impact"] < 55


def test_route_predicts_clear_cases():
    model = arc_router.train(LEDGER)
    # low BFS expansions -> bfs (no headroom)
    assert arc_router.route(_entry("x", "?", 150, 350)["features"], model)["predicted"] == "bfs"
    # ample headroom + low impact -> cell_count
    assert arc_router.route(_entry("x", "?", 15, 2000)["features"], model)["predicted"] == "cell_count"
    # ample headroom + high impact -> region_count
    assert arc_router.route(_entry("x", "?", 95, 3000)["features"], model)["predicted"] == "region_count"


def test_leave_one_out_generalises_on_separable_data():
    # The honest generalisation test: train on N-1, predict the held-out game.
    loo = arc_router.leave_one_out(LEDGER)
    assert loo["accuracy"] == 1.0
    assert loo["n"] == len(LEDGER)


def test_empty_ledger_explores():
    out = arc_router.route(_entry("x", "?", 50, 1000)["features"], arc_router.train([]))
    assert out["decision"] == "explore"
    assert out["predicted"] is None


def test_novel_game_far_from_distribution_explores():
    # A game unlike anything in the ledger (extreme features) must EXPLORE, not blindly exploit.
    model = arc_router.train(LEDGER)
    out = arc_router.route({"cell_impact": 5000, "bfs_expansions": 10, "start_wrong_cells": 5,
                   "start_wrong_regions": 1, "solution_depth": 1, "action_type": "mixed",
                   "spatial": False, "difficulty": "hard"}, model)
    assert out["decision"] == "explore"


def test_record_appends_and_dedupes(tmp_path):
    p = tmp_path / "ledger.json"
    feats = _entry("g1", "region_count", 70, 2200)["features"]
    arc_router.record("g1", feats, "region_count", {}, path=p)
    arc_router.record("g1", feats, "cell_count", {}, path=p)        # same game -> replace, not duplicate
    arc_router.record("g2", feats, "bfs", {}, path=p)
    entries = json.loads(p.read_text())["entries"]
    assert len(entries) == 2
    assert {e["game"]: e["winner"] for e in entries} == {"g1": "cell_count", "g2": "bfs"}


def test_learned_cell_impact_threshold_is_between_classes():
    th = arc_router.learned_cell_impact_threshold(LEDGER)
    assert 22 < th < 55


def test_extract_features_has_expected_keys():
    win = np.zeros((6, 6), dtype=np.int16)
    g = win.copy()
    g[0, 0] = 1
    trans = [types.SimpleNamespace(grid=g, next_grid=win)]
    feats = arc_router.extract_features("nonexistent_game", win, trans, bfs_expansions=1500)
    assert feats["bfs_expansions"] == 1500.0
    assert feats["cell_impact"] == 1.0          # one cell changed in the single transition
    assert set(feats) >= {"cell_impact", "bfs_expansions", "start_wrong_cells",
                          "start_wrong_regions", "solution_depth", "action_type"}
