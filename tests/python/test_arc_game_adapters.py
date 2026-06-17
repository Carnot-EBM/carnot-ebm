"""Unit tests for the ARC GameAdapter registry (python/carnot/agentic/arc_game_adapters.py).

A GameAdapter is a game's per-game RE captured as a plug-in for arc_solver_kit.OfflineSolver
(action_labels / apply / state_key / hand_verifier). These pin tu93's registration + its FRAME-BASED
verifier logic (player->goal Manhattan distance; player=colour 9, goal=colour 14) on synthetic grids,
game-independent. (The end-to-end proof is solve_adaptered('tu93', 1) -> offline_reproduced=True.)

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the ARC solve infrastructure / per-game adapters).
"""
import numpy as np

from carnot.agentic import arc_game_adapters as adapters


def test_tu93_registered_and_structural():
    assert "tu93" in adapters.adaptered_games()
    ad = adapters.get_adapter("tu93")
    assert ad is not None and ad.game == "tu93"
    for cb in (ad.action_labels, ad.apply, ad.state_key, ad.hand_verifier):
        assert callable(cb)
    assert ad.featurize is None          # frame-based RE; learned-verifier featurize is a future upgrade


def _grid(player_xy, goal_xy=(20, 10)):
    g = np.full((64, 64), 5, dtype=np.int16)   # floor colour
    g[goal_xy[1], goal_xy[0]] = 14             # goal = static colour-14 marker
    g[player_xy[1], player_xy[0]] = 9          # player = colour-9 sprite
    return g


def test_tu93_hand_verifier_is_player_goal_distance():
    ad = adapters.get_adapter("tu93")
    far = ad.hand_verifier(None, _grid((10, 10)))      # |20-10| + |10-10| = 10
    assert abs(far - 10.0) < 1e-6
    near = ad.hand_verifier(None, _grid((15, 10)))     # |20-15| = 5 -> closer
    assert near < far
    # missing player or goal -> a large finite default (never crashes the search)
    assert ad.hand_verifier(None, np.full((64, 64), 5, dtype=np.int16)) >= 1000.0


def test_tu93_state_key_distinguishes_player_positions():
    ad = adapters.get_adapter("tu93")
    k1 = ad.state_key(None, _grid((10, 10)))
    k2 = ad.state_key(None, _grid((14, 10)))
    assert k1 != k2                         # different player position -> different dedup key
    assert ad.state_key(None, _grid((10, 10))) == k1   # deterministic
